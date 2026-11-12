"""Tests for tools/index.py: sources config, chunking, mock-embedding builds, incremental
updates, embedding reuse, budgets and degradation. No API key, no network.

    python -m unittest discover -s tools -p 'test_*.py' -v
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
import index  # noqa: E402

DIM = 32
REPO = Path(__file__).resolve().parent.parent


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


class IndexCase(unittest.TestCase):
    """Isolates every global the indexer reads: DB, state dir, backend, dimensions."""

    backend = "mock"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / "memory.db"
        self.home = self.root / "home"
        self.data = self.root / "data"
        patches = {
            "MEMORY_DB": self.db, "DATA_DIR": self.data, "ANAMNESIS_HOME": self.home,
            "EMBED_SPEND_STATE_PATH": self.home / "spend.json", "QUERY_CACHE_PATH": self.home / "q.db",
            "BRANCH_DOCS_STAGING": self.home / "branch-docs", "SOURCES_FILE": None,
            "EMBED_BACKEND": self.backend, "EMBEDDING_MODEL": f"test-{self.backend}", "EMBEDDING_DIM": DIM,
            "_DIM_ENV": None, "BATCH_SIZE": 4, "ALERT_WEBHOOK": None, "_QUERY_CACHE_MEM": {},
            "_QUERY_CACHE_SCHEMA_OK": set(), "_embed_deadline": None, "_run_tokens_used": 0,
            "_run_exhausted": False, "_GEMINI_BACKUP_CLIENT": None,
        }
        for k, v in patches.items():
            p = mock.patch.object(index, k, v)
            p.start()
            self.addCleanup(p.stop)
        env = mock.patch.dict(os.environ, {k: v for k, v in os.environ.items()
                                           if not k.startswith(("ANAMNESIS_", "GEMINI_"))}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    # helpers
    def write(self, rel: str, text: str) -> Path:
        p = self.data / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def config(self, sources: list[dict], **extra) -> dict:
        path = self.root / "sources.json"
        path.write_text(json.dumps({"version": 1, **extra, "sources": sources}), encoding="utf-8")
        return index.load_sources_config(path)

    def rows(self, sql: str, *args):
        with contextlib.closing(sqlite3.connect(str(self.db))) as c:
            return c.execute(sql, args).fetchall()

    def fixture(self):
        self.write("memory/alpha.md", "---\nname: Alpha note\ndescription: about quokkas\n---\n"
                   "Quokkas are small marsupials that live on islands near the coast.\n\n"
                   "## Habitat\nThey prefer scrubland and swamps with dense vegetation nearby.\n")
        self.write("memory/feedback-beta.md", "# Beta\nPrefer boring technology over clever technology when shipping.")
        self.write("decisions/log.md", "# Decisions\n\n### 2026-01-01 adopt sqlite\n"
                   "We chose SQLite FTS5 for keyword retrieval because it needs no server.\n")
        self.write("manifests/people.json", json.dumps({"contacts": [
            {"name": "Ada Example", "relationship": "advisor", "notes": "knows compilers"}]}))
        return self.config([
            {"name": "memory", "root": str(self.data / "memory"), "glob": ["*.md"], "store_type": "memory",
             "store_type_rules": [{"match": "feedback*", "store_type": "feedback"}]},
            {"name": "decisions", "root": str(self.data / "decisions"), "store_type": "decision"},
            {"name": "manifests", "root": str(self.data / "manifests"), "glob": ["*.json"], "store_type": "contact"},
        ])


class MockEmbeddingTests(IndexCase):
    def test_deterministic_unit_vectors(self):
        a, b, c = index.mock_embedding("same"), index.mock_embedding("same"), index.mock_embedding("other")
        self.assertEqual(len(a), DIM)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertAlmostEqual(sum(x * x for x in a), 1.0, places=5)

    def test_embed_batch_and_query_use_mock_without_client(self):
        self.assertEqual(len(index.embed_batch(None, ["x", "y"])), 2)
        self.assertEqual(index.embed_query(None, "q"), index.mock_embedding("q"))
        self.assertIsNone(index.get_indexing_client())

    def test_pack_roundtrip(self):
        v = [0.5, 0.25, -1.0]
        self.assertEqual(index.unpack_embedding(index.pack_embedding(v)), v)


class ChunkerTests(IndexCase):
    def test_markdown_frontmatter_title_description_topics(self):
        text = "---\nname: My Note\ndescription: short summary\ntopics:\n  - alpha\n  - beta\n---\n" \
               "Body text long enough to survive the minimum section size filter."
        chunks = index.chunk_markdown(text, "x/note.md")
        self.assertEqual(chunks[0]["title"], "My Note")
        self.assertTrue(chunks[0]["content"].startswith("short summary"))
        self.assertEqual(chunks[0]["topic_ids"], "alpha beta")
        self.assertNotIn("name: My Note", chunks[0]["content"])

    def test_code_fence_is_never_split(self):
        fence = "```python\n" + "\n".join(f"line_{i} = {i}" for i in range(120)) + "\n```"
        text = "# T\n\n" + "intro paragraph. " * 40 + "\n\n" + fence + "\n\n" + "outro. " * 40
        parts = index.chunk_markdown(text, "d.md", max_chars=500)
        holders = [p for p in parts if "line_0 = 0" in p["content"]]
        self.assertEqual(len(holders), 1)
        self.assertIn("line_119 = 119", holders[0]["content"])

    def test_json_prose_twin(self):
        data = {"mistakes": [{"mistake_id": "m1", "symptom": "disk full", "root_cause": "no rotation",
                              "topic_ids": ["ops"]}]}
        (chunk,) = index.chunk_json_entries(data, "m.json")
        self.assertIn("Root cause: no rotation", chunk["content"])
        self.assertEqual(chunk["topic_ids"], "ops")
        self.assertEqual(json.loads(chunk["metadata"])["mistake_id"], "m1")

    def test_json_generic_fallback(self):
        chunks = index.chunk_json_entries({"note": "hello world", "empty": ""}, "x.json")
        self.assertEqual([c["title"] for c in chunks], ["note"])
        self.assertEqual(index.chunk_json_entries([{"title": "A", "k": "v"}], "x.json")[0]["title"], "A")

    def test_facts_chunker_overview_and_bullets(self):
        bullets = "\n".join(f"- capability number {i} does something rather specific and useful" for i in range(6))
        text = f"# Facts\nPurpose of this project is demonstrating things.\n\n## Capabilities\n{bullets}\n"
        titles = [c["title"] for c in index.chunk_facts(text, "p/facts.md")]
        self.assertEqual(titles[0], "facts - Overview")
        self.assertEqual(sum(t.startswith("Capabilities: capability number") for t in titles), 6)

    def test_article_chunker_strips_jsx_and_has_metadata_chunk(self):
        text = "---\ntitle: Post\ntags: [a, b]\ndate: 2026-01-01\n---\n<Widget prop=\"1\" />\n" \
               "![img](x.png)\n## Section\nReal article text that is long enough to be kept as a chunk."
        chunks = index.chunk_article(text, "post.mdx")
        self.assertIn("Tags: a, b", chunks[0]["content"])
        self.assertTrue(all("Widget" not in c["content"] and "x.png" not in c["content"] for c in chunks))


class SourcesConfigTests(IndexCase):
    def test_builtin_default_points_at_data_dir(self):
        cfg = index.load_sources_config()
        self.assertEqual([s["name"] for s in cfg["sources"]], ["memory", "decisions", "manifests"])
        self.assertEqual(cfg["sources"][0]["root"], self.data / "memory")

    def test_example_config_validates(self):
        if not importlib.util.find_spec("yaml"):
            self.skipTest("PyYAML not installed")
        cfg = index.load_sources_config(REPO / "anamnesis.sources.example.yaml")
        self.assertEqual({s["kind"] for s in cfg["sources"]}, {"files", "git_metadata", "git_branch_docs"})
        self.assertEqual(cfg["sources"][0]["root"], self.data / "memory")  # ${ANAMNESIS_DATA_DIR} resolved

    def test_invalid_configs_are_rejected(self):
        bad = [
            ({"sources": [{"root": "x", "colour": "red"}]}, "unknown key"),
            ({"sources": []}, "non-empty"),
            ({"sources": [{"store_type": "memory"}]}, "root"),
            ({"sources": [{"root": "x", "store_type": "Bad Type"}]}, "store_type"),
            ({"sources": [{"root": "$NO_SUCH_VAR_XYZ/x"}]}, "undefined"),
            ({"sources": [{"root": "x", "chunker": "magic"}]}, "chunker"),
            ({"version": 2, "sources": [{"root": "x"}]}, "version"),
            ({"nope": 1, "sources": [{"root": "x"}]}, "top-level"),
            ({"sources": [{"root": "https://example.com/docs"}]}, "URLs are not allowed"),
            ({"sources": [{"root": "ftp://host/dir"}]}, "URLs are not allowed"),
            ({"sources": [{"root": "file:///etc"}]}, "URLs are not allowed"),
            ({"sources": [{"kind": "git_branch_docs", "repo": "git://host/r.git"}]}, "URLs are not allowed"),
            ({"sources": [{"kind": "git_metadata", "repos": ["ssh://host/r.git"]}]}, "URLs are not allowed"),
        ]
        for cfg, needle in bad:
            with self.subTest(cfg=cfg), self.assertRaises(index.ConfigError) as cm:
                index._validate_config(cfg, self.root)
            self.assertIn(needle, str(cm.exception))

    def test_exclude_rules_and_disabled_sources(self):
        self.write("memory/keep.md", "# Keep\nThis file should be indexed by the files source.")
        self.write("memory/_archive/old.md", "# Old\nThis file lives in an excluded archive directory.")
        self.write("memory/skip-me.md", "# Skip\nThis one is excluded by a name pattern in config.")
        self.write("other/x.md", "# X\nThis belongs to a source that is disabled in the config.")
        cfg = self.config([
            {"root": str(self.data / "memory"), "exclude": ["_archive", "skip-*"]},
            {"root": str(self.data / "other"), "enabled": False},
        ])
        names = sorted(Path(s["path"]).name for s in index.collect_sources(cfg))
        self.assertEqual(names, ["keep.md"])

    def test_store_type_rules_weight_and_project(self):
        self.write("memory/feedback-x.md", "# F\nFeedback body that is long enough for the chunker.")
        self.write("memory/plain.md", "# P\nPlain body that is long enough for the chunker.")
        cfg = self.config([{"root": str(self.data / "memory"), "store_type": "memory", "project": "proj-a",
                            "weight": 2.0, "store_type_rules": [{"match": "feedback*", "store_type": "feedback"}]}],
                          store_weights={"feedback": 1.5})
        by = {Path(s["path"]).name: s for s in index.collect_sources(cfg)}
        self.assertEqual(by["feedback-x.md"]["store_type"], "feedback")
        self.assertEqual(by["plain.md"]["store_type"], "memory")
        self.assertEqual(by["plain.md"]["project"], "proj-a")
        chunks = index._build_new_chunks_for_source(by["feedback-x.md"])
        self.assertAlmostEqual(chunks[0]["weight"], 1.5 * 2.0)

    def test_retired_sources_are_skipped(self):
        self.write("memory/gone.md", "---\nretirement_status: retired\n---\nThis memory was retired long ago.")
        self.write("memory/live.md", "---\npriority: invariant\n---\nThis memory is still live and relevant.")
        cfg = self.config([{"root": str(self.data / "memory")}])
        srcs = index.collect_sources(cfg)
        self.assertEqual([Path(s["path"]).name for s in srcs], ["live.md"])
        self.assertEqual(srcs[0]["priority"], "invariant")

    def test_symlink_escaping_root_and_oversized_files_are_skipped(self):
        outside = self.root / "outside.md"
        outside.write_text("# Secret\nContent outside the configured source root must not be indexed.")
        self.write("memory/ok.md", "# Ok\nA normal file inside the root that is fine to index.")
        self.write("memory/big.md", "# Big\n" + "x" * 500)
        link = self.data / "memory" / "link.md"
        try:
            link.symlink_to(outside)
        except OSError:
            self.skipTest("symlinks unavailable")
        cfg = self.config([{"root": str(self.data / "memory")}])
        with quiet():
            names = sorted(Path(s["path"]).name for s in index.collect_sources(cfg))
        self.assertEqual(names, ["big.md", "ok.md"])  # link.md escapes the root
        with mock.patch.object(index, "MAX_FILE_BYTES", 200), quiet():
            names = sorted(Path(s["path"]).name for s in index.collect_sources(cfg))
        self.assertEqual(names, ["ok.md"])  # big.md exceeds the size cap
        allow = self.config([{"root": str(self.data / "memory"), "follow_symlinks": True}])
        self.assertIn("link.md", [Path(s["path"]).name for s in index.collect_sources(allow)])


class BuildTests(IndexCase):
    def build(self, cfg):
        with quiet():
            index.build_index(config=cfg)

    def update(self, cfg, **kw):
        with quiet():
            index.update_index(config=cfg, **kw)

    def test_build_indexes_fixture_with_embeddings_and_fts(self):
        cfg = self.fixture()
        self.build(cfg)
        total, embedded = self.rows("SELECT COUNT(*), COUNT(embedding) FROM chunks")[0]
        self.assertGreaterEqual(total, 4)
        self.assertEqual(total, embedded)
        for (blob,) in self.rows("SELECT embedding FROM chunks"):
            self.assertEqual(len(blob), DIM * 4)
        hits = self.rows("SELECT c.title, c.store_type FROM chunks_fts f JOIN chunks c ON c.id = f.rowid "
                         "WHERE chunks_fts MATCH 'quokkas'")
        self.assertEqual({h[1] for h in hits}, {"memory"})
        self.assertEqual({t for (t,) in self.rows("SELECT DISTINCT store_type FROM chunks")},
                         {"memory", "feedback", "decision", "contact"})
        meta = dict(self.rows("SELECT key, value FROM index_meta"))
        self.assertEqual(meta["embedding_model"], "test-mock")
        self.assertEqual(int(meta["embedding_dim"]), DIM)
        self.assertIn("last_updated", meta)
        (twin,) = self.rows("SELECT metadata FROM chunks WHERE store_type = 'contact'")
        self.assertEqual(json.loads(twin[0])["name"], "Ada Example")

    def test_rebuild_and_update_are_idempotent(self):
        cfg = self.fixture()
        self.build(cfg)
        first = self.rows("SELECT chunk_id, content, embedding FROM chunks ORDER BY chunk_id")
        self.build(cfg)
        self.assertEqual(self.rows("SELECT chunk_id, content, embedding FROM chunks ORDER BY chunk_id"), first)
        ids = self.rows("SELECT id FROM chunks ORDER BY id")
        self.update(cfg)
        self.assertEqual(self.rows("SELECT id FROM chunks ORDER BY id"), ids)
        self.assertEqual(self.rows("SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'sqlite'")[0][0], 1)
        self.assertEqual(self.rows("PRAGMA integrity_check")[0][0], "ok")

    def test_update_touches_only_changed_sources(self):
        cfg = self.fixture()
        self.build(cfg)
        sql = "SELECT source_path, id, content FROM chunks WHERE store_type='decision'"
        before = {r[0]: r[1:] for r in self.rows(sql)}
        alpha = self.data / "memory" / "alpha.md"
        alpha.write_text(alpha.read_text() + "\n## Diet\nThey eat grasses, leaves and succulent plants every day.\n")
        os.utime(alpha, (alpha.stat().st_atime, alpha.stat().st_mtime + 10))
        self.write("memory/gamma.md", "# Gamma\nA brand new memory about pelicans arriving after the first build.")
        (self.data / "memory" / "feedback-beta.md").unlink()
        self.update(cfg)
        self.assertEqual({r[0]: r[1:] for r in self.rows(sql)}, before)
        self.assertEqual(len(self.rows("SELECT 1 FROM chunks_fts WHERE chunks_fts MATCH 'succulent'")), 1)
        self.assertEqual(len(self.rows("SELECT 1 FROM chunks_fts WHERE chunks_fts MATCH 'pelicans'")), 1)
        self.assertEqual(len(self.rows("SELECT 1 FROM chunks_fts WHERE chunks_fts MATCH 'boring'")), 0)
        self.assertEqual(self.rows("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL")[0][0], 0)
        self.assertEqual(self.rows("SELECT COUNT(*) FROM source_meta")[0][0], 4)

    def test_dry_run_writes_nothing(self):
        cfg = self.fixture()
        self.build(cfg)
        self.write("memory/new.md", "# New\nA file that appears after the build and must not be written.")
        snapshot = self.rows("SELECT chunk_id FROM chunks ORDER BY chunk_id")
        self.update(cfg, dry_run=True)
        self.assertEqual(self.rows("SELECT chunk_id FROM chunks ORDER BY chunk_id"), snapshot)

    def test_moved_file_reuses_embeddings_without_embedding_again(self):
        cfg = self.fixture()
        self.build(cfg)
        shutil.move(str(self.data / "memory" / "alpha.md"), str(self.data / "memory" / "alpha-moved.md"))
        with mock.patch.object(index, "embed_batch", wraps=index.embed_batch) as spy:
            self.build(cfg)  # same text under a new path: corpus-wide reuse, zero embed calls
        embedded_texts = [t for call in spy.call_args_list for t in call.args[1]]
        self.assertEqual(embedded_texts, [])
        self.assertEqual(self.rows("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL")[0][0], 0)

    def test_sidecar_chunks_survive_rebuild(self):
        cfg = self.fixture()
        self.build(cfg)
        conn = index.init_db(self.db)
        conn.execute("INSERT INTO chunks (chunk_id, store_type, source_path, title, content, indexed_at) "
                     "VALUES ('clip:1', 'conversation_clip', '/x', 'clip', 'clipped words zebra', 'now')")
        conn.commit()
        conn.close()
        self.build(cfg)
        self.assertEqual(self.rows("SELECT COUNT(*) FROM chunks WHERE chunk_id = 'clip:1'")[0][0], 1)
        self.assertEqual(len(self.rows("SELECT 1 FROM chunks_fts WHERE chunks_fts MATCH 'zebra'")), 1)
        self.update(cfg)  # update leaves it alone and backfills its missing vector
        self.assertEqual(self.rows("SELECT length(embedding) FROM chunks WHERE chunk_id = 'clip:1'")[0][0], DIM * 4)

    def test_refuses_to_index_into_a_database_from_another_model(self):
        cfg = self.fixture()
        self.build(cfg)
        with mock.patch.object(index, "EMBEDDING_MODEL", "some-other-model"):
            with self.assertRaises(SystemExit):
                self.update(cfg)

    def test_search_returns_expected_chunk_and_survives_hostile_queries(self):
        cfg = self.fixture()
        self.build(cfg)
        res = index.search("quokkas marsupials", limit=3, quiet=True)
        self.assertTrue(res)
        self.assertTrue(res[0]["source_path"].endswith("alpha.md"))
        for q in ['foo" OR "bar', "NEAR(a b)", "title:secret*", "'; DROP TABLE chunks;--", "!!"]:
            index.search(q, quiet=True)  # must not raise
        self.assertGreater(self.rows("SELECT COUNT(*) FROM chunks")[0][0], 0)

    def test_project_boost_prefers_named_project(self):
        self.write("memory/a.md", "# A\nThe deploy pipeline runs nightly for the orchard application.")
        self.write("memory/b.md", "# B\nThe deploy pipeline runs nightly for the harvest application.")
        cfg = self.config([{"root": str(self.data / "memory"), "glob": ["a.md"], "project": "orchard"},
                           {"root": str(self.data / "memory"), "glob": ["b.md"], "project": "harvest"}])
        self.build(cfg)
        conn = sqlite3.connect(str(self.db))
        aliases = index._project_aliases(conn, {}, ["shared"])
        conn.close()
        self.assertEqual(index._project_boost("deploy pipeline orchard", "orchard", "", aliases, []), 2.0)
        self.assertEqual(index._project_boost("deploy pipeline orchard", "harvest", "", aliases, []), 0.5)
        self.assertEqual(index._project_boost("deploy pipeline orchard", "shared", "", aliases, ["shared"]), 1.0)
        self.assertEqual(index._project_boost("something else", "harvest", "", aliases, []), 1.0)

    def test_corrupt_json_is_skipped_not_fatal(self):
        self.write("manifests/bad.json", "{not json")
        self.write("manifests/good.json", json.dumps({"k": "a value that is present"}))
        cfg = self.config([{"root": str(self.data / "manifests"), "glob": ["*.json"], "store_type": "index"}])
        self.build(cfg)
        self.assertEqual([r[0] for r in self.rows("SELECT title FROM chunks")], ["k"])


class RecallIntegrationTests(IndexCase):
    def test_recall_library_reads_mock_built_db_via_index_embed_query(self):
        import recall
        self.build_fixture()
        with mock.patch.object(index, "get_embed_client", return_value=None):
            hits = recall.recall("quokkas", 3, db=self.db, embed=lambda q: index.embed_query(None, q))
        self.assertTrue(hits and hits[0]["source_path"].endswith("alpha.md"))
        kw = recall.recall("quokkas", 3, db=self.db, embed=None)  # keyword-only path
        self.assertTrue(kw)

    def build_fixture(self):
        cfg = self.fixture()
        with quiet():
            index.build_index(config=cfg)


class DegradeAndBudgetTests(IndexCase):
    backend = "gemini"

    def fake_gemini(self, fail_if=None, calls=None):
        def call(_client, texts):
            if calls is not None:
                calls.append(list(texts))
            if fail_if and fail_if(texts):
                raise RuntimeError("400 INVALID_ARGUMENT: text too long")
            resp = mock.MagicMock()
            resp.embeddings = [mock.MagicMock(values=index.mock_embedding(t)) for t in texts]
            resp.usage_metadata = None
            return resp
        return call

    def test_backoff_retries_transient_errors_with_exponential_waits(self):
        attempts = []

        def flaky(c, texts):
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("429 RESOURCE_EXHAUSTED quota")
            return self.fake_gemini()(c, texts)
        with mock.patch.object(index, "_gemini_embed_call", flaky), \
                mock.patch.object(index.time, "sleep") as sleep, quiet():
            out = index.embed_batch(object(), ["a"], max_retries=3)
        self.assertEqual(len(out), 1)
        waits = [c.args[0] for c in sleep.call_args_list]
        self.assertEqual(len(waits), 2)
        self.assertTrue(5 <= waits[0] < 5.5 and 10 <= waits[1] < 10.5, waits)

    def test_non_transient_errors_are_not_retried(self):
        with mock.patch.object(index, "_gemini_embed_call", side_effect=RuntimeError("403 PERMISSION_DENIED")), \
                mock.patch.object(index.time, "sleep") as sleep:
            with self.assertRaises(RuntimeError):
                index.embed_batch(object(), ["a"], max_retries=3)
        sleep.assert_not_called()

    def test_backup_key_is_tried_once_after_rate_limit_exhaustion(self):
        primary, backup = object(), object()

        def call(client, texts):
            if client is primary:
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            return self.fake_gemini()(client, texts)
        os.environ["GEMINI_API_KEY_BACKUP"] = "backup-test-value"
        with mock.patch.object(index, "_gemini_embed_call", call), \
                mock.patch.object(index, "get_gemini_client", return_value=backup), \
                mock.patch.object(index.time, "sleep"), quiet():
            out = index.embed_batch(primary, ["a"], max_retries=2)
        self.assertEqual(len(out), 1)

    def test_poison_text_is_isolated_by_bisection(self):
        fake = self.fake_gemini(fail_if=lambda texts: any("POISON" in t for t in texts))
        texts = [f"t{i}" for i in range(7)] + ["POISON"]
        with mock.patch.object(index, "_gemini_embed_call", fake), quiet():
            vecs = index.embed_with_degrade(object(), texts)
        self.assertEqual([v is None for v in vecs], [False] * 7 + [True])

    def test_build_lands_every_chunk_when_embedding_partly_fails(self):
        for i in range(6):
            self.write(f"memory/n{i}.md", f"# Note {i}\n" + ("filler sentence about topic. " * 5) +
                       ("POISON marker" if i == 3 else ""))
        cfg = self.config([{"root": str(self.data / "memory")}])
        os.environ["GEMINI_API_KEY"] = "test-value-not-a-key"
        fake = self.fake_gemini(fail_if=lambda texts: any("POISON" in t for t in texts))
        with mock.patch.object(index, "_gemini_embed_call", fake), mock.patch.object(index, "get_gemini_client"), quiet():
            index.build_index(config=cfg)
        total, embedded = self.rows("SELECT COUNT(*), COUNT(embedding) FROM chunks")[0]
        self.assertEqual((total, embedded), (6, 5))
        self.assertEqual(len(self.rows("SELECT 1 FROM chunks_fts WHERE chunks_fts MATCH 'POISON'")), 1)  # still searchable
        with mock.patch.object(index, "_gemini_embed_call", fake), mock.patch.object(index, "get_gemini_client"), quiet():
            index.backfill_embeddings()  # still poisoned: stays NULL, does not crash
        self.assertEqual(self.rows("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL")[0][0], 1)

    def test_run_budget_degrades_to_fts_only_then_backfill_heals(self):
        for i in range(5):
            self.write(f"memory/n{i}.md", f"# Note {i}\n" + "distinct filler words for this note number. " * 4)
        cfg = self.config([{"root": str(self.data / "memory")}])
        os.environ["GEMINI_API_KEY"] = "test-value-not-a-key"
        fake = self.fake_gemini()
        with mock.patch.object(index, "_gemini_embed_call", fake), mock.patch.object(index, "get_gemini_client"), \
                mock.patch.object(index, "EMBED_RUN_TOKEN_CAP", 600), quiet(), \
                contextlib.redirect_stderr(io.StringIO()):
            index.build_index(config=cfg)
        total, embedded = self.rows("SELECT COUNT(*), COUNT(embedding) FROM chunks")[0]
        self.assertEqual(total, 5)
        self.assertLess(embedded, 5)
        with mock.patch.object(index, "_gemini_embed_call", fake), mock.patch.object(index, "get_gemini_client"), quiet():
            index.backfill_embeddings()
        self.assertEqual(self.rows("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL")[0][0], 0)

    def test_monthly_cap_persists_and_blocks(self):
        state = {"month": index._current_month(), "tokens_used": 10_000, "alerts_sent": {}}
        index.EMBED_SPEND_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        index.EMBED_SPEND_STATE_PATH.write_text(json.dumps(state))
        with mock.patch.object(index, "EMBED_MONTHLY_TOKEN_CAP", 10_050), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(index.EmbedBudgetExceeded):
                index._reserve_embed_budget(100)
        self.assertEqual(json.loads(index.EMBED_SPEND_STATE_PATH.read_text())["tokens_used"], 10_000)

    def test_missing_key_is_a_clean_cli_error(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("GEMINI_", "ANAMNESIS_"))}
        env.update(ANAMNESIS_HOME=str(self.home), ANAMNESIS_DB=str(self.db), ANAMNESIS_DATA_DIR=str(self.data),
                   ANAMNESIS_EMBED_BACKEND="gemini")
        self.write("memory/a.md", "# A\nSome text long enough to be a chunk of its own here.")
        r = subprocess.run([sys.executable, str(REPO / "tools" / "index.py"), "build"], env=env,
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("GEMINI_API_KEY", r.stderr)


class CliTests(IndexCase):
    def run_cli(self, *args, sources=None):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("GEMINI_", "ANAMNESIS_"))}
        env.update(ANAMNESIS_HOME=str(self.home), ANAMNESIS_DB=str(self.db), ANAMNESIS_DATA_DIR=str(self.data),
                   ANAMNESIS_EMBED_BACKEND="mock", ANAMNESIS_EMBEDDING_DIM=str(DIM))
        if sources:
            env["ANAMNESIS_SOURCES"] = str(sources)
        return subprocess.run([sys.executable, str(REPO / "tools" / "index.py"), *args], env=env,
                              capture_output=True, text=True)

    def test_build_search_stats_end_to_end(self):
        self.write("memory/alpha.md", "# Alpha\nQuokkas are small marsupials that live on islands.")
        r = self.run_cli("build")
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        r = self.run_cli("search", "quokkas")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("alpha.md", r.stdout)
        self.assertIn("Embedding coverage: 1/1", self.run_cli("stats").stdout)

    def test_bare_invocation_builds_then_updates(self):
        self.write("memory/alpha.md", "# Alpha\nQuokkas are small marsupials that live on islands.")
        self.assertIn("Running full build", self.run_cli().stdout)
        self.assertIn("No changes detected", self.run_cli().stdout)

    def test_bad_sources_config_exits_2(self):
        bad = self.root / "bad.json"
        bad.write_text(json.dumps({"sources": [{"root": "x", "bogus": 1}]}))
        r = self.run_cli("sources", sources=bad)
        self.assertEqual(r.returncode, 2)
        self.assertIn("unknown key", r.stderr)


@unittest.skipUnless(shutil.which("git"), "git not installed")
class GitSourcesTests(IndexCase):
    def git(self, cwd, *args):
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.com",
                   GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.com", GIT_CONFIG_GLOBAL=os.devnull)
        subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True)

    def make_repo(self):
        origin, work = self.root / "origin.git", self.root / "work"
        self.git(self.root, "init", "--bare", "-b", "main", str(origin))
        self.git(self.root, "clone", str(origin), str(work))
        self.git(work, "checkout", "-b", "main")
        (work / "docs").mkdir()
        (work / "docs" / "main.md").write_text("# Main\nA document that is present on the main branch itself.\n")
        for args in (("add", "."), ("commit", "-m", "init"), ("push", "origin", "main")):
            self.git(work, *args)
        self.git(work, "checkout", "-b", "feature/x")
        (work / "docs" / "proposal.md").write_text("# Proposal\nAn unmerged proposal about wombat migration routes.\n")
        for args in (("add", "."), ("commit", "-m", "proposal"), ("push", "origin", "feature/x"),
                     ("checkout", "main"), ("fetch", "origin")):
            self.git(work, *args)
        return work

    def test_branch_only_docs_are_staged_with_provenance_and_metadata_chunk(self):
        work = self.make_repo()
        cfg = self.config([
            {"kind": "git_branch_docs", "repo": str(work), "glob": ["docs/*.md"], "base_ref": "origin/main",
             "project": "demo"},
            {"kind": "git_metadata", "repos": [str(work)], "project": "demo"},
        ])
        with quiet():
            srcs = index.collect_sources(cfg)
        self.assertEqual([Path(s["path"]).name for s in srcs], ["proposal.md"])
        staged = Path(srcs[0]["path"])
        self.assertIn("PROPOSED work, not merged", staged.read_text())
        self.assertTrue(str(staged).startswith(str((self.home / "branch-docs").resolve())))
        (meta,) = index.collect_git_metadata(cfg)
        self.assertEqual(meta["title"], "work - Git Metadata")
        self.assertIn("Total commits:", meta["content"])
        with quiet():
            index.build_index(config=cfg)
        self.assertEqual(len(self.rows("SELECT 1 FROM chunks_fts WHERE chunks_fts MATCH 'wombat'")), 1)
        self.assertEqual(self.rows("SELECT COUNT(*) FROM chunks WHERE store_type = 'project_meta'")[0][0], 1)


if __name__ == "__main__":
    unittest.main()
