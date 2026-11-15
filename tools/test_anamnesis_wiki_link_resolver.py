#!/usr/bin/env python3
"""Hermetic tests for wiki_link_resolver.py.

Exercises the classifier + apply path against a synthetic SQLite DB and a synthetic
tempdir of markdown files. Fictional data only; no network.

Checks:
  (a) an exact-stem [[slug]] resolves to the right chunk pair
  (b) a prefix-stripped alias ([[prefer-small-prs]] for feedback_prefer_small_prs.md)
  (c) a slug-shaped `name:` frontmatter alias resolves
  (d) a [[slug]] with no matching file is dangling, never inserted
  (e) a [[slug]] that collides across two files is ambiguous, never inserted
  (f) apply is idempotent
  (g) other relation types and the markdown files are never touched
  (h) a self-reference is bucketed apart from dangling
  (i) an unindexed source file's occurrences are individually accounted for
  (j) concurrent apply_edges calls produce exactly one row per (from,to) pair
  (k) every occurrence lands in exactly one bucket
  (l) a file indexed twice (symlinked dir) is treated as indexed, tagged duplicate_*

    python -m unittest discover -s tools -p 'test_*.py' -v
"""
import importlib.util
import re
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "wiki_link_resolver.py"

spec = importlib.util.spec_from_file_location("wiki_link_resolver", MODULE_PATH)
wlr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wlr)  # type: ignore

def make_db(path: Path):
    # isolation_level=None: manual transaction control, matching production
    # (main() connects the same way) so BEGIN IMMEDIATE in apply_edges is
    # the only transaction boundary in play.
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.executescript(
        """
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT UNIQUE NOT NULL,
            store_type TEXT NOT NULL,
            source_path TEXT NOT NULL,
            title TEXT,
            content TEXT NOT NULL,
            indexed_at TEXT NOT NULL
        );
        CREATE TABLE nugget_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_chunk_id INTEGER NOT NULL,
            to_chunk_id INTEGER,
            relation_type TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            source_text TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )
    # pre-existing causal edges the resolver must never touch
    conn.execute(
        "INSERT INTO nugget_links (from_chunk_id, to_chunk_id, relation_type, source_text) "
        "VALUES (1, NULL, 'broke_because', 'pre-existing prose entry')"
    )
    conn.commit()
    return conn


def index_file(conn, source_path: str, store_type="memory", chunk_id_prefix="memory:test"):
    stem = Path(source_path).stem
    conn.execute(
        "INSERT INTO chunks (chunk_id, store_type, source_path, title, content, indexed_at) "
        "VALUES (?, ?, ?, ?, 'x', datetime('now'))",
        (f"{chunk_id_prefix}:{stem}:0", store_type, source_path, stem),
    )
    conn.commit()
    return conn.execute("SELECT id FROM chunks WHERE source_path=?", (source_path,)).fetchone()[0]


def run_scenarios(check):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        memory_dir = td / "memory"
        memory_dir.mkdir()
        db_path = td / "test.db"

        # a: exact-stem match
        (memory_dir / "feedback_source_a.md").write_text(
            "---\nname: some prose title\n---\nSee [[feedback_target_a]].\n"
        )
        (memory_dir / "feedback_target_a.md").write_text("---\nname: target a\n---\nBody.\n")

        # b: prefix-stripped alias
        (memory_dir / "feedback_prefer_small_prs.md").write_text(
            "---\nname: Prefer small pull requests\n---\nBody.\n"
        )
        (memory_dir / "feedback_source_b.md").write_text(
            "---\nname: prose\n---\nRelated: [[prefer-small-prs]].\n"
        )

        # c: name: alias
        (memory_dir / "user_profile.md").write_text(
            "---\nname: profile-details\n---\nBody.\n"
        )
        (memory_dir / "feedback_source_c.md").write_text(
            "---\nname: prose\n---\nSee [[profile-details]].\n"
        )

        # d: dangling (no matching file at all)
        (memory_dir / "feedback_source_d.md").write_text(
            "---\nname: prose\n---\nNot yet written: [[future-topic-nobody-made]].\n"
        )

        # e: genuine ambiguity -- two distinct files share a stripped alias
        (memory_dir / "feedback_dup_x.md").write_text("---\nname: dup one\n---\nBody.\n")
        (memory_dir / "reference_dup_x.md").write_text("---\nname: dup two\n---\nBody.\n")
        (memory_dir / "feedback_source_e.md").write_text(
            "---\nname: prose\n---\nAmbiguous: [[dup_x]].\n"
        )

        # h: self-reference (a file linking to its own slug)
        (memory_dir / "feedback_source_h.md").write_text(
            "---\nname: prose\n---\nSee also [[feedback_source_h]] above.\n"
        )

        # i: source file has wikilinks but is deliberately left UNINDEXED
        # (no chunks row) -- simulates a freshly-written file not yet
        # picked up by the indexer.
        (memory_dir / "feedback_source_i_unindexed.md").write_text(
            "---\nname: prose\n---\nSee [[feedback_target_a]] and [[also-not-real]].\n"
        )

        # l: a file physically indexed TWICE under two distinct
        # chunks.source_path rows -- reproduces the real symlinked-dir vs
        # resolved-dir split via an actual filesystem symlink, so
        # Path.resolve() genuinely agrees the two rows are one file.
        (memory_dir / "feedback_dup_indexed.md").write_text(
            "---\nname: prose\n---\nLinks out to [[feedback_target_a]].\n"
        )
        alt_dir = td / "alt_memory_path"
        alt_dir.mkdir()
        alt_path = alt_dir / "feedback_dup_indexed.md"
        alt_path.symlink_to(memory_dir / "feedback_dup_indexed.md")
        (memory_dir / "feedback_source_l.md").write_text(
            "---\nname: prose\n---\nSee [[feedback_dup_indexed]] for detail.\n"
        )

        # m: pipe-alias form [[slug|Display Text]] resolves on slug before pipe
        (memory_dir / "project_first_idea.md").write_text(
            "---\nname: First idea\n---\nBody.\n"
        )
        (memory_dir / "feedback_source_m.md").write_text(
            "---\nname: prose\n---\nSee [[project_first_idea|Anamnesis memory]].\n"
        )

        # n: .md-suffixed slug form [[slug.md]] strips .md and resolves
        (memory_dir / "feedback_source_n.md").write_text(
            "---\nname: prose\n---\nSee [[feedback_target_a.md]].\n"
        )

        # o: spaced non-slug prose forms land in ignored_non_slug bucket
        (memory_dir / "feedback_source_o.md").write_text(
            "---\nname: prose\n---\nMentions [[skill some-tool]] and [[the onboarding upload funnel]].\n"
        )

        conn = make_db(db_path)
        for f in memory_dir.glob("*.md"):
            if f.stem == "feedback_source_i_unindexed":
                continue  # deliberately not indexed, for check (i)
            index_file(conn, str(f))
        # index the SAME physical file a second time under its alt-dir path
        # (distinct chunk_id prefix -- chunk_id is UNIQUE, and this is a
        # different row/source_path for the same on-disk file, exactly
        # reproducing the real symlinked-vs-resolved double-index)
        index_file(conn, str(alt_path), chunk_id_prefix="memory:test-alt")

        mtimes_before = {f: f.stat().st_mtime for f in memory_dir.glob("*.md")}

        result = wlr.classify(conn, memory_dir)

        resolved_by_slug = {(r["from_file"], r["raw_slug"]): r for r in result["resolved"]}

        check(
            "a: exact-stem resolves",
            resolved_by_slug.get(("feedback_source_a", "feedback_target_a"), {}).get("to_file")
            == "feedback_target_a",
        )
        check(
            "b: prefix-stripped alias resolves",
            resolved_by_slug.get(("feedback_source_b", "prefer-small-prs"), {}).get("to_file")
            == "feedback_prefer_small_prs",
        )
        check(
            "c: name: frontmatter alias resolves",
            resolved_by_slug.get(("feedback_source_c", "profile-details"), {}).get("to_file")
            == "user_profile",
        )
        check(
            "m: pipe-alias resolves to target slug",
            resolved_by_slug.get(("feedback_source_m", "project_first_idea|Anamnesis memory"), {}).get("to_file")
            == "project_first_idea",
        )
        check(
            "n: .md suffix resolves to target slug",
            resolved_by_slug.get(("feedback_source_n", "feedback_target_a.md"), {}).get("to_file")
            == "feedback_target_a",
        )

        dangling_slugs = {(r["from_file"], r["raw_slug"]) for r in result["dangling"]}
        check(
            "d: no-match slug reported dangling, not resolved",
            ("feedback_source_d", "future-topic-nobody-made") in dangling_slugs
            and ("feedback_source_d", "future-topic-nobody-made") not in resolved_by_slug,
        )

        ambiguous_slugs = {(r["from_file"], r["raw_slug"]) for r in result["ambiguous"]}
        check(
            "e: colliding alias reported ambiguous, never resolved",
            ("feedback_source_e", "dup_x") in ambiguous_slugs
            and ("feedback_source_e", "dup_x") not in resolved_by_slug,
        )

        check(
            "only 'references' would be inserted (broke_because/fixed_by untouched pre-apply)",
            conn.execute("SELECT COUNT(*) FROM nugget_links WHERE relation_type != 'broke_because'").fetchone()[0]
            == 0,
        )

        # f: apply + idempotency
        inserted1, skipped1 = wlr.apply_edges(conn, result["resolved_deduped"])
        check("f: first apply inserts >0 edges", inserted1 > 0, f"inserted={inserted1}")
        check(
            "f: first apply skips 0 existing",
            skipped1 == 0,
        )
        result2 = wlr.classify(conn, memory_dir)
        inserted2, skipped2 = wlr.apply_edges(conn, result2["resolved_deduped"])
        check("f: second apply inserts 0 new edges", inserted2 == 0, f"inserted={inserted2}")
        check(
            "f: second apply skips all as already-existing",
            skipped2 == inserted1,
            f"skipped2={skipped2} inserted1={inserted1}",
        )

        ref_count = conn.execute("SELECT COUNT(*) FROM nugget_links WHERE relation_type='references'").fetchone()[0]
        check("f: references count matches inserted1", ref_count == inserted1)

        causal_count = conn.execute(
            "SELECT COUNT(*) FROM nugget_links WHERE relation_type IN ('broke_because','fixed_by')"
        ).fetchone()[0]
        check("g: broke_because/fixed_by rows untouched", causal_count == 1, f"count={causal_count}")

        mtimes_after = {f: f.stat().st_mtime for f in memory_dir.glob("*.md")}
        check("g: no markdown mtimes changed", mtimes_before == mtimes_after)

        # h: self-reference bucketed separately from dangling
        self_ref_slugs = {(r["from_file"], r["raw_slug"]) for r in result["self_references"]}
        check(
            "h: self-reference goes in self_references, not dangling",
            ("feedback_source_h", "feedback_source_h") in self_ref_slugs
            and ("feedback_source_h", "feedback_source_h") not in dangling_slugs,
        )

        # i: unindexed source file's occurrences are individually accounted
        # for, not silently dropped
        unindexed_slugs = {(r["from_file"], r["raw_slug"]) for r in result["skipped_unindexed_source"]}
        check(
            "i: unindexed source's wikilinks land in skipped_unindexed_source",
            ("feedback_source_i_unindexed", "feedback_target_a") in unindexed_slugs
            and ("feedback_source_i_unindexed", "also-not-real") in unindexed_slugs,
        )
        check(
            "i: unindexed source's wikilinks are NOT silently resolved/dangling",
            ("feedback_source_i_unindexed", "feedback_target_a") not in resolved_by_slug
            and ("feedback_source_i_unindexed", "also-not-real") not in dangling_slugs,
        )

        # o: spaced non-slug prose forms land in ignored_non_slug
        ignored_slugs = {(r["from_file"], r["raw_slug"]) for r in result["ignored_non_slug"]}
        check(
            "o: spaced non-slug prose lands in ignored_non_slug bucket",
            ("feedback_source_o", "skill some-tool") in ignored_slugs
            and ("feedback_source_o", "the onboarding upload funnel") in ignored_slugs,
        )

        # l: double-indexed file resolves normally in BOTH roles, tagged
        # with the real cause instead of a false "not indexed" reason
        as_target = resolved_by_slug.get(("feedback_source_l", "feedback_dup_indexed"))
        check(
            "l: link TO a double-indexed file resolves (not skipped_target_not_indexed)",
            as_target is not None and as_target.get("to_file") == "feedback_dup_indexed",
        )
        check(
            "l: resolved-as-target record carries duplicate_target, names dedupe (not reindex)",
            as_target is not None
            and "duplicate_target" in as_target
            and "dedupe" in as_target["duplicate_target"]
            and "not indexed" not in as_target["duplicate_target"],
        )
        as_source = resolved_by_slug.get(("feedback_dup_indexed", "feedback_target_a"))
        check(
            "l: link FROM a double-indexed file resolves (not skipped_unindexed_source)",
            as_source is not None and as_source.get("to_file") == "feedback_target_a",
        )
        check(
            "l: resolved-as-source record carries duplicate_source, names dedupe (not reindex)",
            as_source is not None
            and "duplicate_source" in as_source
            and "dedupe" in as_source["duplicate_source"]
            and "not indexed" not in as_source["duplicate_source"],
        )
        check(
            "l: duplicate_indexed_files reports the file with 2 source_path variants",
            result["duplicate_indexed_files"].get("feedback_dup_indexed", {}).get("source_path_variants")
            and len(result["duplicate_indexed_files"]["feedback_dup_indexed"]["source_path_variants"]) == 2,
        )
        no_false_unindexed = not any(
            r["from_file"] == "feedback_dup_indexed" for r in result["skipped_unindexed_source"]
        )
        no_false_target_unindexed = not any(
            r.get("from_file") == "feedback_source_l" and r["raw_slug"] == "feedback_dup_indexed"
            for r in result["skipped_target_not_indexed"]
        )
        check(
            "l: double-indexed file never appears in either false-negative bucket",
            no_false_unindexed and no_false_target_unindexed,
        )

        # k: bucket accounting is exhaustive -- sum of every bucket equals
        # the raw regex match count across every file in memory_dir
        raw_occurrence_count = sum(
            len(re.findall(r"\[\[[^\]\n]+\]\]", f.read_text())) for f in memory_dir.glob("*.md")
        )
        bucket_sum = (
            len(result["resolved"])
            + len(result["ambiguous"])
            + len(result["dangling"])
            + len(result["self_references"])
            + len(result["ignored_non_slug"])
            + len(result["skipped_target_not_indexed"])
            + len(result["skipped_unindexed_source"])
        )
        check(
            "k: bucket sum equals raw wikilink occurrence count (nothing dropped)",
            bucket_sum == raw_occurrence_count,
            f"bucket_sum={bucket_sum} raw={raw_occurrence_count}",
        )

        conn.close()

        # j: concurrent apply -- two separate connections to the same
        # on-disk DB race to insert an overlapping edge set. The unique
        # index + BEGIN IMMEDIATE must guarantee exactly one references
        # row per (from,to) pair regardless of interleaving.
        conn_probe = sqlite3.connect(str(db_path))
        conn_probe.execute("DELETE FROM nugget_links WHERE relation_type='references'")
        conn_probe.commit()
        conn_probe.close()

        # check_same_thread=False: these connections are deliberately used
        # from a background thread below, to race two apply_edges calls
        # against each other. busy_timeout gives the loser of the
        # BEGIN IMMEDIATE race a chance to wait-then-retry instead of
        # failing instantly, so the test exercises "serializes correctly"
        # rather than just "the loser errors out".
        conn_a = wlr.sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
        conn_b = wlr.sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
        conn_a.execute("PRAGMA busy_timeout=2000")
        conn_b.execute("PRAGMA busy_timeout=2000")
        result_for_race = wlr.classify(conn_a, memory_dir)
        edges = result_for_race["resolved_deduped"]

        race_errors = []
        race_results = {}

        def run_apply(conn, key):
            try:
                race_results[key] = wlr.apply_edges(conn, edges)
            except Exception as exc:  # sqlite3.OperationalError on lock contention is acceptable
                race_errors.append((key, exc))

        t1 = threading.Thread(target=run_apply, args=(conn_a, "a"))
        t2 = threading.Thread(target=run_apply, args=(conn_b, "b"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        conn_a.close()
        conn_b.close()

        conn_check = sqlite3.connect(str(db_path))
        for from_id, to_id in {(e["from_chunk_id"], e["to_chunk_id"]) for e in edges}:
            n = conn_check.execute(
                "SELECT COUNT(*) FROM nugget_links WHERE from_chunk_id=? AND to_chunk_id=? AND relation_type='references'",
                (from_id, to_id),
            ).fetchone()[0]
            check(f"j: no double-insert for edge ({from_id}->{to_id}) under concurrent apply", n == 1, f"n={n}")
        idx_exists = conn_check.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_nugget_links_references_unique'"
        ).fetchone()
        check("j: structural unique index was created", idx_exists is not None)
        conn_check.close()


class TestWikiLinkResolver(unittest.TestCase):
    def test_all_scenarios(self):
        def check(name, cond, detail=""):
            with self.subTest(name):
                self.assertTrue(cond, detail)

        run_scenarios(check)


if __name__ == "__main__":
    unittest.main()
