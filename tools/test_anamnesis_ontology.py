#!/usr/bin/env python3
"""Tests for the ontology layer (coordinates, compiler, queries, evidence), the dictionary and link validity.

Self-contained and fictional: every test builds its own dictionary, registries and
in-memory SQLite database. No network, no API keys, stdlib only.

    python -m unittest discover -s tools -p 'test_*.py' -v
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import anamnesis_dict as adict  # noqa: E402
import anamnesis_ontology as ontology  # noqa: E402
import anamnesis_validity as validity  # noqa: E402

ENV_KEYS = ("ANAMNESIS_DICT_PATH", "ANAMNESIS_REPO_ROOT", "ANAMNESIS_HOME", "ANAMNESIS_DATA_DIR",
            "ANAMNESIS_DB", "ANAMNESIS_PATH_ROOTS")

FIXTURE_DICTIONARY = {
    "version": "test",
    "projects": [
        {"name": "alpha-app", "aliases": ["alpha"], "description": "fictional app", "enabled": True},
        {"name": "beta-site", "aliases": ["beta"], "description": "fictional site", "enabled": True},
    ],
    "store_types": {
        "decision": {"weight": 1.5, "resolve_target": True, "evidence_source": True},
        "solution": {"weight": 1.2, "resolve_target": True, "evidence_source": True},
        "mistake": {"weight": 1.3, "resolve_target": True, "evidence_source": True},
        "memory": {"weight": 1.0, "evidence_source": True},
    },
    "ontology": {
        "node_kinds": ["Project", "Service", "Machine", "Capability", "Decision", "Policy",
                       "Person", "Incident", "Topic"],
        "predicates": ["RUNS_ON", "HOSTS_SERVICE", "DEPENDS_ON", "USES_CAPABILITY",
                       "PROVIDES_CAPABILITY", "FALLBACK_TO", "REPLICATES_TO", "GOVERNS_SCOPE",
                       "DECIDED_FOR", "SUPERSEDES", "ENFORCED_BY", "OBSERVED_IN",
                       "CONTACT_FOR_PROJECT"],
        "polysemy_blocklist": ["tidy", "work"],
        "exclude_path_markers": ["/mirrors/"],
    },
    "dev_root_candidates": ["code", "src"],
    "canonical_default_project": "alpha-app",
    "registries": {
        "rules_files": ["RULES.md"],
        "decision_logs": ["decisions.md"],
        "capabilities": ["capabilities.json"],
        "capability_edges": ["edges.json"],
        "contacts": ["contacts.json"],
        "entities": ["entities.json"],
        "enforcement": {"dirs": ["tests"], "extensions": [".py"], "telemetry": ["firings.jsonl"]},
    },
}

ENTITIES = {
    "entities": [
        {"id": "machine/host-one", "kind": "Machine", "name": "host-one", "aliases": ["h1"]},
        {"id": "service/web", "kind": "Service", "name": "Web service", "aliases": ["web"]},
        {"id": "service/web-standby", "kind": "Service", "name": "Web standby"},
        {"id": "service/batch", "kind": "Service", "name": "Batch runner"},
        {"id": "incident/2026-02-01-web-outage", "kind": "Incident", "name": "web outage",
         "aliases": ["web-outage"]},
        {"id": "service/bogus", "kind": "NotAKind", "name": "bad entity"},
    ],
    "relations": [
        {"subject": "service/web", "predicate": "RUNS_ON", "object": "machine/host-one"},
        {"subject": "service/batch", "predicate": "RUNS_ON", "object": "machine/host-one"},
        {"subject": "service/web", "predicate": "FALLBACK_TO", "object": "service/web-standby"},
        {"subject": "project/alpha-app", "predicate": "DEPENDS_ON", "object": "service/web"},
        {"subject": "incident/2026-02-01-web-outage", "predicate": "DECIDED_FOR", "object": "service/web"},
        {"subject": "service/web", "predicate": "NOT_A_PREDICATE", "object": "machine/host-one"},
    ],
}

RULES_MD = """# Rules

## Reviews required
<!-- policy: id=review-required scope=project/alpha-app enforced_by=required_reviews -->
Every change lands through a reviewed pull request.

## Keep notes short
<!-- policy: id=short-notes scope=project/beta-site -->
Notes should stay under a page.
"""

DECISIONS_MD = """# Decisions

### 2026-01-01 — use sqlite for storage
**Topic:** storage
**Scope:** alpha-app
**Origin:** human
**Decision:** sqlite.

### 2026-03-01 — switch storage to postgres
**Topic:** storage
**Supersedes:** 2026-01-01
**Origin:** human

### 2026-04-01 — pick a queue
**Topic:** messaging
**Origin:** human

### 2026-04-02 — pick another queue
**Topic:** messaging
**Origin:** agent_inferred
"""

SCHEMA = """
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT, chunk_id TEXT UNIQUE, store_type TEXT NOT NULL,
    source_path TEXT NOT NULL, title TEXT, content TEXT NOT NULL, project TEXT,
    indexed_at TEXT NOT NULL, retirement_status TEXT DEFAULT 'active', metadata TEXT DEFAULT '');
CREATE TABLE nugget_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT, from_chunk_id INTEGER NOT NULL, to_chunk_id INTEGER,
    relation_type TEXT NOT NULL, confidence REAL DEFAULT 1.0, source_text TEXT, created_at TEXT);
"""


class OntologyCase(unittest.TestCase):
    """Isolated repo root + dictionary + database per test."""

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ENV_KEYS}
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "tests").mkdir()
        (self.root / "dictionary.json").write_text(json.dumps(FIXTURE_DICTIONARY), encoding="utf-8")
        (self.root / "entities.json").write_text(json.dumps(ENTITIES), encoding="utf-8")
        (self.root / "RULES.md").write_text(RULES_MD, encoding="utf-8")
        (self.root / "decisions.md").write_text(DECISIONS_MD, encoding="utf-8")
        (self.root / "tests" / "test_gate.py").write_text("# checks required_reviews\n", encoding="utf-8")
        os.environ["ANAMNESIS_DICT_PATH"] = str(self.root / "dictionary.json")
        os.environ["ANAMNESIS_REPO_ROOT"] = str(self.root)
        os.environ["ANAMNESIS_HOME"] = str(self.root / "home")
        os.environ.pop("ANAMNESIS_PATH_ROOTS", None)
        adict.reset_cache()
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        validity.ensure_validity_schema(self.conn)
        ontology.ensure_ontology_schema(self.conn)

    def tearDown(self):
        self.conn.close()
        self._tmp.cleanup()
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        adict.reset_cache()

    def add_chunk(self, cid, store, path, title, content, project="alpha-app", indexed="2026-05-01T00:00:00Z"):
        self.conn.execute(
            "INSERT INTO chunks (id, chunk_id, store_type, source_path, title, content, project, indexed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (cid[0], cid[1], store, path, title, content, project, indexed),
        )
        self.conn.commit()

    def compile(self, **kw):
        return ontology.compile_ontology(self.conn, repo_root=self.root, **kw)


class TestDictionary(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ENV_KEYS}

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        adict.reset_cache()

    def test_shipped_dictionary_is_valid_and_complete(self):
        for k in ("ANAMNESIS_DICT_PATH", "ANAMNESIS_DATA_DIR"):
            os.environ.pop(k, None)
        adict.reset_cache()
        d = adict.load_dictionary(strict=True)
        self.assertIn("projects", d)
        self.assertIn("registries", d)
        weights = adict.get_store_weights()
        self.assertEqual(weights["decision"], 1.5)
        self.assertTrue(all(w > 0 for w in weights.values()))
        kinds = adict.get_ontology_node_kinds()
        self.assertNotIn("Concept", kinds, "amorphous prose buckets stay out of the ontology")
        self.assertTrue({"RUNS_ON", "DEPENDS_ON", "SUPERSEDES"} <= adict.get_ontology_predicates())
        self.assertIn("example-app", adict.get_all_projects())
        self.assertNotIn("legacy-importer", adict.get_all_projects())
        self.assertIn("legacy-importer", adict.get_all_projects(enabled_only=False))

    def test_shipped_dictionary_compiles_cleanly(self):
        """The shipped example must compile with zero warnings against an empty database."""
        for k in ("ANAMNESIS_DICT_PATH", "ANAMNESIS_DATA_DIR", "ANAMNESIS_REPO_ROOT"):
            os.environ.pop(k, None)
        adict.reset_cache()
        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA)
        stats = ontology.compile_ontology(conn)
        conn.close()
        self.assertEqual(stats["warnings"], 0, ontology.LAST_COMPILE_WARNINGS)
        self.assertGreater(stats["nodes"], 5)

    def test_missing_dictionary_falls_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as td:
            os.environ["ANAMNESIS_DICT_PATH"] = str(Path(td) / "nope.json")
            adict.reset_cache()
            self.assertEqual(adict.get_all_projects(), [])
            self.assertIn("Project", adict.get_ontology_node_kinds())
            with self.assertRaises(FileNotFoundError):
                adict.load_dictionary(strict=True)

    def test_home_and_db_resolution_follow_env(self):
        os.environ["ANAMNESIS_HOME"] = "/tmp/example-home"
        os.environ.pop("ANAMNESIS_DB", None)
        self.assertEqual(adict.db_path(), Path("/tmp/example-home/memory.db"))
        os.environ["ANAMNESIS_DB"] = "/tmp/other.db"
        self.assertEqual(adict.db_path(), Path("/tmp/other.db"))


class TestGuardrails(OntologyCase):
    def test_only_the_compiler_writes_ontology_tables(self):
        """Bootstrap-loop guard: no other tool may INSERT/UPDATE/DELETE ontology_* rows."""
        write_re = re.compile(
            r"\b(?:INSERT\s+(?:OR\s+\w+\s+)?INTO|UPDATE|DELETE\s+FROM)\s+ontology_(?:nodes|claims|evidence)\b",
            re.IGNORECASE,
        )
        offenders = [
            f.name for f in sorted(SCRIPT_DIR.glob("*.py"))
            if not f.name.startswith("test_") and f.name != "anamnesis_ontology.py"
            and write_re.search(f.read_text(encoding="utf-8", errors="replace"))
        ]
        self.assertEqual(offenders, [])

    def test_no_check_constraints_in_projection_tables(self):
        for tbl in ("ontology_nodes", "ontology_claims", "ontology_evidence"):
            ddl = self.conn.execute("SELECT sql FROM sqlite_master WHERE name=?", (tbl,)).fetchone()[0]
            self.assertNotIn("CHECK", ddl.upper())


class TestCanonicalPaths(OntologyCase):
    def test_same_file_on_two_machines_gets_one_chunk_key(self):
        a = ontology.compute_chunk_key("/srv/alice/code/alpha-app/docs/guide.md", "Setup")
        b = ontology.compute_chunk_key("/mnt/bob/src/alpha-app/docs/guide.md", "Setup")
        c = ontology.compute_chunk_key("alpha-app/docs/guide.md", "Setup")
        self.assertEqual(a, b)
        self.assertEqual(a, c)

    def test_roots_are_configurable_not_hardcoded(self):
        raw = "/srv/alice/workspace/alpha-app/notes.md"
        self.assertEqual(ontology.canonicalize_source_path(raw), raw)  # unknown root: untouched
        os.environ["ANAMNESIS_PATH_ROOTS"] = "workspace"
        adict.reset_cache()
        self.assertEqual(ontology.canonicalize_source_path(raw), "alpha-app/notes.md")

    def test_default_project_prefix_only_when_configured(self):
        self.assertEqual(ontology.canonicalize_source_path("docs/a.md"), "alpha-app/docs/a.md")
        d = dict(FIXTURE_DICTIONARY, canonical_default_project="")
        (self.root / "dictionary.json").write_text(json.dumps(d), encoding="utf-8")
        adict.reset_cache()
        self.assertEqual(ontology.canonicalize_source_path("docs/a.md"), "docs/a.md")


class TestSupersession(OntologyCase):
    def test_sibling_chunks_never_supersede_each_other(self):
        ts = "2026-10-19T12:00:00Z"
        self.add_chunk((1, "doc:1"), "project_doc", "docs/arch.md", "System Overview", "Part 1", indexed=ts)
        self.add_chunk((2, "doc:2"), "project_doc", "docs/arch.md", "System Overview", "Part 2", indexed=ts)
        self.add_chunk((3, "dec:1"), "decision", "docs/log.md", "Storage Engine Policy",
                       "**Date:** 2026-05-01\nOld policy", indexed=ts)
        self.add_chunk((4, "dec:2"), "decision", "docs/log.md", "Storage Engine Policy",
                       "**Date:** 2026-08-11\nNew policy", indexed=ts)
        res = validity.compute_supersessions(self.conn)
        self.assertEqual(res["sibling_collisions_skipped"], 1)
        self.assertEqual(len(res["verdicts"]), 1)
        self.assertEqual(res["verdicts"][0]["superseded_chunk_id"], 3)
        self.assertEqual(res["verdicts"][0]["superseding_chunk_id"], 4)


class TestCompiler(OntologyCase):
    def test_registry_only_compile_works_without_a_chunks_table(self):
        conn = sqlite3.connect(":memory:")
        stats = ontology.compile_ontology(conn, repo_root=self.root, link_evidence=False)
        self.assertGreater(stats["nodes"], 0)

    def test_registries_compile_and_bad_entries_warn(self):
        stats = self.compile(link_evidence=False)
        self.assertGreater(stats["nodes"], 8)
        self.assertEqual(stats["warnings"], 2)  # bad kind + bad predicate
        text = " ".join(ontology.LAST_COMPILE_WARNINGS)
        self.assertIn("NotAKind", text)
        self.assertIn("NOT_A_PREDICATE", text)
        ids = {r[0] for r in self.conn.execute("SELECT node_id FROM ontology_nodes")}
        self.assertIn("project/alpha-app", ids)
        self.assertIn("service/web", ids)
        self.assertNotIn("service/bogus", ids)

    def test_compile_is_idempotent_and_rebuildable(self):
        first = self.compile(link_evidence=False)
        second = self.compile(link_evidence=False)
        self.assertEqual(first, second)

    def test_capabilities_edges_and_contacts(self):
        (self.root / "capabilities.json").write_text(json.dumps({"projects": {
            "alpha-app": {"capabilities": [{"capability_id": "search", "description": "full text"}]}}}))
        (self.root / "edges.json").write_text(json.dumps({"edges": [
            {"consumer": "beta-site", "provider": "alpha-app", "capability_id": "search",
             "status": "active"},
            {"consumer": "all", "provider": "alpha-app", "capability_id": "search", "status": "active"}]}))
        (self.root / "contacts.json").write_text(json.dumps({"contacts": [
            {"name": "Pat Example", "role": "owner", "projects": ["alpha-app"]}]}))
        self.compile(link_evidence=False)
        preds = {(r["subject_id"], r["predicate"], r["object_id_or_literal"]) for r in
                 self.conn.execute("SELECT * FROM ontology_claims")}
        self.assertIn(("project/alpha-app", "PROVIDES_CAPABILITY", "capability/search"), preds)
        self.assertIn(("project/beta-site", "DEPENDS_ON", "project/alpha-app"), preds)
        self.assertIn(("person/pat-example", "CONTACT_FOR_PROJECT", "project/alpha-app"), preds)
        # "all" never makes a project depend on itself
        self.assertNotIn(("project/alpha-app", "DEPENDS_ON", "project/alpha-app"), preds)


class TestImpact(OntologyCase):
    def test_failure_blast_radius_fallbacks_spofs_and_incidents(self):
        self.add_chunk((10, "mistake:outage"), "mistake", "notes/failures.md", "web outage on host-one",
                       "host-one lost power and the web service went down", project="alpha-app")
        ontology.backfill_chunk_coordinates(self.conn, only_missing=False)
        self.compile()

        res = ontology.query_ontology_impact(self.conn, "h1", mode="failure", depth=2)
        self.assertEqual(res["entity"]["node_id"], "machine/host-one")
        direct = {d["affected_id"] for d in res["direct_impact"]}
        self.assertEqual(direct, {"service/web", "service/batch"})
        self.assertIn("project/alpha-app", {t["affected_id"] for t in res["transitive_impact"]})
        spofs = {s["node_id"] for s in res["single_points_of_failure"]}
        self.assertIn("service/batch", spofs, "no FALLBACK_TO claim -> single point of failure")
        self.assertNotIn("service/web", spofs)
        self.assertIn("service/web-standby", {m["target_id"] for m in res["mitigations_and_fallbacks"]})
        self.assertGreaterEqual(len(res["historical_incidents"]), 1)

    def test_dependencies_and_inspect_modes(self):
        self.compile(link_evidence=False)
        deps = ontology.query_ontology_impact(self.conn, "alpha", mode="dependencies", depth=3)
        self.assertEqual(deps["entity"]["node_id"], "project/alpha-app")
        targets = {d["to_id"] for d in deps["dependencies"]}
        self.assertTrue({"service/web", "machine/host-one"} <= targets)
        ins = ontology.query_ontology_impact(self.conn, "service/web", mode="inspect")
        self.assertTrue(ins["outbound_claims"] and ins["inbound_claims"])
        self.assertIn("error", ontology.query_ontology_impact(self.conn, "nothing-here"))


class TestAudit(OntologyCase):
    def test_enforced_vs_paper_rules_and_supersession_lineage(self):
        self.compile(link_evidence=False)
        audit = ontology.query_ontology_audit(self.conn, scope="all")
        enforced = {p["policy_id"] for p in audit["enforced_policies"]}
        paper = {p["policy_id"] for p in audit["paper_rules"]}
        self.assertEqual(enforced, {"policy/review-required"})
        self.assertEqual(paper, {"policy/short-notes"})
        old = self.conn.execute(
            "SELECT governance_status, valid_to FROM ontology_claims "
            "WHERE subject_id='topic/storage' AND governance_status='superseded'").fetchone()
        self.assertIsNotNone(old)
        self.assertEqual(old["valid_to"], "2026-03-01")
        self.assertTrue(audit["superseded_lineage"])

    def test_contradictions_flag_unsuperseded_topic_collisions_and_attribution(self):
        self.compile(link_evidence=False)
        found = ontology.query_ontology_audit(self.conn, scope="contradictions")["contradictions"]
        self.assertEqual({c["subject_id"] for c in found}, {"topic/messaging"},
                         "superseded storage decisions must not collide")
        self.assertTrue(found[0]["attribution_violation"], "human vs agent_inferred is flagged")

    def test_telemetry_makes_a_rule_observed(self):
        (self.root / "firings.jsonl").write_text(
            '{"rule": "short-notes", "ts": "2026-06-01T00:00:00Z"}\nnot json\n', encoding="utf-8")
        self.compile(link_evidence=False)
        audit = ontology.query_ontology_audit(self.conn, scope="unverified")
        by_id = {p["policy_id"]: p for p in audit["enforced_policies"]}
        self.assertEqual(by_id["policy/short-notes"]["verification_status"], "observed")

    def test_link_health_reports_dangling_foreign_keys(self):
        self.add_chunk((1, "a:0"), "memory", "n/a.md", "A", "alpha text")
        self.conn.execute("INSERT INTO nugget_links (from_chunk_id, to_chunk_id, relation_type, source_text) "
                          "VALUES (1, 424242, 'fixed_by', 'x')")
        self.conn.commit()
        self.compile(link_evidence=False)
        health = ontology.query_ontology_audit(self.conn, scope="link_health")["nugget_links_health"]
        self.assertEqual(health["dangling_fk"], 1)


class TestEvidenceLinker(OntologyCase):
    def test_co_mentions_are_evidence_only_never_claims(self):
        self.add_chunk((1, "decision:recap"), "decision", "notes/recap.md", "Weekly recap",
                       "alpha-app and beta-site were discussed together; we agreed to tidy early.",
                       project="")
        ontology.backfill_chunk_coordinates(self.conn, only_missing=False)
        self.compile(link_evidence=True)
        roles = {r[0]: r[1] for r in self.conn.execute("SELECT node_id, relation_role FROM ontology_evidence")}
        self.assertEqual(roles.get("project/alpha-app"), "co_mentioned_in_chunk")
        self.assertEqual(roles.get("project/beta-site"), "co_mentioned_in_chunk")
        deps = ontology.query_ontology_impact(self.conn, "project/beta-site", mode="dependencies")
        self.assertNotIn("project/alpha-app", {d["to_id"] for d in deps["dependencies"]})

    def test_polysemy_blocklist_words_do_not_match_bare(self):
        (self.root / "capabilities.json").write_text(json.dumps({"projects": {
            "tidy": {"capabilities": []}}}))
        self.add_chunk((1, "m:0"), "memory", "notes/x.md", "Misc", "we should tidy the loop on this", project="")
        self.compile(link_evidence=True)
        nodes = {r[0] for r in self.conn.execute("SELECT node_id FROM ontology_evidence")}
        self.assertNotIn("project/tidy", nodes)

    def test_ambient_context_and_query_entities(self):
        self.compile(link_evidence=False)
        ctx = ontology.get_ambient_project_context(self.conn, "alpha-app")
        self.assertEqual(ctx["node_id"], "project/alpha-app")
        self.assertTrue(ctx["outbound_claims"])
        ents = ontology.get_ambient_query_entities(self.conn, "what runs on host-one?")
        self.assertEqual(ents[0]["node_id"], "machine/host-one")
        self.assertIsNone(ontology.get_ambient_project_context(self.conn, "unknown-project"))


class TestLinkSurvival(OntologyCase):
    def test_reindex_with_new_top_section_orphans_no_links(self):
        body = "Always prefer the GNU sync binary over the bundled one to prevent stalled delta transfers."
        self.add_chunk((10, "solution:abc:0"), "solution", "/srv/alice/code/alpha-app/docs/solutions/sync.md",
                       "Use the GNU sync tool for delta transfers", body)
        self.add_chunk((1, "extracts:1"), "session_extracts", "docs/sessions/f1.json",
                       "Failed approach: bundled sync stalled", "the bundled sync hung")
        self.conn.execute(
            "INSERT INTO nugget_links (id, from_chunk_id, to_chunk_id, relation_type, confidence, source_text) "
            "VALUES (100, 1, 999999, 'fixed_by', 0.95, "
            "'Prefer the GNU sync binary over the bundled one for delta transfers')")
        self.conn.commit()

        stats = ontology.repair_nugget_links(self.conn)
        self.assertEqual(stats["resolved"], 1)
        row = self.conn.execute("SELECT * FROM nugget_links WHERE id=100").fetchone()
        self.assertEqual(row["to_chunk_id"], 10)
        saved_key = row["to_chunk_key"]

        # Destructive re-index on another machine: new rowids, a new section on top.
        self.conn.execute("DELETE FROM chunks WHERE id = 10")
        self.add_chunk((50, "solution:abc:0"), "solution", "/mnt/bob/src/alpha-app/docs/solutions/sync.md",
                       "Introduction added at top", "Intro text")
        self.add_chunk((51, "solution:abc:1"), "solution", "/mnt/bob/src/alpha-app/docs/solutions/sync.md",
                       "Use the GNU sync tool for delta transfers", body)
        counts = ontology.sync_nugget_links_after_reindex(self.conn)
        self.assertEqual(counts["resolved"], 1)
        after = self.conn.execute("SELECT * FROM nugget_links WHERE id=100").fetchone()
        self.assertEqual(after["to_chunk_id"], 51)
        self.assertEqual(after["to_chunk_key"], saved_key)
        self.assertEqual(after["resolution_status"], "resolved")

    def test_worktree_mirrors_never_become_targets(self):
        body = "Always prefer the GNU sync binary over the bundled one to prevent stalled delta transfers."
        self.add_chunk((1, "s:canon"), "solution", "alpha-app/docs/s.md", "Use GNU sync for delta transfers", body)
        self.add_chunk((2, "s:mirror"), "solution", "/srv/alice/code/x/mirrors/b1/alpha-app/docs/s.md",
                       "Use GNU sync for delta transfers", body)
        self.add_chunk((3, "f:1"), "session_extracts", "f.json", "Failed", "hung")
        self.conn.execute(
            "INSERT INTO nugget_links (id, from_chunk_id, to_chunk_id, relation_type, source_text) "
            "VALUES (1, 3, NULL, 'fixed_by', "
            "'Prefer the GNU sync binary over the bundled one for delta transfers')")
        self.conn.commit()
        ontology.repair_nugget_links(self.conn)
        row = self.conn.execute("SELECT to_chunk_id, resolution_status FROM nugget_links WHERE id=1").fetchone()
        self.assertEqual((row["to_chunk_id"], row["resolution_status"]), (1, "resolved"))


class TestFreshDatabase(unittest.TestCase):
    def test_sync_is_noop_without_nugget_links_table(self):
        conn = sqlite3.connect(":memory:")
        self.assertEqual(ontology.sync_nugget_links_after_reindex(conn),
                         {"resolved": 0, "ambiguous": 0, "unresolved": 0})


if __name__ == "__main__":
    unittest.main()
