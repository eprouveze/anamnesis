#!/usr/bin/env python3
"""Tests for anamnesis_validity: bi-temporal validity (shadow mode first).

Covers schema idempotency, shadow-mode no-mutation, supersession keying, the live
flag, the promotion gate and the apply/shadow CLI. Fictional data, temp files only.

    python -m unittest discover -s tools -p 'test_*.py' -v
"""
import contextlib
import io
import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anamnesis_validity as validity  # noqa: E402

ENV_KEYS = ("ANAMNESIS_HOME", "ANAMNESIS_DB", "ANAMNESIS_VALIDITY_SHADOW", "ANAMNESIS_VALIDITY_LIVE_FILE",
            "ANAMNESIS_VALIDITY_LIVE", "ANAMNESIS_VALIDITY_GATE_DAYS", "ANAMNESIS_VALIDITY_MIN_COVERAGE")

LINKS_DDL = ("CREATE TABLE nugget_links (id INTEGER PRIMARY KEY, from_chunk_id INTEGER, "
             "to_chunk_id INTEGER, relation_type TEXT, confidence REAL, source_text TEXT, created_at TEXT)")
CHUNKS_DDL = ("CREATE TABLE chunks (id INTEGER PRIMARY KEY, chunk_id TEXT, store_type TEXT, "
              "source_path TEXT, title TEXT, content TEXT, indexed_at TEXT, "
              "retirement_status TEXT DEFAULT 'active', metadata TEXT DEFAULT '')")


class ValidityCase(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ENV_KEYS}
        for k in ENV_KEYS:
            os.environ.pop(k, None)
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        os.environ["ANAMNESIS_HOME"] = str(self.tmp / "home")
        self.db = self.tmp / "memory.db"

    def tearDown(self):
        self._tmp.cleanup()
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def make_db(self, with_link=True, path=None):
        conn = sqlite3.connect(str(path or self.db))
        conn.row_factory = sqlite3.Row
        conn.executescript(CHUNKS_DDL + ";" + LINKS_DDL)
        conn.execute("INSERT INTO chunks (id, chunk_id, store_type, source_path, title, content, indexed_at) "
                     "VALUES (10,'decision:old','decision','decisions.md','Ship the feature now','old','2026-01-01T00:00:00')")
        conn.execute("INSERT INTO chunks (id, chunk_id, store_type, source_path, title, content, indexed_at) "
                     "VALUES (11,'decision:new','decision','decisions.md','Ship the feature now','new','2026-01-02T00:00:00')")
        if with_link:
            conn.execute("INSERT INTO nugget_links (id, from_chunk_id, to_chunk_id, relation_type) "
                         "VALUES (1, 10, 11, 'superseded_by')")
        conn.commit()
        return conn


class TestSchemaAndKeys(ValidityCase):
    def test_ensure_schema_is_idempotent_and_tolerant(self):
        conn = sqlite3.connect(":memory:")
        validity.ensure_validity_schema(conn)  # no table yet: no-op, no error
        conn.execute(LINKS_DDL)
        self.assertFalse(validity.has_validity_columns(conn))
        validity.ensure_validity_schema(conn)
        self.assertTrue(validity.has_validity_columns(conn))
        validity.ensure_validity_schema(conn)

    def test_session_extracts_and_noise_titles_are_not_keyable(self):
        self.assertIsNone(validity.supersession_key(
            {"store_type": "session_extracts", "source_path": "f.json", "title": "Tool pattern: Bash (useful)"}))
        for noise in ("unknown", "2026-01-01", "short"):
            self.assertIsNone(validity.supersession_key({"source_path": "a.md", "title": noise}), noise)
        self.assertEqual(validity.supersession_key({"source_path": "a.md", "title": "Real Identity Title"}),
                         "a_md|Real Identity Title")
        self.assertEqual(validity.supersession_key({"source_path": "a.md", "title": "x" * 12,
                                              "metadata": '{"id": "stable-1"}'}), "a_md|stable-1")

    def test_default_paths_live_under_anamnesis_home(self):
        self.assertEqual(validity.shadow_path(), self.tmp / "home" / "validity-shadow.json")
        self.assertEqual(validity.validity_live_flag_path(), self.tmp / "home" / ".validity-live")
        self.assertEqual(validity.memory_db_path(), self.tmp / "home" / "memory.db")


class TestShadowSafety(ValidityCase):
    def test_compute_excludes_session_extracts_and_detects_real_supersession(self):
        conn = self.make_db(with_link=False)
        conn.execute("INSERT INTO chunks (id, chunk_id, store_type, source_path, title, content, indexed_at) "
                     "VALUES (1,'session_extracts:t:s1:a','session_extracts','f.json','Tool pattern: Bash (useful)','a','2026-01-01T00:00:00')")
        conn.execute("INSERT INTO chunks (id, chunk_id, store_type, source_path, title, content, indexed_at) "
                     "VALUES (2,'session_extracts:t:s1:b','session_extracts','f.json','Tool pattern: Bash (useful)','b','2026-01-02T00:00:00')")
        r = validity.compute_supersessions(conn)
        keys = [v["key"] for v in r["verdicts"]]
        self.assertFalse(any("Tool pattern" in k for k in keys))
        self.assertTrue(any("Ship the feature now" in k for k in keys))
        self.assertGreater(r["key_coverage"], 0.0)

    def test_compute_never_mutates_links(self):
        conn = self.make_db()
        before = conn.execute("SELECT * FROM nugget_links").fetchall()
        validity.compute_supersessions(conn)
        self.assertEqual([tuple(r) for r in before],
                         [tuple(r) for r in conn.execute("SELECT * FROM nugget_links")])

    def test_validity_clause_is_empty_in_shadow_and_filters_when_live(self):
        conn = self.make_db()
        self.assertEqual(validity.link_validity_clause(conn, "nl"), "")  # no columns, shadow
        validity.ensure_validity_schema(conn)
        self.assertEqual(validity.link_validity_clause(conn, "nl"), "")  # columns, still shadow
        os.environ["ANAMNESIS_VALIDITY_LIVE"] = "1"
        self.assertIn("valid_to IS NULL", validity.link_validity_clause(conn, "nl"))

    def test_live_flag_roundtrip_and_empty_file_is_not_live(self):
        self.assertFalse(validity.is_validity_live())
        validity.set_validity_live(True)
        self.assertTrue(validity.is_validity_live())
        validity.set_validity_live(False)
        self.assertFalse(validity.is_validity_live())
        validity.validity_live_flag_path().parent.mkdir(parents=True, exist_ok=True)
        validity.validity_live_flag_path().write_text("")
        self.assertFalse(validity.is_validity_live())


class TestApply(ValidityCase):
    def test_apply_closes_links_then_live_filter_hides_them(self):
        conn = self.make_db()
        applied = validity.apply_supersessions(conn)
        self.assertEqual((applied["applied_verdicts"], applied["closed_links"]), (1, 1))
        row = conn.execute("SELECT valid_to, superseded_by FROM nugget_links WHERE id=1").fetchone()
        self.assertEqual(row["valid_to"], "2026-01-02T00:00:00")
        self.assertEqual(row["superseded_by"], "decision:new")
        validity.set_validity_live(True)
        clause = validity.link_validity_clause(conn, "nl")
        self.assertEqual(conn.execute(f"SELECT COUNT(*) FROM nugget_links nl WHERE 1=1 {clause}").fetchone()[0], 0)

    def test_apply_refuses_when_a_false_supersession_exists(self):
        conn = self.make_db()
        conn.execute("UPDATE chunks SET retirement_status='retired' WHERE id=11")  # newer is retired
        res = validity.apply_supersessions(conn)
        self.assertEqual(res["skipped_reason"], "false_supersessions_detected")
        self.assertEqual(res["closed_links"], 0)
        self.assertIsNone(conn.execute("SELECT valid_to FROM nugget_links WHERE id=1").fetchone()[0])


class TestGateAndCli(ValidityCase):
    def shadow(self, days, false=0, cov=0.72):
        started = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        validity.write_shadow({"started_at": started, "verdicts": [], "false_supersessions": false,
                         "key_coverage": cov, "db": str(self.db.resolve())})

    def test_gate_decision_reasons(self):
        self.assertFalse(validity.gate_decision({})["pass"])
        self.shadow(5)
        self.assertIn("need 30d", " ".join(validity.gate_decision()["reasons"]))
        self.shadow(31, false=3)
        self.assertIn("false supersession", " ".join(validity.gate_decision()["reasons"]))
        self.shadow(31, cov=0.1)
        self.assertIn("coverage", " ".join(validity.gate_decision()["reasons"]))
        self.shadow(31)
        self.assertTrue(validity.gate_decision()["pass"])
        os.environ["ANAMNESIS_VALIDITY_GATE_DAYS"] = "60"
        self.assertFalse(validity.gate_decision()["pass"])

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = validity.main(["--db", str(self.db), *argv])
        return rc, out.getvalue(), err.getvalue()

    def test_gate_command_promotes_only_on_pass(self):
        self.shadow(31)
        rc, out, _ = self.run_cli("gate")
        self.assertEqual(rc, 0)
        self.assertTrue(validity.is_validity_live())
        self.shadow(31, false=2)
        rc, out, _ = self.run_cli("gate")
        self.assertEqual(rc, 1)
        self.assertFalse(validity.is_validity_live(), "a failing gate must also switch back to shadow")

    def test_shadow_command_records_a_window_and_keeps_its_start(self):
        self.make_db()
        rc, out, _ = self.run_cli("shadow")
        self.assertEqual(rc, 0)
        first = validity.load_shadow()
        self.assertTrue(first["started_at"])
        self.assertEqual(len(first["verdicts"]), 1)
        self.run_cli("shadow")
        self.assertEqual(validity.load_shadow()["started_at"], first["started_at"])
        self.assertFalse(validity.is_validity_live(), "recording a shadow run must never go live")

    def test_apply_is_gated_never_sets_flag_and_warns_on_zero_closed(self):
        self.make_db(with_link=False)
        rc, _, err = self.run_cli("apply")
        self.assertEqual(rc, 2)
        self.assertIn("gate not satisfied", err)

        self.shadow(31)
        rc, out, err = self.run_cli("apply")
        self.assertEqual(rc, 0, err)
        self.assertIn("WARNING ZERO-CLOSED", out)
        self.assertFalse(validity.is_validity_live())

        self.db.unlink()
        self.make_db(with_link=True)
        rc, out, err = self.run_cli("apply")
        self.assertEqual(rc, 0, err)
        self.assertNotIn("ZERO-CLOSED", out)
        self.assertIn("1 nugget_links closed", out)
        self.assertFalse(validity.is_validity_live())
        conn = sqlite3.connect(str(self.db))
        self.assertTrue(conn.execute("SELECT valid_to FROM nugget_links WHERE id=1").fetchone()[0])

    def test_switching_database_restarts_the_shadow_window(self):
        self.make_db()
        self.shadow(31)
        other = self.db.with_name("other.db")
        self.make_db(path=other).close()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(validity.main(["--db", str(other), "shadow"]), 0)
        rec = validity.load_shadow()
        self.assertEqual(rec["db"], str(other.resolve()))
        self.assertIn("need 30d", " ".join(validity.gate_decision()["reasons"]))

    def test_apply_refuses_a_database_the_shadow_run_did_not_measure(self):
        self.make_db()
        self.shadow(31)
        other = self.db.with_name("other.db")
        self.make_db(path=other).close()
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = validity.main(["--db", str(other), "apply"])
        self.assertEqual(rc, 2)
        self.assertIn("shadow run measured", err.getvalue())


if __name__ == "__main__":
    unittest.main()
