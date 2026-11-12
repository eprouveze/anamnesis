#!/usr/bin/env python3
"""Tests for decay.py — propose-only value-decay review. No network, no API key.

    python -m unittest discover -s tools -p 'test_*.py' -v
"""
import contextlib
import hashlib
import io
import json
import os
import re
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anamnesis_llm  # noqa: E402
import decay  # noqa: E402

AS_OF = datetime(2026, 6, 1, tzinfo=timezone.utc)


def cand(cid, store="memory", observed="2026-01-01T00:00:00Z", content="Some durable fact."):
    return {"candidate_id": cid, "store_type": store, "source_path": f"notes/{cid}.md", "title": cid,
            "content": content, "observed_at": observed, "protected": store in decay.PROTECTED_TYPES,
            "content_sha256": hashlib.sha256(content.encode()).hexdigest()}


def judgment(cid, action="keep", scores=None):
    return {"candidate_id": cid, "scores": scores or {"continued_relevance": 3, "uniqueness": 2,
                                                      "temporal_validity": 4},
            "confidence": 0.8, "recommended_action": action, "rationale": "r", "evidence_needed": "e"}


class TestSelection(unittest.TestCase):
    def test_age_floor_and_missing_timestamps(self):
        pool = [cand("old"), cand("fresh", observed="2026-05-30T00:00:00Z"), cand("nodate", observed="")]
        picked = decay.select_candidates(pool, AS_OF, 30, 10, 900, 24000)
        self.assertEqual([c["candidate_id"] for c in picked], ["old"])

    def test_bounds_and_truncation(self):
        pool = [cand(f"c{i}", content="x" * 5000) for i in range(10)]
        picked = decay.select_candidates(pool, AS_OF, 0, 3, 300, 24000)
        self.assertEqual(len(picked), 3)
        self.assertTrue(all(len(c["content"]) == 300 for c in picked))
        tiny = decay.select_candidates(pool, AS_OF, 0, 10, 300, 1000)
        self.assertLess(len(tiny), 10)

    def test_rotation_is_stable_within_a_period(self):
        pool = [cand(f"c{i}") for i in range(8)]
        a = decay.select_candidates(pool, AS_OF, 0, 4, 900, 24000)
        b = decay.select_candidates(list(reversed(pool)), AS_OF, 0, 4, 900, 24000)
        self.assertEqual([c["candidate_id"] for c in a], [c["candidate_id"] for c in b])


class TestJudgments(unittest.TestCase):
    def test_valid_and_protected_review_archive_is_downgraded(self):
        pool = [cand("a"), cand("b", store="decision")]
        out = decay.validate_judgments(
            {"schema_version": 1, "judgments": [judgment("a", "review_archive"),
                                                judgment("b", "review_archive")]}, pool)
        by = {j["candidate_id"]: j for j in out}
        self.assertEqual(by["a"]["recommended_action"], "review_archive")
        self.assertEqual(by["b"]["recommended_action"], "verify")
        self.assertTrue(by["b"]["safety_adjustment"])

    def test_rejects_malformed_payloads(self):
        pool = [cand("a")]
        bad = [
            {"schema_version": 2, "judgments": []},
            {"schema_version": 1, "judgments": []},  # missing judgment for a
            {"schema_version": 1, "judgments": [judgment("zzz")]},
            {"schema_version": 1, "judgments": [judgment("a"), judgment("a")]},
            {"schema_version": 1, "judgments": [judgment("a", "delete")]},
            {"schema_version": 1, "judgments": [judgment("a", scores={"continued_relevance": 9,
                                                                       "uniqueness": 1, "temporal_validity": 1})]},
        ]
        for payload in bad:
            with self.assertRaises(ValueError, msg=str(payload)[:60]):
                decay.validate_judgments(payload, pool)

    def test_parse_model_json_strips_code_fences(self):
        self.assertEqual(decay.parse_model_json('```json\n{"a": 1}\n```'), {"a": 1})


class TestCli(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ("ANAMNESIS_HOME", "ANAMNESIS_DB")}
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        os.environ["ANAMNESIS_HOME"] = str(self.tmp / "home")
        self.db = self.tmp / "memory.db"
        conn = sqlite3.connect(self.db)
        conn.execute("CREATE TABLE chunks (chunk_id TEXT, store_type TEXT, source_path TEXT, title TEXT, "
                     "content TEXT, created_at TEXT, modified_at TEXT, indexed_at TEXT, retirement_status TEXT)")
        conn.execute("INSERT INTO chunks VALUES ('m1','memory','notes/m1.md','Old fact','body one',"
                     "'2025-01-01T00:00:00Z','2025-01-02T00:00:00Z','2025-01-03T00:00:00Z','active')")
        conn.execute("INSERT INTO chunks VALUES ('d1','decision','notes/d1.md','A decision','body two',"
                     "'2025-01-01T00:00:00Z',NULL,'2025-01-03T00:00:00Z','active')")
        conn.execute("INSERT INTO chunks VALUES ('r1','memory','notes/r1.md','Retired','body',"
                     "'2025-01-01T00:00:00Z',NULL,'2025-01-03T00:00:00Z','retired')")
        conn.execute("INSERT INTO chunks VALUES ('s1','session_topic','notes/s.md','Session','body',"
                     "'2025-01-01T00:00:00Z',NULL,'2025-01-03T00:00:00Z','active')")
        conn.commit()
        conn.close()
        self.before = self.db.read_bytes()

    def tearDown(self):
        self._tmp.cleanup()
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = decay.main(["--db", str(self.db), "--as-of", "2026-06-01T00:00:00Z", *argv])
        return rc, out.getvalue(), err.getvalue()

    def test_default_run_makes_no_model_call_and_does_not_touch_the_db(self):
        with mock.patch.object(anamnesis_llm, "complete", side_effect=AssertionError("model called")):
            rc, out, err = self.run_main()
        self.assertEqual(rc, 0, err)
        self.assertIn("mode=propose candidates=2 judged=0", out)
        self.assertEqual(self.db.read_bytes(), self.before)
        reports = list((self.tmp / "home" / "decay-reports").glob("*.json"))
        self.assertEqual(len(reports), 1)
        report = json.loads(reports[0].read_text())
        self.assertFalse(report["mutation_capability"])
        self.assertEqual({c["candidate_id"] for c in report["candidates"]}, {"m1", "d1"})

    def test_judge_uses_the_configured_model_and_stays_propose_only(self):
        reply = json.dumps({"schema_version": 1, "judgments": [judgment("m1", "review_archive"),
                                                               judgment("d1", "review_archive")]})
        with mock.patch.object(anamnesis_llm, "complete", return_value=reply) as m:
            rc, out, err = self.run_main("--judge", "--model", "test-model")
        self.assertEqual(rc, 0, err)
        self.assertEqual(m.call_args.args[1], "test-model")
        self.assertIn("judged=2", out)
        md = next((self.tmp / "home" / "decay-reports").glob("*.md")).read_text()
        self.assertIn("PROPOSE ONLY", md)
        self.assertIn("protected store type cannot receive review_archive", md)
        self.assertEqual(self.db.read_bytes(), self.before)

    def test_unavailable_model_fails_loudly_without_writing_a_report(self):
        with mock.patch.object(anamnesis_llm, "complete", side_effect=anamnesis_llm.LLMUnavailable("no key")):
            rc, _, err = self.run_main("--judge")
        self.assertEqual(rc, 2)
        self.assertIn("no key", err)
        self.assertFalse((self.tmp / "home" / "decay-reports").exists())

    def test_judge_and_judgments_are_mutually_exclusive(self):
        rc, _, err = self.run_main("--judge", "--judgments", str(self.tmp / "x.json"))
        self.assertEqual(rc, 2)

    def test_source_has_no_mutating_sql(self):
        src = Path(decay.__file__).read_text(encoding="utf-8")
        self.assertIsNone(
            re.search(r"\b(INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM|DROP\s+TABLE|ALTER\s+TABLE)\b",
                      src, re.IGNORECASE),
            "decay.py must not contain mutating SQL")


if __name__ == "__main__":
    unittest.main()
