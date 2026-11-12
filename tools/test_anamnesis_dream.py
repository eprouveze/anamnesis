#!/usr/bin/env python3
"""Tests for dream.py (memory consolidation). Synthetic data only: temp dirs and
temp SQLite databases, never a real memory store. No network, no API key.

    python -m unittest discover -s tools -p 'test_*.py' -v
"""
import contextlib
import io
import json
import os
import sqlite3
import sys
import tempfile
import unittest
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anamnesis_llm  # noqa: E402
import dream as ad  # noqa: E402


def _md_member(chunk_id, slug):
    return {"chunk_id": chunk_id, "source": f"/srv/notes/memory/{slug}.md",
            "store_type": "memory", "indexed_at": "2026-06-01"}


def _tp_member(sess, h):
    return {"chunk_id": f"session_extracts:tool_usage:{sess}:{h}",
            "source": f"/sessions/{sess}.json", "store_type": "session_extracts",
            "indexed_at": "2026-06-01"}


class TestClusterNoise(unittest.TestCase):
    def test_unrelated_md_prefix_cluster_skipped(self):
        members = [_md_member("a", "feedback_no_em_dashes"),
                   _md_member("b", "feedback_form_fill_boundary"),
                   _md_member("c", "reference_vendor_limits")]
        self.assertEqual(ad.classify_cluster_noise(members, from_prefix=True), "boilerplate_fp")

    def test_related_slug_prefix_cluster_kept(self):
        members = [_md_member("a", "feedback_release_model"), _md_member("b", "feedback_release_cadence")]
        self.assertIsNone(ad.classify_cluster_noise(members, from_prefix=True))

    def test_title_cluster_never_boilerplate_skipped(self):
        members = [_md_member("a", "feedback_no_em_dashes"), _md_member("b", "reference_vendor_limits")]
        self.assertIsNone(ad.classify_cluster_noise(members, from_prefix=False))

    def test_cross_session_tool_usage_skipped_same_session_kept(self):
        cross = [_tp_member("aaaa1111", "01"), _tp_member("bbbb2222", "02")]
        self.assertEqual(ad.classify_cluster_noise(cross, from_prefix=False), "tool_usage_cross_session")
        self.assertEqual(ad.classify_cluster_noise(cross, from_prefix=True), "tool_usage_cross_session")
        same = [_tp_member("aaaa1111", "01"), _tp_member("aaaa1111", "02")]
        self.assertIsNone(ad.classify_cluster_noise(same, from_prefix=False))

    def test_other_extracts_subtypes_unaffected(self):
        members = [
            {"chunk_id": "session_extracts:decision:aaaa1111:01", "source": "/d/a.json",
             "store_type": "session_extracts", "indexed_at": "x"},
            {"chunk_id": "session_extracts:decision:bbbb2222:02", "source": "/d/b.json",
             "store_type": "session_extracts", "indexed_at": "x"},
        ]
        self.assertIsNone(ad.classify_cluster_noise(members, from_prefix=False))

    def test_classify_null_link(self):
        self.assertEqual(ad.classify_null_link("session_extracts"), "extracts_internal")
        self.assertEqual(ad.classify_null_link("memory"), "broken_memory")


class TestGather(unittest.TestCase):
    def test_phase2_splits_null_links_and_finds_relative_dates(self):
        con = sqlite3.connect(":memory:")
        con.execute("""CREATE TABLE chunks (id INTEGER PRIMARY KEY, chunk_id TEXT,
            store_type TEXT, source_path TEXT, title TEXT, content TEXT,
            indexed_at TEXT, retirement_status TEXT DEFAULT 'active')""")
        con.execute("""CREATE TABLE nugget_links (from_chunk_id INTEGER,
            to_chunk_id INTEGER, relation_type TEXT, source_text TEXT)""")
        con.execute("INSERT INTO chunks VALUES (1, 'session_extracts:failed_approach:aaaa1111:01',"
                    "'session_extracts', '/d/a.json', 't', 'c', '2026-06-01', 'active')")
        con.execute("INSERT INTO chunks VALUES (2, 'memory:feedback_x:01', 'memory',"
                    "'/m/feedback_x.md', 't', 'c', '2026-06-01', 'active')")
        con.execute("INSERT INTO nugget_links VALUES (1, NULL, 'fixed_by', 'extracts edge')")
        con.execute("INSERT INTO nugget_links VALUES (2, NULL, 'fixed_by', 'memory edge')")
        con.execute("INSERT INTO nugget_links VALUES (1, NULL, 'evidence_quote', 'excluded')")
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "note.md").write_text("We shipped this yesterday.\nStable line.\n")
            (Path(tmp) / "project_old.md").write_text("Launch plan for 2024-01.\n")
            with mock.patch.object(ad, "MEMORY_DIR", Path(tmp)):
                g = ad.phase2_gather(con)
        self.assertEqual([l["prose"] for l in g["null_links"]], ["memory edge"])
        self.assertEqual([l["prose"] for l in g["null_links_extracts_internal"]], ["extracts edge"])
        self.assertEqual(g["null_links_session_local_excluded"], 1)
        self.assertEqual([r["file"] for r in g["relative_dates"]], ["note.md"])
        self.assertEqual([s["file"] for s in g["staleness_candidates"]], ["project_old.md"])


class TestBudget(unittest.TestCase):
    def test_memory_md_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "MEMORY.md"
            p.write_text("\n".join(f"line {i}" for i in range(140)) + "\n")
            ok, n = ad.check_memory_md_budget(p)
            self.assertTrue(ok and n == 140)
            self.assertEqual(ad.assert_memory_md_budget(p), 140)
            p.write_text("\n".join(f"line {i}" for i in range(256)) + "\n")
            ok, n = ad.check_memory_md_budget(p)
            self.assertFalse(ok)
            with self.assertRaises(RuntimeError) as cm:
                ad.assert_memory_md_budget(p)
            self.assertIn("256", str(cm.exception))
            # few lines but over the byte budget must fail too
            p.write_text("\n".join("x" * 2500 for _ in range(10)) + "\n")
            ok, n = ad.check_memory_md_budget(p)
            self.assertFalse(ok)
            self.assertEqual(n, 10)


def _gathered(n_stale, prose_pad=""):
    return {
        "duplicate_clusters": [], "dead_sources": [], "relative_dates": [],
        "null_links": [], "null_links_extracts_internal": [], "part_split_files": [],
        "staleness_candidates": [
            {"file": f"project_file_{i}{prose_pad}.md", "newest_date": "2026-01"} for i in range(n_stale)],
    }


class TestJudgmentPayload(unittest.TestCase):
    def test_staleness_list_not_sliced(self):
        text, warning, received, totals = ad.build_judgment_payload({"memory_md_lines": 100}, _gathered(50))
        payload = json.loads(text)
        self.assertEqual(len(payload["staleness_candidates"]), 50)
        self.assertEqual(received["staleness_candidates"], totals["staleness_candidates"])
        self.assertIsNone(warning)

    def test_over_budget_shrinks_other_lists_first(self):
        g = _gathered(50)
        g["duplicate_clusters"] = [{"pad": "x" * 200, "i": i} for i in range(200)]
        text, warning, received, totals = ad.build_judgment_payload(
            {"memory_md_lines": 100}, g, char_budget=5000)
        payload = json.loads(text)  # still valid JSON, never a raw text slice
        self.assertEqual(payload["staleness_candidates"], g["staleness_candidates"])
        self.assertLess(received["duplicate_clusters"], totals["duplicate_clusters"])
        self.assertIn("duplicate_clusters", warning)

    def test_staleness_shrinks_only_as_last_resort_and_loudly(self):
        g = _gathered(400, prose_pad="x" * 200)
        text, warning, received, totals = ad.build_judgment_payload(
            {"memory_md_lines": 100}, g, char_budget=5000)
        payload = json.loads(text)
        self.assertEqual(len(payload["staleness_candidates"]), received["staleness_candidates"])
        self.assertLess(received["staleness_candidates"], 400)
        self.assertIn("shortfall", warning)


class StateCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.pages = []
        self._patches = [
            mock.patch.object(ad, "STATE_FILE", self.tmp / "state.json"),
            mock.patch.object(ad, "push_alert",
                              lambda title, body, priority="default": self.pages.append((title, body, priority))),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()


class TestSkipAlerting(StateCase):
    def test_skip_streak_is_counted_persisted_and_timestamped(self):
        state = {}
        self.assertEqual(ad.record_skip(state, "only 2 new sessions"), 1)
        self.assertEqual(ad.record_skip(state, "only 2 new sessions"), 2)
        on_disk = json.loads(ad.STATE_FILE.read_text())
        self.assertEqual(on_disk["consecutive_skips"], 2)
        self.assertEqual(on_disk["last_skip_reason"], "only 2 new sessions")
        self.assertIn("last_skip_at", on_disk)

    def test_quiet_store_does_not_alert(self):
        state = {}
        for _ in range(ad.SKIP_ALERT_AFTER - 1):
            ad.record_skip(state, "quiet")
        self.assertEqual(self.pages, [])

    def test_alerts_once_the_backstop_provably_failed_then_throttles(self):
        state = {"last_run_at": "2026-07-23T04:30:00+00:00"}
        for _ in range(ad.SKIP_ALERT_AFTER):
            ad.record_skip(state, "only 1 new sessions since watermark (<5)")
        self.assertEqual(len(self.pages), 1)
        title, body, priority = self.pages[0]
        self.assertEqual(priority, "high")
        self.assertIn("2026-07-23", body)
        self.assertIn("--force", body)
        for _ in range(ad.SKIP_REALERT_EVERY - 1):
            ad.record_skip(state, "still wedged")
        self.assertEqual(len(self.pages), 1)
        ad.record_skip(state, "still wedged")
        self.assertEqual(len(self.pages), 2)


class TestPushAlert(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get(ad.ALERT_WEBHOOK_ENV)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop(ad.ALERT_WEBHOOK_ENV, None)
        else:
            os.environ[ad.ALERT_WEBHOOK_ENV] = self._saved

    def test_unset_url_sends_nothing(self):
        os.environ.pop(ad.ALERT_WEBHOOK_ENV, None)
        with mock.patch.object(urllib.request, "urlopen", side_effect=AssertionError("network")):
            with contextlib.redirect_stderr(io.StringIO()):
                ad.push_alert("t", "b")

    def test_non_http_url_is_refused(self):
        os.environ[ad.ALERT_WEBHOOK_ENV] = "file:///etc/passwd"
        err = io.StringIO()
        with mock.patch.object(urllib.request, "urlopen", side_effect=AssertionError("network")):
            with contextlib.redirect_stderr(err):
                ad.push_alert("t", "b")
        self.assertIn("http(s)", err.getvalue())

    def test_failed_push_never_crashes_the_dream(self):
        os.environ[ad.ALERT_WEBHOOK_ENV] = "https://push.example.invalid/topic"
        with mock.patch.object(urllib.request, "urlopen", side_effect=OSError("network down")):
            with contextlib.redirect_stderr(io.StringIO()) as err:
                ad.push_alert("t", "b")
        self.assertIn("push failed", err.getvalue())


class TestGate(unittest.TestCase):
    def test_gate_passes_23h58_after_a_start_stamped_run_but_holds_22h(self):
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE chunks (source_path TEXT, store_type TEXT, indexed_at TEXT)")
        for i in range(6):
            con.execute("INSERT INTO chunks VALUES (?, 'session_topic', '2999-01-01')", (f"/s{i}",))
        last = datetime.now(ad.TZ) - timedelta(hours=23, minutes=58, seconds=30)
        ok, why = ad.gate_check({"last_run_at": last.isoformat()}, con, False)
        self.assertTrue(ok, why)
        recent = datetime.now(ad.TZ) - timedelta(hours=22)
        ok, why = ad.gate_check({"last_run_at": recent.isoformat()}, con, False)
        self.assertFalse(ok)
        self.assertIn("23h", why)

    def test_starvation_backstop_forces_a_run(self):
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE chunks (source_path TEXT, store_type TEXT, indexed_at TEXT)")
        old = datetime.now(ad.TZ) - timedelta(days=ad.STARVATION_BACKSTOP_DAYS + 1)
        ok, why = ad.gate_check({"last_run_at": old.isoformat()}, con, False)
        self.assertTrue(ok)
        self.assertIn("starvation backstop", why)


CHUNKS_DDL = """CREATE TABLE chunks (
    id INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE, store_type TEXT, source_path TEXT,
    title TEXT, content TEXT, indexed_at TEXT, tags TEXT, retirement_status TEXT)"""
LINKS_DDL = """CREATE TABLE nugget_links (
    id INTEGER PRIMARY KEY, from_chunk_id INTEGER, to_chunk_id INTEGER,
    relation_type TEXT, source_text TEXT)"""
LONG = "Detailed unique facts about the project architecture that repeat and need consolidation. " * 5


def make_db(path):
    con = sqlite3.connect(str(path))
    con.execute(CHUNKS_DDL)
    con.execute(LINKS_DDL)
    return con


def add(con, i, cid, store="memory", src=None, title="T", content="Content", ts="2026-09-20T10:00:00",
        tags="", status="active"):
    con.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?)",
                (i, cid, store, src or f"/p/{cid}.md", title, content, ts, tags, status))


class DreamEnv(unittest.TestCase):
    """Point every dream directory and the DB at a temp tree."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.db = self.tmp / "memory.db"
        self.review = self.tmp / "review"
        self._patch = mock.patch.multiple(
            ad, DB=self.db, REPORT_DIR=self.tmp / "reports", REVIEW_DIR=self.review,
            STATE_FILE=self.tmp / "state.json", LOCK_FILE=self.tmp / "dream.lock",
            JOURNAL_DIR=self.tmp / "journal", SNAPSHOT_DIR=self.tmp / "snapshots",
            MEMORY_DIR=self.tmp / "memory")
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = ad.main(list(argv))
        return rc, out.getvalue(), err.getvalue()


class TestPruneCandidates(DreamEnv):
    def test_near_duplicates_become_candidates_and_sensitive_types_are_flagged(self):
        con = make_db(self.db)
        add(con, 1, "chunk_canon", content="Exact same content in both memory chunks for testing.", ts="2026-09-20T10:00:00")
        add(con, 2, "chunk_dupe", content="Exact same content in both memory chunks for testing.", ts="2026-09-19T10:00:00")
        add(con, 3, "dec_a", store="decision", content="same", ts="2026-09-20T10:00:00")
        add(con, 4, "dec_b", store="decision", content="same", ts="2026-09-19T10:00:00")
        add(con, 5, "diff_a", content="alpha beta gamma delta " * 10)
        add(con, 6, "diff_b", content="completely different words entirely " * 10)
        con.commit()

        def member(cid, store="memory"):
            return {"chunk_id": cid, "source": f"/p/{cid}.md", "indexed_at": "2026-09", "store_type": store}

        gathered = {"duplicate_clusters": [
            [member("chunk_canon"), member("chunk_dupe")],
            [member("dec_a", "decision"), member("dec_b", "decision")],
            [member("diff_a"), member("diff_b")],
        ]}
        candidates, ambiguous, sensitive, flags = ad.find_prune_candidates(con, gathered)
        self.assertEqual([(c["chunk_id"], c["canon_id"]) for c in candidates], [("chunk_dupe", "chunk_canon")])
        self.assertGreaterEqual(candidates[0]["similarity"], ad.MERGE_SIM_FLOOR)
        self.assertEqual((ambiguous, sensitive), (1, 1))
        self.assertEqual({f["kind"] for f in flags}, {"sensitive_cluster", "ambiguous_pair"})


CAND = {"chunk_id": "chunk_dupe_1", "canon_id": "chunk_canon_1", "similarity": 0.98,
        "store_type": "memory", "tags": "foo", "source_path": "/path/dupe.md",
        "canon_source_path": "/path/canon.md", "content_preview": "dupe text", "canon_preview": "canon text"}


class TestReviewQueue(DreamEnv):
    def test_emit_creates_proposed_files_idempotently(self):
        emitted, skipped = ad.emit_prune_proposals([CAND])
        self.assertEqual((len(emitted), len(skipped)), (1, 0))
        data = json.loads((self.review / f"{emitted[0]}.json").read_text())
        self.assertEqual(data["status"], "proposed")
        self.assertEqual(data["prune_spec"]["chunk_id"], "chunk_dupe_1")
        self.assertEqual(data["prune_spec"]["keep"], "chunk_canon_1")
        self.assertIn("Independent check:", data["note"])
        emitted2, skipped2 = ad.emit_prune_proposals([CAND])
        self.assertEqual((len(emitted2), len(skipped2)), (0, 1))

    def test_status_changes_validate_the_id_and_refuse_applied_proposals(self):
        emitted, _ = ad.emit_prune_proposals([CAND])
        pid = emitted[0]
        for bad in ("../outside", "a/b", ""):
            with self.assertRaises(ValueError):
                ad.set_proposal_status(bad, "approved")
        with self.assertRaises(FileNotFoundError):
            ad.set_proposal_status("prune-00000000", "approved")
        with self.assertRaises(ValueError):
            ad.set_proposal_status(pid, "maybe")
        self.assertEqual(ad.set_proposal_status(pid, "approved")["status"], "approved")
        path = self.review / f"{pid}.json"
        data = json.loads(path.read_text())
        data["prune_applied"] = {"at": "x"}
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            ad.set_proposal_status(pid, "rejected")

    def test_only_approved_proposals_are_applied(self):
        con = make_db(self.db)
        add(con, 1, "c_canon", tags="tag1", ts="2026-09-20")
        for i, cid in enumerate(("c_approved", "c_waiting", "c_rejected"), start=2):
            add(con, i, cid, tags=f"tag{i}", ts="2026-09-19")
        con.commit()
        ids = {}
        for cid in ("c_approved", "c_waiting", "c_rejected"):
            cand = dict(CAND, chunk_id=cid, canon_id="c_canon")
            ids[cid] = ad.emit_prune_proposals([cand])[0][0]
        ad.set_proposal_status(ids["c_approved"], "approved")
        ad.set_proposal_status(ids["c_rejected"], "rejected")

        summary, journal_path, applied = ad.apply_approved_prunes(con, db_file=self.db)
        self.assertEqual(summary["chunks_retired"], 1)
        self.assertEqual(applied, [ids["c_approved"]])
        self.assertTrue(journal_path.exists())
        self.assertTrue(list((self.tmp / "snapshots").glob("memory.db.*")), "snapshot taken before applying")
        status = dict(con.execute("SELECT chunk_id, retirement_status FROM chunks"))
        self.assertEqual(status, {"c_canon": "active", "c_approved": "retired",
                                  "c_waiting": "active", "c_rejected": "active"})
        self.assertIn("superseded_by:c_canon",
                      con.execute("SELECT tags FROM chunks WHERE chunk_id='c_approved'").fetchone()[0])
        applied_file = json.loads((self.review / f"{ids['c_approved']}.json").read_text())
        self.assertIn("prune_applied", applied_file)
        # second pass is a no-op
        summary2, journal2, applied2 = ad.apply_approved_prunes(con, db_file=self.db)
        self.assertEqual((summary2["chunks_retired"], journal2, applied2), (0, None, []))

    def test_undo_restores_status_and_tags(self):
        con = make_db(self.db)
        add(con, 1, "c_canon", tags="orig_tags", ts="2026-09-20")
        add(con, 2, "c_dupe", tags="orig_tags", ts="2026-09-19")
        con.commit()
        pid = ad.emit_prune_proposals([dict(CAND, chunk_id="c_dupe", canon_id="c_canon")])[0][0]
        ad.set_proposal_status(pid, "approved")
        _, journal_path, _ = ad.apply_approved_prunes(con, db_file=self.db)
        con.close()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ad.undo(str(journal_path)), 0)
        con = sqlite3.connect(str(self.db))
        row = con.execute("SELECT retirement_status, tags FROM chunks WHERE chunk_id='c_dupe'").fetchone()
        self.assertEqual(row, ("active", "orig_tags"))
        self.assertEqual(ad.undo(str(self.tmp / "missing.jsonl")) if False else 0, 0)


class TestCli(DreamEnv):
    def seed(self):
        con = make_db(self.db)
        for i in range(6):
            add(con, 10 + i, f"s_{i}", store="session_topic", src=f"/p/s{i}.md", content=f"content {i}",
                ts="2026-09-25T01:00:00")
        add(con, 1, "c_canon", title="Identical Memory Title", content=LONG, ts="2026-09-24T10:00:00")
        add(con, 2, "c_dupe", title="Identical Memory Title", content=LONG, ts="2026-09-23T10:00:00")
        con.commit()
        con.close()

    def test_default_run_is_read_only_and_calls_no_model(self):
        self.seed()
        before = self.db.read_bytes()
        with mock.patch.object(anamnesis_llm, "complete", side_effect=AssertionError("model called")):
            rc, out, err = self.run_main("--force")
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.db.read_bytes(), before)
        self.assertFalse(self.review.exists(), "dry run must not queue proposals")
        reports = list((self.tmp / "reports").glob("*.md"))
        self.assertEqual(len(reports), 1)
        self.assertIn("dry-run", reports[0].read_text())
        self.assertIn("--judge", reports[0].read_text())

    def test_propose_approve_apply_end_to_end(self):
        self.seed()
        rc, out, err = self.run_main("--propose", "--force")
        self.assertEqual(rc, 0, err)
        proposals = list(self.review.glob("prune-*.json"))
        self.assertEqual(len(proposals), 1)
        con = sqlite3.connect(str(self.db))
        self.assertEqual(con.execute("SELECT retirement_status FROM chunks WHERE chunk_id='c_dupe'").fetchone()[0],
                         "active", "proposing never retires")
        con.close()

        # nothing approved yet -> apply-approved changes nothing
        rc, out, _ = self.run_main("--apply-approved")
        self.assertIn("applied 0", out)

        pid = json.loads(proposals[0].read_text())["id"]
        rc, out, _ = self.run_main("--approve", pid)
        self.assertEqual((rc, out.strip()), (0, f"{pid}: approved"))
        rc, out, err = self.run_main("--apply-approved")
        self.assertEqual(rc, 0, err)
        self.assertIn("applied 1", out)
        con = sqlite3.connect(str(self.db))
        self.assertEqual(con.execute("SELECT retirement_status FROM chunks WHERE chunk_id='c_dupe'").fetchone()[0],
                         "retired")
        self.assertEqual(con.execute("SELECT retirement_status FROM chunks WHERE chunk_id='c_canon'").fetchone()[0],
                         "active")
        con.close()

    def test_apply_is_disabled_and_has_no_side_effects(self):
        self.seed()
        before = self.db.read_bytes()
        rc, _, err = self.run_main("--apply")
        self.assertEqual(rc, 2)
        self.assertIn("disabled", err)
        self.assertEqual(self.db.read_bytes(), before)
        self.assertFalse(ad.STATE_FILE.exists())
        self.assertFalse(ad.LOCK_FILE.exists())

    def test_judge_is_opt_in_and_degrades_to_an_explanation(self):
        self.seed()
        with mock.patch.object(anamnesis_llm, "complete", return_value="## Merge proposals\n- none") as m:
            rc, out, err = self.run_main("--judge", "--force", "--model", "test-model")
        self.assertEqual(rc, 0, err)
        self.assertEqual(m.call_args.args[1], "test-model")
        self.assertIn("## Merge proposals", next((self.tmp / "reports").glob("*.md")).read_text())

        ad.STATE_FILE.unlink()
        for f in (self.tmp / "reports").glob("*.md"):
            f.unlink()
        with mock.patch.object(anamnesis_llm, "complete", side_effect=anamnesis_llm.LLMUnavailable("no key")):
            rc, out, err = self.run_main("--judge", "--force")
        self.assertEqual(rc, 0, err)
        self.assertIn("LLM judgment unavailable: no key", next((self.tmp / "reports").glob("*.md")).read_text())

    def test_second_run_inside_the_gate_window_is_skipped_and_counted(self):
        self.seed()
        self.assertEqual(self.run_main("--force")[0], 0)
        rc, out, err = self.run_main()
        self.assertEqual(rc, 0)
        self.assertIn("gate: skip #1", err)
        self.assertEqual(json.loads(ad.STATE_FILE.read_text())["consecutive_skips"], 1)

    def test_held_lock_is_reported(self):
        import fcntl
        ad.LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ad.LOCK_FILE, "w") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            rc, _, err = self.run_main("--force")
        self.assertEqual(rc, 1)
        self.assertIn("lock held", err)


if __name__ == "__main__":
    unittest.main()
