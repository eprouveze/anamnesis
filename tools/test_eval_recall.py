"""Tests for eval_recall.py with an injected search function and the heuristic judge (no network)."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
import eval_recall  # noqa: E402


class EvalTests(unittest.TestCase):
    def test_run_with_heuristic_judge_saves_results_and_history(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            golden = t / "g.json"
            golden.write_text(json.dumps({"questions": [
                {"id": "q1", "question": "where do quokkas live", "ground_truth": "quokkas live on islands", "category": "bio"},
                {"id": "q2", "question": "unknown", "ground_truth": "nothing relevant here at all", "category": "bio"}]}))
            hits = {"where do quokkas live": [{"content": "quokkas live on islands", "store_type": "memory",
                                                "project": "p", "similarity": 0.9}], "unknown": []}
            patches = [mock.patch.object(eval_recall, "GOLDEN_SET", golden),
                       mock.patch.object(eval_recall, "EVAL_DIR", t / "e"),
                       mock.patch.object(eval_recall, "NEW_RESULTS", t / "e" / "results.json"),
                       mock.patch.object(eval_recall, "PREVIOUS_RESULTS", t / "e" / "prev.json"),
                       mock.patch.object(eval_recall, "HISTORY_FILE", t / "e" / "h.csv")]
            for p in patches:
                p.start()
            try:
                out = io.StringIO()
                with redirect_stdout(out):
                    eval_recall.run_assessment(search_fn=lambda q, limit=5: hits[q],
                                               judge=lambda q, gt, ps: eval_recall._heuristic_judgment(gt, ps), sleep_s=0)
                saved = json.loads((t / "e" / "results.json").read_text())
                self.assertEqual(saved["q1"]["judgment"]["recall"], "yes")
                self.assertEqual(saved["q2"]["judgment"]["recall"], "no")
                self.assertIn("Recall:          1/2", out.getvalue())
                self.assertEqual(len((t / "e" / "h.csv").read_text().splitlines()), 2)
            finally:
                for p in patches:
                    p.stop()

    def test_bootstrap_ci_bounds(self):
        m, lo, hi = eval_recall.bootstrap_ci([1.0, 0.0] * 20)
        self.assertAlmostEqual(m, 0.5)
        self.assertTrue(0.3 < lo < 0.5 < hi < 0.7)


if __name__ == "__main__":
    unittest.main()
