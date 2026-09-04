"""Tests for the query-embedding cache in recall.py. No API key, no network:
_embed_remote is stubbed and counted.

    python -m unittest discover -s tools -p 'test_*.py' -v
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
import index  # noqa: E402
import recall  # noqa: E402

VEC = [0.5, 0.25, 0.125, 1.0]  # exactly representable in float32 (the cache stores f32)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name) / "q.db"
        self._patches = [
            mock.patch.object(recall, "QUERY_CACHE", self.cache),
            mock.patch.object(recall, "_MEM", {}),
            mock.patch.object(recall, "_SCHEMA_OK", set()),
        ]
        for p in self._patches:
            p.start()
        self.calls = 0

        def fake_remote(q):
            self.calls += 1
            return list(VEC)
        self._remote = mock.patch.object(recall, "_embed_remote", side_effect=fake_remote)
        self._remote.start()

    def tearDown(self):
        self._remote.stop()
        for p in self._patches:
            p.stop()
        self.tmp.cleanup()

    def test_second_call_hits_in_process_cache(self):
        recall.embed_query("hello world")
        recall.embed_query("hello world")
        self.assertEqual(self.calls, 1)

    def test_sqlite_cache_survives_new_process(self):
        recall.embed_query("hello world")
        recall._MEM.clear()          # simulate a fresh short-lived CLI process
        recall._SCHEMA_OK.clear()
        self.assertEqual(recall.embed_query("hello world"), VEC)
        self.assertEqual(self.calls, 1)

    def test_key_normalises_whitespace_only(self):
        self.assertEqual(recall.cache_key("a  b\n c"), recall.cache_key("a b c"))
        self.assertNotEqual(recall.cache_key("Hello"), recall.cache_key("hello"))

    def test_model_change_reaps_stale_rows(self):
        recall.embed_query("hello world")
        with mock.patch.object(recall, "MODEL", "other-model"), \
             mock.patch.object(recall, "_MEM", {}), mock.patch.object(recall, "_SCHEMA_OK", set()):
            recall.embed_query("hello world")
            self.assertEqual(self.calls, 2)
            rows = sqlite3.connect(str(self.cache)).execute(
                "SELECT model FROM query_embeddings").fetchall()
        self.assertEqual(rows, [("other-model",)])

    def test_cache_failure_is_a_miss_not_an_error(self):
        bad = Path(self.tmp.name) / "missing-dir" / "q.db"
        with mock.patch.object(recall, "QUERY_CACHE", bad):
            self.assertEqual(recall.embed_query("hello world"), VEC)
        self.assertEqual(self.calls, 1)

    def test_failed_embed_is_not_cached(self):
        self._remote.stop()
        with mock.patch.object(recall, "_embed_remote", side_effect=RuntimeError("429")):
            with self.assertRaises(RuntimeError):
                recall.embed_query("hello world")
        self._remote.start()
        self.assertEqual(sqlite3.connect(str(self.cache)).execute(
            "SELECT COUNT(*) FROM query_embeddings").fetchone()[0], 0)


class RecallDegradeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        db = Path(self.tmp.name) / "memory.db"
        conn = index.init_db(db)
        for i, (title, body) in enumerate([("timestamps", "store timestamps as UTC ISO strings"),
                                           ("naming", "prefer kebab-case slugs")]):
            conn.execute("INSERT INTO chunks (id, chunk_id, title, content, store_type, source, weight,"
                         " embedding, indexed_at) VALUES (?,?,?,?,?,?,?,?,?)",
                         (i + 1, f"c{i}", title, body, "memory", "t.md", 1.0, recall.pack(VEC), "now"))
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
        conn.commit()
        conn.close()
        self._db = mock.patch.object(recall, "DB", db)
        self._db.start()
        self._mem = mock.patch.object(recall, "_MEM", {})
        self._mem.start()
        self._cache = mock.patch.object(recall, "QUERY_CACHE", Path(self.tmp.name) / "q.db")
        self._cache.start()

    def tearDown(self):
        for p in (self._cache, self._mem, self._db):
            p.stop()
        self.tmp.cleanup()

    def test_keyword_only_when_embedding_fails(self):
        with mock.patch.object(recall, "_embed_remote", side_effect=RuntimeError("429")):
            rows = recall.recall("timestamps")
        self.assertEqual([r["chunk_id"] for r in rows], ["c0"])
        self.assertIsNone(rows[0]["score"])

    def test_embedding_failure_without_keyword_hit_raises(self):
        with mock.patch.object(recall, "_embed_remote", side_effect=RuntimeError("429")):
            with self.assertRaises(RuntimeError):
                recall.recall("zzzz")

    def test_semantic_path_unchanged(self):
        with mock.patch.object(recall, "_embed_remote", return_value=list(VEC)):
            rows = recall.recall("timestamps")
        self.assertEqual(rows[0]["chunk_id"], "c0")
        self.assertAlmostEqual(rows[0]["score"], 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
