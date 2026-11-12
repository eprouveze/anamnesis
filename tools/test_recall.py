"""Tests for the retrieval core in recall.py. No API key, no network: the embedder is a stub.

    python -m unittest discover -s tools -p 'test_*.py' -v
"""
from __future__ import annotations

import contextlib
import io
import sqlite3
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import recall  # noqa: E402

VEC = [0.5, 0.25, 0.125, 1.0]
ORTHO = [1.0, 0.0, 0.0, 0.0]


def pack(values):
    return struct.pack(f"{len(values)}f", *values)


SCHEMA = """
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT, chunk_id TEXT UNIQUE NOT NULL, store_type TEXT NOT NULL,
    source_path TEXT NOT NULL, title TEXT, content TEXT NOT NULL, embedding BLOB,
    created_at TEXT, modified_at TEXT, project TEXT, topic_ids TEXT, weight REAL DEFAULT 1.0,
    indexed_at TEXT NOT NULL);
CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id, title, content, store_type, project, topic_ids,
    content=chunks, content_rowid=id, tokenize='porter unicode61');
"""


def make_db(path: Path, rows) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA)
    for i, (cid, title, body, stype, project, emb) in enumerate(rows, start=1):
        conn.execute(
            "INSERT INTO chunks (id, chunk_id, store_type, source_path, title, content, embedding,"
            " modified_at, project, topic_ids, weight, indexed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (i, cid, stype, "t.md", title, body, pack(emb) if emb else None,
             "2020-01-01", project, "", 1.0, "2020-01-01T00:00:00Z"))
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
    conn.commit()
    conn.close()


class HelperTests(unittest.TestCase):
    def test_cosine_identical_and_mismatched(self):
        self.assertAlmostEqual(recall.cosine(VEC, VEC), 1.0, places=6)
        self.assertEqual(recall.cosine(VEC, [1.0, 2.0]), 0.0)
        self.assertEqual(recall.cosine([], []), 0.0)

    def test_fts_query_is_safe_for_hostile_input(self):
        q = recall.fts_match_query('foo" OR NEAR(bar) * -baz:')
        self.assertNotIn("*", q)
        self.assertNotIn("(", q)
        self.assertEqual(recall.fts_match_query("!!! ???"), "")

    def test_short_words_dropped_unless_nothing_else(self):
        self.assertEqual(recall.fts_match_query("a to retry"), '"retry"')
        self.assertEqual(recall.fts_match_query("a to"), '"a" OR "to"')


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "memory.db"
        make_db(self.db, [
            ("c0", "timestamps", "store timestamps as UTC ISO strings", "memory", "demo", VEC),
            ("c1", "naming", "prefer kebab-case slugs for filenames", "decision", "demo", ORTHO),
            ("c2", "retries", "timestamps in retry logs help debugging", "decision", "other", ORTHO),
        ])
        self.conn = sqlite3.connect(str(self.db))

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_semantic_rerank_prefers_closest_vector(self):
        res, info = recall.search_chunks(self.conn, "timestamps", embed=lambda q: list(VEC))
        self.assertEqual(info["mode"], "hybrid")
        self.assertEqual(res[0]["chunk_id"], "c0")
        self.assertNotIn("embedding", res[0])

    def test_keyword_only_when_embedder_fails(self):
        def boom(_q):
            raise RuntimeError("429 quota key=sk-SECRET")
        res, info = recall.search_chunks(self.conn, "timestamps", embed=boom)
        self.assertEqual(info["mode"], "fts")
        self.assertEqual(info["embed_error"], "RuntimeError")  # class only, never the message
        self.assertEqual({r["chunk_id"] for r in res}, {"c0", "c2"})

    def test_no_embedder_is_keyword_only(self):
        res, info = recall.search_chunks(self.conn, "kebab")
        self.assertEqual(info["mode"], "fts")
        self.assertEqual([r["chunk_id"] for r in res], ["c1"])

    def test_filters_apply_in_sql_before_the_limit(self):
        res, _ = recall.search_chunks(self.conn, "timestamps", store_types=["decision"])
        self.assertEqual([r["chunk_id"] for r in res], ["c2"])
        res, _ = recall.search_chunks(self.conn, "timestamps", project="demo")
        self.assertEqual([r["chunk_id"] for r in res], ["c0"])

    def test_no_match_and_empty_query(self):
        self.assertEqual(recall.search_chunks(self.conn, "zzzzqqq")[0], [])
        res, info = recall.search_chunks(self.conn, "!!!")
        self.assertEqual((res, info["mode"]), ([], "none"))

    def test_limit(self):
        res, _ = recall.search_chunks(self.conn, "timestamps", limit=1)
        self.assertEqual(len(res), 1)

    def test_candidate_pool_is_best_keyword_matches(self):
        # The densest keyword match sits at the highest rowid, so a rowid-ordered LIMIT
        # would drop it; BM25 ordering must keep it in the pool.
        rows = [(f"d{i}", "t", "timestamps " * (10 if i == 59 else 1) + f"filler {i}", "memory", "demo", None)
                for i in range(60)]
        db = Path(self.tmp.name) / "big.db"
        make_db(db, rows)
        conn = sqlite3.connect(str(db))
        try:
            res, _ = recall.search_chunks(conn, "timestamps", limit=1, fts_limit=10)
        finally:
            conn.close()
        self.assertEqual(res[0]["chunk_id"], "d59")


class CliTests(unittest.TestCase):
    def test_cli_missing_db_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(recall.main(["--db", str(Path(tmp) / "nope.db"), "hello"]), 1)

    def test_cli_keyword_only_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "memory.db"
            make_db(db, [("c0", "timestamps", "store timestamps as UTC", "memory", "demo", None)])
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = recall.main(["--db", str(db), "--no-embed", "--json", "timestamps"])
        self.assertEqual(rc, 0)
        self.assertIn('"c0"', out.getvalue())

    def test_default_db_path_honours_env(self):
        import os
        from unittest import mock
        with mock.patch.dict(os.environ, {"ANAMNESIS_DB": "/x/y.db"}):
            self.assertEqual(recall.default_db_path(), Path("/x/y.db"))
        with mock.patch.dict(os.environ, {"ANAMNESIS_HOME": "/h"}, clear=True):
            self.assertEqual(recall.default_db_path(), Path("/h/memory.db"))


if __name__ == "__main__":
    unittest.main()
