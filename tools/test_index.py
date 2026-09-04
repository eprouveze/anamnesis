"""Unit tests for tools/index.py — offline index building and mock embeddings.

No API key or network required: runs entirely on the standard library.

    python -m unittest discover -s tools -p 'test_*.py' -v
"""
from __future__ import annotations

import math
import os
import sqlite3
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
import index  # noqa: E402
import recall  # noqa: E402


class MockEmbeddingTests(unittest.TestCase):
    def test_mock_embedding_shape_and_determinism(self):
        v1 = index.mock_embedding("test query text", dim=64)
        v2 = index.mock_embedding("test query text", dim=64)
        v3 = index.mock_embedding("completely different text", dim=64)

        self.assertEqual(len(v1), 64)
        self.assertEqual(v1, v2)
        self.assertNotEqual(v1, v3)

        # Verify unit vector normalization
        norm_sq = sum(x * x for x in v1)
        self.assertAlmostEqual(norm_sq, 1.0, places=5)

    def test_embed_all_falls_back_to_mock_when_no_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            vecs = index.embed_all(["hello", "world"])
            self.assertEqual(len(vecs), 2)
            self.assertEqual(len(vecs[0]), index.EMBEDDING_DIM)
            self.assertEqual(len(vecs[1]), index.EMBEDDING_DIM)

    def test_embed_all_supports_explicit_mock_env(self):
        with mock.patch.dict(os.environ, {"ANAMNESIS_MOCK_EMBEDDINGS": "1", "GEMINI_API_KEY": "some-key"}):
            vecs = index.embed_all(["hello"])
            self.assertEqual(len(vecs), 1)
            self.assertEqual(len(vecs[0]), index.EMBEDDING_DIM)

    def test_embed_all_supports_dummy_key(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "dummy"}):
            vecs = index.embed_all(["hello"])
            self.assertEqual(len(vecs), 1)
            self.assertEqual(len(vecs[0]), index.EMBEDDING_DIM)

    def test_embed_all_uses_client_when_key_present(self):
        fake_client = mock.MagicMock()
        fake_response = mock.MagicMock()
        fake_emb = mock.MagicMock()
        fake_emb.values = [0.1] * index.EMBEDDING_DIM
        fake_response.embeddings = [fake_emb]
        fake_client.models.embed_content.return_value = fake_response

        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "valid-api-key"}):
            with mock.patch.dict("sys.modules", {"google": mock.MagicMock(), "google.genai": mock.MagicMock()}):
                import google.genai
                google.genai.Client.return_value = fake_client
                vecs = index.embed_all(["test"])
                self.assertEqual(len(vecs), 1)
                self.assertEqual(vecs[0], [0.1] * index.EMBEDDING_DIM)


class BuildDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "data"
        self.out_db = Path(self.tmp.name) / "memory.db"
        self.query_cache = Path(self.tmp.name) / "queries.db"

        # Create a sample data directory structure
        (self.root / "memory").mkdir(parents=True)
        (self.root / "decisions").mkdir(parents=True)
        (self.root / "manifests").mkdir(parents=True)

        (self.root / "memory" / "sample.md").write_text(
            "---\nname: Sample Note\n---\nFirst paragraph about testing.\n\nSecond paragraph about search.",
            encoding="utf-8",
        )
        (self.root / "decisions" / "d-001.md").write_text(
            "---\nname: Decision Record\n---\nUse SQLite FTS5 for indexing.",
            encoding="utf-8",
        )
        (self.root / "manifests" / "manifest.json").write_text(
            "{\"item1\": {\"title\": \"Manifest Item\", \"desc\": \"Structured manifest entry\"}}",
            encoding="utf-8",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_build_succeeds_without_api_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            index.build(self.root, self.out_db)

        self.assertTrue(self.out_db.exists())
        conn = sqlite3.connect(str(self.out_db))

        # Check chunks
        chunks = conn.execute("SELECT chunk_id, title, content, embedding FROM chunks").fetchall()
        self.assertGreaterEqual(len(chunks), 3)
        for cid, title, content, blob in chunks:
            self.assertTrue(blob is not None)
            floats = struct.unpack(f"{len(blob)//4}f", blob)
            self.assertEqual(len(floats), index.EMBEDDING_DIM)

        # Check FTS5 index
        fts_hits = conn.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'testing'").fetchall()
        self.assertGreaterEqual(len(fts_hits), 1)

        # Check meta table
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        self.assertIn("chunks", meta)
        self.assertIn("built_at", meta)
        self.assertIn("embedding_dim", meta)
        self.assertEqual(int(meta["embedding_dim"]), index.EMBEDDING_DIM)
        conn.close()

    def test_recall_against_mock_built_db(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            index.build(self.root, self.out_db)

        # Recall without API key: keyword-only fallback
        with mock.patch.object(recall, "DB", self.out_db), \
             mock.patch.object(recall, "QUERY_CACHE", self.query_cache), \
             mock.patch.object(recall, "_MEM", {}), \
             mock.patch.dict(os.environ, {}, clear=True):
            res_kw = recall.recall("testing")
            self.assertGreaterEqual(len(res_kw), 1)
            self.assertIsNone(res_kw[0]["score"])

        # Recall with mock embeddings mode: semantic ranking
        with mock.patch.object(recall, "DB", self.out_db), \
             mock.patch.object(recall, "QUERY_CACHE", self.query_cache), \
             mock.patch.object(recall, "_MEM", {}), \
             mock.patch.dict(os.environ, {"ANAMNESIS_MOCK_EMBEDDINGS": "1"}):
            res_mock = recall.recall("testing")
            self.assertGreaterEqual(len(res_mock), 1)
            self.assertIsNotNone(res_mock[0]["score"])


if __name__ == "__main__":
    unittest.main()
