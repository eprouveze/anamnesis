"""Tests for chat_import.py and `index.import_chat_exports` (mock embeddings, no network)."""
from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import chat_import  # noqa: E402
import index  # noqa: E402
from test_index import DIM, IndexCase, quiet  # noqa: E402


class ParseTests(IndexCase):
    def test_claude_export_keeps_only_human_messages(self):
        f = self.root / "conversations.json"
        f.write_text(json.dumps([{"name": "Trip", "created_at": "2026-01-02T03:04:05Z", "chat_messages": [
            {"sender": "human", "text": "plan a trip to lisbon"}, {"sender": "assistant", "text": "sure"}]},
            "junk"]))
        with quiet():
            convs = chat_import.auto_detect_and_parse(f)
        self.assertEqual(len(convs), 1)
        self.assertEqual([m["text"] for m in convs[0]["user_messages"]], ["plan a trip to lisbon"])

    def test_chatgpt_dir_and_bad_files(self):
        d = self.root / "exp" / "chatgpt"
        d.mkdir(parents=True)
        (d / "a.json").write_text(json.dumps({"title": "T", "messages": [{"role": "user", "content": "hello there"}]}))
        (d / "b.json").write_text("{broken")
        (d / "c.json").write_text("[]")
        with quiet():
            convs = chat_import.auto_detect_and_parse(self.root / "exp")
        self.assertEqual([c["title"] for c in convs], ["T"])

    def test_gemini_takeout_activity(self):
        d = self.root / "exp" / "gemini"
        d.mkdir(parents=True)
        (d / "MyActivity.json").write_text(json.dumps([{"title": "Prompted what is fts5", "time": "2026-01-01T00:00:00Z",
                                                        "safeHtmlItem": [{"html": "<p>x</p>"}]}]))
        with quiet():
            convs = chat_import.auto_detect_and_parse(self.root / "exp")
        self.assertEqual(convs[0]["title"], "what is fts5")

    def test_long_conversation_is_split_not_truncated_and_ids_are_stable(self):
        msgs = [{"text": f"message {i} " + "w" * 900, "timestamp": ""} for i in range(10)]
        conv = dict(platform="claude", title="Long", date="", user_messages=msgs, source_file="/x/c.json")
        chunks = chat_import.conversations_to_chunks([conv], project="chat")
        self.assertGreater(len(chunks), 1)
        joined = "\n".join(c["content"] for c in chunks)
        self.assertTrue(all(f"message {i} " in joined for i in range(10)))
        self.assertEqual([c["chunk_id"] for c in chunks], [c["chunk_id"] for c in chat_import.conversations_to_chunks([conv])])
        self.assertEqual(len({c["chunk_id"] for c in chunks}), len(chunks))


class ImportIntoIndexTests(IndexCase):
    def make_export(self):
        d = self.root / "exp" / "claude"
        d.mkdir(parents=True)
        (d / "conversations.json").write_text(json.dumps([{"name": "Gardening", "created_at": "2026-01-02T00:00:00Z",
            "chat_messages": [{"sender": "human", "text": "how do I prune an espalier pear tree"}]}]))
        return self.root / "exp"

    def test_import_is_idempotent_searchable_and_survives_rebuild(self):
        exp = self.make_export()
        with quiet():
            index.import_chat_exports(exp)
            index.import_chat_exports(exp)
        self.assertEqual(self.rows("SELECT COUNT(*) FROM chunks WHERE store_type='chat_export'")[0][0], 1)
        self.assertEqual(len(self.rows("SELECT 1 FROM chunks_fts WHERE chunks_fts MATCH 'espalier'")), 1)
        self.assertEqual(self.rows("SELECT length(embedding) FROM chunks")[0][0], DIM * 4)
        self.write("memory/a.md", "# A\nSome ordinary memory text that is long enough to chunk.")
        with quiet():
            index.build_index(config=self.config([{"root": str(self.data / "memory")}]))
        self.assertEqual(self.rows("SELECT COUNT(*) FROM chunks WHERE store_type='chat_export'")[0][0], 1)

    def test_secrets_are_redacted_before_chunking_and_embedding(self):
        secret = "sk-proj-" + "AbC123dEf456" * 3
        token = "anm_" + "Ab3dEf7hIj9kLm1nOpQr"
        d = self.root / "exp" / "claude"
        d.mkdir(parents=True)
        (d / "conversations.json").write_text(json.dumps([{"name": f"debug {secret}", "created_at": "2026-01-02T00:00:00Z",
            "chat_messages": [{"sender": "human", "text": f"my key is {secret} and token {token} please rotate zucchini"}]}]))
        with quiet():
            index.import_chat_exports(self.root / "exp")
        rows = self.rows("SELECT title, content FROM chunks WHERE store_type='chat_export'")
        self.assertEqual(len(rows), 1)
        blob = " ".join(rows[0])
        self.assertNotIn(secret, blob)
        self.assertNotIn(token, blob)
        self.assertIn("[REDACTED:API_KEY]", blob)
        self.assertIn("[REDACTED:ANAMNESIS_TOKEN]", blob)
        self.assertIn("zucchini", blob)
        self.assertEqual(self.rows("SELECT 1 FROM chunks_fts WHERE chunks_fts MATCH 'zucchini'")[0][0], 1)

    def test_conversations_to_chunks_redacts_directly(self):
        secret = "ghp_" + "a1" * 15
        convs = [{"platform": "claude", "title": "t", "date": "2026-01-01", "source_file": "f",
                  "user_messages": [{"text": f"token {secret}"}]}]
        chunks = chat_import.conversations_to_chunks(convs)
        self.assertTrue(chunks)
        self.assertNotIn(secret, json.dumps(chunks, default=str))


if __name__ == "__main__":
    unittest.main()
