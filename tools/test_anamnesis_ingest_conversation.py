#!/usr/bin/env python3
# Smoke tests for ingest_conversation.py. Stdlib only, no network.
# Builds a throwaway SQLite DB with the same chunks/chunks_fts schema as the real
# index (never touches a real database) and exercises:
#   - dry-run mode never writes
#   - --live mode inserts + is idempotent on re-run (same input -> no duplicate)
#   - secret redaction fires on realistic key formats
#   - missing input file exits non-zero without creating a DB
#
# Run: python3 -m unittest discover -s tools -p 'test_*.py'
from __future__ import annotations

import importlib.util
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

FAKE_GITHUB = "gh" + "p_" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"  # built at runtime so scanners do not flag a fake
FAKE_SLACK = "xo" + "xb-" + "1234567890-1234567890-" + "AbCdEfGhIjKlMnOpQrSt"  # built at runtime so scanners do not flag a fake
FAKE_STRIPE = "sk_" + "live_" + "51" + "ABCDEFGHIJKLMNOPQRSTUVWX"  # built at runtime so scanners do not flag a fake

SCRIPT_PATH = Path(__file__).resolve().parent / "ingest_conversation.py"

SCHEMA_SQL = """
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id TEXT UNIQUE NOT NULL,
    store_type TEXT NOT NULL,
    source_path TEXT NOT NULL,
    title TEXT,
    content TEXT NOT NULL,
    embedding BLOB,
    created_at TEXT,
    modified_at TEXT,
    project TEXT,
    topic_ids TEXT,
    weight REAL DEFAULT 1.0,
    indexed_at TEXT NOT NULL,
    tags TEXT DEFAULT '',
    priority TEXT DEFAULT 'important',
    load_when TEXT DEFAULT '',
    retirement_status TEXT DEFAULT 'active',
    metadata TEXT DEFAULT ''
);
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    chunk_id, title, content, store_type, project, topic_ids
);
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, chunk_id, title, content, store_type, project, topic_ids)
    VALUES (new.id, new.chunk_id, new.title, new.content, new.store_type, new.project, new.topic_ids);
END;
"""

SAMPLE_CONVO = f"""## User
Here is a fake stripe live key {FAKE_STRIPE} for testing \
redaction, and a fake github token {FAKE_GITHUB}.

## Assistant
Got it, I will not leak that.
"""


def load_module():
    spec = importlib.util.spec_from_file_location("ingest_conv", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


class RedactionAndParsingTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_redacts_realistic_keys(self):
        redacted, count = self.mod.redact_content(SAMPLE_CONVO)
        self.assertEqual(count, 2)
        self.assertNotIn(FAKE_STRIPE, redacted)
        self.assertNotIn(FAKE_GITHUB, redacted)
        self.assertIn("[REDACTED:PAYMENT_KEY]", redacted)
        self.assertIn("[REDACTED:GITHUB_TOKEN]", redacted)

    def test_redacts_other_key_shapes(self):
        # Key shapes that must never survive redaction (all dummies).
        dummies = {
            "SLACK_TOKEN": FAKE_SLACK,
            "API_KEY": "sk-proj-" + "AbC123dEf456" * 3,
            "AWS_ACCESS_KEY": "AKIA" + "ABCDEFGHIJKLMNOP",
            "JWT": "eyJhbGciOiJIUzI1NiJ9.eyJyb2xlIjoic2VydmljZV9yb2xlIn0.x_y-z" + "Q" * 40,
            "BEARER_TOKEN": "Bearer " + "abcDEF0123456789" * 2,
            "ANAMNESIS_TOKEN": "anm_" + "Ab3dEf7hIj9kLm1nOpQr",
        }
        for label, dummy in dummies.items():
            redacted, count = self.mod.redact_content(f"token {dummy} here")
            self.assertNotIn(dummy, redacted, label)
            self.assertIn(f"[REDACTED:{label}]", redacted, label)

    def test_more_vendor_shapes_and_generic_assignments(self):
        for dummy in ["sk-ant-api03-" + "Zz9" * 12, "AIza" + "A" * 35, "ghp_" + "a1" * 15, "github_pat_" + "B2" * 15,
                      "hf_" + "c" * 34, "gsk_" + "d4" * 12, "rk_live_" + "e5" * 10, "ASIA" + "ABCDEFGHIJKLMNOP"]:
            redacted, count = self.mod.redact_content(f"x {dummy} y")
            self.assertNotIn(dummy, redacted, dummy)
            self.assertGreaterEqual(count, 1, dummy)
        redacted, _ = self.mod.redact_content('config api_key = "hunter2hunter2" and API-KEY: abcdef123456')
        self.assertNotIn("hunter2hunter2", redacted)
        self.assertNotIn("abcdef123456", redacted)
        generic = {"PRIVATE_KEY", "JWT", "AWS_ACCESS_KEY", "GITHUB_TOKEN", "PAYMENT_KEY", "SLACK_TOKEN",
                   "API_KEY", "ANAMNESIS_TOKEN", "BEARER_TOKEN"}
        self.assertEqual({label for _, label in self.mod.REDACTION_PATTERNS}, generic)

    def test_redacts_private_key_block(self):
        block = "-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJBAKj34GkxFhD90vcNLYLInFEX6Ppy\n-----END RSA PRIVATE KEY-----"
        redacted, count = self.mod.redact_content("key:\n" + block)
        self.assertEqual(count, 1)
        self.assertNotIn("MIIBOgIBAAJBAKj34GkxFhD90vcNLYLInFEX6Ppy", redacted)

    def test_parses_turn_markers(self):
        turns = self.mod.parse_turns(SAMPLE_CONVO)
        self.assertIsNotNone(turns)
        roles = [role for role, _ in turns]
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)

    def test_no_markers_returns_none(self):
        self.assertIsNone(self.mod.parse_turns("just a plain blob of text, no labels here."))

    def test_chunk_id_stable_across_runs(self):
        # chunk_id must not depend on wall-clock time (idempotency requirement).
        id1 = self.mod.build_chunk_id("other", "My Title", 0, "same content")
        id2 = self.mod.build_chunk_id("other", "My Title", 0, "same content")
        self.assertEqual(id1, id2)


class CliIntegrationTests(unittest.TestCase):
    """Exercise the CLI end-to-end against a throwaway DB (never the real one)."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        self.db_path = self.tmp / "test-memory.db"
        conn = sqlite3.connect(str(self.db_path))
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        conn.close()

        self.convo_path = self.tmp / "convo.md"
        self.convo_path.write_text(SAMPLE_CONVO, encoding="utf-8")

    def tearDown(self):
        self.tmpdir.cleanup()

    def run_cli(self, *extra_args):
        return subprocess.run(
            [sys.executable, str(SCRIPT_PATH), str(self.convo_path), *extra_args],
            capture_output=True, text=True,
        )

    def test_dry_run_writes_nothing(self):
        result = self.run_cli("--db", str(self.db_path))
        self.assertEqual(result.returncode, 0)
        self.assertIn("DRY RUN", result.stdout)
        conn = sqlite3.connect(str(self.db_path))
        count = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE store_type='conversation_clip'"
        ).fetchone()[0]
        conn.close()
        self.assertEqual(count, 0, "dry-run must never write to the DB")

    def test_dry_run_output_is_redacted(self):
        result = self.run_cli("--db", str(self.db_path))
        self.assertNotIn(FAKE_STRIPE, result.stdout)
        self.assertIn("[REDACTED:", result.stdout)

    def test_live_inserts_and_is_idempotent(self):
        first = self.run_cli("--live", "--db", str(self.db_path))
        self.assertEqual(first.returncode, 0)
        self.assertIn("inserted", first.stdout)

        conn = sqlite3.connect(str(self.db_path))
        count_after_first = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE store_type='conversation_clip'"
        ).fetchone()[0]
        fts_count_after_first = conn.execute(
            "SELECT COUNT(*) FROM chunks_fts WHERE store_type='conversation_clip'"
        ).fetchone()[0]
        conn.close()
        self.assertEqual(count_after_first, 1)
        self.assertEqual(fts_count_after_first, 1, "FTS mirror trigger must fire")

        second = self.run_cli("--live", "--db", str(self.db_path))
        self.assertEqual(second.returncode, 0)
        self.assertIn("skipped (exists)", second.stdout)

        conn = sqlite3.connect(str(self.db_path))
        count_after_second = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE store_type='conversation_clip'"
        ).fetchone()[0]
        conn.close()
        self.assertEqual(count_after_second, 1, "re-running must be a no-op (idempotent)")

    def test_live_refuses_to_create_missing_db(self):
        missing_db = self.tmp / "does-not-exist.db"
        result = self.run_cli("--live", "--db", str(missing_db))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(missing_db.exists(), "must never silently create a new Anamnesis DB")

    def test_live_refuses_a_non_anamnesis_database(self):
        junk = self.tmp / "junk.db"
        sqlite3.connect(str(junk)).close()
        result = self.run_cli("--live", "--db", str(junk))
        self.assertNotEqual(result.returncode, 0)

    def test_live_works_against_a_real_index_schema(self):
        # The schema produced by index.init_db must accept clips and keep FTS in sync.
        sys.path.insert(0, str(SCRIPT_PATH.parent))
        import index
        db = self.tmp / "real.db"
        index.init_db(db).close()
        result = self.run_cli("--live", "--db", str(db))
        self.assertEqual(result.returncode, 0, result.stderr)
        conn = sqlite3.connect(str(db))
        n = conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'redaction'").fetchone()[0]
        conn.close()
        self.assertEqual(n, 1)

    def test_missing_input_file_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), str(self.tmp / "nope.md"), "--db", str(self.db_path)],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
