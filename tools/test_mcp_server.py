"""Tests for tools/mcp_server.py: auth, scopes, path confinement, tool surface.

No network, no API keys: tokens are generated at runtime, the embedder is disabled, and
the HTTP app is driven in-process with Starlette's TestClient.

    python -m unittest tools/test_mcp_server.py -v
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
import mcp_server as m  # noqa: E402
from fastmcp.exceptions import ToolError  # noqa: E402
from fastmcp.server.auth import AccessToken  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

EXPECTED_TOOLS = {
    "brief", "recall", "recall_precedent", "store_memory", "store_decision", "synthesize", "trace",
    "get_mistakes", "get_topic", "get_decisions", "get_recent", "session_context", "list_files",
    "read_file", "get_capabilities", "ontology_audit", "ontology_impact", "health",
}

MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}

SCHEMA = """
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT, chunk_id TEXT UNIQUE NOT NULL, store_type TEXT NOT NULL,
    source_path TEXT NOT NULL, title TEXT, content TEXT NOT NULL, embedding BLOB,
    created_at TEXT, modified_at TEXT, project TEXT, topic_ids TEXT, weight REAL DEFAULT 1.0,
    indexed_at TEXT NOT NULL);
CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id, title, content, store_type, project, topic_ids,
    content=chunks, content_rowid=id, tokenize='porter unicode61');
CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE source_meta (source_path TEXT PRIMARY KEY, mtime REAL NOT NULL, store_type TEXT NOT NULL,
    project TEXT NOT NULL);
"""
TRIGGER = """
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, chunk_id, title, content, store_type, project, topic_ids)
  VALUES (new.id, new.chunk_id, new.title, new.content, new.store_type, new.project, new.topic_ids);
END;
"""


def rpc(client: TestClient, tool: str, args: dict | None = None, headers: dict | None = None, params=None):
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": tool, "arguments": args or {}}}
    return client.post("/mcp", json=body, headers={**MCP_HEADERS, **(headers or {})}, params=params)


def tool_result(resp) -> dict:
    """The JSON-RPC `result` of a tools/call response (plain JSON or a single SSE event)."""
    text = resp.text
    if not text.lstrip().startswith("{"):
        text = next(l[5:] for l in text.splitlines() if l.startswith("data:"))
    return json.loads(text)["result"]


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


class Env(unittest.TestCase):
    """Points the module's paths at a temp dir and clears auth-related env flags."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.root = root
        self.data = root / "data"
        self.data.mkdir()
        patches = [
            mock.patch.object(m, "HOME", root / "home"),
            mock.patch.object(m, "DATA_DIR", self.data),
            mock.patch.object(m, "DB_PATH", root / "memory.db"),
            mock.patch.object(m, "CLIENTS_PATH", root / "clients.json"),
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch.object(m, "_embedder", lambda: None),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        for var in ("ANAMNESIS_ALLOW_QUERY_TOKEN", "ANAMNESIS_ALLOW_LOCALHOST_NOAUTH",
                    "ANAMNESIS_LOCALHOST_WRITE", "ANAMNESIS_OAUTH_PROVIDER", "ANAMNESIS_FILE_ROOTS"):
            os.environ.pop(var, None)
        m._clients_cache.update(key=None, clients=[])
        self.addCleanup(self.tmp.cleanup)

    def make_db(self, trigger: bool = True):
        conn = sqlite3.connect(str(m.DB_PATH))
        conn.executescript(SCHEMA + (TRIGGER if trigger else ""))
        conn.commit()
        conn.close()


# --------------------------------------------------------------------------------------
# Client registry
# --------------------------------------------------------------------------------------
class ClientStoreTests(Env):
    def test_only_the_hash_is_stored(self):
        token = m.add_client("laptop")
        raw = m.CLIENTS_PATH.read_text()
        self.assertNotIn(token, raw)
        self.assertNotIn(token[4:], raw)
        rec = json.loads(raw)["clients"][0]
        self.assertEqual(rec["token_sha256"], m.hash_token(token))
        self.assertEqual(set(rec), {"name", "token_sha256", "scopes", "created_at"})
        self.assertEqual(rec["scopes"], ["read", "write"])

    def test_file_is_private(self):
        m.add_client("laptop")
        mode = stat.S_IMODE(m.CLIENTS_PATH.stat().st_mode)
        self.assertEqual(mode & 0o077, 0)

    def test_read_only_scope_and_duplicate_names(self):
        m.add_client("viewer", read_only=True)
        self.assertEqual(m.load_clients()[0]["scopes"], ["read"])
        with self.assertRaises(ValueError):
            m.add_client("viewer")
        with self.assertRaises(ValueError):
            m.add_client("bad name!")

    def test_find_client_uses_constant_time_compare(self):
        token = m.add_client("laptop")
        with mock.patch.object(m.hmac, "compare_digest", wraps=m.hmac.compare_digest) as cd:
            self.assertEqual(m.find_client(token)["name"], "laptop")
            self.assertTrue(cd.called)
        self.assertIsNone(m.find_client(token + "x"))
        self.assertIsNone(m.find_client(""))

    def test_tokens_are_unique_and_prefixed(self):
        a, b = m.add_client("a"), m.add_client("b")
        self.assertNotEqual(a, b)
        self.assertTrue(a.startswith("anm_") and len(a) > 40)

    def test_revoke(self):
        token = m.add_client("laptop")
        self.assertTrue(m.revoke_client("laptop"))
        self.assertFalse(m.revoke_client("laptop"))
        self.assertIsNone(m.find_client(token))

    def test_corrupt_or_malformed_file_fails_closed(self):
        m.CLIENTS_PATH.write_text("{not json")
        m.CLIENTS_PATH.chmod(0o600)
        self.assertEqual(m.load_clients(), [])
        m.CLIENTS_PATH.write_text(json.dumps({"clients": [
            {"name": "x", "token_sha256": "short", "scopes": ["read"]},
            {"name": "y", "token_sha256": "0" * 64, "scopes": ["admin"]},
        ]}))
        m.CLIENTS_PATH.chmod(0o600)
        m._clients_cache["key"] = None
        self.assertEqual(m.load_clients(), [])

    def test_cli_add_prints_token_once_and_stores_hash(self):
        import contextlib
        import io
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = m.main(["add-client", "cli-client", "--read-only"])
        self.assertEqual(rc, 0)
        token = out.getvalue().strip()
        self.assertTrue(token.startswith("anm_"))
        self.assertNotIn(token, err.getvalue())
        self.assertNotIn(token, m.CLIENTS_PATH.read_text())
        self.assertEqual(m.find_client(token)["scopes"], ["read"])
        out2 = io.StringIO()
        with contextlib.redirect_stdout(out2):
            m.main(["list-clients"])
        self.assertNotIn(token, out2.getvalue())
        self.assertIn("cli-client", out2.getvalue())


# --------------------------------------------------------------------------------------
# HTTP authentication
# --------------------------------------------------------------------------------------
class HttpAuthTests(Env):
    def setUp(self):
        super().setUp()
        self.rw = m.add_client("writer")
        self.ro = m.add_client("viewer", read_only=True)
        self.addCleanup(setattr, m, "_TRANSPORT", None)
        self.addCleanup(setattr, m.mcp, "auth", None)
        self.app = m.build_http_app(host_origin_protection=False)

    def test_header_token_accepted(self):
        with TestClient(self.app) as c:
            r = rpc(c, "health", headers=bearer(self.rw))
        self.assertEqual(r.status_code, 200)
        self.assertIn('"status"', r.text)

    def test_missing_token_rejected(self):
        with TestClient(self.app) as c:
            self.assertEqual(rpc(c, "health").status_code, 401)

    def test_wrong_token_rejected(self):
        with TestClient(self.app) as c:
            self.assertEqual(rpc(c, "health", headers=bearer("anm_" + "x" * 43)).status_code, 401)
            self.assertEqual(rpc(c, "health", headers={"Authorization": "Basic abc"}).status_code, 401)

    def test_query_token_rejected_by_default(self):
        with TestClient(self.app) as c:
            r = rpc(c, "health", params={"token": self.rw})
        self.assertEqual(r.status_code, 401)
        self.assertEqual(r.json()["error"], "query_token_disabled")
        self.assertNotIn(self.rw, r.text)

    def test_query_token_accepted_only_with_flag_and_warns(self):
        with mock.patch.dict(os.environ, {"ANAMNESIS_ALLOW_QUERY_TOKEN": "1"}), \
                self.assertLogs("anamnesis", level="WARNING") as logs, TestClient(self.app) as c:
            r = rpc(c, "health", params={"token": self.rw})
            bad = rpc(c, "health", params={"token": "anm_" + "y" * 43})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(bad.status_code, 401)
        joined = "\n".join(logs.output)
        self.assertIn("SECURITY", joined)
        self.assertNotIn(self.rw, joined)  # fingerprint only, never the value

    def test_header_wins_over_query_token(self):
        with mock.patch.dict(os.environ, {"ANAMNESIS_ALLOW_QUERY_TOKEN": "1"}), TestClient(self.app) as c:
            r = rpc(c, "health", headers=bearer("anm_" + "z" * 43), params={"token": self.rw})
        self.assertEqual(r.status_code, 401)

    def test_query_token_rejected_even_beside_valid_header(self):
        with TestClient(self.app) as c:
            r = rpc(c, "health", headers=bearer(self.rw), params={"token": self.rw})
        self.assertEqual(r.status_code, 401)

    def test_read_only_client_cannot_write(self):
        with TestClient(self.app) as c:
            for tool, args in (
                ("store_memory", {"type": "user", "title": "t", "content": "c"}),
                ("store_decision", {"title": "t", "context": "c", "decision": "d",
                                    "alternatives": "a", "rationale": "r"}),
            ):
                r = rpc(c, tool, args, headers=bearer(self.ro))
                self.assertIn("read-only", r.text)
                self.assertIn('"isError":true', r.text.replace(" ", ""))
            self.assertFalse((self.data / "memory").exists())
            self.assertFalse((self.data / "decisions").exists())
            self.assertIn('"status"', rpc(c, "health", headers=bearer(self.ro)).text)  # reads work

    def test_write_client_can_write(self):
        with TestClient(self.app) as c:
            r = rpc(c, "store_memory", {"type": "user", "title": "likes tea", "content": "tea"},
                    headers=bearer(self.rw))
        self.assertIn("user_likes_tea.md", r.text)
        self.assertTrue((self.data / "memory" / "user_likes_tea.md").exists())

    def test_revoked_token_stops_working_without_restart(self):
        with TestClient(self.app) as c:
            self.assertEqual(rpc(c, "health", headers=bearer(self.rw)).status_code, 200)
            m.revoke_client("writer")
            self.assertEqual(rpc(c, "health", headers=bearer(self.rw)).status_code, 401)

    def test_unauthenticated_health_route_is_minimal(self):
        with TestClient(self.app) as c:
            body = c.get("/health").json()
        self.assertEqual(set(body), {"status", "uptime_s", "index_updated_at", "index_age_h", "index_stale"})

    def test_token_in_tool_output_never_logged(self):
        with TestClient(self.app) as c, self.assertLogs("anamnesis", level="INFO") as logs:
            rpc(c, "health")
            rpc(c, "health", headers=bearer("anm_" + "w" * 43))
        self.assertNotIn("anm_" + "w" * 43, "\n".join(logs.output))


class PreAuthUnitTests(Env):
    """PreAuthMiddleware against a trivial inner app that records the Authorization header."""

    def run_request(self, *, client=("203.0.113.9", 1234), headers=None, query=b"", path="/mcp"):
        seen = {}

        async def inner(scope, receive, send):
            seen["auth"] = dict(scope["headers"]).get(b"authorization")
            seen["query"] = scope["query_string"]
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        status = {}

        async def send(msg):
            if msg["type"] == "http.response.start":
                status["code"] = msg["status"]

        scope = {"type": "http", "path": path, "query_string": query, "client": client,
                 "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]}
        app = m.PreAuthMiddleware(inner)

        async def go():
            await app(scope, lambda: asyncio.sleep(0), send)
        asyncio.run(go())
        return status.get("code"), seen

    def test_localhost_noauth_is_off_by_default(self):
        code, seen = self.run_request(client=("127.0.0.1", 5))
        self.assertIsNone(seen["auth"])

    def test_localhost_noauth_injects_only_for_direct_loopback(self):
        with mock.patch.dict(os.environ, {"ANAMNESIS_ALLOW_LOCALHOST_NOAUTH": "1"}):
            _, seen = self.run_request(client=("127.0.0.1", 5))
            self.assertEqual(seen["auth"], b"Bearer " + m._LOCALHOST_TOKEN.encode())
            _, seen = self.run_request(client=("203.0.113.9", 5))
            self.assertIsNone(seen["auth"])
            _, seen = self.run_request(client=("127.0.0.1", 5), headers={"X-Forwarded-For": "198.51.100.1"})
            self.assertIsNone(seen["auth"])  # a local reverse proxy must not grant access

    def test_other_paths_untouched(self):
        with mock.patch.dict(os.environ, {"ANAMNESIS_ALLOW_LOCALHOST_NOAUTH": "1"}):
            _, seen = self.run_request(client=("127.0.0.1", 5), path="/health")
            self.assertIsNone(seen["auth"])

    def test_query_token_moved_to_header_and_stripped_when_enabled(self):
        with mock.patch.dict(os.environ, {"ANAMNESIS_ALLOW_QUERY_TOKEN": "1"}):
            _, seen = self.run_request(query=b"a=1&token=abc123&b=2")
        self.assertEqual(seen["auth"], b"Bearer abc123")
        self.assertEqual(seen["query"], b"a=1&b=2")

    def test_query_token_refused_when_disabled(self):
        code, seen = self.run_request(query=b"token=abc123")
        self.assertEqual(code, 401)
        self.assertNotIn("auth", seen)  # inner app never reached

    def test_localhost_verifier_scopes(self):
        v = m.ClientTokenVerifier()
        self.assertIsNone(asyncio.run(v.verify_token(m._LOCALHOST_TOKEN)))  # flag off
        with mock.patch.dict(os.environ, {"ANAMNESIS_ALLOW_LOCALHOST_NOAUTH": "1"}):
            at = asyncio.run(v.verify_token(m._LOCALHOST_TOKEN))
            self.assertEqual(at.scopes, ["read"])
            self.assertIsNone(asyncio.run(v.verify_token(m._LOCALHOST_TOKEN + "x")))
        with mock.patch.dict(os.environ, {"ANAMNESIS_ALLOW_LOCALHOST_NOAUTH": "1",
                                          "ANAMNESIS_LOCALHOST_WRITE": "1"}):
            self.assertEqual(asyncio.run(v.verify_token(m._LOCALHOST_TOKEN)).scopes, ["read", "write"])


class PrincipalTests(Env):
    def test_only_explicit_stdio_marker_is_trusted(self):
        with self.assertRaises(ToolError):  # unset marker (in-process, tests, misuse) fails closed
            m.resolve_principal()
        with mock.patch.object(m, "_TRANSPORT", "http"), self.assertRaises(ToolError):
            m.resolve_principal()
        with mock.patch.object(m, "_TRANSPORT", "stdio"):
            self.assertEqual(m.resolve_principal().name, "stdio")

    def test_forged_marker_does_not_grant_access(self):
        forged = AccessToken(token="t", client_id="evil", scopes=["read", "write"],
                             claims={"anamnesis_marker": "guess", "anamnesis_client": "admin"})
        with self.assertRaises(ToolError):
            m.principal_from_access_token(forged)

    def test_oauth_allowlist_enforced_server_side(self):
        env = {"ANAMNESIS_OAUTH_PROVIDER": "google", "ANAMNESIS_OAUTH_ALLOWED_EMAILS": "Me@Example.com"}
        ok = AccessToken(token="t", client_id="1", scopes=[],
                         claims={"email": "me@example.com", "email_verified": True})
        other = AccessToken(token="t", client_id="2", scopes=[],
                            claims={"email": "other@example.com", "email_verified": True})
        unverified = AccessToken(token="t", client_id="3", scopes=[],
                                 claims={"email": "me@example.com", "email_verified": False})
        with mock.patch.dict(os.environ, env):
            p = m.principal_from_access_token(ok)
            self.assertEqual(p.via, "oauth")
            self.assertFalse(p.can_write)  # OAuth defaults to read-only
            for bad in (other, unverified):
                with self.assertRaises(ToolError):
                    m.principal_from_access_token(bad)
        with mock.patch.dict(os.environ, {**env, "ANAMNESIS_OAUTH_SCOPES": "read,write"}):
            self.assertTrue(m.principal_from_access_token(ok).can_write)

    def test_oauth_token_rejected_when_oauth_not_configured(self):
        at = AccessToken(token="t", client_id="1", scopes=[],
                         claims={"email": "me@example.com", "email_verified": True})
        with self.assertRaises(ToolError):
            m.principal_from_access_token(at)


class OAuthConfigTests(Env):
    def base_env(self, **extra):
        env = {"ANAMNESIS_OAUTH_PROVIDER": "google", "ANAMNESIS_OAUTH_CLIENT_ID": "id.apps.example",
               "ANAMNESIS_OAUTH_CLIENT_SECRET": "not-a-real-secret",
               "ANAMNESIS_OAUTH_BASE_URL": "https://memory.example.test",
               "ANAMNESIS_OAUTH_JWT_KEY": "unit-test-signing-key", "ANAMNESIS_OAUTH_ALLOWED_EMAILS": "me@example.com"}
        env.update(extra)
        return env

    def test_disabled_by_default(self):
        self.assertIsNone(m.build_oauth_provider())
        self.assertIsInstance(m.build_auth(), m.ClientTokenVerifier)

    def test_empty_allowlist_refuses_to_start(self):
        with mock.patch.dict(os.environ, self.base_env(ANAMNESIS_OAUTH_ALLOWED_EMAILS="")):
            with self.assertRaises(SystemExit):
                m.build_oauth_provider()

    def test_missing_credentials_refuse_to_start(self):
        with mock.patch.dict(os.environ, self.base_env(ANAMNESIS_OAUTH_CLIENT_SECRET="")):
            with self.assertRaises(SystemExit):
                m.build_oauth_provider()

    def test_unknown_provider_refused(self):
        with mock.patch.dict(os.environ, self.base_env(ANAMNESIS_OAUTH_PROVIDER="nope")):
            with self.assertRaises(SystemExit):
                m.build_oauth_provider()

    def test_github_provider_builds_and_uses_login_allowlist(self):
        from key_value.aio.stores.memory import MemoryStore
        env = self.base_env(ANAMNESIS_OAUTH_PROVIDER="github", ANAMNESIS_OAUTH_ALLOWED_LOGINS="octo-owner")
        with mock.patch.dict(os.environ, env):
            self.assertEqual(m.oauth_allowlist(), {"octo-owner"})
            self.assertIsNotNone(m.build_oauth_provider(MemoryStore()))
            ok = AccessToken(token="t", client_id="1", scopes=[], claims={"login": "Octo-Owner"})
            bad = AccessToken(token="t", client_id="1", scopes=[], claims={"login": "someone-else"})
            self.assertEqual(m.principal_from_access_token(ok).via, "oauth")
            with self.assertRaises(ToolError):
                m.principal_from_access_token(bad)

    def test_oauth_discovery_routes_served_and_mcp_still_protected(self):
        from key_value.aio.stores.memory import MemoryStore
        with mock.patch.dict(os.environ, self.base_env()):
            self.addCleanup(setattr, m, "_TRANSPORT", None)
            self.addCleanup(setattr, m.mcp, "auth", None)
            app = m.build_http_app(host_origin_protection=False, oauth_storage=MemoryStore())
            with TestClient(app) as c:
                meta = c.get("/.well-known/oauth-authorization-server")
                self.assertEqual(meta.status_code, 200)
                self.assertIn("authorization_endpoint", meta.json())
                self.assertEqual(rpc(c, "health").status_code, 401)

    def test_static_tokens_still_verify_alongside_oauth(self):
        from fastmcp.server.auth import MultiAuth
        from key_value.aio.stores.memory import MemoryStore
        token = m.add_client("headless")
        with mock.patch.dict(os.environ, self.base_env()):
            auth = m.build_auth(oauth_storage=MemoryStore())
            self.assertIsInstance(auth, MultiAuth)
            self.assertEqual(auth.required_scopes, [])  # IdP scopes must not gate static tokens
            at = asyncio.run(auth.verify_token(token))
            self.assertIsNotNone(at)
            self.assertEqual(m.principal_from_access_token(at).name, "headless")
            self.assertIsNone(asyncio.run(auth.verify_token("anm_" + "q" * 43)))


# --------------------------------------------------------------------------------------
# Tool surface
# --------------------------------------------------------------------------------------
class ToolSurfaceTests(Env):
    def test_registered_tools(self):
        names = {t.name for t in asyncio.run(m.mcp.list_tools())}
        self.assertEqual(names, EXPECTED_TOOLS)

    def test_no_operator_specific_arguments(self):
        home = str(Path.home()).lower()
        for t in asyncio.run(m.mcp.list_tools()):
            params = json.dumps(t.parameters).lower()
            for needle in ("token", "password", "secret"):
                self.assertNotIn(needle, params, t.name)
            self.assertNotIn(home, params + (t.description or "").lower(), t.name)


# --------------------------------------------------------------------------------------
# read_file / list_files confinement
# --------------------------------------------------------------------------------------
class HardeningTests(Env):
    def test_every_tool_is_classified_and_unknown_requires_write(self):
        names = {t.name for t in asyncio.run(m.mcp.list_tools())}
        self.assertEqual(m.READ_TOOLS | m.WRITE_TOOLS, names)
        self.assertEqual(m.required_scope("health"), "read")
        self.assertEqual(m.required_scope("store_memory"), "write")
        self.assertEqual(m.required_scope("brand_new_tool"), "write")

    def test_unclassified_tool_blocked_for_read_only_client(self):
        @m.mcp.tool
        def zz_new_tool() -> str:
            return "x"
        self.addCleanup(m.mcp.local_provider.remove_tool, "zz_new_tool")
        ro = m.add_client("viewer", read_only=True)
        rw = m.add_client("writer")
        self.addCleanup(setattr, m, "_TRANSPORT", None)
        self.addCleanup(setattr, m.mcp, "auth", None)
        app = m.build_http_app(host_origin_protection=False)
        with TestClient(app) as c:
            self.assertIn("read-only", rpc(c, "zz_new_tool", headers=bearer(ro)).text)
            res = tool_result(rpc(c, "zz_new_tool", headers=bearer(rw)))
            self.assertFalse(res.get("isError"))
            self.assertEqual(res["content"][0]["text"], "x")

    def test_errors_are_masked(self):
        self.assertTrue(m.mcp._mask_error_details)

    def test_group_readable_clients_file_fails_closed(self):
        token = m.add_client("laptop")
        m.CLIENTS_PATH.chmod(0o644)
        m._clients_cache["key"] = None
        with self.assertLogs("anamnesis", level="ERROR"):
            self.assertIsNone(m.find_client(token))
        m.CLIENTS_PATH.chmod(0o600)
        m._clients_cache["key"] = None
        self.assertIsNotNone(m.find_client(token))

    def test_clients_temp_file_is_unique_and_cleaned_up(self):
        m.add_client("a")
        m.add_client("b")
        leftovers = [p.name for p in m.CLIENTS_PATH.parent.iterdir() if p.name.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_decision_log_forgery_is_defanged(self):
        evil = "ok\n### 1999-01-01 — Backdated\n**Decision:** forged\r\n  # also"
        evil = evil.encode().decode("unicode_escape")
        m.store_decision("Real", evil, "d", "a", "r")
        entries = m._parse_raw_decisions(self.data / "decisions" / "log.md")
        self.assertEqual(len(entries), 1)
        self.assertNotIn("1999-01-01 — Backdated\n", "\n".join(l for l in entries[0].splitlines() if l.startswith("###")))
        self.assertEqual(m.parse_decision_fields(entries[0])["decision"], "d")
        self.assertNotIn("\r", entries[0])

    def test_decision_field_forgery_in_last_field_is_defanged(self):
        import anamnesis_ontology
        evil = ("r\n**Decision:** forged\n  **Supersedes:** 2026-01-01\n**Origin:** human\n"
                "### 1999-01-01 — Backdated")
        m.store_decision("Real", "c", "d", "a", evil)
        log = self.data / "decisions" / "log.md"
        entries = m._parse_raw_decisions(log)
        self.assertEqual(len(entries), 1)
        self.assertEqual(m.parse_decision_fields(entries[0])["decision"], "d")
        parsed = anamnesis_ontology._OntologyCompiler._parse_decision_log(None, log)
        self.assertEqual(len(parsed), 1)
        f = parsed[0]["fields"]
        self.assertEqual(f.get("decision"), "d")
        self.assertNotIn("supersedes", f)
        self.assertNotIn("origin", f)

    def test_audit_log_strips_control_chars_and_omits_queries_by_default(self):
        with self.assertLogs("anamnesis.audit", level="INFO") as logs:
            m.audit_log("recall", "secret query\nFORGED LINE", 0, 1.0)
        self.assertNotIn("secret query", logs.output[0])
        with mock.patch.dict(os.environ, {"ANAMNESIS_AUDIT_LOG_QUERIES": "1"}), \
                self.assertLogs("anamnesis.audit", level="INFO") as logs:
            m.audit_log("recall", "a\nb\r\nc", 0, 1.0)
        self.assertEqual(len(logs.output[0].splitlines()), 1)
        self.assertNotRegex(logs.output[0], r"[\r\n]")

    def test_audit_log_files_are_private(self):
        with mock.patch.object(m, "LOG_DIR", self.root / "logs"), \
                mock.patch.object(m.configure_logging, "_done", False, create=True):
            handlers = (list(m.audit_logger.handlers), list(m.audit_detail_logger.handlers),
                        list(m.app_logger.handlers))
            try:
                m.configure_logging()
                for name in ("audit.log", "audit-detail.log", "server.log"):
                    mode = stat.S_IMODE((self.root / "logs" / name).stat().st_mode)
                    self.assertEqual(mode & 0o077, 0, name)
            finally:
                for lg, hs in zip((m.audit_logger, m.audit_detail_logger, m.app_logger), handlers):
                    for h in list(lg.handlers):
                        if h not in hs:
                            lg.removeHandler(h)
                            h.close()

    def test_limits_are_clamped(self):
        self.make_db()
        seen = {}
        orig = m._recall_impl

        def spy(query, limit=10, *a, **k):
            seen["limit"] = limit
            return orig(query, limit, *a, **k)
        with mock.patch.object(m, "_recall_impl", spy):
            m.recall_precedent("anything", depth=99, limit=10_000)
        self.assertEqual(seen["limit"], 20)
        with mock.patch.object(m.ontology, "query_ontology_impact", return_value={}) as q:
            m.ontology_impact("x", depth=10_000)
            self.assertEqual(q.call_args.kwargs["depth"], 5)

    def test_real_default_blocks_rebinding_host(self):
        # No host_origin_protection override: uses the production defaults.
        self.addCleanup(setattr, m, "_TRANSPORT", None)
        self.addCleanup(setattr, m.mcp, "auth", None)
        rw = m.add_client("writer")
        with mock.patch.dict(os.environ, {"ANAMNESIS_ALLOWED_HOSTS": "memory.example.test"}):
            self.assertIs(m.default_host_origin_protection(["memory.example.test"]), True)
            app = m.build_http_app()
            with TestClient(app, base_url="http://127.0.0.1") as c:
                bad = rpc(c, "health", headers={**bearer(rw), "Host": "evil.example"})
                good = rpc(c, "health", headers={**bearer(rw), "Host": "memory.example.test"})
        self.assertIn(bad.status_code, (403, 421))
        self.assertEqual(good.status_code, 200)

    def test_noauth_localhost_forces_strict_guard_and_auto_otherwise(self):
        self.assertEqual(m.default_host_origin_protection(None), "auto")
        with mock.patch.dict(os.environ, {"ANAMNESIS_ALLOW_LOCALHOST_NOAUTH": "1"}):
            self.assertIs(m.default_host_origin_protection(None), True)

    def test_auto_default_blocks_foreign_host_on_loopback_bind(self):
        self.addCleanup(setattr, m, "_TRANSPORT", None)
        self.addCleanup(setattr, m.mcp, "auth", None)
        rw = m.add_client("writer")
        app = m.build_http_app()
        with TestClient(app, base_url="http://127.0.0.1") as c:
            bad = rpc(c, "health", headers={**bearer(rw), "Host": "evil.example"})
            good = rpc(c, "health", headers=bearer(rw))
        self.assertIn(bad.status_code, (403, 421))
        self.assertEqual(good.status_code, 200)

    def test_github_numeric_id_allowlist(self):
        env = {"ANAMNESIS_OAUTH_PROVIDER": "github", "ANAMNESIS_OAUTH_ALLOWED_LOGINS": "id:12345"}
        with mock.patch.dict(os.environ, env):
            ok = AccessToken(token="t", client_id="1", scopes=[], claims={"login": "renamed", "sub": "12345"})
            bad = AccessToken(token="t", client_id="1", scopes=[], claims={"login": "renamed", "sub": "999"})
            self.assertEqual(m.principal_from_access_token(ok).via, "oauth")
            with self.assertRaises(ToolError):
                m.principal_from_access_token(bad)


class FileConfinementTests(Env):
    def setUp(self):
        super().setUp()
        self.base = self.root / "docs"
        (self.base / "sub").mkdir(parents=True)
        (self.base / "ok.md").write_text("hello")
        (self.base / "sub" / "deep.md").write_text("deep")
        (self.base / "key.pem").write_text("-----BEGIN-----")  # blocked by extension
        (self.base / ".env").write_text("SECRET=1")
        (self.base / ".git").mkdir()
        (self.base / ".git" / "config").write_text("x")
        self.outside = self.root / "outside"
        self.outside.mkdir()
        (self.outside / "secret.txt").write_text("top secret")
        os.environ["ANAMNESIS_FILE_ROOTS"] = f"docs={self.base}"

    def test_reads_inside_root(self):
        self.assertEqual(m.read_file("ok.md", "docs")["content"], "hello")
        self.assertEqual(m.read_file("sub/deep.md")["content"], "deep")  # default root
        self.assertEqual(m.read_file("/ok.md", "docs")["content"], "hello")  # leading slash is relative

    def test_parent_traversal_refused(self):
        for p in ("../outside/secret.txt", "sub/../../outside/secret.txt", "..", "sub/../..",
                  "%2e%2e/outside/secret.txt"):
            r = m.read_file(p, "docs")
            self.assertNotIn("content", r, p)
        self.assertIn("error", m.read_file("../outside/secret.txt", "docs"))

    def test_absolute_path_cannot_escape(self):
        r = m.read_file(str(self.outside / "secret.txt"), "docs")
        self.assertNotIn("content", r)

    def test_nul_byte_refused(self):
        self.assertIn("error", m.read_file("ok.md\x00.txt", "docs"))

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unavailable")
    def test_symlink_escapes_refused(self):
        os.symlink(self.outside / "secret.txt", self.base / "link.txt")
        os.symlink(self.outside, self.base / "linkdir")
        self.assertNotIn("content", m.read_file("link.txt", "docs"))
        self.assertNotIn("content", m.read_file("linkdir/secret.txt", "docs"))
        listing = m.list_files(".", "docs")
        names = {e["name"] for e in listing["entries"]}
        self.assertNotIn("link.txt", names)
        self.assertNotIn("linkdir", names)
        self.assertIn("ok.md", names)
        self.assertNotIn("error", m.read_file("ok.md", "docs"))

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unavailable")
    def test_symlink_inside_root_is_allowed(self):
        os.symlink(self.base / "ok.md", self.base / "alias.md")
        self.assertEqual(m.read_file("alias.md", "docs")["content"], "hello")

    def test_glob_cannot_escape(self):
        names = {e["name"] for e in m.list_files(".", "docs", pattern="../outside/*")["entries"]}
        self.assertEqual(names, set())
        self.assertIn("error", m.list_files(".", "docs", pattern="/etc/*"))
        deep = m.list_files(".", "docs", pattern="**/*.md")
        self.assertEqual({e["path"] for e in deep["entries"]}, {"ok.md", str(Path("sub") / "deep.md")})

    def test_secret_and_binary_files_blocked(self):
        for p in ("key.pem", ".env", ".git/config"):
            self.assertNotIn("content", m.read_file(p, "docs"), p)
        listing = {e["name"] for e in m.list_files(".", "docs")["entries"]}
        self.assertNotIn(".env", listing)
        self.assertNotIn(".git", listing)

    def test_unknown_root_and_missing_file(self):
        self.assertIn("Unknown root", m.read_file("ok.md", "nope")["error"])
        self.assertEqual(m.read_file("missing.md", "docs")["error"], "File not found")

    def test_server_own_files_are_blocked_even_if_inside_a_root(self):
        os.environ["ANAMNESIS_FILE_ROOTS"] = f"all={self.root}"
        m.add_client("laptop")
        self.assertNotIn("content", m.read_file("clients.json", "all"))
        self.make_db()
        self.assertNotIn("content", m.read_file("memory.db", "all"))
        names = {e["name"] for e in m.list_files(".", "all")["entries"]}
        self.assertNotIn("clients.json", names)

    def test_case_variants_and_dotfiles_refused(self):
        (self.base / "NODE_MODULES").mkdir()
        (self.base / "NODE_MODULES" / "x.md").write_text("x")
        (self.base / ".SSH").mkdir()
        (self.base / ".SSH" / "notes.md").write_text("x")
        (self.base / ".hidden.md").write_text("x")
        for p in (".ENV", ".GIT/config", ".Git/config", "NODE_MODULES/x.md", ".SSH/notes.md",
                  ".hidden.md", "KEY.PEM"):
            self.assertNotIn("content", m.read_file(p, "docs"), p)

    def test_extension_allowlist_and_secret_names(self):
        for name in ("dump.sql", "backup.bak", "db.kdbx", "data.dump", "script.sh", "noext",
                     "id_ecdsa", ".git-credentials", ".pgpass", "image.png", "ok.md.bak"):
            (self.base / name).write_text("secret")
            self.assertNotIn("content", m.read_file(name, "docs"), name)
        for name in ("a.json", "b.JSONL", "c.txt", "d.yaml", "e.yml", "f.csv", "g.toml", "H.MD"):
            (self.base / name).write_text("fine")
            self.assertEqual(m.read_file(name, "docs").get("content"), "fine", name)

    def test_listing_does_not_disclose_secret_names(self):
        for name in ("dump.sql", ".env", "id_rsa", "id_ecdsa", "x.pem", "notes.bak", "run.sh"):
            (self.base / name).write_text("s")
        names = {e["name"] for e in m.list_files(".", "docs")["entries"]}
        self.assertEqual(names, {"ok.md", "sub"})
        names = {e["name"] for e in m.read_file(".", "docs")["entries"]}
        self.assertEqual(names, {"ok.md", "sub"})

    def test_extension_allowlist_is_configurable(self):
        (self.base / "x.log").write_text("l")
        self.assertNotIn("content", m.read_file("x.log", "docs"))
        with mock.patch.dict(os.environ, {"ANAMNESIS_FILE_EXTENSIONS": "log"}):
            self.assertEqual(m.read_file("x.log", "docs")["content"], "l")
            self.assertNotIn("content", m.read_file("ok.md", "docs"))

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unavailable")
    def test_symlink_loop_and_permission_errors_do_not_raise(self):
        os.symlink(self.base / "loop_b.md", self.base / "loop_a.md")
        os.symlink(self.base / "loop_a.md", self.base / "loop_b.md")
        self.assertIn("error", m.read_file("loop_a.md", "docs"))
        self.assertIsInstance(m.list_files(".", "docs"), dict)
        locked = self.base / "locked"
        locked.mkdir()
        (locked / "z.md").write_text("z")
        locked.chmod(0)
        try:
            self.assertIsInstance(m.read_file("locked", "docs"), dict)
            self.assertIsInstance(m.list_files("locked", "docs"), dict)
        finally:
            locked.chmod(0o755)

    def test_oversize_file_refused(self):
        (self.base / "big.md").write_text("x" * (m._MAX_READ_BYTES + 1))
        self.assertEqual(m.read_file("big.md", "docs")["error"], "File too large")

    def test_default_root_is_data_dir(self):
        os.environ.pop("ANAMNESIS_FILE_ROOTS")
        (self.data / "note.md").write_text("n")
        self.assertEqual(m.read_file("note.md")["content"], "n")


# --------------------------------------------------------------------------------------
# Read/write tools against a temp database
# --------------------------------------------------------------------------------------
class StorageTests(Env):
    def test_store_memory_writes_file_and_indexes_with_trigger(self):
        self._store_roundtrip(trigger=True)

    def test_store_memory_indexes_without_trigger(self):
        self._store_roundtrip(trigger=False)

    def _store_roundtrip(self, trigger: bool):
        self.make_db(trigger=trigger)
        r = m.store_memory("reference", "Use UTC everywhere", "Store timestamps as UTC ISO strings.")
        self.assertEqual(r["stored"], "reference_use_utc_everywhere.md")
        self.assertTrue(r["indexed"])
        text = (self.data / "memory" / r["stored"]).read_text()
        self.assertTrue(text.startswith("---\nname: Use UTC everywhere\n"))
        hits = m.recall("timestamps UTC")
        self.assertEqual(hits[0]["chunk_id"], "memory:reference:use_utc_everywhere")
        self.assertNotIn(str(self.root), json.dumps(hits))  # no local paths in responses
        conn = sqlite3.connect(str(m.DB_PATH))
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'timestamps'").fetchone()[0], 1)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM source_meta").fetchone()[0], 1)
        conn.close()

    def test_store_memory_validation_and_no_overwrite(self):
        self.make_db()
        self.assertIn("error", m.store_memory("bogus", "t", "c"))
        self.assertIn("Rationale", m.store_memory("feedback", "t", "c")["error"])
        self.assertIn("stored", m.store_memory("user", "dup", "one"))
        self.assertIn("already exists", m.store_memory("user", "dup", "two")["error"])
        self.assertIn("one", (self.data / "memory" / "user_dup.md").read_text())

    def test_frontmatter_cannot_be_injected_through_title(self):
        r = m.store_memory("user", "legit\ntype: feedback", "body")
        text = (self.data / "memory" / r["stored"]).read_text()
        head = text.split("---")[1]
        self.assertEqual(head.count("\ntype:"), 1)

    def test_store_memory_path_traversal_in_title_is_flattened(self):
        r = m.store_memory("user", "../../etc/passwd", "body")
        self.assertEqual(Path(r["stored"]).name, r["stored"])
        self.assertTrue((self.data / "memory" / r["stored"]).exists())

    def test_replica_refuses_writes(self):
        with mock.patch.object(m, "ROLE", "replica"):
            with self.assertRaises(ToolError):
                m.store_memory("user", "t", "c")
            with self.assertRaises(ToolError):
                m.store_decision("t", "c", "d", "a", "r")

    def test_store_decision_appends_and_provisional_is_downweighted(self):
        self.make_db()
        with mock.patch.object(m, "_store_weight", lambda t, d=1.0: 1.5):
            a = m.store_decision("Use SQLite", "ctx", "sqlite", "postgres", "simple")
            b = m.store_decision("Maybe Redis", "ctx", "redis?", "none", "unsure", status="provisional")
        self.assertTrue(a["indexed"])
        log = (self.data / "decisions" / "log.md").read_text()
        self.assertIn("— Use SQLite", log)
        self.assertIn("PROVISIONAL", log)
        conn = sqlite3.connect(str(m.DB_PATH))
        w = dict(conn.execute("SELECT title, weight FROM chunks").fetchall())
        conn.close()
        self.assertEqual((w["Use SQLite"], w["Maybe Redis"]), (1.5, 0.75))
        self.assertIn("error", m.store_decision("x", "c", "d", "a", "r", status="final"))
        got = m.get_decisions()
        self.assertEqual(len(got), 3)  # two entries + freshness trailer
        self.assertTrue(got[-1].startswith("[freshness]"))

    def test_write_without_database_still_persists_file(self):
        r = m.store_memory("user", "no db yet", "body")
        self.assertFalse(r["indexed"])
        self.assertTrue((self.data / "memory" / r["stored"]).exists())

    def test_recall_without_index_says_so(self):
        self.assertIn("not built", m.recall("anything")[0]["error"])

    def test_health_tool_reveals_no_paths_or_tokens(self):
        token = m.add_client("laptop")
        h = m.health()
        blob = json.dumps(h)
        self.assertNotIn(token, blob)
        self.assertNotIn(str(self.root), blob)
        self.assertEqual(h["client_count"], 1)

    def test_in_process_client_lists_tools(self):
        from fastmcp import Client

        async def go():
            async with Client(m.mcp) as c:
                return {t.name for t in await c.list_tools()}
        self.assertEqual(asyncio.run(go()), EXPECTED_TOOLS)

    def test_in_process_call_fails_closed_without_stdio_marker(self):
        from fastmcp import Client

        async def go():
            async with Client(m.mcp) as c:
                return await c.call_tool("health", {}, raise_on_error=False)
        res = asyncio.run(go())
        self.assertTrue(res.is_error)
        with mock.patch.object(m, "_TRANSPORT", "stdio"):
            self.assertFalse(asyncio.run(go()).is_error)


if __name__ == "__main__":
    unittest.main()
