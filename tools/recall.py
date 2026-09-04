#!/usr/bin/env python3
"""anamnesis recall — query a memory.db built by index.py.

Two-stage retrieval, same shape as the production system: FTS5 keyword search to
gather candidates, then semantic rerank by embedding cosine similarity. This is a
minimal reference reader (no server); the production serving layer is an MCP server
that wraps the same query — see docs/architecture.md.

    GEMINI_API_KEY=... python tools/recall.py "your question"

Query embeddings are cached (in-process dict + a small SQLite file next to memory.db,
override with ANAMNESIS_QUERY_CACHE) so a repeated question costs zero API calls and
an exhausted embedding quota degrades to keyword-only results instead of an error.
"""
from __future__ import annotations

import contextlib
import hashlib
import math
import os
import re
import struct
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

DB = Path(os.environ.get("ANAMNESIS_OUT", "memory.db"))
MODEL = os.environ.get("ANAMNESIS_EMBEDDING_MODEL", "gemini-embedding-001")
QUERY_CACHE = Path(os.environ.get("ANAMNESIS_QUERY_CACHE", str(DB.with_suffix(".queries.db"))))

_MEM: dict[str, list[float]] = {}   # in-process cache; bounded so a long-lived caller can't grow unbounded
_MEM_MAX = 512
_SCHEMA_OK: set[str] = set()        # cache paths whose schema was ensured in this process


def pack(values: list[float]) -> bytes:
    return struct.pack(f"{len(values)}f", *values)


def unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob)//4}f", blob))


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


# --- query-embedding cache -------------------------------------------------------------

def cache_key(text: str) -> str:
    # Whitespace-only normalisation. Case and accents change the embedding, so they must
    # change the key — lowercasing would serve one casing's vector for the other.
    norm = " ".join(text.split())
    return hashlib.sha1(f"{MODEL}\n{norm}".encode("utf-8")).hexdigest()


def _cache_conn() -> sqlite3.Connection:
    # Shared by any long-running caller AND short-lived CLI invocations: WAL so readers and
    # the writer never block each other; a short busy timeout because this sits on the
    # interactive path — a locked file means "miss", not "wait".
    conn = sqlite3.connect(str(QUERY_CACHE), timeout=0.5)
    path_key = str(QUERY_CACHE)
    if path_key not in _SCHEMA_OK:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""CREATE TABLE IF NOT EXISTS query_embeddings (
            key TEXT PRIMARY KEY, model TEXT NOT NULL, dim INTEGER NOT NULL,
            embedding BLOB NOT NULL, created_at TEXT NOT NULL)""")
        conn.execute("DELETE FROM query_embeddings WHERE model != ?", (MODEL,))  # reap stale-model rows
        conn.commit()
        _SCHEMA_OK.add(path_key)
    return conn


def _cache_get(key: str) -> list[float] | None:
    try:
        with contextlib.closing(_cache_conn()) as conn:
            row = conn.execute("SELECT embedding, dim FROM query_embeddings WHERE key = ? AND model = ?",
                               (key, MODEL)).fetchone()
        if not row:
            return None
        emb = unpack(row[0])
        if not emb or len(emb) != row[1]:
            return None  # torn/corrupt blob or dimension mismatch → miss
        return emb
    except Exception as e:
        # The cache is an accelerator, never a dependency — but a silent miss hides
        # contention/corruption, so say so once per call.
        print(f"query-cache read failed ({type(e).__name__}: {str(e)[:120]}) — treating as miss",
              file=sys.stderr, flush=True)
        return None


def _cache_put(key: str, emb: list[float]) -> None:
    try:
        with contextlib.closing(_cache_conn()) as conn:
            conn.execute("INSERT OR IGNORE INTO query_embeddings (key, model, dim, embedding, created_at)"
                         " VALUES (?, ?, ?, ?, ?)",
                         (key, MODEL, len(emb), pack(emb),
                          datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")))
            conn.commit()
    except Exception as e:
        print(f"query-cache write failed ({type(e).__name__}: {str(e)[:120]}) — vector not cached",
              file=sys.stderr, flush=True)


def _embed_remote(q: str) -> list[float]:
    """ONE API call, no retry: a 429 raises immediately so the caller can degrade in
    milliseconds instead of sleeping through a quota window."""
    from google import genai
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY not set (https://aistudio.google.com/apikey)")
    client = genai.Client(api_key=key)  # bind — a temporary Client gets GC'd and closes its httpx client mid-call
    r = client.models.embed_content(model=MODEL, contents=[q])
    return list(r.embeddings[0].values)


def embed_query(q: str) -> list[float]:
    """Cache-first: in-process dict, then the SQLite cache, then one API call.
    Cache failures of any kind are misses, never errors. Failed embeds are never cached."""
    key = cache_key(q)
    emb = _MEM.get(key)
    if emb is None:
        emb = _cache_get(key)
    if emb is None:
        emb = _embed_remote(q)
        _cache_put(key, emb)
    if len(_MEM) >= _MEM_MAX:
        _MEM.clear()
    _MEM[key] = emb
    return emb


# --- retrieval -------------------------------------------------------------------------

def recall(query: str, k: int = 5, candidates: int = 50) -> list[dict]:
    if not DB.exists():
        sys.exit(f"{DB} not found — run tools/index.py first.")
    conn = sqlite3.connect(str(DB))
    fts_q = " OR ".join(re.findall(r"\w+", query)) or query
    rows = conn.execute(
        "SELECT c.chunk_id, c.title, c.content, c.store_type, c.weight, c.embedding "
        "FROM chunks_fts f JOIN chunks c ON c.id = f.rowid "
        "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
        (fts_q, candidates),
    ).fetchall()
    fts_hit = bool(rows)
    if not rows:  # fall back to whole corpus if keywords miss
        rows = conn.execute(
            "SELECT chunk_id, title, content, store_type, weight, embedding FROM chunks"
        ).fetchall()
    try:
        qv = embed_query(query)
    except SystemExit:
        raise
    except Exception as e:
        # Quota exhausted / network down: keyword-only results beat no results. Without a
        # keyword hit there is nothing sensible to rank, so surface the error instead.
        if not fts_hit:
            raise
        print(f"embedding unavailable ({type(e).__name__}: {str(e)[:120]}) — keyword-only results",
              file=sys.stderr, flush=True)
        return [{"chunk_id": cid, "title": title, "content": content,
                 "store_type": stype, "score": None}
                for cid, title, content, stype, _w, _e in rows[:k]]
    scored = []
    for cid, title, content, stype, weight, emb in rows:
        score = cosine(qv, unpack(emb)) * (weight or 1.0)
        scored.append({"chunk_id": cid, "title": title, "content": content,
                       "store_type": stype, "score": round(score, 4)})
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:k]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python tools/recall.py "your question"')
    for r in recall(" ".join(sys.argv[1:])):
        print(f"[{r['score']}] ({r['store_type']}) {r['title']}\n    {r['content'][:160].strip()}\n")
