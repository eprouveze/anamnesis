#!/usr/bin/env python3
"""anamnesis recall — query a memory.db built by index.py.

Two-stage retrieval, same shape as the production system: FTS5 keyword search to
gather candidates, then semantic rerank by embedding cosine similarity. This is a
minimal reference reader (no server); the production serving layer is an MCP server
that wraps the same query — see docs/architecture.md.

    GEMINI_API_KEY=... python tools/recall.py "your question"
"""
from __future__ import annotations

import math
import os
import re
import struct
import sqlite3
import sys
from pathlib import Path

DB = Path(os.environ.get("ANAMNESIS_OUT", "memory.db"))
MODEL = os.environ.get("ANAMNESIS_EMBEDDING_MODEL", "gemini-embedding-001")


def unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob)//4}f", blob))


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def embed_query(q: str) -> list[float]:
    from google import genai
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY not set (https://aistudio.google.com/apikey)")
    client = genai.Client(api_key=key)  # bind — a temporary Client gets GC'd and closes its httpx client mid-call
    r = client.models.embed_content(model=MODEL, contents=[q])
    return list(r.embeddings[0].values)


def recall(query: str, k: int = 5, candidates: int = 50) -> list[dict]:
    if not DB.exists():
        sys.exit(f"{DB} not found — run tools/index.py first.")
    conn = sqlite3.connect(str(DB))
    fts_q = " OR ".join(re.findall(r"\w+", query)) or query
    rows = conn.execute(
        "SELECT c.chunk_id, c.title, c.content, c.store_type, c.weight, c.embedding "
        "FROM chunks_fts f JOIN chunks c ON c.id = f.rowid "
        "WHERE chunks_fts MATCH ? LIMIT ?",
        (fts_q, candidates),
    ).fetchall()
    if not rows:  # fall back to whole corpus if keywords miss
        rows = conn.execute(
            "SELECT chunk_id, title, content, store_type, weight, embedding FROM chunks"
        ).fetchall()
    qv = embed_query(query)
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
