#!/usr/bin/env python3
"""anamnesis recall: the retrieval core, plus a small CLI.

Two-stage retrieval over a memory.db built by index.py:

  1. FTS5 keyword search gathers candidates (BM25 order).
  2. If an embedder is available, candidates are reranked by embedding cosine
     similarity x store weight x freshness. If it is not (no key, quota exhausted,
     offline), BM25 rank itself is the relevance signal and the result says so.

The MCP server (tools/mcp_server.py) calls `search_chunks` directly, so the CLI and
the server rank identically. Query embedding goes through index.py (`embed_query`),
which owns the backend choice, the model and the query-embedding cache; this module
never embeds on its own, so it cannot disagree with the vectors stored in the DB.

    python tools/recall.py "how should I store timestamps?"
    python tools/recall.py -n 3 --type decision --project demo "retry policy"
    python tools/recall.py --no-embed "keyword only"

Environment:
  ANAMNESIS_DB     path to memory.db (default: $ANAMNESIS_HOME/memory.db)
  ANAMNESIS_HOME   state directory (default: ~/.anamnesis)
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import os
import re
import sqlite3
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

Embedder = Callable[[str], "list[float]"]


def home_dir() -> Path:
    return Path(os.environ.get("ANAMNESIS_HOME") or (Path.home() / ".anamnesis"))


def default_db_path() -> Path:
    return Path(os.environ.get("ANAMNESIS_DB") or (home_dir() / "memory.db"))


# --- small pure helpers --------------------------------------------------------------

def unpack_embedding(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. Vectors of different lengths come from different models and
    score 0 rather than being silently truncated by zip()."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def freshness_score(timestamp: str | None, now: datetime | None = None) -> float:
    """0..1 recency decay used only for ranking (not a date). Unknown dates are neutral."""
    if not timestamp:
        return 0.5
    try:
        dt = datetime.fromisoformat(str(timestamp)[:10]).replace(tzinfo=timezone.utc)
    except ValueError:
        return 0.5
    days = ((now or datetime.now(timezone.utc)) - dt).days
    if days <= 7:
        return 1.0
    if days <= 30:
        return 0.8
    if days <= 90:
        return 0.5
    return 0.3


def fts_match_query(text: str) -> str:
    """Build a safe FTS5 MATCH expression: quoted words OR-ed together.

    Words of 3+ characters are used (shorter ones are noise); if there are none, fall
    back to whatever words exist. Keeping only word characters neutralises quotes and
    FTS operators in user text, so arbitrary input can never be a syntax error.
    """
    words = re.findall(r"\w+", text)
    chosen = [w for w in words if len(w) > 2] or words
    return " OR ".join(f'"{w}"' for w in chosen)


# --- retrieval -------------------------------------------------------------------------

def search_chunks(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int = 10,
    store_types: Optional[Sequence[str]] = None,
    project: Optional[str] = None,
    embed: Optional[Embedder] = None,
    fts_limit: int = 150,
) -> tuple[list[dict], dict]:
    """FTS candidates, optionally reranked by embeddings.

    Returns (results, info). Each result carries chunk_id, title, content (full),
    source_path, store_type, project, score, similarity, freshness, confidence and the
    raw date columns. `info["mode"]` is "hybrid", "fts" or "none"; `info["embed_error"]`
    says why an available embedder was not used.
    """
    info: dict = {"mode": "none", "candidates": 0}
    match = fts_match_query(query)
    if not match:
        return [], info

    sql = (
        "SELECT c.*, rank FROM chunks_fts fts JOIN chunks c ON c.id = fts.rowid "
        "WHERE chunks_fts MATCH ?"
    )
    params: list = [match]
    if store_types:
        sql += f" AND c.store_type IN ({','.join('?' for _ in store_types)})"
        params.extend(store_types)
    if project:
        sql += " AND c.project = ?"
        params.append(project)
    sql += " ORDER BY rank LIMIT ?"
    params.append(fts_limit)

    cur = conn.cursor()
    cur.row_factory = sqlite3.Row
    rows = cur.execute(sql, params).fetchall()
    info["candidates"] = len(rows)
    if not rows:
        return [], info

    top_rank = abs(rows[0]["rank"])
    fifth_rank = abs(rows[min(4, len(rows) - 1)]["rank"]) if len(rows) > 1 else 0
    fts_confident = top_rank > 20 and (top_rank - fifth_rank) > 3  # skip the embed call

    qvec = None
    if embed is not None and not fts_confident:
        try:
            qvec = embed(query)
        except Exception as e:  # no key, quota, network: keyword-only beats nothing
            info["embed_error"] = type(e).__name__  # class only: messages can carry keys or URLs
    info["mode"] = "hybrid" if qvec else "fts"

    results = []
    for row in rows:
        d = dict(row)
        blob = d.pop("embedding", None)
        if qvec and blob:
            similarity = cosine(qvec, unpack_embedding(blob))
        elif qvec:
            # Hybrid mode, unembedded chunk: keep the BM25 ordering, scaled into 0..0.6 so a
            # keyword-only match ranks alongside (not above) a strong embedding match.
            similarity = 0.6 * abs(d["rank"]) / top_rank if top_rank else 0.3
        else:
            # BM25 is the relevance signal when there is no query vector.
            similarity = abs(d["rank"]) / top_rank if top_rank else 0.3
        fresh = freshness_score(d.get("source_session_at") or d.get("modified_at") or "")
        weight = d.get("weight") or 1.0
        score = similarity * fresh * (weight if qvec else 1.0)
        results.append({
            **d,
            "score": round(score, 3),
            "similarity": round(similarity, 3),
            "freshness": round(fresh, 2),
            "confidence": round(min(score / 1.5, 1.0), 2),
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:limit], info


def default_embedder() -> Optional[Embedder]:
    """Query embedder backed by index.py, or None when it cannot be set up."""
    try:
        idx = importlib.import_module("index")
        client = idx.get_embed_client()
    except Exception as e:
        print(f"embedding unavailable ({type(e).__name__}); keyword-only",
              file=sys.stderr)
        return None
    return lambda q: idx.embed_query(client, q)


def recall(query: str, k: int = 5, *, db: Path | None = None, embed: Optional[Embedder] = None,
           store_types: Optional[Sequence[str]] = None, project: Optional[str] = None) -> list[dict]:
    """Convenience wrapper: open the DB, search, return the ranked results."""
    db = db or default_db_path()
    if not db.exists():
        raise FileNotFoundError(f"{db} not found; run tools/index.py first")
    with contextlib.closing(sqlite3.connect(str(db))) as conn:
        results, _ = search_chunks(conn, query, limit=k, store_types=store_types,
                                   project=project, embed=embed)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Search an Anamnesis memory.db")
    ap.add_argument("query", nargs="+", help="natural-language question")
    ap.add_argument("-n", "--limit", type=int, default=5)
    ap.add_argument("--type", dest="store_types", action="append", help="store type filter (repeatable)")
    ap.add_argument("--project")
    ap.add_argument("--db", type=Path, default=None)
    ap.add_argument("--no-embed", action="store_true", help="keyword-only, no embedding call")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    db = args.db or default_db_path()
    if not db.exists():
        print(f"{db} not found; run tools/index.py first", file=sys.stderr)
        return 1
    embed = None if args.no_embed else default_embedder()
    with contextlib.closing(sqlite3.connect(str(db))) as conn:
        results, info = search_chunks(conn, " ".join(args.query), limit=args.limit,
                                      store_types=args.store_types, project=args.project, embed=embed)
    if args.json:
        slim = [{k: r[k] for k in ("chunk_id", "title", "store_type", "project", "score", "content")}
                for r in results]
        print(json.dumps({"mode": info["mode"], "results": slim}, indent=2))
        return 0
    if not results:
        print("no results")
        return 0
    if info["mode"] == "fts":
        print("(keyword-only ranking)", file=sys.stderr)
    for r in results:
        print(f"[{r['score']}] ({r['store_type']}) {r['title']}\n    {r['content'][:160].strip()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
