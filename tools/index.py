#!/usr/bin/env python3
"""anamnesis indexer — the genericized core.

Reads a self-contained `data/` tree of markdown + JSON, embeds every chunk with a
Gemini embedding model, and writes a `memory.db` (SQLite FTS5 keyword index +
`embedding BLOB` for semantic rerank). No machine-specific paths: the same script
runs on a laptop or a GitHub Actions runner, driven entirely by env vars.

    ANAMNESIS_DATA_ROOT   directory holding memory/ decisions/ manifests/  (default: ./data)
    ANAMNESIS_OUT         output database path                             (default: ./memory.db)
    GEMINI_API_KEY        embedding key (aistudio.google.com/apikey)

    python tools/index.py            # build memory.db from ./data

The DB is a *build artifact*, not primary state — git holds the sources, this
rebuilds the index anywhere. That is the whole point: no primary machine.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

EMBEDDING_MODEL = os.environ.get("ANAMNESIS_EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM = int(os.environ.get("ANAMNESIS_EMBEDDING_DIM", "3072"))
MAX_CHARS = 1500          # target chunk size
BATCH = 32                # embedding batch size

# store_type weights — higher = more trusted at recall time. Decisions are the
# highest-trust content class (they read as authoritative to a downstream agent).
WEIGHTS = {"decision": 1.5, "feedback": 1.2, "memory": 1.0, "reference": 1.0,
           "mistake": 1.1, "contact": 1.0, "index": 0.9}


def now_utc() -> str:
    # Always UTC + Z. Never naive-local — mixing conventions in one column silently
    # breaks freshness comparisons (a lesson learned the hard way; see docs/the-story.md).
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Minimal YAML-ish frontmatter parser (top-level key: value only — no PyYAML dep)."""
    meta: dict = {}
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            for line in text[3:end].splitlines():
                if ":" in line and not line.lstrip().startswith("#"):
                    k, _, v = line.partition(":")
                    meta[k.strip()] = v.strip()
            body = text[end + 4:]
    return meta, body.strip()


def chunk_text(text: str) -> list[str]:
    """Split on blank lines, then greedily pack paragraphs up to MAX_CHARS."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 > MAX_CHARS and cur:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks or [text.strip()]


def collect(root: Path) -> list[dict]:
    """Collect sources from data/{memory,decisions,manifests}. Store type by folder/prefix."""
    out: list[dict] = []
    mem = root / "memory"
    if mem.exists():
        for f in sorted(mem.glob("*.md")):
            stype = "feedback" if f.name.startswith("feedback") else \
                    "reference" if f.name.startswith("reference") else "memory"
            out.append({"path": f, "store_type": stype})
    dec = root / "decisions"
    if dec.exists():
        for f in sorted(dec.rglob("*.md")):
            out.append({"path": f, "store_type": "decision"})
    man = root / "manifests"
    if man.exists():
        for f in sorted(man.glob("*.json")):
            out.append({"path": f, "store_type": "index"})
    return out


def to_chunks(src: dict) -> list[dict]:
    path: Path = src["path"]
    text = path.read_text(encoding="utf-8", errors="replace")
    stype = src["store_type"]
    if path.suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []
        # one chunk per top-level entry; keep it simple and generic
        items = data.values() if isinstance(data, dict) else data
        return [{"title": path.stem, "content": json.dumps(v, ensure_ascii=False)[:MAX_CHARS],
                 "store_type": stype, "source": path.name} for v in items if v]
    meta, body = parse_frontmatter(text)
    title = meta.get("name") or path.stem
    return [{"title": title, "content": c, "store_type": stype, "source": path.name}
            for c in chunk_text(body)]


def embed_all(texts: list[str]) -> list[list[float]]:
    from google import genai
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY not set (get one at https://aistudio.google.com/apikey)")
    client = genai.Client(api_key=key)
    vecs: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        r = client.models.embed_content(model=EMBEDDING_MODEL, contents=batch)
        vecs.extend(list(e.values) for e in r.embeddings)
        print(f"  embedded {min(i + BATCH, len(texts))}/{len(texts)}", flush=True)
    return vecs


def init_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            chunk_id TEXT UNIQUE NOT NULL,
            title TEXT, content TEXT NOT NULL,
            store_type TEXT, source TEXT, weight REAL,
            embedding BLOB, indexed_at TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(chunk_id, title, content, content=chunks, content_rowid=id);
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        """
    )
    return conn


def build(data_root: Path, out: Path) -> None:
    start = time.time()
    srcs = collect(data_root)
    all_chunks: list[dict] = []
    for s in srcs:
        all_chunks.extend(to_chunks(s))
    print(f"Found {len(srcs)} sources → {len(all_chunks)} chunks", flush=True)
    if not all_chunks:
        sys.exit(f"No sources under {data_root} — add markdown to data/memory/ first.")

    vecs = embed_all([c["content"] for c in all_chunks])

    if out.exists():
        out.unlink()
    conn = init_db(out)
    ts = now_utc()
    for i, (c, v) in enumerate(zip(all_chunks, vecs)):
        cid = f"{c['store_type']}:{c['source']}:{i}"
        blob = struct.pack(f"{len(v)}f", *v)
        conn.execute(
            "INSERT INTO chunks (chunk_id,title,content,store_type,source,weight,embedding,indexed_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (cid, c["title"], c["content"], c["store_type"], c["source"],
             WEIGHTS.get(c["store_type"], 1.0), blob, ts),
        )
    conn.execute(
        "INSERT INTO chunks_fts (rowid, chunk_id, title, content)"
        " SELECT id, chunk_id, title, content FROM chunks"
    )
    for k, val in {"chunks": len(all_chunks), "embedding_model": EMBEDDING_MODEL,
                   "embedding_dim": EMBEDDING_DIM, "built_at": ts,
                   "source_commit": os.environ.get("GITHUB_SHA", "")}.items():
        conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (k, str(val)))
    conn.commit()
    conn.close()
    mb = out.stat().st_size / 1024 / 1024
    print(f"Built {out} — {len(all_chunks)} chunks, {mb:.1f} MB, {time.time()-start:.1f}s", flush=True)


if __name__ == "__main__":
    root = Path(os.environ.get("ANAMNESIS_DATA_ROOT", "data"))
    out = Path(os.environ.get("ANAMNESIS_OUT", "memory.db"))
    build(root, out)
