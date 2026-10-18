#!/usr/bin/env python3
"""anamnesis MCP server — direct agent memory integration for Claude Code, Codex, and Cursor.

Exposes Anamnesis tools over standard stdio JSON-RPC via FastMCP / MCP SDK.

Tools:
- recall(query, limit, store_types): Meaning-based search with embedding cosine reranking
- get_recent(limit, store_types): Chronological memory entries, newest first
- get_decisions(limit, query): Fast lookup of authoritative architectural decisions
- store_memory(type, title, content, rationale): Writes memory markdown file & indexes it
- store_decision(title, context, decision, alternatives, rationale, status): Records decision
- health(): System status, DB chunk counts, and freshness metadata

Usage:
  python tools/mcp_server.py
  # Or via uv:
  uv run --with mcp --with google-genai tools/mcp_server.py
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Connect to FastMCP from the official mcp package or standalone fastmcp
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    try:
        from fastmcp import FastMCP
    except ImportError:
        sys.exit(
            "MCP library not found. Please install with:\n"
            "  pip install mcp\n"
            "or run with:\n"
            "  uv run --with mcp tools/mcp_server.py"
        )

# Import local recall engine
sys.path.insert(0, str(Path(__file__).parent))
import recall

mcp = FastMCP("anamnesis")

DB_PATH = Path(os.environ.get("ANAMNESIS_OUT", os.environ.get("ANAMNESIS_DB", "memory.db")))
DATA_ROOT = Path(os.environ.get("ANAMNESIS_DATA_ROOT", "data"))
AUTO_COMMIT = os.environ.get("ANAMNESIS_GIT_AUTOCOMMIT", "1") == "1"


def git_commit(filepath: Path, message: str) -> bool:
    if not AUTO_COMMIT:
        return False
    try:
        repo_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(filepath.parent),
            stderr=subprocess.DEVNULL
        ).decode().strip()
        subprocess.run(["git", "add", str(filepath)], cwd=repo_root, check=True)
        subprocess.run(["git", "commit", "-m", message], cwd=repo_root, check=True)
        return True
    except Exception:
        return False


@mcp.tool()
def recall_memory(query: str, limit: int = 5, store_types: list[str] | None = None) -> list[dict]:
    """Search memory by meaning and relevance.

    Uses FTS5 keyword matching followed by semantic cosine similarity reranking.
    Falls back gracefully to keyword BM25 ranking if offline or quota-limited.

    Args:
        query: Natural language question or search phrase.
        limit: Maximum number of results to return (default: 5).
        store_types: Optional filter by store types (e.g. ['decision', 'feedback', 'reference']).
    """
    results = recall.recall(query, k=limit * 2 if store_types else limit)
    if store_types:
        results = [r for r in results if r.get("store_type") in store_types][:limit]
    return results[:limit]


@mcp.tool()
def get_recent(limit: int = 10, store_types: list[str] | None = None) -> list[dict]:
    """Retrieve the newest memory entries ordered by indexed timestamp.

    Args:
        limit: Number of entries to retrieve (default: 10).
        store_types: Optional filter by store type (e.g. ['feedback', 'decision']).
    """
    if not DB_PATH.exists():
        return [{"error": f"Database {DB_PATH} not found. Run index.py first."}]
    conn = sqlite3.connect(str(DB_PATH))
    query = "SELECT chunk_id, title, content, store_type, indexed_at FROM chunks"
    params = []
    if store_types:
        placeholders = ",".join("?" for _ in store_types)
        query += f" WHERE store_type IN ({placeholders})"
        params.extend(store_types)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [{"chunk_id": r[0], "title": r[1], "content": r[2], "store_type": r[3], "indexed_at": r[4]} for r in rows]


@mcp.tool()
def get_decisions(limit: int = 10, query: str | None = None) -> list[dict]:
    """Retrieve authoritative decisions from the memory store.

    Args:
        limit: Maximum decisions to return (default: 10).
        query: Optional keyword or topic to search within decisions.
    """
    if query:
        return recall.recall(query, k=limit, store_types=["decision"])
    return get_recent(limit=limit, store_types=["decision"])


@mcp.tool()
def store_memory(
    type: str,
    title: str,
    content: str,
    rationale: str | None = None,
) -> dict:
    """Store a persistent memory chunk (markdown file) under data/memory/.

    Types: 'feedback' (corrections & rules), 'reference' (architecture/specs), 'memory' (general knowledge).

    Args:
        type: Category ('feedback', 'reference', 'memory').
        title: Short title describing the memory.
        content: The core fact, pattern, or rule to remember.
        rationale: Why this memory was created (required for feedback).
    """
    valid_types = {"feedback", "reference", "memory", "mistake"}
    if type not in valid_types:
        return {"error": f"Invalid type '{type}'. Must be one of: {sorted(valid_types)}"}
    if type == "feedback" and not (rationale and rationale.strip()):
        return {"error": "Rationale is required for feedback memories."}

    mem_dir = DATA_ROOT / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)

    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:50]
    filename = f"{type}-{slug}.md"
    filepath = mem_dir / filename

    if filepath.exists():
        return {"error": f"Memory file '{filename}' already exists."}

    body = content.strip()
    if rationale and rationale.strip():
        body = f"{body}\n\n**Rationale:** {rationale.strip()}"

    doc = f"""---
name: {title}
description: {content[:100].replace(chr(10), ' ')}
type: {type}
---

{body}
"""
    filepath.write_text(doc, encoding="utf-8")
    committed = git_commit(filepath, f"memory: {title}")

    return {
        "status": "stored",
        "file": str(filepath),
        "committed": committed,
        "note": "Re-run tools/index.py to rebuild embeddings, or query directly via FTS."
    }


@mcp.tool()
def store_decision(
    title: str,
    context: str,
    decision: str,
    alternatives: str,
    rationale: str,
    status: str = "decided",
) -> dict:
    """Record a persistent decision in data/decisions/log.md.

    Args:
        title: Short imperative title of the decision.
        context: What problem was being solved.
        decision: What choice was made.
        alternatives: What other choices were considered and rejected.
        rationale: Why this choice was selected.
        status: 'decided' (for finalized calls) or 'provisional' (for options under review).
    """
    if status not in {"decided", "provisional"}:
        return {"error": "status must be 'decided' or 'provisional'"}

    dec_dir = DATA_ROOT / "decisions"
    dec_dir.mkdir(parents=True, exist_ok=True)
    logfile = dec_dir / "log.md"

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    status_tag = " [PROVISIONAL]" if status == "provisional" else ""
    entry = f"""
## {date_str}: {title}{status_tag}

- **Context**: {context.strip()}
- **Decision**: {decision.strip()}
- **Alternatives**: {alternatives.strip()}
- **Rationale**: {rationale.strip()}
"""
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(entry)

    committed = git_commit(logfile, f"decision: {title}")
    return {"status": "stored", "file": str(logfile), "committed": committed}


@mcp.tool()
def health() -> dict:
    """Check Anamnesis database status, chunk counts, and freshness."""
    if not DB_PATH.exists():
        return {"status": "error", "message": f"Database {DB_PATH} missing. Run index.py."}
    conn = sqlite3.connect(str(DB_PATH))
    chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    embedded = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0]
    meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
    conn.close()

    return {
        "status": "ok",
        "database": str(DB_PATH),
        "size_kb": round(DB_PATH.stat().st_size / 1024, 1),
        "total_chunks": chunks,
        "embedded_chunks": embedded,
        "built_at": meta.get("built_at"),
        "model": meta.get("embedding_model"),
        "source_commit": meta.get("source_commit", ""),
    }


if __name__ == "__main__":
    mcp.run()
