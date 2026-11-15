#!/usr/bin/env python3
"""Clip an external LLM conversation transcript into the memory index.

    python tools/ingest_conversation.py transcript.md --source chatgpt            # dry run (default)
    python tools/ingest_conversation.py transcript.md --source chatgpt --live     # write

Safety: dry-run is the default and live writes need an explicit --live. Secrets that match
well-known key formats are redacted BEFORE anything is printed or stored. Chunks are written
as store_type `conversation_clip` without an embedding (searchable by FTS at once); run
`python tools/index.py backfill` to embed them. To remove every clip again:
    DELETE FROM chunks WHERE store_type = 'conversation_clip';

The database is ANAMNESIS_DB (alias ANAMNESIS_OUT), default $ANAMNESIS_HOME/memory.db, or --db.
It must already exist: this tool never creates a database.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anamnesis_redact import REDACTION_PATTERNS, redact_text  # noqa: E402,F401


def default_db_path() -> Path:
    explicit = os.environ.get("ANAMNESIS_DB") or os.environ.get("ANAMNESIS_OUT")
    if explicit:
        return Path(explicit).expanduser()
    home = os.environ.get("ANAMNESIS_HOME") or str(Path.home() / ".anamnesis")
    return Path(home).expanduser() / "memory.db"


DEFAULT_DB_PATH = default_db_path()

# Secret redaction lives in anamnesis_redact.py (shared with the chat importer).

# Turn marker patterns (case-insensitive)
TURN_PATTERNS = [
    re.compile(r"^##\s*(user|you|human)\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^##\s*(assistant|chatgpt|claude|grok|gemini|ai|model)\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\*\*(user|you|human):\*\*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\*\*(assistant|chatgpt|claude|grok|gemini|ai|model):\*\*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^(user|you|human):", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^(assistant|chatgpt|claude|grok|gemini|ai|model):", re.IGNORECASE | re.MULTILINE),
]

USER_LABELS = {"user", "you", "human"}
ASSISTANT_LABELS = {"assistant", "chatgpt", "claude", "grok", "gemini", "ai", "model"}


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower())[:50].strip("_")
    return slug or "untitled"


def redact_content(content: str) -> tuple[str, int]:
    """Redact secrets from content, return (redacted_text, count)."""
    return redact_text(content)


def parse_turns(content: str) -> list[tuple[str, str]] | None:
    """Parse content into list of (role, text) turns. Returns None if no markers found."""
    lines = content.split("\n")
    turns = []
    current_role = None
    current_lines = []
    found_marker = False

    for line in lines:
        matched = False
        for pattern in TURN_PATTERNS:
            m = pattern.match(line)
            if m:
                label = m.group(1).lower()
                if label in USER_LABELS:
                    role = "user"
                else:
                    role = "assistant"
                if current_role is not None and current_lines:
                    turns.append((current_role, "\n".join(current_lines).strip()))
                current_role = role
                current_lines = []
                found_marker = True
                matched = True
                # Remove the marker from the line content
                rest = pattern.sub("", line).strip()
                if rest:
                    current_lines.append(rest)
                break
        if not matched:
            if current_role is not None:
                current_lines.append(line)
            else:
                # Before first marker, treat as preamble
                current_lines.append(line)

    if current_role is not None and current_lines:
        turns.append((current_role, "\n".join(current_lines).strip()))

    if not found_marker:
        return None
    return turns


def chunk_turns(turns: list[tuple[str, str]], max_chars: int = 3500, min_chars: int = 2000, max_chunks: int = 50) -> list[str]:
    """Group turns into chunks of roughly min_chars-max_chars characters."""
    if not turns:
        return []

    chunks = []
    current_chunk = []
    current_size = 0

    for role, text in turns:
        turn_text = f"{role}: {text}"
        turn_size = len(turn_text)

        if current_size + turn_size > max_chars and current_size >= min_chars:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [turn_text]
            current_size = turn_size
        else:
            current_chunk.append(turn_text)
            current_size += turn_size

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    # If too many chunks, merge them
    if len(chunks) > max_chunks:
        merged = []
        chunk_size = len(chunks) // max_chunks + 1
        for i in range(0, len(chunks), chunk_size):
            merged.append("\n\n".join(chunks[i:i + chunk_size]))
        chunks = merged

    return chunks


def derive_title(content: str, turns: list[tuple[str, str]] | None, custom_title: str | None) -> str:
    """Derive title from first user turn or first line."""
    if custom_title:
        return custom_title

    if turns:
        for role, text in turns:
            if role == "user":
                first_line = text.split("\n")[0].strip()
                if len(first_line) > 80:
                    return first_line[:77] + "..."
                return first_line

    # Fallback to first line of content
    first_line = content.split("\n")[0].strip()
    if len(first_line) > 80:
        return first_line[:77] + "..."
    return first_line


def build_chunk_id(source: str, title: str, index: int, content: str) -> str:
    """Build deterministic chunk ID."""
    content_hash = hashlib.sha1(content.encode()).hexdigest()[:12]
    slug = slugify(title)
    return f"conversation:{source}:{slug}:{index}:{content_hash}"


def check_existing(conn: sqlite3.Connection | None, chunk_id: str) -> bool:
    """Check if chunk_id exists in DB. Returns False if conn is None."""
    if conn is None:
        return False
    try:
        return conn.execute("SELECT 1 FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone() is not None
    except sqlite3.Error:
        return False


def build_chunks(content: str, source: str, title: str, project: str, tags: str, source_path: Path, live: bool) -> list[dict]:
    """Build memory chunk records from content."""
    # Redact content
    redacted_content, redact_count = redact_content(content)
    if redact_count > 0:
        print(f"Redacted {redact_count} potential secret(s)", file=sys.stderr)

    # Parse turns
    turns = parse_turns(redacted_content)

    # Determine chunk texts
    if turns is None or len(redacted_content) <= 4000:
        chunk_texts = [redacted_content]
    else:
        chunk_texts = chunk_turns(turns)

    # Derive title
    derived_title = derive_title(redacted_content, turns, title)

    # Build chunks
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    total_chunks = len(chunk_texts)

    chunks = []
    for i, chunk_text in enumerate(chunk_texts):
        part_suffix = f" (source: {source}, part {i + 1}/{total_chunks})" if total_chunks > 1 else ""
        chunk_title = derived_title + part_suffix

        provenance_header = f"[source: {source} | clipped: {now_iso} | part {i + 1}/{total_chunks}]\n\n"
        full_content = provenance_header + chunk_text

        # Hash the stable chunk_text (pre-timestamp header), not full_content, so
        # chunk_id is deterministic across re-runs of the same input file at
        # different times (required for idempotent re-ingest / dry-run checks).
        chunk_id = build_chunk_id(source, derived_title, i, chunk_text)

        chunk = {
            "chunk_id": chunk_id,
            "store_type": "conversation_clip",
            "source_path": str(source_path),
            "title": chunk_title,
            "content": full_content,
            "created_at": now_iso,
            "modified_at": now_iso,
            "project": project,
            "topic_ids": "",
            "weight": 1.0,
            "indexed_at": now_iso,
            "tags": tags,
        }
        chunks.append(chunk)

    return chunks


INSERT_COLUMNS = ("chunk_id", "store_type", "source_path", "title", "content", "created_at", "modified_at",
                  "project", "topic_ids", "weight", "indexed_at", "tags")


def write_chunks(conn: sqlite3.Connection, chunks: list[dict]) -> tuple[int, int]:
    """Write chunks to the DB. Returns (inserted, skipped). Only columns the table has are
    written, so an older or extended schema works; the FTS triggers mirror each insert."""
    have = {r[1] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()}
    cols = [c for c in INSERT_COLUMNS if c in have]
    inserted = skipped = 0
    for chunk in chunks:
        if check_existing(conn, chunk["chunk_id"]):
            print(f"[LIVE] skipped (exists) {chunk['chunk_id']}")
            skipped += 1
            continue
        conn.execute(f"INSERT INTO chunks ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                     [chunk[c] for c in cols])
        print(f"[LIVE] inserted {chunk['chunk_id']}")
        inserted += 1
    conn.commit()
    return inserted, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clip an external LLM conversation transcript into memory chunks."
    )
    parser.add_argument("input_file", type=str, help="Path to conversation transcript file")
    parser.add_argument("--source", type=str, default="other", choices=["chatgpt", "claude", "grok", "gemini", "other"],
                        help="Provenance label (default: other)")
    parser.add_argument("--title", type=str, default=None, help="Optional custom title")
    parser.add_argument("--project", type=str, default="external", help="Project label (default: external)")
    parser.add_argument("--tags", type=str, default="", help="Comma-separated tags")
    parser.add_argument("--live", action="store_true", help="Actually write to the DB (default: dry-run)")
    parser.add_argument("--db", type=str, default=None, help="Override DB path (default: ANAMNESIS_DB)")

    args = parser.parse_args()

    # Resolve DB path
    db_path = Path(args.db) if args.db else default_db_path()

    # Read input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: input file not found: {args.input_file}", file=sys.stderr)
        return 1

    try:
        content = input_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        return 1

    # Build chunks
    chunks = build_chunks(content, args.source, args.title, args.project, args.tags, input_path.resolve(), args.live)

    if not chunks:
        print("No chunks to process.", file=sys.stderr)
        return 0

    if args.live:
        # Live mode
        if not db_path.exists():
            print(f"Error: DB file does not exist: {db_path}", file=sys.stderr)
            return 1

        try:
            conn = sqlite3.connect(str(db_path))
            conn.execute("SELECT 1 FROM chunks LIMIT 1")
        except sqlite3.Error as e:
            print(f"Error: {db_path} is not an anamnesis database ({e})", file=sys.stderr)
            return 1

        print(f"\n{'=' * 60}")
        print(f"LIVE — writing {len(chunks)} chunk(s) to {db_path}")
        print(f"{'=' * 60}\n")

        inserted, skipped = write_chunks(conn, chunks)
        conn.close()

        print(f"\nSummary: {inserted} inserted, {skipped} skipped (already present)")
        if inserted:
            print("Run `python tools/index.py backfill` to embed the new chunks.")

    else:
        # Dry-run mode
        print(f"\n{'=' * 60}")
        print("DRY RUN — nothing written")
        print(f"{'=' * 60}\n")

        # Try to open DB for existence check (read-only)
        conn = None
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path))
            except sqlite3.Error:
                pass

        for chunk in chunks:
            exists = check_existing(conn, chunk["chunk_id"])
            chunk["_exists_in_db"] = exists
            print(json.dumps(chunk, indent=2, ensure_ascii=False))
            print()

        if conn:
            conn.close()

        print(f"Summary: {len(chunks)} chunk(s) would be created")
        print("Dry run only — no data was written to Anamnesis.")
        print("Re-run with --live to actually store these memories.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
