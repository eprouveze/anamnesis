#!/usr/bin/env python3
"""wiki_link_resolver.py — deterministic [[wiki-link]] edge resolver.

Write-time resolver: match `[[slug]]` references inside memory markdown against
existing memory filenames/slugs, and create first-class edges in `nugget_links`
for the ones that resolve deterministically.

Design constraints:
  - relation_type='references', a SEPARATE namespace from broke_because/fixed_by.
    This script only ever inserts 'references' rows and never touches other
    relation types.
  - >=99% precision, ambiguous matches left UNRESOLVED. A [[slug]] that normalizes
    to more than one distinct markdown file is reported as ambiguous and never
    written as an edge.
  - Dangling forward-refs (a [[slug]] with no matching file at all) are a FEATURE:
    deliberate markers for something worth writing later. They are reported,
    never "fixed", never flagged as an error.
  - NEVER edits markdown files. It only reads .md files and only writes to
    `nugget_links`; it has no code path that opens a markdown file for writing.

Matching rule (deterministic, precision-first):
  1. Every markdown file in the memory dir is one note. Its canonical identity
     is its filename stem (e.g. `feedback_prefer_small_prs`).
  2. Each note is reachable under up to three slug aliases:
       a. its filename stem itself
       b. its filename stem with a known category prefix stripped
          (feedback_/project_meta_/project_/reference_/contact_/user_), because
          `[[prefer-small-prs]]` commonly omits the file's own `feedback_` prefix
       c. its frontmatter `name:` value, when that value is already slug-shaped
          (matches `^[a-zA-Z0-9][a-zA-Z0-9_-]*$`) rather than a prose title
  3. All aliases of all files are folded into one normalized index (lowercase,
     `_`/`-` treated as equivalent). A [[slug]] resolves only when its normalized
     form maps to EXACTLY ONE distinct file across every alias of every file.
  4. A resolved link writes one `references` edge from the chunk-index-0 `chunks`
     row of the source file to the chunk-index-0 row of the target file,
     source_text = the raw `[[slug]]` text. Duplicate (from,to) pairs are deduped
     before insert, and an edge that already exists is skipped, not re-inserted:
     a partial UNIQUE INDEX on (from_chunk_id, to_chunk_id) WHERE
     relation_type='references' plus a single BEGIN IMMEDIATE transaction around
     the whole apply, so a concurrent second --apply cannot double-insert (it
     serializes behind the first, or errors).

Bucket accounting is exhaustive: every `[[slug]]` occurrence found by the regex
lands in exactly one of resolved / ambiguous / dangling / self_references /
ignored_non_slug / skipped_target_not_indexed / skipped_unindexed_source. A source
file that is not yet indexed into `chunks` is NOT silently dropped; its occurrences
are counted individually into skipped_unindexed_source, so
`wikilink_occurrences_total` always equals the sum of every bucket. A `[[slug]]`
that resolves to its OWN source file is a self-reference, counted separately from
`dangling_forward_refs`.

Double-indexed files (the same physical memory file present under >1 distinct
`chunks.source_path`, e.g. via a symlinked directory) are NOT "not indexed":
`resolve_source_paths()` compares `Path.resolve()` across the candidate rows and,
when they agree on disk, treats the file as indexed (one canonical row). The
`duplicate_source`/`duplicate_target` fields and the top-level
`duplicate_indexed_files` map name the real remedy: dedupe the chunk rows, never
"reindex the file".

Default mode is always a dry-run report; `--apply` is required to write edges.

Configuration (environment, overridable by flags):
  ANAMNESIS_DB                   database (default $ANAMNESIS_HOME/memory.db)
  ANAMNESIS_MEMORY_DIR           markdown directory (default $ANAMNESIS_DATA_DIR/memory)
  ANAMNESIS_WIKILINK_PREFIXES    comma-separated category prefixes to strip
                                 (default: feedback_,project_meta_,project_,reference_,contact_,user_)

Usage:
  python tools/wiki_link_resolver.py                       # dry-run against the live DB
  python tools/wiki_link_resolver.py --db copy.db          # dry-run against a copy
  python tools/wiki_link_resolver.py --db copy.db --apply  # write edges into that DB
  python tools/wiki_link_resolver.py --report out.json     # save the full classification
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anamnesis_dict import data_dir, db_path as _default_db_path  # noqa: E402

DB_PATH = _default_db_path()
MEMORY_DIR = Path(os.environ.get("ANAMNESIS_MEMORY_DIR") or (data_dir() / "memory"))

WIKILINK_RE = re.compile(r"\[\[([^\]\n]+)\]\]")
SLUG_SHAPED = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")

# Filename category prefixes used across the memory corpus (feedback_*, project_*,
# reference_*, contact_*, user_*). A [[bracket]] link commonly omits this prefix.
# Longer prefixes must come first (project_meta_ before project_).
_DEFAULT_PREFIXES = "feedback_,project_meta_,project_,reference_,contact_,user_"
KNOWN_PREFIXES = tuple(
    p.strip() for p in os.environ.get("ANAMNESIS_WIKILINK_PREFIXES", _DEFAULT_PREFIXES).split(",") if p.strip()
)


def normalize(s: str) -> str:
    return s.strip().lower().replace("_", "-")


def load_markdown_index(memory_dir: Path) -> dict[str, Path]:
    """filename stem -> Path, for every .md file in memory_dir (top level only)."""
    return {p.stem: p for p in sorted(memory_dir.glob("*.md"))}


def frontmatter_name(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None
    for line in text[3:end].splitlines():
        if line.startswith("name:"):
            return line[len("name:"):].strip()
    return None


def aliases_for_stem(stem: str, path: Path) -> set[str]:
    out = {stem}
    for prefix in KNOWN_PREFIXES:
        if stem.startswith(prefix) and len(stem) > len(prefix):
            out.add(stem[len(prefix):])
    name_val = frontmatter_name(path)
    if name_val and SLUG_SHAPED.match(name_val):
        out.add(name_val)
    return out


def build_normalized_index(stem_index: dict[str, Path]) -> dict[str, set[str]]:
    """normalized-key -> set of distinct filename stems reachable under it."""
    norm: dict[str, set[str]] = {}
    for stem, path in stem_index.items():
        for alias in aliases_for_stem(stem, path):
            norm.setdefault(normalize(alias), set()).add(stem)
    return norm


def resolve_source_paths(
    conn: sqlite3.Connection, stem_index: dict[str, Path], memory_dir: Path
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """filename stem -> the exact source_path string `chunks` has it indexed under,
    plus the set of stems that are DOUBLE-indexed (the same physical file has
    more than one distinct `chunks.source_path` row -- e.g. a symlinked memory
    directory path vs the resolved path pointing at the same file on disk).

    `chunks.source_path` can use either directory variant for the same file, so
    this matches on basename rather than assuming one directory prefix -- still
    an exact per-file match, just not prefix-anchored.

    A basename with >1 distinct source_path candidates is resolved by
    checking whether they're literally the same file on disk
    (`Path.resolve()` equality, following the symlink) -- if so it's a
    duplicate-index situation, not a missing-index one, and the stem is
    reported separately rather than silently treated as unmapped. Only a
    genuine cross-directory basename collision between two DIFFERENT files
    is left unmapped.
    """
    rows = conn.execute("SELECT DISTINCT source_path FROM chunks WHERE source_path LIKE '%.md'").fetchall()
    by_basename: dict[str, list[str]] = {}
    for (p,) in rows:
        by_basename.setdefault(Path(p).name, []).append(p)

    out: dict[str, str] = {}
    duplicate_indexed: dict[str, list[str]] = {}
    for stem in stem_index:
        candidates = sorted(set(by_basename.get(f"{stem}.md", [])))
        if not candidates:
            continue
        if len(candidates) == 1:
            out[stem] = candidates[0]
            continue

        preferred = [p for p in candidates if str(Path(p).parent) in (str(memory_dir), str(memory_dir.resolve()))]
        try:
            on_disk_targets = {Path(p).resolve() for p in candidates if Path(p).exists()}
        except OSError:
            on_disk_targets = set()

        if len(on_disk_targets) == 1:
            # Same physical file, indexed under >1 source_path -- a
            # duplicate-index situation, not a missing-index one. Still
            # pick a canonical path (preferred directory, else the first
            # sorted candidate) so this file resolves normally (it IS
            # indexed -- just twice), while recording the variant paths so
            # classify() can surface the real cause (duplicate chunk rows,
            # remedy = dedupe them) instead of ever claiming "not indexed".
            out[stem] = preferred[0] if preferred else candidates[0]
            duplicate_indexed[stem] = candidates
        elif preferred:
            out[stem] = preferred[0]
        # else: candidates point at genuinely different files on disk that
        # happen to share a basename -- leave unmapped, same conservative
        # behavior as before for this (unobserved) case.
    return out, duplicate_indexed


def first_chunk_id(conn: sqlite3.Connection, source_path: str) -> int | None:
    rows = conn.execute(
        "SELECT id, chunk_id FROM chunks WHERE source_path = ? ORDER BY id LIMIT 50", (source_path,)
    ).fetchall()
    if not rows:
        return None
    for cid, chunk_id in rows:
        if chunk_id.endswith(":0"):
            return cid
    return rows[0][0]


def classify(conn: sqlite3.Connection, memory_dir: Path) -> dict:
    stem_index = load_markdown_index(memory_dir)
    norm_index = build_normalized_index(stem_index)
    indexed_paths, duplicate_indexed = resolve_source_paths(conn, stem_index, memory_dir)

    resolved, ambiguous, dangling, self_refs, skipped, skipped_unindexed, ignored_non_slug = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for stem, path in stem_index.items():
        text = path.read_text(encoding="utf-8", errors="replace")
        occurrences = list(WIKILINK_RE.finditer(text))
        if not occurrences:
            continue

        # A source file with no resolvable chunk id cannot be given a
        # from_chunk_id, so nothing in it can be resolved/ambiguous/dangling
        # yet -- but every occurrence still gets an explicit bucket (never
        # silently dropped from the totals) so a freshly-written,
        # not-yet-reindexed file is visible rather than invisible.
        #
        # A file in `duplicate_indexed` IS indexed (just twice) -- it gets a
        # real source_path/from_id here, not the "not indexed" reason, so
        # its links resolve normally. `duplicate_source` on the record
        # names the real cause for anyone auditing the output: dedupe the
        # chunk rows, don't reindex.
        source_path = indexed_paths.get(stem)
        from_id = first_chunk_id(conn, source_path) if source_path else None
        if from_id is None:
            for m in occurrences:
                raw_inner = m.group(1).strip()
                skipped_unindexed.append(
                    {
                        "from_file": stem,
                        "raw_wikilink": m.group(0),
                        "raw_slug": raw_inner,
                        "reason": "source file not indexed as a chunk (not yet reindexed)",
                    }
                )
            continue

        for m in occurrences:
            raw_inner = m.group(1).strip()
            if "|" in raw_inner:
                raw_slug, display_alias = [p.strip() for p in raw_inner.split("|", 1)]
            else:
                raw_slug = raw_inner
                display_alias = None

            target_slug = raw_slug[:-3] if raw_slug.lower().endswith(".md") else raw_slug

            record = {
                "from_file": stem,
                "raw_wikilink": m.group(0),
                "raw_slug": raw_inner,
            }
            if display_alias:
                record["display_alias"] = display_alias
            if target_slug != raw_slug:
                record["target_slug"] = target_slug

            if not SLUG_SHAPED.match(target_slug):
                record["reason"] = "inner text is not a valid slug (contains spaces or unsupported characters)"
                ignored_non_slug.append(record)
                continue

            key = normalize(target_slug)
            candidates = norm_index.get(key, set())
            if stem in duplicate_indexed:
                record["duplicate_source"] = (
                    f"'{stem}' is indexed under {len(duplicate_indexed[stem])} distinct chunks.source_path "
                    "rows for the same physical file -- not unindexed. Remedy: dedupe the chunk rows, "
                    "don't reindex."
                )

            if not candidates:
                record["reason"] = "no markdown file with this slug exists"
                dangling.append(record)
                continue
            if len(candidates) > 1:
                record["candidates"] = sorted(candidates)
                ambiguous.append(record)
                continue

            target_stem = next(iter(candidates))
            if target_stem == stem:
                # Not a missing memory (that's what `dangling` reserves for
                # deliberate forward references) -- this is a
                # link resolving to its own source file, a distinct case
                # that also doesn't warrant an edge.
                record["reason"] = "self-reference, not an edge"
                self_refs.append(record)
                continue

            target_source_path = indexed_paths.get(target_stem)
            if target_source_path is None:
                record["reason"] = f"target file '{target_stem}' exists but is not indexed as a chunk"
                skipped.append(record)
                continue
            if target_stem in duplicate_indexed:
                record["duplicate_target"] = (
                    f"target '{target_stem}' is indexed under {len(duplicate_indexed[target_stem])} distinct "
                    "chunks.source_path rows for the same physical file -- not unindexed. Remedy: dedupe "
                    "the chunk rows, don't reindex."
                )
            to_id = first_chunk_id(conn, target_source_path)
            if to_id is None:
                record["reason"] = f"target file '{target_stem}' indexed but no chunk row found"
                skipped.append(record)
                continue

            record["to_file"] = target_stem
            record["from_chunk_id"] = from_id
            record["to_chunk_id"] = to_id
            resolved.append(record)

    seen = set()
    deduped_resolved = []
    for r in resolved:
        k = (r["from_chunk_id"], r["to_chunk_id"])
        if k in seen:
            continue
        seen.add(k)
        deduped_resolved.append(r)

    return {
        "markdown_files_scanned": len(stem_index),
        "resolved": resolved,
        "resolved_deduped": deduped_resolved,
        "ambiguous": ambiguous,
        "dangling": dangling,
        "self_references": self_refs,
        "skipped_target_not_indexed": skipped,
        "skipped_unindexed_source": skipped_unindexed,
        "ignored_non_slug": ignored_non_slug,
        # File-level (not occurrence-level) diagnostic: which memory files
        # have >1 `chunks.source_path` row for the same physical file, and
        # the real remedy. These files' wikilink occurrences are still
        # classified normally above (resolved/dangling/etc, tagged with
        # duplicate_source/duplicate_target) -- this section exists so the
        # data-quality issue itself is visible and actionable on its own,
        # separately from any single occurrence.
        "duplicate_indexed_files": {
            stem: {"source_path_variants": variants, "remedy": "dedupe the duplicate chunk rows, do not reindex"}
            for stem, variants in duplicate_indexed.items()
        },
    }


def ensure_unique_index(conn: sqlite3.Connection) -> None:
    """Structural (not check-then-act) guarantee against double-inserting the
    same edge. A partial unique index scoped to relation_type='references'
    so it never constrains the pre-existing broke_because/fixed_by rows
    (many of which legitimately share a from_chunk_id with to_chunk_id
    NULL -- NULLs don't collide under a unique index either way, but the
    WHERE clause keeps this index's blast radius to exactly the rows this
    script owns).
    """
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_nugget_links_references_unique "
        "ON nugget_links(from_chunk_id, to_chunk_id) WHERE relation_type='references'"
    )


def apply_edges(conn: sqlite3.Connection, deduped_resolved: list[dict]) -> tuple[int, int]:
    """Insert the resolved edges. Idempotent under concurrent invocations:
    the partial unique index makes a duplicate edge a constraint violation
    at the DB level (not just a pre-check in this process), and the whole
    batch runs inside one BEGIN IMMEDIATE transaction so a second process
    calling this concurrently blocks (or gets SQLITE_BUSY, surfaced as an
    error rather than silently double-inserting) instead of interleaving
    with this one's SELECT/INSERT pair.
    """
    ensure_unique_index(conn)
    inserted = skipped_existing = 0
    conn.execute("BEGIN IMMEDIATE")
    try:
        for r in deduped_resolved:
            source_text = r.get("raw_wikilink") or f"[[{r['raw_slug']}]]"
            cur = conn.execute(
                "INSERT OR IGNORE INTO nugget_links (from_chunk_id, to_chunk_id, relation_type, confidence, source_text) "
                "VALUES (?, ?, 'references', 1.0, ?)",
                (r["from_chunk_id"], r["to_chunk_id"], source_text),
            )
            if cur.rowcount:
                inserted += 1
            else:
                skipped_existing += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return inserted, skipped_existing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(_default_db_path()), help="sqlite DB to read/write (default: %(default)s)")
    ap.add_argument("--memory-dir", default=str(MEMORY_DIR), help="memory markdown directory (default: %(default)s)")
    ap.add_argument("--apply", action="store_true", help="insert the resolved edges (default: dry-run report only)")
    ap.add_argument("--report", help="write the full classification as JSON to this path")
    args = ap.parse_args()

    db_path = Path(args.db)
    memory_dir = Path(args.memory_dir)
    # isolation_level=None -> autocommit / manual transaction control, so
    # apply_edges' explicit BEGIN IMMEDIATE is the only transaction boundary
    # (no implicit transaction from the driver to fight with).
    conn = sqlite3.connect(str(db_path), isolation_level=None)

    result = classify(conn, memory_dir)

    inserted = skipped_existing = 0
    if args.apply:
        inserted, skipped_existing = apply_edges(conn, result["resolved_deduped"])

    # Buckets are exhaustive and mutually exclusive by construction: this
    # sum must equal the raw regex occurrence count across every file, with
    # nothing dropped in between and self-references counted separately from
    # genuine dangling forward-refs.
    total_occurrences = (
        len(result["resolved"])
        + len(result["ambiguous"])
        + len(result["dangling"])
        + len(result["self_references"])
        + len(result["ignored_non_slug"])
        + len(result["skipped_target_not_indexed"])
        + len(result["skipped_unindexed_source"])
    )
    counts = {
        "db": str(db_path),
        "memory_dir": str(memory_dir),
        "apply": args.apply,
        "markdown_files_scanned": result["markdown_files_scanned"],
        "wikilink_occurrences_total": total_occurrences,
        "resolved_occurrences": len(result["resolved"]),
        "resolved_deduped_edges": len(result["resolved_deduped"]),
        "inserted": inserted,
        "skipped_existing_edge": skipped_existing,
        "ambiguous_left_unresolved": len(result["ambiguous"]),
        "dangling_forward_refs": len(result["dangling"]),
        "self_references": len(result["self_references"]),
        "ignored_non_slug": len(result["ignored_non_slug"]),
        "skipped_target_not_indexed": len(result["skipped_target_not_indexed"]),
        "skipped_unindexed_source": len(result["skipped_unindexed_source"]),
        # File-level count of the double-indexing data-quality issue (not a
        # classification bucket -- these files' occurrences are still
        # counted above under resolved/dangling/etc, tagged with
        # duplicate_source/duplicate_target in --report output).
        "duplicate_indexed_files": len(result["duplicate_indexed_files"]),
    }
    print(json.dumps(counts, indent=2))

    if args.report:
        Path(args.report).write_text(json.dumps({**counts, **result}, indent=2))

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
