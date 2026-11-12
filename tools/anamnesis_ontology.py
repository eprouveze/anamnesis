#!/usr/bin/env python3
"""anamnesis_ontology.py — git-first materialized operational ontology for Anamnesis.

Layers (each builds on the one before):
  - Coordinates: intensional logical coordinates (`heading_slug`, `chunk_key`) on `chunks`
    and `nugget_links` (`target_source_path`, `target_heading_slug`, `to_chunk_key`,
    `resolution_status`). Integer `to_chunk_id` foreign keys break whenever an
    indexer re-inserts a file, so links are also stored by stable coordinate and
    re-bound after every re-index.
  - Compiler: deterministic registry compiler (`compile_ontology`) projecting git-tracked
    registries into rebuildable SQLite views (`ontology_nodes`, `ontology_claims`,
    `ontology_evidence`) with no SQL CHECK enums and mandatory `provenance_origin`.
    Sources are configured in the dictionary's `registries` section (see
    docs/dictionary.md); defaults cover the dictionary's projects, `CLAUDE.md` /
    `AGENTS.md` policy annotations, a decisions log, capabilities, contacts, generic
    entity/relation files, enforcement witnesses and rule-firing telemetry.
  - Queries: query engines behind the `ontology_impact` and `ontology_audit` tools:
      1. `ontology_impact(entity_id, mode='failure'|'inspect'|'dependencies', depth=2)`
      2. `ontology_audit(scope='all'|'resolve'|'contradictions'|'unverified'|'link_health')`
  - Evidence and context: anti-polysemy evidence linker and ambient `ontology_context` enrichment
    for `brief()` and `recall()`.

This module is the ONLY writer of `ontology_*` tables. It is imported by the MCP
server and the indexer; keep it deployed alongside them.

Environment variables (all optional):
  ANAMNESIS_REPO_ROOT   root that relative registry paths resolve against
  ANAMNESIS_PATH_ROOTS  extra home-relative checkout dirs (os.pathsep separated) that
                        canonical source paths strip, besides `dev_root_candidates`
  ANAMNESIS_DB          database for the CLI (default $ANAMNESIS_HOME/memory.db)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from anamnesis_dict import (
    db_path as default_db_path,
    dictionary_path,
    get_ontology_node_kinds,
    get_ontology_predicates,
    get_ontology_setting,
    get_path_roots,
    get_policies,
    get_polysemy_blocklist,
    get_registries,
    get_store_types_with,
    load_dictionary,
    repo_root as default_repo_root,
)

# ---------------------------------------------------------------------------
# Constants & Canonical Coordinate Helpers
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]{4,}")

# Store types link repair may bind prose references to, and store types scanned
# for ontology evidence. Dictionary `store_types.<name>.resolve_target` /
# `.evidence_source` flags override these defaults.
_DEFAULT_RESOLVE_TARGET_STORES = (
    "feedback",
    "solution",
    "mistake",
    "decision",
    "project_meta",
    "project_notes",
)
_DEFAULT_EVIDENCE_STORES = _DEFAULT_RESOLVE_TARGET_STORES + ("memory",)

_HOME_RE_CACHE: Dict[Tuple[str, ...], Tuple["re.Pattern[str]", "re.Pattern[str]"]] = {}


def _resolve_target_stores() -> Tuple[str, ...]:
    return get_store_types_with("resolve_target", _DEFAULT_RESOLVE_TARGET_STORES)


def _evidence_stores() -> Tuple[str, ...]:
    return get_store_types_with("evidence_source", _DEFAULT_EVIDENCE_STORES)


def _is_mirror_path(raw_source_path: Optional[str]) -> bool:
    """True for chunk copies that live in disposable mirrors (e.g. git worktrees).

    Mirror copies must never become link targets or create score ties against the
    canonical file. Markers come from dictionary `ontology.exclude_path_markers`.
    """
    markers = get_ontology_setting("exclude_path_markers", []) or []
    raw = raw_source_path or ""
    return any(m and m in raw for m in markers)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify_heading(title: Optional[str]) -> str:
    """Normalize a chunk title/heading into a stable coordinate slug."""
    s = _SLUG_RE.sub("-", (title or "").strip().lower()).strip("-")
    return s[:120] if s else "root"


def slugify_id(raw: Optional[str]) -> str:
    s = _SLUG_RE.sub("-", (raw or "").strip().lower()).strip("-")
    return s[:80] if s else "unknown"


def _prefix_regexes() -> Tuple["re.Pattern[str]", "re.Pattern[str]"]:
    roots = tuple(get_path_roots())
    cached = _HOME_RE_CACHE.get(roots)
    if cached:
        return cached
    # <two-component absolute prefix, e.g. a home dir> or ~ , then a configured checkout root.
    root_alt = "|".join(re.escape(r) for r in sorted(roots, key=len, reverse=True)) or "(?!)"
    home_prefix = r"(?:/[^/]+/[^/]+|~)"
    checkout = re.compile(rf"^{home_prefix}/(?:{root_alt})/", re.IGNORECASE)
    dotdir = re.compile(rf"^{home_prefix}/(\.[A-Za-z0-9_-]+/.*)$")
    _HOME_RE_CACHE[roots] = (checkout, dotdir)
    return checkout, dotdir


def canonicalize_source_path(source_path: Optional[str]) -> str:
    """Normalize machine-specific checkout prefixes into a portable relative path.

    Prevents the same file indexed from two machines (different home directories,
    different checkout roots) from producing different chunk_keys. Roots come from
    the dictionary (`dev_root_candidates`, `path_roots`) and `ANAMNESIS_PATH_ROOTS`.
    When `canonical_default_project` is set, root-relative `docs/`, `scripts/`,
    `data/` ... paths are prefixed with that project name so they stay distinct
    from other projects' identically named files.
    """
    raw = (source_path or "").strip()
    if not raw:
        return ""
    checkout_re, dotdir_re = _prefix_regexes()
    m_dot = dotdir_re.match(raw)
    if m_dot:
        return m_dot.group(1)
    cleaned = checkout_re.sub("", raw)
    default_project = str(load_dictionary().get("canonical_default_project") or "").strip("/")
    if default_project and cleaned.startswith(
        ("docs/", "scripts/", "data/")
    ):
        return f"{default_project}/{cleaned}"
    return cleaned


def compute_chunk_key(source_path: Optional[str], title: Optional[str]) -> str:
    """Deterministic coordinate key `sha256(canonical_source_path :: heading_slug)[:24]`.

    Deliberately excludes any positional ordinal, so inserting a new section near
    the top of a file never invalidates downstream keys.
    """
    canon_path = canonicalize_source_path(source_path)
    h_slug = slugify_heading(title)
    payload = f"{canon_path}::{h_slug}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def compute_claim_id(
    subject_id: str,
    predicate: str,
    object_id_or_literal: str,
    source_path: str,
) -> str:
    payload = f"{subject_id}::{predicate}::{object_id_or_literal}::{source_path}".encode("utf-8")
    return "claim:" + hashlib.sha256(payload).hexdigest()[:20]


def compute_evidence_id(
    node_id: str,
    target_source_path: str,
    target_heading_slug: str,
    relation_role: str,
) -> str:
    payload = f"{node_id}::{target_source_path}::{target_heading_slug}::{relation_role}".encode("utf-8")
    return "ev:" + hashlib.sha256(payload).hexdigest()[:20]


# ---------------------------------------------------------------------------
# Coordinate and projection schema initialization (Additive, Tolerant-Read, Zero CHECK Enums)
# ---------------------------------------------------------------------------

CHUNK_COORD_COLUMNS = {
    "heading_slug": "TEXT",
    "chunk_key": "TEXT",
}

NUGGET_LINK_COORD_COLUMNS = {
    "target_source_path": "TEXT",
    "target_heading_slug": "TEXT",
    "to_chunk_key": "TEXT",
    "resolution_status": "TEXT DEFAULT 'unresolved'",
}


def has_ontology_tables(conn: sqlite3.Connection) -> bool:
    """Tolerant probe: do the ontology projection tables exist in this DB?"""
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('ontology_nodes', 'ontology_claims', 'ontology_evidence')"
        ).fetchall()
        return len(rows) == 3
    except sqlite3.Error:
        return False


def has_nugget_link_coord_columns(conn: sqlite3.Connection) -> bool:
    try:
        existing = {r[1] for r in conn.execute("PRAGMA table_info(nugget_links)").fetchall()}
        return all(c in existing for c in NUGGET_LINK_COORD_COLUMNS)
    except sqlite3.Error:
        return False


def ensure_ontology_schema(conn: sqlite3.Connection) -> None:
    """Idempotently ensure the coordinate columns and the projection tables.

    No DDL CHECK constraints are used: vocabulary lives in the git-tracked dictionary.
    """
    # 1. chunks coordinate columns (if chunks table exists)
    try:
        chunk_cols = {r[1] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()}
    except sqlite3.Error:
        chunk_cols = set()
    if chunk_cols:
        for col, ddl in CHUNK_COORD_COLUMNS.items():
            if col not in chunk_cols:
                conn.execute(f"ALTER TABLE chunks ADD COLUMN {col} {ddl}")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_chunks_coord ON chunks(source_path, heading_slug)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_chunks_chunk_key ON chunks(chunk_key)")

    # 2. nugget_links coordinate columns (if nugget_links table exists)
    try:
        nl_cols = {r[1] for r in conn.execute("PRAGMA table_info(nugget_links)").fetchall()}
    except sqlite3.Error:
        nl_cols = set()
    if nl_cols:
        for col, ddl in NUGGET_LINK_COORD_COLUMNS.items():
            if col not in nl_cols:
                conn.execute(f"ALTER TABLE nugget_links ADD COLUMN {col} {ddl}")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_nugget_links_coord "
            "ON nugget_links(target_source_path, target_heading_slug)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS ix_nugget_links_to_chunk_key ON nugget_links(to_chunk_key)")

    # 3. Ontology materialized projection tables
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS ontology_nodes (
            node_id TEXT PRIMARY KEY,
            node_kind TEXT NOT NULL,
            canonical_name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            source_registry TEXT NOT NULL,
            source_path TEXT NOT NULL,
            attributes_json TEXT NOT NULL DEFAULT '{}',
            aliases_json TEXT NOT NULL DEFAULT '[]',
            compiled_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_ontology_nodes_kind ON ontology_nodes(node_kind, status);

        CREATE TABLE IF NOT EXISTS ontology_claims (
            claim_id TEXT PRIMARY KEY,
            subject_id TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object_id_or_literal TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            source_path TEXT NOT NULL,
            evidence_span TEXT,
            provenance_origin TEXT NOT NULL,
            asserted_at TEXT,
            valid_from TEXT,
            valid_to TEXT,
            observed_at TEXT,
            confidence REAL NOT NULL DEFAULT 1.0,
            epistemic_status TEXT NOT NULL,
            governance_status TEXT NOT NULL DEFAULT 'active',
            verification_status TEXT NOT NULL DEFAULT 'unverified',
            supersedes_claim_id TEXT,
            compiled_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_ontology_claims_subj
            ON ontology_claims(subject_id, predicate, governance_status);
        CREATE INDEX IF NOT EXISTS ix_ontology_claims_obj
            ON ontology_claims(object_id_or_literal, predicate, governance_status);

        CREATE TABLE IF NOT EXISTS ontology_evidence (
            evidence_id TEXT PRIMARY KEY,
            node_id TEXT NOT NULL,
            claim_id TEXT,
            target_source_path TEXT NOT NULL,
            target_heading_slug TEXT NOT NULL,
            to_chunk_key TEXT,
            relation_role TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.7,
            compiled_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_ontology_evidence_node
            ON ontology_evidence(node_id, relation_role);
        CREATE INDEX IF NOT EXISTS ix_ontology_evidence_coord
            ON ontology_evidence(target_source_path, target_heading_slug);
        CREATE INDEX IF NOT EXISTS ix_ontology_evidence_chunk_key
            ON ontology_evidence(to_chunk_key);
        """
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Chunk Coordinate Backfill & `nugget_links` Foreign-Key Repair
# ---------------------------------------------------------------------------


def backfill_chunk_coordinates(conn: sqlite3.Connection, only_missing: bool = True) -> int:
    """Populate `heading_slug` and `chunk_key` on `chunks` rows.

    When a single file contains multiple chunks with the identical base heading
    (e.g. a >500-word section split into sub-chunks `:0`, `:1`, or repeated section
    titles in an append-only log), occurrences after the first in that file receive
    a deterministic per-heading occurrence suffix (`<slug>#2`, `<slug>#3`, ...).
    Inserting a new section with a new heading at the top of the file does not
    alter the occurrence count of any other heading in the file.
    """
    ensure_ontology_schema(conn)
    if only_missing:
        missing_count = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE chunk_key IS NULL OR chunk_key = ''"
        ).fetchone()[0]
        if missing_count == 0:
            return 0

    rows = conn.execute("SELECT id, source_path, title, chunk_key FROM chunks ORDER BY id ASC").fetchall()
    if not rows:
        return 0

    seen_per_file_heading: Dict[Tuple[str, str], int] = defaultdict(int)
    updates: List[Tuple[str, str, int]] = []
    for cid, sp, title, existing_ck in rows:
        canon_sp = canonicalize_source_path(sp)
        base_slug = slugify_heading(title)
        seen_per_file_heading[(canon_sp, base_slug)] += 1
        occ = seen_per_file_heading[(canon_sp, base_slug)]
        h_slug = base_slug if occ == 1 else f"{base_slug}#{occ}"
        payload = f"{canon_sp}::{h_slug}".encode("utf-8")
        c_key = hashlib.sha256(payload).hexdigest()[:24]
        if not only_missing or not existing_ck or existing_ck != c_key:
            updates.append((h_slug, c_key, cid))

    if updates:
        conn.executemany(
            "UPDATE chunks SET heading_slug = ?, chunk_key = ? WHERE id = ?",
            updates,
        )
        conn.commit()
    return len(updates)


def sync_nugget_links_after_reindex(conn: sqlite3.Connection) -> Dict[str, int]:
    """Re-bind `nugget_links.to_chunk_id` from stable `to_chunk_key` coordinates.

    Whenever the indexer runs `DELETE + INSERT` on a file, the integer
    `chunks.id` changes, but `chunk_key` (`sha256(canonical_source_path::heading_slug)`)
    stays invariant. This function restores `to_chunk_id` and updates
    `resolution_status` ('resolved' when unique, 'ambiguous' when multiple chunks
    share a duplicate key, 'unresolved' when absent).
    """
    has_links = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='nugget_links'"
    ).fetchone()
    if not has_links:
        return {"resolved": 0, "ambiguous": 0, "unresolved": 0}
    ensure_ontology_schema(conn)
    backfill_chunk_coordinates(conn, only_missing=True)

    # Map chunk_key -> list of (id, source_path, heading_slug), skipping mirror copies (worktrees)
    key_map: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)
    for cid, sp, hs, ck in conn.execute(
        "SELECT id, source_path, heading_slug, chunk_key FROM chunks WHERE chunk_key IS NOT NULL"
    ).fetchall():
        if _is_mirror_path(sp):
            continue
        key_map[ck].append((cid, canonicalize_source_path(sp), hs or "root"))

    rows = conn.execute(
        "SELECT id, to_chunk_id, to_chunk_key, target_source_path, target_heading_slug "
        "FROM nugget_links WHERE to_chunk_key IS NOT NULL AND to_chunk_key != ''"
    ).fetchall()

    updates: List[Tuple[Optional[int], str, str, str, int]] = []
    counts = {"resolved": 0, "ambiguous": 0, "unresolved": 0}
    for link_id, cur_to_id, ck, tsp, ths in rows:
        matches = key_map.get(ck, [])
        if len(matches) == 1:
            cid, canon_sp, h_slug = matches[0]
            updates.append((cid, canon_sp, h_slug, "resolved", link_id))
            counts["resolved"] += 1
        elif len(matches) > 1:
            cid, canon_sp, h_slug = matches[0]
            updates.append((cid, canon_sp, h_slug, "ambiguous", link_id))
            counts["ambiguous"] += 1
        else:
            updates.append((None, tsp or "", ths or "", "unresolved", link_id))
            counts["unresolved"] += 1

    if updates:
        conn.executemany(
            "UPDATE nugget_links SET to_chunk_id = ?, target_source_path = ?, "
            "target_heading_slug = ?, resolution_status = ? WHERE id = ?",
            updates,
        )
        conn.commit()
    return counts


def repair_nugget_links(conn: sqlite3.Connection, reevaluate_ambiguous: bool = True) -> Dict[str, int]:
    """Repair dangling or ambiguous `nugget_links.to_chunk_id` rows using BM25 + coordinate binding.

    1. Ensures `chunks.heading_slug` and `chunks.chunk_key` are populated (with intra-file
       duplicate heading disambiguation `#2`, `#3`, ...).
    2. Excludes mirror chunks (paths containing a configured `exclude_path_markers` entry,
       dictionary `ontology.exclude_path_markers`) from candidate targets so branch copies never create artificial
       score ties against the canonical files.
    3. Scores `source_text` (up to 20 informative tokens) against an in-memory IDF-weighted
       inverted index over target knowledge stores (`feedback`, `solution`, `mistake`,
       `decision`, `project_meta`, `project_notes`). Quarantines as `'ambiguous'` only when
       two distinct canonical documents tie within 2% relative BM25 score.
    """
    ensure_ontology_schema(conn)
    backfill_chunk_coordinates(conn, only_missing=False)

    # Build chunk_key multiplicity index over canonical (non-mirror) chunks
    key_counts: Counter[str] = Counter()
    live_chunks_by_id: Dict[int, Tuple[str, str, str]] = {}
    for cid, sp, hs, ck in conn.execute(
        "SELECT id, source_path, heading_slug, chunk_key FROM chunks"
    ).fetchall():
        canon_sp = canonicalize_source_path(sp)
        h_slug = hs or slugify_heading("")
        c_key = ck or compute_chunk_key(sp, h_slug)
        if not _is_mirror_path(sp):
            key_counts[c_key] += 1
            live_chunks_by_id[cid] = (canon_sp, h_slug, c_key)

    # Build fast in-memory inverted index over canonical target knowledge chunks
    target_stores = _resolve_target_stores()
    placeholders = ",".join("?" for _ in target_stores)
    raw_target_rows = conn.execute(
        f"SELECT id, source_path, title, content, heading_slug, chunk_key "
        f"FROM chunks WHERE store_type IN ({placeholders})",
        target_stores,
    ).fetchall()
    target_rows = [r for r in raw_target_rows if not _is_mirror_path(r[1])] or raw_target_rows

    docs: List[Tuple[int, str, str, str]] = []  # (chunk_id, canon_path, heading_slug, chunk_key)
    postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)  # token -> [(doc_idx, tf)]
    doc_lens: List[int] = []

    for doc_idx, (cid, sp, title, content, hs, ck) in enumerate(target_rows):
        canon_sp = canonicalize_source_path(sp)
        h_slug = hs or slugify_heading(title)
        c_key = ck or compute_chunk_key(sp, title)
        docs.append((cid, canon_sp, h_slug, c_key))
        text = f"{title or ''} {title or ''} {(content or '')[:1500]}".lower()
        toks = _TOKEN_RE.findall(text)
        doc_lens.append(max(len(toks), 1))
        tf_counts = Counter(toks)
        for tok, tf in tf_counts.items():
            postings[tok].append((doc_idx, tf))

    n_docs = max(len(docs), 1)
    avg_dl = (sum(doc_lens) / n_docs) if doc_lens else 1.0
    idf: Dict[str, float] = {
        tok: math.log(1.0 + (n_docs - len(plist) + 0.5) / (len(plist) + 0.5))
        for tok, plist in postings.items()
    }

    # Cache resolution by source_text
    text_resolution_cache: Dict[str, Tuple[Optional[int], str, str, str, str]] = {}

    def _resolve_prose(prose: str) -> Tuple[Optional[int], str, str, str, str]:
        if prose in text_resolution_cache:
            return text_resolution_cache[prose]
        q_toks = []
        seen_t: Set[str] = set()
        for t in _TOKEN_RE.findall((prose or "").lower()):
            if t not in seen_t and t in idf:
                seen_t.add(t)
                q_toks.append(t)
            if len(q_toks) >= 20:
                break
        if len(q_toks) < 2:
            res = (None, "", "", "", "unresolved")
            text_resolution_cache[prose] = res
            return res

        scores: Dict[int, float] = defaultdict(float)
        matched_terms: Dict[int, int] = defaultdict(int)
        k1 = 1.5
        b = 0.75
        min_idf = 0.8 if n_docs >= 20 else 0.1
        min_score = 10.0 if n_docs >= 20 else 1.5
        for tok in q_toks:
            w = idf[tok]
            if w < min_idf:
                continue  # skip ultra-common stop tokens
            for doc_idx, tf in postings[tok]:
                dl = doc_lens[doc_idx]
                denom = tf + k1 * (1.0 - b + b * (dl / avg_dl))
                scores[doc_idx] += w * ((tf * (k1 + 1.0)) / denom)
                matched_terms[doc_idx] += 1

        ranked = [
            (sc, doc_idx)
            for doc_idx, sc in scores.items()
            if matched_terms[doc_idx] >= 2 and sc >= min_score
        ]
        if not ranked:
            res = (None, "", "", "", "unresolved")
            text_resolution_cache[prose] = res
            return res

        ranked.sort(reverse=True)
        best_score, best_idx = ranked[0]
        cid, canon_sp, h_slug, c_key = docs[best_idx]

        # Only quarantine as ambiguous if duplicate key or a genuine <2% cross-document score tie
        status = "resolved"
        if key_counts[c_key] > 1:
            status = "ambiguous"
        else:
            for alt_score, alt_idx in ranked[1:5]:
                _, alt_sp, _, alt_key = docs[alt_idx]
                if alt_sp == canon_sp:
                    continue  # same underlying document, not a cross-doc ambiguity
                if (best_score - alt_score) / max(best_score, 1e-6) < 0.02:
                    status = "ambiguous"
                break

        res = (cid, canon_sp, h_slug, c_key, status)
        text_resolution_cache[prose] = res
        return res

    all_links = conn.execute(
        "SELECT id, to_chunk_id, source_text, to_chunk_key, resolution_status FROM nugget_links"
    ).fetchall()

    updates: List[Tuple[Optional[int], str, str, str, str, int]] = []
    stats = {"resolved": 0, "ambiguous": 0, "unresolved": 0, "total": len(all_links)}

    for link_id, to_cid, source_text, existing_ck, cur_status in all_links:
        if (
            to_cid is not None
            and to_cid in live_chunks_by_id
            and not (reevaluate_ambiguous and cur_status == "ambiguous")
        ):
            canon_sp, h_slug, c_key = live_chunks_by_id[to_cid]
            status = "ambiguous" if key_counts[c_key] > 1 else "resolved"
            updates.append((to_cid, canon_sp, h_slug, c_key, status, link_id))
            stats[status] += 1
            continue

        cid, canon_sp, h_slug, c_key, status = _resolve_prose(source_text or "")
        updates.append((cid, canon_sp, h_slug, c_key, status, link_id))
        stats[status] += 1

    if updates:
        conn.executemany(
            "UPDATE nugget_links SET to_chunk_id = ?, target_source_path = ?, "
            "target_heading_slug = ?, to_chunk_key = ?, resolution_status = ? WHERE id = ?",
            updates,
        )
        conn.commit()

    return stats


# ---------------------------------------------------------------------------
# Deterministic Registry Compiler (`compile_ontology`)
# ---------------------------------------------------------------------------

# Default registry sources, used for any key absent from the dictionary's
# `registries` section. Missing files are skipped silently, so these defaults
# are safe for any repository layout. Paths resolve against ANAMNESIS_REPO_ROOT
# (default: the repository root); absolute and `~/` paths are also accepted.
DEFAULT_REGISTRIES: Dict[str, Any] = {
    "rules_files": ["CLAUDE.md", "AGENTS.md"],
    "decision_logs": ["data/decisions/log.md"],
    "capabilities": ["data/capabilities.json"],
    "capability_edges": ["data/capability-edges.json"],
    "contacts": ["data/contacts.json"],
    "entities": ["data/entities.json"],
    "enforcement": {
        "dirs": ["tests", "scripts", "hooks", ".github/workflows"],
        "extensions": [".py", ".sh", ".yml", ".yaml"],
        "telemetry": ["data/rule-firings.jsonl"],
    },
}

# Last compile's non-fatal problems (malformed registry entries that were skipped).
LAST_COMPILE_WARNINGS: List[str] = []

_POLICY_ANNOTATION_RE = re.compile(r"<!--\s*policy:\s*(.*?)\s*-->")
_MD_HEADING_RE = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")
_DECISION_HEADING_RE = re.compile(r"^#{1,4}\s+(\d{4}-\d{2}-\d{2})\s*[—–:\-]+\s*(.+?)\s*$")
_BOLD_FIELD_RE = re.compile(r"^\*\*([^*]+?):?\*\*:?\s*(.*)$")


class _OntologyCompiler:
    def __init__(self, repo_root: Path, home_dir: Path) -> None:
        self.repo_root = repo_root
        self.home_dir = home_dir
        self.compiled_at = now_iso()
        self.valid_node_kinds = get_ontology_node_kinds()
        self.valid_predicates = get_ontology_predicates()
        self.polysemy_blocklist = get_polysemy_blocklist()
        self.registries = {**DEFAULT_REGISTRIES, **get_registries()}
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.claims: Dict[str, Dict[str, Any]] = {}
        self.warnings: List[str] = []

    # -- registry plumbing -------------------------------------------------

    def _path(self, raw: str) -> Path:
        raw = str(raw)
        if raw == "~" or raw.startswith("~/"):
            return self.home_dir / raw[2:]
        p = Path(raw)
        return p if p.is_absolute() else self.repo_root / p

    def _paths(self, key: str) -> List[Path]:
        val = self.registries.get(key) or []
        if isinstance(val, str):
            val = [val]
        return [self._path(v) for v in val]

    def _rel(self, path: Path) -> str:
        """Portable display path: relative to the repo root when inside it."""
        try:
            return path.resolve().relative_to(self.repo_root.resolve()).as_posix()
        except ValueError:
            return path.name

    def _warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def _read_json(self, path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._warn(f"{self._rel(path)}: unreadable registry ({exc.__class__.__name__})")
            return None

    # -- node / claim primitives ---------------------------------------------

    def add_node(
        self,
        node_id: str,
        node_kind: str,
        canonical_name: str,
        source_registry: str,
        source_path: str,
        status: str = "active",
        attributes: Optional[Dict[str, Any]] = None,
        aliases: Optional[List[str]] = None,
    ) -> None:
        if self.valid_node_kinds and node_kind not in self.valid_node_kinds:
            raise ValueError(f"Invalid ontology node_kind {node_kind!r}; not in dictionary.json")
        self.nodes[node_id] = {
            "node_id": node_id,
            "node_kind": node_kind,
            "canonical_name": canonical_name,
            "status": status,
            "source_registry": source_registry,
            "source_path": source_path,
            "attributes_json": json.dumps(attributes or {}, sort_keys=True),
            "aliases_json": json.dumps(sorted(set(aliases or [])), sort_keys=True),
            "compiled_at": self.compiled_at,
        }

    def add_claim(
        self,
        subject_id: str,
        predicate: str,
        object_id_or_literal: str,
        source_kind: str,
        source_path: str,
        provenance_origin: str,
        epistemic_status: str,
        governance_status: str = "active",
        verification_status: str = "enforced",
        confidence: float = 1.0,
        evidence_span: Optional[str] = None,
        asserted_at: Optional[str] = None,
        valid_from: Optional[str] = None,
        valid_to: Optional[str] = None,
        observed_at: Optional[str] = None,
        supersedes_claim_id: Optional[str] = None,
        custom_claim_id: Optional[str] = None,
    ) -> str:
        if self.valid_predicates and predicate not in self.valid_predicates:
            raise ValueError(f"Invalid ontology predicate {predicate!r}; not in dictionary.json")
        cid = custom_claim_id or compute_claim_id(
            subject_id, predicate, object_id_or_literal, source_path
        )
        self.claims[cid] = {
            "claim_id": cid,
            "subject_id": subject_id,
            "predicate": predicate,
            "object_id_or_literal": object_id_or_literal,
            "source_kind": source_kind,
            "source_path": source_path,
            "evidence_span": evidence_span,
            "provenance_origin": provenance_origin,
            "asserted_at": asserted_at,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "observed_at": observed_at,
            "confidence": confidence,
            "epistemic_status": epistemic_status,
            "governance_status": governance_status,
            "verification_status": verification_status,
            "supersedes_claim_id": supersedes_claim_id,
            "compiled_at": self.compiled_at,
        }
        return cid

    def _safe_node(self, **kw: Any) -> bool:
        try:
            self.add_node(**kw)
            return True
        except ValueError as exc:
            self._warn(f"{kw.get('source_path')}: {exc}")
            return False

    def _safe_claim(self, **kw: Any) -> Optional[str]:
        try:
            return self.add_claim(**kw)
        except ValueError as exc:
            self._warn(f"{kw.get('source_path')}: {exc}")
            return None

    # -- registries ----------------------------------------------------------

    def compile_dictionary(self) -> None:
        dict_data = load_dictionary()
        src = self._rel(dictionary_path())
        for proj in dict_data.get("projects", []):
            name = proj["name"]
            pid = f"project/{slugify_id(name)}"
            status = "active" if proj.get("enabled", True) else "archived"
            aliases = list(proj.get("aliases", [])) + [name]
            self.add_node(
                node_id=pid,
                node_kind="Project",
                canonical_name=name,
                source_registry=src,
                source_path=src,
                status=status,
                attributes={"description": proj.get("description", ""), "enabled": proj.get("enabled", True)},
                aliases=aliases,
            )

    def compile_entities(self) -> None:
        """Generic entity/relation registries: {"entities": [...], "relations": [...]}.

        entity:   {"id", "kind", "name", "status", "aliases", "attributes"}
        relation: {"subject", "predicate", "object", "origin", "verification",
                   "governance", "valid_from", "valid_to", "evidence"}
        Entries with an unknown kind/predicate are skipped with a warning.
        """
        for path in self._paths("entities"):
            if not path.exists():
                continue
            data = self._read_json(path)
            if not isinstance(data, dict):
                continue
            src = self._rel(path)
            for ent in data.get("entities", []):
                if not isinstance(ent, dict) or not ent.get("id") or not ent.get("kind"):
                    self._warn(f"{src}: entity needs id and kind: {str(ent)[:80]}")
                    continue
                self._safe_node(
                    node_id=str(ent["id"]),
                    node_kind=str(ent["kind"]),
                    canonical_name=str(ent.get("name") or ent["id"]),
                    source_registry=src,
                    source_path=src,
                    status=str(ent.get("status", "active")),
                    attributes=ent.get("attributes") or {},
                    aliases=[str(a) for a in ent.get("aliases", [])],
                )
            for rel in data.get("relations", []):
                if not isinstance(rel, dict) or not all(rel.get(k) for k in ("subject", "predicate", "object")):
                    self._warn(f"{src}: relation needs subject, predicate, object: {str(rel)[:80]}")
                    continue
                self._safe_claim(
                    subject_id=str(rel["subject"]),
                    predicate=str(rel["predicate"]),
                    object_id_or_literal=str(rel["object"]),
                    source_kind="registry",
                    source_path=src,
                    provenance_origin=str(rel.get("origin", "registry")),
                    epistemic_status="authoritative",
                    governance_status=str(rel.get("governance", "active")),
                    verification_status=str(rel.get("verification", "enforced")),
                    valid_from=rel.get("valid_from"),
                    valid_to=rel.get("valid_to"),
                    evidence_span=rel.get("evidence"),
                )

    def compile_capabilities(self) -> None:
        """Capabilities: {"projects": {name: {"status", "capabilities": [{capability_id,
        description, maturity}]}}}. Edges: {"edges": [{consumer, provider, capability_id,
        status, integration_pattern}]}; a consumer of "all" or "*" expands to every Project."""
        for path in self._paths("capabilities"):
            if not path.exists():
                continue
            cap_data = self._read_json(path)
            if not isinstance(cap_data, dict):
                continue
            src = self._rel(path)
            for proj_name, pbody in cap_data.get("projects", {}).items():
                pid = f"project/{slugify_id(proj_name)}"
                if pid not in self.nodes:
                    self.add_node(
                        node_id=pid,
                        node_kind="Project",
                        canonical_name=proj_name,
                        source_registry=src,
                        source_path=src,
                        status=pbody.get("status", "active"),
                        aliases=[proj_name],
                    )
                for cap in pbody.get("capabilities", []):
                    if not isinstance(cap, dict) or not cap.get("capability_id"):
                        self._warn(f"{src}: capability without capability_id under {proj_name}")
                        continue
                    cid = f"capability/{slugify_id(cap['capability_id'])}"
                    self.add_node(
                        node_id=cid,
                        node_kind="Capability",
                        canonical_name=cap["capability_id"],
                        source_registry=src,
                        source_path=src,
                        status="active",
                        attributes={
                            "description": cap.get("description", ""),
                            "maturity": cap.get("maturity", "production"),
                            "provider_project": pid,
                        },
                        aliases=[cap["capability_id"]],
                    )
                    self.add_claim(
                        subject_id=pid,
                        predicate="PROVIDES_CAPABILITY",
                        object_id_or_literal=cid,
                        source_kind="registry",
                        source_path=src,
                        provenance_origin="registry",
                        epistemic_status="authoritative",
                        verification_status="enforced",
                    )

        for path in self._paths("capability_edges"):
            if not path.exists():
                continue
            edges_data = self._read_json(path)
            if not isinstance(edges_data, dict):
                continue
            src = self._rel(path)
            for edge in edges_data.get("edges", []):
                consumer_raw = str(edge.get("consumer", ""))
                provider_raw = str(edge.get("provider", ""))
                if not consumer_raw or not provider_raw:
                    self._warn(f"{src}: edge needs consumer and provider")
                    continue
                cap_id = f"capability/{slugify_id(edge.get('capability_id', ''))}"
                edge_status = edge.get("status", "active")
                gov_status = "active" if edge_status == "active" else "proposed"
                ver_status = "enforced" if edge_status == "active" else "unverified"
                provider_id = f"project/{slugify_id(provider_raw)}"
                if consumer_raw.lower() in ("all", "*"):
                    consumers = sorted(
                        nid for nid, nd in self.nodes.items()
                        if nd["node_kind"] == "Project" and nd["status"] == "active"
                    )
                else:
                    consumers = [f"project/{slugify_id(consumer_raw)}"]
                for consumer_id in consumers:
                    self.add_claim(
                        subject_id=consumer_id,
                        predicate="USES_CAPABILITY",
                        object_id_or_literal=cap_id,
                        source_kind="registry",
                        source_path=src,
                        provenance_origin="registry",
                        epistemic_status="authoritative",
                        governance_status=gov_status,
                        verification_status=ver_status,
                        evidence_span=edge.get("integration_pattern"),
                    )
                    if edge_status == "active" and consumer_id != provider_id:
                        self.add_claim(
                            subject_id=consumer_id,
                            predicate="DEPENDS_ON",
                            object_id_or_literal=provider_id,
                            source_kind="registry",
                            source_path=src,
                            provenance_origin="registry",
                            epistemic_status="authoritative",
                            governance_status="active",
                            verification_status="enforced",
                            evidence_span=edge.get("integration_pattern"),
                        )

    def compile_contacts(self) -> None:
        """Contacts: {"contacts": [{name, role, organization, projects: [...]}]} (or a bare list)."""
        for path in self._paths("contacts"):
            if not path.exists():
                continue
            cdata = self._read_json(path)
            contacts = cdata.get("contacts", []) if isinstance(cdata, dict) else cdata
            if not isinstance(contacts, list):
                continue
            src = self._rel(path)
            for person in contacts:
                if not isinstance(person, dict) or not person.get("name"):
                    continue
                person_id = f"person/{slugify_id(person['name'])}"
                self.add_node(
                    node_id=person_id,
                    node_kind="Person",
                    canonical_name=person["name"],
                    source_registry=src,
                    source_path=src,
                    status="active",
                    attributes={"role": person.get("role", ""), "org": person.get("organization", "")},
                    aliases=[person["name"]],
                )
                for proj in person.get("projects", []) or []:
                    self._safe_claim(
                        subject_id=person_id,
                        predicate="CONTACT_FOR_PROJECT",
                        object_id_or_literal=f"project/{slugify_id(str(proj))}",
                        source_kind="registry",
                        source_path=src,
                        provenance_origin="registry",
                        epistemic_status="authoritative",
                        verification_status="enforced",
                    )

    # -- policies, enforcement witnesses, telemetry --------------------------

    def _annotated_policies(self) -> List[Dict[str, Any]]:
        """Policies declared inline in rules files (CLAUDE.md, AGENTS.md, ...).

        Annotation, placed under the heading that states the rule:
            <!-- policy: id=no-secrets scope=project/example-app enforced_by=gitleaks,pre-commit -->
        The heading is the policy name; the text up to the next heading is its
        description. `scope` defaults to "global"; `enforced_by` is a comma-separated
        list of tokens searched for in the enforcement directories.
        """
        out: List[Dict[str, Any]] = []
        for path in self._paths("rules_files"):
            if not path.exists():
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            src = self._rel(path)
            heading = ""
            for i, line in enumerate(lines):
                hm = _MD_HEADING_RE.match(line)
                if hm:
                    heading = hm.group(1)
                    continue
                am = _POLICY_ANNOTATION_RE.search(line)
                if not am:
                    continue
                try:
                    tokens = shlex.split(am.group(1))
                except ValueError:
                    self._warn(f"{src}:{i + 1}: unparsable policy annotation")
                    continue
                kv = dict(t.split("=", 1) for t in tokens if "=" in t)
                if not kv.get("id"):
                    self._warn(f"{src}:{i + 1}: policy annotation without id=")
                    continue
                body: List[str] = []
                for nxt in lines[i + 1:]:
                    if _MD_HEADING_RE.match(nxt) or _POLICY_ANNOTATION_RE.search(nxt):
                        break
                    if nxt.strip():
                        body.append(nxt.strip())
                out.append({
                    "id": kv["id"],
                    "name": kv.get("name") or heading or kv["id"],
                    "scope": kv.get("scope", "global"),
                    "source_path": src,
                    "evidence_span": f"{src}#{slugify_heading(heading)}",
                    "description": " ".join(body)[:400],
                    "enforcement_patterns": [t for t in kv.get("enforced_by", "").split(",") if t],
                })
        return out

    def _policies(self) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for pol in self._annotated_policies() + get_policies():
            merged[slugify_id(str(pol["id"]))] = pol  # dictionary entries win over annotations
        norm = []
        for pid, pol in merged.items():
            norm.append({
                "node_id": f"policy/{pid}",
                "rule_id": str(pol["id"]),
                "name": pol.get("name") or str(pol["id"]),
                "scope": pol.get("scope", "global"),
                "source_path": pol.get("source_path", "dictionary.json"),
                "evidence_span": pol.get("evidence_span"),
                "description": pol.get("description", ""),
                "enforcement_patterns": list(pol.get("enforcement_patterns", [])),
                "origin": pol.get("origin", "human"),
                "valid_from": pol.get("valid_from"),
            })
        return norm

    def _enforcement_corpus(self) -> List[Tuple[str, str]]:
        cfg = self.registries.get("enforcement") or {}
        exts = tuple(cfg.get("extensions", DEFAULT_REGISTRIES["enforcement"]["extensions"]))
        files: List[Tuple[str, str]] = []
        for rel_dir in cfg.get("dirs", DEFAULT_REGISTRIES["enforcement"]["dirs"]):
            dpath = self._path(rel_dir)
            if not dpath.is_dir():
                continue
            for fpath in sorted(dpath.iterdir()):
                if fpath.is_file() and fpath.suffix in exts:
                    try:
                        files.append((self._rel(fpath), fpath.read_text(encoding="utf-8", errors="replace")))
                    except OSError:
                        pass
        return files

    def _telemetry(self) -> Tuple[Counter, Dict[str, str], str]:
        """Rule-firing telemetry JSONL: {"rule": id, "ts": iso}. Returns counts, last-seen, path."""
        cfg = self.registries.get("enforcement") or {}
        counts: Counter = Counter()
        last_seen: Dict[str, str] = {}
        used = ""
        for raw in cfg.get("telemetry", DEFAULT_REGISTRIES["enforcement"]["telemetry"]):
            tpath = self._path(raw)
            if not tpath.exists():
                continue
            used = used or self._rel(tpath)
            try:
                for line in tpath.read_text(encoding="utf-8", errors="replace").splitlines():
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rid = str(obj.get("rule") or obj.get("rule_id") or "").strip()
                    if rid:
                        counts[rid] += 1
                        ts = str(obj.get("ts") or obj.get("timestamp") or "")
                        if ts > last_seen.get(rid, ""):
                            last_seen[rid] = ts
            except OSError:
                pass
        return counts, last_seen, used

    def compile_policies(self) -> None:
        search_files = self._enforcement_corpus()
        telemetry_counts, telemetry_last_seen, telemetry_path = self._telemetry()

        for pol in self._policies():
            pid = pol["node_id"]
            patterns = pol["enforcement_patterns"]
            enforcing_files: List[str] = []
            if patterns:
                for rel_p, text in search_files:
                    if any(pat in text or pat in rel_p for pat in patterns):
                        enforcing_files.append(rel_p)

            rule_id = pol["rule_id"]
            fired_count = telemetry_counts.get(rule_id, 0)
            last_fired = telemetry_last_seen.get(rule_id)

            if enforcing_files:
                ver_status = "enforced"
            elif fired_count > 0:
                ver_status = "observed"
            else:
                ver_status = "unverified"

            self.add_node(
                node_id=pid,
                node_kind="Policy",
                canonical_name=pol["name"],
                source_registry=pol["source_path"],
                source_path=pol["source_path"],
                status="active",
                attributes={
                    "description": pol["description"],
                    "scope": pol["scope"],
                    "verification_status": ver_status,
                    "enforcing_file_count": len(enforcing_files),
                    "telemetry_fire_count": fired_count,
                },
                aliases=[rule_id, pol["name"]],
            )
            self.add_claim(
                subject_id=pid,
                predicate="GOVERNS_SCOPE",
                object_id_or_literal=pol["scope"],
                source_kind="policy_doc",
                source_path=pol["source_path"],
                evidence_span=pol["evidence_span"],
                provenance_origin=pol["origin"],
                epistemic_status="authoritative",
                governance_status="active",
                verification_status=ver_status,
                valid_from=pol["valid_from"],
            )

            for ef in enforcing_files[:5]:
                gate_node = f"service/gate-{slugify_id(Path(ef).stem)}"
                if gate_node not in self.nodes:
                    self.add_node(
                        node_id=gate_node,
                        node_kind="Service",
                        canonical_name=f"Gate: {Path(ef).name}",
                        source_registry=ef,
                        source_path=ef,
                        status="active",
                        attributes={"gate_file": ef},
                        aliases=[Path(ef).name],
                    )
                self.add_claim(
                    subject_id=pid,
                    predicate="ENFORCED_BY",
                    object_id_or_literal=gate_node,
                    source_kind="test_gate",
                    source_path=ef,
                    provenance_origin="registry",
                    epistemic_status="verified_code",
                    governance_status="active",
                    verification_status="enforced",
                )

            if fired_count > 0:
                self.add_claim(
                    subject_id=pid,
                    predicate="OBSERVED_IN",
                    object_id_or_literal=f"telemetry:rule-firings:{fired_count}",
                    source_kind="telemetry",
                    source_path=telemetry_path,
                    provenance_origin="telemetry",
                    epistemic_status="observed_runtime",
                    governance_status="active",
                    verification_status="observed",
                    observed_at=last_fired,
                    confidence=0.95,
                )

    # -- decision logs -------------------------------------------------------

    def _parse_decision_log(self, path: Path) -> List[Dict[str, Any]]:
        """Entries headed `### YYYY-MM-DD — title`; `**Field:** value` lines follow.

        Recognised optional fields: Topic, Scope, Supersedes, Origin, Status.
        """
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []
        entries: List[Dict[str, Any]] = []
        cur: Optional[Dict[str, Any]] = None
        for line in lines:
            hm = _DECISION_HEADING_RE.match(line)
            if hm:
                cur = {"date": hm.group(1), "title": hm.group(2), "fields": {}}
                entries.append(cur)
                continue
            if cur is not None:
                fm = _BOLD_FIELD_RE.match(line.strip())
                if fm:
                    cur["fields"][fm.group(1).strip().lower()] = fm.group(2).strip()
        return entries

    def compile_decision_logs(self) -> None:
        valid_topic = not self.valid_node_kinds or "Topic" in self.valid_node_kinds
        for path in self._paths("decision_logs"):
            if not path.exists():
                continue
            src = self._rel(path)
            entries = self._parse_decision_log(path)
            seen: Counter = Counter()
            for e in entries:
                base = f"decision/{e['date']}-{slugify_id(e['title'])[:60]}"
                seen[base] += 1
                e["node_id"] = base if seen[base] == 1 else f"{base}-{seen[base]}"
                e["claim_id"] = "claim:" + hashlib.sha256(e["node_id"].encode()).hexdigest()[:20]

            def _find(token: str) -> Optional[Dict[str, Any]]:
                tok = slugify_id(token)
                hits = [x for x in entries if x["node_id"].split("/", 1)[1] == tok
                        or x["date"] == token.strip()
                        or x["node_id"].split("/", 1)[1].startswith(tok)]
                return hits[0] if len(hits) == 1 else None

            superseded_by: Dict[str, Dict[str, Any]] = {}
            for e in entries:
                for tok in [t for t in re.split(r"[,;]", e["fields"].get("supersedes", "")) if t.strip()]:
                    old = _find(tok)
                    if old is None:
                        self._warn(f"{src}: Supersedes {tok.strip()!r} in {e['node_id']} matches no unique entry")
                    elif old is not e:
                        superseded_by[old["node_id"]] = e

            for e in entries:
                nid = e["node_id"]
                fields = e["fields"]
                newer = superseded_by.get(nid)
                status = "superseded" if newer else "active"
                origin = fields.get("origin", "registry").strip().lower() or "registry"
                topic = fields.get("topic", "").strip()
                scope = fields.get("scope", "").strip()
                self.add_node(
                    node_id=nid,
                    node_kind="Decision",
                    canonical_name=e["title"],
                    source_registry=src,
                    source_path=src,
                    status=status,
                    attributes={"asserted_at": e["date"], "provenance_origin": origin, "topic": topic},
                    aliases=[nid.split("/", 1)[1]],
                )
                common = dict(
                    source_kind="decision_log",
                    source_path=src,
                    evidence_span=nid,
                    provenance_origin=origin,
                    epistemic_status="extracted_prose" if origin == "agent_inferred" else "authoritative",
                    governance_status=status,
                    verification_status="observed",
                    asserted_at=e["date"],
                    valid_from=e["date"],
                    valid_to=newer["date"] if newer else None,
                )
                if topic and valid_topic:
                    tid = f"topic/{slugify_id(topic)}"
                    if tid not in self.nodes:
                        self.add_node(tid, "Topic", topic, src, src, aliases=[topic])
                    # Two active decisions on one topic that do not supersede each other
                    # collide on (topic, GOVERNS_SCOPE) and surface in the contradictions audit.
                    older_claims = [
                        x["claim_id"] for x in entries
                        if superseded_by.get(x["node_id"]) is e
                        and x["fields"].get("topic", "").strip() == topic
                    ]
                    self.add_claim(
                        subject_id=tid,
                        predicate="GOVERNS_SCOPE",
                        object_id_or_literal=nid,
                        supersedes_claim_id=older_claims[0] if older_claims else None,
                        custom_claim_id=e["claim_id"],
                        **common,
                    )
                if scope:
                    scope_id = f"project/{slugify_id(scope)}" if f"project/{slugify_id(scope)}" in self.nodes else scope
                    self._safe_claim(
                        subject_id=nid,
                        predicate="GOVERNS_SCOPE",
                        object_id_or_literal=scope_id,
                        **common,
                    )
                for old in [x for x in entries if superseded_by.get(x["node_id"]) is e]:
                    self._safe_claim(
                        subject_id=nid,
                        predicate="SUPERSEDES",
                        object_id_or_literal=old["node_id"],
                        source_kind="decision_log",
                        source_path=src,
                        provenance_origin=origin,
                        epistemic_status=common["epistemic_status"],
                        governance_status="active",
                        verification_status="observed",
                        asserted_at=e["date"],
                        valid_from=e["date"],
                    )


def _link_chunk_evidence(
    conn: sqlite3.Connection,
    nodes: Dict[str, Dict[str, Any]],
    polysemy_blocklist: Set[str],
    compiled_at: str,
) -> int:
    """Deterministic anti-polysemy evidence linker over high-signal chunk stores.

    Links chunks to canonical `node_id`s in `ontology_evidence` without LLM calls:
      - Bare common words in `polysemy_blocklist` (e.g. `plan`, `close`, `work`)
        are NEVER matched in unstructured prose unless prefixed by namespace
        (`project/...`, `machine/...`, `worker/...`).
      - Chunks mentioning >= 2 entities record `co_mentioned_in_chunk` on
        `ontology_evidence` and are NEVER promoted into `ontology_claims`
        (`DEPENDS_ON` / `RUNS_ON`).
    """
    alias_to_node: Dict[str, str] = {}
    for nid, nd in nodes.items():
        # Always allow the explicit namespace ID (e.g. 'machine/build-host')
        alias_to_node[nid.lower()] = nid
        for al in json.loads(nd["aliases_json"]):
            al_clean = al.strip()
            if len(al_clean) < 3:
                continue
            if al_clean.lower() in polysemy_blocklist:
                continue
            alias_to_node[al_clean.lower()] = nid

    # Build a single compiled regex for safe aliases
    escaped = sorted((re.escape(k) for k in alias_to_node.keys()), key=len, reverse=True)
    if not escaped:
        return 0
    pattern = re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)

    # Scan high-signal stores in chunks
    target_stores = _evidence_stores()
    placeholders = ",".join("?" for _ in target_stores)
    rows = conn.execute(
        f"SELECT source_path, title, content, store_type, project, heading_slug, chunk_key "
        f"FROM chunks WHERE store_type IN ({placeholders})",
        target_stores,
    ).fetchall()

    evidence_rows: Dict[str, Tuple[str, str, Optional[str], str, str, str, str, float, str]] = {}

    for sp, title, content, stype, proj, hs, ck in rows:
        canon_sp = canonicalize_source_path(sp)
        h_slug = hs or slugify_heading(title)
        c_key = ck or compute_chunk_key(sp, title)
        text = f"{title or ''}\n{(content or '')[:2000]}"

        matched_text_nodes: Set[str] = set()
        for m in pattern.finditer(text):
            nid = alias_to_node.get(m.group(1).lower())
            if nid:
                matched_text_nodes.add(nid)

        matched_nodes: Set[str] = set(matched_text_nodes)
        # Also bind project column directly if it maps to a known project node
        if proj:
            proj_nid = f"project/{slugify_id(proj)}"
            if proj_nid in nodes:
                matched_nodes.add(proj_nid)

        if not matched_nodes:
            continue

        is_incident = stype == "mistake" or any(
            w in (title or "").lower() for w in ("outage", "crash-loop", "broke", "incident", "failure")
        )
        for nid in matched_nodes:
            role = "incident_evidence" if is_incident else (
                "co_mentioned_in_chunk" if len(matched_text_nodes) > 1 else (
                    "decision_rationale" if stype == "decision" else "mentioned_in_chunk"
                )
            )
            ev_id = compute_evidence_id(nid, canon_sp, h_slug, role)
            evidence_rows[ev_id] = (
                ev_id,
                nid,
                None,
                canon_sp,
                h_slug,
                c_key,
                role,
                0.85 if is_incident else 0.70,
                compiled_at,
            )

    conn.execute("DELETE FROM ontology_evidence")
    if evidence_rows:
        conn.executemany(
            "INSERT OR REPLACE INTO ontology_evidence "
            "(evidence_id, node_id, claim_id, target_source_path, target_heading_slug, "
            "to_chunk_key, relation_role, confidence, compiled_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            list(evidence_rows.values()),
        )
    return len(evidence_rows)


def compile_ontology(
    conn: sqlite3.Connection,
    repo_root: Optional[Path] = None,
    home_dir: Optional[Path] = None,
    link_evidence: bool = True,
) -> Dict[str, int]:
    """Compile all configured registries into `ontology_*` tables.

    This is the ONLY permitted writer to `ontology_nodes`, `ontology_claims`,
    and `ontology_evidence`. Malformed registry entries are skipped and reported
    through `LAST_COMPILE_WARNINGS` / the `warnings` count, never silently dropped.
    """
    ensure_ontology_schema(conn)
    root = repo_root or default_repo_root()
    home = home_dir or Path.home()

    compiler = _OntologyCompiler(repo_root=root, home_dir=home)
    compiler.compile_dictionary()
    compiler.compile_entities()
    compiler.compile_capabilities()
    compiler.compile_contacts()
    compiler.compile_policies()
    compiler.compile_decision_logs()
    LAST_COMPILE_WARNINGS[:] = compiler.warnings

    conn.execute("DELETE FROM ontology_nodes")
    conn.execute("DELETE FROM ontology_claims")

    conn.executemany(
        "INSERT OR REPLACE INTO ontology_nodes "
        "(node_id, node_kind, canonical_name, status, source_registry, source_path, "
        "attributes_json, aliases_json, compiled_at) "
        "VALUES (:node_id, :node_kind, :canonical_name, :status, :source_registry, :source_path, "
        ":attributes_json, :aliases_json, :compiled_at)",
        list(compiler.nodes.values()),
    )
    conn.executemany(
        "INSERT OR REPLACE INTO ontology_claims "
        "(claim_id, subject_id, predicate, object_id_or_literal, source_kind, source_path, "
        "evidence_span, provenance_origin, asserted_at, valid_from, valid_to, observed_at, "
        "confidence, epistemic_status, governance_status, verification_status, "
        "supersedes_claim_id, compiled_at) "
        "VALUES (:claim_id, :subject_id, :predicate, :object_id_or_literal, :source_kind, :source_path, "
        ":evidence_span, :provenance_origin, :asserted_at, :valid_from, :valid_to, :observed_at, "
        ":confidence, :epistemic_status, :governance_status, :verification_status, "
        ":supersedes_claim_id, :compiled_at)",
        list(compiler.claims.values()),
    )

    ev_count = 0
    if link_evidence:
        ev_count = _link_chunk_evidence(
            conn,
            compiler.nodes,
            compiler.polysemy_blocklist,
            compiler.compiled_at,
        )

    conn.commit()
    return {
        "nodes": len(compiler.nodes),
        "claims": len(compiler.claims),
        "evidence": ev_count,
        "warnings": len(compiler.warnings),
    }


# ---------------------------------------------------------------------------
# Query Engines for `ontology_impact` and `ontology_audit`
# ---------------------------------------------------------------------------


def resolve_entity_id(conn: sqlite3.Connection, raw_query: str) -> Optional[str]:
    """Resolve a canonical `node_id`, canonical name or alias (e.g. `api` -> `service/api`)."""
    if not has_ontology_tables(conn):
        return None
    q = (raw_query or "").strip()
    if not q:
        return None
    row = conn.execute(
        "SELECT node_id FROM ontology_nodes WHERE LOWER(node_id) = LOWER(?) OR LOWER(canonical_name) = LOWER(?)",
        (q, q),
    ).fetchone()
    if row:
        return row[0]

    q_low = q.lower()
    for nid, aliases_json in conn.execute("SELECT node_id, aliases_json FROM ontology_nodes").fetchall():
        try:
            aliases = json.loads(aliases_json or "[]")
        except json.JSONDecodeError:
            aliases = []
        if any(a.lower() == q_low for a in aliases):
            return nid
    return None


def query_ontology_impact(
    conn: sqlite3.Connection,
    entity_id: str,
    mode: str = "failure",
    depth: int = 2,
    include_evidence: bool = True,
) -> Dict[str, Any]:
    """Execute `ontology_impact(entity_id, mode='failure'|'inspect'|'dependencies', depth=2)`."""
    if not has_ontology_tables(conn):
        return {"error": "Ontology projection tables not yet compiled in this database."}

    resolved_id = resolve_entity_id(conn, entity_id)
    if not resolved_id:
        return {"error": f"Entity {entity_id!r} not found in ontology_nodes."}

    depth = max(1, min(int(depth), 5))
    node_row = conn.execute(
        "SELECT node_id, node_kind, canonical_name, status, source_registry, source_path, attributes_json, aliases_json "
        "FROM ontology_nodes WHERE node_id = ?",
        (resolved_id,),
    ).fetchone()
    node_dict = {
        "node_id": node_row[0],
        "node_kind": node_row[1],
        "canonical_name": node_row[2],
        "status": node_row[3],
        "source_registry": node_row[4],
        "source_path": node_row[5],
        "attributes": json.loads(node_row[6] or "{}"),
        "aliases": json.loads(node_row[7] or "[]"),
    }

    # Fetch direct outbound and inbound active claims
    outbound = [
        dict(r)
        for r in conn.execute(
            "SELECT claim_id, subject_id, predicate, object_id_or_literal, source_kind, source_path, "
            "provenance_origin, epistemic_status, governance_status, verification_status "
            "FROM ontology_claims WHERE subject_id = ? AND governance_status = 'active'",
            (resolved_id,),
        ).fetchall()
    ]
    inbound = [
        dict(r)
        for r in conn.execute(
            "SELECT claim_id, subject_id, predicate, object_id_or_literal, source_kind, source_path, "
            "provenance_origin, epistemic_status, governance_status, verification_status "
            "FROM ontology_claims WHERE object_id_or_literal = ? AND governance_status = 'active'",
            (resolved_id,),
        ).fetchall()
    ]

    # Fetch evidence chunks via intensional logical coordinates
    evidence_list: List[Dict[str, Any]] = []
    if include_evidence:
        ev_rows = conn.execute(
            """
            SELECT oe.evidence_id, oe.relation_role, oe.confidence,
                   oe.target_source_path, oe.target_heading_slug, oe.to_chunk_key,
                   c.id AS chunk_row_id, c.chunk_id, c.title, c.store_type, c.project
            FROM ontology_evidence oe
            LEFT JOIN chunks c ON c.chunk_key = oe.to_chunk_key
            WHERE oe.node_id = ?
            ORDER BY CASE oe.relation_role
                WHEN 'incident_evidence' THEN 0
                WHEN 'decision_rationale' THEN 1
                WHEN 'mentioned_in_chunk' THEN 2
                ELSE 3
            END, oe.confidence DESC
            LIMIT 15
            """,
            (resolved_id,),
        ).fetchall()
        for r in ev_rows:
            evidence_list.append({
                "evidence_id": r[0],
                "relation_role": r[1],
                "confidence": r[2],
                "target_source_path": r[3],
                "target_heading_slug": r[4],
                "to_chunk_key": r[5],
                "resolved": r[6] is not None,
                "chunk": (
                    {
                        "id": r[6],
                        "chunk_id": r[7],
                        "title": r[8],
                        "store_type": r[9],
                        "project": r[10],
                    }
                    if r[6] is not None
                    else None
                ),
            })

    if mode == "inspect":
        return {
            "mode": "inspect",
            "entity": node_dict,
            "outbound_claims": outbound,
            "inbound_claims": inbound,
            "prose_evidence": evidence_list,
        }

    # Recursive CTE for dependency & failure impact traversal
    # Note: co_mentioned_in_chunk lives in ontology_evidence, NEVER in ontology_claims,
    # so co-mentions can never leak into dependency or blast-radius walks.
    if mode == "dependencies":
        cte_rows = conn.execute(
            """
            WITH RECURSIVE dep_walk(hop, from_id, predicate, to_id, source_path, verification_status, path) AS (
                SELECT 1, subject_id, predicate, object_id_or_literal, source_path, verification_status,
                       subject_id || '->' || object_id_or_literal
                FROM ontology_claims
                WHERE subject_id = ?
                  AND governance_status = 'active'
                  AND predicate IN ('DEPENDS_ON', 'RUNS_ON', 'USES_CAPABILITY', 'CONSUMES_QUOTA', 'REPLICATES_TO')
                UNION ALL
                SELECT w.hop + 1, c.subject_id, c.predicate, c.object_id_or_literal, c.source_path, c.verification_status,
                       w.path || '->' || c.object_id_or_literal
                FROM ontology_claims c
                JOIN dep_walk w ON c.subject_id = w.to_id
                WHERE w.hop < ?
                  AND c.governance_status = 'active'
                  AND c.predicate IN ('DEPENDS_ON', 'RUNS_ON', 'USES_CAPABILITY', 'PROVIDES_CAPABILITY', 'CONSUMES_QUOTA')
                  AND instr(w.path, c.object_id_or_literal) = 0
            )
            SELECT DISTINCT hop, from_id, predicate, to_id, source_path, verification_status FROM dep_walk ORDER BY hop, from_id
            """,
            (resolved_id, depth),
        ).fetchall()
        return {
            "mode": "dependencies",
            "entity": node_dict,
            "dependencies": [
                {
                    "hop": r[0],
                    "from_id": r[1],
                    "predicate": r[2],
                    "to_id": r[3],
                    "source_path": r[4],
                    "verification_status": r[5],
                }
                for r in cte_rows
            ],
            "prose_evidence": evidence_list,
        }

    # mode == "failure": "What breaks if X disappears right now?"
    impact_rows = conn.execute(
        """
        WITH RECURSIVE blast(hop, affected_id, via_predicate, failed_dependency_id, source_path, verification_status, path) AS (
            SELECT 1, c.subject_id, c.predicate, c.object_id_or_literal, c.source_path, c.verification_status,
                   c.object_id_or_literal || '<-' || c.subject_id
            FROM ontology_claims c
            WHERE c.object_id_or_literal = ?
              AND c.governance_status = 'active'
              AND c.predicate IN ('RUNS_ON', 'DEPENDS_ON', 'USES_CAPABILITY', 'GOVERNS_SCOPE', 'DECIDED_FOR')
            UNION ALL
            SELECT b.hop + 1, c.subject_id, c.predicate, c.object_id_or_literal, c.source_path, c.verification_status,
                   b.path || '<-' || c.subject_id
            FROM ontology_claims c
            JOIN blast b ON c.object_id_or_literal = b.affected_id
            WHERE b.hop < ?
              AND c.governance_status = 'active'
              AND c.predicate IN ('RUNS_ON', 'DEPENDS_ON', 'USES_CAPABILITY', 'PROVIDES_CAPABILITY')
              AND instr(b.path, c.subject_id) = 0
        )
        SELECT DISTINCT b.hop, b.affected_id, n.node_kind, n.canonical_name, n.status,
                        b.via_predicate, b.failed_dependency_id, b.source_path, b.verification_status
        FROM blast b
        LEFT JOIN ontology_nodes n ON n.node_id = b.affected_id
        ORDER BY b.hop, b.affected_id
        """,
        (resolved_id, depth),
    ).fetchall()

    direct_impact: List[Dict[str, Any]] = []
    transitive_impact: List[Dict[str, Any]] = []
    affected_service_ids: Set[str] = {resolved_id}

    for r in impact_rows:
        item = {
            "hop": r[0],
            "affected_id": r[1],
            "node_kind": r[2],
            "canonical_name": r[3],
            "status": r[4],
            "via_predicate": r[5],
            "failed_dependency_id": r[6],
            "source_path": r[7],
            "verification_status": r[8],
        }
        affected_service_ids.add(r[1])
        if r[0] == 1:
            direct_impact.append(item)
        else:
            transitive_impact.append(item)

    # Mitigations / Fallbacks & Replicas for any affected service/node
    id_placeholders = ",".join("?" for _ in affected_service_ids)
    fallback_rows = conn.execute(
        f"SELECT subject_id, predicate, object_id_or_literal, source_path "
        f"FROM ontology_claims "
        f"WHERE subject_id IN ({id_placeholders}) "
        f"AND governance_status = 'active' "
        f"AND predicate IN ('FALLBACK_TO', 'REPLICATES_TO')",
        tuple(affected_service_ids),
    ).fetchall()

    fallback_by_subj: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    mitigations: List[Dict[str, str]] = []
    for subj, pred, obj, sp in fallback_rows:
        entry = {"subject_id": subj, "predicate": pred, "target_id": obj, "source_path": sp}
        if pred == "FALLBACK_TO":
            fallback_by_subj[subj].append(entry)
        mitigations.append(entry)

    # Single Points of Failure: active services/workers directly on `resolved_id` with 0 FALLBACK_TO claims
    spof_kinds = set(get_ontology_setting("spof_kinds", ["Service", "Worker"]) or [])
    spofs: List[Dict[str, Any]] = []
    for item in direct_impact:
        if item["node_kind"] in spof_kinds and item["status"] == "active":
            if not fallback_by_subj.get(item["affected_id"]):
                spofs.append({
                    "node_id": item["affected_id"],
                    "canonical_name": item["canonical_name"],
                    "node_kind": item["node_kind"],
                    "reason": f"Runs on {resolved_id} with 0 active FALLBACK_TO claims",
                    "source_path": item["source_path"],
                })

    # Historical Incidents linked to the entity or its directly affected services
    incident_rows = conn.execute(
        f"SELECT n.node_id, n.canonical_name, n.source_path, c.object_id_or_literal "
        f"FROM ontology_nodes n "
        f"JOIN ontology_claims c ON c.subject_id = n.node_id "
        f"WHERE n.node_kind = 'Incident' AND c.object_id_or_literal IN ({id_placeholders})",
        tuple(affected_service_ids),
    ).fetchall()
    historical_incidents = [
        {
            "incident_id": r[0],
            "title": r[1],
            "source_path": r[2],
            "affected_target": r[3],
        }
        for r in incident_rows
    ]
    for ev in evidence_list:
        if ev["relation_role"] == "incident_evidence" and ev["resolved"]:
            historical_incidents.append({
                "incident_id": ev["chunk"]["chunk_id"],
                "title": ev["chunk"]["title"],
                "source_path": ev["target_source_path"],
                "affected_target": resolved_id,
            })

    unverified_gaps = [
        item for item in (direct_impact + transitive_impact)
        if item.get("verification_status") == "unverified"
    ]

    return {
        "mode": "failure",
        "entity": node_dict,
        "direct_impact": direct_impact,
        "transitive_impact": transitive_impact,
        "mitigations_and_fallbacks": mitigations,
        "single_points_of_failure": spofs,
        "historical_incidents": historical_incidents,
        "unverified_gaps": unverified_gaps,
        "prose_evidence": evidence_list,
    }


def query_ontology_audit(
    conn: sqlite3.Connection,
    scope: str = "all",
    subject_id: Optional[str] = None,
    at_time: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute `ontology_audit(scope='all'|'resolve'|'contradictions'|'unverified'|'link_health')`."""
    if not has_ontology_tables(conn):
        return {"error": "Ontology projection tables not yet compiled in this database."}

    out: Dict[str, Any] = {"scope": scope}

    # 1. Temporal & Supersession Resolution (`scope in ('all', 'resolve')`)
    if scope in ("all", "resolve"):
        resolved_subj = resolve_entity_id(conn, subject_id) if subject_id else None
        params: List[Any] = []
        where_clauses = ["1=1"]
        if resolved_subj:
            where_clauses.append("(c.subject_id = ? OR c.evidence_span = ?)")
            params.extend([resolved_subj, resolved_subj])
        if at_time:
            where_clauses.append("(c.valid_from IS NULL OR c.valid_from <= ?)")
            where_clauses.append("(c.valid_to IS NULL OR c.valid_to > ?)")
            params.extend([at_time, at_time])

        sql = (
            "SELECT c.claim_id, c.subject_id, c.predicate, c.object_id_or_literal, "
            "c.source_kind, c.source_path, c.provenance_origin, c.epistemic_status, "
            "c.governance_status, c.verification_status, c.asserted_at, c.valid_from, "
            "c.valid_to, c.supersedes_claim_id "
            f"FROM ontology_claims c WHERE {' AND '.join(where_clauses)} "
            "AND c.source_kind IN ('policy_doc', 'decision_log') "
            "ORDER BY COALESCE(c.asserted_at, c.valid_from, '') DESC"
        )
        claims = [dict(r) for r in conn.execute(sql, params).fetchall()]
        out["resolved_claims"] = claims
        out["superseded_lineage"] = [c for c in claims if c.get("supersedes_claim_id") or c.get("governance_status") == "superseded"]

    # 2. Structural Contradictions & Attribution Violations (`scope in ('all', 'contradictions')`)
    if scope in ("all", "contradictions"):
        # Formal tuple collision: two active claims on the same
        # (subject_id, predicate) with distinct object_id_or_literal and no supersedes edge
        collision_rows = conn.execute(
            """
            SELECT c1.claim_id AS claim_a, c1.subject_id, c1.predicate,
                   c1.object_id_or_literal AS object_a, c1.source_path AS source_a,
                   c1.provenance_origin AS origin_a, c1.asserted_at AS date_a,
                   c2.claim_id AS claim_b, c2.object_id_or_literal AS object_b,
                   c2.source_path AS source_b, c2.provenance_origin AS origin_b,
                   c2.asserted_at AS date_b
            FROM ontology_claims c1
            JOIN ontology_claims c2
              ON c1.subject_id = c2.subject_id
             AND c1.predicate = c2.predicate
             AND c1.claim_id < c2.claim_id
            WHERE c1.governance_status = 'active'
              AND c2.governance_status = 'active'
              AND c1.source_kind IN ('policy_doc', 'decision_log')
              AND c2.source_kind IN ('policy_doc', 'decision_log')
              AND c1.object_id_or_literal != c2.object_id_or_literal
              AND COALESCE(c1.supersedes_claim_id, '') != c2.claim_id
              AND COALESCE(c2.supersedes_claim_id, '') != c1.claim_id
            """
        ).fetchall()

        contradictions = []
        for r in collision_rows:
            d = dict(r)
            d["attribution_violation"] = (
                (d["origin_a"] == "human" and d["origin_b"] == "agent_inferred")
                or (d["origin_b"] == "human" and d["origin_a"] == "agent_inferred")
            )
            contradictions.append(d)
        out["contradictions"] = contradictions

    # 3. Paper Rules vs Enforced Policies (`scope in ('all', 'unverified')`)
    if scope in ("all", "unverified"):
        policy_nodes = conn.execute(
            "SELECT node_id, canonical_name, source_path, attributes_json "
            "FROM ontology_nodes WHERE node_kind = 'Policy' AND status = 'active' ORDER BY node_id"
        ).fetchall()
        enforced_policies = []
        paper_rules = []
        for nid, cname, sp, attr_json in policy_nodes:
            attrs = json.loads(attr_json or "{}")
            gates = [
                r[0]
                for r in conn.execute(
                    "SELECT object_id_or_literal FROM ontology_claims "
                    "WHERE subject_id = ? AND predicate = 'ENFORCED_BY' AND governance_status = 'active'",
                    (nid,),
                ).fetchall()
            ]
            obs = [
                r[0]
                for r in conn.execute(
                    "SELECT object_id_or_literal FROM ontology_claims "
                    "WHERE subject_id = ? AND predicate = 'OBSERVED_IN' AND governance_status = 'active'",
                    (nid,),
                ).fetchall()
            ]
            record = {
                "policy_id": nid,
                "canonical_name": cname,
                "source_path": sp,
                "scope": attrs.get("scope"),
                "enforced_by": gates,
                "observed_in": obs,
                "verification_status": "enforced" if gates else ("observed" if obs else "unverified"),
            }
            if gates or obs:
                enforced_policies.append(record)
            else:
                paper_rules.append(record)

        out["enforced_policies"] = enforced_policies
        out["paper_rules"] = paper_rules
        out["paper_rule_count"] = len(paper_rules)
        out["enforced_policy_count"] = len(enforced_policies)

    # 4. Link & Coordinate Health (`scope in ('all', 'link_health')`)
    if scope in ("all", "link_health"):
        nl_stats: Dict[str, int] = {"resolved": 0, "ambiguous": 0, "unresolved": 0, "dangling_fk": 0}
        if has_nugget_link_coord_columns(conn):
            for status_val, cnt in conn.execute(
                "SELECT COALESCE(resolution_status, 'unresolved'), COUNT(*) FROM nugget_links GROUP BY COALESCE(resolution_status, 'unresolved')"
            ).fetchall():
                nl_stats[status_val] = cnt
        nl_stats["dangling_fk"] = conn.execute(
            "SELECT COUNT(*) FROM nugget_links nl LEFT JOIN chunks c ON c.id = nl.to_chunk_id "
            "WHERE nl.to_chunk_id IS NOT NULL AND c.id IS NULL"
        ).fetchone()[0]
        out["nugget_links_health"] = nl_stats

    return out


# ---------------------------------------------------------------------------
# Ambient Context Enrichment for `brief()` and `recall()`
# ---------------------------------------------------------------------------


def get_ambient_project_context(conn: sqlite3.Connection, project_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """Tolerant-read helper returning compiled `ontology_context` for `brief(project=...)`."""
    if not project_name or not has_ontology_tables(conn):
        return None
    pid = resolve_entity_id(conn, f"project/{slugify_id(project_name)}") or resolve_entity_id(conn, project_name)
    if not pid:
        return None
    impact = query_ontology_impact(conn, pid, mode="inspect", include_evidence=False)
    if "error" in impact:
        return None
    audit = query_ontology_audit(conn, scope="unverified")
    in_scope = {pid, *(get_ontology_setting("global_scopes", []) or [])}
    project_paper_rules = [r for r in audit.get("paper_rules", []) if r.get("scope") in in_scope]
    return {
        "node_id": pid,
        "canonical_name": impact["entity"]["canonical_name"],
        "outbound_claims": impact["outbound_claims"],
        "inbound_claims": impact["inbound_claims"],
        "paper_rules_in_scope": project_paper_rules,
    }


def get_ambient_query_entities(conn: sqlite3.Connection, query: str) -> List[Dict[str, Any]]:
    """Tolerant-read helper matching canonical ontology entities inside a `recall(query)` string."""
    if not query or not has_ontology_tables(conn):
        return []
    polysemy = get_polysemy_blocklist()
    q_low = query.lower()
    matched_ids: List[str] = []
    for nid, cname, nkind, aliases_json in conn.execute(
        "SELECT node_id, canonical_name, node_kind, aliases_json FROM ontology_nodes WHERE status = 'active'"
    ).fetchall():
        candidates = [nid.lower(), cname.lower()]
        try:
            candidates.extend(a.lower() for a in json.loads(aliases_json or "[]") if a.lower() not in polysemy)
        except json.JSONDecodeError:
            pass
        if any(len(c) >= 3 and re.search(r"\b" + re.escape(c) + r"\b", q_low) for c in candidates):
            matched_ids.append(nid)
        if len(matched_ids) >= 3:
            break

    results: List[Dict[str, Any]] = []
    for nid in matched_ids:
        claims = [
            {"predicate": r[0], "object": r[1], "verification_status": r[2]}
            for r in conn.execute(
                "SELECT predicate, object_id_or_literal, verification_status "
                "FROM ontology_claims WHERE subject_id = ? AND governance_status = 'active' LIMIT 8",
                (nid,),
            ).fetchall()
        ]
        results.append({"node_id": nid, "active_claims": claims})
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Anamnesis ontology compiler and link-repair CLI")
    parser.add_argument("--db", default=str(default_db_path()))
    parser.add_argument("--repair-links", action="store_true", help="Run chunk coordinate backfill & nugget_links FK repair")
    parser.add_argument("--compile", action="store_true", help="Compile the configured registries into ontology_* tables")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    ensure_ontology_schema(conn)
    if args.repair_links:
        updated_chunks = backfill_chunk_coordinates(conn, only_missing=False)
        link_stats = repair_nugget_links(conn)
        print(f"Link repair complete: {updated_chunks} chunks keyed, nugget_links={link_stats}")
    if args.compile:
        comp_stats = compile_ontology(conn)
        print(f"Compile complete: {comp_stats}")
    conn.close()
