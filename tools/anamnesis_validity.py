#!/usr/bin/env python3
"""Anamnesis link validity — bi-temporal validity (shadow mode first).

Single source of truth for the bi-temporal supersession schema columns, the
supersession-key derivation, the shadow-file contract and the live-flag gate.
Imported by the MCP server (ensure schema, tolerant-read on graph walks) and by
dream.py (tolerant-read on link queries). It also ships a small CLI:

  python tools/anamnesis_validity.py shadow    # compute verdicts, record them in the shadow file
  python tools/anamnesis_validity.py status    # show shadow window, false supersessions, live flag
  python tools/anamnesis_validity.py gate      # promote to LIVE only if the shadow window passed
  python tools/anamnesis_validity.py apply     # close superseded links (never touches the live flag)

HARD CONSTRAINT: SHADOW MODE FIRST. Computing verdicts never mutates live
`nugget_links` validity; verdicts go to a shadow file. The live flag is the only
thing that changes retrieval behaviour, and the gate (`gate_decision`) is the only
code that flips it: mechanical, deterministic and reboot-persistent (see
is_validity_live()).

Design notes:
  - Supersession keys = file-slug + frontmatter identity ONLY, never chunk titles
    across files (titles drift; false collisions silently hide valid memories).
    Within a single source_path, the title is the frontmatter identity (the
    heading), so an exact (source_path, title) match is the conservative key.
  - Deterministic, not embedding-based: cheap insurance, auditable.
  - False supersession is the main risk (quiet recall suppression), hence shadow
    mode + a promotion gate, never live closure by hand-waving.

Environment variables (all optional):
  ANAMNESIS_HOME            state dir (default ~/.anamnesis)
  ANAMNESIS_DB              database (default $ANAMNESIS_HOME/memory.db)
  ANAMNESIS_VALIDITY_SHADOW       shadow file (default $ANAMNESIS_HOME/validity-shadow.json)
  ANAMNESIS_VALIDITY_LIVE_FILE    live flag file (default $ANAMNESIS_HOME/.validity-live)
  ANAMNESIS_VALIDITY_LIVE         "1"/"true"/"yes" forces LIVE for this process
  ANAMNESIS_VALIDITY_GATE_DAYS    shadow window required before promotion (default 30)
  ANAMNESIS_VALIDITY_MIN_COVERAGE minimum key coverage to promote (default 0.5)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anamnesis_dict import anamnesis_home, db_path  # noqa: E402

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

VALIDITY_COLUMNS = ("valid_from", "valid_to", "recorded_at", "superseded_by")

VALIDITY_COLUMN_DDL = {
    "valid_from": "TEXT",
    "valid_to": "TEXT",
    "recorded_at": "TEXT",
    "superseded_by": "INTEGER",
}

# ---------------------------------------------------------------------------
# Paths / flags
# ---------------------------------------------------------------------------


def _home() -> Path:
    return anamnesis_home()


def shadow_path() -> Path:
    """Where shadow verdicts are logged. No live data is touched."""
    return Path(os.environ.get("ANAMNESIS_VALIDITY_SHADOW", str(_home() / "validity-shadow.json")))


def validity_live_flag_path() -> Path:
    """Reboot-persistent live flag file. Presence (non-empty) == LIVE."""
    return Path(os.environ.get("ANAMNESIS_VALIDITY_LIVE_FILE", str(_home() / ".validity-live")))


def is_validity_live() -> bool:
    """Is the bi-temporal closure LIVE (i.e. should retrieval suppress stale links)?

    True iff the env override is set OR the reboot-persistent flag file exists and
    is non-empty. Default is False = SHADOW (no live closure). This is the single
    gate that changes retrieval behaviour, and it is mechanical + persistent.
    """
    if os.environ.get("ANAMNESIS_VALIDITY_LIVE", "").strip() in ("1", "true", "yes"):
        return True
    p = validity_live_flag_path()
    try:
        return p.exists() and p.stat().st_size > 0
    except OSError:
        return False


def set_validity_live(live: bool) -> None:
    """Flip the reboot-persistent flag. Only the gate (and tests) should call this."""
    p = validity_live_flag_path()
    if live:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("1\n")
    else:
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def memory_db_path() -> Path:
    return db_path()


# ---------------------------------------------------------------------------
# Schema (additive, idempotent)
# ---------------------------------------------------------------------------


def has_validity_columns(conn: sqlite3.Connection) -> bool:
    """Tolerant probe: do the bi-temporal columns exist on nugget_links yet?"""
    try:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(nugget_links)").fetchall()}
    except sqlite3.Error:
        return False
    return all(c in existing for c in VALIDITY_COLUMNS)


def ensure_validity_schema(conn: sqlite3.Connection) -> None:
    """Add the four bi-temporal columns to nugget_links if missing.

    Additive + idempotent (ALTER TABLE ADD COLUMN is safe on SQLite; it does not
    rewrite existing rows and existing readers using SELECT nl.* / specific cols
    are unaffected). This is deliverable 1 — the schema — and it does NOT apply
    any verdict; it only makes room to record them.
    """
    try:
        conn.execute("SELECT 1 FROM nugget_links LIMIT 0")
    except sqlite3.Error:
        # Table doesn't exist yet (fresh DB) — nothing to migrate; the indexer
        # creates it. Nothing to do here.
        return
    existing = {row[1] for row in conn.execute("PRAGMA table_info(nugget_links)").fetchall()}
    changed = False
    for col, typ in VALIDITY_COLUMN_DDL.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE nugget_links ADD COLUMN {col} {typ}")
            changed = True
    if changed:
        conn.commit()


# ---------------------------------------------------------------------------
# Supersession keys (file-slug + frontmatter ONLY)
# ---------------------------------------------------------------------------

# Stores whose titles are GENERATED per-note (not stable frontmatter identities)
# are excluded from supersession keying. session_extracts titles are
# "<type>: <short field>" (e.g. "Tool pattern: Bash (useful)") — the same title
# maps to genuinely DIFFERENT notes (distinct content hashes), so keying on the
# title manufactures false supersessions. These notes are immutable point-in-time
# observations, not updateable memories, so bi-temporal validity does not apply.
SUPERSEDE_EXCLUDED_STORES = frozenset({"session_extracts"})


def _slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")


def frontmatter_identity(row: sqlite3.Row | dict) -> str:
    """Stable frontmatter identity for a chunk, or '' when unresolvable.

    Prefers an explicit stable id in the metadata JSON (id/slug/key/decision_id),
    then falls back to the normalized title. We do NOT use raw prose content as a
    key (never chunk prose). An empty string means "cannot key" — such a
    chunk is excluded from supersession (conservative: better to miss a
    supersession than to close a valid memory).

    Noise titles ("unknown", "untitled", bare dates) are treated as unresolvable:
    a generic identity would group unrelated chunks under one key and manufacture
    false supersessions — the exact failure the shadow window is meant to catch.
    """
    d = dict(row) if not isinstance(row, dict) else row
    if (d.get("store_type") or "") in SUPERSEDE_EXCLUDED_STORES:
        return ""
    meta = d.get("metadata")
    if isinstance(meta, str) and meta.strip():
        try:
            obj = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            obj = None
        if isinstance(obj, dict):
            for k in ("id", "slug", "key", "decision_id", "chunk_key"):
                if obj.get(k):
                    return str(obj[k]).strip()
    title = (d.get("title") or "").strip()
    if not title:
        return ""
    low = title.lower()
    if low in ("unknown", "untitled", "n/a", "na", "none", "(untitled)"):
        return ""
    if len(title) < 8:
        return ""
    if re.fullmatch(r"[\d\-–—/.:\s]+", title):
        return ""  # bare date/timestamp — not a stable identity
    return title


def supersession_key(row: sqlite3.Row | dict) -> str | None:
    """Return the deterministic supersession key, or None if not keyable.

    key = "<file-slug>|<frontmatter-identity>". Both halves must be non-empty.
    """
    d = dict(row) if not isinstance(row, dict) else row
    slug = _slugify(str(d.get("source_path") or ""))
    fm = frontmatter_identity(d)
    if not slug or not fm:
        return None
    return f"{slug}|{fm}"


# ---------------------------------------------------------------------------
# Shadow file contract
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_shadow() -> dict:
    """Load the shadow file, or return a fresh empty contract."""
    p = shadow_path()
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "started_at": None,
        "verdicts": [],
        "false_supersessions": 0,
        "key_coverage": None,
        "total_links": None,
    }


def write_shadow(data: dict) -> None:
    p = shadow_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


# ---------------------------------------------------------------------------
# Supersession detection (READ-ONLY — never mutates live data)
# ---------------------------------------------------------------------------


_DATE_LINE_RE = re.compile(
    r"(?:\*\*Date\*\*|\bDate|\basserted_at|\bvalid_from)\s*:\s*\**\s*(\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}(?::\d{2})?)?)",
    re.IGNORECASE,
)


def _chronological_stamp(d: dict) -> str:
    """Return the most specific chronological timestamp for a chunk.

    Prefers explicit date/asserted_at/valid_from in metadata JSON, then an
    explicit Date: YYYY-MM-DD line in content, and finally falls back to
    indexed_at.
    """
    meta = d.get("metadata")
    if isinstance(meta, str) and meta.strip():
        try:
            obj = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            obj = None
        if isinstance(obj, dict):
            for k in ("asserted_at", "valid_from", "date", "created_at"):
                if obj.get(k):
                    return str(obj[k]).strip()
    content = d.get("content") or ""
    m = _DATE_LINE_RE.search(content)
    if m:
        return m.group(1).strip()
    return (d.get("indexed_at") or "").strip()


def compute_supersessions(conn: sqlite3.Connection) -> dict:
    """Compute supersession verdicts + key coverage WITHOUT writing anything.

    Returns a dict with the verdict list and the key-coverage metric. This is the
    whole shadow-mode computation: deterministic, read-only, conservative.

    Rule: within a single (file-slug, frontmatter) key, an older chunk with the
    SAME key, DIFFERENT content, and a STRICTLY EARLIER chronological stamp is
    superseded by a newer chunk. Co-existing sibling chunks from the same file
    snapshot that share an identical timestamp (e.g. long sections split across
    multiple chunks or repeated template headings) are never treated as
    supersessions.
    """
    # Only keyable chunks participate.
    rows = conn.execute(
        "SELECT id, chunk_id, store_type, source_path, title, content, indexed_at, "
        "retirement_status, metadata FROM chunks"
    ).fetchall()

    keyed: dict[str, list[dict]] = {}
    keyable = 0
    for r in rows:
        d = dict(r)
        k = supersession_key(d)
        if k is None:
            continue
        keyable += 1
        d["_chrono_stamp"] = _chronological_stamp(d)
        keyed.setdefault(k, []).append(d)

    verdicts: list[dict] = []
    sibling_collisions_skipped = 0
    for k, members in keyed.items():
        if len(members) < 2:
            continue
        members.sort(key=lambda m: (m.get("_chrono_stamp") or "", m.get("id") or 0))
        # Consecutive same-key pairs, older -> newer, different content.
        for older, newer in zip(members, members[1:]):
            if (older.get("content") or "") == (newer.get("content") or ""):
                continue  # identical content: re-index artifact, not a supersession
            older_stamp = older.get("_chrono_stamp") or ""
            newer_stamp = newer.get("_chrono_stamp") or ""
            if older_stamp == newer_stamp:
                # Sibling chunks from the same file snapshot (split sections or
                # repeated subheadings) share an identical timestamp and are NOT
                # temporal supersessions.
                sibling_collisions_skipped += 1
                continue
            # Conservative: skip if the "newer" is retired and the "older" is active
            # (superseding an active memory with a retired one is a false supersession).
            false = (
                (newer.get("retirement_status") or "active") != "active"
                and (older.get("retirement_status") or "active") == "active"
            )
            verdicts.append({
                "key": k,
                "superseded_chunk_id": older["id"],
                "superseded_chunk_key": older.get("chunk_id"),
                "superseding_chunk_id": newer["id"],
                "superseding_chunk_key": newer.get("chunk_id"),
                "superseded_indexed_at": older.get("indexed_at"),
                "superseding_indexed_at": newer.get("indexed_at"),
                "superseded_chrono_stamp": older_stamp,
                "superseding_chrono_stamp": newer_stamp,
                "false_supersession": false,
            })

    total_links = conn.execute("SELECT COUNT(*) FROM nugget_links").fetchone()[0]

    total_chunks = len(rows)
    coverage = (keyable / total_chunks) if total_chunks else 0.0

    return {
        "verdicts": verdicts,
        "false_supersession_count": sum(1 for v in verdicts if v["false_supersession"]),
        "sibling_collisions_skipped": sibling_collisions_skipped,
        "key_coverage": round(coverage, 4),
        "total_links": total_links,
        "total_chunks": total_chunks,
        "keyable_chunks": keyable,
    }


def apply_supersessions(conn: sqlite3.Connection, result: dict | None = None) -> dict:
    """Apply verified supersession verdicts to live `nugget_links`.

    Closes any `nugget_links` row touching a superseded chunk (`from_chunk_id` or
    `to_chunk_id`) by setting `valid_to`, `superseded_by`, and `recorded_at` when
    not already closed. Refuses to mutate if any verdict is flagged as a false
    supersession.
    """
    ensure_validity_schema(conn)
    if result is None:
        result = compute_supersessions(conn)
    if result.get("false_supersession_count", 0) > 0:
        return {
            **result,
            "applied_verdicts": 0,
            "closed_links": 0,
            "skipped_reason": "false_supersessions_detected",
        }

    now = _now_iso()
    applied_verdicts = 0
    closed_links = 0
    touching_links = 0  # links (open OR already closed) that touch a superseded chunk
    for v in result.get("verdicts", []):
        if v.get("false_supersession"):
            continue
        old_id = v.get("superseded_chunk_id")
        new_key = v.get("superseding_chunk_key") or str(v.get("superseding_chunk_id") or "")
        valid_to = v.get("superseding_chrono_stamp") or v.get("superseding_indexed_at") or now
        if not old_id:
            continue
        touching_links += conn.execute(
            "SELECT COUNT(*) FROM nugget_links WHERE from_chunk_id = ? OR to_chunk_id = ?",
            (old_id, old_id),
        ).fetchone()[0]
        cur = conn.execute(
            """
            UPDATE nugget_links
               SET valid_to = ?,
                   superseded_by = ?,
                   recorded_at = COALESCE(NULLIF(recorded_at, ''), ?)
             WHERE (from_chunk_id = ? OR to_chunk_id = ?)
               AND (valid_to IS NULL OR valid_to = '')
            """,
            (valid_to, new_key, now, old_id, old_id),
        )
        closed_links += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        applied_verdicts += 1

    conn.commit()
    return {
        **result,
        "applied_verdicts": applied_verdicts,
        "closed_links": closed_links,
        "touching_links": touching_links,
    }


# ---------------------------------------------------------------------------
# Tolerant-read helper
# ---------------------------------------------------------------------------


def link_validity_clause(conn: sqlite3.Connection, alias: str = "nl") -> str:
    """Return a SQL WHERE fragment that suppresses superseded links — or '' in shadow.

    Shadow mode (default): return '' — NO live closure. Only when is_validity_live() is
    True AND the columns exist do we return a clause excluding superseded rows.
    Tolerant: if the columns don't exist yet, returns '' regardless of flag.
    """
    if not is_validity_live():
        return ""
    if not has_validity_columns(conn):
        return ""
    return f" AND ({alias}.valid_to IS NULL OR {alias}.valid_to = '')"



# ---------------------------------------------------------------------------
# Promotion gate + CLI
# ---------------------------------------------------------------------------

DEFAULT_GATE_DAYS = 30
DEFAULT_MIN_COVERAGE = 0.5


def gate_decision(data: dict | None = None, now: datetime | None = None) -> dict:
    """Mechanical promotion verdict from a shadow record. Pure: writes nothing.

    PASS needs ALL of: a recorded shadow start at least ANAMNESIS_VALIDITY_GATE_DAYS
    old, zero false supersessions, and key coverage >= ANAMNESIS_VALIDITY_MIN_COVERAGE.
    """
    data = load_shadow() if data is None else data
    now = now or datetime.now(timezone.utc)
    days_required = int(os.environ.get("ANAMNESIS_VALIDITY_GATE_DAYS", DEFAULT_GATE_DAYS))
    min_cov = float(os.environ.get("ANAMNESIS_VALIDITY_MIN_COVERAGE", DEFAULT_MIN_COVERAGE))
    reasons: list[str] = []

    started = None
    raw_start = data.get("started_at")
    if raw_start:
        try:
            started = datetime.fromisoformat(str(raw_start).replace("Z", "+00:00"))
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
        except ValueError:
            started = None
    age_days = (now - started).total_seconds() / 86400 if started else None
    if age_days is None:
        reasons.append("shadow window has not started (run `shadow` first)")
    elif age_days < days_required:
        reasons.append(f"shadow window is {age_days:.1f}d old, need {days_required}d")
    false_n = int(data.get("false_supersessions") or 0)
    if false_n:
        reasons.append(f"{false_n} false supersession(s) recorded")
    coverage = data.get("key_coverage")
    if coverage is None or float(coverage) < min_cov:
        reasons.append(f"key coverage {coverage!r} below {min_cov}")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "age_days": None if age_days is None else round(age_days, 2),
        "days_required": days_required,
        "false_supersessions": false_n,
        "key_coverage": coverage,
    }


def _db_identity(path: Path | None) -> str:
    return str(Path(path or memory_db_path()).expanduser().resolve())


def record_shadow_run(conn: sqlite3.Connection, db_path: Path | None = None) -> dict:
    """Compute verdicts (read-only) and persist them to the shadow file.

    The shadow window starts at the first recorded run and is preserved across
    later runs; counters reflect the latest run. The database the run measured is
    recorded so `apply` cannot act on a different one.
    """
    result = compute_supersessions(conn)
    shadow = load_shadow()
    shadow["db"] = _db_identity(db_path)
    shadow["started_at"] = shadow.get("started_at") or _now_iso()
    shadow["updated_at"] = _now_iso()
    shadow["verdicts"] = result["verdicts"]
    shadow["false_supersessions"] = result["false_supersession_count"]
    shadow["key_coverage"] = result["key_coverage"]
    shadow["total_links"] = result["total_links"]
    write_shadow(shadow)
    return shadow


def _open_db(path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path or memory_db_path()))
    conn.row_factory = sqlite3.Row
    return conn


def _cmd_shadow(args) -> int:
    conn = _open_db(args.db)
    try:
        shadow = record_shadow_run(conn, args.db)
    finally:
        conn.close()
    print(f"shadow: {len(shadow['verdicts'])} verdict(s), "
          f"{shadow['false_supersessions']} false, coverage {shadow['key_coverage']}")
    return 0


def _cmd_status(args) -> int:
    verdict = gate_decision()
    print(json.dumps({"live": is_validity_live(), "gate": verdict}, indent=2))
    return 0


def _cmd_gate(args) -> int:
    verdict = gate_decision()
    if verdict["pass"]:
        set_validity_live(True)
        print("gate: PASS — validity is now LIVE (retrieval suppresses superseded links)")
        return 0
    set_validity_live(False)
    print("gate: FAIL — staying in shadow mode:\n  - " + "\n  - ".join(verdict["reasons"]))
    return 1


def _cmd_apply(args) -> int:
    """Close superseded links. Gated on the same verdict; never sets the live flag."""
    verdict = gate_decision()
    if not verdict["pass"] and not getattr(args, "force", False):
        print("apply refused (gate not satisfied; use --force to override):\n  - "
              + "\n  - ".join(verdict["reasons"]), file=sys.stderr)
        return 2
    force = getattr(args, "force", False)
    target = getattr(args, "db", None)
    if not force:
        recorded = load_shadow().get("db")
        if recorded != _db_identity(target):
            print(f"apply refused: the shadow run measured {recorded!r}, not "
                  f"{_db_identity(target)!r} (re-run `shadow` against it, or --force)", file=sys.stderr)
            return 2
    conn = _open_db(target)
    try:
        if not force:
            min_cov = float(os.environ.get("ANAMNESIS_VALIDITY_MIN_COVERAGE", DEFAULT_MIN_COVERAGE))
            live = compute_supersessions(conn)
            if live["key_coverage"] is None or float(live["key_coverage"]) < min_cov:
                print(f"apply refused: target key coverage {live['key_coverage']!r} below {min_cov}",
                      file=sys.stderr)
                return 2
        res = apply_supersessions(conn)
    finally:
        conn.close()
    if res.get("skipped_reason"):
        print(f"apply skipped: {res['skipped_reason']}", file=sys.stderr)
        return 2
    print(f"apply: {res['applied_verdicts']} verdict(s) applied, "
          f"{res['closed_links']} nugget_links closed")
    if res["applied_verdicts"] and res["closed_links"] == 0:
        print("WARNING ZERO-CLOSED: verdicts exist but no link touches a superseded chunk")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Anamnesis bi-temporal supersession (shadow-first)")
    ap.add_argument("--db", type=Path, default=None, help="database (default: ANAMNESIS_DB)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("shadow", help="compute verdicts and record them in the shadow file")
    sub.add_parser("status", help="show gate status and live flag")
    sub.add_parser("gate", help="promote to LIVE if the shadow window passed")
    p_apply = sub.add_parser("apply", help="close superseded links (gated)")
    p_apply.add_argument("--force", action="store_true", help="skip the gate check")
    args = ap.parse_args(argv)
    return {"shadow": _cmd_shadow, "status": _cmd_status,
            "gate": _cmd_gate, "apply": _cmd_apply}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
