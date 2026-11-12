#!/usr/bin/env python3
"""anamnesis indexer: sources -> chunks -> embeddings -> memory.db (SQLite FTS5 + vectors).

Reads the files named by a sources config (see `anamnesis.sources.example.yaml`), chunks
them into semantic units, embeds every chunk and stores everything in one SQLite database
for two-stage retrieval (FTS5 keyword candidates, then embedding rerank).

    python tools/index.py build [--tags]       full rebuild from the sources config
    python tools/index.py update [--dry-run]   incremental: only changed / new / removed sources
    python tools/index.py backfill             embed chunks that still lack a vector
    python tools/index.py search "query"       two-stage search (prints ranked chunks)
    python tools/index.py synthesize "query"   search + LLM synthesis with citations
    python tools/index.py stats                index statistics
    python tools/index.py sources              show the resolved sources (no writes)
    python tools/index.py import-chats PATH    import ChatGPT / Claude / Gemini exports
    python tools/index.py                      same as `update` (builds when no DB exists)

Configuration is environment variables plus one optional sources file; nothing is hardcoded
to a machine, a user or a host. Secrets are only ever read from the environment.

    ANAMNESIS_HOME              state directory (spend counters, query cache)   ~/.anamnesis
    ANAMNESIS_DATA_DIR          git-tracked memory tree                         <repo>/data
    ANAMNESIS_DB                database path (alias: ANAMNESIS_OUT)            $ANAMNESIS_HOME/memory.db
    ANAMNESIS_SOURCES           sources config, YAML or JSON                    built-in: data/{memory,decisions,manifests}
    ANAMNESIS_EMBED_BACKEND     gemini | local | mock                           gemini
    GEMINI_API_KEY              required for backend=gemini (and tags/synthesis)
    GEMINI_API_KEY_BACKUP       optional second key tried once after rate-limit exhaustion

The database is a derived artifact: git holds the sources, this script rebuilds the index
anywhere. Vector spaces of different embedding models are not interchangeable, so the model
name is recorded in `index_meta` and indexing into a database built with another model is refused.
"""
from __future__ import annotations

import argparse
import contextlib
import fnmatch
import hashlib
import json
import math
import os
import random
import re
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:  # POSIX only; the spend-state lock degrades to a no-op elsewhere
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anamnesis_lock import connect_with_lock_warning  # noqa: E402
import anamnesis_ontology as ontology  # noqa: E402

try:
    from anamnesis_dict import get_store_weights  # noqa: E402
except Exception:  # pragma: no cover - dictionary module is optional
    get_store_weights = None  # type: ignore[assignment]

# --------------------------------------------------------------------------------------
# Configuration (environment only)
# --------------------------------------------------------------------------------------
HOME = Path.home()
REPO_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, *fallbacks: str) -> str | None:
    for n in (name, *fallbacks):
        v = os.environ.get(n)
        if v:
            return v
    return None


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


ANAMNESIS_HOME = Path(_env("ANAMNESIS_HOME") or HOME / ".anamnesis").expanduser()
MEMORY_DIR = ANAMNESIS_HOME  # alias kept for callers that import it
ANAMNESIS_STATE_DIR = ANAMNESIS_HOME
DATA_DIR = Path(_env("ANAMNESIS_DATA_DIR", "ANAMNESIS_DATA_ROOT") or REPO_ROOT / "data").expanduser()
MEMORY_DB = Path(_env("ANAMNESIS_DB", "ANAMNESIS_OUT") or ANAMNESIS_HOME / "memory.db").expanduser()
SOURCES_FILE = _env("ANAMNESIS_SOURCES")
DEFAULT_PROJECT = _env("ANAMNESIS_DEFAULT_PROJECT", "ANAMNESIS_PROJECT") or "default"

# Embedding backends. Vector spaces are NOT interchangeable: switching backend or model
# requires re-embedding the whole index (enforced by _assert_index_model).
EMBED_BACKEND = "mock" if _truthy(os.environ.get("ANAMNESIS_MOCK_EMBEDDINGS")) else \
    (os.environ.get("ANAMNESIS_EMBED_BACKEND") or "gemini").strip().lower()
_DIM_ENV = os.environ.get("ANAMNESIS_EMBEDDING_DIM")
if EMBED_BACKEND == "gemini":
    EMBEDDING_MODEL = os.environ.get("ANAMNESIS_EMBEDDING_MODEL", "gemini-embedding-001")
    EMBEDDING_DIM = int(_DIM_ENV or "3072")
elif EMBED_BACKEND == "local":
    # Optional sentence-transformers model, run on CPU: free, no data egress.
    EMBEDDING_MODEL = os.environ.get("ANAMNESIS_LOCAL_MODEL", "google/embeddinggemma-300m")
    EMBEDDING_DIM = int(_DIM_ENV or "768")
elif EMBED_BACKEND == "mock":
    # Deterministic hash-derived vectors: CI and tests, no key, no network, no semantics.
    EMBEDDING_MODEL = "mock-sha256-lcg-v1"
    EMBEDDING_DIM = int(_DIM_ENV or "3072")
else:
    raise RuntimeError(f"ANAMNESIS_EMBED_BACKEND must be 'gemini', 'local' or 'mock', got {EMBED_BACKEND!r}")

BATCH_SIZE = int(os.environ.get("ANAMNESIS_EMBED_BATCH", "20"))
GENERATION_MODEL = os.environ.get("ANAMNESIS_GENERATION_MODEL", "gemini-3.8-flash")
MAX_FILE_BYTES = int(os.environ.get("ANAMNESIS_MAX_FILE_BYTES", str(5 * 1024 * 1024)))
EMBED_RETRY_BASE_S = float(os.environ.get("ANAMNESIS_EMBED_BACKOFF_S", "5"))
EMBED_RETRY_MAX_WAIT_S = float(os.environ.get("ANAMNESIS_EMBED_RETRY_MAX_WAIT_S", "120"))

# Embedding spend guard (gemini backend only). Two independent controls:
#   1. corpus-wide reuse (_resolve_corpus_reuse): identical text is never embedded twice,
#      wherever it lives, so moving or restaging files costs nothing;
#   2. a hard monthly + per-run budget, enforced in embed_batch_budgeted.
# The accounting unit is the UTF-8 BYTE length of the text, not an estimated token count:
# no subword tokenizer emits more tokens than input bytes, so a byte count can only
# over-count and never lets a batch slip under a cap it will actually bust (len//4 can,
# for CJK, code and punctuation-heavy text). The defaults are therefore roughly 4x the
# token figures you would configure if you counted tokens.
EMBED_SPEND_STATE_PATH = Path(os.environ.get("ANAMNESIS_EMBED_SPEND_STATE",
                                             str(ANAMNESIS_HOME / "embed-spend-state.json")))
EMBED_MONTHLY_TOKEN_CAP = int(os.environ.get("ANAMNESIS_EMBED_MONTHLY_TOKEN_CAP", "160000000"))
EMBED_RUN_TOKEN_CAP = int(os.environ.get("ANAMNESIS_EMBED_RUN_TOKEN_CAP", "40000000"))
ALERT_WEBHOOK = os.environ.get("ANAMNESIS_ALERT_WEBHOOK")  # optional http(s) URL, JSON POST

DEFAULT_STORE_WEIGHTS = {
    "decision": 1.5, "feedback": 1.2, "mistake": 1.1, "memory": 1.0, "reference": 1.0,
    "contact": 1.0, "capability": 1.0, "project_doc": 1.0, "project_meta": 1.5,
    "session_log": 1.0, "decision_log": 1.5, "transcript": 0.9, "chat_export": 1.1,
    "conversation_clip": 1.0, "index": 0.9,
}


def _load_store_weights() -> dict:
    weights = dict(DEFAULT_STORE_WEIGHTS)
    if get_store_weights is not None:
        try:
            weights.update(get_store_weights())
        except Exception:  # dictionary file absent or malformed: built-in defaults stand
            pass
    return weights


# Mutated in place when a sources config adds `store_weights`, so importers see the update.
STORE_WEIGHTS = _load_store_weights()

# Rows written by side-car tools (chat import, conversation clipping), not derived from the
# sources config. A full rebuild keeps them; everything else is re-derived from files.
SIDECAR_STORE_TYPES = ("chat_export", "conversation_clip")


def now_utc() -> str:
    """UTC with a Z suffix. Never naive-local: mixed conventions in one column silently
    break any MAX()/freshness comparison across the timezone offset."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8"), usedforsecurity=False).hexdigest()  # identity key, not security


# --------------------------------------------------------------------------------------
# Embedding clients and backends
# --------------------------------------------------------------------------------------
_GEMINI_BACKUP_CLIENT = None


def get_gemini_client(which: str = "primary"):
    """Gemini client from the environment (GEMINI_API_KEY, or GEMINI_API_KEY_BACKUP for `backup`)."""
    from google import genai
    var = "GEMINI_API_KEY" if which == "primary" else "GEMINI_API_KEY_BACKUP"
    key = os.environ.get(var)
    if not key:
        raise RuntimeError(f"{var} is not set (get a key at https://aistudio.google.com/apikey), "
                           f"or use ANAMNESIS_EMBED_BACKEND=local|mock")
    return genai.Client(api_key=key)


def get_embed_client():
    """Client for the embedding layer: None for local/mock (no API), else a Gemini client."""
    return None if EMBED_BACKEND in ("local", "mock") else get_gemini_client()


def get_indexing_client():
    """Client used by indexing runs. None for local/mock; a Gemini client otherwise
    (raises when the key is missing, so a misconfigured run fails before doing any work)."""
    return get_embed_client()


def get_tag_client():
    """Gemini client for optional auto-tagging; None (and a note) when unavailable."""
    try:
        return get_gemini_client()
    except Exception as e:
        print(f"  auto-tagging skipped: {e}", flush=True)
        return None


def mock_embedding(text: str, dim: int | None = None) -> list[float]:
    """Deterministic unit vector derived from a hash of `text` (CI / offline use)."""
    dim = EMBEDDING_DIM if dim is None else dim
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    vec: list[float] = []
    val = seed
    norm_sq = 0.0
    for _ in range(dim):
        val = (val * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        f = (val / 0xFFFFFFFFFFFFFFFF) * 2.0 - 1.0
        vec.append(f)
        norm_sq += f * f
    inv = 1.0 / math.sqrt(norm_sq) if norm_sq > 0 else 1.0
    return [x * inv for x in vec]


_LOCAL_MODEL = None


def _local_model():
    """Lazy-load the sentence-transformers model in float32 on CPU (float16 can silently
    degrade vectors)."""
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        import torch
        from sentence_transformers import SentenceTransformer
        threads = os.environ.get("ANAMNESIS_EMBED_THREADS")
        if threads:
            torch.set_num_threads(int(threads))
        _LOCAL_MODEL = SentenceTransformer(EMBEDDING_MODEL, device="cpu",
                                           model_kwargs={"torch_dtype": torch.float32})
        _LOCAL_MODEL.max_seq_length = int(os.environ.get("ANAMNESIS_LOCAL_MAX_SEQ", "2048"))
    return _LOCAL_MODEL


def _local_doc_text(embed_text: str) -> str:
    """Map _embed_text()'s canonical string to the `title: ... | text: ...` document format
    used by EmbeddingGemma-style models. _embed_text() itself is unchanged: embed_hash (the
    reuse key) is computed from it."""
    header, _, content = embed_text.partition("\n")
    parts = header.split(" | ", 2)
    if len(parts) == 3 and parts[0].startswith("Project: ") and parts[1].startswith("Type: "):
        return f"title: {parts[2] or 'none'} | text: {parts[0]} | {parts[1]}\n{content}"
    return f"title: none | text: {embed_text}"


def _local_encode(texts: list[str], query: bool = False) -> list[list[float]]:
    model = _local_model()
    if query:
        prompts = getattr(model, "prompts", {}) or {}
        name = next((p for p in ("query", "SearchQuery", "Retrieval-query") if p in prompts), None)
        kwargs = {"prompt_name": name} if name else {}
        embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, **kwargs)
    else:
        embs = model.encode([_local_doc_text(t) for t in texts], batch_size=8,
                            normalize_embeddings=True, convert_to_numpy=True)
    out = [e.tolist() for e in embs]
    if any(v != v for e in out for v in e):
        raise RuntimeError("local embedder returned NaN")
    return out


_TRANSIENT_RE = re.compile(r"\b(429|500|502|503|504)\b|RESOURCE_EXHAUSTED|UNAVAILABLE|DEADLINE_EXCEEDED|"
                           r"timed? ?out|connection (reset|aborted|error)", re.IGNORECASE)
_RATE_LIMIT_RE = re.compile(r"\b429\b|RESOURCE_EXHAUSTED", re.IGNORECASE)


def _gemini_embed_call(client, texts: list[str]):
    kwargs = {"model": EMBEDDING_MODEL, "contents": texts}
    if _DIM_ENV:  # only request a reduced dimensionality when explicitly asked for
        from google.genai import types
        kwargs["config"] = types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM)
    return client.models.embed_content(**kwargs)


def embed_batch(client, texts: list[str], max_retries: int = 3,
                usage_out: dict | None = None) -> list[list[float]]:
    """Embed a batch of texts with retry and exponential backoff.

    Transient failures (rate limit, 5xx, timeouts) are retried up to `max_retries` attempts;
    a rate-limit message's own "retry in N s" hint is honoured (capped). After the last
    attempt, a rate-limited call gets ONE more try on GEMINI_API_KEY_BACKUP when set.
    usage_out, if given, receives {"tokens": <API-reported count or None>}.
    """
    global _GEMINI_BACKUP_CLIENT
    if EMBED_BACKEND == "mock":
        if usage_out is not None:
            usage_out["tokens"] = 0
        return [mock_embedding(t) for t in texts]
    if EMBED_BACKEND == "local":
        if usage_out is not None:
            usage_out["tokens"] = 0  # free: nothing to reconcile against the spend budget
        return _local_encode(texts)

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            result = _gemini_embed_call(client, texts)
            if usage_out is not None:
                usage_out["tokens"] = _extract_token_usage(result)
            return [list(e.values) for e in result.embeddings]
        except Exception as e:
            last_exc = e
            err = str(e)
            if not _TRANSIENT_RE.search(err) or attempt >= max_retries - 1:
                break
            hint = re.search(r"retry.*?(\d+\.?\d*)\s*s", err, re.IGNORECASE)
            wait = float(hint.group(1)) if hint else EMBED_RETRY_BASE_S * (2 ** attempt)
            wait = min(wait, EMBED_RETRY_MAX_WAIT_S) + random.uniform(0, 0.25)
            print(f"    Transient embed error; waiting {wait:.0f}s (attempt {attempt + 1}/{max_retries})...",
                  flush=True)
            time.sleep(wait)
    assert last_exc is not None
    if max_retries > 1 and _RATE_LIMIT_RE.search(str(last_exc)) and os.environ.get("GEMINI_API_KEY_BACKUP"):
        try:
            if _GEMINI_BACKUP_CLIENT is None:
                _GEMINI_BACKUP_CLIENT = get_gemini_client("backup")
            print("    Primary key rate-limited; trying the backup key once.", flush=True)
            result = _gemini_embed_call(_GEMINI_BACKUP_CLIENT, texts)
            if usage_out is not None:
                usage_out["tokens"] = _extract_token_usage(result)
            return [list(e.values) for e in result.embeddings]
        except Exception as e:
            last_exc = e
    raise last_exc


class EmbedBudgetExceeded(Exception):
    """Raised by embed_batch_budgeted BEFORE an API call that would push this run or this
    calendar month over its embedding cap. Callers degrade to NULL embeddings (searchable via
    FTS immediately); backfill_embeddings() fills them in later, budget permitting."""


def _estimate_tokens(texts: list[str]) -> int:
    """Pre-flight upper bound used to gate spend BEFORE the call (UTF-8 byte length)."""
    if not texts:
        return 0
    return max(1, sum(len(t.encode("utf-8")) for t in texts))


def _extract_token_usage(result) -> int | None:
    """Best-effort real token count from an embed response; None means "use the estimate"."""
    for attr in ("usage_metadata", "usageMetadata"):
        meta = getattr(result, attr, None)
        if meta is None:
            continue
        for field in ("total_token_count", "totalTokenCount", "prompt_token_count", "promptTokenCount"):
            val = getattr(meta, field, None)
            if isinstance(val, int):
                return val
    return None


def _current_month(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m")


def _embed_spend_state_load() -> dict:
    try:
        return json.loads(EMBED_SPEND_STATE_PATH.read_text())
    except Exception:
        return {}


def _embed_spend_state_save(state: dict) -> None:
    """Atomic write (tmp + os.replace): this file is read-modify-written on every batch."""
    EMBED_SPEND_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(EMBED_SPEND_STATE_PATH.parent), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, EMBED_SPEND_STATE_PATH)


def _embed_spend_state_for_month(state: dict, month: str) -> dict:
    """Roll `state` over to `month`: usage resets, per-month alert history is kept."""
    if state.get("month") != month:
        return {"month": month, "tokens_used": 0, "alerts_sent": state.get("alerts_sent", {})}
    return state


def _send_embed_cap_alert(kind: str, month: str, tokens: int, cap: int) -> None:
    """One alert per (month, kind): stderr always, plus ANAMNESIS_ALERT_WEBHOOK if configured.
    The caller owns the dedupe check; this function always sends."""
    pct = 100 * tokens / cap if cap else 0
    if kind == "cap":
        title = "Anamnesis embed spend: MONTHLY CAP HIT"
        body = (f"{month}: {tokens:,}/{cap:,} embedding units ({pct:.0f}%). Embedding stopped for the "
                f"rest of this run; remaining chunks are left without vectors until back under budget.")
    elif kind == "run-cap":
        title = "Anamnesis embed spend: RUN CAP HIT"
        body = (f"{month}: a single run would use {tokens:,} units, over the per-run cap of {cap:,}. "
                f"Embedding stopped for the rest of this run.")
    else:
        title = "Anamnesis embed spend: 80% of monthly cap"
        body = f"{month}: {tokens:,}/{cap:,} embedding units used ({pct:.0f}%)."
    print(f"  ALERT {title} - {body}", file=sys.stderr, flush=True)
    if ALERT_WEBHOOK and ALERT_WEBHOOK.lower().startswith(("http://", "https://")):
        try:
            req = urllib.request.Request(ALERT_WEBHOOK, data=json.dumps({"title": title, "text": body}).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=10).close()  # noqa: S310 - scheme checked above
        except Exception as e:
            print(f"  alert webhook failed ({type(e).__name__}: {str(e)[:120]})", file=sys.stderr, flush=True)


_run_tokens_used = 0     # reset at the start of each build/update/backfill run
_run_exhausted = False   # sticky for the rest of the top-level run
_embed_deadline = None   # time.time() cutoff for embedding in this run; None = unlimited


def _reset_run_embed_budget(budget_env: str = "ANAMNESIS_EMBED_BUDGET_S") -> None:
    """Reset per-run counters and arm the optional wall-clock budget.

    With ANAMNESIS_EMBED_BUDGET_S set, embedding loops stop once the budget is spent and the
    run still lands every source with whatever vectors it has (chunks without a vector are
    FTS-searchable at once; `backfill` fills them on later runs). Without it a long run that
    is killed by an outer timeout would land nothing.
    """
    global _run_tokens_used, _run_exhausted, _embed_deadline
    _run_tokens_used = 0
    _run_exhausted = False
    try:
        raw = os.environ.get(budget_env)
        if raw is None and budget_env != "ANAMNESIS_EMBED_BUDGET_S":
            raw = os.environ.get("ANAMNESIS_EMBED_BUDGET_S", "0")
        budget = float(raw or "0")
    except ValueError:
        budget = 0.0
    _embed_deadline = time.time() + budget if budget > 0 else None


def _embed_time_up() -> bool:
    return _embed_deadline is not None and time.time() >= _embed_deadline


def _mark_run_exhausted() -> None:
    """Keep the per-run cap tripped for the REST of the run, not just the loop that hit it
    (a later, smaller call must not sneak a batch through after the run already stopped)."""
    global _run_exhausted
    _run_exhausted = True


def _embed_spend_lock_path() -> Path:
    return EMBED_SPEND_STATE_PATH.with_name(EMBED_SPEND_STATE_PATH.name + ".lock")


@contextlib.contextmanager
def _embed_spend_locked():
    """Exclusive flock around a read-modify-write of the spend-state file, so two concurrent
    indexer processes cannot both read the same balance and both pass the cap check."""
    lock_path = _embed_spend_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as fh:
        if fcntl is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _reserve_embed_budget(estimated_tokens: int) -> None:
    """Reserve `estimated_tokens` against the per-run and monthly budgets BEFORE the API call.

    Raises EmbedBudgetExceeded and changes nothing if either cap would be exceeded. On success
    the in-process run counter and the persisted monthly counter are both incremented; the
    caller must then call exactly one of _release_embed_reservation (call failed, nothing
    spent) or _reconcile_embed_usage (call succeeded, correct to the real figure).
    """
    global _run_tokens_used
    if _run_exhausted:
        raise EmbedBudgetExceeded("per-run cap already exceeded earlier this run")

    month = _current_month()
    projected_run = _run_tokens_used + estimated_tokens
    if projected_run > EMBED_RUN_TOKEN_CAP:
        print(f"  EMBED RUN CAP: {projected_run:,} units would exceed ANAMNESIS_EMBED_RUN_TOKEN_CAP="
              f"{EMBED_RUN_TOKEN_CAP:,}; stopping embedding for the rest of this run.",
              file=sys.stderr, flush=True)
        _send_embed_cap_alert("run-cap", month, projected_run, EMBED_RUN_TOKEN_CAP)
        _mark_run_exhausted()
        raise EmbedBudgetExceeded(f"per-run cap {EMBED_RUN_TOKEN_CAP:,} exceeded")

    with _embed_spend_locked():
        state = _embed_spend_state_for_month(_embed_spend_state_load(), month)
        projected_month = state.get("tokens_used", 0) + estimated_tokens
        alerts_sent = state.setdefault("alerts_sent", {}).setdefault(month, {})

        if projected_month > EMBED_MONTHLY_TOKEN_CAP:
            print(f"  EMBED MONTHLY CAP: {projected_month:,} units would exceed "
                  f"ANAMNESIS_EMBED_MONTHLY_TOKEN_CAP={EMBED_MONTHLY_TOKEN_CAP:,} for {month}; "
                  f"stopping embedding for this run.", file=sys.stderr, flush=True)
            if not alerts_sent.get("cap"):
                _send_embed_cap_alert("cap", month, projected_month, EMBED_MONTHLY_TOKEN_CAP)
                alerts_sent["cap"] = True
                _embed_spend_state_save(state)
            _mark_run_exhausted()
            raise EmbedBudgetExceeded(f"monthly cap {EMBED_MONTHLY_TOKEN_CAP:,} exceeded for {month}")

        state["tokens_used"] = projected_month  # reserve durably before the API call
        if projected_month >= 0.8 * EMBED_MONTHLY_TOKEN_CAP and not alerts_sent.get("80pct"):
            _send_embed_cap_alert("80pct", month, projected_month, EMBED_MONTHLY_TOKEN_CAP)
            alerts_sent["80pct"] = True
        _embed_spend_state_save(state)

    _run_tokens_used = projected_run


def _release_embed_reservation(tokens: int) -> None:
    """Undo a reservation when the API call that followed it failed outright."""
    global _run_tokens_used
    _run_tokens_used = max(0, _run_tokens_used - tokens)
    with _embed_spend_locked():
        state = _embed_spend_state_for_month(_embed_spend_state_load(), _current_month())
        state["tokens_used"] = max(0, state.get("tokens_used", 0) - tokens)
        _embed_spend_state_save(state)


def _reconcile_embed_usage(estimated_tokens: int, actual_tokens: int) -> None:
    """Correct a reservation to the real cost once an embed call succeeds."""
    delta = actual_tokens - estimated_tokens
    if delta == 0:
        return
    global _run_tokens_used
    _run_tokens_used = max(0, _run_tokens_used + delta)
    with _embed_spend_locked():
        state = _embed_spend_state_for_month(_embed_spend_state_load(), _current_month())
        state["tokens_used"] = max(0, state.get("tokens_used", 0) + delta)
        _embed_spend_state_save(state)


def embed_batch_budgeted(client, texts: list[str], max_retries: int = 3) -> list[list[float]]:
    """Every INDEXING embed call goes through here: reserve against the caps, call, then
    reconcile to the real cost (or release the reservation on failure). The spend caps guard
    the paid gemini backend only; local and mock are free. embed_query() (one short, cached
    string at interactive time) deliberately bypasses this and degrades to FTS-only on failure.
    """
    if EMBED_BACKEND != "gemini":
        return embed_batch(client, texts, max_retries=max_retries)
    estimated = _estimate_tokens(texts)
    _reserve_embed_budget(estimated)
    try:
        usage_out: dict = {}
        result = embed_batch(client, texts, max_retries=max_retries, usage_out=usage_out)
    except Exception:
        _release_embed_reservation(estimated)
        raise
    actual = usage_out.get("tokens")
    if actual is not None:
        _reconcile_embed_usage(estimated, actual)
    return result


_POISON_RE = re.compile(r"\b400\b|INVALID_ARGUMENT|too long|exceed|invalid", re.IGNORECASE)


def embed_with_degrade(client, texts: list[str]) -> list[list[float] | None]:
    """Embed `texts`, degrading instead of failing: returns one vector-or-None per text.

    If a batch is rejected as invalid (e.g. one oversized text), it is bisected so only the
    offending text loses its vector. Any other failure leaves the whole batch as None (the
    chunks still land in FTS and `backfill` retries them). Budget exhaustion propagates.
    """
    if not texts:
        return []
    try:
        vecs = embed_batch_budgeted(client, texts)
    except EmbedBudgetExceeded:
        raise
    except Exception as e:
        if len(texts) > 1 and _POISON_RE.search(str(e)) and not _TRANSIENT_RE.search(str(e)):
            mid = len(texts) // 2
            return embed_with_degrade(client, texts[:mid]) + embed_with_degrade(client, texts[mid:])
        print(f"  ERROR embedding {len(texts)} text(s): {str(e)[:200]}", flush=True)
        return [None] * len(texts)
    if len(vecs) != len(texts):
        print(f"  WARN: embedder returned {len(vecs)} vectors for {len(texts)} texts; "
              f"the remainder is left unembedded", flush=True)
    return list(vecs) + [None] * (len(texts) - len(vecs))


def _embed_pending(client, batch: list[dict], state: dict) -> None:
    """Embed the chunks in `batch` that lack a vector, honouring the budgets.

    Sets c["_embedding_blob"] to packed bytes or None. `state` carries "exhausted" (stop
    embedding for the rest of the run) and "failed" (count of chunks left without a vector).
    """
    todo = [c for c in batch if c.get("_embedding_blob") is None]
    if not todo or state["exhausted"]:
        return
    if _embed_time_up():
        print("  EMBED TIME BUDGET reached; landing the rest without vectors "
              "(backfill fills them on later runs)", flush=True)
        state["exhausted"] = True
        return
    try:
        vecs = embed_with_degrade(client, [_embed_text(c) for c in todo])
    except EmbedBudgetExceeded as e:
        print(f"  EMBED BUDGET EXCEEDED; leaving remaining chunks unembedded this run ({e})", flush=True)
        state["exhausted"] = True
        return
    for c, v in zip(todo, vecs):
        if v is not None and _valid_embedding(v):
            c["_embedding_blob"] = pack_embedding(v)
        else:
            c["_embedding_blob"] = None
            state["failed"] += 1


# --------------------------------------------------------------------------------------
# Query-embedding cache
# --------------------------------------------------------------------------------------
QUERY_CACHE_PATH = Path(os.environ.get("ANAMNESIS_QUERY_CACHE", str(ANAMNESIS_HOME / "query-embeddings.db")))
_QUERY_CACHE_MEM: dict[str, list[float]] = {}
_QUERY_CACHE_MEM_MAX = 512
_QUERY_CACHE_SCHEMA_OK: set[str] = set()


def _query_cache_key(text: str) -> str:
    # Whitespace-only normalisation: case and accents change the embedding, so they must
    # change the key. The model is part of the key, so a model switch never serves stale vectors.
    norm = " ".join(text.split())
    return hashlib.sha256(f"{EMBEDDING_MODEL}\n{norm}".encode("utf-8")).hexdigest()


def _query_cache_conn() -> sqlite3.Connection:
    # WAL so readers and the writer never block each other; a short busy timeout because this
    # sits on the interactive path: a locked file means "miss", not "wait".
    QUERY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(QUERY_CACHE_PATH), timeout=0.5)
    path_key = str(QUERY_CACHE_PATH)
    if path_key not in _QUERY_CACHE_SCHEMA_OK:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""CREATE TABLE IF NOT EXISTS query_embeddings (
            key TEXT PRIMARY KEY, model TEXT NOT NULL, dim INTEGER NOT NULL,
            embedding BLOB NOT NULL, created_at TEXT NOT NULL)""")
        conn.execute("DELETE FROM query_embeddings WHERE model != ?", (EMBEDDING_MODEL,))
        conn.commit()
        _QUERY_CACHE_SCHEMA_OK.add(path_key)
    return conn


def _query_cache_get(key: str) -> list[float] | None:
    try:
        with contextlib.closing(_query_cache_conn()) as conn:
            row = conn.execute("SELECT embedding, dim FROM query_embeddings WHERE key = ? AND model = ?",
                               (key, EMBEDDING_MODEL)).fetchone()
        if not row:
            return None
        emb = unpack_embedding(row[0])
        if not emb or len(emb) != row[1]:
            return None  # torn blob or dimension mismatch: miss
        return emb
    except Exception as e:
        # The cache is an accelerator, never a dependency, but a silent miss hides contention.
        print(f"  query-cache read failed ({type(e).__name__}: {str(e)[:120]}); treating as miss",
              file=sys.stderr, flush=True)
        return None


def _query_cache_put(key: str, emb: list[float]) -> None:
    try:
        with contextlib.closing(_query_cache_conn()) as conn:
            conn.execute("INSERT OR IGNORE INTO query_embeddings (key, model, dim, embedding, created_at)"
                         " VALUES (?, ?, ?, ?, ?)",
                         (key, EMBEDDING_MODEL, len(emb), pack_embedding(emb), now_utc()))
            conn.commit()
    except Exception as e:
        print(f"  query-cache write failed ({type(e).__name__}: {str(e)[:120]}); vector not cached",
              file=sys.stderr, flush=True)


def embed_query(client, text: str) -> list[float]:
    """Embed ONE interactive query: cache-first, and never sleep on a rate limit.

    Returns a vector from the in-process cache, else the shared SQLite cache, else ONE API
    call with max_retries=1 (a 429 raises immediately, so callers degrade to FTS-only in
    milliseconds instead of inheriting the bulk backoff). Cache failures are misses, never
    errors. Failed embeds are never cached. The mock backend is deterministic and uncached.
    """
    if EMBED_BACKEND == "mock":
        return mock_embedding(text)
    key = _query_cache_key(text)
    emb = _QUERY_CACHE_MEM.get(key)
    if emb is None:
        emb = _query_cache_get(key)
    if emb is None:
        if EMBED_BACKEND == "local":
            emb = _local_encode([text], query=True)[0]
        else:
            emb = embed_batch(client, [text], max_retries=1)[0]
        _query_cache_put(key, emb)
    if len(_QUERY_CACHE_MEM) >= _QUERY_CACHE_MEM_MAX:
        _QUERY_CACHE_MEM.clear()
    _QUERY_CACHE_MEM[key] = emb
    return emb


def pack_embedding(values: list[float]) -> bytes:
    """Pack a float list into a binary blob for SQLite storage."""
    return struct.pack(f"{len(values)}f", *values)


def unpack_embedding(blob: bytes) -> list[float]:
    """Unpack a binary blob into a float list."""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob[:n * 4]))


def _embed_text(c: dict) -> str:
    """Canonical text fed to the embedding model for a chunk. Single source of truth: the
    reuse cache keys on exactly what a fresh embed would produce."""
    return f"Project: {c['project']} | Type: {c['store_type']} | {c['title']}\n{c['content']}"


def _valid_embedding(emb) -> bool:
    """True only for a usable vector (right dimension, no NaN)."""
    return emb is not None and len(emb) == EMBEDDING_DIM and not any(v != v for v in emb)


def _embed_hash(c: dict) -> str:
    """md5 of the canonical embed text: the reuse lookup key."""
    return _md5(_embed_text(c))


def _backfill_embed_hashes(conn: sqlite3.Connection) -> int:
    """Fill embed_hash for rows embedded before that column existed. Local only (no API
    call): recomputed from the stored fields, so it cannot drift from a fresh _embed_hash()."""
    pending = conn.execute("SELECT id, project, store_type, title, content FROM chunks "
                           "WHERE embed_hash IS NULL AND embedding IS NOT NULL").fetchall()
    if not pending:
        return 0
    for rid, project, store_type, title, content in pending:
        h = _embed_hash({"project": project, "store_type": store_type, "title": title, "content": content})
        conn.execute("UPDATE chunks SET embed_hash = ? WHERE id = ?", (h, rid))
    conn.commit()
    print(f"  embed_hash backfill: {len(pending)} row(s) hashed locally (no API calls)", flush=True)
    return len(pending)


def _resolve_corpus_reuse(conn: sqlite3.Connection, chunks: list[dict]) -> int:
    """Corpus-wide embedding reuse: a chunk whose EXACT embed text was already embedded
    anywhere in the corpus (not just under the same source_path) reuses that vector.

    md5 is a lookup index, never proof of equality: every candidate row sharing a hash is
    kept and the canonical text is compared byte for byte before a vector is reused, so a
    hash collision falls through to a fresh embed instead of reusing the wrong vector.

    Mutates each chunk dict: sets "_embed_hash" always, and "_embedding_blob" on a hit.
    Only SELECTs, so it is safe during a dry run. Returns the number of new hits.
    """
    hash_to_ids: dict[str, list[int]] = {}
    for h, rid in conn.execute("SELECT embed_hash, id FROM chunks WHERE embed_hash IS NOT NULL "
                               "AND embedding IS NOT NULL AND length(embedding) = ?", (EMBEDDING_DIM * 4,)):
        hash_to_ids.setdefault(h, []).append(rid)

    need_ids: set[int] = set()
    for c in chunks:
        if "_embed_hash" not in c:
            c["_embed_hash"] = _embed_hash(c)
        if c.get("_embedding_blob") is not None:
            continue
        need_ids.update(hash_to_ids.get(c["_embed_hash"], ()))

    candidate_text_by_id: dict[int, str] = {}
    blob_by_id: dict[int, bytes] = {}
    need_list = sorted(need_ids)
    for i in range(0, len(need_list), 500):
        id_batch = need_list[i:i + 500]
        placeholders = ",".join("?" * len(id_batch))
        for rid, project, store_type, title, content, emb in conn.execute(
                f"SELECT id, project, store_type, title, content, embedding FROM chunks "
                f"WHERE id IN ({placeholders})", id_batch):
            candidate_text_by_id[rid] = _embed_text(
                {"project": project, "store_type": store_type, "title": title, "content": content})
            blob_by_id[rid] = emb

    hits = 0
    for c in chunks:
        if c.get("_embedding_blob") is not None:
            continue
        target_text = _embed_text(c)
        for rid in hash_to_ids.get(c["_embed_hash"], ()):
            if candidate_text_by_id.get(rid) == target_text:
                c["_embedding_blob"] = blob_by_id[rid]
                hits += 1
                break
    return hits


def _assert_index_model(conn: sqlite3.Connection) -> None:
    """Refuse to index into a database embedded with another model: the text-hash reuse paths
    would otherwise keep (and relabel) the old model's vectors without a single embed call."""
    row = conn.execute("SELECT value FROM index_meta WHERE key = 'embedding_model'").fetchone()
    wrong_dim = conn.execute("SELECT 1 FROM chunks WHERE embedding IS NOT NULL AND length(embedding) != ? LIMIT 1",
                             (EMBEDDING_DIM * 4,)).fetchone()
    if (row and row[0] != EMBEDDING_MODEL) or wrong_dim:
        raise SystemExit(
            f"index holds {row[0] if row else 'unknown-model'} vectors but ANAMNESIS_EMBED_BACKEND={EMBED_BACKEND} "
            f"embeds with {EMBEDDING_MODEL} ({EMBEDDING_DIM}d). Rebuild the database with `build` "
            f"(delete {MEMORY_DB} first), or set the backend/model to match the index.")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Mismatched lengths (vectors from two models) score 0 instead of
    zip() silently truncating to a meaningless number."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def freshness_score(modified_at: str) -> float:
    """Score based on recency of modification (date prefix of an ISO timestamp)."""
    if not modified_at:
        return 0.5
    try:
        mod = datetime.fromisoformat(modified_at[:10])
        days = (datetime.now(timezone.utc).replace(tzinfo=None) - mod).days
        if days <= 7:
            return 1.0
        if days <= 30:
            return 0.8
        if days <= 90:
            return 0.5
        return 0.3
    except (ValueError, TypeError):
        return 0.5


# --------------------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------------------
def init_db(db_path: Path, label: str = "index.py") -> sqlite3.Connection:
    """Create the SQLite database (FTS5 + vector storage) and run pending schema migrations."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = connect_with_lock_warning(db_path, label=label)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT UNIQUE NOT NULL,
            store_type TEXT NOT NULL,
            source_path TEXT NOT NULL,
            title TEXT,
            content TEXT NOT NULL,
            embedding BLOB,
            created_at TEXT,
            modified_at TEXT,
            project TEXT,
            topic_ids TEXT,
            weight REAL DEFAULT 1.0,
            indexed_at TEXT NOT NULL
        );

        -- freshness reporting reads MAX(indexed_at); without this it is a full scan
        CREATE INDEX IF NOT EXISTS idx_chunks_indexed_at ON chunks(indexed_at);

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_id,
            title,
            content,
            store_type,
            project,
            topic_ids,
            content=chunks,
            content_rowid=id,
            tokenize='porter unicode61'
        );

        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, chunk_id, title, content, store_type, project, topic_ids)
            VALUES (new.id, new.chunk_id, new.title, new.content, new.store_type, new.project, new.topic_ids);
        END;

        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, title, content, store_type, project, topic_ids)
            VALUES ('delete', old.id, old.chunk_id, old.title, old.content, old.store_type, old.project, old.topic_ids);
        END;

        CREATE TABLE IF NOT EXISTS index_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE TABLE IF NOT EXISTS source_meta (
            source_path TEXT PRIMARY KEY,
            mtime REAL NOT NULL,
            store_type TEXT NOT NULL,
            project TEXT NOT NULL
        );
    """)
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
    for col, ddl in (("tags", "TEXT DEFAULT ''"), ("priority", "TEXT DEFAULT 'important'"),
                     ("load_when", "TEXT DEFAULT ''"), ("retirement_status", "TEXT DEFAULT 'active'"),
                     ("metadata", "TEXT DEFAULT ''"), ("live_source", "TEXT DEFAULT ''"),
                     ("embed_hash", "TEXT")):
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE chunks ADD COLUMN {col} {ddl}")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_embed_hash ON chunks(embed_hash)")
    # update_index() runs `WHERE source_path = ?` once per changed source and once per removed one.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source_path ON chunks(source_path)")
    conn.commit()
    return conn


# --------------------------------------------------------------------------------------
# Chunkers
# --------------------------------------------------------------------------------------
def _parse_yaml_frontmatter(text: str) -> tuple[dict, str]:
    """Parse simple YAML frontmatter (scalars and `- item` lists). Returns (meta, body)."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    fm_text = text[3:end].strip()
    body = text[end + 4:].strip()
    meta: dict = {}
    current_key = None
    current_list = None
    for line in fm_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and current_key and current_list is not None:
            current_list.append(stripped[2:].strip().strip('"').strip("'"))
            continue
        if ":" in stripped:
            if current_key and current_list is not None:
                meta[current_key] = current_list
                current_list = None
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if not val:
                current_key = key
                current_list = []
            else:
                meta[key] = val
                current_key = key
                current_list = None
    if current_key and current_list is not None:
        meta[current_key] = current_list
    return meta, body


def _fm_topic_ids(fm: dict) -> str:
    """Space-separated topic ids from frontmatter `topics` / `capabilities` / `tags`."""
    for key in ("topics", "capabilities", "tags"):
        raw = fm.get(key)
        if isinstance(raw, list) and raw:
            return " ".join(str(c) for c in raw if c)
        if isinstance(raw, str) and raw.strip():
            return " ".join(t.strip() for t in raw.strip("[]").split(",") if t.strip())
    return ""


def _split_respecting_structure(text: str, max_chars: int = 2000, overlap_chars: int = 150) -> list[str]:
    """Split text at meaningful boundaries, preserving structural elements.

    1. identify atomic blocks (code fences, tables, contiguous lists);
    2. split the rest on paragraph boundaries;
    3. greedily combine segments up to max_chars;
    4. never split inside an atomic block (an oversized block stays whole);
    5. prepend a short overlap from the previous chunk for embedding context.
    """
    if len(text) <= max_chars:
        return [text]

    atomic_blocks = []
    for m in re.finditer(r'^```[^\n]*\n.*?^```', text, re.MULTILINE | re.DOTALL):
        atomic_blocks.append(('code', m.start(), m.end(), m.group()))
    for m in re.finditer(r'(?:^\|.*\|$\n?){2,}', text, re.MULTILINE):
        atomic_blocks.append(('table', m.start(), m.end(), m.group()))
    for m in re.finditer(r'(?:^[\s]*[-*]\s.*$\n?){3,}', text, re.MULTILINE):
        atomic_blocks.append(('list', m.start(), m.end(), m.group()))
    for m in re.finditer(r'(?:^[\s]*\d+[.)]\s.*$\n?){3,}', text, re.MULTILINE):
        atomic_blocks.append(('numlist', m.start(), m.end(), m.group()))

    atomic_blocks.sort(key=lambda x: (x[1], -(x[2] - x[1])))
    filtered = []
    last_end = -1
    for block in atomic_blocks:
        if block[1] >= last_end:
            filtered.append(block)
            last_end = block[2]
    atomic_blocks = filtered

    segments = []
    pos = 0
    for _btype, bstart, bend, bcontent in atomic_blocks:
        before = text[pos:bstart]
        if before.strip():
            for para in re.split(r'\n\n+', before):
                if para.strip():
                    segments.append(('para', para.strip()))
        segments.append(('atomic', bcontent.strip()))
        pos = bend
    remaining = text[pos:]
    if remaining.strip():
        for para in re.split(r'\n\n+', remaining):
            if para.strip():
                segments.append(('para', para.strip()))

    chunks = []
    current_parts = []
    current_len = 0
    for stype, content in segments:
        content_len = len(content)
        if stype == 'atomic' and content_len > max_chars:
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_len = 0
            chunks.append(content)
            continue
        new_len = current_len + content_len + (2 if current_parts else 0)
        if new_len > max_chars and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = []
            current_len = 0
        current_parts.append(content)
        current_len += content_len + (2 if len(current_parts) > 1 else 0)
    if current_parts:
        chunks.append("\n\n".join(current_parts))

    if overlap_chars > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            overlap_text = chunks[i - 1][-overlap_chars:]
            for marker in ['. ', '.\n', '\n']:
                idx = overlap_text.find(marker)
                if idx != -1 and idx < len(overlap_text) - 20:
                    overlap_text = overlap_text[idx + len(marker):]
                    break
            overlapped.append(f"<!-- overlap -->{overlap_text}\n\n{chunks[i]}")
        chunks = overlapped

    return chunks if chunks else [text]


MIN_SECTION_CHARS = 30


def chunk_markdown(text: str, source_path: str, max_chars: int = 2000) -> list[dict]:
    """Split markdown into chunks by headings (levels 1-3), then by paragraphs.

    Frontmatter is parsed, not indexed raw: `name` titles an untitled first section,
    `description` is prepended to the first chunk, and `topics`/`capabilities`/`tags`
    become the chunks' topic_ids.
    """
    fm, body = _parse_yaml_frontmatter(text)
    stem = Path(source_path).stem
    first_title = str(fm.get("name") or stem)
    topic_ids = _fm_topic_ids(fm)
    desc = fm.get("description")

    sections = []
    current_title = first_title
    current_lines: list[str] = []
    for line in body.splitlines():
        if re.match(r'^#{1,3}\s', line) and current_lines:
            sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line.lstrip("#").strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))

    chunks = []
    for title, content in sections:
        if not content or len(content) < MIN_SECTION_CHARS:
            continue
        sub_chunks = _split_respecting_structure(content, max_chars)
        for i, chunk_text in enumerate(sub_chunks):
            chunk_title = title if len(sub_chunks) == 1 else f"{title} (part {i + 1})"
            chunks.append({"title": chunk_title, "content": chunk_text})

    if not chunks and body.strip():
        sub_chunks = _split_respecting_structure(body, max_chars)
        for i, chunk_text in enumerate(sub_chunks):
            chunks.append({"title": first_title if len(sub_chunks) == 1 else f"{first_title} (part {i + 1})",
                           "content": chunk_text})
    if chunks and desc and isinstance(desc, str):
        chunks[0]["content"] = f"{desc}\n\n{chunks[0]['content']}"
    if topic_ids:
        for c in chunks:
            c["topic_ids"] = topic_ids
    return chunks


def _flush_facts_section(chunks: list, title: str, lines: list[str]):
    """Flush a section: the whole section, plus one chunk per substantial bullet when the
    section is bullet-heavy (a fact list packs many distinct facts under one heading)."""
    content = "\n".join(lines).strip()
    if not content or len(content) < 15:
        return
    bullets = [ln for ln in lines if ln.strip().startswith("- ")]
    chunks.append({"title": title, "content": content})
    if len(bullets) >= 5:
        for bullet in bullets:
            bullet_text = bullet.strip().lstrip("- ").strip()
            if len(bullet_text) > 40:
                chunks.append({"title": f"{title}: {bullet_text[:80]}", "content": bullet_text})


def chunk_facts(text: str, source_path: str) -> list[dict]:
    """Fine-grained chunker for fact sheets (project summaries, capability notes): one chunk
    per heading section, one per bullet in bullet-heavy sections, and an "Overview" chunk
    combining the first three sections for queries that span several aspects."""
    fm, body = _parse_yaml_frontmatter(text)
    topic_ids = _fm_topic_ids(fm)
    parse_text = body if body.strip() else text

    chunks: list[dict] = []
    sections = []
    current_title = Path(source_path).stem
    current_lines: list[str] = []
    for line in parse_text.splitlines():
        if re.match(r'^#{1,3}\s', line):
            if current_lines:
                sections.append((current_title, list(current_lines)))
                _flush_facts_section(chunks, current_title, current_lines)
            current_title = line.lstrip("#").strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, list(current_lines)))
        _flush_facts_section(chunks, current_title, current_lines)

    if len(sections) >= 2:
        summary_lines: list[str] = []
        for _, lines in sections[:3]:
            summary_lines.extend(lines)
        summary = "\n".join(summary_lines).strip()
        if summary:
            chunks.insert(0, {"title": f"{Path(source_path).stem} - Overview", "content": summary})
    if not chunks and text.strip():
        chunks.append({"title": Path(source_path).stem, "content": text})
    if topic_ids:
        for c in chunks:
            c.setdefault("topic_ids", topic_ids)
    return chunks


def chunk_article(text: str, source_path: str) -> list[dict]:
    """Chunk an article / blog post (MD or MDX): a metadata chunk (title, description, tags,
    date) for keyword matching, then structure-aware body chunks. JSX components and image
    references are stripped from the body."""
    meta, body = _parse_yaml_frontmatter(text)
    body = re.sub(r'<[A-Z][^>]*/>', '', body)
    body = re.sub(r'<[A-Z][A-Za-z]*\b[^>]*>.*?</[A-Z][A-Za-z]*>', '', body, flags=re.DOTALL)
    body = re.sub(r'!\[.*?\]\(.*?\)', '', body)

    title = str(meta.get("title") or meta.get("name") or Path(source_path).stem)
    tags = meta.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip().strip('"') for t in tags.strip("[]").split(",") if t.strip()]
    published = meta.get("publishedAt") or meta.get("date") or ""
    meta_content = (f"Title: {title}\nDescription: {meta.get('description', '')}\n"
                    f"Tags: {', '.join(tags)}\nPublished: {published}\n")
    tag_ids = " ".join(tags)
    chunks = [{"title": f"{title} - metadata", "content": meta_content, "topic_ids": tag_ids}]

    sections = []
    heading = title
    lines: list[str] = []
    for line in body.splitlines():
        if re.match(r'^#{1,3}\s', line) and lines:
            sections.append((heading, "\n".join(lines).strip()))
            heading = line.lstrip("#").strip()
            lines = [line]
        else:
            lines.append(line)
    if lines:
        sections.append((heading, "\n".join(lines).strip()))
    for heading, section_text in sections:
        if not section_text or len(section_text) < MIN_SECTION_CHARS:
            continue
        subs = _split_respecting_structure(section_text, max_chars=1500)
        for i, chunk_text in enumerate(subs):
            chunks.append({"title": f"{title} - {heading}" + (f" (part {i + 1})" if len(subs) > 1 else ""),
                           "content": chunk_text, "topic_ids": tag_ids})
    return chunks


def _render_json_entry_prose(entry: dict, entry_type: str) -> tuple[str, str, str]:
    """Render a structured JSON entry as labelled prose for BM25 and embedding retrieval.

    One chunk, two representations: prose for the retriever (this text, stored as `content`),
    the original structured entry for the reasoner (stored in the `metadata` column).
    Returns (title, content_prose, topic_ids_str).
    """
    lines: list[str] = []
    topic_ids = ""

    if entry_type == "mistakes":
        title = entry.get("mistake_id") or entry.get("title", "unknown")
        lines.append(f"Mistake: {entry.get('symptom', title)}")
        for label, key in (("Symptom", "symptom"), ("Root cause", "root_cause"),
                           ("False assumption", "false_assumption"), ("Trigger pattern", "trigger_pattern"),
                           ("Correction", "correction"), ("Prevention rule", "prevention_rule"),
                           ("Severity", "severity")):
            if entry.get(key):
                lines.append(f"{label}: {entry[key]}")
        t_ids = entry.get("topic_ids", [])
        if isinstance(t_ids, list) and t_ids:
            topic_ids = " ".join(map(str, t_ids))
            lines.append(f"Applies to topics: {', '.join(map(str, t_ids))}")
        if entry.get("evidence"):
            lines.append(f"Evidence: {entry['evidence']}")
        return str(title), "\n".join(lines), topic_ids

    if entry_type == "contacts":
        name = entry.get("name", "unknown")
        lines.append(f"Contact: {name}")
        for label, key in (("Relationship", "relationship"), ("Context", "context"), ("Notes", "notes")):
            if entry.get(key):
                lines.append(f"{label}: {entry[key]}")
        projects = entry.get("projects", [])
        if isinstance(projects, list) and projects:
            lines.append(f"Projects: {', '.join(map(str, projects))}")
        return str(name), "\n".join(lines), topic_ids

    if entry_type == "topics":
        label = entry.get("label", "unknown")
        lines.append(f"Topic: {label}")
        for lab, key in (("Project scope", "project_scope"), ("Related topics", "related_topics"),
                         ("Associated memories", "memory_ids")):
            val = entry.get(key, [])
            if isinstance(val, list) and val:
                lines.append(f"{lab}: {', '.join(map(str, val))}")
        return str(label), "\n".join(lines), topic_ids

    if entry_type == "capabilities":
        proj_name = entry.get("proj_name", "unknown")
        proj_data = entry.get("proj_data", {})
        lines.append(f"Project: {proj_name}")
        if proj_data.get("status"):
            lines.append(f"Status: {proj_data['status']}")
        caps = proj_data.get("capabilities", [])
        if isinstance(caps, list) and caps:
            lines.append("Capabilities:")
            for cap in caps:
                lines.append(f"- Capability: {cap.get('capability_id', '')}")
                if cap.get("description"):
                    lines.append(f"  Description: {cap['description']}")
                for lab, key in (("Inputs", "inputs"), ("Outputs", "outputs"), ("Constraints", "constraints"),
                                 ("Reusable assets", "reusable_assets")):
                    if cap.get(key):
                        lines.append(f"  {lab}: {', '.join(map(str, cap[key]))}")
                if cap.get("maturity"):
                    lines.append(f"  Maturity: {cap['maturity']}")
        return f"{proj_name} capabilities", "\n".join(lines), topic_ids

    if entry_type == "edges":
        consumer = entry.get("consumer", "")
        provider = entry.get("provider", "")
        cap_id = entry.get("capability_id", "")
        title = f"{consumer} -> {provider} ({cap_id})" if cap_id else entry.get("title", "edge")
        lines.append(f"Capability edge: {consumer} consumes {cap_id} from {provider}")
        if entry.get("integration_pattern"):
            lines.append(f"Integration pattern: {entry['integration_pattern']}")
        if entry.get("status"):
            lines.append(f"Status: {entry['status']}")
        return str(title), "\n".join(lines), topic_ids

    title = entry.get("title") or entry.get("name") or entry.get("id", "unknown")
    return str(title), json.dumps(entry, indent=2, ensure_ascii=False), topic_ids


def _generic_prose(value) -> str:
    """Labelled prose for an arbitrary JSON value (dict -> `key: value` lines)."""
    if isinstance(value, dict):
        out = []
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                v = json.dumps(v, ensure_ascii=False)
            out.append(f"{k}: {v}")
        return "\n".join(out)
    if isinstance(value, list):
        return "\n".join(f"- {json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v}"
                         for v in value)
    return str(value)


_METADATA_CAP = 20000


def _twin(chunk: dict, entry) -> dict:
    """Attach the structured original of a prose-rendered entry (the "twin")."""
    try:
        raw = json.dumps(entry, ensure_ascii=False)
    except (TypeError, ValueError):
        return chunk
    if len(raw) <= _METADATA_CAP:
        chunk["metadata"] = raw
    return chunk


def chunk_json_entries(data, source_path: str) -> list[dict]:
    """Split JSON data into per-entry chunks with a prose representation and a structured twin.

    Known list keys (mistakes, contacts, edges, topics, projects) get purpose-built prose;
    objects/solutions keep their JSON text; any other top-level structure falls back to one
    labelled-prose chunk per top-level entry.
    """
    chunks: list[dict] = []
    stem = Path(source_path).stem
    if isinstance(data, dict):
        handled = False
        for key in ("mistakes", "contacts", "edges"):
            if isinstance(data.get(key), list):
                handled = True
                for entry in data[key]:
                    if not isinstance(entry, dict):
                        continue
                    title, prose, topic_ids = _render_json_entry_prose(entry, key)
                    chunk = {"title": title, "content": prose}
                    if topic_ids:
                        chunk["topic_ids"] = topic_ids
                    chunks.append(_twin(chunk, entry))
        for key in ("objects", "solutions"):
            if isinstance(data.get(key), list):
                handled = True
                for entry in data[key]:
                    if not isinstance(entry, dict):
                        continue
                    title = entry.get("title") or entry.get("name") or entry.get("mistake_id") or entry.get("id", "unknown")
                    chunks.append(_twin({"title": str(title), "content": json.dumps(entry, indent=2, ensure_ascii=False)},
                                        entry))
        if isinstance(data.get("topics"), dict):
            handled = True
            for _tid, topic_data in data["topics"].items():
                if isinstance(topic_data, dict):
                    title, prose, _ = _render_json_entry_prose(topic_data, "topics")
                    chunks.append(_twin({"title": title, "content": prose}, topic_data))
        if isinstance(data.get("projects"), dict):
            handled = True
            for proj_name, proj_data in data["projects"].items():
                if isinstance(proj_data, dict):
                    title, prose, _ = _render_json_entry_prose({"proj_name": proj_name, "proj_data": proj_data},
                                                               "capabilities")
                    chunks.append(_twin({"title": title, "content": prose}, proj_data))
        if not handled:
            for key, value in data.items():
                if value in (None, "", [], {}):
                    continue
                chunks.append(_twin({"title": str(key), "content": _generic_prose(value)}, value))
    elif isinstance(data, list):
        for i, entry in enumerate(data):
            if entry in (None, "", [], {}):
                continue
            title = (entry.get("title") or entry.get("name") or entry.get("id")) if isinstance(entry, dict) else None
            chunks.append(_twin({"title": str(title or f"{stem} #{i + 1}"), "content": _generic_prose(entry)}, entry))
    return chunks


# --------------------------------------------------------------------------------------
# Sources config
# --------------------------------------------------------------------------------------
class ConfigError(ValueError):
    """The sources config is malformed."""


CHUNKERS = ("auto", "markdown", "facts", "article", "json")
KINDS = ("files", "git_metadata", "git_branch_docs")
_COMMON_KEYS = {"name", "kind", "enabled", "project"}
_KIND_KEYS = {
    "files": {"root", "glob", "exclude", "store_type", "store_type_rules", "chunker", "weight",
              "follow_symlinks", "max_files"},
    "git_metadata": {"repos", "store_type"},
    "git_branch_docs": {"repo", "glob", "base_ref", "store_type", "max_files"},
}
_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _builtin_config() -> dict:
    """What `index.py` indexes with no ANAMNESIS_SOURCES: the git-tracked data/ tree."""
    return {
        "version": 1,
        "defaults": {"project": DEFAULT_PROJECT},
        "sources": [
            {"name": "memory", "root": str(DATA_DIR / "memory"), "glob": ["*.md"], "store_type": "memory",
             "store_type_rules": [{"match": "feedback*", "store_type": "feedback"},
                                  {"match": "reference*", "store_type": "reference"}]},
            {"name": "decisions", "root": str(DATA_DIR / "decisions"), "glob": ["**/*.md"],
             "store_type": "decision"},
            {"name": "manifests", "root": str(DATA_DIR / "manifests"), "glob": ["*.json"],
             "store_type": "index"},
        ],
    }


def _expand(value: str, field: str) -> str:
    """Expand ~, $VAR and ${VAR}. ANAMNESIS_DATA_DIR / ANAMNESIS_HOME resolve to the effective
    values even when unset in the environment. An undefined variable is an error (an empty
    expansion would silently turn `$X/memory` into `/memory`)."""
    known = {"ANAMNESIS_DATA_DIR": str(DATA_DIR), "ANAMNESIS_HOME": str(ANAMNESIS_HOME)}

    def sub(m: re.Match) -> str:
        name = m.group(1) or m.group(2)
        val = os.environ.get(name) or known.get(name)
        if val is None:
            raise ConfigError(f"{field}: undefined environment variable ${name}")
        return val
    return os.path.expanduser(re.sub(r"\$\{(\w+)\}|\$(\w+)", sub, value))


_URL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://|^(?:file|data|javascript):", re.IGNORECASE)


def _reject_url(value: str, field: str) -> str:
    """Sources are filesystem paths only: URL schemes (http, ftp, file:, ...) are rejected."""
    if _URL_RE.match(value.strip()):
        raise ConfigError(f"{field}: URLs are not allowed, use a filesystem path ({value!r})")
    return value


def _as_list(v, field: str) -> list:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, list) and all(isinstance(x, str) for x in v):
        return list(v)
    raise ConfigError(f"{field}: expected a string or a list of strings")


def _validate_config(cfg, base: Path) -> dict:
    """Validate and normalise a parsed sources config (relative roots resolved against `base`)."""
    if not isinstance(cfg, dict):
        raise ConfigError("sources config must be a mapping at the top level")
    unknown = set(cfg) - {"version", "defaults", "store_weights", "project_aliases", "shared_projects", "sources"}
    if unknown:
        raise ConfigError(f"unknown top-level key(s): {sorted(unknown)}")
    if cfg.get("version", 1) != 1:
        raise ConfigError(f"unsupported config version {cfg.get('version')!r} (this indexer reads version 1)")
    defaults = cfg.get("defaults") or {}
    if not isinstance(defaults, dict) or set(defaults) - {"project", "exclude"}:
        raise ConfigError("defaults: allowed keys are project, exclude")
    def_project = str(defaults.get("project") or DEFAULT_PROJECT)
    def_exclude = _as_list(defaults.get("exclude"), "defaults.exclude")

    weights = cfg.get("store_weights") or {}
    if not isinstance(weights, dict) or not all(isinstance(v, (int, float)) and v > 0 for v in weights.values()):
        raise ConfigError("store_weights: expected a mapping of store_type -> positive number")
    aliases = cfg.get("project_aliases") or {}
    if not isinstance(aliases, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in aliases.items()):
        raise ConfigError("project_aliases: expected a mapping of alias -> project name")
    shared = _as_list(cfg.get("shared_projects", ["shared"]), "shared_projects")

    raw_sources = cfg.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ConfigError("sources: expected a non-empty list")
    out = []
    for i, s in enumerate(raw_sources):
        where = f"sources[{i}]"
        if not isinstance(s, dict):
            raise ConfigError(f"{where}: expected a mapping")
        kind = s.get("kind", "files")
        if kind not in KINDS:
            raise ConfigError(f"{where}: kind must be one of {KINDS}")
        bad = set(s) - _COMMON_KEYS - _KIND_KEYS[kind]
        if bad:
            raise ConfigError(f"{where}: unknown key(s) for kind {kind}: {sorted(bad)}")
        n = {"kind": kind, "name": str(s.get("name") or f"{kind}-{i}"), "enabled": bool(s.get("enabled", True)),
             "project": str(s.get("project") or def_project)}
        stype = s.get("store_type", {"files": "memory", "git_metadata": "project_meta",
                                     "git_branch_docs": "project_doc"}[kind])
        if not isinstance(stype, str) or not _TYPE_RE.match(stype):
            raise ConfigError(f"{where}.store_type: expected a lowercase identifier like 'decision_log'")
        n["store_type"] = stype
        if kind == "files":
            if not s.get("root"):
                raise ConfigError(f"{where}: `root` is required")
            root = Path(_expand(_reject_url(str(s["root"]), f"{where}.root"), f"{where}.root"))
            n["root"] = root if root.is_absolute() else (base / root)
            n["glob"] = _as_list(s.get("glob"), f"{where}.glob") or ["**/*.md"]
            n["exclude"] = def_exclude + _as_list(s.get("exclude"), f"{where}.exclude")
            chunker = s.get("chunker", "auto")
            if chunker not in CHUNKERS:
                raise ConfigError(f"{where}.chunker: must be one of {CHUNKERS}")
            n["chunker"] = chunker
            w = s.get("weight", 1.0)
            if not isinstance(w, (int, float)) or w <= 0:
                raise ConfigError(f"{where}.weight: expected a positive number")
            n["weight"] = float(w)
            rules = s.get("store_type_rules") or []
            if not isinstance(rules, list):
                raise ConfigError(f"{where}.store_type_rules: expected a list")
            n["store_type_rules"] = []
            for j, r in enumerate(rules):
                if not (isinstance(r, dict) and set(r) == {"match", "store_type"}
                        and isinstance(r["match"], str) and isinstance(r["store_type"], str)
                        and _TYPE_RE.match(r["store_type"])):
                    raise ConfigError(f"{where}.store_type_rules[{j}]: expected {{match: <glob>, store_type: <name>}}")
                n["store_type_rules"].append((r["match"], r["store_type"]))
            n["follow_symlinks"] = bool(s.get("follow_symlinks", False))
            n["max_files"] = int(s.get("max_files", 50000))
        elif kind == "git_metadata":
            repos = _as_list(s.get("repos"), f"{where}.repos")
            if not repos:
                raise ConfigError(f"{where}: `repos` is required")
            n["repos"] = []
            for r in repos:
                rp = Path(_expand(_reject_url(r, f"{where}.repos"), f"{where}.repos"))
                n["repos"].append(rp if rp.is_absolute() else base / rp)
        else:  # git_branch_docs
            if not s.get("repo"):
                raise ConfigError(f"{where}: `repo` is required")
            repo = Path(_expand(_reject_url(str(s["repo"]), f"{where}.repo"), f"{where}.repo"))
            n["repo"] = repo if repo.is_absolute() else base / repo
            n["glob"] = _as_list(s.get("glob"), f"{where}.glob") or ["docs/**/*.md"]
            n["base_ref"] = str(s.get("base_ref", "origin/main"))
            n["max_files"] = int(s.get("max_files", 1500))
        out.append(n)
    return {"sources": out, "store_weights": dict(weights), "project_aliases": dict(aliases),
            "shared_projects": shared}


def load_sources_config(path: str | os.PathLike | None = None) -> dict:
    """Load + validate the sources config (`path`, else ANAMNESIS_SOURCES, else the built-in
    data/ layout). Applies the config's store_weights to STORE_WEIGHTS."""
    path = path or SOURCES_FILE
    if not path:
        cfg = _validate_config(_builtin_config(), DATA_DIR)
    else:
        p = Path(str(path)).expanduser()
        try:
            raw = p.read_text(encoding="utf-8")
        except OSError as e:
            raise ConfigError(f"cannot read sources config {p}: {e}") from e
        if p.suffix.lower() == ".json":
            parsed = json.loads(raw)
        else:
            try:
                import yaml
            except ImportError as e:
                raise ConfigError("PyYAML is required for a YAML sources config (pip install pyyaml), "
                                  "or use a .json config") from e
            parsed = yaml.safe_load(raw)
        cfg = _validate_config(parsed, p.resolve().parent)
    STORE_WEIGHTS.update(cfg["store_weights"])
    return cfg


def _excluded(rel: str, patterns: list[str]) -> bool:
    """Exclude test against a root-relative posix path. A bare name (no wildcard, no slash)
    excludes any path component of that name; otherwise fnmatch, where `*` crosses `/`."""
    parts = rel.split("/")
    for pat in patterns:
        if not any(ch in pat for ch in "*?[/"):
            if pat in parts:
                return True
            continue
        if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch("/" + rel, pat):
            return True
        if pat.startswith("**/") and fnmatch.fnmatch(rel, pat[3:]):
            return True
    return False


def _store_type_for(src: dict, rel: str) -> str:
    name = rel.rsplit("/", 1)[-1]
    for pat, stype in src["store_type_rules"]:
        if fnmatch.fnmatch(rel if "/" in pat else name, pat):
            return stype
    return src["store_type"]


def _collect_files(src: dict) -> list[dict]:
    root: Path = src["root"]
    if not root.exists():
        return []
    root_real = root.resolve()
    files = [root] if root.is_file() else sorted({f for pat in src["glob"] for f in root.glob(pat)})
    out: list[dict] = []
    for f in files:
        if len(out) >= src["max_files"]:
            print(f"  source {src['name']}: hit max_files={src['max_files']}, stopping", flush=True)
            break
        if not f.is_file():
            continue
        rel = f.name if root.is_file() else f.relative_to(root).as_posix()
        if _excluded(rel, src["exclude"]):
            continue
        if not src["follow_symlinks"] and not root.is_file():
            try:
                f.resolve().relative_to(root_real)
            except ValueError:
                print(f"  SKIP {f}: symlink leaves the source root (set follow_symlinks: true to allow)", flush=True)
                continue
        try:
            size = f.stat().st_size
        except OSError:
            continue
        if size > MAX_FILE_BYTES:
            print(f"  SKIP {f}: {size} bytes exceeds ANAMNESIS_MAX_FILE_BYTES={MAX_FILE_BYTES}", flush=True)
            continue
        entry = {"path": str(f), "store_type": _store_type_for(src, rel), "project": src["project"],
                 "chunker": src["chunker"], "weight": src["weight"], "source_name": src["name"]}
        if f.suffix.lower() in (".md", ".mdx", ".markdown", ".txt"):
            try:
                fm, _ = _parse_yaml_frontmatter(f.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                fm = {}
            if fm.get("retirement_status", "active") == "retired":
                continue  # retired sources are not indexed
            load_when = fm.get("load_when", "")
            entry["priority"] = fm.get("priority", "important")
            entry["retirement_status"] = fm.get("retirement_status", "active")
            entry["load_when"] = ", ".join(load_when) if isinstance(load_when, list) else load_when
            entry["live_source"] = fm.get("live_source", "")
        out.append(entry)
    return out


def _git_cmd(path: str, args: list[str]) -> str:
    """Run a git command with a timeout; return stdout or '' on any failure."""
    try:
        r = subprocess.run(["git", "-C", path] + args, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


BRANCH_DOCS_STAGING = ANAMNESIS_HOME / "branch-docs"


def collect_branch_only_docs(src: dict) -> list[dict]:
    """Index docs that exist ONLY on remote branches, never on the base branch.

    Every other collector walks a working tree, so unmerged work that was pushed to a branch
    but never checked out here is invisible to memory, and the more work is delivered by
    branch + pull request, the more of it goes unseen. This stages matching files from each
    remote branch (not on `base_ref`) into BRANCH_DOCS_STAGING, OUTSIDE the repo so the main
    sweep never indexes them twice, with a provenance header marking them PROPOSED: unmerged
    means not merged. The staging area is rebuilt each run, so a branch deleted upstream
    stops being remembered as live.
    """
    repo = str(src["repo"])
    if not (Path(repo) / ".git").exists():
        return []
    refs = _git_cmd(repo, ["for-each-ref", "--format=%(refname:short)", "refs/remotes/origin"])
    branches = [b for b in refs.splitlines() if b and b not in ("origin/HEAD", src["base_ref"])]
    if not branches:
        return []
    staging = BRANCH_DOCS_STAGING / re.sub(r"[^A-Za-z0-9._-]+", "_", src["name"])
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    staging_real = staging.resolve()

    sources: list[dict] = []
    seen: set[str] = set()
    for br in branches:
        if len(sources) >= src["max_files"]:
            print(f"  branch-docs: hit max_files={src['max_files']}, stopping (remaining branches unscanned)")
            break
        changed = _git_cmd(repo, ["diff", "--name-only", "--diff-filter=AM", f"{src['base_ref']}...{br}"])
        for rel in changed.splitlines():
            if rel in seen or not any(fnmatch.fnmatch(rel, g) for g in src["glob"]):
                continue
            body = _git_cmd(repo, ["show", f"{br}:{rel}"])
            if not body.strip():
                continue
            safe_br = br.replace("origin/", "").replace("/", "__")
            dest = (staging / safe_br / rel).resolve()
            try:
                dest.relative_to(staging_real)
            except ValueError:
                continue  # never write outside the staging area
            seen.add(rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(
                f"<!-- PROVENANCE: this document exists only on branch `{br}`, not on {src['base_ref']}. "
                f"It is PROPOSED work, not merged. -->\n\n{body}", encoding="utf-8")
            sources.append({"path": str(dest), "store_type": src["store_type"], "project": src["project"],
                            "chunker": "auto", "weight": 1.0, "source_name": src["name"]})
    if sources:
        print(f"  branch-docs: {len(sources)} doc(s) indexed from {len(branches)} remote branch(es)")
    return sources


def collect_git_metadata(config: dict | None = None) -> list[dict]:
    """Synthetic per-repository chunks (first commit, commit count, last commit, branch,
    contributor count) for every `git_metadata` source in the config."""
    config = config or load_sources_config()
    chunks = []
    for src in config["sources"]:
        if src["kind"] != "git_metadata" or not src["enabled"]:
            continue
        for repo in src["repos"]:
            if not (repo / ".git").exists():
                continue
            path = str(repo)
            proj = repo.name
            first = _git_cmd(path, ["log", "--reverse", "--format=%ci"])
            contributors = _git_cmd(path, ["shortlog", "-sn", "--all"])
            content = (f"Project: {proj}\n"
                       f"First commit: {first.splitlines()[0] if first else ''}\n"
                       f"Total commits: {_git_cmd(path, ['rev-list', '--count', 'HEAD'])}\n"
                       f"Last commit: {_git_cmd(path, ['log', '-1', '--format=%ci (%cr)'])}\n"
                       f"Primary branch: {_git_cmd(path, ['symbolic-ref', '--short', 'HEAD'])}\n"
                       f"Contributors: {len(contributors.splitlines()) if contributors else 0}")
            chunks.append({"path": f"git:{proj}", "store_type": src["store_type"], "project": src["project"],
                           "title": f"{proj} - Git Metadata", "content": content})
    return chunks


def collect_sources(config: dict | None = None) -> list[dict]:
    """All file-backed sources to index, per the sources config."""
    config = config or load_sources_config()
    sources: list[dict] = []
    for src in config["sources"]:
        if not src["enabled"]:
            continue
        if src["kind"] == "files":
            sources.extend(_collect_files(src))
        elif src["kind"] == "git_branch_docs":
            try:  # a git failure must not cost the local sweep
                sources.extend(collect_branch_only_docs(src))
            except Exception as exc:
                print(f"  branch-docs: SKIPPED ({str(exc)[:140]}); local sources unaffected")
    seen: set[str] = set()
    unique = []
    for s in sources:  # overlapping globs / roots: first definition wins
        if s["path"] not in seen:
            seen.add(s["path"])
            unique.append(s)
    return unique


# --------------------------------------------------------------------------------------
# Build / update
# --------------------------------------------------------------------------------------
def _chunk_source(src: dict):
    """Chunk one source file. Returns (chunks, weight, modified) or None to skip."""
    path = Path(src["path"])
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        mtime = path.stat().st_mtime
    except Exception as e:
        print(f"  SKIP {path}: {e}")
        return None
    modified = datetime.fromtimestamp(mtime, timezone.utc).strftime("%Y-%m-%d")

    chunker = src.get("chunker", "auto")
    if chunker == "auto":
        chunker = "json" if path.suffix.lower() == ".json" else "markdown"
    if chunker == "json":
        try:
            chunks = chunk_json_entries(json.loads(text), str(path))
        except json.JSONDecodeError:
            return None
    elif chunker == "facts":
        chunks = chunk_facts(text, str(path))
    elif chunker == "article":
        chunks = chunk_article(text, str(path))
    else:
        chunks = chunk_markdown(text, str(path))

    weight = STORE_WEIGHTS.get(src["store_type"], 1.0) * float(src.get("weight", 1.0))
    return chunks, weight, modified


def _build_new_chunks_for_source(src: dict):
    """Chunk dicts (with positional chunk_ids) for a source. None = skip (unreadable / bad
    JSON); [] = the file legitimately yields no chunks (so its stale chunks get deleted)."""
    res = _chunk_source(src)
    if res is None:
        return None
    chunks, weight, modified = res
    path = Path(src["path"])
    path_hash = _md5(str(path))[:8]
    return [{
        "chunk_id": f"{src['store_type']}:{path_hash}:{path.stem}:{i}",
        "store_type": src["store_type"],
        "source_path": str(path),
        "title": chunk["title"],
        "content": chunk["content"],
        "modified_at": modified,
        "project": src["project"],
        "weight": weight,
        "topic_ids": chunk.get("topic_ids", ""),
        "metadata": chunk.get("metadata", ""),
        "priority": src.get("priority", "important"),
        "load_when": src.get("load_when", ""),
        "retirement_status": src.get("retirement_status", "active"),
        "live_source": src.get("live_source", ""),
    } for i, chunk in enumerate(chunks)]


def _chunk_columns(conn: sqlite3.Connection) -> set[str]:
    return {r[1] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()}


def _insert_chunk(conn, c: dict, now: str, cols: set[str] | None = None):
    """Insert one resolved chunk. c['_embedding_blob'] is packed bytes or None. Logical
    coordinate columns (added by the ontology layer) are filled when the table has them."""
    cols = cols if cols is not None else _chunk_columns(conn)
    names = ["chunk_id", "store_type", "source_path", "title", "content", "embedding", "modified_at",
             "project", "topic_ids", "weight", "indexed_at", "priority", "load_when",
             "retirement_status", "live_source", "embed_hash", "metadata"]
    vals = [c["chunk_id"], c["store_type"], c["source_path"], c["title"], c["content"],
            c.get("_embedding_blob"), c["modified_at"], c["project"], c.get("topic_ids", ""), c["weight"],
            now, c.get("priority", "important"), c.get("load_when", ""),
            c.get("retirement_status", "active"), c.get("live_source", ""),
            c.get("_embed_hash") or _embed_hash(c), c.get("metadata", "")]
    if "chunk_key" in cols and "heading_slug" in cols:
        names += ["heading_slug", "chunk_key"]
        vals += [ontology.slugify_heading(c["title"]), ontology.compute_chunk_key(c["source_path"], c["title"])]
    conn.execute(f"INSERT INTO chunks ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})", vals)


def _dedupe_chunk_ids(chunks: list[dict]) -> list[dict]:
    """Collapse duplicate chunk_ids (last write wins). chunks_fts is an external-content FTS5
    table kept in sync by AFTER INSERT / AFTER DELETE triggers; a REPLACE would delete the old
    row without firing the delete trigger (recursive_triggers is off), stranding its FTS entry
    and desynchronising the index. Deduplicating first makes the collision impossible."""
    seen: dict[str, int] = {}
    out: list[dict] = []
    for c in chunks:
        cid = c["chunk_id"]
        if cid in seen:
            out[seen[cid]] = c
        else:
            seen[cid] = len(out)
            out.append(c)
    if len(out) != len(chunks):
        print(f"  collapsed {len(chunks) - len(out)} duplicate chunk_id(s) before insert (FTS-integrity guard)",
              flush=True)
    return out


def _git_meta_chunk(gm: dict) -> dict:
    path_hash = _md5(gm["path"])[:8]
    return {"chunk_id": f"{gm['store_type']}:{path_hash}:{gm['project']}:0", "store_type": gm["store_type"],
            "source_path": gm["path"], "title": gm["title"], "content": gm["content"],
            "modified_at": now_utc()[:10], "project": gm["project"],
            "weight": STORE_WEIGHTS.get(gm["store_type"], 1.0)}


def generate_tags_batch(client, chunks: list[dict], batch_size: int = 15) -> dict[int, str]:
    """Generate hierarchical tags (topic/subtopic) for chunks with a Gemini generation model."""
    from google.genai import types
    results: dict[int, str] = {}
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        parts = [f"[{j}] Title: {c['title']}\nContent: {c['content'][:300]}" for j, c in enumerate(batch)]
        prompt = ("Extract 2-4 hierarchical tags (format: topic/subtopic) for each numbered text chunk below. "
                  "Examples: deployment/ci, memory/retrieval, billing/invoices.\n"
                  "Return one line per chunk: the chunk number followed by the tags comma-separated.\n"
                  "Format: [0] tag1/sub1, tag2/sub2\n\n" + "\n\n".join(parts))
        try:
            response = client.models.generate_content(
                model=GENERATION_MODEL, contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=1024))
            for line in (response.text or "").strip().splitlines():
                m = re.match(r'\[(\d+)\]\s*(.*)', line.strip())
                if m:
                    results[i + int(m.group(1))] = m.group(2).strip()
        except Exception as e:
            print(f"    Tag generation failed for batch {i}: {e}", flush=True)
    return results


def _set_meta(conn: sqlite3.Connection, **kv) -> None:
    for k, v in kv.items():
        conn.execute("INSERT OR REPLACE INTO index_meta VALUES (?, ?)", (k, str(v)))


def _run_ontology(conn: sqlite3.Connection, phase: str) -> None:
    """Ontology hooks: logical chunk coordinates, link re-binding after a re-index (only when
    the ontology module offers it and its link table exists), and the ontology projection.
    Never fails the index run."""
    try:
        ontology.backfill_chunk_coordinates(conn, only_missing=True)
        sync = next((getattr(ontology, n) for n in dir(ontology)
                     if re.fullmatch(r"sync_\w*links_after_reindex", n)), None)
        has_links = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name LIKE '%\\_links' ESCAPE '\\'").fetchone()
        if sync and has_links:
            sync(conn)
        ontology.compile_ontology(conn)
    except Exception as e:
        print(f"  WARN: ontology compilation after {phase} failed: {e}", flush=True)


def build_index(tags: bool = False, config: dict | None = None):
    """Full rebuild of the file-derived chunks from the sources config.

    Rows written by side-car tools (chat imports, clipped conversations) are kept.
    Embeddings are saved incrementally after each batch, so an interrupted run keeps its
    progress; chunks whose exact text is already embedded are reused, not re-embedded.
    """
    print("Building memory index...", flush=True)
    start = time.time()
    config = config or load_sources_config()

    client = get_indexing_client()
    conn = init_db(MEMORY_DB)
    _assert_index_model(conn)
    _reset_run_embed_budget()
    _backfill_embed_hashes(conn)

    sources = collect_sources(config)
    print(f"Found {len(sources)} sources to index", flush=True)

    all_chunks: list[dict] = []
    for src in sources:
        built = _build_new_chunks_for_source(src)
        if built:
            all_chunks.extend(built)
    git_meta = collect_git_metadata(config)
    all_chunks.extend(_git_meta_chunk(gm) for gm in git_meta)
    print(f"Created {len(all_chunks)} chunks from {len(sources)} sources + {len(git_meta)} git metadata", flush=True)

    all_chunks = _dedupe_chunk_ids(all_chunks)

    # Corpus-wide reuse is resolved BEFORE the destructive DELETE below, so it sees the table
    # as it stood before the rebuild.
    reuse_hits = _resolve_corpus_reuse(conn, all_chunks)

    keep = ",".join("?" * len(SIDECAR_STORE_TYPES))
    conn.execute(f"DELETE FROM chunks WHERE store_type NOT IN ({keep})", SIDECAR_STORE_TYPES)
    conn.execute("DELETE FROM source_meta")
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")

    print(f"Embedding {len(all_chunks)} chunks (batch size {BATCH_SIZE}, "
          f"{reuse_hits} reused via corpus-wide hash match)...", flush=True)
    embed_start = time.time()
    now = now_utc()
    cols = _chunk_columns(conn)
    state = {"exhausted": False, "failed": 0}
    done = 0
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        _embed_pending(client, batch, state)
        for chunk in batch:
            _insert_chunk(conn, chunk, now, cols)
        conn.commit()  # incremental save after each batch
        done += len(batch)
        print(f"  {done}/{len(all_chunks)} chunks saved", flush=True)
    print(f"Embedding took {time.time() - embed_start:.1f}s ({state['failed']} failures, "
          f"{reuse_hits} reused via corpus-wide hash match)", flush=True)

    # Belt and braces: the earlier 'rebuild' ran before the inserts, so it cannot heal drift
    # introduced during them. This one can.
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    conn.commit()

    if tags:
        tag_client = client if client is not None else get_tag_client()
        if tag_client is not None:
            print(f"Auto-tagging {len(all_chunks)} chunks...", flush=True)
            tag_start = time.time()
            tag_map = generate_tags_batch(tag_client, all_chunks)
            for idx, tag_str in tag_map.items():
                if idx < len(all_chunks):
                    conn.execute("UPDATE chunks SET tags = ? WHERE chunk_id = ?", (tag_str, all_chunks[idx]["chunk_id"]))
            conn.commit()
            print(f"Tagged {len(tag_map)}/{len(all_chunks)} chunks in {time.time() - tag_start:.1f}s", flush=True)

    for src in sources:  # source_meta drives incremental updates
        p = Path(src["path"])
        if p.exists():
            conn.execute("INSERT OR REPLACE INTO source_meta (source_path, mtime, store_type, project) "
                         "VALUES (?, ?, ?, ?)", (src["path"], p.stat().st_mtime, src["store_type"], src["project"]))

    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    _set_meta(conn, last_built=now, last_updated=datetime.now(timezone.utc).isoformat(), total_chunks=total,
              total_sources=len(sources), embedding_model=EMBEDDING_MODEL, embedding_dim=EMBEDDING_DIM,
              embed_backend=EMBED_BACKEND)
    conn.commit()
    _run_ontology(conn, "build")

    print(f"\nIndex built: {len(all_chunks)} chunks from {len(sources)} sources in {time.time() - start:.1f}s", flush=True)
    print(f"Database: {MEMORY_DB} ({MEMORY_DB.stat().st_size / 1024 / 1024:.1f} MB)", flush=True)
    conn.close()


def update_index(dry_run: bool = False, config: dict | None = None):
    """Incremental update: re-chunk only new / modified sources, drop removed ones.

    Content-aware: each changed file's chunks are diffed against the stored rows, unchanged
    chunks are left untouched, and only genuinely new text is embedded. Everything is planned
    and embedded first, then applied in ONE transaction.
    """
    if not MEMORY_DB.exists():
        if dry_run:
            print("No existing index; a full build would run.")
            return
        print("No existing index. Running full build instead.")
        build_index(config=config)
        return

    print("Incremental index update...", flush=True)
    start = time.time()
    config = config or load_sources_config()

    client = get_indexing_client()
    conn = init_db(MEMORY_DB)
    _assert_index_model(conn)
    _reset_run_embed_budget()
    _backfill_embed_hashes(conn)

    sources = collect_sources(config)
    source_paths = {s["path"] for s in sources}
    existing = {row[0]: row[1] for row in conn.execute("SELECT source_path, mtime FROM source_meta")}

    to_add, to_update, to_remove = [], [], []
    for src in sources:
        path = Path(src["path"])
        if not path.exists():
            continue
        mtime = path.stat().st_mtime
        if src["path"] not in existing:
            to_add.append((src, mtime))
        elif mtime > existing[src["path"]]:
            to_update.append((src, mtime))
    for old_path in existing:
        if old_path not in source_paths:
            to_remove.append(old_path)

    git_meta = collect_git_metadata(config)
    if len(to_add) + len(to_update) + len(to_remove) == 0 and not git_meta:
        print("No changes detected. Index is current.")
        conn.close()
        backfill_embeddings(_reset_budget=False)  # still heal side-car rows lacking a vector
        return
    print(f"Changes: {len(to_add)} new, {len(to_update)} modified, {len(to_remove)} removed sources", flush=True)

    plan = []                 # [{path, src, mtime, changed, new, stale_ids, kept, skipped}]
    chunks_to_embed = []      # changed + new chunks with no reuse hit
    reuse_hits = 0
    for src, mtime in to_add + to_update:
        path = Path(src["path"])
        new_chunks = _build_new_chunks_for_source(src)
        if new_chunks is None:
            # Unreadable / bad JSON: leave existing rows alone, but still bump source_meta so the
            # file is not reclassified as changed on every run.
            plan.append({"path": str(path), "src": src, "mtime": mtime, "changed": [], "new": [],
                         "stale_ids": [], "kept": 0, "skipped": True})
            continue
        new_chunks = _dedupe_chunk_ids(new_chunks)

        existing_content: dict[str, str] = {}
        reuse: dict[str, bytes] = {}
        for cid, content, title, proj, stype, emb in conn.execute(
                "SELECT chunk_id, content, title, project, store_type, embedding FROM chunks WHERE source_path = ?",
                (str(path),)).fetchall():
            existing_content[cid] = content
            if emb is not None and content is not None and len(emb) == EMBEDDING_DIM * 4:
                key = _md5(_embed_text({"project": proj, "store_type": stype, "title": title, "content": content}))
                reuse.setdefault(key, emb)

        new_ids = set()
        changed, new = [], []
        kept = 0
        for c in new_chunks:
            new_ids.add(c["chunk_id"])
            if c["chunk_id"] in existing_content and existing_content[c["chunk_id"]] == c["content"]:
                kept += 1
                continue  # identical chunk: leave the row completely untouched
            key = _embed_hash(c)
            c["_embed_hash"] = key
            if key in reuse:
                c["_embedding_blob"] = reuse[key]
                reuse_hits += 1
            else:
                c["_embedding_blob"] = None
                chunks_to_embed.append(c)
            (changed if c["chunk_id"] in existing_content else new).append(c)
        plan.append({"path": str(path), "src": src, "mtime": mtime, "changed": changed, "new": new,
                     "stale_ids": [cid for cid in existing_content if cid not in new_ids],
                     "kept": kept, "skipped": False})

    # Corpus-wide reuse: old content that merely moved to a new path costs nothing.
    corpus_reuse_hits = _resolve_corpus_reuse(conn, chunks_to_embed)
    needing_api = [c for c in chunks_to_embed if c.get("_embedding_blob") is None]

    total_kept = sum(p["kept"] for p in plan)
    total_changed = sum(len(p["changed"]) for p in plan)
    total_new = sum(len(p["new"]) for p in plan)
    total_stale = sum(len(p["stale_ids"]) for p in plan)

    if dry_run:
        print("\n=== DRY RUN (no writes, no embedding API calls) ===", flush=True)
        for p in plan:
            if p["skipped"]:
                print(f"  SKIP   {Path(p['path']).name} (unreadable)")
                continue
            nc, nn, nd = len(p["changed"]), len(p["new"]), len(p["stale_ids"])
            if nc or nn or nd:
                print(f"  {Path(p['path']).name}: kept={p['kept']} changed={nc} new={nn} deleted={nd}")
        for old_path in to_remove:
            print(f"  REMOVE {old_path} (source gone)")
        print(f"\nTotals: kept={total_kept} changed={total_changed} new={total_new} deleted={total_stale} "
              f"removed_sources={len(to_remove)}; embeddings needed={len(needing_api)} "
              f"(same-path reuse={reuse_hits}, corpus-wide reuse={corpus_reuse_hits})", flush=True)
        conn.close()
        return

    state = {"exhausted": False, "failed": 0}
    if needing_api:
        print(f"Embedding {len(needing_api)} changed/new chunks (reused same-path={reuse_hits} "
              f"corpus-wide={corpus_reuse_hits}, skipped {total_kept} unchanged)...", flush=True)
        embed_start = time.time()
        done = 0
        for i in range(0, len(needing_api), BATCH_SIZE):
            batch = needing_api[i:i + BATCH_SIZE]
            _embed_pending(client, batch, state)
            done += len(batch)
            print(f"  {done}/{len(needing_api)} embedded ({time.time() - embed_start:.0f}s)", flush=True)

    now = now_utc()
    landed_without_embedding = 0
    cols = _chunk_columns(conn)
    try:
        for old_path in to_remove:
            conn.execute("DELETE FROM chunks WHERE source_path = ?", (old_path,))
            conn.execute("DELETE FROM source_meta WHERE source_path = ?", (old_path,))
        for p in plan:
            for cid in p["stale_ids"]:
                conn.execute("DELETE FROM chunks WHERE chunk_id = ?", (cid,))
            # Changed: DELETE + INSERT (there is no AFTER UPDATE trigger, so an in-place UPDATE
            # would desync FTS5). New content always lands, even with a NULL embedding: it is
            # FTS-searchable at once and `backfill` fills the vector later. Keeping the old row
            # instead would strand the new text forever, since source_meta advances below.
            for c in p["changed"]:
                landed_without_embedding += c.get("_embedding_blob") is None
                conn.execute("DELETE FROM chunks WHERE chunk_id = ?", (c["chunk_id"],))
                _insert_chunk(conn, c, now, cols)
            for c in p["new"]:
                landed_without_embedding += c.get("_embedding_blob") is None
                _insert_chunk(conn, c, now, cols)
        for p in plan:  # bump every processed source, incl. no-op touches
            s = p["src"]
            conn.execute("INSERT OR REPLACE INTO source_meta (source_path, mtime, store_type, project) "
                         "VALUES (?, ?, ?, ?)", (s["path"], p["mtime"], s["store_type"], s["project"]))
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    print(f"Content-aware apply: kept={total_kept} changed={total_changed} new={total_new} deleted={total_stale} "
          f"removed_sources={len(to_remove)} landed_without_embedding={landed_without_embedding}", flush=True)

    for gm in git_meta:  # cheap refresh; re-embedded only when the content changed
        c = _git_meta_chunk(gm)
        row = conn.execute("SELECT content FROM chunks WHERE chunk_id = ?", (c["chunk_id"],)).fetchone()
        if row and row[0] == c["content"]:
            continue
        c["_embed_hash"] = _embed_hash(c)
        _embed_pending(client, [c], state)
        conn.execute("DELETE FROM chunks WHERE chunk_id = ?", (c["chunk_id"],))
        _insert_chunk(conn, c, now, cols)
    conn.commit()

    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    source_count = conn.execute("SELECT COUNT(*) FROM source_meta").fetchone()[0]
    # Explicit UTC: consumers report this as the index date.
    _set_meta(conn, last_updated=datetime.now(timezone.utc).isoformat(), total_chunks=total,
              total_sources=source_count, embedding_model=EMBEDDING_MODEL, embedding_dim=EMBEDDING_DIM,
              embed_backend=EMBED_BACKEND)
    conn.commit()
    _run_ontology(conn, "update")

    print(f"\nIncremental update: {total_changed + total_new} chunks (re-)written in {time.time() - start:.1f}s", flush=True)
    print(f"Index now has {total} chunks from {source_count} sources", flush=True)
    conn.close()

    # Heal chunks that landed (or were inserted by a side-car tool) without a vector. Same
    # run, so the per-run budget is NOT reset. Short-circuits when nothing is pending.
    backfill_embeddings(_reset_budget=False)


def backfill_embeddings(_reset_budget: bool = True):
    """Embed chunks that still lack a vector (an earlier run stopped on its budget, the API
    failed, or a side-car tool inserted rows without embedding). Respects the spend budget."""
    if not MEMORY_DB.exists():
        print("No index. Run build first.")
        return

    client = get_indexing_client()
    conn = init_db(MEMORY_DB, label="index.py backfill_embeddings")
    _assert_index_model(conn)
    _backfill_embed_hashes(conn)
    if _reset_budget:
        _reset_run_embed_budget(budget_env="ANAMNESIS_BACKFILL_BUDGET_S")

    missing = conn.execute("SELECT id, chunk_id, title, content, store_type, project, embed_hash "
                           "FROM chunks WHERE embedding IS NULL").fetchall()
    if not missing:
        print("All chunks have embeddings. Nothing to backfill.")
        conn.close()
        return

    pending = [{"id": r[0], "chunk_id": r[1], "title": r[2], "content": r[3], "store_type": r[4],
                "project": r[5], "_embed_hash": r[6]} for r in missing]
    for c in pending:
        if not c["_embed_hash"]:
            c["_embed_hash"] = _embed_hash(c)
    reuse_hits = _resolve_corpus_reuse(conn, pending)
    if reuse_hits:
        for c in pending:
            if c.get("_embedding_blob") is not None:
                conn.execute("UPDATE chunks SET embedding = ?, embed_hash = ? WHERE id = ?",
                             (c["_embedding_blob"], c["_embed_hash"], c["id"]))
        conn.commit()
        print(f"  corpus-wide reuse: {reuse_hits} chunk(s) filled with no API call", flush=True)

    still_missing = [c for c in pending if c.get("_embedding_blob") is None]
    if not still_missing:
        print("All chunks have embeddings after reuse. Nothing more to backfill.")
        conn.close()
        return

    print(f"Backfilling {len(still_missing)} chunks missing embeddings...", flush=True)
    start = time.time()
    state = {"exhausted": False, "failed": 0}
    filled = 0
    for i in range(0, len(still_missing), BATCH_SIZE):
        batch = still_missing[i:i + BATCH_SIZE]
        _embed_pending(client, batch, state)
        if state["exhausted"]:
            break
        for c in batch:
            if c.get("_embedding_blob") is not None:
                conn.execute("UPDATE chunks SET embedding = ?, embed_hash = ? WHERE id = ?",
                             (c["_embedding_blob"], c["_embed_hash"], c["id"]))
                filled += 1
        conn.commit()
        print(f"  {filled}/{len(still_missing)} backfilled ({filled / max(time.time() - start, 1e-6):.1f} chunks/s)",
              flush=True)

    remaining = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL").fetchone()[0]
    print(f"\nBackfill done: {filled} filled in {time.time() - start:.1f}s ({reuse_hits} via corpus-wide reuse). "
          f"Remaining without embeddings: {remaining}", flush=True)
    conn.close()


def import_chat_exports(path, project: str | None = None):
    """Import chat exports from ChatGPT, Claude or Gemini (user messages only) as `chat_export` chunks."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from chat_import import auto_detect_and_parse, conversations_to_chunks, import_summary

    print(f"Importing chat exports from: {path}", flush=True)
    conversations = auto_detect_and_parse(path)
    if not conversations:
        print("No conversations found.")
        return

    print(f"\n{import_summary(conversations)}", flush=True)
    chunks = conversations_to_chunks(conversations, project=project or "chat")
    chunks = _dedupe_chunk_ids(chunks)
    print(f"Chunking: {len(chunks)} chunks from {len(conversations)} conversations", flush=True)

    client = get_indexing_client()
    conn = init_db(MEMORY_DB)
    _assert_index_model(conn)
    now = now_utc()
    _reset_run_embed_budget()
    # Reuse BEFORE deleting: a re-import of an unchanged export must reuse the old vectors.
    reuse_hits = _resolve_corpus_reuse(conn, chunks)
    old_count = conn.execute("SELECT COUNT(*) FROM chunks WHERE store_type = 'chat_export'").fetchone()[0]
    if old_count:
        conn.execute("DELETE FROM chunks WHERE store_type = 'chat_export'")
        conn.commit()
        print(f"Removed {old_count} existing chat_export chunks", flush=True)

    cols = _chunk_columns(conn)
    state = {"exhausted": False, "failed": 0}
    saved = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        _embed_pending(client, batch, state)
        for chunk in batch:
            _insert_chunk(conn, chunk, now, cols)
        conn.commit()
        saved += len(batch)
        print(f"  {saved}/{len(chunks)} chunks saved ({reuse_hits} reused via corpus-wide hash match)", flush=True)

    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    _set_meta(conn, last_updated=datetime.now(timezone.utc).isoformat(), total_chunks=total)
    conn.commit()
    conn.close()
    print(f"\nImport complete: {saved} chunks ({state['failed']} embedding failures)", flush=True)
    print(f"Database: {MEMORY_DB}", flush=True)


# --------------------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------------------
def _project_aliases(conn, config_aliases: dict, shared: list[str]) -> dict:
    """alias -> project: the config's aliases plus each project name found in the index."""
    aliases = {k.lower(): v for k, v in config_aliases.items()}
    try:
        for (p,) in conn.execute("SELECT DISTINCT project FROM chunks WHERE project IS NOT NULL"):
            if p and len(p) >= 3 and p != DEFAULT_PROJECT and p not in shared:
                aliases.setdefault(p.lower(), p)
    except sqlite3.Error:
        pass
    return aliases


def _project_boost(query: str, chunk_project: str, chunk_content: str, aliases: dict | None = None,
                   shared: list[str] | None = None) -> float:
    """Boost chunks of the project the query names; penalise other projects; shared projects
    are cross-cutting (no penalty). Aliases match on word boundaries."""
    if not aliases:
        return 1.0
    q_lower = query.lower()
    matched = None
    for alias, project in aliases.items():
        if re.search(rf"(?<![\w-]){re.escape(alias)}(?![\w-])", q_lower):
            matched = project
            break
    if matched is None:
        return 1.0
    if chunk_project == matched:
        return 2.0
    if chunk_project in (shared or []):
        return 1.15 if matched.lower() in chunk_content.lower() else 1.0
    return 0.5


_STOPWORDS = {"what", "when", "where", "which", "that", "this", "with", "from", "have", "does", "will",
              "about", "after", "before", "during", "between", "into", "through", "than", "then", "them",
              "they", "their", "there", "these", "those", "been", "being", "were", "would", "could",
              "should", "must", "also", "more", "most", "some", "many", "much", "very", "only", "just",
              "your", "know"}


def _keyword_overlap_boost(query: str, chunk_content: str) -> float:
    """Boost chunks sharing significant query terms (4+ chars, not stopwords): 1.0x at 0%
    overlap up to 1.3x at 100%."""
    terms = [w.lower().strip("?.,!") for w in query.split()
             if len(w) >= 4 and w.lower().strip("?.,!") not in _STOPWORDS]
    if not terms:
        return 1.0
    content_lower = chunk_content.lower()
    return 1.0 + (sum(1 for t in terms if t in content_lower) / len(terms)) * 0.3


def search(query: str, limit: int = 5, quiet: bool = False):
    """Two-stage search: FTS5 keyword candidates, then embedding rerank (FTS-only when
    embeddings are unavailable or the keyword match is already decisive)."""
    if not MEMORY_DB.exists():
        print(f"ERROR: Index not built at {MEMORY_DB}. Run: python tools/index.py build")
        sys.exit(1)

    def out(*a):
        if not quiet:
            print(*a)

    start = time.time()
    conn = sqlite3.connect(str(MEMORY_DB))
    conn.row_factory = sqlite3.Row

    # Quote each term so FTS5 never parses user text as syntax (columns, NEAR, quotes).
    words = re.findall(r"\w+", query)
    terms = [w for w in words if len(w) > 2] or words
    if not terms:
        conn.close()
        return []
    fts_query = " OR ".join('"' + w.replace('"', '""') + '"' for w in terms)
    candidates = conn.execute("""
        SELECT c.*, rank
        FROM chunks_fts fts
        JOIN chunks c ON c.id = fts.rowid
        WHERE chunks_fts MATCH ?
        ORDER BY rank
        LIMIT 150
    """, (fts_query,)).fetchall()
    fts_time = time.time() - start

    if not candidates:
        out(f"No results for: {query}")
        conn.close()
        return []

    # BM25 rank is negative; more negative = better. A strong, well-separated top hit skips
    # the embedding call.
    top_rank = abs(candidates[0]["rank"]) or 1e-9
    fifth_rank = abs(candidates[min(4, len(candidates) - 1)]["rank"]) if len(candidates) > 1 else 0
    fts_confident = top_rank > 20 and (top_rank - fifth_rank) > 3

    query_emb = None
    if not fts_confident and EMBED_BACKEND != "mock":  # mock vectors carry no semantics: rank by BM25
        try:
            query_emb = embed_query(get_embed_client(), query)
        except Exception as e:  # no key / SDK / quota / cache: FTS-only, immediately
            print(f"  embedding unavailable ({type(e).__name__}: {str(e)[:80]}); FTS-only", file=sys.stderr)

    try:
        cfg = load_sources_config()
        aliases = _project_aliases(conn, cfg["project_aliases"], cfg["shared_projects"])
        shared = cfg["shared_projects"]
    except Exception:
        aliases, shared = _project_aliases(conn, {}, ["shared"]), ["shared"]

    priority_mult_by = {"invariant": 1.5, "important": 1.0, "helpful": 0.8}
    keys = candidates[0].keys()
    results = []
    for row in candidates:
        if "retirement_status" in keys and row["retirement_status"] == "retired":
            continue
        if query_emb is not None:
            blob = row["embedding"]
            similarity = cosine_similarity(query_emb, unpack_embedding(blob)) if blob else 0.3
        else:
            similarity = abs(row["rank"]) / top_rank  # BM25 is the relevance signal
        priority = row["priority"] if "priority" in keys and row["priority"] else "important"
        priority_mult = priority_mult_by.get(priority, 1.0)
        fresh = freshness_score(row["modified_at"])
        if query_emb is not None:
            score = (similarity * row["weight"] * fresh * _project_boost(query, row["project"], row["content"],
                                                                         aliases, shared)
                     * _keyword_overlap_boost(query, row["content"]) * priority_mult)
        else:
            score = similarity * fresh * priority_mult
        results.append({
            "chunk_id": row["chunk_id"], "title": row["title"], "content": row["content"][:500],
            "source_path": row["source_path"], "store_type": row["store_type"], "project": row["project"],
            "score": score, "similarity": similarity, "freshness": fresh, "weight": row["weight"],
            "priority": priority, "live_source": row["live_source"] if "live_source" in keys else "",
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    seen_sources: set[str] = set()
    deduped = []
    for r in results:  # keep the highest-scoring chunk per source
        if r["source_path"] not in seen_sources:
            seen_sources.add(r["source_path"])
            deduped.append(r)
    results = deduped

    total_time = time.time() - start
    rerank_mode = "embedding" if query_emb is not None else "FTS-only"
    header = (f"\n## /recall results for: \"{query}\"\n"
              f"FTS: {len(candidates)} candidates in {fts_time * 1000:.0f}ms | Rerank: {rerank_mode} | "
              f"Total: {total_time * 1000:.0f}ms\n")

    # Epistemic-gap suppression: a weak best match means "nothing relevant in memory".
    EPISTEMIC_GAP_THRESHOLD = 0.35
    if results and results[0]["similarity"] < EPISTEMIC_GAP_THRESHOLD:
        out(header)
        out("WARNING: EPISTEMIC GAP - no relevant memory found for this query.")
        out(f"Best match similarity: {results[0]['similarity']:.2f} (threshold: {EPISTEMIC_GAP_THRESHOLD})")
        out("This may be a known unknown or a topic not yet captured in memory.\n")
        for r in results[:limit]:
            r["epistemic_gap"] = True
        conn.close()
        return results[:limit]

    out(header)
    if len(results) >= 2 and results[0]["score"] - results[1]["score"] > 0.25:  # clear winner: drop noise
        cutoff = results[0]["score"] - 0.3
        results = [r for r in results if r["score"] >= cutoff]

    for i, r in enumerate(results[:limit]):
        confidence = min(r["score"] / 1.5, 1.0)
        out(f"### Result {i + 1} (confidence: {confidence:.2f}) {'CAUTION: low confidence' if confidence < 0.4 else ''}")
        out(f"**Claim**: {r['title']}")
        out(f"**Evidence**: {r['source_path']}")
        out(f"**Type**: {r['store_type']} | **Project**: {r['project']}")
        out(f"**Score breakdown**: similarity={r['similarity']:.2f} x weight={r['weight']}x x freshness={r['freshness']:.1f}")
        if r.get("live_source"):
            out(f"**Live source**: {r['live_source']} - this memory is a snapshot, not the current value")
        out(f"```\n{r['content'][:300]}...\n```\n")

    conn.close()
    return results[:limit]


def synthesize(query: str, limit: int = 8) -> str:
    """Search + LLM synthesis: a coherent answer with inline [source: path] citations."""
    from google.genai import types

    results = search(query, limit=limit)
    if not results:
        return f"No relevant memories found for: {query}"

    conn = sqlite3.connect(str(MEMORY_DB))
    conn.row_factory = sqlite3.Row
    context_parts = []
    for r in results:
        row = conn.execute("SELECT content FROM chunks WHERE chunk_id = ?", (r["chunk_id"],)).fetchone()
        context_parts.append(f"[source: {r['source_path']}]\n{(row['content'] if row else r['content'])[:1500]}")
    conn.close()

    system_prompt = ("Synthesize these knowledge sources into a direct, coherent answer. "
                     "Cite each claim with [source: filepath]. If sources conflict, note the conflict.")
    user_prompt = f"Question: {query}\n\nKnowledge sources:\n\n" + "\n\n---\n\n".join(context_parts)
    try:
        response = get_gemini_client().models.generate_content(
            model=GENERATION_MODEL, contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.3,
                                               max_output_tokens=2048))
        answer = response.text or "Synthesis failed: empty response"
    except Exception as e:
        answer = f"Synthesis failed: {e}"

    print(f"\n## Synthesized answer for: \"{query}\"\n")
    print(answer)
    return answer


def show_stats():
    """Print index statistics."""
    if not MEMORY_DB.exists():
        print("No index built yet.")
        return
    conn = sqlite3.connect(str(MEMORY_DB))
    conn.row_factory = sqlite3.Row
    meta = {row["key"]: row["value"] for row in conn.execute("SELECT * FROM index_meta").fetchall()}
    print("Memory Index Stats\n==================")
    print(f"Database: {MEMORY_DB} ({MEMORY_DB.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"Last built: {meta.get('last_built', 'never')}")
    print(f"Last updated: {meta.get('last_updated', 'never')}")
    print(f"Total chunks: {meta.get('total_chunks', 0)}")
    print(f"Total sources: {meta.get('total_sources', 0)}")
    print(f"Embedding model: {meta.get('embedding_model', 'unknown')} ({meta.get('embedding_dim', 0)}d)")
    print("\nBy store type:")
    for row in conn.execute("SELECT store_type, COUNT(*) AS cnt FROM chunks GROUP BY store_type ORDER BY cnt DESC"):
        print(f"  {row[0]}: {row[1]}")
    print("\nBy project:")
    for row in conn.execute("SELECT project, COUNT(*) AS cnt FROM chunks GROUP BY project ORDER BY cnt DESC"):
        print(f"  {row[0]}: {row[1]}")
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    with_emb = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0]
    print(f"\nEmbedding coverage: {with_emb}/{total} ({(with_emb / total * 100) if total else 0:.0f}%)")
    conn.close()


def show_sources(config: dict | None = None):
    """Print the resolved sources (no writes): what the next build would index."""
    config = config or load_sources_config()
    print(f"Sources config: {SOURCES_FILE or '(built-in: ' + str(DATA_DIR) + ')'}")
    for s in config["sources"]:
        extra = f" root={s['root']}" if s["kind"] == "files" else ""
        print(f"  [{s['kind']}] {s['name']} enabled={s['enabled']} store_type={s['store_type']} "
              f"project={s['project']}{extra}")
    files = collect_sources(config)
    by_type: dict[str, int] = {}
    for f in files:
        by_type[f["store_type"]] = by_type.get(f["store_type"], 0) + 1
    print(f"\n{len(files)} file(s): " + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())))


def _audit_direct(tool: str, query: str, n_results: int, latency_ms: float):
    """Opt-in (ANAMNESIS_AUDIT=1) JSONL audit of direct CLI calls under $ANAMNESIS_HOME/logs.
    Off by default because it records the (truncated) query text. Never fails a search."""
    if not _truthy(os.environ.get("ANAMNESIS_AUDIT")):
        return
    try:
        log_dir = Path(os.environ.get("ANAMNESIS_LOG_DIR", str(ANAMNESIS_HOME / "logs")))
        log_dir.mkdir(parents=True, exist_ok=True)
        entry = {"ts": now_utc(), "tool": tool, "query": query[:100], "results": n_results,
                 "latency_ms": round(latency_ms, 1), "method": "direct"}
        with open(log_dir / "audit.log", "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="index.py", description="anamnesis indexer")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("build", help="full rebuild from the sources config")
    p.add_argument("--tags", action="store_true", default=_truthy(os.environ.get("ANAMNESIS_AUTOTAG")),
                   help="generate LLM tags per chunk (needs GEMINI_API_KEY; off by default)")
    p.add_argument("--skip-tags", action="store_true", help=argparse.SUPPRESS)
    p = sub.add_parser("update", help="incremental update (builds when no database exists)")
    p.add_argument("--dry-run", action="store_true")
    sub.add_parser("backfill", help="embed chunks that lack a vector")
    for name in ("search", "synthesize"):
        p = sub.add_parser(name)
        p.add_argument("query", nargs="+")
    sub.add_parser("stats")
    sub.add_parser("sources", help="show resolved sources (no writes)")
    p = sub.add_parser("import-chats", help="import ChatGPT / Claude / Gemini exports")
    p.add_argument("path")
    p.add_argument("--project", default=None)
    args = ap.parse_args(argv)
    cmd = args.cmd or "update"
    dry = getattr(args, "dry_run", False)

    try:
        if cmd == "build":
            build_index(tags=bool(args.tags and not args.skip_tags))
        elif cmd == "update":
            update_index(dry_run=dry)
        elif cmd == "backfill":
            backfill_embeddings()
        elif cmd in ("search", "synthesize"):
            q = " ".join(args.query)
            t0 = time.time()
            if cmd == "search":
                _audit_direct("recall", q, len(search(q) or []), (time.time() - t0) * 1000)
            else:
                synthesize(q)
                _audit_direct("synthesize", q, 0, (time.time() - t0) * 1000)
        elif cmd == "import-chats":
            import_chat_exports(args.path, project=args.project)
        elif cmd == "stats":
            show_stats()
        elif cmd == "sources":
            show_sources()
    except ConfigError as e:
        print(f"sources config error: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:  # e.g. missing GEMINI_API_KEY for the gemini backend
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
