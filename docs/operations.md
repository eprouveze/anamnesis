# Operations

## Build the index

CI does it on push to `data/**` and nightly. By hand:
```bash
gh workflow run index --repo <you>/anamnesis
# or locally (the identical script the runner uses):
GEMINI_API_KEY=... python tools/index.py       # → memory.db
```

## Query

```bash
GEMINI_API_KEY=... python tools/recall.py "your question"
```

Query embeddings are cached in `memory.db.queries.db` next to `memory.db` (override with
`ANAMNESIS_QUERY_CACHE`); a repeated question costs no API call. Rows are keyed by model,
so changing `ANAMNESIS_EMBEDDING_MODEL` never serves a stale vector; old rows are tiny and
are left in place. If the embedding call fails (no key, quota, network), recall prints a
warning to stderr and returns the keyword matches unranked (`score: null`, shown as `[kw]`
by the CLI) rather than failing. Delete the file to reset.

## Test

```bash
python -m unittest discover -s tools -p 'test_*.py'    # no API key needed
```

## Verify an artifact before trusting it

```bash
gh release download --repo <you>/anamnesis -p memory.db
python3 - <<'PY'
import sqlite3, os
db="memory.db"; c=sqlite3.connect(db)
n=c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
e=c.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0]
assert n>0 and e==n and os.path.getsize(db)>20_000
print(f"OK: chunks={n} embedded={e}")
PY
```

## Add a memory

The commit *is* the write:
```bash
cat > data/memory/reference-my-note.md <<'EOF'
---
name: my note
description: one-line summary used for recall relevance
type: reference
---
The fact you want the agent to remember.
EOF
git add data/memory/reference-my-note.md && git commit -m "memory: my note" && git push
```

## Failure modes → response

| Symptom | Response |
|---|---|
| `index` red at **integrity floor** | Build produced an empty/partly-embedded DB — not published. Check the embed step (key/quota). Last good Release keeps serving. |
| Embed step red | Usually `GEMINI_API_KEY` quota/expiry. Fix the secret, re-run. Fails closed. |
| Replica stale | Check its pull timer + compare served build vs latest Release. Add a freshness alarm. |
| GitHub down | Writes queue as local commits; reads keep serving the last artifact. Nothing lost. |
| Everything down | `git clone` anywhere + `tools/anamnesis-rebuild.sh` → fresh `memory.db`. RPO = last push. |

## Guardrails

- One writer per dataset, enforced by git ordering — never add a write path that bypasses it.
- A replica never serves an unverified artifact.
- The pipeline is protected (CODEOWNERS) — it holds the embedding key's blast radius.
