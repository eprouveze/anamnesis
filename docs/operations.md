# Operations

## Build the index

CI does it on push to `data/**` and nightly. By hand:
```bash
gh workflow run index --repo <you>/anamnesis
# or locally (the identical script the runner uses):
python tools/index.py build        # full rebuild → $ANAMNESIS_DB
python tools/index.py              # incremental update (builds if no DB)
python tools/index.py sources      # show resolved sources, write nothing
python tools/index.py stats
python tools/index.py import-chats ./export.json --project my-app
```

Key environment (full list in each module's docstring):

| Var | Default | Purpose |
|---|---|---|
| `ANAMNESIS_HOME` | `~/.anamnesis` | state: db, clients, caches, logs |
| `ANAMNESIS_DATA_DIR` | `<repo>/data` | git-tracked memory |
| `ANAMNESIS_DB` | `$ANAMNESIS_HOME/memory.db` | the index |
| `ANAMNESIS_SOURCES` | built-in `data/` roots | YAML sources config (see `anamnesis.sources.example.yaml`) |
| `ANAMNESIS_EMBED_BACKEND` | `gemini` | `gemini` · `local` (sentence-transformers) · `mock` (CI/tests) |
| `ANAMNESIS_EMBED_MONTHLY_TOKEN_CAP` / `_RUN_TOKEN_CAP` | 160M / 40M bytes | embedding spend caps |
| `ANAMNESIS_ALERT_WEBHOOK` | — | JSON POST on cap/partial-embed alerts |
| `GEMINI_API_KEY`, `GEMINI_API_KEY_BACKUP` | — | environment only |

## Query

```bash
python tools/recall.py "your question"
python tools/index.py synthesize "what did we decide about X?"
```

Query embeddings are cached in `$ANAMNESIS_HOME/query-embeddings.db`, keyed by model. If
the embedding call fails (no key, quota, network), recall returns keyword matches
unranked rather than failing.

## Serve over MCP

```bash
python tools/mcp_server.py --transport stdio         # local agent, trusted
python tools/mcp_server.py                           # HTTP on 127.0.0.1:7777/mcp
python tools/mcp_server.py add-client <name> [--read-only]
python tools/mcp_server.py list-clients
python tools/mcp_server.py revoke-client <name>      # takes effect without restart
```

Put TLS (reverse proxy or tunnel) in front and set `ANAMNESIS_ALLOWED_HOSTS` to the public
hostname. For GUI connectors that cannot send headers, enable OAuth rather than
query-string tokens:

```bash
export ANAMNESIS_OAUTH_PROVIDER=google
export ANAMNESIS_OAUTH_CLIENT_ID=... ANAMNESIS_OAUTH_CLIENT_SECRET=...
export ANAMNESIS_OAUTH_BASE_URL=https://memory.example.com
export ANAMNESIS_OAUTH_ALLOWED_EMAILS=you@example.com    # required; enforced on every call
```

A replica (`ANAMNESIS_ROLE=replica`) serves reads and refuses `store_*`.

## Maintenance (propose-only by default)

```bash
python tools/dream.py            # dry-run report; --propose queues, --apply-approved applies only approved (with undo)
python tools/decay.py            # value-decay proposals for curated facts
python tools/eval_recall.py      # recall quality vs. a golden set
```

## Test

```bash
python -m unittest discover -s tools -p 'test_*.py'    # no API key, no network
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
- A replica never serves an unverified artifact (`pull-and-serve.sh` fails closed; `ANAMNESIS_ALLOW_UNSIGNED=1` is an explicit, logged override).
- The pipeline is protected (CODEOWNERS) — it holds the embedding key's blast radius.
