# anamnesis

**Persistent memory for an AI agent, built so no single machine is required to write or
refresh it.** The *memory* pillar of
[Stewardship Engineering](https://github.com/eprouveze/stewardship-engineering) —
alongside [heartbeat](https://github.com/eprouveze/heartbeat) (the loop) and
[rightmodel](https://github.com/eprouveze/rightmodel) (routing & trust).

An agent that operates as a custodian of your work over months needs memory that is
durable, resilient, and yours. This is the genericized core of one such system.

> **The DB is a build artifact. Git is the write-ahead log. The writer is a pipeline.**

Memory entries are **git commits** (markdown/JSON under `data/`). A **GitHub Actions
pipeline** indexes and embeds them into a `memory.db` (SQLite FTS5 + vector rerank),
cryptographically signs the build artifact with **keyless Sigstore/Cosign**, and
publishes it as a **Release artifact**. Every machine is a disposable replica that pulls
the artifact, verifies its provenance signature, and serves it. Lose any machine — or all of
them — and the write path still works (git commits queue offline) and the index rebuilds anywhere.

> These are working tools with rough edges, not a polished product. The author runs a
> heavily-extended, private instance of this daily; this repo is the genericized core you
> can build on.

## Why it's shaped this way

Most "agent memory" designs quietly depend on one machine or one hosted database being up.
When that primary dies, writing and refreshing memory stops. The trick here: **if your
sources already live in git, the database is a *derived* artifact** — rebuild it anywhere
from the sources plus an embeddings cache, and the "primary machine" concept disappears.

## Quickstart

```bash
git clone https://github.com/eprouveze/anamnesis && cd anamnesis
export GEMINI_API_KEY=...          # https://aistudio.google.com/apikey
pip install google-genai

python tools/index.py                       # build memory.db from data/
python tools/recall.py "how should I store timestamps?"
```

To run it as a pipeline: add `GEMINI_API_KEY` as an Actions secret, commit a memory under
`data/memory/`, and push — the `index` workflow builds, signs, and publishes a `memory.db` Release.
Point your replicas at it with `tools/pull-and-serve.sh`.

## What's here

| Path | Role |
|---|---|
| `tools/index.py` | the indexer — `data/` → embed → `memory.db` (env-driven, runs anywhere) |
| `tools/recall.py` | reference reader — FTS5 candidates → embedding rerank; query embeddings cached, keyword-only fallback when the API is unavailable |
| `tools/test_recall.py` | unit tests for the cache and fallback — stdlib only, no API key |
| `tools/pull-and-serve.sh` | replica: pull latest Release → verify (Cosign OIDC + SHA256) → atomic swap |
| `tools/anamnesis-rebuild.sh` | local twin of the pipeline (recovery when CI is down) |
| `.github/workflows/index.yml` | the pipeline — index + sign with Cosign + publish on push + nightly |
| `.github/workflows/smoke.yml` | prove your key works in CI (one embedding call) |
| `.github/workflows/test.yml` | unit tests on every PR — no secrets needed |
| `data/` | your memory (markdown), decisions, and structured manifests — the example corpus ships so it builds out of the box |
| `docs/` | [architecture](docs/architecture.md) · [operations](docs/operations.md) |

## Design in one diagram

```
WRITE (any device, offline-ok)    BUILD & SIGN (no machine required)      SERVE (disposable replicas)
  memory = git commit to data/ ─►  GitHub Actions (concurrency:1) ──────►  pull latest Release →
  (phone, laptop, agent, CI)       index + embed + Cosign sign → memory.db  verify signature → atomic swap
```

## Security: Preventing Belief Breaches

When your memory database is a downloadable file, an attacker who compromises a write token
could publish a poisoned memory artifact with a matching checksum. The agent would read it,
trust it, and act on it.

To prevent this, every release artifact is cryptographically signed using **keyless Sigstore/Cosign**
during the GitHub Actions run. Replicas run `tools/pull-and-serve.sh` to verify that the artifact
was built by *this exact repository workflow on `main`* before swapping it into service.

## Serving to an Agent via MCP (Claude Code, Codex, Cursor)

Anamnesis includes a native **Model Context Protocol (MCP)** server in `tools/mcp_server.py`.

### 1. Add to Claude Code
```bash
claude mcp add anamnesis -e GEMINI_API_KEY=your_key -- python /path/to/anamnesis/tools/mcp_server.py
```
Or via `uv` (zero virtualenv setup required):
```bash
claude mcp add anamnesis -e GEMINI_API_KEY=your_key -- uv run --with mcp --with google-genai /path/to/anamnesis/tools/mcp_server.py
```

### 2. Add to Cursor / Windsurf / Codex
Add to your `mcp.json` or `.cursor/mcp.json`:
```json
{
  "mcpServers": {
    "anamnesis": {
      "command": "python",
      "args": ["/absolute/path/to/anamnesis/tools/mcp_server.py"],
      "env": {
        "GEMINI_API_KEY": "your_gemini_api_key",
        "ANAMNESIS_OUT": "/path/to/anamnesis/memory.db"
      }
    }
  }
}
```

Exposed MCP tools:
- `recall_memory(query, limit, store_types)`: Semantic search with cosine reranking and offline keyword fallback.
- `store_memory(type, title, content, rationale)`: Stores a markdown memory chunk and indexes it.
- `store_decision(title, context, decision, alternatives, rationale)`: Appends an authoritative architectural decision.
- `get_recent(limit, store_types)`: Chronological memory retrieval.
- `get_decisions(limit, query)`: Filtered decision search.
- `health()`: Verifies database size, chunk counts, and build commit.

## License

MIT — see [LICENSE](LICENSE).
