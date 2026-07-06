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
pipeline** indexes and embeds them into a `memory.db` (SQLite FTS5 + vector rerank) and
publishes it as a **Release artifact**. Every machine is a disposable replica that pulls
the artifact and serves it. Lose any machine — or all of them — and the write path still
works (git commits queue offline) and the index rebuilds anywhere.

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
`data/memory/`, and push — the `index` workflow builds and publishes a `memory.db` Release.
Point your replicas at it with `tools/pull-and-serve.sh`.

## What's here

| Path | Role |
|---|---|
| `tools/index.py` | the indexer — `data/` → embed → `memory.db` (env-driven, runs anywhere) |
| `tools/recall.py` | reference reader — FTS5 candidates → embedding rerank |
| `tools/pull-and-serve.sh` | replica: pull latest Release → verify → atomic swap |
| `tools/anamnesis-rebuild.sh` | local twin of the pipeline (recovery when CI is down) |
| `.github/workflows/index.yml` | the pipeline — index + publish on push + nightly |
| `.github/workflows/smoke.yml` | prove your key works in CI (one embedding call) |
| `data/` | your memory (markdown), decisions, and structured manifests — the example corpus ships so it builds out of the box |
| `docs/` | [architecture](docs/architecture.md) · [operations](docs/operations.md) |

## Design in one diagram

```
WRITE (any device, offline-ok)    BUILD (no machine required)        SERVE (disposable replicas)
  memory = git commit to data/ ─►  GitHub Actions (concurrency:1) ─►  pull latest Release →
  (phone, laptop, agent, CI)       index + embed → memory.db         verify → atomic swap → serve
```

## Serving to an agent

`recall.py` is a reference reader. In production you serve `memory.db` over the
[Model Context Protocol](https://modelcontextprotocol.io) so an agent can call `recall`
directly — see [docs/architecture.md](docs/architecture.md#serving). The MCP server is
intentionally out of this core kit (it's the easy part); the hard part is the durable,
no-primary write/refresh path, which is what this repo is about.

## License

MIT — see [LICENSE](LICENSE).
