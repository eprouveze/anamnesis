# Architecture

Three pillars. The design goal is singular: **no single machine is required for the write
or refresh path.**

## 1. Write = git commit

A memory or decision is a markdown/JSON file committed under `data/`. This works from any
device with git, **offline** — commits queue locally and push on reconnect. Git commit
ordering serializes writers, so **split-brain is structurally impossible**: there is no
"promote a new primary" step to get wrong. One writer per dataset, enforced by git itself.

`data/` layout:
- `data/memory/*.md` — memories. `feedback*`, `reference*`, else `memory` — weighted at recall.
- `data/decisions/*.md` — decisions; highest trust (they read as authoritative to an agent).
- `data/manifests/*.json` — structured stores (contacts, topics, capabilities, …).

## 2. Build = a pipeline (`.github/workflows/index.yml`)

On push to `data/**` (and nightly), a GitHub Actions job runs `tools/index.py`:
checkout → embed each chunk (`gemini-embedding-001`, dim 3072) → write `memory.db`
(SQLite FTS5 keyword index + `embedding BLOB`) → **integrity floor** (non-empty,
fully embedded, min size) → publish `memory.db` + `manifest.json` as a **Release asset**.
`concurrency: index` serializes runs.

The indexer takes **no machine-specific paths** — it is driven by `ANAMNESIS_DATA_ROOT`,
`ANAMNESIS_OUT`, and `GEMINI_API_KEY` — so the *same script* runs on a runner or a laptop.
`tools/anamnesis-rebuild.sh` is that local twin: any machine can hand-produce a release when
the CI itself is the broken thing.

## 3. Serve = disposable replicas

Each serving machine runs `tools/pull-and-serve.sh` on a timer: fetch the latest Release →
**verify** (sanity floor; add signature verification in production) → download-to-temp →
**atomic `mv`** into place. N replicas, none individually required. In production you front
them with a small failover proxy so a reader always hits a live one.

### Serving

`tools/recall.py` is a reference reader (FTS5 candidates → embedding cosine rerank). To give
an agent direct access, wrap the same query in a [Model Context Protocol](https://modelcontextprotocol.io)
server exposing a `recall` tool over `memory.db`. That server is deliberately outside this
core kit — it's the easy part. The hard part, and the point of this repo, is the durable
no-primary write/refresh path above.

## Latency & consistency

Reads are local and fast. Write→recallable is roughly *commit + CI + replica-pull* (minutes),
i.e. eventually consistent across sessions. The only thing given up is instant
read-your-own-write on a primary — rarely needed, since a session already holds what it just
wrote. Memory is a *cross-session* substrate.

## Security model

Making any machine a writer to a shared, agent-trusted memory enlarges the poisoning surface.
Controls, in rough priority:

| Control | Why |
|---|---|
| **Protect the pipeline** — CODEOWNERS/branch-protection on `.github/**` + `tools/**`; data commits touch only `data/**` | The workflow holds the embedding key; a data-write credential must not be able to edit it. |
| **Verify artifacts before serving** — signature + sanity floor before the atomic swap | A poisoned release would otherwise become trusted context fleet-wide. Prefer a signature whose key never touches the CI runner (e.g. cosign / sigstore). |
| **Quarantine high-trust writes** — decisions land on a review branch, merge after a window | A single injected "decision" must not silently become authoritative for every future session. |
| **Attributable writes** — carry the writer's identity in each commit | So a bad entry is traceable to who/what wrote it. |
| **Split credentials** — replicas pull with a read-only, releases-scoped token; writers use a separate write token | A compromised read-only replica must not gain write access. |

None of these are exotic; they're the difference between a memory an agent can trust and one
an attacker can rewrite.
