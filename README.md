# anamnesis

**Persistent memory for an AI agent, built so no single machine is required to write or
refresh it.** The *memory* pillar of
[Stewardship Engineering](https://github.com/eprouveze/stewardship-engineering) —
alongside [heartbeat](https://github.com/eprouveze/heartbeat) (the loop) and
[rightmodel](https://github.com/eprouveze/rightmodel) (routing & trust).

An agent that operates as a custodian of your work over months needs memory that is
durable, resilient, and yours.

> **The DB is a build artifact. Git is the write-ahead log. The writer is a pipeline.**

Memory entries are **git commits** (markdown/JSON under `data/`, plus any other roots you
list in a sources config). The indexer chunks and embeds them into a `memory.db` (SQLite
FTS5 + vector rerank + a materialized ontology). A pipeline can build it in CI, sign it
with **keyless Sigstore/Cosign** and publish it as a Release; every machine is then a
disposable replica that verifies and serves it. Or run one primary writer and serve it
directly over MCP. Either way the database is rebuildable from git.

> Working tools with rough edges, not a polished product.

## Quickstart

```bash
git clone https://github.com/eprouveze/anamnesis && cd anamnesis
python3.12 -m venv .venv && . .venv/bin/activate   # Python 3.12+
pip install -r requirements.txt

export GEMINI_API_KEY=...                    # or ANAMNESIS_EMBED_BACKEND=mock / local
# local backend only: pip install sentence-transformers torch
python tools/index.py build                  # data/ → ~/.anamnesis/memory.db
python tools/recall.py "how should I store timestamps?"
```

To index more than `data/`, copy `anamnesis.sources.example.yaml`, list your roots, and
point `ANAMNESIS_SOURCES` at it.

## What's here

| Path | Role |
|---|---|
| `tools/index.py` | indexer: sources config → chunk → embed (Gemini, local sentence-transformers, or mock) → `memory.db`; spend caps, retry/backoff, partial-embed degrade, chat-export import |
| `tools/recall.py` | retrieval core (FTS5 candidates → embedding rerank, keyword fallback) + CLI |
| `tools/mcp_server.py` | MCP server (stdio or streamable HTTP): 18 tools, hashed per-client tokens, optional OAuth |
| `tools/anamnesis_ontology.py` | registry compiler → `ontology_nodes/claims/evidence`; impact & audit queries; stable link coordinates |
| `tools/anamnesis_dict.py` | loads `data/dictionary.json` (projects, node kinds, predicates, store weights, registries) |
| `tools/anamnesis_validity.py` | link-validity / supersession schema used by recall |
| `tools/anamnesis_redact.py` | shared secret-redaction patterns for imported text |
| `tools/anamnesis_llm.py` | opt-in LLM helper for dream/decay judging (no default remote provider) |
| `tools/dream.py` | periodic memory consolidation (orient → gather → consolidate → prune), dry-run by default |
| `tools/decay.py` | propose-only value-decay review of curated facts |
| `tools/wiki_link_resolver.py` | deterministic `[[wiki-link]]` edge resolver |
| `tools/chat_import.py`, `tools/ingest_conversation.py` | import chat exports / clip LLM conversations into memory (with secret redaction) |
| `tools/eval_recall.py` | recall quality eval against a golden set (LLM or heuristic judge) |
| `tools/pull-and-serve.sh` | replica: pull latest Release → verify (Cosign OIDC + SHA256) → atomic swap |
| `tools/anamnesis-rebuild.sh` | local twin of the CI pipeline |
| `.github/workflows/` | `test` (every PR, no secrets) · `index` (build + sign + publish on main) · `smoke` (manual key check) |
| `data/` | example memory, decisions, manifests and `dictionary.json` — builds out of the box |
| `docs/` | [architecture](docs/architecture.md) · [operations](docs/operations.md) · [dictionary](docs/dictionary.md) |

## Serving to an agent via MCP

Local (stdio — trusted, no token):
```bash
# single quotes: the client stores the ${GEMINI_API_KEY} reference and expands it at launch,
# so the key itself never lands in its config file
claude mcp add anamnesis -e 'GEMINI_API_KEY=${GEMINI_API_KEY}' -- \
  /path/to/anamnesis/.venv/bin/python /path/to/anamnesis/tools/mcp_server.py --transport stdio
```

Remote (HTTP): run `python tools/mcp_server.py` (binds `127.0.0.1:7777/mcp`; put your own
TLS proxy or tunnel in front), create a token per client, and send it as a header:
```bash
python tools/mcp_server.py add-client laptop            # prints the token once; stores only its SHA-256
python tools/mcp_server.py add-client dashboard --read-only
claude mcp add --transport http anamnesis https://memory.example.com/mcp \
  --header "Authorization: Bearer $ANAMNESIS_TOKEN"
```

Tools: `brief`, `recall`, `recall_precedent`, `store_memory`, `store_decision`,
`synthesize`, `trace`, `get_mistakes`, `get_topic`, `get_decisions`, `get_recent`,
`session_context`, `list_files`, `read_file`, `get_capabilities`, `ontology_audit`,
`ontology_impact`, `health`.

## Security model

- **Header tokens only by default.** A token in a URL ends up in configs, proxy logs and
  transcripts, so `?token=` is refused unless `ANAMNESIS_ALLOW_QUERY_TOKEN=1`.
- **Per-client, hashed, scoped.** `clients.json` (mode 0600) stores SHA-256 hashes compared
  with `hmac.compare_digest`; revoke one client without touching others, no restart needed.
  Read-only clients cannot call `store_*`.
- **OAuth for GUI connectors** (claude.ai, ChatGPT): `ANAMNESIS_OAUTH_PROVIDER=google|github`
  with a mandatory server-side allowlist of emails/logins. Headless agents keep using
  bearer tokens on the same endpoint.
- **Fail closed.** Binds to loopback; unauthenticated loopback access is opt-in and
  read-only; `read_file`/`list_files` are confined to configured roots (symlink-safe) and
  refuse secrets, the DB and the clients file; tool output never contains host paths.
- **Signed artifacts.** In the CI topology every release is Cosign-signed by a publish job
  that runs no repository code; `pull-and-serve.sh` refuses any artifact not signed by this
  repository's workflow on `main` (override only with `ANAMNESIS_ALLOW_UNSIGNED=1`).
  Untrusted runs (PRs, forks, branches) build with mock embeddings and never see a secret.
- **Secrets from the environment only** — never read from dotfiles or secret stores.

## License

MIT — see [LICENSE](LICENSE).
