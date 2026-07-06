# Decisions log (example)

### 2026-01-01 — memory is a git-native build artifact
**Context:** a single primary machine was the sole writer/indexer; its outage froze all memory.
**Decision:** writes become git commits; a CI pipeline indexes+embeds; every machine is a replica.
**Alternatives:** promote a second machine to primary (still a machine); a managed vector DB (third-party).
**Rationale:** no machine is required for the write/refresh path; sources stay in your git.
