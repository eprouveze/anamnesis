# Decisions log (example)

### 2026-01-01 — memory is a git-native build artifact
**Context:** any design where one machine or one hosted DB is the sole writer/indexer makes that component a single point of failure for the write path.
**Decision:** writes become git commits; a CI pipeline indexes+embeds; every machine is a disposable replica.
**Alternatives:** promote a second machine to primary (still a machine); a managed vector DB (a third-party dependency).
**Rationale:** no machine is required for the write/refresh path; the sources stay in your git.
