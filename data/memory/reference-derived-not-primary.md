---
name: treat the index as a derived artifact
description: if sources live in git, the database is a build artifact, not primary state
type: reference
---
When the sources of truth already live in version control, the search index is a
*derived* artifact. Rebuild it anywhere from the sources plus an embeddings cache,
and the concept of a "primary machine" disappears — which is the whole design.
