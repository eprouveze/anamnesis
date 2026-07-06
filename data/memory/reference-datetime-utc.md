---
name: always store timestamps in UTC
description: mixing naive-local and UTC timestamps in one column breaks comparisons
type: reference
---
Store every timestamp as UTC with a `Z` suffix. If one writer records naive local
time and another records UTC into the same column, any `MAX()` or freshness
comparison silently spans the timezone offset and misleads. Normalize on write.
