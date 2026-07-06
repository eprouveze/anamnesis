#!/usr/bin/env bash
# Replica: pull the latest Release → verify → atomic swap. Run on a timer.
# Use a READ-ONLY, releases-scoped token — never a write-capable PAT.
set -euo pipefail
: "${ANAMNESIS_REPO:?e.g. youruser/anamnesis}"
: "${GH_TOKEN:?read-only releases-scoped token}"
DEST="${ANAMNESIS_DB:-$HOME/.anamnesis/memory.db}"; mkdir -p "$(dirname "$DEST")"
tmp="$(mktemp)"
gh release download --repo "$ANAMNESIS_REPO" -p memory.db -O "$tmp"
# Verify before trusting: minimal SQLite + non-empty sanity floor.
python3 - "$tmp" <<'PY'
import sqlite3, sys, os
db = sys.argv[1]
assert os.path.getsize(db) > 20_000, "artifact too small"
assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM chunks").fetchone()[0] > 0
print("verified")
PY
mv -f "$tmp" "$DEST"   # atomic swap
echo "served: $DEST"
