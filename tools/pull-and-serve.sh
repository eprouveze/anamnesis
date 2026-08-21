#!/usr/bin/env bash
# Replica: pull the latest Release → verify (Cosign OIDC + SHA256) → atomic swap.
# Run on a timer (cron or launchd).
# Use a READ-ONLY, releases-scoped token — never a write-capable PAT.
set -euo pipefail

: "${ANAMNESIS_REPO:=eprouveze/anamnesis}"
: "${GH_TOKEN:?read-only releases-scoped token}"

DEST="${ANAMNESIS_DB:-$HOME/.anamnesis/memory.db}"
mkdir -p "$(dirname "$DEST")"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

# 1. Download release bundle
gh release download --repo "$ANAMNESIS_REPO" -p memory.db -p manifest.json -p cosign.bundle -D "$TMP_DIR"

# 2. Cryptographic signature check (Keyless OIDC bound to this repo's main branch)
if command -v cosign >/dev/null 2>&1; then
    EXPECTED_IDENTITY="https://github.com/${ANAMNESIS_REPO}/.github/workflows/index.yml@refs/heads/main"
    cosign verify-blob \
      --bundle "$TMP_DIR/cosign.bundle" \
      --certificate-identity-regexp "^${EXPECTED_IDENTITY}$" \
      --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
      "$TMP_DIR/memory.db" >/dev/null
    echo "Signature verified via Sigstore/Cosign (Identity: $EXPECTED_IDENTITY)"
else
    echo "WARN: cosign not found on PATH; skipping cryptographic signature check" >&2
fi

# 3. Checksum verification against manifest + sanity floor
python3 - "$TMP_DIR" <<'PY'
import json, hashlib, sys, os, sqlite3
tmp = sys.argv[1]
db_path = os.path.join(tmp, "memory.db")
manifest_path = os.path.join(tmp, "manifest.json")

assert os.path.exists(db_path), "memory.db missing"
assert os.path.exists(manifest_path), "manifest.json missing"

manifest = json.load(open(manifest_path))
actual_sha = hashlib.sha256(open(db_path, "rb").read()).hexdigest()
assert actual_sha == manifest["sha256"], f"SHA256 mismatch: {actual_sha} vs {manifest['sha256']}"

# Sanity floor check
assert os.path.getsize(db_path) > 20_000, "Artifact size below 20KB sanity floor"
chunks = sqlite3.connect(db_path).execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
assert chunks > 0, "Artifact contains zero chunks"
print(f"Verified artifact integrity: {chunks} chunks, {len(open(db_path, 'rb').read())} bytes")
PY

# 4. Atomic swap
mv -f "$TMP_DIR/memory.db" "$DEST"
echo "Served verified memory: $DEST"
