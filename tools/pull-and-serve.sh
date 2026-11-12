#!/usr/bin/env bash
# Replica: pull the latest Release → verify (Cosign OIDC + SHA256) → atomic swap.
# Run on a timer (cron or launchd).
set -euo pipefail

: "${ANAMNESIS_REPO:=eprouveze/anamnesis}"
DEST="${ANAMNESIS_DB:-$HOME/.anamnesis/memory.db}"
mkdir -p "$(dirname "$DEST")"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

fail() { echo "PULL-VERIFY FAIL: $*" >&2; exit 1; }

command -v gh >/dev/null || fail "gh CLI required (brew/apt install gh)"
TAG="$(gh release view --repo "$ANAMNESIS_REPO" --json tagName -q .tagName 2>/dev/null || true)"
[ -n "$TAG" ] || fail "No release found on $ANAMNESIS_REPO"

# 1. Fast-path: Check manifest before downloading large database
gh release download "$TAG" --repo "$ANAMNESIS_REPO" -p manifest.json -D "$TMP_DIR" >/dev/null 2>&1 || fail "manifest.json missing from release"

NEW_SHA="$(python3 -c "import json; m=json.load(open('$TMP_DIR/manifest.json')); print(m.get('source_commit', m.get('sha256', 'none')))")"
if [ -f "${DEST}.manifest.json" ]; then
  CUR_SHA="$(python3 -c "import json; m=json.load(open('${DEST}.manifest.json')); print(m.get('source_commit', m.get('sha256', 'none')))" 2>/dev/null || echo "old")"
  if [ "$NEW_SHA" = "$CUR_SHA" ] && [ "$NEW_SHA" != "none" ]; then
    echo "Anamnesis replica up to date ($TAG: $NEW_SHA)"
    exit 0
  fi
fi

# 2. Download release bundle (memory.db + signature bundle)
gh release download "$TAG" --repo "$ANAMNESIS_REPO" -p memory.db -D "$TMP_DIR" >/dev/null 2>&1 || fail "memory.db missing from release"
gh release download "$TAG" --repo "$ANAMNESIS_REPO" -p cosign.bundle -D "$TMP_DIR" >/dev/null 2>&1 || true

# 3. Signature check: the artifact must have been signed by this repository's index
# workflow on main (keyless OIDC). The manifest checksum below only proves integrity,
# not origin, so an unsigned artifact is refused unless ANAMNESIS_ALLOW_UNSIGNED=1.
BUNDLE="$TMP_DIR/cosign.bundle"
if command -v cosign >/dev/null 2>&1 && [ -s "$BUNDLE" ]; then
  EXPECTED_IDENTITY="https://github.com/${ANAMNESIS_REPO}/.github/workflows/index.yml@refs/heads/main"
  cosign verify-blob \
    --bundle "$BUNDLE" \
    --certificate-identity "$EXPECTED_IDENTITY" \
    --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
    "$TMP_DIR/memory.db" >/dev/null || fail "signature verification failed"
  echo "Signature verified (identity: $EXPECTED_IDENTITY)"
elif [ "${ANAMNESIS_ALLOW_UNSIGNED:-0}" = "1" ]; then
  echo "WARN: serving WITHOUT signature verification (ANAMNESIS_ALLOW_UNSIGNED=1)" >&2
else
  command -v cosign >/dev/null 2>&1 || fail "cosign not installed (or set ANAMNESIS_ALLOW_UNSIGNED=1 to accept unsigned artifacts)"
  fail "release has no cosign.bundle (or set ANAMNESIS_ALLOW_UNSIGNED=1)"
fi

# 4. Checksum verification against manifest + sanity floor
python3 - "$TMP_DIR" <<'PY'
import json, hashlib, sys, os, sqlite3
tmp = sys.argv[1]
db_path = os.path.join(tmp, "memory.db")
manifest_path = os.path.join(tmp, "manifest.json")

assert os.path.exists(db_path), "memory.db missing"
assert os.path.exists(manifest_path), "manifest.json missing"

manifest = json.load(open(manifest_path))
actual_sha = hashlib.sha256(open(db_path, "rb").read()).hexdigest()
want_sha = manifest["sha256"]["memory.db"] if isinstance(manifest.get("sha256"), dict) else manifest.get("sha256")
if want_sha:
    assert actual_sha == want_sha, f"SHA256 mismatch: {actual_sha} vs {want_sha}"

# Sanity floor check
assert os.path.getsize(db_path) > 20_000, "Artifact size below 20KB sanity floor"
chunks = sqlite3.connect(db_path).execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
assert chunks > 0, "Artifact contains zero chunks"
print(f"Verified artifact integrity: {chunks} chunks, {os.path.getsize(db_path)} bytes")
PY

# 5. Atomic swap on the same filesystem
STAGE="$(dirname "$DEST")/.memory.db.incoming.$$"
cp "$TMP_DIR/memory.db" "$STAGE"
mv -f "$STAGE" "$DEST"
cp "$TMP_DIR/manifest.json" "${DEST}.manifest.json"
echo "Served verified memory: $DEST ($TAG)"
