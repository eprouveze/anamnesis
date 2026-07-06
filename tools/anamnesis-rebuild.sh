#!/usr/bin/env bash
# Local twin of the CI pipeline: rebuild memory.db on ANY machine (recovery path
# when GitHub Actions itself is unavailable). Same script the runner uses.
set -euo pipefail
: "${GEMINI_API_KEY:?get one at https://aistudio.google.com/apikey}"
pip install --quiet google-genai 2>/dev/null || true
ANAMNESIS_DATA_ROOT="${ANAMNESIS_DATA_ROOT:-data}" ANAMNESIS_OUT="${ANAMNESIS_OUT:-memory.db}" \
  python3 tools/index.py
echo "rebuilt memory.db from ${ANAMNESIS_DATA_ROOT:-data}/"
