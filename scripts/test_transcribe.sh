#!/bin/bash
# Smoke-test the /api/transcribe endpoint.
set -euo pipefail
HOST="${1:-http://localhost:8000}"
AUDIO_FILE="${2:?usage: test_transcribe.sh [host] <path-to-wav>}"

curl -sS -X POST "$HOST/api/transcribe" \
  -F "audio=@${AUDIO_FILE}" | python3 -m json.tool
