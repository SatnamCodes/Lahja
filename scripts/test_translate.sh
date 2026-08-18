#!/bin/bash
# Smoke-test the /api/translate endpoint.
set -euo pipefail
HOST="${1:-http://localhost:8000}"
TEXT="${2:-Nwng bubagra tamwi?}"

curl -sS -X POST "$HOST/api/translate" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"$TEXT\", \"source_language\": \"trp\", \"target_language\": \"eng\"}" | python3 -m json.tool
