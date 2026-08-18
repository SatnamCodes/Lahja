#!/bin/bash
# Smoke-test the /api/speak endpoint.
set -euo pipefail
HOST="${1:-http://localhost:8000}"
TEXT="${2:-Nwng bubagra tamwi?}"

curl -sS -X POST "$HOST/api/speak" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"$TEXT\", \"language\": \"trp\"}" | python3 -m json.tool
