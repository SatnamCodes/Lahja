#!/bin/bash
# Smoke-test the /api/chat endpoint. Requires GROQ_API_KEY to be set on the server.
set -euo pipefail
HOST="${1:-http://localhost:8000}"
TEXT="${2:-Nwng bubagra tamwi?}"

curl -sS -X POST "$HOST/api/chat" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"$TEXT\"}" | python3 -m json.tool
