#!/bin/bash
# Starts both halves of Lahja for a demo: the FastAPI backend (port 8000,
# every AI engine) and the Next.js UI (port 3000, proxies /api and /health
# to the backend - see Lahja/next.config.ts). Ctrl+C stops both.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! node -e 'process.exit(process.versions.node.split(".").map(Number)[0] >= 20 ? 0 : 1)' 2>/dev/null; then
  echo "Lahja/ needs Node >=20.9 (found $(node --version 2>/dev/null || echo 'none')); see Lahja/README.md." >&2
  exit 1
fi

export COQUI_TOS_AGREED=1
python3 -m uvicorn service.app:app --host 0.0.0.0 --port "${PORT:-8000}" &
BACKEND_PID=$!
trap 'kill "$BACKEND_PID" 2>/dev/null' EXIT

cd Lahja
[ -d node_modules ] || npm install
npm run dev
