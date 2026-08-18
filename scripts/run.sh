#!/bin/bash
# Start the LAHJA TTS service.
set -euo pipefail
cd "$(dirname "$0")/.."
# XTTS v2 is Coqui Public Model License (non-commercial). Setting this
# skips its interactive prompt; only appropriate for non-commercial use
# (a hackathon demo qualifies). See https://coqui.ai/cpml
export COQUI_TOS_AGREED=1
exec python3 -m uvicorn service.app:app --host 0.0.0.0 --port "${PORT:-8000}"
