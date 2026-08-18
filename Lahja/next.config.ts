import type { NextConfig } from "next"

// The FastAPI backend (service/app.py, see ../scripts/run.sh) owns every AI
// engine: XTTS voice cloning, the Kokborok<->English NMT model, the Groq
// chat bridge, and the phoneme/Whisper ASR tiers. Proxying /api and /audio
// through Next's dev/prod server means the browser only ever talks to one
// origin, so there's no CORS setup required for the demo.
const BACKEND_URL = process.env.LAHJA_BACKEND_URL ?? "http://localhost:8000"

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${BACKEND_URL}/api/:path*` },
      { source: "/audio/:path*", destination: `${BACKEND_URL}/audio/:path*` },
      { source: "/health", destination: `${BACKEND_URL}/health` },
    ]
  },
}

export default nextConfig
