# Lahja web UI

The primary demo frontend for Lahja — a digital language layer for
Kokborok (ISO 639-3 `trp`). Next.js 16 + shadcn/ui. See the [root
README](../README.md) for the overall project and what each AI feature
actually does.

This app has **no AI of its own** — every result on the page comes from
the FastAPI backend in [`../service`](../service). `/api/:path*`,
`/audio/:path*`, and `/health` are proxied straight through to it (see
[`next.config.ts`](./next.config.ts)), so the browser only ever talks to
one origin and there's nothing to configure for a same-machine demo.

## Run

Needs **Node >=20.9** (`node --version`; use `nvm install 20` if you're on
an older system Node — Next 16 refuses to run on Node 18).

```bash
# from the repo root
./scripts/run_demo.sh
```

That starts the backend (port 8000) and this app (port 3000, `npm install`
on first run) together and stops both on Ctrl+C. Open
`http://localhost:3000`. Or run the two halves yourself in separate
terminals:

```bash
./scripts/run.sh          # backend, from the repo root
cd Lahja && npm install && npm run dev   # this app
```

The nav bar's status dot polls `/health` and shows whether the backend is
reachable — check that first if a panel errors out.

To point this UI at a backend running elsewhere, set `LAHJA_BACKEND_URL`
before `npm run build`/`next dev` (it's read at build/dev-server start, not
in the browser):

```bash
LAHJA_BACKEND_URL=http://your-host:8000 npm run dev
```

## Layout

```
app/page.tsx                        four feature sections + "how it works"
components/site/
  feature-text-to-speech.tsx        Feature 1: POST /api/speak
  feature-speech-to-text.tsx        Feature 4: POST /api/transcribe
  feature-translate.tsx             Feature 2: POST /api/translate
  feature-chat.tsx                  Feature 3: POST /api/chat
  method-badge.tsx                  shared confidence/method badge
  how-it-works.tsx                  static copy of the tiered-fallback design
hooks/
  use-audio-recorder.ts             MediaRecorder wrapper for the mic
  use-backend-health.ts             polls /health for the nav status dot
lib/
  api.ts                            fetch wrappers + ApiError
  kokborok.ts                       shared constants (sample phrase, tiers)
```

Every panel shows the real `method` and `confidence` the backend returned —
there is no client-side mock or placeholder path. If the backend is down or
a model fails to load, the panel shows the backend's actual error message
rather than a fake result.

## Commands

```bash
npm run dev         # dev server, hot reload
npm run build        # production build
npm run start         # serve the production build
npm run lint          # eslint
npm run typecheck     # tsc --noEmit
npm run format         # prettier --write
```
