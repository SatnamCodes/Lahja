# Lahja

A digital language layer for Kokborok (ISO 639-3: `trp`).

## Feature 1: Text(Kokborok) -> Speech(Kokborok)

No native Kokborok TTS model exists, so this service approximates one two
ways and is always honest about which:

- **`xtts_zero_shot`** (primary) — Coqui XTTS v2, zero-shot voice cloning
  against a short Kokborok reference clip. Input text is Romanized
  (Latin-script) Kokborok, so it's driven through XTTS's English
  grapheme-to-phoneme path as the closest available fit. Works from as
  little as ~10-20s of reference audio, no training required.
- **`mms_bridge_zero_shot`** (fallback) — Meta's `facebook/mms-tts-ben`
  (Bengali) as a phonetically-adjacent bridge language, pretrained weights
  only. Used automatically if no reference audio is available or XTTS
  fails to load. **Caveat:** this model's tokenizer is native Bengali
  script; it currently produces empty output (and the engine skips it) on
  Romanized Kokborok text without a Latin->Bengali transliteration step,
  which isn't wired up yet. In practice XTTS is the only backend that
  works directly on our Latin-script input.
- **`mms_fine_tuned`** (stretch) — the same MMS bridge model fine-tuned on
  the aligned Kokborok batch, once available at
  `models/mms_finetuned/ben/`.

Every response from `/api/speak` reports which method actually produced
the audio via the `method` field, plus a heuristic `confidence`.

### Setup

```bash
pip3 install --break-system-packages -r requirements.txt
```

Drop reference audio (WAV) into `data/audio/`:
- Early on: the ~10-20s clip you have available.
- Later: the larger aligned batch (~30-100 short utterances) as it arrives.

### Run

```bash
./scripts/run.sh
# or: PORT=8080 ./scripts/run.sh
```

`scripts/run.sh` sets `COQUI_TOS_AGREED=1`, which accepts XTTS v2's
non-commercial CPML license non-interactively (fine for a hackathon demo,
not for commercial use). If you run uvicorn directly instead, set that
env var yourself or the first request will hang waiting for a TOS prompt
on stdin. First request also downloads XTTS v2 (~2GB) and can take a
minute or two; subsequent requests are sub-second on GPU.

### API

```
POST /api/speak
{"text": "Nwng bubagra tamwi?", "language": "trp"}

-> {"audio_url": "http://localhost:8000/audio/<hash>.wav",
    "confidence": 0.45,
    "method": "xtts_zero_shot"}
```

```bash
./scripts/test_speak.sh                 # http://localhost:8000
./scripts/test_speak.sh http://host:port "custom text"
```

`GET /health` reports service status and whether it's running on `cuda`
or `cpu`.

Set `LAHJA_PUBLIC_BASE_URL` (default `http://localhost:8000`) if the
service is reachable at a different host/port than it binds to.
