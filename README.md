# Lahja

A digital language layer for Kokborok (ISO 639-3: `trp`), spoken by over a
million people in Tripura and neighboring Bangladesh. Kokborok has no
native entry in any major speech or language model: no TTS voice, no ASR,
no NLLB-200 language code, no LLM that speaks it. Lahja is four features
built to close that gap as honestly as possible - every response names the
real model tier that produced it and a heuristic confidence, rather than
presenting a best-effort bridge as if it were native support.

## Architecture

```
service/        FastAPI backend - the only place any model actually runs
  tts_engine.py     Feature 1: text -> speech (XTTS zero-shot voice clone)
  mt_engine.py       Feature 2: Kokborok <-> English (fine-tuned NLLB-200)
  chat_engine.py      Feature 3: Kokborok Q&A (MT-bridged LLM)
  asr_engine.py        Feature 4: speech -> text (phoneme/Whisper bridge)
Lahja/           Primary web UI - Next.js, proxies /api to the backend
frontend/        Fallback UI - plain HTML/CSS/JS, no build step, served
                  directly by FastAPI at http://localhost:8000/
asr/             Whisper-small + LoRA fine-tuning pipeline for Feature 4
                  (own venv, own README - see asr/README.md)
data/, models/   Reference audio, translated manifests, model checkpoints
```

## Quickstart (demo)

```bash
pip3 install --break-system-packages -r requirements.txt
export GROQ_API_KEY=your-key-here          # free: console.groq.com/keys
export HF_TOKEN=hf_your_token_here         # after accepting the gated repo, see Feature 2
./scripts/run_demo.sh                       # backend :8000 + web UI :3000
```

Opens `http://localhost:3000`. A 90-second walkthrough: type the sample
Kokborok phrase and hit **Speak it** (XTTS clones a real Kokborok voice from
a reference clip in `data/audio/`) → flip to **Translate** and see the
fine-tuned NLLB-200 model turn it into English and back → **Ask Lahja**
a question in Kokborok and watch it bridge through an LLM and back → record
a clip in **Speech → Text** and see the honest IPA-phoneme fallback, since
no Kokborok ASR model exists yet. The "Under the hood" section at the
bottom of the page lays out the full tiered-fallback design.

`Lahja/` needs Node >=20.9 (`nvm install 20` if your system Node is older -
Next 16 will not start on Node 18). No frontend build step is required to
use the backend directly: `./scripts/run.sh` alone serves the no-build
fallback UI from `frontend/` at `http://localhost:8000/`.

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

For longer source recordings (e.g. multi-minute narrations), run
`python3 scripts/prepare_audio.py` first to segment them into short clips,
then `python3 scripts/select_reference_clips.py` to pick the final
`data/audio/reference_*.wav` set. The second step matters more than it
looks: it embeds every candidate clip with a real speaker-verification
model (resemblyzer) and keeps only the ones that are actually the same
speaker - source recordings with a narrator plus dramatized character
voices (common in dubbed/scripted audio) will otherwise get blended into
one inconsistent XTTS speaker embedding, which measurably hurts voice
cloning quality. Naturalness beyond this is capped by not having any
Kokborok-specific training data (see Feature 4) - reference curation
improves voice consistency, not the underlying accent/prosody model.

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

## Feature 2: Text(Kokborok) <-> Text(English) translation

Kokborok has no entry in standard NLLB-200 either, so this uses
[`MWirelabs/kokborok-mt`](https://huggingface.co/MWirelabs/kokborok-mt) -
`nllb-200-distilled-600M` fine-tuned with a custom `trp_Latn` token on
~36k parallel sentences (CC-BY-4.0).

**This repo is gated on Hugging Face** (free, but requires agreeing to
share contact info). Before the first request:

1. Log in to Hugging Face and accept the terms at
   https://huggingface.co/MWirelabs/kokborok-mt
2. Create an access token at https://huggingface.co/settings/tokens
3. `pip3 install --break-system-packages huggingface_hub` (if not already
   pulled in transitively) and either run `huggingface-cli login`, or export
   the token so `transformers` picks it up automatically:
   ```bash
   export HF_TOKEN=hf_your_token_here
   ```

Without this, `/api/translate` and `/api/chat` (which depends on it for the
trp<->eng bridge) fail with a 503 pointing at the gated-repo error.

```
POST /api/translate
{"text": "Nwng bubagra tamwi?", "source_language": "trp", "target_language": "eng"}

-> {"translated_text": "How are you?", "confidence": 0.7, "method": "kokborok_mt_nllb"}
```

```bash
./scripts/test_translate.sh
```

## Feature 3: Kokborok chatbot (Kokborok question -> Kokborok answer)

No LLM has native Kokborok fluency, so this bridges through English: the
question is translated trp->eng with the same MT model, sent to a hosted
LLM, and the answer translated back eng->trp. Requires a free Groq API key
(https://console.groq.com/keys):

```bash
export GROQ_API_KEY=your-key-here
# optional: export GROQ_MODEL=openai/gpt-oss-120b (default)
```

```
POST /api/chat
{"text": "Nwng bubagra tamwi?"}

-> {"answer": "<Kokborok answer>", "english_bridge": "<English answer, for debugging>",
    "confidence": 0.5, "method": "mt_bridge_llm"}
```

```bash
./scripts/test_chat.sh
```

## Feature 4: Kokborok speech -> text (ASR)

No native or bridge-language Kokborok ASR model exists (Meta MMS's
1,107-language ASR set doesn't include `trp`). `data/asr_manifest.jsonl`
now has 273 labeled clips (segmented from narrated Kokborok audio in
`data/asr_dataset/`, transcribed in Bengali script), and a LoRA fine-tuning
pipeline for them lives in [`asr/`](asr/) - speaker-disjoint train/test
splits, a required baseline-vs-fine-tuned comparison, and WER/CER logged to
`asr/results/metrics.jsonl` per run. No checkpoint has been trained and
deployed from it yet (`models/whisper_finetuned/trp/` is still empty), so
`/api/transcribe` currently falls through to the zero-shot tiers below. It
tries three tiers, in order, and is honest in the response about which one
answered:

1. **`whisper_fine_tuned`** (confidence `0.6`) - a real checkpoint fine-tuned
   on labeled Kokborok audio. Not available yet; see below.
2. **`phoneme_zero_shot_bridge`** (confidence `0.3`) - a language-agnostic
   IPA phoneme recognizer
   ([`facebook/wav2vec2-lv-60-espeak-cv-ft`](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft)).
   Needs no Kokborok training data at all - it reports the sounds it hears
   as IPA phoneme symbols instead of guessing words in a language it's
   never seen. Not real Kokborok orthography, but a human can usually
   reconstruct actual words from phonemes faster than from a
   wrong-language word guess. Setup (one-time):
   ```bash
   pip3 install --break-system-packages phonemizer   # already in requirements.txt
   python3 scripts/convert_phoneme_model.py
   ```
   The convert step exists because this model's weights only ship as
   `pytorch_model.bin`, and `transformers` refuses to `torch.load()` that
   on torch < 2.6 (CVE-2025-32434) - this repo stays on torch 2.5.x for
   XTTS compatibility, so the script re-saves the weights as `safetensors`
   locally instead of upgrading torch.
3. **`whisper_zero_shot_bridge`** (confidence `0.15`) - generic
   multilingual Whisper doing its own language ID, as a last resort if the
   phoneme model isn't set up. Its output is confident words in whatever
   language it guesses (Hindi/Devanagari script in testing), which is
   actively misleading for Kokborok audio - used only when the phoneme
   tier is unavailable.

**To produce a real checkpoint**, either:

- the quick path - `python3 scripts/finetune_whisper.py`, a plain full
  fine-tune reading `data/asr_manifest.jsonl` directly; or
- the rigorous path - the LoRA pipeline in [`asr/`](asr/) (its own venv,
  speaker-disjoint splits, baseline comparison, logged WER/CER). Its
  adapter checkpoints land in `asr/checkpoints/`, not
  `models/whisper_finetuned/trp/`, so merging an adapter back into a
  standalone checkpoint there is the one remaining step before
  `ASREngine` picks it up automatically and prefers it over both
  zero-shot tiers.

`scripts/prepare_audio.py` segments long recordings in `data/audio/*.mp3`
into short clips and writes a fill-in-the-blank
`data/asr_manifest_template.jsonl` to start a new manifest from, if you're
adding more labeled audio.

```
POST /api/transcribe   (multipart, field name "audio")

-> {"text": "s a j ð e j...", "confidence": 0.3, "method": "phoneme_zero_shot_bridge"}
```

```bash
./scripts/test_transcribe.sh http://localhost:8000 data/audio/clip001.wav
```
