# Lahja ASR — Kokborok speech-to-text

Feature: **Audio (Kokborok) → Text (Kokborok)**.

Scope is deliberately narrow. This component does **not** do TTS (that is
Lahja's [`service/`](../service) package, which it neither imports nor
modifies) and does **not** translate to English — those are separate
components.

Approach: **Whisper-small fine-tuned with LoRA**. Whisper has no Kokborok
(`trp`) language token, so it is driven through a proxy language token —
`en` by default, since our transcripts are Romanized Latin script. This is
an approximation, in the same spirit as the TTS side's bridge languages, and
the proxy actually used is recorded in every metrics record.

## Setup

The ASR component keeps its **own** virtualenv. Lahja's root
`requirements.txt` pins `transformers==4.57.6` for `coqui-tts`, which
conflicts with the current `transformers` needed for Whisper fine-tuning, so
the two environments must stay separate.

```bash
cd asr
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

Verified working on: Python 3.14, torch 2.13+cu130, transformers 5.15,
peft 0.20, on a 6 GB RTX 4050 Laptop GPU.

## Data

`data/raw/` is **read-only** — nothing in this repo ever writes to it. Two
layouts are auto-detected:

| Layout | Shape |
| --- | --- |
| `pairs` | `data/raw/spk01/utt000.wav` + `data/raw/spk01/utt000.txt` |
| `manifest` | `data/raw/manifest.csv` with `audio_path,transcript,speaker_id` |

Speaker attribution matters more than it looks: it drives the split, so a
wrong `data.speaker_strategy` silently leaks a speaker between train and
test and reports a WER that is far too good. Set it explicitly:

- `parent_dir` — one folder per speaker (default)
- `filename_prefix` — `spk01_utt000.wav`, split on `data.speaker_delimiter`
- `single` — everything is one speaker (blocks a real split)

Clips must be **≤ 30 s**; Whisper's encoder is fixed at 30 s and would
silently truncate longer audio, desynchronizing it from the transcript.
Longer recordings need segmenting first.

## Workflow

```bash
cd asr

# 1. Build speaker-disjoint splits from data/raw -> data/processed
./.venv/bin/python scripts/prepare_data.py

# 2. ALWAYS smoke-test before a real run (~25 s, synthetic data, 2 steps)
./.venv/bin/python scripts/smoke_test.py

# 3. Train (resumable — rerun the same command after an interruption)
./.venv/bin/python scripts/train.py

# 4. Score. Prefer the fast check while iterating:
./.venv/bin/python scripts/evaluate.py --set eval.limit=20
./.venv/bin/python scripts/evaluate.py            # full test split
```

Everything is config-driven — no paths or hyperparameters are hardcoded in a
script body. Override any key from the CLI:

```bash
./.venv/bin/python scripts/train.py --set train.batch_size=2 --set train.max_steps=500
```

### Establish a baseline first

Before trusting a fine-tune, score the untuned model on the same split. If
LoRA doesn't beat this, the training isn't helping:

```bash
./.venv/bin/python scripts/evaluate.py --adapter none
```

## Splits are disjoint by speaker

Splitting by utterance would put the same voice in train and test, and on a
small corpus the model then recognizes the *speaker* rather than the
language. `split_by_speaker` assigns whole speakers, largest first, to
whichever split is furthest below its target share.

It **refuses** to produce a leaky split: with fewer speakers than non-empty
splits it raises rather than silently overlapping. `split.allow_speaker_overlap:
true` forces an utterance-level fallback, but the resulting WER is optimistic
and should never be reported as a real result.

`data/processed/split_summary.json` records the per-split speaker lists and
an overlap check that must always read empty.

## Results

Every training and evaluation run appends one JSON line to
`results/metrics.jsonl` (WER, CER, speakers per split, language proxy,
device, LoRA settings, resumed-from checkpoint), so runs stay comparable long
after the console scrollback is gone. Smoke runs write to a *separate*
`data/fixtures/smoke_metrics.jsonl`, so synthetic-data scores can never
contaminate the real history.

Each eval also writes a per-utterance `results/predictions_*.jsonl` with
reference/hypothesis pairs and per-utterance WER. **Read it.** Whisper
fine-tuned on a small corpus produces fluent, confident, and entirely wrong
transcripts; an aggregate WER hides that, and the dump is how you catch it.

## Layout

```
asr/
├── configs/
│   ├── whisper_small_lora.yaml   # the real run
│   └── smoke.yaml                # tiny everything, synthetic fixture
├── scripts/
│   ├── prepare_data.py           # raw -> speaker-disjoint splits
│   ├── train.py                  # LoRA fine-tune, resumable
│   ├── evaluate.py               # WER/CER + prediction dump
│   ├── smoke_test.py             # full pipeline, tiny
│   └── make_fixture.py           # synthetic corpus (NOT real language data)
├── src/kokborok_asr/
│   ├── manifest.py               # corpus discovery, both layouts
│   ├── splitting.py              # speaker-disjoint splitting
│   ├── data.py                   # audio loading, dataset, collator
│   ├── modeling.py               # Whisper + LoRA construction
│   ├── evaluation.py             # shared generate + score path
│   ├── metrics.py                # WER/CER, metrics.jsonl
│   ├── text.py                   # transcript normalization
│   └── config.py                 # YAML + --set overrides
├── data/{raw,processed}/         # raw is READ-ONLY
├── checkpoints/
└── results/metrics.jsonl
```

## Notes for the 6 GB GPU

Defaults are tuned for a 6 GB card: batch size 4 with gradient accumulation
to an effective 16, gradient checkpointing on, bf16 where supported. If you
hit OOM, lower `train.batch_size` and raise
`train.gradient_accumulation_steps` to keep the effective batch constant.
Whisper-medium and large will not fit for training on this card.

## Fixture data is not language data

`scripts/make_fixture.py` generates tones and noise with mechanically
recombined word forms, purely so the pipeline is runnable before real
recordings exist. It carries no linguistic information and its WER is
meaningless — never report a metric computed on it.
