# Lahja ASR — Kokborok speech-to-text

Feature: **Audio (Kokborok) → Text (Kokborok)**.

Scope is deliberately narrow. This component does **not** do TTS (that is
Lahja's [`service/`](../service) package, which it neither imports nor
modifies) and does **not** translate to English — those are separate
components.

Approach: **Whisper-small fine-tuned with LoRA**. Whisper has no Kokborok
(`trp`) language token, so it is driven through a proxy language token —
**`bn` (Bengali)**, because our transcripts are in **Bengali script**
(Kokborok is written in either Bengali or Latin script; this corpus uses
Bengali). Bengali shares the target script and most of the subword inventory,
so the tokenizer emits a sane number of tokens per word instead of falling
back to bytes. It is a script and tokenizer match, **not** a claim that
Bengali is phonetically adjacent to Kokborok — an approximation in the same
spirit as the TTS side's bridge languages. The proxy actually used is recorded
in every metrics record.

Note that Lahja's TTS side uses *Romanized* Kokborok, so the two components do
not share a script. Don't assume either convention when moving text between
them.

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

`data/raw/` is **read-only** — nothing in this repo ever writes to it. Three
layouts are auto-detected (in `auto` mode, by the manifest's suffix):

| Layout | Shape |
| --- | --- |
| `pairs` | `data/raw/spk01/utt000.wav` + `data/raw/spk01/utt000.txt` |
| `manifest` | `data/raw/manifest.csv` with `audio_path,transcript,speaker_id` |
| `jsonl` | one JSON object per line: `{"audio": ..., "text": ..., "speaker_id": ...}` |

The Kokborok corpus uses `jsonl`, and its manifest lives **outside** `raw_dir`
— at the repo root, built by `scripts/build_asr_dataset.py` then
`scripts/filter_bengali_script.py` one level above `asr/`. Three config keys
make that work, and each fails *silently* if wrong:

- **`paths.manifest_file`** — read a manifest from outside `raw_dir`. Still
  read-only; `raw_dir`'s guarantee is untouched.
- **`data.audio_root`** — base for relative audio paths. The jsonl manifest
  stores repo-root-relative paths while itself living in `<repo>/data/`, so
  resolving against its own directory looks for
  `<repo>/data/data/asr_dataset/...`. Nothing raises: `validate()` drops every
  row as "audio file missing" and the error then blames the wrong thing.
- **`data.utt_id_prefix`** — namespaces utterance ids. Clips outside `raw_dir`
  get bare-filename ids, unique only while one flat directory is the sole
  source; a second source with its own `clip_0000.wav` would otherwise have its
  rows quietly dropped as duplicates.

Group attribution matters more than it looks: it drives the split, so a wrong
`data.speaker_strategy` silently leaks a group between train and test and
reports a WER that is far too good. Set it explicitly:

- `manifest_field` — trust the row's own `speaker_id`/`speaker` (what the
  Kokborok config uses; `build_asr_dataset.py` writes an episode id per clip)
- `parent_dir` — one folder per speaker
- `filename_prefix` — `spk01_utt000.wav`, split on `data.speaker_delimiter`
- `single` — everything is one speaker (blocks a real split)

On the Kokborok corpus specifically, **neither path-derived strategy works and
neither errors**: `parent_dir` returns `asr_dataset` for every clip (they sit
outside `raw_dir`, so its guard never fires) and `filename_prefix` returns
`clip` (because `clip_0000` contains the default `_` delimiter). Both collapse
18 groups into 1 without a warning.

### Splits are grouped by episode, not by speaker

The corpus is a **single narrator**, so a speaker-disjoint split is impossible
from this source. Splits are disjoint by **programme episode** instead
(`split.group_by: episode`, 18 groups) — a genuine topic/section holdout, but
train and test still share a voice, so **reported WER/CER are optimistic** and
are not an estimate of real-world WER on an unseen speaker.
`split.speaker_optimistic: true` records that in every metrics row. See
[`PLAN.md`](../PLAN.md) §3 for the full caveat and the recommended 6-fold
grouped cross-validation.

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
