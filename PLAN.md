# Lahja ASR — data state and training plan

Status as of 2026-08-18. This file records what the Kokborok ASR training data
actually is (verified, not assumed), the split decision forced by having one
speaker, and the specific code changes needed to feed it into `asr/`.

Nothing in `asr/` has been changed yet. This is the plan, not the changelog.

---

## 1. Confirmed data state

**Training data: [`data/asr_manifest_clean.jsonl`](data/asr_manifest_clean.jsonl) — 235 clips, 32.9 minutes.**

| | rows | duration |
| --- | --- | --- |
| `data/asr_manifest.jsonl` (all captioned speech) | 273 | 36.7 min |
| dropped: non-Bengali script corruption | 38 | 3.8 min |
| **`data/asr_manifest_clean.jsonl` (use this)** | **235** | **32.9 min** |

Row format is `{"audio": "data/asr_dataset/clip_NNNN.wav", "text": "..."}`, paths
relative to the **repo root**. Clips are 16 kHz mono WAV, 1.0–19.0 s
(mean 8.1 s), regenerable from source via
[`scripts/build_asr_dataset.py`](scripts/build_asr_dataset.py) — the WAVs
themselves are gitignored ([`.gitignore:229`](.gitignore#L229)).

### Provenance: one recording, one voice

All 18 episode MP3s and `Kok-Borok_Words_of_Life_complete.mp3` are **the same
underlying audio**. Verified by cross-correlating per-episode energy envelopes
against the full hour: each episode matches at r = 0.975–0.999 (best competing
peak ≤ 0.25), at monotonically increasing offsets that tile the complete file
end to end with gaps ≤ 0.05 s (MP3 frame padding). Episode 001 starts at 0.00 s,
episode 018 ends at 3683.16 s, and the file is 3683.15 s long. **There is no
audio in the episodes absent from the complete file, and none in the complete
file absent from the episodes.**

One caveat on how far that evidence reaches: cross-correlation proves there is a
single *recording*, so no extra speakers are hiding in the per-episode files. It
does not by itself prove a single *voice within* that recording. A separate
acoustic check supports the single-narrator reading — median F0 across all 235
clips is 188 Hz with a clean unimodal distribution (IQR 172–204 Hz, no second
mode), and spectral centroid is tight (median 1804 Hz, IQR 1677–2001). That is
consistent with one narrator throughout. It is a heuristic, not speaker
verification; if a proper embedding pass (e.g. `resemblyzer`, already in the root
`requirements.txt`) ever reveals 2+ distinct voices, revisit §3 — that would make
a genuine speaker-disjoint split possible and this whole section moot.

**Working assumption: one speaker, `spk_all`.**

### Script: Bengali, not Romanized Latin

The transcripts are **Bengali script** (Kokborok is written in both Bengali and
Latin; this source uses Bengali). All 235 rows are pure Bengali script — the 38
rows carrying Devanagari, Telugu, or stray Latin were removed to produce the
clean manifest.

This contradicts an assumption currently written into three places in `asr/`,
all of which say the transcripts are Romanized Latin:
`configs/whisper_small_lora.yaml` (`model.language: en`), the
[`text.py`](asr/src/kokborok_asr/text.py) module docstring, and
[`asr/README.md`](asr/README.md). See §4 for the fixes.

**The Whisper proxy language token must be `bn`, not `en`.** Whisper has no
`trp` token; Bengali is the correct proxy for Bengali-script output, and it is a
far better one than English — `bn` shares the target script and much of the
subword inventory, so the tokenizer emits a sane number of tokens per word
instead of byte-fallback soup.

### Corpus size reality check

32.9 minutes is small, and the text is sparser than the duration suggests: 2,239
word tokens, 1,193 unique, **75% of the vocabulary appears exactly once**. Mean
9.5 words per utterance. Expect the model to learn acoustics and phonotactics,
not vocabulary.

---

## 2. Known gaps

Coverage of the 61.4-minute source program is **54%** (32.9 min labeled). The
remaining 28.5 min is 3.8 min dropped for script corruption plus 24.7 min never
captioned as speech at all — the 138 `[মিউজিক]` / `[সশব্দ হাসি]` caption entries
plus 3 over-long spans. Much of that genuinely is music (two of the 18 sections
are titled "Song"), so it is not all recoverable.

Coverage is very uneven across the program:

| Well covered | Thinly covered |
| --- | --- |
| 002 *Jesus the Mighty One* 96% · 011 *What is a Christian* 96% · 016 *The Prodigal Son* 92% · 005 *Stronger than Demons* 84% | **007 *On Calvary's Hill* 5%** · 017 *The Prodigal Returns* 10% · 010 *Hear the Gospel Call* 11% · 003 *Song* 22% · 009 *Jesus is Coming Again* 27% |

**Episode 007 "On Calvary's Hill" is the one place hand-retranscription adds
real data.** It had 14 captioned utterances; 13 were script-corrupted (the
auto-captioner switched to a Devanagari model mid-section) and were dropped,
leaving **one usable clip — 0.1 min out of 3.3 min of audio**. Because §1
establishes the episodes are the same audio as the complete file, re-segmenting
anything recovers no new audio: the only way to add data from this source is a
human typing transcripts. Episode 007 is where that effort has the highest
yield, followed by 017 and 010.

The 208 empty-text rows in
[`data/asr_manifest_template.jsonl`](data/asr_manifest_template.jsonl) are a
to-do list for exactly that work, generated once by
[`scripts/prepare_audio.py`](scripts/prepare_audio.py) and never filled in. No
per-episode transcript has ever existed in this repo — no `.srt`, `.vtt`, or
per-episode `.txt` on disk, and nothing matching them in any commit on any
branch. Those 208 segments cover the same audio as the clean manifest, so they
duplicate rather than extend it; their value is as annotation targets for the
uncovered 46%, not as a second data source.

---

## 3. Decision: how to split 235 clips from one speaker

### The constraint

[`splitting.py`](asr/src/kokborok_asr/splitting.py) is speaker-disjoint by
design and refuses to build a leaky split by accident. With one speaker it
cannot produce a 3-way split: `split_by_speaker()` raises `SplitError` when
`n_speakers < len(active_splits)`, and 1 < 3. The module's own docstring states
the reason — an utterance-level split on a small corpus means "the model
recognizes the speaker, not the language."

### Decision: train now, with an episode-grouped split — not a random utterance split, and not waiting

**Do not wait for more speakers.** Waiting yields nothing measurable, and 32.9
minutes is workable for the intended method: LoRA on Whisper-small is adapting
~1% of parameters, not learning a language from scratch. Treat this run as a
bootstrap that produces a baseline number and a working pipeline, not a
shippable model.

**But do not use a plain random utterance-level split either.** On this corpus
that is the worst available option, and worse than the generic warning in
`splitting.py` implies. The 235 clips are *contiguous slices of one continuous
recording*, cut at caption boundaries. A random shuffle puts clip 0042 in train
and clip 0043 in test — adjacent seconds of the same sentence, sharing speaker,
room acoustics, microphone, topic, and often overlapping words. That leaks far
more than speaker identity.

**Use the 18 episode sections as split groups instead.** Every clip's timestamp
maps to exactly one episode (the boundaries are established in §1), so each clip
gets a group id like `ep007`. Holding out whole episodes gives topic- and
section-disjoint evaluation for free, which is a real generalization test even
though the voice is shared.

The elegant part: this needs **no new splitting code and no leaky code path**.
Populate `Utterance.speaker_id` with the episode id and the existing
`split_by_speaker()` greedy largest-first assignment works unchanged, produces a
valid 3-way split from 18 groups, and keeps the `summarize()` leak check
meaningful (it will correctly report zero *group* overlap).
**`split.allow_speaker_overlap` stays `false`** — the fallback
`_utterance_level_split()` is never entered, and nobody has to remember to turn
a leak flag back off later.

### Caveat that must be reported with every number from this data

> WER and CER from this corpus are **optimistic**. Train and test share a single
> speaker, recording session, microphone, and acoustic environment, so the model
> can partially memorize this narrator's voice. Splits are disjoint by program
> section, not by speaker. These numbers measure *"can it generalize to unseen
> passages from this one narrator"* — they are a valid relative signal for
> comparing runs, and **not** an estimate of real-world Kokborok WER on an unseen
> speaker. Any external claim about accuracy needs a held-out speaker that does
> not exist yet.

This caveat belongs in the run record, not just in this file: write the group
granularity into `results/metrics.jsonl` for every run (see §4).

### Test-set size problem, and the fix

A single 70/15/15 split over 18 unevenly sized groups leaves a test set of
roughly 30–40 clips (~5 min) — too small for a stable WER, and its composition
would be dominated by whichever episodes happened to land in it.

**Recommended: 6-fold grouped cross-validation over the 18 episodes** (3
episodes per fold) once the pipeline runs end to end. Every clip gets evaluated
exactly once, the reported WER averages over the whole corpus instead of one
lucky 15%, and fold-to-fold variance becomes an honest error bar. Cost is 6
training runs; on 32.9 minutes with LoRA that is cheap. Use the single
70/15/15 split for the first smoke run and iteration, then report
cross-validated numbers.

Also report a bootstrap confidence interval over utterances alongside any point
WER. On ~40 test clips the CI will be wide, and showing it is more honest than a
bare percentage.

---

## 4. Path forward

Phased, so each step produces something checkable.

**Phase 1 — wire up the data (no training).** Make `manifest.py` read the clean
manifest (§5), add episode group ids, run `scripts/prepare_data.py`, and confirm
the split summary shows 235 utterances, 18 groups, zero group overlap, and
sensible per-split hours. Deliverable: `asr/data/processed/*.jsonl`.

**Phase 2 — config fixes.** In `asr/configs/whisper_small_lora.yaml`:

- `model.language: en` → **`bn`**. Update the neighbouring comment: the proxy is
  Bengali because the transcripts *are* Bengali script, not because it is
  phonetically adjacent.
- `data.speaker_strategy: parent_dir` → **`manifest_field`** (new, §5), or
  `single` if episode grouping is deferred. **Do not leave it at `parent_dir`** —
  see the silent-failure note in §5.
- `data.audio_root` (new, §5) → repo root, so `data/asr_dataset/...` resolves.
- `text.punctuation`: add the Bengali danda `।` and double danda `॥`. The current
  235 rows contain **zero** punctuation, so this is a no-op today, but any
  hand-typed transcript will use `।`, and `normalize()` would otherwise leave it
  glued to the preceding word and inflate WER.
- Record the split granularity in metrics: log `group_by: episode` and
  `n_groups: 18` so a future reader of `results/metrics.jsonl` can tell these
  numbers are section-disjoint, not speaker-disjoint.
- `text.lowercase: true` can stay — it is a harmless no-op on Bengali script
  (verified: no row changes under `.lower()`).
- `data.max_duration: 30.0` / `min_duration: 0.2` need no change; clips are
  1.0–19.0 s.

**Phase 3 — baseline run.** Zero-shot Whisper-small with `language: bn` on the
test fold *before* fine-tuning, so the LoRA run has something to beat. Expect a
poor number; record it anyway.

**Phase 4 — LoRA fine-tune**, then 6-fold grouped CV per §3.

**Phase 5 — grow the data.** In priority order:

1. Hand-transcribe episode **007** (3.3 min, currently 1 clip), then **017**
   and **010**. Highest yield per minute of human effort; the segment WAVs are
   regenerable via `scripts/prepare_audio.py`.
2. Hand-correct the 38 dropped rows rather than discarding them — the audio is
   fine, only the caption text is corrupt, so this recovers 3.8 min cheaply.
3. **Record a second speaker.** This is the only change that makes a real
   speaker-disjoint split — and therefore a defensible WER — possible at all.
   Even 10 minutes from one additional voice, reserved entirely as a test set,
   is worth more for evaluation than another hour from this narrator.

---

## 5. Changes needed in `asr/src/kokborok_asr/manifest.py`

Goal: accept `data/asr_manifest_clean.jsonl` directly as a third input format,
alongside the existing `manifest` (CSV/TSV) and `pairs` layouts, without
disturbing either.

### 5.1 New `jsonl` layout

- Add `"asr_manifest_clean.jsonl"`, `"asr_manifest.jsonl"`, and
  `"manifest.jsonl"` to the auto-detected names (`MANIFEST_NAMES`, or a parallel
  `JSONL_MANIFEST_NAMES`), and accept `layout: "jsonl"` in `discover()`
  alongside `auto` / `manifest` / `pairs`. In `auto`, dispatch on the suffix of
  whichever manifest is found rather than assuming CSV.
- Add `_load_from_jsonl(manifest_path, raw_dir, strategy, delimiter, *, audio_root, default_speaker)`
  mirroring `_load_from_manifest`: read line by line, skip blank lines,
  `json.loads` each. **Raise `CorpusError` naming the 1-based line number on
  malformed JSON** — there is no JSON path in this module today, and a bare
  `JSONDecodeError` from line 187 of a 235-line file is a bad debugging
  experience.

### 5.2 Field-name mapping (the reason a new loader is needed at all)

The clean manifest uses `audio` / `text`; `Utterance` and the CSV loader use
`audio_path` / `transcript`. Accept aliases:

| `Utterance` field | accepted JSONL keys |
| --- | --- |
| `audio_path` | `audio`, `audio_path`, `path`, `wav` |
| `transcript` | `text`, `transcript`, `sentence` |
| `speaker_id` | `speaker_id`, `speaker` (optional; falls back to strategy) |

Reject a row that has neither an audio key nor a text key with a `CorpusError`
that lists the keys actually present — silently skipping such rows would let a
schema typo shrink the corpus without anyone noticing.

### 5.3 `audio_root` — the trap that would break this silently

**This is the one change most likely to be missed.** Paths in
`asr_manifest_clean.jsonl` are relative to the **repo root**
(`data/asr_dataset/clip_0000.wav`), while the manifest file itself lives in
`<repo>/data/`. `_load_from_manifest` resolves relative paths against
`manifest_path.parent`, which here yields
`<repo>/data/data/asr_dataset/clip_0000.wav` — wrong for all 235 rows. Nothing
raises: `validate()` drops every row as "audio file missing", and `discover()`
then fails with the generic "No usable utterances found", pointing at the wrong
problem entirely.

Add an explicit `audio_root: Path | None` parameter (new config key
`data.audio_root`), defaulting to `manifest_path.parent` to preserve current CSV
behaviour, and set it to the repo root for this manifest.

### 5.4 Reading a manifest that lives outside `raw_dir`

`asr/`'s `project_root()` is the `asr/` directory, so `paths.raw_dir: data/raw`
means `asr/data/raw` — while this manifest lives at `<repo>/data/`. Two knock-on
effects:

- `discover()` only ever looks inside `raw_dir`. Add an optional
  `manifest_path: Path | None = None` parameter (new config key
  `paths.manifest_file`, e.g. `../../data/asr_manifest_clean.jsonl`) that
  bypasses `_find_manifest()` when set. Preferred over symlinking, which breaks
  differently on other machines. The `data/raw` read-only invariant is
  untouched — this only ever reads.
- `make_utt_id()` computes a path relative to `raw_dir`; for clips outside it,
  `relative_to` raises and it silently falls back to `Path(audio_path.name)`, so
  ids become `clip_0000` … `clip_0410`. That happens to be safe here because all
  clips sit in one flat directory with unique names — but it is fragile: add a
  second source with its own `clip_0000.wav` and the ids collide, at which point
  `validate()` discards the duplicates as "duplicate utt_id" and the corpus
  quietly shrinks. Add a `data.utt_id_prefix` (or derive a prefix from the
  manifest stem) and namespace the ids now, before there is a second source.

### 5.5 Speaker attribution — another silent failure

Leaving `speaker_strategy: parent_dir` (the current config default) does **not**
raise on this data. Because the clips are outside `raw_dir`, `_speaker_from_path`
hits the `except ValueError` branch, falls back to the full path, and returns
`rel.parts[-2]` — so every clip gets the speaker id **`asr_dataset`**. One
speaker, no error, a meaningless id, and the `parent_dir` guard that exists
precisely to catch this never fires because the path *does* have ≥ 2 parts.

`filename_prefix` fails the same way: `clip_0000` contains the default `_`
delimiter, so it splits happily and returns **`clip`** for every row. Both were
checked against the real resolved paths — neither raises, and both collapse the
corpus to one bogus group id. This must be set explicitly; there is no default
here that fails loudly.

Two options:

- **Minimum:** set `speaker_strategy: single` → every clip becomes `spk_all`.
  Honest, and blocks a 3-way split exactly as `splitting.py` intends.
- **Recommended, per §3:** add a `manifest_field` strategy that trusts a
  `speaker_id` present in the row, and have
  [`scripts/build_asr_dataset.py`](scripts/build_asr_dataset.py) write
  `"speaker_id": "ep007"` per clip, derived from its timestamp and the episode
  boundaries. §5.2 already reads the field, so **`manifest.py` needs no episode
  logic of its own** — the audio→episode attribution belongs in the builder,
  which is the only place that knows the timestamps.

Note the naming friction this creates: `speaker_id` would hold an episode id,
and `summarize()` would report `"speakers": ["ep001", ...]`. Renaming the field
to `group_id` throughout (`Utterance`, `splitting.py`, `summarize()`) with
`speaker_id` kept as a read alias is cleaner and makes the §3 caveat structurally
obvious rather than a comment someone has to find. It is a wider change than
this task needs, so it is deliberately deferred — but it should be done before
a second real speaker arrives and the overloaded field becomes ambiguous.

### 5.6 What needs no change

- `validate()` — no empty transcripts in the clean manifest, no duplicate ids
  under the current flat layout.
- `probe_duration()` — reads WAV headers via `soundfile`, works as-is.
- `Utterance.to_json` / `from_json` — the internal schema is fine; only the
  input adapter changes.
- The `pairs` and CSV `manifest` loaders — untouched, and worth keeping a test
  on to prove the new layout did not disturb them.
