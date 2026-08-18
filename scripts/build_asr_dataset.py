"""
Build a Whisper fine-tuning manifest from data/audio/kokborok_transcript.txt,
a timestamped transcript aligned to data/audio/Kok-Borok_Words_of_Life_complete.mp3.

CAVEAT (read before trusting the output): this transcript contains bracketed
non-speech tags ([মিউজিক] "music", [সশব্দ হাসি] "audible laughter") and at
least one line that randomly switches into Telugu script mid-sentence -
both are hallmarks of an auto-generated captioning tool (e.g. YouTube
auto-captions), not a human transcriber. Until a human confirms this
transcript is accurate, treat any model fine-tuned on it as trained on
plausibly-noisy pseudo-labels, not verified ground truth.

What this script does:
  1. Parse timestamp lines (MM:SS or H:MM:SS) and the caption text that
     follows each one.
  2. Drop bracketed-only captions (non-speech tags) and anything empty.
  3. Slice data/audio/Kok-Borok_Words_of_Life_complete.mp3 at each
     [timestamp, next_timestamp) span, skipping spans that are too short
     or implausibly long (missing timestamp markers).
  4. Attribute each clip to one of the 18 "Words of Life" episodes by where
     its timestamp falls in the concatenated programme (see EPISODE GROUPS
     below), and emit that as "speaker_id".
  5. Write clips to data/asr_dataset/ and a manifest to
     data/asr_manifest.jsonl, in the {"audio": ..., "text": ...} format
     scripts/finetune_whisper.py expects, plus the "speaker_id" group tag.

EPISODE GROUPS (why "speaker_id" holds an episode number):
    The whole programme is one narrator, so a speaker-disjoint train/val/test
    split is impossible from this source alone. The 18 episodes are used as
    split *groups* instead: holding out whole episodes gives topic- and
    section-disjoint evaluation, which is a real generalization test even
    though the voice is shared. Populating "speaker_id" with an episode id
    lets asr/'s existing speaker-disjoint splitter do that unchanged, with no
    leaky code path enabled. The field name is inherited from that splitter
    and is deliberately not renamed yet - see PLAN.md sections 3 and 5.5.

    Episode boundaries are cumulative durations of the 18 per-episode mp3s.
    That is valid because those 18 files are exactly the complete recording,
    in filename order: verified by cross-correlating per-episode energy
    envelopes against the full hour (r = 0.975-0.999, consecutive gaps
    <= 0.05s, 001 starting at 0.00s and 018 ending at 3683.16s of a 3683.15s
    file). WER from this data is still speaker-optimistic; PLAN.md section 3
    carries the caveat that must accompany any number derived from it.

Usage:
    python3 scripts/build_asr_dataset.py
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import librosa
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from service import config  # noqa: E402

TRANSCRIPT_PATH = config.DATA_DIR / "audio" / "kokborok_transcript.txt"
SOURCE_AUDIO = config.DATA_DIR / "audio" / "Kok-Borok_Words_of_Life_complete.mp3"
OUTPUT_DIR = config.DATA_DIR / "asr_dataset"
# The 18 per-episode files. Zero-padded numbering means filename order is
# also numeric order, which is the order they were concatenated in.
EPISODE_GLOB = "Kok-Borok Words of Life *.mp3"

TIMESTAMP_RE = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$")
EPISODE_NUM_RE = re.compile(r"Words of Life (\d{3})\b")
MIN_SEC, MAX_SEC = 1.0, 20.0


def parse_timestamp(line: str):
    m = TIMESTAMP_RE.match(line.strip())
    if not m:
        return None
    a, b, c = m.groups()
    if c is None:
        return int(a) * 60 + int(b)
    return int(a) * 3600 + int(b) * 60 + int(c)


def is_real_speech(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    # Drop lines that are only bracketed tags, e.g. "[মিউজিক]" or "[সশব্দ হাসি]"
    stripped = re.sub(r"\[[^\]]*\]", "", text).strip()
    return bool(stripped)


def episode_boundaries():
    """[(episode_id, start_sec, end_sec)] for the 18 episodes, in order.

    Durations come from the mp3 headers (no decode). Fails loudly rather than
    guessing: a missing episode file or an unparsable number would silently
    shift every later boundary and mislabel whole stretches of the corpus.
    """
    paths = sorted((config.DATA_DIR / "audio").glob(EPISODE_GLOB))
    if not paths:
        raise SystemExit(
            f"No per-episode mp3s matching {EPISODE_GLOB!r} in {config.DATA_DIR / 'audio'}; "
            f"cannot attribute clips to episode groups."
        )
    bounds = []
    cursor = 0.0
    for path in paths:
        m = EPISODE_NUM_RE.search(path.name)
        if not m:
            raise SystemExit(f"Cannot read an episode number from {path.name!r}")
        duration = sf.info(str(path)).duration
        bounds.append((f"ep{m.group(1)}", cursor, cursor + duration))
        cursor += duration
    return bounds


def episode_for(seconds: float, bounds) -> str:
    """Which episode a programme-relative timestamp falls in."""
    for episode_id, start, end in bounds:
        if start <= seconds < end:
            return episode_id
    # Past the last boundary: the concatenation is ~0.7s shorter than the sum
    # of the episode durations (per-file mp3 frame padding), so a timestamp in
    # the final fraction of a second legitimately lands here.
    return bounds[-1][0]


def parse_transcript():
    lines = TRANSCRIPT_PATH.read_text(encoding="utf-8").splitlines()
    entries = []  # (start_sec, text)
    pending_start = None
    pending_text_lines = []

    def flush():
        if pending_start is not None:
            text = " ".join(pending_text_lines).strip()
            entries.append((pending_start, text))

    for line in lines:
        ts = parse_timestamp(line)
        if ts is not None:
            flush()
            pending_start = ts
            pending_text_lines = []
        elif line.strip() and line.strip() != "TRANSCRIPT":
            pending_text_lines.append(line.strip())
    flush()

    return entries


def main():
    if not TRANSCRIPT_PATH.exists():
        raise SystemExit(f"No transcript at {TRANSCRIPT_PATH}")
    if not SOURCE_AUDIO.exists():
        raise SystemExit(f"No source audio at {SOURCE_AUDIO}")

    entries = parse_transcript()
    print(f"Parsed {len(entries)} timestamped entries from transcript")

    bounds = episode_boundaries()
    print(f"Episode groups: {len(bounds)} episodes spanning {bounds[-1][2]:.1f}s "
          f"({bounds[0][0]}..{bounds[-1][0]})")

    print(f"Loading {SOURCE_AUDIO.name}...")
    audio, sr = librosa.load(str(SOURCE_AUDIO), sr=16000, mono=True)
    total_duration = len(audio) / sr

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    skipped_nonspeech = 0
    skipped_duration = 0

    for i, (start, text) in enumerate(entries):
        end = entries[i + 1][0] if i + 1 < len(entries) else total_duration
        duration = end - start

        if not is_real_speech(text):
            skipped_nonspeech += 1
            continue
        if duration < MIN_SEC or duration > MAX_SEC:
            skipped_duration += 1
            continue

        start_sample = int(start * sr)
        end_sample = int(min(end, total_duration) * sr)
        clip = audio[start_sample:end_sample]

        clip_path = OUTPUT_DIR / f"clip_{i:04d}.wav"
        sf.write(str(clip_path), clip, sr)
        manifest_rows.append({
            "audio": str(clip_path.relative_to(config.BASE_DIR)),
            "text": text,
            # Split group, not a real speaker - see EPISODE GROUPS above.
            "speaker_id": episode_for(start, bounds),
        })

    with open(config.ASR_MANIFEST_PATH, "w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nKept {len(manifest_rows)} clips")
    print(f"Skipped {skipped_nonspeech} non-speech (music/laughter/empty) entries")
    print(f"Skipped {skipped_duration} entries outside [{MIN_SEC}, {MAX_SEC}]s duration")
    print(f"Manifest written to {config.ASR_MANIFEST_PATH}")

    counts = Counter(row["speaker_id"] for row in manifest_rows)
    print(f"\nClips per episode group ({len(counts)} of {len(bounds)} episodes represented):")
    for episode_id, _, _ in bounds:
        print(f"  {episode_id}: {counts.get(episode_id, 0)}")


if __name__ == "__main__":
    main()
