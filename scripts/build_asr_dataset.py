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
  4. Write clips to data/asr_dataset/ and a manifest to
     data/asr_manifest.jsonl, in the {"audio": ..., "text": ...} format
     scripts/finetune_whisper.py expects.

Usage:
    python3 scripts/build_asr_dataset.py
"""

import json
import re
import sys
from pathlib import Path

import librosa
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from service import config  # noqa: E402

TRANSCRIPT_PATH = config.DATA_DIR / "audio" / "kokborok_transcript.txt"
SOURCE_AUDIO = config.DATA_DIR / "audio" / "Kok-Borok_Words_of_Life_complete.mp3"
OUTPUT_DIR = config.DATA_DIR / "asr_dataset"

TIMESTAMP_RE = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$")
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
        manifest_rows.append({"audio": str(clip_path.relative_to(config.BASE_DIR)), "text": text})

    with open(config.ASR_MANIFEST_PATH, "w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nKept {len(manifest_rows)} clips")
    print(f"Skipped {skipped_nonspeech} non-speech (music/laughter/empty) entries")
    print(f"Skipped {skipped_duration} entries outside [{MIN_SEC}, {MAX_SEC}]s duration")
    print(f"Manifest written to {config.ASR_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
