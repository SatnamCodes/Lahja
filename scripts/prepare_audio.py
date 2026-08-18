"""
Segment the long Kok-Borok "Words of Life" mp3s in data/audio/ into short,
utterance-length clips - useful two ways:

1. A handful of clean segments get copied straight into data/audio/*.wav as
   XTTS zero-shot voice-cloning reference clips (no transcript needed).
2. Every segment is written to data/audio_segments/ and listed in
   data/asr_manifest_template.jsonl with an empty "text" field, ready for a
   Kokborok speaker to fill in. Segmenting first turns "transcribe a 3
   minute sermon" into "transcribe fifteen 5-10s clips", which is the
   difference between impractical and doable by hand.

Tracks with "Song" in the title are skipped for the XTTS reference picks
(singing is a bad voice-cloning reference) but still segmented and listed
in the manifest template in case their spoken intros are useful.

Usage:
    python3 scripts/prepare_audio.py
"""

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from service import config  # noqa: E402

SEGMENTS_DIR = config.DATA_DIR / "audio_segments"
MANIFEST_TEMPLATE_PATH = config.DATA_DIR / "asr_manifest_template.jsonl"

MIN_SEGMENT_SEC = 3.0
MAX_SEGMENT_SEC = 15.0
TOP_DB = 30  # silence threshold; lower = stricter about what counts as silence
NUM_REFERENCE_CLIPS = 3
MIN_REFERENCE_SEC = 8.0  # cleaner, longer clips make better voice-cloning refs


def merge_intervals(intervals: np.ndarray, sr: int) -> list[tuple[int, int]]:
    """Merge librosa's non-silent intervals into utterance-length chunks."""
    min_len = int(MIN_SEGMENT_SEC * sr)
    max_len = int(MAX_SEGMENT_SEC * sr)
    merged = []
    start, end = None, None
    for s, e in intervals:
        if start is None:
            start, end = s, e
            continue
        if e - start <= max_len:
            end = e
        else:
            if end - start >= min_len:
                merged.append((start, end))
            start, end = s, e
    if start is not None and end - start >= min_len:
        merged.append((start, end))
    return merged


def main():
    mp3_files = sorted(config.AUDIO_REF_DIR.glob("*.mp3"))
    if not mp3_files:
        raise SystemExit(f"No mp3 files found in {config.AUDIO_REF_DIR}")

    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    reference_candidates = []

    for mp3_path in mp3_files:
        is_song = "song" in mp3_path.stem.lower()
        print(f"Processing {mp3_path.name} (song={is_song})...")
        audio, sr = librosa.load(str(mp3_path), sr=16000, mono=True)
        intervals = librosa.effects.split(audio, top_db=TOP_DB)
        if len(intervals) == 0:
            print(f"  no non-silent audio detected, skipping")
            continue
        chunks = merge_intervals(intervals, sr)

        stem_dir = SEGMENTS_DIR / mp3_path.stem.replace(" ", "_")
        stem_dir.mkdir(parents=True, exist_ok=True)
        for i, (start, end) in enumerate(chunks):
            seg_path = stem_dir / f"seg_{i:03d}.wav"
            sf.write(str(seg_path), audio[start:end], sr)
            duration = (end - start) / sr
            manifest_rows.append({"audio": str(seg_path.relative_to(config.BASE_DIR)), "text": ""})
            if not is_song and duration >= MIN_REFERENCE_SEC:
                reference_candidates.append((duration, seg_path))
        print(f"  wrote {len(chunks)} segments to {stem_dir}")

    with open(MANIFEST_TEMPLATE_PATH, "w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(manifest_rows)} segments to manifest template: {MANIFEST_TEMPLATE_PATH}")
    print("Fill in the \"text\" field for each line (a Kokborok speaker listens to the")
    print("clip and types what's said), then rename to data/asr_manifest.jsonl and run")
    print("scripts/finetune_whisper.py.")

    # Pick a spread of clean, mid-length clips as XTTS reference audio.
    reference_candidates.sort(key=lambda t: abs(t[0] - 12.0))  # prefer clips near ~12s
    picked = reference_candidates[:NUM_REFERENCE_CLIPS]
    if not picked:
        print("\nNo clips long/clean enough for XTTS reference audio - skipping that step.")
        return

    print(f"\nCopying {len(picked)} clips into {config.AUDIO_REF_DIR} as XTTS reference audio:")
    for i, (duration, seg_path) in enumerate(picked):
        dest = config.AUDIO_REF_DIR / f"reference_{i:02d}.wav"
        audio, sr = librosa.load(str(seg_path), sr=None)
        sf.write(str(dest), audio, sr)
        print(f"  {dest.name}  <- {seg_path.relative_to(config.BASE_DIR)}  ({duration:.1f}s)")


if __name__ == "__main__":
    main()
