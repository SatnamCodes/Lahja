"""
Pick XTTS reference clips by actual speaker-embedding consistency, not just
duration. scripts/prepare_audio.py originally grabbed the first 3 segments
near ~12s regardless of whether they're even the same speaker - GRN's
dramatized "Words of Life" narrations often mix a narrator with character
voices, and XTTS blending inconsistent speakers into one embedding actively
hurts clone quality.

This embeds every candidate segment with a real speaker-verification model
(resemblyzer), finds the medoid (the clip most similar to all others - i.e.
the most "typical" speaker in the set), and keeps only clips that are
tightly clustered around it - which should reliably isolate the single
recurring narrator.

Usage:
    python3 scripts/select_reference_clips.py
"""

import sys
from pathlib import Path

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from service import config  # noqa: E402

SEGMENTS_DIR = config.DATA_DIR / "audio_segments"
MIN_SEC, MAX_SEC = 6.0, 18.0
NUM_REFERENCE_CLIPS = 3
SIMILARITY_FLOOR = 0.75  # below this, a clip is probably a different speaker


def main():
    import soundfile as sf

    candidates = []
    for wav_path in sorted(SEGMENTS_DIR.rglob("*.wav")):
        if "song" in wav_path.parent.name.lower():
            continue
        info = sf.info(str(wav_path))
        duration = info.frames / info.samplerate
        if MIN_SEC <= duration <= MAX_SEC:
            candidates.append((wav_path, duration))

    if len(candidates) < NUM_REFERENCE_CLIPS:
        raise SystemExit(f"Only {len(candidates)} candidate clips in range - need more segments.")

    print(f"Embedding {len(candidates)} candidate clips with resemblyzer...")
    encoder = VoiceEncoder()
    embeddings = []
    for wav_path, _ in candidates:
        wav = preprocess_wav(str(wav_path))
        embeddings.append(encoder.embed_utterance(wav))
    embeddings = np.stack(embeddings)

    # Cosine similarity matrix
    norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sim = norm @ norm.T

    # Medoid = clip with highest mean similarity to all others (most "typical" speaker)
    mean_sim = sim.mean(axis=1)
    medoid_idx = int(np.argmax(mean_sim))
    medoid_path = candidates[medoid_idx][0]
    print(f"\nMedoid (most typical speaker): {medoid_path.relative_to(config.BASE_DIR)}")

    sims_to_medoid = sim[medoid_idx]
    ranked = sorted(range(len(candidates)), key=lambda i: -sims_to_medoid[i])

    print("\nTop candidates by similarity to medoid:")
    kept = []
    for i in ranked[:15]:
        path, duration = candidates[i]
        flag = "" if sims_to_medoid[i] >= SIMILARITY_FLOOR else "  <- below floor, likely different speaker"
        print(f"  {sims_to_medoid[i]:.3f}  {duration:5.1f}s  {path.relative_to(config.BASE_DIR)}{flag}")
        if sims_to_medoid[i] >= SIMILARITY_FLOOR:
            kept.append(path)

    n_below_floor = sum(1 for i in range(len(candidates)) if sims_to_medoid[i] < SIMILARITY_FLOOR)
    print(f"\n{n_below_floor}/{len(candidates)} candidates fall below the similarity floor "
          f"({SIMILARITY_FLOOR}) - likely a second speaker or noisy segment mixed into the recordings.")

    # Prefer diversity of source episode among the qualifying clips, so the
    # reference set covers more phonetic variety instead of 3 adjacent
    # seconds of the same recording.
    picked = []
    seen_tracks = set()
    for path in kept:
        track = path.parent.name
        if track in seen_tracks:
            continue
        picked.append(path)
        seen_tracks.add(track)
        if len(picked) == NUM_REFERENCE_CLIPS:
            break
    if len(picked) < NUM_REFERENCE_CLIPS:
        picked = kept[:NUM_REFERENCE_CLIPS]  # not enough distinct tracks - fall back to top matches
    if len(picked) < NUM_REFERENCE_CLIPS:
        raise SystemExit("Not enough consistent same-speaker clips found above the similarity floor.")

    # Clear out the old auto-picked references, replace with the vetted set.
    for old in config.AUDIO_REF_DIR.glob("reference_*.wav"):
        old.unlink()

    print(f"\nWriting {len(picked)} vetted reference clips to {config.AUDIO_REF_DIR}:")
    for i, src in enumerate(picked):
        dest = config.AUDIO_REF_DIR / f"reference_{i:02d}.wav"
        audio, sr = sf.read(str(src))
        sf.write(str(dest), audio, sr)
        print(f"  {dest.name}  <- {src.relative_to(config.BASE_DIR)}")


if __name__ == "__main__":
    main()
