#!/usr/bin/env python3
"""Diagnostic QA: what fraction of each clip's words appear in a reference lexicon.

    python scripts/check_vocab_coverage.py
    python scripts/check_vocab_coverage.py --manifest ../data/asr_manifest_clean.jsonl
    python scripts/check_vocab_coverage.py --limit 40 --threshold 0.3

Read-only, and deliberately *not* wired into the training pipeline: neither
reference CSV has audio attached, so nothing here is training data. This exists
to rank clips by how unfamiliar their transcript looks, so the ones most likely
to hold caption errors float to the top of a manual listening queue.

Low coverage is a *smell*, not a verdict. A clip scores low for any of:

  * a genuine caption error (what we are hunting),
  * a rare or compound Kokborok word the reference lexicon simply lacks,
  * inflection - the lexicon is mostly citation forms, so a legitimately
    inflected word counts as a miss.

The reference lexicons are small (a few thousand types) next to any real
vocabulary, so absolute coverage numbers mean little. The *ranking* is the
useful output.

SCRIPT MISMATCH WARNING: the bundled reference CSVs are Romanized Latin
Kokborok, while data/asr_manifest_clean.jsonl is Bengali script. Latin
vocabulary cannot match Bengali transcripts, so coverage collapses to ~0 and
the ranking carries no signal. The script detects this and says so loudly
rather than printing a table of zeroes that looks like a finding. To get real
signal, one side has to be transliterated into the other's script first.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kokborok_asr.text import normalize  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("check_vocab_coverage")

ASR_DIR = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ASR_DIR / "data" / "reference"

# (path, column holding Kokborok text) - both are text-only, no audio.
REFERENCE_CSVS = (
    (REFERENCE_DIR / "kokborok_navigation_dataset.csv", "Kokborok"),
    (REFERENCE_DIR / "kokborok_lexicon.csv", "kokborok"),
)

# Manifest paths are relative to the REPO ROOT (one level above asr/), not to
# asr/ and not to the manifest's own directory - a row reads
# "data/asr_dataset/clip_0000.wav" while the manifest itself sits in data/.
REPO_ROOT = ASR_DIR.parent
DEFAULT_MANIFEST = REPO_ROOT / "data" / "asr_manifest_clean.jsonl"


def script_of(word: str) -> str:
    """Dominant Unicode script of a word, for the mismatch check."""
    counts: dict[str, int] = {}
    for ch in word:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        tag = name.split()[0] if name else "OTHER"
        tag = {"LATIN": "Latin", "BENGALI": "Bengali", "DEVANAGARI": "Devanagari"}.get(tag, tag.title())
        counts[tag] = counts.get(tag, 0) + 1
    return max(counts, key=counts.get) if counts else "none"


def script_profile(words) -> dict[str, int]:
    profile: dict[str, int] = {}
    for w in words:
        s = script_of(w)
        profile[s] = profile.get(s, 0) + 1
    return profile


def load_vocabulary(csv_specs) -> tuple[set[str], dict[str, int]]:
    """Normalized word set drawn from the Kokborok column of each reference CSV.

    Entries are phrases as often as single words, so every entry is split into
    words - a transcript word matches if it appears anywhere in the lexicon.
    """
    vocab: set[str] = set()
    per_file: dict[str, int] = {}
    for path, column in csv_specs:
        if not path.exists():
            raise SystemExit(
                f"Reference CSV not found: {path}\n"
                f"Expected both reference files under {REFERENCE_DIR}."
            )
        before = len(vocab)
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or column not in reader.fieldnames:
                raise SystemExit(
                    f"{path.name} has no {column!r} column; found: {reader.fieldnames}"
                )
            rows = 0
            for row in reader:
                rows += 1
                # Same normalizer the training targets use - no second
                # implementation to drift out of sync with text.py.
                for word in normalize(row.get(column) or "").split():
                    if word:
                        vocab.add(word)
        per_file[path.name] = len(vocab) - before
        logger.info(
            "%s: %d rows -> %d new vocabulary words (running total %d)",
            path.name, rows, len(vocab) - before, len(vocab),
        )
    return vocab, per_file


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Manifest not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: malformed JSON - {exc}") from exc
    if not rows:
        raise SystemExit(f"No rows in {path}")
    return rows


def score(rows: list[dict], vocab: set[str]) -> list[dict]:
    """Per-clip coverage: fraction of transcript words present in vocab."""
    scored = []
    for row in rows:
        text = row.get("text") or row.get("transcript") or ""
        words = normalize(text).split()
        if not words:
            continue
        missing = [w for w in words if w not in vocab]
        scored.append(
            {
                "audio": row.get("audio") or row.get("audio_path") or "?",
                "text": text,
                "n_words": len(words),
                "n_missing": len(missing),
                "coverage": (len(words) - len(missing)) / len(words),
                "missing": missing,
            }
        )
    # Lowest coverage first; ties broken by longer clips first, since a 1-word
    # clip at 0% is far weaker evidence than a 20-word clip at 0%.
    scored.sort(key=lambda r: (r["coverage"], -r["n_words"]))
    return scored


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST,
        help=f"JSONL manifest with audio/text rows (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3,
        help="Coverage below this counts as 'worth a manual listen' (default: 0.3)",
    )
    parser.add_argument(
        "--limit", type=int, default=10,
        help="How many lowest-coverage clips to list (default: 10; 0 for all)",
    )
    parser.add_argument(
        "--csv-out", type=Path, default=None,
        help="Also write the full ranking to this CSV",
    )
    args = parser.parse_args()

    vocab, per_file = load_vocabulary(REFERENCE_CSVS)
    rows = load_manifest(args.manifest)
    logger.info("Manifest: %s (%d rows)", args.manifest, len(rows))
    scored = score(rows, vocab)

    # --- script sanity check, before reporting anything as a finding ---------
    vocab_scripts = script_profile(vocab)
    manifest_words = [w for r in scored for w in normalize(r["text"]).split()]
    text_scripts = script_profile(manifest_words)
    vocab_main = max(vocab_scripts, key=vocab_scripts.get)
    text_main = max(text_scripts, key=text_scripts.get)

    print()
    print("=" * 72)
    print("REFERENCE VOCABULARY")
    print("=" * 72)
    for name, added in per_file.items():
        print(f"  {name:38s} +{added:6d} words")
    print(f"  {'combined unique vocabulary':38s} ={len(vocab):6d} words")
    print(f"  dominant script: {vocab_main}  {dict(sorted(vocab_scripts.items(), key=lambda kv: -kv[1]))}")
    print(f"  transcript script: {text_main}  {dict(sorted(text_scripts.items(), key=lambda kv: -kv[1]))}")

    mismatch = vocab_main != text_main
    if mismatch:
        print()
        print("!" * 72)
        print(f"!! SCRIPT MISMATCH: lexicon is {vocab_main}, transcripts are {text_main}.")
        print("!! Coverage below is ~0 for mechanical reasons, NOT because the")
        print("!! transcripts are bad, and the ranking carries no diagnostic signal.")
        print("!! Fix by transliterating one side into the other's script first.")
        print("!" * 72)

    total_words = sum(r["n_words"] for r in scored)
    total_hits = sum(r["n_words"] - r["n_missing"] for r in scored)
    mean_cov = sum(r["coverage"] for r in scored) / len(scored)
    below = [r for r in scored if r["coverage"] < args.threshold]
    zero = [r for r in scored if r["coverage"] == 0.0]

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  clips scored                     : {len(scored)}")
    print(f"  mean per-clip coverage           : {mean_cov:.1%}")
    print(f"  corpus-level coverage            : {total_hits}/{total_words} words = "
          f"{(total_hits / total_words if total_words else 0):.1%}")
    print(f"  clips below {args.threshold:.0%} (manual listen) : {len(below)} "
          f"({len(below) / len(scored):.0%} of corpus)")
    print(f"  clips at exactly 0%              : {len(zero)}")

    n = len(scored) if args.limit == 0 else min(args.limit, len(scored))
    print()
    print("=" * 72)
    print(f"LOWEST-COVERAGE CLIPS (worst {n} of {len(scored)})")
    print("=" * 72)
    for i, r in enumerate(scored[:n], 1):
        print(f"{i:3d}. {r['coverage']:6.1%}  {r['n_words'] - r['n_missing']}/{r['n_words']} words  "
              f"{Path(r['audio']).name}")
        print(f"       text   : {r['text']}")
        if r["missing"]:
            shown = " ".join(r["missing"][:12])
            more = f" (+{len(r['missing']) - 12} more)" if len(r["missing"]) > 12 else ""
            print(f"       missing: {shown}{more}")

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["audio", "coverage", "n_words", "n_missing", "text", "missing_words"])
            for r in scored:
                writer.writerow([
                    r["audio"], f"{r['coverage']:.4f}", r["n_words"], r["n_missing"],
                    r["text"], " ".join(r["missing"]),
                ])
        print(f"\nFull ranking written to {args.csv_out}")

    if mismatch:
        print("\nExiting 2: script mismatch makes these numbers non-diagnostic.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
