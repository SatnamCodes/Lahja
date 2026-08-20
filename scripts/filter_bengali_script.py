"""Drop rows whose transcript contains non-Bengali script, producing the
clean training manifest from the raw one.

    python3 scripts/filter_bengali_script.py

Kokborok is written in Bengali or Latin script; this source uses Bengali. The
auto-generated transcript behind data/asr_manifest.jsonl repeatedly switches
into the wrong script mid-programme - Devanagari for a long stretch of episode
007, plus scattered Telugu and bare English words - which is a captioning
artifact, not dialect variation. Those rows have good audio and corrupt text,
so they are removed from training rather than trusted as pseudo-labels.

Any row containing at least one alphabetic character outside the Bengali block
is dropped whole. Dropping the row rather than the offending word is
deliberate: a caption that switched scripts mid-sentence is unreliable
throughout, not just where the script changed.

The dropped rows are recoverable data, not garbage - re-transcribing them by
hand is the cheapest way to grow this corpus, since the audio is already
segmented. See PLAN.md section 2.
"""

import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from service import config  # noqa: E402

INPUT_PATH = config.ASR_MANIFEST_PATH
OUTPUT_PATH = config.DATA_DIR / "asr_manifest_clean.jsonl"

BENGALI_RANGE = (0x0980, 0x09FF)
# Ranges seen in this transcript, named so the report says something useful.
KNOWN_RANGES = {
    "Bengali": (0x0980, 0x09FF),
    "Devanagari": (0x0900, 0x097F),
    "Telugu": (0x0C00, 0x0C7F),
}


def script_tag(ch: str) -> str:
    code = ord(ch)
    for name, (lo, hi) in KNOWN_RANGES.items():
        if lo <= code <= hi:
            return name
    if ch.isascii():
        return "Latin"
    return unicodedata.name(ch, "UNKNOWN").split()[0].title()


def offending_scripts(text: str) -> set[str]:
    """Scripts present in ``text`` other than Bengali, ignoring non-letters."""
    found = set()
    for ch in text:
        if not ch.isalpha():
            continue
        if BENGALI_RANGE[0] <= ord(ch) <= BENGALI_RANGE[1]:
            continue
        found.add(script_tag(ch))
    return found


def main():
    if not INPUT_PATH.exists():
        raise SystemExit(f"No manifest at {INPUT_PATH}; run scripts/build_asr_dataset.py first")

    kept, dropped = [], []
    with open(INPUT_PATH, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{INPUT_PATH}:{lineno}: malformed JSON - {exc}") from exc
            bad = offending_scripts(row.get("text", ""))
            (dropped if bad else kept).append((row, bad))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        for row, _ in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Read    {len(kept) + len(dropped)} rows from {INPUT_PATH.name}")
    print(f"Dropped {len(dropped)} rows containing non-Bengali script")
    for scripts, n in Counter(tuple(sorted(b)) for _, b in dropped).most_common():
        print(f"  {'+'.join(scripts):24s} {n}")
    print(f"Kept    {len(kept)} rows -> {OUTPUT_PATH}")

    by_group = Counter(row.get("speaker_id", "?") for row, _ in dropped)
    if by_group:
        print("\nDropped rows by episode group (re-transcription targets):")
        for group, n in sorted(by_group.items()):
            print(f"  {group}: {n}")


if __name__ == "__main__":
    main()
