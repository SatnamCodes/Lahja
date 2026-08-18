#!/usr/bin/env python3
"""Build speaker-disjoint train/val/test splits from the raw corpus.

Reads data/raw (never writes to it) and emits JSONL split files plus a
summary into data/processed.

    python scripts/prepare_data.py
    python scripts/prepare_data.py --set data.speaker_strategy=filename_prefix
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kokborok_asr import manifest, splitting  # noqa: E402
from kokborok_asr.config import add_config_args, load_config  # noqa: E402
from kokborok_asr.paths import write_split  # noqa: E402
from kokborok_asr.text import normalizer_from_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("prepare_data")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    parser.add_argument(
        "--dry-run", action="store_true", help="Report what would be written, write nothing"
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    raw_dir = cfg.resolve_path("paths.raw_dir", "data/raw")
    processed_dir = cfg.resolve_path("paths.processed_dir", "data/processed")

    logger.info("Raw corpus (read-only): %s", raw_dir)

    utterances = manifest.discover(
        raw_dir,
        layout=cfg.get("data.layout", "auto"),
        speaker_strategy=cfg.get("data.speaker_strategy", "parent_dir"),
        speaker_delimiter=cfg.get("data.speaker_delimiter", "_"),
        probe_durations=cfg.get("data.probe_durations", True),
    )
    utterances, problems = manifest.validate(utterances)
    for problem in problems:
        logger.warning("%s", problem)
    if not utterances:
        logger.error("Every discovered utterance was rejected; nothing to split.")
        return 1

    normalize = normalizer_from_config(cfg)
    empty_after_norm = [u.utt_id for u in utterances if not normalize(u.transcript)]
    if empty_after_norm:
        logger.warning(
            "%d transcript(s) normalize to empty text (e.g. %s); they will train on nothing.",
            len(empty_after_norm), ", ".join(empty_after_norm[:3]),
        )

    speakers = splitting.group_by_speaker(utterances)
    durations = [u.duration for u in utterances if u.duration]
    logger.info(
        "Discovered %d utterances from %d speaker(s)%s",
        len(utterances), len(speakers),
        f", {sum(durations) / 3600:.3f} h total" if durations else "",
    )

    splits = splitting.split_by_speaker(
        utterances,
        ratios=cfg.get("split.ratios", {"train": 0.7, "val": 0.15, "test": 0.15}),
        seed=int(cfg.get("split.seed", 1234)),
        allow_speaker_overlap=bool(cfg.get("split.allow_speaker_overlap", False)),
    )

    summary = splitting.summarize(splits)
    overlap = summary.pop("_speaker_overlap", {})
    for name in sorted(splits):
        info = summary[name]
        logger.info(
            "  %-5s %4d utts | %2d speaker(s) %s%s",
            name, info["utterances"], info["n_speakers"], info["speakers"],
            f" | {info['hours']:.3f} h" if info["hours"] else "",
        )
    if overlap:
        logger.error("SPEAKER LEAK between splits: %s", overlap)
        if not cfg.get("split.allow_speaker_overlap", False):
            return 1

    if args.dry_run:
        logger.info("--dry-run: no files written")
        return 0

    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, utts in splits.items():
        path = write_split(processed_dir / f"{name}.jsonl", utts)
        logger.info("Wrote %s (%d utterances)", path, len(utts))

    summary_path = processed_dir / "split_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_name": cfg.get("run_name"),
                "raw_dir": str(raw_dir),
                "config": str(cfg.source),
                "n_utterances": len(utterances),
                "n_speakers": len(speakers),
                "splits": summary,
                "speaker_overlap": overlap,
                "split_seed": cfg.get("split.seed", 1234),
                "problems": problems,
            },
            indent=2, ensure_ascii=False, default=str,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
