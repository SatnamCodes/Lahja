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
    # Optional: a manifest and audio tree outside raw_dir (the Kokborok
    # manifest lives at the repo root, one level above asr/). Both are
    # read-only; raw_dir's read-only guarantee is unaffected.
    manifest_file = (
        cfg.resolve_path("paths.manifest_file") if cfg.get("paths.manifest_file") else None
    )
    audio_root = cfg.resolve_path("data.audio_root") if cfg.get("data.audio_root") else None

    logger.info("Raw corpus (read-only): %s", raw_dir)
    if manifest_file:
        logger.info("Manifest (read-only): %s", manifest_file)
    if audio_root:
        logger.info("Audio root for relative paths: %s", audio_root)

    utterances = manifest.discover(
        raw_dir,
        layout=cfg.get("data.layout", "auto"),
        speaker_strategy=cfg.get("data.speaker_strategy", "parent_dir"),
        speaker_delimiter=cfg.get("data.speaker_delimiter", "_"),
        probe_durations=cfg.get("data.probe_durations", True),
        manifest_path=manifest_file,
        audio_root=audio_root,
        utt_id_prefix=cfg.get("data.utt_id_prefix"),
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
    # "group" not "speaker": on this corpus the split groups are programme
    # episodes from a single narrator (split.group_by), so calling them
    # speakers in the log would misrepresent what the split guarantees.
    group_by = cfg.get("split.group_by", "speaker")
    logger.info(
        "Discovered %d utterances in %d %s group(s)%s",
        len(utterances), len(speakers), group_by,
        f", {sum(durations) / 3600:.3f} h ({sum(durations) / 60:.1f} min) total"
        if durations else "",
    )

    # A declared group count that no longer matches reality means the corpus
    # changed under the config. Fail rather than silently splitting on
    # something other than what metrics.jsonl will claim was used.
    expected_groups = cfg.get("split.n_groups")
    if expected_groups is not None and int(expected_groups) != len(speakers):
        logger.error(
            "split.n_groups says %s but the corpus has %d %s group(s): %s.\n"
            "Either the data changed or the config is stale - every WER logged "
            "would be labelled with the wrong grouping. Fix one of the two.",
            expected_groups, len(speakers), group_by, sorted(speakers),
        )
        return 1

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
            "  %-5s %4d utts | %2d %s group(s) %s%s",
            name, info["utterances"], info["n_speakers"], group_by, info["speakers"],
            f" | {info['hours'] * 60:.1f} min" if info["hours"] else "",
        )
    if overlap:
        logger.error("%s LEAK between splits: %s", group_by.upper(), overlap)
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
                "manifest_file": str(manifest_file) if manifest_file else None,
                "n_utterances": len(utterances),
                "n_speakers": len(speakers),
                "group_by": group_by,
                "n_groups": len(speakers),
                "speaker_optimistic": bool(cfg.get("split.speaker_optimistic", False)),
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
