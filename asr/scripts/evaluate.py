#!/usr/bin/env python3
"""Score a trained adapter on a split: WER + CER, logged to metrics.jsonl.

    python scripts/evaluate.py                        # full test split
    python scripts/evaluate.py --set eval.limit=20    # fast check while iterating
    python scripts/evaluate.py --split val --adapter checkpoints/.../checkpoint-300

Always writes a per-utterance reference/hypothesis dump next to the metrics,
because an aggregate WER hides the fluent-but-wrong transcripts Whisper tends
to produce when fine-tuned on a small corpus.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kokborok_asr.config import add_config_args, load_config  # noqa: E402
from kokborok_asr.evaluation import log_sample_predictions, run_evaluation  # noqa: E402
from kokborok_asr.metrics import append_metrics, write_predictions  # noqa: E402
from kokborok_asr.modeling import build_model, build_processor, resolve_device  # noqa: E402
from kokborok_asr.paths import read_split  # noqa: E402
from kokborok_asr.text import normalizer_from_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("evaluate")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    parser.add_argument("--split", default=None, help="Split to score (default: eval.split)")
    parser.add_argument(
        "--adapter", default=None,
        help="LoRA adapter dir (default: train.output_dir). Pass 'none' for the untuned baseline.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N utterances")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    processed_dir = cfg.resolve_path("paths.processed_dir", "data/processed")
    metrics_file = cfg.resolve_path("paths.metrics_file", "results/metrics.jsonl")
    results_dir = cfg.resolve_path("paths.results_dir", "results")

    split_name = args.split or cfg.get("eval.split", "test")
    utterances = read_split(processed_dir / f"{split_name}.jsonl")

    adapter_arg = args.adapter if args.adapter is not None else str(cfg.get("train.output_dir", ""))
    adapter_path = None
    if adapter_arg and adapter_arg.lower() != "none":
        candidate = Path(adapter_arg)
        if not candidate.is_absolute():
            candidate = Path(__file__).resolve().parents[1] / candidate
        if (candidate / "adapter_config.json").exists():
            adapter_path = candidate
        else:
            logger.warning(
                "No adapter_config.json in %s - evaluating the UNTUNED base model. "
                "Train first, or pass --adapter none to silence this.", candidate,
            )

    processor = build_processor(cfg)
    model = build_model(cfg, for_training=False, adapter_path=adapter_path)
    device = resolve_device(cfg.get("model.device", "auto"))

    limit = args.limit if args.limit is not None else cfg.get("eval.limit")
    normalize = normalizer_from_config(cfg)

    started = time.time()
    outcome = run_evaluation(
        model, processor, utterances,
        normalizer=normalize,
        batch_size=int(cfg.get("eval.batch_size", 8)),
        max_new_tokens=int(cfg.get("eval.max_new_tokens", 128)),
        num_beams=int(cfg.get("eval.num_beams", 1)),
        device=device,
        limit=int(limit) if limit else None,
    )
    elapsed = time.time() - started

    predictions = outcome.pop("predictions")
    log_sample_predictions(predictions)

    tag = "baseline" if adapter_path is None else "tuned"
    pred_path = write_predictions(
        results_dir / f"predictions_{cfg.get('run_name', 'run')}_{split_name}_{tag}.jsonl",
        predictions,
    )

    append_metrics(
        metrics_file,
        {
            "event": "evaluate",
            "run_name": cfg.get("run_name"),
            "config": str(cfg.source),
            "model": cfg.get("model.name"),
            "language_proxy": cfg.get("model.language"),
            "adapter": str(adapter_path) if adapter_path else None,
            "is_baseline": adapter_path is None,
            "eval_split": split_name,
            "eval_speakers": sorted({u.speaker_id for u in utterances}),
            "limit": int(limit) if limit else None,
            "device": device,
            "eval_seconds": round(elapsed, 2),
            "predictions_file": str(pred_path),
            **outcome,
        },
    )

    logger.info(
        "%s | %s | WER=%.4f CER=%.4f over %d utterance(s)",
        cfg.get("run_name"), split_name, outcome["wer"], outcome["cer"], outcome["n_utterances"],
    )
    logger.info("Per-utterance dump: %s", pred_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
