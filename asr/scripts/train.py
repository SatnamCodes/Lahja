#!/usr/bin/env python3
"""Fine-tune Whisper with LoRA on the Kokborok splits. Resumable.

    python scripts/train.py
    python scripts/train.py --set train.batch_size=2 --set train.max_steps=200
    python scripts/train.py --resume checkpoints/whisper_small_lora_trp/checkpoint-300

Resuming is the default: with `train.resume: auto` the newest checkpoint in
train.output_dir is picked up automatically, so an interrupted run continues
rather than restarting.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kokborok_asr.config import add_config_args, load_config  # noqa: E402
from kokborok_asr.data import DataCollatorSpeechSeq2SeqWithPadding, WhisperASRDataset  # noqa: E402
from kokborok_asr.evaluation import log_sample_predictions, run_evaluation  # noqa: E402
from kokborok_asr.metrics import append_metrics, split_provenance, write_predictions  # noqa: E402
from kokborok_asr.modeling import build_model, build_processor, resolve_device, resolve_dtype  # noqa: E402
from kokborok_asr.paths import read_split  # noqa: E402
from kokborok_asr.text import normalizer_from_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train")


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def resolve_resume(resume_cfg, output_dir: Path, cli_resume: str | None):
    """Return a checkpoint path (or None) honouring CLI > config."""
    from transformers.trainer_utils import get_last_checkpoint

    choice = cli_resume if cli_resume is not None else str(resume_cfg)
    if choice in ("none", "None", "false", "False", ""):
        return None
    if choice == "auto":
        if output_dir.exists():
            last = get_last_checkpoint(str(output_dir))
            if last:
                logger.info("Resuming from newest checkpoint: %s", last)
                return last
        logger.info("No existing checkpoint in %s; starting fresh", output_dir)
        return None
    path = Path(choice)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    if not path.exists():
        raise FileNotFoundError(f"--resume checkpoint not found: {path}")
    logger.info("Resuming from: %s", path)
    return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    parser.add_argument(
        "--resume", default=None,
        help="auto | none | path/to/checkpoint (overrides train.resume)",
    )
    args = parser.parse_args()

    import torch
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    cfg = load_config(args.config, args.overrides)
    root = Path(__file__).resolve().parents[1]
    processed_dir = cfg.resolve_path("paths.processed_dir", "data/processed")
    output_dir = cfg.resolve_path("train.output_dir", "checkpoints/run")
    metrics_file = cfg.resolve_path("paths.metrics_file", "results/metrics.jsonl")

    train_utts = read_split(processed_dir / "train.jsonl")
    val_path = processed_dir / "val.jsonl"
    val_utts = read_split(val_path) if val_path.exists() else []
    logger.info("train=%d utterances | val=%d utterances", len(train_utts), len(val_utts))

    processor = build_processor(cfg)
    model = build_model(cfg, for_training=True)

    normalize = normalizer_from_config(cfg)
    ds_kwargs = dict(
        max_duration=cfg.get("data.max_duration", 30.0),
        min_duration=cfg.get("data.min_duration", 0.2),
    )
    train_ds = WhisperASRDataset(train_utts, processor, normalize, **ds_kwargs)
    eval_ds = WhisperASRDataset(val_utts, processor, normalize, **ds_kwargs) if val_utts else None
    if len(train_ds) == 0:
        logger.error("Every training utterance was filtered out by the duration bounds.")
        return 1

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    device = resolve_device(cfg.get("model.device", "auto"))
    dtype = resolve_dtype(device, cfg.get("train.precision", "auto"))
    use_fp16 = device == "cuda" and dtype == torch.float16
    use_bf16 = device == "cuda" and dtype == torch.bfloat16

    eval_strategy = str(cfg.get("train.eval_strategy", "steps")) if eval_ds else "no"
    max_steps = int(cfg.get("train.max_steps", -1))

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(cfg.get("train.batch_size", 4)),
        per_device_eval_batch_size=int(cfg.get("train.eval_batch_size", cfg.get("train.batch_size", 4))),
        gradient_accumulation_steps=int(cfg.get("train.gradient_accumulation_steps", 4)),
        learning_rate=float(cfg.get("train.learning_rate", 1e-3)),
        num_train_epochs=float(cfg.get("train.num_train_epochs", 8)),
        max_steps=max_steps,
        warmup_steps=int(cfg.get("train.warmup_steps", 50)),
        logging_steps=int(cfg.get("train.logging_steps", 10)),
        eval_strategy=eval_strategy,
        eval_steps=int(cfg.get("train.eval_steps", 100)),
        save_strategy=str(cfg.get("train.save_strategy", "steps")),
        save_steps=int(cfg.get("train.save_steps", 100)),
        save_total_limit=int(cfg.get("train.save_total_limit", 3)),
        gradient_checkpointing=bool(cfg.get("train.gradient_checkpointing", True)),
        fp16=use_fp16,
        bf16=use_bf16,
        seed=int(cfg.get("train.seed", 1234)),
        dataloader_num_workers=int(cfg.get("train.dataloader_num_workers", 2)),
        # Required for PEFT: the collator emits columns the signature check
        # would otherwise strip, and the loss key must be declared.
        remove_unused_columns=False,
        label_names=["labels"],
        predict_with_generate=False,
        report_to=list(cfg.get("train.report_to", [])),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=processor.feature_extractor,
    )

    resume_from = resolve_resume(cfg.get("train.resume", "auto"), output_dir, args.resume)

    started = time.time()
    result = trainer.train(resume_from_checkpoint=resume_from)
    elapsed = time.time() - started
    logger.info("Training finished in %.1fs", elapsed)

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    logger.info("Saved adapter + processor to %s", output_dir)

    record = {
        "event": "train",
        "run_name": cfg.get("run_name"),
        # How splits were grouped; every WER below is relative to this.
        **split_provenance(cfg),
        "config": str(cfg.source),
        "git_commit": git_commit(),
        "model": cfg.get("model.name"),
        "language_proxy": cfg.get("model.language"),
        "device": device,
        "precision": str(dtype).replace("torch.", ""),
        "lora": cfg.section("lora"),
        "n_train_utterances": len(train_ds),
        "n_val_utterances": len(eval_ds) if eval_ds else 0,
        "train_speakers": sorted({u.speaker_id for u in train_utts}),
        "val_speakers": sorted({u.speaker_id for u in val_utts}),
        "resumed_from": resume_from,
        "global_step": int(result.global_step),
        "train_loss": float(result.training_loss),
        "train_seconds": round(elapsed, 2),
        "checkpoint": str(output_dir),
    }

    # WER/CER from a real generate pass, using the same code path as
    # scripts/evaluate.py so the numbers are directly comparable.
    if cfg.get("train.final_eval", True) and eval_ds:
        split_name = cfg.get("train.final_eval_split", "val")
        logger.info("Running final generate-based eval on '%s'...", split_name)
        eval_utts = val_utts if split_name == "val" else read_split(processed_dir / f"{split_name}.jsonl")
        model.config.use_cache = True
        outcome = run_evaluation(
            model, processor, eval_utts,
            normalizer=normalize,
            batch_size=int(cfg.get("eval.batch_size", 8)),
            max_new_tokens=int(cfg.get("eval.max_new_tokens", 128)),
            num_beams=int(cfg.get("eval.num_beams", 1)),
            device=device,
        )
        predictions = outcome.pop("predictions")
        log_sample_predictions(predictions)
        pred_path = cfg.resolve_path("paths.results_dir", "results") / (
            f"predictions_{cfg.get('run_name', 'run')}_{split_name}.jsonl"
        )
        write_predictions(pred_path, predictions)
        record.update({"eval_split": split_name, "predictions_file": str(pred_path), **outcome})
        logger.info("Final %s WER=%.4f CER=%.4f", split_name, outcome["wer"], outcome["cer"])

    append_metrics(metrics_file, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
