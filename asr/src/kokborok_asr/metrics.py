"""WER/CER computation and the append-only metrics log.

Every training or evaluation run appends one JSON line to
``results/metrics.jsonl`` so runs stay comparable after the console
scrollback is gone.
"""

from __future__ import annotations

import json
import logging
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


def compute_wer_cer(references: Sequence[str], hypotheses: Sequence[str]) -> dict[str, float]:
    """Word and character error rates over a whole split.

    Computed corpus-level (total edits / total reference length), not as a
    mean of per-utterance rates, which would over-weight short utterances.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"references and hypotheses differ in length: {len(references)} vs {len(hypotheses)}"
        )
    pairs = [(r, h) for r, h in zip(references, hypotheses) if r.strip()]
    if not pairs:
        logger.warning("No non-empty references; returning NaN error rates")
        return {"wer": float("nan"), "cer": float("nan"), "n_scored": 0}

    import jiwer

    refs = [r for r, _ in pairs]
    hyps = [h for _, h in pairs]
    wer_out = jiwer.process_words(refs, hyps)
    cer_out = jiwer.process_characters(refs, hyps)
    return {
        "wer": float(wer_out.wer),
        "cer": float(cer_out.cer),
        "n_scored": len(pairs),
        "word_substitutions": int(wer_out.substitutions),
        "word_deletions": int(wer_out.deletions),
        "word_insertions": int(wer_out.insertions),
    }


def split_provenance(cfg) -> dict[str, Any]:
    """How the train/val/test split was grouped, for the run record.

    Every WER/CER number is only interpretable alongside this. The Kokborok
    corpus is a single narrator, so splits are disjoint by programme episode
    rather than by speaker: train and test share a voice and the scores are
    correspondingly optimistic. Recording it per run means a future reader of
    metrics.jsonl cannot mistake these for speaker-disjoint numbers.
    """
    return {
        "group_by": cfg.get("split.group_by", "speaker"),
        "n_groups": cfg.get("split.n_groups"),
        "speaker_optimistic": bool(cfg.get("split.speaker_optimistic", False)),
    }


def append_metrics(metrics_path: Path, record: dict[str, Any]) -> Path:
    """Append one run record to results/metrics.jsonl."""
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        **record,
    }
    with metrics_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    logger.info(
        "Logged metrics to %s (wer=%s cer=%s)",
        metrics_path,
        _fmt(payload.get("wer")),
        _fmt(payload.get("cer")),
    )
    return metrics_path


def _fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else str(value)


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Dump per-utterance reference/hypothesis pairs for eyeballing.

    Whisper fine-tuned on a tiny corpus can produce fluent, confident, and
    entirely wrong transcripts. An aggregate WER hides that; this file is
    how you catch it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return path
