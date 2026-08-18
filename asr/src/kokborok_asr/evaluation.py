"""Shared evaluation: generate transcripts, score WER/CER, dump samples.

Both ``scripts/train.py`` (at the end of a run) and ``scripts/evaluate.py``
call ``run_evaluation`` so a number logged during training is produced by
exactly the same code path as a standalone eval.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from .data import WHISPER_SAMPLE_RATE, load_audio
from .manifest import Utterance
from .metrics import compute_wer_cer
from .modeling import resolve_device, resolve_dtype

logger = logging.getLogger(__name__)


def run_evaluation(
    model,
    processor,
    utterances: list[Utterance],
    *,
    normalizer: Callable[[str], str],
    batch_size: int = 8,
    max_new_tokens: int = 128,
    num_beams: int = 1,
    device: str | None = None,
    limit: int | None = None,
    progress_every: int = 5,
) -> dict:
    """Transcribe ``utterances`` and score them against their references."""
    import torch

    if limit is not None and limit > 0:
        utterances = utterances[:limit]
    if not utterances:
        raise ValueError("No utterances to evaluate")

    device = device or resolve_device()
    dtype = next(model.parameters()).dtype
    model.eval()

    references: list[str] = []
    hypotheses: list[str] = []
    rows: list[dict] = []

    n_batches = (len(utterances) + batch_size - 1) // batch_size
    for b in range(n_batches):
        chunk = utterances[b * batch_size : (b + 1) * batch_size]
        audios = [load_audio(u.audio_path) for u in chunk]
        inputs = processor.feature_extractor(
            audios, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt"
        )
        features = inputs.input_features.to(device=device, dtype=dtype)

        with torch.no_grad():
            generated = model.generate(
                input_features=features,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
        decoded = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)

        for u, hyp in zip(chunk, decoded):
            ref_norm = normalizer(u.transcript)
            hyp_norm = normalizer(hyp)
            references.append(ref_norm)
            hypotheses.append(hyp_norm)
            rows.append(
                {
                    "utt_id": u.utt_id,
                    "speaker_id": u.speaker_id,
                    "duration": u.duration,
                    "reference": ref_norm,
                    "hypothesis": hyp_norm,
                    "hypothesis_raw": hyp,
                }
            )

        if progress_every and (b + 1) % progress_every == 0:
            logger.info("Evaluated %d/%d batches", b + 1, n_batches)

    scores = compute_wer_cer(references, hypotheses)

    # Per-utterance WER, so the worst offenders are easy to find in the dump.
    import jiwer

    for row in rows:
        if row["reference"].strip():
            row["wer"] = float(jiwer.wer(row["reference"], row["hypothesis"]))

    empty = sum(1 for h in hypotheses if not h.strip())
    if empty:
        logger.warning("%d/%d hypotheses were empty", empty, len(hypotheses))

    return {
        **scores,
        "n_utterances": len(utterances),
        "n_empty_hypotheses": empty,
        "predictions": rows,
    }


def log_sample_predictions(rows: list[dict], k: int = 5) -> None:
    """Print a few reference/hypothesis pairs so failures are visible."""
    logger.info("--- sample predictions ---")
    for row in rows[:k]:
        logger.info("  [%s] ref: %s", row["utt_id"], row["reference"])
        logger.info("  [%s] hyp: %s", row["utt_id"], row["hypothesis"])
