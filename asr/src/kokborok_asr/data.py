"""Audio loading, the torch Dataset, and the padding collator for Whisper."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .manifest import Utterance

logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16_000


def load_audio(path: Path, target_sr: int = WHISPER_SAMPLE_RATE) -> np.ndarray:
    """Load an audio file as mono float32 at ``target_sr``."""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(audio, dtype=np.float32)


class WhisperASRDataset:
    """Maps utterances to Whisper log-mel features and token labels.

    Feature extraction happens lazily per item rather than being cached up
    front, which keeps memory flat on a 6 GB / 14 GB box at the cost of some
    CPU per epoch. Set ``dataloader_num_workers`` in the config to hide it.
    """

    def __init__(
        self,
        utterances: list[Utterance],
        processor,
        normalizer: Callable[[str], str] | None = None,
        *,
        max_duration: float | None = 30.0,
        min_duration: float | None = 0.2,
        max_label_length: int = 448,
    ):
        self.processor = processor
        self.normalizer = normalizer or (lambda s: s)
        self.max_label_length = max_label_length
        self.utterances = self._filter(utterances, min_duration, max_duration)

    @staticmethod
    def _filter(
        utterances: list[Utterance], min_duration: float | None, max_duration: float | None
    ) -> list[Utterance]:
        kept: list[Utterance] = []
        for u in utterances:
            d = u.duration
            if d is not None:
                if min_duration is not None and d < min_duration:
                    logger.warning("Dropping %s: %.2fs shorter than min_duration", u.utt_id, d)
                    continue
                if max_duration is not None and d > max_duration:
                    # Whisper's encoder is fixed at 30s; longer clips get
                    # truncated silently, so the transcript would not match.
                    logger.warning(
                        "Dropping %s: %.2fs exceeds max_duration (Whisper truncates at 30s; "
                        "segment this clip instead)", u.utt_id, d
                    )
                    continue
            kept.append(u)
        return kept

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, index: int) -> dict[str, Any]:
        u = self.utterances[index]
        audio = load_audio(u.audio_path)
        features = self.processor.feature_extractor(
            audio, sampling_rate=WHISPER_SAMPLE_RATE
        ).input_features[0]

        text = self.normalizer(u.transcript)
        labels = self.processor.tokenizer(text).input_ids
        if len(labels) > self.max_label_length:
            logger.warning("Truncating labels for %s (%d tokens)", u.utt_id, len(labels))
            labels = labels[: self.max_label_length]

        return {"input_features": features, "labels": labels, "utt_id": u.utt_id}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Pads mel features and label sequences into a batch."""

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Padding must not contribute to the loss.
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # The tokenizer prepends the decoder start token, but the model adds
        # it again when shifting labels right; drop the duplicate.
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
