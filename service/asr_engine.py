"""
Kokborok (trp) speech-to-text.

No native or bridge-language Kokborok ASR model exists (Meta MMS's
1,107-language ASR set does not include trp, unlike its TTS counterpart used
elsewhere in this service). Until a fine-tuned checkpoint is available at
WHISPER_FINETUNED_DIR (produced by scripts/finetune_whisper.py once labeled
Kokborok audio is available), this falls back to general-purpose
multilingual Whisper doing its own language detection - not tuned to
Kokborok phonetics or the Romanized orthography used elsewhere in this
service, so it's returned at low confidence and clearly labeled as such.
"""

import logging
from pathlib import Path
from typing import Optional

from . import config

logger = logging.getLogger("lahja.asr")


class TranscriptionResult:
    def __init__(self, text: str, confidence: float, method: str):
        self.text = text
        self.confidence = confidence
        self.method = method


class ASREngine:
    def __init__(self):
        self._model = None
        self._processor = None
        self._device: Optional[str] = None
        self._load_failed = False
        self._fine_tuned = False

    @property
    def device(self) -> str:
        if self._device is None:
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"
            logger.info("ASREngine device: %s", self._device)
        return self._device

    def _load(self):
        if self._model is not None or self._load_failed:
            return self._model, self._processor
        try:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            checkpoint = config.WHISPER_FINETUNED_DIR if config.WHISPER_FINETUNED_DIR.exists() else None
            source = str(checkpoint) if checkpoint else config.WHISPER_MODEL_NAME
            self._fine_tuned = checkpoint is not None
            logger.info(
                "Loading Whisper ASR model from %s (fine_tuned=%s) on %s...",
                source, self._fine_tuned, self.device,
            )
            self._processor = WhisperProcessor.from_pretrained(source)
            self._model = WhisperForConditionalGeneration.from_pretrained(source).to(self.device)
        except Exception:
            logger.exception("Whisper ASR model failed to load")
            self._load_failed = True
            self._model = None
            self._processor = None
        return self._model, self._processor

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        import librosa

        model, processor = self._load()
        if model is None:
            raise RuntimeError(
                "No ASR backend available: check network access for the Whisper "
                f"model download, or provide a fine-tuned checkpoint at "
                f"{config.WHISPER_FINETUNED_DIR}"
            )

        audio, _ = librosa.load(str(audio_path), sr=16000, mono=True)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(self.device)

        predicted_ids = model.generate(**inputs)
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

        method = config.METHOD_ASR_FINE_TUNED if self._fine_tuned else config.METHOD_ASR_ZERO_SHOT_BRIDGE
        confidence = 0.6 if self._fine_tuned else 0.15
        return TranscriptionResult(text, confidence=confidence, method=method)


engine = ASREngine()
