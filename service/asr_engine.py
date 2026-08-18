"""
Kokborok (trp) speech-to-text.

No native or bridge-language Kokborok ASR model exists, and no labeled
Kokborok audio is available to fine-tune with. transcribe() tries, in
order, and returns the first that succeeds:

  1. fine_tuned: a real Whisper checkpoint fine-tuned on labeled Kokborok
     audio, once one exists at WHISPER_FINETUNED_DIR (see
     scripts/finetune_whisper.py). Not available yet.
  2. phoneme_zero_shot_bridge: a language-agnostic IPA phoneme recognizer
     (facebook/wav2vec2-lv-60-espeak-cv-ft). Needs no Kokborok training
     data - it reports the sounds it hears as IPA symbols instead of
     guessing words in a language it's never seen. Still not Kokborok
     orthography, but honest about what it is, and a human can usually
     reconstruct real words from phonemes faster than from a wrong-language
     Whisper guess. Requires `pip install phonemizer` and a one-time
     `scripts/convert_phoneme_model.py` run (see config.py for why).
  3. whisper_zero_shot_bridge: generic multilingual Whisper doing its own
     language ID, as a last resort if espeak-ng isn't installed. Its output
     is confident words in whatever language it guesses - actively
     misleading for Kokborok audio, so it's ranked last and given the
     lowest confidence.
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
        self._device: Optional[str] = None
        self._whisper_model = None
        self._whisper_processor = None
        self._whisper_fine_tuned = False
        self._whisper_load_failed = False
        self._phoneme_model = None
        self._phoneme_processor = None
        self._phoneme_load_failed = False

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

    def _load_whisper(self):
        if self._whisper_model is not None or self._whisper_load_failed:
            return self._whisper_model, self._whisper_processor
        try:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            checkpoint = config.WHISPER_FINETUNED_DIR if config.WHISPER_FINETUNED_DIR.exists() else None
            source = str(checkpoint) if checkpoint else config.WHISPER_MODEL_NAME
            self._whisper_fine_tuned = checkpoint is not None
            logger.info(
                "Loading Whisper ASR model from %s (fine_tuned=%s) on %s...",
                source, self._whisper_fine_tuned, self.device,
            )
            self._whisper_processor = WhisperProcessor.from_pretrained(source)
            self._whisper_model = WhisperForConditionalGeneration.from_pretrained(source).to(self.device)
        except Exception:
            logger.exception("Whisper ASR model failed to load")
            self._whisper_load_failed = True
            self._whisper_model = None
            self._whisper_processor = None
        return self._whisper_model, self._whisper_processor

    def _load_phoneme_model(self):
        if self._phoneme_model is not None or self._phoneme_load_failed:
            return self._phoneme_model, self._phoneme_processor
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            source = (
                str(config.PHONEME_ASR_LOCAL_DIR)
                if config.PHONEME_ASR_LOCAL_DIR.exists()
                else config.PHONEME_ASR_MODEL_NAME
            )
            logger.info("Loading phoneme ASR model from %s on %s...", source, self.device)
            self._phoneme_processor = Wav2Vec2Processor.from_pretrained(source)
            self._phoneme_model = Wav2Vec2ForCTC.from_pretrained(source).to(self.device)
        except Exception:
            logger.exception(
                "Phoneme ASR model failed to load (needs `pip install phonemizer`, and "
                "scripts/convert_phoneme_model.py run once to produce a safetensors "
                "checkpoint at PHONEME_ASR_LOCAL_DIR)"
            )
            self._phoneme_load_failed = True
            self._phoneme_model = None
            self._phoneme_processor = None
        return self._phoneme_model, self._phoneme_processor

    def _transcribe_fine_tuned_or_whisper_bridge(self, audio, sr: int) -> Optional[TranscriptionResult]:
        import torch

        model, processor = self._load_whisper()
        if model is None:
            return None
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            predicted_ids = model.generate(**inputs)
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        if self._whisper_fine_tuned:
            return TranscriptionResult(text, confidence=0.6, method=config.METHOD_ASR_FINE_TUNED)
        return TranscriptionResult(text, confidence=0.15, method=config.METHOD_ASR_ZERO_SHOT_BRIDGE)

    def _transcribe_phoneme_bridge(self, audio, sr: int) -> Optional[TranscriptionResult]:
        import torch

        model, processor = self._load_phoneme_model()
        if model is None:
            return None
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        phonemes = processor.batch_decode(predicted_ids)[0].strip()
        if not phonemes:
            return None
        return TranscriptionResult(phonemes, confidence=0.3, method=config.METHOD_ASR_PHONEME_BRIDGE)

    def _load_audio(self, audio_path: Path):
        """
        Decode audio_path to a 16kHz mono array. Browser mic recordings
        arrive as webm/opus, which libsndfile (what librosa.load tries
        first) can't open at all ("Format not recognised") - so every input
        goes through ffmpeg to a normalized wav first rather than relying on
        librosa's limited built-in format support.
        """
        import subprocess
        import tempfile

        import librosa

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            normalized_path = Path(tmp.name)
        try:
            proc = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(audio_path),
                    "-ar", "16000", "-ac", "1", "-f", "wav", str(normalized_path),
                ],
                capture_output=True,
                timeout=30,
            )
            if proc.returncode != 0 or not normalized_path.exists() or normalized_path.stat().st_size == 0:
                raise ValueError(
                    f"Could not decode uploaded audio (ffmpeg: {proc.stderr.decode(errors='replace')[-300:]})"
                )
            return librosa.load(str(normalized_path), sr=16000, mono=True)
        finally:
            normalized_path.unlink(missing_ok=True)

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        audio, sr = self._load_audio(audio_path)

        result = self._transcribe_fine_tuned_or_whisper_bridge(audio, sr)
        if result is not None and result.method == config.METHOD_ASR_FINE_TUNED:
            return result

        phoneme_result = self._transcribe_phoneme_bridge(audio, sr)
        if phoneme_result is not None:
            return phoneme_result

        if result is not None:
            return result

        raise RuntimeError(
            "No ASR backend available: check network access for model downloads, "
            "and for the phoneme bridge, that espeak-ng is installed "
            "(`sudo apt-get install espeak-ng`)."
        )


engine = ASREngine()
