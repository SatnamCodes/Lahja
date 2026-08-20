"""
Kokborok (trp) TTS engine.

No native Kokborok TTS model exists, so every result here is an
approximation and is labeled with the method that actually produced it:

  - xtts_zero_shot: Coqui XTTS v2, zero-shot voice cloning against a short
    Kokborok reference clip. Kokborok is Romanized (Latin script) in our
    input text, and XTTS has no "trp" tokenizer, so we drive it with the
    English grapheme-to-phoneme path as the closest available fit. This is
    the primary path: it needs no training and can start from as little as
    ~10-20s of reference audio.
  - mms_bridge_zero_shot: Meta MMS-TTS for Bengali (or Assamese) used as a
    phonetically-adjacent bridge language, pretrained weights only. Falls
    back to this if XTTS has no reference audio or fails to load.
  - mms_fine_tuned: the same MMS bridge model after fine-tuning on the
    aligned Kokborok batch. Used automatically once a checkpoint exists at
    models/mms_finetuned/<bridge>/.

speak() tries these in order and returns the first that succeeds, so the
service degrades gracefully rather than failing outright.
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional

from . import config

logger = logging.getLogger("lahja.tts")


class SynthesisResult:
    def __init__(self, file_path: Path, confidence: float, method: str):
        self.file_path = file_path
        self.confidence = confidence
        self.method = method


class TTSEngine:
    def __init__(self):
        self._device: Optional[str] = None
        self._xtts = None
        self._xtts_load_failed = False
        self._mms_models: dict[str, tuple] = {}
        self._mms_load_failed: set[str] = set()

    @property
    def device(self) -> str:
        if self._device is None:
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"
            logger.info("TTSEngine device: %s", self._device)
        return self._device

    def _reference_clips(self) -> list[Path]:
        return sorted(config.AUDIO_REF_DIR.glob("*.wav"))

    def _pick_speaker_reference(self) -> Optional[list[str]]:
        clips = self._reference_clips()
        if not clips:
            return None
        # XTTS blends multiple reference wavs into one speaker embedding;
        # cap at 3 so this stays fast even once the larger batch lands.
        return [str(p) for p in clips[:3]]

    def _load_xtts(self):
        if self._xtts is not None or self._xtts_load_failed:
            return self._xtts
        try:
            from TTS.api import TTS
            logger.info("Loading XTTS v2 on %s...", self.device)
            self._xtts = TTS(config.XTTS_MODEL_NAME).to(self.device)
        except Exception:
            logger.exception("XTTS v2 failed to load")
            self._xtts_load_failed = True
            self._xtts = None
        return self._xtts

    def _finetuned_checkpoint(self, bridge: str) -> Optional[Path]:
        path = config.BASE_DIR / "models" / "mms_finetuned" / bridge
        return path if path.exists() else None

    def _load_mms(self, bridge: str):
        if bridge in self._mms_models:
            return self._mms_models[bridge]
        if bridge in self._mms_load_failed:
            return None
        try:
            from transformers import VitsModel, AutoTokenizer
            checkpoint = self._finetuned_checkpoint(bridge)
            source = str(checkpoint) if checkpoint else config.MMS_MODEL_CANDIDATES[bridge]
            logger.info("Loading MMS-TTS bridge model from %s...", source)
            model = VitsModel.from_pretrained(source).to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(source)
            fine_tuned = checkpoint is not None
            self._mms_models[bridge] = (model, tokenizer, fine_tuned)
            return self._mms_models[bridge]
        except Exception:
            logger.exception("MMS-TTS bridge model %s failed to load", bridge)
            self._mms_load_failed.add(bridge)
            return None

    def _output_path(self, text: str, method: str) -> Path:
        stem = hashlib.sha1(f"{method}:{text}".encode("utf-8")).hexdigest()[:16]
        return config.OUTPUT_DIR / f"{stem}.wav"

    def _synthesize_xtts(self, text: str, language: str) -> Optional[SynthesisResult]:
        speaker_wavs = self._pick_speaker_reference()
        if not speaker_wavs:
            logger.info("No reference audio in data/audio/ - skipping XTTS zero-shot")
            return None
        tts = self._load_xtts()
        if tts is None:
            return None
        # trp text has no XTTS tokenizer, so it's driven through XTTS's
        # English phoneme path as an approximation (low confidence). Real
        # English text takes that same path natively - XTTS genuinely
        # supports English - so it's labeled and scored as what it is: real
        # synthesis, just in the cloned Kokborok narrator's voice rather
        # than a dedicated English speaker.
        is_native_english = language == "eng"
        method = config.METHOD_XTTS_ENGLISH_NATIVE if is_native_english else config.METHOD_XTTS_ZERO_SHOT
        confidence = 0.85 if is_native_english else 0.45
        out_path = self._output_path(text, method)
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wavs,
            language=config.XTTS_BRIDGE_LANGUAGE,
            file_path=str(out_path),
        )
        return SynthesisResult(out_path, confidence=confidence, method=method)

    def _synthesize_mms(self, text: str) -> Optional[SynthesisResult]:
        import torch
        import scipy.io.wavfile as wavfile

        bridge = config.MMS_DEFAULT_BRIDGE
        loaded = self._load_mms(bridge)
        if loaded is None:
            return None
        model, tokenizer, fine_tuned = loaded
        inputs = tokenizer(text, return_tensors="pt")
        if inputs.input_ids.shape[-1] == 0:
            # The bridge model's tokenizer is native-script (e.g. Bengali
            # Unicode) and Kokborok text here is Romanized Latin script, so
            # every character got dropped as out-of-vocab. Without a
            # Latin->Bengali transliteration step this bridge can't help;
            # skip rather than run the model on an empty sequence.
            logger.warning(
                "MMS bridge tokenizer produced no tokens for %r (expects native script); skipping",
                text,
            )
            return None
        inputs = inputs.to(self.device)
        with torch.no_grad():
            output = model(**inputs).waveform
        method = config.METHOD_MMS_FINE_TUNED if fine_tuned else config.METHOD_MMS_BRIDGE
        confidence = 0.65 if fine_tuned else 0.35
        out_path = self._output_path(text, method)
        wavfile.write(
            str(out_path),
            rate=model.config.sampling_rate,
            data=output.squeeze().cpu().numpy(),
        )
        return SynthesisResult(out_path, confidence=confidence, method=method)

    def speak(self, text: str, language: str = "trp") -> SynthesisResult:
        text = text.strip()
        if not text:
            raise ValueError("text must not be empty")

        result = self._synthesize_xtts(text, language)
        if result is not None:
            return result

        result = self._synthesize_mms(text)
        if result is not None:
            return result

        raise RuntimeError(
            "No TTS backend available: put a short Kokborok reference clip "
            "in data/audio/*.wav for XTTS zero-shot voice cloning, or check "
            "network access for the MMS bridge model download."
        )


engine = TTSEngine()
