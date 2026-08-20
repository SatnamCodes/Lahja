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

from . import config, device_utils

logger = logging.getLogger("lahja.asr")


class TranscriptionResult:
    def __init__(self, text: str, confidence: float, method: str):
        self.text = text
        self.confidence = confidence
        self.method = method



WEBM_TARGET_SR = 16000


def _load_any_audio(audio_path, target_sr: int = WEBM_TARGET_SR):
    """Decode any browser-supplied audio to mono float32 at ``target_sr``.

    librosa/libsndfile handle wav/flac/ogg but NOT WebM, and MediaRecorder
    emits `audio/webm` (Opus) by default in Chrome, Edge and Brave - so a clip
    recorded in the browser failed with "Format not recognised" while an
    uploaded .wav worked. librosa's usual escape hatch (audioread -> system
    ffmpeg) is no help here either: this box has an incomplete FFmpeg install.

    PyAV is the fallback because its wheels bundle their own FFmpeg libraries,
    so WebM/Opus decodes with no system packages and no sudo.
    """
    import numpy as np

    try:
        import librosa

        return librosa.load(str(audio_path), sr=target_sr, mono=True)
    except Exception as exc:  # noqa: BLE001 - any decode failure falls through
        logger.info("libsndfile could not decode %s (%s); trying PyAV",
                    audio_path.name, type(exc).__name__)

    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            f"Could not decode {audio_path.name}: libsndfile does not support this "
            f"container (browser recordings are WebM/Opus) and PyAV is not installed. "
            f"Install it with `pip install av`."
        ) from exc

    with av.open(str(audio_path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise RuntimeError(f"No audio stream found in {audio_path.name}")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=target_sr)
        chunks = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        # Flush the resampler; without this the tail of short clips is lost.
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))

    if not chunks:
        raise RuntimeError(f"Decoded no audio from {audio_path.name}")
    audio = np.concatenate(chunks).astype("float32") / 32768.0
    return audio, target_sr


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
        # Each model records where it ACTUALLY landed: one may fall back to
        # CPU on VRAM pressure while the other stays on GPU, and inputs must
        # be moved to the same device as their model.
        self._whisper_device: Optional[str] = None
        self._phoneme_device: Optional[str] = None

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
            self._whisper_model, self._whisper_device = device_utils.to_device_or_cpu(
                WhisperForConditionalGeneration.from_pretrained(source),
                self.device, what="the Whisper ASR model",
            )
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
            self._phoneme_model, self._phoneme_device = device_utils.to_device_or_cpu(
                Wav2Vec2ForCTC.from_pretrained(source),
                self.device, what="the phoneme ASR model",
            )
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
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(
            self._whisper_device or self.device
        )
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
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(
            self._phoneme_device or self.device
        )
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        phonemes = processor.batch_decode(predicted_ids)[0].strip()
        if not phonemes:
            return None
        return TranscriptionResult(phonemes, confidence=0.3, method=config.METHOD_ASR_PHONEME_BRIDGE)

    def _load_audio(self, audio_path: Path):
        """Decode ``audio_path`` to a 16kHz mono array.

        Browser mic recordings arrive as webm/opus, which libsndfile (what
        librosa.load tries first) cannot open at all ("Format not
        recognised"), so a container-agnostic decoder is required.

        Prefer the ffmpeg CLI when it is installed: it handles the widest set
        of containers and normalizes to wav in one pass. But it is NOT always
        present - a plain `pip install -r requirements.txt` brings no ffmpeg
        binary, and this project's own dev box has an incomplete FFmpeg - so
        a missing binary must fall through rather than 500. _load_any_audio()
        covers that with PyAV, whose wheels bundle their own FFmpeg
        libraries and need no system packages.
        """
        import shutil

        if shutil.which("ffmpeg"):
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
                if proc.returncode == 0 and normalized_path.exists() and normalized_path.stat().st_size > 0:
                    return librosa.load(str(normalized_path), sr=16000, mono=True)
                # ffmpeg exists but could not decode this input: fall through
                # to PyAV rather than failing, then surface PyAV's error.
                logger.info(
                    "ffmpeg could not decode %s (exit %s); trying the in-process decoder",
                    audio_path.name, proc.returncode,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                logger.info("ffmpeg unavailable or failed for %s (%s); trying the "
                            "in-process decoder", audio_path.name, type(exc).__name__)
            finally:
                normalized_path.unlink(missing_ok=True)
        else:
            logger.info("No ffmpeg binary on PATH; decoding %s in-process via PyAV",
                        audio_path.name)

        return _load_any_audio(audio_path, target_sr=16000)

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
