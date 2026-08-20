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

from . import config, device_utils, translit

logger = logging.getLogger("lahja.tts")



def _trim_silence(waveform, sr: int, threshold: float = 0.02, pad_ms: int = 40):
    """Strip near-silent head and tail, keeping a small pad.

    Frame-wise RMS rather than per-sample: a single loud sample in an
    otherwise silent stretch should not count as speech.
    """
    import numpy as np

    if waveform.size == 0:
        return waveform
    hop = max(1, int(0.01 * sr))
    n = len(waveform) // hop * hop
    if n < hop:
        return waveform
    frames = waveform[:n].reshape(-1, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    loud = np.flatnonzero(rms > threshold)
    if loud.size == 0:
        return waveform  # all quiet - return as-is rather than an empty clip
    pad = int(pad_ms / 1000 * sr)
    start = max(0, loud[0] * hop - pad)
    end = min(len(waveform), (loud[-1] + 1) * hop + pad)
    return waveform[start:end]


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
        self._mms_device: Optional[str] = None

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
            model, mms_device = device_utils.to_device_or_cpu(
                VitsModel.from_pretrained(source), self.device,
                what=f"the MMS-TTS {bridge} bridge model",
            )
            self._mms_device = mms_device
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
        out_path = self._output_path(text, config.METHOD_XTTS_ZERO_SHOT)
        try:
            tts.tts_to_file(
                text=text,
                speaker_wav=speaker_wavs,
                language=config.XTTS_BRIDGE_LANGUAGE,
                file_path=str(out_path),
            )
        except Exception:
            # Synthesis can fail even after the model loads fine - most often
            # because coqui-tts reads the reference clip through torchcodec,
            # which needs a complete FFmpeg shared-library set. Returning None
            # rather than raising is what makes speak()'s documented "falls
            # back to MMS if XTTS ... fails" actually hold: before this, a
            # synthesis-time error escaped speak() and the MMS bridge was
            # never tried at all.
            logger.exception("XTTS v2 synthesis failed; falling back to the MMS bridge")
            return None
        return SynthesisResult(out_path, confidence=0.45, method=config.METHOD_XTTS_ZERO_SHOT)

    def _synthesize_mms(self, text: str) -> Optional[SynthesisResult]:
        import numpy as np
        import torch
        import scipy.io.wavfile as wavfile

        bridge = config.MMS_DEFAULT_BRIDGE
        loaded = self._load_mms(bridge)
        if loaded is None:
            return None
        model, tokenizer, fine_tuned = loaded

        # The bridge tokenizer is native-script (Bengali). Romanized Kokborok
        # - what the UI and the MT model both use - would tokenize to nothing,
        # so transliterate first. Bengali-script input is passed through
        # untouched by latin_to_bengali(), so this is safe unconditionally.
        synth_text = text
        if translit.looks_romanized(text):
            synth_text = translit.latin_to_bengali(text)
            logger.info("Transliterated Romanized input for the Bengali bridge: %r -> %r",
                        text, synth_text)

        inputs = tokenizer(synth_text, return_tensors="pt")
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
        inputs = inputs.to(self._mms_device or self.device)
        with torch.no_grad():
            output = model(**inputs).waveform
        method = config.METHOD_MMS_FINE_TUNED if fine_tuned else config.METHOD_MMS_BRIDGE
        confidence = 0.65 if fine_tuned else 0.35
        out_path = self._output_path(text, method)
        waveform = output.squeeze().cpu().numpy()
        # VITS emits float32. Writing that straight out produces a 32-bit
        # float WAV, which browsers cannot decode - <audio> reports a valid
        # duration but plays silence. Convert to 16-bit PCM, which every
        # browser handles. Peak-normalize first (only when it would clip or
        # when the signal is quiet) so the result is audible without
        # distorting.
        # Trim leading/trailing silence. VITS pads the utterance, and on a
        # ~2s clip a 0.4s silent head reads as "nothing happened" - especially
        # since the UI starts playback automatically.
        waveform = _trim_silence(waveform, sr=model.config.sampling_rate)
        peak = float(np.abs(waveform).max()) if waveform.size else 0.0
        if peak > 0:
            waveform = waveform / peak * 0.95
        wavfile.write(
            str(out_path),
            rate=model.config.sampling_rate,
            data=(waveform * 32767.0).astype(np.int16),
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
