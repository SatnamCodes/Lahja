"""
Speech-to-speech translation: chains the three existing engines
(asr_engine -> mt_engine -> tts_engine) at request time rather than being
its own trained model - there is no data anywhere in this project pairing
Kokborok speech with speech in another language, so a dedicated end-to-end
model isn't just unbuilt, it's currently untrainable.

Only eng -> trp is supported right now. trp -> eng is refused with a clear
explanation rather than silently producing nonsense: none of the live ASR
tiers (phoneme IPA, wrong-language Whisper guess) output real Kokborok
orthography, and feeding either into the trp_Latn-trained MT model would be
garbage-in-garbage-out. Once the fine-tuned ASR tier exists (see
asr_engine.py), trp -> eng becomes trivial to add here - same shape, other
direction.

Every stage's own method and confidence is returned alongside the final
result, and the overall confidence is the product of the stage confidences,
not just the last stage's - a translated, resynthesized result is never
more trustworthy than its weakest link.
"""

import logging
from pathlib import Path
from typing import Optional

from . import config
from .asr_engine import engine as asr_engine
from .mt_engine import engine as mt_engine
from .tts_engine import engine as tts_engine

logger = logging.getLogger("lahja.speech_translate")

METHOD_ENGLISH_ASR = "whisper_english_asr"


class StageInfo:
    def __init__(self, method: str, confidence: float, text: Optional[str] = None):
        self.method = method
        self.confidence = confidence
        self.text = text


class SpeechTranslateResult:
    def __init__(
        self,
        audio_path: Path,
        source_text: str,
        translated_text: str,
        confidence: float,
        stages: list[StageInfo],
    ):
        self.audio_path = audio_path
        self.source_text = source_text
        self.translated_text = translated_text
        self.confidence = confidence
        self.stages = stages


class SpeechTranslateEngine:
    def __init__(self):
        self._whisper_en_model = None
        self._whisper_en_processor = None
        self._whisper_en_load_failed = False
        self._device: Optional[str] = None

    @property
    def device(self) -> str:
        if self._device is None:
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"
        return self._device

    def _load_whisper_en(self):
        """A real English ASR pass, not a bridge - Whisper is genuinely
        strong on English, so this is labeled and scored as native quality,
        unlike every Kokborok ASR tier in asr_engine.py."""
        if self._whisper_en_model is not None or self._whisper_en_load_failed:
            return self._whisper_en_model, self._whisper_en_processor
        try:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            self._whisper_en_processor = WhisperProcessor.from_pretrained(config.WHISPER_MODEL_NAME)
            self._whisper_en_model = WhisperForConditionalGeneration.from_pretrained(
                config.WHISPER_MODEL_NAME
            ).to(self.device)
        except Exception:
            logger.exception("Whisper (English) failed to load")
            self._whisper_en_load_failed = True
        return self._whisper_en_model, self._whisper_en_processor

    def _transcribe_english(self, audio_path: Path) -> StageInfo:
        import librosa
        import torch

        model, processor = self._load_whisper_en()
        if model is None:
            raise RuntimeError("Whisper (English) is unavailable - check network access for the model download.")

        audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(self.device)
        forced_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
        with torch.no_grad():
            predicted_ids = model.generate(**inputs, forced_decoder_ids=forced_ids)
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        return StageInfo(METHOD_ENGLISH_ASR, confidence=0.9, text=text)

    def _transcribe_kokborok(self, audio_path: Path) -> StageInfo:
        result = asr_engine.transcribe(audio_path)
        if result.method != config.METHOD_ASR_FINE_TUNED:
            raise RuntimeError(
                "Kokborok speech-to-speech isn't available yet: the only live Kokborok ASR "
                f"tiers are '{config.METHOD_ASR_PHONEME_BRIDGE}' (IPA phonemes, not real "
                f"orthography) and '{config.METHOD_ASR_ZERO_SHOT_BRIDGE}' (a wrong-language "
                "guess) - neither is valid input for the translator, so chaining them would "
                "silently produce nonsense instead of a real translation. This direction "
                "activates automatically once the fine-tuned ASR tier is live."
            )
        return StageInfo(result.method, result.confidence, text=result.text)

    def translate_speech(
        self, audio_path: Path, source_language: str, target_language: str
    ) -> SpeechTranslateResult:
        if source_language == target_language:
            raise ValueError("source_language and target_language must differ")

        if source_language == "eng":
            asr_stage = self._transcribe_english(audio_path)
        elif source_language == "trp":
            asr_stage = self._transcribe_kokborok(audio_path)
        else:
            raise ValueError(f"Unsupported source_language '{source_language}'; expected 'eng' or 'trp'")

        if not asr_stage.text:
            raise ValueError("No speech detected in the uploaded audio")

        src_code = config.MT_LANG_CODE_MAP[source_language]
        tgt_code = config.MT_LANG_CODE_MAP[target_language]
        mt_result = mt_engine.translate(asr_stage.text, src_code, tgt_code)
        mt_stage = StageInfo(mt_result.method, mt_result.confidence, text=mt_result.text)

        tts_result = tts_engine.speak(mt_result.text, target_language)
        tts_stage = StageInfo(tts_result.method, tts_result.confidence)

        stages = [asr_stage, mt_stage, tts_stage]
        overall_confidence = asr_stage.confidence * mt_stage.confidence * tts_stage.confidence

        return SpeechTranslateResult(
            audio_path=tts_result.file_path,
            source_text=asr_stage.text,
            translated_text=mt_result.text,
            confidence=round(overall_confidence, 3),
            stages=stages,
        )


engine = SpeechTranslateEngine()
