"""
Kokborok (trp) <-> English machine translation.

Standard NLLB-200 has no Kokborok entry, so this uses MWirelabs/kokborok-mt:
nllb-200-distilled-600M fine-tuned with a custom trp_Latn language token on
~36k parallel sentences (CC-BY-4.0). See
https://huggingface.co/MWirelabs/kokborok-mt for training details and
reported BLEU/COMET scores.
"""

import logging
from typing import Optional

from . import config

logger = logging.getLogger("lahja.mt")


class TranslationResult:
    def __init__(self, text: str, confidence: float, method: str):
        self.text = text
        self.confidence = confidence
        self.method = method


class MTEngine:
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device: Optional[str] = None
        self._load_failed = False

    @property
    def device(self) -> str:
        if self._device is None:
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"
            logger.info("MTEngine device: %s", self._device)
        return self._device

    def _load(self):
        if self._model is not None or self._load_failed:
            return self._model, self._tokenizer
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            logger.info("Loading Kokborok MT model %s on %s...", config.MT_MODEL_NAME, self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(config.MT_MODEL_NAME)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(config.MT_MODEL_NAME).to(self.device)
        except Exception:
            logger.exception("Kokborok MT model failed to load")
            self._load_failed = True
            self._model = None
            self._tokenizer = None
        return self._model, self._tokenizer

    def translate(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        text = text.strip()
        if not text:
            raise ValueError("text must not be empty")

        model, tokenizer = self._load()
        if model is None:
            raise RuntimeError(
                f"Kokborok MT model unavailable: check network access to download "
                f"{config.MT_MODEL_NAME} from Hugging Face"
            )

        tokenizer.src_lang = source_lang
        inputs = tokenizer(text, return_tensors="pt").to(self.device)
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)
        output_tokens = model.generate(
            **inputs, forced_bos_token_id=forced_bos_token_id, max_new_tokens=256
        )
        translated = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0].strip()
        return TranslationResult(translated, confidence=0.7, method=config.METHOD_MT_KOKBOROK_MT)


engine = MTEngine()
