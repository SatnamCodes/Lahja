import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
AUDIO_REF_DIR = DATA_DIR / "audio"  # reference/training clips: early ~10-20s bootstrap,
                                     # then the larger aligned batch (~30-100 utterances)
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_REF_DIR.mkdir(parents=True, exist_ok=True)

# Public host:port the audio_url should be built against.
PUBLIC_BASE_URL = os.environ.get("LAHJA_PUBLIC_BASE_URL", "http://localhost:8000")

# XTTS v2 has no "trp" (Kokborok) tokenizer. Kokborok is written here in
# Romanized Latin script, so we use the English grapheme-to-phoneme path as
# the closest available approximation for a Latin-script input.
XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_BRIDGE_LANGUAGE = "en"

# Track B: phonetically-adjacent bridge language models (Bengali primary,
# Assamese as an alternate), used zero-shot until/unless we fine-tune on the
# aligned Kokborok batch that arrives later.
MMS_MODEL_CANDIDATES = {
    "ben": "facebook/mms-tts-ben",
    "asm": "facebook/mms-tts-asm",
}
MMS_DEFAULT_BRIDGE = "ben"

METHOD_XTTS_ZERO_SHOT = "xtts_zero_shot"
METHOD_MMS_BRIDGE = "mms_bridge_zero_shot"
METHOD_MMS_FINE_TUNED = "mms_fine_tuned"

# --- Machine Translation (Kokborok <-> English) ---
# NLLB-200 has no native Kokborok entry either, but MWirelabs/kokborok-mt is
# nllb-200-distilled-600M fine-tuned with a custom trp_Latn token on ~36k
# parallel sentences (CC-BY-4.0): https://huggingface.co/MWirelabs/kokborok-mt
MT_MODEL_NAME = "MWirelabs/kokborok-mt"
MT_LANG_TRP = "trp_Latn"
MT_LANG_ENG = "eng_Latn"
MT_LANG_CODE_MAP = {"trp": MT_LANG_TRP, "eng": MT_LANG_ENG, "en": MT_LANG_ENG}
METHOD_MT_KOKBOROK_MT = "kokborok_mt_nllb"

# --- Chatbot (Kokborok question -> Kokborok answer, bridged via English) ---
# No LLM has native Kokborok fluency, so we translate trp->eng with the MT
# model above, ask a free-tier hosted LLM, then translate the answer back.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
METHOD_CHAT_MT_BRIDGE = "mt_bridge_llm"

# --- ASR (Kokborok speech -> text) ---
# No native or bridge-language Kokborok ASR model exists (Meta MMS's
# 1,107-language ASR set does not include trp). Falls back to general
# multilingual Whisper (low confidence) until a fine-tuned checkpoint is
# placed at WHISPER_FINETUNED_DIR - see scripts/finetune_whisper.py, which
# expects a manifest at data/asr_manifest.jsonl.
WHISPER_MODEL_NAME = "openai/whisper-small"
WHISPER_FINETUNED_DIR = BASE_DIR / "models" / "whisper_finetuned" / "trp"
ASR_MANIFEST_PATH = DATA_DIR / "asr_manifest.jsonl"
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
METHOD_ASR_FINE_TUNED = "whisper_fine_tuned"
METHOD_ASR_ZERO_SHOT_BRIDGE = "whisper_zero_shot_bridge"
