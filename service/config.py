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
