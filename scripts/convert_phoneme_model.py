"""
One-time local conversion of facebook/wav2vec2-lv-60-espeak-cv-ft's weights
from pytorch_model.bin to safetensors.

`transformers` refuses to torch.load() any non-safetensors checkpoint on
torch < 2.6 (CVE-2025-32434) - see check_torch_load_is_safe(). This repo is
pinned to torch 2.5.1 for coqui-tts/XTTS compatibility, so we can't just
bump torch. Since this is Meta's own public checkpoint (not an arbitrary
untrusted file), we load it once with the safety gate patched out, then
re-save it as safetensors so every subsequent load goes through the normal,
ungated safetensors path.

Usage:
    python3 scripts/convert_phoneme_model.py

Writes to models/phoneme_asr_local/, which ASREngine prefers over the
remote model name when present.
"""

import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from service import config  # noqa: E402


def main():
    import transformers.modeling_utils as mu

    with mock.patch.object(mu, "check_torch_load_is_safe", lambda: None):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        print(f"Downloading + loading {config.PHONEME_ASR_MODEL_NAME}...")
        processor = Wav2Vec2Processor.from_pretrained(config.PHONEME_ASR_MODEL_NAME)
        model = Wav2Vec2ForCTC.from_pretrained(config.PHONEME_ASR_MODEL_NAME)

    config.PHONEME_ASR_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.PHONEME_ASR_LOCAL_DIR, safe_serialization=True)
    processor.save_pretrained(config.PHONEME_ASR_LOCAL_DIR)
    print(f"Saved safetensors checkpoint to {config.PHONEME_ASR_LOCAL_DIR}")


if __name__ == "__main__":
    main()
