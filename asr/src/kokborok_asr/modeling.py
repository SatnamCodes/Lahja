"""Whisper + LoRA model construction.

Whisper has no Kokborok (``trp``) language token, so we drive it through a
proxy language token — the same honest-approximation approach Lahja's TTS
side takes. ``model.language`` in the config controls which proxy is used;
``en`` is the default because our transcripts are Romanized Latin script.
Whatever is chosen gets recorded in ``results/metrics.jsonl``.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_device(preference: str = "auto") -> str:
    import torch

    if preference != "auto":
        return preference
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(device: str, precision: str = "auto"):
    """Pick a compute dtype. fp16 on consumer NVIDIA, fp32 on CPU."""
    import torch

    if precision == "fp32" or device == "cpu":
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    # auto: bf16 only where it is natively supported (Ampere+).
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_processor(cfg):
    from transformers import WhisperProcessor

    name = cfg.get("model.name", "openai/whisper-small")
    language = cfg.get("model.language", "en")
    task = cfg.get("model.task", "transcribe")
    logger.info("Loading processor %s (language=%s task=%s)", name, language, task)
    return WhisperProcessor.from_pretrained(name, language=language, task=task)


def build_model(cfg, *, for_training: bool = True, adapter_path: str | Path | None = None):
    """Load Whisper, optionally attaching a LoRA adapter."""
    import torch
    from transformers import WhisperForConditionalGeneration

    name = cfg.get("model.name", "openai/whisper-small")
    device = resolve_device(cfg.get("model.device", "auto"))
    language = cfg.get("model.language", "en")
    task = cfg.get("model.task", "transcribe")

    logger.info("Loading %s on %s", name, device)
    model = WhisperForConditionalGeneration.from_pretrained(name)

    # Let the model generate freely in the chosen proxy language: Whisper's
    # defaults suppress tokens and pin decoder ids in ways that fight a
    # fine-tune on an unseen language.
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    try:
        model.generation_config.language = language
        model.generation_config.task = task
    except Exception:  # noqa: BLE001 - older/newer generation configs differ
        logger.warning("Could not pin generation language/task on this transformers version")

    if adapter_path is not None:
        from peft import PeftModel

        logger.info("Attaching LoRA adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, str(adapter_path))
        if not for_training:
            model = model.merge_and_unload()
    elif for_training:
        model = attach_lora(cfg, model)

    if for_training:
        model.config.use_cache = False
        if cfg.get("train.gradient_checkpointing", True):
            model.enable_input_require_grads()
    else:
        model.config.use_cache = True
        dtype = resolve_dtype(device, cfg.get("model.precision", "auto"))
        model = model.to(device=device, dtype=dtype)
        model.eval()

    return model


def attach_lora(cfg, model):
    """Wrap Whisper in a LoRA adapter as configured."""
    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=int(cfg.get("lora.r", 32)),
        lora_alpha=int(cfg.get("lora.alpha", 64)),
        target_modules=list(cfg.get("lora.target_modules", ["q_proj", "v_proj"])),
        lora_dropout=float(cfg.get("lora.dropout", 0.05)),
        bias=cfg.get("lora.bias", "none"),
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "LoRA attached: %s trainable / %s total params (%.3f%%)",
        f"{trainable:,}", f"{total:,}", 100 * trainable / max(total, 1),
    )
    return model
