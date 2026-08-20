"""Shared helpers for placing models on a device with graceful degradation.

Lahja runs four independent models - MT (~600M params), Whisper ASR, a
wav2vec2 phoneme recognizer, and an MMS/VITS vocoder - in a single process.
On a 6 GB consumer GPU they do not all fit, and whichever one happens to load
last hits CUDA OOM. Before this module, that surfaced in confusing ways: the
feature either failed outright or silently dropped to a much worse tier (the
phoneme ASR OOM-ing meant transcribe() fell through to untuned Whisper and
returned repetition-loop garbage at confidence 0.15).

Falling back to CPU keeps the better model, trading latency for correctness,
and matches the "degrade gracefully rather than fail outright" contract the
engine modules already document.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("lahja.device")


def is_oom(exc: BaseException) -> bool:
    """True for CUDA out-of-memory, across torch versions.

    ``torch.OutOfMemoryError`` only exists on newer torch; older versions
    raise a plain ``RuntimeError`` carrying the detail in its message, so
    check both rather than relying on the type alone.
    """
    try:
        import torch

        oom_type = getattr(torch, "OutOfMemoryError", None)
        if oom_type is not None and isinstance(exc, oom_type):
            return True
    except Exception:  # noqa: BLE001 - torch may be absent entirely
        pass
    return "out of memory" in str(exc).lower()


def release_vram() -> None:
    """Drop cached CUDA blocks left behind by a failed ``.to(device)``."""
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - best-effort cleanup
        pass


def to_device_or_cpu(model: Any, device: str, *, what: str) -> tuple[Any, str]:
    """Move ``model`` to ``device``, falling back to CPU on CUDA OOM.

    Returns ``(model, actual_device)`` so the caller can record where the
    model really landed - inputs must be moved to the same place, and a
    silently-wrong device is worse than a slow one.
    """
    try:
        return model.to(device), device
    except Exception as exc:
        if not is_oom(exc):
            raise
        logger.warning(
            "Not enough VRAM for %s on %s (%s); falling back to CPU. "
            "This will be slower but keeps the better model.",
            what, device, type(exc).__name__,
        )
        release_vram()
        return model.to("cpu"), "cpu"
