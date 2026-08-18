"""Transcript normalization.

Kokborok is written in either Bengali or Latin script. The ASR corpus here is
**Bengali script** (data/asr_manifest_clean.jsonl), which is why the Whisper
proxy language token is ``bn``. Lahja's TTS side uses Romanized Kokborok
instead, so the two components do not share a script - do not assume either
convention when reading transcripts from the other.

Normalization is deliberately conservative: we lowercase, strip punctuation
and collapse whitespace, but never touch the letters themselves, so we don't
quietly destroy distinctions the language makes. ``lowercase`` is a no-op on
Bengali script and is kept only for Romanized input (where it matters, e.g.
the ``w`` vowel in "Nwng", "tamwi").

The default punctuation set below is ASCII-only. Bengali-script transcripts
use the danda ``।`` and double danda ``॥`` as sentence punctuation; configs
handling Bengali text should add them via ``text.punctuation`` (the shipped
configs/whisper_small_lora.yaml does), or the danda stays attached to the
preceding word and counts as a word error at scoring time.
"""

from __future__ import annotations

import functools
import re
import unicodedata

# Punctuation we strip for scoring. Apostrophes/hyphens are kept by default
# because they can be lexical in Romanized Kokborok rather than decorative.
_DEFAULT_PUNCT = r"""!"#$%&()*+,./:;<=>?@[\]^_`{|}~"""

_WHITESPACE_RE = re.compile(r"\s+")


def normalize(
    text: str,
    *,
    lowercase: bool = True,
    strip_punctuation: bool = True,
    punctuation: str = _DEFAULT_PUNCT,
    unicode_form: str = "NFC",
) -> str:
    """Normalize a transcript for training targets and WER/CER scoring."""
    if text is None:
        return ""
    out = unicodedata.normalize(unicode_form, str(text))
    if lowercase:
        out = out.lower()
    if strip_punctuation and punctuation:
        table = {ord(ch): " " for ch in punctuation}
        out = out.translate(table)
    return _WHITESPACE_RE.sub(" ", out).strip()


def normalizer_from_config(cfg) -> callable:
    """Build a normalize() partial from the ``text`` section of a config.

    Returns a :func:`functools.partial`, not a lambda, and that matters: the
    normalizer is held by ``WhisperASRDataset``, which torch pickles when
    ``train.dataloader_num_workers`` > 0 spawns worker processes. A lambda is a
    local object and cannot be pickled, so training died at step 0 with
    "Can't pickle local object ... normalizer_from_config.<locals>.<lambda>".
    A partial over a module-level function pickles cleanly. The smoke config
    runs with 0 workers, so it never serializes the dataset and cannot catch a
    regression here - keep this picklable by hand.
    """
    section = cfg.section("text")
    return functools.partial(
        normalize,
        lowercase=section.get("lowercase", True),
        strip_punctuation=section.get("strip_punctuation", True),
        punctuation=section.get("punctuation", _DEFAULT_PUNCT),
        unicode_form=section.get("unicode_form", "NFC"),
    )
