"""Transcript normalization.

Kokborok here is Romanized (Latin script), matching the convention the TTS
side of Lahja already uses. Normalization is deliberately conservative: we
lowercase, strip punctuation and collapse whitespace, but never touch the
letters themselves, so we don't quietly destroy distinctions the language
makes (e.g. the ``w`` vowel in "Nwng", "tamwi").
"""

from __future__ import annotations

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
    """Build a normalize() partial from the ``text`` section of a config."""
    section = cfg.section("text")
    return lambda s: normalize(
        s,
        lowercase=section.get("lowercase", True),
        strip_punctuation=section.get("strip_punctuation", True),
        punctuation=section.get("punctuation", _DEFAULT_PUNCT),
        unicode_form=section.get("unicode_form", "NFC"),
    )
