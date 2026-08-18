"""Romanized Kokborok -> Bengali script transliteration.

Why this exists: Lahja's UI and MT model both use Romanized Kokborok
(``trp_Latn``), but the only available TTS bridge is Meta's MMS-TTS for
Bengali, whose tokenizer is native-script. Feeding it Latin text drops every
character, so tts_engine skipped the bridge and reported "No TTS backend
available" for perfectly valid input. This module is the missing step that
comment called for.

It is an APPROXIMATION, deliberately so. Kokborok has no single official
romanization, and Bengali orthography carries distinctions Kokborok does not
use (and vice versa). The goal is a phoneme string close enough that a
Bengali voice model produces recognisable Kokborok speech - not scholarly
transliteration. Anything that needs to be exact should use Bengali-script
input directly.

Mapping rationale, checked against the Bengali-script transcript in
data/audio/kokborok_transcript.txt:

  * Bengali consonants carry an inherent vowel, so "borok" (people) is
    ব+র+ক = বরক with no vowel signs at all. Only a/i/u/e take a matra.
  * Kokborok's distinctive "w" (a central vowel, roughly schwa) has no
    Bengali equivalent. It maps to the inherent vowel - i.e. nothing - which
    is what the transcript does: "nwng" (you) appears as নং.
  * Word-final "ng" is the anusvara ং, not a full ঙ.
  * Adjacent consonants get an explicit hasant (্) to form a conjunct, so
    "gra" becomes গ্রা.
"""

from __future__ import annotations

import re

# Digraphs first - longest match wins, so "kh" never parses as k + h.
CONSONANTS: list[tuple[str, str]] = [
    ("chh", "ছ"),
    ("ch", "চ"),
    ("kh", "খ"),
    ("gh", "ঘ"),
    ("jh", "ঝ"),
    ("th", "থ"),
    ("dh", "ধ"),
    ("ph", "ফ"),
    ("bh", "ভ"),
    ("sh", "শ"),
    ("ng", "ং"),
    ("ny", "ঞ"),
    ("k", "ক"),
    ("g", "গ"),
    ("j", "জ"),
    ("t", "ত"),
    ("d", "দ"),
    ("n", "ন"),
    ("p", "প"),
    ("f", "ফ"),
    ("b", "ব"),
    ("m", "ম"),
    ("y", "য়"),
    ("r", "র"),
    ("l", "ল"),
    ("s", "স"),
    ("h", "হ"),
    ("w", "ও"),  # only when standing as a consonant/glide, see _is_glide
    ("v", "ভ"),
    ("z", "জ"),
    ("c", "ক"),
    ("q", "ক"),
    ("x", "ক্স"),
]

# Independent forms, used word-initially or after another vowel.
VOWELS_INDEPENDENT: dict[str, str] = {
    "a": "আ",
    "i": "ই",
    "u": "উ",
    "e": "এ",
    "o": "ও",
    "w": "অ",
}

# Dependent forms (matras) attached to the preceding consonant. "o" and "w"
# ride the consonant's inherent vowel, so they add nothing.
VOWELS_MATRA: dict[str, str] = {
    "a": "া",
    "i": "ি",
    "u": "ু",
    "e": "ে",
    "o": "",
    "w": "",
}

HASANT = "্"
VOWEL_CHARS = set("aeiouw")
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")


def _match_consonant(text: str, pos: int) -> tuple[str, str] | None:
    lowered = text.lower()
    for latin, bengali in CONSONANTS:
        if lowered.startswith(latin, pos):
            return latin, bengali
    return None


def transliterate_word(word: str) -> str:
    """Transliterate a single Romanized Kokborok word to Bengali script."""
    out: list[str] = []
    pos = 0
    prev_was_consonant = False
    lowered = word.lower()
    n = len(lowered)

    while pos < n:
        ch = lowered[pos]

        if ch in ("'", "’", "-"):
            pos += 1
            continue

        if ch in VOWEL_CHARS:
            # "w" before another vowel: word-initially it is a real glide
            # ("wa-") and takes ও, but after a consonant it contributes
            # nothing and the following vowel stands independent - which is
            # how the transcript spells "bagwi" as বাগই (গ + ই), not বাগওয়ই.
            if ch == "w" and pos + 1 < n and lowered[pos + 1] in VOWEL_CHARS - {"w"}:
                if not prev_was_consonant:
                    out.append("ও")
                prev_was_consonant = False
                pos += 1
                continue
            if prev_was_consonant:
                out.append(VOWELS_MATRA[ch])
            else:
                out.append(VOWELS_INDEPENDENT[ch])
            prev_was_consonant = False
            pos += 1
            continue

        matched = _match_consonant(lowered, pos)
        if matched is None:
            # Unknown character: pass it through rather than silently dropping
            # it, so debugging a bad mapping is possible.
            out.append(word[pos])
            prev_was_consonant = False
            pos += 1
            continue

        latin, bengali = matched
        # Two consonants in a row form a conjunct. Anusvara ং is a final/nasal
        # mark, not a joinable consonant, so it never takes a hasant.
        if prev_was_consonant and bengali != "ং" and out and out[-1] != "ং":
            out.append(HASANT)
        out.append(bengali)
        prev_was_consonant = bengali != "ং"
        pos += len(latin)

    return "".join(out)


def latin_to_bengali(text: str) -> str:
    """Transliterate Romanized text, leaving non-Latin runs untouched.

    Text already in Bengali script passes through unchanged, so this is safe
    to call unconditionally.
    """
    return _LATIN_WORD.sub(lambda m: transliterate_word(m.group(0)), text)


def looks_romanized(text: str) -> bool:
    """True if the text is predominantly Latin letters.

    Used to decide whether transliteration is needed at all; Bengali-script
    input must never be run through it.
    """
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    bengali = sum(1 for c in text if 0x0980 <= ord(c) <= 0x09FF)
    return latin > bengali
