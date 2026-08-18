"""Speaker-disjoint train/val/test splitting.

Splitting by utterance would put the same voice in train and test, which
inflates the scores badly on a small corpus — the model recognizes the
speaker, not the language. Every split here is disjoint by speaker, and the
module refuses to produce a leaky split by accident.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict

from .manifest import Utterance

logger = logging.getLogger(__name__)


class SplitError(RuntimeError):
    """Raised when a speaker-disjoint split isn't possible as configured."""


def _stable_key(speaker: str, seed: int) -> str:
    """Deterministic per-speaker shuffle key (stable across machines/runs)."""
    return hashlib.sha256(f"{seed}:{speaker}".encode("utf-8")).hexdigest()


def group_by_speaker(utterances: list[Utterance]) -> dict[str, list[Utterance]]:
    groups: dict[str, list[Utterance]] = defaultdict(list)
    for u in utterances:
        groups[u.speaker_id].append(u)
    return dict(groups)


def split_by_speaker(
    utterances: list[Utterance],
    ratios: dict[str, float],
    *,
    seed: int = 1234,
    allow_speaker_overlap: bool = False,
) -> dict[str, list[Utterance]]:
    """Partition utterances into splits with no speaker shared between them.

    Speakers are assigned greedily, largest first, to whichever split is
    furthest below its target share of utterances. This keeps the split
    proportions close to the requested ratios even when speakers contribute
    very different numbers of utterances.
    """
    if not utterances:
        raise SplitError("No utterances to split")

    active = {name: float(r) for name, r in ratios.items() if float(r) > 0}
    if not active:
        raise SplitError(f"All split ratios are zero: {ratios}")
    total_ratio = sum(active.values())
    if abs(total_ratio - 1.0) > 1e-6:
        logger.warning("Split ratios sum to %.4f, normalizing to 1.0", total_ratio)
        active = {k: v / total_ratio for k, v in active.items()}

    groups = group_by_speaker(utterances)
    n_speakers = len(groups)

    if n_speakers < len(active):
        message = (
            f"Cannot build a speaker-disjoint {'/'.join(active)} split: the corpus has "
            f"only {n_speakers} speaker(s) ({', '.join(sorted(groups))}) but "
            f"{len(active)} non-empty splits were requested.\n"
            f"Options: (a) add recordings from more speakers, (b) reduce the number of "
            f"splits (e.g. set split.ratios.val to 0), or (c) set "
            f"split.allow_speaker_overlap: true to fall back to an utterance-level "
            f"split - which leaks speakers across splits and will report optimistic WER."
        )
        if not allow_speaker_overlap:
            raise SplitError(message)
        logger.warning("%s\nProceeding with a LEAKY utterance-level split as configured.", message)
        return _utterance_level_split(utterances, active, seed)

    total = len(utterances)
    targets = {name: ratio * total for name, ratio in active.items()}
    assigned: dict[str, list[Utterance]] = {name: [] for name in active}

    ordered = sorted(
        groups.items(),
        key=lambda kv: (-len(kv[1]), _stable_key(kv[0], seed)),
    )

    for speaker, utts in ordered:
        empty = [name for name in active if not assigned[name]]
        if empty:
            # Seed every split with at least one speaker before balancing,
            # so a small corpus can't leave val/test empty.
            choice = min(empty, key=lambda name: (-targets[name], name))
        else:
            choice = max(
                active,
                key=lambda name: (targets[name] - len(assigned[name]), name),
            )
        assigned[choice].extend(utts)

    for name, utts in assigned.items():
        if not utts:
            raise SplitError(
                f"Split '{name}' ended up empty. With {n_speakers} speakers, try "
                f"adjusting split.ratios or recording more speakers."
            )
        utts.sort(key=lambda u: u.utt_id)

    return assigned


def _utterance_level_split(
    utterances: list[Utterance], ratios: dict[str, float], seed: int
) -> dict[str, list[Utterance]]:
    """Leaky fallback: only used when explicitly allowed in the config."""
    ordered = sorted(utterances, key=lambda u: _stable_key(u.utt_id, seed))
    assigned: dict[str, list[Utterance]] = {name: [] for name in ratios}
    names = list(ratios)
    boundaries: list[tuple[str, int]] = []
    start = 0
    for i, name in enumerate(names):
        end = len(ordered) if i == len(names) - 1 else start + int(round(ratios[name] * len(ordered)))
        boundaries.append((name, end))
        start = end
    start = 0
    for name, end in boundaries:
        assigned[name] = ordered[start:end]
        start = end
    return assigned


def summarize(splits: dict[str, list[Utterance]]) -> dict[str, dict]:
    """Per-split stats, including a leak check that must always report 0."""
    speaker_sets = {name: {u.speaker_id for u in utts} for name, utts in splits.items()}
    summary: dict[str, dict] = {}
    for name, utts in splits.items():
        durations = [u.duration for u in utts if u.duration]
        summary[name] = {
            "utterances": len(utts),
            "speakers": sorted(speaker_sets[name]),
            "n_speakers": len(speaker_sets[name]),
            "hours": round(sum(durations) / 3600.0, 4) if durations else None,
        }
    overlaps = {}
    names = sorted(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = speaker_sets[a] & speaker_sets[b]
            if shared:
                overlaps[f"{a}|{b}"] = sorted(shared)
    summary["_speaker_overlap"] = overlaps
    return summary
