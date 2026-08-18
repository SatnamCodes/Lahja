"""Import bootstrap and split-file IO for the scripts in ``scripts/``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .manifest import Utterance


def ensure_importable() -> Path:
    """Put ``asr/src`` on sys.path so scripts run without installation."""
    src = Path(__file__).resolve().parents[1]
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


def write_split(path: Path, utterances: list[Utterance], root: Path | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for u in utterances:
            fh.write(json.dumps(u.to_json(root), ensure_ascii=False) + "\n")
    return path


def read_split(path: Path, root: Path | None = None) -> list[Utterance]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}\nRun scripts/prepare_data.py first."
        )
    utterances: list[Utterance] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                utterances.append(Utterance.from_json(json.loads(line), root))
    if not utterances:
        raise ValueError(f"Split file is empty: {path}")
    return utterances
