"""YAML config loading with dotted-key CLI overrides.

Every script in this repo is driven by a YAML file plus optional
``--set key.path=value`` overrides, so no path or hyperparameter is ever
hardcoded in a script body.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

_MISSING = object()


def _coerce(raw: str) -> Any:
    """Parse an override value as YAML, so ints/floats/bools/lists work."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


class Config:
    """A nested dict with dotted-path access."""

    def __init__(self, data: dict[str, Any], source: Path | None = None):
        self._data = data
        self.source = source

    # -- access ---------------------------------------------------------
    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, path: str) -> Any:
        value = self.get(path, _MISSING)
        if value is _MISSING:
            raise KeyError(
                f"Required config key '{path}' is missing"
                + (f" from {self.source}" if self.source else "")
            )
        return value

    def set(self, path: str, value: Any) -> None:
        parts = path.split(".")
        node = self._data
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = value

    def section(self, path: str) -> dict[str, Any]:
        value = self.get(path, {})
        return value if isinstance(value, dict) else {}

    # -- misc -----------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def resolve_path(self, path: str, default: Any = None) -> Path:
        """Resolve a config path value relative to the project root."""
        value = self.get(path, default)
        if value is None:
            raise KeyError(f"Required path config key '{path}' is missing")
        p = Path(str(value)).expanduser()
        return p if p.is_absolute() else (project_root() / p)

    def __repr__(self) -> str:
        return f"Config({json.dumps(self._data, indent=2, default=str)})"


def project_root() -> Path:
    """Repo root: .../asr (this file lives at asr/src/kokborok_asr/config.py)."""
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML config and apply ``key.path=value`` overrides."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = project_root() / p
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}: {p}")

    cfg = Config(data, source=p)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.path=value, got: {item!r}")
        key, _, raw = item.partition("=")
        cfg.set(key.strip(), _coerce(raw.strip()))
    return cfg


def add_config_args(parser) -> None:
    """Attach the standard --config/--set flags to an argparse parser."""
    parser.add_argument(
        "--config",
        default="configs/whisper_small_lora.yaml",
        help="Path to the YAML config (default: configs/whisper_small_lora.yaml)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY.PATH=VALUE",
        help="Override a config value; repeatable. Example: --set train.batch_size=2",
    )
