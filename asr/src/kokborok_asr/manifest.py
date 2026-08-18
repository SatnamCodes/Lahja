"""Discovery and loading of the raw corpus.

``data/raw/`` is treated as strictly read-only: nothing in this module ever
writes to it. Two on-disk layouts are supported, auto-detected unless the
config pins one, because the exact shape of the Kokborok corpus isn't fixed
yet:

``manifest``
    A single CSV/TSV with columns ``audio_path,transcript,speaker_id``
    (extra columns are carried through as metadata).

``pairs``
    One audio file plus a matching transcript file per utterance, e.g.
    ``spk01/utt_0007.wav`` + ``spk01/utt_0007.txt``.

``jsonl``
    One JSON object per line, as produced by the repo-root
    ``scripts/build_asr_dataset.py``::

        {"audio": "data/asr_dataset/clip_0000.wav", "text": "...", "speaker_id": "ep001"}

    Key names are flexible (see ``AUDIO_KEYS``/``TEXT_KEYS``/``SPEAKER_KEYS``).
    Relative audio paths in this layout are resolved against ``audio_root``,
    **not** against the manifest's own directory: the Lahja manifests store
    repo-root-relative paths while themselves living in ``<repo>/data/``, so
    resolving them locally would silently look for
    ``<repo>/data/data/asr_dataset/...`` and report every clip as missing.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a")
TRANSCRIPT_EXTENSIONS = (".txt", ".lab", ".trans")
MANIFEST_NAMES = ("manifest.csv", "manifest.tsv", "metadata.csv", "metadata.tsv")
JSONL_MANIFEST_NAMES = ("manifest.jsonl", "asr_manifest_clean.jsonl", "asr_manifest.jsonl")
JSONL_SUFFIXES = (".jsonl", ".ndjson")

# Accepted row keys in the jsonl layout, in priority order.
AUDIO_KEYS = ("audio", "audio_path", "path", "wav")
TEXT_KEYS = ("text", "transcript", "sentence")
SPEAKER_KEYS = ("speaker_id", "speaker")


class CorpusError(RuntimeError):
    """Raised when the raw corpus can't be interpreted."""


@dataclass(frozen=True)
class Utterance:
    utt_id: str
    audio_path: Path
    transcript: str
    speaker_id: str
    duration: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self, root: Path | None = None) -> dict[str, Any]:
        path = self.audio_path
        if root is not None:
            try:
                path = path.relative_to(root)
            except ValueError:
                pass
        return {
            "utt_id": self.utt_id,
            "audio_path": str(path),
            "transcript": self.transcript,
            "speaker_id": self.speaker_id,
            "duration": self.duration,
            **({"meta": self.meta} if self.meta else {}),
        }

    @staticmethod
    def from_json(row: dict[str, Any], root: Path | None = None) -> "Utterance":
        path = Path(row["audio_path"])
        if root is not None and not path.is_absolute():
            path = root / path
        return Utterance(
            utt_id=row["utt_id"],
            audio_path=path,
            transcript=row["transcript"],
            speaker_id=str(row["speaker_id"]),
            duration=row.get("duration"),
            meta=row.get("meta", {}) or {},
        )


# ---------------------------------------------------------------------------
# speaker attribution
# ---------------------------------------------------------------------------

def _speaker_from_path(audio_path: Path, raw_dir: Path, strategy: str, delimiter: str) -> str:
    """Derive a speaker id from an audio path.

    Speaker identity drives the train/val/test split, so getting this wrong
    silently leaks a speaker across splits and inflates the scores. When in
    doubt we fail loudly rather than guess.
    """
    if strategy == "single":
        return "spk_all"
    if strategy == "manifest_field":
        # Only reached when the row carried no speaker field at all - the
        # loaders use the row's value when present. Failing here is right:
        # falling back to a path-derived guess is what silently produces one
        # bogus group for a whole corpus.
        raise CorpusError(
            f"speaker strategy 'manifest_field' requires a "
            f"{'/'.join(SPEAKER_KEYS)} value on every row, but {audio_path.name} "
            f"has none. Either add the field (see scripts/build_asr_dataset.py, "
            f"which writes an episode id per clip) or pick a different "
            f"data.speaker_strategy."
        )
    if strategy == "parent_dir":
        try:
            rel = audio_path.relative_to(raw_dir)
        except ValueError:
            rel = audio_path
        if len(rel.parts) < 2:
            raise CorpusError(
                f"speaker strategy 'parent_dir' needs each clip in a speaker folder, "
                f"but {audio_path} sits directly in {raw_dir}. Use "
                f"data.speaker_strategy: filename_prefix or single instead."
            )
        return rel.parts[-2]
    if strategy == "filename_prefix":
        stem = audio_path.stem
        if delimiter not in stem:
            raise CorpusError(
                f"speaker strategy 'filename_prefix' expects a {delimiter!r} in the "
                f"filename (e.g. spk01{delimiter}utt003.wav), but got {audio_path.name}"
            )
        return stem.split(delimiter, 1)[0]
    raise CorpusError(f"Unknown data.speaker_strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# duration probing
# ---------------------------------------------------------------------------

def make_utt_id(audio_path: Path, raw_dir: Path, prefix: str | None = None) -> str:
    """A corpus-unique utterance id derived from the path.

    The bare filename stem is not enough: layouts that put one folder per
    speaker reuse the same names (``spk01/utt000.wav``, ``spk02/utt000.wav``),
    so stem-only ids collide and the duplicate check silently drops every
    speaker but the first. Using the raw-dir-relative path keeps ids unique.

    Clips that live *outside* ``raw_dir`` - every row of a jsonl manifest
    pointing at the repo-root dataset - fall back to the bare filename, which
    is only unique as long as one flat directory is the sole source. ``prefix``
    (``data.utt_id_prefix``) namespaces those ids so a second source adding its
    own ``clip_0000.wav`` collides visibly at config time instead of having its
    rows quietly discarded by :func:`validate` as duplicates.
    """
    try:
        rel = audio_path.resolve().relative_to(raw_dir.resolve())
    except ValueError:
        rel = Path(audio_path.name)
    base = "-".join([*rel.parts[:-1], rel.stem]) or rel.stem
    return f"{prefix}-{base}" if prefix else base


def probe_duration(path: Path) -> float | None:
    """Read the duration from the audio header (cheap: no full decode)."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.samplerate:
            return round(info.frames / float(info.samplerate), 4)
    except Exception:  # noqa: BLE001 - probing is best-effort
        logger.debug("Could not probe duration for %s", path, exc_info=True)
    return None


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def _find_manifest(raw_dir: Path) -> Path | None:
    for name in (*MANIFEST_NAMES, *JSONL_MANIFEST_NAMES):
        candidate = raw_dir / name
        if candidate.exists():
            return candidate
    return None


def layout_for(manifest_path: Path) -> str:
    """Which loader a manifest file needs, from its suffix."""
    return "jsonl" if manifest_path.suffix.lower() in JSONL_SUFFIXES else "manifest"


def _resolve_audio(raw_path: str, audio_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else (audio_root / path).resolve()


def _first_key(row: dict[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return key
    return None


def _load_from_jsonl(
    manifest_path: Path,
    raw_dir: Path,
    strategy: str,
    delimiter: str,
    *,
    audio_root: Path,
    utt_id_prefix: str | None = None,
) -> list[Utterance]:
    """Load one JSON object per line. See the module docstring for the shape."""
    utterances: list[Utterance] = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusError(
                    f"{manifest_path}:{lineno}: malformed JSON - {exc.msg} "
                    f"(at column {exc.colno})"
                ) from exc
            if not isinstance(row, dict):
                raise CorpusError(
                    f"{manifest_path}:{lineno}: expected a JSON object per line, "
                    f"got {type(row).__name__}"
                )

            audio_key = _first_key(row, AUDIO_KEYS)
            text_key = _first_key(row, TEXT_KEYS)
            if audio_key is None or text_key is None:
                missing = "audio" if audio_key is None else "transcript"
                expected = AUDIO_KEYS if audio_key is None else TEXT_KEYS
                raise CorpusError(
                    f"{manifest_path}:{lineno}: no {missing} key. Expected one of "
                    f"{list(expected)}; row has {sorted(row)}."
                )

            audio_path = _resolve_audio(row[audio_key], audio_root)
            speaker_key = _first_key(row, SPEAKER_KEYS)
            if speaker_key is not None:
                speaker = str(row[speaker_key]).strip()
            else:
                speaker = _speaker_from_path(audio_path, raw_dir, strategy, delimiter)

            consumed = {audio_key, text_key, speaker_key}
            extra = {k: v for k, v in row.items() if k not in consumed}
            utterances.append(
                Utterance(
                    utt_id=make_utt_id(audio_path, raw_dir, utt_id_prefix)
                    or f"utt_{lineno:06d}",
                    audio_path=audio_path,
                    transcript=str(row[text_key]).strip(),
                    speaker_id=speaker,
                    meta=extra,
                )
            )
    return utterances


def _load_from_manifest(
    manifest_path: Path,
    raw_dir: Path,
    strategy: str,
    delimiter: str,
    *,
    audio_root: Path | None = None,
    utt_id_prefix: str | None = None,
) -> list[Utterance]:
    delim = "\t" if manifest_path.suffix.lower() == ".tsv" else ","
    # Default keeps the pre-existing behaviour for CSV manifests, whose paths
    # have always been relative to the manifest's own directory.
    audio_root = audio_root or manifest_path.parent
    utterances: list[Utterance] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delim)
        if reader.fieldnames is None:
            raise CorpusError(f"Manifest {manifest_path} is empty")
        fields = {f.strip(): f for f in reader.fieldnames if f}
        if "audio_path" not in fields or "transcript" not in fields:
            raise CorpusError(
                f"Manifest {manifest_path} must have 'audio_path' and 'transcript' "
                f"columns; found: {sorted(fields)}"
            )
        has_speaker = "speaker_id" in fields
        for i, row in enumerate(reader):
            raw_path = (row.get(fields["audio_path"]) or "").strip()
            if not raw_path:
                continue
            audio_path = _resolve_audio(raw_path, audio_root)
            transcript = (row.get(fields["transcript"]) or "").strip()
            if has_speaker and (row.get(fields["speaker_id"]) or "").strip():
                speaker = row[fields["speaker_id"]].strip()
            else:
                speaker = _speaker_from_path(audio_path, raw_dir, strategy, delimiter)
            extra = {
                k: v for k, v in row.items()
                if k and k not in {fields.get("audio_path"), fields.get("transcript"), fields.get("speaker_id")}
            }
            utterances.append(
                Utterance(
                    utt_id=make_utt_id(audio_path, raw_dir, utt_id_prefix) or f"utt_{i:06d}",
                    audio_path=audio_path,
                    transcript=transcript,
                    speaker_id=str(speaker),
                    meta=extra,
                )
            )
    return utterances


def _load_from_pairs(raw_dir: Path, strategy: str, delimiter: str) -> list[Utterance]:
    audio_files = sorted(
        p for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
    utterances: list[Utterance] = []
    for audio_path in audio_files:
        transcript_path = None
        for ext in TRANSCRIPT_EXTENSIONS:
            candidate = audio_path.with_suffix(ext)
            if candidate.exists():
                transcript_path = candidate
                break
        if transcript_path is None:
            logger.warning("No transcript beside %s - skipping", audio_path)
            continue
        transcript = transcript_path.read_text(encoding="utf-8").strip()
        utterances.append(
            Utterance(
                utt_id=make_utt_id(audio_path, raw_dir),
                audio_path=audio_path.resolve(),
                transcript=transcript,
                speaker_id=_speaker_from_path(audio_path, raw_dir, strategy, delimiter),
            )
        )
    return utterances


def discover(
    raw_dir: Path,
    *,
    layout: str = "auto",
    speaker_strategy: str = "parent_dir",
    speaker_delimiter: str = "_",
    probe_durations: bool = True,
    manifest_path: Path | None = None,
    audio_root: Path | None = None,
    utt_id_prefix: str | None = None,
) -> list[Utterance]:
    """Load every utterance from a read-only raw corpus.

    ``manifest_path`` (``paths.manifest_file``) reads a manifest that lives
    outside ``raw_dir`` - the Lahja manifests sit at ``<repo>/data/`` while
    ``raw_dir`` is ``asr/data/raw``. Nothing is written either way; the
    read-only guarantee on ``raw_dir`` is unaffected.

    ``audio_root`` (``data.audio_root``) is the base for relative audio paths
    in a manifest. For the jsonl layout it must be given explicitly and is the
    repo root; CSV manifests keep resolving against their own directory.
    """
    raw_dir = Path(raw_dir).resolve()
    if not raw_dir.exists():
        raise CorpusError(f"Raw data directory does not exist: {raw_dir}")

    if manifest_path is not None:
        manifest_path = Path(manifest_path).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = (raw_dir / manifest_path).resolve()
        if not manifest_path.exists():
            raise CorpusError(f"paths.manifest_file does not exist: {manifest_path}")
    else:
        manifest_path = _find_manifest(raw_dir)

    if layout == "auto":
        layout = layout_for(manifest_path) if manifest_path else "pairs"

    if layout in ("manifest", "jsonl"):
        if manifest_path is None:
            raise CorpusError(
                f"layout {layout!r} requested but none of "
                f"{(*MANIFEST_NAMES, *JSONL_MANIFEST_NAMES)} found in {raw_dir}. "
                f"Set paths.manifest_file to read one from elsewhere."
            )
        logger.info("Reading corpus from %s manifest: %s", layout, manifest_path)
        if layout == "jsonl":
            if audio_root is None:
                raise CorpusError(
                    f"layout 'jsonl' needs data.audio_root: audio paths in "
                    f"{manifest_path.name} are relative to a project root, not to the "
                    f"manifest's own directory, so resolving them without it would "
                    f"look for files that do not exist and report every clip missing."
                )
            utterances = _load_from_jsonl(
                manifest_path, raw_dir, speaker_strategy, speaker_delimiter,
                audio_root=Path(audio_root).expanduser().resolve(),
                utt_id_prefix=utt_id_prefix,
            )
        else:
            utterances = _load_from_manifest(
                manifest_path, raw_dir, speaker_strategy, speaker_delimiter,
                audio_root=Path(audio_root).expanduser().resolve() if audio_root else None,
                utt_id_prefix=utt_id_prefix,
            )
    elif layout == "pairs":
        logger.info("Reading corpus as audio/transcript pairs under: %s", raw_dir)
        utterances = _load_from_pairs(raw_dir, speaker_strategy, speaker_delimiter)
    else:
        raise CorpusError(
            f"Unknown data.layout: {layout!r} (expected auto/manifest/jsonl/pairs)"
        )

    if not utterances:
        raise CorpusError(
            f"No usable utterances found in {raw_dir}. Expected a manifest "
            f"({'/'.join((*MANIFEST_NAMES, *JSONL_MANIFEST_NAMES))}) or audio files with "
            f"matching {'/'.join(TRANSCRIPT_EXTENSIONS)} transcripts."
        )

    if probe_durations:
        utterances = [replace(u, duration=probe_duration(u.audio_path)) for u in utterances]
    return utterances


def validate(utterances: Iterable[Utterance]) -> tuple[list[Utterance], list[str]]:
    """Drop unusable utterances, returning (kept, human-readable problems)."""
    kept: list[Utterance] = []
    problems: list[str] = []
    seen_ids: set[str] = set()
    for u in utterances:
        if not u.audio_path.exists():
            problems.append(f"{u.utt_id}: audio file missing ({u.audio_path})")
            continue
        if not u.transcript.strip():
            problems.append(f"{u.utt_id}: empty transcript")
            continue
        if u.utt_id in seen_ids:
            problems.append(f"{u.utt_id}: duplicate utt_id - skipping the later one")
            continue
        seen_ids.add(u.utt_id)
        kept.append(u)
    return kept, problems
