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
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a")
TRANSCRIPT_EXTENSIONS = (".txt", ".lab", ".trans")
MANIFEST_NAMES = ("manifest.csv", "manifest.tsv", "metadata.csv", "metadata.tsv")


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

def make_utt_id(audio_path: Path, raw_dir: Path) -> str:
    """A corpus-unique utterance id derived from the path.

    The bare filename stem is not enough: layouts that put one folder per
    speaker reuse the same names (``spk01/utt000.wav``, ``spk02/utt000.wav``),
    so stem-only ids collide and the duplicate check silently drops every
    speaker but the first. Using the raw-dir-relative path keeps ids unique.
    """
    try:
        rel = audio_path.resolve().relative_to(raw_dir.resolve())
    except ValueError:
        rel = Path(audio_path.name)
    return "-".join([*rel.parts[:-1], rel.stem]) or rel.stem


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
    for name in MANIFEST_NAMES:
        candidate = raw_dir / name
        if candidate.exists():
            return candidate
    return None


def _load_from_manifest(manifest_path: Path, raw_dir: Path, strategy: str, delimiter: str) -> list[Utterance]:
    delim = "\t" if manifest_path.suffix.lower() == ".tsv" else ","
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
            audio_path = Path(raw_path)
            if not audio_path.is_absolute():
                audio_path = (manifest_path.parent / audio_path).resolve()
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
                    utt_id=make_utt_id(audio_path, raw_dir) or f"utt_{i:06d}",
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
) -> list[Utterance]:
    """Load every utterance from a read-only raw corpus directory."""
    raw_dir = Path(raw_dir).resolve()
    if not raw_dir.exists():
        raise CorpusError(f"Raw data directory does not exist: {raw_dir}")

    manifest_path = _find_manifest(raw_dir)
    if layout == "auto":
        layout = "manifest" if manifest_path else "pairs"

    if layout == "manifest":
        if manifest_path is None:
            raise CorpusError(
                f"layout 'manifest' requested but none of {MANIFEST_NAMES} found in {raw_dir}"
            )
        logger.info("Reading corpus from manifest: %s", manifest_path)
        utterances = _load_from_manifest(manifest_path, raw_dir, speaker_strategy, speaker_delimiter)
    elif layout == "pairs":
        logger.info("Reading corpus as audio/transcript pairs under: %s", raw_dir)
        utterances = _load_from_pairs(raw_dir, speaker_strategy, speaker_delimiter)
    else:
        raise CorpusError(f"Unknown data.layout: {layout!r} (expected auto/manifest/pairs)")

    if not utterances:
        raise CorpusError(
            f"No usable utterances found in {raw_dir}. Expected either a manifest CSV "
            f"({'/'.join(MANIFEST_NAMES)}) or audio files with matching "
            f"{'/'.join(TRANSCRIPT_EXTENSIONS)} transcripts."
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
