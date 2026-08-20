#!/usr/bin/env python3
"""Assertions for the jsonl manifest layout and its config knobs.

    python scripts/test_manifest_jsonl.py

Standalone (no pytest - it is not a dependency of this component). Exits 0 if
every check passes, 1 on the first failure, with the failing expectation named.

The centrepiece is the audio_root check. Relative paths in
data/asr_manifest_clean.jsonl are repo-root-relative while the manifest itself
lives in <repo>/data/, so resolving them against the manifest's own directory
yields <repo>/data/data/asr_dataset/... - a path that does not exist. Nothing
raises when that happens: validate() drops every row as "audio file missing"
and discover() then blames the wrong thing entirely. So this asserts both
directions - that audio_root=<repo> finds real files on disk, and that the
naive resolution does not - because a test that only checks "no exception" would
pass just as happily while loading zero usable clips.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kokborok_asr import manifest as M  # noqa: E402

ASR_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ASR_DIR.parent
RAW_DIR = ASR_DIR / "data" / "raw"
CLEAN_MANIFEST = REPO_ROOT / "data" / "asr_manifest_clean.jsonl"

failures: list[str] = []
checks = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global checks
    checks += 1
    if condition:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}" + (f"\n       {detail}" if detail else ""))
        failures.append(label)


def expect_error(label: str, fn, *, contains: list[str]) -> None:
    """Assert fn() raises CorpusError whose message mentions each fragment."""
    global checks
    checks += 1
    try:
        fn()
    except M.CorpusError as exc:
        missing = [frag for frag in contains if frag not in str(exc)]
        if missing:
            print(f"  FAIL {label}\n       raised, but message lacks {missing}\n       got: {exc}")
            failures.append(label)
        else:
            print(f"  ok   {label}")
        return
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL {label}\n       raised {type(exc).__name__}, expected CorpusError: {exc}")
        failures.append(label)
        return
    print(f"  FAIL {label}\n       no exception raised")
    failures.append(label)


def write_jsonl(tmp: Path, name: str, rows: list) -> Path:
    path = tmp / name
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write((row if isinstance(row, str) else json.dumps(row, ensure_ascii=False)) + "\n")
    return path


def main() -> int:
    if not CLEAN_MANIFEST.exists():
        print(f"SKIP: {CLEAN_MANIFEST} not found; run scripts/build_asr_dataset.py "
              f"and scripts/filter_bengali_script.py from the repo root first.")
        return 1

    tmp_dir = tempfile.mkdtemp(prefix="manifest_jsonl_test_")
    tmp = Path(tmp_dir)

    # -- 1. audio_root: the real manifest, resolved correctly -----------------
    print("\n[1] audio_root resolves repo-root-relative paths to real files")
    utts = M.discover(
        RAW_DIR,
        layout="jsonl",
        manifest_path=CLEAN_MANIFEST,
        audio_root=REPO_ROOT,
        speaker_strategy="manifest_field",
        probe_durations=True,
    )
    check("loaded 235 utterances", len(utts) == 235, f"got {len(utts)}")
    on_disk = [u for u in utts if u.audio_path.exists()]
    check(
        "every audio_path exists ON DISK (not merely constructed)",
        len(on_disk) == len(utts),
        f"{len(utts) - len(on_disk)} of {len(utts)} missing; first bad: "
        f"{next((u.audio_path for u in utts if not u.audio_path.exists()), None)}",
    )
    check(
        "durations probed from real audio",
        all(u.duration and u.duration > 0 for u in utts),
        "some utterances have no duration - probe_duration() could not read them",
    )
    kept, problems = M.validate(utts)
    check("validate() keeps all 235", len(kept) == 235, f"kept {len(kept)}, problems: {problems[:3]}")

    # -- 2. audio_root: the naive resolution must FAIL ------------------------
    # Without this, check 1 could pass for the wrong reason.
    print("\n[2] the naive resolution (manifest's own dir) does NOT find the files")
    naive = M._resolve_audio("data/asr_dataset/clip_0000.wav", CLEAN_MANIFEST.parent)
    correct = M._resolve_audio("data/asr_dataset/clip_0000.wav", REPO_ROOT)
    check("naive path is the doubled data/data/ path", naive.parts[-3:-1] == ("data", "asr_dataset")
          and "data/data/" in str(naive).replace("\\", "/"), f"got {naive}")
    check("naive path does not exist", not naive.exists(), f"unexpectedly exists: {naive}")
    check("audio_root path does exist", correct.exists(), f"missing: {correct}")
    expect_error(
        "layout jsonl without audio_root is refused outright",
        lambda: M.discover(RAW_DIR, layout="jsonl", manifest_path=CLEAN_MANIFEST),
        contains=["data.audio_root"],
    )

    # -- 3. speaker_id from the row (manifest_field) --------------------------
    print("\n[3] manifest_field speaker strategy")
    groups = sorted({u.speaker_id for u in utts})
    check("18 distinct groups", len(groups) == 18, f"got {len(groups)}: {groups}")
    check("ids are ep001..ep018", groups == [f"ep{i:03d}" for i in range(1, 19)], f"got {groups}")
    check(
        "no group id leaked from the path",
        "asr_dataset" not in groups,
        "speaker fell back to the parent directory instead of the row field",
    )
    expect_error(
        "manifest_field with no speaker key on the row fails loudly",
        lambda: M.discover(
            RAW_DIR, layout="jsonl", audio_root=REPO_ROOT,
            speaker_strategy="manifest_field", probe_durations=False,
            manifest_path=write_jsonl(tmp, "nospeaker.jsonl", [
                {"audio": "data/asr_dataset/clip_0000.wav", "text": "ক"},
            ]),
        ),
        contains=["manifest_field", "speaker_id"],
    )

    # -- 4. malformed JSON names the 1-based line number ---------------------
    print("\n[4] malformed JSON reports its line number")
    bad = write_jsonl(tmp, "bad.jsonl", [
        {"audio": "data/asr_dataset/clip_0000.wav", "text": "ক"},
        {"audio": "data/asr_dataset/clip_0001.wav", "text": "খ"},
        '{"audio": "data/asr_dataset/clip_0002.wav", "text": OOPS}',
    ])
    expect_error(
        "CorpusError names line 3",
        lambda: M.discover(RAW_DIR, layout="jsonl", manifest_path=bad, audio_root=REPO_ROOT,
                           speaker_strategy="single", probe_durations=False),
        contains=["bad.jsonl:3", "malformed JSON"],
    )
    not_object = write_jsonl(tmp, "notobj.jsonl", ['["a", "b"]'])
    expect_error(
        "a JSON array line is rejected, not silently skipped",
        lambda: M.discover(RAW_DIR, layout="jsonl", manifest_path=not_object, audio_root=REPO_ROOT,
                           speaker_strategy="single", probe_durations=False),
        contains=["notobj.jsonl:1", "expected a JSON object"],
    )

    # -- 5. field aliases ----------------------------------------------------
    print("\n[5] field aliases")
    aliased = write_jsonl(tmp, "aliases.jsonl", [
        {"audio": "data/asr_dataset/clip_0000.wav", "text": "a", "speaker_id": "g1"},
        {"audio_path": "data/asr_dataset/clip_0001.wav", "transcript": "b", "speaker": "g2"},
        {"path": "data/asr_dataset/clip_0002.wav", "sentence": "c"},
        # clip_0004, not clip_0003: clip indices are transcript entry numbers,
        # so they are non-contiguous wherever an entry was dropped as
        # non-speech (entry 3 is the 00:28 [music] tag).
        {"wav": "data/asr_dataset/clip_0004.wav", "text": "d"},
    ])
    alias_utts = M.discover(RAW_DIR, layout="jsonl", manifest_path=aliased, audio_root=REPO_ROOT,
                            speaker_strategy="single", probe_durations=False)
    check("all 4 alias spellings load", len(alias_utts) == 4, f"got {len(alias_utts)}")
    check("transcripts read via every text alias",
          [u.transcript for u in alias_utts] == ["a", "b", "c", "d"],
          f"got {[u.transcript for u in alias_utts]}")
    check("speaker_id and speaker both honoured",
          [u.speaker_id for u in alias_utts[:2]] == ["g1", "g2"],
          f"got {[u.speaker_id for u in alias_utts[:2]]}")
    check("rows without a speaker key fall back to the strategy",
          [u.speaker_id for u in alias_utts[2:]] == ["spk_all", "spk_all"],
          f"got {[u.speaker_id for u in alias_utts[2:]]}")
    check("all alias-resolved paths exist on disk",
          all(u.audio_path.exists() for u in alias_utts))

    # -- 6. rows missing a required key --------------------------------------
    print("\n[6] rows missing audio or text are rejected, listing the keys present")
    expect_error(
        "missing text key names the row's actual keys",
        lambda: M.discover(RAW_DIR, layout="jsonl", audio_root=REPO_ROOT,
                           speaker_strategy="single", probe_durations=False,
                           manifest_path=write_jsonl(tmp, "notext.jsonl", [
                               {"audio": "data/asr_dataset/clip_0000.wav", "spk": "x"},
                           ])),
        contains=["notext.jsonl:1", "transcript", "'audio'", "'spk'"],
    )
    expect_error(
        "missing audio key names the row's actual keys",
        lambda: M.discover(RAW_DIR, layout="jsonl", audio_root=REPO_ROOT,
                           speaker_strategy="single", probe_durations=False,
                           manifest_path=write_jsonl(tmp, "noaudio.jsonl", [
                               {"text": "ক", "note": "y"},
                           ])),
        contains=["noaudio.jsonl:1", "audio", "'note'"],
    )

    # -- 7. utt_id_prefix namespacing ---------------------------------------
    print("\n[7] utt_id_prefix namespaces ids")
    plain = M.discover(RAW_DIR, layout="jsonl", manifest_path=CLEAN_MANIFEST, audio_root=REPO_ROOT,
                       speaker_strategy="manifest_field", probe_durations=False)
    prefixed = M.discover(RAW_DIR, layout="jsonl", manifest_path=CLEAN_MANIFEST, audio_root=REPO_ROOT,
                          speaker_strategy="manifest_field", probe_durations=False,
                          utt_id_prefix="wol")
    check("unprefixed ids are bare stems", plain[0].utt_id == "clip_0000", f"got {plain[0].utt_id}")
    check("prefixed ids carry the prefix", prefixed[0].utt_id == "wol-clip_0000",
          f"got {prefixed[0].utt_id}")
    check("ids stay unique under prefixing", len({u.utt_id for u in prefixed}) == len(prefixed))
    check("prefix does not disturb paths", prefixed[0].audio_path == plain[0].audio_path)

    # -- 8. auto layout dispatch by suffix ----------------------------------
    print("\n[8] auto mode dispatches on suffix")
    check("layout_for(.jsonl) == jsonl", M.layout_for(Path("a/manifest.jsonl")) == "jsonl")
    check("layout_for(.ndjson) == jsonl", M.layout_for(Path("a/x.ndjson")) == "jsonl")
    check("layout_for(.csv) == manifest", M.layout_for(Path("a/manifest.csv")) == "manifest")
    check("layout_for(.tsv) == manifest", M.layout_for(Path("a/metadata.tsv")) == "manifest")
    auto = M.discover(RAW_DIR, layout="auto", manifest_path=CLEAN_MANIFEST, audio_root=REPO_ROOT,
                      speaker_strategy="manifest_field", probe_durations=False)
    check("auto picks the jsonl loader for the real manifest", len(auto) == 235, f"got {len(auto)}")
    check("asr_manifest_clean.jsonl is an auto-detected name",
          "asr_manifest_clean.jsonl" in M.JSONL_MANIFEST_NAMES)
    expect_error(
        "a missing paths.manifest_file is reported as such",
        lambda: M.discover(RAW_DIR, layout="jsonl", audio_root=REPO_ROOT,
                           manifest_path=tmp / "does_not_exist.jsonl"),
        contains=["paths.manifest_file", "does not exist"],
    )

    # -- 9. regression: the pre-existing layouts still work ------------------
    print("\n[9] regression - csv manifest and pairs layouts unchanged")
    csv_root = tmp / "csvcorpus"
    (csv_root / "spk01").mkdir(parents=True)
    src_wav = REPO_ROOT / "data" / "asr_dataset" / "clip_0000.wav"
    (csv_root / "spk01" / "utt000.wav").write_bytes(src_wav.read_bytes())
    (csv_root / "manifest.csv").write_text(
        "audio_path,transcript,speaker_id\nspk01/utt000.wav,hello,spk01\n", encoding="utf-8"
    )
    csv_utts = M.discover(csv_root, layout="auto", speaker_strategy="parent_dir",
                          probe_durations=False)
    check("csv manifest still auto-detected", len(csv_utts) == 1, f"got {len(csv_utts)}")
    check("csv paths still resolve against the manifest's own dir",
          csv_utts[0].audio_path.exists(), f"missing: {csv_utts[0].audio_path}")
    check("csv speaker_id column still honoured", csv_utts[0].speaker_id == "spk01")

    pairs_root = tmp / "pairscorpus"
    (pairs_root / "spk02").mkdir(parents=True)
    (pairs_root / "spk02" / "utt000.wav").write_bytes(src_wav.read_bytes())
    (pairs_root / "spk02" / "utt000.txt").write_text("hi", encoding="utf-8")
    pair_utts = M.discover(pairs_root, layout="auto", speaker_strategy="parent_dir",
                           probe_durations=False)
    check("pairs layout still works", len(pair_utts) == 1, f"got {len(pair_utts)}")
    check("pairs speaker from parent dir", pair_utts[0].speaker_id == "spk02")

    print(f"\n{'=' * 60}")
    if failures:
        print(f"FAILED {len(failures)}/{checks} checks:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"All {checks} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
