#!/usr/bin/env python3
"""Run the whole pipeline tiny, to catch errors before a real training run.

Generates a synthetic fixture corpus, prepares splits, trains 2 steps, and
evaluates 2 utterances. Passing means the plumbing works -- it says nothing
about model quality. Run this before every full training run:

    python scripts/smoke_test.py
    python scripts/smoke_test.py --set model.name=openai/whisper-small
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kokborok_asr.config import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("smoke_test")

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run_step(name: str, argv: list[str]) -> tuple[bool, float, str]:
    logger.info("=== %s ===", name)
    logger.info("$ %s", " ".join(argv))
    started = time.time()
    proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.time() - started
    if proc.returncode != 0:
        # Show the tail of both streams: tracebacks land on stderr, but the
        # scripts log their diagnostics through logging (also stderr).
        logger.error("%s FAILED in %.1fs (exit %d)", name, elapsed, proc.returncode)
        for stream_name, stream in (("stdout", proc.stdout), ("stderr", proc.stderr)):
            tail = (stream or "").strip().splitlines()[-25:]
            if tail:
                logger.error("--- %s tail ---\n%s", stream_name, "\n".join(tail))
        return False, elapsed, proc.stderr or proc.stdout
    logger.info("%s ok (%.1fs)", name, elapsed)
    return True, elapsed, proc.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--keep", action="store_true", help="Keep the fixture artifacts afterwards")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    fixture_raw = cfg.resolve_path("paths.raw_dir", "data/fixtures/raw")
    processed = cfg.resolve_path("paths.processed_dir", "data/fixtures/processed")
    checkpoints = cfg.resolve_path("train.output_dir", "data/fixtures/checkpoints/smoke")

    real_raw = (ROOT / "data" / "raw").resolve()
    if fixture_raw.resolve() == real_raw:
        logger.error("Refusing to run: smoke config points at the real data/raw.")
        return 1

    passthrough = [item for pair in (("--set", o) for o in args.overrides) for item in pair]
    python = sys.executable

    steps = [
        ("fixture", [python, str(SCRIPTS / "make_fixture.py"), "--out", str(fixture_raw), "--force"]),
        ("prepare_data", [python, str(SCRIPTS / "prepare_data.py"), "--config", args.config, *passthrough]),
        ("train", [python, str(SCRIPTS / "train.py"), "--config", args.config, *passthrough]),
        ("evaluate", [python, str(SCRIPTS / "evaluate.py"), "--config", args.config, *passthrough]),
    ]

    results = []
    for name, argv in steps:
        ok, elapsed, _ = run_step(name, argv)
        results.append((name, ok, elapsed))
        if not ok:
            logger.error("SMOKE TEST FAILED at '%s'. Do not launch a full run yet.", name)
            return 1

    total = sum(e for _, _, e in results)
    logger.info("=" * 56)
    for name, ok, elapsed in results:
        logger.info("  %-14s %s  %5.1fs", name, "PASS" if ok else "FAIL", elapsed)
    logger.info("SMOKE TEST PASSED in %.1fs - pipeline is wired correctly.", total)
    logger.info("Reminder: fixture audio is synthetic tones; its WER is meaningless.")

    if not args.keep:
        for path in (processed, checkpoints):
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
        logger.info("Cleaned fixture artifacts (use --keep to retain them)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
