#!/usr/bin/env python3
"""Local strict-sync verification of training validation exports.

Companion to tools/validation_exports.py: while a training box runs,
its HF uploader ships every-100th-game replays (per category) to the
campaign repo under validate_exports/. This tool pulls the new ones
and plays each back in REAL Wesnoth (validate_replay_wesnoth's
minimized, isolated-userdata harness -- no focus stealing), one at a
time, and appends a verdict per replay to a results CSV. Any OOS is
the signal the sim diverged from Wesnoth on a live training
distribution -- surface it, don't average it away.

Usage:
    python tools/run_validation_batch.py               # local dir only
    python tools/run_validation_batch.py --hf-pull     # fetch first

Idempotent: replays already in the results CSV are skipped, so run
it periodically during a campaign (each playback takes ~1-2 min).
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger("run_validation_batch")

DEFAULT_DIR = Path("training/validate_exports")
DEFAULT_TOKEN_FILE = Path.home() / ".hf_token_wesnoth"


def _hf_pull(dest: Path, repo: str, token_file: Path) -> int:
    """Download new validate_exports/* files from the campaign repo
    into `dest`, preserving the category subdirs. Returns #new."""
    from huggingface_hub import HfApi, hf_hub_download
    token = token_file.read_text(encoding="utf-8").strip()
    api = HfApi(token=token)
    n_new = 0
    for rf in api.list_repo_files(repo, repo_type="model"):
        if not rf.startswith("validate_exports/"):
            continue
        rel = rf[len("validate_exports/"):]
        out = dest / rel
        if out.exists():
            continue
        got = hf_hub_download(repo, rf, repo_type="model", token=token)
        out.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(got, out)
        n_new += 1
        log.info(f"pulled {rel}")
    return n_new


def main(argv) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR,
                    help="Exports root (category subdirs of .bz2).")
    ap.add_argument("--hf-pull", action="store_true",
                    help="Pull new exports from the campaign HF repo "
                         "before validating.")
    ap.add_argument("--hf-repo", default="momom2/wesnoth-tier-a")
    ap.add_argument("--hf-token-file", type=Path,
                    default=DEFAULT_TOKEN_FILE)
    ap.add_argument("--results", type=Path, default=None,
                    help="Results CSV (default: <dir>/results.csv).")
    ap.add_argument("--timeout", type=float, default=420.0)
    ap.add_argument("--limit", type=int, default=0,
                    help="Validate at most N new replays this run "
                         "(0 = all).")
    args = ap.parse_args(argv[1:])

    if args.hf_pull:
        n = _hf_pull(args.dir, args.hf_repo, args.hf_token_file)
        log.info(f"hf pull: {n} new file(s)")

    results_path = args.results or (args.dir / "results.csv")
    done = set()
    if results_path.exists():
        with results_path.open(encoding="utf-8", newline="") as f:
            done = {row["replay"] for row in csv.DictReader(f)}

    todo = [p for p in sorted(args.dir.rglob("*.bz2"))
            if p.relative_to(args.dir).as_posix() not in done]
    if args.limit > 0:
        todo = todo[:args.limit]
    if not todo:
        log.info("nothing new to validate")
        return 0
    log.info(f"{len(todo)} replay(s) to validate "
             f"(~{len(todo) * 1.5:.0f} min)")

    from tools.validate_replay_wesnoth import validate_in_wesnoth
    new_row = not results_path.exists()
    n_clean = n_oos = 0
    with results_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new_row:
            w.writerow(["replay", "category", "verdict",
                        "detail", "validated_at"])
        for p in todo:
            rel = p.relative_to(args.dir).as_posix()
            cat = rel.split("/", 1)[0] if "/" in rel else "?"
            log.info(f"=== validating {rel}")
            try:
                offending = validate_in_wesnoth(p, timeout=args.timeout)
            except Exception as e:                    # noqa: BLE001
                offending = [f"HARNESS ERROR: {e}"]
            verdict = "clean" if not offending else "OOS"
            if offending:
                n_oos += 1
                for line in offending[:5]:
                    log.error(f"  {line}")
            else:
                n_clean += 1
            w.writerow([rel, cat, verdict,
                        " | ".join(offending[:3]),
                        time.strftime("%FT%T")])
            f.flush()
    log.info(f"batch done: {n_clean} clean, {n_oos} OOS "
             f"(results: {results_path})")
    return 1 if n_oos else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
