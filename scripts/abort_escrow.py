#!/usr/bin/env python3
"""Escrow a final sweep to HF after a terminal training failure.

User ruling 2026-09-02: on any terminal path of the self-play launch flows
(a tripwire ABORTED_* marker, the relaunch cap, a smoke or test failure)
the campaign checkpoint, its .holdout, the trainer CSV and a tar of the
pins, probes and logs go to HF before the box stops: a stopped box's disk
survives, but resuming needs its host's GPU to be free, so nothing may be
hostage to that. The launch scripts then stop the box with
scripts/box/box_stop.py, which uses the key Vast gives each box.

Inputs: $WORKDIR/.hf_token (no escrow without it). Env: WORKDIR (default
/workspace), REPO_ROOT, CAMPAIGN_FILE, HF_PREFIX, HF_REPO. `--dry-run`
prints the uploads and touches nothing remote.
"""
from __future__ import annotations

import os
import sys
import tarfile
import time
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", "/workspace"))
REPO_ROOT = Path(os.environ.get("REPO_ROOT", str(WORKDIR / "wai")))
CAMPAIGN_FILE = os.environ.get("CAMPAIGN_FILE", "tier_a_campaign.pt")
HF_REPO = os.environ.get("HF_REPO", "momom2/wesnoth-model-checkpoints")
HF_PREFIX = os.environ.get("HF_PREFIX", "tier-b/")


def log(msg: str) -> None:
    line = f"{time.strftime('%FT%TZ', time.gmtime())} abort_escrow: {msg}"
    print(line, flush=True)
    try:
        with (WORKDIR / "abort_escrow.log").open("a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip() or None
    except OSError:
        return None


def final_escrow(dry_run: bool) -> None:
    token = _read(WORKDIR / ".hf_token")
    if not token:
        log("no .hf_token -- skipping final escrow")
        return
    ckpt_dir = REPO_ROOT / "training" / "checkpoints"
    files = [
        (ckpt_dir / CAMPAIGN_FILE, f"{CAMPAIGN_FILE}"),
        (ckpt_dir / f"{CAMPAIGN_FILE}.holdout", f"{CAMPAIGN_FILE}.holdout"),
        (REPO_ROOT / "training" / "logs" / "trainer_history_local.csv",
         "trainer_history_local.csv"),
    ]
    tar_path = WORKDIR / "abort_escrow.tar.gz"
    with tarfile.open(tar_path, "w:gz") as t:
        for name in ("pins.log", "probes", "profiles", "train.log",
                     "onstart.log", "upload.log", "watchdog.log",
                     # gauntlet logs: a tests/smoke/rust failure must
                     # be diagnosable from the escrow alone (VG3:
                     # ABORTED_tests with no pytest log = box restart)
                     "pytest_full.log", "smoke.log", "rust_build.log",
                     "armVG_driver.log", "anchor_build.log"):
            p = WORKDIR / name
            if p.exists():
                t.add(p, arcname=name)
        for p in WORKDIR.glob("ABORTED_*"):
            t.add(p, arcname=p.name)
    files.append((tar_path, "abort_escrow.tar.gz"))
    if dry_run:
        for src, dst in files:
            log(f"DRY-RUN would upload {src} -> {HF_PREFIX}{dst} "
                f"(exists={src.exists()})")
        return
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    for src, dst in files:
        if not src.exists():
            continue
        try:
            api.upload_file(path_or_fileobj=str(src),
                            path_in_repo=HF_PREFIX + dst, repo_id=HF_REPO)
            log(f"escrowed {dst}")
        except Exception as e:  # noqa: BLE001 -- keep going; stopping
            # the box after this matters more than any one file.
            # The type only: an HTTP error's text can quote a URL.
            log(f"escrow FAILED for {dst}: {type(e).__name__}")


def main(argv) -> int:
    dry_run = "--dry-run" in argv
    log(f"terminal failure escrow (dry_run={dry_run}, campaign={CAMPAIGN_FILE})")
    final_escrow(dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
