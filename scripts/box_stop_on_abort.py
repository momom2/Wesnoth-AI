#!/usr/bin/env python3
"""Stop (not destroy) the Vast box after a terminal training failure.

User ruling 2026-09-02: a tripwire that saves the model but not the
credit is half a tripwire -- arm VG idled ~8h (~$3.6) after its
K-collapse abort. On any terminal path (tripwire ABORTED_* marker,
relaunch cap exhausted, smoke/test failure) the supervisor now:

  1. escrows a final sweep to HF (campaign checkpoint + .holdout +
     trainer CSV + a tar of pins/probes/logs) -- a stopped box's
     disk survives, but resuming needs the host's GPU to be free,
     so nothing may be hostage to that;
  2. puts the instance in the STOPPED state via the Vast REST API
     (what `vastai stop instance` does). Storage billing continues
     (~$0.005/h at 25 GB); GPU billing stops.

Inputs (files, written at provision time by the launch flow):
  $WORKDIR/.vast_api_key   the account API key
  $WORKDIR/.instance_id    this box's contract id (fallback: env
                           CONTAINER_ID)
  $WORKDIR/.hf_token       for the final escrow (optional)
Env: WORKDIR (default /workspace), CAMPAIGN_FILE, HF_PREFIX, HF_REPO.
`--dry-run` prints the actions and touches nothing remote.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", "/workspace"))
REPO_ROOT = Path(os.environ.get("REPO_ROOT", str(WORKDIR / "wai")))
CAMPAIGN_FILE = os.environ.get("CAMPAIGN_FILE", "tier_a_campaign.pt")
HF_REPO = os.environ.get("HF_REPO", "momom2/wesnoth-model-checkpoints")
HF_PREFIX = os.environ.get("HF_PREFIX", "tier-b/")
VAST_API = "https://console.vast.ai/api/v0"


def log(msg: str) -> None:
    line = f"{time.strftime('%FT%TZ', time.gmtime())} stop_on_abort: {msg}"
    print(line, flush=True)
    try:
        with (WORKDIR / "stop_on_abort.log").open("a") as f:
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
            # the box matters more than any one file.
            log(f"escrow FAILED for {dst}: {e!r}")


def stop_instance(dry_run: bool) -> int:
    key = _read(WORKDIR / ".vast_api_key")
    iid = _read(WORKDIR / ".instance_id") or os.environ.get("CONTAINER_ID")
    if not key or not iid:
        log(f"cannot stop: api_key={'ok' if key else 'MISSING'} "
            f"instance_id={iid or 'MISSING'} -- box keeps billing")
        return 2
    if dry_run:
        log(f"DRY-RUN would PUT {VAST_API}/instances/{iid}/ state=stopped")
        return 0
    try:
        import requests
        r = requests.put(f"{VAST_API}/instances/{iid}/",
                         params={"api_key": key},
                         json={"state": "stopped"}, timeout=60)
        log(f"stop request: HTTP {r.status_code} {r.text[:200]}")
        if r.ok:
            return 0
    except Exception as e:  # noqa: BLE001
        log(f"REST stop failed: {e!r}; trying the CLI")
    rc = subprocess.run(
        [sys.executable, "-m", "vastai", "--api-key", key,
         "stop", "instance", str(iid)],
        capture_output=True, text=True).returncode
    log(f"CLI stop rc={rc}")
    return rc


def main(argv) -> int:
    dry_run = "--dry-run" in argv
    log(f"terminal failure handler start (dry_run={dry_run}, "
        f"campaign={CAMPAIGN_FILE})")
    final_escrow(dry_run)
    return stop_instance(dry_run)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
