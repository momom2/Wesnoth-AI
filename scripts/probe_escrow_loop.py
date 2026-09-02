#!/usr/bin/env python3
"""Escrow the non-checkpoint leg artifacts every 30 min: pins.log,
probe games, per-pin signal profiles, and the run logs. The
checkpoint + CSV go through hf_upload_loop; this loop covers what
that one does not, so a dead box loses at most half an hour of
probe/profile evidence.

Env: WORKDIR (default /workspace), HF_PREFIX, HF_REPO,
ESCROW_EVERY (seconds, default 1800). Token at $WORKDIR/.hf_token.
"""
from __future__ import annotations

import os
import tarfile
import time
import traceback
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", "/workspace"))
HF_REPO = os.environ.get("HF_REPO", "momom2/wesnoth-model-checkpoints")
HF_PREFIX = os.environ.get("HF_PREFIX", "tier-b/")
EVERY = int(os.environ.get("ESCROW_EVERY", "1800"))
ITEMS = ("pins.log", "probes", "profiles", "train.log", "pinloop.log",
         "armVG_driver.log", "watchdog.log", "upload.log")


def main() -> int:
    from huggingface_hub import HfApi
    api = HfApi(token=(WORKDIR / ".hf_token").read_text().strip())
    tar = WORKDIR / "probe_escrow.tar.gz"
    while True:
        try:
            with tarfile.open(tar, "w:gz") as t:
                for name in ITEMS:
                    p = WORKDIR / name
                    if p.exists():
                        t.add(p, arcname=name)
            api.upload_file(path_or_fileobj=str(tar),
                            path_in_repo=HF_PREFIX + "probe_escrow.tar.gz",
                            repo_id=HF_REPO)
            print(time.strftime("%FT%TZ", time.gmtime()),
                  "probe escrow uploaded", flush=True)
        except Exception:  # noqa: BLE001 -- keep the loop alive
            traceback.print_exc()
        time.sleep(EVERY)


if __name__ == "__main__":
    raise SystemExit(main())
