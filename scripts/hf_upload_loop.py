#!/usr/bin/env python3
"""Periodic checkpoint uploader for rented GPU nodes.

Uploads the campaign checkpoint + trainer-history CSV to a (private)
Hugging Face model repo, immediately at start and then every
UPLOAD_EVERY seconds. Each upload is a Hub commit, so the checkpoint
HISTORY is preserved — that's the input for the Elo-vs-compute ladder.

Why this exists (2026-07-03): a stopped/outbid Vast instance's disk is
unreachable until the machine frees up, so anything not exported is
hostage. Pull-based grabbing depends on the laptop being awake;
push-based export from the node does not.

Config (env, or files next to WORKDIR for tokenless templates):
  HF_TOKEN   -- fine-grained write token scoped to the target repo,
                or a token file at $WORKDIR/.hf_token
  HF_REPO    -- target repo id, or a file at $WORKDIR/.hf_repo
                (default: momom2/wesnoth-tier-a)
  WORKDIR    -- defaults to /workspace
Run from the repo root (paths below are repo-relative).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

UPLOAD_EVERY = int(os.environ.get("HF_UPLOAD_EVERY", "1800"))
WORKDIR = Path(os.environ.get("WORKDIR", "/workspace"))

FILES = [
    ("training/checkpoints/tier_a_campaign.pt", "tier_a_campaign.pt"),
    ("training/logs/trainer_history_local.csv", "trainer_history_local.csv"),
]


def _read_opt(env: str, fallback_file: Path) -> str:
    v = os.environ.get(env, "").strip()
    if not v and fallback_file.exists():
        v = fallback_file.read_text(encoding="utf-8").strip()
    return v


def main() -> int:
    token = _read_opt("HF_TOKEN", WORKDIR / ".hf_token")
    repo = _read_opt("HF_REPO", WORKDIR / ".hf_repo") \
        or "momom2/wesnoth-tier-a"
    if not token:
        print("hf_upload_loop: no HF_TOKEN / .hf_token; exiting.",
              flush=True)
        return 1
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo, private=True, exist_ok=True)
    print(f"hf_upload_loop: uploading to {repo} every "
          f"{UPLOAD_EVERY}s", flush=True)
    last_sig = None
    while True:
        try:
            # Skip the upload when nothing changed (cheap mtime+size
            # signature) -- keeps the Hub history meaningful.
            sig = tuple(
                (p, os.path.getmtime(p), os.path.getsize(p))
                for p, _ in FILES if os.path.exists(p)
            )
            if sig and sig != last_sig:
                for src, dst in FILES:
                    if os.path.exists(src):
                        api.upload_file(
                            path_or_fileobj=src, path_in_repo=dst,
                            repo_id=repo, repo_type="model")
                last_sig = sig
                print(f"hf_upload_loop: uploaded at "
                      f"{time.strftime('%FT%TZ', time.gmtime())}",
                      flush=True)
        except Exception as e:                      # noqa: BLE001
            # Transient network/Hub errors must not kill the loop --
            # the next cycle retries.
            print(f"hf_upload_loop: upload failed: {e}", flush=True)
        time.sleep(UPLOAD_EVERY)


if __name__ == "__main__":
    sys.exit(main())
