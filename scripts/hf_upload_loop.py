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

HARD TIMEOUTS (2026-08-15): on the tcs2 leg this loop wedged SILENTLY
for ~15 hours — the process stayed alive but hung inside a large-
checkpoint `upload_file` call (no timeout anywhere in that stack), so
every 30-min escrow cycle after it was skipped while training kept
saving locally. The stall watchdog only monitors the TRAINING
process, so nothing noticed. Every upload now runs in a SUBPROCESS
with a hard kill (`subprocess.run(timeout=...)`) — 15 min for the
large checkpoint, 5 min for small files — logs LOUDLY on timeout,
and the loop continues; the failed file retries next cycle because
the change-signature only advances on a fully successful sweep. A
heartbeat line prints EVERY cycle (even "nothing to upload") so
staleness is greppable:  grep "hf_upload_loop: cycle" upload.log.

Config (env, or files next to WORKDIR for tokenless templates):
  HF_TOKEN   -- fine-grained write token scoped to the target repo,
                or a token file at $WORKDIR/.hf_token
  HF_REPO    -- target repo id, or a file at $WORKDIR/.hf_repo
                (default: momom2/wesnoth-model-checkpoints)
  HF_PREFIX  -- folder prefix inside the repo for every upload
                (default: tier-b/ -- the repo sorts lineages into
                tier-a/ and tier-b/ folders since 2026-08-06)
  WORKDIR    -- defaults to /workspace
  HF_UPLOAD_TIMEOUT_LARGE / _SMALL -- seconds (default 900 / 300);
                "large" applies over HF_UPLOAD_LARGE_BYTES (50 MB).
Run from the repo root (paths below are repo-relative).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict

UPLOAD_EVERY = int(os.environ.get("HF_UPLOAD_EVERY", "1800"))
WORKDIR = Path(os.environ.get("WORKDIR", "/workspace"))

TIMEOUT_LARGE = int(os.environ.get("HF_UPLOAD_TIMEOUT_LARGE", "900"))
TIMEOUT_SMALL = int(os.environ.get("HF_UPLOAD_TIMEOUT_SMALL", "300"))
LARGE_BYTES = int(os.environ.get("HF_UPLOAD_LARGE_BYTES",
                                 str(50 * 1024 * 1024)))

# The campaign's identity, shared with vast_onstart.sh's CAMPAIGN_FILE.
# It names both the local rolling checkpoint and the HF escrow object,
# so a run on a different lineage MUST override it -- otherwise this
# loop uploads that run's weights over the reserved `tier_a_campaign.pt`
# escrow. Kept as one variable so the two can never drift apart.
CAMPAIGN_FILE = os.environ.get("CAMPAIGN_FILE", "tier_a_campaign.pt")

# Repo folder for this lineage. All training is Tier-b since
# 2026-08-05 (user decision), so tier-b/ is the default; a tier-a
# revival run must override. Applied to EVERY path_in_repo below.
HF_PREFIX = os.environ.get("HF_PREFIX", "tier-b/")

FILES = [
    (f"training/checkpoints/{CAMPAIGN_FILE}", CAMPAIGN_FILE),
    ("training/logs/trainer_history_local.csv", "trainer_history_local.csv"),
    # Persisted holdout probe (2026-07-18): without it, a destroyed/
    # reseeded box resamples the probe and the holdout-CE curve loses
    # cross-box comparability (the balance-exhaustion stop of
    # 2026-07-19 nearly stranded it on an unreachable disk).
    (f"training/checkpoints/{CAMPAIGN_FILE}.holdout",
     f"{CAMPAIGN_FILE}.holdout"),
    # Human-holdout CE probe curve (scripts/holdout_probe_loop.py) --
    # the handoff observable; losing it to an outbid would blind the
    # A1/F1 verdict.
    ("training/logs/holdout_probe.csv", "holdout_probe.csv"),
]
# HF_EXTRA_FILES="src:dst,src:dst" adds run-specific artifacts (e.g.
# the supervised pass: supervised.pt + its eval curve, 2026-07-16).
for _pair in os.environ.get("HF_EXTRA_FILES", "").split(","):
    if ":" in _pair:
        _src, _dst = _pair.split(":", 1)
        if _src.strip() and _dst.strip():
            FILES.append((_src.strip(), _dst.strip()))


def _read_opt(env: str, fallback_file: Path) -> str:
    v = os.environ.get(env, "").strip()
    if not v and fallback_file.exists():
        v = fallback_file.read_text(encoding="utf-8").strip()
    return v


# ---------------------------------------------------------------------
# Timeout-bounded upload (the 2026-08-15 hardening)
# ---------------------------------------------------------------------

# The subprocess body. Module-level so the regression test can swap
# in a hanging stub and exercise the REAL kill path.
_CHILD_CODE = (
    "import os, sys\n"
    "from huggingface_hub import HfApi\n"
    "HfApi(token=os.environ['HF_UPLOAD_TOKEN']).upload_file(\n"
    "    path_or_fileobj=sys.argv[1], path_in_repo=sys.argv[2],\n"
    "    repo_id=sys.argv[3], repo_type='model')\n"
)


def upload_with_timeout(src: str, path_in_repo: str, repo: str,
                        token: str, timeout_s: int) -> bool:
    """One upload in a killable subprocess. True on success; False on
    timeout (child SIGKILLed) or nonzero exit -- both logged loudly,
    neither raises: the caller's cycle continues and the change-
    signature logic retries the file next cycle. The token travels
    via the child's env, never argv."""
    env = {**os.environ, "HF_UPLOAD_TOKEN": token}
    try:
        r = subprocess.run(
            [sys.executable, "-c", _CHILD_CODE, str(src),
             path_in_repo, repo],
            env=env, timeout=timeout_s, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        print(f"hf_upload_loop: UPLOAD TIMED OUT after {timeout_s}s: "
              f"{src} -> {path_in_repo}; child killed, file retries "
              f"next cycle. ESCROW IS DEGRADED until a cycle "
              f"succeeds.", flush=True)
        return False
    if r.returncode != 0:
        print(f"hf_upload_loop: upload FAILED rc={r.returncode}: "
              f"{src} -> {path_in_repo}: "
              f"{(r.stderr or '')[-300:]}", flush=True)
        return False
    return True


def _timeout_for(src: str) -> int:
    try:
        return (TIMEOUT_LARGE if os.path.getsize(src) > LARGE_BYTES
                else TIMEOUT_SMALL)
    except OSError:
        return TIMEOUT_SMALL


# ---------------------------------------------------------------------
# One escrow cycle (extracted for testability -- the regression test
# drives this exact function with a stub uploader)
# ---------------------------------------------------------------------

def run_cycle(uploader: Callable[[str, str], bool],
              state: Dict) -> None:
    """One sweep: campaign files (signature-gated, all-or-retry),
    validation exports (each once), games-log tarball. `uploader(src,
    dst_in_repo) -> bool`; False means the file didn't land and the
    relevant signature must NOT advance. Ends with the heartbeat
    line -- every cycle prints exactly one `cycle ...` line, so
    `grep 'hf_upload_loop: cycle'` measures staleness."""
    did = []
    try:
        # Campaign file set: skip when nothing changed (cheap
        # mtime+size signature) -- keeps the Hub history meaningful.
        sig = tuple(
            (p, os.path.getmtime(p), os.path.getsize(p))
            for p, _ in FILES if os.path.exists(p)
        )
        if sig and sig != state.get("last_sig"):
            ok = True
            for src, dst in FILES:
                if os.path.exists(src):
                    ok = uploader(src, HF_PREFIX + dst) and ok
            if ok:
                state["last_sig"] = sig
                did.append("campaign set")
            else:
                did.append("campaign set INCOMPLETE (retries)")
        # Validation replay exports: every-Nth-per-category
        # strict-sync replays; each NEW file once (unique names).
        vdir = Path("training/validate_exports")
        if vdir.is_dir():
            seen = state.setdefault("uploaded_validation", set())
            sweep = sorted((vdir / "bundles").glob("*.tar")) + \
                sorted(f for f in vdir.rglob("*.bz2")
                       if "bundles" not in f.parts)
            for f in sweep:
                rel = f.relative_to(vdir).as_posix()
                if rel in seen:
                    continue
                try:
                    if uploader(str(f),
                                f"{HF_PREFIX}validate_exports/{rel}"):
                        seen.add(rel)
                        did.append(f"export {rel}")
                except FileNotFoundError:
                    continue      # bundled away mid-sweep
        # Per-game logs: ONE rolling tarball, signature over
        # (count, total size).
        gdir = Path(os.environ.get("GAME_LOG_DIR",
                                   "training/logs/games"))
        if gdir.is_dir():
            gfiles = sorted(gdir.rglob("games.jsonl"))
            gsig = (len(gfiles),
                    sum(os.path.getsize(f) for f in gfiles))
            if gfiles and gsig != state.get("last_games_sig"):
                import tarfile
                tar_path = gdir.parent / "games_log.tar.gz"
                with tarfile.open(tar_path, "w:gz") as tf:
                    for f in gfiles:
                        tf.add(str(f),
                               arcname=f.relative_to(gdir).as_posix())
                if uploader(str(tar_path),
                            HF_PREFIX + "games_log.tar.gz"):
                    state["last_games_sig"] = gsig
                    did.append(f"games_log ({gsig[0]} files)")
    except Exception as e:                          # noqa: BLE001
        # Transient FS/Hub errors must not kill the loop -- the next
        # cycle retries.
        did.append(f"ERROR {e!r}")
    # Heartbeat: exactly one line per cycle, always.
    stamp = time.strftime("%FT%TZ", time.gmtime())
    print(f"hf_upload_loop: cycle at {stamp}: "
          f"{'; '.join(did) if did else 'OK, nothing to upload'}",
          flush=True)


def main() -> int:
    token = _read_opt("HF_TOKEN", WORKDIR / ".hf_token")
    repo = _read_opt("HF_REPO", WORKDIR / ".hf_repo") \
        or "momom2/wesnoth-model-checkpoints"
    if not token:
        print("hf_upload_loop: no HF_TOKEN / .hf_token; exiting.",
              flush=True)
        return 1
    from huggingface_hub import HfApi
    try:
        HfApi(token=token).create_repo(repo, private=True,
                                       exist_ok=True)
    except Exception as e:                          # noqa: BLE001
        # A repo-scoped fine-grained token (the recommended setup)
        # can't create repos at all -- the repo is pre-created in the
        # UI and this call just gets a 403. Uploads still work.
        print(f"hf_upload_loop: create_repo skipped ({e})", flush=True)
    print(f"hf_upload_loop: uploading to {repo} every "
          f"{UPLOAD_EVERY}s (timeouts {TIMEOUT_LARGE}s/"
          f"{TIMEOUT_SMALL}s large/small)", flush=True)

    def _uploader(src: str, dst_in_repo: str) -> bool:
        return upload_with_timeout(src, dst_in_repo, repo, token,
                                   _timeout_for(src))

    state: Dict = {}
    while True:
        run_cycle(_uploader, state)
        time.sleep(UPLOAD_EVERY)


if __name__ == "__main__":
    sys.exit(main())
