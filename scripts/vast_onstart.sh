#!/usr/bin/env bash
# Vast.ai on-start script for the Tier-a Phase 2 calibration run
# (docs/tier_a_runbook.md). Paste into the template's "On-start
# Script" box. It runs at EVERY instance (re)start, so it encodes the
# first-launch vs preemption-resume distinction that must not be
# fumbled: --reset-decision-step ONLY on the very first launch.
#
# Layout: /workspace persists across interruptible stop/restart
# cycles on Vast (it is the instance's disk; it is lost only if the
# instance is DESTROYED -- download checkpoints before destroying).
set -uo pipefail
# Vast images differ in data dir + where python lives (observed
# 2026-07-02 on vastai/pytorch:cuda-13.0.3-auto: no /workspace at
# onstart time, and python sits in /venv/main which only interactive
# shells activate). Be robust to both.
WORKDIR="${DATA_DIRECTORY:-/workspace}"
export WORKDIR   # read by hf_upload_loop.py and the HF seed block
mkdir -p "$WORKDIR"
exec >> "$WORKDIR/onstart.log" 2>&1
echo "==== onstart $(date -u +%FT%TZ) ===="

cd "$WORKDIR"

# Resolve python: prefer the image's venv, then conda, then PATH.
if [ -x /venv/main/bin/python ]; then
    export PATH="/venv/main/bin:$PATH"
elif [ -x /opt/conda/bin/python ]; then
    export PATH="/opt/conda/bin:$PATH"
fi
PY="$(command -v python || command -v python3 || true)"
if [ -z "$PY" ]; then
    echo "[onstart] FATAL: no python found on PATH/venv/conda"; exit 1
fi
echo "[onstart] using python: $PY"

# Hard requirements: CUDA torch + Python >= 3.11 (project floor).
"$PY" - <<'EOF' || { echo "[onstart] FATAL: env check failed"; exit 1; }
import sys, torch
assert sys.version_info >= (3, 11), f"need Python >=3.11, got {sys.version}"
assert torch.cuda.is_available(), "no CUDA device visible"
print(f"[onstart] python {sys.version.split()[0]}, torch {torch.__version__}, "
      f"gpu {torch.cuda.get_device_name(0)}, "
      f"vcpus reported: {__import__('os').cpu_count()}")
EOF

if [ ! -d Wesnoth-AI ]; then
    git clone --depth 1 https://github.com/momom2/Wesnoth-AI.git || exit 1
fi
cd Wesnoth-AI
# Pick up fixes pushed since the instance was created. --ff-only so a
# locally-dirtied tree (shouldn't happen; checkpoints write to an
# untracked path) fails loudly instead of merging silently.
git pull --ff-only || echo "[onstart] WARN: git pull failed; running existing checkout"

# A tripwire abort (exit 4 = all-draws, 5 = holdout stall) needs a
# human decision -- do NOT auto-relaunch over it.
if ls "$WORKDIR"/ABORTED_* >/dev/null 2>&1; then
    echo "[onstart] ABORTED_* marker present -- NOT relaunching."
    echo "[onstart] Read the tail of $WORKDIR/train.log, diagnose,"
    echo "[onstart] delete the marker, then restart the instance."
    exit 0
fi

# First launch warm-starts from the committed 5M grow WITH the anneal
# reset; any restart after that resumes the campaign checkpoint
# WITHOUT it. The campaign file doubles as the marker (it exists iff
# training has saved at least once; the save is atomic + .bak'd).
CAMPAIGN=training/checkpoints/tier_a_campaign.pt

# Seed the campaign from HF Hub on a FRESH instance so a brand-new
# node RESUMES the campaign instead of silently starting over
# (2026-07-05 incident: token scp'd after onstart had already begun a
# fresh --reset-decision-step run). Requires HF_TOKEN in the template
# env (pass at create time: vastai create instance ... --env '-e
# HF_TOKEN=hf_...') or $WORKDIR/.hf_token pre-seeded some other way.
# HF_SEED_FILE selects WHICH repo file seeds the campaign (default:
# the rolling campaign checkpoint). Pass -e HF_SEED_FILE=... at
# create time to start a run from a different escrowed checkpoint,
# e.g. human_value_allgames.pt (the 2026-07-09 human-corpus value
# fine-tune: late-game AUC 0.89 vs the old head's ~0.50). It lands
# AS the local campaign file, so the resume path (no anneal reset,
# decision_step carried) applies and the uploader's tier_a_campaign
# escrow rolls forward from it.
if [ ! -f "$CAMPAIGN" ]; then
    if [ -n "${HF_TOKEN:-}" ] || [ -f "$WORKDIR/.hf_token" ]; then
        "$PY" -m pip install --quiet huggingface_hub || true
        HF_SEED_TOKEN="${HF_TOKEN:-}" \
        HF_SEED_FILE="${HF_SEED_FILE:-tier_a_campaign.pt}" \
        "$PY" - <<'EOF' && echo "[onstart] seeded campaign from HF" \
            || echo "[onstart] HF seed unavailable (first campaign?)"
import os, pathlib, shutil, sys
from huggingface_hub import hf_hub_download
tok = os.environ.get("HF_SEED_TOKEN") or pathlib.Path(
    os.environ.get("WORKDIR", "/workspace"), ".hf_token"
).read_text().strip()
fname = os.environ.get("HF_SEED_FILE", "tier_a_campaign.pt")
try:
    p = hf_hub_download("momom2/wesnoth-tier-a", fname, token=tok)
except Exception as e:                                  # noqa: BLE001
    print(f"[onstart] hf seed download failed: {e}")
    sys.exit(1)
dst = pathlib.Path("training/checkpoints/tier_a_campaign.pt")
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(p, dst)
print(f"[onstart] seed file: {fname}")
EOF
    fi
fi
if [ -f "$CAMPAIGN" ]; then
    CKPT_IN="$CAMPAIGN"; RESET=""
    echo "[onstart] RESUME from $CAMPAIGN (no --reset-decision-step)"
else
    CKPT_IN=training/checkpoints/tier_a_5m.pt
    RESET="--reset-decision-step"
    echo "[onstart] FIRST LAUNCH from $CKPT_IN (+anneal reset)"
fi

# ---- Periodic checkpoint export (Hugging Face Hub) ------------------
# Opt-in: put a fine-grained write token (scoped to ONE model repo) in
# $WORKDIR/.hf_token, or set HF_TOKEN in the template env. Uploads the
# campaign checkpoint + CSV immediately and every 30 min -- a stopped
# instance's disk is unreachable (learned 2026-07-03), so anything not
# pushed off the node is hostage to the next outbid.
pkill -f 'hf_upload_loo[p].py' 2>/dev/null || true
if [ -n "${HF_TOKEN:-}" ] || [ -f "$WORKDIR/.hf_token" ]; then
    "$PY" -m pip install --quiet huggingface_hub || true
    WORKDIR="$WORKDIR" nohup "$PY" scripts/hf_upload_loop.py \
        >> "$WORKDIR/hf_upload.log" 2>&1 &
    echo "[onstart] HF checkpoint uploader ON (see hf_upload.log)"
else
    echo "[onstart] HF uploader off (no HF_TOKEN / $WORKDIR/.hf_token)"
fi

# Rotate a bloated train.log (the 2026-07-03 fd-leak spammed 134MB of
# tracebacks; keep restarts snappy and greps fast).
if [ -f "$WORKDIR/train.log" ] && \
        [ "$(stat -c%s "$WORKDIR/train.log")" -gt 50000000 ]; then
    mv "$WORKDIR/train.log" "$WORKDIR/train.log.1"
    echo "[onstart] rotated oversized train.log -> train.log.1"
fi

# Both spellings: torch <=2.7 reads PYTORCH_CUDA_ALLOC_CONF, newer
# reads PYTORCH_ALLOC_CONF.
# SPOOL_WORKERS / TRAIN_BATCH: GPU-memory knobs, overridable via
# -e at create time for smaller cards (16GB: 12 / 48; the 24GB
# defaults 16 / 64 measured ~17GB with creep).
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# GPU memory budget (24GB card, learned from the 2026-07-06 OOM):
# each spool worker pins a ~560MB CUDA context + model, so
# 24 workers (13.7GB) + trainer peak at batch 128 (~10GB) OOM'd
# mid-train-step (and allocator thrash made the step take 22 min).
# 16 workers (~9GB) + batch 64 (~5GB) leaves real headroom.
# 48 actor processes x (ctrl/resp queues + shipped experience pipes)
# exceed the container's default 1024-fd soft limit (observed
# 2026-07-02: OSError errno 24 in multiprocessing resource_sharer).
ulimit -n 65536 2>/dev/null || ulimit -n 4096 2>/dev/null || true
echo "[onstart] fd limit: $(ulimit -n)"
nohup bash -c "
  '$PY' tools/sim_self_play.py --device cuda \
    --mcts --mcts-sims 32 \
    --d-model 256 --num-layers 6 --num-heads 8 --d-ff 1024 \
    --replay-buffer --replay-updates 16 --value-coef 1.0 \
    --replay-minibatch 128 --replay-capacity 24000 \
    --train-batch-size ${TRAIN_BATCH:-64} --mcts-batch-size 16 \
    --mini-ratio ${MINI_RATIO:-0.5} --drill-ratio ${DRILL_RATIO:-0.3} \
    --mcts-aux-score --mcts-moves-left \
    --mcts-moves-left-utility 0.2 \
    ${AUX_VALUE_BONUS:+--mcts-aux-value-bonus $AUX_VALUE_BONUS} \
    ${FOGLESS_RATIO:+--fogless-ratio $FOGLESS_RATIO} \
    --value-label-smoothing 0.02 \
    --holdout-size 512 --holdout-per-game-cap 64 \
    ${HUMAN_ANCHOR_FILE:+--human-anchor-file $HUMAN_ANCHOR_FILE} \
    ${HUMAN_ANCHOR_UPDATES:+--human-anchor-updates $HUMAN_ANCHOR_UPDATES} \
    ${HUMAN_ANCHOR_BATCH:+--human-anchor-batch $HUMAN_ANCHOR_BATCH} \
    ${DRAW_VALUE_WEIGHT:+--draw-value-weight $DRAW_VALUE_WEIGHT} \
    --abort-decisive-rate 0.05 --abort-window 40 \
    --abort-holdout-stall 150 \
    --spool-workers ${SPOOL_WORKERS:-16} --games-per-iter ${SPOOL_WORKERS:-16} \
    $RESET \
    --checkpoint-in  $CKPT_IN \
    --checkpoint-out $CAMPAIGN \
    --iterations 100000 --save-every 2 --log-level INFO \
    >> '$WORKDIR/train.log' 2>&1
  rc=\$?
  echo \"[onstart] training exited rc=\$rc at \$(date -u +%FT%TZ)\" >> '$WORKDIR/train.log'
  if [ \$rc -ge 3 ]; then touch '$WORKDIR/ABORTED_'\$rc; fi
" >/dev/null 2>&1 &
echo "[onstart] training launched (tail -f $WORKDIR/train.log)"
