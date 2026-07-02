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
if [ -f "$CAMPAIGN" ]; then
    CKPT_IN="$CAMPAIGN"; RESET=""
    echo "[onstart] RESUME from $CAMPAIGN (no --reset-decision-step)"
else
    CKPT_IN=training/checkpoints/tier_a_5m.pt
    RESET="--reset-decision-step"
    echo "[onstart] FIRST LAUNCH from $CKPT_IN (+anneal reset)"
fi

export PYTORCH_ALLOC_CONF=expandable_segments:True
nohup bash -c "
  '$PY' tools/sim_self_play.py --device cuda \
    --mcts --mcts-sims 32 \
    --d-model 256 --num-layers 6 --num-heads 8 --d-ff 1024 \
    --replay-buffer --replay-updates 16 --value-coef 1.0 \
    --replay-minibatch 128 --replay-capacity 6000 \
    --train-batch-size 128 --mcts-batch-size 16 \
    --mini-ratio 0.5 --drill-ratio 0.3 \
    --holdout-size 512 \
    --abort-decisive-rate 0.05 --abort-window 40 \
    --abort-holdout-stall 150 \
    --actor-pool 48 --games-per-iter 48 \
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
