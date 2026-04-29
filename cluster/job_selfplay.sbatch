#!/usr/bin/env bash
#SBATCH --job-name=wai-selfplay
#SBATCH --partition=ENSTA-l40s
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:55:00
#SBATCH --output=training/logs/selfplay-slurm-%j.log
#SBATCH --error=training/logs/selfplay-slurm-%j.log
#
# Self-play training of the Wesnoth model on one GPU.
# Submit with:  sbatch cluster/job_selfplay.sbatch
# Stop chain:   touch training/checkpoints/selfplay_done.flag
# Status:       squeue --me
# Cancel:       scancel <jobid>

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "[job] node:       $(hostname)"
echo "[job] job_id:     ${SLURM_JOB_ID}"
echo "[job] start:      $(date -Is)"
echo "[job] gpu(s):"
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>&1 | sed 's/^/[job]   /'

# shellcheck disable=SC1091
source .venv/bin/activate

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available inside the job — check driver / wheel"
print(f"[job] torch:    {torch.__version__}")
print(f"[job] device:   {torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
PY

mkdir -p training/checkpoints training/logs

# Latest-checkpoint pick:
#   1. existing self-play checkpoint (resume the chain)
#   2. else final supervised epoch checkpoint (warmstart from
#      supervised) -- glob for the highest epoch number so we don't
#      have to know EPOCHS from job.sbatch
#   3. else generic supervised.pt (mid-supervised checkpoint)
#   4. else fall through to random init (a warning, not an error,
#      so a fresh tree can still kick off if the user wants)
SP_CKPT=training/checkpoints/sim_selfplay.pt
CKPT_FLAG=""
if [ -f "${SP_CKPT}" ]; then
    CKPT_FLAG="--checkpoint-in ${SP_CKPT}"
    echo "[job] resuming self-play from ${SP_CKPT}"
else
    LATEST_SUP=$(ls -1 training/checkpoints/supervised_epoch*.pt 2>/dev/null \
                 | sort -V | tail -n 1 || true)
    if [ -n "${LATEST_SUP}" ]; then
        CKPT_FLAG="--checkpoint-in ${LATEST_SUP}"
        echo "[job] warmstart from ${LATEST_SUP}"
    elif [ -f training/checkpoints/supervised.pt ]; then
        CKPT_FLAG="--checkpoint-in training/checkpoints/supervised.pt"
        echo "[job] warmstart from training/checkpoints/supervised.pt"
    else
        echo "[job] WARN: no checkpoint found; starting from random init"
    fi
fi

# --workers 6: rollout worker threads. 8 CPUs allocated; one for the
# main thread (running gradient compute / queue management), one for
# OS slack, six driving rollouts in parallel. Each worker spends
# most of its time in encoder Python work + CPU policy forward; on
# GPU the forward happens off-thread (cuda kernels), so workers
# overlap with one another's encode steps.
#
# --iterations 1000: a large finite ceiling. Walltime cuts us off
# well before this (single iter is seconds-to-minutes); the chain
# resubmits from the latest checkpoint. Setting this huge means
# we never naturally terminate; only the manual `selfplay_done.flag`
# below stops the chain.
#
# --save-every 5: checkpoint every 5 iterations. Balances "lose
# little on SIGTERM" against "don't thrash the disk". A walltime
# kill mid-iteration loses at most ~5 iterations of progress.
#
# --max-turns 60: standard 2p Wesnoth games rarely run past turn 60
# and most decide by turn 40. Cap at 60 so a stuck game doesn't
# starve worker threads on a single rollout.
#
# --reward-config cluster/configs/reward_selfplay.json: tuned
# defaults (see the file's _about field). Tweak the JSON between
# chain links to retune without touching code.
ITERATIONS=1000
WORKERS=6
GAMES_PER_ITER=8
MAX_TURNS=60

TRAINER_EXIT=0
python -u tools/sim_self_play.py \
    --replay-pool replays_dataset \
    --checkpoint-out "${SP_CKPT}" \
    --device cuda \
    --iterations ${ITERATIONS} \
    --games-per-iter ${GAMES_PER_ITER} \
    --max-turns ${MAX_TURNS} \
    --workers ${WORKERS} \
    --save-every 5 \
    --reward-config cluster/configs/reward_selfplay.json \
    ${CKPT_FLAG} || TRAINER_EXIT=$?

echo "[job] trainer exit code: ${TRAINER_EXIT}"

# NO auto-resubmit. Same reasoning as cluster/job.sbatch:
#   1. bash doesn't reliably survive SLURM walltime kills, so the
#      chain didn't actually fire on walltime hits anyway.
#   2. Self-throttling on a shared cluster with daily quotas --
#      operator decides when to continue.
# Continue with: GUI "Sync + Continue" or
# `bash cluster/run.sh continue selfplay`.
echo "[job] not auto-resubmitting; use 'run.sh continue selfplay' or"
echo "[job] the GUI's 'Sync + Continue' button to start the next link."
