#!/usr/bin/env bash
#SBATCH --job-name=wai-supervised
#SBATCH --partition=ENSTA-l40s
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:55:00
#SBATCH --output=training/logs/slurm-%j.log
#SBATCH --error=training/logs/slurm-%j.log
#
# Supervised training of the Wesnoth model on one GPU.
# Submit with:  sbatch cluster/job.sbatch
# Status:       squeue --me        (or  bash cluster/run.sh status)
# Cancel:       scancel <jobid>    (or  bash cluster/run.sh stop)

set -euo pipefail

# When sbatch is invoked from the project root, SLURM_SUBMIT_DIR is the
# project root; that's the path our scripts assume. cd-ing there here
# makes the script callable from anywhere.
cd "${SLURM_SUBMIT_DIR}"

echo "[job] node:       $(hostname)"
echo "[job] job_id:     ${SLURM_JOB_ID}"
echo "[job] start:      $(date -Is)"
echo "[job] gpu(s):"
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>&1 | sed 's/^/[job]   /'

# Activate the venv (built by cluster/setup.sh on the login node — same
# filesystem on compute nodes since /home is shared).
# shellcheck disable=SC1091
source .venv/bin/activate

# Final sanity check before the long run: torch must see the GPU.
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available inside the job — check driver / wheel"
print(f"[job] torch:    {torch.__version__}")
print(f"[job] device:   {torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
PY

mkdir -p training/checkpoints training/logs

# Hyperparameters identical to the local CPU baseline. `--device cuda`
# routes the model + encoder onto the GPU; the encoder's Python state-
# building still runs on the CPU side of each step, but on a compute
# node with 8 cores per task that's no longer the bottleneck.
#
# Auto-resume across day-boundary jobs: the student QoS caps a single
# job at 24h, so we run for 23:55 and rely on chained submissions for
# longer training. Each fresh submission inherits whatever the previous
# job's checkpoint left behind (model + encoder + optimizer state +
# step/pair counters). If no checkpoint exists yet, --resume's path
# check inside the trainer just falls through and starts from scratch.
RESUME_FLAG=""
if [ -f training/checkpoints/supervised.pt ]; then
    RESUME_FLAG="--resume training/checkpoints/supervised.pt"
    echo "[job] resuming from training/checkpoints/supervised.pt"
fi

# Resubmission chain. Student QoS limits us to 4h per job (slide 16
# of the cluster doc); training likely needs more. Strategy: when this
# job ends, if the work isn't complete (no `done.flag`), submit a
# follow-up that will auto-resume from the checkpoint we just wrote.
# MaxJobs=1 + MaxSubmit=5 means at most one resubmit can be queued at
# a time, which is exactly what we want.
#
# When walltime hits, SLURM SIGTERMs python; the trainer doesn't catch
# it, so the last periodic checkpoint (every 500 steps, ~30s on GPU)
# is what survives. Worst-case lost work is ~500 steps. exit 143 from
# python = SIGTERM-killed; we count that as a normal end-of-job.
TRAINER_EXIT=0
# --workers 6: prefetch encode_raw in 6 subprocesses while the main
# thread runs encode_from_raw + model forward/backward on the GPU.
# Hides the CPU encode work (~3-4ms/pair) behind the GPU step time.
# 8 CPUs allocated → 6 workers + 1 main + 1 OS slack. Workers
# inherit a frozen vocab snapshot pre-seeded from unit_stats.json
# (356 unit types) so out-of-vocab clamps stay rare.
#
# Resume re-uses the saved encoder vocab dict, so a follow-up job
# in the chain doesn't re-seed (would only matter for fresh runs
# anyway — vocab IDs in the checkpoint are the source of truth).
# EPOCHS and the sentinel snapshot path go together: the trainer
# writes `supervised_epoch{EPOCHS-1}.pt` after the last epoch. Bump
# both atomically here so we don't desync (running with --epochs N
# but checking for epoch{M}.pt would chain forever even on success).
EPOCHS=10
EPOCHS_LAST_SENTINEL=training/checkpoints/supervised_epoch$((EPOCHS - 1)).pt

python -u tools/supervised_train.py replays_dataset \
    --bs 32 \
    --lr 1e-4 \
    --epochs ${EPOCHS} \
    --device cuda \
    --log-every 100 \
    --ckpt-every 500 \
    --workers 6 \
    --checkpoint training/checkpoints/supervised.pt \
    ${RESUME_FLAG} || TRAINER_EXIT=$?

echo "[job] trainer exit code: ${TRAINER_EXIT}"

# "Done" detection: trainer finished all $EPOCHS epochs => the last
# per-epoch snapshot exists. Drop a done.flag so the operator (and
# any future automation) sees training as complete.
if [ -f "${EPOCHS_LAST_SENTINEL}" ]; then
    touch training/checkpoints/done.flag
    echo "[job] training complete (${EPOCHS_LAST_SENTINEL} present)"
fi

# NO auto-resubmit. Two reasons:
#   1. The bash script doesn't reliably survive SLURM's walltime
#      kill (SIGTERM goes to the whole step; bash exits before
#      reaching this code), so the chain didn't actually fire on
#      walltime hits anyway.
#   2. Self-throttling -- on a shared cluster with daily quotas,
#      auto-resubmission is impolite. The operator decides when to
#      continue: GUI "Sync + Continue" or
#      `bash cluster/run.sh continue supervised`.
echo "[job] not auto-resubmitting; use 'run.sh continue supervised' or"
echo "[job] the GUI's 'Sync + Continue' button to start the next link."
