#!/usr/bin/env bash
#SBATCH --job-name=wai-profile
#SBATCH --partition=ENSTA-l40s
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=training/logs/profile-slurm-%j.log
#SBATCH --error=training/logs/profile-slurm-%j.log
#
# One-shot profiling run of the self-play loop on a real GPU.
# Wall time: 25 minutes -- enough for 1 warmup + 3 profiled iters
# at production settings, plus artifact dumps.
#
# Submit with:  sbatch cluster/job_profile.sbatch
# Status:       squeue --me
# Cancel:       scancel <jobid>
#
# Output artifacts land in:
#   training/profiles/<jobid>/
#     summary.txt              -- read this first
#     phase_timing.csv         -- per-iter rollout vs train_step
#     cprofile.prof            -- snakeviz / py-spy fodder
#     cprofile_cumulative.txt  -- top-30 by cumtime
#     cprofile_self.txt        -- top-30 by tottime
#     nvidia_smi.csv           -- GPU utilization timeline
#     torch_trace.json         -- (only if USE_TORCH_PROFILE=1)
#     args.json / env.json     -- the invocation + version info

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
assert torch.cuda.is_available(), "CUDA not available inside the job"
print(f"[job] torch:    {torch.__version__}")
print(f"[job] device:   {torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
PY

# Match the production self-play sbatch's defaults so the profile
# reflects the actual workload. Each env var here can be overridden
# from the command line:
#   sbatch --export=ALL,ITERATIONS=5,WORKERS=4 cluster/job_profile.sbatch
ITERATIONS="${ITERATIONS:-3}"
WARMUP_ITERS="${WARMUP_ITERS:-1}"
GAMES_PER_ITER="${GAMES_PER_ITER:-8}"
MAX_TURNS="${MAX_TURNS:-60}"
WORKERS="${WORKERS:-6}"
SEED="${SEED:-0}"
REWARD_CONFIG="${REWARD_CONFIG:-cluster/configs/reward_selfplay.json}"
USE_TORCH_PROFILE="${USE_TORCH_PROFILE:-0}"
NVIDIA_SMI_INTERVAL="${NVIDIA_SMI_INTERVAL:-1.0}"

# Auto-pick the freshest self-play checkpoint, falling back to the
# latest supervised epoch -- same logic as job_selfplay.sbatch.
SP_CKPT=training/checkpoints/sim_selfplay.pt
CKPT_ARG=""
if [ -f "${SP_CKPT}" ]; then
    CKPT_ARG="--checkpoint ${SP_CKPT}"
    echo "[job] profiling against ${SP_CKPT}"
else
    LATEST_SUP=$(ls -1 training/checkpoints/supervised_epoch*.pt 2>/dev/null \
                 | sort -V | tail -n 1 || true)
    if [ -n "${LATEST_SUP}" ]; then
        CKPT_ARG="--checkpoint ${LATEST_SUP}"
        echo "[job] profiling against ${LATEST_SUP} (no self-play ckpt yet)"
    else
        echo "[job] WARN: no checkpoint; profiling from random init"
    fi
fi

# MCTS overrides. Off by default; flip USE_MCTS=1 to profile the
# tree-search workload (very different from REINFORCE).
MCTS_ARGS=()
if [ "${USE_MCTS:-0}" = "1" ] || [ "${USE_MCTS:-}" = "true" ]; then
    MCTS_ARGS=(--mcts
               --mcts-sims    "${MCTS_SIMS:-50}"
               --mcts-c-puct  "${MCTS_C_PUCT:-1.5}")
fi

# torch.profiler is opt-in because it's expensive (~30-50% overhead
# and a multi-MB JSON trace). Set USE_TORCH_PROFILE=1 to enable.
TORCH_PROFILE_ARGS=()
if [ "${USE_TORCH_PROFILE}" = "1" ]; then
    TORCH_PROFILE_ARGS=(--torch-profile)
fi

# FORCED_FACTION: same handling as job_selfplay.sbatch. Empty -> use
# module default; "none" -> explicit disable; anything else -> lock.
FORCED_FACTION_ARGS=()
if [ -n "${FORCED_FACTION:-}" ]; then
    FORCED_FACTION_ARGS=(--forced-faction "${FORCED_FACTION}")
fi

echo "[job] params: iters=${ITERATIONS} warmup=${WARMUP_ITERS} " \
     "games/iter=${GAMES_PER_ITER} max_turns=${MAX_TURNS} " \
     "workers=${WORKERS} seed=${SEED} " \
     "reward=${REWARD_CONFIG} " \
     "torch_profile=${USE_TORCH_PROFILE} mcts=${USE_MCTS:-0}"

mkdir -p training/profiles training/logs

PROFILE_DIR="training/profiles/${SLURM_JOB_ID}"

python -u tools/profile_selfplay.py \
    --device cuda \
    --output-dir "${PROFILE_DIR}" \
    --iterations            "${ITERATIONS}" \
    --warmup-iterations     "${WARMUP_ITERS}" \
    --games-per-iter        "${GAMES_PER_ITER}" \
    --max-turns             "${MAX_TURNS}" \
    --workers               "${WORKERS}" \
    --seed                  "${SEED}" \
    --reward-config         "${REWARD_CONFIG}" \
    --nvidia-smi-interval   "${NVIDIA_SMI_INTERVAL}" \
    "${FORCED_FACTION_ARGS[@]}" \
    "${MCTS_ARGS[@]}" \
    "${TORCH_PROFILE_ARGS[@]}" \
    ${CKPT_ARG}

echo "[job] artifacts in ${PROFILE_DIR}:"
ls -la "${PROFILE_DIR}" 2>&1 | sed 's/^/[job]   /'
echo "[job] end: $(date -Is)"
echo ""
echo "[job] pull artifacts back with:"
echo "[job]   scp -r mesogip_outside:~/wesnoth-ai/${PROFILE_DIR} training/profiles/"
echo "[job] then open the summary:"
echo "[job]   cat ${PROFILE_DIR}/summary.txt"
