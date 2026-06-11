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
# --iterations 100000: a very large safety ceiling. The real exit
# criterion is --time-budget below: the trainer saves a final
# checkpoint when wall-time is exhausted, so SIGTERM from SLURM
# is no longer the loop-exit mechanism. The chain resubmits from
# the saved checkpoint via `bash cluster/run.sh continue selfplay`
# (or the GUI's "Sync + Continue" button); only the manual
# `selfplay_done.flag` stops the chain entirely.
#
# --time-budget 03:50:00: 3h50m. Walltime ceiling above is 03:55:00,
# so this leaves 5min headroom for the trainer to save + exit
# cleanly, well before SLURM hard-kills the process. With the old
# iter-based bound, walltime kills mid-iteration could lose up to
# `save-every` iterations of progress -- now the loop exits at
# its own checkpoint boundary so worst-case loss is one iteration
# of rollout.
#
# --save-every 5: checkpoint every 5 iterations. Now mostly
# defensive (matters only if the process is force-killed mid-iter,
# e.g. operator hits Stop). Time-budget exit always saves before
# return.
#
# --max-turns 60: standard 2p Wesnoth games rarely run past turn 60
# and most decide by turn 40. Cap at 60 so a stuck game doesn't
# starve worker threads on a single rollout.
#
# --reward-config cluster/configs/reward_selfplay.json: tuned
# defaults (see the file's _about field). Tweak the JSON between
# chain links to retune without touching code.
# Each knob accepts an env var override (sbatch --export=KEY=VAL).
# The defaults below run if the operator submits without --export,
# which matches the "Start self-play" button's behavior.
ITERATIONS="${ITERATIONS:-100000}"
TIME_BUDGET="${TIME_BUDGET:-03:50:00}"
WORKERS="${WORKERS:-6}"
GAMES_PER_ITER="${GAMES_PER_ITER:-8}"
MAX_TURNS="${MAX_TURNS:-200}"
SAVE_EVERY="${SAVE_EVERY:-5}"
SEED="${SEED:-0}"
REWARD_CONFIG="${REWARD_CONFIG:-cluster/configs/reward_selfplay.json}"
# FORCED_FACTION: empty = use sim_self_play module default
# (Knalgan Alliance). "none" = explicit disable. Anything else =
# lock to that faction.
#
# Default-era faction names contain a space ("Knalgan Alliance"),
# so we MUST pass the value as a single argv token. A flat string
# `FORCED_FACTION_ARG="--forced-faction ${FORCED_FACTION}"`
# expanded unquoted at the call site word-splits on the space and
# argparse sees "Alliance" as a stray positional ("unrecognized
# arguments: Alliance"). Use a bash array so the space-containing
# value stays a single token through expansion.
FORCED_FACTION_ARGS=()
if [ -n "${FORCED_FACTION:-}" ]; then
    FORCED_FACTION_ARGS=(--forced-faction "${FORCED_FACTION}")
fi

# MCTS overrides. USE_MCTS={1,true} flips action selection to
# AlphaZero-style tree search: `MCTS_SIMS` simulations per move,
# `MCTS_C_PUCT` exploration constant, `MCTS_BATCH_SIZE` for
# batched leaf eval. Defaults match the sim_self_play.py CLI
# defaults; per-step shaping reward is ignored in MCTS mode.
MCTS_ARGS=()
if [ "${USE_MCTS:-0}" = "1" ] || [ "${USE_MCTS:-}" = "true" ]; then
    MCTS_ARGS=(--mcts
               --mcts-sims     "${MCTS_SIMS:-50}"
               --mcts-c-puct   "${MCTS_C_PUCT:-1.5}"
               --mcts-batch-size "${MCTS_BATCH_SIZE:-1}")
fi

# Mini-maps mode. MINI_MAPS={1,true} restricts scenario sampling to
# the 5 smallest Ladder maps (Sablestone Delta, Weldyn Channel, Den
# of Onis, Swamp of Dread, Hamlets) -- ~690-870 cells vs the full
# pool's 690-2352. Used for the engagement-curriculum phase: smaller
# maps mean leaders start ~12-15 hexes apart, reducing the long-
# march cost that has been suspected of dominating the policy
# gradient. Off by default (full pool). See
# scenario_pool.MINI_MAP_SCENARIO_IDS.
MINI_MAPS_ARGS=()
if [ "${MINI_MAPS:-0}" = "1" ] || [ "${MINI_MAPS:-}" = "true" ]; then
    MINI_MAPS_ARGS=(--mini-maps)
fi

# Mini/ladder MIX. MINI_RATIO is a float in [0,1]: fraction of games
# per iter that sample from the mini-maps pool (rest go to ladder).
# Ignored when MINI_MAPS=1 (already 100% mini). Set MINI_RATIO=0.3
# to get ~30% mini games per iter alongside ladder.
MINI_RATIO_ARGS=()
if [ -n "${MINI_RATIO:-}" ] && [ "${MINI_RATIO}" != "0" ] && [ "${MINI_RATIO}" != "0.0" ]; then
    MINI_RATIO_ARGS=(--mini-ratio "${MINI_RATIO}")
fi

echo "[job] params: iters=${ITERATIONS} time_budget=${TIME_BUDGET} " \
     "games/iter=${GAMES_PER_ITER} max_turns=${MAX_TURNS} " \
     "workers=${WORKERS} save_every=${SAVE_EVERY} seed=${SEED} " \
     "forced_faction='${FORCED_FACTION:-(default)}' " \
     "reward=${REWARD_CONFIG} " \
     "mini_maps=${MINI_MAPS:-0} mini_ratio=${MINI_RATIO:-0} " \
     "mcts=${USE_MCTS:-0} (sims=${MCTS_SIMS:-50}, c_puct=${MCTS_C_PUCT:-1.5})"

TRAINER_EXIT=0
python -u tools/sim_self_play.py \
    --checkpoint-out "${SP_CKPT}" \
    --device cuda \
    --iterations ${ITERATIONS} \
    --time-budget "${TIME_BUDGET}" \
    --games-per-iter ${GAMES_PER_ITER} \
    --max-turns ${MAX_TURNS} \
    --workers ${WORKERS} \
    --save-every ${SAVE_EVERY} \
    --seed ${SEED} \
    --reward-config "${REWARD_CONFIG}" \
    "${FORCED_FACTION_ARGS[@]}" \
    "${MCTS_ARGS[@]}" \
    "${MINI_MAPS_ARGS[@]}" \
    "${MINI_RATIO_ARGS[@]}" \
    ${CKPT_FLAG} || TRAINER_EXIT=$?

echo "[job] trainer exit code: ${TRAINER_EXIT}"

# Storage hygiene. ENSTA flagged that accumulating GB of stale
# checkpoints + per-job SLURM logs over weeks could trip a quota and
# cut access. Each chain link runs the comprehensive cleanup:
#   - copy sim_selfplay.pt -> sim_selfplay_archive_<ts>.pt if the
#     freshest existing archive is older than the stride (default 6h)
#   - apply tiered retention to archives (hourly < 24h, daily < 7d,
#     weekly < 30d, monthly thereafter)
#   - drop legacy checkpoint_<N>.pt + *.bak
#   - keep only the freshest supervised_epoch*.pt
#   - keep latest 10 selfplay-slurm-*.log + slurm-*.log
#
# Idempotent + safe to run any time. See tools/cleanup.py for the
# full policy (the docstring is the spec).
if [ -f tools/cleanup.py ]; then
    python -u tools/cleanup.py --location cluster --yes \
        --keep-logs 10 || \
        echo "[job] tools/cleanup.py failed (non-fatal); continuing"
fi

# NO auto-resubmit. Same reasoning as cluster/job.sbatch:
#   1. bash doesn't reliably survive SLURM walltime kills, so the
#      chain didn't actually fire on walltime hits anyway.
#   2. Self-throttling on a shared cluster with daily quotas --
#      operator decides when to continue.
# Continue with: GUI "Sync + Continue" or
# `bash cluster/run.sh continue selfplay`.
echo "[job] not auto-resubmitting; use 'run.sh continue selfplay' or"
echo "[job] the GUI's 'Sync + Continue' button to start the next link."
