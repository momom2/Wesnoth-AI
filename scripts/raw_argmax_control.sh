#!/usr/bin/env bash
# The raw-argmax control (review 2026-09-04): does "search beats the
# raw seed" survive when the raw seed plays argmax instead of
# sampling? Two 40-game matches on the seed's own weights:
#   A. seed raw-argmax   vs seed raw-sampling   (tools/raw_player.py)
#   B. seed+MCTS-32      vs seed raw-argmax     (catalog procedure)
# Run ON an eval box after scripts/eval_box_setup.sh staged the seed
# as training/checkpoints/seed.pt. Writes /workspace/control/DONE
# when both fits are on disk; games accumulate under
# /workspace/control/{A_argmax_vs_sample,B_search_vs_argmax}.
set -uo pipefail
[ -x /venv/main/bin/python ] && export PATH=/venv/main/bin:$PATH
cd /workspace/Wesnoth-AI
OUT=/workspace/control
mkdir -p "$OUT"
SEED=training/checkpoints/seed.pt
GAMES="${GAMES:-40}"
JOBS="${JOBS:-10}"
DEV="${DEV:-cuda}"

python tools/run_elo_batch.py --label-a seed_argmax --spec-a "$SEED" \
    --label-b seed_sample --spec-b "$SEED" \
    --outdir "$OUT/A_argmax_vs_sample" --games "$GAMES" \
    --mcts-sims 0 --raw-temperature-a 0 \
    --device "$DEV" --jobs "$JOBS" --time-budget-min 120 \
    2>&1 | tee -a "$OUT/A.log"
python tools/elo_collect.py "$OUT/A_argmax_vs_sample" --no-catalog \
    --save-json "$OUT/A_fit.json" 2>&1 | tee -a "$OUT/A.log"

python tools/run_elo_batch.py --label-a seed_mcts32 --spec-a "$SEED" \
    --label-b seed_argmax --spec-b "$SEED" \
    --outdir "$OUT/B_search_vs_argmax" --games "$GAMES" \
    --mcts-sims-a 32 --mcts-sims-b 0 --raw-temperature-b 0 \
    --no-turn-search \
    --device "$DEV" --jobs "$JOBS" --time-budget-min 150 \
    2>&1 | tee -a "$OUT/B.log"
python tools/elo_collect.py "$OUT/B_search_vs_argmax" --no-catalog \
    --save-json "$OUT/B_fit.json" 2>&1 | tee -a "$OUT/B.log"

touch "$OUT/DONE"
echo CONTROL_DONE
