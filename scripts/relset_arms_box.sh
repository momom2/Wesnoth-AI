#!/usr/bin/env bash
# Relevant-set imitation experiment, two arms
# (docs/model_cost_study_20260905.md section 7, pre-registered).
#
# From the seed weights, half an epoch of imitation each, same seed and
# pair stream:
#   control  full-board hex basis
#   relset   relevant-set hex basis (--relevant-set-hexes)
# then PURE matches, raw:t0 on both sides, sides alternated, ladder maps:
#   control vs seed   800 games
#   relset  vs seed   800 games
#   relset  vs control 400 games
#
# Run ON a 4090 box (docs/box_specs.md amendments 2026-09-04: 24 cores,
# 30+ GB RAM, $0.33/h) after
#   bash Wesnoth-AI/scripts/eval_box_setup.sh \
#       tier-b/a3/seed_imit_tierb_start.pt=seed.pt
# staged the seed as training/checkpoints/seed.pt. The imitation corpus
# is staged here from HF (tier-b/replays_dataset_imitation_dedup_20260908.tar.gz,
# token in /workspace/.hf_token). Everything lands under
# /workspace/relset/; finished stages leave a DONE marker and are
# skipped on re-entry; /workspace/relset/DONE closes the run.
#
# Cost (study section 7, box time at $0.33/h):
#   arms          2 x 4.6 h (1.26M pairs at 76 pairs/s)   $3.0
#   800-game raw matches, 2 x ~40 min                     $0.72
#   400-game arm-vs-arm match, ~20 min                    $0.18
#   bring-up, pulls                                       ~$0.3
#   total about $4.3, one box-day (full-epoch arms: $7.5)
# The holdout eval every 50k pairs replays 150 holdout games per eval
# (stratified reservoir); in the relevant-set arm each replayed pair
# builds its label in the subset basis, so budget up to ~1 h more for
# that arm's 25 evals.
set -uo pipefail
[ -x /venv/main/bin/python ] && export PATH=/venv/main/bin:$PATH
cd /workspace/Wesnoth-AI
OUT=/workspace/relset
mkdir -p "$OUT"
SEED=training/checkpoints/seed.pt
DATASET=replays_dataset_imitation

PAIRS="${PAIRS:-1260000}"          # 0.5 epoch of the 2.515M winner-side pairs
RUN_SEED="${RUN_SEED:-20260905}"   # file order + value subsampling, both arms
WORKERS="${WORKERS:-20}"           # encode workers; the run is CPU-bound
EVAL_EVERY="${EVAL_EVERY:-50000}"
GAMES_VS_SEED="${GAMES_VS_SEED:-800}"
GAMES_ARMS="${GAMES_ARMS:-400}"
JOBS="${JOBS:-10}"
DEV="${DEV:-cuda}"

if [ ! -f "$SEED" ]; then
    echo "missing $SEED: run scripts/eval_box_setup.sh first" >&2
    exit 1
fi
if [ -f /workspace/.hf_token ]; then
    export HF_TOKEN="$(cat /workspace/.hf_token)"
fi
if [ ! -f "$DATASET/manifest.jsonl" ]; then
    python - <<'EOF' || { echo "dataset staging failed" >&2; exit 1; }
import pathlib, tarfile
from huggingface_hub import hf_hub_download
# The fog-annotated corpus (2026-09-07: fog/shroud per side, 20 games
# quarantined); the tarball carries its top-level folder.
p = hf_hub_download("momom2/wesnoth-model-checkpoints",
                    "tier-b/replays_dataset_imitation_dedup_20260908.tar.gz")
dst = pathlib.Path("replays_dataset_imitation")
with tarfile.open(p, "r:gz") as tf:
    tf.extractall(".")
print(f"imitation dataset: {len(list(dst.glob('*.json.gz')))} games")
EOF
fi
{ nproc; free -g | head -2; nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null; git rev-parse HEAD; } > "$OUT/box.txt" 2>&1

# ---- arms ------------------------------------------------------------
# 15M seed arch (wesnoth_ai/model.py:262-275). Imitation mode as the
# seed's own run: winners-only policy CE, per-game weighting, outcome
# value supervision, manifest holdout; --reinit-value-head off. Fresh
# optimizer and counters from --init-from; the cosine schedule steps
# per epoch, so the LR is flat over the half epoch.
run_arm() {                       # run_arm NAME [extra flags...]
    local name="$1"; shift
    local dir="$OUT/$name"
    if [ -f "$dir/DONE" ]; then echo "arm $name already done"; return 0; fi
    mkdir -p "$dir"
    python tools/supervised_train.py "$DATASET" \
        --checkpoint "$dir/arm.pt" \
        --init-from "$SEED" \
        --imitation-config configs/imitation.json \
        --d-model 384 --num-layers 8 --num-heads 12 --d-ff 1536 \
        --epochs 1 --max-pairs "$PAIRS" --seed "$RUN_SEED" \
        --bs 64 --lr 1e-4 --device "$DEV" --workers "$WORKERS" \
        --eval-every "$EVAL_EVERY" --eval-pairs 1200 \
        --eval-pairs-per-game 8 --eval-sample-seed 0 \
        --ckpt-every 2000 --log-every 100 \
        "$@" 2>&1 | tee -a "$dir/train.log"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ] || [ ! -f "$dir/arm.pt" ]; then
        echo "arm $name FAILED rc=$rc" >&2; return 1
    fi
    touch "$dir/DONE"
}

run_arm control || exit 1
run_arm relset --relevant-set-hexes || exit 1

# ---- matches ---------------------------------------------------------
# raw:t0 on both sides (--mcts-sims 0, temperature 0). The relevant-set
# arm's checkpoint carries relevant_set_hexes=True; eval_sim's loader
# peeks it and builds both encoders in that basis, so no
# --relevant-set-a/-b flag is passed (those force the basis on a
# checkpoint that was not trained in it). The result files record the
# effective basis per side (basis_a/basis_b).
#
# A match is DONE only when run_elo_batch exited 0 AND the outdir holds
# the pre-registered number of decisive results: a time-budget cut, a
# memory stop or crashed children all leave run_elo_batch at exit 0
# with a short outdir, and elo_collect fits whatever is there. Short
# means exit 1 here; re-entry resumes the match from its files.
decisive_results() {              # decisive_results OUTDIR -> count of win/loss files
    python - "$1" <<'EOF'
import json, pathlib, sys
n = 0
for p in pathlib.Path(sys.argv[1]).glob("game_*.json"):
    try:
        n += json.loads(p.read_text(encoding="utf-8")).get("outcome_a") in ("win", "loss")
    except Exception:
        pass
print(n)
EOF
}

run_match() {                     # run_match NAME LABEL_A SPEC_A LABEL_B SPEC_B GAMES SEED_BASE BUDGET_MIN
    local name="$1" la="$2" sa="$3" lb="$4" sb="$5" games="$6" sb0="$7" budget="$8"
    local dir="$OUT/$name"
    if [ -f "$dir.DONE" ]; then echo "match $name already done"; return 0; fi
    python tools/run_elo_batch.py --label-a "$la" --spec-a "$sa" \
        --label-b "$lb" --spec-b "$sb" \
        --outdir "$dir" --games "$games" --seed-base "$sb0" \
        --mcts-sims 0 --raw-temperature-a 0 --raw-temperature-b 0 \
        --persistent-workers --shared-inference --no-infer-compile --device "$DEV" --jobs "$JOBS" \
        --time-budget-min "$budget" 2>&1 | tee -a "$dir.log"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "match $name: run_elo_batch FAILED rc=$rc" >&2; return 1
    fi
    local have
    have=$(decisive_results "$dir")
    if [ "$have" -lt "$games" ]; then
        echo "match $name SHORT: $have/$games decisive results; re-run to continue" >&2
        return 1
    fi
    python tools/elo_collect.py "$dir" --no-catalog \
        --save-json "$dir.fit.json" 2>&1 | tee -a "$dir.log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ] || [ ! -f "$dir.fit.json" ]; then
        echo "match $name: elo_collect FAILED rc=$rc" >&2; return 1
    fi
    touch "$dir.DONE"
}

CONTROL="$OUT/control/arm.pt"
RELSET="$OUT/relset/arm.pt"
run_match control_vs_seed control "$CONTROL" seed_t0 "$SEED" "$GAMES_VS_SEED" 10000 120 || exit 1
run_match relset_vs_seed  relset  "$RELSET"  seed_t0 "$SEED" "$GAMES_VS_SEED" 20000 120 || exit 1
run_match relset_vs_control relset "$RELSET" control "$CONTROL" "$GAMES_ARMS" 30000 60 || exit 1

# What to pull: $OUT/{control,relset}/{arm.pt,arm_eval.jsonl,train.log},
# $OUT/*.fit.json, $OUT/*.log, $OUT/box.txt. Read
# target_masked_ce / target_masked_top1 at equal pairs from the two
# arm_eval.jsonl files (the unmasked ce is not comparable across
# bases), then the three fits with their SE.
touch "$OUT/DONE"
echo RELSET_DONE
