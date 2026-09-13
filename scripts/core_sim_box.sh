#!/usr/bin/env bash
# The simulator on the Rust-owned state, timed on the eval path
# (docs/rust_port_plan.md phase 4): the reference player against
# itself, 40 games, shared inference, 20 workers, with the core off
# (`WESNOTH_RUST_CORE=0`, the Python state of record) and on, twice
# each, then one lone game under py-spy per mode. The observation
# kernels stay on in both modes. Records under $OUT.
#
#   OUT=/workspace/coresim CKPT=training/checkpoints/relset.pt bash scripts/core_sim_box.sh
set -uo pipefail
OUT="${OUT:-/workspace/coresim}"
CKPT="${CKPT:-training/checkpoints/relset.pt}"
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
python -c "import wesnoth_core; assert wesnoth_core.__phase__ >= 8, wesnoth_core.__phase__" || { echo "wheel predates phase 8"; exit 1; }
{ nproc --all; cat /sys/fs/cgroup/cpu.max 2>/dev/null; nvidia-smi --query-gpu=name --format=csv,noheader; } > "$OUT/box.txt" 2>&1

timed_match() {                   # timed_match MODE CORE_FLAG
    local mode="$1"
    local flag="$2"
    local dir="$OUT/games_$mode"
    local t0
    t0=$(date +%s)
    WESNOTH_RUST_CORE="$flag" python tools/run_elo_batch.py --label-a relset --spec-a "$CKPT" \
        --label-b relset_ref --spec-b "$CKPT" \
        --outdir "$dir" --games 40 --max-extra-games 0 --seed-base 20000 \
        --mcts-sims 0 --raw-temperature-a 0 --raw-temperature-b 0 \
        --persistent-workers --shared-inference --no-infer-compile --device cuda \
        --jobs 20 --inference-max-batch 20 \
        --time-budget-min 40 2>&1 | grep --line-buffered -v "wesnoth_core is not importable" > "$OUT/$mode.log"
    echo "$mode $(( $(date +%s) - t0 )) s $(ls "$dir"/game_*.json 2>/dev/null | wc -l) games" | tee "$OUT/$mode.wall"
    cp "$dir"/.inference_server_*.json "$OUT/$mode.server_stats.json" 2>/dev/null || true
}
timed_match P 0
timed_match C 1
timed_match P2 0
timed_match C2 1

for mode in on off; do
    flag=$([ "$mode" = on ] && echo 1 || echo 0)
    setsid bash -c "sleep 100000 | python -u tools/eval_inference_server.py --spec $CKPT --device cuda --window-ms 1.5 --max-batch 4 \
        --stats-out $OUT/prof_server_$mode.json --label prof > $OUT/prof_server_$mode.out 2> $OUT/prof_server_$mode.err" &
    SPID=$!
    for i in $(seq 1 120); do grep -q "^__ADDR__ " "$OUT/prof_server_$mode.out" && break; sleep 2; done
    ADDR=$(grep "^__ADDR__ " "$OUT/prof_server_$mode.out" | head -1 | sed "s/^__ADDR__ //")
    mkdir -p "$OUT/prof_games_$mode"
    t0=$(date +%s)
    WESNOTH_RUST_CORE="$flag" py-spy record --duration 200 --format raw --idle --nonblocking -o "$OUT/pyspy_$mode.txt" -- \
        python tools/elo_eval_game.py relset "$CKPT" relset_ref "$CKPT" 1 20001 "$OUT/prof_games_$mode" \
        --max-turns 200 --mcts-sims 0 --raw-temperature-a 0 --raw-temperature-b 0 \
        --inference-address-a "$ADDR" --inference-address-b "$ADDR" --infer-bf16 --infer-packed-trunk \
        --log-level INFO > "$OUT/prof_game_$mode.log" 2>&1
    echo "lone_$mode $(( $(date +%s) - t0 )) s" | tee "$OUT/lone_$mode.wall"
    python tools/pyspy_summary.py "$OUT/pyspy_$mode.txt" --top 25 > "$OUT/pyspy_$mode.summary.txt" 2>&1 || true
    kill -- -$SPID 2>/dev/null; pkill -f "eval_inference_server.py --spec $CKPT" 2>/dev/null; sleep 2
done
cat "$OUT"/*.wall
touch "$OUT/ALL_DONE"
echo CORE_SIM_DONE
