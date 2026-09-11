#!/usr/bin/env bash
# Eval-path throughput, one factor at a time (plan 1.5, 2026-09-11).
# The same 40-game raw:t0 match (seed against itself, seed base 20000,
# no replacements) is timed in four modes on one box, then one worker
# is profiled with py-spy. Modes:
#   A  persistent workers, Python enumeration      (today's match script)
#   B  A + --shared-inference                      (measured 2026-09-05: 215 -> 145 s on a 4090)
#   C  B + the Rust wesnoth_core wheel             (masks and enumeration in Rust)
#   D  C without --shared-inference                (the wheel alone)
# ON THE BOX after /workspace/Wesnoth-AI is staged with the seed at
# training/checkpoints/seed.pt:  bash eval_profile_box.sh
# Records under /workspace/evalprof/: <mode>.log, <mode>.wall (seconds),
# the server stats json per mode, pyspy_*.txt and the summaries;
# uploaded to HF tier-b/eval_profile_20260911/ at the end.
set -uo pipefail
OUT=/workspace/evalprof
mkdir -p "$OUT"
cd /workspace/Wesnoth-AI
export HF_TOKEN="$(tr -d '\r\n' < /workspace/.hf_token)" HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
SEED=training/checkpoints/seed.pt
GAMES="${GAMES:-40}"
JOBS="${JOBS:-10}"
{ nproc --all; cat /sys/fs/cgroup/cpu.max 2>/dev/null; nvidia-smi --query-gpu=name --format=csv,noheader; python -c "import wesnoth_core" 2>/dev/null && echo "wesnoth_core present" || echo "wesnoth_core absent"; } > "$OUT/box.txt" 2>&1

timed_match() {                   # timed_match MODE [extra run_elo_batch flags...]
    local mode="$1"; shift
    if [ -f "$OUT/$mode.wall" ]; then echo "mode $mode already done"; return 0; fi
    local dir="$OUT/games_$mode"
    local t0=$(date +%s)
    python tools/run_elo_batch.py --label-a seed --spec-a "$SEED" --label-b seed_ref --spec-b "$SEED" \
        --outdir "$dir" --games "$GAMES" --max-extra-games 0 --seed-base 20000 \
        --mcts-sims 0 --raw-temperature-a 0 --raw-temperature-b 0 \
        --persistent-workers --no-infer-compile --device cuda --jobs "$JOBS" \
        --time-budget-min 40 "$@" 2>&1 | grep --line-buffered -v "wesnoth_core is not importable" > "$OUT/$mode.log"
    local t1=$(date +%s)
    echo $((t1 - t0)) > "$OUT/$mode.wall"
    ls "$dir"/game_*.json 2>/dev/null | wc -l > "$OUT/$mode.games"
    cp "$dir"/.inference_server_*.json "$OUT/$mode.server_stats.json" 2>/dev/null || true
    echo "mode $mode: $((t1 - t0)) s for $(cat "$OUT/$mode.games") games"
}

timed_match A
timed_match B --shared-inference

# The Rust core (bench_box.sh recipe).
if ! python -c "import wesnoth_core" 2>/dev/null; then
    command -v cc >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq gcc) >/dev/null 2>&1 || true
    command -v cargo >/dev/null 2>&1 || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1
    export PATH="$HOME/.cargo/bin:$PATH"
    python -m pip install -q maturin >/dev/null 2>&1
    python -m pip install -q rust/wesnoth_core 2>&1 | tail -1
fi
python -c "import wesnoth_core; print('wesnoth_core built')" > "$OUT/rust.txt" 2>&1 || echo "wesnoth_core build FAILED" | tee -a "$OUT/rust.txt"

timed_match C --shared-inference
timed_match D

# One worker under py-spy (parent mode; attach is refused on Vast), the
# per-process path so the forward shows up beside the Python.
python -m pip install -q py-spy >/dev/null 2>&1
mkdir -p "$OUT/pyspy_games"
for flag in "" "--gil"; do
    tag=$([ -n "$flag" ] && echo gil || echo all)
    py-spy record --duration 150 --format raw --threads --idle --nonblocking $flag -o "$OUT/pyspy_$tag.txt" -- \
        python tools/elo_eval_game.py seed "$SEED" seed_ref "$SEED" 1 20001 "$OUT/pyspy_games" \
        --max-turns 200 --mcts-sims 0 --raw-temperature-a 0 --raw-temperature-b 0 \
        --device cuda --no-infer-compile --log-level INFO > "$OUT/pyspy_$tag.log" 2>&1
    python tools/pyspy_summary.py "$OUT/pyspy_$tag.txt" > "$OUT/pyspy_$tag.summary.txt" 2>&1 || true
done

python - <<'EOF'
import glob, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for p in sorted(glob.glob("/workspace/evalprof/*")):
    if os.path.isfile(p) and os.path.getsize(p) < 50_000_000:
        api.upload_file(path_or_fileobj=p, path_in_repo="tier-b/eval_profile_20260911/" + os.path.basename(p),
                        repo_id="momom2/wesnoth-model-checkpoints")
print("uploaded", flush=True)
EOF
touch "$OUT/ALL_DONE"
echo EVAL_PROFILE_DONE
