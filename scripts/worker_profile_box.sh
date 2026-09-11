#!/usr/bin/env bash
# The shared-inference eval worker under py-spy (plan 1.2, 2026-09-11):
# what a worker's Python per decision is made of once the forward and
# the priors live on the server. Starts one inference server by hand,
# plays one raw:t0 game through it under py-spy (parent mode; attach
# is refused on Vast), once per observation-kernel mode in MODES
# (on = WESNOTH_RUST_OBSERVE=1, off = 0; same game, same box),
# summarizes, uploads to HF $HF_DIR.
# Expects /workspace/Wesnoth-AI staged with training/checkpoints/seed2.pt
# and the Rust core built (seed2_relset_box.sh does both first).
set -uo pipefail
OUT="${OUT:-/workspace/workerprof}"
HF_DIR="${HF_DIR:-tier-b/worker_profile_20260911}"
MODES="${MODES:-on off}"
mkdir -p "$OUT"
cd /workspace/Wesnoth-AI
export HF_TOKEN="$(tr -d '\r\n' < /workspace/.hf_token)" HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 PATH="$HOME/.cargo/bin:$PATH"
python -m pip install -q py-spy >/dev/null 2>&1
CKPT=training/checkpoints/seed2.pt
# The server serves until its stdin closes (the driver holds a pipe);
# a sleep on the pipe keeps it alive here, killed with the group.
setsid bash -c "sleep 100000 | python -u tools/eval_inference_server.py --spec $CKPT --device cuda --window-ms 1.5 --max-batch 4 \
    --stats-out $OUT/server_stats.json --label prof > $OUT/server.out 2> $OUT/server.err" &
SERVER_PID=$!
for i in $(seq 1 120); do grep -q "^__ADDR__ " "$OUT/server.out" && break; sleep 2; done
ADDR=$(grep "^__ADDR__ " "$OUT/server.out" | head -1 | sed "s/^__ADDR__ //")
[ -n "$ADDR" ] || { echo "server gave no address" >&2; cat "$OUT/server.err" | tail -20; kill $SERVER_PID; exit 1; }
echo "server at $ADDR"
for mode in $MODES; do
    observe=$([ "$mode" = on ] && echo 1 || echo 0)
    mkdir -p "$OUT/games_$mode"
    t0=$(date +%s)
    WESNOTH_RUST_OBSERVE="$observe" py-spy record --duration 200 --format raw --threads --idle --nonblocking -o "$OUT/pyspy_$mode.txt" -- \
        python tools/elo_eval_game.py seed2 "$CKPT" seed2_ref "$CKPT" 1 20001 "$OUT/games_$mode" \
        --max-turns 200 --mcts-sims 0 --raw-temperature-a 0 --raw-temperature-b 0 \
        --inference-address-a "$ADDR" --inference-address-b "$ADDR" --infer-bf16 --infer-packed-trunk \
        --log-level INFO > "$OUT/game_$mode.log" 2>&1
    echo "$mode $(( $(date +%s) - t0 )) s" >> "$OUT/walls.txt"
    python tools/pyspy_summary.py "$OUT/pyspy_$mode.txt" > "$OUT/pyspy_$mode.summary.txt" 2>&1 || true
done
kill -- -$SERVER_PID 2>/dev/null; pkill -f "tools/eval_inference_server.py --spec $CKPT" 2>/dev/null; sleep 2
python - "$OUT" "$HF_DIR" <<'EOF'
import glob, os, sys
from huggingface_hub import HfApi
out, hf_dir = sys.argv[1], sys.argv[2]
api = HfApi(token=os.environ["HF_TOKEN"])
for p in sorted(glob.glob(out + "/*")):
    if os.path.isfile(p) and os.path.getsize(p) < 50_000_000:
        api.upload_file(path_or_fileobj=p, path_in_repo=hf_dir + "/" + os.path.basename(p),
                        repo_id="momom2/wesnoth-model-checkpoints")
print("uploaded", flush=True)
EOF
touch "$OUT/ALL_DONE"
echo WORKER_PROFILE_DONE
