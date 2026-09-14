#!/usr/bin/env bash
# Two measurements the levers left after phase 1 rest on, on one 4090:
#
#   1. tools/bench_serve_graph.py: where a serve batch's 16 ms goes --
#      kernel launches against device busy time, the seam's stage
#      seconds, and the forward eager against the same forward replayed
#      from a CUDA graph (what a static-shape serve path would leave).
#   2. tools/bench_train_step.py on PRODUCTION's experience format:
#      `--source pool` runs one real pool iteration, whose experiences
#      carry the actor's packed masks (commit 5c877f8), against the
#      bench-state source that rebuilds them on the host. The 26.6 ms
#      policy-loss figure on record (docs/box_specs.md "Training path
#      cost") was measured on the latter, before the masks shipped.
#   3. The static-shape serve path (wesnoth_ai/graphed_serve.py) on both
#      production paths, one factor each: a 40-game raw:t0 match of
#      relset against itself through the shared server with and
#      without --graphed-serve (walls, the server's counters, the
#      graphed summary in its stats file), and one pool iteration of
#      48 actors and games with and without it (the committed bf16
#      packed configuration).
#
# Expects /workspace/.hf_token (chmod 600). Records under
# /workspace/servegraph, uploaded to HF $HF_DIR after each step.
set -uo pipefail
[ -x /venv/main/bin/python ] && export PATH=/venv/main/bin:$PATH
OUT=/workspace/servegraph
HF_DIR="${HF_DIR:-tier-b/serve_graph_20260914}"
STAGE="${STAGE:-tier-b/staging/stage_20260914c.tar.gz}"
GAMES="${GAMES:-48}"
SIMS="${SIMS:-32}"
DPH="${DPH:-0.37}"
SKIP_MICRO="${SKIP_MICRO:-0}"       # 1: skip the serve microbenchmark
SKIP_TRAIN="${SKIP_TRAIN:-0}"       # 1: skip the two trainer benches
SKIP_AB="${SKIP_AB:-0}"             # 1: skip the eval and pool A/B arms
mkdir -p "$OUT"
cd /workspace
export HF_TOKEN="$(tr -d '\r\n' < /workspace/.hf_token)" HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export HF_DIR
python -m pip install -q huggingface_hub psutil scipy pytest 2>&1 | tail -1 || true

upload() {
    python - <<'PY' 2>/dev/null || true
import os, glob
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for p in sorted(glob.glob("/workspace/servegraph/*")):
    if os.path.isfile(p) and os.path.getsize(p) < 50_000_000:
        api.upload_file(path_or_fileobj=p,
                        path_in_repo=os.environ["HF_DIR"] + "/" + os.path.basename(p),
                        repo_id="momom2/wesnoth-model-checkpoints")
PY
}

# ---- code, wheel, box facts -----------------------------------------
rm -rf Wesnoth-AI && mkdir -p Wesnoth-AI
python - "$STAGE" <<'PY'
import sys, tarfile
from huggingface_hub import hf_hub_download
p = hf_hub_download("momom2/wesnoth-model-checkpoints", sys.argv[1])
with tarfile.open(p, "r:gz") as tf:
    tf.extractall("/workspace/Wesnoth-AI")
print("code staged")
PY
cd /workspace/Wesnoth-AI
command -v cc >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq gcc) >/dev/null 2>&1 || true
command -v cargo >/dev/null 2>&1 || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1
export PATH="$HOME/.cargo/bin:$PATH"
python -m pip install -q maturin >/dev/null 2>&1
touch rust/wesnoth_core/src/*.rs
python -m pip install --force-reinstall --no-deps rust/wesnoth_core > "$OUT/build.log" 2>&1
python -c "import wesnoth_core; p = wesnoth_core.__phase__; print('wheel phase', p); \
assert p >= 10, f'wheel is phase {p}; the source declares 10'" | tee -a "$OUT/build.log" \
    || { echo BUILD_FAILED | tee -a "$OUT/build.log"; upload; touch "$OUT/ALL_DONE"; exit 1; }
{ nproc --all; cat /sys/fs/cgroup/cpu.max 2>/dev/null; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
  free -m | head -2; python -c "import torch; print('torch', torch.__version__)"; } > "$OUT/box.txt" 2>&1
cat "$OUT/box.txt"
upload

CKPT=training/checkpoints/relset.pt
[ -f "$CKPT" ] || python - <<'PY'
import os, pathlib
from huggingface_hub import hf_hub_download
dst = pathlib.Path("training/checkpoints"); dst.mkdir(parents=True, exist_ok=True)
src = hf_hub_download("momom2/wesnoth-model-checkpoints", "tier-b/seed2_relset_20260911/arm_epoch0.pt",
                      token=os.environ["HF_TOKEN"])
(dst / "relset.pt").write_bytes(pathlib.Path(src).read_bytes())
print("reference player staged")
PY
if [ ! -d replays_dataset_imitation ]; then
    python - <<'PY'
import pathlib, tarfile
from huggingface_hub import hf_hub_download
p = hf_hub_download("momom2/wesnoth-model-checkpoints",
                    "tier-b/replays_dataset_imitation_dedup_20260908.tar.gz")
with tarfile.open(p, "r:gz") as tf:
    tf.extractall(".")
print("corpus staged", len(list(pathlib.Path("replays_dataset_imitation").glob("*.json.gz"))))
PY
fi

# ---- 1. the serve batch, eager and graphed ---------------------------
if [ "$SKIP_MICRO" != "1" ]; then
python tools/bench_serve_graph.py --checkpoint "$CKPT" --device cuda \
    --batch-sizes 8,16 --states 32 --repeats 50 \
    --out "$OUT/serve_graph.json" > "$OUT/serve_graph.log" 2>&1
tail -12 "$OUT/serve_graph.log"
upload
fi

# ---- 2. the trainer on production's experiences ----------------------
if [ "$SKIP_TRAIN" != "1" ]; then
python tools/bench_train_step.py --checkpoint "$CKPT" --device cuda \
    --source pool --pool-actors 16 --pool-games 16 --pool-max-turns 12 --pool-timeout 900 \
    --precisions bf16 --batch-sizes 16 --n-list 1024 --repeats 2 \
    --experiences-out "$OUT/pool_experiences.pkl" \
    --out "$OUT/train_pool.json" --md "$OUT/train_pool.md" > "$OUT/train_pool.log" 2>&1
tail -25 "$OUT/train_pool.log"
python - <<'PY' | tee "$OUT/train_pool_masks.txt"
import pickle
exps = pickle.load(open("/workspace/servegraph/pool_experiences.pkl", "rb"))
with_masks = sum(1 for e in exps if getattr(e, "masks", None) is not None)
print(f"pool experiences {len(exps)}, with shipped masks {with_masks}")
PY
rm -f "$OUT/pool_experiences.pkl"
upload
python tools/bench_train_step.py --checkpoint "$CKPT" --device cuda \
    --source bench --states 200 \
    --precisions bf16 --batch-sizes 16 --n-list 1024 --repeats 2 \
    --out "$OUT/train_bench.json" --md "$OUT/train_bench.md" > "$OUT/train_bench.log" 2>&1
tail -25 "$OUT/train_bench.log"
upload
fi

if [ "$SKIP_AB" != "1" ]; then
# ---- 3a. the eval path with and without the graphed server ----------
eval_arm() {                     # eval_arm NAME [--graphed-serve]
    local name="$1"; shift
    local dir="$OUT/eval_games_$name"
    local t0
    t0=$(date +%s)
    python tools/run_elo_batch.py --label-a relset --spec-a "$CKPT" \
        --label-b relset_ref --spec-b "$CKPT" \
        --outdir "$dir" --games 40 --max-extra-games 0 --seed-base 20000 \
        --mcts-sims 0 --raw-temperature-a 0 --raw-temperature-b 0 \
        --persistent-workers --shared-inference --no-infer-compile --device cuda \
        --jobs 20 --inference-max-batch 20 "$@" \
        --time-budget-min 30 2>&1 | grep --line-buffered -v "wesnoth_core is not importable" > "$OUT/eval_$name.log"
    echo "$name $* $(( $(date +%s) - t0 )) s $(ls "$dir"/game_*.json 2>/dev/null | wc -l) games" | tee -a "$OUT/eval.walls"
    mkdir -p "$OUT/eval_stats_$name"
    cp "$dir"/.inference_server_*.json "$OUT/eval_stats_$name/" 2>/dev/null || true
    grep -h "inference server .*requests in .*batches\|graphed serve" "$OUT/eval_$name.log" | tail -3 | tee -a "$OUT/eval.counters"
}
eval_arm eager_a
eval_arm graphed_a --graphed-serve
eval_arm eager_b
eval_arm graphed_b --graphed-serve
# With the server this fast it idles a third of its wall waiting for
# the workers' requests: do more workers pay now? (They did not while
# the server was the bound, docs/box_specs.md 2026-09-13.)
eval_arm graphed_j28 --graphed-serve --jobs 28 --inference-max-batch 28
tar czf "$OUT/eval_stats.tar.gz" -C "$OUT" $(cd "$OUT" && ls -d eval_stats_* 2>/dev/null) 2>/dev/null || true
upload

# ---- 3b. the pool with and without the graphed server ---------------
for arm in eager graphed; do
    extra=""
    [ "$arm" = graphed ] && extra="--graphed-serve"
    python tools/bench_pool.py --checkpoint "$CKPT" \
        --actors "$GAMES" --games "$GAMES" --sims "$SIMS" \
        --leaf-batch 16 --max-turns 30 \
        --dollars-per-hour "$DPH" --server-priors \
        --infer-bf16 --packed-trunk --packed-embed $extra \
        > "$OUT/pool_$arm.json" 2> "$OUT/pool_$arm.log"
    tail -4 "$OUT/pool_$arm.log"
    upload
done
python - <<'PY' | tee "$OUT/pool_verdict.txt"
import json
def rd(n):
    try:
        return json.load(open(f"/workspace/servegraph/pool_{n}.json"))
    except Exception:
        return None
a, b = rd("eager"), rd("graphed")
if not (a and b):
    print("one arm missing; no verdict")
else:
    for k in ("leaves_per_s", "saturated_leaves_per_s", "games_per_dollar"):
        x, y = a.get(k), b.get(k)
        if x and y:
            print(f"{k:26s} eager {x:10.1f}   graphed {y:10.1f}   {y / x:.2f}x")
    print("graphed summary:", b.get("graphed_serve_summary"))
PY
upload
fi
touch "$OUT/ALL_DONE"
upload
echo SERVE_GRAPH_DONE
