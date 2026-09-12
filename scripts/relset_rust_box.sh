#!/usr/bin/env bash
# The relevant-set basis on the Rust path, certified and timed
# (2026-09-12, user order "complete the Rust port, including for eval
# and pool"). Unattended through HF (scripts/rent_box.py create
# --onstart relset_rust_box.sh):
#   1. stage the code tarball STAGE and the reference checkpoint;
#   2. build the wheel (phase 5 kernels);
#   3. the certification suites (relevant set, observation,
#      enumeration, encoding, seam, priors);
#   4. the reference player against itself, 40 games, shared
#      inference, the observation kernels off and on (the eval path
#      in the relevant-set basis before and after);
#   5. one game under py-spy per mode (the worker's own Python).
# Records under /workspace/relrust/, uploaded to HF $HF_DIR.
set -uo pipefail
OUT=/workspace/relrust
HF_DIR="${HF_DIR:-tier-b/relset_rust_20260912}"
STAGE="${STAGE:-tier-b/staging/stage_20260912a.tar.gz}"
mkdir -p "$OUT"
cd /workspace
export HF_TOKEN="$(tr -d '\r\n' < /workspace/.hf_token)" HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
python -m pip install -q huggingface_hub psutil py-spy scipy pytest 2>&1 | grep -v "WARNING: Running pip" | tail -1 || true
progress() {
    { date -u; ls "$OUT"; tail -3 "$OUT"/*.log 2>/dev/null | tail -12; } > "$OUT/progress.txt"
    python - <<'EOF' 2>/dev/null
import os
from huggingface_hub import HfApi
HfApi(token=os.environ["HF_TOKEN"]).upload_file(
    path_or_fileobj="/workspace/relrust/progress.txt", path_in_repo=os.environ["HF_DIR"] + "/progress.txt",
    repo_id="momom2/wesnoth-model-checkpoints")
EOF
}
export HF_DIR
if [ ! -d Wesnoth-AI/tools ]; then
python - "$STAGE" <<'EOF'
import os, sys, tarfile
from huggingface_hub import hf_hub_download
p = hf_hub_download("momom2/wesnoth-model-checkpoints", sys.argv[1])
os.makedirs("/workspace/Wesnoth-AI", exist_ok=True)
with tarfile.open(p, "r:gz") as tf:
    tf.extractall("/workspace/Wesnoth-AI")
print("code staged", flush=True)
EOF
fi
cd /workspace/Wesnoth-AI
python - <<'EOF'
import pathlib
from huggingface_hub import hf_hub_download
dst = pathlib.Path("training/checkpoints"); dst.mkdir(parents=True, exist_ok=True)
if not (dst / "relset.pt").exists():
    p = hf_hub_download("momom2/wesnoth-model-checkpoints", "tier-b/seed2_relset_20260911/arm_epoch0.pt")
    (dst / "relset.pt").write_bytes(pathlib.Path(p).read_bytes())
print("relset staged", flush=True)
EOF
progress
# 2. The wheel.
command -v cc >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq gcc) >/dev/null 2>&1 || true
command -v cargo >/dev/null 2>&1 || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1
export PATH="$HOME/.cargo/bin:$PATH"
python -m pip install -q maturin >/dev/null 2>&1
python -m pip install --force-reinstall --no-deps rust/wesnoth_core > "$OUT/build.log" 2>&1
python -c "import wesnoth_core; print('wesnoth_core phase', wesnoth_core.__phase__, sorted(n for n in dir(wesnoth_core) if not n.startswith('_')))" >> "$OUT/build.log" 2>&1 || echo "BUILD FAILED" >> "$OUT/build.log"
tail -3 "$OUT/build.log"
{ nproc --all; cat /sys/fs/cgroup/cpu.max 2>/dev/null; nvidia-smi --query-gpu=name --format=csv,noheader; tail -1 "$OUT/build.log"; } > "$OUT/box.txt" 2>&1
progress
if grep -q "BUILD FAILED" "$OUT/build.log"; then
    python - <<'EOF'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(path_or_fileobj="/workspace/relrust/build.log", path_in_repo=os.environ["HF_DIR"] + "/build.log",
                repo_id="momom2/wesnoth-model-checkpoints")
EOF
    touch "$OUT/ALL_DONE"; echo BUILD_FAILED; exit 1
fi
# 3. Certification.
python -m pytest tests/test_rust_relevant_set.py tests/test_rust_observe.py tests/test_rust_enumerate.py \
    tests/test_rust_encode_raw.py tests/test_rust_reach.py tests/test_enumerate_vectorized.py \
    tests/test_inference_seam.py tests/test_server_priors.py tests/test_visibility.py tests/test_raw_player_compact.py \
    -q -p no:cacheprovider > "$OUT/tests.log" 2>&1
tail -3 "$OUT/tests.log"
progress
# 4. The reference player against itself, kernels off and on.
CKPT=training/checkpoints/relset.pt
timed_match() {                   # timed_match MODE OBSERVE_FLAG
    local mode="$1" flag="$2"
    local dir="$OUT/games_$mode"
    local t0=$(date +%s)
    WESNOTH_RUST_OBSERVE="$flag" python tools/run_elo_batch.py --label-a relset --spec-a "$CKPT" \
        --label-b relset_ref --spec-b "$CKPT" \
        --outdir "$dir" --games 40 --max-extra-games 0 --seed-base 20000 \
        --mcts-sims 0 --raw-temperature-a 0 --raw-temperature-b 0 \
        --persistent-workers --shared-inference --no-infer-compile --device cuda \
        --jobs 20 --inference-max-batch 20 \
        --time-budget-min 40 2>&1 | grep --line-buffered -v "wesnoth_core is not importable" > "$OUT/$mode.log"
    echo "$mode $(( $(date +%s) - t0 )) s $(ls "$dir"/game_*.json 2>/dev/null | wc -l) games" | tee "$OUT/$mode.wall"
    cp "$dir"/.inference_server_*.json "$OUT/$mode.server_stats.json" 2>/dev/null || true
    progress
}
timed_match P 0
timed_match O 1
timed_match P2 0
timed_match O2 1
# 5. One game under py-spy per mode.
for mode in on off; do
    observe=$([ "$mode" = on ] && echo 1 || echo 0)
    setsid bash -c "sleep 100000 | python -u tools/eval_inference_server.py --spec $CKPT --device cuda --window-ms 1.5 --max-batch 4 \
        --stats-out $OUT/prof_server_$mode.json --label prof > $OUT/prof_server_$mode.out 2> $OUT/prof_server_$mode.err" &
    SPID=$!
    for i in $(seq 1 120); do grep -q "^__ADDR__ " "$OUT/prof_server_$mode.out" && break; sleep 2; done
    ADDR=$(grep "^__ADDR__ " "$OUT/prof_server_$mode.out" | head -1 | sed "s/^__ADDR__ //")
    mkdir -p "$OUT/prof_games_$mode"
    WESNOTH_RUST_OBSERVE="$observe" py-spy record --duration 200 --format raw --idle --nonblocking -o "$OUT/pyspy_$mode.txt" -- \
        python tools/elo_eval_game.py relset "$CKPT" relset_ref "$CKPT" 1 20001 "$OUT/prof_games_$mode" \
        --max-turns 200 --mcts-sims 0 --raw-temperature-a 0 --raw-temperature-b 0 \
        --inference-address-a "$ADDR" --inference-address-b "$ADDR" --infer-bf16 --infer-packed-trunk \
        --log-level INFO > "$OUT/prof_game_$mode.log" 2>&1
    python tools/pyspy_summary.py "$OUT/pyspy_$mode.txt" --top 20 > "$OUT/pyspy_$mode.summary.txt" 2>&1 || true
    kill -- -$SPID 2>/dev/null; pkill -f "eval_inference_server.py --spec $CKPT" 2>/dev/null; sleep 2
done
python - <<'EOF'
import glob, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for p in sorted(glob.glob("/workspace/relrust/*")):
    if os.path.isfile(p) and os.path.getsize(p) < 50_000_000:
        api.upload_file(path_or_fileobj=p, path_in_repo=os.environ["HF_DIR"] + "/" + os.path.basename(p),
                        repo_id="momom2/wesnoth-model-checkpoints")
print("uploaded", flush=True)
EOF
touch "$OUT/ALL_DONE"
progress
echo RELSET_RUST_DONE
