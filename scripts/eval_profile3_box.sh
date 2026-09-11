#!/usr/bin/env bash
# The observation kernel measured (plan 1.2, pre-registered in BACKLOG
# 2026-09-11): the same 40-game raw:t0 match (seed2 against itself,
# seed base 20000, no replacements; shared inference, packed embed,
# the Rust core, 20 workers) with the kernel off and on. One factor:
# WESNOTH_RUST_OBSERVE=0 forces the Python passes in the same code.
#   P   observation off
#   O   observation on
# Unattended through HF (scripts/seed2_onstart.sh pattern); records
# under /workspace/evalprof3/, uploaded to tier-b/eval_profile3_20260911/.
set -uo pipefail
OUT=/workspace/evalprof3
mkdir -p "$OUT"
cd /workspace
export HF_TOKEN="$(tr -d '\r\n' < /workspace/.hf_token)" HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
python -m pip install -q huggingface_hub psutil 2>&1 | grep -v "WARNING: Running pip" | tail -1 || true
STAGE="${STAGE:-tier-b/staging/stage_20260911c.tar.gz}"
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
if not (dst / "seed2.pt").exists():
    p = hf_hub_download("momom2/wesnoth-model-checkpoints", "tier-b/seed2.pt")
    (dst / "seed2.pt").write_bytes(pathlib.Path(p).read_bytes())
print("seed2 staged", flush=True)
EOF
if ! python -c "import wesnoth_core; assert wesnoth_core.__phase__ >= 4" 2>/dev/null; then
    command -v cc >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq gcc) >/dev/null 2>&1 || true
    command -v cargo >/dev/null 2>&1 || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1
    export PATH="$HOME/.cargo/bin:$PATH"
    python -m pip install -q maturin >/dev/null 2>&1
    python -m pip install -q --force-reinstall --no-deps rust/wesnoth_core 2>&1 | tail -1
fi
python -c "import wesnoth_core; print('wesnoth_core phase', wesnoth_core.__phase__)" > "$OUT/rust.txt" 2>&1 || echo "wesnoth_core build FAILED" | tee -a "$OUT/rust.txt"
{ nproc --all; cat /sys/fs/cgroup/cpu.max 2>/dev/null; nvidia-smi --query-gpu=name --format=csv,noheader; cat "$OUT/rust.txt"; } > "$OUT/box.txt" 2>&1

CKPT=training/checkpoints/seed2.pt
GAMES="${GAMES:-40}"
JOBS="${JOBS:-20}"
progress() {
    { date -u; for m in P O P2 O2; do [ -f "$OUT/$m.wall" ] && echo "$m $(cat "$OUT/$m.wall") s"; done;
      cat "$OUT/rust.txt"; tail -2 "$OUT"/*.log 2>/dev/null | tail -8; } > "$OUT/progress.txt"
    python - <<'EOF'
import os
from huggingface_hub import HfApi
HfApi(token=os.environ["HF_TOKEN"]).upload_file(
    path_or_fileobj="/workspace/evalprof3/progress.txt", path_in_repo="tier-b/eval_profile3_20260911/progress.txt",
    repo_id="momom2/wesnoth-model-checkpoints")
EOF
}
timed_match() {                   # timed_match MODE OBSERVE_FLAG
    local mode="$1" flag="$2"
    if [ -f "$OUT/$mode.wall" ]; then echo "mode $mode already done"; return 0; fi
    local dir="$OUT/games_$mode"
    local t0=$(date +%s)
    WESNOTH_RUST_OBSERVE="$flag" python tools/run_elo_batch.py --label-a seed2 --spec-a "$CKPT" \
        --label-b seed2_ref --spec-b "$CKPT" \
        --outdir "$dir" --games "$GAMES" --max-extra-games 0 --seed-base 20000 \
        --mcts-sims 0 --raw-temperature-a 0 --raw-temperature-b 0 \
        --persistent-workers --shared-inference --no-infer-compile --device cuda \
        --jobs "$JOBS" --inference-max-batch "$JOBS" \
        --time-budget-min 40 2>&1 | grep --line-buffered -v "wesnoth_core is not importable" > "$OUT/$mode.log"
    local t1=$(date +%s)
    echo $((t1 - t0)) > "$OUT/$mode.wall"
    ls "$dir"/game_*.json 2>/dev/null | wc -l > "$OUT/$mode.games"
    cp "$dir"/.inference_server_*.json "$OUT/$mode.server_stats.json" 2>/dev/null || true
    echo "mode $mode: $((t1 - t0)) s for $(cat "$OUT/$mode.games") games"
    progress
}
progress
timed_match P 0
timed_match O 1
timed_match P2 0
timed_match O2 1

python - <<'EOF'
import glob, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for p in sorted(glob.glob("/workspace/evalprof3/*")):
    if os.path.isfile(p) and os.path.getsize(p) < 50_000_000:
        api.upload_file(path_or_fileobj=p, path_in_repo="tier-b/eval_profile3_20260911/" + os.path.basename(p),
                        repo_id="momom2/wesnoth-model-checkpoints")
print("uploaded", flush=True)
EOF
touch "$OUT/ALL_DONE"
progress
echo EVAL_PROFILE3_DONE
