#!/usr/bin/env bash
# Eval-path throughput, round 2 (plan 1.5, 2026-09-11): the eval
# server's host-side levers, one at a time, on the same 40-game
# raw:t0 match (seed against itself, seed base 20000, no replacements,
# shared inference, the Rust core, 20 workers):
#   F0  packed embed off (round 1's mode F)
#   H   packed embed on (the pool's default; -4.9 ms host per batch)
#   I   H + the compiled packed layer loop (--compile-packed)
# Unattended: fetched and run by the box's onstart through HF
# (scripts/seed2_onstart.sh pattern). Records under /workspace/evalprof2/,
# uploaded to HF tier-b/eval_profile2_20260911/ at the end.
set -uo pipefail
OUT=/workspace/evalprof2
mkdir -p "$OUT"
cd /workspace
export HF_TOKEN="$(tr -d '\r\n' < /workspace/.hf_token)" HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
python -m pip install -q huggingface_hub psutil 2>&1 | grep -v "WARNING: Running pip" | tail -1 || true
if [ ! -d Wesnoth-AI/tools ]; then
python - <<'EOF'
import shutil, tarfile
from huggingface_hub import hf_hub_download
p = hf_hub_download("momom2/wesnoth-model-checkpoints", "tier-b/staging/stage_20260911b.tar.gz")
import os; os.makedirs("/workspace/Wesnoth-AI", exist_ok=True)
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
if not (dst / "seed.pt").exists():
    p = hf_hub_download("momom2/wesnoth-model-checkpoints", "tier-b/a3/seed_imit_tierb_start.pt")
    (dst / "seed.pt").write_bytes(pathlib.Path(p).read_bytes())
print("seed staged", flush=True)
EOF
if ! python -c "import wesnoth_core" 2>/dev/null; then
    command -v cc >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq gcc) >/dev/null 2>&1 || true
    command -v cargo >/dev/null 2>&1 || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1
    export PATH="$HOME/.cargo/bin:$PATH"
    python -m pip install -q maturin >/dev/null 2>&1
    python -m pip install -q rust/wesnoth_core 2>&1 | tail -1
fi
python -c "import wesnoth_core; print('wesnoth_core built')" > "$OUT/rust.txt" 2>&1 || echo "wesnoth_core build FAILED" | tee -a "$OUT/rust.txt"
{ nproc --all; cat /sys/fs/cgroup/cpu.max 2>/dev/null; nvidia-smi --query-gpu=name --format=csv,noheader; cat "$OUT/rust.txt"; } > "$OUT/box.txt" 2>&1

SEED=training/checkpoints/seed.pt
GAMES="${GAMES:-40}"
JOBS="${JOBS:-20}"
progress() {
    { date -u; ls "$OUT"; for m in F0 H I; do [ -f "$OUT/$m.wall" ] && echo "$m $(cat "$OUT/$m.wall") s"; done;
      tail -3 "$OUT"/*.log 2>/dev/null | tail -12; } > "$OUT/progress.txt"
    python - <<'EOF'
import os
from huggingface_hub import HfApi
HfApi(token=os.environ["HF_TOKEN"]).upload_file(
    path_or_fileobj="/workspace/evalprof2/progress.txt", path_in_repo="tier-b/eval_profile2_20260911/progress.txt",
    repo_id="momom2/wesnoth-model-checkpoints")
EOF
}
timed_match() {                   # timed_match MODE [extra run_elo_batch flags...]
    local mode="$1"; shift
    if [ -f "$OUT/$mode.wall" ]; then echo "mode $mode already done"; return 0; fi
    local dir="$OUT/games_$mode"
    local t0=$(date +%s)
    python tools/run_elo_batch.py --label-a seed --spec-a "$SEED" --label-b seed_ref --spec-b "$SEED" \
        --outdir "$dir" --games "$GAMES" --max-extra-games 0 --seed-base 20000 \
        --mcts-sims 0 --raw-temperature-a 0 --raw-temperature-b 0 \
        --persistent-workers --shared-inference --no-infer-compile --device cuda \
        --jobs "$JOBS" --inference-max-batch "$JOBS" \
        --time-budget-min 40 "$@" 2>&1 | grep --line-buffered -v "wesnoth_core is not importable" > "$OUT/$mode.log"
    local t1=$(date +%s)
    echo $((t1 - t0)) > "$OUT/$mode.wall"
    ls "$dir"/game_*.json 2>/dev/null | wc -l > "$OUT/$mode.games"
    cp "$dir"/.inference_server_*.json "$OUT/$mode.server_stats.json" 2>/dev/null || true
    cp "$dir"/.inference_server_*.log "$OUT/$mode.server.log" 2>/dev/null || true
    echo "mode $mode: $((t1 - t0)) s for $(cat "$OUT/$mode.games") games"
    progress
}
progress
timed_match F0 --no-packed-embed
timed_match H
timed_match I --compile-packed

python - <<'EOF'
import glob, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for p in sorted(glob.glob("/workspace/evalprof2/*")):
    if os.path.isfile(p) and os.path.getsize(p) < 50_000_000:
        api.upload_file(path_or_fileobj=p, path_in_repo="tier-b/eval_profile2_20260911/" + os.path.basename(p),
                        repo_id="momom2/wesnoth-model-checkpoints")
print("uploaded", flush=True)
EOF
touch "$OUT/ALL_DONE"
progress
echo EVAL_PROFILE2_DONE
