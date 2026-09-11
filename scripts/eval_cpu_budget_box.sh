#!/usr/bin/env bash
# Is the crowded eval box CPU-bound? (plan 1.2 follow-up, 2026-09-11)
# The same 40-game shared-inference match as eval_profile3_box.sh, once
# per observation-kernel mode, bracketed by the container cgroup's
# cpu.stat (usage_usec = CPU seconds actually consumed, throttled_usec
# = time the quota held us back) and the server stats. Records under
# /workspace/evalcpu/, uploaded to HF tier-b/eval_cpu_20260911/.
set -uo pipefail
OUT=/workspace/evalcpu
mkdir -p "$OUT"
cd /workspace/Wesnoth-AI
export HF_TOKEN="$(tr -d '\r\n' < /workspace/.hf_token)" HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 PATH="$HOME/.cargo/bin:$PATH"
CKPT=training/checkpoints/seed2.pt
GAMES="${GAMES:-40}"
JOBS="${JOBS:-20}"
{ nproc --all; cat /sys/fs/cgroup/cpu.max 2>/dev/null; nvidia-smi --query-gpu=name --format=csv,noheader; } > "$OUT/box.txt" 2>&1
cpustat() { grep -E "^(usage_usec|user_usec|system_usec|nr_throttled|throttled_usec)" /sys/fs/cgroup/cpu.stat 2>/dev/null | tr "\n" " "; echo; }
timed_match() {                   # timed_match MODE OBSERVE_FLAG
    local mode="$1" flag="$2"
    local dir="$OUT/games_$mode"
    echo "$mode before $(cpustat)" >> "$OUT/cpu.txt"
    local t0=$(date +%s.%N)
    WESNOTH_RUST_OBSERVE="$flag" python tools/run_elo_batch.py --label-a seed2 --spec-a "$CKPT" \
        --label-b seed2_ref --spec-b "$CKPT" \
        --outdir "$dir" --games "$GAMES" --max-extra-games 0 --seed-base 20000 \
        --mcts-sims 0 --raw-temperature-a 0 --raw-temperature-b 0 \
        --persistent-workers --shared-inference --no-infer-compile --device cuda \
        --jobs "$JOBS" --inference-max-batch "$JOBS" \
        --time-budget-min 40 2>&1 | grep --line-buffered -v "wesnoth_core is not importable" > "$OUT/$mode.log"
    local t1=$(date +%s.%N)
    echo "$mode after  $(cpustat)" >> "$OUT/cpu.txt"
    echo "$mode wall $(python -c "print(round($t1 - $t0, 1))") s" >> "$OUT/cpu.txt"
    cp "$dir"/.inference_server_*.json "$OUT/$mode.server_stats.json" 2>/dev/null || true
}
timed_match P 0
timed_match O 1
cat "$OUT/cpu.txt"
python - <<'EOF'
import glob, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for p in sorted(glob.glob("/workspace/evalcpu/*")):
    if os.path.isfile(p) and os.path.getsize(p) < 50_000_000:
        api.upload_file(path_or_fileobj=p, path_in_repo="tier-b/eval_cpu_20260911/" + os.path.basename(p),
                        repo_id="momom2/wesnoth-model-checkpoints")
print("uploaded", flush=True)
EOF
touch "$OUT/ALL_DONE"
echo EVAL_CPU_DONE
