#!/usr/bin/env bash
# The serve batch cap: does coalescing up to 64 leaves per serve batch
# (four actor requests) raise the pool's saturated rate, and does a
# 4090 then reach plan 1.3's 3,000 leaves per second? The committed
# bf16 packed configuration, 48 actors and games, 32 evaluations, two
# interleaved pairs of --max-batch 16 against 64, then 96, 64 graphed
# and a smoke of the reference checkpoint at 16, on a single-tenant
# 4090 host. Rule, predictions and cost: docs/serve_batch_prereg_20260920.md.
#
# Expects /workspace/.hf_token (chmod 600). Records under
# /workspace/serve_batch, uploaded to HF $HF_DIR after each arm.
set -uo pipefail
[ -x /venv/main/bin/python ] && export PATH=/venv/main/bin:$PATH
OUT=/workspace/serve_batch
HF_DIR="${HF_DIR:-tier-b/serve_batch_20260920}"
STAGE="${STAGE:-tier-b/staging/stage_20260920a.tar.gz}"
GAMES="${GAMES:-48}"
SIMS="${SIMS:-32}"
DPH="${DPH:-0.47}"
EXTRA_ARMS="${EXTRA_ARMS:-1}"
mkdir -p "$OUT"
cd /workspace
export HF_TOKEN="$(tr -d '\r\n' < /workspace/.hf_token)" HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export HF_DIR
python -m pip install -q huggingface_hub psutil scipy 2>&1 | tail -1 || true

upload() {
    python - <<'PY' 2>/dev/null || true
import os, glob
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for p in sorted(glob.glob("/workspace/serve_batch/*")):
    if os.path.isfile(p) and os.path.getsize(p) < 50_000_000:
        api.upload_file(path_or_fileobj=p,
                        path_in_repo=os.environ["HF_DIR"] + "/" + os.path.basename(p),
                        repo_id="momom2/wesnoth-model-checkpoints")
PY
}

box_facts() {                    # box_facts LABEL: appended to box.txt
    { echo "== $1 $(date -u +%H:%M:%S)"; cat /proc/loadavg;
      nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader;
      free -m | sed -n 2p; } >> "$OUT/box.txt" 2>&1
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
    || { echo BUILD_FAILED | tee -a "$OUT/build.log"; touch "$OUT/ALL_DONE"; upload; exit 1; }
{ echo "cores(nproc) $(nproc)"; echo "cores(all) $(nproc --all)";
  grep -m1 "model name" /proc/cpuinfo; echo "cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)";
  echo "pids.max $(cat /sys/fs/cgroup/pids.max 2>/dev/null || echo n/a)";
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader;
  free -m | head -2; python -c "import torch; print('torch', torch.__version__)";
  git -C /workspace/Wesnoth-AI log -1 --oneline 2>/dev/null || echo "no git";
  python tools/kernel_status.py 2>/dev/null | tail -8; } > "$OUT/box.txt" 2>&1
box_facts idle
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


REF=$(python tools/reference_player.py --ensure 2>/dev/null | tail -1)
echo "reference checkpoint: $REF" | tee -a "$OUT/box.txt"

# ---- the pool arms --------------------------------------------------
pool_arm() {                     # pool_arm NAME CHECKPOINT MAX_BATCH [extra bench_pool flags]
    local name="$1" ckpt="$2" mb="$3"; shift 3
    local t0
    box_facts "pool_$name before"
    t0=$(date +%s)
    python tools/bench_pool.py --checkpoint "$ckpt" \
        --actors "$GAMES" --games "$GAMES" --sims "$SIMS" \
        --leaf-batch 16 --max-batch "$mb" --max-turns 30 \
        --dollars-per-hour "$DPH" --server-priors \
        --infer-bf16 --packed-trunk --packed-embed "$@" \
        > "$OUT/pool_$name.json" 2> "$OUT/pool_$name.log"
    echo "$name max_batch=$mb $* $(( $(date +%s) - t0 )) s" | tee -a "$OUT/pool.walls"
    tail -4 "$OUT/pool_$name.log"
    upload
}
pool_arm b16_a "$CKPT" 16
pool_arm b64_a "$CKPT" 64
pool_arm b16_b "$CKPT" 16
pool_arm b64_b "$CKPT" 64
# The extra arms only when the core pairs say the cap matters: both 64
# arms at least 1.10x their 16 arm on the saturated column. A flat pair
# ends the box here, about $0.30 spent.
pairs_up=$(python - <<'EOF'
import json
def sat(n):
    try:
        return json.load(open(f"/workspace/serve_batch/pool_{n}.json"))["saturated_leaves_per_s"]
    except Exception:
        return 0.0
ok = all(sat(f"b64_{p}") >= 1.10 * sat(f"b16_{p}") > 0 for p in ("a", "b"))
print("1" if ok else "0")
EOF
)
if [ "$EXTRA_ARMS" = "1" ] && [ "$pairs_up" = "1" ]; then
    pool_arm b96_a "$CKPT" 96
    pool_arm b64g_a "$CKPT" 64 --graphed-serve
    pool_arm ref16_a "$REF" 16
else
    echo "extra arms skipped (pairs_up=$pairs_up, EXTRA_ARMS=$EXTRA_ARMS)" | tee -a "$OUT/pool.walls"
fi

# ---- the verdict under the pre-registered rule ----------------------
python - <<'PY' | tee "$OUT/verdict.txt"
import json
OUT = "/workspace/serve_batch"

def rd(n):
    try:
        return json.load(open(f"{OUT}/pool_{n}.json"))
    except Exception:
        return None

import re

def serve_line(n):
    """leaves per batch and device ms per leaf from the pool log's serve-stages line."""
    try:
        text = open(f"{OUT}/pool_{n}.log", encoding="utf-8", errors="replace").read()
    except Exception:
        return None, None
    lb = re.findall(r"leaves/batch=([0-9.]+)", text)
    ms = re.findall(r"\(([0-9.]+) ms/leaf\)", text)
    return (float(lb[-1]) if lb else None), (float(ms[-1]) if ms else None)

arms = ["b16_a", "b64_a", "b16_b", "b64_b", "b96_a", "b64g_a", "ref16_a"]
print("POOL (48 actors and games, 32 sims, bf16 packed, eager unless noted)")
for n in arms:
    r = rd(n)
    if not r:
        print(f"  {n:8s} missing"); continue
    lb, ms = serve_line(n)
    print(f"  {n:8s} max_batch {r.get('max_batch')!s:>3} saturated {r['saturated_leaves_per_s']:8.0f} "
          f"iteration {r['leaves_per_s']:8.0f} games/$ {r.get('games_per_dollar', float('nan')):7.1f} "
          f"leaves/batch {lb} device ms/leaf {ms} queue {r.get('queue_depth')} tokens/leaf {r.get('tokens_per_leaf')}")
b16 = [rd("b16_a"), rd("b16_b")]
quiet = None
if all(b16):
    a, b = (e["saturated_leaves_per_s"] for e in b16)
    spread = abs(a - b) / ((a + b) / 2)
    quiet = spread <= 0.05
    print(f"  b16 saturated repeat: {a:.0f} against {b:.0f}, spread {spread:.1%} -> {'QUIET' if quiet else 'NOISY'}")
default64 = True
for pair in ("a", "b"):
    e, g = rd(f"b16_{pair}"), rd(f"b64_{pair}")
    if not (e and g):
        print(f"  pair {pair}: an arm is missing"); default64 = False; continue
    sat = g["saturated_leaves_per_s"] / e["saturated_leaves_per_s"]
    it = g["leaves_per_s"] / e["leaves_per_s"]
    gpd = g["games_per_dollar"] / e["games_per_dollar"]
    ok = sat >= 1.15 and gpd >= 0.97
    print(f"  pair {pair}: saturated {sat:.3f}x (>=1.15), iteration {it:.3f}x, games/$ {gpd:.3f}x (>=0.97) -> {'pass' if ok else 'FAIL'}")
    default64 = default64 and ok
best = max((rd(n)["saturated_leaves_per_s"] for n in arms if rd(n)), default=0.0)
print(f"DEFAULT 64: {'YES' if (default64 and quiet) else 'NO'} (quiet={quiet})")
print(f"PLAN 1.3 TARGET (3,000 saturated on a 4090): {'MET' if best >= 3000 else 'NOT MET'} (best {best:.0f})")
PY
touch "$OUT/ALL_DONE"
upload
echo SERVE_BATCH_DONE
