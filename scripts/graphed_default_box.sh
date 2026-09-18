#!/usr/bin/env bash
# The A/B that decides whether --graphed-serve becomes the default: the
# graphed server (wesnoth_ai/graphed_serve.py) against the eager one on
# a SINGLE-TENANT 4090 host, the committed bf16 packed configuration,
# both production paths, two interleaved pairs each.
#
# The 2026-09-14 pair (docs/box_specs.md "The serve batch is
# launch-bound") read 1.30x saturated on the pool and 27.0 -> 12.7 ms
# of infer per eval batch, on a 128-core shared host whose eager pool
# arms swung 474..1,032 iteration leaves/s across the day. Only the
# saturated column was stable, and only within about 1.3x. A quiet box
# repeats it within 5% ("Phase 1's exit").
#
# Pre-registered rule (2026-09-18), evaluated by the verdict step:
#   QUIET   the two eager pool arms' saturated leaves/s agree within 5%.
#           If not, the box is noisy: no default changes, the pair is
#           reported as noise.
#   POOL ON graphed / eager saturated leaves/s >= 1.15 in BOTH pairs,
#           games per dollar >= 0.97 in both, fallbacks under 5% of
#           served batches, no "graphed serve failed" in the logs.
#   EVAL ON graphed / eager server infer seconds per batch <= 0.75 in
#           BOTH pairs (the server's own counters), the walls not slower
#           in either.
#   Anything else keeps that path's default OFF.
#
# Expects /workspace/.hf_token (chmod 600). Records under
# /workspace/graphed_default, uploaded to HF $HF_DIR after each step.
set -uo pipefail
[ -x /venv/main/bin/python ] && export PATH=/venv/main/bin:$PATH
OUT=/workspace/graphed_default
HF_DIR="${HF_DIR:-tier-b/graphed_default_20260918}"
STAGE="${STAGE:-tier-b/staging/stage_20260918a.tar.gz}"
GAMES="${GAMES:-48}"
SIMS="${SIMS:-32}"
DPH="${DPH:-0.47}"
SKIP_POOL="${SKIP_POOL:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
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
for p in sorted(glob.glob("/workspace/graphed_default/*")):
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
    || { echo BUILD_FAILED | tee -a "$OUT/build.log"; upload; touch "$OUT/ALL_DONE"; exit 1; }
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

# ---- 1. the pool, two interleaved pairs -----------------------------
pool_arm() {                     # pool_arm NAME [--graphed-serve]
    local name="$1"; shift
    local t0
    box_facts "pool_$name before"
    t0=$(date +%s)
    python tools/bench_pool.py --checkpoint "$CKPT" \
        --actors "$GAMES" --games "$GAMES" --sims "$SIMS" \
        --leaf-batch 16 --max-turns 30 \
        --dollars-per-hour "$DPH" --server-priors \
        --infer-bf16 --packed-trunk --packed-embed "$@" \
        > "$OUT/pool_$name.json" 2> "$OUT/pool_$name.log"
    echo "$name $* $(( $(date +%s) - t0 )) s" | tee -a "$OUT/pool.walls"
    tail -4 "$OUT/pool_$name.log"
    upload
}
if [ "$SKIP_POOL" != "1" ]; then
pool_arm eager_a
pool_arm graphed_a --graphed-serve
pool_arm eager_b
pool_arm graphed_b --graphed-serve
fi

# ---- 2. the eval path, two interleaved pairs ------------------------
eval_arm() {                     # eval_arm NAME [--graphed-serve]
    local name="$1"; shift
    local dir="$OUT/eval_games_$name"
    local t0
    box_facts "eval_$name before"
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
    grep -h "inference server .*requests in .*batches\|graphed serve" "$OUT/eval_$name.log" | tail -3 \
        | sed "s/^/$name /" | tee -a "$OUT/eval.counters"
}
if [ "$SKIP_EVAL" != "1" ]; then
eval_arm eager_a
eval_arm graphed_a --graphed-serve
eval_arm eager_b
eval_arm graphed_b --graphed-serve
tar czf "$OUT/eval_stats.tar.gz" -C "$OUT" $(cd "$OUT" && ls -d eval_stats_* 2>/dev/null) 2>/dev/null || true
upload
fi

# ---- 3. the verdict under the pre-registered rule --------------------
python - <<'PY' | tee "$OUT/verdict.txt"
import json, re, os
OUT = "/workspace/graphed_default"

def rd(n):
    try:
        return json.load(open(f"{OUT}/pool_{n}.json"))
    except Exception:
        return None

def disabled(n):
    try:
        return "graphed serve failed" in open(f"{OUT}/pool_{n}.log").read()
    except Exception:
        return False

print("POOL (48 actors and games, 32 sims, bf16 packed)")
pool_on, quiet = True, True
eager = [rd("eager_a"), rd("eager_b")]
if all(eager):
    a, b = (e["saturated_leaves_per_s"] for e in eager)
    spread = abs(a - b) / ((a + b) / 2)
    quiet = spread <= 0.05
    print(f"  eager saturated repeat: {a:.0f} against {b:.0f}, spread {spread:.1%} -> {'QUIET' if quiet else 'NOISY'}")
for pair in ("a", "b"):
    e, g = rd(f"eager_{pair}"), rd(f"graphed_{pair}")
    if not (e and g):
        print(f"  pair {pair}: an arm is missing"); pool_on = False; continue
    for k in ("leaves_per_s", "saturated_leaves_per_s", "games_per_dollar"):
        print(f"  pair {pair} {k:24s} eager {e[k]:9.1f}  graphed {g[k]:9.1f}  {g[k] / e[k]:.3f}x")
    sat = g["saturated_leaves_per_s"] / e["saturated_leaves_per_s"]
    gpd = g["games_per_dollar"] / e["games_per_dollar"]
    s = g.get("graphed_serve_summary") or {}
    fb = sum((s.get("fallbacks") or {}).values())
    served = s.get("served") or 0
    fb_frac = fb / served if served else 1.0
    print(f"  pair {pair} graphed: served {served}, fallbacks {fb} ({fb_frac:.1%}), capture_s {s.get('capture_s')}, "
          f"graphs {s.get('graphs')}, disabled line {disabled(f'graphed_{pair}')}")
    ok = sat >= 1.15 and gpd >= 0.97 and fb_frac < 0.05 and s.get("graphs") is True and not disabled(f"graphed_{pair}")
    print(f"  pair {pair}: saturated {sat:.3f}x (>=1.15), games/$ {gpd:.3f}x (>=0.97) -> {'pass' if ok else 'FAIL'}")
    pool_on = pool_on and ok
print(f"POOL default: {'ON' if (pool_on and quiet) else 'OFF'}"
      + ("" if quiet else " (box noisy; the pair is not evidence)"))

print("EVAL (40-game raw:t0 relset against itself, 20 workers)")
counters = {}
try:
    for line in open(f"{OUT}/eval.counters"):
        m = re.match(r"(\S+) .*?(\d+) requests in (\d+) batches.*?infer ([\d.]+)s", line)
        if m:                                   # the biggest server of the arm is the match's
            name, row = m.group(1), (int(m.group(2)), int(m.group(3)), float(m.group(4)))
            if name not in counters or row[0] > counters[name][0]:
                counters[name] = row
except Exception:
    pass
walls = {}
try:
    for line in open(f"{OUT}/eval.walls"):
        parts = line.split()
        walls[parts[0]] = (int(parts[-4]), int(parts[-2]))
except Exception:
    pass
eval_on = bool(counters)
for pair in ("a", "b"):
    e, g = counters.get(f"eager_{pair}"), counters.get(f"graphed_{pair}")
    we, wg = walls.get(f"eager_{pair}"), walls.get(f"graphed_{pair}")
    if not (e and g and we and wg):
        print(f"  pair {pair}: a record is missing"); eval_on = False; continue
    pe, pg = 1000 * e[2] / e[1], 1000 * g[2] / g[1]
    ok = pg / pe <= 0.75 and wg[0] <= we[0] and wg[1] == 40 and we[1] == 40
    print(f"  pair {pair} infer per batch eager {pe:.1f} ms ({e[1]} batches, mean {e[0] / e[1]:.1f}) "
          f"graphed {pg:.1f} ms ({g[1]} batches, mean {g[0] / g[1]:.1f})  {pg / pe:.3f}x (<=0.75); "
          f"walls {we[0]} -> {wg[0]} s, games {we[1]}/{wg[1]} -> {'pass' if ok else 'FAIL'}")
    eval_on = eval_on and ok
print(f"EVAL default: {'ON' if eval_on else 'OFF'}")
PY
upload
touch "$OUT/ALL_DONE"
upload
echo GRAPHED_DEFAULT_DONE
