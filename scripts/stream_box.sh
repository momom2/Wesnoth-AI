#!/usr/bin/env bash
# Continuous generation against the barrier iteration, on one
# single-tenant 4090 host (docs/continuous_generation_20260918.md,
# "The measurement"). The committed bf16 packed configuration, 32
# evaluations, 48 actors; three arms, twice each, interleaved:
#
#   barrier  bench_pool --actors 48 --games 48                (one iteration)
#   form_a   bench_pool --actors 48 --games 96                (two games per actor)
#   stream   bench_pool --actors 48 --games 48 --stream --rounds 4 --step-seconds 20
#
# Pre-registered (2026-09-18): the number is games per hour, the
# stream's over its windows (each plus the STEP_SECONDS idle that
# stands in for a step, the drain left out), the barrier's over its
# iteration plus the same STEP_SECONDS (its step runs with the actors
# idle). PASS when the stream reads at least 1.25x the barrier in both
# pairs, its straddle mean lies in 0.7-1.5 and no window timed out.
# Otherwise the stream stays opt-in with the reading on record.
#
# Expects /workspace/.hf_token (chmod 600). Records under
# /workspace/stream, uploaded to HF $HF_DIR after each arm.
set -uo pipefail
[ -x /venv/main/bin/python ] && export PATH=/venv/main/bin:$PATH
OUT=/workspace/stream
HF_DIR="${HF_DIR:-tier-b/stream_20260918}"
STAGE="${STAGE:-tier-b/staging/stage_20260918b.tar.gz}"
ACTORS="${ACTORS:-48}"
SIMS="${SIMS:-32}"
DPH="${DPH:-0.40}"
ROUNDS="${ROUNDS:-4}"
STEP_SECONDS="${STEP_SECONDS:-20}"
PAIRS="${PAIRS:-2}"
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
for p in sorted(glob.glob("/workspace/stream/*")):
    if os.path.isfile(p) and os.path.getsize(p) < 50_000_000:
        api.upload_file(path_or_fileobj=p,
                        path_in_repo=os.environ["HF_DIR"] + "/" + os.path.basename(p),
                        repo_id="momom2/wesnoth-model-checkpoints")
PY
}

box_facts() {
    { echo "== $1 $(date -u +%H:%M:%S)"; cat /proc/loadavg;
      nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader;
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
{ echo "cores(all) $(nproc --all)"; grep -m1 "model name" /proc/cpuinfo;
  echo "cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)";
  echo "pids.max $(cat /sys/fs/cgroup/pids.max 2>/dev/null || echo n/a)";
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader;
  free -m | head -2; python -c "import torch; print('torch', torch.__version__)";
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

# ---- the arms, interleaved ------------------------------------------
arm() {                          # arm NAME GAMES [extra bench_pool args]
    local name="$1" games="$2"; shift 2
    local t0
    box_facts "$name before"
    t0=$(date +%s)
    python tools/bench_pool.py --checkpoint "$CKPT" \
        --actors "$ACTORS" --games "$games" --sims "$SIMS" \
        --leaf-batch 16 --max-turns 30 --iteration-timeout 1500 \
        --dollars-per-hour "$DPH" --server-priors \
        --infer-bf16 --packed-trunk --packed-embed "$@" \
        > "$OUT/pool_$name.json" 2> "$OUT/pool_$name.log"
    echo "$name games=$games $* $(( $(date +%s) - t0 )) s" | tee -a "$OUT/pool.walls"
    grep -h "stream window\|stream closed\|serve stages" "$OUT/pool_$name.log" | tail -6 \
        | sed 's/^.*actor_[a-z]* INFO //'
    upload
}
for pair in $(seq 1 "$PAIRS"); do
    arm "barrier_$pair" "$ACTORS"
    arm "stream_$pair" "$ACTORS" --stream --rounds "$ROUNDS" --step-seconds "$STEP_SECONDS"
    arm "forma_$pair" "$(( 2 * ACTORS ))"
done

# ---- the verdict under the pre-registered rule ----------------------
python - "$PAIRS" "$STEP_SECONDS" <<'PY' | tee "$OUT/verdict.txt"
import json, sys
OUT = "/workspace/stream"
pairs = int(sys.argv[1])
step_s = float(sys.argv[2])

def rd(n):
    try:
        return json.load(open(f"{OUT}/pool_{n}.json"))
    except Exception:
        return None

ok_all = True
for p in range(1, pairs + 1):
    b, s, a = rd(f"barrier_{p}"), rd(f"stream_{p}"), rd(f"forma_{p}")
    print(f"pair {p}")
    for name, d in (("barrier", b), ("form A", a), ("stream", s)):
        if not d:
            print(f"  {name:8s} missing"); continue
        print(f"  {name:8s} games {d['games_completed']:4d} in {d['gen_seconds']:6.0f}s  "
              f"games/h {d['games_per_hour']:7.1f}  leaves/s {d['leaves_per_s']:7.0f}  "
              f"saturated {d['saturated_leaves_per_s'] or 0:7.0f}  "
              f"decisive {d['decisive']}  truncated {d['truncated']}")
    if not (b and s):
        ok_all = False; continue
    barrier_gph = 3600.0 * b["games_completed"] / (b["gen_seconds"] + step_s)
    stream_gph = s.get("window_games_per_hour") or 0.0
    ratio = stream_gph / barrier_gph
    forma_gph = (3600.0 * a["games_completed"] / (a["gen_seconds"] + step_s)) if a else None
    print(f"  with a {step_s:.0f}s step per iteration: barrier {barrier_gph:.1f} games/h, "
          f"form A {forma_gph and round(forma_gph, 1)} games/h "
          f"({forma_gph and round(forma_gph / barrier_gph, 3)}x); stream windows "
          f"{stream_gph:.1f} games/h with the same idle")
    rounds = s.get("stream_rounds") or []
    straddle = [r["straddle_mean"] for r in rounds if r.get("straddle_mean") is not None]
    mean_straddle = sum(straddle) / len(straddle) if straddle else None
    timed_out = any(r.get("timed_out") for r in rounds)
    for r in rounds:
        print(f"    window {r['round']}: {r['games']} games {r['seconds']:.0f}s "
              f"{r['leaves_per_s']:.0f} leaves/s saturated {r['saturated_leaves_per_s'] or 0:.0f} "
              f"queue {r['queue_depth'] or 0:.1f} game s p50 {r['game_seconds_p50'] or 0:.0f} "
              f"straddle mean {r['straddle_mean']:.2f} max {r['straddle_max']} "
              f"share {r['straddled_share']:.2f}")
    # the roof: the barrier's saturated rate over its leaves per game
    roof = (b["saturated_leaves_per_s"] * 3600.0 / (b["forwards"] / max(1, b["games_completed"]))
            if b.get("saturated_leaves_per_s") and b.get("forwards") else None)
    ok = (ratio >= 1.25 and mean_straddle is not None and 0.7 <= mean_straddle <= 1.5
          and not timed_out)
    print(f"  stream / barrier games per hour {ratio:.3f}x (>=1.25); straddle mean "
          f"{mean_straddle}; timed out {timed_out}; roof from the barrier's saturated rate "
          f"{roof and round(roof, 1)} games/h -> {'pass' if ok else 'FAIL'}")
    ok_all = ok_all and ok
print("STREAM:", "PASS" if ok_all else "FAIL")
PY
upload
touch "$OUT/ALL_DONE"
upload
echo STREAM_DONE
