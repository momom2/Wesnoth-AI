#!/usr/bin/env bash
# Phase 1's exit measurement (docs/plan_20260904.md 4): the pool's
# generation rate and searched games per dollar in the reference
# player's relevant-set basis, the same on the full-board twin so the
# basis multiplier is measured on ONE box, and the per-call costs step
# 1.2 owes (step, fork, encode; Python against the Rust-owned state).
#
# Arms, in order (each writes its own JSON under $OUT):
#   micro      tools/bench_core.py: fork / step / encode per call
#   smoke      the MCTS self-play guard with the Rust-owned state on
#   rel_py     pool, relevant-set basis, Python state of record
#   rel_core   pool, relevant-set basis, Rust-owned state
#   full_py    pool, full-board twin (seed2 at one pass), Python state
#
#   OUT=/workspace/phase1 DPH=0.16 ACTORS=19 bash scripts/phase1_exit_box.sh
set -uo pipefail
OUT="${OUT:-/workspace/phase1}"
DPH="${DPH:-0.16}"                 # the box's price per hour
ACTORS="${ACTORS:-19}"
GAMES="${GAMES:-19}"
SIMS="${SIMS:-32}"
LEAF_BATCH="${LEAF_BATCH:-16}"
MAX_TURNS="${MAX_TURNS:-30}"
REL="${REL:-training/checkpoints/relset.pt}"
FULL="${FULL:-training/checkpoints/seed2_e0.pt}"
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
python -c "import wesnoth_core; assert wesnoth_core.__phase__ >= 8, wesnoth_core.__phase__" \
    || { echo "wheel predates phase 8"; exit 1; }
{ date -u; nproc --all; cat /sys/fs/cgroup/cpu.max 2>/dev/null; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } > "$OUT/box.txt" 2>&1
cat "$OUT/box.txt"

# The full-board twin of the reference player (seed2 at one pass).
if [ ! -f "$FULL" ]; then
    HF_TOKEN="$(tr -d '\r\n' < /workspace/.hf_token)" python - "$FULL" <<'PY'
import os, sys, pathlib
from huggingface_hub import hf_hub_download
dst = pathlib.Path(sys.argv[1]); dst.parent.mkdir(parents=True, exist_ok=True)
src = hf_hub_download("momom2/wesnoth-model-checkpoints",
                      "tier-b/clean_seed_20260909/arm_epoch0.pt",
                      token=os.environ["HF_TOKEN"])
dst.write_bytes(pathlib.Path(src).read_bytes())
print("full-board twin staged", dst.stat().st_size)
PY
fi

# 1. Per-call costs (plan 1.2's acceptance).
python tools/bench_core.py --states 8 --repeat 40 --out "$OUT/micro.json" > "$OUT/micro.log" 2>&1
tail -20 "$OUT/micro.log"

# 2. The search guard with the Rust-owned state on: a crash here is
# cheaper to find than inside a pool arm.
WESNOTH_RUST_CORE=1 timeout 1200 python -m pytest tests/test_sim_self_play_smoke.py tests/test_actor_pool_smoke.py tests/test_mcts.py \
    tests/test_sim_determinism.py tests/test_fork_isolation.py tests/test_game_core.py -q -p no:cacheprovider -m "" \
    > "$OUT/smoke_core.log" 2>&1
echo "smoke(core on): $(tail -1 "$OUT/smoke_core.log")"

pool_arm() {                       # pool_arm NAME CHECKPOINT CORE_FLAG
    local name="$1"
    local ckpt="$2"
    local flag="$3"
    local t0
    t0=$(date +%s)
    WESNOTH_RUST_CORE="$flag" timeout 2400 python tools/bench_pool.py --checkpoint "$ckpt" \
        --actors "$ACTORS" --games "$GAMES" --sims "$SIMS" --leaf-batch "$LEAF_BATCH" \
        --max-batch "$LEAF_BATCH" --max-turns "$MAX_TURNS" --dollars-per-hour "$DPH" \
        --server-priors --infer-bf16 --packed-trunk --packed-embed \
        --device cuda --out "$OUT/pool_$name.json" > "$OUT/pool_$name.log" 2>&1
    echo "$name wall $(( $(date +%s) - t0 )) s"
    python - "$OUT/pool_$name.json" "$name" <<'PY' 2>/dev/null || tail -5 "$OUT/pool_$name.log"
import json, sys
d = json.load(open(sys.argv[1]))
keys = ("leaves_per_s", "saturated_leaves_per_s", "games_per_hour", "games_per_dollar",
        "tokens_per_leaf", "forwards", "decisions", "gen_seconds", "games_completed")
print(sys.argv[2], {k: (round(v, 1) if isinstance(v, float) else v)
                    for k, v in ((k, d.get(k)) for k in keys) if v is not None})
PY
}
pool_arm rel_py "$REL" 0
pool_arm rel_core "$REL" 1
pool_arm full_py "$FULL" 0

echo "--- phase 1 exit"
for f in "$OUT"/pool_*.json; do
    python - "$f" <<'PY' 2>/dev/null
import json, os, sys
d = json.load(open(sys.argv[1]))
print(os.path.basename(sys.argv[1]),
      "leaves/s", round(d.get("leaves_per_s") or 0, 1),
      "saturated", round(d.get("saturated_leaves_per_s") or 0, 1),
      "games/h", round(d.get("games_per_hour") or 0, 1),
      "games/$", round(d.get("games_per_dollar") or 0, 1),
      "tokens/leaf", round(d.get("tokens_per_leaf") or 0, 1))
PY
done
touch "$OUT/ALL_DONE"
echo PHASE1_EXIT_DONE
