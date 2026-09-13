#!/usr/bin/env bash
# Where the pool actor's per-leaf time goes, with the phase-8 wheel
# (docs/box_specs.md "Phase 1's exit" left this unexplained: an actor
# spends 27-58 ms of its own CPU per leaf while the simulator's
# fork+step+encode is under 1 ms).
#
#   1. build the wheel, assert phase 8;
#   2. tools/bench_leaf.py: the components of one leaf expansion;
#   3. py-spy over a whole pool iteration WITH its subprocesses, so the
#      actors' own stacks are sampled, not just the parent's;
#   4. the same pool iteration's rate, for the leaves/s the profile
#      belongs to.
# Everything lands under /workspace/actorprof and is uploaded to HF.
set -uo pipefail
OUT=/workspace/actorprof
HF_DIR="${HF_DIR:-tier-b/actor_profile_20260913}"
STAGE="${STAGE:-tier-b/staging/stage_20260913a.tar.gz}"
ACTORS="${ACTORS:-19}"
GAMES="${GAMES:-19}"
mkdir -p "$OUT"
cd /workspace
export HF_TOKEN="$(tr -d '\r\n' < /workspace/.hf_token)" HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
python -m pip install -q huggingface_hub psutil py-spy scipy pytest 2>&1 | tail -1 || true

upload() {
    python - "$1" <<'PY' 2>/dev/null || true
import os, sys, glob
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for p in sorted(glob.glob(sys.argv[1])):
    if os.path.isfile(p) and os.path.getsize(p) < 50_000_000:
        api.upload_file(path_or_fileobj=p, path_in_repo=os.environ["HF_DIR"] + "/" + os.path.basename(p),
                        repo_id="momom2/wesnoth-model-checkpoints")
PY
}
export HF_DIR

# 1. Code and wheel.
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
python -c "import wesnoth_core; assert wesnoth_core.__phase__ >= 8, wesnoth_core.__phase__; print('wheel phase', wesnoth_core.__phase__)" | tee -a "$OUT/build.log" \
    || { echo BUILD_FAILED | tee -a "$OUT/build.log"; upload "$OUT/*"; touch "$OUT/ALL_DONE"; exit 1; }
{ nproc --all; cat /sys/fs/cgroup/cpu.max 2>/dev/null; nvidia-smi --query-gpu=name --format=csv,noheader; } > "$OUT/box.txt" 2>&1
cat "$OUT/box.txt"
upload "$OUT/*"

# The reference player, for a realistic vocabulary and a real pool run.
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

# The corpus the midgame states come from.
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

# 2. The components of one leaf expansion.
python tools/bench_leaf.py --states 3 --repeat 20 --device cuda --profile \
    --out "$OUT/leaf.json" > "$OUT/leaf.log" 2>&1
tail -40 "$OUT/leaf.log"
upload "$OUT/*"

# 3 + 4. One pool iteration under py-spy, following the actor subprocesses.
t0=$(date +%s)
python tools/bench_pool.py --checkpoint "$CKPT" \
    --actors "$ACTORS" --games "$GAMES" --sims 32 --leaf-batch 16 --max-batch 16 \
    --max-turns 30 --dollars-per-hour "${DPH:-0.335}" --device cuda --iteration-timeout 1500 \
    --server-priors --infer-bf16 --packed-trunk --packed-embed \
    --out "$OUT/pool.json" > "$OUT/pool.log" 2>&1 &
POOL_PID=$!
sleep 75                                  # let the actors reach steady state
py-spy record --pid "$POOL_PID" --subprocesses --duration 180 --rate 120 --idle --nonblocking \
    --format raw -o "$OUT/pyspy_pool.txt" > "$OUT/pyspy.log" 2>&1
wait "$POOL_PID"
echo "pool wall $(( $(date +%s) - t0 )) s" | tee "$OUT/pool.wall"
python tools/pyspy_summary.py "$OUT/pyspy_pool.txt" --top 40 > "$OUT/pyspy_pool.summary.txt" 2>&1 || true
head -45 "$OUT/pyspy_pool.summary.txt"
python - <<'PY' 2>/dev/null || true
import json
d = json.load(open("/workspace/actorprof/pool.json"))
print({k: d.get(k) for k in ("leaves_per_s", "saturated_leaves_per_s", "games_per_hour",
                             "games_per_dollar", "tokens_per_leaf", "games_completed")})
PY
upload "$OUT/*"
touch "$OUT/ALL_DONE"
echo ACTOR_PROFILE_DONE
