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
#
# Expects /workspace/.hf_token (chmod 600). Records under
# /workspace/servegraph, uploaded to HF $HF_DIR after each step.
set -uo pipefail
[ -x /venv/main/bin/python ] && export PATH=/venv/main/bin:$PATH
OUT=/workspace/servegraph
HF_DIR="${HF_DIR:-tier-b/serve_graph_20260914}"
STAGE="${STAGE:-tier-b/staging/stage_20260914b.tar.gz}"
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
python tools/bench_serve_graph.py --checkpoint "$CKPT" --device cuda \
    --batch-sizes 8,16 --states 32 --repeats 50 \
    --out "$OUT/serve_graph.json" > "$OUT/serve_graph.log" 2>&1
tail -12 "$OUT/serve_graph.log"
upload

# ---- 2. the trainer on production's experiences ----------------------
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
touch "$OUT/ALL_DONE"
upload
echo SERVE_GRAPH_DONE
