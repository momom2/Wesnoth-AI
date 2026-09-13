#!/usr/bin/env bash
# Verification round for the changes that can only run where the Rust
# wheel builds (the laptop cannot execute freshly built binaries):
#   1. build the wheel, assert the phase;
#   2. the suites that exercise the Rust-owned state, with the core as
#      the state of record AND with the Python state, since the recruit
#      rejection and the movement-class key are about the difference;
#   3. tools/diff_core.py over a corpus slice: the movement-class cache
#      is now keyed on the defense table's CONTENT, and a mistake there
#      shows up as a wrong movement cost, which the sweep catches.
# Records under /workspace/verify, uploaded to HF.
set -uo pipefail
OUT=/workspace/verify
HF_DIR="${HF_DIR:-tier-b/verify_20260913}"
STAGE="${STAGE:-tier-b/staging/stage_20260913b.tar.gz}"
REPLAYS="${REPLAYS:-400}"
mkdir -p "$OUT"
cd /workspace
export HF_TOKEN="$(tr -d '\r\n' < /workspace/.hf_token)" HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
python -m pip install -q huggingface_hub pytest psutil scipy 2>&1 | tail -1 || true

upload() {
    python - <<'PY' 2>/dev/null || true
import os, glob
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for p in sorted(glob.glob("/workspace/verify/*")):
    if os.path.isfile(p) and os.path.getsize(p) < 50_000_000:
        api.upload_file(path_or_fileobj=p, path_in_repo=os.environ["HF_DIR"] + "/" + os.path.basename(p),
                        repo_id="momom2/wesnoth-model-checkpoints")
PY
}
export HF_DIR

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
    || { echo BUILD_FAILED | tee -a "$OUT/build.log"; upload; touch "$OUT/ALL_DONE"; exit 1; }
{ nproc --all; nvidia-smi --query-gpu=name --format=csv,noheader; } > "$OUT/box.txt" 2>&1
upload

SUITES="tests/test_game_core.py tests/test_recruit_rejection.py tests/test_sentinel_value.py \
tests/test_encoder_transfer.py tests/test_train_perf.py tests/test_imitation_flat_batch.py \
tests/test_sim_determinism.py tests/test_fork_isolation.py tests/test_cli_help.py"

echo "--- suites, Python state of record"
python -m pytest $SUITES -q -p no:cacheprovider -m "" > "$OUT/tests_python.log" 2>&1
tail -3 "$OUT/tests_python.log"
echo "--- suites, Rust-owned state of record"
WESNOTH_RUST_CORE=1 python -m pytest $SUITES -q -p no:cacheprovider -m "" > "$OUT/tests_core.log" 2>&1
tail -3 "$OUT/tests_core.log"
upload

# The corpus the sweep needs.
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

echo "--- diff_core over $REPLAYS replays"
python tools/diff_core.py replays_dataset_imitation --limit "$REPLAYS" > "$OUT/diff_core.log" 2>&1
grep -v WARNING "$OUT/diff_core.log" | head -8
upload
touch "$OUT/ALL_DONE"
echo VERIFY_DONE
