#!/usr/bin/env bash
# The turn-level value gap under the reference player: the screen on
# the first 60 holdout boundary positions, the confirmation of its
# nominal hits, the verdict. Bars, predictions and cost are
# pre-registered in docs/turn_gap_ref_prereg_20260921.md.
#
# Expects /workspace/.hf_token (chmod 600). Records under
# /workspace/turn_gap, uploaded to HF $HF_DIR after each step; the
# tool rewrites <out>.partial.json after every completed position, so
# a cut stage is still read.
set -uo pipefail
[ -x /venv/main/bin/python ] && export PATH=/venv/main/bin:$PATH
OUT=/workspace/turn_gap
HF_DIR="${HF_DIR:-tier-b/turn_gap_ref_20260921}"
STAGE="${STAGE:-tier-b/staging/stage_20260921a.tar.gz}"
JOBS="${JOBS:-24}"
N_STATES="${N_STATES:-60}"
SEED="${SEED:-21}"
SCREEN_TIMEOUT="${SCREEN_TIMEOUT:-3.5h}"
CONFIRM_TIMEOUT="${CONFIRM_TIMEOUT:-2h}"
DPH="${DPH:-0.56}"
mkdir -p "$OUT"
cd /workspace
export HF_TOKEN="$(tr -d '\r\n' < /workspace/.hf_token)" HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export HF_DIR
python -m pip install -q huggingface_hub pytest psutil scipy 2>&1 | tail -1 || true

upload() {
    python - <<'PY' 2>/dev/null || true
import os, glob
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for p in sorted(glob.glob("/workspace/turn_gap/*")):
    if os.path.isfile(p) and os.path.getsize(p) < 50_000_000:
        api.upload_file(path_or_fileobj=p,
                        path_in_repo=os.environ["HF_DIR"] + "/" + os.path.basename(p),
                        repo_id="momom2/wesnoth-model-checkpoints")
PY
}

finish() {                       # finish REASON: the marker every exit leaves
    echo "$1" | tee -a "$OUT/status.txt"
    touch "$OUT/ALL_DONE"
    upload
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
if ! command -v cc >/dev/null 2>&1; then
    (apt-get update -qq && apt-get install -y -qq gcc) > "$OUT/cc_install.log" 2>&1 \
        || { conda install -y -q -c conda-forge c-compiler >> "$OUT/cc_install.log" 2>&1 \
             && ln -sf "$(ls /opt/conda/bin/x86_64-conda-linux-gnu-cc 2>/dev/null | head -1)" /usr/local/bin/cc; } \
        || true
fi
{ echo "cc: $(command -v cc || echo none)"; cc --version 2>&1 | head -1; } > "$OUT/build.log"
command -v cargo >/dev/null 2>&1 || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1
export PATH="$HOME/.cargo/bin:$PATH"
python -m pip install -q maturin >/dev/null 2>&1
touch rust/wesnoth_core/src/*.rs
python -m pip install --force-reinstall --no-deps rust/wesnoth_core >> "$OUT/build.log" 2>&1
python -c "import wesnoth_core; p = wesnoth_core.__phase__; print('wheel phase', p); \
assert p >= 10, f'wheel is phase {p}; the source declares 10'" | tee -a "$OUT/build.log" \
    || { echo BUILD_FAILED | tee -a "$OUT/build.log"; finish BUILD_FAILED; exit 1; }
{ echo "cores(all) $(nproc --all)"; grep -m1 "model name" /proc/cpuinfo;
  echo "cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)";
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader;
  free -m | head -2; python -c "import torch; print('torch', torch.__version__)";
  python tools/kernel_status.py 2>/dev/null | tail -8; } > "$OUT/box.txt" 2>&1
cat "$OUT/box.txt"
upload

# ---- the corpus the manifest rebuilds its positions from, and the reference
if [ ! -d replays_dataset_imitation ]; then
    python - <<'CORPUS'
import pathlib, tarfile
from huggingface_hub import hf_hub_download
p = hf_hub_download("momom2/wesnoth-model-checkpoints",
                    "tier-b/replays_dataset_imitation_dedup_20260908.tar.gz")
with tarfile.open(p, "r:gz") as tf:
    tf.extractall(".")
print("corpus staged", len(list(pathlib.Path("replays_dataset_imitation").glob("*.json.gz"))))
CORPUS
fi
python tools/reference_player.py --ensure | tee -a "$OUT/build.log" \
    || { finish REFERENCE_MISSING; exit 1; }
python tools/reference_player.py | tee "$OUT/reference.json"

# ---- 1. the screen ----------------------------------------------------
COMMON=(--reference --device cuda --jobs "$JOBS" --shared-inference --no-infer-compile
        --playout-temperature 0.5 --cap-turns 30 --seed "$SEED" --dollars-per-hour "$DPH"
        --gap-threshold 0.25)
t0=$(date +%s)
timeout "$SCREEN_TIMEOUT" python tools/turn_gap.py "${COMMON[@]}" \
    --n-states "$N_STATES" --alternatives 4 --temperature 1.0 \
    --playouts 40 --rounds 10 --drop-z 2 --stop-z 2 --stop-margin 0 \
    --out "$OUT/screen.json" 2>&1 | grep --line-buffered -v "wesnoth_core is not importable" > "$OUT/screen.log"
rc=${PIPESTATUS[0]}
echo "screen rc=$rc $(( $(date +%s) - t0 )) s" | tee -a "$OUT/walls.txt"
SCREEN="$OUT/screen.json"
[ -f "$SCREEN" ] || SCREEN="$OUT/screen.partial.json"
[ -f "$SCREEN" ] || { finish "SCREEN_FAILED rc=$rc"; exit 1; }
python tools/turn_gap.py --summarize "$SCREEN" --dollars-per-hour "$DPH" | tee "$OUT/screen_summary.md"
upload

# ---- 2. the confirmation of the nominal hits --------------------------
nominal=$(python - "$SCREEN" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(sum(1 for r in d["positions"] if r["gap"] >= 0.25))
PY
)
CONFIRM=""
if [ "$nominal" -gt 0 ]; then
    t0=$(date +%s)
    timeout "$CONFIRM_TIMEOUT" python tools/turn_gap.py "${COMMON[@]}" \
        --confirm-from "$SCREEN" --confirm-top 1 \
        --playouts 160 --rounds 20 --drop-z 2 --stop-z 2 --stop-margin 0.10 \
        --out "$OUT/confirm.json" 2>&1 | grep --line-buffered -v "wesnoth_core is not importable" > "$OUT/confirm.log"
    rc=${PIPESTATUS[0]}
    echo "confirm rc=$rc nominal=$nominal $(( $(date +%s) - t0 )) s" | tee -a "$OUT/walls.txt"
    CONFIRM="$OUT/confirm.json"
    [ -f "$CONFIRM" ] || CONFIRM="$OUT/confirm.partial.json"
    [ -f "$CONFIRM" ] && python tools/turn_gap.py --summarize "$CONFIRM" --dollars-per-hour "$DPH" | tee "$OUT/confirm_summary.md"
else
    echo "no nominal hit in the screen; nothing to confirm" | tee -a "$OUT/walls.txt"
fi
upload

# ---- the verdict ------------------------------------------------------
python tools/analysis/turn_gap_verdict.py "$SCREEN" ${CONFIRM:+"$CONFIRM"} \
    --n-positions "$N_STATES" --json "$OUT/verdict.json" | tee "$OUT/verdict.txt"
finish TURN_GAP_DONE
echo TURN_GAP_DONE
