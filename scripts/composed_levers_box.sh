#!/usr/bin/env bash
# Do the day's two levers add? The terrain-set arm under the end_turn
# offset -1.5 against the reference under the same offset, the
# composed player against today's reference, and the arm's self-pin,
# 800 decisive games each, under docs/composed_levers_prereg_20260919.md.
# Records under /workspace/composed, uploaded to HF $HF_DIR after each
# match; every exit leaves ALL_DONE.
set -uo pipefail
[ -x /venv/main/bin/python ] && export PATH=/venv/main/bin:$PATH
OUT=/workspace/composed
HF_DIR="${HF_DIR:-tier-b/composed_levers_20260919}"
STAGE="${STAGE:-tier-b/staging/stage_20260919b.tar.gz}"
ARM_HF="${ARM_HF:-tier-b/terrain_multi_hot_20260919/arm_epoch0.pt}"
REF_HF="${REF_HF:-tier-b/seed2_relset_20260911/arm_epoch0.pt}"
JOBS="${JOBS:-20}"
MATCH_DECISIVE="${MATCH_DECISIVE:-800}"
MATCH_EXTRA="${MATCH_EXTRA:-500}"
OFFSET="${OFFSET:--1.5}"
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
for p in sorted(glob.glob("/workspace/composed/*")):
    if os.path.isfile(p) and os.path.getsize(p) < 50_000_000:
        api.upload_file(path_or_fileobj=p,
                        path_in_repo=os.environ["HF_DIR"] + "/" + os.path.basename(p),
                        repo_id="momom2/wesnoth-model-checkpoints")
PY
}
die() {
    echo "FAILED: $1" | tee -a "$OUT/FAILED"
    touch "$OUT/ALL_DONE"; upload; exit 1
}

rm -rf Wesnoth-AI && mkdir -p Wesnoth-AI
python - "$STAGE" <<'PY' || die "code staging"
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
    || die "wheel build (build.log)"
{ echo "cores(all) $(nproc --all)"; grep -m1 "model name" /proc/cpuinfo;
  echo "cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)";
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader;
  free -m | head -2; python -c "import torch; print('torch', torch.__version__)";
  python tools/kernel_status.py 2>/dev/null | tail -8; } > "$OUT/box.txt" 2>&1
cat "$OUT/box.txt"
upload

python - "$ARM_HF" "$REF_HF" <<'PY' || die "checkpoint staging"
import os, pathlib, sys
from huggingface_hub import hf_hub_download
dst = pathlib.Path("training/checkpoints"); dst.mkdir(parents=True, exist_ok=True)
for remote, local in ((sys.argv[1], "terrain_e1.pt"), (sys.argv[2], "relset.pt")):
    src = hf_hub_download("momom2/wesnoth-model-checkpoints", remote, token=os.environ["HF_TOKEN"])
    (dst / local).write_bytes(pathlib.Path(src).read_bytes())
    print("staged", local, "from", remote)
PY
ARM=training/checkpoints/terrain_e1.pt
REF=training/checkpoints/relset.pt

match() {                        # match NAME LABEL_A SPEC_A LABEL_B SPEC_B SEED_BASE [decode flags]
    local name="$1" la="$2" a="$3" lb="$4" b="$5" seed_base="$6"; shift 6
    local dir="$OUT/games_$name"
    local t0
    t0=$(date +%s)
    python tools/run_elo_batch.py --label-a "$la" --spec-a "$a" --label-b "$lb" --spec-b "$b" \
        --outdir "$dir" --games "$MATCH_DECISIVE" --max-extra-games "$MATCH_EXTRA" --seed-base "$seed_base" \
        --mcts-sims 0 --raw-temperature-a 0 --raw-temperature-b 0 "$@" \
        --persistent-workers --shared-inference --no-infer-compile --device cuda \
        --jobs "$JOBS" --inference-max-batch "$JOBS" \
        --time-budget-min 120 2>&1 | grep --line-buffered -v "wesnoth_core is not importable" > "$OUT/match_$name.log"
    echo "$name seed_base=$seed_base $* $(( $(date +%s) - t0 )) s $(ls "$dir"/game_*.json 2>/dev/null | wc -l) games" | tee -a "$OUT/match.walls"
    python tools/analysis/endturn_readout.py "$dir" | tee "$OUT/readout_$name.txt"
    python tools/elo_collect.py "$dir" --no-catalog --save-json "$OUT/$name.fit.json" 2>&1 | tail -4 | tee -a "$OUT/match_$name.log"
    upload
}

match composed_vs_refdecode terrain_e1 "$ARM" relset "$REF" 47000 \
    --raw-end-turn-offset-a "$OFFSET" --raw-end-turn-offset-b "$OFFSET"
match composed_vs_ref terrain_e1 "$ARM" relset "$REF" 48000 --raw-end-turn-offset-a "$OFFSET"
match arm_selfpin terrain_e1 "$ARM" terrainmirror_e1 "$ARM" 49000

python tools/analysis/endturn_readout.py "$OUT"/games_* | tee "$OUT/verdict.txt"
touch "$OUT/ALL_DONE"
upload
echo COMPOSED_DONE
