#!/usr/bin/env bash
# Test 1 of the training-signal panel: end_turn decided at the actor
# level, against the reference player at raw:t0, with the end_turn
# logit offset as the attribution arm. Bars, predictions and cost are
# pre-registered in docs/endturn_rule_prereg_20260919.md; this script
# runs the screens, the 800-decisive match and (on a pass) the
# attribution arm, and writes the verdict under those bars.
#
# Expects /workspace/.hf_token (chmod 600). Records under
# /workspace/endturn, uploaded to HF $HF_DIR after each step.
set -uo pipefail
[ -x /venv/main/bin/python ] && export PATH=/venv/main/bin:$PATH
OUT=/workspace/endturn
HF_DIR="${HF_DIR:-tier-b/endturn_rule_20260919}"
STAGE="${STAGE:-tier-b/staging/stage_20260919a.tar.gz}"
JOBS="${JOBS:-20}"
SCREEN_GAMES="${SCREEN_GAMES:-40}"
MATCH_DECISIVE="${MATCH_DECISIVE:-800}"
MATCH_EXTRA="${MATCH_EXTRA:-500}"
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
for p in sorted(glob.glob("/workspace/endturn/*")):
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
{ echo "cores(all) $(nproc --all)"; grep -m1 "model name" /proc/cpuinfo;
  echo "cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)";
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader;
  free -m | head -2; python -c "import torch; print('torch', torch.__version__)";
  python tools/kernel_status.py 2>/dev/null | tail -8; } > "$OUT/box.txt" 2>&1
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

# ---- one match: player A under DECODE against raw:t0 ----------------
match() {                        # match NAME SEED_BASE GAMES EXTRA [player A decode flags]
    local name="$1" seed_base="$2" games="$3" extra="$4"; shift 4
    local dir="$OUT/games_$name"
    local t0
    t0=$(date +%s)
    python tools/run_elo_batch.py --label-a "$name" --spec-a "$CKPT" \
        --label-b relset --spec-b "$CKPT" \
        --outdir "$dir" --games "$games" --max-extra-games "$extra" --seed-base "$seed_base" \
        --mcts-sims 0 --raw-temperature-a 0 --raw-temperature-b 0 "$@" \
        --persistent-workers --shared-inference --no-infer-compile --device cuda \
        --jobs "$JOBS" --inference-max-batch "$JOBS" \
        --time-budget-min 120 2>&1 | grep --line-buffered -v "wesnoth_core is not importable" > "$OUT/match_$name.log"
    echo "$name seed_base=$seed_base games=$games $* $(( $(date +%s) - t0 )) s $(ls "$dir"/game_*.json 2>/dev/null | wc -l) games" | tee -a "$OUT/match.walls"
    python tools/analysis/endturn_readout.py "$dir" | tee "$OUT/readout_$name.txt"
    upload
}

# ---- 1. the screens: the rule and the two offsets, 40 games each -----
match screen_endm 41000 "$SCREEN_GAMES" 0 --raw-end-turn-a actor
match screen_eo075 41100 "$SCREEN_GAMES" 0 --raw-end-turn-offset-a -0.75
match screen_eo150 41200 "$SCREEN_GAMES" 0 --raw-end-turn-offset-a -1.5

# Kill 1: the rule's decisions per side-turn at least 3% above raw:t0's
# in the same games, else the rule does not fire and the match is not
# played.
if ! python tools/analysis/endturn_readout.py "$OUT/games_screen_endm" --require-fire 1.03; then
    echo "KILL 1: the rule does not fire (decisions per side-turn under 1.03x raw:t0's)" | tee "$OUT/verdict.txt"
    upload; touch "$OUT/ALL_DONE"; upload; exit 0
fi

# ---- 2. the rule to 800 decisive ------------------------------------
match endm 42000 "$MATCH_DECISIVE" "$MATCH_EXTRA" --raw-end-turn-a actor

# ---- 3. the attribution arm, on a pass only --------------------------
if python tools/analysis/endturn_readout.py "$OUT/games_endm" --require-pass 0.535; then
    best=$(python tools/analysis/endturn_readout.py "$OUT/games_screen_eo075" "$OUT/games_screen_eo150" --best-p)
    case "$best" in
        *eo150*) offset=-1.5 ;;
        *)       offset=-0.75 ;;
    esac
    match "eo$offset" 43000 "$MATCH_DECISIVE" "$MATCH_EXTRA" --raw-end-turn-offset-a "$offset"
fi

# ---- the verdict -----------------------------------------------------
python tools/analysis/endturn_readout.py "$OUT"/games_* --verdict | tee "$OUT/verdict.txt"
upload
touch "$OUT/ALL_DONE"
upload
echo ENDTURN_DONE
