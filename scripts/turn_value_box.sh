#!/usr/bin/env bash
# The turn-ranking value function under the reference player (`obs8` at
# raw:t0+eo-1.5): bars, predictions and cost in
# docs/turn_value_prereg_20260925.md. Stages, each cut at twice its
# estimate:
#   1. validation: the 2026-09-23 turn-gap procedure on the first 60
#      holdout boundary positions, screen then confirmation;
#   2. training data: candidate turns and one playout each at up to 15
#      turn starts of each of the 800 recorded games of obs8 against
#      terrain (tools/turn_value_data.py);
#   3. features of every candidate's pre-end_turn state through the
#      frozen trunk, the two arms fitted, the verdict.
# Expects /workspace/.hf_token (chmod 600). Records under
# /workspace/turn_value, uploaded to HF $HF_DIR after each stage and
# every 30 minutes during the data stage. Re-entry skips finished
# stages and continues the data log where it stopped. Every exit, clean
# or not, leaves ALL_DONE on HF and stops the instance (`stop_self`,
# with the id and key Vast puts in the container).
set -uo pipefail
[ -x /venv/main/bin/python ] && export PATH=/venv/main/bin:$PATH
WORKDIR=/workspace
OUT=$WORKDIR/turn_value
GAMES=$WORKDIR/obs_games
STAGE="${STAGE:-tier-b/staging/stage_20260925v.tar.gz}"
export HF_DIR="${HF_DIR:-tier-b/turn_value_20260925}"
GAMES_TAR="${GAMES_TAR:-tier-b/observation_retrain_20260924/games_obs_e1_vs_terrain.tar.gz}"
JOBS="${JOBS:-24}"
SEED="${SEED:-25}"
DPH="${DPH:-0.56}"
SCREEN_TIMEOUT="${SCREEN_TIMEOUT:-75m}"
CONFIRM_TIMEOUT="${CONFIRM_TIMEOUT:-75m}"
DATA_TIMEOUT="${DATA_TIMEOUT:-7.6h}"
FIT_TIMEOUT="${FIT_TIMEOUT:-1h}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export HF_HUB_DISABLE_XET=1
mkdir -p "$OUT"
cd $WORKDIR
export HF_TOKEN="$(tr -d '\r\n' < $WORKDIR/.hf_token)"
python -m pip install -q huggingface_hub psutil pytest scipy 2>&1 | grep -v "WARNING: Running pip" | tail -1 || true

upload() {                       # every record; the data log and the caches too
python - <<'EOF' 2>/dev/null || true
import glob, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for p in sorted(glob.glob("/workspace/turn_value/*")):
    if os.path.isfile(p) and os.path.getsize(p) < 1_000_000_000:
        api.upload_file(path_or_fileobj=p, path_in_repo=os.environ["HF_DIR"] + "/" + os.path.basename(p),
                        repo_id="momom2/wesnoth-model-checkpoints")
EOF
}
stop_self() {                    # stop this instance: its GPU stops billing, its disk stays
    local id="${CONTAINER_ID:-}" key="${CONTAINER_API_KEY:-}"
    if [ -z "$id" ] || [ -z "$key" ]; then
        id=$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^CONTAINER_ID=//p' | head -1)
        key=$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^CONTAINER_API_KEY=//p' | head -1)
    fi
    if [ -z "$id" ] || [ -z "$key" ]; then
        echo "stop_self $(date -u +%FT%TZ): no instance id or key in the environment; the laptop watcher stops the box" >> "$OUT/stop.log"
        upload
        return
    fi
    echo "stop_self $(date -u +%FT%TZ): stopping instance $id" >> "$OUT/stop.log"
    upload
    INSTANCE_KEY="$key" python - "$id" >> "$OUT/stop.log" 2>&1 <<'EOF'
import os, sys
import requests
r = requests.put(f"https://console.vast.ai/api/v0/instances/{sys.argv[1]}/",
                 params={"api_key": os.environ["INSTANCE_KEY"]}, json={"state": "stopped"}, timeout=60)
print("stop:", r.status_code, r.text[:200])
EOF
}
finish() {                       # finish REASON: every exit leaves ALL_DONE, then stops the box
    echo "$1 $(date -u +%FT%TZ)" | tee -a "$OUT/status.txt"
    touch "$OUT/ALL_DONE"
    stop_self
    exit "${2:-0}"
}
stage_wall() {                   # stage_wall NAME RC T0
    echo "$1 rc=$2 $(( $(date +%s) - $3 )) s" | tee -a "$OUT/walls.txt"
}

# ---- code, wheel, box facts -----------------------------------------
if [ ! -f Wesnoth-AI/.staged_from ] || [ "$(cat Wesnoth-AI/.staged_from)" != "$STAGE" ]; then
rm -rf Wesnoth-AI && mkdir -p Wesnoth-AI
python - "$STAGE" <<'EOF' || finish "CODE_STAGING_FAILED" 1
import sys, tarfile
from huggingface_hub import hf_hub_download
p = hf_hub_download("momom2/wesnoth-model-checkpoints", sys.argv[1])
with tarfile.open(p, "r:gz") as tf:
    tf.extractall("/workspace/Wesnoth-AI")
print("code staged", flush=True)
EOF
echo "$STAGE" > Wesnoth-AI/.staged_from
fi
cd Wesnoth-AI
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
want_phase=$(grep -oP '__phase__",\s*\K[0-9]+' rust/wesnoth_core/src/lib.rs | head -1)
python -c "import wesnoth_core, sys; p = wesnoth_core.__phase__; print('wheel phase', p, 'source', sys.argv[1]); \
assert str(p) == sys.argv[1]" "$want_phase" | tee -a "$OUT/build.log" \
    || finish "BUILD_FAILED (build.log)" 1
{ echo "cores(all) $(nproc --all)"; grep -m1 "model name" /proc/cpuinfo;
  echo "cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)"; free -g | head -2;
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader;
  python -c "import torch; print('torch', torch.__version__)";
  python -c "import wesnoth_ai; from wesnoth_ai.constants import OBSERVATION_EPOCH as E; print('code', wesnoth_ai.__version__, 'epoch', E)";
  python tools/kernel_status.py 2>/dev/null | tail -8; } > "$OUT/box.txt" 2>&1
cat "$OUT/box.txt"

# ---- the corpus the manifest positions rebuild from, the reference, the games
if [ ! -f "$OUT/STAGED" ]; then
python tools/reference_player.py --ensure >> "$OUT/build.log" 2>&1 || finish "REFERENCE_MISSING" 1
python tools/reference_player.py > "$OUT/reference.json"
python - "$GAMES_TAR" <<'EOF' || finish "DATA_STAGING_FAILED" 1
import pathlib, sys, tarfile
from huggingface_hub import hf_hub_download
p = hf_hub_download("momom2/wesnoth-model-checkpoints",
                    "tier-b/replays_dataset_imitation_dedup_20260908.tar.gz")
with tarfile.open(p, "r:gz") as tf:
    tf.extractall(".")
print("corpus staged", len(list(pathlib.Path("replays_dataset_imitation").glob("*.json.gz"))))
p = hf_hub_download("momom2/wesnoth-model-checkpoints", sys.argv[1])
games = pathlib.Path("/workspace/obs_games")
games.mkdir(exist_ok=True)
with tarfile.open(p, "r:gz") as tf:
    tf.extractall(games)
records = list(games.rglob("*.game.jsonl.gz"))
print("games staged", len(records))
assert len(records) == 800, len(records)
EOF
touch "$OUT/STAGED"
fi
GAMES_DIR=$(dirname "$(find "$GAMES" -name '*.game.jsonl.gz' | head -1)")

# ---- the tools against the built wheel, before anything is spent on them
python -m pytest tests/test_turn_value.py tests/test_turn_gap.py tests/test_game_record.py \
    -q -p no:cacheprovider -m "" > "$OUT/tests.log" 2>&1
rc=$?
tail -3 "$OUT/tests.log"
[ "$rc" -eq 0 ] || finish "TESTS_FAILED (tests.log)" 1
upload

# ---- 1. validation: the screen, then the confirmation of its hits ----
COMMON=(--reference --device cuda --jobs "$JOBS" --shared-inference --no-infer-compile
        --playout-temperature 0.5 --cap-turns 30 --seed "$SEED" --dollars-per-hour "$DPH"
        --gap-threshold 0.25)
if [ ! -f "$OUT/screen.json" ]; then
    t0=$(date +%s)
    timeout "$SCREEN_TIMEOUT" python tools/turn_gap.py "${COMMON[@]}" \
        --n-states 60 --alternatives 4 --temperature 1.0 \
        --playouts 40 --rounds 10 --drop-z 2 --stop-z 2 --stop-margin 0 \
        --out "$OUT/screen.json" > "$OUT/screen.log" 2>&1
    stage_wall screen $? "$t0"
fi
SCREEN="$OUT/screen.json"
[ -f "$SCREEN" ] || SCREEN="$OUT/screen.partial.json"
[ -f "$SCREEN" ] || finish "SCREEN_FAILED (screen.log)" 1
python tools/turn_gap.py --summarize "$SCREEN" --dollars-per-hour "$DPH" > "$OUT/screen_summary.md"
upload
CONFIRM="$OUT/confirm.json"
if [ ! -f "$CONFIRM" ]; then
    nominal=$(python -c "import json,sys; print(sum(r['gap'] >= 0.25 for r in json.load(open(sys.argv[1]))['positions']))" "$SCREEN")
    if [ "$nominal" -gt 0 ]; then
        t0=$(date +%s)
        timeout "$CONFIRM_TIMEOUT" python tools/turn_gap.py "${COMMON[@]}" \
            --confirm-from "$SCREEN" --confirm-top 1 \
            --playouts 160 --rounds 20 --drop-z 2 --stop-z 2 --stop-margin 0.10 \
            --out "$CONFIRM" > "$OUT/confirm.log" 2>&1
        stage_wall "confirm nominal=$nominal" $? "$t0"
    else
        echo "no nominal hit in the screen; nothing to confirm" | tee -a "$OUT/walls.txt"
    fi
fi
[ -f "$CONFIRM" ] || CONFIRM="$OUT/confirm.partial.json"
[ -f "$CONFIRM" ] && python tools/turn_gap.py --summarize "$CONFIRM" --dollars-per-hour "$DPH" > "$OUT/confirm_summary.md"
upload

# ---- 2. training data, uploaded every 30 minutes while it grows -------
if [ ! -f "$OUT/DATA_DONE" ]; then
    ( while sleep 1800; do upload; done ) &
    escrow=$!
    t0=$(date +%s)
    timeout "$DATA_TIMEOUT" python tools/turn_value_data.py --reference --games-dir "$GAMES_DIR" \
        --device cuda --jobs "$JOBS" --shared-inference --no-infer-compile --seed "$SEED" \
        --out "$OUT/data.jsonl.gz" >> "$OUT/data.log" 2>&1
    rc=$?
    stage_wall data "$rc" "$t0"
    kill "$escrow" 2>/dev/null
    touch "$OUT/DATA_DONE"
    upload
fi

# ---- 3. features, the arms, the verdict ------------------------------
t0=$(date +%s)
FEAT=(python tools/turn_value.py features --reference --device cuda --jobs "$JOBS")
[ -f "$OUT/train.pt" ] || timeout "$FIT_TIMEOUT" "${FEAT[@]}" --games-dir "$GAMES_DIR" \
    --positions "$OUT/data.jsonl.gz" --out "$OUT/train.pt" >> "$OUT/features.log" 2>&1 \
    || finish "FEATURES_FAILED (features.log)" 1
"${FEAT[@]}" --positions "$SCREEN" --out "$OUT/screen.pt" >> "$OUT/features.log" 2>&1 \
    || finish "FEATURES_FAILED (features.log)" 1
PRIMARY=()
if [ -f "$CONFIRM" ]; then
    "${FEAT[@]}" --positions "$CONFIRM" --out "$OUT/confirm.pt" >> "$OUT/features.log" 2>&1 \
        || finish "FEATURES_FAILED (features.log)" 1
    PRIMARY=(--primary "$CONFIRM=$OUT/confirm.pt")
fi
stage_wall features 0 "$t0"
python tools/turn_value.py fit --train "$OUT/train.pt" --out-dir "$OUT" >> "$OUT/fit.log" 2>&1 \
    || finish "FIT_FAILED (fit.log)" 1
python tools/turn_value.py evaluate --heads "$OUT" "${PRIMARY[@]}" \
    --secondary "$SCREEN=$OUT/screen.pt" --train "$OUT/train.pt" \
    --out "$OUT/verdict.json" > "$OUT/verdict.md.log" 2>&1 \
    || finish "EVALUATE_FAILED (verdict.md.log)" 1
finish TURN_VALUE_DONE
