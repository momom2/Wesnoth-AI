#!/usr/bin/env bash
# How often each XOD rewrite class fires on played games, before the
# census of docs/xod_dominance_design_20260924.md section 7: the
# reference plays itself (terrain at raw:t0+eo-1.5 on both sides, the
# census's 60-turn cap, every game recorded whole by the eval path,
# tools/game_record.py), then tools/analysis/dominance_count.py reads
# those games and the imitation corpus's training split under every
# combination of relaxations, and keeps every fight distribution it
# computed.
#
# Each stage is cut at twice its estimate (GAMES_PER_MIN and
# COUNT_GAMES_PER_MIN, measured 2026-09-24: about 2 s per game per core
# for the count on 20 corpus games; the match rate is the 2026-09-13
# 40-game walls, 42-74 s at 10 jobs). The tools write as they go and a
# background loop uploads /workspace/dominance every UPLOAD_EVERY_S, so a
# cut or a dead box keeps what it produced. Every exit leaves ALL_DONE.
#
# Expects /workspace/.hf_token (chmod 600). Records under
# /workspace/dominance, uploaded to HF $HF_DIR.
set -uo pipefail
[ -x /venv/main/bin/python ] && export PATH=/venv/main/bin:$PATH
OUT=/workspace/dominance
HF_DIR="${HF_DIR:-tier-b/dominance_count_20260924}"
STAGE="${STAGE:-tier-b/staging/stage_20260924x.tar.gz}"
JOBS="${JOBS:-20}"
GAMES="${GAMES:-2000}"                 # decisive games wanted
EXTRA="${EXTRA:-1000}"                 # capped games replaced, at most
MAX_TURNS="${MAX_TURNS:-60}"
MATCH_EST_MIN="${MATCH_EST_MIN:-50}"   # 2,000 games at about 40 a minute
COUNT_EST_MIN="${COUNT_EST_MIN:-25}"   # about 19,000 games at 2 s each over 32 cores
UPLOAD_EVERY_S="${UPLOAD_EVERY_S:-600}"
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
for p in sorted(glob.glob("/workspace/dominance/*")):
    if os.path.isfile(p) and not p.endswith(".tmp") and os.path.getsize(p) < 900_000_000:
        api.upload_file(path_or_fileobj=p,
                        path_in_repo=os.environ["HF_DIR"] + "/" + os.path.basename(p),
                        repo_id="momom2/wesnoth-model-checkpoints")
PY
}

FINISHED=0
UPLOADER=""
finish() {                       # finish REASON: the marker every exit leaves, once
    [ "$FINISHED" = 1 ] && return
    FINISHED=1
    [ -n "$UPLOADER" ] && kill "$UPLOADER" 2>/dev/null
    echo "$1" | tee -a "$OUT/status.txt"
    touch "$OUT/ALL_DONE"
    upload
}
trap 'finish "EXITED rc=$?"' EXIT
trap 'finish INTERRUPTED; exit 130' INT TERM HUP

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
want_phase=$(grep -oP '__phase__",\s*\K[0-9]+' rust/wesnoth_core/src/lib.rs | head -1)
python -c "import wesnoth_core, sys; p = wesnoth_core.__phase__; print('wheel phase', p, 'source', sys.argv[1]); \
assert str(p) == sys.argv[1]" "$want_phase" | tee -a "$OUT/build.log" \
    || { finish BUILD_FAILED; exit 1; }
{ echo "cores(all) $(nproc --all)"; grep -m1 "model name" /proc/cpuinfo;
  echo "cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)";
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader;
  free -m | head -2; python -c "import torch; print('torch', torch.__version__)";
  python -c "import wesnoth_ai; print('code version', wesnoth_ai.__version__)";
  python tools/kernel_status.py 2>/dev/null | tail -8; } > "$OUT/box.txt" 2>&1
cat "$OUT/box.txt"
upload
( while sleep "$UPLOAD_EVERY_S"; do upload; done ) &
UPLOADER=$!

# ---- the corpus and the reference -----------------------------------
python - <<'CORPUS'
import pathlib, tarfile
from huggingface_hub import hf_hub_download
p = hf_hub_download("momom2/wesnoth-model-checkpoints",
                    "tier-b/replays_dataset_imitation_dedup_20260908.tar.gz")
with tarfile.open(p, "r:gz") as tf:
    tf.extractall(".")
print("corpus staged", len(list(pathlib.Path("replays_dataset_imitation").glob("*.json.gz"))))
CORPUS
python tools/reference_player.py --ensure | tee -a "$OUT/build.log" || { finish REFERENCE_MISSING; exit 1; }

# ---- 1. the reference against itself, every game recorded -----------
GAMES_DIR="$OUT/games_ref"
t0=$(date +%s)
# shellcheck disable=SC2046
python tools/run_elo_batch.py $(python tools/reference_player.py --flags a) \
    $(python tools/reference_player.py --flags b | sed 's/--label-b terrain/--label-b terrainmirror/') \
    --outdir "$GAMES_DIR" --games "$GAMES" --max-extra-games "$EXTRA" --seed-base 60000 \
    --max-turns "$MAX_TURNS" --mcts-sims 0 --raw-temperature-a 0 --raw-temperature-b 0 \
    --persistent-workers --shared-inference --no-infer-compile --device cuda \
    --jobs "$JOBS" --inference-max-batch "$JOBS" --time-budget-min $((2 * MATCH_EST_MIN)) \
    2>&1 | grep --line-buffered -v "wesnoth_core is not importable" > "$OUT/match.log"
n_rec=$(ls "$GAMES_DIR"/*.game.jsonl.gz 2>/dev/null | wc -l)
echo "match $(( $(date +%s) - t0 )) s, $n_rec recorded games (estimate ${MATCH_EST_MIN} min)" | tee -a "$OUT/walls.txt"
tar -czf "$OUT/games_ref_records.tar.gz" -C "$GAMES_DIR" $(cd "$GAMES_DIR" && ls *.game.jsonl.gz) 2>/dev/null
upload
[ "$n_rec" -gt 0 ] || { finish NO_GAMES; exit 1; }

# ---- 2. the count over both sources ---------------------------------
n_human=$(python - <<'PY'
import json
rows = [json.loads(l) for l in open("replays_dataset_imitation/manifest.jsonl")]
print(sum(1 for r in rows if not r.get("holdout")))
PY
)
t0=$(date +%s)
timeout "$((2 * COUNT_EST_MIN))m" python tools/analysis/dominance_count.py \
    --corpus replays_dataset_imitation --games "$n_human" \
    --records "$GAMES_DIR/*.game.jsonl.gz" --jobs "$(nproc)" \
    --out "$OUT/counts.json" --outcomes-out "$OUT/outcomes.jsonl.gz" --write-every 200 \
    > "$OUT/count.md" 2> "$OUT/count.log"
rc=$?
echo "count rc=$rc $(( $(date +%s) - t0 )) s over $n_human human games and $n_rec generated (estimate ${COUNT_EST_MIN} min)" \
    | tee -a "$OUT/walls.txt"
cat "$OUT/count.md"
finish "DONE rc=$rc"
