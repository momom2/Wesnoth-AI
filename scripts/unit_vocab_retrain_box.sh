#!/usr/bin/env bash
# The reference's recipe with every reachable unit type on its own row of
# the type embedding (tools/unit_vocab.py) and the neutral side's commands
# out of the imitation pairs. `obs8` has 157 of its 356 type names on the
# overflow row, 76 of the 190 unit types our games can field among them.
# Everything else is `scripts/observation_retrain_box.sh` to the letter:
# the relevant-set basis, --terrain-multi-hot, configs/imitation.json,
# batch 64, lr 1e-4, cosine over 4 epochs stopped after STOP_AFTER_EPOCH
# passes, the deduplicated corpus with the manifest split, fog gate on,
# pre-encoded records, the same run seed. Then:
#   the per-phase value evaluation of the checkpoint;
#   the arm against the reference player (`obs8`), both at the reference
#     decode (raw:t0+eo-1.5), PURE, 800 decisive games, each side served
#     in its own encoding by its own inference server; the match is read
#     only once it holds its 800 decisive games.
# Bars and predictions: docs/unit_vocab_retrain_prereg_20260925.md.
# Unattended: fetched and started by the box's onstart through HF.
# Every 30 minutes the escrow uploads the latest checkpoint, epoch
# checkpoints, log and curve to HF $HF_DIR. Re-entry resumes from the
# latest checkpoint, continuing its pass where it was cut
# (tools/supervised_train.py `PassPosition`). Every exit, clean or not,
# leaves ALL_DONE on HF and then stops the instance (`stop_self`, with the
# id and key Vast puts in the container). Never `set -x`: the HF token and
# the instance key are in the environment.
set -uo pipefail
WORKDIR=/workspace
OUT=$WORKDIR/vocabretrain
ENC=$WORKDIR/encoded_vocab
STAGE="${STAGE:-tier-b/staging/stage_20260925u.tar.gz}"
EPOCHS="${EPOCHS:-4}"
STOP_AFTER_EPOCH="${STOP_AFTER_EPOCH:-1}"
RUN_SEED="${RUN_SEED:-20260909}"
WORKERS="${WORKERS:-30}"
GAMES="${GAMES:-800}"
JOBS="${JOBS:-20}"
export HF_DIR="${HF_DIR:-tier-b/unit_vocab_retrain_20260925}"
ARCH="--d-model 384 --num-layers 8 --num-heads 12 --d-ff 1536"
EVAL="--eval-every 50000 --eval-pairs 1200 --eval-pairs-per-game 8 --eval-sample-seed 0"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export HF_HUB_DISABLE_XET=1
mkdir -p "$OUT"
cd $WORKDIR
export HF_TOKEN="$(tr -d '\r\n' < $WORKDIR/.hf_token)"
python -m pip install -q huggingface_hub psutil pytest scipy 2>&1 | grep -v "WARNING: Running pip" | tail -1 || true

upload_small() {
python - <<'EOF' 2>/dev/null || true
import glob, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for p in sorted(glob.glob("/workspace/vocabretrain/*")):
    if os.path.isfile(p) and os.path.getsize(p) < 50_000_000 and not p.endswith(".pt"):
        api.upload_file(path_or_fileobj=p, path_in_repo=os.environ["HF_DIR"] + "/" + os.path.basename(p),
                        repo_id="momom2/wesnoth-model-checkpoints")
EOF
}
instance_var() {                 # instance_var NAME: from our environment, else from PID 1's
    local value="${!1:-}"
    [ -n "$value" ] || value=$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n "s/^$1=//p" | head -1)
    printf '%s' "$value"
}
stop_self() {                    # stop this instance: its GPU stops billing, its disk stays
    local id key
    id=$(instance_var CONTAINER_ID)
    key=$(instance_var CONTAINER_API_KEY)
    if [ -z "$id" ] || [ -z "$key" ]; then
        echo "stop_self $(date -u +%FT%TZ): no instance id or key in the environment; the laptop watcher stops the box" >> "$OUT/stop.log"
        upload_small
        return
    fi
    echo "stop_self $(date -u +%FT%TZ): stopping instance $id" >> "$OUT/stop.log"
    upload_small
    # The key travels in the environment and in the Authorization header.
    # Only status codes and exception type names are printed, and stderr
    # is dropped: a traceback can carry the URL, which holds the key in
    # the query form.
    INSTANCE_KEY="$key" timeout -k 30s 5m python - "$id" >> "$OUT/stop.log" 2>/dev/null <<'EOF'
import os
import sys
import time

import requests

URL = f"https://console.vast.ai/api/v0/instances/{sys.argv[1]}/"
KEY = os.environ["INSTANCE_KEY"]
AUTH = {"bearer": {"headers": {"Authorization": f"Bearer {KEY}"}},
        "query": {"params": {"api_key": KEY}}}
form = "bearer"
for attempt in range(1, 4):
    try:
        status = requests.put(URL, json={"state": "stopped"}, timeout=60, **AUTH[form]).status_code
    except Exception as e:  # noqa: BLE001 - the type name only
        print(f"stop attempt {attempt} ({form}): {type(e).__name__}", flush=True)
        time.sleep(20)
        continue
    print(f"stop attempt {attempt} ({form}): HTTP {status}", flush=True)
    if 200 <= status < 300:
        sys.exit(0)
    # Vast answers 404, not 401, to a key it does not accept: the key in
    # the query, as the vastai SDK also sends it, tried once.
    if 400 <= status < 500 and status != 429:
        if form == "query":
            sys.exit(1)
        form = "query"
        continue
    time.sleep(20)
sys.exit(1)
EOF
}
die() {                          # die REASON: the run stops, the box does not idle
    echo "FAILED: $1" | tee -a "$OUT/FAILED"
    touch "$OUT/ALL_DONE"
    upload_small
    stop_self
    exit 1
}

if [ ! -f Wesnoth-AI/.staged_from ] || [ "$(cat Wesnoth-AI/.staged_from)" != "$STAGE" ]; then
python - "$STAGE" <<'EOF' || die "code staging"
import os, sys, tarfile
from huggingface_hub import hf_hub_download
p = hf_hub_download("momom2/wesnoth-model-checkpoints", sys.argv[1])
os.makedirs("/workspace/Wesnoth-AI", exist_ok=True)
with tarfile.open(p, "r:gz") as tf:
    tf.extractall("/workspace/Wesnoth-AI")
print("code staged", flush=True)
EOF
echo "$STAGE" > Wesnoth-AI/.staged_from
fi
cd Wesnoth-AI

# ---- the Rust wheel (the pre-encoder, the trainer's probe and eval all run its kernels)
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
    || die "wheel build (build.log)"
{ nproc --all; grep -m1 "model name" /proc/cpuinfo; cat /sys/fs/cgroup/cpu.max 2>/dev/null; free -g | head -2;
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader;
  python -c "import torch; print('torch', torch.__version__)";
  python -c "import wesnoth_ai; from wesnoth_ai.constants import OBSERVATION_EPOCH as E; print('code', wesnoth_ai.__version__, 'epoch', E)";
  python tools/kernel_status.py 2>/dev/null | tail -8; } > "$OUT/box.txt" 2>&1
upload_small

# ---- the encoder and the observation against the Rust kernels, before anything trains
python -m pytest tests/test_unit_vocab.py tests/test_game_record.py \
    tests/test_time_of_day_features.py tests/test_terrain_multi_hot.py tests/test_rust_encode_raw.py \
    tests/test_game_core.py tests/test_vision.py tests/test_rust_observe.py \
    -q -p no:cacheprovider -m "" > "$OUT/tests_obs.log" 2>&1
rc_tests=$?
tail -3 "$OUT/tests_obs.log"
upload_small
[ "$rc_tests" -eq 0 ] || die "vocabulary, pair, encoder or observation tests (tests_obs.log)"

if [ ! -f "$OUT/STAGED" ]; then
python tools/reference_player.py --ensure >> "$OUT/build.log" 2>&1 || die "reference staging"
python - <<'EOF' || die "corpus staging"
import json, tarfile
from huggingface_hub import hf_hub_download
p = hf_hub_download("momom2/wesnoth-model-checkpoints",
                    "tier-b/replays_dataset_imitation_dedup_20260908.tar.gz")
with tarfile.open(p, "r:gz") as tf:
    tf.extractall(".")
rows = [json.loads(l) for l in open("replays_dataset_imitation/manifest.jsonl")]
print("corpus", len(rows), "games, holdout", sum(1 for r in rows if r.get("holdout")), flush=True)
EOF
touch "$OUT/STAGED"
fi

if [ ! -f "$OUT/fresh_vocab.pt" ]; then
python - <<'EOF' || die "vocab"
import torch
from tools.unit_vocab import seed_vocab
from wesnoth_ai.encoder import GameStateEncoder, names_on_overflow_row
enc = GameStateEncoder(d_model=32)
seed_vocab(enc)                                 # refuses a set that reaches the overflow row
assert len(enc.unit_type_to_id) == 190, len(enc.unit_type_to_id)
assert names_on_overflow_row(enc.unit_type_to_id) == []
torch.save({"unit_type_to_id": dict(enc.unit_type_to_id),
            "faction_to_id": dict(enc.faction_to_id)}, "/workspace/vocabretrain/fresh_vocab.pt")
print("fresh vocab:", len(enc.unit_type_to_id), "types,", len(enc.faction_to_id), "factions", flush=True)
EOF
fi

if [ ! -f "$ENC/PREENCODE_DONE" ]; then
    t0=$(date +%s)
    python tools/preencode_corpus.py --dataset replays_dataset_imitation --out "$ENC" \
        --vocab-from "$OUT/fresh_vocab.pt" --fog-hides-enemy-villages --relevant-set-hexes \
        --terrain-multi-hot --workers "$WORKERS" \
        2>&1 | grep --line-buffered -v "wesnoth_core is not importable" | tee -a "$OUT/preencode.log" | tail -3
    echo "preencode wall $(( $(date +%s) - t0 )) s" | tee -a "$OUT/preencode.log"
    grep -q "PREENCODE_DONE" "$OUT/preencode.log" && touch "$ENC/PREENCODE_DONE" \
        || die "pre-encoding (preencode.log)"
fi

escrow() {
python - <<'EOF'
import glob, os, shutil, time
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
out = "/workspace/vocabretrain"
hf_dir = os.environ["HF_DIR"]


def upload(path, name):
    for attempt in range(3):
        try:
            api.upload_file(path_or_fileobj=path, path_in_repo=f"{hf_dir}/{name}",
                            repo_id="momom2/wesnoth-model-checkpoints")
            return True
        except Exception as e:                      # noqa: BLE001 - the loop retries next round
            print("upload failed", name, attempt, type(e).__name__, str(e)[:120], flush=True)
            time.sleep(30)
    return False


files = ["train.log", "arm_eval.jsonl", "arm_signal.jsonl", "arm_prof.json",
         "preencode.log", "box.txt", "progress.txt", "build.log",
         "tests_obs.log", "phase_obs.json", "phase_obs.md", "STOPPED_AFTER_EPOCH", "match.walls"]
files += sorted(os.path.basename(p) for p in glob.glob(out + "/arm_epoch*.pt"))
files += sorted(os.path.basename(p) for p in glob.glob(out + "/*.fit.json"))
files += sorted(os.path.basename(p) for p in glob.glob(out + "/*.server_stats.json"))
files += sorted(os.path.basename(p) for p in glob.glob(out + "/timing_*.txt"))
for name in files:
    p = os.path.join(out, name)
    if not os.path.exists(p):
        continue
    marker = p + ".escrowed"
    if name.endswith(".pt") and os.path.exists(marker):
        continue
    if upload(p, name):
        open(marker, "w").close()
latest = os.path.join(out, "arm.pt")
if os.path.exists(latest) and not os.path.exists(out + "/DONE"):
    snap = os.path.join(out, "arm_latest_snapshot.pt")
    shutil.copyfile(latest, snap)
    if upload(snap, "arm_latest.pt"):
        print("ESCROW_OK", time.strftime("%Y-%m-%d %H:%M"), flush=True)
EOF
}
progress() {
    { date -u; grep -o "epoch=[0-9]* step=[0-9]*.*pairs=[0-9]* rate=[0-9.]*/s wall=[0-9.]*m" "$OUT/train.log" 2>/dev/null | tail -1 | sed "s/avg_loss.*pairs=/pairs=/";
      grep "EVAL\[epoch" "$OUT/train.log" 2>/dev/null | cut -c1-160; ls "$OUT" | tr "\n" " "; echo;
      tail -2 "$OUT"/*.log 2>/dev/null | tail -8; } > "$OUT/progress.txt"
}
( while [ ! -f "$OUT/ALL_DONE" ]; do sleep 1800; progress; escrow >> "$OUT/escrow.log" 2>&1; done ) &
ESCROW_PID=$!

# Stop after STOP_AFTER_EPOCH like the reference: kill the trainer once
# that epoch's checkpoint and holdout eval are written.
( until grep -q "EVAL\[epoch$((STOP_AFTER_EPOCH - 1))-end\]" "$OUT/train.log" 2>/dev/null; do sleep 20; done
  for p in $(pgrep -f "^python tools/supervised_train"); do kill $p; done; sleep 15
  pgrep -f "^python tools/supervised_train" >/dev/null && pkill -9 -f "^python tools/supervised_train"
  echo "trainer stopped $(date -u +%H:%M) after epoch $STOP_AFTER_EPOCH" > "$OUT/STOPPED_AFTER_EPOCH" ) &

if [ -n "${RESUME_HF:-}" ] && [ ! -f "$OUT/arm.pt" ] && [ ! -f "$OUT/DONE" ]; then
python - <<EOF || die "resume download"
import shutil
from huggingface_hub import hf_hub_download
shutil.copyfile(hf_hub_download("momom2/wesnoth-model-checkpoints", "$RESUME_HF"), "$OUT/arm.pt")
print("resuming from $RESUME_HF", flush=True)
EOF
fi
[ -f "$OUT/arm_epoch$((STOP_AFTER_EPOCH - 1)).pt" ] && touch "$OUT/DONE"
if [ ! -f "$OUT/DONE" ]; then
    RESUME=""
    [ -f "$OUT/arm.pt" ] && RESUME="--resume $OUT/arm.pt"
    t0=$(date +%s)
    python tools/supervised_train.py replays_dataset_imitation \
        --checkpoint "$OUT/arm.pt" $RESUME \
        --imitation-config configs/imitation.json $ARCH --relevant-set-hexes --terrain-multi-hot \
        --epochs "$EPOCHS" --seed "$RUN_SEED" \
        --bs 64 --lr 1e-4 --device cuda --workers 0 --preencoded "$ENC" \
        $EVAL --ckpt-every 2000 --log-every 100 \
        2>&1 | grep --line-buffered -v "wesnoth_core is not importable" | tee -a "$OUT/train.log"
    echo "training wall $(( $(date +%s) - t0 )) s" | tee -a "$OUT/train.log"
    if [ ! -f "$OUT/arm_epoch$((STOP_AFTER_EPOCH - 1)).pt" ]; then
        escrow >> "$OUT/escrow.log" 2>&1
        die "training ended without the epoch-$STOP_AFTER_EPOCH checkpoint"
    fi
    touch "$OUT/DONE"
fi
CKPT="$OUT/arm_epoch$((STOP_AFTER_EPOCH - 1)).pt"
escrow >> "$OUT/escrow.log" 2>&1

[ -f "$OUT/phase_obs.json" ] || python tools/analysis/value_head_by_phase.py --checkpoint "$CKPT" \
    --jobs 20 --device cuda --out "$OUT/phase_obs.json" 2>&1 | grep -v "wesnoth_core is not importable" | tail -10

decisive_results() {
    python - "$1" <<'EOF'
import json, pathlib, sys
n = 0
for p in pathlib.Path(sys.argv[1]).glob("game_*.json"):
    try:
        n += json.loads(p.read_text(encoding="utf-8")).get("outcome_a") in ("win", "loss")
    except Exception:
        pass
print(n)
EOF
}
match() {                         # match NAME SPEC_A SPEC_B GAMES SEED_BASE MAX_EXTRA
    local name="$1" a="$2" b="$3" games="$4" sb="$5" extra="$6"
    local dir="$OUT/games_$name"
    if [ -f "$OUT/$name.fit.json" ] || [ -f "$OUT/timing_$name.txt" ]; then echo "match $name done"; return 0; fi
    local t0=$(date +%s)
    python tools/run_elo_batch.py --label-a "${name%%_vs_*}" --spec-a "$a" --label-b "${name##*_vs_}" --spec-b "$b" \
        --outdir "$dir" --games "$games" --max-extra-games "$extra" --seed-base "$sb" \
        --mcts-sims 0 --raw-temperature-a 0 --raw-temperature-b 0 \
        --raw-end-turn-offset-a "$EO" --raw-end-turn-offset-b "$EO" \
        --persistent-workers --shared-inference --no-infer-compile --device cuda --jobs "$JOBS" \
        --time-budget-min 60 2>&1 | grep --line-buffered -v "wesnoth_core is not importable" | tee -a "$OUT/$name.log"
    local t1=$(date +%s)
    echo "$name: $((t1 - t0)) s, $(ls "$dir"/game_*.json 2>/dev/null | wc -l) games, $(decisive_results "$dir") decisive" | tee "$OUT/timing_$name.txt" | tee -a "$OUT/match.walls"
    cp "$dir"/.inference_server_*.json "$OUT/$name.server_stats.json" 2>/dev/null || true
    if [ "$games" -ge 100 ]; then
        python tools/elo_collect.py "$dir" --no-catalog --save-json "$OUT/$name.fit.json" 2>&1 | tee -a "$OUT/$name.log"
    fi
}
E="e$STOP_AFTER_EPOCH"
EO=$(python -c "import json; print(json.load(open('configs/reference_player.json'))['decode']['raw_end_turn_offset'])")
REF=$(python -c "import json; print(json.load(open('configs/reference_player.json'))['checkpoint_local'])")
ARM="training/checkpoints/vocab_$E.pt"
cp "$CKPT" "$ARM"
NAME="vocab_${E}_vs_obs8"
match "$NAME" "$ARM" "$REF" "$GAMES" 70000 1500
# A match short of its decisive games (failed games, a dead server, the
# time budget) runs once more in the same directory, which resumes it;
# a match still short is recorded as cut and its fit left unread.
if [ "$(decisive_results "$OUT/games_$NAME")" -lt "$GAMES" ]; then
    rm -f "$OUT/$NAME.fit.json" "$OUT/timing_$NAME.txt"
    match "$NAME" "$ARM" "$REF" "$GAMES" 70000 1500
fi
n_decisive=$(decisive_results "$OUT/games_$NAME")
if [ "$n_decisive" -lt "$GAMES" ]; then
    echo "MATCH_CUT: $n_decisive of $GAMES decisive games; the fit is not the verdict" | tee -a "$OUT/match.walls"
fi
kill $ESCROW_PID 2>/dev/null
touch "$OUT/ALL_DONE"
progress
escrow >> "$OUT/escrow.log" 2>&1
upload_small
echo UNIT_VOCAB_RETRAIN_DONE
stop_self
