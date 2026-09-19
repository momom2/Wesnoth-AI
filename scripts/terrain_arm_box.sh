#!/usr/bin/env bash
# The terrain-set arm: the reference player's recipe from scratch with
# the hex's terrain carried as its full set from the engine's aliases
# (`--terrain-multi-hot`, encoder.terrain_tokens; the one-class view
# labelled 86% of the Ladder pool's forest-overlay hexes as something
# other than forest, tools/analysis/hide_cover_census.py), everything
# else `scripts/seed2_relset_box.sh` to the letter: the relevant-set
# basis, configs/imitation.json, batch 64, lr 1e-4, cosine over 4
# epochs, stopped after STOP_AFTER_EPOCH passes, the deduplicated
# corpus with the manifest split, fog gate on, pre-encoded records,
# the same run seed. Then:
#   the per-phase value evaluation of the checkpoint;
#   the arm against the reference player (`relset`, the same recipe
#     at one pass), PURE raw:t0, 800 decisive games, each side served
#     in its own encoding by its own inference server;
#   40-game self-timings of both, for the throughput number.
# Bars and predictions: docs/terrain_multi_hot_prereg_20260919.md.
# Unattended: fetched and started by the box's onstart through HF.
# Every 30 minutes the escrow uploads the latest checkpoint, epoch
# checkpoints, log and curve to HF $HF_DIR. Re-entry resumes from the
# latest checkpoint. Every exit, clean or not, leaves ALL_DONE on HF
# so the watcher on the laptop pulls and destroys the box.
set -uo pipefail
WORKDIR=/workspace
OUT=$WORKDIR/terrain
ENC=$WORKDIR/encoded_terrain
STAGE="${STAGE:-tier-b/staging/stage_20260919b.tar.gz}"
EPOCHS="${EPOCHS:-4}"
STOP_AFTER_EPOCH="${STOP_AFTER_EPOCH:-1}"
REF_HF="${REF_HF:-tier-b/seed2_relset_20260911/arm_epoch0.pt}"
RUN_SEED="${RUN_SEED:-20260909}"
WORKERS="${WORKERS:-30}"
GAMES="${GAMES:-800}"
JOBS="${JOBS:-20}"
export HF_DIR="${HF_DIR:-tier-b/terrain_multi_hot_20260919}"
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
for p in sorted(glob.glob("/workspace/terrain/*")):
    if os.path.isfile(p) and os.path.getsize(p) < 50_000_000 and not p.endswith(".pt"):
        api.upload_file(path_or_fileobj=p, path_in_repo=os.environ["HF_DIR"] + "/" + os.path.basename(p),
                        repo_id="momom2/wesnoth-model-checkpoints")
EOF
}
die() {                          # die REASON: the run stops, the box does not idle
    echo "FAILED: $1" | tee -a "$OUT/FAILED"
    touch "$OUT/ALL_DONE"
    upload_small
    exit 1
}

if [ ! -d Wesnoth-AI/tools ]; then
python - "$STAGE" <<'EOF' || die "code staging"
import os, sys, tarfile
from huggingface_hub import hf_hub_download
p = hf_hub_download("momom2/wesnoth-model-checkpoints", sys.argv[1])
os.makedirs("/workspace/Wesnoth-AI", exist_ok=True)
with tarfile.open(p, "r:gz") as tf:
    tf.extractall("/workspace/Wesnoth-AI")
print("code staged", flush=True)
EOF
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
python -c "import wesnoth_core; p = wesnoth_core.__phase__; print('wheel phase', p); \
assert p >= 10, f'wheel is phase {p}; the source declares 10'" | tee -a "$OUT/build.log" \
    || die "wheel build (build.log)"
{ nproc --all; grep -m1 "model name" /proc/cpuinfo; cat /sys/fs/cgroup/cpu.max 2>/dev/null; free -g | head -2;
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader;
  python -c "import torch; print('torch', torch.__version__)";
  python tools/kernel_status.py 2>/dev/null | tail -8; } > "$OUT/box.txt" 2>&1
upload_small

# ---- the encoder under the flag against the Rust kernels, recorded before anything trains
python -m pytest tests/test_terrain_multi_hot.py tests/test_rust_encode_raw.py tests/test_game_core.py \
    -q -p no:cacheprovider -m "" > "$OUT/tests_terrain.log" 2>&1
tail -3 "$OUT/tests_terrain.log"
grep -q " failed" "$OUT/tests_terrain.log" && echo "TESTS FAILED (tests_terrain.log); the run goes on, the verdict carries the flag" | tee -a "$OUT/tests_terrain.log"
upload_small

if [ ! -f "$OUT/STAGED" ]; then
python - "$REF_HF" <<'EOF' || die "corpus or reference staging"
import json, pathlib, sys, tarfile
from huggingface_hub import hf_hub_download
dst = pathlib.Path("training/checkpoints"); dst.mkdir(parents=True, exist_ok=True)
p = hf_hub_download("momom2/wesnoth-model-checkpoints", sys.argv[1])
(dst / "relset.pt").write_bytes(pathlib.Path(p).read_bytes())
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
import pathlib
import torch
from wesnoth_ai.encoder import GameStateEncoder
from tools.supervised_train import _seed_vocab_from_unit_stats
enc = GameStateEncoder(d_model=32)
_seed_vocab_from_unit_stats(enc, pathlib.Path("unit_stats.json"))
torch.save({"unit_type_to_id": dict(enc.unit_type_to_id),
            "faction_to_id": dict(enc.faction_to_id)}, "/workspace/terrain/fresh_vocab.pt")
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
out = "/workspace/terrain"
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


files = ["train.log", "arm_eval.jsonl", "preencode.log", "box.txt", "progress.txt", "build.log",
         "tests_terrain.log", "phase_terrain.json", "phase_terrain.md", "STOPPED_AFTER_EPOCH", "match.walls"]
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

[ -f "$OUT/phase_terrain.json" ] || python tools/analysis/value_head_by_phase.py --checkpoint "$CKPT" \
    --jobs 20 --device cuda --out "$OUT/phase_terrain.json" 2>&1 | grep -v "wesnoth_core is not importable" | tail -10

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
        --persistent-workers --shared-inference --no-infer-compile --device cuda --jobs "$JOBS" \
        --time-budget-min 150 2>&1 | grep --line-buffered -v "wesnoth_core is not importable" | tee -a "$OUT/$name.log"
    local t1=$(date +%s)
    echo "$name: $((t1 - t0)) s, $(ls "$dir"/game_*.json 2>/dev/null | wc -l) games, $(decisive_results "$dir") decisive" | tee "$OUT/timing_$name.txt" | tee -a "$OUT/match.walls"
    cp "$dir"/.inference_server_*.json "$OUT/$name.server_stats.json" 2>/dev/null || true
    if [ "$games" -ge 100 ]; then
        python tools/elo_collect.py "$dir" --no-catalog --save-json "$OUT/$name.fit.json" 2>&1 | tee -a "$OUT/$name.log"
    fi
}
E="e$STOP_AFTER_EPOCH"
ARM="training/checkpoints/terrain_$E.pt"
REF="training/checkpoints/relset.pt"
cp "$CKPT" "$ARM"
match "terrain_${E}_vs_relset" "$ARM" "$REF" "$GAMES" 61000 1500
# Self-timings: the batch runner keys result files by label, so a
# side against itself needs two labels (2026-09-19: one label refused).
match "terrain_${E}_vs_terrainmirror_$E" "$ARM" "$ARM" 40 62000 0
match "relset_vs_relsetmirror" "$REF" "$REF" 40 62000 0
kill $ESCROW_PID 2>/dev/null
touch "$OUT/ALL_DONE"
progress
escrow >> "$OUT/escrow.log" 2>&1
upload_small
echo TERRAIN_ARM_DONE
