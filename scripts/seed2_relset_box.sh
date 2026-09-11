#!/usr/bin/env bash
# Plan 1.4, the relevant-set arm, judged the clean way (user order
# 2026-09-11): seed2's twin trained FROM SCRATCH in the relevant-set
# hex basis (`--relevant-set-hexes`, docs/model_cost_study_20260905.md
# section 2: 2.7x fewer tokens per leaf), the same recipe as seed2 to
# the letter (configs/imitation.json, batch 64, lr 1e-4, cosine over
# 4 epochs, stopped after STOP_AFTER_EPOCH passes, the deduplicated
# corpus with the manifest split, fog gate on, pre-encoded records;
# user order 2026-09-11: one pass, judged against seed2's own one-pass
# checkpoint SEED2_HF), then:
#   the per-phase value evaluation of the checkpoint;
#   seed2_relset against seed2, PURE raw:t0, 800 decisive games
#     (each side served in its own basis by its own inference server);
#   40-game self-timings of both, for the throughput number.
# Unattended: fetched and started by the box's onstart through HF.
# Every 30 minutes the escrow uploads the latest checkpoint, epoch
# checkpoints, log and curve to HF tier-b/seed2_relset_20260911/.
# Re-entry resumes from the latest checkpoint.
set -uo pipefail
WORKDIR=/workspace
OUT=$WORKDIR/relset
ENC=$WORKDIR/encoded_relset
EPOCHS="${EPOCHS:-4}"
STOP_AFTER_EPOCH="${STOP_AFTER_EPOCH:-1}"
SEED2_HF="${SEED2_HF:-tier-b/clean_seed_20260909/arm_epoch$((STOP_AFTER_EPOCH - 1)).pt}"
RUN_SEED="${RUN_SEED:-20260909}"
WORKERS="${WORKERS:-30}"
GAMES="${GAMES:-800}"
JOBS="${JOBS:-20}"
export HF_DIR=tier-b/seed2_relset_20260911
ARCH="--d-model 384 --num-layers 8 --num-heads 12 --d-ff 1536"
EVAL="--eval-every 50000 --eval-pairs 1200 --eval-pairs-per-game 8 --eval-sample-seed 0"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export HF_HUB_DISABLE_XET=1
mkdir -p "$OUT"
cd $WORKDIR
export HF_TOKEN="$(tr -d '\r\n' < $WORKDIR/.hf_token)"
python -m pip install -q huggingface_hub psutil 2>&1 | grep -v "WARNING: Running pip" | tail -1 || true
if [ ! -d Wesnoth-AI/tools ]; then
python - <<'EOF'
import os, tarfile
from huggingface_hub import hf_hub_download
p = hf_hub_download("momom2/wesnoth-model-checkpoints", "tier-b/staging/stage_20260911b.tar.gz")
os.makedirs("/workspace/Wesnoth-AI", exist_ok=True)
with tarfile.open(p, "r:gz") as tf:
    tf.extractall("/workspace/Wesnoth-AI")
print("code staged", flush=True)
EOF
fi
cd Wesnoth-AI

if [ ! -f "$OUT/STAGED" ]; then
python - <<'EOF' || { echo "staging failed" >&2; exit 1; }
import json, pathlib, tarfile
from huggingface_hub import hf_hub_download
dst = pathlib.Path("training/checkpoints"); dst.mkdir(parents=True, exist_ok=True)
for remote, local in (("tier-b/seed2.pt", "seed2.pt"),):
    p = hf_hub_download("momom2/wesnoth-model-checkpoints", remote)
    (dst / local).write_bytes(pathlib.Path(p).read_bytes())
p = hf_hub_download("momom2/wesnoth-model-checkpoints",
                    "tier-b/replays_dataset_imitation_dedup_20260908.tar.gz")
with tarfile.open(p, "r:gz") as tf:
    tf.extractall(".")
rows = [json.loads(l) for l in open("replays_dataset_imitation/manifest.jsonl")]
print("corpus", len(rows), "games, holdout", sum(1 for r in rows if r.get("holdout")), flush=True)
EOF
touch "$OUT/STAGED"
fi
if ! python -c "import wesnoth_core" 2>/dev/null; then
    command -v cc >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq gcc) >/dev/null 2>&1 || true
    command -v cargo >/dev/null 2>&1 || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1
    export PATH="$HOME/.cargo/bin:$PATH"
    python -m pip install -q maturin >/dev/null 2>&1
    python -m pip install -q rust/wesnoth_core 2>&1 | tail -1
fi
{ nproc --all; cat /sys/fs/cgroup/cpu.max 2>/dev/null; free -g | head -2; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; python -c "import wesnoth_core; print('wesnoth_core built')" 2>&1; } > "$OUT/box.txt" 2>&1

if [ ! -f "$OUT/fresh_vocab.pt" ]; then
python - <<'EOF' || { echo "vocab failed" >&2; exit 1; }
import pathlib
import torch
from wesnoth_ai.encoder import GameStateEncoder
from tools.supervised_train import _seed_vocab_from_unit_stats
enc = GameStateEncoder(d_model=32)
_seed_vocab_from_unit_stats(enc, pathlib.Path("unit_stats.json"))
torch.save({"unit_type_to_id": dict(enc.unit_type_to_id),
            "faction_to_id": dict(enc.faction_to_id)}, "/workspace/relset/fresh_vocab.pt")
print("fresh vocab:", len(enc.unit_type_to_id), "types,", len(enc.faction_to_id), "factions", flush=True)
EOF
fi

if [ ! -f "$ENC/PREENCODE_DONE" ]; then
    python tools/preencode_corpus.py --dataset replays_dataset_imitation --out "$ENC" \
        --vocab-from "$OUT/fresh_vocab.pt" --fog-hides-enemy-villages --relevant-set-hexes \
        --workers "$WORKERS" \
        2>&1 | grep --line-buffered -v "wesnoth_core is not importable" | tee -a "$OUT/preencode.log" | tail -3
    grep -q "PREENCODE_DONE" "$OUT/preencode.log" && touch "$ENC/PREENCODE_DONE" \
        || { echo "pre-encoding failed" >&2; exit 1; }
fi

escrow() {
python - <<'EOF'
import glob, os, shutil, time
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
out = "/workspace/relset"
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


files = ["train.log", "arm_eval.jsonl", "preencode.log", "box.txt", "progress.txt",
         "phase_seed2_relset.json", "phase_seed2_relset.md", "STOPPED_AFTER_EPOCH"]
files += sorted(os.path.basename(p) for p in glob.glob(out + "/arm_epoch*.pt"))
files += sorted(os.path.basename(p) for p in glob.glob(out + "/*.fit.json"))
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

# Stop after STOP_AFTER_EPOCH like seed2: kill the trainer once that
# epoch's checkpoint and holdout eval are written.
( until grep -q "EVAL\[epoch$((STOP_AFTER_EPOCH - 1))-end\]" "$OUT/train.log" 2>/dev/null; do sleep 20; done
  for p in $(pgrep -f "^python tools/supervised_train"); do kill $p; done; sleep 15
  pgrep -f "^python tools/supervised_train" >/dev/null && pkill -9 -f "^python tools/supervised_train"
  echo "trainer stopped $(date -u +%H:%M) after epoch $STOP_AFTER_EPOCH" > "$OUT/STOPPED_AFTER_EPOCH" ) &

if [ -n "${RESUME_HF:-}" ] && [ ! -f "$OUT/arm.pt" ] && [ ! -f "$OUT/DONE" ]; then
python - <<EOF || { echo "resume download failed" >&2; exit 1; }
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
    python tools/supervised_train.py replays_dataset_imitation \
        --checkpoint "$OUT/arm.pt" $RESUME \
        --imitation-config configs/imitation.json $ARCH --relevant-set-hexes \
        --epochs "$EPOCHS" --seed "$RUN_SEED" \
        --bs 64 --lr 1e-4 --device cuda --workers 0 --preencoded "$ENC" \
        $EVAL --ckpt-every 2000 --log-every 100 \
        2>&1 | grep --line-buffered -v "wesnoth_core is not importable" | tee -a "$OUT/train.log"
    if [ ! -f "$OUT/arm_epoch$((STOP_AFTER_EPOCH - 1)).pt" ]; then
        echo "training ended without the epoch-$STOP_AFTER_EPOCH checkpoint" >&2
        escrow >> "$OUT/escrow.log" 2>&1; exit 1
    fi
    touch "$OUT/DONE"
fi
CKPT="$OUT/arm_epoch$((STOP_AFTER_EPOCH - 1)).pt"
cp "$CKPT" training/checkpoints/seed2_relset.pt
escrow >> "$OUT/escrow.log" 2>&1

[ -f "$OUT/phase_seed2_relset.json" ] || python tools/analysis/value_head_by_phase.py --checkpoint "$CKPT" \
    --jobs 20 --device cuda --out "$OUT/phase_seed2_relset.json" 2>&1 | grep -v "wesnoth_core is not importable" | tail -10

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
    echo "$name: $((t1 - t0)) s, $(ls "$dir"/game_*.json 2>/dev/null | wc -l) games, $(decisive_results "$dir") decisive" | tee "$OUT/timing_$name.txt"
    cp "$dir"/.inference_server_*.json "$OUT/$name.server_stats.json" 2>/dev/null || true
    if [ "$games" -ge 100 ]; then
        python tools/elo_collect.py "$dir" --no-catalog --save-json "$OUT/$name.fit.json" 2>&1 | tee -a "$OUT/$name.log"
    fi
}
E="e$STOP_AFTER_EPOCH"
SEED2="training/checkpoints/seed2_$E.pt"
RELSET="training/checkpoints/seed2_relset_$E.pt"
cp "$CKPT" "$RELSET"
[ -f "$SEED2" ] || python - "$SEED2_HF" "$SEED2" <<'EOF'
import shutil, sys
from huggingface_hub import hf_hub_download
shutil.copyfile(hf_hub_download("momom2/wesnoth-model-checkpoints", sys.argv[1]), sys.argv[2])
print("opponent staged:", sys.argv[1], flush=True)
EOF
match "seed2_relset_${E}_vs_seed2_$E" "$RELSET" "$SEED2" "$GAMES" 60000 1500
match "relself_${E}_vs_relself_$E" "$RELSET" "$RELSET" 40 20000 0
match "seed2self_${E}_vs_seed2self_$E" "$SEED2" "$SEED2" 40 20000 0
kill $ESCROW_PID 2>/dev/null
touch "$OUT/ALL_DONE"
progress
escrow >> "$OUT/escrow.log" 2>&1
python - <<'EOF'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for name in ("seed2_relset_vs_seed2.log", "relself_vs_relself.server_stats.json", "seed2self_vs_seed2self.server_stats.json", "escrow.log"):
    p = "/workspace/relset/" + name
    if os.path.exists(p):
        api.upload_file(path_or_fileobj=p, path_in_repo=f"{os.environ['HF_DIR']}/{name}", repo_id="momom2/wesnoth-model-checkpoints")
print("final uploads done", flush=True)
EOF
echo SEED2_RELSET_DONE
