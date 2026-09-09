#!/usr/bin/env bash
# The clean imitation seed (user order 2026-09-09): a 15M network
# trained FROM SCRATCH on the human corpus with every 2026-09-08 fix
# in place -- the deduplicated corpus (17,019 games), the manifest
# split, fog recorded per side, global feature 5 gated by fog (the
# default for a fresh network), pre-encoded records. Same recipe as
# the seed's imitation pass otherwise (configs/imitation.json, batch
# 64, lr 1e-4 with the cosine schedule over the epochs, value loss
# weight 1.0, 16 value states per game), several epochs because a
# fresh trunk needs them (the 2026-08-08 A/B: 3.449 holdout CE after
# one epoch from random against 3.107 warm).
#
# ON THE BOX after /workspace/{stage.tar.gz,.hf_token}:
#   bash clean_seed_retrain_box.sh
# Every stage writes as it runs; an escrow loop uploads the latest
# checkpoint, every epoch-end checkpoint, the log and the eval curve
# to HF tier-b/clean_seed_20260909/ every 30 minutes; re-entry after
# a box death resumes from the latest checkpoint.
set -uo pipefail
WORKDIR=/workspace
OUT=$WORKDIR/clean_seed
ENC=$WORKDIR/encoded_gated
EPOCHS="${EPOCHS:-5}"
RUN_SEED="${RUN_SEED:-20260909}"
WORKERS="${WORKERS:-22}"
HF_DIR=tier-b/clean_seed_20260909
ARCH="--d-model 384 --num-layers 8 --num-heads 12 --d-ff 1536"
EVAL="--eval-every 50000 --eval-pairs 1200 --eval-pairs-per-game 8 --eval-sample-seed 0"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
mkdir -p "$OUT"
cd $WORKDIR
if [ ! -d Wesnoth-AI/tools ]; then
    mkdir -p Wesnoth-AI && tar xzf $WORKDIR/stage.tar.gz -C Wesnoth-AI
fi
cd Wesnoth-AI
export HF_TOKEN="$(cat $WORKDIR/.hf_token)"
python -m pip install -q huggingface_hub psutil 2>&1 | grep -v "WARNING: Running pip" | tail -1 || true

if [ ! -f "$OUT/STAGED" ]; then
python - <<'EOF' || { echo "staging failed" >&2; exit 1; }
import json, pathlib, tarfile
from huggingface_hub import hf_hub_download
dst = pathlib.Path("training/checkpoints"); dst.mkdir(parents=True, exist_ok=True)
p = hf_hub_download("momom2/wesnoth-model-checkpoints", "tier-b/a3/seed_imit_tierb_start.pt")
(dst / "seed.pt").write_bytes(pathlib.Path(p).read_bytes())
p = hf_hub_download("momom2/wesnoth-model-checkpoints",
                    "tier-b/replays_dataset_imitation_dedup_20260908.tar.gz")
with tarfile.open(p, "r:gz") as tf:
    tf.extractall(".")
rows = [json.loads(l) for l in open("replays_dataset_imitation/manifest.jsonl")]
print("corpus", len(rows), "games, holdout", sum(1 for r in rows if r.get("holdout")),
      "fog off", sum(1 for r in rows if r.get("fog") is False), flush=True)
EOF
touch "$OUT/STAGED"
fi
{ nproc --all; free -g | head -2; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } > "$OUT/box.txt" 2>&1

# The vocab a fresh network starts from (unit_stats seeding, the same
# dicts the trainer builds), saved so the pre-encoder can read it.
if [ ! -f "$OUT/fresh_vocab.pt" ]; then
python - <<'EOF' || { echo "vocab failed" >&2; exit 1; }
import torch
from wesnoth_ai.encoder import GameStateEncoder
from tools.supervised_train import _seed_vocab_from_unit_stats
enc = GameStateEncoder(d_model=32)
_seed_vocab_from_unit_stats(enc, "unit_stats.json")
torch.save({"unit_type_to_id": dict(enc.unit_type_to_id),
            "faction_to_id": dict(enc.faction_to_id)}, "/workspace/clean_seed/fresh_vocab.pt")
seed = torch.load("training/checkpoints/seed.pt", map_location="cpu", weights_only=False)
same = (dict(seed.get("unit_type_to_id", {})) == dict(enc.unit_type_to_id)
        and dict(seed.get("faction_to_id", {})) == dict(enc.faction_to_id))
print("fresh vocab:", len(enc.unit_type_to_id), "types,", len(enc.faction_to_id),
      "factions; identical to the seed's:", same, flush=True)
EOF
fi

if [ ! -f "$ENC/PREENCODE_DONE" ]; then
    python tools/preencode_corpus.py --dataset replays_dataset_imitation --out "$ENC" \
        --vocab-from "$OUT/fresh_vocab.pt" --fog-hides-enemy-villages --workers "$WORKERS" \
        2>&1 | grep -v "wesnoth_core is not importable" | tee -a "$OUT/preencode.log" | tail -3
    grep -q "PREENCODE_DONE" "$OUT/preencode.log" && touch "$ENC/PREENCODE_DONE" \
        || { echo "pre-encoding failed" >&2; exit 1; }
fi

escrow() {
python - <<'EOF'
import glob, os, shutil, time
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
out = "/workspace/clean_seed"
hf_dir = os.environ.get("HF_DIR", "tier-b/clean_seed_20260909")
files = ["train.log", "arm_eval.jsonl", "preencode.log", "box.txt"]
files += sorted(os.path.basename(p) for p in glob.glob(out + "/arm_epoch*.pt"))
for name in files:
    p = os.path.join(out, name)
    if not os.path.exists(p):
        continue
    marker = p + ".escrowed"
    if name.endswith(".pt") and os.path.exists(marker):
        continue
    api.upload_file(path_or_fileobj=p, path_in_repo=f"{hf_dir}/{name}",
                    repo_id="momom2/wesnoth-model-checkpoints")
    open(marker, "w").close()
latest = os.path.join(out, "arm.pt")
if os.path.exists(latest):
    snap = os.path.join(out, "arm_latest_snapshot.pt")
    shutil.copyfile(latest, snap)                # a copy, so the trainer's next save cannot tear it
    api.upload_file(path_or_fileobj=snap, path_in_repo=f"{hf_dir}/arm_latest.pt",
                    repo_id="momom2/wesnoth-model-checkpoints")
print("ESCROW_OK", time.strftime("%Y-%m-%d %H:%M"), flush=True)
EOF
}
export HF_DIR
( while [ ! -f "$OUT/DONE" ]; do sleep 1800; escrow >> "$OUT/escrow.log" 2>&1; done ) &
ESCROW_PID=$!

if [ ! -f "$OUT/DONE" ]; then
    RESUME=""
    [ -f "$OUT/arm.pt" ] && RESUME="--resume $OUT/arm.pt"
    python tools/supervised_train.py replays_dataset_imitation \
        --checkpoint "$OUT/arm.pt" $RESUME \
        --imitation-config configs/imitation.json $ARCH \
        --epochs "$EPOCHS" --seed "$RUN_SEED" \
        --bs 64 --lr 1e-4 --device cuda --workers 0 --preencoded "$ENC" \
        $EVAL --ckpt-every 2000 --log-every 100 \
        2>&1 | grep -v "wesnoth_core is not importable" | tee -a "$OUT/train.log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ] || [ ! -f "$OUT/arm.pt" ]; then
        echo "training FAILED rc=$rc" >&2
        escrow >> "$OUT/escrow.log" 2>&1
        exit 1
    fi
    touch "$OUT/DONE"
fi
kill $ESCROW_PID 2>/dev/null
escrow >> "$OUT/escrow.log" 2>&1
python tools/analysis/value_head_by_phase.py --checkpoint "$OUT/arm.pt" --jobs 20 --device cuda \
    --out "$OUT/phase_clean_seed.json" 2>&1 | grep -v "wesnoth_core is not importable" | tail -10
python - <<'EOF'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for name in ("phase_clean_seed.json", "phase_clean_seed.md"):
    p = "/workspace/clean_seed/" + name
    if os.path.exists(p):
        api.upload_file(path_or_fileobj=p, path_in_repo=f"{os.environ['HF_DIR']}/{name}",
                        repo_id="momom2/wesnoth-model-checkpoints")
EOF
touch "$OUT/ALL_DONE"
echo CLEAN_SEED_DONE
