#!/usr/bin/env bash
# Value-head study arms (docs/value_head_study_20260907.md, user's plan
# 2026-09-08). ON THE BOX after /workspace/{stage.tar.gz,.hf_token}:
#   value_head_plus_1        the seed's recipe, half an epoch, repaired corpus
#   value_head_plus_material same, the value head also reading material
# then the per-phase evaluation of the seed, the earlier control arm
# (unrepaired corpus), and both new arms, plus each arm's holdout eval.
# Every stage writes as it runs (train.log, arm_eval.jsonl, partial
# json) and touches a DONE marker; re-entry skips finished stages.
set -uo pipefail
WORKDIR=/workspace
OUT=$WORKDIR/vh
PAIRS="${PAIRS:-1260000}"
RUN_SEED="${RUN_SEED:-20260905}"
WORKERS="${WORKERS:-14}"
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
import pathlib, tarfile
from huggingface_hub import hf_hub_download
dst = pathlib.Path("training/checkpoints"); dst.mkdir(parents=True, exist_ok=True)
for remote, local in (("tier-b/a3/seed_imit_tierb_start.pt", "seed.pt"),
                      ("tier-b/relset_arms_20260905/control_arm.pt", "control_arm_20260905.pt")):
    p = hf_hub_download("momom2/wesnoth-model-checkpoints", remote)
    (dst / local).write_bytes(pathlib.Path(p).read_bytes())
    print("staged", local, (dst / local).stat().st_size, flush=True)
p = hf_hub_download("momom2/wesnoth-model-checkpoints",
                    "tier-b/replays_dataset_imitation_dedup_20260908.tar.gz")
with tarfile.open(p, "r:gz") as tf:
    tf.extractall(".")
import json
rows = [json.loads(l) for l in open("replays_dataset_imitation/manifest.jsonl")]
print("corpus", len(rows), "games, fog off", sum(1 for r in rows if r.get("fog") is False), flush=True)
EOF
touch "$OUT/STAGED"
fi
{ nproc; free -g | head -2; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } > "$OUT/box.txt" 2>&1

run_arm() {                       # run_arm NAME [extra flags...]
    local name="$1"; shift
    local dir="$OUT/$name"
    if [ -f "$dir/DONE" ]; then echo "arm $name already done"; return 0; fi
    mkdir -p "$dir"
    python tools/supervised_train.py replays_dataset_imitation \
        --checkpoint "$dir/arm.pt" --init-from training/checkpoints/seed.pt \
        --imitation-config configs/imitation.json $ARCH \
        --epochs 1 --max-pairs "$PAIRS" --seed "$RUN_SEED" \
        --bs 64 --lr 1e-4 --device cuda --workers "$WORKERS" \
        $EVAL --ckpt-every 2000 --log-every 100 \
        "$@" 2>&1 | tee -a "$dir/train.log"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ] || [ ! -f "$dir/arm.pt" ]; then
        echo "arm $name FAILED rc=$rc" >&2; return 1
    fi
    touch "$dir/DONE"
}

phase_eval() {                    # phase_eval NAME CKPT
    local name="$1" ckpt="$2"
    if [ -f "$OUT/phase_$name.json" ]; then echo "phase eval $name already done"; return 0; fi
    python tools/analysis/value_head_by_phase.py --checkpoint "$ckpt" --jobs 20 --device cuda \
        --out "$OUT/phase_$name.json" 2>&1 | grep -v "wesnoth_core is not importable" | tail -12
}

run_arm value_head_plus_1
run_arm value_head_plus_material --value-material
phase_eval seed training/checkpoints/seed.pt
phase_eval control_arm_20260905 training/checkpoints/control_arm_20260905.pt
phase_eval value_head_plus_1 "$OUT/value_head_plus_1/arm.pt"
phase_eval value_head_plus_material "$OUT/value_head_plus_material/arm.pt"
touch "$OUT/ALL_DONE"
echo VH_ARMS_DONE
