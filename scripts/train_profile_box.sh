#!/usr/bin/env bash
# The imitation trainer timed or profiled on a box (2026-09-11, user:
# "if there is Python, there is room for optimization"): a short
# from-scratch run on a pre-encoded corpus already on the box (the
# relevant-set records of the relset arm; the loop is the same in both
# bases), no holdout probe, so the number is the training loop's own.
#   MODE=time   plain run: pairs per second from the trainer's log
#   MODE=pyspy  the same run sampled in parent mode with idle time
#               kept, so GPU waits show as the frames that block on them
#   CODE        the checkout to run (default /workspace/Wesnoth-AI)
# Records under $OUT, uploaded to HF $HF_DIR. Needs an idle GPU: a run
# beside a training arm is starved of GPU memory and means nothing.
set -uo pipefail
MODE="${MODE:-time}"
CODE="${CODE:-/workspace/Wesnoth-AI}"
OUT="${OUT:-/workspace/trainprof}"
HF_DIR="${HF_DIR:-tier-b/train_profile_20260911}"
ENC="${ENC:-/workspace/encoded_relset}"
PAIRS="${PAIRS:-30000}"
mkdir -p "$OUT"
cd "$CODE"
export HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 PATH="$HOME/.cargo/bin:$PATH"
ARCH="--d-model 384 --num-layers 8 --num-heads 12 --d-ff 1536"
TRAIN="python tools/supervised_train.py replays_dataset_imitation \
    --checkpoint $OUT/arm.pt --imitation-config configs/imitation.json $ARCH --relevant-set-hexes \
    --epochs 1 --seed 20260909 --bs 64 --lr 1e-4 --device cuda --workers 0 --preencoded $ENC \
    --eval-every 100000000 --ckpt-every 1000000 --log-every 50 --max-pairs $PAIRS"
t0=$(date +%s)
if [ "$MODE" = pyspy ]; then
    python -m pip install -q py-spy >/dev/null 2>&1
    py-spy record --duration 400 --format raw --idle --nonblocking -o "$OUT/pyspy_train.txt" -- \
        $TRAIN 2>&1 | grep --line-buffered -v "wesnoth_core is not importable" > "$OUT/train.log"
    python tools/pyspy_summary.py "$OUT/pyspy_train.txt" --top 25 > "$OUT/pyspy_train.summary.txt" 2>&1 || true
else
    $TRAIN 2>&1 | grep --line-buffered -v "wesnoth_core is not importable" > "$OUT/train.log"
fi
{ echo "mode $MODE code $CODE wall $(( $(date +%s) - t0 )) s";
  grep -o "pairs=[0-9]* rate=[0-9.]*/s wall=[0-9.]*m" "$OUT/train.log" | tail -3; } > "$OUT/wall.txt"
cat "$OUT/wall.txt"
python - "$OUT" "$HF_DIR" <<'EOF'
import glob, os, sys
from huggingface_hub import HfApi
out, hf_dir = sys.argv[1], sys.argv[2]
api = HfApi(token=os.environ["HF_TOKEN"])
for p in sorted(glob.glob(out + "/*")):
    if os.path.isfile(p) and not p.endswith(".pt") and os.path.getsize(p) < 50_000_000:
        api.upload_file(path_or_fileobj=p, path_in_repo=hf_dir + "/" + os.path.basename(p),
                        repo_id="momom2/wesnoth-model-checkpoints")
print("uploaded", flush=True)
EOF
touch "$OUT/ALL_DONE"
echo TRAIN_PROFILE_DONE
