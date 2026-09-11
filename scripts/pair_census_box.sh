#!/usr/bin/env bash
# Pair census (2026-09-11 night): seed2's epoch trained on 2,491,171
# pairs and the relset twin's on 2,825,379 with the same 16,650 files,
# the same recipe and the same seed; the manifest's winner actions
# plus end_turns plus the expected value-only states give 2.83M. Does
# the full-board corpus lose pairs silently in the trainer (encode and
# flush failures are logged at DEBUG)? The same first FILES files of
# the seeded order, one epoch each, through the trainer of the arm's
# checkout, with the trainer's logger at DEBUG: full board (pre-encoded
# here) against the relevant-set records. Runs after the trainer
# timings (TRAINER_TIMINGS_DONE). Records under /workspace/census/,
# uploaded to HF tier-b/pair_census_20260911/.
set -uo pipefail
OUT=/workspace/census
FILES="${FILES:-300}"
mkdir -p "$OUT"
cd /workspace/Wesnoth-AI
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 PATH="$HOME/.cargo/bin:$PATH"
ARCH="--d-model 384 --num-layers 8 --num-heads 12 --d-ff 1536"
if [ ! -f /workspace/encoded_full/PREENCODE_DONE ]; then
    python tools/preencode_corpus.py --dataset replays_dataset_imitation --out /workspace/encoded_full \
        --vocab-from /workspace/relset/fresh_vocab.pt --fog-hides-enemy-villages --workers 30 \
        2>&1 | grep --line-buffered -v "wesnoth_core is not importable" | tee "$OUT/preencode_full.log" | tail -2
    grep -q "PREENCODE_DONE" "$OUT/preencode_full.log" && touch /workspace/encoded_full/PREENCODE_DONE
fi
census() {                        # census NAME CORPUS EXTRA_FLAGS
    local name="$1" corpus="$2" extra="$3"
    python - "$name" "$corpus" "$FILES" $extra <<'EOF' 2>&1 | grep --line-buffered -v "wesnoth_core is not importable" > "$OUT/$name.log"
import logging, sys
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
for n in ("supervised_train", "tools.supervised_train", "__main__"):
    logging.getLogger(n).setLevel(logging.DEBUG)
name, corpus, files, extra = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
from tools.supervised_train import main
argv = ["supervised_train.py", "replays_dataset_imitation", "--checkpoint", f"/workspace/census/{name}.pt",
        "--imitation-config", "configs/imitation.json", "--d-model", "384", "--num-layers", "8",
        "--num-heads", "12", "--d-ff", "1536", "--epochs", "1", "--seed", "20260909", "--bs", "64",
        "--lr", "1e-4", "--device", "cuda", "--workers", "0", "--preencoded", corpus,
        "--eval-every", "100000000", "--ckpt-every", "1000000", "--log-every", "500",
        "--max-replays", files] + extra
sys.exit(main(argv))
EOF
    { echo "== $name"; grep -c "encode failed" "$OUT/$name.log"; grep -c "batch flush failed" "$OUT/$name.log";
      grep -c "file_error" "$OUT/$name.log"; grep "epoch accounting\|Training on" "$OUT/$name.log";
      grep "failed" "$OUT/$name.log" | sort | uniq -c | sort -rn | head -5; } >> "$OUT/summary.txt"
}
census full /workspace/encoded_full ""
census relset /workspace/encoded_relset "--relevant-set-hexes"
cat "$OUT/summary.txt"
python - "$OUT" <<'EOF'
import glob, os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for p in sorted(glob.glob(sys.argv[1] + "/*")):
    if os.path.isfile(p) and not p.endswith(".pt") and os.path.getsize(p) < 50_000_000:
        api.upload_file(path_or_fileobj=p, path_in_repo="tier-b/pair_census_20260911/" + os.path.basename(p),
                        repo_id="momom2/wesnoth-model-checkpoints")
print("uploaded", flush=True)
EOF
touch "$OUT/ALL_DONE"
echo PAIR_CENSUS_DONE
