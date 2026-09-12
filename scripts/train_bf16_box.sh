#!/usr/bin/env bash
# bf16 training timed against fp32 (2026-09-12, user order "make
# training work in bf16"): the full-board corpus pre-encoded from the
# reference checkpoint's vocab, then 30k pairs from scratch with the
# batched trainer in fp32 and under --bf16 (scripts/train_profile_box.sh
# MODE=time), on an idle GPU. Expects the code staged, the replay
# corpus under replays_dataset_imitation and relset.pt staged. Records
# under /workspace/bf16/, uploaded to HF $HF_DIR.
set -uo pipefail
OUT=/workspace/bf16
HF_DIR="${HF_DIR:-tier-b/train_bf16_20260912}"
mkdir -p "$OUT"
cd /workspace/Wesnoth-AI
export HF_TOKEN="$(tr -d '\r\n' < /workspace/.hf_token)" HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 PATH="$HOME/.cargo/bin:$PATH"
ENC=/workspace/encoded_full_fresh
if [ ! -f "$ENC/PREENCODE_DONE" ]; then
python - <<'EOF2'
import pathlib
import torch
from wesnoth_ai.encoder import GameStateEncoder
from tools.supervised_train import _seed_vocab_from_unit_stats
enc = GameStateEncoder(d_model=32)
_seed_vocab_from_unit_stats(enc, pathlib.Path("unit_stats.json"))
torch.save({"unit_type_to_id": dict(enc.unit_type_to_id), "faction_to_id": dict(enc.faction_to_id)},
           "/workspace/fresh_vocab.pt")
print("fresh vocab:", len(enc.unit_type_to_id), "types", flush=True)
EOF2
    python tools/preencode_corpus.py --dataset replays_dataset_imitation --out "$ENC" \
        --vocab-from /workspace/fresh_vocab.pt --fog-hides-enemy-villages --workers "$(nproc --all)" \
        2>&1 | grep --line-buffered -v "wesnoth_core is not importable" | tee "$OUT/preencode.log" | tail -2
    grep -q "PREENCODE_DONE" "$OUT/preencode.log" && touch "$ENC/PREENCODE_DONE"
fi
for run in fp32 bf16; do
    extra=$([ "$run" = bf16 ] && echo "--bf16" || echo "")
    MODE=time CODE=/workspace/Wesnoth-AI OUT="$OUT/$run" HF_DIR="$HF_DIR/$run" ENC="$ENC" BASIS="" EXTRA="$extra" PAIRS=30000 \
        bash /workspace/train_profile_box.sh > "$OUT/$run.log" 2>&1
    cat "$OUT/$run/wall.txt"
    grep "avg_loss" "$OUT/$run/train.log" | tail -2 | cut -c1-140
    nvidia-smi --query-gpu=memory.used --format=csv,noheader >> "$OUT/$run/wall.txt"
done
python - "$OUT" "$HF_DIR" <<'EOF'
import glob, os, sys
from huggingface_hub import HfApi
out, hf_dir = sys.argv[1], sys.argv[2]
api = HfApi(token=os.environ["HF_TOKEN"])
for p in sorted(glob.glob(out + "/*")):
    if os.path.isfile(p) and os.path.getsize(p) < 50_000_000:
        api.upload_file(path_or_fileobj=p, path_in_repo=hf_dir + "/" + os.path.basename(p),
                        repo_id="momom2/wesnoth-model-checkpoints")
print("uploaded", flush=True)
EOF
touch "$OUT/ALL_DONE"
echo TRAIN_BF16_DONE
