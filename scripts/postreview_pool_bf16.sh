#!/usr/bin/env bash
# The pool's serve-process arms in the COMMITTED configuration --
# az_loop's defaults: bf16 inference, the packed varlen trunk, packed
# embed. scripts/postreview_box.sh phase 2 ran the same pair at
# bench_pool's own defaults (fp32, unpacked), where the GPU owns 92% of
# the serve time and a second process has no room to pay; this pair
# asks the question where it was meant to be asked. Runs on the same
# box after that script's ALL_DONE marker; records go to the same HF
# folder as pool_bf16_p{1,2}.json and pool_bf16_verdict.txt.
set -u
OUT=/workspace/postreview
HF_DIR=tier-b/postreview_20260914
GAMES="${GAMES:-48}"
SIMS="${SIMS:-32}"
LEAF_BATCH="${LEAF_BATCH:-16}"
MAX_TURNS="${MAX_TURNS:-30}"
DPH="${DPH:-0.34}"

while [ ! -f "$OUT/ALL_DONE" ]; do sleep 20; done
cd /workspace/Wesnoth-AI
export HF_TOKEN="$(tr -d '\r\n' < /workspace/.hf_token)" HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export HF_DIR

upload() {
    python - <<'PY' 2>/dev/null || true
import os, glob
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for p in sorted(glob.glob("/workspace/postreview/pool_bf16_*")) + ["/workspace/postreview/ALL_DONE_BF16"]:
    if os.path.isfile(p) and os.path.getsize(p) < 50_000_000:
        api.upload_file(path_or_fileobj=p,
                        path_in_repo=os.environ["HF_DIR"] + "/" + os.path.basename(p),
                        repo_id="momom2/wesnoth-model-checkpoints")
PY
}

for np_ in 1 2; do
    echo "=== serve_processes=$np_ (bf16, packed trunk, packed embed) ==="
    python tools/bench_pool.py --checkpoint training/checkpoints/relset.pt \
        --actors "$GAMES" --games "$GAMES" --sims "$SIMS" \
        --leaf-batch "$LEAF_BATCH" --max-turns "$MAX_TURNS" \
        --dollars-per-hour "$DPH" --server-priors \
        --infer-bf16 --packed-trunk --packed-embed \
        --serve-processes "$np_" \
        > "$OUT/pool_bf16_p$np_.json" 2> "$OUT/pool_bf16_p$np_.log"
    tail -3 "$OUT/pool_bf16_p$np_.log"
    upload
done
python - <<'PY' | tee "$OUT/pool_bf16_verdict.txt"
import json
def rd(n):
    try:
        return json.load(open(f"/workspace/postreview/pool_bf16_p{n}.json"))
    except Exception:
        return None
a, b = rd(1), rd(2)
if not (a and b):
    print("one arm missing; no verdict")
else:
    for k in ("leaves_per_s", "saturated_leaves_per_s", "games_per_dollar"):
        x, y = a.get(k), b.get(k)
        if x and y:
            print(f"{k:26s} 1 proc {x:10.1f}   2 proc {y:10.1f}   {y / x:.2f}x")
    print("\nbf16 + packed trunk + packed embed, the committed configuration. "
          "Expectation on record was 1.3-1.5x. Under 1.1x, drop the idea and say so.")
PY
touch "$OUT/ALL_DONE_BF16"
upload
echo POOL_BF16_DONE
