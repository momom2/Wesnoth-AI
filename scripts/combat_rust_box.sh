#!/usr/bin/env bash
# The Rust combat kernel certified (2026-09-12, port plan phase 3a):
# the fuzz and [mp_checkup] tests, then the imitation corpus
# reconstructed through diff_replay with the kernel on and off, one
# shard per core, and the two divergence lists compared. Expects the
# code staged and the wheel built (relset_rust_box.sh does both);
# stages the corpus from HF. Records under /workspace/combatrust/,
# uploaded to HF $HF_DIR.
set -uo pipefail
OUT=/workspace/combatrust
HF_DIR="${HF_DIR:-tier-b/combat_rust_20260912}"
CORPUS_TGZ="${CORPUS_TGZ:-tier-b/replays_dataset_imitation_dedup_20260908.tar.gz}"
mkdir -p "$OUT"
cd /workspace/Wesnoth-AI
export HF_TOKEN="$(tr -d '\r\n' < /workspace/.hf_token)" HF_HUB_DISABLE_XET=1 HF_DIR
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PATH="$HOME/.cargo/bin:$PATH"
python -c "import wesnoth_core; assert wesnoth_core.__phase__ >= 6, wesnoth_core.__phase__" || {
    python -m pip install --force-reinstall --no-deps rust/wesnoth_core > "$OUT/build.log" 2>&1
}
python -c "import wesnoth_core; print('phase', wesnoth_core.__phase__)" | tee "$OUT/box.txt"
nproc --all >> "$OUT/box.txt"
# 1. Tests.
python -m pytest tests/test_rust_combat.py tests/test_combat_seed_alignment.py tests/test_combat_rules.py \
    tests/test_sim_determinism.py -q -p no:cacheprovider > "$OUT/tests.log" 2>&1
tail -2 "$OUT/tests.log"
# 2. The corpus.
if [ ! -f replays_dataset_imitation/manifest.jsonl ]; then
python - "$CORPUS_TGZ" <<'EOF'
import sys, tarfile
from huggingface_hub import hf_hub_download
p = hf_hub_download("momom2/wesnoth-model-checkpoints", sys.argv[1])
with tarfile.open(p, "r:gz") as tf:
    tf.extractall(".")
print("corpus staged", flush=True)
EOF
fi
ls replays_dataset_imitation/*.json.gz | sort > "$OUT/files.txt"
N=$(wc -l < "$OUT/files.txt"); S=$(nproc --all); echo "corpus $N files, $S shards" | tee -a "$OUT/box.txt"
split -n l/"$S" -d -a 2 "$OUT/files.txt" "$OUT/shard_"
sweep() {                          # sweep MODE RUST_COMBAT_FLAG
    local mode="$1" flag="$2"
    mkdir -p "$OUT/$mode"
    local t0=$(date +%s)
    for f in "$OUT"/shard_*; do
        local k=$(basename "$f")
        WESNOTH_RUST_COMBAT="$flag" python tools/diff_replay.py $(cat "$f") --all-divergences \
            > "$OUT/$mode/$k.log" 2>&1 &
    done
    wait
    echo "$mode wall $(( $(date +%s) - t0 )) s" | tee -a "$OUT/box.txt"
    grep -h "^diff_replay:" "$OUT/$mode"/*.log > "$OUT/$mode.summary.txt"
    python - "$OUT/$mode" > "$OUT/$mode.divergences.txt" <<'EOF'
import glob, re, sys
tot = clean = div = 0
lines = []
for p in sorted(glob.glob(sys.argv[1] + "/*.log")):
    for l in open(p, errors="replace"):
        m = re.match(r"diff_replay: (\d+) replays, (\d+) clean, (\d+) with divergence", l)
        if m:
            tot += int(m.group(1)); clean += int(m.group(2)); div += int(m.group(3))
        elif "diverg" in l.lower() or l.startswith("  "):
            lines.append(l.rstrip())
print(f"TOTAL {tot} replays, {clean} clean, {div} with divergences")
print("\n".join(sorted(set(lines))))
EOF
    head -1 "$OUT/$mode.divergences.txt"
}
sweep rust 1
sweep python 0
if diff <(tail -n +2 "$OUT/rust.divergences.txt") <(tail -n +2 "$OUT/python.divergences.txt") > "$OUT/diff.txt"; then
    echo "SWEEPS IDENTICAL" | tee -a "$OUT/box.txt"
else
    echo "SWEEPS DIFFER ($(wc -l < "$OUT/diff.txt") lines)" | tee -a "$OUT/box.txt"
fi
python - <<'EOF'
import glob, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for p in sorted(glob.glob("/workspace/combatrust/*")):
    if os.path.isfile(p) and os.path.getsize(p) < 50_000_000:
        api.upload_file(path_or_fileobj=p, path_in_repo=os.environ["HF_DIR"] + "/" + os.path.basename(p),
                        repo_id="momom2/wesnoth-model-checkpoints")
print("uploaded", flush=True)
EOF
touch "$OUT/ALL_DONE"
echo COMBAT_RUST_DONE
