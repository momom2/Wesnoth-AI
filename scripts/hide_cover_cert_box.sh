#!/usr/bin/env bash
# Certification for the hide-cover root fix (2026-09-13).
#
# Cover for ambush / concealment / submerge is now decided by the
# engine's own `[hides] [filter_location]` terrain globs
# (`terrain_resolver.hides_cover`) instead of a hand-rolled overlay
# table's defense keys, which gave NO cover on 19% of the shipped maps'
# forest-overlay hexes and 25% of their village-overlay hexes.
#
# More hexes now hide units, so more moves can be stopped by an ambush
# and more units are fog-hidden. That is a behaviour change on the
# reconstruction path, so the acceptance test is the whole corpus:
#   1. tools/diff_replay.py over every replay, sharded -- does the sim
#      still accept every recorded command as legal?
#   2. tools/diff_core.py -- do the Rust-owned state and the Python
#      applier still agree (the core's baked cover flags changed too)?
#   3. the suites.
# Records under /workspace/hidecert, uploaded to HF.
set -uo pipefail
OUT=/workspace/hidecert
HF_DIR="${HF_DIR:-tier-b/hide_cover_20260913}"
STAGE="${STAGE:-tier-b/staging/stage_20260913c.tar.gz}"
SHARDS="${SHARDS:-24}"
CORE_REPLAYS="${CORE_REPLAYS:-600}"
mkdir -p "$OUT"
cd /workspace
export HF_TOKEN="$(tr -d '\r\n' < /workspace/.hf_token)" HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
python -m pip install -q huggingface_hub pytest psutil scipy 2>&1 | tail -1 || true

upload() {
    python - <<'PY' 2>/dev/null || true
import os, glob
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for p in sorted(glob.glob("/workspace/hidecert/*")):
    if os.path.isfile(p) and os.path.getsize(p) < 50_000_000:
        api.upload_file(path_or_fileobj=p, path_in_repo=os.environ["HF_DIR"] + "/" + os.path.basename(p),
                        repo_id="momom2/wesnoth-model-checkpoints")
PY
}
export HF_DIR

rm -rf Wesnoth-AI && mkdir -p Wesnoth-AI
python - "$STAGE" <<'PY'
import sys, tarfile
from huggingface_hub import hf_hub_download
p = hf_hub_download("momom2/wesnoth-model-checkpoints", sys.argv[1])
with tarfile.open(p, "r:gz") as tf:
    tf.extractall("/workspace/Wesnoth-AI")
print("code staged")
PY
cd /workspace/Wesnoth-AI
command -v cc >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq gcc) >/dev/null 2>&1 || true
command -v cargo >/dev/null 2>&1 || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1
export PATH="$HOME/.cargo/bin:$PATH"
python -m pip install -q maturin >/dev/null 2>&1
touch rust/wesnoth_core/src/*.rs
python -m pip install --force-reinstall --no-deps rust/wesnoth_core > "$OUT/build.log" 2>&1
python -c "import wesnoth_core; assert wesnoth_core.__phase__ >= 8; print('wheel phase', wesnoth_core.__phase__)" | tee -a "$OUT/build.log" \
    || { echo BUILD_FAILED | tee -a "$OUT/build.log"; upload; touch "$OUT/ALL_DONE"; exit 1; }
{ nproc --all; nvidia-smi --query-gpu=name --format=csv,noheader; } > "$OUT/box.txt" 2>&1
upload

python - <<'PY'
import pathlib, tarfile
from huggingface_hub import hf_hub_download
if not pathlib.Path("replays_dataset_imitation").exists():
    p = hf_hub_download("momom2/wesnoth-model-checkpoints",
                        "tier-b/replays_dataset_imitation_dedup_20260908.tar.gz")
    with tarfile.open(p, "r:gz") as tf:
        tf.extractall(".")
print("corpus", len(list(pathlib.Path("replays_dataset_imitation").glob("*.json.gz"))))
PY

# 1. The whole corpus through the reconstructor, sharded.
ls replays_dataset_imitation/*.json.gz > "$OUT/files.txt"
rm -f "$OUT"/shard_*
split -n "l/$SHARDS" -d -a 2 "$OUT/files.txt" "$OUT/shard_"
t0=$(date +%s)
for f in "$OUT"/shard_??; do
    ( xargs -a "$f" python tools/diff_replay.py > "$f.log" 2>&1 || true ) &
done
wait
echo "diff_replay wall $(( $(date +%s) - t0 )) s" | tee "$OUT/replay.wall"
python - <<'PY' | tee "$OUT/replay_summary.txt"
import glob, re
tot = clean = div = 0
lines = []
for log in sorted(glob.glob("/workspace/hidecert/shard_??.log")):
    for line in open(log, encoding="utf-8", errors="replace"):
        m = re.search(r"(\d+)\s+replays?.*?(\d+)\s+clean", line)
        if "replays" in line and ("clean" in line or "diverg" in line):
            lines.append(line.rstrip())
        for pat, key in ((r"(\d+) replays", "tot"), (r"(\d+) clean", "clean"),
                         (r"(\d+) with divergences", "div")):
            mm = re.search(pat, line)
            if mm:
                if key == "tot": tot += int(mm.group(1))
                elif key == "clean": clean += int(mm.group(1))
                else: div += int(mm.group(1))
print(f"diff_replay: {tot} replays, {clean} clean, {div} with divergences")
for l in lines[:12]:
    print("  " + l)
PY
grep -h "DIVERGENCE\|divergence" "$OUT"/shard_??.log 2>/dev/null | head -25 > "$OUT/replay_divergences.txt" || true
head -25 "$OUT/replay_divergences.txt"
upload

# 2. Core against the Python applier (the core's cover flags changed too).
python tools/diff_core.py replays_dataset_imitation --limit "$CORE_REPLAYS" > "$OUT/diff_core.log" 2>&1
grep -v WARNING "$OUT/diff_core.log" | head -6
upload

# 3. The suites, both states of record.
SUITES="tests/test_hide_cover.py tests/test_visibility.py tests/test_game_core.py \
tests/test_recruit_rejection.py tests/test_fork_isolation.py tests/test_sim_determinism.py \
tests/test_rust_observe.py tests/test_rust_relevant_set.py tests/test_rust_enumerate.py"
python -m pytest $SUITES -q -p no:cacheprovider -m "" > "$OUT/tests_python.log" 2>&1
tail -3 "$OUT/tests_python.log"
WESNOTH_RUST_CORE=1 python -m pytest $SUITES -q -p no:cacheprovider -m "" > "$OUT/tests_core.log" 2>&1
tail -3 "$OUT/tests_core.log"
upload
touch "$OUT/ALL_DONE"
echo HIDE_COVER_CERT_DONE
