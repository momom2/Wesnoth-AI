#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# Step 5 of docs/rust_core_port_20260928.md: the Rust core certified against
# the Python applier over the whole imitation corpus on a CPU box. Every
# replay is set up and applied by both (tools/diff_core.py), the two states
# compared after the setup and every command (the map and the event state at
# each init_side and end_turn), every tenth player decision encoded by both
# in three views (the full board, the full board with the terrain set, and
# obs8's), and before every attack the defender's weapon choice and the
# attack's outcome distributions compared. Then the core's answers are
# compared with the engine's answers the hidden-unit oracle recorded
# (tools/hidden_units_oracle.py --recorded).
# Nothing is trained and no GPU is used.
#
# Before renting, from the laptop's repository root:
#     python tools/stage_code.py --out /tmp/stage.tar.gz \
#         --script scripts/core_certify_box.sh --upload tier-b/staging/stage_DATE.tar.gz
#
# Cost, measured on the laptop 2026-09-28: 3.8 ms per command with the
# encodings (12 replays, 3,884 commands, 395 decisions encoded three ways),
# so the corpus's ~6.3M commands are ~6.7 core-hours: ~13 minutes on 32
# cores, plus the bring-up (the Rust wheel's build, the 17,019-replay
# corpus). DIFF_CUT_MIN bounds the sweep at about 6 times that.
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md).
# Never `set -x`: the HF token and the instance key are in the environment.
set -uo pipefail
WORKDIR=/workspace
OUT=$WORKDIR/core_certify
STAGE="${STAGE:-}"
CORPUS_TAR="${CORPUS_TAR:-tier-b/replays_dataset_imitation_dedup_20260908.tar.gz}"
SHARDS="${SHARDS:-$(nproc)}"
ENCODE_EVERY="${ENCODE_EVERY:-10}"
export HF_DIR="${HF_DIR:-tier-b/core_certify_20260928}"
DIFF_CUT_MIN="${DIFF_CUT_MIN:-90}"
BOX_MAX_H="${BOX_MAX_H:-3}"
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }

box_init
[ -n "$STAGE" ] || box_finish "NO_STAGE: build the code stage (tools/stage_code.py) and pass STAGE" 1
box_restore build.log || box_finish "RESTORE_FAILED (restore.log)" 1
box_pip huggingface_hub psutil pytest numpy || echo "pip install failed (pip.log)"

box_stage_code || box_finish "CODE_STAGING_FAILED (staging.log)" 1
cd "$BOX_REPO" || box_finish "CODE_STAGING_FAILED (no $BOX_REPO)" 1
box_build_wheel || box_finish "WHEEL_BUILD_FAILED (build.log)" 1
# The encoder imports torch; a CPU image without it gets the CPU wheel.
python -c "import torch" 2>/dev/null || timeout -k 30s 15m python -m pip install -q torch \
    --index-url https://download.pytorch.org/whl/cpu >> "$OUT/pip.log" 2>&1 \
    || box_finish "TORCH_INSTALL_FAILED (pip.log)" 1
box_facts > "$OUT/box.txt.tmp" 2>&1
mv -f "$OUT/box.txt.tmp" "$OUT/box.txt"
box_monitor_start

# ---- the corpus
if [ ! -f replays_dataset_imitation/manifest.jsonl ]; then
    box_bounded corpus 30 staging.log python - "$CORPUS_TAR" <<'EOF' \
        || box_finish "CORPUS_FAILED rc=$BOX_RC (staging.log)" 1
import sys, tarfile
from huggingface_hub import hf_hub_download
path = hf_hub_download("momom2/wesnoth-model-checkpoints", sys.argv[1])
with tarfile.open(path, "r:gz") as tf:
    tf.extractall(".")
print("corpus files", sum(1 for n in tf.getnames() if n.endswith(".json.gz")), flush=True)
EOF
fi
[ -f replays_dataset_imitation/manifest.jsonl ] || box_finish "CORPUS_FAILED: no replays_dataset_imitation/manifest.jsonl (staging.log)" 1

# ---- the core's tests, the corpus present
if ! box_marked_this_stage "$BOX_STATE/TESTED"; then
    box_bounded tests 20 tests.log python -m pytest tests/test_game_core.py tests/test_rust_units.py \
        tests/test_rust_terrain.py tests/test_turn_events.py tests/test_diff_core.py -q -p no:cacheprovider
    [ "$BOX_RC" -eq 0 ] || box_finish "TESTS_FAILED rc=$BOX_RC (tests.log)" 1
    box_mark "$BOX_STATE/TESTED"
fi

# ---- the sweep, one diff_core per shard of the corpus
export OMP_NUM_THREADS=1
ls replays_dataset_imitation/*.json.gz > "$OUT/files.txt"
rm -f "$OUT"/shard_??*
split -n "l/$SHARDS" -d -a 3 "$OUT/files.txt" "$OUT/shard_"
# shellcheck disable=SC2016 # expanded by the inner shell, which gets them as $0 and $1
box_bounded sweep "$DIFF_CUT_MIN" sweep.log bash -c '
    for f in "$0"/shard_[0-9][0-9][0-9]; do
        ( xargs -a "$f" python tools/diff_core.py --every 1 --encode-every "$1" --outcomes > "$f.log" 2>&1;
          echo "$(date -u +%FT%TZ) $(basename "$f") rc=$?" >> "$0/progress.log" ) &
    done
    wait' "$OUT" "$ENCODE_EVERY"
[ "$BOX_WHY" = "ok" ] || box_finish "SWEEP_${BOX_WHY^^} rc=$BOX_RC (sweep.log, progress.log)" 1
python - "$OUT" <<'EOF' || box_finish "SUMMARY_FAILED" 1
import glob, re, sys
from collections import Counter
out = sys.argv[1]
replays = clean = divergent = 0
kinds = Counter()
divergences = []
for log in sorted(glob.glob(f"{out}/shard_[0-9][0-9][0-9].log")):
    lines = open(log, encoding="utf-8", errors="replace").read().splitlines()
    for k, line in enumerate(lines):
        m = re.match(r"diff_core: (\d+) replays, (\d+) clean, (\d+) with divergences", line)
        if m:
            replays += int(m[1]); clean += int(m[2]); divergent += int(m[3])
            if k + 1 < len(lines):
                for part in lines[k + 1].strip().split(", "):
                    name, _, v = part.partition("=")
                    if v.isdigit():
                        kinds[name] += int(v)
        elif line.startswith("  ") and ".json.gz" in line:
            divergences.append(line.strip())
text = "\n".join([f"core certification: {replays} replays, {clean} clean, {divergent} with divergences",
                  "  " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()))]
                 + ["  " + d[:400] for d in divergences[:300]])
print(text)
open(f"{out}/summary.txt", "w", encoding="utf-8").write(text + "\n")
EOF

# ---- the engine's recorded answers on hidden units and vision
# shellcheck disable=SC2016 # expanded by the inner shell, which gets the output directory as $0
box_bounded oracle 10 oracle.log bash -c '
    python tools/hidden_units_oracle.py --recorded training/metrics/fidelity/hidden_units_oracle_20260920.json \
        --log-level WARNING --out "$0/hidden_units_recorded.json"
    python tools/hidden_units_oracle.py --recorded training/metrics/fidelity/hidden_units_oracle_vision_20260924.json \
        --log-level WARNING --out "$0/vision_recorded.json"' "$OUT"
box_finish "CORE_CERTIFY_DONE $(head -n 1 "$OUT/summary.txt"); oracle rc=$BOX_RC"
