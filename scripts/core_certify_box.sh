#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# Step 5 of docs/rust_core_port_20260928.md: the Rust core certified against
# the Python applier over the whole imitation corpus on a CPU box, the corpus
# rebuilt from the raw replays at the stage's version (the one the retrain
# trains on: tools/build_imitation_dataset.py over RAW_TAR). Every
# replay is set up and applied by both (tools/diff_core.py), the two states
# compared after the setup and every command (the map and the event state at
# each init_side and end_turn), every tenth player decision encoded by both
# in three views (the full board, the full board with the terrain set, and
# obs8's), before every attack the defender's weapon choice and the
# attack's outcome distributions compared, and after every command each
# player side's sighting record compared with tools/sighting_oracle.py.
# The parity encoding's own columns have no second builder: tests cover
# them. Then the core's answers are compared with the engine's answers the
# hidden-unit oracle recorded (tools/hidden_units_oracle.py --recorded).
# Nothing is trained and no GPU is used.
#
# Before renting, from the laptop's repository root:
#     python tools/stage_code.py --out /tmp/stage.tar.gz \
#         --script scripts/core_certify_box.sh --upload tier-b/staging/stage_DATE.tar.gz
#
# Cost, measured on the laptop 2026-09-30 over 10 fogged corpus replays
# (3,018 commands): 9 ms per command with every option above (one decision
# in ten encoded), so the corpus's ~5.5M commands are ~14 core-hours: ~25
# minutes on 64 hardware threads, plus the bring-up (the Rust wheel's build,
# the corpus's build from the raw replays: 1.2-1.8 core-hours).
# DIFF_CUT_MIN bounds the sweep at 4 times that.
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md).
# Never `set -x`: the HF token and the instance key are in the environment.
# box-needs: disk_gb=60 ram_gb=48 cores=48
set -uo pipefail
WORKDIR=/workspace
OUT=$WORKDIR/core_certify
STAGE="${STAGE:-}"
RAW_TAR="${RAW_TAR:-tier-b/corpus_v3/raw_corpus_20260929.tar}"
SHARDS="${SHARDS:-}"                      # default: box_workers (cores, memory-bounded), after bring-up
ENCODE_EVERY="${ENCODE_EVERY:-10}"
export HF_DIR="${HF_DIR:-tier-b/core_certify_20260930}"
DIFF_CUT_MIN="${DIFF_CUT_MIN:-120}"
BOX_MAX_H="${BOX_MAX_H:-3.5}"
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }

box_init
[ -n "$STAGE" ] || box_finish "NO_STAGE: build the code stage (tools/stage_code.py) and pass STAGE" 1
box_bind_run_stage
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

# ---- the corpus: the raw replays, then the build (a marker after each)
if ! box_marked_this_stage "$BOX_STATE/INPUTS_DONE"; then
    box_bounded inputs 20 staging.log python - "$RAW_TAR" <<'EOF' \
        || box_finish "INPUTS_FAILED rc=$BOX_RC (staging.log)" 1
import sys, tarfile
from huggingface_hub import hf_hub_download
path = hf_hub_download("momom2/wesnoth-model-checkpoints", sys.argv[1])
with tarfile.open(path, "r") as tf:
    tf.extractall(".")
    n = sum(1 for name in tf.getnames() if name.endswith(".bz2"))
print("raw replays", n, flush=True)
EOF
    box_mark "$BOX_STATE/INPUTS_DONE"
fi
if ! box_marked_this_stage "$BOX_STATE/CORPUS_DONE"; then
    rm -rf replays_dataset_imitation replays_dataset_imitation_duplicates
    from=$(box_size "$OUT/corpus_build.log")
    box_bounded --stall "$OUT/corpus_build.log" 15 corpus 60 corpus_build.log \
        python tools/build_imitation_dataset.py --raw-root . --out replays_dataset_imitation \
        --workers "$(box_workers)"
    tail -c "+$(( from + 1 ))" "$OUT/corpus_build.log" | grep "BUILD_DONE" \
        || box_finish "CORPUS_${BOX_WHY^^} rc=$BOX_RC (corpus_build.log)" 1
    # The crash barrier the retrain's corpus passes: every candidate
    # accounted for, under 1% failed, at the stage's corpus version.
    box_bounded corpus-check 10 corpus_build.log \
        python tools/build_imitation_dataset.py --out replays_dataset_imitation --check "$OUT/corpus_summary.json" \
        || box_finish "CORPUS_BARRIER rc=$BOX_RC (corpus_build.log, corpus_summary.json)" 1
    box_mark "$BOX_STATE/CORPUS_DONE"
fi
[ -f replays_dataset_imitation/manifest.jsonl ] || box_finish "CORPUS_FAILED: no replays_dataset_imitation/manifest.jsonl (corpus_build.log)" 1

# ---- the core's tests, the corpus present
if ! box_marked_this_stage "$BOX_STATE/TESTED"; then
    box_bounded tests 20 tests.log python -m pytest tests/test_game_core.py tests/test_rust_units.py \
        tests/test_rust_terrain.py tests/test_turn_events.py tests/test_diff_core.py \
        tests/test_delayed_shroud.py -q -p no:cacheprovider
    [ "$BOX_RC" -eq 0 ] || box_finish "TESTS_FAILED rc=$BOX_RC (tests.log)" 1
    box_mark "$BOX_STATE/TESTED"
fi

# ---- the sweep, one diff_core per shard of the corpus
# The shard lists and logs live in one folder, uploaded as one tarball a
# round (a file each would cost a model-host commit each); a marker after
# a sweep whose every replay has its verdict keeps a re-entry from redoing
# it. In the sweep the applier runs
# its rules in Python (the Rust kernels off), so that no rule is the core
# compared with itself; the core does not read these switches.
export OMP_NUM_THREADS=1
SW=$OUT/shards
mkdir -p "$SW"
box_upload_dir shards "$SW"
if ! box_marked_this_stage "$BOX_STATE/SWEPT"; then
    [ -n "$SHARDS" ] || SHARDS=$(box_workers)
    echo "shards $SHARDS" >> "$OUT/box.txt"
    rm -f "$SW"/shard_* "$SW/progress.log"
    find replays_dataset_imitation -maxdepth 1 -name '*.json.gz' | sort > "$SW/files.txt"
    split -n "l/$SHARDS" -d -a 3 "$SW/files.txt" "$SW/shard_"
    # shellcheck disable=SC2016 # expanded by the inner shell, which gets them as $0 and $1
    box_bounded sweep "$DIFF_CUT_MIN" sweep.log bash -c '
        for f in "$0"/shard_[0-9][0-9][0-9]; do
            ( xargs -a "$f" env WESNOTH_RUST=0 WESNOTH_RUST_OBSERVE=0 WESNOTH_RUST_COMBAT=0 \
                  python tools/diff_core.py --every 1 --encode-every "$1" --outcomes --sightings > "$f.log" 2>&1
              rc=$?
              echo "$(date -u +%FT%TZ) $(basename "$f") rc=$rc" >> "$0/progress.log" ) &
        done
        wait' "$SW" "$ENCODE_EVERY"
    [ "$BOX_WHY" = "ok" ] || box_finish "SWEEP_${BOX_WHY^^} rc=$BOX_RC (sweep.log, shards/progress.log)" 1
fi
# The summary counts replays against the file list: a shard that died
# without its summary line reads INCOMPLETE, never clean.
python - "$SW" "$OUT/summary.txt" <<'PYEOF' || box_finish "SUMMARY_FAILED" 1
import glob, re, sys
from collections import Counter
shards, out = sys.argv[1:3]
expected = sum(1 for line in open(f"{shards}/files.txt", encoding="utf-8") if line.strip())
replays = clean = divergent = 0
kinds = Counter()
divergences = []
for log in sorted(glob.glob(f"{shards}/shard_[0-9][0-9][0-9].log")):
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
verdict = "INCOMPLETE" if replays != expected or not expected else "DIVERGENT" if divergent else "CLEAN"
text = "\n".join([f"core certification {verdict}: {replays} of {expected} replays, {clean} clean, "
                  f"{divergent} with divergences",
                  "  " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()))]
                 + ["  " + d[:400] for d in divergences[:300]])
print(text)
open(out, "w", encoding="utf-8").write(text + "\n")
PYEOF
# A sweep with a replay short of its verdict is redone on re-entry.
if ! head -n 1 "$OUT/summary.txt" | grep -q " INCOMPLETE:"; then
    box_marked_this_stage "$BOX_STATE/SWEPT" || box_mark "$BOX_STATE/SWEPT"
fi

# ---- the engine's recorded answers on hidden units and vision
# shellcheck disable=SC2016 # expanded by the inner shell, which gets the output directory as $0
box_bounded oracle 10 oracle.log bash -c '
    rc=0
    python tools/hidden_units_oracle.py --recorded training/metrics/fidelity/hidden_units_oracle_20260920.json \
        --log-level WARNING --out "$0/hidden_units_recorded.json" || rc=1
    python tools/hidden_units_oracle.py --recorded training/metrics/fidelity/hidden_units_oracle_vision_20260924.json \
        --log-level WARNING --out "$0/vision_recorded.json" || rc=1
    exit "$rc"' "$OUT"
oracle_rc=$BOX_RC
summary=$(head -n 1 "$OUT/summary.txt")
if [[ $summary == *" CLEAN:"* ]] && [ "$oracle_rc" -eq 0 ]; then
    box_finish "CORE_CERTIFY_DONE $summary; oracle rc=0"
fi
box_finish "CORE_CERTIFY_FAILED $summary; oracle rc=$oracle_rc" 1
