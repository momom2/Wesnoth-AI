#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# The imitation corpus rebuilt from the raw replays under the version-2
# rules (docs/corpus_v2_20260926.md): tools/build_imitation_dataset.py over
# every candidate of the dispositions ledger, on a CPU box. No GPU, no
# training: the new corpus goes to HF as one tarball, laid out like
# tier-b/replays_dataset_imitation_dedup_20260908.tar.gz, beside the
# builder's log and its summary.
#
# Before renting, from the laptop's repository root (the raw replays live
# only there):
#     python tools/stage_raw_corpus.py --out /tmp/raw_corpus.tar \
#         --upload tier-b/corpus_v2/raw_corpus_20260926.tar
#     python tools/stage_code.py --out /tmp/stage.tar.gz \
#         --script scripts/corpus_v2_rebuild_box.sh --upload tier-b/staging/stage_DATE.tar.gz
#
# Cost, measured: 0.23-0.34 s per candidate on one laptop core (extraction
# plus the labeller's reconstruction; runs of 150 and 250 games,
# 2026-09-26), so the 19,367 candidates are 1.2-1.8 core-hours: 5 to 7
# minutes on 16 cores, plus the bring-up, 0.23 GiB down and about 60 MiB
# up (the 2026-09 corpus's game files are 62 MiB). BUILD_CUT_MIN bounds
# the build at 8 times that.
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md).
# Never `set -x`: the HF token and the instance key are in the environment.
set -uo pipefail
WORKDIR=/workspace
OUT=$WORKDIR/corpus_v2
STAGE="${STAGE:-}"
RAW_TAR="${RAW_TAR:-tier-b/corpus_v2/raw_corpus_20260926.tar}"
WORKERS="${WORKERS:-$(nproc)}"
export HF_DIR="${HF_DIR:-tier-b/corpus_v2_20260926}"
BUILD_CUT_MIN="${BUILD_CUT_MIN:-60}"             # estimated 7 on 16 cores
BUILD_STALL_MIN="${BUILD_STALL_MIN:-15}"         # the builder logs every 1,000 candidates
BOX_MAX_H="${BOX_MAX_H:-2}"
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }

CORPUS=$WORKDIR/corpus_v2_build/replays_dataset_imitation

box_init
[ -n "$STAGE" ] || box_finish "NO_STAGE: build the code stage (tools/stage_code.py) and pass STAGE" 1
box_restore build.log || box_finish "RESTORE_FAILED (restore.log)" 1
box_pip huggingface_hub psutil pytest || echo "pip install failed (pip.log)"

box_stage_code || box_finish "CODE_STAGING_FAILED (staging.log)" 1
cd "$BOX_REPO" || box_finish "CODE_STAGING_FAILED (no $BOX_REPO)" 1
box_facts > "$OUT/box.txt.tmp" 2>&1
mv -f "$OUT/box.txt.tmp" "$OUT/box.txt"
box_monitor_start

# ---- the rules under test before anything is built
if ! box_marked_this_stage "$BOX_STATE/TESTED"; then
    box_bounded tests 15 tests.log python -m pytest tests/test_corpus_v2.py \
        tests/test_corpus_split_hygiene.py tests/test_scenario_cfg_case.py -q -p no:cacheprovider
    [ "$BOX_RC" -eq 0 ] || box_finish "TESTS_FAILED rc=$BOX_RC (tests.log)" 1
    box_mark "$BOX_STATE/TESTED"
fi

# ---- the inputs: the ledger and the candidates' raw replays
if [ ! -f training/logs/replay_dispositions.jsonl.gz ]; then
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
fi

# ---- the build (again on a machine without it: the corpus goes up whole)
if [ ! -f "$CORPUS/manifest.jsonl" ]; then
    box_bounded --stall "$OUT/build.log" "$BUILD_STALL_MIN" build "$BUILD_CUT_MIN" build.log \
        python tools/build_imitation_dataset.py --raw-root . --out "$CORPUS" --workers "$WORKERS"
    grep "BUILD_DONE" "$OUT/build.log" | tail -n 1 > "$OUT/summary.txt.tmp"
    if [ -s "$OUT/summary.txt.tmp" ]; then
        mv -f "$OUT/summary.txt.tmp" "$OUT/summary.txt"
    else
        box_finish "BUILD_${BOX_WHY^^} rc=$BOX_RC (build.log)" 1
    fi
    for f in manifest.jsonl outcomes.jsonl quarantined.jsonl duplicates.jsonl errors.jsonl; do
        cp -f "$CORPUS/$f" "$OUT/$f"
    done
fi
box_upload_dir replays_dataset_imitation_v2 "$CORPUS"
box_finish "CORPUS_V2_DONE $(cat "$OUT/summary.txt")"
