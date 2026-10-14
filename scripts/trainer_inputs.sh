# shellcheck shell=bash
# trainer_inputs.sh: the sequence trainer's inputs on a box, for the scripts
# that time or train it from own1 under round 2's preset
# (scripts/trainer_speed_box.sh, scripts/trainer_acceptance_box.sh). Sourced
# from the code stage after box_init and box_stage_code (it is not part of the
# box library, which box_onstart.sh fetches before the stage); `trainer_inputs`
# brings up, each step once per code stage on the machine (re-entry skips what
# is done):
#   own1 from HF, its SHA-256 checked (INIT_HF, INIT_SHA256 -> INIT_PT);
#   the corpus rebuilt at version 5 from RAW_TAR (CORPUS) and pre-encoded with
#     own1's vocabularies (SEQ);
#   round 2's own sequences (OWN): OWN_CACHE from HF, else built as round 2
#     built them (SOURCE_TARS' games, own1's 64-slot sides labelled by own1 at
#     0 slots) and sent to HF_DIR as own_sequences_r2.tar.gz (CACHE_OWN=0:
#     not sent);
#   the columns of both (tools/columnize_sequences.py, every position checked
#     against its pickle), their manifests in OUT/columns_NAME.json.
# Expected on a 4090 box (training/metrics/trainer_speed_2_20261014/walls.txt):
# about 10 minutes with the own sequences from HF, 50 when they are built.
# Never `set -x`: the HF token is in the environment.

WORKDIR=${WORKDIR:-/workspace}
RAW_TAR="${RAW_TAR:-tier-b/corpus_v3/raw_corpus_20260929.tar}"
INIT_HF="${INIT_HF:-tier-b/memory_own_trajectories_20261011/own.pt}"
INIT_SHA256="${INIT_SHA256:-70925e770d43496e3ea8f9523656f201b71ada228a0362b0eb0a600c8c107f80}"
SOURCE_TARS="${SOURCE_TARS:-tier-b/own_confirm_20261011/games_own64_vs_p3.tar.gz \
tier-b/own_confirm_20261011/games_own64_vs_own0.tar.gz \
tier-b/own_confirm_20261011/games_own64_selfplay.tar.gz \
tier-b/memory_own_round2_20261011/games_gen_own1_selfplay.tar.gz}"
CACHE_OWN="${CACHE_OWN:-1}"
OWN_CACHE="${OWN_CACHE:-tier-b/trainer_speed_20261013b/own_sequences_r2.tar.gz}"
BUILD_CUT_MIN="${BUILD_CUT_MIN:-60}"
BUILD_STALL_MIN="${BUILD_STALL_MIN:-15}"
PREENCODE_CUT_MIN="${PREENCODE_CUT_MIN:-150}"
PREENCODE_STALL_MIN="${PREENCODE_STALL_MIN:-20}"
CAND_CUT_MIN="${CAND_CUT_MIN:-60}"
CAND_STALL_MIN="${CAND_STALL_MIN:-20}"
TARGETS_CUT_MIN="${TARGETS_CUT_MIN:-120}"
TARGETS_STALL_MIN="${TARGETS_STALL_MIN:-20}"
COLUMNS_CUT_MIN="${COLUMNS_CUT_MIN:-40}"
COLUMNS_STALL_MIN="${COLUMNS_STALL_MIN:-10}"
RAW=$WORKDIR/raw
CORPUS=$WORKDIR/corpus
SEQ=$WORKDIR/sequences
OWN_GAMES=$WORKDIR/own_games_r2
CAND=$WORKDIR/own_candidates_r2
OWN=$WORKDIR/own_sequences_r2
INIT_PT=$WORKDIR/own1.pt

trainer_init_checkpoint() {      # own1, its SHA-256 checked
    if { [ -f "$INIT_PT" ] && echo "$INIT_SHA256  $INIT_PT" | sha256sum -c --status; }; then
        return 0
    fi
    box_bounded init 20 staging.log python - "$INIT_HF" "$INIT_PT" <<'EOF' \
        || box_finish "INIT_CHECKPOINT_FAILED ($INIT_HF; staging.log)" 1
import os, shutil, sys
from huggingface_hub import hf_hub_download
shutil.copyfile(hf_hub_download("momom2/wesnoth-model-checkpoints", sys.argv[1]), sys.argv[2] + ".tmp")
os.replace(sys.argv[2] + ".tmp", sys.argv[2])
EOF
    echo "$INIT_SHA256  $INIT_PT" | sha256sum -c --status || box_finish "INIT_CHECKPOINT_SHA256 ($INIT_HF)" 1
}

trainer_corpus() {               # the corpus at version 5 and its sequences with own1's vocabularies
    local from
    if ! box_marked_this_stage "$BOX_STATE/CORPUS_DONE"; then
        box_bounded raw 30 staging.log python - "$RAW_TAR" "$RAW" <<'EOF' || box_finish "RAW_FAILED (staging.log)" 1
import sys, tarfile
from huggingface_hub import hf_hub_download
with tarfile.open(hf_hub_download("momom2/wesnoth-model-checkpoints", sys.argv[1]), "r") as tf:
    tf.extractall(sys.argv[2])
    print("raw replays", sum(1 for name in tf.getnames() if name.endswith(".bz2")), flush=True)
EOF
        rm -rf "$CORPUS" "${CORPUS}_duplicates"
        from=$(box_size "$OUT/corpus_build.log")
        box_bounded --stall "$OUT/corpus_build.log" "$BUILD_STALL_MIN" corpus "$BUILD_CUT_MIN" corpus_build.log \
            python tools/build_imitation_dataset.py --raw-root "$RAW" --out "$CORPUS" --workers "$WORKERS" \
            --dispositions "$RAW/training/logs/replay_dispositions.jsonl.gz"
        tail -c "+$(( from + 1 ))" "$OUT/corpus_build.log" | grep -q "BUILD_DONE" \
            || box_finish "CORPUS_${BOX_WHY^^} rc=$BOX_RC (corpus_build.log)" 1
        box_bounded corpus-check 10 corpus_build.log \
            python tools/build_imitation_dataset.py --out "$CORPUS" --check "$OUT/corpus_summary.json" \
            --dispositions "$RAW/training/logs/replay_dispositions.jsonl.gz" \
            || box_finish "CORPUS_BARRIER rc=$BOX_RC (corpus_build.log)" 1
        box_mark "$BOX_STATE/CORPUS_DONE"
    fi
    if ! box_marked_this_stage "$SEQ/SEQUENCES_DONE"; then
        box_marked_this_stage "$SEQ/STARTED" || { rm -rf "$SEQ"; box_mark "$SEQ/STARTED"; } \
            || box_finish "SEQUENCES_UNWRITABLE ($SEQ)" 1
        from=$(box_size "$OUT/preencode.log")
        box_bounded --stall "$OUT/preencode.log" "$PREENCODE_STALL_MIN" preencode "$PREENCODE_CUT_MIN" preencode.log \
            python tools/preencode_sequences.py --dataset "$CORPUS" --out "$SEQ" --workers "$WORKERS" \
            --vocab-from "$INIT_PT"
        tail -c "+$(( from + 1 ))" "$OUT/preencode.log" | grep -q "SEQUENCES_DONE" \
            || box_finish "PREENCODE_${BOX_WHY^^} rc=$BOX_RC (preencode.log)" 1
        box_mark "$SEQ/SEQUENCES_DONE"
    fi
}

trainer_own_sequences() {        # round 2's own sequences: from HF when an earlier session sent them, else built and sent
    local from
    if ! box_marked_this_stage "$OWN/OWN_DONE"; then
        box_bounded own-restore 30 staging.log python - "$OWN_CACHE" "$WORKDIR" <<'EOF'
import sys, tarfile
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
try:
    path = hf_hub_download("momom2/wesnoth-model-checkpoints", sys.argv[1])
except EntryNotFoundError:
    print("no cached own sequences on HF", flush=True)
    sys.exit(3)
with tarfile.open(path, "r:gz") as tf:
    if hasattr(tarfile, "data_filter"):
        tf.extractall(sys.argv[2], filter="data")
    else:
        tf.extractall(sys.argv[2])
print("own sequences from HF", flush=True)
EOF
        if [ "$BOX_RC" -eq 0 ] && [ -f "$OWN/sequence_manifest.json" ]; then
            box_mark "$OWN/OWN_DONE"
        else
            rm -rf "$OWN" "$CAND"
            # shellcheck disable=SC2086 # SOURCE_TARS is a list of HF paths
            box_bounded own-inputs 30 staging.log python - "$OWN_GAMES" $SOURCE_TARS <<'EOF' \
                || box_finish "OWN_INPUTS_FAILED (staging.log)" 1
import sys, tarfile
from huggingface_hub import hf_hub_download
for path in sys.argv[2:]:
    with tarfile.open(hf_hub_download("momom2/wesnoth-model-checkpoints", path), "r:gz") as tf:
        tf.extractall(sys.argv[1])
        print(path, sum(1 for name in tf.getnames() if name.endswith(".game.jsonl.gz")), "game records", flush=True)
EOF
            from=$(box_size "$OUT/candidates.log")
            box_bounded --stall "$OUT/candidates.log" "$CAND_STALL_MIN" candidates "$CAND_CUT_MIN" candidates.log \
                python tools/own_trajectories.py --games "$OWN_GAMES"/games_* --teacher "$INIT_PT" \
                --player-sha256 "$INIT_SHA256" --slots 64 --out "$CAND" --workers "$WORKERS"
            tail -c "+$(( from + 1 ))" "$OUT/candidates.log" | grep -q "CANDIDATES_DONE" \
                || box_finish "CANDIDATES_${BOX_WHY^^} rc=$BOX_RC (candidates.log)" 1
            box_gpu_ok || box_finish "GPU_UNRESPONSIVE before the targets: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
            from=$(box_size "$OUT/targets.log")
            box_bounded --stall "$OUT/targets.log" "$TARGETS_STALL_MIN" targets "$TARGETS_CUT_MIN" targets.log \
                python tools/own_targets.py --candidates "$CAND" --teacher "$INIT_PT" --out "$OWN" --device cuda
            tail -c "+$(( from + 1 ))" "$OUT/targets.log" | grep -q "OWN_TARGETS_DONE" \
                || box_finish "TARGETS_${BOX_WHY^^} rc=$BOX_RC (targets.log)" 1
            box_mark "$OWN/OWN_DONE"
            if [ "$CACHE_OWN" = 1 ]; then
                box_upload_dir own_sequences_r2 "$OWN"
                box_upload
            fi
        fi
    fi
    cp -f "$OWN/sequence_manifest.json" "$OUT/own_sequence_manifest.json" || box_finish "OWN_MANIFEST_MISSING" 1
}

trainer_columns() {              # the columns of the corpus's and the own sequences
    local dir name from
    for dir in "$SEQ" "$OWN"; do
        name=$(basename "$dir")
        if ! box_marked_this_stage "$dir/COLUMNS_DONE"; then
            from=$(box_size "$OUT/columns.log")
            box_bounded --stall "$OUT/columns.log" "$COLUMNS_STALL_MIN" "columns $name" "$COLUMNS_CUT_MIN" \
                columns.log python tools/columnize_sequences.py --sequences "$dir" --workers "$WORKERS"
            case $BOX_RC in
                0) tail -c "+$(( from + 1 ))" "$OUT/columns.log" | grep -q "COLUMNS_DONE" \
                       || box_finish "COLUMNS_UNFINISHED $name (columns.log)" 1 ;;
                2) box_finish "COLUMNS_MISMATCH $name: games whose columns differ from their pickles (columns.log)" 1 ;;
                *) box_finish "COLUMNS_${BOX_WHY^^} rc=$BOX_RC $name (columns.log)" 1 ;;
            esac
            box_mark "$dir/COLUMNS_DONE"
        fi
        timeout 1m python -c "import json, sys; m = json.load(open(sys.argv[1])); m.pop('games'); print(json.dumps(m))" \
            "$dir/columns_manifest.json" > "$OUT/columns_$name.json" || box_finish "COLUMNS_MANIFEST_MISSING $name" 1
    done
}

trainer_inputs() {               # every input above, in order (run from BOX_REPO)
    trainer_init_checkpoint
    trainer_corpus
    trainer_own_sequences
    trainer_columns
    box_upload
}
