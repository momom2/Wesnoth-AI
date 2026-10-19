# shellcheck shell=bash
# shellcheck source-path=SCRIPTDIR
# trainer_inputs.sh: the sequence trainer's inputs on a box, for the scripts
# that time or train it from own1 under round 2's preset
# (scripts/trainer_speed_box.sh, scripts/trainer_acceptance_box.sh). Sourced
# from the code stage after box_init and box_stage_code (it is not part of the
# box library, which box_onstart.sh fetches before the stage); `trainer_inputs`
# brings up, through the input cache (scripts/box_inputs.sh), each step once
# per code stage on the machine (re-entry skips what is done):
#   own1 from HF, its SHA-256 checked (INIT_HF, INIT_SHA256 -> INIT_PT);
#   round 2's source games (SOURCE_TARS) in OWN_GAMES, the own sequences' key;
#   round 2's own sequences (OWN), restored in the background while the corpus
#     is brought up: the cache's entry, else the legacy tarball OWN_LEGACY
#     (the round-3 bench's, core phase 32; empty: none), checked against the
#     key and then sent as the entry, else built as round 2 built them
#     (own1's 64-slot sides labelled by own1 at 0 slots) and sent;
#   the corpus at version 5 (CORPUS): the cache's entry, else rebuilt from
#     RAW_TAR and sent;
#   its sequences with own1's vocabularies (SEQ), pre-encoded here
#     (CACHE_SEQUENCES=1: through the cache too);
#   the columns of both (tools/columnize_sequences.py, every position checked
#     against its pickle), their manifests in OUT/columns_NAME.json.
# Expected on a 4090 box with the corpus and the own sequences in the cache,
# from the 2026-10-13 to 2026-10-15 trainer boxes' walls.txt: own1 5-18 s, the
# corpus 5 s, the source games 6 s, the pre-encoding 218-347 s and its
# columns 53-72 s, the own restore (139-181 s) hidden behind them: 4.8-7.5
# minutes, against 7.5-12 before. Building the own sequences adds about 41
# minutes (candidates 154 s, targets 2,286 s on 2026-10-13).
# Never `set -x`: the HF token is in the environment.

WORKDIR=${WORKDIR:-/workspace}
RAW_TAR="${RAW_TAR:-tier-b/corpus_v3/raw_corpus_20260929.tar}"
INIT_HF="${INIT_HF:-tier-b/memory_own_trajectories_20261011/own.pt}"
INIT_SHA256="${INIT_SHA256:-70925e770d43496e3ea8f9523656f201b71ada228a0362b0eb0a600c8c107f80}"
SOURCE_TARS="${SOURCE_TARS:-tier-b/own_confirm_20261011/games_own64_vs_p3.tar.gz \
tier-b/own_confirm_20261011/games_own64_vs_own0.tar.gz \
tier-b/own_confirm_20261011/games_own64_selfplay.tar.gz \
tier-b/memory_own_round2_20261011/games_gen_own1_selfplay.tar.gz}"
OWN_LEGACY="${OWN_LEGACY-tier-b/trainer_speed_3_20261015b/own_sequences_r2.tar.gz}"
CAND_CUT_MIN="${CAND_CUT_MIN:-60}"
CAND_STALL_MIN="${CAND_STALL_MIN:-20}"
TARGETS_CUT_MIN="${TARGETS_CUT_MIN:-120}"
TARGETS_STALL_MIN="${TARGETS_STALL_MIN:-20}"
RAW=$WORKDIR/raw
CORPUS=$WORKDIR/corpus
SEQ=$WORKDIR/sequences
OWN_GAMES=$WORKDIR/own_games_r2
CAND=$WORKDIR/own_candidates_r2
OWN=$WORKDIR/own_sequences_r2
INIT_PT=$WORKDIR/own1.pt
# shellcheck source=box_inputs.sh
. "$BOX_REPO/scripts/box_inputs.sh" || box_finish "CODE_STAGING_FAILED (no scripts/box_inputs.sh)" 1

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

trainer_own_build() {            # round 2's own sequences built as round 2 built them
    local from
    rm -rf "$OWN" "$CAND"
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
}

trainer_own_sequences() {        # round 2's own sequences: the background restore's, else built; sent when not restored
    inputs_own_end
    if ! box_marked_this_stage "$OWN/OWN_DONE"; then
        case $INPUTS_OUTCOME in
            restored) box_mark "$OWN/COLUMNS_DONE" || box_finish "OWN_UNWRITABLE ($OWN)" 1 ;;
            adopted) ;;
            *) trainer_own_build ;;
        esac
        box_mark "$OWN/OWN_DONE" || box_finish "OWN_UNWRITABLE ($OWN)" 1
        if [ "$INPUTS_OUTCOME" != restored ]; then
            inputs_columns "$OWN"
            inputs_send own "$OWN"
        fi
    fi
    cp -f "$OWN/sequence_manifest.json" "$OUT/own_sequence_manifest.json" || box_finish "OWN_MANIFEST_MISSING" 1
}

trainer_inputs() {               # every input above, in order (run from BOX_REPO)
    trainer_init_checkpoint
    # shellcheck disable=SC2086 # SOURCE_TARS is a list of HF paths
    inputs_games "$OWN_GAMES" $SOURCE_TARS
    inputs_own_begin "$OWN" "$INIT_PT" "$INIT_SHA256" 64 "$OWN_LEGACY" "$OWN_GAMES"/games_*
    inputs_corpus "$CORPUS" "$RAW_TAR"
    inputs_sequences "$SEQ" "$CORPUS" "$INIT_PT"
    trainer_own_sequences
    inputs_columns "$SEQ"
    inputs_columns "$OWN"
    box_upload
}
