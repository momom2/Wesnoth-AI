# shellcheck shell=bash
# box_inputs.sh: the inputs a training box builds -- the corpus, its sequences,
# the memory player's own sequences, their columns -- each restored from the
# input cache on the model host (tools/input_cache.py, docs/box_runbook.md
# "The input cache") when an entry holds its key, else built here and sent to
# the cache in the background, so the next box restores it (box_finish ends a
# send still running; the next box that misses sends again). Sourced from the
# code stage after box_init and box_stage_code, the working directory
# BOX_REPO; the caller sets OUT and WORKERS. The steps, each once per code
# stage on the machine (re-entry skips what is done):
#   inputs_games DIR HF_TAR...    match tarballs unpacked into DIR
#   inputs_own_begin OWN ...      the own sequences' key; their restore starts
#                                 in the background, beside the corpus's steps
#   inputs_corpus CORPUS RAW_TAR  the corpus at the code's version
#   inputs_sequences SEQ CORPUS VOCAB   its sequences, and their barrier
#   inputs_own_end                waits for the own restore: INPUTS_OUTCOME
#   inputs_send KIND DIR          an input built here, sent as an entry
#   inputs_columns DIR            the columns of a sequence directory
#
# Switches:
#   INPUT_CACHE=1      0: every input built here, nothing restored or sent.
#   CACHE_SEQUENCES=0  1: the corpus's sequences go through the cache too.
#                      With their columns they are about 18 GB (3.1 kB a
#                      position pickled and 1.3 kB in columns, measured on 77
#                      games, times 4.1 million positions); at the rate the
#                      2026-10 boxes fetched the own sequences' 6 GB from the
#                      host (33-43 MB/s), restoring them takes about as long
#                      as building them (270-460 s).
# Records: cache.txt (each input's outcome, the size of each sequence
# directory, the entries a sent one supersedes), cache.log, walls.txt.
# Never `set -x`: the HF token is in the environment.

INPUT_CACHE="${INPUT_CACHE:-1}"
CACHE_SEQUENCES="${CACHE_SEQUENCES:-0}"
RAW="${RAW:-$WORKDIR/raw}"
INPUTS_KEYS=$BOX_STATE/input_keys
CACHE_RESTORE_CUT_MIN="${CACHE_RESTORE_CUT_MIN:-40}"
CACHE_SEND_CUT_MIN="${CACHE_SEND_CUT_MIN:-120}"
BUILD_CUT_MIN="${BUILD_CUT_MIN:-60}"
BUILD_STALL_MIN="${BUILD_STALL_MIN:-15}"             # the builder logs every 1,000 candidates
PREENCODE_CUT_MIN="${PREENCODE_CUT_MIN:-150}"
PREENCODE_STALL_MIN="${PREENCODE_STALL_MIN:-20}"     # the pre-encoder logs every 200 games
COLUMNS_CUT_MIN="${COLUMNS_CUT_MIN:-40}"
COLUMNS_STALL_MIN="${COLUMNS_STALL_MIN:-10}"
INPUTS_OUTCOME=""
INPUTS_OWN_PID=""

inputs_key() {                   # inputs_key KIND ARGS...: KIND's key file in INPUTS_KEYS, from ARGS
    local kind=$1
    shift
    mkdir -p "$INPUTS_KEYS" || return 1
    box_bounded "cache-key $kind" 15 cache.log python tools/input_cache.py key "$kind" "$@" \
        --out "$INPUTS_KEYS/$kind.json"
}

inputs_restore() {               # inputs_restore KIND DIR ARGS...: DIR from KIND's entry; true when restored or adopted
    local kind=$1 dir=$2
    shift 2
    INPUTS_OUTCOME=off
    rm -f "$INPUTS_KEYS/$kind.refused"
    [ "$INPUT_CACHE" = 1 ] || return 1
    box_bounded "cache-restore $kind" "$CACHE_RESTORE_CUT_MIN" cache.log python tools/input_cache.py \
        --record "$OUT/cache.txt" restore --key "$INPUTS_KEYS/$kind.json" --into "$dir" "$@"
    case $BOX_RC in
        0) INPUTS_OUTCOME=restored ;;
        4) INPUTS_OUTCOME=adopted ;;
        3) INPUTS_OUTCOME=absent ;;
        2) INPUTS_OUTCOME=refused
           : > "$INPUTS_KEYS/$kind.refused" ;;   # the entry sent from here replaces it
        *) INPUTS_OUTCOME=failed
           box_log cache.txt "CACHE_RESTORE_FAILED $kind rc=$BOX_RC $BOX_WHY (cache.log): built here" ;;
    esac
    [ "$INPUTS_OUTCOME" = restored ] || [ "$INPUTS_OUTCOME" = adopted ]
}

inputs_send() {                  # inputs_send KIND DIR ARGS...: DIR sent as KIND's entry, in the background
    local kind=$1 dir=$2 replace=()
    shift 2
    [ "$INPUT_CACHE" = 1 ] || return 0
    [ ! -f "$INPUTS_KEYS/$kind.refused" ] || replace=(--replace)
    (
        RANDOM=$BASHPID          # the step's verdict file is named from $$ and $RANDOM
        box_bounded "cache-send $kind" "$CACHE_SEND_CUT_MIN" cache.log python tools/input_cache.py \
            --record "$OUT/cache.txt" store --key "$INPUTS_KEYS/$kind.json" --from "$dir" "$@" \
            ${replace[@]+"${replace[@]}"} \
            || box_log cache.txt "CACHE_SEND_FAILED $kind rc=$BOX_RC $BOX_WHY (cache.log)"
    ) < /dev/null 8>&- &
    BOX_ASYNC_PIDS+=("$!")
}

inputs_games() {                 # inputs_games DIR HF_TAR...: the match tarballs unpacked into DIR
    local dir=$1 marker
    shift
    marker=$BOX_STATE/GAMES_DONE_$(basename "$dir")
    box_marked_this_stage "$marker" && return 0
    rm -rf "$dir"
    box_bounded games 30 staging.log python tools/input_cache.py unpack --into "$dir" "$@" \
        || box_finish "GAMES_FAILED rc=$BOX_RC (staging.log)" 1
    box_mark "$marker" || box_finish "GAMES_UNWRITABLE ($marker)" 1
}

inputs_corpus() {                # inputs_corpus CORPUS RAW_TAR: the corpus from the cache, else built from RAW_TAR's replays; its summary in OUT/corpus_summary.json
    local corpus=$1 raw_tar=$2 ledger=$RAW/training/logs/replay_dispositions.jsonl.gz from
    box_marked_this_stage "$BOX_STATE/CORPUS_DONE" && return 0
    inputs_key corpus --raw "$raw_tar" || box_finish "CACHE_KEY_FAILED corpus rc=$BOX_RC (cache.log)" 1
    if ! inputs_restore corpus "$corpus" --facts "$OUT/corpus_summary.json"; then
        rm -rf "$corpus" "${corpus}_duplicates" "$RAW"
        box_bounded raw 30 staging.log python tools/input_cache.py fetch-raw --key "$INPUTS_KEYS/corpus.json" \
            --into "$RAW" || box_finish "RAW_FAILED rc=$BOX_RC (staging.log)" 1
        from=$(box_size "$OUT/corpus_build.log")
        box_bounded --stall "$OUT/corpus_build.log" "$BUILD_STALL_MIN" corpus "$BUILD_CUT_MIN" corpus_build.log \
            python tools/build_imitation_dataset.py --raw-root "$RAW" --out "$corpus" --workers "$WORKERS" \
            --dispositions "$ledger"
        tail -c "+$(( from + 1 ))" "$OUT/corpus_build.log" | grep -q "BUILD_DONE" \
            || box_finish "CORPUS_${BOX_WHY^^} rc=$BOX_RC (corpus_build.log)" 1
        # The crash barrier: every candidate accounted for, under 1% failed.
        box_bounded corpus-check 10 corpus_build.log python tools/build_imitation_dataset.py --out "$corpus" \
            --check "$OUT/corpus_summary.json" --dispositions "$ledger" \
            || box_finish "CORPUS_BARRIER rc=$BOX_RC (corpus_build.log, corpus_summary.json)" 1
        inputs_send corpus "$corpus" --facts "$OUT/corpus_summary.json"
    fi
    box_mark "$BOX_STATE/CORPUS_DONE" || box_finish "CORPUS_UNWRITABLE ($BOX_STATE)" 1
}

inputs_sequences() {             # inputs_sequences SEQ CORPUS VOCAB: the corpus's sequences under VOCAB's vocabularies (a checkpoint, or "fresh"); their barrier in OUT/sequence_summary.json
    local seq=$1 corpus=$2 vocab=$3 from vocab_key=(--fresh-vocab) vocab_encode=()
    [ "$vocab" = fresh ] || { vocab_key=(--vocab-from "$vocab"); vocab_encode=(--vocab-from "$vocab"); }
    box_marked_this_stage "$seq/SEQUENCES_DONE" && return 0
    INPUTS_OUTCOME=off
    if [ "$CACHE_SEQUENCES" = 1 ]; then
        inputs_key sequences --corpus-key "$INPUTS_KEYS/corpus.json" "${vocab_key[@]}" \
            || box_finish "CACHE_KEY_FAILED sequences rc=$BOX_RC (cache.log)" 1
        inputs_restore sequences "$seq"
    fi
    if [ "$INPUTS_OUTCOME" = restored ]; then
        { box_mark "$seq/STARTED" && box_mark "$seq/COLUMNS_DONE"; } || box_finish "SEQUENCES_UNWRITABLE ($seq)" 1
    else
        box_marked_this_stage "$seq/STARTED" || { rm -rf "$seq"; box_mark "$seq/STARTED"; } \
            || box_finish "SEQUENCES_UNWRITABLE ($seq)" 1   # records another stage's code built
        from=$(box_size "$OUT/preencode.log")
        box_bounded --stall "$OUT/preencode.log" "$PREENCODE_STALL_MIN" preencode "$PREENCODE_CUT_MIN" \
            preencode.log python tools/preencode_sequences.py --dataset "$corpus" --out "$seq" --workers "$WORKERS" \
            ${vocab_encode[@]+"${vocab_encode[@]}"}
        tail -c "+$(( from + 1 ))" "$OUT/preencode.log" | grep -q "SEQUENCES_DONE" \
            || box_finish "PREENCODE_${BOX_WHY^^} rc=$BOX_RC (preencode.log)" 1
    fi
    # The crash barrier: under 0.5% of games skipped, posterior errors under 0.1%.
    box_bounded preencode-check 10 preencode.log python - "$seq" "$OUT/sequence_summary.json" <<'EOF' \
        || box_finish "PREENCODE_BARRIER rc=$BOX_RC (preencode.log, sequence_summary.json)" 1
import json, pathlib, sys
from tools.preencode_sequences import load_manifest
man = load_manifest(pathlib.Path(sys.argv[1]))
t = man["totals"]
summary = {"manifest_games": man["n_manifest_games"], "games": man["n_games"], "errors": len(man["errors"]),
           "fingerprint": man["fingerprint"], "core_phase": man["core_phase"],
           "corpus_version": man["corpus_version"], **t}
pathlib.Path(sys.argv[2]).write_text(json.dumps(summary, indent=1), encoding="utf-8")
print("sequences", summary, flush=True)
assert man["n_games"] + len(man["errors"]) == man["n_manifest_games"], "games unaccounted for"
assert len(man["errors"]) < 0.005 * man["n_manifest_games"], "0.5% or more of the games skipped"
assert t.get("posterior_errors", 0) < 0.001 * max(1, t.get("posteriors", 0)), "posterior errors at 0.1% or more"
EOF
    box_mark "$seq/SEQUENCES_DONE" || box_finish "SEQUENCES_UNWRITABLE ($seq)" 1
    if [ "$CACHE_SEQUENCES" = 1 ] && [ "$INPUTS_OUTCOME" != restored ]; then
        inputs_columns "$seq"
        inputs_send sequences "$seq"
    fi
}

inputs_own_begin() {             # inputs_own_begin OWN TEACHER_PT PLAYER_SHA256 SLOTS LEGACY GAME_DIR...: the own key, its restore started in the background (LEGACY: a tarball of OWN's name to adopt when no entry holds the key, "" for none)
    local own=$1 teacher=$2 player=$3 slots=$4 legacy=$5 legacy_args=()
    shift 5
    INPUTS_OWN_PID=""
    box_marked_this_stage "$own/OWN_DONE" && return 0
    inputs_key own --teacher "$teacher" --player-sha256 "$player" --slots "$slots" --games "$@" \
        || box_finish "CACHE_KEY_FAILED own rc=$BOX_RC (cache.log)" 1
    [ -z "$legacy" ] || legacy_args=(--legacy "$legacy" --legacy-top "$(basename "$own")")
    rm -f "$INPUTS_KEYS/own.outcome"
    (
        RANDOM=$BASHPID          # the step's verdict file is named from $$ and $RANDOM
        inputs_restore own "$own" ${legacy_args[@]+"${legacy_args[@]}"}
        printf '%s\n' "$INPUTS_OUTCOME" > "$INPUTS_KEYS/own.outcome"
    ) < /dev/null 8>&- &
    INPUTS_OWN_PID=$!
    BOX_ASYNC_PIDS+=("$INPUTS_OWN_PID")
}

inputs_own_end() {               # inputs_own_end: the own restore waited for; INPUTS_OUTCOME (restored, adopted, absent, refused, failed, off; done when nothing ran)
    INPUTS_OUTCOME=done
    [ -n "$INPUTS_OWN_PID" ] || return 0
    wait "$INPUTS_OWN_PID"
    INPUTS_OWN_PID=""
    INPUTS_OUTCOME=$(cat "$INPUTS_KEYS/own.outcome" 2>/dev/null) || INPUTS_OUTCOME=failed
    [ -n "$INPUTS_OUTCOME" ] || INPUTS_OUTCOME=failed
}

inputs_columns() {               # inputs_columns DIR: DIR's columns, every position checked against its pickle; their manifest in OUT/columns_NAME.json
    local dir=$1 name from
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
        box_mark "$dir/COLUMNS_DONE" || box_finish "COLUMNS_UNWRITABLE ($dir)" 1
    fi
    timeout 1m python -c "import json, sys; m = json.load(open(sys.argv[1])); m.pop('games'); print(json.dumps(m))" \
        "$dir/columns_manifest.json" > "$OUT/columns_$name.json" || box_finish "COLUMNS_MANIFEST_MISSING $name" 1
    box_log cache.txt "SIZE $name: $(du -sb "$dir" 2>/dev/null | cut -f1) bytes, its records and columns"
}
