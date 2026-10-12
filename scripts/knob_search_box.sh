#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# The search over the reference's decode knobs
# (docs/decode_knobs_prereg_20261012.md), on one box:
#   the tests of the decode, its flags, the estimand guards and the search
#   driver, on the wheel built from the stage;
#   the reference `own1` (configs/reference_player.json, its SHA-256
#   checked);
#   successive halving by tools/knob_search.py: 24 knob vectors x 100
#   decisive games against the reference, the best 8 x 300 more, the best
#   2 x 800 more, PURE, the Ladder maps with both factions drawn
#   uniformly, every game recorded whole; one match per bounded step, each
#   vector's games going up as one tarball after its match;
#   the confirmation: the winner x 800 decisive games on seed base 250000,
#   fitted by tools/elo_collect.py (confirm.fit.json, and the reading in
#   search.jsonl).
#   Stage 2 does not start when today's decode (kv00) lies outside two
#   standard errors of 50% in stage 1 (the pre-registration's null arm):
#   the entry finishes NULL_ARM_SUSPECT; IGNORE_NULL_ARM=1 continues.
#
# Box: an RTX 4090 with at least 24 usable cores for the 20 workers
# (docs/box_specs.md). Expected wall 2.4 to 3.4 h: bring-up and tests about
# 15 minutes; 7,200 decisive games at 0.87 to 1.34 s a game (800-decisive
# matches of own1 at 64 slots took 692 to 1,070 s on 2026-10-11,
# training/metrics/own_confirm_20261011 and memory_own_round2_20261011),
# 1.7 to 2.7 h; an estimated 40 s to bring up each of the 35 matches' servers and
# workers, 0.4 h; the final upload a few minutes. Cost at $0.42-0.63 an
# hour: $1.0-2.1 expected; BOX_MAX_H 4 caps it at $1.7-2.5 plus the final
# round. Identical match repeats have differed by 1.8x, which would pass
# the switch: a cut entry resumes on re-entry, the driver reading every count
# from the games directories. In walls.txt a stage's step that leaves more
# of its stage to play reads rc=10 (the driver's "one match done").
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md): the
# onstart of scripts/rent_box.py fetches the library of STAGE, then this
# script. Every step has a bound; the dead-man's switch finishes the entry
# after BOX_MAX_H hours whatever it is doing. Records go to HF $HF_DIR
# every 15 minutes and after each match. Re-entry, on this machine or a new
# one (files absent here come back from HF), resumes the search where its
# records stop. A run belongs to the code stage that began it (RUN_STAGE):
# another stage continues it only with RESUME_OTHER_STAGE=1. Every exit
# past that check, clean or not, uploads the records with ALL_DONE last and
# stops the instance; an entry refused there stops it and sends nothing.
# Never `set -x`: the HF token and the instance key are in the environment.
# box-needs: disk_gb=40 ram_gb=48 gpu_ram_gb=24 cores=24 gpu=4090
set -uo pipefail
WORKDIR=/workspace
OUT=$WORKDIR/knobsearch
STAGE="${STAGE:-}"
JOBS="${JOBS:-20}"
IGNORE_NULL_ARM="${IGNORE_NULL_ARM:-0}"
export HF_DIR="${HF_DIR:-tier-b/decode_knobs_20261012}"
# Bounds in minutes, from the pre-registration's "Cost" and the measured
# match rates: one match of 100, 300 or 800 decisive games, about 1.6 times
# its expected wall at the slower rate, the run_elo_batch time budget below it.
TESTS_CUT_MIN="${TESTS_CUT_MIN:-20}"
MATCH_CUT_MIN=(0 15 30 45)                      # by stage; the confirmation takes stage 3's
BUDGET_MIN=(0 10 22 35)
BOX_ROUND_MIN="${BOX_ROUND_MIN:-15}"
BOX_MAX_H="${BOX_MAX_H:-4}"
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }

VECTORS=$(seq -f 'kv%02g' 0 23)

notes() {                        # the search's progress, for the finish reason
    local done
    done=$(grep -c '"event": "match"' "$OUT/search.jsonl" 2>/dev/null)
    echo "matches recorded: ${done:-0}; rankings: $(grep -c '"event": "ranking"' "$OUT/search.jsonl" 2>/dev/null)"
}
# shellcheck disable=SC2317 # called by the library, in the reason of an unexpected exit
box_notes() { notes; }
box_on_round() {                 # progress.txt, before each upload round
    { date -u
      notes
      tail -n 3 "$OUT/search.jsonl" 2>/dev/null | cut -c 1-400
      tail -n 12 "$OUT/walls.txt" 2>/dev/null
      tail -n 4 "$OUT/search.log" 2>/dev/null
    } > "$OUT/progress.txt.tmp" && mv -f "$OUT/progress.txt.tmp" "$OUT/progress.txt"
}

box_init
[ -n "$STAGE" ] || box_finish "NO_STAGE: build the code stage (tools/stage_code.py) and pass STAGE" 1
box_bind_run_stage

# ---- the records of an earlier entry: the search's log and fit, each vector's games as the tarball it went up as
restored=(search.jsonl confirm.fit.json winner_decode.json search.log)
for name in $VECTORS confirm; do                 # a directory already here is not fetched again
    [ -d "$OUT/games_$name" ] || restored+=("games_$name.tar.gz")
done
box_restore "${restored[@]}" || box_finish "RESTORE_FAILED (restore.log)" 1
for name in $VECTORS confirm; do
    if [ -f "$OUT/games_$name.tar.gz" ]; then
        [ -d "$OUT/games_$name" ] || timeout -k 30s 10m tar -xzf "$OUT/games_$name.tar.gz" -C "$OUT" \
            || box_finish "GAMES_RESTORE_FAILED ($name)" 1
        rm -f "$OUT/games_$name.tar.gz"
        # The directory is what HF holds: it goes up again only once it changes.
        box_mark_landed "games_$name.tar.gz" "$OUT/games_$name" \
            || echo "games_$name will go up again (restore.log)"
    fi
    box_upload_dir "games_$name" "$OUT/games_$name"
done
box_pip huggingface_hub psutil pytest scipy requests || echo "pip install failed (pip.log)"

# ---- the code and the Rust wheel (the matches run on the core; the slope reads it)
box_stage_code || box_finish "CODE_STAGING_FAILED (staging.log)" 1
cd "$BOX_REPO" || box_finish "CODE_STAGING_FAILED (no $BOX_REPO)" 1
box_build_wheel || box_finish "BUILD_FAILED (build.log)" 1
box_facts > "$OUT/box.txt.tmp" 2>&1
mv -f "$OUT/box.txt.tmp" "$OUT/box.txt"
timeout 2m python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || box_finish "NO_CUDA (box.txt)" 1
box_upload_async
box_monitor_start

# ---- the crash barrier: the decode, its flags, the guards and the driver, on this wheel and this GPU
if ! box_marked_this_stage "$BOX_STATE/TESTED"; then
    : > "$OUT/tests.log"
    box_bounded tests "$TESTS_CUT_MIN" tests.log python -m pytest tests/test_decode_knobs.py \
        tests/test_knob_search.py tests/test_raw_player.py tests/test_end_turn_rule.py \
        tests/test_reference_player.py tests/test_attack_pathfinding.py tests/test_game_core.py \
        tests/test_parity_integration.py tests/test_match_memory.py tests/test_eval_workers.py \
        tests/test_eval_inference_server.py tests/test_eval_match_failures.py tests/test_eval_refusals.py \
        tests/test_game_record.py tests/test_elo_collect.py -q -p no:cacheprovider -m ""
    tail -n 3 "$OUT/tests.log"
    [ "$BOX_RC" -eq 0 ] || box_finish "TESTS_FAILED rc=$BOX_RC (tests.log)" 1
    box_mark "$BOX_STATE/TESTED"
fi

# ---- the reference: its checkpoint (SHA-256 checked); the search plays it on both sides
box_bounded reference 15 staging.log python tools/reference_player.py --ensure \
    || box_finish "REFERENCE_MISSING rc=$BOX_RC (staging.log)" 1
REF_LABEL=$(timeout 1m python -c "import json; print(json.load(open('configs/reference_player.json'))['label'])") \
    || box_finish "REFERENCE_CONFIG_UNREADABLE (configs/reference_player.json)" 1
[ "$REF_LABEL" = own1 ] \
    || box_finish "REFERENCE_CHANGED: configs/reference_player.json names $REF_LABEL; this search tunes own1" 1
# The vectors and the first commands, recorded before any game.
timeout 5m python tools/knob_search.py --root "$OUT" --plan > "$OUT/plan.txt" 2>&1 \
    || box_finish "PLAN_FAILED (plan.txt)" 1

# ---- the search: one match per step, its games up after it
NULL_ARM_FLAG=()
[ "$IGNORE_NULL_ARM" != 1 ] || NULL_ARM_FLAG=(--ignore-null-arm)
search_step() {                  # search_step NAME CUT BUDGET ARGS...: one bounded call of the driver; its exit code
    local name=$1 cut=$2 budget=$3
    shift 3
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before $name: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    box_bounded "$name" "$cut" search.log python tools/knob_search.py --root "$OUT" "$@" \
        --jobs "$JOBS" --time-budget-min "$budget" ${NULL_ARM_FLAG[@]+"${NULL_ARM_FLAG[@]}"}
    return "$BOX_RC"
}
for stage in 1 2 3; do
    while :; do
        search_step "stage $stage" "${MATCH_CUT_MIN[stage]}" "${BUDGET_MIN[stage]}" --stage "$stage" --one
        rc=$?
        box_upload_async
        case $rc in
            0) break ;;
            10) ;;
            5) box_on_round
               box_finish "NULL_ARM_SUSPECT: today's decode outside two standard errors of 50% in stage 1 \
(search.jsonl); IGNORE_NULL_ARM=1 goes on" 1 ;;
            *) box_finish "SEARCH_FAILED stage $stage rc=$rc $BOX_WHY $(notes) (search.log, kv*.log)" 1 ;;
        esac
    done
done
if [ ! -f "$OUT/confirm.fit.json" ] || ! grep -q '"event": "confirmation"' "$OUT/search.jsonl"; then
    search_step confirmation "${MATCH_CUT_MIN[3]}" "${BUDGET_MIN[3]}" --confirm \
        || box_finish "CONFIRMATION_FAILED rc=$BOX_RC $BOX_WHY (search.log, confirm.log)" 1
fi
box_on_round
box_finish "KNOB_SEARCH_DONE $(notes); $(grep '"event": "confirmation"' "$OUT/search.jsonl" | tail -n 1 | cut -c 1-300)"
