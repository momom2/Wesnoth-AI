#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# Match L on one box (docs/recruit_rule_l_prereg_20261021.md; the design is
# docs/training_design_20261019/current_design.md, section 4 row "Box A: L"):
#   the CPU tests of the rule path and of the generation path, on the wheel
#   built from the stage, the reference's checkpoint, and the frozen rule
#   (RULE, its content checked against its hash, its seal beside it);
#   then own1 against itself on the pool configuration (120 games in flight,
#   six a worker process, the reference decode and 64 memory slots on both
#   sides, no coin), the rule on the player each game's seat coin chooses
#   (P 1/2, keyed by the seed), N_L decisive games in matches of CHUNK_GAMES
#   so each fails alone. Each match's rows are extracted
#   (tools/recruit_rule_readout.py extract), its games sent up as a tarball,
#   and the readout of the matches done rewritten (readout_l.json): the
#   hooked seat's Elo against the seal, the null line, fire rates by
#   faction, I-cap, the seat coin. A match that fails (a crash, a failed
#   game, its rows) is left out and the next one runs; two in a row stop the
#   run. A readout that fails a check stops it too: the later matches would
#   measure a path the checks refused.
#
# Seeds. The matches play 700000 to 700000 + N_L - 1 in chunks, a capped game
# of slot i replaced by seed + k x 1,000,000. Checked 2026-10-21 on main: no
# script, document or training/metrics record names a seed from 700000 to
# 739999, or one of those plus a multiple of 1,000,000; the seeds taken are
# 300000-312799 (own2's base match), 500000-506399 and 600000-659999 (the
# near-tie coins, their replacements 1.5 and 1.6 million up). N_L is refused
# above 40,000, so L stays below 740000.
#
# Expected wall on a 4090 with an EPYC 7702 (31 usable cores), from batch 1's
# box (training/metrics/near_tie_coins_20261020/walls.txt, stages.txt): entry
# to the first match 5.6 minutes (staging, wheel, tests 216 s); games at
# 22,500 to 22,900 an hour, each match's start-up and last wave included
# (match a 6,400 in 1,007 s, b05 10,000 in 1,601 s): 37,800 games 99 to
# 101 minutes; four extractions and readouts about 4 minutes (the coin rows
# of 10,000 games took 26 s); the final upload a few minutes. In all about
# 1.9 hours, $1.10 at $0.577 an hour; on a host 1.41x slower about 2.6
# hours. BOX_MAX_H 3.5.
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md).
# Records go up every 30 minutes, after each match and at the finish, which
# uploads before it stops the instance. Re-entry restores the readout and
# the rows of the matches done, skips them, and plays the next; a match that
# failed stays out (its .failed file: delete it on HF and on the disk to
# play it again).
# Never `set -x`: the HF token and the instance key are in the environment.
# box-needs: disk_gb=60 ram_gb=64 gpu_ram_gb=24 cores=32 gpu=4090
set -uo pipefail
WORKDIR=/workspace
STAGE="${STAGE:-}"
HF_DIR="${HF_DIR:-tier-b/recruit_rule_l_20261021}"
export HF_DIR
BOX_MAX_H="${BOX_MAX_H:-3.5}"
JOBS="${JOBS:-120}"
WORKER_GAMES="${WORKER_GAMES:-6}"
MIN_FREE_MB="${MIN_FREE_MB:-300}"
GAMES_PER_H="${GAMES_PER_H:-22500}"            # the pool's measured rate, for the steps' bounds
N_L="${N_L:-37800}"
CHUNK_GAMES="${CHUNK_GAMES:-10000}"
L_SEED_BASE="${L_SEED_BASE:-700000}"
RULE="${RULE:-configs/recruit_rule_bd_13f8f00e4a7c.json}"
TESTS_CUT_MIN="${TESTS_CUT_MIN:-40}"
ROWS_CUT_MIN="${ROWS_CUT_MIN:-30}"
READOUT_CUT_MIN="${READOUT_CUT_MIN:-30}"
OUT=$WORKDIR/recruit_rule_l
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }

CHUNKS=()                        # the matches: names, games, seed bases (the plan)
CHUNK_SIZES=()
CHUNK_BASES=()
RESTORE=()

# ---- the readers (begin)
verdict() {                      # verdict FILE: a readout's verdict and outcome, "not read" without one
    timeout 1m python -c "import json, sys; r = json.load(open(sys.argv[1])); print(r['verdict'], r.get('outcome', 'no reading'))" \
        "$1" 2>/dev/null || echo "not read"
}
# ---- the readers (end)
# shellcheck disable=SC2317 # called by the library, in the reason of an unexpected exit
box_notes() { notes; }
notes() {                        # what is done, for a finish reason
    local done_=0 name
    for name in "${CHUNKS[@]}"; do
        ! box_marked_this_stage "$OUT/match_$name.done" || done_=$(( done_ + 1 ))
    done
    echo "$done_ of ${#CHUNKS[@]} matches, readout $(verdict "$OUT/readout_l.json")"
}
box_on_round() {                 # progress.txt, before each upload round
    { date -u
      notes
      tail -n 2 "$OUT/matches.txt" 2>/dev/null
    } > "$OUT/progress.txt.tmp" && mv -f "$OUT/progress.txt.tmp" "$OUT/progress.txt"
}

box_init
[ -n "$STAGE" ] || box_finish "NO_STAGE: build the code stage (tools/stage_code.py) and pass STAGE" 1

# ---- the plan (begin): tests/test_recruit_rule_box.py runs this block and the stages with the library stubbed
plan_chunks() {
    local off=0 c=0 size
    while (( off < N_L )); do
        size=$CHUNK_GAMES
        (( off + size <= N_L )) || size=$(( N_L - off ))
        CHUNKS+=("$(printf 'l%02d' "$c")")
        CHUNK_SIZES+=("$size")
        CHUNK_BASES+=("$(( L_SEED_BASE + off ))")
        off=$(( off + size ))
        c=$(( c + 1 ))
    done
}
for v in N_L CHUNK_GAMES L_SEED_BASE JOBS; do
    if ! [[ ${!v} =~ ^[1-9][0-9]*$ ]]; then
        box_finish "BAD_PLAN: $v=${!v} is not a positive integer" 1
    fi
done
(( N_L <= 40000 )) || box_finish "BAD_PLAN: N_L=$N_L: L's seeds would pass 740000" 1
plan_chunks
RESTORE=(readout_l.json matches.txt matches_failed.txt)
for name in "${CHUNKS[@]}"; do
    RESTORE+=("rows_$name.jsonl" "match_$name.done" "match_$name.failed")
done
# ---- the plan (end)

box_bind_run_stage
box_restore "${RESTORE[@]}" || box_finish "RESTORE_FAILED (restore.log)" 1
box_pip huggingface_hub psutil pytest requests numpy || echo "pip install failed (pip.log)"

# ---- the code, the Rust wheel, the reference, the rule
box_stage_code || box_finish "CODE_STAGING_FAILED (staging.log)" 1
cd "$BOX_REPO" || box_finish "CODE_STAGING_FAILED (no $BOX_REPO)" 1
box_build_wheel || box_finish "BUILD_FAILED (build.log)" 1
box_facts > "$OUT/box.txt.tmp" 2>&1
mv -f "$OUT/box.txt.tmp" "$OUT/box.txt"
timeout 2m python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || box_finish "NO_CUDA (box.txt)" 1
box_upload
box_bounded reference 15 staging.log python tools/reference_player.py --ensure \
    || box_finish "REFERENCE_MISSING rc=$BOX_RC (staging.log)" 1
REF_PT=$BOX_REPO/$(timeout 1m python -c "import json; print(json.load(open('configs/reference_player.json'))['checkpoint_local'])") \
    || box_finish "REFERENCE_CONFIG_UNREADABLE (configs/reference_player.json)" 1
[ -f "$REF_PT" ] || box_finish "REFERENCE_MISSING ($REF_PT)" 1
RULE=$BOX_REPO/$RULE
timeout 1m python -c "import sys; from tools.recruit_rule import RecruitRule; print(RecruitRule.load(sys.argv[1]).hash)" \
    "$RULE" >> "$OUT/box.txt" 2>&1 && [ -f "${RULE%.json}_readout.json" ] \
    || box_finish "RULE_REFUSED ($RULE: its content against its hash, its seal beside it; box.txt)" 1

# ---- the CPU tests of the rule path and of the generation path, on this wheel: the gate
if ! box_marked_this_stage "$BOX_STATE/TESTED"; then
    : > "$OUT/tests.log"
    box_bounded tests "$TESTS_CUT_MIN" tests.log python -m pytest tests/test_recruit_rule.py tests/test_recruit_rule_box.py \
        tests/test_coin_games.py tests/test_near_tie_coins.py tests/test_coin_labels.py tests/test_game_record.py \
        tests/test_raw_player.py tests/test_raw_player_compact.py tests/test_end_turn_rule.py tests/test_decision_state.py \
        tests/test_match_memory.py tests/test_server_priors.py tests/test_eval_workers.py tests/test_eval_inference_server.py \
        tests/test_eval_match_failures.py tests/test_generation_identity.py tests/test_elo_collect.py tests/test_cli_help.py \
        -q -p no:cacheprovider -m ""
    tail -n 3 "$OUT/tests.log"
    [ "$BOX_RC" -eq 0 ] || box_finish "TESTS_FAILED rc=$BOX_RC (tests.log)" 1
    box_mark "$BOX_STATE/TESTED"
fi

# ---- the sides: the reference's flags under the labels own1a and own1b, its checkpoint by path
side_flags() {                   # side_flags SIDE LABEL: one flag or value per line
    local side=$1 label=$2 flags i
    mapfile -t flags < <(timeout 1m python tools/reference_player.py --flags "$side" | tr ' ' '\n')
    for (( i = 0; i + 1 < ${#flags[@]}; i++ )); do
        case ${flags[i]} in
            "--label-$side") flags[i + 1]=$label ;;
            "--spec-$side") flags[i + 1]=$REF_PT ;;
        esac
    done
    printf '%s\n' "${flags[@]}"
}
mapfile -t SIDE_A < <(side_flags a own1a)
mapfile -t SIDE_B < <(side_flags b own1b)
[[ " ${SIDE_A[*]} " == *" --spec-a $REF_PT "* && " ${SIDE_A[*]} " == *" --raw-end-turn-offset-a -1.5 "* \
    && " ${SIDE_B[*]} " == *" --memory-b 64 "* && " ${SIDE_B[*]} " == *" --label-b own1b "* ]] \
    || box_finish "REFERENCE_FLAGS_FAILED (tools/reference_player.py --flags): ${SIDE_A[*]} | ${SIDE_B[*]}" 1
box_monitor_start

# ---- the stages (begin)
POOL=(--mcts-sims 0 --raw-temperature-a 0 --raw-temperature-b 0 --persistent-workers --shared-inference
      --no-infer-compile --device cuda --jobs "$JOBS" --worker-games "$WORKER_GAMES" --min-free-mb "$MIN_FREE_MB")
count_games() {                  # count_games DIR: "<result files> <decisive>"
    timeout 5m python - "$1" <<'EOF'
import json, pathlib, sys
files = list(pathlib.Path(sys.argv[1]).glob("game_*.json"))
decisive = 0
for p in files:
    try:
        decisive += json.loads(p.read_text(encoding="utf-8")).get("outcome_a") in ("win", "loss")
    except Exception:
        pass
print(len(files), decisive)
EOF
}
play() {                         # play NAME GAMES SEED_BASE: one match into WORKDIR/games_NAME; false when it failed
    local name=$1 games=$2 base=$3 dir="$WORKDIR/games_$1" est t0 rc why failed f
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before match $name: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    est=$(( games * 60 / GAMES_PER_H + 1 ))     # minutes at the measured rate
    t0=$(box_now)
    box_bounded "match $name" "$(( 2 * est + 15 ))" "match_$name.log" python tools/run_elo_batch.py \
        "${SIDE_A[@]}" "${SIDE_B[@]}" --outdir "$dir" --games "$games" --max-extra-games "$(( games / 2 ))" \
        --seed-base "$base" "${POOL[@]}" --time-budget-min "$(( 3 * est / 2 + 10 ))" --recruit-rule "$RULE"
    rc=$BOX_RC why=$BOX_WHY
    failed=$(find "$dir" -maxdepth 1 -name 'failed_*.json' 2>/dev/null | wc -l)
    echo "$(box_stamp) $name: games $games from seed $base, rc=$rc $why, $(( $(box_now) - t0 )) s," \
         "results and decisive $(count_games "$dir"), failed $failed" >> "$OUT/matches.txt"
    for f in "$dir"/.inference_server_*.json "$dir"/.worker_stages.json; do
        [ ! -f "$f" ] || cp -f "$f" "$OUT/$name.${f##*/.}"
    done
    # run_elo_batch exits 3 or 4 when its games ended short of the decisive
    # results asked for (the time budget, the replacement guard): the games
    # played are read. Any other exit, or a failed game, fails the match.
    case $rc in
        0|3|4) [ "$failed" -eq 0 ] ;;
        *) return 1 ;;
    esac
}
rows_of() {                      # rows_of NAME: the rule rows of games_NAME
    box_bounded "rule rows $1" "$ROWS_CUT_MIN" rows.log python tools/recruit_rule_readout.py extract \
        --match "$1=$WORKDIR/games_$1" --rule "$RULE" --out "$OUT/rows_$1.jsonl"
}
read_out() {                     # read_out MATCH...: the readout of these matches into readout_l.json, its report appended to readout_l.log
    local m rows=()
    for m in "$@"; do
        rows+=("$OUT/rows_$m.jsonl")
    done
    echo "== $(box_stamp) readout_l: $*" >> "$OUT/readout_l.log"
    box_bounded "readout l" "$READOUT_CUT_MIN" readout_l.log python tools/recruit_rule_readout.py readout \
        --rows "${rows[@]}" --rule "$RULE" --plan "$OUT/matches.txt" --save-json "$OUT/readout_l.json"
}
match_failed() {                 # match_failed NAME WHY: NAME is left out, and says why
    echo "MATCH_FAILED $1: $2" | tee -a "$OUT/matches_failed.txt"
    echo "$2" > "$OUT/match_$1.failed"
}

stage_l() {                      # the matches; returns when the run ends
    local k name done_=() fails=0 fresh v
    for k in "${!CHUNKS[@]}"; do
        name=${CHUNKS[$k]}
        fresh=0
        if box_marked_this_stage "$OUT/match_$name.done"; then
            done_+=("$name")
        elif [ -f "$OUT/match_$name.failed" ]; then
            echo "LEFT_OUT $name: it failed at an earlier entry, $(cat "$OUT/match_$name.failed")" >> "$OUT/matches_failed.txt"
            continue
        elif ! play "$name" "${CHUNK_SIZES[$k]}" "${CHUNK_BASES[$k]}"; then
            match_failed "$name" "the match failed (matches.txt, match_$name.log)"
        elif ! rows_of "$name"; then
            match_failed "$name" "its rows failed, rc=$BOX_RC $BOX_WHY (rows.log)"
        else
            box_upload_dir "games_$name" "$WORKDIR/games_$name"
            box_mark "$OUT/match_$name.done"
            done_+=("$name")
            fresh=1
        fi
        if [ ! -f "$OUT/match_$name.done" ]; then
            fails=$(( fails + 1 ))
            (( fails < 2 )) || { STAGE_L="STOPPED: two matches failed in a row (matches_failed.txt)"; return; }
            continue
        fi
        fails=0
        [ "$fresh" = 1 ] || continue         # a match done at an earlier entry: its readout is restored
        read_out "${done_[@]}" || echo "READOUT_FAILED after $name rc=$BOX_RC (readout_l.log)" >> "$OUT/matches_failed.txt"
        box_upload_async
        v=$(verdict "$OUT/readout_l.json")
        [[ $v == PASS* ]] || { STAGE_L="CHECK_FAILED after $name: $v (readout_l.log)"; return; }
    done
    if [ "${#done_[@]}" -gt 0 ] && [ ! -f "$OUT/readout_l.json" ]; then
        read_out "${done_[@]}"
    fi
    STAGE_L="DONE: ${#done_[@]} of ${#CHUNKS[@]} matches read"
}

STAGE_L="not run"
stage_l
REASON="$STAGE_L, readout $(verdict "$OUT/readout_l.json")"
# ---- the stages (end)
case $STAGE_L in
    DONE*) box_finish "RECRUIT_RULE_L_DONE $REASON" ;;
    *) box_finish "RECRUIT_RULE_L_INCOMPLETE $REASON" 1 ;;
esac
