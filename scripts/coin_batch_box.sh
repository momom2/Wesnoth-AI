#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# Near-tie coins on one box: the attack known answer and batch 1
# (docs/near_tie_coins_prereg_20261020.md; the design is
# docs/training_design_20261019/current_design.md 3.1, 3.2, 3.7 and stages
# S0(a), S0(c) and S1 of section 4):
#   the CPU tests of the coin path and of the generation path, on the wheel
#   built from the stage, and the reference's checkpoint (SHA-256 checked);
#   stage A, S0(a): own1 against itself on the pool configuration (120 games in
#   flight, six a worker process, the reference decode on both sides), the
#   attack offset ATTACK_OFFSET on the player the game's attack coin chooses
#   (P 1/2, keyed by the seed), N_A games; their luck walk, their coin rows,
#   elo_collect's fit of A against B (a report: about 0 when the hook moves
#   between the players) and the readout (tools/coin_labels.py): the attack
#   coin's label, the hooked player's Elo, the dice control. A stage A that
#   fails or reads outside a window holds stage B (GATE_B_ON_A=1, the design's
#   rule for S0(a): fix the path before any coin game);
#   stage B, batch 1: own1 self-play with near-tie coins on both sides at band
#   COIN_BAND and propensity COIN_P, N_B games in matches of their own: the
#   first SMOKE_GAMES (the smoke, S0(c)), then CHUNK_GAMES each. Each match is
#   walked, its coin rows and plant rows written (tools/coin_plant.py), its
#   games sent up as a tarball, and the readout of the matches done rewritten
#   (readout_b.json): the mean label pooled, by mover, by 0.05-nat margin bin
#   and by pair type, I-coin, I-log, the band decisions per side-game, the
#   coin plant per mover side, the dice control. The smoke is a gate: unless
#   its own readout (readout_smoke.json) passes every check at 3 standard
#   errors, the entry finishes before the rest of the batch and says which
#   check failed. A later match whose run fails (a crash, a failed game) is
#   left out and the next one runs; two in a row stop the stage.
#
# Seeds. Stage A plays seeds 500000 to 500000 + N_A - 1, the smoke and
# batch 1 600000 to 600000 + N_B - 1, a capped game of slot i replaced by
# seed + k x 1,000,000. Checked 2026-10-20: no script, document or
# training/metrics record of the repository names a seed base above 250000
# (the decode-knob confirmation, 250000 to 250799), and the own2 match the
# lead runs apart plays 300000 to 312799 (+ k x 1,000,000); N_B is refused
# above 300,000, so batch 1 stays below 900000.
#
# Expected wall on a 4090 with 31-32 usable cores: bring-up, wheel and tests
# 20-25 minutes (scripts/fp8_inference_box.sh's estimate); stage A
# 6,400 games at the pool's steady 21,966 games an hour (ref_p120x6,
# docs/training_design_20261019/analysis/round7_provenance_price_tail.out)
# 17.5 minutes, its walk about 1.5 (the 2026-10-19 walk read 800 games in
# 9-11 s on 31 workers), the rest 2: about 22 minutes; stage B 100,000 games
# 4.55 hours, eleven matches' start-up and last wave 11 x 36 s (16 s and 20 s
# in that record) 7 minutes, their walks 21 minutes, the coin and plant rows
# three passes over the records at about 1 ms a game (0.7 ms read on the
# laptop for a record without coins) 5 minutes, eleven readouts about 5: about
# 5.2 hours; the final upload a few minutes. In all about 6.0 hours, $2.5 at
# $0.42 an hour; on a host 1.41x slower about 8.0 hours. BOX_MAX_H 9.5.
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md).
# Records go up every 30 minutes, after each match and at the finish, which
# uploads before it stops the instance. Re-entry restores the readouts and
# the rows of the matches done, skips them, and plays the next; a stage or
# match that failed stays out (its .failed file: delete it on HF and on the disk to play it
# again).
# Never `set -x`: the HF token and the instance key are in the environment.
# box-needs: disk_gb=60 ram_gb=64 gpu_ram_gb=24 cores=32 gpu=4090
set -uo pipefail
WORKDIR=/workspace
STAGE="${STAGE:-}"
HF_DIR="${HF_DIR:-tier-b/near_tie_coins_20261020}"
export HF_DIR
BOX_MAX_H="${BOX_MAX_H:-9.5}"
BOX_WORKER_GB="${BOX_WORKER_GB:-0.5}"           # a walking worker holds about 230 MB (match_luck_box.sh)
JOBS="${JOBS:-120}"
WORKER_GAMES="${WORKER_GAMES:-6}"
MIN_FREE_MB="${MIN_FREE_MB:-300}"
GAMES_PER_H="${GAMES_PER_H:-21966}"            # the pool's steady rate, for the steps' bounds
N_A="${N_A:-6400}"
A_SEED_BASE="${A_SEED_BASE:-500000}"
ATTACK_OFFSET="${ATTACK_OFFSET:-1}"
GATE_B_ON_A="${GATE_B_ON_A:-1}"
N_B="${N_B:-100000}"
SMOKE_GAMES="${SMOKE_GAMES:-2000}"
CHUNK_GAMES="${CHUNK_GAMES:-10000}"
B_SEED_BASE="${B_SEED_BASE:-600000}"
COIN_BAND="${COIN_BAND:-0.35}"
COIN_P="${COIN_P:-0.25}"
BETA="${BETA:-configs/luck_beta_20261019.json}"
TESTS_CUT_MIN="${TESTS_CUT_MIN:-40}"
WALK_CUT_MIN="${WALK_CUT_MIN:-60}"
WALK_STALL_MIN="${WALK_STALL_MIN:-10}"         # no row for this long: the walk is cut
ROWS_CUT_MIN="${ROWS_CUT_MIN:-30}"
READOUT_CUT_MIN="${READOUT_CUT_MIN:-30}"
OUT=$WORKDIR/near_tie_coins
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }

CHUNKS=()                        # stage B's matches: names, games, seed bases (the plan)
CHUNK_SIZES=()
CHUNK_BASES=()
RESTORE=()

# ---- the readers (begin)
verdict() {                      # verdict FILE: a readout's verdict, "not read" without one
    timeout 1m python -c "import json, sys; print(json.load(open(sys.argv[1]))['verdict'])" "$1" 2>/dev/null \
        || echo "not read"
}
# ---- the readers (end)
# shellcheck disable=SC2317 # called by the library, in the reason of an unexpected exit
box_notes() { notes; }
notes() {                        # what is done, for a finish reason
    local done_=0 name
    for name in "${CHUNKS[@]}"; do
        ! box_marked_this_stage "$OUT/match_$name.done" || done_=$(( done_ + 1 ))
    done
    echo "stage A: $(verdict "$OUT/readout_a.json"); stage B: $done_ of ${#CHUNKS[@]} matches, $(verdict "$OUT/readout_b.json")"
}
box_on_round() {                 # progress.txt, before each upload round
    { date -u
      notes
      tail -n 2 "$OUT/matches.txt" 2>/dev/null
    } > "$OUT/progress.txt.tmp" && mv -f "$OUT/progress.txt.tmp" "$OUT/progress.txt"
}

box_init
[ -n "$STAGE" ] || box_finish "NO_STAGE: build the code stage (tools/stage_code.py) and pass STAGE" 1

# ---- the plan (begin): tests/test_coin_batch_box.py runs this block and the stages with the library stubbed
plan_chunks() {
    local off=0 c=0 size
    while (( off < N_B )); do
        size=$(( c == 0 ? SMOKE_GAMES : CHUNK_GAMES ))
        (( off + size <= N_B )) || size=$(( N_B - off ))
        CHUNKS+=("$(printf 'b%02d' "$c")")
        CHUNK_SIZES+=("$size")
        CHUNK_BASES+=("$(( B_SEED_BASE + off ))")
        off=$(( off + size ))
        c=$(( c + 1 ))
    done
}
for v in N_A N_B SMOKE_GAMES CHUNK_GAMES A_SEED_BASE B_SEED_BASE JOBS; do
    if ! [[ ${!v} =~ ^[1-9][0-9]*$ ]]; then
        box_finish "BAD_PLAN: $v=${!v} is not a positive integer" 1
    fi
done
(( N_B <= 300000 )) || box_finish "BAD_PLAN: N_B=$N_B: batch 1's seeds would pass 900000" 1
plan_chunks
RESTORE=(readout_a.json elo_a.json rows_a.jsonl coins_a.jsonl stage_a.done stage_a.failed
         readout_smoke.json readout_b.json matches.txt matches_failed.txt)
for name in "${CHUNKS[@]}"; do
    RESTORE+=("rows_$name.jsonl" "coins_$name.jsonl" "plant_$name.jsonl" "match_$name.done" "match_$name.failed")
done
# ---- the plan (end)

box_bind_run_stage
box_restore "${RESTORE[@]}" || box_finish "RESTORE_FAILED (restore.log)" 1
box_pip huggingface_hub psutil pytest requests numpy || echo "pip install failed (pip.log)"

# ---- the code, the Rust wheel, the reference
box_stage_code || box_finish "CODE_STAGING_FAILED (staging.log)" 1
cd "$BOX_REPO" || box_finish "CODE_STAGING_FAILED (no $BOX_REPO)" 1
box_build_wheel || box_finish "BUILD_FAILED (build.log)" 1
box_facts > "$OUT/box.txt.tmp" 2>&1
mv -f "$OUT/box.txt.tmp" "$OUT/box.txt"
timeout 2m python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || box_finish "NO_CUDA (box.txt)" 1
WALK_JOBS=$(box_workers)
echo "walk jobs $WALK_JOBS" >> "$OUT/box.txt"
box_upload
box_bounded reference 15 staging.log python tools/reference_player.py --ensure \
    || box_finish "REFERENCE_MISSING rc=$BOX_RC (staging.log)" 1
REF_PT=$BOX_REPO/$(timeout 1m python -c "import json; print(json.load(open('configs/reference_player.json'))['checkpoint_local'])") \
    || box_finish "REFERENCE_CONFIG_UNREADABLE (configs/reference_player.json)" 1
[ -f "$REF_PT" ] || box_finish "REFERENCE_MISSING ($REF_PT)" 1
[ -f "$BETA" ] || box_finish "BETA_MISSING ($BETA)" 1

# ---- the CPU tests of the coin path and of the generation path, on this wheel: the gate
if ! box_marked_this_stage "$BOX_STATE/TESTED"; then
    : > "$OUT/tests.log"
    box_bounded tests "$TESTS_CUT_MIN" tests.log python -m pytest tests/test_near_tie_coins.py \
        tests/test_coin_labels.py tests/test_coin_games.py tests/test_coin_batch_box.py tests/test_match_luck.py \
        tests/test_game_record.py tests/test_raw_player.py tests/test_raw_player_compact.py tests/test_end_turn_rule.py \
        tests/test_decision_state.py tests/test_match_memory.py tests/test_server_priors.py tests/test_eval_workers.py \
        tests/test_eval_inference_server.py tests/test_eval_match_failures.py tests/test_generation_identity.py \
        tests/test_elo_collect.py tests/test_cli_help.py -q -p no:cacheprovider -m ""
    tail -n 3 "$OUT/tests.log"
    [ "$BOX_RC" -eq 0 ] || box_finish "TESTS_FAILED rc=$BOX_RC (tests.log)" 1
    box_mark "$BOX_STATE/TESTED"
fi

# ---- the self-play sides: the reference's flags under the labels own1a and own1b, its checkpoint by path
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
COINS=(--coin-band-a "$COIN_BAND" --coin-p-a "$COIN_P" --coin-band-b "$COIN_BAND" --coin-p-b "$COIN_P")
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
play() {                         # play NAME GAMES SEED_BASE FLAGS...: one match into WORKDIR/games_NAME; false when it failed
    local name=$1 games=$2 base=$3 dir="$WORKDIR/games_$1" est t0 rc why failed f
    shift 3
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before match $name: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    est=$(( games * 60 / GAMES_PER_H + 1 ))     # minutes at the steady rate
    t0=$(box_now)
    box_bounded "match $name" "$(( 2 * est + 15 ))" "match_$name.log" python tools/run_elo_batch.py \
        "${SIDE_A[@]}" "${SIDE_B[@]}" --outdir "$dir" --games "$games" --max-extra-games "$(( games / 2 ))" \
        --seed-base "$base" "${POOL[@]}" --time-budget-min "$(( 3 * est / 2 + 10 ))" "$@"
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
rows_of() {                      # rows_of NAME [plant]: the walk, the coin rows and the plant rows of games_NAME
    local name=$1 dir="$WORKDIR/games_$1"
    touch "$OUT/rows_$name.jsonl"
    box_bounded --stall "$OUT/rows_$name.jsonl" "$WALK_STALL_MIN" "walk $name" "$WALK_CUT_MIN" walk.log \
        python tools/match_luck.py walk --out "$OUT/rows_$name.jsonl" --jobs "$WALK_JOBS" --match "$name=$dir" \
        || return 1
    box_bounded "coin rows $name" "$ROWS_CUT_MIN" rows.log python tools/coin_labels.py extract \
        --match "$name=$dir" --out "$OUT/coins_$name.jsonl" || return 1
    [ "${2:-}" = plant ] || return 0
    box_bounded "plant rows $name" "$ROWS_CUT_MIN" rows.log python tools/coin_plant.py plant \
        --match "$name=$dir" --out "$OUT/plant_$name.jsonl"
}
read_out() {                     # read_out NAME PLANT MATCH...: the readout of these matches into NAME.json, its report appended to NAME.log
    local out=$1 plant=$2 m coins=() rows=() plants=()
    shift 2
    for m in "$@"; do
        coins+=("$OUT/coins_$m.jsonl")
        rows+=("$OUT/rows_$m.jsonl")
        [ "$plant" != plant ] || plants+=("$OUT/plant_$m.jsonl")
    done
    [ "${#plants[@]}" -eq 0 ] || plants=(--plant "${plants[@]}")
    echo "== $(box_stamp) $out: $*" >> "$OUT/$out.log"
    box_bounded "readout $out" "$READOUT_CUT_MIN" "$out.log" python tools/coin_labels.py readout \
        --coins "${coins[@]}" --rows "${rows[@]}" --beta "$BETA" --band "$COIN_BAND" \
        ${plants[@]+"${plants[@]}"} --plan "$OUT/matches.txt" --save-json "$OUT/$out.json"
}
match_failed() {                 # match_failed NAME WHY: NAME is left out, and says why
    echo "MATCH_FAILED $1: $2" | tee -a "$OUT/matches_failed.txt"
    echo "$2" > "$OUT/match_$1.failed"
}

stage_a() {                      # S0(a): the attack known answer, once
    box_marked_this_stage "$OUT/stage_a.done" && return 0
    [ ! -f "$OUT/stage_a.failed" ] || return 1
    if ! play a "$N_A" "$A_SEED_BASE" --attack-coin-offset "$ATTACK_OFFSET"; then
        match_failed a "the match failed (matches.txt, match_a.log)"
        echo "match" > "$OUT/stage_a.failed"
        return 1
    fi
    box_upload_dir games_a "$WORKDIR/games_a"
    if ! rows_of a; then
        match_failed a "its rows failed, rc=$BOX_RC $BOX_WHY (walk.log, rows.log)"
        echo "rows" > "$OUT/stage_a.failed"
        return 1
    fi
    box_bounded "elo a" 15 elo_a.log python tools/elo_collect.py "$WORKDIR/games_a" --no-catalog \
        --save-json "$OUT/elo_a.json" || echo "ELO_A_FAILED rc=$BOX_RC (elo_a.log)" >> "$OUT/matches_failed.txt"
    read_out readout_a noplant a || { echo "readout" > "$OUT/stage_a.failed"; return 1; }
    box_mark "$OUT/stage_a.done"
    box_upload_async
}

stage_b() {                      # batch 1, the smoke first; returns when the stage ends
    local k name done_=() fails=0 smoke fresh
    for k in "${!CHUNKS[@]}"; do
        name=${CHUNKS[$k]}
        fresh=0
        if box_marked_this_stage "$OUT/match_$name.done"; then
            done_+=("$name")
        elif [ -f "$OUT/match_$name.failed" ]; then
            echo "LEFT_OUT $name: it failed at an earlier entry, $(cat "$OUT/match_$name.failed")" >> "$OUT/matches_failed.txt"
            [ "$k" -gt 0 ] || { STAGE_B="SMOKE_FAILED: $(cat "$OUT/match_$name.failed")"; return; }
            continue
        elif ! play "$name" "${CHUNK_SIZES[$k]}" "${CHUNK_BASES[$k]}" "${COINS[@]}"; then
            match_failed "$name" "the match failed (matches.txt, match_$name.log)"
        elif ! rows_of "$name" plant; then
            match_failed "$name" "its rows failed, rc=$BOX_RC $BOX_WHY (walk.log, rows.log)"
        else
            box_upload_dir "games_$name" "$WORKDIR/games_$name"
            box_mark "$OUT/match_$name.done"
            done_+=("$name")
            fresh=1
        fi
        if [ ! -f "$OUT/match_$name.done" ]; then
            [ "$k" -gt 0 ] || { STAGE_B="SMOKE_FAILED: $(cat "$OUT/match_$name.failed")"; return; }
            fails=$(( fails + 1 ))
            (( fails < 2 )) || { STAGE_B="STAGE_B_STOPPED: two matches failed in a row (matches_failed.txt)"; return; }
            continue
        fi
        fails=0
        if [ "$k" -eq 0 ]; then              # the smoke's own readout is the gate
            [ -f "$OUT/readout_smoke.json" ] || read_out readout_smoke plant "$name"
            smoke=$(verdict "$OUT/readout_smoke.json")
            [ "$smoke" = PASS ] || { STAGE_B="SMOKE_GATE_FAILED: $smoke (readout_smoke.log)"; return; }
        fi
        [ "$fresh" = 1 ] || continue         # a match done at an earlier entry: its readout is restored
        read_out readout_b plant "${done_[@]}" || echo "READOUT_FAILED after $name rc=$BOX_RC (readout_b.log)" \
            >> "$OUT/matches_failed.txt"
        box_upload_async
    done
    if [ "${#done_[@]}" -gt 0 ] && [ ! -f "$OUT/readout_b.json" ]; then
        read_out readout_b plant "${done_[@]}"
    fi
    STAGE_B="BATCH_DONE: ${#done_[@]} of ${#CHUNKS[@]} matches read"
}

STAGE_B="not run"
stage_a || true
VERDICT_A=$(verdict "$OUT/readout_a.json")
if [ "$GATE_B_ON_A" = 1 ] && [ "$VERDICT_A" != PASS ]; then
    STAGE_B="HELD: stage A reads '$VERDICT_A'"
else
    stage_b
fi
VERDICT_B=$(verdict "$OUT/readout_b.json")
REASON="stage A: $VERDICT_A; stage B: $STAGE_B, readout $VERDICT_B"
# ---- the stages (end)
case $STAGE_B in
    BATCH_DONE*) box_finish "NEAR_TIE_COINS_DONE $REASON" ;;
    *) box_finish "NEAR_TIE_COINS_INCOMPLETE $REASON" 1 ;;
esac
