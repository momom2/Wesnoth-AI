#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# fp8 products for self-play generation, on one box
# (docs/fp8_inference_prereg_20261019.md, branch feature/generation-speed-6):
#   the tests of the generation path and of this branch on the wheel built
#   from the stage, the reference's checkpoint (its SHA-256 checked), the
#   earlier rounds' CUDA tests (recorded), then the fp8 tests on this GPU
#   and the reference's weights: the gate. Unless every fp8 test passes,
#   none skipped, the entry finishes there (FP8_GATE_FAILED);
#   the speed arms: production-configured matches of the reference against
#   itself, 120 games in flight with six a worker process, the same games
#   as round 5's (seeds from SEED_BASE), each server appending its counters
#   to a timeline every TIMELINE_S seconds, in this order:
#     ref_1 fp8c_1 bf16e fp8e ref_2 profile_ref profile_fp8c fp8c_2 ref_3
#   ref is round 5's fastest configuration (the compiled loop on the bf16
#   residual stream), fp8c the same with fp8 products on the one server both
#   sides share, bf16e and fp8e the two without the compiled loop, profile_*
#   ref and fp8c with PROFILE_BATCHES batches under torch.profiler (the
#   device time by kernel class; not timed). Each arm fails alone: a match
#   that crashes or loses a game, a server that fails a request, a compiled
#   loop that fell back to eager, an fp8 arm whose server did not play fp8
#   (tools/fp8_inference_readout.py speed, speed.json rewritten after every
#   arm, arms_failed.txt) leaves the next arm to run;
#   the play check: the reference with fp8 (side A, reffp8) against the
#   reference without it (side B, ref), both on ref's configuration at the
#   reference decode, PLAY_GAMES decisive games from PLAY_SEED_BASE, every
#   game recorded whole and its games sent up as one tarball; fitted
#   (tools/elo_collect.py) and read (tools/fp8_inference_readout.py play:
#   PASS, KILL below -2 standard errors, NOT_READ when short).
#
# The readings are the servers' counters, never a match's wall: identical
# repeats have differed 1.6-1.8x in wall on a shared host. Each speed arm is
# read over a steady window of its server's timeline (from 20 s after the
# first request, 60 s, inside the 120 games in flight) and over the
# server's whole serving wall (round 5's reading). The three ref runs,
# before, between and after the others, bracket the drift: the session is
# valid when each lies within VALID_SPREAD of their mean.
#
# Expected wall on a 4090 with 31-32 usable cores (round 5's walls,
# training/metrics/generation_speed_r5_20261015/walls.txt): bring-up, wheel
# and tests 20-25 minutes, the fp8 tests 5-10 (inductor compiles the fp8
# loop twice), the nine arms 2 to 3 minutes each (600 games; a compiled
# server compiles for about a minute before serving), the play check 12 to
# 16 minutes (3,200 decisive games at round 5's 4.8 games a second, slower
# with two servers on the GPU), the final upload a few minutes: about 1.2
# hours, $0.50-0.55 at $0.42-0.46 an hour. BOX_MAX_H 2.5 caps it at
# $1.05-1.15 plus the final round.
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md). No
# periodic upload rounds while the arms run (they would share the machine
# with the timing): the records go up after each arm, after the play check
# and at the finish, which uploads before it stops the instance. Re-entry
# skips the arms that are done and resumes the play check in its directory.
# Never `set -x`: the HF token and the instance key are in the environment.
# box-needs: disk_gb=60 ram_gb=64 gpu_ram_gb=24 cores=32 gpu=4090
set -uo pipefail
WORKDIR=/workspace
STAGE="${STAGE:-}"
HF_DIR="${HF_DIR:-tier-b/fp8_inference_20261019}"
export HF_DIR
BOX_MAX_H="${BOX_MAX_H:-2.5}"
JOBS="${JOBS:-120}"
GAMES="${GAMES:-400}"
GAMES_PER_JOB="${GAMES_PER_JOB:-5}"
TIMELINE_S="${TIMELINE_S:-5}"
PROFILE_BATCHES="${PROFILE_BATCHES:-40}"       # a profiled arm's batches under torch.profiler
PROFILE_AFTER="${PROFILE_AFTER:-400}"           # the batches each profiled server serves first
# The speed arms play round 5's games: no committed result carries a seed
# from 130000 to 130999 but the generation speed runs'.
SEED_BASE="${SEED_BASE:-130000}"
# The play check plays seeds 140000 to 143199 and replaces a capped game of
# slot i with seed 140000 + i + k * 1,000,000; no committed result, branch
# or tag of 2026-10-19 names a seed from 131000 to 1,149,999 but the box
# probe's 131000-131999.
PLAY_SEED_BASE="${PLAY_SEED_BASE:-140000}"
PLAY_GAMES="${PLAY_GAMES:-3200}"
PLAY_EXTRA="${PLAY_EXTRA:-1500}"
VALID_SPREAD="${VALID_SPREAD:-0.05}"
TESTS_CUT_MIN="${TESTS_CUT_MIN:-40}"
CUDA_CUT_MIN="${CUDA_CUT_MIN:-30}"
ARM_CUT_MIN="${ARM_CUT_MIN:-30}"
MATCH_BUDGET_MIN="${MATCH_BUDGET_MIN:-22}"     # a speed match stops admitting games here; the arm's cut is ARM_CUT_MIN
PLAY_BUDGET_MIN="${PLAY_BUDGET_MIN:-45}"       # the play check stops admitting games here
PLAY_CUT_MIN="${PLAY_CUT_MIN:-70}"
OUT=$WORKDIR/fp8_inference
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }

# ---- the arm list (begin): tests/test_fp8_inference_box.py reads it
REF="--compile-packed --bf16-residual"
FP8="--fp8-inference-a --fp8-inference-b"
PROFILE="--inference-profile-batches $PROFILE_BATCHES --inference-profile-after $PROFILE_AFTER"
ARM_SPECS=(
    "ref_1 $REF"
    "fp8c_1 $REF $FP8"
    "bf16e --bf16-residual"
    "fp8e --bf16-residual $FP8"
    "ref_2 $REF"
    "profile_ref $REF $PROFILE"
    "profile_fp8c $REF $FP8 $PROFILE"
    "fp8c_2 $REF $FP8"
    "ref_3 $REF"
)
ARMS=""
for spec in "${ARM_SPECS[@]}"; do ARMS="$ARMS ${spec%% *}"; done
# The play check: side A's server plays fp8, side B's does not.
PLAY_FLAGS="$REF --fp8-inference-a"
PLAY_LABEL_A=reffp8
PLAY_LABEL_B=ref
# ---- the arm list (end)

# shellcheck disable=SC2317 # called through box_notes, by the library
notes() {                        # what is done, for the finish reason
    local name done_=() play="not read"
    for name in $ARMS; do
        ! box_marked_this_stage "$OUT/arm_$name.done" || done_+=("$name")
    done
    [ ! -f "$OUT/play.json" ] || play="read"
    echo "done: ${done_[*]:-nothing}; play: $play"
}
# shellcheck disable=SC2317 # called by the library, in the reason of an unexpected exit
box_notes() { notes; }

box_init
[ -n "$STAGE" ] || box_finish "NO_STAGE: build the code stage (tools/stage_code.py) and pass STAGE" 1
box_bind_run_stage
box_restore speed.json tests.log cuda_tests.log fp8_cuda_tests.log play.fit.json play.json timing_play.txt \
    || box_finish "RESTORE_FAILED (restore.log)" 1
if [ ! -d "$OUT/games_play" ]; then        # the play check's games as the tarball they went up as
    box_restore games_play.tar.gz || box_finish "RESTORE_FAILED (restore.log)" 1
fi
if [ -f "$OUT/games_play.tar.gz" ]; then
    [ -d "$OUT/games_play" ] || timeout -k 30s 10m tar -xzf "$OUT/games_play.tar.gz" -C "$OUT" \
        || box_finish "PLAY_RESTORE_FAILED (games_play.tar.gz)" 1
    rm -f "$OUT/games_play.tar.gz"
    box_mark_landed games_play.tar.gz "$OUT/games_play" || echo "games_play will go up again (restore.log)"
fi
box_pip huggingface_hub psutil pytest scipy requests || echo "pip install failed (pip.log)"

# ---- the code and the Rust wheel
box_stage_code || box_finish "CODE_STAGING_FAILED (staging.log)" 1
cd "$BOX_REPO" || box_finish "CODE_STAGING_FAILED (no $BOX_REPO)" 1
box_build_wheel || box_finish "BUILD_FAILED (build.log)" 1
box_facts > "$OUT/box.txt.tmp" 2>&1
mv -f "$OUT/box.txt.tmp" "$OUT/box.txt"
timeout 2m python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || box_finish "NO_CUDA (box.txt)" 1
box_upload

# ---- the reference: its checkpoint (SHA-256 checked), before the tests that read its weights
box_bounded reference 15 staging.log python tools/reference_player.py --ensure \
    || box_finish "REFERENCE_MISSING rc=$BOX_RC (staging.log)" 1
REF_PT=$BOX_REPO/$(timeout 1m python -c "import json; print(json.load(open('configs/reference_player.json'))['checkpoint_local'])") \
    || box_finish "REFERENCE_CONFIG_UNREADABLE (configs/reference_player.json)" 1
[ -f "$REF_PT" ] || box_finish "REFERENCE_MISSING ($REF_PT)" 1

# ---- the tests of the generation path and of this branch, on this wheel; the earlier CUDA tests are recorded
if ! box_marked_this_stage "$BOX_STATE/TESTED"; then
    : > "$OUT/tests.log"
    box_bounded tests "$TESTS_CUT_MIN" tests.log python -m pytest tests/test_generation_identity.py \
        tests/test_serve_stage.py tests/test_server_profile.py tests/test_packed_compile.py \
        tests/test_view_cache.py tests/test_core_masks.py tests/test_core_view.py tests/test_decision_state.py \
        tests/test_eval_workers.py tests/test_eval_inference_server.py tests/test_stage_timing.py \
        tests/test_match_memory.py tests/test_server_priors.py tests/test_game_core.py tests/test_memory_model.py \
        tests/test_eval_match_failures.py tests/test_raw_player.py tests/test_inference_seam.py \
        tests/test_leaf_wire.py tests/test_fp8_linear.py tests/test_fp8_inference.py tests/test_fp8_provenance.py \
        tests/test_box_probe.py tests/test_fp8_inference_readout.py tests/test_elo_collect.py \
        tests/test_game_record.py -q -p no:cacheprovider -m ""
    tail -n 3 "$OUT/tests.log"
    [ "$BOX_RC" -eq 0 ] || box_finish "TESTS_FAILED rc=$BOX_RC (tests.log)" 1
    : > "$OUT/cuda_tests.log"
    box_bounded cuda-tests 25 cuda_tests.log python -m pytest tests/test_server_priors_cuda.py \
        tests/test_packed_compile_cuda.py \
        tests/test_match_memory.py::test_the_served_memory_batch_is_the_slot_memorys_padding \
        "tests/test_serve_stage.py::test_on_cuda_the_rust_stage_answers_as_the_python_path" \
        "tests/test_serve_stage.py::test_on_cuda_the_kept_memory_plays_as_the_wire" \
        -q -p no:cacheprovider -m ""
    tail -n 3 "$OUT/cuda_tests.log"
    box_mark "$BOX_STATE/TESTED"
fi
CUDA_TESTS=$(grep -Eo "[0-9]+ (passed|failed|error|errors|skipped)" "$OUT/cuda_tests.log" 2>/dev/null | tr '\n' ' ')

# ---- the gate: the fp8 tests on this GPU and the reference's weights, every one passed, none skipped
if ! box_marked_this_stage "$BOX_STATE/FP8_GATE"; then
    : > "$OUT/fp8_cuda_tests.log"
    box_bounded fp8-cuda-tests "$CUDA_CUT_MIN" fp8_cuda_tests.log python -m pytest tests/test_fp8_inference_cuda.py \
        -q -s -p no:cacheprovider -m ""
    tail -n 3 "$OUT/fp8_cuda_tests.log"
    FP8_TESTS=$(grep -Eo "[0-9]+ (passed|failed|error|errors|skipped)" "$OUT/fp8_cuda_tests.log" | tr '\n' ' ')
    { [ "$BOX_RC" -eq 0 ] && [[ $FP8_TESTS == *passed* ]] && [[ $FP8_TESTS != *skipped* ]]; } \
        || box_finish "FP8_GATE_FAILED rc=$BOX_RC: ${FP8_TESTS:-no summary} (fp8_cuda_tests.log)" 1
    box_mark "$BOX_STATE/FP8_GATE"
fi
FP8_TESTS=$(grep -Eo "[0-9]+ (passed|failed|error|errors|skipped)" "$OUT/fp8_cuda_tests.log" 2>/dev/null | tr '\n' ' ')
box_upload

# ---- the self-play sides' flags
side_flags() {                   # side_flags SIDE LABEL: the reference's flags for SIDE, one per line, under LABEL, its checkpoint by absolute path
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
mapfile -t SIDE_A < <(side_flags a refa)
mapfile -t SIDE_B < <(side_flags b refb)
mapfile -t PLAY_A < <(side_flags a "$PLAY_LABEL_A")
mapfile -t PLAY_B < <(side_flags b "$PLAY_LABEL_B")
[[ " ${SIDE_A[*]} " == *" --spec-a $REF_PT "* && " ${SIDE_B[*]} " == *" --memory-b 64 "* \
    && " ${PLAY_A[*]} " == *" --label-a $PLAY_LABEL_A "* && " ${PLAY_B[*]} " == *" --spec-b $REF_PT "* ]] \
    || box_finish "REFERENCE_FLAGS_FAILED (tools/reference_player.py --flags): ${SIDE_A[*]} | ${PLAY_B[*]}" 1

# ---- the arms (begin): tests/test_fp8_inference_box.py runs this block with the library stubbed
summarize() {                    # speed.json from the arms so far
    box_bounded summary 5 summary.log python tools/fp8_inference_readout.py speed "$OUT" --spread "$VALID_SPREAD"
}
match_flags() {                  # match_flags: what every match of this run passes to the match driver
    printf '%s\n' --mcts-sims 0 --raw-temperature-a 0 --raw-temperature-b 0 --persistent-workers \
        --shared-inference --no-infer-compile --device cuda --jobs "$JOBS" --worker-games 6 --min-free-mb 300 \
        --inference-timeline-s "$TIMELINE_S"
}
mapfile -t MATCH_FLAGS < <(match_flags)
arm() {                          # arm NAME FLAGS...: one speed match, once; false when it failed or lost a game
    local name=$1 dir="$WORKDIR/games_$1" t0 f games failed rc why
    shift
    ! box_marked_this_stage "$OUT/arm_$name.done" || return 0
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before arm $name: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    rm -rf "$dir"
    games=$(( GAMES_PER_JOB * JOBS > GAMES ? GAMES_PER_JOB * JOBS : GAMES ))
    echo "$(box_stamp) $name: jobs $JOBS, games $games, $*" >> "$OUT/arm_$name.log"
    t0=$(box_now)
    box_bounded "arm $name" "$ARM_CUT_MIN" "arm_$name.log" python tools/run_elo_batch.py \
        "${SIDE_A[@]}" "${SIDE_B[@]}" --outdir "$dir" --games "$games" --max-extra-games 0 \
        --seed-base "$SEED_BASE" "${MATCH_FLAGS[@]}" --time-budget-min "$MATCH_BUDGET_MIN" "$@"
    rc=$BOX_RC why=$BOX_WHY              # the summary's own step resets BOX_RC
    failed=$(find "$dir" -maxdepth 1 -name 'failed_*.json' 2>/dev/null | wc -l)
    printf '{"arm": "%s", "jobs": %d, "flags": "%s", "slots": %d, "wall_s": %d, "games": %d, "failed_games": %d, "rc": %d, "why": "%s"}\n' \
        "$name" "$JOBS" "$*" "$games" "$(( $(box_now) - t0 ))" \
        "$(find "$dir" -maxdepth 1 -name 'game_*.json' 2>/dev/null | wc -l)" "$failed" "$rc" "$why" \
        > "$OUT/arm_$name.json"
    for f in "$dir"/.inference_server_*.json "$dir"/.inference_server_*.timeline.jsonl "$dir"/.worker_stages.json; do
        [ ! -f "$f" ] || cp -f "$f" "$OUT/arm_${name}.${f##*/.}"
    done
    box_mark "$OUT/arm_$name.done"
    summarize
    box_upload
    # run_elo_batch exits 3 or 4 when its games end without enough decisive
    # results (no replacements here): the timing is whole. Any other exit,
    # or a failed game, fails the arm; the readout fails it for its servers.
    case $rc in
        0|3|4) ;;
        *) echo "ARM_FAILED $name rc=$rc $why (arm_$name.log)" | tee -a "$OUT/arms_failed.txt"
           return 1 ;;
    esac
    if [ "$failed" -gt 0 ]; then
        echo "ARM_FAILED $name: $failed failed games (arm_$name.log, games_$name/failed_*.json)" \
            | tee -a "$OUT/arms_failed.txt"
        return 1
    fi
}
decisive_results() {             # decisive_results DIR: the games in DIR that ended in a win or a loss
    timeout 2m python - "$1" <<'EOF'
import json, pathlib, sys
n = 0
for p in pathlib.Path(sys.argv[1]).glob("game_*.json"):
    try:
        n += json.loads(p.read_text(encoding="utf-8")).get("outcome_a") in ("win", "loss")
    except Exception:
        pass
print(n)
EOF
}
play_check() {                   # the play check, resumed in its directory, then its fit and its reading
    local dir="$OUT/games_play" t0 f rc rc_flag=()
    [ ! -f "$OUT/play.json" ] || return 0
    box_upload_dir games_play "$dir"
    box_upload_hold play.fit.json games_play.tar.gz
    box_upload_hold play.json games_play.tar.gz
    if [ ! -f "$OUT/timing_play.txt" ]; then           # the games; a re-entry after them redoes only the reading
        box_gpu_ok || box_finish "GPU_UNRESPONSIVE before the play check: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
        echo "$(box_stamp) play: seed base $PLAY_SEED_BASE, $PLAY_GAMES games, $PLAY_FLAGS" >> "$OUT/play.log"
        t0=$(box_now)
        # shellcheck disable=SC2086 # PLAY_FLAGS is a list of flags
        box_bounded "match play" "$PLAY_CUT_MIN" play.log python tools/run_elo_batch.py \
            "${PLAY_A[@]}" "${PLAY_B[@]}" --outdir "$dir" --games "$PLAY_GAMES" --max-extra-games "$PLAY_EXTRA" \
            --seed-base "$PLAY_SEED_BASE" "${MATCH_FLAGS[@]}" --time-budget-min "$PLAY_BUDGET_MIN" $PLAY_FLAGS
        rc=$BOX_RC
        echo "play: $(( $(box_now) - t0 )) s, $(find "$dir" -maxdepth 1 -name 'game_*.json' 2>/dev/null | wc -l) games," \
             "$(decisive_results "$dir") decisive, rc=$rc $BOX_WHY" > "$OUT/timing_play.txt"
        for f in "$dir"/.inference_server_*.json "$dir"/.inference_server_*.timeline.jsonl; do
            [ ! -f "$f" ] || cp -f "$f" "$OUT/play.server_${f##*/.inference_server_}"
        done
    fi
    rc=$(sed -n 's/.* rc=\([0-9]*\) .*/\1/p' "$OUT/timing_play.txt" | tail -n 1)
    [ -z "$rc" ] || rc_flag=(--rc "$rc")
    box_bounded "fit play" 10 play.log python tools/elo_collect.py "$dir" --no-catalog --save-json "$OUT/play.fit.json" \
        || echo "PLAY_FIT_FAILED rc=$BOX_RC (play.log)" | tee -a "$OUT/arms_failed.txt"
    box_bounded "read play" 5 play.log python tools/fp8_inference_readout.py play "$OUT" --games-dir "$dir" \
        --fit "$OUT/play.fit.json" --games "$PLAY_GAMES" --fp8-label "$PLAY_LABEL_A" --bf16-label "$PLAY_LABEL_B" \
        ${rc_flag[@]+"${rc_flag[@]}"} || echo "PLAY_READ_FAILED rc=$BOX_RC (play.log)" | tee -a "$OUT/arms_failed.txt"
}
summarize
for spec in "${ARM_SPECS[@]}"; do
    read -r name flags <<< "$spec"
    # shellcheck disable=SC2086 # flags is a list of flags
    arm "$name" $flags || true
done
summarize
play_check
box_upload
SUMMARY=$(tail -n 1 "$OUT/summary.log" | cut -c1-500)
PLAY=$(timeout 1m python -c "import json, sys; r = json.load(open(sys.argv[1])); \
print(r['verdict'], r.get('elo_fp8'), '+-', r.get('se'), 'Elo,', r['decisive'], 'decisive')" "$OUT/play.json" 2>/dev/null)
FAILED_ARMS=$(awk '$1 == "ARM_FAILED" {print $2}' "$OUT/arms_failed.txt" 2>/dev/null | tr -d ':' | sort -u | tr '\n' ' ')
# ---- the arms (end)
REASON="$SUMMARY; play: ${PLAY:-not read}; fp8 tests: ${FP8_TESTS:-none}; cuda tests: ${CUDA_TESTS:-none}; failed: ${FAILED_ARMS:-none}"
# A play check without a verdict leaves the question open: the run failed.
case ${PLAY:-NOT_READ} in
    PASS*|KILL*) box_finish "FP8_INFERENCE_DONE $REASON" ;;
    *) box_finish "FP8_INFERENCE_INCOMPLETE $REASON" 1 ;;
esac
