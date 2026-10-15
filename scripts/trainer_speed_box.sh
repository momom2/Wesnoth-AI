#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# The sequence trainer's speed on one box, rounds 3 and 4 (docs/trainer_speed_20261013.md,
# "Round 3", "Round 4"): the fastest configuration of main (the packed trunk,
# kept activations, the columnar records, the window batched) against the
# trunk's variants (wesnoth_ai/trunk_modes.py), fp8 products among them, and a
# sweep of the step's streams, timed under round 2's training arguments
# (scripts/memory_own_trajectories_box.sh, ROUND=2: from own1, its own
# game-sides beside the corpus's) by tools/bench_sequence_step.py.
#
# One code stage, STAGE (tools/stage_code.py --script scripts/trainer_speed_box.sh).
# Its default path is main's to the bit (the CPU traces of the round-3 doc),
# so main's fastest configuration runs from this stage as the reference arm.
#
# The steps:
#   the tests of this path on this wheel: the CPU-path tests gate the run; the
#     CUDA tests (tests/test_trainer_speed_cuda.py, tests/test_trunk_modes_cuda.py)
#     are recorded in cuda_tests.log, with what they print (whether the fused
#     ReLU's GEMM is the plain linear's to the bit, the fp8 trunk's errors and
#     its product against the CPU emulation), and named in the finish line;
#   the trainer's inputs (scripts/trainer_inputs.sh: own1, the corpus at
#     version 5 and its sequences, round 2's own sequences from HF, the columns);
#   the activation budget's calibration (memory.json; CALIBRATE=0: none):
#     CALIBRATE_FLAGS (default B, the accepted configuration, at
#     `--activation-budget-gb auto`) on the pass (tools/bench_sequence_step.py
#     memory): every window's tokens scanned, 5 warm-up and CALIBRATE_WINDOWS
#     timed windows from the pass's start (positions a second, peaks), then
#     CALIBRATE_WORST of the pass's largest windows run where the pass meets
#     them; memory.json gives the peaks, the planner's figures, whether every
#     window fit, and the flags a real run should use;
#   the gradient check (gradcheck_NAME.json): each candidate against the
#     reference flags, FASTEST, in a process of its own, from the same weights
#     at rate 0 over 3 windows, a crash barrier per candidate: a candidate that
#     fails it, or whose check does not finish, is left out of the timing and
#     the others go on;
#   the timing (interleave.jsonl, speed.json; TIMING=0: none): every arm a trainer of its own in
#     one process, the records read first, the arms taking turns window by
#     window (tools/bench_interleave.py; an arm that fails to build or in a
#     window is recorded and the others go on): ref (FASTEST), each passing candidate,
#     and ref2 (FASTEST again, the A/A pair: the session is valid when ref2's
#     median window-time ratio to ref is within VALID_SPREAD of 1); WARMUP
#     windows each (the first holds any compile), ROUNDS rounds, PROFILE_WINDOWS
#     profiled per arm;
#   the streams sweep (sweep.jsonl, sweep.json; SWEEP_STREAMS empty: none): the
#     step's streams at SWEEP_FLAGS (default FASTEST; set it to the fastest arm
#     that passed), each count with half its streams on own game-sides, every
#     arm a trainer of its own in one process as in the timing (an arm that
#     runs out of memory is recorded and left out): positions a second, the
#     window's peak device memory (every arm's weights and optimizer state
#     resident, about 0.3 GB each) and the time steps kept under the activation
#     budget. Timing only: the learning rate a larger step needs is the
#     acceptance's question (scripts/trainer_acceptance_box.sh);
#   the records to HF, the instance stopped. A timing that leaves no summary is
#     named in the finish line; the sweep runs all the same.
#
# The candidates (CANDIDATES, a list of the names; default relu and compiled),
# on FASTEST: relu (--fused-relu) and compiled (--compiled-trunk, inductor);
# bf16 (--trunk-bf16), bf16_relu, bf16_compiled, fp8 (--fp8-trunk) and
# compiled_fp8 by name (bf16 failed its CUDA test, fp8 its gradient check).
#
# Expected wall on a 4090 (the round-4 walls.txt): bring-up and tests 15
# minutes, the inputs 10 with the own sequences from HF, the calibration 10
# (the scan of the 2,000,000-position pass's windows, about 5, the compile, 45
# windows at about 0.35 s, 6 of the largest), the gradient check 2 (two
# processes of two trainers each), the timing 5 (four trainers, 5 + 40 + 3
# windows each), the sweep 12 (three trainers, 5 + 20 + 1 windows of 0.4, 0.8
# and 1.6 s or more): about 55 minutes, $0.40 at $0.42 an hour; the
# calibration alone (TIMING=0 CANDIDATES= SWEEP_STREAMS=) about 35 minutes,
# $0.25. 40 minutes more if the own sequences must be built again. BOX_MAX_H 2
# caps it at about $0.85 plus the final round.
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md). No
# periodic upload rounds while the arms run (they would share the machine
# with the timing): the records go up after each stage and at the finish.
# Re-entry skips the stages that are done. Never `set -x`: the HF token and
# the instance key are in the environment.
# box-needs: disk_gb=150 ram_gb=64 gpu_ram_gb=24 cores=32 gpu=4090
set -uo pipefail
WORKDIR=/workspace
STAGE="${STAGE:-}"
HF_DIR="${HF_DIR:-tier-b/trainer_speed_4_20261016}"
export HF_DIR
BOX_MAX_H="${BOX_MAX_H:-2}"
RUN_SEED="${RUN_SEED:-20261012}"                # round 2's
LR="${LR:-2.8e-5}"
PASS_POSITIONS="${PASS_POSITIONS:-2000000}"
DECAY_FROM="${DECAY_FROM:-0.5}"
PROBE_EVERY="${PROBE_EVERY:-1200000}"
WARMUP="${WARMUP:-5}"
ROUNDS="${ROUNDS:-40}"
PROFILE_WINDOWS="${PROFILE_WINDOWS:-3}"
BUDGET_GB="${BUDGET_GB:-19.18}"                 # the first round's packed_budget arm
BYTES_PER_TOKEN="${BYTES_PER_TOKEN:-102139}"
VALID_SPREAD="${VALID_SPREAD:-0.03}"
CANDIDATES="${CANDIDATES-relu compiled}"
TIMING="${TIMING:-1}"
CALIBRATE="${CALIBRATE:-1}"
CALIBRATE_FLAGS="${CALIBRATE_FLAGS:---packed-trunk --columns --window-batched --compiled-trunk --activation-budget-gb auto}"
CALIBRATE_WINDOWS="${CALIBRATE_WINDOWS:-40}"
CALIBRATE_WORST="${CALIBRATE_WORST:-6}"
CALIBRATE_CUT_MIN="${CALIBRATE_CUT_MIN:-40}"
SWEEP_STREAMS="${SWEEP_STREAMS:-32 64 128}"
SWEEP_FLAGS="${SWEEP_FLAGS:-}"                  # default FASTEST
SWEEP_WARMUP="${SWEEP_WARMUP:-5}"
SWEEP_ROUNDS="${SWEEP_ROUNDS:-20}"
WORKERS="${WORKERS:-}"
TESTS_CUT_MIN="${TESTS_CUT_MIN:-40}"
GRADCHECK_CUT_MIN="${GRADCHECK_CUT_MIN:-25}"
TIMING_CUT_MIN="${TIMING_CUT_MIN:-45}"
SWEEP_CUT_MIN="${SWEEP_CUT_MIN:-30}"
OUT=$WORKDIR/trainer_speed
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }

notes() {                        # what is done, for the finish reason
    local done_=()
    local name
    for name in $CANDIDATES; do
        [ ! -f "$OUT/gradcheck_$name.json" ] || done_+=("gradcheck_$name")
    done
    [ ! -f "$OUT/memory.json" ] || done_+=(calibration)
    [ ! -f "$OUT/speed.json" ] || done_+=(timing)
    [ ! -f "$OUT/sweep.json" ] || done_+=(sweep)
    echo "done: ${done_[*]:-nothing}"
}
# shellcheck disable=SC2317 # called by the library, in the reason of an unexpected exit
box_notes() { notes; }

box_init
[ -n "$WORKERS" ] || WORKERS=$(box_workers)
[ -n "$STAGE" ] || box_finish "NO_STAGE: build the code stage (tools/stage_code.py) and pass STAGE" 1
box_bind_run_stage
# shellcheck disable=SC2046 # one name per candidate
box_restore memory.json speed.json interleave.jsonl sweep.json sweep.jsonl cuda_tests.log tests.log \
    $(for name in $CANDIDATES; do echo "gradcheck_$name.json"; done) || box_finish "RESTORE_FAILED (restore.log)" 1
box_pip huggingface_hub psutil pytest scipy requests || echo "pip install failed (pip.log)"

# ---- the code and the Rust wheel
box_stage_code || box_finish "CODE_STAGING_FAILED (staging.log)" 1
cd "$BOX_REPO" || box_finish "CODE_STAGING_FAILED (no $BOX_REPO)" 1
# shellcheck source=trainer_inputs.sh
. "$BOX_REPO/scripts/trainer_inputs.sh" || box_finish "CODE_STAGING_FAILED (no scripts/trainer_inputs.sh)" 1
box_build_wheel || box_finish "BUILD_FAILED (build.log)" 1
box_facts > "$OUT/box.txt.tmp" 2>&1
mv -f "$OUT/box.txt.tmp" "$OUT/box.txt"
timeout 2m python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || box_finish "NO_CUDA (box.txt)" 1
box_upload

# ---- the tests: the CPU paths gate the run, the CUDA tests are recorded
if ! box_marked_this_stage "$BOX_STATE/TESTED"; then
    : > "$OUT/tests.log"
    box_bounded tests "$TESTS_CUT_MIN" tests.log python -m pytest tests/test_sequence_train.py \
        tests/test_sequence_speed.py tests/test_sequence_columns.py tests/test_sequence_launches.py \
        tests/test_own_loss.py tests/test_own_trajectories.py tests/test_memory_model.py \
        tests/test_packed_trunk.py tests/test_packed_trunk_cuda.py tests/test_packed_embed.py \
        tests/test_preencode_sequences.py tests/test_memory_learner.py tests/test_trunk_modes.py \
        tests/test_trunk_variants.py tests/test_bench_interleave.py tests/test_trainer_acceptance_readout.py \
        tests/test_fp8_linear.py tests/test_rules_version.py tests/test_train_every.py tests/test_memory_plan.py \
        tests/test_bench_gradcheck.py \
        -q -p no:cacheprovider -m ""
    tail -n 3 "$OUT/tests.log"
    [ "$BOX_RC" -eq 0 ] || box_finish "TESTS_FAILED rc=$BOX_RC (tests.log)" 1
    : > "$OUT/cuda_tests.log"
    box_bounded cuda-tests 20 cuda_tests.log python -m pytest tests/test_trainer_speed_cuda.py \
        tests/test_trunk_modes_cuda.py -q -s -p no:cacheprovider -m ""
    tail -n 3 "$OUT/cuda_tests.log"
    box_mark "$BOX_STATE/TESTED"
fi
CUDA_TESTS=$(grep -Eo "[0-9]+ (passed|failed|error|errors|skipped)" "$OUT/cuda_tests.log" | tr '\n' ' ')
box_upload

trainer_inputs

# ---- the arms (begin): tests/test_trainer_speed_box.py runs this block with the library stubbed
TRAIN_ARGS=(--sequences "$SEQ" --dataset "$CORPUS" --own-sequences "$OWN" --seed "$RUN_SEED" --device cuda
            --lr "$LR" --warmup-steps 0 --decay-from "$DECAY_FROM" --pass-positions "$PASS_POSITIONS"
            --probe-every "$PROBE_EVERY" --barrier-positions 1000000 --probe-ks "0,64" --probe-at-start
            --init-from "$INIT_PT" --init-other-records)
FASTEST="--packed-trunk --activation-budget-gb $BUDGET_GB --activation-bytes-per-token $BYTES_PER_TOKEN --columns --window-batched"
read -r -a CANDIDATE_NAMES <<< "$CANDIDATES"
candidate_flags() {              # candidate_flags NAME: the flags NAME adds to FASTEST
    case $1 in
        bf16) echo "--trunk-bf16" ;;
        relu) echo "--fused-relu" ;;
        compiled) echo "--compiled-trunk" ;;
        bf16_relu) echo "--trunk-bf16 --fused-relu" ;;
        bf16_compiled) echo "--trunk-bf16 --compiled-trunk" ;;
        fp8) echo "--fp8-trunk" ;;
        compiled_fp8) echo "--compiled-trunk --fp8-trunk" ;;
        *) box_finish "UNKNOWN_CANDIDATE $1 (CANDIDATES)" 1 ;;
    esac
}
for name in "${CANDIDATE_NAMES[@]}"; do
    candidate_flags "$name" > /dev/null
done

# The calibration: CALIBRATE_FLAGS on the pass's first and largest windows.
if [ "$CALIBRATE" = 1 ] && [ ! -f "$OUT/memory.json" ]; then
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before the calibration: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    rm -f "$OUT/memory.json.pending"
    box_bounded memory "$CALIBRATE_CUT_MIN" memory.log python tools/bench_sequence_step.py memory \
        --out "$OUT/memory.json.pending" "--flags=$CALIBRATE_FLAGS" --windows "$CALIBRATE_WINDOWS" \
        --worst "$CALIBRATE_WORST" -- "${TRAIN_ARGS[@]}" --out "$OUT/memory_arm.pt"
    if [ "$BOX_RC" -eq 0 ]; then
        mv -f "$OUT/memory.json.pending" "$OUT/memory.json"
    else
        echo "CALIBRATION_FAILED rc=$BOX_RC $BOX_WHY (memory.log)" | tee -a "$OUT/arms_failed.txt"
    fi
    box_upload
fi
gradcheck_failed() {             # gradcheck_failed NAME: NAME's check did not finish, so NAME does not pass
    echo "GRADCHECK_FAILED $1 rc=$BOX_RC $BOX_WHY (gradcheck_$1.log)" | tee -a "$OUT/arms_failed.txt"
    printf '{"pass": false, "rc": %d, "why": "%s", "log": "gradcheck_%s.log", "candidates": []}\n' "$BOX_RC" \
        "$BOX_WHY" "$1" > "$OUT/gradcheck_$1.json"
}
passed() {                       # passed NAME: NAME's own gradient check passed its flags
    timeout 1m python - "$OUT/gradcheck_$1.json" "$FASTEST $(candidate_flags "$1")" <<'EOF'
import json, sys
check = json.load(open(sys.argv[1], encoding="utf-8"))
want = " ".join(sys.argv[2].split())
sys.exit(0 if any(" ".join(c["candidate"].split()) == want and c["pass"] for c in check.get("candidates", [])) else 1)
EOF
}

# The gradient check: each candidate against FASTEST in a process of its own, a crash barrier per candidate.
for name in "${CANDIDATE_NAMES[@]}"; do
    [ ! -f "$OUT/gradcheck_$name.json" ] || continue
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before the gradient check of $name: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    rm -f "$OUT/gradcheck_$name.json.pending"
    box_bounded "gradcheck $name" "$GRADCHECK_CUT_MIN" "gradcheck_$name.log" python tools/bench_sequence_step.py \
        gradcheck --out "$OUT/gradcheck_$name.json.pending" "--reference=$FASTEST" \
        "--candidate=$FASTEST $(candidate_flags "$name")" -- "${TRAIN_ARGS[@]}" --out "$OUT/gradcheck_arm_$name.pt"
    case $BOX_RC in
        0|8) mv -f "$OUT/gradcheck_$name.json.pending" "$OUT/gradcheck_$name.json" || gradcheck_failed "$name" ;;
        *) gradcheck_failed "$name" ;;
    esac
    box_upload
done

# The timing: ref, every candidate that passed, ref2, interleaved in one process.
if [ "$TIMING" = 1 ] && [ ! -f "$OUT/speed.json" ]; then
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before the timing: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    arms=("--arm=ref=$FASTEST")
    for name in "${CANDIDATE_NAMES[@]}"; do
        if passed "$name"; then
            arms+=("--arm=$name=$FASTEST $(candidate_flags "$name")")
        else
            echo "LEFT_OUT $name: its gradient check did not pass (gradcheck_$name.json)" | tee -a "$OUT/arms_failed.txt"
        fi
    done
    arms+=("--arm=ref2=$FASTEST")
    rm -f "$OUT/interleave.jsonl"
    box_bounded timing "$TIMING_CUT_MIN" timing.log python tools/bench_sequence_step.py interleave \
        --out "$OUT/interleave.jsonl" "${arms[@]}" --reference ref --warmup "$WARMUP" --rounds "$ROUNDS" \
        --profile-windows "$PROFILE_WINDOWS" --spread "$VALID_SPREAD" --summary "$OUT/speed.json" \
        -- "${TRAIN_ARGS[@]}"
    if [ "$BOX_RC" -ne 0 ] && [ -f "$OUT/interleave.jsonl" ]; then
        # The windows timed before the failure stand: their summary from the rows.
        echo "TIMING_FAILED rc=$BOX_RC $BOX_WHY (timing.log)" | tee -a "$OUT/arms_failed.txt"
        box_bounded timing-summary 5 timing.log python tools/bench_sequence_step.py interleave-summary \
            "$OUT/interleave.jsonl" --reference ref --spread "$VALID_SPREAD" --out "$OUT/speed.json"
    fi
    if [ "$BOX_RC" -ne 0 ] && [ ! -f "$OUT/speed.json" ]; then
        echo "TIMING_FAILED rc=$BOX_RC $BOX_WHY: no summary (timing.log)" | tee -a "$OUT/arms_failed.txt"
    fi
    box_upload
fi

# The streams sweep: SWEEP_FLAGS at each count of SWEEP_STREAMS, half of them own.
read -r -a SWEEP_COUNTS <<< "$SWEEP_STREAMS"
if [ "${#SWEEP_COUNTS[@]}" -gt 0 ] && [ ! -f "$OUT/sweep.json" ]; then
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before the sweep: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    arms=()
    for n in "${SWEEP_COUNTS[@]}"; do
        arms+=("--arm=s$n=${SWEEP_FLAGS:-$FASTEST} --streams $n --own-streams $(( n / 2 ))")
    done
    rm -f "$OUT/sweep.jsonl"
    box_bounded sweep "$SWEEP_CUT_MIN" sweep.log python tools/bench_sequence_step.py interleave \
        --out "$OUT/sweep.jsonl" "${arms[@]}" --reference "s${SWEEP_COUNTS[0]}" --warmup "$SWEEP_WARMUP" \
        --rounds "$SWEEP_ROUNDS" --profile-windows 1 --not-production --summary "$OUT/sweep.json" \
        -- "${TRAIN_ARGS[@]}"
    if [ "$BOX_RC" -ne 0 ] && [ -f "$OUT/sweep.jsonl" ]; then
        echo "SWEEP_FAILED rc=$BOX_RC $BOX_WHY (sweep.log)" | tee -a "$OUT/arms_failed.txt"
        box_bounded sweep-summary 5 sweep.log python tools/bench_sequence_step.py interleave-summary \
            "$OUT/sweep.jsonl" --reference "s${SWEEP_COUNTS[0]}" --out "$OUT/sweep.json"
    elif [ "$BOX_RC" -ne 0 ]; then
        echo "SWEEP_FAILED rc=$BOX_RC $BOX_WHY: no rows (sweep.log)" | tee -a "$OUT/arms_failed.txt"
    fi
    box_upload
fi
SUMMARY=$(timeout 1m python - "$OUT/speed.json" "$OUT/sweep.json" "$OUT/memory.json" <<'EOF'
import json, os, sys
out = {"timing": None}
if os.path.exists(sys.argv[3]):
    m = json.load(open(sys.argv[3], encoding="utf-8"))
    out["calibration"] = {"fits": m["fits"], "positions_per_second": m["timed"]["positions_per_second"],
                          "peak_reserved_gb": m["peak_reserved_gb_max"], "headroom_gb": m["headroom_gb"],
                          "largest_window_run": m["largest_window_run"], "flags": m["flags"]}
if os.path.exists(sys.argv[1]):
    s = json.load(open(sys.argv[1], encoding="utf-8"))
    rates = {a: v["positions_per_second"] for a, v in s["arms"].items()}
    speedups = {a: (v.get("against_reference") or {}).get("speedup_median") for a, v in s["arms"].items()
                if a != "ref"}
    out.update(timing={"valid": s["valid"], "rates": rates, "speedup_median": speedups})
if os.path.exists(sys.argv[2]):
    w = json.load(open(sys.argv[2], encoding="utf-8"))
    out["sweep"] = {a: [v["positions_per_second"], v["peak_gb_max"]] for a, v in w["arms"].items()}
print(json.dumps(out))
EOF
)
FAILED_STEPS=$(cut -d' ' -f1-2 "$OUT/arms_failed.txt" 2>/dev/null | tr '\n' ' ')
[ -f "$OUT/speed.json" ] || [ -f "$OUT/sweep.json" ] || [ -f "$OUT/memory.json" ] \
    || box_finish "NOTHING_MEASURED: no memory.json, speed.json or sweep.json; left out: ${FAILED_STEPS:-none} $(notes)" 1
box_finish "TRAINER_SPEED_DONE ${SUMMARY:0:1200}; cuda tests: ${CUDA_TESTS:-none}; left out: ${FAILED_STEPS:-none}"
# ---- the arms (end)
