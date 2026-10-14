#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# The sequence trainer's speed on one box, round 3 (docs/trainer_speed_20261013.md,
# "Round 3"): the fastest configuration of main (the packed trunk, kept
# activations, the columnar records, the window batched) against the trunk's
# variants (wesnoth_ai/trunk_modes.py), timed under round 2's training
# arguments (scripts/memory_own_trajectories_box.sh, ROUND=2: from own1, its
# own game-sides beside the corpus's) by tools/bench_sequence_step.py.
#
# One code stage, STAGE (tools/stage_code.py --script scripts/trainer_speed_box.sh).
# Its default path is main's to the bit (the CPU traces of the round-3 doc),
# so main's fastest configuration runs from this stage as the reference arm.
#
# The steps:
#   the tests of this path on this wheel: the CPU-path tests gate the run; the
#     CUDA tests (tests/test_trainer_speed_cuda.py, tests/test_trunk_modes_cuda.py)
#     are recorded in cuda_tests.log, with what they print (whether the fused
#     ReLU's GEMM is the plain linear's to the bit), and named in the finish line;
#   the trainer's inputs (scripts/trainer_inputs.sh: own1, the corpus at
#     version 5 and its sequences, round 2's own sequences from HF, the columns);
#   the gradient check (gradcheck.json): every candidate against the reference
#     flags, FASTEST, from the same weights at rate 0 over 3 windows, a crash
#     barrier per candidate: a candidate that fails it, or a check that does not
#     finish, leaves that arm, or every candidate arm, out of the timing;
#   the timing (interleave.jsonl, speed.json): every arm a trainer of its own in
#     one process, the records read first, the arms taking turns window by
#     window (tools/bench_interleave.py): ref (FASTEST), each passing candidate,
#     and ref2 (FASTEST again, the A/A pair: the session is valid when ref2's
#     median window-time ratio to ref is within VALID_SPREAD of 1); WARMUP
#     windows each (the first holds any compile), ROUNDS rounds, PROFILE_WINDOWS
#     profiled per arm;
#   the records to HF, the instance stopped.
#
# The candidates, on FASTEST: bf16 (--trunk-bf16), relu (--fused-relu),
# compiled (--compiled-trunk, inductor), bf16_relu and bf16_compiled.
#
# Expected wall on a 4090 (round 2's walls.txt): bring-up and tests 15
# minutes, the inputs 10 with the own sequences from HF, the gradient check 4
# (six trainers, two compiles), the timing 10 (seven trainers built and their
# records read, 5 + 40 + 3 windows each at about 0.4 s, the compiles, the
# profiles): about 40 minutes, $0.30 at $0.42 an hour; 80 minutes and $0.60
# if the own sequences must be built again. BOX_MAX_H 1.5 caps it at about
# $0.65 plus the final round.
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
HF_DIR="${HF_DIR:-tier-b/trainer_speed_3_20261015}"
export HF_DIR
BOX_MAX_H="${BOX_MAX_H:-1.5}"
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
WORKERS="${WORKERS:-}"
TESTS_CUT_MIN="${TESTS_CUT_MIN:-40}"
GRADCHECK_CUT_MIN="${GRADCHECK_CUT_MIN:-25}"
TIMING_CUT_MIN="${TIMING_CUT_MIN:-45}"
OUT=$WORKDIR/trainer_speed
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }

notes() {                        # what is done, for the finish reason
    local done_=()
    [ ! -f "$OUT/gradcheck.json" ] || done_+=(gradcheck)
    [ ! -f "$OUT/speed.json" ] || done_+=(timing)
    echo "done: ${done_[*]:-nothing}"
}
# shellcheck disable=SC2317 # called by the library, in the reason of an unexpected exit
box_notes() { notes; }

box_init
[ -n "$WORKERS" ] || WORKERS=$(box_workers)
[ -n "$STAGE" ] || box_finish "NO_STAGE: build the code stage (tools/stage_code.py) and pass STAGE" 1
box_bind_run_stage
box_restore speed.json gradcheck.json interleave.jsonl cuda_tests.log tests.log \
    || box_finish "RESTORE_FAILED (restore.log)" 1
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
CANDIDATES=(bf16 relu compiled bf16_relu bf16_compiled)
candidate_flags() {              # candidate_flags NAME: the flags NAME adds to FASTEST
    case $1 in
        bf16) echo "--trunk-bf16" ;;
        relu) echo "--fused-relu" ;;
        compiled) echo "--compiled-trunk" ;;
        bf16_relu) echo "--trunk-bf16 --fused-relu" ;;
        bf16_compiled) echo "--trunk-bf16 --compiled-trunk" ;;
    esac
}
gradcheck_failed() {             # gradcheck_failed: a check that did not finish passes no candidate
    echo "GRADCHECK_FAILED rc=$BOX_RC $BOX_WHY (gradcheck.log)" | tee -a "$OUT/arms_failed.txt"
    printf '{"pass": false, "rc": %d, "why": "%s", "log": "gradcheck.log", "candidates": []}\n' "$BOX_RC" "$BOX_WHY" \
        > "$OUT/gradcheck.json"
}
passed() {                       # passed NAME: the gradient check passed NAME's flags
    timeout 1m python - "$OUT/gradcheck.json" "$FASTEST $(candidate_flags "$1")" <<'EOF'
import json, sys
check = json.load(open(sys.argv[1], encoding="utf-8"))
want = " ".join(sys.argv[2].split())
sys.exit(0 if any(" ".join(c["candidate"].split()) == want and c["pass"] for c in check.get("candidates", [])) else 1)
EOF
}

# The gradient check: every candidate against FASTEST, a crash barrier per candidate.
if [ ! -f "$OUT/gradcheck.json" ]; then
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before the gradient check: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    rm -f "$OUT/gradcheck.json.pending"
    check_args=()
    for name in "${CANDIDATES[@]}"; do
        check_args+=("--candidate=$FASTEST $(candidate_flags "$name")")
    done
    box_bounded gradcheck "$GRADCHECK_CUT_MIN" gradcheck.log python tools/bench_sequence_step.py gradcheck \
        --out "$OUT/gradcheck.json.pending" "--reference=$FASTEST" "${check_args[@]}" -- "${TRAIN_ARGS[@]}" \
        --out "$OUT/gradcheck_arm.pt"
    case $BOX_RC in
        0|8) mv -f "$OUT/gradcheck.json.pending" "$OUT/gradcheck.json" || gradcheck_failed ;;
        *) gradcheck_failed ;;
    esac
    box_upload
fi

# The timing: ref, every candidate that passed, ref2, interleaved in one process.
if [ ! -f "$OUT/speed.json" ]; then
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before the timing: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    arms=("--arm=ref=$FASTEST")
    for name in "${CANDIDATES[@]}"; do
        if passed "$name"; then
            arms+=("--arm=$name=$FASTEST $(candidate_flags "$name")")
        else
            echo "LEFT_OUT $name: its gradient check did not pass (gradcheck.json)" | tee -a "$OUT/arms_failed.txt"
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
    box_upload
fi
[ -f "$OUT/speed.json" ] || box_finish "TIMING_FAILED: no speed.json (timing.log) $(notes)" 1
SUMMARY=$(timeout 1m python - "$OUT/speed.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
rates = {a: v["positions_per_second"] for a, v in s["arms"].items()}
speedups = {a: (v.get("against_reference") or {}).get("speedup_median") for a, v in s["arms"].items() if a != "ref"}
print(json.dumps({"valid": s["valid"], "rates": rates, "speedup_median": speedups}))
EOF
)
FAILED_STEPS=$(cut -d' ' -f1-2 "$OUT/arms_failed.txt" 2>/dev/null | tr '\n' ' ')
box_finish "TRAINER_SPEED_DONE ${SUMMARY:0:500}; cuda tests: ${CUDA_TESTS:-none}; left out: ${FAILED_STEPS:-none}"
# ---- the arms (end)
