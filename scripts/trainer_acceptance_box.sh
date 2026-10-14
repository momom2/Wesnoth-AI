#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# The training acceptance of the sequence trainer's arithmetic-changing
# speedups (docs/trainer_speed_20261013.md, "Round 3: the training
# acceptance"): three short passes from own1 under round 2's preset, the same
# seed, each PASS_POSITIONS positions and its final holdout probe at 0 and 64
# slots:
#   A   the padded trunk, today's arithmetic (A_FLAGS: the columnar records and
#       the padded trunk's activation budget, both the default's arithmetic to
#       the bit, to shorten the pass);
#   B   the fastest configuration that changes the arithmetic (B_FLAGS: the
#       packed trunk, the window batched and the trunk variants; set it from
#       the round-3 bench's speed.json, the fastest arm whose gradient check
#       passed);
#   A2  A again: the run-to-run nondeterminism of today's GPU training.
# The order is A, B, A2. Each pass writes its probe's per-game-side sums
# (<out>.probe_sides.jsonl), and tools/trainer_acceptance_readout.py pairs B
# against A and A2 against A over the same holdout game-sides (acceptance.json,
# acceptance.txt): B passes when, for ce_all and belief at 0 and 64 slots,
# |B - A| is no larger than |A2 - A| plus two paired standard errors of B - A.
# The criterion is the lead's to pre-register; the readout also gives the
# one-sided reading and every difference with its error.
#
# Before the passes, a crash barrier: B's gradients against A's from the same
# weights at rate 0 over 3 windows (tools/bench_sequence_step.py gradcheck,
# gradcheck.json); anything but a pass ends the run before money goes into
# the passes.
#
# One code stage, STAGE (tools/stage_code.py --script scripts/trainer_acceptance_box.sh).
# The trainer's inputs come from scripts/box/trainer_inputs.sh (own1, the
# corpus at version 5 and its sequences, round 2's own sequences from HF's
# cache, the columns); a box that ran scripts/trainer_speed_box.sh from the
# same stage has them already.
#
# Expected wall on a 4090: bring-up and tests 15 minutes, the inputs 10, the
# barrier 3; A and A2 about 36 minutes each (500,000 positions at about 280
# a second, round 1's padded arm with its budget, then a probe of about 5.4
# minutes, round 2's); B about 13 (about 1,300 positions a second or more,
# its compile, the probe): about 1 hour 55 minutes, $0.80 at $0.42 an hour.
# A pass is cut at PASS_CUT_MIN (70: 1.5 times A's estimate) or after
# PASS_STALL_MIN without a log line. BOX_MAX_H 3 caps it at about $1.30 plus
# the final round.
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md). A
# pass interrupted by a re-entry resumes from its last checkpoint (every
# 250,000 positions). The records go to HF after each pass, the passes'
# checkpoints with them (their .stable.pt stay on the machine). Never
# `set -x`: the HF token and the instance key are in the environment.
# box-needs: disk_gb=150 ram_gb=64 gpu_ram_gb=24 cores=32 gpu=4090
set -uo pipefail
WORKDIR=/workspace
STAGE="${STAGE:-}"
HF_DIR="${HF_DIR:-tier-b/trainer_acceptance_20261015}"
export HF_DIR
BOX_MAX_H="${BOX_MAX_H:-3}"
RUN_SEED="${RUN_SEED:-20261012}"                # round 2's
LR="${LR:-2.8e-5}"
PASS_POSITIONS="${PASS_POSITIONS:-500000}"
DECAY_FROM="${DECAY_FROM:-0.5}"
A_FLAGS="${A_FLAGS:---columns --activation-budget-gb 15.97 --activation-bytes-per-token 150329}"
B_FLAGS="${B_FLAGS:---packed-trunk --columns --window-batched --activation-budget-gb 19.18 --activation-bytes-per-token 102139 --trunk-bf16 --compiled-trunk}"
WORKERS="${WORKERS:-}"
TESTS_CUT_MIN="${TESTS_CUT_MIN:-40}"
GRADCHECK_CUT_MIN="${GRADCHECK_CUT_MIN:-20}"
PASS_CUT_MIN="${PASS_CUT_MIN:-70}"
PASS_STALL_MIN="${PASS_STALL_MIN:-15}"
OUT=$WORKDIR/trainer_acceptance
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }
# shellcheck source=box/trainer_inputs.sh
. "${BOX_LIB:-$WORKDIR/box}/trainer_inputs.sh" || { echo "no trainer_inputs.sh beside the box library"; exit 1; }
PASSES="A B A2"

notes() {                        # what is done, for the finish reason
    local name done_=()
    [ ! -f "$OUT/gradcheck.json" ] || done_+=(gradcheck)
    for name in $PASSES; do
        ! box_marked_this_stage "$OUT/pass_$name.done" || done_+=("$name")
    done
    echo "done: ${done_[*]:-nothing}"
}
# shellcheck disable=SC2317 # called by the library, in the reason of an unexpected exit
box_notes() { notes; }

box_init
[ -n "$WORKERS" ] || WORKERS=$(box_workers)
[ -n "$STAGE" ] || box_finish "NO_STAGE: build the code stage (tools/stage_code.py) and pass STAGE" 1
box_bind_run_stage
# shellcheck disable=SC2046 # one name per pass and file
box_restore gradcheck.json cuda_tests.log tests.log $(for p in $PASSES; do echo "pass_$p.probe_sides.jsonl"; done) \
    || box_finish "RESTORE_FAILED (restore.log)" 1
box_pip huggingface_hub psutil pytest scipy requests || echo "pip install failed (pip.log)"
for name in $PASSES; do
    box_upload_skip "pass_$name.stable.pt"
done

# ---- the code and the Rust wheel
box_stage_code || box_finish "CODE_STAGING_FAILED (staging.log)" 1
cd "$BOX_REPO" || box_finish "CODE_STAGING_FAILED (no $BOX_REPO)" 1
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
        tests/test_own_loss.py tests/test_packed_trunk.py tests/test_trunk_modes.py tests/test_trunk_variants.py \
        tests/test_trainer_acceptance_readout.py -q -p no:cacheprovider -m ""
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

# ---- the passes (begin): tests/test_trainer_acceptance_box.py runs this block with the library stubbed
COMMON=(--sequences "$SEQ" --dataset "$CORPUS" --own-sequences "$OWN" --seed "$RUN_SEED" --device cuda
        --lr "$LR" --warmup-steps 0 --decay-from "$DECAY_FROM" --pass-positions "$PASS_POSITIONS"
        --probe-every 1200000 --barrier-positions 1000000 --probe-ks "0,64")
INIT=(--init-from "$INIT_PT" --init-other-records)
pass_flags() {                   # pass_flags NAME: the flags of pass NAME
    case $1 in
        A|A2) echo "$A_FLAGS" ;;
        B) echo "$B_FLAGS" ;;
    esac
}
final_probe() {                  # final_probe NAME: the pass's final probe is written
    grep -q '"final": true' "$OUT/pass_$1.probe_sides.jsonl" 2>/dev/null
}
train_pass() {                   # train_pass NAME: the pass, once; resumed from its checkpoint after a re-entry
    local name=$1
    local -a start=("${INIT[@]}")
    ! box_marked_this_stage "$OUT/pass_$name.done" || return 0
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before pass $name: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    [ ! -f "$OUT/pass_$name.pt" ] || start=(--resume)
    # shellcheck disable=SC2046 # the pass's flags are a list
    box_bounded --stall "$OUT/pass_$name.log" "$PASS_STALL_MIN" "pass $name" "$PASS_CUT_MIN" "pass_$name.log" \
        python tools/sequence_train.py "${COMMON[@]}" "${start[@]}" --out "$OUT/pass_$name.pt" $(pass_flags "$name")
    if [ "$BOX_RC" -ne 0 ] || ! final_probe "$name"; then
        box_upload
        box_finish "PASS_FAILED $name rc=$BOX_RC $BOX_WHY (pass_$name.log) $(notes)" 1
    fi
    box_mark "$OUT/pass_$name.done"
    box_upload
}

# The barrier: B's gradients against A's, from the same weights at rate 0.
if [ ! -f "$OUT/gradcheck.json" ]; then
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before the gradient check: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    rm -f "$OUT/gradcheck.json.pending"
    box_bounded gradcheck "$GRADCHECK_CUT_MIN" gradcheck.log python tools/bench_sequence_step.py gradcheck \
        --out "$OUT/gradcheck.json.pending" "--reference=$A_FLAGS" "--candidate=$B_FLAGS" -- "${COMMON[@]}" \
        "${INIT[@]}" --out "$OUT/gradcheck_arm.pt"
    case $BOX_RC in
        0) mv -f "$OUT/gradcheck.json.pending" "$OUT/gradcheck.json" ;;
        8) mv -f "$OUT/gradcheck.json.pending" "$OUT/gradcheck.json"
           box_upload
           box_finish "B_GRADCHECK_FAILED: B's gradients are not A's within the check (gradcheck.json)" 1 ;;
        *) box_upload
           box_finish "B_GRADCHECK_UNFINISHED rc=$BOX_RC $BOX_WHY (gradcheck.log)" 1 ;;
    esac
    box_upload
fi

for name in $PASSES; do
    train_pass "$name"
done

box_bounded readout 5 readout.log python tools/trainer_acceptance_readout.py --a "$OUT/pass_A.probe_sides.jsonl" \
    --a2 "$OUT/pass_A2.probe_sides.jsonl" --b "$OUT/pass_B.probe_sides.jsonl" --out "$OUT/acceptance.json" \
    || box_finish "READOUT_FAILED rc=$BOX_RC (readout.log)" 1
cp -f "$OUT/readout.log" "$OUT/acceptance.txt"
VERDICT=$(timeout 1m python - "$OUT" <<'EOF'
import json, pathlib, re, sys
out = pathlib.Path(sys.argv[1])
r = json.loads((out / "acceptance.json").read_text(encoding="utf-8"))
rates = {}
for name in ("A", "B", "A2"):
    m = re.findall(r"SEQUENCE_TRAIN_DONE (\d+) positions, \d+ steps in (\d+) s",
                   (out / f"pass_{name}.log").read_text(encoding="utf-8", errors="replace"))
    rates[name] = round(int(m[-1][0]) / max(1, int(m[-1][1])), 1) if m else None
print(json.dumps({"pass": r["pass"], "not_worse": r["not_worse"], "positions_per_second": rates}))
EOF
)
box_finish "TRAINER_ACCEPTANCE_DONE ${VERDICT:0:400}; cuda tests: ${CUDA_TESTS:-none}"
# ---- the passes (end)
