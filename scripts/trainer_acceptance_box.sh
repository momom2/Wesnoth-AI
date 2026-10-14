#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# The training acceptance of the sequence trainer's speed options that change
# the numbers or the method (docs/trainer_speed_20261013.md, "Round 3: the
# training acceptance", "Round 4"): short passes from own1 under round 2's
# preset, the same seed, each PASS_POSITIONS positions walked and its final
# holdout probe at 0 and 64 slots. The arms (ARMS, a list of the names; A is
# required, every other can be left out):
#   A      the padded trunk, today's arithmetic (A_FLAGS: the columnar records
#          and the padded trunk's activation budget, both the default's
#          arithmetic to the bit, to shorten the pass);
#   A2     A again: the run-to-run nondeterminism of today's GPU training;
#   B      the fastest configuration that changes the arithmetic (B_FLAGS:
#          the packed trunk, the window batched and the trunk variants; set it
#          from the bench's speed.json, the fastest arm whose gradient check
#          passed);
#   B8     B with fp8 products (B8_FLAGS: B_FLAGS without --fused-relu, which
#          fp8 excludes, plus --fp8-trunk);
#   B64    B at BIG_STREAMS streams (half of them own), the learning rate LR
#          times the square root of BIG_STREAMS / 32, the same positions (so
#          half the steps at 64);
#   BHALF  B training one window in SUBSET_EVERY (--train-every): the others
#          run the forward only, carrying the memory, so each game-side trains
#          on the decisions of its trained windows, about half of them, and the
#          pass walks the same positions as B's.
# The order is A, B, A2, B8, B64, BHALF. Each pass writes its probe's
# per-game-side sums (<out>.probe_sides.jsonl), and
# tools/trainer_acceptance_readout.py pairs every arm against A over the same
# holdout game-sides, with A2 - A as the noise (acceptance.json,
# acceptance.txt): an arm passes when, for ce_all and belief at 0 and 64 slots,
# |arm - A| is no larger than |A2 - A| plus two paired standard errors of
# arm - A; it also reads each arm's difference from B (its own factor). The
# criterion is the lead's to pre-register; the readout gives every difference
# with its error.
#
# Before the passes, a crash barrier: B's and B8's gradients against A's from
# the same weights at rate 0 over 3 windows (tools/bench_sequence_step.py
# gradcheck, gradcheck.json). B failing it, or a check that does not finish,
# ends the run before money goes into the passes; B8 failing it leaves B8
# out. B64 and BHALF are B's arithmetic: B's check covers them.
#
# A pass that fails ends the run when it is A's (nothing to compare against);
# any other is recorded in arms_failed.txt and the run goes on, and a re-entry
# leaves it failed (delete its pass_NAME.failed to run it again). A pass
# interrupted by a re-entry resumes from its last checkpoint (every 250,000
# positions).
#
# One code stage, STAGE (tools/stage_code.py --script scripts/trainer_acceptance_box.sh).
# The trainer's inputs come from scripts/trainer_inputs.sh (own1, the
# corpus at version 5 and its sequences, round 2's own sequences from HF's
# cache, the columns); a box that ran scripts/trainer_speed_box.sh from the
# same stage has them already.
#
# Expected wall on a 4090, each pass's training plus its final probe of about
# 5.4 minutes (round 2's): bring-up and tests 15 minutes, the inputs 10, the
# barrier 4; A and A2 about 36 minutes each (500,000 positions at about 280 a
# second, round 1's padded arm with its budget); B about 13 (main's fastest
# measured 1,242 positions a second, the compile); B8 about 15 (fp8's speed at
# these sizes is the bench's to measure; this assumes 1,000 a second); B64
# about 12 and BHALF about 11 (forward-only windows cost about a third of a
# trained one). All six about 2 hours 30 minutes, $1.07 at $0.42 an hour: the
# fixed part $0.20, A and A2 $0.25 each, B $0.09, B8 $0.11, B64 $0.08, BHALF
# $0.08. A pass is cut at PASS_CUT_MIN (70: about twice A's estimate) or after
# PASS_STALL_MIN without a log line. BOX_MAX_H 4 caps it at about $1.70 plus
# the final round.
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md). The
# records go to HF after each pass, the passes' checkpoints with them (their
# .stable.pt stay on the machine). Never `set -x`: the HF token and the
# instance key are in the environment.
# box-needs: disk_gb=150 ram_gb=64 gpu_ram_gb=24 cores=32 gpu=4090
set -uo pipefail
WORKDIR=/workspace
STAGE="${STAGE:-}"
HF_DIR="${HF_DIR:-tier-b/trainer_acceptance_20261016}"
export HF_DIR
BOX_MAX_H="${BOX_MAX_H:-4}"
RUN_SEED="${RUN_SEED:-20261012}"                # round 2's
LR="${LR:-2.8e-5}"
PASS_POSITIONS="${PASS_POSITIONS:-500000}"
DECAY_FROM="${DECAY_FROM:-0.5}"
ARMS="${ARMS:-A B A2 B8 B64 BHALF}"
A_FLAGS="${A_FLAGS:---columns --activation-budget-gb 15.97 --activation-bytes-per-token 150329}"
B_FLAGS="${B_FLAGS:---packed-trunk --columns --window-batched --activation-budget-gb 19.18 --activation-bytes-per-token 102139 --compiled-trunk}"
B8_FLAGS="${B8_FLAGS:-}"                        # default: B_FLAGS without --fused-relu, plus --fp8-trunk
BIG_STREAMS="${BIG_STREAMS:-64}"
SUBSET_EVERY="${SUBSET_EVERY:-2}"
WORKERS="${WORKERS:-}"
TESTS_CUT_MIN="${TESTS_CUT_MIN:-40}"
GRADCHECK_CUT_MIN="${GRADCHECK_CUT_MIN:-25}"
PASS_CUT_MIN="${PASS_CUT_MIN:-70}"
PASS_STALL_MIN="${PASS_STALL_MIN:-15}"
OUT=$WORKDIR/trainer_acceptance
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }

notes() {                        # what is done, for the finish reason
    local name done_=()
    [ ! -f "$OUT/gradcheck.json" ] || done_+=(gradcheck)
    for name in $ARMS; do
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
box_restore gradcheck.json cuda_tests.log tests.log arms_failed.txt \
    $(for p in $ARMS; do echo "pass_$p.probe_sides.jsonl"; done) || box_finish "RESTORE_FAILED (restore.log)" 1
box_pip huggingface_hub psutil pytest scipy requests || echo "pip install failed (pip.log)"
for name in $ARMS; do
    box_upload_skip "pass_$name.stable.pt"
done

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
        tests/test_own_loss.py tests/test_packed_trunk.py tests/test_trunk_modes.py tests/test_trunk_variants.py \
        tests/test_trainer_acceptance_readout.py tests/test_fp8_linear.py tests/test_rules_version.py \
        tests/test_train_every.py -q -p no:cacheprovider -m ""
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
        --warmup-steps 0 --decay-from "$DECAY_FROM" --pass-positions "$PASS_POSITIONS"
        --probe-every 1200000 --barrier-positions 1000000 --probe-ks "0,64")
INIT=(--init-from "$INIT_PT" --init-other-records)
if [ -z "$B8_FLAGS" ]; then
    read -r -a b_words <<< "$B_FLAGS"
    for word in "${b_words[@]}"; do
        [ "$word" = --fused-relu ] || B8_FLAGS+="$word "
    done
    B8_FLAGS+="--fp8-trunk"
fi
BIG_LR=$(awk -v lr="$LR" -v n="$BIG_STREAMS" 'BEGIN { printf "%.6g", lr * sqrt(n / 32) }')
read -r -a ARM_NAMES <<< "$ARMS"
case " $ARMS " in *" A "*) ;; *) box_finish "NO_ARM_A: every arm is read against A (ARMS=$ARMS)" 1 ;; esac
pass_flags() {                   # pass_flags NAME: the flags of pass NAME
    case $1 in
        A|A2) echo "--lr $LR $A_FLAGS" ;;
        B) echo "--lr $LR $B_FLAGS" ;;
        B8) echo "--lr $LR $B8_FLAGS" ;;
        B64) echo "--lr $BIG_LR $B_FLAGS --streams $BIG_STREAMS --own-streams $(( BIG_STREAMS / 2 ))" ;;
        BHALF) echo "--lr $LR $B_FLAGS --train-every $SUBSET_EVERY" ;;
        *) return 1 ;;
    esac
}
for name in "${ARM_NAMES[@]}"; do
    pass_flags "$name" > /dev/null || box_finish "UNKNOWN_ARM $name (ARMS)" 1
done
in_arms() { case " $ARMS " in *" $1 "*) return 0 ;; esac; return 1; }
final_probe() {                  # final_probe NAME: the pass's final probe is written
    grep -q '"final": true' "$OUT/pass_$1.probe_sides.jsonl" 2>/dev/null
}
left_out() {                     # left_out NAME WHY: NAME runs no pass, and says why
    echo "LEFT_OUT $1: $2" | tee -a "$OUT/arms_failed.txt"
    echo "$2" > "$OUT/pass_$1.failed"
}
train_pass() {                   # train_pass NAME: the pass, once; resumed from its checkpoint after a re-entry
    local name=$1
    local -a start=("${INIT[@]}")
    ! box_marked_this_stage "$OUT/pass_$name.done" || return 0
    [ ! -f "$OUT/pass_$name.failed" ] || return 1
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before pass $name: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    [ ! -f "$OUT/pass_$name.pt" ] || start=(--resume)
    # shellcheck disable=SC2046 # the pass's flags are a list
    box_bounded --stall "$OUT/pass_$name.log" "$PASS_STALL_MIN" "pass $name" "$PASS_CUT_MIN" "pass_$name.log" \
        python tools/sequence_train.py "${COMMON[@]}" "${start[@]}" --out "$OUT/pass_$name.pt" $(pass_flags "$name")
    if [ "$BOX_RC" -ne 0 ] || ! final_probe "$name"; then
        box_upload
        [ "$name" != A ] || box_finish "PASS_FAILED A rc=$BOX_RC $BOX_WHY (pass_A.log) $(notes)" 1
        left_out "$name" "its pass failed, rc=$BOX_RC $BOX_WHY (pass_$name.log)"
        return 1
    fi
    box_mark "$OUT/pass_$name.done"
    box_upload
}
check_passed() {                 # check_passed FLAGS: the gradient check passed the candidate FLAGS
    timeout 1m python - "$OUT/gradcheck.json" "$1" <<'EOF'
import json, sys
check = json.load(open(sys.argv[1], encoding="utf-8"))
want = " ".join(sys.argv[2].split())
sys.exit(0 if any(" ".join(c["candidate"].split()) == want and c["pass"] for c in check.get("candidates", [])) else 1)
EOF
}

# The barrier: B's and B8's gradients against A's, from the same weights at rate 0.
if [ ! -f "$OUT/gradcheck.json" ]; then
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before the gradient check: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    rm -f "$OUT/gradcheck.json.pending"
    candidates=("--candidate=$B_FLAGS")
    ! in_arms B8 || candidates+=("--candidate=$B8_FLAGS")
    box_bounded gradcheck "$GRADCHECK_CUT_MIN" gradcheck.log python tools/bench_sequence_step.py gradcheck \
        --out "$OUT/gradcheck.json.pending" "--reference=$A_FLAGS" "${candidates[@]}" -- "${COMMON[@]}" \
        --lr "$LR" "${INIT[@]}" --out "$OUT/gradcheck_arm.pt"
    case $BOX_RC in
        0|8) mv -f "$OUT/gradcheck.json.pending" "$OUT/gradcheck.json" ;;
        *) box_upload
           box_finish "B_GRADCHECK_UNFINISHED rc=$BOX_RC $BOX_WHY (gradcheck.log)" 1 ;;
    esac
    box_upload
fi
check_passed "$B_FLAGS" || box_finish "B_GRADCHECK_FAILED: B's gradients are not A's within the check (gradcheck.json)" 1
if in_arms B8 && [ ! -f "$OUT/pass_B8.failed" ] && ! check_passed "$B8_FLAGS"; then
    left_out B8 "its gradient check did not pass (gradcheck.json)"
fi

for name in "${ARM_NAMES[@]}"; do
    train_pass "$name" || true
done

readout_args=(--a "$OUT/pass_A.probe_sides.jsonl")
! box_marked_this_stage "$OUT/pass_A2.done" || readout_args+=(--a2 "$OUT/pass_A2.probe_sides.jsonl")
for name in "${ARM_NAMES[@]}"; do
    case $name in A|A2) continue ;; esac
    ! box_marked_this_stage "$OUT/pass_$name.done" || readout_args+=(--arm "$name=$OUT/pass_$name.probe_sides.jsonl")
done
case " ${readout_args[*]} " in
    *" --arm "*) ;;
    *) box_finish "NO_ARM_TRAINED: no arm but A and A2 finished $(notes)" 1 ;;
esac
box_bounded readout 5 readout.log python tools/trainer_acceptance_readout.py "${readout_args[@]}" \
    --out "$OUT/acceptance.json" || box_finish "READOUT_FAILED rc=$BOX_RC (readout.log)" 1
cp -f "$OUT/readout.log" "$OUT/acceptance.txt"
VERDICT=$(timeout 1m python - "$OUT" "$ARMS" <<'EOF'
import json, pathlib, re, sys
out = pathlib.Path(sys.argv[1])
r = json.loads((out / "acceptance.json").read_text(encoding="utf-8"))
rates = {}
for name in sys.argv[2].split():
    log = out / f"pass_{name}.log"
    m = re.findall(r"SEQUENCE_TRAIN_DONE (\d+) positions, \d+ steps in (\d+) s",
                   log.read_text(encoding="utf-8", errors="replace")) if log.exists() else []
    rates[name] = round(int(m[-1][0]) / max(1, int(m[-1][1])), 1) if m else None
print(json.dumps({"pass": {a: v["pass"] for a, v in r["arms"].items()},
                  "not_worse": {a: v["not_worse"] for a, v in r["arms"].items()}, "positions_per_second": rates}))
EOF
)
FAILED_STEPS=$(cut -d' ' -f1-2 "$OUT/arms_failed.txt" 2>/dev/null | tr '\n' ' ')
box_finish "TRAINER_ACCEPTANCE_DONE ${VERDICT:0:600}; cuda tests: ${CUDA_TESTS:-none}; left out: ${FAILED_STEPS:-none}"
# ---- the passes (end)
