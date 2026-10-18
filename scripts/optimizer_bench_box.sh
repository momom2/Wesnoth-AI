#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# The optimizer bench (wesnoth_ai/optimizers.py, tools/sequence_train.py
# --optimizer): imitation passes from scratch, the reference architecture
# under configuration B (B_FLAGS: the packed trunk, the columns, the window
# batched, the compiled trunk, the activation budget sized from the device),
# each over the same first PASS_POSITIONS corpus positions in the same order
# under the same seed, the recipe's 300-step warm-up, a holdout probe at about
# half the pass (PROBE_EVERY: the first window past it; the next would fall at
# the pass's end, beside the final probe) and at its end, at 0 and 64 slots.
#
# The arms (ARMS, names OPTIMIZER_RATE: the rate lo, mid or hi of the
# optimizer's grid, mid2 the middle rate again), in this order by default so
# that every candidate's middle rate is done first:
#   adamw          the recipe's AdamW (LR_GRID: 1.4e-4, 2.8e-4, 5.6e-4);
#   muon, soap     on the trunk's matrices, adamw_hygiene elsewhere, the same
#                  grid (Muon's update matches AdamW's RMS, SOAP is Adam in a
#                  rotated basis);
#   schedulefree   schedule-free AdamW, SF_GRID (1x, 3x and 10x the AdamW
#                  grid's middle: the package's README puts schedule-free's
#                  rate at 1x to 10x a scheduled one's), no cooldown and no
#                  weight average (its own average replaces both);
#   adamw_hygiene  AdamW, beta2 0.95, no decay on the 1-D weights, the
#                  embeddings and the memory's rows, the same grid;
#   adamw_mid2     adamw_mid again: the run-to-run noise of GPU training.
# Every scheduled arm cools down linearly to 0 over the pass's last 20%
# (DECAY_FROM 0.8), so each is read at a trained endpoint, and keeps a
# weight average (EMA_DECAY 0.998, about 500 steps of the pass's 1,954).
#
# After the arms, the averages of the best two arms with one (EMA_ARMS auto:
# the lowest final EMA_KEY, the policy's holdout cross-entropy at 64 slots;
# or a list of names, empty for none) are probed: each pass resumed when finished, which
# probes it again, at its average (`--resume --probe-weights ema`). Then
# tools/optimizer_bench_readout.py reads every finished arm against
# adamw_mid, with adamw_mid2 - adamw_mid as the noise, its differences paired
# over holdout games from the probes' per-game-side sums, at both probes; the
# averages against their arm's live weights; each arm's positions a second,
# peak device memory and optimizer state (bench.json, bench.txt). The
# readings' criterion is the lead's to pre-register.
#
# Every arm fails alone: a pass that fails is recorded in arms_failed.txt
# (and pass_NAME.failed, which a re-entry leaves failed) and the others run;
# so does an average's probe. A pass interrupted by a re-entry resumes from
# its last checkpoint (every 250,000 positions); the checkpoints stay on the
# machine, so on another machine an unfinished pass starts again, its partial
# records cleared, and a finished one keeps its records and markers from HF.
#
# One code stage, STAGE (tools/stage_code.py --script scripts/optimizer_bench_box.sh).
# The trainer's inputs come from scripts/trainer_inputs.sh (own1 for the
# vocabularies, the corpus at version 5 from the input cache and its
# sequences, the columns; the own sequences it also brings up are not read
# here).
#
# Expected wall on a 4090, from configuration B's measured 1,477 positions a
# second: a pass's 1,000,448 positions 11.3 minutes, the compile about 1.5,
# its two probes about 6.2 (185 s each at 0 and 64 slots, the acceptance's),
# the start 1: about 20 minutes an arm, SOAP's about 22 (its projections and
# its refresh every 10 steps, estimated at 13% of a step: no measurement
# yet). Bring-up and tests 15 minutes, the inputs 5 to 8 (the corpus and the
# own sequences from the input cache), the 16 arms 5 hours 25
# minutes, the two averages' probes 8 minutes: about 6 hours, $2.50 at $0.42
# an hour; each arm $0.14. A pass is cut at PASS_CUT_MIN (45: about twice
# its estimate) or after PASS_STALL_MIN without a log line. BOX_MAX_H 8 caps
# it at about $3.40 plus the final round.
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md). The
# records go to HF after each pass. Never `set -x`: the HF token and the
# instance key are in the environment.
# box-needs: disk_gb=150 ram_gb=64 gpu_ram_gb=24 cores=32 gpu=4090
set -uo pipefail
WORKDIR=/workspace
STAGE="${STAGE:-}"
HF_DIR="${HF_DIR:-tier-b/optimizer_bench_20261016}"
export HF_DIR
BOX_MAX_H="${BOX_MAX_H:-8}"
RUN_SEED="${RUN_SEED:-20260929}"                # the recipe's
PASS_POSITIONS="${PASS_POSITIONS:-1000000}"
PROBE_EVERY="${PROBE_EVERY:-501000}"
DECAY_FROM="${DECAY_FROM:-0.8}"
EMA_DECAY="${EMA_DECAY:-0.998}"
EMA_ARMS="${EMA_ARMS-auto}"                 # empty: none
EMA_KEY="${EMA_KEY:-ce_all_k64}"
LR_GRID="${LR_GRID:-1.4e-4 2.8e-4 5.6e-4}"
SF_GRID="${SF_GRID:-2.8e-4 8.4e-4 2.8e-3}"
ARMS="${ARMS:-adamw_mid muon_mid soap_mid schedulefree_mid adamw_hygiene_mid adamw_lo adamw_hi muon_lo muon_hi \
soap_lo soap_hi schedulefree_lo schedulefree_hi adamw_hygiene_lo adamw_hygiene_hi adamw_mid2}"
B_FLAGS="${B_FLAGS:---packed-trunk --columns --window-batched --compiled-trunk --activation-budget-gb auto}"
REFERENCE=adamw_mid
REPEAT=adamw_mid2
WORKERS="${WORKERS:-}"
TESTS_CUT_MIN="${TESTS_CUT_MIN:-40}"
PASS_CUT_MIN="${PASS_CUT_MIN:-45}"
PASS_STALL_MIN="${PASS_STALL_MIN:-15}"
EMA_CUT_MIN="${EMA_CUT_MIN:-20}"
OUT=$WORKDIR/optimizer_bench
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }

notes() {                        # what is done, for the finish reason
    local name done_=()
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
box_restore cuda_tests.log tests.log arms_failed.txt $(for p in $ARMS; do
        echo "pass_$p.done pass_$p.failed pass_$p.probe_sides.jsonl pass_$p.probe.jsonl pass_$p.steps.jsonl"
        echo "ema_$p.done pass_$p.ema.probe_sides.jsonl pass_$p.ema.probe.jsonl"
    done) || box_finish "RESTORE_FAILED (restore.log)" 1
box_pip huggingface_hub psutil pytest scipy requests schedulefree==1.4.1 || echo "pip install failed (pip.log)"
for name in $ARMS; do
    box_upload_skip "pass_$name.pt"
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
    box_bounded tests "$TESTS_CUT_MIN" tests.log python -m pytest tests/test_optimizers.py \
        tests/test_optimizer_passes.py tests/test_optimizer_bench_box.py tests/test_sequence_train.py \
        tests/test_sequence_speed.py tests/test_sequence_columns.py tests/test_sequence_launches.py \
        tests/test_packed_trunk.py tests/test_trunk_modes.py tests/test_trunk_variants.py tests/test_rules_version.py \
        tests/test_train_every.py tests/test_memory_plan.py tests/test_input_cache.py \
        -q -p no:cacheprovider -m ""
    tail -n 3 "$OUT/tests.log"
    [ "$BOX_RC" -eq 0 ] || box_finish "TESTS_FAILED rc=$BOX_RC (tests.log)" 1
    : > "$OUT/cuda_tests.log"
    box_bounded cuda-tests 20 cuda_tests.log python -m pytest tests/test_optimizers_cuda.py \
        tests/test_trainer_speed_cuda.py tests/test_trunk_modes_cuda.py -q -s -p no:cacheprovider -m ""
    tail -n 3 "$OUT/cuda_tests.log"
    box_mark "$BOX_STATE/TESTED"
fi
CUDA_TESTS=$(grep -Eo "[0-9]+ (passed|failed|error|errors|skipped)" "$OUT/cuda_tests.log" | tr '\n' ' ')
box_upload

trainer_inputs

# ---- the arms (begin): tests/test_optimizer_bench_box.py runs this block with the library stubbed
COMMON=(--sequences "$SEQ" --dataset "$CORPUS" --seed "$RUN_SEED" --device cuda
        --pass-positions "$PASS_POSITIONS" --probe-every "$PROBE_EVERY"
        --barrier-positions "$(( PASS_POSITIONS * 2 ))" --probe-ks "0,64")
read -r -a ARM_NAMES <<< "$ARMS"
read -r -a LRS <<< "$LR_GRID"
read -r -a SF_LRS <<< "$SF_GRID"
if [ "${#LRS[@]}" -ne 3 ] || [ "${#SF_LRS[@]}" -ne 3 ]; then
    box_finish "BAD_GRID: three rates each (LR_GRID, SF_GRID)" 1
fi
arm_flags() {                    # arm_flags NAME: the trainer flags of arm NAME
    local optimizer=${1%_*} rate=${1##*_} i
    case $rate in
        lo) i=0 ;;
        mid|mid2) i=1 ;;
        hi) i=2 ;;
        *) return 1 ;;
    esac
    case $optimizer in
        adamw|adamw_hygiene|muon|soap)
            echo "--optimizer $optimizer --lr ${LRS[$i]} $B_FLAGS --decay-from $DECAY_FROM --weight-ema $EMA_DECAY" ;;
        schedulefree)
            echo "--optimizer schedulefree --lr ${SF_LRS[$i]} $B_FLAGS" ;;
        *) return 1 ;;
    esac
}
for name in "${ARM_NAMES[@]}"; do
    arm_flags "$name" > /dev/null || box_finish "UNKNOWN_ARM $name (ARMS)" 1
done
final_probe() {                  # final_probe NAME: the pass's final probe is written
    grep -q '"final": true' "$OUT/pass_$1.probe_sides.jsonl" 2>/dev/null
}
left_out() {                     # left_out NAME WHY: NAME runs no further, and says why
    echo "LEFT_OUT $1: $2" | tee -a "$OUT/arms_failed.txt"
    echo "$2" > "$OUT/pass_$1.failed"
}
train_pass() {                   # train_pass NAME: the pass, once; resumed from its checkpoint after a re-entry
    local name=$1
    local -a start=()
    ! box_marked_this_stage "$OUT/pass_$name.done" || return 0
    [ ! -f "$OUT/pass_$name.failed" ] || return 1
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before pass $name: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    if [ -f "$OUT/pass_$name.pt" ]; then
        start=(--resume)
    else                         # a pass from its start: no rows of an earlier attempt in its records
        rm -f "$OUT/pass_$name".{probe,probe_sides,steps,signal,ema.probe,ema.probe_sides}.jsonl
    fi
    # shellcheck disable=SC2046 # the arm's flags are a list
    box_bounded --stall "$OUT/pass_$name.log" "$PASS_STALL_MIN" "pass $name" "$PASS_CUT_MIN" "pass_$name.log" \
        python tools/sequence_train.py "${COMMON[@]}" ${start[@]+"${start[@]}"} --out "$OUT/pass_$name.pt" \
        $(arm_flags "$name")
    if [ "$BOX_RC" -ne 0 ] || ! final_probe "$name"; then
        box_upload
        left_out "$name" "its pass failed, rc=$BOX_RC $BOX_WHY (pass_$name.log)"
        return 1
    fi
    box_mark "$OUT/pass_$name.done"
    box_upload
}
for name in "${ARM_NAMES[@]}"; do
    train_pass "$name" || true
done

# The averages: the best two arms that keep one (EMA_ARMS auto), each finished pass probed again at its average.
DONE_ARMS=()
for name in "${ARM_NAMES[@]}"; do
    ! box_marked_this_stage "$OUT/pass_$name.done" || DONE_ARMS+=("$name")
done
[ "${#DONE_ARMS[@]}" -gt 0 ] || box_finish "NO_ARM_TRAINED $(notes)" 1
if [ "$EMA_ARMS" = auto ]; then
    EMA_ARMS=$(timeout 1m python - "$OUT" "$EMA_KEY" "${DONE_ARMS[@]}" <<'EOF'
import json, pathlib, sys
out, key, names = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
reading, _, run = key.rpartition("_")
scores = {}
for name in names:
    if name.startswith("schedulefree"):
        continue                     # no weight average: its own average is what its probe reads
    rows = [json.loads(line) for line in (out / f"pass_{name}.probe.jsonl").read_text().splitlines() if line.strip()]
    finals = [r for r in rows if r.get("final") and r.get("weights", "live") == "live"]
    if finals and finals[-1].get(run, {}).get(reading) is not None:
        scores[name] = finals[-1][run][reading]
print(" ".join(sorted(scores, key=scores.get)[:2]))
EOF
    ) || EMA_ARMS=""
fi
for name in $EMA_ARMS; do
    ! box_marked_this_stage "$OUT/ema_$name.done" || continue
    case " ${DONE_ARMS[*]} " in *" $name "*) ;; *) echo "EMA_SKIPPED $name: its pass is not done" \
        | tee -a "$OUT/arms_failed.txt"; continue ;; esac
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before the average of $name: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    # shellcheck disable=SC2046 # the arm's flags are a list
    box_bounded "ema $name" "$EMA_CUT_MIN" "ema_$name.log" python tools/sequence_train.py "${COMMON[@]}" --resume \
        --out "$OUT/pass_$name.pt" $(arm_flags "$name") --probe-weights ema
    if [ "$BOX_RC" -eq 0 ] && grep -q '"final": true' "$OUT/pass_$name.ema.probe_sides.jsonl" 2>/dev/null; then
        box_mark "$OUT/ema_$name.done"
    else
        echo "EMA_FAILED $name rc=$BOX_RC $BOX_WHY (ema_$name.log)" | tee -a "$OUT/arms_failed.txt"
    fi
    box_upload
done

box_bounded readout 5 readout.log python tools/optimizer_bench_readout.py --dir "$OUT" --reference "$REFERENCE" \
    --repeat "$REPEAT" --arms "${DONE_ARMS[@]}" --out "$OUT/bench.json" \
    || box_finish "READOUT_FAILED rc=$BOX_RC (readout.log) $(notes)" 1
cp -f "$OUT/readout.log" "$OUT/bench.txt"
SUMMARY=$(timeout 1m python - "$OUT/bench.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
def diff(arm, probe, key):
    d = arm["probes"].get(probe, {}).get(key, {}).get("minus", {})
    return None if d.get("diff") is None else [round(d["diff"], 5), None if d["se"] is None else round(d["se"], 5)]
print(json.dumps({name: {"ce_k64": diff(a, "final", "ce_all_k64"), "pos_s": a["positions_per_second"]}
                  for name, a in r["arms"].items()}))
EOF
)
FAILED_STEPS=$(cut -d' ' -f1-2 "$OUT/arms_failed.txt" 2>/dev/null | tr '\n' ' ')
box_finish "OPTIMIZER_BENCH_DONE ${SUMMARY:0:1500}; ema: ${EMA_ARMS:-none}; cuda tests: ${CUDA_TESTS:-none}; left out: ${FAILED_STEPS:-none}"
# ---- the arms (end)
