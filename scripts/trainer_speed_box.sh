#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# The sequence trainer's speed on one box (docs/trainer_speed_20261013.md,
# "Benchmark"): the step of the base code and of this stage, its flags off
# and on, timed under round 2's training arguments
# (scripts/memory_own_trajectories_box.sh, ROUND=2: from own1, its own
# game-sides beside the corpus's) by tools/bench_sequence_step.py.
#
# Two code stages: STAGE, this branch's (tools/stage_code.py --script
# scripts/trainer_speed_box.sh), and BASE_STAGE, the commit the work branched
# from (exp/memory-own-trajectories at bd5f33d), staged from a checkout of it
# with tools/stage_code.py --out ... --upload tier-b/staging/<name>.tar.gz. The
# base stage is unpacked beside this one and its trainer timed with this
# stage's bench (--repo); both use this stage's Rust wheel (the work changes
# no Rust).
#
# The steps:
#   the tests of this path on this wheel and GPU: the CPU-path tests gate the
#     run; the CUDA staging tests (tests/test_trainer_speed_cuda.py) are
#     recorded in cuda_tests.log and named in the finish line;
#   own1 from HF, its SHA-256 checked; the corpus rebuilt at version 5 and
#     pre-encoded with own1's vocabularies;
#   round 2's own sequences: own_sequences_r2.tar.gz from HF_DIR when an
#     earlier session left it there, else built as round 2 built them (the
#     confirmation's three matches and round 2's generation match, own1's
#     64-slot sides, labelled by own1 at 0 slots) and sent to HF for the
#     next session (CACHE_OWN=0: not sent);
#   the packed trunk's gradient check (gradcheck.json): its gradients against
#     the padded trunk's at dropout 0, a crash barrier; a failure leaves the
#     packed arms out;
#   the arms, each WARMUP untimed windows, WINDOWS timed and PROFILE_WINDOWS
#     profiled (arm_NAME.jsonl, with the trainer's rows in
#     arm_NAME.steps.jsonl):
#       base_1, this_1, base_2, this_2: the base code and this stage's
#         defaults, alternated; the session is valid when the two base runs
#         agree within 3% (VALID_SPREAD);
#       cal_padded (short): a budget of CAL_PADDED_GB, the activation bytes
#         a padded token measured;
#       budget: this stage with the activation budget that fits the worst
#         window under PEAK_GB (22 GiB) after this_1's peak and MARGIN_GB;
#       packed: --packed-trunk; cal_packed (short) and packed_budget as
#         above on the packed trunk;
#       preload: the fastest configuration with every record read first;
#   compare (cmp_A_vs_B.json) and speed.json: positions a second per arm and
#     each change's speedup, median and 95% interval over the windows;
#   the records to HF, the instance stopped.
#
# Expected wall on a 4090 (round 2's walls.txt): bring-up and tests 15
# minutes, inputs 2, corpus and pre-encoding 7, the own sequences 55 the first
# time (candidates 3, targets 48, the upload a few) or 5 from HF, the
# gradient check 3, eight arms of about 6 minutes and two calibrations of 3:
# 2.4 hours the first time, 1.5 after; $1.0 and $0.6 at $0.42 an hour.
# BOX_MAX_H 4 caps the first at about $1.7 plus the final round.
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md). No
# periodic upload rounds while the arms run (they would share the machine
# with the timing): the records go up after each stage and at the finish.
# Re-entry skips the stages and arms that are done. Never `set -x`: the HF
# token and the instance key are in the environment.
# box-needs: disk_gb=150 ram_gb=64 gpu_ram_gb=24 cores=32 gpu=4090
set -uo pipefail
WORKDIR=/workspace
STAGE="${STAGE:-}"
BASE_STAGE="${BASE_STAGE:-}"
HF_DIR="${HF_DIR:-tier-b/trainer_speed_20261013}"
export HF_DIR
BOX_MAX_H="${BOX_MAX_H:-4}"
RAW_TAR="${RAW_TAR:-tier-b/corpus_v3/raw_corpus_20260929.tar}"
INIT_HF="${INIT_HF:-tier-b/memory_own_trajectories_20261011/own.pt}"
INIT_SHA256="${INIT_SHA256:-70925e770d43496e3ea8f9523656f201b71ada228a0362b0eb0a600c8c107f80}"
SOURCE_TARS="${SOURCE_TARS:-tier-b/own_confirm_20261011/games_own64_vs_p3.tar.gz \
tier-b/own_confirm_20261011/games_own64_vs_own0.tar.gz \
tier-b/own_confirm_20261011/games_own64_selfplay.tar.gz \
tier-b/memory_own_round2_20261011/games_gen_own1_selfplay.tar.gz}"
RUN_SEED="${RUN_SEED:-20261012}"                # round 2's
LR="${LR:-2.8e-5}"
PASS_POSITIONS="${PASS_POSITIONS:-2000000}"
DECAY_FROM="${DECAY_FROM:-0.5}"
PROBE_EVERY="${PROBE_EVERY:-1200000}"
CACHE_OWN="${CACHE_OWN:-1}"
WARMUP="${WARMUP:-20}"
WINDOWS="${WINDOWS:-80}"
PROFILE_WINDOWS="${PROFILE_WINDOWS:-3}"
CAL_WARMUP="${CAL_WARMUP:-3}"
CAL_WINDOWS="${CAL_WINDOWS:-10}"
CAL_PADDED_GB="${CAL_PADDED_GB:-4}"             # about one padded time step (docs/design_constants.md)
CAL_PACKED_GB="${CAL_PACKED_GB:-2.5}"           # about two packed time steps
PEAK_GB="${PEAK_GB:-22}"
MARGIN_GB="${MARGIN_GB:-1}"
VALID_SPREAD="${VALID_SPREAD:-0.03}"
WORKERS="${WORKERS:-}"
# Bounds in minutes.
TESTS_CUT_MIN="${TESTS_CUT_MIN:-40}"
BUILD_CUT_MIN="${BUILD_CUT_MIN:-60}"
BUILD_STALL_MIN="${BUILD_STALL_MIN:-15}"
PREENCODE_CUT_MIN="${PREENCODE_CUT_MIN:-150}"
PREENCODE_STALL_MIN="${PREENCODE_STALL_MIN:-20}"
CAND_CUT_MIN="${CAND_CUT_MIN:-60}"
CAND_STALL_MIN="${CAND_STALL_MIN:-20}"
TARGETS_CUT_MIN="${TARGETS_CUT_MIN:-120}"
TARGETS_STALL_MIN="${TARGETS_STALL_MIN:-20}"
ARM_CUT_MIN="${ARM_CUT_MIN:-25}"
OUT=$WORKDIR/trainer_speed
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }

BASE_REPO=$WORKDIR/base_repo
RAW=$WORKDIR/raw
CORPUS=$WORKDIR/corpus
SEQ=$WORKDIR/sequences
OWN_GAMES=$WORKDIR/own_games_r2
CAND=$WORKDIR/own_candidates_r2
OWN=$WORKDIR/own_sequences_r2
INIT_PT=$WORKDIR/own1.pt
ARMS="base_1 this_1 base_2 this_2 cal_padded budget packed cal_packed packed_budget preload"

notes() {                        # what is done, for the finish reason
    local name done_=()
    [ ! -f "$OUT/gradcheck.json" ] || done_+=(gradcheck)
    for name in $ARMS; do
        [ ! -f "$OUT/arm_$name.done" ] || done_+=("$name")
    done
    echo "done: ${done_[*]:-nothing}"
}
# shellcheck disable=SC2317 # called by the library, in the reason of an unexpected exit
box_notes() { notes; }

box_init
[ -n "$WORKERS" ] || WORKERS=$(box_workers)
[ -n "$STAGE" ] || box_finish "NO_STAGE: build the code stage (tools/stage_code.py) and pass STAGE" 1
[ -n "$BASE_STAGE" ] || box_finish "NO_BASE_STAGE: stage the base commit and pass BASE_STAGE" 1
box_bind_run_stage
box_restore speed.json gradcheck.json cuda_tests.log tests.log || box_finish "RESTORE_FAILED (restore.log)" 1
box_pip huggingface_hub psutil pytest scipy requests || echo "pip install failed (pip.log)"

# ---- the code, the base code and the Rust wheel
box_stage_code || box_finish "CODE_STAGING_FAILED (staging.log)" 1
cd "$BOX_REPO" || box_finish "CODE_STAGING_FAILED (no $BOX_REPO)" 1
box_build_wheel || box_finish "BUILD_FAILED (build.log)" 1
if [ "$(cat "$BASE_REPO/.staged_from" 2>/dev/null)" != "$BASE_STAGE" ]; then
    rm -rf "$BASE_REPO.tmp"
    box_bounded base-stage 15 staging.log python "$BOX_LIB/box_stage.py" fetch "$BASE_STAGE" --into "$BASE_REPO.tmp" \
        || box_finish "BASE_STAGING_FAILED (staging.log)" 1
    { printf '%s\n' "$BASE_STAGE" > "$BASE_REPO.tmp/.staged_from" && rm -rf "$BASE_REPO" \
        && mv "$BASE_REPO.tmp" "$BASE_REPO"; } || box_finish "BASE_STAGING_FAILED (unpacking)" 1
fi
[ -f "$BASE_REPO/tools/sequence_train.py" ] || box_finish "BASE_STAGING_FAILED (no trainer in $BASE_STAGE)" 1
box_facts > "$OUT/box.txt.tmp" 2>&1
mv -f "$OUT/box.txt.tmp" "$OUT/box.txt"
timeout 2m python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || box_finish "NO_CUDA (box.txt)" 1
box_upload

# ---- the tests: the CPU paths gate the run, the CUDA staging tests are recorded
if ! box_marked_this_stage "$BOX_STATE/TESTED"; then
    : > "$OUT/tests.log"
    box_bounded tests "$TESTS_CUT_MIN" tests.log python -m pytest tests/test_sequence_train.py \
        tests/test_sequence_speed.py tests/test_own_loss.py tests/test_own_trajectories.py \
        tests/test_memory_model.py tests/test_packed_trunk.py tests/test_packed_trunk_cuda.py \
        tests/test_packed_embed.py tests/test_preencode_sequences.py tests/test_memory_learner.py \
        -q -p no:cacheprovider -m ""
    tail -n 3 "$OUT/tests.log"
    [ "$BOX_RC" -eq 0 ] || box_finish "TESTS_FAILED rc=$BOX_RC (tests.log)" 1
    : > "$OUT/cuda_tests.log"
    box_bounded cuda-tests 15 cuda_tests.log python -m pytest tests/test_trainer_speed_cuda.py \
        -q -p no:cacheprovider -m ""
    tail -n 3 "$OUT/cuda_tests.log"
    box_mark "$BOX_STATE/TESTED"
fi
CUDA_TESTS=$(grep -Eo "[0-9]+ (passed|failed|error|errors|skipped)" "$OUT/cuda_tests.log" | tr '\n' ' ')
box_upload

# ---- own1, its SHA-256 checked
if ! { [ -f "$INIT_PT" ] && echo "$INIT_SHA256  $INIT_PT" | sha256sum -c --status; }; then
    box_bounded init 20 staging.log python - "$INIT_HF" "$INIT_PT" <<'EOF' \
        || box_finish "INIT_CHECKPOINT_FAILED ($INIT_HF; staging.log)" 1
import os, shutil, sys
from huggingface_hub import hf_hub_download
shutil.copyfile(hf_hub_download("momom2/wesnoth-model-checkpoints", sys.argv[1]), sys.argv[2] + ".tmp")
os.replace(sys.argv[2] + ".tmp", sys.argv[2])
EOF
    echo "$INIT_SHA256  $INIT_PT" | sha256sum -c --status || box_finish "INIT_CHECKPOINT_SHA256 ($INIT_HF)" 1
fi

# ---- the corpus at version 5 and its sequences with own1's vocabularies
if ! box_marked_this_stage "$BOX_STATE/CORPUS_DONE"; then
    box_bounded raw 30 staging.log python - "$RAW_TAR" "$RAW" <<'EOF' || box_finish "RAW_FAILED (staging.log)" 1
import sys, tarfile
from huggingface_hub import hf_hub_download
with tarfile.open(hf_hub_download("momom2/wesnoth-model-checkpoints", sys.argv[1]), "r") as tf:
    tf.extractall(sys.argv[2])
    print("raw replays", sum(1 for name in tf.getnames() if name.endswith(".bz2")), flush=True)
EOF
    rm -rf "$CORPUS" "${CORPUS}_duplicates"
    from=$(box_size "$OUT/corpus_build.log")
    box_bounded --stall "$OUT/corpus_build.log" "$BUILD_STALL_MIN" corpus "$BUILD_CUT_MIN" corpus_build.log \
        python tools/build_imitation_dataset.py --raw-root "$RAW" --out "$CORPUS" --workers "$WORKERS"
    tail -c "+$(( from + 1 ))" "$OUT/corpus_build.log" | grep -q "BUILD_DONE" \
        || box_finish "CORPUS_${BOX_WHY^^} rc=$BOX_RC (corpus_build.log)" 1
    box_bounded corpus-check 10 corpus_build.log \
        python tools/build_imitation_dataset.py --out "$CORPUS" --check "$OUT/corpus_summary.json" \
        || box_finish "CORPUS_BARRIER rc=$BOX_RC (corpus_build.log)" 1
    box_mark "$BOX_STATE/CORPUS_DONE"
fi
if ! box_marked_this_stage "$SEQ/SEQUENCES_DONE"; then
    box_marked_this_stage "$SEQ/STARTED" || { rm -rf "$SEQ"; box_mark "$SEQ/STARTED"; } \
        || box_finish "SEQUENCES_UNWRITABLE ($SEQ)" 1
    from=$(box_size "$OUT/preencode.log")
    box_bounded --stall "$OUT/preencode.log" "$PREENCODE_STALL_MIN" preencode "$PREENCODE_CUT_MIN" preencode.log \
        python tools/preencode_sequences.py --dataset "$CORPUS" --out "$SEQ" --workers "$WORKERS" \
        --vocab-from "$INIT_PT"
    tail -c "+$(( from + 1 ))" "$OUT/preencode.log" | grep -q "SEQUENCES_DONE" \
        || box_finish "PREENCODE_${BOX_WHY^^} rc=$BOX_RC (preencode.log)" 1
    box_mark "$SEQ/SEQUENCES_DONE"
fi

# ---- round 2's own sequences: from HF when an earlier session sent them, else built and sent
if ! box_marked_this_stage "$OWN/OWN_DONE"; then
    box_bounded own-restore 30 staging.log python - "$HF_DIR/own_sequences_r2.tar.gz" "$WORKDIR" <<'EOF'
import sys, tarfile
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
try:
    path = hf_hub_download("momom2/wesnoth-model-checkpoints", sys.argv[1])
except EntryNotFoundError:
    print("no cached own sequences on HF", flush=True)
    sys.exit(3)
with tarfile.open(path, "r:gz") as tf:
    if hasattr(tarfile, "data_filter"):
        tf.extractall(sys.argv[2], filter="data")
    else:
        tf.extractall(sys.argv[2])
print("own sequences from HF", flush=True)
EOF
    if [ "$BOX_RC" -eq 0 ] && [ -f "$OWN/sequence_manifest.json" ]; then
        box_mark "$OWN/OWN_DONE"
    else
        rm -rf "$OWN" "$CAND"
        # shellcheck disable=SC2086 # SOURCE_TARS is a list of HF paths
        box_bounded own-inputs 30 staging.log python - "$OWN_GAMES" $SOURCE_TARS <<'EOF' \
            || box_finish "OWN_INPUTS_FAILED (staging.log)" 1
import sys, tarfile
from huggingface_hub import hf_hub_download
for path in sys.argv[2:]:
    with tarfile.open(hf_hub_download("momom2/wesnoth-model-checkpoints", path), "r:gz") as tf:
        tf.extractall(sys.argv[1])
        print(path, sum(1 for name in tf.getnames() if name.endswith(".game.jsonl.gz")), "game records", flush=True)
EOF
        from=$(box_size "$OUT/candidates.log")
        box_bounded --stall "$OUT/candidates.log" "$CAND_STALL_MIN" candidates "$CAND_CUT_MIN" candidates.log \
            python tools/own_trajectories.py --games "$OWN_GAMES"/games_* --teacher "$INIT_PT" \
            --player-sha256 "$INIT_SHA256" --slots 64 --out "$CAND" --workers "$WORKERS"
        tail -c "+$(( from + 1 ))" "$OUT/candidates.log" | grep -q "CANDIDATES_DONE" \
            || box_finish "CANDIDATES_${BOX_WHY^^} rc=$BOX_RC (candidates.log)" 1
        box_gpu_ok || box_finish "GPU_UNRESPONSIVE before the targets: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
        from=$(box_size "$OUT/targets.log")
        box_bounded --stall "$OUT/targets.log" "$TARGETS_STALL_MIN" targets "$TARGETS_CUT_MIN" targets.log \
            python tools/own_targets.py --candidates "$CAND" --teacher "$INIT_PT" --out "$OWN" --device cuda
        tail -c "+$(( from + 1 ))" "$OUT/targets.log" | grep -q "OWN_TARGETS_DONE" \
            || box_finish "TARGETS_${BOX_WHY^^} rc=$BOX_RC (targets.log)" 1
        box_mark "$OWN/OWN_DONE"
        if [ "$CACHE_OWN" = 1 ]; then
            box_upload_dir own_sequences_r2 "$OWN"
            box_upload
        fi
    fi
fi
cp -f "$OWN/sequence_manifest.json" "$OUT/own_sequence_manifest.json" || box_finish "OWN_MANIFEST_MISSING" 1

# ---- the arms
TRAIN_ARGS=(--sequences "$SEQ" --dataset "$CORPUS" --own-sequences "$OWN" --seed "$RUN_SEED" --device cuda
            --lr "$LR" --warmup-steps 0 --decay-from "$DECAY_FROM" --pass-positions "$PASS_POSITIONS"
            --probe-every "$PROBE_EVERY" --barrier-positions 1000000 --probe-ks 0,64 --probe-at-start
            --init-from "$INIT_PT" --init-other-records)
arm() {                          # arm NAME REPO WARMUP WINDOWS PROFILE BENCH_FLAGS TRAIN_FLAGS...: one bench run, once
    local name=$1 repo=$2 warmup=$3 windows=$4 profile=$5 bench_flags=$6
    shift 6
    [ ! -f "$OUT/arm_$name.done" ] || return 0
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before arm $name: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    rm -f "$OUT/arm_$name.jsonl" "$OUT/arm_$name.steps.jsonl" "$OUT/arm_$name.smi.csv"
    cd "$repo" || box_finish "NO_REPO $repo" 1
    # shellcheck disable=SC2086 # bench_flags is a list of flags
    box_bounded "arm $name" "$ARM_CUT_MIN" "arm_$name.log" python "$BOX_REPO/tools/bench_sequence_step.py" run \
        --out "$OUT/arm_$name.jsonl" --repo "$repo" --warmup "$warmup" --windows "$windows" \
        --profile-windows "$profile" $bench_flags -- "${TRAIN_ARGS[@]}" --out "$OUT/arm_$name.pt" "$@"
    cd "$BOX_REPO" || box_finish "NO_REPO $BOX_REPO" 1
    [ "$BOX_RC" -eq 0 ] || box_finish "ARM_FAILED $name rc=$BOX_RC $BOX_WHY (arm_$name.log) $(notes)" 1
    box_mark "$OUT/arm_$name.done"
}
window_stat() {                  # window_stat ARM STAT: over the arm's timed windows, the max peak or the measured bytes per kept token
    timeout 1m python - "$OUT/arm_$1.jsonl" "$2" <<'EOF'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
win = [r for r in rows if r["kind"] == "window"]
if sys.argv[2] == "peak":
    print(max(r["peak_gb"] for r in win))
else:
    kept = [r for r in win if r.get("kept_steps")]
    tokens = sum(r["kept_tokens"] for r in kept)
    print(round(sum(r["kept_gb"] for r in kept) * 2 ** 30 / tokens) if tokens else "none")
EOF
}
budget_for() {                   # budget_for ARM: the GiB of activations that fit the arm's worst window under PEAK_GB
    timeout 1m python -c "import sys; b = $PEAK_GB - $MARGIN_GB - float(sys.argv[1]); print(round(b, 2) if b > 0 else 0)" \
        "$(window_stat "$1" peak)"
}
budget_arms() {                  # budget_arms CAL NAME BASE_ARM CAL_GB TRAIN_FLAGS...: calibrate the bytes per token, then time the budget
    local cal=$1 name=$2 base=$3 cal_gb=$4 bpt budget
    shift 4
    arm "$cal" "$BOX_REPO" "$CAL_WARMUP" "$CAL_WINDOWS" 0 "" "$@" --activation-budget-gb "$cal_gb"
    bpt=$(window_stat "$cal" bpt)
    budget=$(budget_for "$base")
    echo "$name: $bpt bytes per kept token measured, budget $budget GiB" | tee -a "$OUT/budgets.txt"
    if [[ ! $bpt =~ ^[0-9]+$ ]] || [ "$budget" = 0 ]; then
        echo "$name: no budget to time" | tee -a "$OUT/budgets.txt"
        return 1
    fi
    # Five percent over the measurement: a token kept beyond the estimate costs memory, not numbers.
    arm "$name" "$BOX_REPO" "$WARMUP" "$WINDOWS" "$PROFILE_WINDOWS" "" "$@" --activation-budget-gb "$budget" \
        --activation-bytes-per-token "$(( bpt * 105 / 100 ))"
}

if [ ! -f "$OUT/gradcheck.json" ]; then
    box_bounded gradcheck 20 gradcheck.log python tools/bench_sequence_step.py gradcheck \
        --out "$OUT/gradcheck.json.pending" -- "${TRAIN_ARGS[@]}" --out "$OUT/gradcheck_arm.pt"
    case $BOX_RC in
        0|8) mv -f "$OUT/gradcheck.json.pending" "$OUT/gradcheck.json" ;;
        *) box_finish "GRADCHECK_${BOX_WHY^^} rc=$BOX_RC (gradcheck.log)" 1 ;;
    esac
    box_upload
fi
PACKED_OK=$(timeout 1m python -c "import json, sys; print(int(json.load(open(sys.argv[1]))['pass']))" "$OUT/gradcheck.json")

arm base_1 "$BASE_REPO" "$WARMUP" "$WINDOWS" "$PROFILE_WINDOWS" ""
arm this_1 "$BOX_REPO" "$WARMUP" "$WINDOWS" "$PROFILE_WINDOWS" ""
arm base_2 "$BASE_REPO" "$WARMUP" "$WINDOWS" "$PROFILE_WINDOWS" ""
arm this_2 "$BOX_REPO" "$WARMUP" "$WINDOWS" "$PROFILE_WINDOWS" ""
box_upload
FASTEST=this_1
FASTEST_FLAGS=()
if budget_arms cal_padded budget this_1 "$CAL_PADDED_GB"; then
    FASTEST=budget
    FASTEST_FLAGS=(--activation-budget-gb "$(budget_for this_1)" \
                   --activation-bytes-per-token "$(( $(window_stat cal_padded bpt) * 105 / 100 ))")
fi
box_upload
if [ "$PACKED_OK" = 1 ]; then
    arm packed "$BOX_REPO" "$WARMUP" "$WINDOWS" "$PROFILE_WINDOWS" "" --packed-trunk
    FASTEST=packed
    FASTEST_FLAGS=(--packed-trunk)
    if budget_arms cal_packed packed_budget packed "$CAL_PACKED_GB" --packed-trunk; then
        FASTEST=packed_budget
        FASTEST_FLAGS=(--packed-trunk --activation-budget-gb "$(budget_for packed)" \
                       --activation-bytes-per-token "$(( $(window_stat cal_packed bpt) * 105 / 100 ))")
    fi
    box_upload
fi
arm preload "$BOX_REPO" "$WARMUP" "$WINDOWS" 0 "--preload" ${FASTEST_FLAGS[@]+"${FASTEST_FLAGS[@]}"}
echo "preload: $FASTEST's flags (${FASTEST_FLAGS[*]})" >> "$OUT/budgets.txt"

# ---- the comparisons and speed.json
compare() {                      # compare A B FIELD...: B's speedup over A, the fields allowed to differ
    local a=$1 b=$2
    shift 2
    [ -f "$OUT/arm_$a.done" ] && [ -f "$OUT/arm_$b.done" ] || return 0
    box_bounded "compare $a $b" 5 compare.log python tools/bench_sequence_step.py compare \
        "$OUT/arm_$a.jsonl" "$OUT/arm_$b.jsonl" --allow "$@" --out "$OUT/cmp_${a}_vs_$b.json"
}
compare base_1 base_2
compare this_1 this_2
compare base_1 this_1 code
compare base_2 this_2 code
compare this_1 budget activation_budget_gb activation_bytes_per_token
compare this_1 packed trunk
compare this_1 packed_budget trunk activation_budget_gb activation_bytes_per_token
compare base_1 "$FASTEST" code trunk activation_budget_gb activation_bytes_per_token
compare "$FASTEST" preload bench
box_bounded summary 5 compare.log python - "$OUT" "$VALID_SPREAD" "$PEAK_GB" <<'EOF'
import json, pathlib, sys
out, spread, peak_cap = pathlib.Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
arms, cmps = {}, {}
for p in sorted(out.glob("arm_*.jsonl")):
    if p.name.endswith(".steps.jsonl"):
        continue
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = next((r for r in rows if r["kind"] == "total"), None)
    win = [r for r in rows if r["kind"] == "window"]
    if total is None:
        continue
    peaks = [r["peak_gb"] for r in win if r.get("peak_gb") is not None]
    arm = arms[p.stem[4:]] = {"positions_per_second": total["positions_per_second"], "windows": total["windows"],
                              "peak_gb_max": max(peaks) if peaks else None,
                              "kept_steps_mean": sum(r.get("kept_steps") or 0 for r in win) / max(1, len(win))}
    for r in rows:
        if r["kind"] in ("gpu", "profile"):
            arm[r["kind"]] = r
for p in sorted(out.glob("cmp_*.json")):
    c = json.loads(p.read_text(encoding="utf-8"))
    cmps[p.stem[4:]] = {k: c[k] for k in ("speedup_total", "speedup_median", "speedup_ci95")}
base = cmps.get("base_1_vs_base_2")
summary = {"arms": arms, "comparisons": cmps,
           "valid": base is not None and abs(base["speedup_total"] - 1.0) <= spread,
           "budget_peaks_under_cap": {a: arms[a]["peak_gb_max"] <= peak_cap
                                      for a in ("budget", "packed_budget") if a in arms},
           "gradcheck": json.loads((out / "gradcheck.json").read_text(encoding="utf-8"))["pass"]}
(out / "speed.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print(json.dumps({k: summary[k] for k in ("valid", "budget_peaks_under_cap", "gradcheck")}),
      {a: v["positions_per_second"] for a, v in arms.items()}, flush=True)
EOF
SUMMARY=$(tail -n 1 "$OUT/compare.log" | cut -c1-400)
box_finish "TRAINER_SPEED_DONE $SUMMARY; cuda tests: ${CUDA_TESTS:-none}"
