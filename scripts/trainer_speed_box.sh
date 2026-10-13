#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# The sequence trainer's speed on one box, round 2 (docs/trainer_speed_20261013.md,
# "Round 2"): the first round's fastest configuration (the packed trunk with
# kept activations, the pickled records streamed) against this stage's
# changes, timed under round 2's training arguments
# (scripts/memory_own_trajectories_box.sh, ROUND=2: from own1, its own
# game-sides beside the corpus's) by tools/bench_sequence_step.py.
#
# Two code stages: STAGE, this branch's (tools/stage_code.py --script
# scripts/trainer_speed_box.sh), and BASE_STAGE, the first round's
# (feature/trainer-speed at e2d2877), staged from a checkout of it with
# tools/stage_code.py --out ... --upload tier-b/staging/<name>.tar.gz. The
# base stage is unpacked beside this one and its trainer timed with this
# stage's bench (--repo); both use this stage's Rust wheel (the work changes
# no Rust).
#
# Every arm runs the first round's fastest flags, FLAGS: the packed trunk and
# the activation budget its packed_budget arm timed (BUDGET_GB, with the bytes
# a kept token measured, BYTES_PER_TOKEN; training/metrics/
# trainer_speed_20261013/budgets.txt), so every arm keeps the same time steps.
#
# The steps:
#   the tests of this path on this wheel and GPU: the CPU-path tests gate the
#     run; the CUDA tests (tests/test_trainer_speed_cuda.py and the CUDA tests
#     of tests/test_sequence_columns.py) are recorded in cuda_tests.log and
#     named in the finish line;
#   own1 from HF, its SHA-256 checked; the corpus rebuilt at version 5 and
#     pre-encoded with own1's vocabularies;
#   round 2's own sequences: OWN_CACHE from HF (the first round's), else built
#     as round 2 built them (the confirmation's three matches and round 2's
#     generation match, own1's 64-slot sides, labelled by own1 at 0 slots)
#     and sent to HF_DIR as own_sequences_r2.tar.gz (CACHE_OWN=0: not sent);
#   the columns of the corpus's and the own sequences
#     (tools/columnize_sequences.py, every position checked against its pickle);
#   the arms, each WARMUP untimed windows, WINDOWS timed and PROFILE_WINDOWS
#     profiled (arm_NAME.jsonl, with the trainer's rows in
#     arm_NAME.steps.jsonl); an arm whose process fails after its total row
#     (the profile) keeps its timed windows:
#       base_1, this_1, base_2, this_2: the first round's code and this
#         stage's, alternated, first (this stage's launch changes, which keep
#         the arithmetic); the session is valid when the two base runs agree
#         within 3% (VALID_SPREAD), and one of them failing ends the run;
#       the window-batched forward's gradient check (gradcheck.json): its
#         gradients against the stepped forward's, both on the packed trunk,
#         a crash barrier; anything but a pass leaves the batched arms out;
#       columns: this stage reading the columnar records (--columns);
#       batched_pickles: --window-batched on the pickled records;
#       batched: --columns --window-batched, every change together;
#       preload: the fastest of columns and batched with every record read
#         first (what the columnar loader still costs);
#       this_3: this_1 again, after every flag arm; the flag arms'
#         comparisons with this_1 hold when this_1 and this_3 agree within
#         VALID_SPREAD;
#     a flag arm, the gradient check or this_3 that fails is recorded in
#     arms_failed.txt and the bench goes on without it;
#   compare (cmp_A_vs_B.json) and speed.json: positions a second per arm and
#     each change's speedup, median and 95% interval over the windows;
#   the records to HF, the instance stopped.
#
# Expected wall on a 4090 (the first round's walls.txt): bring-up and tests 15
# minutes, inputs 2, corpus and pre-encoding 7, the own sequences 5 from HF,
# the columns 5, the gradient check 2, nine arms of 2 to 4 minutes: about 1
# hour, $0.45 at $0.42 an hour; 2.2 hours and $0.95 if the own sequences must
# be built again. BOX_MAX_H 3 caps it at about $1.3 plus the final round.
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
HF_DIR="${HF_DIR:-tier-b/trainer_speed_2_20261014}"
export HF_DIR
BOX_MAX_H="${BOX_MAX_H:-3}"
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
OWN_CACHE="${OWN_CACHE:-tier-b/trainer_speed_20261013/own_sequences_r2.tar.gz}"
WARMUP="${WARMUP:-20}"
WINDOWS="${WINDOWS:-80}"
PROFILE_WINDOWS="${PROFILE_WINDOWS:-3}"
BUDGET_GB="${BUDGET_GB:-19.18}"                 # the first round's packed_budget arm
BYTES_PER_TOKEN="${BYTES_PER_TOKEN:-102139}"
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
COLUMNS_CUT_MIN="${COLUMNS_CUT_MIN:-40}"
COLUMNS_STALL_MIN="${COLUMNS_STALL_MIN:-10}"
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
ARMS="base_1 this_1 base_2 this_2 columns batched_pickles batched preload this_3"

notes() {                        # what is done, for the finish reason
    local name done_=()
    [ ! -f "$OUT/gradcheck.json" ] || done_+=(gradcheck)
    for name in $ARMS; do
        ! box_marked_this_stage "$OUT/arm_$name.done" || done_+=("$name")
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
        tests/test_sequence_speed.py tests/test_sequence_columns.py tests/test_sequence_launches.py \
        tests/test_own_loss.py tests/test_own_trajectories.py tests/test_memory_model.py \
        tests/test_packed_trunk.py tests/test_packed_trunk_cuda.py tests/test_packed_embed.py \
        tests/test_preencode_sequences.py tests/test_memory_learner.py \
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
        python tools/build_imitation_dataset.py --raw-root "$RAW" --out "$CORPUS" --workers "$WORKERS" \
        --dispositions "$RAW/training/logs/replay_dispositions.jsonl.gz"
    tail -c "+$(( from + 1 ))" "$OUT/corpus_build.log" | grep -q "BUILD_DONE" \
        || box_finish "CORPUS_${BOX_WHY^^} rc=$BOX_RC (corpus_build.log)" 1
    box_bounded corpus-check 10 corpus_build.log \
        python tools/build_imitation_dataset.py --out "$CORPUS" --check "$OUT/corpus_summary.json" \
        --dispositions "$RAW/training/logs/replay_dispositions.jsonl.gz" \
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
    box_bounded own-restore 30 staging.log python - "$OWN_CACHE" "$WORKDIR" <<'EOF'
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

# ---- the columns of the corpus's and the own sequences, every position checked against its pickle
for dir in "$SEQ" "$OWN"; do
    name=$(basename "$dir")
    if ! box_marked_this_stage "$dir/COLUMNS_DONE"; then
        from=$(box_size "$OUT/columns.log")
        box_bounded --stall "$OUT/columns.log" "$COLUMNS_STALL_MIN" "columns $name" "$COLUMNS_CUT_MIN" \
            columns.log python tools/columnize_sequences.py --sequences "$dir" --workers "$WORKERS"
        case $BOX_RC in
            0) tail -c "+$(( from + 1 ))" "$OUT/columns.log" | grep -q "COLUMNS_DONE" \
                   || box_finish "COLUMNS_UNFINISHED $name (columns.log)" 1 ;;
            2) box_finish "COLUMNS_MISMATCH $name: games whose columns differ from their pickles (columns.log)" 1 ;;
            *) box_finish "COLUMNS_${BOX_WHY^^} rc=$BOX_RC $name (columns.log)" 1 ;;
        esac
        box_mark "$dir/COLUMNS_DONE"
    fi
    timeout 1m python -c "import json, sys; m = json.load(open(sys.argv[1])); m.pop('games'); print(json.dumps(m))" \
        "$dir/columns_manifest.json" > "$OUT/columns_$name.json" || box_finish "COLUMNS_MANIFEST_MISSING $name" 1
done
box_upload

# ---- the arms (begin): tests/test_trainer_speed_box.py runs this block with the library stubbed
TRAIN_ARGS=(--sequences "$SEQ" --dataset "$CORPUS" --own-sequences "$OWN" --seed "$RUN_SEED" --device cuda
            --lr "$LR" --warmup-steps 0 --decay-from "$DECAY_FROM" --pass-positions "$PASS_POSITIONS"
            --probe-every "$PROBE_EVERY" --barrier-positions 1000000 --probe-ks "0,64" --probe-at-start
            --init-from "$INIT_PT" --init-other-records)
FLAGS=(--packed-trunk --activation-budget-gb "$BUDGET_GB" --activation-bytes-per-token "$BYTES_PER_TOKEN")
arm() {                          # arm NAME REPO BENCH_FLAGS TRAIN_FLAGS...: one bench run of FLAGS and TRAIN_FLAGS, once; false when it failed
    local name=$1 repo=$2 bench_flags=$3
    shift 3
    ! box_marked_this_stage "$OUT/arm_$name.done" || return 0
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before arm $name: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    rm -f "$OUT/arm_$name.jsonl" "$OUT/arm_$name.steps.jsonl" "$OUT/arm_$name.smi.csv"
    cd "$repo" || box_finish "NO_REPO $repo" 1
    # shellcheck disable=SC2086 # bench_flags is a list of flags
    box_bounded "arm $name" "$ARM_CUT_MIN" "arm_$name.log" python "$BOX_REPO/tools/bench_sequence_step.py" run \
        --out "$OUT/arm_$name.jsonl" --repo "$repo" --warmup "$WARMUP" --windows "$WINDOWS" \
        --profile-windows "$PROFILE_WINDOWS" $bench_flags -- "${TRAIN_ARGS[@]}" --out "$OUT/arm_$name.pt" \
        "${FLAGS[@]}" "$@"
    cd "$BOX_REPO" || box_finish "NO_REPO $BOX_REPO" 1
    if [ "$BOX_RC" -ne 0 ]; then
        if ! grep -q '"kind": "total"' "$OUT/arm_$name.jsonl" 2>/dev/null; then
            echo "ARM_FAILED $name rc=$BOX_RC $BOX_WHY (arm_$name.log)" | tee -a "$OUT/arms_failed.txt"
            return 1
        fi
        # The timed windows and their total are written: what failed came after them (the profile).
        echo "AFTER_TOTAL_FAILED $name rc=$BOX_RC $BOX_WHY (arm_$name.log)" | tee -a "$OUT/arms_failed.txt"
    fi
    box_mark "$OUT/arm_$name.done"
}
arm_rate() {                     # arm_rate ARM: its positions a second, 0 when it has no total
    timeout 1m python - "$OUT/arm_$1.jsonl" <<'EOF'
import json, sys
try:
    rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
    print(next(r["positions_per_second"] for r in rows if r["kind"] == "total"))
except (OSError, StopIteration, ValueError):
    print(0)
EOF
}
gradcheck_failed() {             # gradcheck_failed: a check that did not finish passes nothing
    echo "GRADCHECK_FAILED gradcheck rc=$BOX_RC $BOX_WHY (gradcheck.log)" | tee -a "$OUT/arms_failed.txt"
    printf '{"pass": false, "rc": %d, "why": "%s", "log": "gradcheck.log"}\n' "$BOX_RC" "$BOX_WHY" \
        > "$OUT/gradcheck.json"
}

# The base and this stage, alternated: the comparison the session is for, before anything
# that could end it. A later arm that fails (out of memory, say) is recorded in
# arms_failed.txt and the bench goes on without it.
for name in base_1 this_1 base_2 this_2; do
    repo=$BOX_REPO
    [ "${name%_*}" = this ] || repo=$BASE_REPO
    arm "$name" "$repo" "" || box_finish "ARM_FAILED $name (arm_$name.log) $(notes)" 1
done
box_upload
# The window-batched forward's gradient check, a crash barrier: anything but a pass (exit 0)
# leaves the batched arms out, a check that did not finish (an exception, a cut) included.
if [ ! -f "$OUT/gradcheck.json" ]; then
    rm -f "$OUT/gradcheck.json.pending"
    box_bounded gradcheck 20 gradcheck.log python tools/bench_sequence_step.py gradcheck \
        --out "$OUT/gradcheck.json.pending" --reference=--packed-trunk \
        "--candidate=--packed-trunk --window-batched" -- "${TRAIN_ARGS[@]}" --out "$OUT/gradcheck_arm.pt"
    case $BOX_RC in
        0|8) mv -f "$OUT/gradcheck.json.pending" "$OUT/gradcheck.json" || gradcheck_failed ;;
        *) gradcheck_failed ;;
    esac
    box_upload
fi
BATCHED_OK=$(timeout 1m python -c "import json, sys; print(int(json.load(open(sys.argv[1]))['pass']))" \
    "$OUT/gradcheck.json")
FASTEST=this_1
FASTEST_FLAGS=()
if arm columns "$BOX_REPO" "" --columns; then
    FASTEST=columns
    FASTEST_FLAGS=(--columns)
fi
if [ "$BATCHED_OK" = 1 ]; then
    arm batched_pickles "$BOX_REPO" "" --window-batched || true
    if arm batched "$BOX_REPO" "" --columns --window-batched \
            && timeout 1m python -c "import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)" \
               "$(arm_rate batched)" "$(arm_rate "$FASTEST")"; then
        FASTEST=batched
        FASTEST_FLAGS=(--columns --window-batched)
    fi
fi
box_upload
if [ "$FASTEST" != this_1 ]; then
    arm preload "$BOX_REPO" "--preload" "${FASTEST_FLAGS[@]}" \
        || echo "the preload arm failed: no reading of the columnar loader's cost" >> "$OUT/budgets.txt"
    echo "preload: $FASTEST's flags (${FASTEST_FLAGS[*]})" >> "$OUT/budgets.txt"
fi
# this_1 again after every flag arm: their comparisons with this_1 hold when the two agree.
arm this_3 "$BOX_REPO" "" || echo "the last arm failed: the flag arms' drift is not read" >> "$OUT/budgets.txt"

# ---- the comparisons and speed.json
compare() {                      # compare A B FIELD...: B's speedup over A, the fields allowed to differ
    local a=$1 b=$2
    shift 2
    if [ ! -f "$OUT/arm_$a.done" ] || [ ! -f "$OUT/arm_$b.done" ]; then
        return 0
    fi
    box_bounded "compare $a $b" 5 compare.log python tools/bench_sequence_step.py compare \
        "$OUT/arm_$a.jsonl" "$OUT/arm_$b.jsonl" --allow "$@" --out "$OUT/cmp_${a}_vs_$b.json"
}
compare base_1 base_2
compare this_1 this_2
compare this_1 this_3
compare base_1 this_1 code
compare base_2 this_2 code
compare this_1 columns columns
compare this_1 batched_pickles window_batched
compare columns batched window_batched
compare base_1 batched code columns window_batched
compare "$FASTEST" preload bench
box_bounded summary 5 compare.log python - "$OUT" "$VALID_SPREAD" <<'EOF'
import json, pathlib, sys
out, spread = pathlib.Path(sys.argv[1]), float(sys.argv[2])
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

    def mean(key):
        vals = [r[key] for r in win if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    arm = arms[p.stem[4:]] = {"positions_per_second": total["positions_per_second"], "windows": total["windows"],
                              "peak_gb_max": max(peaks) if peaks else None,
                              "kept_steps_mean": sum(r.get("kept_steps") or 0 for r in win) / max(1, len(win)),
                              "gc_seconds_mean": mean("gc_seconds"), "decode_seconds_mean": mean("decode_seconds"),
                              "loader_wait_seconds_mean": mean("loader_wait_seconds")}
    for r in rows:
        if r["kind"] in ("gpu", "profile", "preload"):
            arm[r["kind"]] = r
for p in sorted(out.glob("cmp_*.json")):
    c = json.loads(p.read_text(encoding="utf-8"))
    cmps[p.stem[4:]] = {k: c[k] for k in ("speedup_total", "speedup_median", "speedup_ci95")}


def agree(name):
    return name in cmps and abs(cmps[name]["speedup_total"] - 1.0) <= spread


check = out / "gradcheck.json"
summary = {"arms": arms, "comparisons": cmps,
           # base against this holds when the two base runs agree; the flag arms against
           # this_1 when this_1 and this_3, which brackets them, agree.
           "valid": agree("base_1_vs_base_2"), "flags_valid": agree("this_1_vs_this_3"),
           "gradcheck": json.loads(check.read_text(encoding="utf-8"))["pass"] if check.exists() else None}
(out / "speed.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print(json.dumps({k: summary[k] for k in ("valid", "flags_valid", "gradcheck")}),
      {a: v["positions_per_second"] for a, v in arms.items()}, flush=True)
EOF
SUMMARY=$(tail -n 1 "$OUT/compare.log" | cut -c1-400)
FAILED_STEPS=$(awk '{print $2 "(" $1 ")"}' "$OUT/arms_failed.txt" 2>/dev/null | tr '\n' ' ')
box_finish "TRAINER_SPEED_DONE $SUMMARY; cuda tests: ${CUDA_TESTS:-none}; failed: ${FAILED_STEPS:-none}"
# ---- the arms (end)
