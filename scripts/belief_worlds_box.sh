#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# The rollout re-grade in belief worlds (docs/belief_worlds_prereg_20261012.md)
# on one box:
#   the tests of the worlds, the re-grade and the paths its rollouts use, on
#   the wheel built from the stage;
#   the inputs, as the re-grade's box took them (scripts/rollout_regrade_box.sh):
#     own2 from HF (OWN2_HF, its SHA-256 checked against OWN2_SHA256); step
#     1's rebuild.jsonl from bench.tar.gz, which must equal the committed copy;
#     the benchmark's records (validation.json) and the corpus it was played
#     from;
#   the six arms (tools/rollout_regrade.py --world, wesnoth_ai/belief_worlds.py)
#     in the order TU, O2, RB, RU, LS, T: 8 rollouts per benchmark candidate,
#     both sides own2 at 0 slots at raw:t0.5+eo-1.5, reads 2 and 4, the dice
#     of the re-grade's salts in every arm, each arm's records written per
#     candidate as produced (rollouts_<world>.jsonl); an arm that leaves a
#     candidate incomplete is resumed once, which plays only the candidates
#     without a complete record;
#   the readout (tools/belief_readout.py): per arm the selection gain of the
#     HP margin at read 2, the paired differences TU-O2, T-TU, RB-RU, RU-O2
#     and LS-O2, and the readings (READING_REFUSED when an arm lacks a
#     complete record for a candidate).
#
# Box: an RTX 4090 with at least 32 usable cores (docs/box_specs.md).
# Expected wall about 75 minutes: bring-up and tests 15 (as the 2026-10-12
# re-grade), inputs 3 (about 480 MB), the six arms 55 (the re-grade's arm
# took 540 s on 28 workers, 2026-10-12, training/metrics/rollout_regrade_20261012;
# RB adds one CPU forward per candidate on each worker's own copy of own2),
# the readout 1, the uploads 3. Identical repeats have differed by 1.8x,
# which would bring the arms to 1.7 h. Cost at $0.37-0.46 an hour: about
# $0.5 expected; BOX_MAX_H 2.5 caps it at $0.93-1.15 plus the final round.
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md): the
# onstart of scripts/rent_box.py fetches the library of STAGE, then this
# script. Every step has a bound; the dead-man's switch finishes the entry
# after BOX_MAX_H hours whatever it is doing. Records go to HF $HF_DIR every
# 30 minutes and after each arm. Re-entry, on this machine or a new one
# (files absent here come back from HF), resumes each arm from its
# rollouts file and skips a readout already made. A run belongs to the code
# stage that began it (RUN_STAGE): another stage continues it only with
# RESUME_OTHER_STAGE=1. Every exit past that check, clean or not, uploads the
# records with ALL_DONE last and stops the instance.
# Never `set -x`: the HF token and the instance key are in the environment.
# box-needs: disk_gb=40 ram_gb=64 gpu_ram_gb=24 cores=32 gpu=4090
set -uo pipefail
WORKDIR=/workspace
OUT=$WORKDIR/beliefworlds
STAGE="${STAGE:-}"
OWN2_HF="${OWN2_HF:-tier-b/memory_own_round2_20261011/own.pt}"
OWN2_SHA256="${OWN2_SHA256:-aea027b2696a9fbce0ebcdc0a8483c9df7d35957934b81b0c62093a02cdd9129}"
BENCH_TAR="${BENCH_TAR:-tier-b/critic_step1_20261009/bench.tar.gz}"
BENCH_HF="${BENCH_HF:-tier-b/turn_value_20260925}"
BENCH_CORPUS_TAR="${BENCH_CORPUS_TAR:-tier-b/replays_dataset_imitation_dedup_20260908.tar.gz}"
WORKERS="${WORKERS:-}"                           # default: box_workers less 3 (the server, the parent, the uploads), at most 48
export HF_DIR="${HF_DIR:-tier-b/belief_worlds_20261012}"
# Bounds in minutes, from the estimates above.
TESTS_CUT_MIN="${TESTS_CUT_MIN:-20}"
INPUTS_CUT_MIN="${INPUTS_CUT_MIN:-20}"
ARM_CUT_MIN="${ARM_CUT_MIN:-30}"                 # estimated 9 an arm
ARM_STALL_MIN="${ARM_STALL_MIN:-10}"             # the re-grade logs every 20 candidates
BOX_MAX_H="${BOX_MAX_H:-2.5}"
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }

INPUTS=$WORKDIR/regrade_inputs                   # own2.pt, rebuild.jsonl, validation.json, the corpus
OWN2_PT=$INPUTS/own2.pt
STATIC=training/metrics/critic_step1_20261009/rebuild.jsonl    # in the staged repository
ARMS=(TU:true_units O2:none RB:roster_belief RU:roster_uniform LS:last_seen T:true)

reading_of() {                   # the readout's printed reading, or why there is none
    grep -ho "READING_REFUSED.*\|READING .*" "$OUT/readout.log" 2>/dev/null | tail -1 | cut -c1-300
}
arm_note() {                     # arm_note LABEL: the arm's last REGRADE_ line
    grep -ho "REGRADE_[A-Z]*" "$OUT/regrade_$1.log" 2>/dev/null | tail -1
}
notes() {
    local entry label line="arms:"
    for entry in "${ARMS[@]}"; do
        label=${entry%%:*}
        line+=" $label $(arm_note "$label")"
    done
    echo "$line; readout: $(reading_of)"
}
# shellcheck disable=SC2317 # called by the library, in the reason of an unexpected exit
box_notes() { notes; }
box_on_round() {                 # progress.txt, before each upload round
    local entry world
    { date -u
      notes
      for entry in "${ARMS[@]}"; do
          world=${entry#*:}
          [ ! -f "$OUT/rollouts_$world.jsonl" ] \
              || echo "rollouts_$world.jsonl: $(wc -l < "$OUT/rollouts_$world.jsonl") candidate records"
      done
      tail -n 2 "$OUT"/*.log 2>/dev/null | tail -n 8
    } > "$OUT/progress.txt.tmp" && mv -f "$OUT/progress.txt.tmp" "$OUT/progress.txt"
}

box_init
[ -n "$STAGE" ] || box_finish "NO_STAGE: build the code stage (tools/stage_code.py) and pass STAGE" 1
box_bind_run_stage

# ---- the records of an earlier entry
restored=(belief_readout.json belief_readout.md belief_readout_refused.json readout.log)
for entry in "${ARMS[@]}"; do
    restored+=("rollouts_${entry#*:}.jsonl" "regrade_summary_${entry#*:}.json" "regrade_${entry%%:*}.log")
done
box_restore "${restored[@]}" || box_finish "RESTORE_FAILED (restore.log)" 1
box_pip huggingface_hub psutil pytest scipy requests || echo "pip install failed (pip.log)"

# ---- the code and the Rust wheel
box_stage_code || box_finish "CODE_STAGING_FAILED (staging.log)" 1
cd "$BOX_REPO" || box_finish "CODE_STAGING_FAILED (no $BOX_REPO)" 1
box_build_wheel || box_finish "BUILD_FAILED (build.log)" 1
box_facts > "$OUT/box.txt.tmp" 2>&1
mv -f "$OUT/box.txt.tmp" "$OUT/box.txt"
timeout 2m python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || box_finish "NO_CUDA (box.txt)" 1
if [ -z "$WORKERS" ]; then
    WORKERS=$(( $(box_workers) - 3 ))
    (( WORKERS <= 48 )) || WORKERS=48
    (( WORKERS >= 4 )) || WORKERS=4
fi
echo "re-grade workers: $WORKERS" >> "$OUT/box.txt"
box_upload_async
box_monitor_start

# ---- the crash barrier: the worlds', the re-grade's and its rollouts' paths' tests, on this wheel
if ! box_marked_this_stage "$BOX_STATE/TESTED"; then
    : > "$OUT/tests.log"
    box_bounded tests "$TESTS_CUT_MIN" tests.log python -m pytest tests/test_belief_worlds.py \
        tests/test_rollout_regrade.py tests/test_critic_bench.py tests/test_turn_bench_stats.py \
        tests/test_lookahead_player.py tests/test_game_core.py tests/test_vision.py tests/test_parity_integration.py \
        tests/test_raw_player.py tests/test_eval_inference_server.py -q -p no:cacheprovider -m ""
    tail -n 3 "$OUT/tests.log"
    [ "$BOX_RC" -eq 0 ] || box_finish "TESTS_FAILED rc=$BOX_RC (tests.log)" 1
    box_mark "$BOX_STATE/TESTED"
fi

# ---- the inputs
if ! box_marked_this_stage "$BOX_STATE/INPUTS_DONE"; then
    box_bounded inputs "$INPUTS_CUT_MIN" staging.log python - "$INPUTS" "$OWN2_HF" "$BENCH_TAR" "$BENCH_HF" \
        "$BENCH_CORPUS_TAR" <<'EOF' || box_finish "INPUTS_FAILED rc=$BOX_RC (staging.log)" 1
import pathlib, shutil, sys, tarfile
from huggingface_hub import hf_hub_download
REPO = "momom2/wesnoth-model-checkpoints"
inputs, own2, bench_tar, bench_hf, corpus_tar = sys.argv[1:]
inputs = pathlib.Path(inputs)
inputs.mkdir(parents=True, exist_ok=True)
shutil.copyfile(hf_hub_download(REPO, own2), inputs / "own2.pt")
with tarfile.open(hf_hub_download(REPO, bench_tar), "r:gz") as tf:
    member = tf.extractfile("bench/rebuild.jsonl")
    (inputs / "rebuild.jsonl").write_bytes(member.read())
shutil.copyfile(hf_hub_download(REPO, f"{bench_hf}/validation.json"), inputs / "validation.json")
shutil.rmtree(inputs / "replays_dataset_imitation", ignore_errors=True)
with tarfile.open(hf_hub_download(REPO, corpus_tar), "r:gz") as tf:
    tf.extractall(inputs)
assert (inputs / "replays_dataset_imitation" / "manifest.jsonl").is_file(), "no manifest in the benchmark's corpus"
print("inputs ready", flush=True)
EOF
    echo "$OWN2_SHA256  $OWN2_PT" | sha256sum -c --status || box_finish "OWN2_SHA256_MISMATCH ($OWN2_HF)" 1
    cmp -s "$INPUTS/rebuild.jsonl" "$STATIC" \
        || box_finish "STEP1_REBUILD_DIFFERS: $BENCH_TAR's rebuild.jsonl is not the committed $STATIC" 1
    box_mark "$BOX_STATE/INPUTS_DONE"
fi

# ---- the six arms, each resumed once when it leaves a candidate incomplete
arm_run() {                      # arm_run LABEL WORLD: one attempt, resumed from the arm's records
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before arm $1: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    box_bounded --stall "$OUT/regrade_$1.log" "$ARM_STALL_MIN" "arm $1" "$ARM_CUT_MIN" "regrade_$1.log" \
        python tools/rollout_regrade.py --world "$2" --records "$INPUTS/validation.json" \
        --dataset "$INPUTS/replays_dataset_imitation" --checkpoint "$OWN2_PT" --step1 "$INPUTS/rebuild.jsonl" \
        --out "$OUT" --workers "$WORKERS" --device cuda
}
for entry in "${ARMS[@]}"; do
    label=${entry%%:*}
    world=${entry#*:}
    if ! grep -q "REGRADE_DONE" "$OUT/regrade_$label.log" 2>/dev/null; then
        arm_run "$label" "$world"
        [ "$BOX_RC" -eq 0 ] || arm_run "$label" "$world"
        box_upload_async
    fi
done
ARMS_NOTE=$(notes | cut -d';' -f1)

# ---- the readout
if [ ! -f "$OUT/belief_readout.json" ]; then
    : > "$OUT/readout.log"
    box_bounded readout 15 readout.log python tools/belief_readout.py --dir "$OUT" --out "$OUT"
    box_upload_async
fi
READING=$(reading_of)
box_on_round
[[ $READING == "READING "* ]] \
    || box_finish "NO_READING: ${READING:-the readout printed none} (readout.log); $ARMS_NOTE" 1
box_finish "BELIEF_WORLDS_DONE $ARMS_NOTE; $READING"
