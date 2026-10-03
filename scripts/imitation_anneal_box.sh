#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# The reference's imitation training continued under the anneal rule
# (docs/imitation_anneal_prereg_20261003.md), on one box:
#   the tests of the trainer, the rule and memory play, on the wheel built
#     from the stage;
#   the corpus rebuilt at version 5 from the raw replays (RAW_TAR) and
#     pre-encoded, as for the parity-memory passes;
#   the law's points of the two earlier passes, replayed from their settings
#     (training/metrics/imitation_anneal_20261003/lineage.json, tools/lr_law.py),
#     and pass 2's checkpoint where its lowering began;
#   hold passes at the peak rate under the anneal rule (tools/sequence_train.py
#     --anneal-rule THRESHOLD), each a new pass from the previous one's end, at
#     most HOLD_PASSES: a pass that finishes (exit 0) hands over to the next;
#     one the rule stops (exit 6) is where the holding ends, and so is one whose
#     latest probes left the law (exit 7), which the finish reason flags;
#   the lowering: a straight line from the peak rate to 0 over LOWER_POSITIONS,
#     from where the holding ended; its final probe is the candidate's holdout
#     cross-entropy;
#   one match, PURE, 800 decisive games: the candidate at 64 slots against the
#     reference (seed base MATCH_SEED_BASE, 90000 by default).
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md): the
# onstart of scripts/rent_box.py fetches the library of STAGE, then this
# script. Every step has a bound; the dead-man's switch finishes the entry
# after BOX_MAX_H hours whatever it is doing. Records go to HF $HF_DIR every
# 30 minutes and at the end, the match's games as one tarball. Re-entry, on
# this machine or a new one (files absent here come back from HF), skips
# finished passes and continues the running one from its checkpoint; a new
# machine rebuilds the corpus and the sequences, which are deterministic.
# Every exit past the stage check uploads the records with ALL_DONE last and
# stops the instance.
# Never `set -x`: the HF token and the instance key are in the environment.
# box-needs: disk_gb=120 ram_gb=64 gpu_ram_gb=24 cores=32 gpu=4090
set -uo pipefail
WORKDIR=/workspace
OUT=$WORKDIR/imitationanneal
SEQ=$WORKDIR/sequences
STAGE="${STAGE:-}"
RAW_TAR="${RAW_TAR:-tier-b/corpus_v3/raw_corpus_20260929.tar}"
LINEAGE=training/metrics/imitation_anneal_20261003/lineage.json
RUN_SEED="${RUN_SEED:-20261003}"                 # hold pass K draws its order from RUN_SEED + K
THRESHOLD="${THRESHOLD:-0.03}"                   # the anneal rule: the loss an epoch at the peak must still gain
HOLD_PASSES="${HOLD_PASSES:-2}"                  # the spending cap: hold passes at most
LOWER_POSITIONS="${LOWER_POSITIONS:-2017864}"    # the lowering: half an epoch
MATCH_SEED_BASE="${MATCH_SEED_BASE:-90000}"      # disjoint from every earlier match
WORKERS="${WORKERS:-}"                           # default: box_workers (cores, memory-bounded), after box_init
GAMES="${GAMES:-800}"
JOBS="${JOBS:-20}"
export HF_DIR="${HF_DIR:-tier-b/imitation_anneal_20261003}"
# Bounds in minutes, from the pre-registration's "Cost".
BUILD_CUT_MIN="${BUILD_CUT_MIN:-60}"             # estimated 10 on 32 cores
BUILD_STALL_MIN="${BUILD_STALL_MIN:-15}"         # the builder logs every 1,000 candidates
PREENCODE_CUT_MIN="${PREENCODE_CUT_MIN:-150}"    # 6 on pass 2's box
PREENCODE_STALL_MIN="${PREENCODE_STALL_MIN:-20}" # the pre-encoder logs every 200 games
TRAIN_CUT_MIN="${TRAIN_CUT_MIN:-540}"            # a pass: 336 minutes on pass 2's box
TRAIN_STALL_MIN="${TRAIN_STALL_MIN:-40}"         # the trainer logs every minute; a probe runs silent
MATCH_CUT_MIN="${MATCH_CUT_MIN:-90}"             # the match stops itself at 60 (--time-budget-min), a game at 20 more
BOX_MAX_H="${BOX_MAX_H:-20}"                     # 1.37 times the 14.6 box-hours estimated
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }

CORPUS=replays_dataset_imitation
START_PT=$WORKDIR/start.pt                       # outside OUT: it does not go up again
POINTS=$OUT/law_points.jsonl
ARM=training/checkpoints/anneal_candidate.pt
MATCH=cand64_vs_ref

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
notes() {                        # where the training stands, for the finish reason
    local f line=""
    for f in "$OUT"/train_*.log; do
        [ -f "$f" ] || continue
        line="${f##*/train_}: $(grep -o "positions [0-9]*/[0-9]* steps [0-9]*" "$f" | tail -1)"
    done
    echo "training: ${line:-not started}${LEFT_LAW:+; the probes left the law in hold pass $LEFT_LAW}"
}
# shellcheck disable=SC2317 # called by the library, in the reason of an unexpected exit
box_notes() { notes; }
box_on_round() {                 # progress.txt, before each upload round
    { date -u
      notes
      grep -h "ANNEAL_RULE" "$OUT"/train_*.log 2>/dev/null | cut -c1-400 | tail -n 2
      grep -o "[0-9]*/[0-9]* games, [0-9]* positions.*" "$OUT/preencode.log" 2>/dev/null | tail -1
      find "$OUT" -maxdepth 1 -type f -printf '%f ' 2>/dev/null; echo
      tail -n 2 "$OUT"/*.log 2>/dev/null | tail -n 8
    } > "$OUT/progress.txt.tmp" && mv -f "$OUT/progress.txt.tmp" "$OUT/progress.txt"
}

box_init
[ -n "$WORKERS" ] || WORKERS=$(box_workers)
[ -n "$STAGE" ] || box_finish "NO_STAGE: build the code stage (tools/stage_code.py) and pass STAGE" 1
box_bind_run_stage
restored=(law_seed.jsonl start_areas.json corpus_summary.json sequence_summary.json sequence_manifest.json
          LOWER_DONE lower.pt lower.probe.jsonl lower.signal.jsonl train_lower.log)
for (( k = 1; k <= HOLD_PASSES; k++ )); do
    restored+=("hold$k.rc" "hold$k.pt" "hold$k.probe.jsonl" "hold$k.anneal.jsonl" "hold$k.signal.jsonl"
               "train_hold$k.log")
    box_upload_hold "hold$k.rc" "hold$k.pt" "hold$k.probe.jsonl"
done
box_restore "${restored[@]}" || box_finish "RESTORE_FAILED (restore.log)" 1
box_upload_hold LOWER_DONE lower.pt lower.probe.jsonl
box_pip huggingface_hub psutil pytest scipy requests || echo "pip install failed (pip.log)"

# ---- the code and the Rust wheel (the corpus, the pre-encoding and the match run on the core)
box_stage_code || box_finish "CODE_STAGING_FAILED (staging.log)" 1
cd "$BOX_REPO" || box_finish "CODE_STAGING_FAILED (no $BOX_REPO)" 1
box_build_wheel || box_finish "BUILD_FAILED (build.log)" 1
box_facts > "$OUT/box.txt.tmp" 2>&1
mv -f "$OUT/box.txt.tmp" "$OUT/box.txt"
timeout 2m python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || box_finish "NO_CUDA (box.txt)" 1
box_upload_async
box_monitor_start

# ---- the crash barrier: the recipe's, the rule's and memory play's tests on this wheel
if ! box_marked_this_stage "$BOX_STATE/TESTED"; then
    : > "$OUT/tests.log"
    box_bounded tests 40 tests.log python -m pytest tests/test_game_core.py tests/test_vision.py \
        tests/test_delayed_shroud.py tests/test_parity_integration.py tests/test_faction_posterior.py \
        tests/test_sighting_record.py tests/test_memory_model.py tests/test_preencode_sequences.py \
        tests/test_sequence_train.py tests/test_lr_law.py tests/test_match_memory.py tests/test_unit_vocab.py \
        tests/test_corpus_v3.py tests/test_no_shroud.py -q -p no:cacheprovider -m ""
    tail -n 3 "$OUT/tests.log"
    [ "$BOX_RC" -eq 0 ] || box_finish "TESTS_FAILED rc=$BOX_RC (tests.log)" 1
    box_mark "$BOX_STATE/TESTED"
fi

# ---- the reference and the raw replays
box_bounded reference 15 staging.log python tools/reference_player.py --ensure \
    || box_finish "REFERENCE_MISSING rc=$BOX_RC (staging.log)" 1
if ! box_marked_this_stage "$BOX_STATE/INPUTS_DONE"; then
    box_bounded inputs 20 staging.log python - "$RAW_TAR" <<'EOF' \
        || box_finish "INPUTS_FAILED rc=$BOX_RC (staging.log)" 1
import sys, tarfile
from huggingface_hub import hf_hub_download
path = hf_hub_download("momom2/wesnoth-model-checkpoints", sys.argv[1])
with tarfile.open(path, "r") as tf:
    tf.extractall(".")
    n = sum(1 for name in tf.getnames() if name.endswith(".bz2"))
print("raw replays", n, flush=True)
EOF
    box_mark "$BOX_STATE/INPUTS_DONE"
fi

# ---- the corpus at version 5; the crash barrier: every candidate accounted for, under 1% failed.
# The raw replays and the corpus live in the staged repository, which a new
# stage replaces, so the marker names its stage.
if ! box_marked_this_stage "$BOX_STATE/CORPUS_DONE" && [ ! -f "$OUT/LOWER_DONE" ]; then
    rm -rf "$CORPUS" "${CORPUS}_duplicates"
    from=$(box_size "$OUT/corpus_build.log")
    box_bounded --stall "$OUT/corpus_build.log" "$BUILD_STALL_MIN" corpus "$BUILD_CUT_MIN" corpus_build.log \
        python tools/build_imitation_dataset.py --raw-root . --out "$CORPUS" --workers "$WORKERS"
    tail -c "+$(( from + 1 ))" "$OUT/corpus_build.log" | grep -q "BUILD_DONE" \
        || box_finish "CORPUS_${BOX_WHY^^} rc=$BOX_RC (corpus_build.log)" 1
    box_bounded corpus-check 10 corpus_build.log \
        python tools/build_imitation_dataset.py --out "$CORPUS" --check "$OUT/corpus_summary.json" \
        || box_finish "CORPUS_BARRIER rc=$BOX_RC (corpus_build.log, corpus_summary.json)" 1
    box_mark "$BOX_STATE/CORPUS_DONE"
fi

# ---- the sequences; the crash barrier: under 0.5% of games skipped, posterior errors under 0.1%
if ! box_marked_this_stage "$SEQ/SEQUENCES_DONE" && [ ! -f "$OUT/LOWER_DONE" ]; then
    if ! box_marked_this_stage "$SEQ/STARTED"; then          # records another stage's code built
        rm -rf "$SEQ"
        box_mark "$SEQ/STARTED" || box_finish "SEQUENCES_UNWRITABLE ($SEQ)" 1
    fi
    from=$(box_size "$OUT/preencode.log")
    box_bounded --stall "$OUT/preencode.log" "$PREENCODE_STALL_MIN" preencode "$PREENCODE_CUT_MIN" preencode.log \
        python tools/preencode_sequences.py --dataset "$CORPUS" --out "$SEQ" --workers "$WORKERS"
    tail -n 3 "$OUT/preencode.log"
    tail -c "+$(( from + 1 ))" "$OUT/preencode.log" | grep -q "SEQUENCES_DONE" \
        || box_finish "PREENCODE_${BOX_WHY^^} rc=$BOX_RC (preencode.log)" 1
    box_bounded preencode-check 10 preencode.log python - "$SEQ" "$OUT/sequence_summary.json" <<'EOF' \
        || box_finish "PREENCODE_BARRIER rc=$BOX_RC (preencode.log, sequence_summary.json)" 1
import json, pathlib, sys
from tools.preencode_sequences import load_manifest
man = load_manifest(pathlib.Path(sys.argv[1]))
t = man["totals"]
summary = {"manifest_games": man["n_manifest_games"], "games": man["n_games"], "errors": len(man["errors"]),
           "fingerprint": man["fingerprint"], "core_phase": man["core_phase"],
           "corpus_version": man["corpus_version"], **t}
pathlib.Path(sys.argv[2]).write_text(json.dumps(summary, indent=1), encoding="utf-8")
print("sequences", summary, flush=True)
assert man["n_games"] + len(man["errors"]) == man["n_manifest_games"], "games unaccounted for"
assert len(man["errors"]) < 0.005 * man["n_manifest_games"], "0.5% or more of the games skipped"
assert t.get("posterior_errors", 0) < 0.001 * max(1, t.get("posteriors", 0)), "posterior errors at 0.1% or more"
EOF
    if ! { cp -f "$SEQ/sequence_manifest.json" "$OUT/sequence_manifest.json.tmp" \
            && mv -f "$OUT/sequence_manifest.json.tmp" "$OUT/sequence_manifest.json"; }; then
        box_finish "SEQUENCE_MANIFEST_COPY_FAILED" 1
    fi
    box_mark "$SEQ/SEQUENCES_DONE"
fi

# ---- the starting checkpoint and the law's points of the earlier passes
if [ ! -f "$OUT/LOWER_DONE" ] && [ ! -f "$START_PT" ] \
        && { [ ! -f "$OUT/hold1.pt" ] || [ ! -f "$OUT/law_seed.jsonl" ] || [ ! -f "$OUT/start_areas.json" ]; }; then
    box_bounded start 30 staging.log python - "$LINEAGE" "$START_PT" <<'EOF' \
        || box_finish "START_MISSING rc=$BOX_RC (staging.log)" 1
import json, os, shutil, sys
from huggingface_hub import hf_hub_download
lineage = json.load(open(sys.argv[1], encoding="utf-8"))
tmp = sys.argv[2] + ".tmp"
shutil.copyfile(hf_hub_download("momom2/wesnoth-model-checkpoints", lineage["start_hf"]), tmp)
os.replace(tmp, sys.argv[2])
print("the first hold pass starts from", lineage["start_hf"], flush=True)
EOF
fi
if [ ! -f "$OUT/LOWER_DONE" ] && { [ ! -f "$OUT/law_seed.jsonl" ] || [ ! -f "$OUT/start_areas.json" ]; }; then
    box_bounded law-seed 20 staging.log python - "$LINEAGE" "$START_PT" "$OUT" <<'EOF' \
        || box_finish "LAW_SEED_FAILED rc=$BOX_RC (staging.log)" 1
import json, os, sys
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download
from tools import lr_law
lineage = json.load(open(sys.argv[1], encoding="utf-8"))
passes = [dict(p, probes=hf_hub_download("momom2/wesnoth-model-checkpoints", p["probes_hf"]))
          for p in lineage["passes"]]
start = torch.load(sys.argv[2], map_location="cpu", weights_only=True)
state, meta = start["sequence_resume"]["state"], start["training_meta"]
own = next(p for p in passes if p["name"] == lineage["start_pass"])
assert "areas" not in state, "the start checkpoint carries its areas: it needs no replay"
assert meta.get("decay_from") == own["decay_from"] and state["positions"] >= own["decay_from"] * own["total_positions"], \
    "the start checkpoint is not where the start pass's lowering began"
result = lr_law.replay(passes, "k64")
out = Path(sys.argv[3])
lr_law.write_points(out / "law_seed.jsonl.tmp", result["points"])
os.replace(out / "law_seed.jsonl.tmp", out / "law_seed.jsonl")
areas = result["areas"][own["name"]][int(state["steps"]) - 1].to_dict()
(out / "start_areas.json.tmp").write_text(json.dumps(areas), encoding="utf-8")
os.replace(out / "start_areas.json.tmp", out / "start_areas.json")
print(f"the law's seed: {len(result['points'])} probes; the start checkpoint at {state['steps']} steps of "
      f"{own['name']}, areas {areas}", flush=True)
EOF
fi

# ---- the hold passes under the anneal rule
build_points() {                 # build_points K: the law's points before hold pass K (the seed, then holds 1..K-1)
    local k
    cp -f "$OUT/law_seed.jsonl" "$POINTS.tmp" || return 1
    for (( k = 1; k < $1; k++ )); do
        timeout 5m python tools/lr_law.py append "$OUT/hold$k.probe.jsonl" --key k64 --source "hold$k" \
            --out "$POINTS.tmp" || return 1
    done
    mv -f "$POINTS.tmp" "$POINTS"
}
hold_attempt() {                 # hold_attempt K MINUTES: hold pass K, continuing hold$K.pt when present; sets BOX_RC, BOX_WHY
    local k=$1 start=()
    if [ -f "$OUT/hold$k.pt" ]; then
        start=(--resume)
    elif [ "$k" -eq 1 ]; then
        start=(--init-from "$START_PT" --initial-areas "$(cat "$OUT/start_areas.json")")
    else
        start=(--init-from "$OUT/hold$(( k - 1 )).pt")
    fi
    box_bounded --stall "$OUT/train_hold$k.log" "$TRAIN_STALL_MIN" "hold $k" "$2" "train_hold$k.log" \
        python tools/sequence_train.py --sequences "$SEQ" --dataset "$CORPUS" --out "$OUT/hold$k.pt" \
        --seed $(( RUN_SEED + k )) --device cuda --warmup-steps 0 --anneal-rule "$THRESHOLD" \
        --law-points "$POINTS" --law-key k64 "${start[@]}"
}
BRANCH=""                        # the checkpoint the lowering starts from
LEFT_LAW=""                      # the hold pass whose probes left the law, if one did
for (( k = 1; k <= HOLD_PASSES; k++ )); do
    if [ ! -f "$OUT/hold$k.rc" ]; then
        [ ! -f "$OUT/LOWER_DONE" ] || break       # the lowering is done: there is nothing left to hold
        build_points "$k" || box_finish "LAW_POINTS_FAILED before hold pass $k" 1
        from=$(box_size "$OUT/train_hold$k.log")
        deadline=$(( $(date +%s) + TRAIN_CUT_MIN * 60 ))
        hold_attempt "$k" "$TRAIN_CUT_MIN"
        left=$(( (deadline - $(date +%s)) / 60 ))
        if [ "$BOX_RC" -ne 0 ] && [ "$BOX_RC" -ne 6 ] && [ "$BOX_RC" -ne 7 ] && [ "$BOX_RC" -ne 3 ] \
                && [ "$BOX_WHY" = failed ] && [ "$left" -ge 60 ] && [ -f "$OUT/hold$k.pt" ]; then
            hold_attempt "$k" "$left"    # a crash retries once, from the last periodic checkpoint
        fi
        case $BOX_RC in
            0) tail -c "+$(( from + 1 ))" "$OUT/train_hold$k.log" | grep -q "SEQUENCE_TRAIN_DONE" \
                   || box_finish "HOLD_UNFINISHED pass $k: exit 0 without SEQUENCE_TRAIN_DONE (train_hold$k.log)" 1 ;;
            6|7) ;;
            3) box_finish "MEMORY_BARRIER_FAILED in hold pass $k (train_hold$k.log) $(notes)" 1 ;;
            *) box_finish "HOLD_${BOX_WHY^^} pass $k rc=$BOX_RC (train_hold$k.log; it continues from hold$k.pt on re-entry) $(notes)" 1 ;;
        esac
        echo "$BOX_RC" > "$OUT/hold$k.rc.tmp" && mv -f "$OUT/hold$k.rc.tmp" "$OUT/hold$k.rc"
        box_upload_async
    fi
    rc=$(cat "$OUT/hold$k.rc")
    BRANCH="$OUT/hold$k.pt"
    if [ "$rc" = 7 ]; then
        LEFT_LAW=$k
        echo "hold pass $k: its latest probes left the law; the holding ends here (hold$k.anneal.jsonl)"
        break
    fi
    [ "$rc" != 6 ] && continue
    echo "the anneal rule lowers the rate in hold pass $k: $(grep -h ANNEAL_RULE "$OUT/train_hold$k.log" | tail -1 | cut -c1-300)"
    break
done
if [ ! -f "$OUT/LOWER_DONE" ] && { [ -z "$BRANCH" ] || [ ! -f "$BRANCH" ]; }; then
    box_finish "NO_HOLD_CHECKPOINT (${BRANCH:-none})" 1
fi

# ---- the lowering: a straight line from the peak to 0 over LOWER_POSITIONS
lower_attempt() {                # lower_attempt MINUTES: the lowering, continuing lower.pt when present; sets BOX_RC, BOX_WHY
    local start=(--init-from "$BRANCH")
    [ ! -f "$OUT/lower.pt" ] || start=(--resume)
    box_bounded --stall "$OUT/train_lower.log" "$TRAIN_STALL_MIN" lower "$1" train_lower.log \
        python tools/sequence_train.py --sequences "$SEQ" --dataset "$CORPUS" --out "$OUT/lower.pt" \
        --seed $(( RUN_SEED + 100 )) --device cuda --warmup-steps 0 --decay-from 0 \
        --pass-positions "$LOWER_POSITIONS" "${start[@]}"
}
if [ ! -f "$OUT/LOWER_DONE" ]; then
    from=$(box_size "$OUT/train_lower.log")
    deadline=$(( $(date +%s) + TRAIN_CUT_MIN * 60 ))
    lower_attempt "$TRAIN_CUT_MIN"
    left=$(( (deadline - $(date +%s)) / 60 ))
    if [ "$BOX_RC" -ne 0 ] && [ "$BOX_RC" -ne 3 ] && [ "$BOX_WHY" = failed ] && [ "$left" -ge 60 ] \
            && [ -f "$OUT/lower.pt" ]; then
        lower_attempt "$left"            # a crash retries once, from the last periodic checkpoint
    fi
    if [ "$BOX_RC" -ne 0 ] || ! tail -c "+$(( from + 1 ))" "$OUT/train_lower.log" | grep -q "SEQUENCE_TRAIN_DONE"; then
        box_finish "LOWERING_${BOX_WHY^^} rc=$BOX_RC (train_lower.log; it continues from lower.pt on re-entry) $(notes)" 1
    fi
    box_mark "$OUT/LOWER_DONE"
    box_upload_async
fi
tail -n 1 "$OUT/lower.probe.jsonl" | cut -c1-600

# ---- the match: the candidate at 64 slots against the reference
match() {                        # match NAME GAMES SEED_BASE MAX_EXTRA ARGS...: one attempt, resumed in its directory, then its fit
    local name="$1" games="$2" sb="$3" extra="$4" dir="$OUT/games_$1" t0 f
    shift 4
    if [ -f "$OUT/$name.fit.json" ]; then echo "match $name done"; return 0; fi
    if [ ! -f "$OUT/timing_$name.txt" ]; then          # the games; a re-entry after them redoes only the fit
        t0=$(date +%s)
        box_bounded "match $name" "$MATCH_CUT_MIN" "$name.log" \
            python tools/run_elo_batch.py "$@" \
            --outdir "$dir" --games "$games" --max-extra-games "$extra" --seed-base "$sb" \
            --mcts-sims 0 --raw-temperature-a 0 --raw-temperature-b 0 \
            --persistent-workers --shared-inference --no-infer-compile --device cuda --jobs "$JOBS" \
            --time-budget-min 60
        echo "$name: $(( $(date +%s) - t0 )) s, $(find "$dir" -maxdepth 1 -name 'game_*.json' 2>/dev/null | wc -l) games," \
             "$(decisive_results "$dir") decisive, rc=$BOX_RC $BOX_WHY" | tee "$OUT/timing_$name.txt" | tee -a "$OUT/match.walls"
        for f in "$dir"/.inference_server_*.json; do
            [ -f "$f" ] && cp -f "$f" "$OUT/$name.server_${f##*/.inference_server_}"
        done
    fi
    box_bounded "fit $name" 10 "$name.log" \
        python tools/elo_collect.py "$dir" --no-catalog --save-json "$OUT/$name.fit.json"
}
EO=$(timeout 1m python -c "import json; print(json.load(open('configs/reference_player.json'))['decode']['raw_end_turn_offset'])") \
    || box_finish "REFERENCE_CONFIG_UNREADABLE (configs/reference_player.json)" 1
mapfile -t REF_B < <(timeout 1m python tools/reference_player.py --flags b | tr ' ' '\n')
[ "${#REF_B[@]}" -ge 4 ] || box_finish "REFERENCE_FLAGS_FAILED (tools/reference_player.py --flags b)" 1
restored=("$MATCH.fit.json" "timing_$MATCH.txt")
[ -d "$OUT/games_$MATCH" ] || restored+=("games_$MATCH.tar.gz")
box_restore "${restored[@]}" || box_finish "RESTORE_FAILED (restore.log)" 1
if [ -f "$OUT/games_$MATCH.tar.gz" ]; then       # the games come back as the tarball their directory went up as
    [ -d "$OUT/games_$MATCH" ] || timeout -k 30s 10m tar -xzf "$OUT/games_$MATCH.tar.gz" -C "$OUT" \
        || box_finish "MATCH_RESTORE_FAILED ($MATCH)" 1
    rm -f "$OUT/games_$MATCH.tar.gz"
    box_mark_landed "games_$MATCH.tar.gz" "$OUT/games_$MATCH" || echo "games_$MATCH will go up again (restore.log)"
fi
gpu_or_finish() {                # gpu_or_finish NAME: before match NAME, the GPU answers or the entry ends
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before match $1: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
}
MATCH_RESULT=""                  # this entry's verdict on the match: empty when it holds its decisive games
play() {                         # play NAME SEED_BASE ARGS...: the match, once more when short, then its verdict
    local name="$1" sb="$2" decisive rc
    shift 2
    box_upload_dir "games_$name" "$OUT/games_$name"
    box_upload_hold "$name.fit.json" "games_$name.tar.gz"
    box_upload_hold "timing_$name.txt" "games_$name.tar.gz"
    [ -f "$OUT/$name.fit.json" ] || gpu_or_finish "$name"
    match "$name" "$GAMES" "$sb" 1500 "$@"
    decisive=$(decisive_results "$OUT/games_$name")
    if [[ $decisive =~ ^[0-9]+$ ]] && [ "$decisive" -lt "$GAMES" ]; then
        # The short attempt's fit and timing go, here and on HF, before the second.
        rm -f "$OUT/$name.fit.json" "$OUT/timing_$name.txt"
        box_clear "$name.fit.json" "timing_$name.txt" \
            || echo "the short attempt's fit of $name may stay on HF until the second lands (upload.log)"
        gpu_or_finish "$name"
        match "$name" "$GAMES" "$sb" 1500 "$@"
        decisive=$(decisive_results "$OUT/games_$name")
    fi
    rc=$(sed -n 's/.* rc=\([0-9]*\) .*/\1/p' "$OUT/timing_$name.txt" 2>/dev/null | tail -n 1)
    if ! [[ $decisive =~ ^[0-9]+$ ]]; then
        MATCH_RESULT="MATCH_FAILED: its games could not be counted ($name.log)"
    elif [ ! -f "$OUT/$name.fit.json" ]; then
        MATCH_RESULT="MATCH_FAILED: no fit ($name.log)"
    elif [ "$decisive" -lt "$GAMES" ]; then
        # run_elo_batch: 3 or 4, out of time or of replacements, and 124, the
        # step's own cut, leave a match short; any other exit is a defect.
        case ${rc:-none} in
            0|3|4|124) MATCH_RESULT="MATCH_CUT: $decisive of $GAMES decisive games (rc=$rc); the fit is not read" ;;
            *) MATCH_RESULT="MATCH_FAILED: rc=${rc:-none} ($name.log, games_$name/failed_*.json)" ;;
        esac
    fi
    [ -z "$MATCH_RESULT" ] || echo "$MATCH_RESULT" | tee -a "$OUT/match.walls"
    box_upload_async
}
mkdir -p training/checkpoints
cp "$OUT/lower.pt" "$ARM" || box_finish "ARM_COPY_FAILED (lower.pt)" 1
play "$MATCH" "$MATCH_SEED_BASE" --label-a cand64 --spec-a "$ARM" --memory-a 64 \
    --raw-end-turn-offset-a "$EO" "${REF_B[@]}"
box_on_round
case $MATCH_RESULT in
    "") box_finish "IMITATION_ANNEAL_DONE $(notes)" ;;
    MATCH_CUT*) box_finish "IMITATION_ANNEAL_DONE $(notes); $MATCH_RESULT" ;;
    *) box_finish "IMITATION_ANNEAL_MATCH_FAILED $(notes); $MATCH_RESULT" 1 ;;
esac
