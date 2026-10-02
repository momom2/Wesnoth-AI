#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# Why the reference's memory costs strength in play
# (docs/memory_in_play_prereg_20261002.md), on one box:
#   the tests of the counterfactual tool and of memory play, on the wheel
#     built from the stage;
#   the reference checkpoint, the pass-2 match records of 64 slots against
#     0 (PASS2), and the corpus rebuilt at version 5 from the raw replays
#     (RAW_TAR) for its holdout games;
#   tools/analysis/memory_counterfactual.py, each decision read with the
#     memory carried at the reference's slots and at 0 slots: the 64-slot
#     player's decisions (own), the 0-slot player's (other), the holdout's
#     human decisions (human), CF_PROCS processes each;
#   two matches, PURE, 800 decisive games, before the readings:
#     M1. 64 slots against 0, both at the end_turn offset -2.0 (seed base
#         MATCH_SEED_BASE, 88000 by default);
#     M2. 0 slots at -2.0 against 0 slots at -1.5 (the base + 1000).
#   A match is read only once it holds its decisive games.
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md):
# the onstart of scripts/rent_box.py fetches the library of STAGE, then
# this script. Every step has a bound; the dead-man's switch finishes the
# entry after BOX_MAX_H hours whatever it is doing. Records go to HF
# $HF_DIR every 30 minutes and at the end, each match's games as one
# tarball. Re-entry skips finished steps. Every exit past the stage check
# uploads the records with ALL_DONE last and stops the instance.
# Never `set -x`: the HF token and the instance key are in the environment.
# box-needs: disk_gb=60 ram_gb=64 gpu_ram_gb=24 cores=32 gpu=4090
set -uo pipefail
WORKDIR=/workspace
OUT=$WORKDIR/memoryinplay
STAGE="${STAGE:-}"
RAW_TAR="${RAW_TAR:-tier-b/corpus_v3/raw_corpus_20260929.tar}"
PASS2="${PASS2:-tier-b/parity_memory_pass2_20261002}"
MATCH_SEED_BASE="${MATCH_SEED_BASE:-88000}"
CF_PROCS="${CF_PROCS:-16}"
WORKERS="${WORKERS:-}"                           # default: box_workers (cores, memory-bounded), after box_init
GAMES="${GAMES:-800}"
JOBS="${JOBS:-20}"
export HF_DIR="${HF_DIR:-tier-b/memory_in_play_20261002}"
# Bounds in minutes, from the pre-registration's "Cost".
BUILD_CUT_MIN="${BUILD_CUT_MIN:-60}"             # estimated 10 on 32 cores
BUILD_STALL_MIN="${BUILD_STALL_MIN:-15}"         # the builder logs every 1,000 candidates
CF_CUT_MIN="${CF_CUT_MIN:-90}"                   # estimated 7 to 25 per source
MATCH_CUT_MIN="${MATCH_CUT_MIN:-90}"             # the match stops itself at 60 (--time-budget-min), a game at 20 more
BOX_MAX_H="${BOX_MAX_H:-4}"                      # 2.4 times the 100 minutes estimated at most
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }

CORPUS=replays_dataset_imitation
PASS2_DIR=$WORKDIR/pass2
MATCHES="m1_mem64_vs_mem0_eo2 m2_eo2_vs_eo15"

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
notes() {                        # what is done, for the finish reason
    local done_=() f
    for f in cf_own cf_other cf_human; do [ -f "$OUT/$f.jsonl" ] && done_+=("$f"); done
    for f in $MATCHES; do [ -f "$OUT/$f.fit.json" ] && done_+=("$f"); done
    echo "done: ${done_[*]:-nothing}"
}
# shellcheck disable=SC2317 # called by the library, in the reason of an unexpected exit
box_notes() { notes; }
box_on_round() {                 # progress.txt, before each upload round
    { date -u
      notes
      for f in "$OUT"/cf_*.log; do [ -f "$f" ] && grep "memory_counterfactual" "$f" | tail -n 1; done
      find "$OUT" -maxdepth 1 -type f -printf '%f ' 2>/dev/null; echo
      tail -n 2 "$OUT"/*.log 2>/dev/null | tail -n 8
    } > "$OUT/progress.txt.tmp" && mv -f "$OUT/progress.txt.tmp" "$OUT/progress.txt"
}

box_init
[ -n "$WORKERS" ] || WORKERS=$(box_workers)
[ -n "$STAGE" ] || box_finish "NO_STAGE: build the code stage (tools/stage_code.py) and pass STAGE" 1
box_bind_run_stage
box_restore cf_own.jsonl cf_other.jsonl cf_human.jsonl corpus_summary.json \
    || box_finish "RESTORE_FAILED (restore.log)" 1
box_pip huggingface_hub psutil pytest scipy requests || echo "pip install failed (pip.log)"

# ---- the code and the Rust wheel (the corpus, the readings and the matches run on the core)
box_stage_code || box_finish "CODE_STAGING_FAILED (staging.log)" 1
cd "$BOX_REPO" || box_finish "CODE_STAGING_FAILED (no $BOX_REPO)" 1
box_build_wheel || box_finish "BUILD_FAILED (build.log)" 1
box_facts > "$OUT/box.txt.tmp" 2>&1
mv -f "$OUT/box.txt.tmp" "$OUT/box.txt"
timeout 2m python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || box_finish "NO_CUDA (box.txt)" 1
box_upload_async
box_monitor_start

# ---- the crash barrier: the tool's and memory play's tests on this wheel
if ! box_marked_this_stage "$BOX_STATE/TESTED"; then
    : > "$OUT/tests.log"
    box_bounded tests 20 tests.log python -m pytest tests/test_memory_counterfactual.py \
        tests/test_match_memory.py tests/test_memory_model.py -q -p no:cacheprovider -m ""
    tail -n 3 "$OUT/tests.log"
    [ "$BOX_RC" -eq 0 ] || box_finish "TESTS_FAILED rc=$BOX_RC (tests.log)" 1
    box_mark "$BOX_STATE/TESTED"
fi

# ---- the reference, its decode, the pass-2 records and the raw replays
box_bounded reference 15 staging.log python tools/reference_player.py --ensure \
    || box_finish "REFERENCE_MISSING rc=$BOX_RC (staging.log)" 1
read -r REF SLOTS EO < <(timeout 1m python -c "import json; r = json.load(open('configs/reference_player.json')); \
print(r['checkpoint_local'], r['memory_slots'], r['decode']['raw_end_turn_offset'])") \
    || box_finish "REFERENCE_CONFIG_UNREADABLE (configs/reference_player.json)" 1
[ -n "$EO" ] || box_finish "REFERENCE_CONFIG_UNREADABLE (configs/reference_player.json)" 1
if ! box_marked_this_stage "$BOX_STATE/INPUTS_DONE"; then
    box_bounded inputs 20 staging.log python - "$RAW_TAR" "$PASS2" "$PASS2_DIR" <<'EOF' \
        || box_finish "INPUTS_FAILED rc=$BOX_RC (staging.log)" 1
import sys, tarfile
from huggingface_hub import hf_hub_download
repo = "momom2/wesnoth-model-checkpoints"
with tarfile.open(hf_hub_download(repo, sys.argv[1]), "r") as tf:
    tf.extractall(".")
    print("raw replays", sum(1 for name in tf.getnames() if name.endswith(".bz2")), flush=True)
with tarfile.open(hf_hub_download(repo, sys.argv[2] + "/games_arm64_vs_arm0.tar.gz"), "r:gz") as tf:
    tf.extractall(sys.argv[3])
    print("pass-2 records", sum(1 for name in tf.getnames() if name.endswith(".game.jsonl.gz")), flush=True)
EOF
    box_mark "$BOX_STATE/INPUTS_DONE"
fi

# ---- the corpus at version 5, for its holdout games; the crash barrier: every candidate
# accounted for, under 1% failed. The raw replays and the corpus live in the staged
# repository, which a new stage replaces, so the marker names its stage.
if ! box_marked_this_stage "$BOX_STATE/CORPUS_DONE" && [ ! -f "$OUT/cf_human.jsonl" ]; then
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

# ---- the counterfactual step: a source that fails is noted and the entry goes on. The
# shards write next to cf_NAME.jsonl in OUT, so a cut leaves their rows for the upload.
CF_FAILED=0                      # this entry's failed readings
gpu_or_finish() {                # gpu_or_finish NAME: before step NAME, the GPU answers or the entry ends
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before $1: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
}
counterfactual() {               # counterfactual NAME ARGS...: one source's rows into cf_NAME.jsonl
    local name="$1"
    shift
    if [ -f "$OUT/cf_$name.jsonl" ]; then echo "counterfactual $name done"; return 0; fi
    gpu_or_finish "counterfactual $name"
    box_bounded "counterfactual $name" "$CF_CUT_MIN" "cf_$name.log" \
        python tools/analysis/memory_counterfactual.py --checkpoint "$REF" --slots "$SLOTS" \
        --offset "$EO" --procs "$CF_PROCS" --device cuda --out "$OUT/cf_$name.jsonl" "$@"
    if [ "$BOX_RC" -eq 0 ] && [ -f "$OUT/cf_$name.jsonl" ]; then
        echo "counterfactual $name: $(wc -l < "$OUT/cf_$name.jsonl") rows"
    else
        echo "COUNTERFACTUAL_FAILED $name: rc=$BOX_RC $BOX_WHY (cf_$name.log)" | tee -a "$OUT/failures.txt"
        CF_FAILED=$(( CF_FAILED + 1 ))
    fi
    box_upload_async
}
# ---- the matches
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
MATCHES_FAILED=0                 # this entry's verdicts on the matches (play)
MATCHES_CUT=0
play() {                         # play NAME SEED_BASE ARGS...: the match, once more when short, then its verdict
    local name="$1" sb="$2" decisive rc
    shift 2
    box_upload_dir "games_$name" "$OUT/games_$name"
    box_upload_hold "$name.fit.json" "games_$name.tar.gz"
    box_upload_hold "timing_$name.txt" "games_$name.tar.gz"
    [ -f "$OUT/$name.fit.json" ] || gpu_or_finish "match $name"
    match "$name" "$GAMES" "$sb" 1500 "$@"
    decisive=$(decisive_results "$OUT/games_$name")
    if [[ $decisive =~ ^[0-9]+$ ]] && [ "$decisive" -lt "$GAMES" ]; then
        # The short attempt's fit and timing go, here and on HF, before the second.
        rm -f "$OUT/$name.fit.json" "$OUT/timing_$name.txt"
        box_clear "$name.fit.json" "timing_$name.txt" \
            || echo "the short attempt's fit of $name may stay on HF until the second lands (upload.log)"
        gpu_or_finish "match $name"
        match "$name" "$GAMES" "$sb" 1500 "$@"
        decisive=$(decisive_results "$OUT/games_$name")
    fi
    rc=$(sed -n 's/.* rc=\([0-9]*\) .*/\1/p' "$OUT/timing_$name.txt" 2>/dev/null | tail -n 1)
    if ! [[ $decisive =~ ^[0-9]+$ ]]; then
        echo "MATCH_FAILED $name: its games could not be counted ($name.log)" | tee -a "$OUT/match.walls"
        MATCHES_FAILED=$(( MATCHES_FAILED + 1 ))
    elif [ ! -f "$OUT/$name.fit.json" ]; then
        echo "MATCH_FAILED $name: no fit ($name.log)" | tee -a "$OUT/match.walls"
        MATCHES_FAILED=$(( MATCHES_FAILED + 1 ))
    elif [ "$decisive" -lt "$GAMES" ]; then
        # run_elo_batch: 3 or 4, out of time or of replacements, and 124, the
        # step's own cut, leave a match short; any other exit is a defect.
        case ${rc:-none} in
            0|3|4|124)
                echo "MATCH_CUT $name: $decisive of $GAMES decisive games (rc=$rc); the fit is not read" \
                    | tee -a "$OUT/match.walls"
                MATCHES_CUT=$(( MATCHES_CUT + 1 )) ;;
            *)
                echo "MATCH_FAILED $name: rc=${rc:-none} ($name.log, games_$name/failed_*.json)" \
                    | tee -a "$OUT/match.walls"
                MATCHES_FAILED=$(( MATCHES_FAILED + 1 )) ;;
        esac
    fi
    box_upload_async
}
restored=()
for name in $MATCHES; do                 # a match directory already here is not fetched again
    restored+=("$name.fit.json" "timing_$name.txt")
    [ -d "$OUT/games_$name" ] || restored+=("games_$name.tar.gz")
done
box_restore "${restored[@]}" || box_finish "RESTORE_FAILED (restore.log)" 1
for name in $MATCHES; do                 # a match's games come back as the tarball its directory went up as
    if [ -f "$OUT/games_$name.tar.gz" ]; then
        [ -d "$OUT/games_$name" ] || timeout -k 30s 10m tar -xzf "$OUT/games_$name.tar.gz" -C "$OUT" \
            || box_finish "MATCH_RESTORE_FAILED ($name)" 1
        rm -f "$OUT/games_$name.tar.gz"
        box_mark_landed "games_$name.tar.gz" "$OUT/games_$name" \
            || echo "games_$name will go up again (restore.log)"
    fi
done
play m1_mem64_vs_mem0_eo2 "$MATCH_SEED_BASE" \
    --label-a mem64eo2 --spec-a "$REF" --memory-a "$SLOTS" --raw-end-turn-offset-a -2.0 \
    --label-b mem0eo2 --spec-b "$REF" --memory-b 0 --raw-end-turn-offset-b -2.0
play m2_eo2_vs_eo15 $(( MATCH_SEED_BASE + 1000 )) \
    --label-a mem0eo2 --spec-a "$REF" --memory-a 0 --raw-end-turn-offset-a -2.0 \
    --label-b mem0eo15 --spec-b "$REF" --memory-b 0 --raw-end-turn-offset-b "$EO"
# ---- the counterfactual readings, after the matches, whose length is known
counterfactual own --games "$PASS2_DIR/games_arm64_vs_arm0" --player arm64
counterfactual other --games "$PASS2_DIR/games_arm64_vs_arm0" --player arm0
counterfactual human --corpus "$CORPUS" --holdout

box_on_round
[ "$MATCHES_FAILED" -eq 0 ] && [ "$CF_FAILED" -eq 0 ] \
    || box_finish "MEMORY_IN_PLAY_FAILED $(notes): $MATCHES_FAILED matches failed, $CF_FAILED readings failed (match.walls, failures.txt)" 1
box_finish "MEMORY_IN_PLAY_DONE $(notes); $MATCHES_CUT matches cut"
