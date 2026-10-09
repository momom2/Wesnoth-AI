#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# Why `parity3`'s memory costs it strength in play
# (docs/memory_in_play_parity3_prereg_20261009.md), on one box:
#   the tests of the match path, the per-turn memory reset and the
#     counterfactual reader, on the wheel built from the stage;
#   three matches against `parity3` at 0 slots at the reference decode
#     (raw:t0+eo-1.5), PURE, 800 decisive games each, the Ladder maps with
#     fog and both factions drawn uniformly, every game recorded whole:
#     A1. `parity3` at 64 slots, end_turn offset -2.5 (seed base
#         MATCH_SEED_BASE, 104000 by default);
#     A2. `parity3` at 64 slots with its memory reset at each of its turns,
#         offset -1.5 (the base + 1000);
#     A3. `parity3` at 64 slots, offset -3.5 (the base + 2000);
#   each fitted (tools/elo_collect.py), its games up as one tarball, and its
#     tempo read (tools/analysis/turn_tempo.py);
#   the counterfactual readings (tools/analysis/memory_counterfactual.py,
#     CF_PROCS processes each): `parity3` at the reference decode, each
#     decision read with its memory carried at 64 slots and at 0, on the
#     64-slot player's decisions (own) and the 0-slot player's (other) in
#     the 810 games of the 64-against-0 baselines match (BASELINES), and on
#     both sides' decisions in the holdout games of the corpus rebuilt at
#     version 5 from the raw replays (RAW_TAR; human);
#   the readout (tools/analysis/memory_counterfactual_readout.py) of all of it.
#   A match is read only once it holds its decisive games.
#
# Box: an RTX 4090 with at least 32 usable cores (20 match workers; the
# corpus build and 16 reader processes). Expected wall: 1.5 to 2.2 hours.
# Bring-up and tests about 10 minutes; each match 13 to 18 minutes (the
# baselines box, training/metrics/parity3_baselines_20261008/match.walls:
# 64 slots against 0, 810 games in 750 s; a memory player against itself,
# 1,005 to 1,093 s); the raw replays and the corpus about 15 (10 to build on
# 32 cores, docs/imitation_anneal_prereg_20261003.md); the readings 20 to 50
# (the 2026-10-02 pre-registration's estimate for a source of the same size,
# never measured: own holds 179,448 recorded commands, other 207,081, the
# holdout a few hundred games); tempo, readout and final upload about 5.
# Identical match repeats have differed by 1.8x, which would bring the run
# to about 2.8 hours. Cost at $0.42-0.63 an hour: $0.62-1.40 expected,
# $1.20-1.80 at 2.8 h; BOX_MAX_H 4 caps it at $1.68-2.52 plus the final round.
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md): the
# onstart of scripts/rent_box.py fetches the library of STAGE, then this
# script. Every step has a bound; the dead-man's switch finishes the entry
# after BOX_MAX_H hours whatever it is doing. Records go to HF $HF_DIR every
# 30 minutes and after each match and each reading, each match's games as
# one tarball. Re-entry, on this machine or a new one (files absent here come
# back from HF), skips fitted matches and finished readings and resumes a
# match in its directory. A run belongs to the code stage that began it
# (RUN_STAGE): another stage continues it only with RESUME_OTHER_STAGE=1.
# Every exit past that check, clean or not, uploads the records with ALL_DONE
# last and stops the instance; an entry refused there stops it and sends
# nothing.
# Never `set -x`: the HF token and the instance key are in the environment.
# box-needs: disk_gb=60 ram_gb=64 gpu_ram_gb=24 cores=32 gpu=4090
set -uo pipefail
WORKDIR=/workspace
OUT=$WORKDIR/memoryinplay3
STAGE="${STAGE:-}"
RAW_TAR="${RAW_TAR:-tier-b/corpus_v3/raw_corpus_20260929.tar}"
BASELINES="${BASELINES:-tier-b/parity3_baselines_20261008/games_slots64_vs_slots0.tar.gz}"
# A match with seed base S plays seeds S to S+799 and replaces a capped game
# of slot i with seed S + i + k * 1,000,000. Scanned 2026-10-09: every seed
# base named in a branch or tag of the repository is 103000 or below (the
# material look-ahead gate's), and no committed game file carries a seed
# from 104000 to 106999.
MATCH_SEED_BASE="${MATCH_SEED_BASE:-104000}"
CF_PROCS="${CF_PROCS:-16}"
WORKERS="${WORKERS:-}"                           # default: box_workers (cores, memory-bounded), after box_init
GAMES="${GAMES:-800}"
JOBS="${JOBS:-20}"
export HF_DIR="${HF_DIR:-tier-b/memory_in_play_parity3_20261009}"
# Bounds in minutes, from the pre-registration's "Cost".
TESTS_CUT_MIN="${TESTS_CUT_MIN:-20}"             # the baselines box's list ran in 33-41 s; the added files take about a minute on the laptop
BUILD_CUT_MIN="${BUILD_CUT_MIN:-60}"             # estimated 10 on 32 cores
BUILD_STALL_MIN="${BUILD_STALL_MIN:-15}"         # the builder logs every 1,000 candidates
MATCH_CUT_MIN="${MATCH_CUT_MIN:-90}"             # the match stops itself at 60 (--time-budget-min), a game at 20 more
CF_CUT_MIN="${CF_CUT_MIN:-90}"                   # estimated 7 to 25 per source
BOX_MAX_H="${BOX_MAX_H:-4}"                      # 1.4 times the 2.8 hours of matches at 1.8x
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }

CORPUS=replays_dataset_imitation
BASE_DIR=$WORKDIR/baselines                      # outside OUT: the records stay on HF where they are
MATCHES="a1_eo25 a2_reset a3_eo35"
SOURCES="own other human"
OPPONENT=parity3_slots0

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
notes() {                        # the matches fitted and the readings done so far, for the finish reason
    local name done_=()
    for name in $MATCHES; do
        [ ! -f "$OUT/$name.fit.json" ] || done_+=("$name")
    done
    for name in $SOURCES; do
        [ ! -f "$OUT/cf_$name.jsonl" ] || done_+=("cf_$name")
    done
    [ ! -f "$OUT/readout.json" ] || done_+=(readout)
    echo "done: ${done_[*]:-nothing}"
}
# shellcheck disable=SC2317 # called by the library, in the reason of an unexpected exit
box_notes() { notes; }
box_on_round() {                 # progress.txt, before each upload round
    local name f
    { date -u
      notes
      cat "$OUT/match.walls" 2>/dev/null
      for name in $MATCHES; do
          [ ! -d "$OUT/games_$name" ] \
              || echo "$name: $(find "$OUT/games_$name" -maxdepth 1 -name 'game_*.json' | wc -l) games"
      done
      for f in "$OUT"/cf_*.log; do [ -f "$f" ] && grep "memory_counterfactual" "$f" | tail -n 1; done
      find "$OUT" -maxdepth 1 -type f -printf '%f ' 2>/dev/null; echo
      tail -n 2 "$OUT"/*.log 2>/dev/null | tail -n 8
    } > "$OUT/progress.txt.tmp" && mv -f "$OUT/progress.txt.tmp" "$OUT/progress.txt"
}

box_init
[ -n "$WORKERS" ] || WORKERS=$(box_workers)
[ -n "$STAGE" ] || box_finish "NO_STAGE: build the code stage (tools/stage_code.py) and pass STAGE" 1
box_bind_run_stage

# ---- the records of an earlier entry: fits, timings, logs, readings, and each match's games as the tarball it went up as
restored=(match.walls corpus_summary.json readout.json)
for name in $MATCHES; do                 # a match directory already here is not fetched again
    restored+=("$name.fit.json" "timing_$name.txt" "$name.log" "tempo_$name.json")
    [ -d "$OUT/games_$name" ] || restored+=("games_$name.tar.gz")
done
for name in $SOURCES; do
    restored+=("cf_$name.jsonl")
done
box_restore "${restored[@]}" || box_finish "RESTORE_FAILED (restore.log)" 1
for name in $MATCHES; do
    if [ -f "$OUT/games_$name.tar.gz" ]; then
        [ -d "$OUT/games_$name" ] || timeout -k 30s 10m tar -xzf "$OUT/games_$name.tar.gz" -C "$OUT" \
            || box_finish "MATCH_RESTORE_FAILED ($name)" 1
        rm -f "$OUT/games_$name.tar.gz"
        # The directory is what HF holds: it goes up again only once it changes.
        box_mark_landed "games_$name.tar.gz" "$OUT/games_$name" \
            || echo "games_$name will go up again (restore.log)"
    fi
done
box_pip huggingface_hub psutil pytest scipy requests || echo "pip install failed (pip.log)"

# ---- the code and the Rust wheel (the matches, the corpus and the readings run on the core)
box_stage_code || box_finish "CODE_STAGING_FAILED (staging.log)" 1
cd "$BOX_REPO" || box_finish "CODE_STAGING_FAILED (no $BOX_REPO)" 1
box_build_wheel || box_finish "BUILD_FAILED (build.log)" 1
box_facts > "$OUT/box.txt.tmp" 2>&1
mv -f "$OUT/box.txt.tmp" "$OUT/box.txt"
timeout 2m python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || box_finish "NO_CUDA (box.txt)" 1
box_upload_async
box_monitor_start

# ---- the crash barrier: the match path's, the reset's and the reader's tests on this wheel and this GPU
if ! box_marked_this_stage "$BOX_STATE/TESTED"; then
    : > "$OUT/tests.log"
    box_bounded tests "$TESTS_CUT_MIN" tests.log python -m pytest tests/test_game_core.py tests/test_vision.py \
        tests/test_parity_integration.py tests/test_sighting_record.py tests/test_faction_posterior.py \
        tests/test_memory_model.py tests/test_match_memory.py tests/test_memory_reset.py \
        tests/test_memory_counterfactual.py tests/test_raw_player.py tests/test_reference_player.py \
        tests/test_eval_workers.py tests/test_eval_inference_server.py tests/test_eval_match_failures.py \
        tests/test_game_record.py tests/test_elo_collect.py tests/test_corpus_v3.py tests/test_no_shroud.py \
        -q -p no:cacheprovider -m ""
    tail -n 3 "$OUT/tests.log"
    [ "$BOX_RC" -eq 0 ] || box_finish "TESTS_FAILED rc=$BOX_RC (tests.log)" 1
    box_mark "$BOX_STATE/TESTED"
fi

# ---- the reference: its checkpoint (SHA-256 checked) and each side's flags
box_bounded reference 15 staging.log python tools/reference_player.py --ensure \
    || box_finish "REFERENCE_MISSING rc=$BOX_RC (staging.log)" 1
read -r REF_LABEL REF_PT EO < <(timeout 1m python -c "import json; r = json.load(open('configs/reference_player.json')); \
print(r['label'], r['checkpoint_local'], r['decode']['raw_end_turn_offset'])") \
    || box_finish "REFERENCE_CONFIG_UNREADABLE (configs/reference_player.json)" 1
[ "$REF_LABEL" = parity3 ] \
    || box_finish "REFERENCE_CHANGED: configs/reference_player.json names $REF_LABEL; these matches measure parity3" 1
[ "$EO" = -1.5 ] || box_finish "REFERENCE_DECODE_CHANGED: offset $EO; the arms are set against -1.5" 1
ref_side() {                     # ref_side SIDE LABEL SLOTS [OFFSET]: the reference's match flags for SIDE, one per line, under LABEL at SLOTS slots (and OFFSET)
    local side=$1 label=$2 slots=$3 offset=${4:-} flags i
    mapfile -t flags < <(timeout 1m python tools/reference_player.py --flags "$side" | tr ' ' '\n')
    for (( i = 0; i + 1 < ${#flags[@]}; i++ )); do
        case ${flags[i]} in
            "--label-$side") flags[i + 1]=$label ;;
            "--memory-$side") flags[i + 1]=$slots ;;
            "--raw-end-turn-offset-$side") [ -z "$offset" ] || flags[i + 1]=$offset ;;
        esac
    done
    printf '%s\n' "${flags[@]}"
}
side_ok() {                      # side_ok SIDE SLOTS OFFSET FLAGS...: the flags play the reference's checkpoint at SLOTS slots and offset OFFSET
    local side=$1 slots=$2 offset=$3
    shift 3
    [[ " $* " == *" --spec-$side $REF_PT "* && " $* " == *" --memory-$side $slots "* \
        && " $* " == *" --raw-end-turn-offset-$side $offset "* ]]
}
mapfile -t A1 < <(ref_side a parity3_eo25 64 -2.5)
mapfile -t A2 < <(ref_side a parity3_mr 64)
A2+=(--memory-reset-a)
mapfile -t A3 < <(ref_side a parity3_eo35 64 -3.5)
mapfile -t SLOTS0_B < <(ref_side b "$OPPONENT" 0)
if ! { side_ok a 64 -2.5 "${A1[@]}" && side_ok a 64 "$EO" "${A2[@]}" && side_ok a 64 -3.5 "${A3[@]}" \
        && side_ok b 0 "$EO" "${SLOTS0_B[@]}"; }; then
    box_finish "REFERENCE_FLAGS_FAILED (tools/reference_player.py --flags)" 1
fi

# ---- the matches
match() {                        # match NAME GAMES SEED_BASE MAX_EXTRA ARGS...: one attempt, resumed in its directory, then its fit
    local name="$1" games="$2" sb="$3" extra="$4" dir="$OUT/games_$1" t0 f
    shift 4
    if [ -f "$OUT/$name.fit.json" ]; then echo "match $name done"; return 0; fi
    if [ ! -f "$OUT/timing_$name.txt" ]; then          # the games; a re-entry after them redoes only the fit
        echo "$(date -u +%FT%TZ) $name: seed base $sb, $*" >> "$OUT/$name.log"
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
gpu_or_finish() {                # gpu_or_finish STEP: before STEP, the GPU answers or the entry ends
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before $1: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
}
MATCHES_FAILED=0                 # this entry's verdicts on the matches (play)
MATCHES_CUT=0
play() {                         # play NAME SEED_BASE ARGS...: the match, once more when short, then its verdict and tempo
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
        # step's own cut, leave a match short; any other exit is a defect
        # (1: games failed or a server died; a signal; a usage error).
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
    if [ -d "$OUT/games_$name" ] && [ ! -f "$OUT/tempo_$name.json" ]; then
        box_bounded "tempo $name" 10 tempo.log \
            python tools/analysis/turn_tempo.py --games "$OUT/games_$name" --json "$OUT/tempo_$name.json" \
            || echo "tempo $name failed (tempo.log)"
    fi
    box_upload_async
}
play a1_eo25 "$MATCH_SEED_BASE" "${A1[@]}" "${SLOTS0_B[@]}"
play a2_reset $(( MATCH_SEED_BASE + 1000 )) "${A2[@]}" "${SLOTS0_B[@]}"
play a3_eo35 $(( MATCH_SEED_BASE + 2000 )) "${A3[@]}" "${SLOTS0_B[@]}"

# ---- the readings' inputs: the baselines match records and the raw replays
if ! box_marked_this_stage "$BOX_STATE/INPUTS_DONE"; then
    box_bounded inputs 20 staging.log python - "$RAW_TAR" "$BASELINES" "$BASE_DIR" <<'EOF' \
        || box_finish "INPUTS_FAILED rc=$BOX_RC (staging.log)" 1
import sys, tarfile
from huggingface_hub import hf_hub_download
repo = "momom2/wesnoth-model-checkpoints"
with tarfile.open(hf_hub_download(repo, sys.argv[1]), "r") as tf:
    tf.extractall(".")
    print("raw replays", sum(1 for name in tf.getnames() if name.endswith(".bz2")), flush=True)
with tarfile.open(hf_hub_download(repo, sys.argv[2]), "r:gz") as tf:
    tf.extractall(sys.argv[3])
    print("baselines records", sum(1 for name in tf.getnames() if name.endswith(".game.jsonl.gz")), flush=True)
EOF
    box_mark "$BOX_STATE/INPUTS_DONE"
fi
BASE_GAMES=$BASE_DIR/games_slots64_vs_slots0

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

# ---- the counterfactual readings: a source that fails is noted and the entry goes on. The
# shards write next to cf_NAME.jsonl in OUT, so a cut leaves their rows for the upload.
CF_FAILED=0                      # this entry's failed readings
counterfactual() {               # counterfactual NAME ARGS...: one source's rows into cf_NAME.jsonl
    local name="$1"
    shift
    if [ -f "$OUT/cf_$name.jsonl" ]; then echo "counterfactual $name done"; return 0; fi
    gpu_or_finish "counterfactual $name"
    box_bounded "counterfactual $name" "$CF_CUT_MIN" "cf_$name.log" \
        python tools/analysis/memory_counterfactual.py --checkpoint "$REF_PT" --slots 64 \
        --offset "$EO" --procs "$CF_PROCS" --device cuda --out "$OUT/cf_$name.jsonl" "$@"
    if [ "$BOX_RC" -eq 0 ] && [ -f "$OUT/cf_$name.jsonl" ]; then
        echo "counterfactual $name: $(wc -l < "$OUT/cf_$name.jsonl") rows"
    else
        echo "COUNTERFACTUAL_FAILED $name: rc=$BOX_RC $BOX_WHY (cf_$name.log)" | tee -a "$OUT/failures.txt"
        CF_FAILED=$(( CF_FAILED + 1 ))
    fi
    box_upload_async
}
counterfactual own --games "$BASE_GAMES" --player parity3
counterfactual other --games "$BASE_GAMES" --player "$OPPONENT"
counterfactual human --corpus "$CORPUS" --holdout
if [ ! -f "$OUT/tempo_baselines.json" ]; then
    box_bounded "tempo baselines" 10 tempo.log \
        python tools/analysis/turn_tempo.py --games "$BASE_GAMES" --json "$OUT/tempo_baselines.json" \
        || echo "tempo baselines failed (tempo.log)"
fi

# ---- the readout, once every reading and every fit is here
have_all=1
for name in $SOURCES; do [ -f "$OUT/cf_$name.jsonl" ] || have_all=0; done
for name in $MATCHES; do [ -f "$OUT/$name.fit.json" ] || have_all=0; done
if [ "$have_all" -eq 1 ] && [ ! -f "$OUT/readout.json" ]; then
    box_bounded readout 20 readout.log python tools/analysis/memory_counterfactual_readout.py \
        own="$OUT/cf_own.jsonl" other="$OUT/cf_other.jsonl" human="$OUT/cf_human.jsonl" \
        --fit A1="$OUT/a1_eo25.fit.json" --fit A2="$OUT/a2_reset.fit.json" --fit A3="$OUT/a3_eo35.fit.json" \
        --opponent "$OPPONENT" --json "$OUT/readout.json" \
        || echo "READOUT_FAILED rc=$BOX_RC (readout.log)" | tee -a "$OUT/failures.txt"
fi

box_on_round
if [ "$MATCHES_FAILED" -ne 0 ] || [ "$CF_FAILED" -ne 0 ]; then
    box_finish "MEMORY_IN_PLAY_FAILED $(notes): $MATCHES_FAILED matches failed, $CF_FAILED readings failed (match.walls, failures.txt)" 1
fi
box_finish "MEMORY_IN_PLAY_DONE $(notes); $MATCHES_CUT matches cut"
