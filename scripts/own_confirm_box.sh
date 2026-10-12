#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# The confirmation of round 1 of the memory trained on its own trajectories
# (docs/own_confirm_prereg_20261011.md), on one box:
#   the tests of the match path (as the baselines run them), on the wheel
#   built from the stage;
#   the tuned checkpoint from HF (TUNED_HF, its SHA-256 checked against
#   TUNED_SHA256) and the reference `parity3` (configs/reference_player.json,
#   its SHA-256 checked);
#   four matches, PURE, both sides at the reference decode (raw:t0+eo-1.5),
#   the Ladder maps with both factions drawn uniformly, 800 decisive games
#   each, every game recorded whole:
#     1. own0_vs_p3slots0: the tuned checkpoint at 0 slots against parity3 at
#        0 slots (seed base MATCH_SEED_BASE, 109000 by default): round 1's
#        M2 on fresh seeds;
#     2. own64_vs_p3: the tuned checkpoint at 64 slots against parity3 at 64
#        slots, the reference (the base + 1000);
#     3. own64_vs_own0: the tuned checkpoint at 64 slots against itself at 0
#        slots (the base + 2000): the memory's cost within the tuned weights;
#     4. own64_selfplay: the tuned checkpoint at 64 slots against itself (the
#        base + 3000): the tuned player's own trajectories for round 2.
#   A match is read only once it holds its decisive games; each is fitted
#   (tools/elo_collect.py) and its games go up as one tarball.
#
# Box: an RTX 4090 with at least 24 usable cores for the 20 workers
# (docs/box_specs.md). Expected wall about 1.2 h: bring-up and tests about
# 15 minutes, each match 11 to 13 minutes (round 1's two matches against
# parity3 at 0 slots: 699 and 654 s for 800 games, none capped,
# training/metrics/memory_own_trajectories_20261011/match.walls), the
# final upload a few minutes. Identical match repeats have differed by
# 1.8x, which would bring it to 1.9 h. Cost at $0.42-0.63 an hour:
# $0.50-0.76 expected; BOX_MAX_H 2.5 caps it at $1.05-1.58 plus the final
# round.
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md): the
# onstart of scripts/rent_box.py fetches the library of STAGE, then this
# script. Every step has a bound; the dead-man's switch finishes the entry
# after BOX_MAX_H hours whatever it is doing. Records go to HF $HF_DIR every
# 30 minutes and after each match, each match's games as one tarball.
# Re-entry, on this machine or a new one (files absent here come back from
# HF), skips fitted matches and resumes a match in its directory. A run
# belongs to the code stage that began it (RUN_STAGE): another stage
# continues it only with RESUME_OTHER_STAGE=1. Every exit past that check,
# clean or not, uploads the records with ALL_DONE last and stops the
# instance; an entry refused there stops it and sends nothing.
# Never `set -x`: the HF token and the instance key are in the environment.
# box-needs: disk_gb=40 ram_gb=48 gpu_ram_gb=24 cores=24 gpu=4090
set -uo pipefail
WORKDIR=/workspace
OUT=$WORKDIR/ownconfirm
STAGE="${STAGE:-}"
# A match with seed base S plays seeds S to S+799 and replaces a capped game
# of slot i with seed S + i + k * 1,000,000; every seed base named in a
# branch or tag of the repository on 2026-10-11 is 108000 or below, and no
# committed result carries a seed from 109000 to 112999.
MATCH_SEED_BASE="${MATCH_SEED_BASE:-109000}"
TUNED_HF="${TUNED_HF:-tier-b/memory_own_trajectories_20261011/own.pt}"
TUNED_SHA256="${TUNED_SHA256:-70925e770d43496e3ea8f9523656f201b71ada228a0362b0eb0a600c8c107f80}"
GAMES="${GAMES:-800}"
JOBS="${JOBS:-20}"
export HF_DIR="${HF_DIR:-tier-b/own_confirm_20261011}"
# Bounds in minutes, from the pre-registration's "Cost".
TESTS_CUT_MIN="${TESTS_CUT_MIN:-20}"             # none of these tests takes 7 s on CI
MATCH_CUT_MIN="${MATCH_CUT_MIN:-90}"             # the match stops itself at 60 (--time-budget-min), a game at 20 more
BOX_MAX_H="${BOX_MAX_H:-2.5}"                    # above the 1.9 box-hours of matches at 1.8x
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }

MATCHES="own0_vs_p3slots0 own64_vs_p3 own64_vs_own0 own64_selfplay"

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
notes() {                        # the matches fitted so far, for the finish reason
    local name fitted=()
    for name in $MATCHES; do
        [ ! -f "$OUT/$name.fit.json" ] || fitted+=("$name")
    done
    echo "fitted: ${fitted[*]:-none}"
}
# shellcheck disable=SC2317 # called by the library, in the reason of an unexpected exit
box_notes() { notes; }
box_on_round() {                 # progress.txt, before each upload round
    local name
    { date -u
      notes
      cat "$OUT/match.walls" 2>/dev/null
      for name in $MATCHES; do
          [ ! -d "$OUT/games_$name" ] \
              || echo "$name: $(find "$OUT/games_$name" -maxdepth 1 -name 'game_*.json' | wc -l) games"
      done
      find "$OUT" -maxdepth 1 -type f -printf '%f ' 2>/dev/null; echo
      tail -n 2 "$OUT"/*.log 2>/dev/null | tail -n 8
    } > "$OUT/progress.txt.tmp" && mv -f "$OUT/progress.txt.tmp" "$OUT/progress.txt"
}

box_init
[ -n "$STAGE" ] || box_finish "NO_STAGE: build the code stage (tools/stage_code.py) and pass STAGE" 1
box_bind_run_stage

# ---- the records of an earlier entry: fits, timings, logs, and each match's games as the tarball it went up as
restored=(match.walls)
for name in $MATCHES; do                 # a match directory already here is not fetched again
    restored+=("$name.fit.json" "timing_$name.txt" "$name.log")
    [ -d "$OUT/games_$name" ] || restored+=("games_$name.tar.gz")
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

# ---- the code and the Rust wheel (the matches run on the core)
box_stage_code || box_finish "CODE_STAGING_FAILED (staging.log)" 1
cd "$BOX_REPO" || box_finish "CODE_STAGING_FAILED (no $BOX_REPO)" 1
box_build_wheel || box_finish "BUILD_FAILED (build.log)" 1
box_facts > "$OUT/box.txt.tmp" 2>&1
mv -f "$OUT/box.txt.tmp" "$OUT/box.txt"
timeout 2m python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || box_finish "NO_CUDA (box.txt)" 1
box_upload_async
box_monitor_start

# ---- the crash barrier: the match path's tests on this wheel and this GPU
if ! box_marked_this_stage "$BOX_STATE/TESTED"; then
    : > "$OUT/tests.log"
    box_bounded tests "$TESTS_CUT_MIN" tests.log python -m pytest tests/test_game_core.py tests/test_vision.py \
        tests/test_parity_integration.py tests/test_sighting_record.py tests/test_faction_posterior.py \
        tests/test_match_memory.py tests/test_raw_player.py tests/test_reference_player.py \
        tests/test_eval_workers.py tests/test_eval_inference_server.py tests/test_eval_match_failures.py \
        tests/test_game_record.py tests/test_elo_collect.py -q -p no:cacheprovider -m ""
    tail -n 3 "$OUT/tests.log"
    [ "$BOX_RC" -eq 0 ] || box_finish "TESTS_FAILED rc=$BOX_RC (tests.log)" 1
    box_mark "$BOX_STATE/TESTED"
fi

# ---- the reference: its checkpoint (SHA-256 checked) and its sides' flags
box_bounded reference 15 staging.log python tools/reference_player.py --ensure \
    || box_finish "REFERENCE_MISSING rc=$BOX_RC (staging.log)" 1
read -r REF_LABEL REF_PT EO < <(timeout 1m python -c "import json; r = json.load(open('configs/reference_player.json')); \
print(r['label'], r['checkpoint_local'], r['decode']['raw_end_turn_offset'])") \
    || box_finish "REFERENCE_CONFIG_UNREADABLE (configs/reference_player.json)" 1
[ "$REF_LABEL" = parity3 ] \
    || box_finish "REFERENCE_CHANGED: configs/reference_player.json names $REF_LABEL; these matches measure parity3" 1
ref_side() {                     # ref_side SIDE LABEL SLOTS: the reference's match flags for SIDE, one per line, under LABEL with its memory at SLOTS
    local side=$1 label=$2 slots=$3 flags i
    mapfile -t flags < <(timeout 1m python tools/reference_player.py --flags "$side" | tr ' ' '\n')
    for (( i = 0; i + 1 < ${#flags[@]}; i++ )); do
        case ${flags[i]} in
            "--label-$side") flags[i + 1]=$label ;;
            "--memory-$side") flags[i + 1]=$slots ;;
        esac
    done
    printf '%s\n' "${flags[@]}"
}
side_ok() {                      # side_ok SIDE SLOTS FLAGS...: the flags play the reference's checkpoint at SLOTS slots under its decode
    local side=$1 slots=$2
    shift 2
    [[ " $* " == *" --spec-$side $REF_PT "* && " $* " == *" --memory-$side $slots "* \
        && " $* " == *" --raw-end-turn-offset-$side $EO "* ]]
}
mapfile -t P3_64_B < <(ref_side b parity3 64)
mapfile -t P3_0_B < <(ref_side b parity3_slots0 0)
if ! { side_ok b 64 "${P3_64_B[@]}" && side_ok b 0 "${P3_0_B[@]}"; }; then
    box_finish "REFERENCE_FLAGS_FAILED (tools/reference_player.py --flags)" 1
fi

# ---- the tuned checkpoint: from HF, its SHA-256 checked
TUNED_PT="$WORKDIR/tuned_own.pt"
if ! { [ -f "$TUNED_PT" ] && echo "$TUNED_SHA256  $TUNED_PT" | sha256sum -c --status; }; then
    box_bounded tuned 20 staging.log python - "$TUNED_HF" "$TUNED_PT" <<'EOF' \
        || box_finish "TUNED_FETCH_FAILED rc=$BOX_RC (staging.log)" 1
import shutil, sys
from huggingface_hub import hf_hub_download
shutil.copyfile(hf_hub_download("momom2/wesnoth-model-checkpoints", sys.argv[1]), sys.argv[2])
EOF
    echo "$TUNED_SHA256  $TUNED_PT" | sha256sum -c --status \
        || box_finish "TUNED_SHA256_MISMATCH ($TUNED_HF)" 1
fi
tuned_side() {                   # tuned_side SIDE LABEL SLOTS: the reference's flags for SIDE with the tuned checkpoint
    local side=$1 label=$2 slots=$3 flags i
    mapfile -t flags < <(ref_side "$side" "$label" "$slots")
    for (( i = 0; i + 1 < ${#flags[@]}; i++ )); do
        [ "${flags[i]}" != "--spec-$side" ] || flags[i + 1]=$TUNED_PT
    done
    printf '%s\n' "${flags[@]}"
}
tuned_ok() {                     # tuned_ok SIDE SLOTS FLAGS...: the flags play the tuned checkpoint at SLOTS under the decode
    local side=$1 slots=$2
    shift 2
    [[ " $* " == *" --spec-$side $TUNED_PT "* && " $* " == *" --memory-$side $slots "* \
        && " $* " == *" --raw-end-turn-offset-$side $EO "* ]]
}
mapfile -t OWN0_A < <(tuned_side a own0 0)
mapfile -t OWN64_A < <(tuned_side a own64 64)
mapfile -t OWN0_B < <(tuned_side b own0 0)
mapfile -t OWN64A_A < <(tuned_side a own64a 64)
mapfile -t OWN64B_B < <(tuned_side b own64b 64)
if ! { tuned_ok a 0 "${OWN0_A[@]}" && tuned_ok a 64 "${OWN64_A[@]}" && tuned_ok b 0 "${OWN0_B[@]}" \
        && tuned_ok a 64 "${OWN64A_A[@]}" && tuned_ok b 64 "${OWN64B_B[@]}"; }; then
    box_finish "TUNED_FLAGS_FAILED (tools/reference_player.py --flags)" 1
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
gpu_or_finish() {                # gpu_or_finish NAME: before match NAME, the GPU answers or the entry ends
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before match $1: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
}
MATCHES_FAILED=0                 # this entry's verdicts on the matches (play)
MATCHES_CUT=0
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
    box_upload_async
}
play own0_vs_p3slots0 "$MATCH_SEED_BASE" "${OWN0_A[@]}" "${P3_0_B[@]}"
play own64_vs_p3 $(( MATCH_SEED_BASE + 1000 )) "${OWN64_A[@]}" "${P3_64_B[@]}"
play own64_vs_own0 $(( MATCH_SEED_BASE + 2000 )) "${OWN64_A[@]}" "${OWN0_B[@]}"
play own64_selfplay $(( MATCH_SEED_BASE + 3000 )) "${OWN64A_A[@]}" "${OWN64B_B[@]}"
box_on_round
[ "$MATCHES_FAILED" -eq 0 ] \
    || box_finish "OWN_CONFIRM_FAILED $(notes); $MATCHES_FAILED failed, $MATCHES_CUT cut (match.walls)" 1
box_finish "OWN_CONFIRM_DONE $(notes); $MATCHES_CUT cut"
