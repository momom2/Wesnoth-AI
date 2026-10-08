#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# Gate matches of the look-ahead player against the reference `parity3`, on
# one box. LOOKAHEAD_ARMS lists the arms as NAME:CONFIG:SEED_BASE triples
# (space-separated), and each arm is one match:
#   side A: `parity3` (configs/reference_player.json, its SHA-256 checked) at
#     64 slots and the reference decode, through the look-ahead player
#     (tools/lookahead_player.py) configured by CONFIG, a file of the
#     repository; labelled la_NAME;
#   side B: `parity3` at 64 slots, raw, at the reference decode
#     (raw:t0+eo-1.5);
#   PURE, 800 decisive games, sides alternated, the Ladder maps with fog and
#   both factions drawn uniformly, seeds SEED_BASE to SEED_BASE + 799 (the
#   replacement of a capped game of slot i plays SEED_BASE + i + k * 1,000,000),
#   every game recorded whole. A match is read only once it holds its
#   decisive games; it is fitted (tools/elo_collect.py), its look-ahead
#   telemetry pooled into NAME.lookahead.json (tools/lookahead_gate.py
#   summarize: flip rate by kind and the kind played instead, evaluator
#   states and seconds per decision, failed expansions by reason), and its
#   games go up as one tarball.
# Before the matches, on the wheel built from the stage: the tests of the
# look-ahead player and of the match path; each arm's configuration loaded,
# a critic's checkpoint fetched from the HF path its configuration names and
# checked against the SHA-256 it records (tools/lookahead_gate.py ensure);
# 50 decisions of each arm timed (tools/lookahead_timing.py, prior on the
# CPU; recorded in NAME.decision_timing.json, not gating).
#
# The arm configured now (run Q7, docs/lookahead_material_gate_prereg_20261009.md):
#   material:configs/lookahead_material_gate.json:103000
# the material evaluator, k 8, c 1, sigma 0.1, every decision kind, the
# observed world, untuned. A later gate (a critic) passes its own
# LOOKAHEAD_ARMS, HF_DIR, MATCH_BUDGET_MIN, MATCH_CUT_MIN and BOX_MAX_H from
# its pre-registration.
#
# Box: an RTX 4090 with at least 24 usable cores for the 20 workers
# (docs/box_specs.md; the parity-memory and anneal matches ran 20 workers on
# 31 usable EPYC cores). Expected wall for the material arm: about 45
# minutes. Bring-up and tests about 15 minutes (8 from the entry to the end
# of the tests on the anneal box, and these tests add the look-ahead's);
# the reference's fetch and the 50 timed decisions about 3; the match about
# 23 minutes: the 944-game raw match of 2026-10-04 took 929 s at 20 workers
# (training/metrics/imitation_anneal_20261003/box/match.walls), and the
# material look-ahead adds about 35 ms of worker CPU per decision (measured
# on a laptop core, docs/lookahead_gate_prereg_draft.md "Cost"), 268
# decisions per game side, so 9.4 s a game and 7.4 minutes over 944 games
# at 20 workers; the fit, the summary and the final upload a few minutes.
# Identical match repeats have differed by 1.8x, which would bring the match
# to 41 minutes and the run to about 1.05 h. Cost at $0.42-0.63 an hour:
# $0.32-0.47 expected, $0.44-0.66 at 1.05 h; BOX_MAX_H 1.5 caps it at
# $0.63-0.95 plus the final round.
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md): the
# onstart of scripts/rent_box.py fetches the library of STAGE, then this
# script. Every step has a bound; the dead-man's switch finishes the entry
# after BOX_MAX_H hours whatever it is doing. Records go to HF $HF_DIR every
# 30 minutes and after each match, each match's games as one tarball.
# Re-entry, on this machine or a new one (files absent here come back from
# HF), skips fitted matches and timed arms and resumes a match in its
# directory. A run belongs to the code stage that began it (RUN_STAGE):
# another stage continues it only with RESUME_OTHER_STAGE=1. Every exit past
# that check, clean or not, uploads the records with ALL_DONE last and stops
# the instance; an entry refused there stops it and sends nothing.
# Never `set -x`: the HF token and the instance key are in the environment.
# box-needs: disk_gb=40 ram_gb=48 gpu_ram_gb=24 cores=24 gpu=4090
set -uo pipefail
WORKDIR=/workspace
OUT=$WORKDIR/lookaheadgate
STAGE="${STAGE:-}"
# Seed bases: the parity3 baselines play 100000 to 102799; every other seed
# base named in the repository's history on 2026-10-09, this run's 103000
# aside, is 90000 or below, and no committed game record has a seed in
# 103000-103999 modulo 1,000,000.
LOOKAHEAD_ARMS="${LOOKAHEAD_ARMS:-material:configs/lookahead_material_gate.json:103000}"
GAMES="${GAMES:-800}"
JOBS="${JOBS:-20}"
SLOTS=64                                         # both sides' memory, the reference's
export HF_DIR="${HF_DIR:-tier-b/lookahead_material_gate_20261009}"
# Bounds in minutes, from the pre-registration's "Cost".
TESTS_CUT_MIN="${TESTS_CUT_MIN:-20}"
ENSURE_CUT_MIN="${ENSURE_CUT_MIN:-15}"           # a critic's download
DECISION_TIMING_CUT_MIN="${DECISION_TIMING_CUT_MIN:-15}"  # about a minute: 170 decisions with the prior on the CPU
MATCH_BUDGET_MIN="${MATCH_BUDGET_MIN:-60}"       # the match stops itself here (--time-budget-min)
MATCH_CUT_MIN="${MATCH_CUT_MIN:-90}"             # ...and a game in flight at 20 more
BOX_MAX_H="${BOX_MAX_H:-1.5}"                    # 2 times the 0.75 box-hours expected; above the 1.05 of a match at 1.8x
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }

# ---- the arms
read -r -a ARM_TRIPLES <<< "$LOOKAHEAD_ARMS"
ARM_NAMES=()
ARM_CONFIGS=()
ARM_SEEDS=()
for triple in ${ARM_TRIPLES[@]+"${ARM_TRIPLES[@]}"}; do
    IFS=: read -r name config sb _ <<< "$triple"
    ARM_NAMES+=("${name:-}")
    ARM_CONFIGS+=("${config:-}")
    ARM_SEEDS+=("${sb:-}")
done
arms_problem() {                 # prints what is wrong with LOOKAHEAD_ARMS, nothing when it is sound
    local i j gap
    [ "${#ARM_NAMES[@]}" -gt 0 ] || { echo "no arm"; return; }
    for (( i = 0; i < ${#ARM_NAMES[@]}; i++ )); do
        [[ ${ARM_TRIPLES[i]} =~ ^[^:]+:[^:]+:[^:]+$ ]] \
            || { echo "'${ARM_TRIPLES[i]}' is not NAME:CONFIG:SEED_BASE"; return; }
        [[ ${ARM_NAMES[i]} =~ ^[A-Za-z0-9_-]+$ ]] \
            || { echo "arm name '${ARM_NAMES[i]}' is not made of letters, digits, _ and -"; return; }
        [[ ${ARM_CONFIGS[i]} != /* && ${ARM_CONFIGS[i]} != *..* ]] \
            || { echo "arm ${ARM_NAMES[i]}: its configuration ${ARM_CONFIGS[i]} is not a path inside the repository"; return; }
        [[ ${ARM_SEEDS[i]} =~ ^(0|[1-9][0-9]*)$ ]] && (( ARM_SEEDS[i] + GAMES <= 1000000 )) \
            || { echo "arm ${ARM_NAMES[i]}: seed base '${ARM_SEEDS[i]}' is not a number (no leading 0) of at most 1,000,000 - $GAMES (replacements add multiples of 1,000,000)"; return; }
        for (( j = 0; j < i; j++ )); do
            [ "${ARM_NAMES[i]}" != "${ARM_NAMES[j]}" ] || { echo "arm ${ARM_NAMES[i]} is named twice"; return; }
            gap=$(( ARM_SEEDS[i] - ARM_SEEDS[j] ))
            (( gap >= GAMES || -gap >= GAMES )) \
                || { echo "arms ${ARM_NAMES[j]} and ${ARM_NAMES[i]} share seeds (bases ${ARM_SEEDS[j]} and ${ARM_SEEDS[i]}, $GAMES games each)"; return; }
        done
    done
}

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
    for name in "${ARM_NAMES[@]}"; do
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
      for name in "${ARM_NAMES[@]}"; do
          [ ! -d "$OUT/games_$name" ] \
              || echo "$name: $(find "$OUT/games_$name" -maxdepth 1 -name 'game_*.json' | wc -l) games"
      done
      find "$OUT" -maxdepth 1 -type f -printf '%f ' 2>/dev/null; echo
      tail -n 2 "$OUT"/*.log 2>/dev/null | tail -n 8
    } > "$OUT/progress.txt.tmp" && mv -f "$OUT/progress.txt.tmp" "$OUT/progress.txt"
}

box_init
[ -n "$STAGE" ] || box_finish "NO_STAGE: build the code stage (tools/stage_code.py) and pass STAGE" 1
problem=$(arms_problem)
[ -z "$problem" ] || box_finish "BAD_ARMS: $problem (LOOKAHEAD_ARMS='$LOOKAHEAD_ARMS')" 2
box_bind_run_stage

# ---- the records of an earlier entry: fits, timings, summaries, logs, and each match's games as the tarball it went up as
restored=(match.walls arms.txt)
for name in "${ARM_NAMES[@]}"; do         # a match directory already here is not fetched again
    restored+=("$name.fit.json" "timing_$name.txt" "$name.log" "$name.lookahead.json" "$name.decision_timing.json")
    [ -d "$OUT/games_$name" ] || restored+=("games_$name.tar.gz")
done
box_restore "${restored[@]}" || box_finish "RESTORE_FAILED (restore.log)" 1
for name in "${ARM_NAMES[@]}"; do
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

# ---- the code and the Rust wheel (the matches and the look-ahead run on the core)
box_stage_code || box_finish "CODE_STAGING_FAILED (staging.log)" 1
cd "$BOX_REPO" || box_finish "CODE_STAGING_FAILED (no $BOX_REPO)" 1
box_build_wheel || box_finish "BUILD_FAILED (build.log)" 1
box_facts > "$OUT/box.txt.tmp" 2>&1
mv -f "$OUT/box.txt.tmp" "$OUT/box.txt"
timeout 2m python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || box_finish "NO_CUDA (box.txt)" 1
box_upload_async
box_monitor_start

# ---- the crash barrier: the look-ahead's and the match path's tests on this wheel and this GPU
if ! box_marked_this_stage "$BOX_STATE/TESTED"; then
    : > "$OUT/tests.log"
    box_bounded tests "$TESTS_CUT_MIN" tests.log python -m pytest tests/test_lookahead_player.py \
        tests/test_lookahead_gate.py tests/test_combat_outcomes.py tests/test_game_core.py tests/test_vision.py \
        tests/test_parity_integration.py tests/test_sighting_record.py tests/test_faction_posterior.py \
        tests/test_match_memory.py tests/test_raw_player.py tests/test_reference_player.py \
        tests/test_eval_workers.py tests/test_eval_inference_server.py tests/test_eval_match_failures.py \
        tests/test_game_record.py tests/test_elo_collect.py -q -p no:cacheprovider -m ""
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
mapfile -t REF_B < <(ref_side b "$REF_LABEL" "$SLOTS")
side_ok b "$SLOTS" "${REF_B[@]}" || box_finish "REFERENCE_FLAGS_FAILED (tools/reference_player.py --flags b)" 1

# ---- each arm: its configuration ready (a critic fetched and checked), its procedure tag, 50 decisions timed
for (( i = 0; i < ${#ARM_NAMES[@]}; i++ )); do
    name=${ARM_NAMES[i]} config=${ARM_CONFIGS[i]}
    box_bounded "ensure $name" "$ENSURE_CUT_MIN" staging.log python tools/lookahead_gate.py ensure "$config" \
        || box_finish "ARM_NOT_READY $name rc=$BOX_RC: $config (staging.log)" 1
    tag=$(timeout 5m python tools/lookahead_gate.py ensure "$config" 2>> "$OUT/staging.log") \
        || box_finish "ARM_NOT_READY $name: $config (staging.log)" 1
    echo "$(box_stamp) arm $name: config $config (sha256 $(sha256sum "$config" | cut -c1-64)), seed base ${ARM_SEEDS[i]}, procedure $tag, label la_$name" \
        | tee -a "$OUT/arms.txt"
    if [ ! -f "$OUT/$name.decision_timing.json" ]; then
        box_bounded "decision timing $name" "$DECISION_TIMING_CUT_MIN" "$name.decision_timing.log" \
            python tools/lookahead_timing.py --checkpoint "$REF_PT" --config "$config" \
            --end-turn-offset "$EO" --decisions 50 --out "$OUT/$name.decision_timing.json" \
            || echo "the decision timing of $name did not finish: rc=$BOX_RC $BOX_WHY ($name.decision_timing.log); not gating" \
                | tee -a "$OUT/match.walls"
    fi
done
box_upload_async

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
            --time-budget-min "$MATCH_BUDGET_MIN"
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
SUMMARIES_FAILED=0
play() {                         # play NAME SEED_BASE ARGS...: the match, once more when short, its verdict, its telemetry
    local name="$1" sb="$2" decisive rc
    shift 2
    box_upload_dir "games_$name" "$OUT/games_$name"
    box_upload_hold "$name.fit.json" "games_$name.tar.gz"
    box_upload_hold "timing_$name.txt" "games_$name.tar.gz"
    box_upload_hold "$name.lookahead.json" "games_$name.tar.gz"
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
    # The telemetry of every game played, whatever the verdict.
    if [ -d "$OUT/games_$name" ] && ! box_bounded "summary $name" 10 "$name.log" \
            python tools/lookahead_gate.py summarize "$OUT/games_$name" --out "$OUT/$name.lookahead.json"; then
        echo "SUMMARY_FAILED $name: rc=$BOX_RC $BOX_WHY ($name.log)" | tee -a "$OUT/match.walls"
        SUMMARIES_FAILED=$(( SUMMARIES_FAILED + 1 ))
    fi
    box_upload_async
}
for (( i = 0; i < ${#ARM_NAMES[@]}; i++ )); do
    name=${ARM_NAMES[i]}
    mapfile -t ARM_A < <(ref_side a "la_$name" "$SLOTS")
    side_ok a "$SLOTS" "${ARM_A[@]}" || box_finish "REFERENCE_FLAGS_FAILED (tools/reference_player.py --flags a)" 1
    play "$name" "${ARM_SEEDS[i]}" "${ARM_A[@]}" --lookahead-a "${ARM_CONFIGS[i]}" "${REF_B[@]}"
done
box_on_round
[ "$(( MATCHES_FAILED + SUMMARIES_FAILED ))" -eq 0 ] \
    || box_finish "LOOKAHEAD_GATE_FAILED $(notes); $MATCHES_FAILED failed, $MATCHES_CUT cut, $SUMMARIES_FAILED summaries failed (match.walls)" 1
box_finish "LOOKAHEAD_GATE_DONE $(notes); $MATCHES_CUT cut"
