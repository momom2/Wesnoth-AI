#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# Stage S0(b) of the training design (docs/own2_base_prereg_20261020.md,
# docs/training_design_20261019/current_design.md section 4), on one box:
#   the tests of the match path (as the confirmation ran them), on the wheel
#   built from the stage;
#   `own2` from HF (OWN2_HF, its SHA-256 checked against OWN2_SHA256) and
#   the reference `own1` (configs/reference_player.json, its SHA-256
#   checked);
#   one match, PURE, `own2` at 64 slots on side A against `own1` at 64
#   slots on side B, both at the reference decode (raw:t0+eo-1.5), the
#   Ladder maps with both factions drawn uniformly, GAMES (12,800) decisive
#   games from seed base MATCH_SEED_BASE (300000), every game recorded
#   whole; then its fit (tools/elo_collect.py, `own1` anchored at 0) and the
#   pre-registered reading (reading.json, and the finish reason).
# A short attempt (a game failed, a server died, the memory guard) is played
# once more in its directory while the dead-man's switch leaves time for it.
#
# Seeds: the match plays seeds S to S+12,799 and replaces a capped game of
# slot i with seed S + i + k * 1,000,000, so its seeds modulo 1,000,000 are
# S to S+12,799. On 2026-10-20 (origin/main 41e2f05) the highest seed base
# named in a tracked file is 250000 (an 800-game match), and no tracked
# file names a seed (a `"seed":` field or a game file's name) from 300000
# to 312799 modulo 1,000,000.
#
# Box: an RTX 4090 with at least 24 usable cores for the 20 workers
# (docs/box_specs.md). Expected wall about 4.4 h: entry to the first game
# 132 s at the confirmation (2026-10-11); the match 4.25 h, 16 times round
# 2's R1, the same two checkpoints at the same decode (956 s for 800
# decisive games of 801 on an EPYC 7702 box,
# training/metrics/memory_own_round2_20261011/match.walls); the fit about a
# minute (4 s for 800 games); the games' tarball about 80 MB (R1's 5.0 MB
# for 801 games). Cost at $0.42-0.46 an hour: $1.85-2.02. The match stops
# starting games at MATCH_BUDGET_MIN (330 minutes, 1.29 times the expected
# 255), its last games end within MATCH_GRACE_MIN; BOX_MAX_H 6.5, 1.48
# times the expected box-hours, caps the run at $2.73-2.99 plus the final
# round.
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md): the
# onstart of scripts/rent_box.py fetches the library of STAGE, then this
# script. Every step has a bound; the dead-man's switch finishes the entry
# after BOX_MAX_H hours whatever it is doing. Records go to HF $HF_DIR every
# 30 minutes and after the match, the games as one tarball that grows with
# each round. Re-entry, on this machine or a new one (files absent here come
# back from HF), resumes the match in its directory, or only fits and reads
# a complete one. A run belongs to the code stage that began it
# (RUN_STAGE): another stage continues it only with RESUME_OTHER_STAGE=1.
# Every exit past that check, clean or not, uploads the records with
# ALL_DONE last and stops the instance; an entry refused there stops it and
# sends nothing.
# Never `set -x`: the HF token and the instance key are in the environment.
# box-needs: disk_gb=40 ram_gb=48 gpu_ram_gb=24 cores=24 gpu=4090
set -uo pipefail
WORKDIR=/workspace
OUT=$WORKDIR/own2base
STAGE="${STAGE:-}"
# ---- the configuration (begin): tests/test_own2_base_box.py reads the defaults from here
MATCH_SEED_BASE="${MATCH_SEED_BASE:-300000}"
GAMES="${GAMES:-12800}"
MAX_EXTRA="${MAX_EXTRA:-1500}"                   # replacement guard; R1 capped 1 of 801 games, the confirmation 7 of 807 at most
JOBS="${JOBS:-20}"
OWN2_HF="${OWN2_HF:-tier-b/memory_own_round2_20261011/own.pt}"
OWN2_SHA256="${OWN2_SHA256:-aea027b2696a9fbce0ebcdc0a8483c9df7d35957934b81b0c62093a02cdd9129}"
# Bounds in minutes, from the pre-registration's "Cost".
MATCH_BUDGET_MIN="${MATCH_BUDGET_MIN:-330}"      # --time-budget-min of an attempt: no game starts past it
MATCH_GRACE_MIN="${MATCH_GRACE_MIN:-30}"         # the attempt's cut past its budget: a game's 20-minute timeout, the servers' exit
FINISH_RESERVE_MIN="${FINISH_RESERVE_MIN:-15}"   # kept on the switch after an attempt's cut: the fit, the reading, the final round's start
MIN_ATTEMPT_MIN="${MIN_ATTEMPT_MIN:-10}"         # an attempt with less budget than this does not start
BOX_MAX_H="${BOX_MAX_H:-6.5}"
# ---- the configuration (end)
TESTS_CUT_MIN="${TESTS_CUT_MIN:-20}"             # the confirmation's ran in 48 s
export HF_DIR="${HF_DIR:-tier-b/own2_base_20261020}"
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }

NAME=own2_vs_own1
DIR=$OUT/games_$NAME

decisive_results() {             # decisive_results DIR: the games in DIR that ended in a win or a loss
    timeout 5m python - "$1" <<'EOF' | tr -d '\r'
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
notes() {                        # the match's state, for the finish reason
    echo "games: $(find "$DIR" -maxdepth 1 -name 'game_*.json' 2>/dev/null | wc -l), fit: $([ -f "$OUT/$NAME.fit.json" ] && echo yes || echo no)"
}
# shellcheck disable=SC2317 # called by the library, in the reason of an unexpected exit
box_notes() { notes; }
box_on_round() {                 # progress.txt, before each upload round
    { date -u
      notes
      cat "$OUT/match.walls" 2>/dev/null
      find "$OUT" -maxdepth 1 -type f -printf '%f ' 2>/dev/null; echo
      tail -n 3 "$OUT/$NAME.log" 2>/dev/null
    } > "$OUT/progress.txt.tmp" && mv -f "$OUT/progress.txt.tmp" "$OUT/progress.txt"
}

box_init
[ -n "$STAGE" ] || box_finish "NO_STAGE: build the code stage (tools/stage_code.py) and pass STAGE" 1
box_bind_run_stage

# ---- the records of an earlier entry: the fit, the timing, the log, and the games as the tarball they went up as
restored=(match.walls "$NAME.fit.json" "timing_$NAME.txt" "$NAME.log")
[ -d "$DIR" ] || restored+=("games_$NAME.tar.gz")       # a match directory already here is not fetched again
box_restore "${restored[@]}" || box_finish "RESTORE_FAILED (restore.log)" 1
if [ -f "$OUT/games_$NAME.tar.gz" ]; then
    [ -d "$DIR" ] || timeout -k 30s 15m tar -xzf "$OUT/games_$NAME.tar.gz" -C "$OUT" \
        || box_finish "MATCH_RESTORE_FAILED ($NAME)" 1
    rm -f "$OUT/games_$NAME.tar.gz"
    # The directory is what HF holds: it goes up again only once it changes.
    box_mark_landed "games_$NAME.tar.gz" "$DIR" || echo "games_$NAME will go up again (restore.log)"
fi
box_pip huggingface_hub psutil pytest scipy requests || echo "pip install failed (pip.log)"

# ---- the code and the Rust wheel (the match runs on the core)
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

# ---- the two checkpoints: the reference's (SHA-256 checked by the tool) and own2's from HF
box_bounded reference 15 staging.log python tools/reference_player.py --ensure \
    || box_finish "REFERENCE_MISSING rc=$BOX_RC (staging.log)" 1
OWN2_PT="$WORKDIR/own2.pt"
if ! { [ -f "$OWN2_PT" ] && echo "$OWN2_SHA256  $OWN2_PT" | sha256sum -c --status; }; then
    box_bounded own2 20 staging.log python - "$OWN2_HF" "$OWN2_PT" <<'EOF' \
        || box_finish "OWN2_FETCH_FAILED rc=$BOX_RC (staging.log)" 1
import shutil, sys
from huggingface_hub import hf_hub_download
shutil.copyfile(hf_hub_download("momom2/wesnoth-model-checkpoints", sys.argv[1]), sys.argv[2])
EOF
    echo "$OWN2_SHA256  $OWN2_PT" | sha256sum -c --status \
        || box_finish "OWN2_SHA256_MISMATCH ($OWN2_HF)" 1
fi

# ---- the sides (begin): tests/test_own2_base_box.py runs this block and the match's with the library stubbed
read -r REF_LABEL REF_PT EO SLOTS < <(timeout 1m python -c "import json; r = json.load(open('configs/reference_player.json')); \
print(r['label'], r['checkpoint_local'], r['decode']['raw_end_turn_offset'], r['memory_slots'])" | tr -d '\r') \
    || box_finish "REFERENCE_CONFIG_UNREADABLE (configs/reference_player.json)" 1
[ "$REF_LABEL" = own1 ] \
    || box_finish "REFERENCE_CHANGED: configs/reference_player.json names $REF_LABEL; this match measures own2 against own1" 1
[ "$SLOTS" = 64 ] \
    || box_finish "REFERENCE_CHANGED: the reference plays $SLOTS memory slots; this match measures both sides at 64" 1
ref_flags() {                    # ref_flags SIDE: the reference's match flags for SIDE, one per line
    timeout 1m python tools/reference_player.py --flags "$1" | tr -d '\r' | tr ' ' '\n'
}
side_ok() {                      # side_ok SIDE LABEL SPEC FLAGS...: the flags play SPEC under LABEL at 64 slots at the reference decode
    local side=$1 label=$2 spec=$3
    shift 3
    [[ " $* " == *" --label-$side $label "* && " $* " == *" --spec-$side $spec "* \
        && " $* " == *" --memory-$side 64 "* && " $* " == *" --raw-end-turn-offset-$side $EO "* ]]
}
mapfile -t OWN1_B < <(ref_flags b)
mapfile -t OWN2_A < <(ref_flags a)
for (( i = 0; i + 1 < ${#OWN2_A[@]}; i++ )); do
    case ${OWN2_A[i]} in
        --label-a) OWN2_A[i + 1]=own2 ;;
        --spec-a) OWN2_A[i + 1]=$OWN2_PT ;;
    esac
done
if ! { side_ok b own1 "$REF_PT" "${OWN1_B[@]}" && side_ok a own2 "$OWN2_PT" "${OWN2_A[@]}"; }; then
    box_finish "SIDE_FLAGS_FAILED (tools/reference_player.py --flags)" 1
fi
# ---- the sides (end)

# ---- the match (begin)
match_budget_min() {             # the time budget of an attempt starting now: MATCH_BUDGET_MIN, less when the switch leaves less
    local at left
    at=$(cat "$BOX_OUT/tmp/deadman.at" 2>/dev/null)   # the epoch at which the dead-man's switch fires (boxlib.sh)
    [[ $at =~ ^[0-9]+$ ]] || { echo "$MATCH_BUDGET_MIN"; return; }
    left=$(( (at - $(box_now)) / 60 - MATCH_GRACE_MIN - FINISH_RESERVE_MIN ))
    echo $(( left < MATCH_BUDGET_MIN ? left : MATCH_BUDGET_MIN ))
}
fit() {                          # the fit of the games on disk, own1 anchored at 0
    box_bounded "fit $NAME" 10 "$NAME.log" \
        python tools/elo_collect.py "$DIR" --no-catalog --anchor own1 --save-json "$OUT/$NAME.fit.json"
}
attempt() {                      # attempt BUDGET_MIN: the match's games, resumed in its directory, its timing, then its fit
    local budget=$1 t0 f
    echo "$(date -u +%FT%TZ) $NAME: seed base $MATCH_SEED_BASE, budget $budget min, ${OWN2_A[*]} ${OWN1_B[*]}" >> "$OUT/$NAME.log"
    t0=$(box_now)
    box_bounded "match $NAME" $(( budget + MATCH_GRACE_MIN )) "$NAME.log" \
        python tools/run_elo_batch.py "${OWN2_A[@]}" "${OWN1_B[@]}" \
        --outdir "$DIR" --games "$GAMES" --max-extra-games "$MAX_EXTRA" --seed-base "$MATCH_SEED_BASE" \
        --mcts-sims 0 --raw-temperature-a 0 --raw-temperature-b 0 \
        --persistent-workers --shared-inference --no-infer-compile --device cuda --jobs "$JOBS" \
        --time-budget-min "$budget"
    echo "$NAME: $(( $(box_now) - t0 )) s, $(find "$DIR" -maxdepth 1 -name 'game_*.json' 2>/dev/null | wc -l) games," \
         "$(decisive_results "$DIR") decisive, rc=$BOX_RC $BOX_WHY" | tee "$OUT/timing_$NAME.txt" | tee -a "$OUT/match.walls"
    for f in "$DIR"/.inference_server_*.json; do
        [ -f "$f" ] && cp -f "$f" "$OUT/$NAME.server_${f##*/.inference_server_}"
    done
    fit
}
read_match() {                   # the pre-registered reading of the fit (reading.json); prints its line
    timeout 2m python - "$OUT/$NAME.fit.json" "$OUT/reading.json" "$GAMES" <<'EOF'
import json, math, sys
fit = json.load(open(sys.argv[1]))
pure = fit["tables"]["PURE (decisive only, primary)"]
elo, se = pure["own2"]["elo"], pure["own2"]["se"]
assert pure["own1"]["elo"] == 0.0, "own1 is the anchor"
RECORD, RECORD_SE = 16.5, 12.3           # R1, docs/memory_own_round2_prereg_20261011.md "Measured"
half = 2 * math.hypot(RECORD_SE, se)
lower = elo - 2 * se
reading = {"elo": elo, "se": se, "lower": lower, "upper": elo + 2 * se, "games": int(sys.argv[3]),
           "n_games": fit.get("n_games"), "base": "own2" if lower > 0 else "own1",
           "record_window": [RECORD - half, RECORD + half],
           "agrees_with_record": abs(elo - RECORD) <= half}
json.dump(reading, open(sys.argv[2], "w"), indent=1)
print(f"own2 {elo:+.1f} +- {se:.1f} Elo over own1, lower bound {lower:+.1f}: base {reading['base']}; "
      f"record window [{RECORD - half:+.1f}, {RECORD + half:+.1f}]: "
      f"{'inside' if reading['agrees_with_record'] else 'OUTSIDE'}")
EOF
}
box_upload_dir "games_$NAME" "$DIR"
for f in "$NAME.fit.json" "timing_$NAME.txt" reading.json; do
    box_upload_hold "$f" "games_$NAME.tar.gz"
done
ATTEMPTS=0
NO_TIME=""
while :; do
    decisive=$(decisive_results "$DIR")
    [[ $decisive =~ ^[0-9]+$ ]] || break
    if [ "$decisive" -ge "$GAMES" ]; then
        [ -f "$OUT/$NAME.fit.json" ] || fit       # complete games whose fit an earlier entry did not reach
        break
    fi
    [ "$ATTEMPTS" -lt 2 ] || break
    if [ -f "$OUT/$NAME.fit.json" ] || [ -f "$OUT/timing_$NAME.txt" ]; then
        # A short attempt's fit and timing go, here and on HF, before the next attempt.
        rm -f "$OUT/$NAME.fit.json" "$OUT/timing_$NAME.txt" "$OUT/reading.json"
        box_clear "$NAME.fit.json" "timing_$NAME.txt" reading.json \
            || echo "the short attempt's fit may stay on HF until the next lands (upload.log)"
    fi
    budget=$(match_budget_min)
    if [ "$budget" -lt "$MIN_ATTEMPT_MIN" ]; then
        NO_TIME="the switch leaves $budget minutes of budget, under $MIN_ATTEMPT_MIN"
        break
    fi
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before the match: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    attempt "$budget"
    ATTEMPTS=$(( ATTEMPTS + 1 ))
    box_upload_async
done
box_on_round
rc=$(sed -n 's/.* rc=\([0-9]*\) .*/\1/p' "$OUT/timing_$NAME.txt" 2>/dev/null | tail -n 1)
if ! [[ $decisive =~ ^[0-9]+$ ]]; then
    WHY="MATCH_FAILED $NAME: its games could not be counted ($NAME.log)"
elif [ "$decisive" -ge "$GAMES" ] && [ -f "$OUT/$NAME.fit.json" ]; then
    READING=$(read_match | tr -d '\r') || box_finish "OWN2_BASE_FAILED: READOUT_FAILED on $decisive decisive games (reading.json)" 1
    echo "MATCH_READ $NAME: $READING" | tee -a "$OUT/match.walls"
    box_finish "OWN2_BASE_DONE $READING"
elif [ "$decisive" -ge "$GAMES" ]; then
    WHY="MATCH_FAILED $NAME: $decisive decisive games, no fit ($NAME.log)"
elif [ -n "$NO_TIME" ]; then
    WHY="MATCH_CUT $NAME: $decisive of $GAMES decisive games; $NO_TIME; a re-entry with time resumes it"
else
    # run_elo_batch: 3 or 4, out of time or of replacements, and 124, the
    # step's own cut, leave a match short; any other exit is a defect
    # (1: games failed or a server died; a signal; a usage error).
    case ${rc:-none} in
        0|3|4|124) WHY="MATCH_CUT $NAME: $decisive of $GAMES decisive games (rc=$rc); the fit is not read" ;;
        *) WHY="MATCH_FAILED $NAME: rc=${rc:-none}, $decisive of $GAMES decisive games ($NAME.log, games_$NAME/failed_*.json)" ;;
    esac
fi
echo "$WHY" | tee -a "$OUT/match.walls"
case $WHY in
    MATCH_CUT*) box_finish "OWN2_BASE_CUT after $ATTEMPTS attempts this entry: $WHY" 1 ;;
    *) box_finish "OWN2_BASE_FAILED after $ATTEMPTS attempts this entry: $WHY" 1 ;;
esac
# ---- the match (end)
