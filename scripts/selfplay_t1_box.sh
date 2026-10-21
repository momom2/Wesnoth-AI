#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# Stage 1 of the training design, no training (docs/training_design_20261019/
# current_design.md §4 and §6; build_contracts.md), on one box:
#   the CPU tests of the logging, label and readout path and of the generation
#   path, on the wheel built from the stage, and the reference `own2`
#   (configs/reference_player.json, its SHA-256 checked by the tool);
#   self-play: `own2` against itself at raw:t1 (temperature 1 on both seats,
#   no end_turn offset, 64 memory slots), one network under the labels own2a
#   and own2b, N_SELFPLAY (30,000) games in matches of CHUNK_GAMES (5,000) on
#   the pool configuration (120 games in flight, six a worker process),
#   every decision logged (--log-decisions) and every capped game kept
#   (--max-extra-games 0: run_elo_batch's exit 4, the replacement guard
#   spent, is then a complete match). A game's index is its seed less
#   SP_SEED_BASE: indices below HELDOUT_FROM (20,000) are trained on, the
#   rest held out. Each match is walked into side rows as it lands
#   (tools/selfplay_rows.py), its games sent up as a tarball; then the
#   labels of all of them (tools/pg_labels.py: labels.jsonl, fits.json) and
#   the readouts (tools/selfplay_readout.py: readout.json, whose gates are
#   I-sample, legal mass, I-count, dice control and I-V);
#   D0: `own2` at temperature 1 (no offset, 64 slots) on side A against the
#   reference at its decode (raw:t0+eo-1.5, 64 slots) on side B, D0_GAMES
#   (12,800) decisive games, one network on one server, and its fit
#   (tools/elo_collect.py, the reference at 0);
#   the memory report: `own2` at temperature 1 with 64 slots against 0 slots,
#   MEM_GAMES (3,200) decisive games, and its fit.
# A self-play match that fails stops the self-play (the split needs every
# game); D0 and the memory report are played whatever the self-play did. A
# match cut short by its time budget is resumed once in its directory.
#
# Seeds. Self-play 800000 to 829999, D0 830000 to 842799, the memory report
# 845000 to 848199; a capped D0 or memory game of slot i is replaced by seed
# + k x 1,000,000. Taken by tracked records on 2026-10-22: 300000-312799
# (`own2`'s base match), 500000-506399 and 600000-659999 (the near-tie
# coins), 700000-739999 (recruit rule L), each with its replacements.
#
# Expected wall on a 4090 with 31-32 usable cores, at the pool's steady
# 21,966 games an hour at argmax (ref_p120x6, docs/training_design_20261019/
# analysis/round7_provenance_price_tail.out; the temperature-1 game length
# is unmeasured, and this run measures it): bring-up, wheel and tests 20-25
# minutes; self-play 30,000 games 82 minutes, six matches' start-up and last
# wave 4, their walks 6 (800 games in 9-11 s on 31 workers); the labels and
# the readouts, one pass over 30,000 records, about 10; D0 about 13,000 games
# 36 minutes; the memory report about 3,300 games 9; the fits 2. In all
# about 3.0 hours, $1.7 at $0.577 an hour. BOX_MAX_H 5.5, 1.8 times.
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md): the
# onstart of scripts/rent_box.py fetches the library of STAGE, then this
# script. Every step has a bound; the dead-man's switch finishes the entry
# after BOX_MAX_H hours whatever it is doing. Records go to HF $HF_DIR every
# 30 minutes, after each match and at the finish, which uploads with
# ALL_DONE last and stops the instance. Re-entry restores the records and
# the markers, skips what is done (a done match's games come back from HF
# when its directory is absent), and plays the rest; a self-play match that
# failed stays failed (its .failed file: delete it on HF and on the disk to
# play it again).
# Never `set -x`: the HF token and the instance key are in the environment.
# box-needs: disk_gb=80 ram_gb=64 gpu_ram_gb=24 cores=32 gpu=4090
set -uo pipefail
WORKDIR=/workspace
OUT=$WORKDIR/selfplay_t1
STAGE="${STAGE:-}"
# ---- the configuration (begin): tests/test_selfplay_t1_box.py reads the defaults from here
SP_SEED_BASE="${SP_SEED_BASE:-800000}"
N_SELFPLAY="${N_SELFPLAY:-30000}"
CHUNK_GAMES="${CHUNK_GAMES:-5000}"
HELDOUT_FROM="${HELDOUT_FROM:-20000}"
D0_SEED_BASE="${D0_SEED_BASE:-830000}"
D0_GAMES="${D0_GAMES:-12800}"
MEM_SEED_BASE="${MEM_SEED_BASE:-845000}"
MEM_GAMES="${MEM_GAMES:-3200}"
JOBS="${JOBS:-120}"
WORKER_GAMES="${WORKER_GAMES:-6}"
MIN_FREE_MB="${MIN_FREE_MB:-300}"
GAMES_PER_H="${GAMES_PER_H:-21966}"            # the pool's steady rate, for the steps' bounds
BOX_MAX_H="${BOX_MAX_H:-5.5}"
# ---- the configuration (end)
BOX_WORKER_GB="${BOX_WORKER_GB:-0.5}"           # a walking worker holds about 230 MB (match_luck_box.sh)
TESTS_CUT_MIN="${TESTS_CUT_MIN:-40}"
WALK_CUT_MIN="${WALK_CUT_MIN:-30}"
WALK_STALL_MIN="${WALK_STALL_MIN:-10}"         # no row for this long: the walk is cut
LABELS_CUT_MIN="${LABELS_CUT_MIN:-15}"
READOUT_CUT_MIN="${READOUT_CUT_MIN:-60}"
FIT_CUT_MIN="${FIT_CUT_MIN:-15}"
export HF_DIR="${HF_DIR:-tier-b/selfplay_t1_20261022}"
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }

CHUNKS=()                        # the self-play matches: names, games, offsets from SP_SEED_BASE (the plan)
CHUNK_SIZES=()
CHUNK_OFFSETS=()
RESTORE=()
SELFPLAY="not run"
READOUT="not read"
MEASURES=()

# shellcheck disable=SC2317 # called by the library, in the reason of an unexpected exit
box_notes() { notes; }
notes() {                        # what is done, for a finish reason
    local done_=0 name
    for name in "${CHUNKS[@]}"; do
        ! box_marked_this_stage "$OUT/chunk_$name.done" || done_=$(( done_ + 1 ))
    done
    echo "self-play: $done_ of ${#CHUNKS[@]} matches; readout: $READOUT; ${MEASURES[*]:-no measure}"
}
box_on_round() {                 # progress.txt, before each upload round
    { date -u
      notes
      tail -n 2 "$OUT/matches.txt" 2>/dev/null
    } > "$OUT/progress.txt.tmp" && mv -f "$OUT/progress.txt.tmp" "$OUT/progress.txt"
}

box_init
[ -n "$STAGE" ] || box_finish "NO_STAGE: build the code stage (tools/stage_code.py) and pass STAGE" 1

# ---- the plan (begin): tests/test_selfplay_t1_box.py runs this block and the stages with the library stubbed
plan_chunks() {
    local off=0 c=0 size
    while (( off < N_SELFPLAY )); do
        size=$CHUNK_GAMES
        (( off + size <= N_SELFPLAY )) || size=$(( N_SELFPLAY - off ))
        CHUNKS+=("$(printf 's%02d' "$c")")
        CHUNK_SIZES+=("$size")
        CHUNK_OFFSETS+=("$off")
        off=$(( off + size ))
        c=$(( c + 1 ))
    done
}
for v in SP_SEED_BASE N_SELFPLAY CHUNK_GAMES HELDOUT_FROM D0_SEED_BASE D0_GAMES MEM_SEED_BASE MEM_GAMES JOBS \
         WORKER_GAMES GAMES_PER_H; do
    if ! [[ ${!v} =~ ^[1-9][0-9]*$ ]]; then
        box_finish "BAD_PLAN: $v=${!v} is not a positive integer" 1
    fi
done
(( HELDOUT_FROM < N_SELFPLAY )) || box_finish "BAD_PLAN: HELDOUT_FROM=$HELDOUT_FROM leaves no held-out game" 1
(( SP_SEED_BASE + N_SELFPLAY <= D0_SEED_BASE && D0_SEED_BASE + D0_GAMES <= MEM_SEED_BASE
   && MEM_SEED_BASE + MEM_GAMES <= 1000000 )) \
    || box_finish "BAD_PLAN: the seed ranges overlap or pass 1,000,000 (a replacement's offset)" 1
plan_chunks
RESTORE=(matches.txt rows.jsonl labels.jsonl fits.json readout.json readout.done labels.log readout.log)
for name in "${CHUNKS[@]}"; do
    RESTORE+=("walk_$name.jsonl" "rows_$name.jsonl" "chunk_$name.done" "chunk_$name.failed")
done
for name in d0 memory; do
    RESTORE+=("$name.fit.json" "measure_$name.done" "measure_$name.failed")
done
# ---- the plan (end)

box_bind_run_stage
box_restore "${RESTORE[@]}" || box_finish "RESTORE_FAILED (restore.log)" 1
box_pip huggingface_hub psutil pytest scipy requests numpy || echo "pip install failed (pip.log)"

# ---- the code, the Rust wheel, the reference
box_stage_code || box_finish "CODE_STAGING_FAILED (staging.log)" 1
cd "$BOX_REPO" || box_finish "CODE_STAGING_FAILED (no $BOX_REPO)" 1
box_build_wheel || box_finish "BUILD_FAILED (build.log)" 1
box_facts > "$OUT/box.txt.tmp" 2>&1
mv -f "$OUT/box.txt.tmp" "$OUT/box.txt"
timeout 2m python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || box_finish "NO_CUDA (box.txt)" 1
WALK_JOBS=$(box_workers)
echo "walk jobs $WALK_JOBS" >> "$OUT/box.txt"
box_upload_async
box_bounded reference 15 staging.log python tools/reference_player.py --ensure \
    || box_finish "REFERENCE_MISSING rc=$BOX_RC (staging.log)" 1

# ---- the CPU tests of the logging, label and readout path and of the generation path, on this wheel: the gate
if ! box_marked_this_stage "$BOX_STATE/TESTED"; then
    : > "$OUT/tests.log"
    box_bounded tests "$TESTS_CUT_MIN" tests.log python -m pytest tests/test_decision_log.py \
        tests/test_decision_log_record.py tests/test_selfplay_rows.py tests/test_pg_labels.py \
        tests/test_selfplay_readout.py tests/test_selfplay_t1_box.py tests/test_match_luck.py \
        tests/test_game_record.py tests/test_raw_player.py tests/test_raw_player_compact.py \
        tests/test_end_turn_offset.py tests/test_reference_player.py tests/test_match_memory.py \
        tests/test_server_priors.py tests/test_eval_workers.py tests/test_eval_inference_server.py \
        tests/test_eval_match_failures.py tests/test_generation_identity.py tests/test_elo_collect.py \
        tests/test_cli_help.py -q -p no:cacheprovider -m ""
    tail -n 3 "$OUT/tests.log"
    [ "$BOX_RC" -eq 0 ] || box_finish "TESTS_FAILED rc=$BOX_RC (tests.log)" 1
    box_mark "$BOX_STATE/TESTED"
fi
box_monitor_start

# ---- the sides (begin): one network, `own2`, by its path on every side, so each match serves one server
read -r REF_LABEL REF_REL SLOTS EO < <(timeout 1m python -c "import json; r = json.load(open('configs/reference_player.json')); \
print(r['label'], r['checkpoint_local'], r['memory_slots'], r['decode']['raw_end_turn_offset'])" | tr -d '\r') \
    || box_finish "REFERENCE_CONFIG_UNREADABLE (configs/reference_player.json)" 1
[ "$REF_LABEL" = own2 ] \
    || box_finish "REFERENCE_CHANGED: configs/reference_player.json names $REF_LABEL; stage 1 plays own2 (pi_0)" 1
[ "$SLOTS" = 64 ] \
    || box_finish "REFERENCE_CHANGED: the reference plays $SLOTS memory slots; stage 1 plays 64" 1
REF_PT=$BOX_REPO/$REF_REL
[ -f "$REF_PT" ] || box_finish "REFERENCE_MISSING ($REF_PT)" 1
side_flags() {                   # side_flags SIDE LABEL [SLOTS]: the reference's flags under LABEL, by path, at SLOTS, without its end_turn offset
    local side=$1 label=$2 slots=${3:-} flags out=() i
    mapfile -t flags < <(timeout 1m python tools/reference_player.py --flags "$side" | tr -d '\r' | tr ' ' '\n')
    for (( i = 0; i + 1 < ${#flags[@]}; i += 2 )); do
        case ${flags[i]} in
            "--label-$side") out+=("${flags[i]}" "$label") ;;
            "--spec-$side") out+=("${flags[i]}" "$REF_PT") ;;
            "--memory-$side") out+=("${flags[i]}" "${slots:-${flags[i + 1]}}") ;;
            "--raw-end-turn-offset-$side") ;;
            *) out+=("${flags[i]}" "${flags[i + 1]}") ;;
        esac
    done
    printf '%s\n' "${out[@]}"
}
reference_flags() {              # reference_flags SIDE: the reference at its decode, its checkpoint by path
    local side=$1 flags i
    mapfile -t flags < <(timeout 1m python tools/reference_player.py --flags "$side" | tr -d '\r' | tr ' ' '\n')
    for (( i = 0; i + 1 < ${#flags[@]}; i++ )); do
        [ "${flags[i]}" != "--spec-$side" ] || flags[i + 1]=$REF_PT
    done
    printf '%s\n' "${flags[@]}"
}
mapfile -t SP_A < <(side_flags a own2a)
mapfile -t SP_B < <(side_flags b own2b)
mapfile -t D0_A < <(side_flags a own2t1)
mapfile -t D0_B < <(reference_flags b)
mapfile -t MEM_A < <(side_flags a own2m64 64)
mapfile -t MEM_B < <(side_flags b own2m0 0)
sides_ok() {                     # every side plays own2 by its path at 64 slots (the memory report's B at 0), no offset but the reference's
    local sp=" ${SP_A[*]} ${SP_B[*]} " d0a=" ${D0_A[*]} " d0b=" ${D0_B[*]} " mem=" ${MEM_A[*]} ${MEM_B[*]} "
    [[ $sp == *" --spec-a $REF_PT --memory-a 64 "* && $sp == *" --label-b own2b --spec-b $REF_PT --memory-b 64 "*
       && $d0a == *" --spec-a $REF_PT --memory-a 64 "* && $d0b == *" --spec-b $REF_PT --memory-b 64 "*
       && $d0b == *" --raw-end-turn-offset-b $EO "* && $mem == *" --memory-a 64 "* && $mem == *" --memory-b 0 "*
       && "$sp$d0a$mem" != *"--raw-end-turn-offset"* ]]
}
sides_ok || box_finish "SIDE_FLAGS_FAILED (tools/reference_player.py --flags): ${SP_A[*]} | ${D0_B[*]} | ${MEM_B[*]}" 1
# ---- the sides (end)

# ---- the stages (begin)
POOL=(--persistent-workers --shared-inference --no-infer-compile --device cuda --jobs "$JOBS"
      --worker-games "$WORKER_GAMES" --min-free-mb "$MIN_FREE_MB")
T1=(--mcts-sims 0 --raw-temperature-a 1 --raw-temperature-b 1)
count_games() {                  # count_games DIR: "<result files> <decisive>"
    timeout 5m python - "$1" <<'EOF' | tr -d '\r'
import json, pathlib, sys
files = list(pathlib.Path(sys.argv[1]).glob("game_*.json"))
decisive = 0
for p in files:
    try:
        decisive += json.loads(p.read_text(encoding="utf-8")).get("outcome_a") in ("win", "loss")
    except Exception:
        pass
print(len(files), decisive)
EOF
}
play() {                         # play NAME GAMES SEED_BASE EXTRA FLAGS...: one attempt; PLAYED: complete, short or failed
    local name=$1 games=$2 base=$3 extra=$4 dir="$OUT/games_$1" est t0 rc failed f
    shift 4
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before match $name: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    est=$(( (games + extra / 4) * 60 / GAMES_PER_H + 1 ))     # minutes at the steady rate
    t0=$(box_now)
    box_bounded "match $name" "$(( 2 * est + 15 ))" "match_$name.log" python tools/run_elo_batch.py "$@" \
        --outdir "$dir" --games "$games" --max-extra-games "$extra" --seed-base "$base" "${POOL[@]}" \
        --time-budget-min "$(( 3 * est / 2 + 10 ))"
    rc=$BOX_RC
    failed=$(find "$dir" -maxdepth 1 -name 'failed_*.json' 2>/dev/null | wc -l)
    echo "$(box_stamp) $name: games $games from seed $base, rc=$rc $BOX_WHY, $(( $(box_now) - t0 )) s," \
         "results and decisive $(count_games "$dir"), failed $failed" >> "$OUT/matches.txt"
    for f in "$dir"/.inference_server_*.json "$dir"/.worker_stages.json; do
        [ ! -f "$f" ] || cp -f "$f" "$OUT/$name.${f##*/.}"
    done
    # run_elo_batch: 0 every slot decisive; 4 every slot played, some
    # without a result and no replacement left (with --max-extra-games 0,
    # every capped game kept); 3 or a cut, games left to play. Any other
    # exit, or a failed game, fails the match.
    PLAYED=failed
    if [ "$failed" -eq 0 ]; then
        case $rc in
            0|4) PLAYED=complete ;;
            3|124) PLAYED=short ;;
        esac
    fi
}
play_through() {                 # play_through NAME GAMES SEED_BASE EXTRA FLAGS...: a match, resumed once when cut short
    play "$@"
    [ "$PLAYED" != short ] || play "$@"
    [ "$PLAYED" = complete ]
}
restore_games() {                # restore_games NAME: a done match's games, back from HF when its directory is absent
    local name=$1
    [ ! -d "$OUT/games_$name" ] || return 0
    if ! box_restore "games_$name.tar.gz" || [ ! -f "$OUT/games_$name.tar.gz" ]; then
        return 1
    fi
    timeout -k 30s 30m tar -xzf "$OUT/games_$name.tar.gz" -C "$OUT" || return 1
    rm -f "$OUT/games_$name.tar.gz"
    box_mark_landed "games_$name.tar.gz" "$OUT/games_$name" || echo "games_$name will go up again (restore.log)"
}
rows_of() {                      # rows_of NAME: the luck walk and the side rows of games_NAME
    touch "$OUT/walk_$1.jsonl"
    box_bounded --stall "$OUT/walk_$1.jsonl" "$WALK_STALL_MIN" "rows $1" "$WALK_CUT_MIN" rows.log \
        python tools/selfplay_rows.py --games "$OUT/games_$1" --walk "$OUT/walk_$1.jsonl" \
        --out "$OUT/rows_$1.jsonl" --seed-base "$SP_SEED_BASE" --jobs "$WALK_JOBS"
}
selfplay() {                     # the self-play matches, each walked as it lands; SELFPLAY says how it ended
    local k name base
    for k in "${!CHUNKS[@]}"; do
        name=${CHUNKS[$k]}
        base=$(( SP_SEED_BASE + CHUNK_OFFSETS[k] ))
        if box_marked_this_stage "$OUT/chunk_$name.done"; then
            continue
        elif [ -f "$OUT/chunk_$name.failed" ]; then
            SELFPLAY="FAILED: $name failed at an earlier entry, $(cat "$OUT/chunk_$name.failed")"
            return 1
        fi
        if ! play_through "$name" "${CHUNK_SIZES[$k]}" "$base" 0 "${SP_A[@]}" "${SP_B[@]}" "${T1[@]}" --log-decisions; then
            echo "the match ended $PLAYED (matches.txt, match_$name.log)" > "$OUT/chunk_$name.failed"
        elif ! rows_of "$name"; then
            echo "its rows failed, rc=$BOX_RC $BOX_WHY (rows.log)" > "$OUT/chunk_$name.failed"
        fi
        if [ -f "$OUT/chunk_$name.failed" ]; then
            SELFPLAY="FAILED: $name, $(cat "$OUT/chunk_$name.failed")"
            return 1
        fi
        box_upload_dir "games_$name" "$OUT/games_$name"
        box_mark "$OUT/chunk_$name.done"
        box_upload_async
    done
    SELFPLAY="DONE: ${#CHUNKS[@]} matches, $N_SELFPLAY games"
}
gates() {                        # gates: the readout's gates, one word each
    timeout 2m python - "$OUT/readout.json" <<'EOF' 2>/dev/null | tr -d '\r' || echo "unreadable"
import json, sys
r = json.load(open(sys.argv[1]))
print(" ".join(f"{k}={'pass' if r[k]['pass'] else 'FAIL'}"
               for k in ("i_sample", "legal_mass", "i_count", "dice_control", "i_v") if k in r))
EOF
}
read_selfplay() {                # the labels and the readouts of every self-play match; READOUT says what they read
    local name dirs=() walks=() rows=()
    if box_marked_this_stage "$OUT/readout.done"; then
        READOUT=$(gates)
        return 0
    fi
    for name in "${CHUNKS[@]}"; do
        restore_games "$name" || { READOUT="FAILED: games_$name could not be restored (restore.log)"; return 1; }
        dirs+=("$OUT/games_$name")
        walks+=("$OUT/walk_$name.jsonl")
        rows+=("$OUT/rows_$name.jsonl")
    done
    cat "${rows[@]}" > "$OUT/rows.jsonl.tmp" && mv -f "$OUT/rows.jsonl.tmp" "$OUT/rows.jsonl"
    box_bounded labels "$LABELS_CUT_MIN" labels.log python tools/pg_labels.py "$OUT/rows.jsonl" \
        --heldout-from "$HELDOUT_FROM" --out "$OUT/labels.jsonl" --fits "$OUT/fits.json"
    # pg_labels exits 1 when a game had no luck walk: it is left out, named in fits.json.
    if [ "$BOX_RC" -gt 1 ] || [ ! -f "$OUT/labels.jsonl" ]; then
        READOUT="FAILED: the labels, rc=$BOX_RC $BOX_WHY (labels.log)"
        return 1
    fi
    box_bounded readout "$READOUT_CUT_MIN" readout.log python tools/selfplay_readout.py --games "${dirs[@]}" \
        --labels "$OUT/labels.jsonl" --fits "$OUT/fits.json" --walk "${walks[@]}" --out "$OUT/readout.json"
    # selfplay_readout exits 1 when a gate fails: readout.json says which.
    if [ "$BOX_RC" -gt 1 ] || [ ! -f "$OUT/readout.json" ]; then
        READOUT="FAILED: the readout, rc=$BOX_RC $BOX_WHY (readout.log)"
        return 1
    fi
    box_mark "$OUT/readout.done"
    READOUT=$(gates)
    box_upload_async
}
elo_of() {                       # elo_of NAME LABEL: LABEL's PURE Elo in NAME's fit
    timeout 1m python - "$OUT/$1.fit.json" "$2" <<'EOF' 2>/dev/null | tr -d '\r' || echo "unread"
import json, sys
row = json.load(open(sys.argv[1]))["tables"]["PURE (decisive only, primary)"][sys.argv[2]]
print(f"{row['elo']:+.1f} +- {row['se']:.1f}")
EOF
}
measure() {                      # measure NAME GAMES SEED_BASE ANCHOR LABEL FLAGS...: a match of decisive games and its fit
    local name=$1 games=$2 base=$3 anchor=$4 label=$5
    shift 5
    if ! box_marked_this_stage "$OUT/measure_$name.done"; then
        if [ -f "$OUT/measure_$name.failed" ]; then
            MEASURES+=("$name: FAILED at an earlier entry, $(cat "$OUT/measure_$name.failed");")
            return 1
        fi
        if ! play_through "$name" "$games" "$base" $(( games / 2 )) "$@"; then
            echo "the match ended $PLAYED (matches.txt, match_$name.log)" > "$OUT/measure_$name.failed"
        elif ! box_bounded "fit $name" "$FIT_CUT_MIN" "fit_$name.log" python tools/elo_collect.py \
                "$OUT/games_$name" --no-catalog --anchor "$anchor" --save-json "$OUT/$name.fit.json"; then
            echo "its fit failed, rc=$BOX_RC $BOX_WHY (fit_$name.log)" > "$OUT/measure_$name.failed"
        fi
        [ ! -d "$OUT/games_$name" ] || box_upload_dir "games_$name" "$OUT/games_$name"
        if [ -f "$OUT/measure_$name.failed" ]; then
            MEASURES+=("$name: FAILED, $(cat "$OUT/measure_$name.failed");")
            return 1
        fi
        box_mark "$OUT/measure_$name.done"
        box_upload_async
    fi
    MEASURES+=("$name: $label $(elo_of "$name" "$label") Elo against $anchor;")
}

selfplay && read_selfplay
measure d0 "$D0_GAMES" "$D0_SEED_BASE" "$REF_LABEL" own2t1 "${D0_A[@]}" "${D0_B[@]}" \
    --mcts-sims 0 --raw-temperature-a 1 --raw-temperature-b 0
measure memory "$MEM_GAMES" "$MEM_SEED_BASE" own2m0 own2m64 "${MEM_A[@]}" "${MEM_B[@]}" "${T1[@]}"
REASON="self-play: $SELFPLAY; readout: $READOUT; ${MEASURES[*]}"
if [[ $SELFPLAY == DONE* && $READOUT != *FAIL* && " ${MEASURES[*]} " != *FAILED* ]]; then
    box_finish "SELFPLAY_T1_DONE $REASON"
fi
box_finish "SELFPLAY_T1_INCOMPLETE $REASON" 1
# ---- the stages (end)
