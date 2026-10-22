#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# Stage C of the training design, nothing played or kept (docs/
# training_design_20261019/current_design.md §2, §5 and §6; build_contracts.md),
# on one box, from stage 1's records (scripts/selfplay_t1_box.sh, HF
# STAGE1_HF_DIR):
#   the CPU tests of the policy-gradient path on the wheel built from the stage;
#   stage 1's games, walks, labels and readout restored (its gates must have
#   passed), `own2` (pi_0, configs/reference_player.json) ensured;
#   C0: the gradient pass on the first C0_GAMES training games (the replay
#   identity against the logged law and the precision floor, tools/pg_pass.py)
#   and one replay of C0_HELD held-out games at theta_k (tools/pg_eval.py
#   time): the trainer's and the forwards' measured rates, from which the rest
#   is sized (c0_plan, by tools/stage_c_plan.py, the projection stage 1's
#   rate measurement prints);
#   the gradient pass over every training game of stage 1 (20,000 when stage 1
#   ran whole; the 5,000 and 10,000 points of the gain curve are its splits);
#   the stage C driver on it (tools/pg_stage_c.py on tools/pg_eval.py's
#   network_factory): the arms at K 1/4 on half A, the winner's ladder, B, the
#   gain curve, the head-temperature floor and the gate on round 1's
#   candidate; each arm and part fails alone inside it;
#   the planted offline readout (tools/pg_plant.py): delta for 10 SE from the
#   pass's noise, the plant on the first N_PLANT training games written into a
#   copy of the walk's table and read by selfplay_rows, pg_labels and a
#   planted pass, then read back per seat on PLANT_READOUT_FRACTION of the
#   held-out games;
#   the verdict (verdict.json): design §6's decision on the gate and the plant.
# Each part fails alone: a failed C0 or pass stops what needs it (all of what
# follows), a failed stage C driver leaves the plant to run and the other way
# round, and the verdict reads whatever landed. A part's failure stays failed
# at a re-entry (its .failed file: delete it on HF and on the disk to run it
# again); a done part is not run again.
#
# Cost. Inputs: $0.577 an hour (design §4 prices); the trainer at 1,477
# positions a second (training/metrics/trainer_calibration_20261015: the
# sequence trainer in bf16 with the packed, compiled trunk); 475.7 decisions
# a game (the argmax census; stage 1 measures the temperature-1 length and C0
# reads it); a forward costing a third of a trained position to one (the
# design's illustration and bound); stage C's forwards over the 10,000
# held-out games counted from the driver's reads (tools/stage_c_plan.py
# read_plan, which tests/test_pg_stage_c.py checks against the reads the
# driver makes; a KL search priced at 4 readings): 11.25 forward-equivalents
# with every arm, 9.85 trimmed (the critical batches' 68 KL readings on the
# 5 % subsample are 3.4 of them). The projection is tools/stage_c_plan.py's,
# which stage 1 runs on its measured rates (its rates.json) before stage C is
# decided:
#   bring-up, restore, C0 and uploads about 1.0 h; the pass 20,000 x 475.7 /
#   1,477 = 1.79 h; stage C 3.35 h (forwards at 3x) to 8.81 h (at the bound,
#   trimmed); the plant (two 5,000-game passes, grad M over 2,000 held-out
#   games, five forwards of them) 1.37 to 1.97 h. In all 7.5 to 13.6 h, $4.3
#   to $7.8. The pass and the evaluator run fp32 on the padded trunk without
#   compilation, so they may run 2 to 4 times slower than the calibration: at
#   half its rate the run takes about 14.0 h with forwards at 3x ($8.1), 26.1 h
#   at the bound ($15.1). C0 measures both rates; c0_plan trims the arms when
#   the forwards run at the bound (design §6: no heads-only arm, no 5,000-game
#   point) and stops the run, before the pass, when the projected wall does not
#   fit the switch. BOX_MAX_H 15 ($8.66 at most), the recorded rates' bound.
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md): the
# onstart of scripts/rent_box.py fetches the library of STAGE, then this
# script. Every step has a bound, each 1.5 times its estimate from C0's rates
# plus 15 minutes; the dead-man's switch finishes the entry after BOX_MAX_H
# hours whatever it is doing. Records go to HF $HF_DIR every 30 minutes,
# after each part and at the finish, which uploads with ALL_DONE last and
# stops the instance. A part's done marker goes up only after its files have
# landed. Never `set -x`: the HF token and the instance key are in the
# environment.
# box-needs: disk_gb=120 ram_gb=64 gpu_ram_gb=24 cores=32 gpu=4090
set -uo pipefail
WORKDIR=/workspace
OUT=$WORKDIR/stage_c
S1=$WORKDIR/stage1               # stage 1's records, restored from STAGE1_HF_DIR; never uploaded
STAGE="${STAGE:-}"
# ---- the configuration (begin): tests/test_stage_c_box.py reads the defaults from here
STAGE1_HF_DIR="${STAGE1_HF_DIR:-tier-b/selfplay_t1_20261022}"
SP_SEED_BASE="${SP_SEED_BASE:-800000}"         # stage 1's self-play seed base
CHUNK_GAMES="${CHUNK_GAMES:-5000}"             # stage 1's self-play match size
C0_GAMES="${C0_GAMES:-200}"
C0_HELD="${C0_HELD:-100}"
N_PLANT="${N_PLANT:-5000}"
PLANT_READOUT_FRACTION="${PLANT_READOUT_FRACTION:-0.2}"
PLANT_CLASS="${PLANT_CLASS:-attack}"
PRICE_PER_GAME="${PRICE_PER_GAME:-7.76e-5}"    # design §2: c_g, a generated and trained game
ROUND_FIXED="${ROUND_FIXED:-2.42}"             # design §2: c_f's high end
RESERVE_MIN="${RESERVE_MIN:-35}"               # kept for the uploads and the finish when the run is sized
BOX_MAX_H="${BOX_MAX_H:-15}"
# ---- the configuration (end)
BOX_WORKER_GB="${BOX_WORKER_GB:-1.0}"
TESTS_CUT_MIN="${TESTS_CUT_MIN:-40}"
RESTORE_CUT_MIN="${RESTORE_CUT_MIN:-60}"
C0_CUT_MIN="${C0_CUT_MIN:-40}"
SMALL_CUT_MIN="${SMALL_CUT_MIN:-30}"           # the plant's copy, rows and labels
export HF_DIR="${HF_DIR:-tier-b/stage_c_20261022}"
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }

PARTS=(c0 pass stage_c plant)
RESTORE=()
STATE=()                         # the parts' words for the finish

# shellcheck disable=SC2317 # called by the library, in the reason of an unexpected exit
box_notes() { notes; }
notes() { echo "${STATE[*]:-nothing read}"; }
box_on_round() {                 # progress.txt, before each upload round
    { date -u
      notes
    } > "$OUT/progress.txt.tmp" && mv -f "$OUT/progress.txt.tmp" "$OUT/progress.txt"
}

box_init
[ -n "$STAGE" ] || box_finish "NO_STAGE: build the code stage (tools/stage_code.py) and pass STAGE" 1

# ---- the plan (begin): tests/test_stage_c_box.py runs this block and the stages with the library stubbed
for v in SP_SEED_BASE CHUNK_GAMES C0_GAMES C0_HELD N_PLANT RESERVE_MIN; do
    if ! [[ ${!v} =~ ^[1-9][0-9]*$ ]]; then
        box_finish "BAD_PLAN: $v=${!v} is not a positive integer" 1
    fi
done
for v in PLANT_READOUT_FRACTION PRICE_PER_GAME ROUND_FIXED; do
    if ! [[ ${!v} =~ ^[0-9]*\.?[0-9]+(e-?[0-9]+)?$ ]]; then
        box_finish "BAD_PLAN: $v=${!v} is not a positive number" 1
    fi
done
RESTORE=(c0_pass.json c0_pass.pt c0_forwards.json c0_plan.json c0_plan.txt pass.pt stage_c.json plant_grads.pt
         walk_planted.jsonl rows_planted.jsonl labels_planted.jsonl fits_planted.json pass_planted.pt
         plant_readout.json verdict.json)
for part in "${PARTS[@]}"; do
    RESTORE+=("$part.done" "$part.failed")
done
# ---- the plan (end)

box_bind_run_stage
box_restore "${RESTORE[@]}" || box_finish "RESTORE_FAILED (restore.log)" 1
box_pip huggingface_hub psutil pytest scipy requests numpy || echo "pip install failed (pip.log)"

# ---- the code, the Rust wheel, own2
box_stage_code || box_finish "CODE_STAGING_FAILED (staging.log)" 1
cd "$BOX_REPO" || box_finish "CODE_STAGING_FAILED (no $BOX_REPO)" 1
box_build_wheel || box_finish "BUILD_FAILED (build.log)" 1
box_facts > "$OUT/box.txt.tmp" 2>&1
mv -f "$OUT/box.txt.tmp" "$OUT/box.txt"
timeout 2m python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || box_finish "NO_CUDA (box.txt)" 1
WORKERS=$(box_workers)
echo "workers $WORKERS" >> "$OUT/box.txt"
box_upload_async
box_bounded reference 15 staging.log python tools/reference_player.py --ensure \
    || box_finish "REFERENCE_MISSING rc=$BOX_RC (staging.log)" 1
read -r REF_LABEL REF_REL < <(timeout 1m python -c "import json; r = json.load(open('configs/reference_player.json')); \
print(r['label'], r['checkpoint_local'])" | tr -d '\r') \
    || box_finish "REFERENCE_CONFIG_UNREADABLE (configs/reference_player.json)" 1
[ "$REF_LABEL" = own2 ] \
    || box_finish "REFERENCE_CHANGED: configs/reference_player.json names $REF_LABEL; stage 1 played own2 (pi_0)" 1
CKPT=$BOX_REPO/$REF_REL
[ -f "$CKPT" ] || box_finish "REFERENCE_MISSING ($CKPT)" 1

# ---- the CPU tests of the policy-gradient path, on this wheel: the gate
if ! box_marked_this_stage "$BOX_STATE/TESTED"; then
    : > "$OUT/tests.log"
    box_bounded tests "$TESTS_CUT_MIN" tests.log python -m pytest tests/test_pg_pass.py tests/test_pg_eval.py \
        tests/test_pg_positions.py tests/test_pg_loss.py tests/test_pg_step.py tests/test_pg_stage_c.py \
        tests/test_pg_plant.py tests/test_pg_end_to_end.py tests/test_selfplay_rows.py tests/test_pg_labels.py \
        tests/test_decision_log.py tests/test_decision_log_record.py tests/test_stage_c_box.py \
        tests/test_match_luck.py tests/test_game_record.py -q -p no:cacheprovider -m ""
    tail -n 3 "$OUT/tests.log"
    [ "$BOX_RC" -eq 0 ] || box_finish "TESTS_FAILED rc=$BOX_RC (tests.log)" 1
    box_mark "$BOX_STATE/TESTED"
fi
box_monitor_start

# ---- the stages (begin)
PLAN_N=0 HELDOUT_FROM=0 CHUNKS=() DIRS=()
restore_stage1() {               # stage 1's records into S1: its plan, labels, readout, walks and games; false with S1_WHY
    local names=(plan.txt labels.jsonl fits.json readout.json) off c name missing=()
    mkdir -p "$S1"
    box_bounded "restore stage1" "$RESTORE_CUT_MIN" restore.log python "$BOX_LIB/box_upload.py" --out "$S1" \
        --hf-dir "$STAGE1_HF_DIR" --restore "${names[@]}"
    read -r PLAN_N HELDOUT_FROM _ < "$S1/plan.txt" 2>/dev/null || { S1_WHY="no plan.txt"; return 1; }
    if ! [[ $PLAN_N =~ ^[1-9][0-9]*$ && $HELDOUT_FROM =~ ^[1-9][0-9]*$ ]]; then
        S1_WHY="stage 1's plan.txt reads '$PLAN_N $HELDOUT_FROM': its self-play did not run"
        return 1
    fi
    off=0 c=0
    while (( off < PLAN_N )); do
        CHUNKS+=("$(printf 's%02d' "$c")")
        off=$(( off + CHUNK_GAMES ))
        c=$(( c + 1 ))
    done
    for name in "${CHUNKS[@]}"; do
        names+=("walk_$name.jsonl")
        [ -d "$S1/games_$name" ] || names+=("games_$name.tar.gz")
    done
    box_bounded "restore stage1" "$RESTORE_CUT_MIN" restore.log python "$BOX_LIB/box_upload.py" --out "$S1" \
        --hf-dir "$STAGE1_HF_DIR" --restore "${names[@]}"
    for name in "${CHUNKS[@]}"; do
        if [ ! -d "$S1/games_$name" ] && [ -f "$S1/games_$name.tar.gz" ]; then
            # Into a directory of its own, renamed whole on success: a cut or failed extraction leaves no
            # partial games_$name to be counted complete.
            rm -rf "$S1/.extract_$name" && mkdir -p "$S1/.extract_$name"
            if timeout -k 30s 30m tar -xzf "$S1/games_$name.tar.gz" -C "$S1/.extract_$name" \
                    && [ -d "$S1/.extract_$name/games_$name" ]; then
                mv "$S1/.extract_$name/games_$name" "$S1/games_$name" && rm -f "$S1/games_$name.tar.gz"
            fi
            rm -rf "$S1/.extract_$name"
        fi
        [ -d "$S1/games_$name" ] || missing+=("games_$name")
        [ -f "$S1/walk_$name.jsonl" ] || missing+=("walk_$name.jsonl")
        DIRS+=("$S1/games_$name")
    done
    for name in labels.jsonl readout.json; do
        [ -f "$S1/$name" ] || missing+=("$name")
    done
    if (( ${#missing[@]} )); then
        S1_WHY="missing from $STAGE1_HF_DIR: ${missing[*]}"
        return 1
    fi
    for name in "${CHUNKS[@]}"; do cat "$S1/walk_$name.jsonl"; done > "$S1/walk.jsonl.tmp" \
        && mv -f "$S1/walk.jsonl.tmp" "$S1/walk.jsonl"
    S1_WHY=$(timeout 2m python - "$S1/readout.json" <<'EOF' | tr -d '\r'
import json, sys
r = json.load(open(sys.argv[1]))
failed = [k for k in ("i_sample", "legal_mass", "i_count", "dice_control", "i_v") if not r.get(k, {}).get("pass")]
print(("gates failed: " + " ".join(failed)) if failed else "")
EOF
)
    [ -z "$S1_WHY" ]
}
failed_part() {                  # failed_part PART WHY...: the part's .failed file and its word
    local why="${*:2}"
    echo "$why" > "$OUT/$1.failed"
    STATE+=("$1: FAILED, $why;")
}
skip_part() {                    # skip_part PART: true when PART is done (or failed) at an earlier entry
    if box_marked_this_stage "$OUT/$1.done"; then
        STATE+=("$1: done;")
        return 0
    fi
    if [ -f "$OUT/$1.failed" ]; then
        STATE+=("$1: FAILED at an earlier entry, $(cat "$OUT/$1.failed");")
        return 0
    fi
    return 1
}
part_done() {                    # part_done PART FILE...: the marker, held until FILEs land
    box_upload_hold "$1.done" "${@:2}"
    box_mark "$OUT/$1.done"
    STATE+=("$1: done;")
    box_upload_async
}
ok() { box_marked_this_stage "$OUT/$1.done"; }
c0() {                           # C0: the replay identity, the floor and the two rates
    skip_part c0 && return 0
    box_bounded "c0 pass" "$C0_CUT_MIN" c0.log python tools/pg_pass.py --checkpoint "$CKPT" --games "${DIRS[@]}" \
        --labels "$S1/labels.jsonl" --train-limit "$C0_GAMES" --out "$OUT/c0_pass.pt" --device cuda \
        --workers "$WORKERS"
    pass_refused c0 "$C0_GAMES training games" c0.log c0_pass || return 1
    local limit
    limit=$(timeout 1m python -c "import json, sys; print(json.load(open(sys.argv[1]))['replay']['limit'])" \
        "$OUT/c0_pass.json" 2>/dev/null | tr -d '\r')
    [[ $limit =~ ^[0-9.eE+-]+$ ]] || { failed_part c0 "c0_pass.json names no replay limit"; return 1; }
    box_bounded "c0 forwards" "$C0_CUT_MIN" c0.log python tools/pg_eval.py time --checkpoint "$CKPT" \
        --games "${DIRS[@]}" --labels "$S1/labels.jsonl" --held-games "$C0_HELD" --device cuda --workers "$WORKERS" \
        --replay-kl-limit "$limit" --out "$OUT/c0_forwards.json"
    case $BOX_RC in
        0) ;;
        3) failed_part c0 "the evaluator's replay identity failed on $C0_HELD held-out games, limit $limit" \
               "(c0.log, c0_forwards.json)"; return 1 ;;
        *) failed_part c0 "the C0 forwards ended rc=$BOX_RC $BOX_WHY (c0.log)"; return 1 ;;
    esac
    part_done c0 c0_pass.json c0_pass.pt c0_forwards.json
}
pass_refused() {                 # pass_refused PART WHAT LOG STEM: false (PART failed, saying why) unless the pass's BOX_RC is 0
    case $BOX_RC in
        0) return 0 ;;
        3) failed_part "$1" "the replay identity failed on $2 ($3, $4.replay_failed.pt)" ;;
        4) failed_part "$1" "the precision floor's bound lifts the replay limit past its cap: stage 1 should serve" \
               "in fp32 ($3, $4.floor_refused.json)" ;;
        5) failed_part "$1" "the packed and padded trunks disagree in fp32 ($3, $4.path_failed.json)" ;;
        *) failed_part "$1" "the pass ended rc=$BOX_RC $BOX_WHY ($3)" ;;
    esac
    return 1
}
c0_plan() {                      # the arms and the steps' bounds from C0's rates (tools/stage_c_plan.py); false when the run does not fit
    local at left
    if [ ! -f "$OUT/c0_plan.txt" ]; then
        at=$(cat "$BOX_OUT/tmp/deadman.at" 2>/dev/null)
        [[ $at =~ ^[0-9]+$ ]] || at=$(( $(box_now) + 3600 * 24 ))
        left=$(( (at - $(box_now)) / 60 - RESERVE_MIN ))
        timeout 2m python tools/stage_c_plan.py --pass-summary "$OUT/c0_pass.json" --forwards "$OUT/c0_forwards.json" \
            --labels "$S1/labels.jsonl" --n-plant "$N_PLANT" --readout-fraction "$PLANT_READOUT_FRACTION" \
            --left-min "$left" --out "$OUT/c0_plan.json" --line > "$OUT/c0_plan.txt.tmp" || return 1
        mv -f "$OUT/c0_plan.txt.tmp" "$OUT/c0_plan.txt"
    fi
    read -r TRIM PLAN_MIN LEFT_MIN PASS_CUT STAGE_C_CUT PLANT_CUT N_TRAIN _ < "$OUT/c0_plan.txt"
    STATE+=("plan: $PLAN_MIN min projected of $LEFT_MIN, trim=$TRIM;")
    (( PLAN_MIN <= LEFT_MIN ))
}
gradient_pass() {                # the pass over every training game
    skip_part pass && return 0
    box_bounded pass "$PASS_CUT" pass.log python tools/pg_pass.py --checkpoint "$CKPT" --games "${DIRS[@]}" \
        --labels "$S1/labels.jsonl" --out "$OUT/pass.pt" --device cuda --workers "$WORKERS"
    pass_refused pass "$N_TRAIN training games" pass.log pass || return 1
    part_done pass pass.pt
}
stage_c() {                      # the stage C driver: arms, ladder, B, gain curve, floor, gate
    local arms=(sgd reduced_adamw muon heads_only probe_span) splits=(2 4) dirs
    skip_part stage_c && return 0
    if [ "$TRIM" = 1 ]; then
        arms=(sgd reduced_adamw muon probe_span) splits=(4)
    fi
    dirs=$(IFS=,; echo "${DIRS[*]}")
    box_bounded "stage c" "$STAGE_C_CUT" stage_c.log python tools/pg_stage_c.py --pass "$OUT/pass.pt" \
        --factory tools.pg_eval:network_factory --factory-arg "checkpoint=$CKPT" --factory-arg "games=$dirs" \
        --factory-arg "labels=$S1/labels.jsonl" --factory-arg device=cuda --factory-arg "workers=$WORKERS" \
        --price-per-game "$PRICE_PER_GAME" --round-fixed-cost "$ROUND_FIXED" --arms "${arms[@]}" \
        --curve-splits "${splits[@]}" --out "$OUT/stage_c.json"
    if [ "$BOX_RC" -ne 0 ] || [ ! -f "$OUT/stage_c.json" ]; then
        [ ! -f "$OUT/stage_c.json" ] || box_upload_async
        failed_part stage_c "the driver ended rc=$BOX_RC $BOX_WHY (stage_c.log; stage_c.json keeps its parts)"
        return 1
    fi
    part_done stage_c stage_c.json
}
plant_step() {                   # plant_step NAME CUT CMD...: one step of the plant; PLANT_WHY on failure
    box_bounded "plant $1" "$2" plant.log "${@:3}" && return 0
    # pg_labels exits 3 when a game had no luck walk: both its sides are left out.
    [ "$1" = labels ] && [ "$BOX_RC" -eq 3 ] && return 0
    PLANT_WHY="plant $1 ended rc=$BOX_RC $BOX_WHY (plant.log)"
    return 1
}
plant() {                        # the planted offline readout
    skip_part plant && return 0
    local half=$(( PLANT_CUT / 2 ))
    PLANT_WHY=""
    if [ ! -f "$OUT/plant_grads.pt" ]; then
        plant_step delta "$half" python tools/pg_plant.py delta --checkpoint "$CKPT" --games "${DIRS[@]}" \
            --labels "$S1/labels.jsonl" --pass "$OUT/pass.pt" --walk "$S1/walk.jsonl" --seed-base "$SP_SEED_BASE" \
            --train-games "$N_PLANT" --class "$PLANT_CLASS" --readout-fraction "$PLANT_READOUT_FRACTION" \
            --device cuda --workers "$WORKERS" --out "$OUT/plant_grads.pt" && box_upload_async
    fi
    [ -n "$PLANT_WHY" ] || [ -f "$OUT/walk_planted.jsonl" ] || plant_step copy "$SMALL_CUT_MIN" \
        python tools/pg_plant.py plant --walk "$S1/walk.jsonl" --games "${DIRS[@]}" --grads "$OUT/plant_grads.pt" \
        --out "$OUT/walk_planted.jsonl"
    [ -n "$PLANT_WHY" ] || plant_step rows "$SMALL_CUT_MIN" python tools/selfplay_rows.py --no-walk \
        --walk "$OUT/walk_planted.jsonl" --out "$OUT/rows_planted.jsonl" --seed-base "$SP_SEED_BASE" --max-errors 1000
    [ -n "$PLANT_WHY" ] || plant_step labels "$SMALL_CUT_MIN" python tools/pg_labels.py "$OUT/rows_planted.jsonl" \
        --heldout-from "$HELDOUT_FROM" --out "$OUT/labels_planted.jsonl" --fits "$OUT/fits_planted.json"
    [ -n "$PLANT_WHY" ] || [ -f "$OUT/pass_planted.pt" ] || plant_step pass "$half" python tools/pg_pass.py \
        --checkpoint "$CKPT" --games "${DIRS[@]}" --labels "$OUT/labels_planted.jsonl" --out "$OUT/pass_planted.pt" \
        --device cuda --workers "$WORKERS"
    [ -n "$PLANT_WHY" ] || plant_step readout "$half" python tools/pg_plant.py readout --checkpoint "$CKPT" \
        --games "${DIRS[@]}" --labels "$OUT/labels_planted.jsonl" --pass "$OUT/pass_planted.pt" \
        --grads "$OUT/plant_grads.pt" --main-pass "$OUT/pass.pt" --device cuda --workers "$WORKERS" \
        --out "$OUT/plant_readout.json"
    if [ -n "$PLANT_WHY" ]; then
        failed_part plant "$PLANT_WHY"
        return 1
    fi
    part_done plant plant_readout.json plant_grads.pt pass_planted.pt labels_planted.jsonl
}
verdict() {                      # verdict.json: design §6's decision on what landed; VERDICT its line
    VERDICT=$(timeout 2m python - "$OUT" <<'EOF' | tr -d '\r'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
def read(name):
    p = out / name
    return json.loads(p.read_text()) if p.is_file() else None
c, plant, c0 = read("stage_c.json"), read("plant_readout.json"), read("c0_plan.json")
gate = (c or {}).get("gate", {}).get("verdict")
if gate in (None, "unread"):
    nxt = "stage C did not read its gate (stage_c.json's errors, or no driver readout): fix it and run it again"
elif gate == "neither":
    nxt = "no rounds: the user chooses the $1,000 case, sized from the curve, or a stop"
elif plant is None:
    nxt = "the planted readout did not land: run it before P and rounds"
elif plant.get("verdict") != "pass":
    nxt = "the planted readout missed: debug before P and rounds"
else:
    nxt = {"resolved": "P (the planted step played), then rounds",
           "rounds_at_size": "rounds at the size the gain curve names"}.get(gate, "no verdict")
v = {"gate": (c or {}).get("gate"), "optimizer": (c or {}).get("optimizer"), "stage_c_errors": (c or {}).get("errors"),
     "plant": None if plant is None else {"verdict": plant["verdict"], "seats": plant["seats"], "delta": plant["delta"]},
     "c0": c0, "next": nxt}
(out / "verdict.json").write_text(json.dumps(v, indent=1))
print(f"gate={gate} plant={None if plant is None else plant['verdict']} next: {nxt}")
EOF
) || VERDICT="unreadable"
    box_upload_async
}

S1_WHY=""
if ! restore_stage1; then
    box_finish "STAGE1_UNUSABLE: $S1_WHY (restore.log)" 1
fi
if c0 && ok c0; then
    if ! c0_plan; then
        verdict
        box_finish "C0_OVER_SWITCH ${STATE[*]} (c0_plan.json)" 1
    fi
    if gradient_pass && ok pass; then
        stage_c
        plant
    fi
fi
verdict
REASON="${STATE[*]} verdict: $VERDICT"
if ok c0 && ok pass && ok stage_c && ok plant; then
    box_finish "STAGE_C_DONE $REASON"
fi
box_finish "STAGE_C_INCOMPLETE $REASON" 1
# ---- the stages (end)
