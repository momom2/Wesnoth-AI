#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# The memory trained on its own trajectories, one round on one box. ROUND
# picks the preset (each of its variables can be given in the environment
# instead):
#   ROUND=1 (the default): docs/memory_own_trajectories_prereg_20261011.md,
#     as run on 2026-10-11: init, teacher and player `parity3`, its 5,475
#     recorded 64-slot game-sides, matches M1 and M2 against `parity3` at 0
#     slots (seed bases 107000, 108000);
#   ROUND=2: docs/memory_own_round2_prereg_20261011.md: init, teacher and
#     player `own1`, its 64-slot sides of the confirmation's three matches
#     and of a generation match played here first (`own1` at 64 slots
#     against itself, seed base 113000), matches R1, R2 and R3 (114000,
#     115000, 116000).
# The parameters: the init checkpoint (INIT_HF, INIT_SHA256),
# the teacher (TEACHER_HF, TEACHER_SHA256; default the init), the player
# whose sides are selected (PLAYER_SHA256, PLAYER_SLOTS; default the init at
# 64 slots; the labelling reads the policy that played, so it must be the
# teacher), the own-trajectory sources (SOURCE_TARS, match tarballs on HF),
# an optional generation match (GEN_MATCH) and the evaluation matches
# (EVAL_MATCHES). A match is "NAME SEED_BASE ROLE_A SLOTS_A LABEL_A ROLE_B
# SLOTS_B LABEL_B", matches separated by ';', a role being `tuned` (this
# run's checkpoint), `init` or `teacher`; each side plays the reference's
# decode (configs/reference_player.json, raw:t0+eo-1.5) with its checkpoint
# swapped in.
#
# The steps:
#   the tests of this path, on the wheel built from the stage;
#   the init and teacher checkpoints from HF, their SHA-256 checked;
#   the raw replays (RAW_TAR) and the sources; the generation match, when
#     there is one, its games recorded whole and up as one tarball, a
#     source too;
#   the corpus rebuilt at version 5 and pre-encoded with the init's
#     vocabularies;
#   the own trajectories: the player's game-sides at its slots without a
#     per-turn reset, selected by the results' checkpoint SHA-256, memory and
#     reset fields (tools/own_trajectories.py), each decision labelled with
#     the teacher's raw distribution at 0 slots, top 16 renormalised, on the
#     GPU (tools/own_targets.py); the replay barrier (95% of the recorded
#     decisions in turns 6 and later reproduced) stops the box before
#     training;
#   the fine-tune from the init (tools/sequence_train.py --own-sequences):
#     equal numbers of own and human positions per update, every own position
#     at weight 1, one pass of PASS_POSITIONS (about 1 million of each), the
#     learning rate at a tenth of the recipe's peak, falling linearly to 0
#     over the pass's second half; the starting weights probed first;
#   the holdout and agreement barriers against the init checkpoint
#     (tools/own_barriers.py): a failed one stops the box before any match;
#   the evaluation matches, PURE, 800 decisive games each, the Ladder maps
#     with both factions drawn uniformly, every game recorded whole, each
#     fitted (tools/elo_collect.py), its games up as one tarball.
# Known and left as it is (none in round 1's five recorded matches): a
# recruit the engine relocated after a rejection would count as one forward
# more and its position would show the rejection bit early
# (tools/own_trajectories.py).
#
# Seeds: a match with seed base S plays seeds S to S+799 and replaces a capped
# game of slot i with seed S + i + k * 1,000,000. Scanned 2026-10-11 over every
# branch and tag: the seed bases scripts and documents name, with the offsets
# scripts add, end at 112799 (the confirmation's 109000 + 3000), and no
# committed game file carries a seed from 107000 to 116999, replacements
# included (round 1 took 107000-108799 on its box).
#
# Box: an RTX 4090 with at least 32 usable cores and 64 GB (the corpus build,
# the pre-encodings and the 20 match workers), 150 GB of disk. Expected wall,
# from round 1's box (training/metrics/memory_own_trajectories_20261011/
# walls.txt, 4.5 hours in all):
#   round 1: bring-up and tests 10 minutes, inputs 1, corpus and
#     pre-encoding 7, candidates 4, targets 59 (390 positions a second for
#     1.4 million), the pass 169 (2 million positions at 217 a second, its
#     three probes included), barriers 2, two matches 23, final upload 5;
#     4.6 hours, $1.9-2.9 at $0.42-0.63 an hour; BOX_MAX_H 7;
#   round 2: the same, but a generation match of about 15 minutes (the
#     confirmation's self-play: 862 s), about 4,800 own game-sides and 1.3
#     million positions (round 1's 256 a game-side; candidates 4 minutes,
#     targets 56), and three matches of 12 to 14 minutes each (the
#     confirmation's: 692 to 862 s): 5.2 hours, $2.2-3.3; BOX_MAX_H 8
#     (1.5 times the estimate) caps it at $3.4-5.0 plus the final round.
#     The generation match now plays 120 games in flight (GEN_JOBS): about
#     6 minutes at round 2's measured 8,150 games an hour.
#   Identical match repeats have differed by 1.8x.
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md): the
# onstart of scripts/rent_box.py fetches the library of STAGE, then this
# script (pass ROUND with --env). Every step has a bound; the dead-man's
# switch finishes the entry after BOX_MAX_H hours whatever it is doing.
# Records go to HF $HF_DIR every 30 minutes and after each milestone: the
# fine-tune's checkpoint as soon as its pass is done, each match's games as
# one tarball. Re-entry, on this machine or a new one (files absent here come
# back from HF), continues the pass from its checkpoint, skips what is done
# and resumes a match in its directory; a new machine rebuilds the corpus,
# its sequences and the own records when a step still needs them. A run
# belongs to the code stage that began it (RUN_STAGE): another stage
# continues it only with RESUME_OTHER_STAGE=1. Every exit past that check,
# clean or not, uploads the records with ALL_DONE last and stops the
# instance; an entry refused there stops it and sends nothing.
# Never `set -x`: the HF token and the instance key are in the environment.
# box-needs: disk_gb=150 ram_gb=64 gpu_ram_gb=24 cores=32 gpu=4090
set -uo pipefail
WORKDIR=/workspace
STAGE="${STAGE:-}"
ROUND="${ROUND:-1}"
# ---- round presets (begin)
case $ROUND in
    1)
        : "${HF_DIR:=tier-b/memory_own_trajectories_20261011}"
        : "${INIT_HF:=tier-b/imitation_anneal_20261003/lower.pt}"
        : "${INIT_SHA256:=efba0a78518fe3c1273b408f7471f457353598482cb66e977f55a8db47f8d5ec}"
        : "${SOURCE_TARS:=tier-b/parity3_baselines_20261008/games_selfpin.tar.gz \
tier-b/parity3_baselines_20261008/games_slots64_vs_slots16.tar.gz \
tier-b/parity3_baselines_20261008/games_slots64_vs_slots0.tar.gz \
tier-b/memory_in_play_parity3_20261009/games_a1_eo25.tar.gz \
tier-b/memory_in_play_parity3_20261009/games_a3_eo35.tar.gz}"
        : "${GEN_MATCH:=}"
        : "${EVAL_MATCHES:=m1_own64 107000 tuned 64 own64 init 0 parity3_slots0;\
m2_own0 108000 tuned 0 own0 init 0 parity3_slots0}"
        : "${RUN_SEED:=20261011}"
        : "${BOX_MAX_H:=7}"
        ;;
    2)
        : "${HF_DIR:=tier-b/memory_own_round2_20261011}"
        : "${INIT_HF:=tier-b/memory_own_trajectories_20261011/own.pt}"
        : "${INIT_SHA256:=70925e770d43496e3ea8f9523656f201b71ada228a0362b0eb0a600c8c107f80}"
        : "${SOURCE_TARS:=tier-b/own_confirm_20261011/games_own64_vs_p3.tar.gz \
tier-b/own_confirm_20261011/games_own64_vs_own0.tar.gz \
tier-b/own_confirm_20261011/games_own64_selfplay.tar.gz}"
        : "${GEN_MATCH:=gen_own1_selfplay 113000 init 64 own1a init 64 own1b}"
        : "${EVAL_MATCHES:=r1_own2_64_vs_own1_64 114000 tuned 64 own2_64 init 64 own1;\
r2_own2_0_vs_own1_0 115000 tuned 0 own2_0 init 0 own1_slots0;\
r3_own2_64_vs_own2_0 116000 tuned 64 own2_64 tuned 0 own2_0}"
        : "${RUN_SEED:=20261012}"         # another order of the corpus than round 1's: other human positions
        : "${BOX_MAX_H:=8}"
        ;;
    *)
        echo "ROUND=$ROUND has no preset (1 or 2): give every round variable in the environment"
        ;;
esac
: "${INIT_HF:=}" "${INIT_SHA256:=}" "${SOURCE_TARS:=}" "${GEN_MATCH:=}" "${EVAL_MATCHES:=}" "${RUN_SEED:=}"
: "${TEACHER_HF:=$INIT_HF}"
: "${TEACHER_SHA256:=${INIT_SHA256:-}}"
: "${PLAYER_SHA256:=${INIT_SHA256:-}}"
: "${PLAYER_SLOTS:=64}"
# ---- round presets (end)
export HF_DIR
OUT=$WORKDIR/memoryown_r$ROUND
SEQ=$WORKDIR/sequences
OWN_GAMES=$WORKDIR/own_games_r$ROUND
CAND=$WORKDIR/own_candidates_r$ROUND
OWN=$WORKDIR/own_sequences_r$ROUND
RAW_TAR="${RAW_TAR:-tier-b/corpus_v3/raw_corpus_20260929.tar}"
LR="${LR:-2.8e-5}"                               # a tenth of the recipe's peak, 2.8e-4
PASS_POSITIONS="${PASS_POSITIONS:-2000000}"      # about 1 million own and 1 million human positions
DECAY_FROM="${DECAY_FROM:-0.5}"                  # the rate falls linearly to 0 over the pass's second half
PROBE_EVERY="${PROBE_EVERY:-1200000}"            # one probe inside the pass (the recipe's memory barrier), then the final one
# 1: the pass trains on the packed trunk (--packed-trunk: no dropout, other arithmetic than the padded
# trunk's; docs/trainer_speed_20261013.md); 0: the padded trunk, as rounds 1 and 2 ran.
PACKED_TRUNK="${PACKED_TRUNK:-0}"
WORKERS="${WORKERS:-}"                           # default: box_workers (cores, memory-bounded), after box_init
GAMES="${GAMES:-800}"
JOBS="${JOBS:-20}"                               # the evaluation matches' games in flight, on the eager server
# The generation match's games in flight and its speed flags. None of them
# changes a game's play (tests/test_generation_identity.py), only how fast
# the games are played: 120 in flight, six per worker process, the fastest
# configuration measured (training/metrics/generation_speed_r2_20261013/,
# 8,150-8,210 games an hour against 3,700-3,790 at 20); the round 3 bench
# (scripts/generation_speed_box.sh) names the configuration that replaces it.
GEN_JOBS="${GEN_JOBS:-120}"
GEN_SPEED_FLAGS="${GEN_SPEED_FLAGS:---worker-games 6}"
# Bounds in minutes, from the estimates above.
TESTS_CUT_MIN="${TESTS_CUT_MIN:-40}"
BUILD_CUT_MIN="${BUILD_CUT_MIN:-60}"             # 1 on round 1's box
BUILD_STALL_MIN="${BUILD_STALL_MIN:-15}"         # the builder logs every 1,000 candidates
PREENCODE_CUT_MIN="${PREENCODE_CUT_MIN:-150}"    # 6 on round 1's box
PREENCODE_STALL_MIN="${PREENCODE_STALL_MIN:-20}" # the pre-encoder logs every 200 games
CAND_CUT_MIN="${CAND_CUT_MIN:-60}"               # 4 on round 1's box
CAND_STALL_MIN="${CAND_STALL_MIN:-20}"           # the candidates' step logs every 100 games
TARGETS_CUT_MIN="${TARGETS_CUT_MIN:-120}"        # 59 on round 1's box
TARGETS_STALL_MIN="${TARGETS_STALL_MIN:-20}"     # the labelling logs every 100 games
TRAIN_CUT_MIN="${TRAIN_CUT_MIN:-260}"            # 169 on round 1's box
TRAIN_STALL_MIN="${TRAIN_STALL_MIN:-40}"         # the trainer logs every minute; a probe runs silent
BARRIER_CUT_MIN="${BARRIER_CUT_MIN:-30}"         # 2 on round 1's box
MATCH_CUT_MIN="${MATCH_CUT_MIN:-90}"             # the match stops itself at 60 (--time-budget-min), a game at 20 more
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }

CORPUS=replays_dataset_imitation
ARM=training/checkpoints/own_candidate.pt
INIT_PT=$WORKDIR/init_r$ROUND.pt
TEACHER_PT=$INIT_PT
[ "$TEACHER_SHA256" = "$INIT_SHA256" ] || TEACHER_PT=$WORKDIR/teacher_r$ROUND.pt
match_names() {                  # match_names SPECS: the names of the matches of SPECS, one per line
    local spec name specs=()
    IFS=';' read -ra specs <<< "$1"
    for spec in "${specs[@]}"; do
        read -r name _ <<< "$spec"
        [ -z "$name" ] || echo "$name"
    done
}
GEN_NAME=$(match_names "$GEN_MATCH")
EVAL_NAMES=$(match_names "$EVAL_MATCHES" | tr '\n' ' ')
MATCHES="$GEN_NAME $EVAL_NAMES"

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
    local name done_=() line=""
    [ ! -f "$OUT/own_sequence_summary.json" ] || done_+=(own)
    [ ! -f "$OUT/TRAIN_DONE" ] || done_+=(train)
    [ ! -f "$OUT/barriers.json" ] || done_+=(barriers)
    for name in $MATCHES; do
        [ ! -f "$OUT/$name.fit.json" ] || done_+=("$name")
    done
    [ ! -f "$OUT/train.log" ] || line=$(grep -o "positions [0-9]*/[0-9]* steps [0-9]*" "$OUT/train.log" | tail -1)
    echo "round $ROUND, done: ${done_[*]:-nothing}${line:+; training at $line}"
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
      grep -h "games, [0-9]* \(candidates\|positions\)" "$OUT/candidates.log" "$OUT/targets.log" 2>/dev/null | tail -n 2
      find "$OUT" -maxdepth 1 -type f -printf '%f ' 2>/dev/null; echo
      tail -n 2 "$OUT"/*.log 2>/dev/null | tail -n 8
    } > "$OUT/progress.txt.tmp" && mv -f "$OUT/progress.txt.tmp" "$OUT/progress.txt"
}

box_init
[ -n "$WORKERS" ] || WORKERS=$(box_workers)
[ -n "$STAGE" ] || box_finish "NO_STAGE: build the code stage (tools/stage_code.py) and pass STAGE" 1
for v in INIT_HF INIT_SHA256 TEACHER_HF TEACHER_SHA256 PLAYER_SHA256 SOURCE_TARS EVAL_MATCHES RUN_SEED; do
    [ -n "${!v:-}" ] || box_finish "NO_PRESET: ROUND=$ROUND leaves $v unset" 1
done
# The labelling reads the policy that played each trajectory (its segmentation and its replay).
[ "$PLAYER_SHA256" = "$TEACHER_SHA256" ] \
    || box_finish "PLAYER_IS_NOT_THE_TEACHER: the labelling reads the player's policy, here the teacher's" 1
box_bind_run_stage

# ---- the records of an earlier entry
restored=(corpus_summary.json sequence_summary.json own_candidates_summary.json own_sequence_summary.json
          replay.json TRAIN_DONE own.pt own.probe.jsonl own.signal.jsonl own.steps.jsonl train.log barriers.json
          match.walls)
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
        box_mark_landed "games_$name.tar.gz" "$OUT/games_$name" \
            || echo "games_$name will go up again (restore.log)"
    fi
done
box_upload_skip own.stable.pt                    # the weights where the rate began to fall: kept on this machine
box_upload_hold TRAIN_DONE own.pt own.probe.jsonl
box_pip huggingface_hub psutil pytest scipy requests || echo "pip install failed (pip.log)"

# ---- the code and the Rust wheel
box_stage_code || box_finish "CODE_STAGING_FAILED (staging.log)" 1
cd "$BOX_REPO" || box_finish "CODE_STAGING_FAILED (no $BOX_REPO)" 1
box_build_wheel || box_finish "BUILD_FAILED (build.log)" 1
box_facts > "$OUT/box.txt.tmp" 2>&1
mv -f "$OUT/box.txt.tmp" "$OUT/box.txt"
timeout 2m python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || box_finish "NO_CUDA (box.txt)" 1
box_upload_async
box_monitor_start

# ---- the crash barrier: this path's tests on this wheel and this GPU
if ! box_marked_this_stage "$BOX_STATE/TESTED"; then
    : > "$OUT/tests.log"
    box_bounded tests "$TESTS_CUT_MIN" tests.log python -m pytest tests/test_own_trajectories.py \
        tests/test_sequence_train.py tests/test_sequence_speed.py tests/test_own_loss.py \
        tests/test_preencode_sequences.py tests/test_lr_law.py \
        tests/test_memory_model.py tests/test_match_memory.py tests/test_server_priors.py \
        tests/test_game_record.py tests/test_attack_pathfinding.py tests/test_game_core.py tests/test_vision.py \
        tests/test_parity_integration.py tests/test_sighting_record.py tests/test_faction_posterior.py \
        tests/test_unit_vocab.py tests/test_raw_player.py tests/test_reference_player.py \
        tests/test_eval_workers.py tests/test_eval_inference_server.py tests/test_elo_collect.py \
        tests/test_signal_telemetry.py tests/test_own_rounds.py \
        tests/test_corpus_v3.py tests/test_no_shroud.py -q -p no:cacheprovider -m ""
    tail -n 3 "$OUT/tests.log"
    [ "$BOX_RC" -eq 0 ] || box_finish "TESTS_FAILED rc=$BOX_RC (tests.log)" 1
    box_mark "$BOX_STATE/TESTED"
fi

# ---- the decode (the reference's), and the init and teacher checkpoints, their SHA-256 checked
EO=$(timeout 1m python -c "import json; print(json.load(open('configs/reference_player.json'))['decode']['raw_end_turn_offset'])") \
    || box_finish "REFERENCE_CONFIG_UNREADABLE (configs/reference_player.json)" 1
[ "$EO" = -1.5 ] || box_finish "REFERENCE_DECODE_CHANGED: offset $EO; the rounds are pre-registered at -1.5" 1
fetch_checkpoint() {             # fetch_checkpoint HF_PATH SHA256 DEST NAME: DEST holds HF_PATH's file, its SHA-256 checked
    local hf=$1 sha=$2 dest=$3 name=$4
    if [ -f "$dest" ] && echo "$sha  $dest" | sha256sum -c --status; then return 0; fi
    box_bounded "$name" 20 staging.log python - "$hf" "$dest" <<'EOF' || return 1
import os, shutil, sys
from huggingface_hub import hf_hub_download
shutil.copyfile(hf_hub_download("momom2/wesnoth-model-checkpoints", sys.argv[1]), sys.argv[2] + ".tmp")
os.replace(sys.argv[2] + ".tmp", sys.argv[2])
EOF
    echo "$sha  $dest" | sha256sum -c --status
}
fetch_checkpoint "$INIT_HF" "$INIT_SHA256" "$INIT_PT" init \
    || box_finish "INIT_CHECKPOINT_FAILED ($INIT_HF against $INIT_SHA256; staging.log)" 1
if [ "$TEACHER_PT" != "$INIT_PT" ]; then
    fetch_checkpoint "$TEACHER_HF" "$TEACHER_SHA256" "$TEACHER_PT" teacher \
        || box_finish "TEACHER_CHECKPOINT_FAILED ($TEACHER_HF against $TEACHER_SHA256; staging.log)" 1
fi

# ---- match sides (begin)
side_flags() {                   # side_flags SIDE LABEL SLOTS SPEC: the reference's flags for SIDE, one per line, under LABEL at SLOTS slots with the checkpoint SPEC
    local side=$1 label=$2 slots=$3 spec=$4 flags i
    mapfile -t flags < <(timeout 1m python tools/reference_player.py --flags "$side" | tr -d '\r' | tr ' ' '\n')
    for (( i = 0; i + 1 < ${#flags[@]}; i++ )); do
        case ${flags[i]} in
            "--label-$side") flags[i + 1]=$label ;;
            "--memory-$side") flags[i + 1]=$slots ;;
            "--spec-$side") flags[i + 1]=$spec ;;
        esac
    done
    printf '%s\n' "${flags[@]}"
}
flags_ok() {                     # flags_ok SIDE SLOTS SPEC FLAGS...: the flags play SPEC at SLOTS slots under the decode
    local side=$1 slots=$2 spec=$3
    shift 3
    [[ " $* " == *" --spec-$side $spec "* && " $* " == *" --memory-$side $slots "* \
        && " $* " == *" --raw-end-turn-offset-$side $EO "* ]]
}
role_spec() {                    # role_spec ROLE: the checkpoint a side plays: tuned (this run's), init or teacher
    case $1 in
        tuned) echo "$ARM" ;;
        init) echo "$INIT_PT" ;;
        teacher) echo "$TEACHER_PT" ;;
        *) return 1 ;;
    esac
}
MATCH_ARGS=()
match_args() {                   # match_args SPEC: M_NAME, M_SEED and MATCH_ARGS (both sides' flags) of "NAME SEED ROLE SLOTS LABEL ROLE SLOTS LABEL"
    local ra sa la rb sb lb spec_a spec_b a b
    read -r M_NAME M_SEED ra sa la rb sb lb <<< "$1"
    [[ -n ${lb:-} && $M_SEED =~ ^[0-9]+$ && $sa =~ ^[0-9]+$ && $sb =~ ^[0-9]+$ ]] || return 1
    spec_a=$(role_spec "$ra") || return 1
    spec_b=$(role_spec "$rb") || return 1
    mapfile -t a < <(side_flags a "$la" "$sa" "$spec_a")
    mapfile -t b < <(side_flags b "$lb" "$sb" "$spec_b")
    flags_ok a "$sa" "$spec_a" "${a[@]}" || return 1
    flags_ok b "$sb" "$spec_b" "${b[@]}" || return 1
    MATCH_ARGS=("${a[@]}" "${b[@]}")
}
# ---- match sides (end)

# ---- the matches' machinery
MATCH_JOBS=$JOBS                 # the match's games in flight and speed flags (the generation match sets its own)
MATCH_SPEED=()
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
            --persistent-workers --shared-inference --no-infer-compile --device cuda --jobs "$MATCH_JOBS" \
            ${MATCH_SPEED[@]+"${MATCH_SPEED[@]}"} --time-budget-min 60
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

# What the steps still need: the pass needs the corpus, its sequences and the
# own records; the barriers need the own records only.
NEED_CORPUS=1
NEED_OWN=1
[ ! -f "$OUT/TRAIN_DONE" ] || NEED_CORPUS=0
[ ! -f "$OUT/barriers.json" ] || { NEED_CORPUS=0; NEED_OWN=0; }

# ---- the inputs: the raw replays and the recorded matches
if [ "$NEED_OWN" = 1 ] && ! box_marked_this_stage "$BOX_STATE/INPUTS_DONE_r$ROUND"; then
    # shellcheck disable=SC2086 # SOURCE_TARS is a list of HF paths
    box_bounded inputs 30 staging.log python - "$RAW_TAR" "$OWN_GAMES" $SOURCE_TARS <<'EOF' \
        || box_finish "INPUTS_FAILED rc=$BOX_RC (staging.log)" 1
import sys, tarfile
from huggingface_hub import hf_hub_download
repo = "momom2/wesnoth-model-checkpoints"
with tarfile.open(hf_hub_download(repo, sys.argv[1]), "r") as tf:
    tf.extractall(".")
    print("raw replays", sum(1 for name in tf.getnames() if name.endswith(".bz2")), flush=True)
for path in sys.argv[3:]:
    with tarfile.open(hf_hub_download(repo, path), "r:gz") as tf:
        tf.extractall(sys.argv[2])
        print(path, sum(1 for name in tf.getnames() if name.endswith(".game.jsonl.gz")), "game records", flush=True)
EOF
    box_mark "$BOX_STATE/INPUTS_DONE_r$ROUND"
fi

# ---- the generation match: the player's new trajectories, a source of this round
GEN_DIRS=()
if [ -n "$GEN_MATCH" ]; then
    match_args "$GEN_MATCH" || box_finish "GEN_MATCH_UNREADABLE ($GEN_MATCH)" 1
    if [ "$NEED_OWN" = 1 ] && ! box_marked_this_stage "$CAND/CANDIDATES_DONE"; then
        MATCH_JOBS=$GEN_JOBS
        read -r -a MATCH_SPEED <<< "$GEN_SPEED_FLAGS"
        play "$M_NAME" "$M_SEED" "${MATCH_ARGS[@]}"
        MATCH_JOBS=$JOBS
        MATCH_SPEED=()
        if [ "$MATCHES_FAILED" -ne 0 ] || [ "$MATCHES_CUT" -ne 0 ]; then
            box_finish "GEN_MATCH_SHORT: $M_NAME failed or was cut (match.walls) $(notes)" 1
        fi
    fi
    GEN_DIRS=("$OUT/games_$M_NAME")
fi

# ---- the corpus at version 5; the crash barrier: every candidate accounted for, under 1% failed
if [ "$NEED_CORPUS" = 1 ] && ! box_marked_this_stage "$BOX_STATE/CORPUS_DONE"; then
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

# ---- the sequences with the init's vocabularies; the crash barrier: under 0.5% of games skipped,
# posterior errors under 0.1%
if [ "$NEED_CORPUS" = 1 ] && ! box_marked_this_stage "$SEQ/SEQUENCES_DONE"; then
    if ! box_marked_this_stage "$SEQ/STARTED"; then          # records another stage's code built
        rm -rf "$SEQ"
        box_mark "$SEQ/STARTED" || box_finish "SEQUENCES_UNWRITABLE ($SEQ)" 1
    fi
    from=$(box_size "$OUT/preencode.log")
    box_bounded --stall "$OUT/preencode.log" "$PREENCODE_STALL_MIN" preencode "$PREENCODE_CUT_MIN" preencode.log \
        python tools/preencode_sequences.py --dataset "$CORPUS" --out "$SEQ" --workers "$WORKERS" \
        --vocab-from "$INIT_PT"
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
    box_mark "$SEQ/SEQUENCES_DONE"
fi

# ---- the own candidates; the crash barrier: every own game accounted for, under 2% of them dropped
if [ "$NEED_OWN" = 1 ] && ! box_marked_this_stage "$CAND/CANDIDATES_DONE"; then
    if ! box_marked_this_stage "$CAND/STARTED"; then
        rm -rf "$CAND"
        box_mark "$CAND/STARTED" || box_finish "CANDIDATES_UNWRITABLE ($CAND)" 1
    fi
    from=$(box_size "$OUT/candidates.log")
    box_bounded --stall "$OUT/candidates.log" "$CAND_STALL_MIN" candidates "$CAND_CUT_MIN" candidates.log \
        python tools/own_trajectories.py --games "$OWN_GAMES"/games_* ${GEN_DIRS[@]+"${GEN_DIRS[@]}"} \
        --teacher "$TEACHER_PT" --player-sha256 "$PLAYER_SHA256" --slots "$PLAYER_SLOTS" \
        --out "$CAND" --workers "$WORKERS"
    tail -c "+$(( from + 1 ))" "$OUT/candidates.log" | grep -q "CANDIDATES_DONE" \
        || box_finish "CANDIDATES_${BOX_WHY^^} rc=$BOX_RC (candidates.log)" 1
    box_bounded candidates-check 10 candidates.log python - "$CAND" "$OUT/own_candidates_summary.json" <<'EOF' \
        || box_finish "CANDIDATES_BARRIER rc=$BOX_RC (candidates.log, own_candidates_summary.json)" 1
import collections, json, pathlib, sys
from tools.own_trajectories import read_candidates_manifest
man = read_candidates_manifest(pathlib.Path(sys.argv[1]))
reasons = collections.Counter(e.split(":")[1].strip() if e.startswith("error:") else e for e in man["errors"].values())
summary = {"selection": man["selection"], "games": man["n_games"], "done": man["n_done"],
           "errors": len(man["errors"]), "error_kinds": dict(reasons), "teacher_sha256": man["teacher_sha256"],
           "player_sha256": man["player_sha256"], "core_phase": man["core_phase"], **man["totals"]}
pathlib.Path(sys.argv[2]).write_text(json.dumps(summary, indent=1), encoding="utf-8")
print("own candidates", summary, flush=True)
assert man["n_done"] + len(man["errors"]) == man["n_games"], "own games unaccounted for"
assert len(man["errors"]) < 0.02 * max(1, man["n_games"]), "2% or more of the own games dropped"
EOF
    box_mark "$CAND/CANDIDATES_DONE"
fi

# ---- the teacher's targets; the crash barriers: under 5% of the own game-sides dropped, the
# training game-sides hold their half of the pass, and the replay reproduces 95% of the late decisions
if [ "$NEED_OWN" = 1 ] && ! box_marked_this_stage "$OWN/OWN_DONE"; then
    if ! box_marked_this_stage "$OWN/STARTED"; then
        rm -rf "$OWN"
        box_mark "$OWN/STARTED" || box_finish "OWN_UNWRITABLE ($OWN)" 1
    fi
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before the targets: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    from=$(box_size "$OUT/targets.log")
    box_bounded --stall "$OUT/targets.log" "$TARGETS_STALL_MIN" targets "$TARGETS_CUT_MIN" targets.log \
        python tools/own_targets.py --candidates "$CAND" --teacher "$TEACHER_PT" --out "$OWN" --device cuda
    tail -c "+$(( from + 1 ))" "$OUT/targets.log" | grep -q "OWN_TARGETS_DONE" \
        || box_finish "TARGETS_${BOX_WHY^^} rc=$BOX_RC (targets.log)" 1
    box_bounded targets-check 10 targets.log python - "$OWN" "$OUT/own_sequence_summary.json" "$PASS_POSITIONS" <<'EOF' \
        || box_finish "TARGETS_BARRIER rc=$BOX_RC (targets.log, own_sequence_summary.json)" 1
import json, pathlib, sys
from tools.preencode_sequences import load_manifest
man = load_manifest(pathlib.Path(sys.argv[1]))
t = man["totals"]
train = sum(v for r in man["games"].values() if not r["holdout"] for k, v in r.items() if k.startswith("positions_side"))
held = sum(v for r in man["games"].values() if r["holdout"] for k, v in r.items() if k.startswith("positions_side"))
rate = lambda a, b: round(t.get(a, 0) / max(1, t.get(b, 0)), 4)
summary = {"games": man["n_games"], "train_positions": train, "holdout_positions": held,
           "repro": rate("repro", "decisions"), "repro_late": rate("repro_late", "decisions_late"),
           "agree": rate("agree", "decisions"), "agree_late": rate("agree_late", "decisions_late"),
           "approach_merged": rate("approach_merged", "approach_pairs"), **t}
pathlib.Path(sys.argv[2]).write_text(json.dumps(summary, indent=1), encoding="utf-8")
print("own sequences", summary, flush=True)
dropped = sum(v for k, v in t.items() if k.startswith("dropped: "))
assert dropped < 0.05 * max(1, t.get("game_sides", 0)), "5% or more of the own game-sides dropped"
assert train >= int(sys.argv[3]) // 2, "the own training game-sides hold less than their half of the pass"
EOF
    # The replay barrier (pre-registered): the teacher at the player's slots, its memory carried,
    # reproduces at least 95% of the recorded decisions in turns 6 and later.
    box_bounded replay-check 10 targets.log python - "$OWN" "$OUT/replay.json" <<'EOF' \
        || box_finish "REPLAY_BARRIER_FAILED rc=$BOX_RC (replay.json, targets.log)" 1
import json, pathlib, sys
from tools.own_barriers import replay_barrier
from tools.preencode_sequences import load_manifest
result = replay_barrier(load_manifest(pathlib.Path(sys.argv[1]))["totals"])
pathlib.Path(sys.argv[2]).write_text(json.dumps(result, indent=1), encoding="utf-8")
print("replay", result, flush=True)
sys.exit(0 if result["pass"] else 1)
EOF
    if ! { cp -f "$OWN/sequence_manifest.json" "$OUT/own_sequence_manifest.json.tmp" \
            && mv -f "$OUT/own_sequence_manifest.json.tmp" "$OUT/own_sequence_manifest.json"; }; then
        box_finish "OWN_MANIFEST_COPY_FAILED" 1
    fi
    box_mark "$OWN/OWN_DONE"
    box_upload_async
fi

# ---- the fine-tune from the init checkpoint
train_attempt() {                # train_attempt MINUTES: the pass, continuing own.pt when present; sets BOX_RC, BOX_WHY
    local start=(--init-from "$INIT_PT" --init-other-records) trunk=()
    [ ! -f "$OUT/own.pt" ] || start=(--resume)
    if [ "$PACKED_TRUNK" = 1 ]; then
        trunk=(--packed-trunk)
    fi
    box_bounded --stall "$OUT/train.log" "$TRAIN_STALL_MIN" train "$1" train.log \
        python tools/sequence_train.py --sequences "$SEQ" --dataset "$CORPUS" --own-sequences "$OWN" \
        --out "$OUT/own.pt" --seed "$RUN_SEED" --device cuda --lr "$LR" --warmup-steps 0 \
        --decay-from "$DECAY_FROM" --pass-positions "$PASS_POSITIONS" --probe-every "$PROBE_EVERY" \
        --barrier-positions 1000000 --probe-ks 0,64 --probe-at-start "${start[@]}" ${trunk[@]+"${trunk[@]}"}
}
if [ ! -f "$OUT/TRAIN_DONE" ]; then
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before the pass: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    from=$(box_size "$OUT/train.log")
    deadline=$(( $(date +%s) + TRAIN_CUT_MIN * 60 ))
    train_attempt "$TRAIN_CUT_MIN"
    left=$(( (deadline - $(date +%s)) / 60 ))
    if [ "$BOX_RC" -ne 0 ] && [ "$BOX_RC" -ne 3 ] && [ "$BOX_WHY" = failed ] && [ "$left" -ge 60 ] \
            && [ -f "$OUT/own.pt" ]; then
        train_attempt "$left"            # a crash retries once, from the last periodic checkpoint
    fi
    case $BOX_RC in
        0) tail -c "+$(( from + 1 ))" "$OUT/train.log" | grep -q "SEQUENCE_TRAIN_DONE" \
               || box_finish "TRAIN_UNFINISHED: exit 0 without SEQUENCE_TRAIN_DONE (train.log) $(notes)" 1 ;;
        3) box_finish "MEMORY_BARRIER_FAILED in the pass (train.log) $(notes)" 1 ;;
        *) box_finish "TRAIN_${BOX_WHY^^} rc=$BOX_RC (train.log; it continues from own.pt on re-entry) $(notes)" 1 ;;
    esac
    box_mark "$OUT/TRAIN_DONE"
    box_upload_async                     # the checkpoint goes to HF now, before the barriers and the matches
fi
tail -n 1 "$OUT/own.probe.jsonl" | cut -c1-600

# ---- the crash barriers against the init checkpoint, before any match
if [ ! -f "$OUT/barriers.json" ]; then
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before the barriers: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    box_bounded barriers "$BARRIER_CUT_MIN" barriers.log python tools/own_barriers.py --student "$OUT/own.pt" \
        --init "$INIT_PT" --own-sequences "$OWN" --probes "$OUT/own.probe.jsonl" \
        --out "$OUT/barriers.json.pending" --device cuda
    case $BOX_RC in
        0) mv -f "$OUT/barriers.json.pending" "$OUT/barriers.json" ;;
        8) mv -f "$OUT/barriers.json.pending" "$OUT/barriers_failed.json"
           box_finish "BARRIER_FAILED (barriers_failed.json, barriers.log) $(notes)" 1 ;;
        *) box_finish "BARRIERS_${BOX_WHY^^} rc=$BOX_RC (barriers.log) $(notes)" 1 ;;
    esac
    box_upload_async
fi
timeout 1m python -c "import json, sys; sys.exit(0 if json.load(open(sys.argv[1]))['pass'] else 1)" "$OUT/barriers.json" \
    || box_finish "BARRIER_FAILED (barriers.json) $(notes)" 1

# ---- the evaluation matches
mkdir -p training/checkpoints
cp -f "$OUT/own.pt" "$ARM" || box_finish "ARM_COPY_FAILED (own.pt)" 1
MATCHES_FAILED=0
MATCHES_CUT=0
IFS=';' read -ra EVAL_SPECS <<< "$EVAL_MATCHES"
for spec in "${EVAL_SPECS[@]}"; do
    [ -n "${spec// /}" ] || continue
    match_args "$spec" || box_finish "MATCH_UNREADABLE ($spec)" 1
    play "$M_NAME" "$M_SEED" "${MATCH_ARGS[@]}"
done
box_on_round
[ "$MATCHES_FAILED" -eq 0 ] \
    || box_finish "MEMORY_OWN_MATCHES_FAILED: $MATCHES_FAILED failed, $MATCHES_CUT cut (match.walls) $(notes)" 1
[ "$MATCHES_CUT" -eq 0 ] \
    || box_finish "MEMORY_OWN_DONE with $MATCHES_CUT matches cut short (match.walls) $(notes)"
box_finish "MEMORY_OWN_DONE $(notes)"
