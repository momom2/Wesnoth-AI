#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# The memory trained on its own trajectories, round 1
# (docs/memory_own_trajectories_prereg_20261011.md), on one box:
#   the tests of this path (the own trajectories, the trainer, the
#     pre-encoder, memory play, the server priors, the game records, the
#     match path), on the wheel built from the stage;
#   the corpus rebuilt at version 5 from the raw replays (RAW_TAR) and
#     pre-encoded with `parity3`'s vocabularies, as for its retrain;
#   the own trajectories: every game-side `parity3` played at 64 slots
#     without a per-turn reset in the recorded matches (GAME_TARS), selected
#     by the results' checkpoint SHA-256, memory and reset fields
#     (tools/own_trajectories.py), each decision labelled with the frozen
#     `parity3`'s raw distribution at 0 slots, top 16 renormalised, on the
#     GPU (tools/own_targets.py);
#   the fine-tune from `parity3` (tools/sequence_train.py --own-sequences):
#     equal numbers of own and human positions per update, one pass of
#     PASS_POSITIONS (about 1 million of each), the learning rate at a tenth
#     of the recipe's peak, falling linearly to 0 over the pass's second
#     half; the starting weights probed first (`--probe-at-start`);
#   the crash barriers (tools/own_barriers.py): a failed one stops the box
#     before any match;
#   two matches against `parity3` at 0 slots, PURE, both sides at the
#     reference decode (raw:t0+eo-1.5), the Ladder maps with both factions
#     drawn uniformly, 800 decisive games each, every game recorded whole:
#     M1. the tuned checkpoint at 64 slots (seed base M1_SEED_BASE, 107000);
#     M2. the tuned checkpoint at 0 slots (seed base M2_SEED_BASE, 108000);
#   each fitted (tools/elo_collect.py), its games up as one tarball.
#
# Box: an RTX 4090 with at least 32 usable cores and 64 GB (the corpus build,
# the pre-encodings and the 20 match workers), 150 GB of disk (the corpus,
# its sequences, the own candidates and records). Expected wall: 4.7 hours.
# Bring-up and tests about 20 minutes (the anneal box's list ran in 8; this
# one adds the slow tier of the trainer's tests); the inputs, the corpus and
# its pre-encoding about 30 (the anneal box: 25); the own candidates about 10
# (15 ms a position on one laptop core, on the order of 1.6 million
# positions over about 30 box cores, which run the corpus pre-encoder about
# 3 times faster than the laptop); their labelling about 25 (two forwards a
# decision, the player's slots and 0, about 3.2 million in batches of 512 at
# the eval server's few thousand forwards a second, and the host's work per
# decision); the pass about 155 (2 million positions at the 4.24 ms of
# `parity3`'s hold pass 1, and three probes at 0 and 64 slots of about 4
# minutes each); the barriers about 5; each match 13 to 18 (the baselines
# box: 750 to 1,093 s); the final upload about 5. Identical match repeats
# have differed by 1.8x. Cost at $0.42-0.63 an hour: $2.0-3.0 expected;
# BOX_MAX_H 7 caps it at $2.9-4.4 plus the final round.
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md): the
# onstart of scripts/rent_box.py fetches the library of STAGE, then this
# script. Every step has a bound; the dead-man's switch finishes the entry
# after BOX_MAX_H hours whatever it is doing. Records go to HF $HF_DIR every
# 30 minutes and after each milestone: the fine-tune's checkpoint as soon as
# its pass is done, each match's games as one tarball. Re-entry, on this
# machine or a new one (files absent here come back from HF), continues the
# pass from its checkpoint, skips what is done and resumes a match in its
# directory; a new machine rebuilds the corpus, its sequences and the own
# records when a step still needs them. A run belongs to the code stage that
# began it (RUN_STAGE): another stage continues it only with
# RESUME_OTHER_STAGE=1. Every exit past that check, clean or not, uploads the
# records with ALL_DONE last and stops the instance; an entry refused there
# stops it and sends nothing.
# Never `set -x`: the HF token and the instance key are in the environment.
# box-needs: disk_gb=150 ram_gb=64 gpu_ram_gb=24 cores=32 gpu=4090
set -uo pipefail
WORKDIR=/workspace
OUT=$WORKDIR/memoryown
SEQ=$WORKDIR/sequences
OWN_GAMES=$WORKDIR/own_games
CAND=$WORKDIR/own_candidates
OWN=$WORKDIR/own_sequences
STAGE="${STAGE:-}"
RAW_TAR="${RAW_TAR:-tier-b/corpus_v3/raw_corpus_20260929.tar}"
# The recorded matches the own trajectories come from (the pre-registration's).
GAME_TARS="${GAME_TARS:-tier-b/parity3_baselines_20261008/games_selfpin.tar.gz \
tier-b/parity3_baselines_20261008/games_slots64_vs_slots16.tar.gz \
tier-b/parity3_baselines_20261008/games_slots64_vs_slots0.tar.gz \
tier-b/memory_in_play_parity3_20261009/games_a1_eo25.tar.gz \
tier-b/memory_in_play_parity3_20261009/games_a3_eo35.tar.gz}"
RUN_SEED="${RUN_SEED:-20261011}"
LR="${LR:-2.8e-5}"                               # a tenth of the recipe's peak, 2.8e-4
PASS_POSITIONS="${PASS_POSITIONS:-2000000}"      # about 1 million own and 1 million human positions
DECAY_FROM="${DECAY_FROM:-0.5}"                  # the rate falls linearly to 0 over the pass's second half
PROBE_EVERY="${PROBE_EVERY:-1200000}"            # one probe inside the pass (the recipe's memory barrier), then the final one
# A match with seed base S plays seeds S to S+799 and replaces a capped game
# of slot i with seed S + i + k * 1,000,000. Scanned 2026-10-11: every seed
# base a script or document names in a branch or tag of the repository (113
# refs), with the offsets the scripts add, ends below 107000 (the memory-in-
# play box's 106000 + 799), and no game file committed in any of them
# carries a seed from 107000 to 108999, replacements included.
M1_SEED_BASE="${M1_SEED_BASE:-107000}"
M2_SEED_BASE="${M2_SEED_BASE:-108000}"
WORKERS="${WORKERS:-}"                           # default: box_workers (cores, memory-bounded), after box_init
GAMES="${GAMES:-800}"
JOBS="${JOBS:-20}"
export HF_DIR="${HF_DIR:-tier-b/memory_own_trajectories_20261011}"
# Bounds in minutes, from the estimates above.
TESTS_CUT_MIN="${TESTS_CUT_MIN:-40}"
BUILD_CUT_MIN="${BUILD_CUT_MIN:-60}"             # estimated 10 on 32 cores
BUILD_STALL_MIN="${BUILD_STALL_MIN:-15}"         # the builder logs every 1,000 candidates
PREENCODE_CUT_MIN="${PREENCODE_CUT_MIN:-150}"    # 6 on pass 2's box
PREENCODE_STALL_MIN="${PREENCODE_STALL_MIN:-20}" # the pre-encoder logs every 200 games
CAND_CUT_MIN="${CAND_CUT_MIN:-60}"               # estimated 10
CAND_STALL_MIN="${CAND_STALL_MIN:-20}"           # the candidates' step logs every 100 games
TARGETS_CUT_MIN="${TARGETS_CUT_MIN:-90}"         # estimated 25
TARGETS_STALL_MIN="${TARGETS_STALL_MIN:-20}"     # the labelling logs every 100 games
TRAIN_CUT_MIN="${TRAIN_CUT_MIN:-240}"            # estimated 155
TRAIN_STALL_MIN="${TRAIN_STALL_MIN:-40}"         # the trainer logs every minute; a probe runs silent
BARRIER_CUT_MIN="${BARRIER_CUT_MIN:-30}"         # estimated 5
MATCH_CUT_MIN="${MATCH_CUT_MIN:-90}"             # the match stops itself at 60 (--time-budget-min), a game at 20 more
BOX_MAX_H="${BOX_MAX_H:-7}"                      # 1.5 times the 4.7 box-hours expected
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }

CORPUS=replays_dataset_imitation
ARM=training/checkpoints/own_candidate.pt
MATCHES="m1_own64 m2_own0"
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
notes() {                        # what is done, for the finish reason
    local name done_=()
    [ ! -f "$OUT/own_sequence_summary.json" ] || done_+=(own)
    [ ! -f "$OUT/TRAIN_DONE" ] || done_+=(train)
    [ ! -f "$OUT/barriers.json" ] || done_+=(barriers)
    for name in $MATCHES; do
        [ ! -f "$OUT/$name.fit.json" ] || done_+=("$name")
    done
    local line=""
    [ ! -f "$OUT/train.log" ] || line=$(grep -o "positions [0-9]*/[0-9]* steps [0-9]*" "$OUT/train.log" | tail -1)
    echo "done: ${done_[*]:-nothing}${line:+; training at $line}"
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
box_bind_run_stage

# ---- the records of an earlier entry
restored=(corpus_summary.json sequence_summary.json own_candidates_summary.json own_sequence_summary.json
          TRAIN_DONE own.pt own.probe.jsonl own.signal.jsonl own.steps.jsonl train.log barriers.json match.walls)
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
        tests/test_sequence_train.py tests/test_preencode_sequences.py tests/test_lr_law.py \
        tests/test_memory_model.py tests/test_match_memory.py tests/test_server_priors.py \
        tests/test_game_record.py tests/test_attack_pathfinding.py tests/test_game_core.py tests/test_vision.py \
        tests/test_parity_integration.py tests/test_sighting_record.py tests/test_faction_posterior.py \
        tests/test_unit_vocab.py tests/test_raw_player.py tests/test_reference_player.py \
        tests/test_eval_workers.py tests/test_eval_inference_server.py tests/test_elo_collect.py \
        tests/test_corpus_v3.py tests/test_no_shroud.py -q -p no:cacheprovider -m ""
    tail -n 3 "$OUT/tests.log"
    [ "$BOX_RC" -eq 0 ] || box_finish "TESTS_FAILED rc=$BOX_RC (tests.log)" 1
    box_mark "$BOX_STATE/TESTED"
fi

# ---- the reference: its checkpoint (SHA-256 checked) and its decode
box_bounded reference 15 staging.log python tools/reference_player.py --ensure \
    || box_finish "REFERENCE_MISSING rc=$BOX_RC (staging.log)" 1
read -r REF_LABEL REF_PT EO < <(timeout 1m python -c "import json; r = json.load(open('configs/reference_player.json')); \
print(r['label'], r['checkpoint_local'], r['decode']['raw_end_turn_offset'])") \
    || box_finish "REFERENCE_CONFIG_UNREADABLE (configs/reference_player.json)" 1
[ "$REF_LABEL" = parity3 ] \
    || box_finish "REFERENCE_CHANGED: configs/reference_player.json names $REF_LABEL; this run tunes parity3" 1
[ "$EO" = -1.5 ] || box_finish "REFERENCE_DECODE_CHANGED: offset $EO; the matches are pre-registered at -1.5" 1

# What the steps still need: the pass needs the corpus, its sequences and the
# own records; the barriers need the own records only.
NEED_CORPUS=1
NEED_OWN=1
[ ! -f "$OUT/TRAIN_DONE" ] || NEED_CORPUS=0
[ ! -f "$OUT/barriers.json" ] || { NEED_CORPUS=0; NEED_OWN=0; }

# ---- the inputs: the raw replays and the recorded matches
if [ "$NEED_OWN" = 1 ] && ! box_marked_this_stage "$BOX_STATE/INPUTS_DONE"; then
    # shellcheck disable=SC2086 # GAME_TARS is a list of HF paths
    box_bounded inputs 30 staging.log python - "$RAW_TAR" "$OWN_GAMES" $GAME_TARS <<'EOF' \
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
    box_mark "$BOX_STATE/INPUTS_DONE"
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

# ---- the sequences with parity3's vocabularies; the crash barrier: under 0.5% of games skipped,
# posterior errors under 0.1%
if [ "$NEED_CORPUS" = 1 ] && ! box_marked_this_stage "$SEQ/SEQUENCES_DONE"; then
    if ! box_marked_this_stage "$SEQ/STARTED"; then          # records another stage's code built
        rm -rf "$SEQ"
        box_mark "$SEQ/STARTED" || box_finish "SEQUENCES_UNWRITABLE ($SEQ)" 1
    fi
    from=$(box_size "$OUT/preencode.log")
    box_bounded --stall "$OUT/preencode.log" "$PREENCODE_STALL_MIN" preencode "$PREENCODE_CUT_MIN" preencode.log \
        python tools/preencode_sequences.py --dataset "$CORPUS" --out "$SEQ" --workers "$WORKERS" \
        --vocab-from "$REF_PT"
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
        python tools/own_trajectories.py --games "$OWN_GAMES"/games_* --teacher "$REF_PT" --slots 64 \
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
           "core_phase": man["core_phase"], **man["totals"]}
pathlib.Path(sys.argv[2]).write_text(json.dumps(summary, indent=1), encoding="utf-8")
print("own candidates", summary, flush=True)
assert man["n_done"] + len(man["errors"]) == man["n_games"], "own games unaccounted for"
assert len(man["errors"]) < 0.02 * max(1, man["n_games"]), "2% or more of the own games dropped"
EOF
    box_mark "$CAND/CANDIDATES_DONE"
fi

# ---- the teacher's targets; the crash barrier: under 5% of the own game-sides dropped, and the
# training game-sides hold their half of the pass
if [ "$NEED_OWN" = 1 ] && ! box_marked_this_stage "$OWN/OWN_DONE"; then
    if ! box_marked_this_stage "$OWN/STARTED"; then
        rm -rf "$OWN"
        box_mark "$OWN/STARTED" || box_finish "OWN_UNWRITABLE ($OWN)" 1
    fi
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before the targets: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    from=$(box_size "$OUT/targets.log")
    box_bounded --stall "$OUT/targets.log" "$TARGETS_STALL_MIN" targets "$TARGETS_CUT_MIN" targets.log \
        python tools/own_targets.py --candidates "$CAND" --teacher "$REF_PT" --out "$OWN" --device cuda
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
    if ! { cp -f "$OWN/sequence_manifest.json" "$OUT/own_sequence_manifest.json.tmp" \
            && mv -f "$OUT/own_sequence_manifest.json.tmp" "$OUT/own_sequence_manifest.json"; }; then
        box_finish "OWN_MANIFEST_COPY_FAILED" 1
    fi
    box_mark "$OWN/OWN_DONE"
    box_upload_async
fi

# ---- the fine-tune from parity3
train_attempt() {                # train_attempt MINUTES: the pass, continuing own.pt when present; sets BOX_RC, BOX_WHY
    local start=(--init-from "$REF_PT" --init-other-records)
    [ ! -f "$OUT/own.pt" ] || start=(--resume)
    box_bounded --stall "$OUT/train.log" "$TRAIN_STALL_MIN" train "$1" train.log \
        python tools/sequence_train.py --sequences "$SEQ" --dataset "$CORPUS" --own-sequences "$OWN" \
        --out "$OUT/own.pt" --seed "$RUN_SEED" --device cuda --lr "$LR" --warmup-steps 0 \
        --decay-from "$DECAY_FROM" --pass-positions "$PASS_POSITIONS" --probe-every "$PROBE_EVERY" \
        --barrier-positions 1000000 --probe-ks 0,64 --probe-at-start "${start[@]}"
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

# ---- the crash barriers, before any match
if [ ! -f "$OUT/barriers.json" ]; then
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before the barriers: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    box_bounded barriers "$BARRIER_CUT_MIN" barriers.log python tools/own_barriers.py --student "$OUT/own.pt" \
        --teacher "$REF_PT" --own-sequences "$OWN" --probes "$OUT/own.probe.jsonl" \
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

# ---- the matches: the tuned checkpoint at 64 and at 0 slots against parity3 at 0 slots
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
mapfile -t SLOTS0_B < <(ref_side b "$OPPONENT" 0)
[[ " ${SLOTS0_B[*]} " == *" --spec-b $REF_PT "* && " ${SLOTS0_B[*]} " == *" --memory-b 0 "* \
    && " ${SLOTS0_B[*]} " == *" --raw-end-turn-offset-b $EO "* ]] \
    || box_finish "REFERENCE_FLAGS_FAILED (tools/reference_player.py --flags b)" 1
mkdir -p training/checkpoints
cp -f "$OUT/own.pt" "$ARM" || box_finish "ARM_COPY_FAILED (own.pt)" 1
M1=(--label-a own64 --spec-a "$ARM" --memory-a 64 --raw-end-turn-offset-a "$EO")
M2=(--label-a own0 --spec-a "$ARM" --memory-a 0 --raw-end-turn-offset-a "$EO")

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
play m1_own64 "$M1_SEED_BASE" "${M1[@]}" "${SLOTS0_B[@]}"
play m2_own0 "$M2_SEED_BASE" "${M2[@]}" "${SLOTS0_B[@]}"
box_on_round
[ "$MATCHES_FAILED" -eq 0 ] \
    || box_finish "MEMORY_OWN_MATCHES_FAILED: $MATCHES_FAILED failed, $MATCHES_CUT cut (match.walls) $(notes)" 1
[ "$MATCHES_CUT" -eq 0 ] \
    || box_finish "MEMORY_OWN_DONE with $MATCHES_CUT matches cut short (match.walls) $(notes)"
box_finish "MEMORY_OWN_DONE $(notes)"
