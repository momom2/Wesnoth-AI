#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# The parity-memory retrain (docs/parity_memory_design_20260929.md; bars
# and predictions in docs/parity_memory_prereg_20260929.md), on one box:
#   the tests of the core, the encoding, the pre-encoding, the sequence
#     trainer and memory serving, on the wheel built from the stage;
#   the corpus rebuilt at version 5 from the raw replays (RAW_TAR);
#   the fresh vocabulary: 190 unit types, none on the overflow row;
#   every decision of both sides pre-encoded, game by game
#     (tools/preencode_sequences.py);
#   the pass (tools/sequence_train.py), which stops itself (exit 3) when
#     the memory's crash barrier fails after 500,000 positions;
#   `obs8`'s holdout cross-entropy on the same decisions (tools/holdout_ce.py),
#     for the barrier "the recipe broke";
#   four matches, PURE, both sides at the reference decode, the Ladder maps
#     with factions drawn uniformly and assigned openly, 800 decisive games:
#     1. the arm at 64 slots against obs8 (seed base 80000): the verdict;
#     2. the arm at 64 against the arm at 0 (81000): the memory's share;
#     3. the arm at 16 against the arm at 0 (82000): the curve's shape;
#     4. obs8 against itself (83000): the self-pin.
#   A match is read only once it holds its decisive games.
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md):
# the onstart of scripts/rent_box.py fetches the library of STAGE, then
# this script. Every step has a bound; the dead-man's switch finishes the
# entry after BOX_MAX_H hours whatever it is doing. Records go to HF
# $HF_DIR every 30 minutes and at the end, each match's games as one
# tarball. Re-entry, on this machine or a new one (files absent here come
# back from HF), skips finished steps and continues the pass where its
# checkpoint stood; a new machine rebuilds the corpus and the sequences,
# which are deterministic. A run belongs to the code stage that began it
# (RUN_STAGE, boxlib `box_bind_run_stage`): another stage continues it,
# its finished steps kept, only with RESUME_OTHER_STAGE=1, and a stage that
# changes the data or the pass takes a new HF_DIR. Every exit, clean or
# not, uploads the records with ALL_DONE last and stops the instance.
# Never `set -x`: the HF token and the instance key are in the environment.
# box-needs: disk_gb=120 ram_gb=64 gpu_ram_gb=24 cores=32
set -uo pipefail
WORKDIR=/workspace
OUT=$WORKDIR/paritymemory
SEQ=$WORKDIR/sequences
STAGE="${STAGE:-}"
RAW_TAR="${RAW_TAR:-tier-b/corpus_v3/raw_corpus_20260929.tar}"
RUN_SEED="${RUN_SEED:-20260929}"
WORKERS="${WORKERS:-}"                           # default: box_workers (cores, memory-bounded), after box_init
GAMES="${GAMES:-800}"
JOBS="${JOBS:-20}"
export HF_DIR="${HF_DIR:-tier-b/parity_memory_20260930}"
# Bounds in minutes, from the pre-registration's "Cost".
BUILD_CUT_MIN="${BUILD_CUT_MIN:-60}"             # estimated 10 on 32 cores
BUILD_STALL_MIN="${BUILD_STALL_MIN:-15}"         # the builder logs every 1,000 candidates
PREENCODE_CUT_MIN="${PREENCODE_CUT_MIN:-150}"    # estimated 50-60
PREENCODE_STALL_MIN="${PREENCODE_STALL_MIN:-20}" # the pre-encoder logs every 200 games
TRAIN_CUT_MIN="${TRAIN_CUT_MIN:-1260}"           # the pass: 11-14 hours
TRAIN_STALL_MIN="${TRAIN_STALL_MIN:-40}"         # the trainer logs every minute; a probe runs silent
CE_CUT_MIN="${CE_CUT_MIN:-60}"                   # estimated 15
MATCH_CUT_MIN="${MATCH_CUT_MIN:-90}"             # the match stops itself at 60 (--time-budget-min), a game at 20 more
BOX_MAX_H="${BOX_MAX_H:-28}"                     # 1.5 times the 18 box-hours estimated
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }

CORPUS=replays_dataset_imitation
CKPT="$OUT/arm.pt"
ARM=training/checkpoints/parity_memory.pt

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
notes() {                        # the pass's positions, for the finish reason
    local line
    line=$(grep -o "positions [0-9]*/[0-9]* steps [0-9]*" "$OUT/train.log" 2>/dev/null | tail -1)
    echo "pass: ${line:-not started}"
}
# shellcheck disable=SC2317 # called by the library, in the reason of an unexpected exit
box_notes() { notes; }
box_on_round() {                 # progress.txt, before each upload round
    { date -u
      grep -o "positions [0-9]*/[0-9]*.*positions/s" "$OUT/train.log" 2>/dev/null | tail -1
      grep "PROBE" "$OUT/train.log" 2>/dev/null | cut -c1-300 | tail -n 2
      grep -o "[0-9]*/[0-9]* games, [0-9]* positions.*" "$OUT/preencode.log" 2>/dev/null | tail -1
      find "$OUT" -maxdepth 1 -type f -printf '%f ' 2>/dev/null; echo
      tail -n 2 "$OUT"/*.log 2>/dev/null | tail -n 8
    } > "$OUT/progress.txt.tmp" && mv -f "$OUT/progress.txt.tmp" "$OUT/progress.txt"
}

box_init
[ -n "$WORKERS" ] || WORKERS=$(box_workers)
[ -n "$STAGE" ] || box_finish "NO_STAGE: build the code stage (tools/stage_code.py) and pass STAGE" 1
box_bind_run_stage
box_restore DONE arm.pt arm.probe.jsonl arm.signal.jsonl train.log corpus_summary.json \
    sequence_summary.json sequence_manifest.json obs8_holdout_ce.json barrier.txt \
    || box_finish "RESTORE_FAILED (restore.log)" 1
box_upload_hold DONE arm.pt
box_pip huggingface_hub psutil pytest scipy requests || echo "pip install failed (pip.log)"

# ---- the code and the Rust wheel (the corpus, the pre-encoding and the matches run on the core)
box_stage_code || box_finish "CODE_STAGING_FAILED (staging.log)" 1
cd "$BOX_REPO" || box_finish "CODE_STAGING_FAILED (no $BOX_REPO)" 1
box_build_wheel || box_finish "BUILD_FAILED (build.log)" 1
box_facts > "$OUT/box.txt.tmp" 2>&1
mv -f "$OUT/box.txt.tmp" "$OUT/box.txt"
timeout 2m python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || box_finish "NO_CUDA (box.txt)" 1
box_upload_async
box_monitor_start

# ---- the crash barrier before the corpus: the recipe's tests on this wheel
if ! box_marked_this_stage "$BOX_STATE/TESTED"; then
    : > "$OUT/tests.log"
    box_bounded tests 40 tests.log python -m pytest tests/test_game_core.py tests/test_vision.py \
        tests/test_delayed_shroud.py tests/test_parity_integration.py tests/test_faction_posterior.py \
        tests/test_sighting_record.py tests/test_memory_model.py tests/test_preencode_sequences.py \
        tests/test_sequence_train.py tests/test_match_memory.py tests/test_unit_vocab.py \
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
# Once the pass is done it serves only obs8's holdout cross-entropy. The
# raw replays and the corpus live in the staged repository, which a new
# stage replaces, so both markers name their stage.
if ! box_marked_this_stage "$BOX_STATE/CORPUS_DONE" \
        && { [ ! -f "$OUT/DONE" ] || [ ! -f "$OUT/obs8_holdout_ce.json" ]; }; then
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

# ---- the fresh vocabulary: 190 unit types, the 47 variations on their base type's rows
box_bounded vocab 10 staging.log python - <<'EOF' || box_finish "VOCAB_FAILED rc=$BOX_RC (staging.log)" 1
from tools.preencode_sequences import fresh_vocab
from wesnoth_ai.encoder import names_on_overflow_row
types, factions = fresh_vocab()
assert len(set(types.values())) == 190, len(set(types.values()))
assert names_on_overflow_row(types) == []
print("fresh vocab:", len(types), "names,", len(set(types.values())), "rows,", len(factions), "factions", flush=True)
EOF

# ---- the sequences; the crash barrier: under 0.5% of games skipped, posterior errors under 0.1%
if ! box_marked_this_stage "$SEQ/SEQUENCES_DONE" && [ ! -f "$OUT/DONE" ]; then
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

# ---- the pass; exit 3 is the memory's crash barrier failing (the checkpoint keeps it, so a
# re-entry stops again), exit 4 a pass that ended short of its positions, exit 5 a run of
# non-finite steps. The pass is done when this entry's attempt exits 0 having logged
# SEQUENCE_TRAIN_DONE.
train_attempt() {                # train_attempt MINUTES: the pass, continuing arm.pt when present; sets BOX_RC, BOX_WHY
    local resume=()
    [ -f "$CKPT" ] && resume=(--resume)
    box_bounded --stall "$OUT/train.log" "$TRAIN_STALL_MIN" train "$1" train.log \
        python tools/sequence_train.py --sequences "$SEQ" --dataset "$CORPUS" --out "$CKPT" \
        --seed "$RUN_SEED" --device cuda ${resume[@]+"${resume[@]}"}
}
train_verdict() {                # after an attempt: stop on the pass's own verdicts
    [ "$BOX_RC" -ne 3 ] || box_finish "MEMORY_BARRIER_FAILED (train.log, arm.probe.jsonl) $(notes)" 1
    [ "$BOX_RC" -ne 5 ] || box_finish "NONFINITE_TRAINING (train.log) $(notes)" 1
}
if [ ! -f "$OUT/DONE" ]; then
    deadline=$(( $(date +%s) + TRAIN_CUT_MIN * 60 ))
    from=$(box_size "$OUT/train.log")
    train_attempt "$TRAIN_CUT_MIN"
    train_verdict
    left=$(( (deadline - $(date +%s)) / 60 ))
    if [ "$BOX_RC" -ne 0 ] && [ "$BOX_WHY" = failed ] && [ "$left" -ge 60 ] && [ -f "$CKPT" ]; then
        from=$(box_size "$OUT/train.log")
        train_attempt "$left"            # a crash retries once, from the last periodic checkpoint
        train_verdict
    fi
    [ "$BOX_RC" -ne 4 ] || box_finish "PASS_INCOMPLETE (train.log) $(notes)" 1
    if [ "$BOX_RC" -ne 0 ] || ! tail -c "+$(( from + 1 ))" "$OUT/train.log" | grep -q "SEQUENCE_TRAIN_DONE"; then
        box_finish "TRAINING_${BOX_WHY^^} rc=$BOX_RC (train.log; the pass continues from arm.pt on re-entry) $(notes)" 1
    fi
    box_mark "$OUT/DONE"
fi
box_upload_async

# ---- the recipe barrier: the arm at 0 slots against obs8 on the same holdout decisions
REF=$(timeout 1m python -c "import json; print(json.load(open('configs/reference_player.json'))['checkpoint_local'])") \
    || box_finish "REFERENCE_CONFIG_UNREADABLE (configs/reference_player.json)" 1
if [ ! -f "$OUT/obs8_holdout_ce.json" ]; then           # the holdout games the pre-encoding kept, as the probe reads
    box_bounded holdout-ce "$CE_CUT_MIN" holdout_ce.log \
        python tools/holdout_ce.py "$REF" --dataset "$CORPUS" --sequences "$OUT" --out "$OUT/obs8_holdout_ce.json" \
        || echo "obs8's holdout cross-entropy failed: rc=$BOX_RC $BOX_WHY (holdout_ce.log)"
fi
[ -f "$OUT/barrier.txt" ] || [ ! -f "$OUT/obs8_holdout_ce.json" ] \
    || { timeout 2m python - "$OUT/arm.probe.jsonl" "$OUT/obs8_holdout_ce.json" > "$OUT/barrier.txt.tmp" \
         && mv -f "$OUT/barrier.txt.tmp" "$OUT/barrier.txt"; } <<'EOF'
import json, math, sys
probe = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()][-1]
obs8 = json.load(open(sys.argv[2], encoding="utf-8"))
arm, ref = probe["k0"]["ce_all"], obs8["ce_all"]
counts = (f"arm at 0 slots over {probe['k0']['n_decisions']} decisions, obs8 over {obs8['n_decisions']} "
          f"({obs8.get('unscored_decisions', 0)} unscored)")
if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (arm, ref)) \
        or probe["k0"].get("n_nonfinite") or obs8.get("n_nonfinite"):
    print(f"holdout CE: NONFINITE, investigate before reading the matches; {counts}")
elif probe["k0"]["n_decisions"] != obs8["n_decisions"] + obs8.get("unscored_decisions", 0):
    print(f"holdout CE: DECISIONS_DIFFER, the two did not read the same decisions; {counts}")
else:
    gap = arm - ref
    print(f"holdout CE: arm {arm:.4f}, obs8 {ref:.4f}, {counts}; gap {gap:+.4f} nat: "
          + ("RECIPE_BROKE, investigate before reading the matches" if gap > 0.05 else "within the barrier"))
EOF
cat "$OUT/barrier.txt" 2>/dev/null || echo "no barrier line: obs8's holdout cross-entropy is missing (holdout_ce.log)"

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
    match "$name" "$GAMES" "$sb" 1500 "$@"
    if [ "$(decisive_results "$OUT/games_$name")" -lt "$GAMES" ]; then
        rm -f "$OUT/$name.fit.json" "$OUT/timing_$name.txt"
        match "$name" "$GAMES" "$sb" 1500 "$@"
    fi
    decisive=$(decisive_results "$OUT/games_$name")
    rc=$(sed -n 's/.* rc=\([0-9]*\) .*/\1/p' "$OUT/timing_$name.txt" 2>/dev/null | tail -n 1)
    if [ ! -f "$OUT/$name.fit.json" ]; then
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
EO=$(timeout 1m python -c "import json; print(json.load(open('configs/reference_player.json'))['decode']['raw_end_turn_offset'])") \
    || box_finish "REFERENCE_CONFIG_UNREADABLE (configs/reference_player.json)" 1
mapfile -t REF_B < <(timeout 1m python tools/reference_player.py --flags b | tr ' ' '\n')
mapfile -t REF_A < <(timeout 1m python tools/reference_player.py --flags a | tr ' ' '\n')
if [ "${#REF_A[@]}" -lt 4 ] || [ "${#REF_B[@]}" -lt 4 ]; then
    box_finish "REFERENCE_FLAGS_FAILED (tools/reference_player.py --flags)" 1
fi
MATCHES="arm64_vs_obs8 arm64_vs_arm0 arm16_vs_arm0 obs8a_vs_obs8b"
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
        # The directory is what HF holds: it goes up again only once it changes.
        timeout -k 30s 2m python "$BOX_LIB/box_upload.py" --out "$OUT" --hf-dir "$HF_DIR" \
            --landed "games_$name.tar.gz" "$OUT/games_$name" >> "$OUT/restore.log" 2>&1 \
            || echo "games_$name will go up again (restore.log)"
    fi
done
mkdir -p training/checkpoints
cp "$CKPT" "$ARM" || box_finish "ARM_COPY_FAILED ($CKPT)" 1
play arm64_vs_obs8 80000 --label-a arm64 --spec-a "$ARM" --memory-a 64 --raw-end-turn-offset-a "$EO" "${REF_B[@]}"
play arm64_vs_arm0 81000 --label-a arm64 --spec-a "$ARM" --memory-a 64 --raw-end-turn-offset-a "$EO" \
    --label-b arm0 --spec-b "$ARM" --memory-b 0 --raw-end-turn-offset-b "$EO"
play arm16_vs_arm0 82000 --label-a arm16 --spec-a "$ARM" --memory-a 16 --raw-end-turn-offset-a "$EO" \
    --label-b arm0 --spec-b "$ARM" --memory-b 0 --raw-end-turn-offset-b "$EO"
play obs8a_vs_obs8b 83000 "${REF_A[@]/#obs8/obs8a}" "${REF_B[@]/#obs8/obs8b}"
box_on_round
[ "$MATCHES_FAILED" -eq 0 ] \
    || box_finish "PARITY_MEMORY_MATCHES_FAILED $(notes) $MATCHES_FAILED failed, $MATCHES_CUT cut (match.walls)" 1
box_finish "PARITY_MEMORY_DONE $(notes) $MATCHES_CUT matches cut"
