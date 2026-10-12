#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# Rollouts from the mover's own view, re-grading the turn benchmark
# (docs/rollout_regrade_prereg_20261012.md), and the end_turn offset match,
# on one box:
#   the tests of the re-grade and of the match path, on the wheel built from
#   the stage;
#   the inputs: own2 from HF (OWN2_HF, its SHA-256 checked against
#     OWN2_SHA256); step 1's benchmark directory (bench.tar.gz), whose
#     rebuild.jsonl must equal the committed copy the readout and its tests
#     read; the benchmark's records (validation.json) and the corpus it was
#     played from, from which the states are rebuilt (bench.tar.gz holds the
#     rebuild's checks and encodings, not its states); the reference own1
#     (configs/reference_player.json, its SHA-256 checked);
#   the re-grade (tools/rollout_regrade.py): 8 rollouts per benchmark
#     candidate in the mover's observed world, both sides own2 at 0 slots at
#     raw:t0.5+eo-1.5, reads 2 and 4, written per candidate as produced; a run
#     that leaves a candidate incomplete is resumed once, which plays only
#     the candidates without a complete record;
#   the readout (tools/rollout_readout.py): step 1's statistics per grader,
#     the continue split, and the reading of the HP margin at read 2: Pass,
#     Pass, tempo only, or Fail (READING_REFUSED when a candidate has no
#     complete record);
#   the offset match own1_eo25 (own1 at 64 slots at raw:t0+eo-2.5, side A)
#     against the reference (own1 at 64 slots at raw:t0+eo-1.5, side B):
#     PURE, the Ladder maps with both factions drawn uniformly, 800 decisive
#     games, seed base MATCH_SEED_BASE, every game recorded whole; played once
#     more when short; its fit (tools/elo_collect.py).
# The match runs whatever the readout says; the finish reason names both.
#
# Box: an RTX 4090 with at least 32 usable cores (docs/box_specs.md).
# Expected wall about 50 minutes: bring-up and tests 15 (as the 2026-10-11
# runs), inputs 3 (about 480 MB), the re-grade 8 to 12, the readout 1, the
# match 13 to 16 (round 2's 64-slot matches: 796 to 956 s for 800 games,
# training/metrics/memory_own_round2_20261011/match.walls), the uploads 3.
# The re-grade: 7,728 rollouts of about 51 decisions and 2 value reads each
# (53 forwards per rollout at benchmark position 103 on the laptop,
# 2026-10-12), about 410,000 forwards; the worker's own time per decision
# measured 6.1 ms on the laptop (its wall less the forward round trips), so
# the shared server binds: 11 ms per batch of 4.3 and 15 ms per batch of 8.9
# on the 2026-10-11 match servers, about 7 + 0.86 ms per leaf, which gives
# 600 to 1,000 forwards a second at batches of 25 to 40: 7 to 12 minutes.
# Identical match repeats have differed by 1.8x, which would bring it to
# 1.5 h. Cost at $0.42-0.63 an hour: $0.35-0.53 expected; BOX_MAX_H 2 caps
# it at $0.84-1.26 plus the final round.
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md): the
# onstart of scripts/rent_box.py fetches the library of STAGE, then this
# script. Every step has a bound; the dead-man's switch finishes the entry
# after BOX_MAX_H hours whatever it is doing. Records go to HF $HF_DIR every
# 30 minutes and after each milestone, the match's games as one tarball.
# Re-entry, on this machine or a new one (files absent here come back from
# HF), resumes the re-grade from its rollouts.jsonl, skips a readout and a
# fit already made and resumes the match in its directory. A run belongs to
# the code stage that began it (RUN_STAGE): another stage continues it only
# with RESUME_OTHER_STAGE=1. Every exit past that check, clean or not,
# uploads the records with ALL_DONE last and stops the instance.
# Never `set -x`: the HF token and the instance key are in the environment.
# box-needs: disk_gb=40 ram_gb=64 gpu_ram_gb=24 cores=32 gpu=4090
set -uo pipefail
WORKDIR=/workspace
OUT=$WORKDIR/rolloutregrade
STAGE="${STAGE:-}"
OWN2_HF="${OWN2_HF:-tier-b/memory_own_round2_20261011/own.pt}"
OWN2_SHA256="${OWN2_SHA256:-aea027b2696a9fbce0ebcdc0a8483c9df7d35957934b81b0c62093a02cdd9129}"
BENCH_TAR="${BENCH_TAR:-tier-b/critic_step1_20261009/bench.tar.gz}"
BENCH_HF="${BENCH_HF:-tier-b/turn_value_20260925}"
BENCH_CORPUS_TAR="${BENCH_CORPUS_TAR:-tier-b/replays_dataset_imitation_dedup_20260908.tar.gz}"
# A match with seed base S plays seeds S to S+799 and replaces a capped game
# of slot i with seed S + i + k * 1,000,000. On 2026-10-12 no branch or tag
# of the repository names a seed base from 119201 to 120999 other than this
# run's pre-registration (the largest below it is 119000, the withdrawn RL
# rounds', which plays up to 119799), and no committed file writes a seed
# from 120000 to 120999.
MATCH_SEED_BASE="${MATCH_SEED_BASE:-120000}"
GAMES="${GAMES:-800}"
JOBS="${JOBS:-20}"
WORKERS="${WORKERS:-}"                           # default: box_workers less 3 (the server, the parent, the uploads), at most 48
export HF_DIR="${HF_DIR:-tier-b/rollout_regrade_20261012}"
# Bounds in minutes, from the estimates above.
TESTS_CUT_MIN="${TESTS_CUT_MIN:-20}"
INPUTS_CUT_MIN="${INPUTS_CUT_MIN:-20}"
REGRADE_CUT_MIN="${REGRADE_CUT_MIN:-30}"         # estimated 7 to 12
REGRADE_STALL_MIN="${REGRADE_STALL_MIN:-10}"     # the re-grade logs every 20 candidates
MATCH_CUT_MIN="${MATCH_CUT_MIN:-60}"             # the match stops itself at 40 (--time-budget-min), a game at 20 more
BOX_MAX_H="${BOX_MAX_H:-2}"
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }

INPUTS=$WORKDIR/regrade_inputs                   # own2.pt, rebuild.jsonl, validation.json, the corpus
OWN2_PT=$INPUTS/own2.pt
STATIC=training/metrics/critic_step1_20261009/rebuild.jsonl    # in the staged repository
MATCH=own1_eo25_vs_own1

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
reading_of() {                   # the readout's printed reading, or why there is none
    grep -ho "READING_REFUSED.*\|READING [A-Za-z, ]*" "$OUT/readout.log" 2>/dev/null | tail -1 | cut -c1-200
}
notes() {
    echo "regrade: $(grep -ho 'REGRADE_[A-Z]*' "$OUT/regrade.log" 2>/dev/null | tail -1); readout: $(reading_of);" \
         "match: $([ -f "$OUT/$MATCH.fit.json" ] && echo fitted || echo "not fitted")"
}
# shellcheck disable=SC2317 # called by the library, in the reason of an unexpected exit
box_notes() { notes; }
box_on_round() {                 # progress.txt, before each upload round
    { date -u
      notes
      [ ! -f "$OUT/rollouts.jsonl" ] || echo "rollouts.jsonl: $(wc -l < "$OUT/rollouts.jsonl") candidate records"
      cat "$OUT/match.walls" 2>/dev/null
      [ ! -d "$OUT/games_$MATCH" ] || echo "$MATCH: $(find "$OUT/games_$MATCH" -maxdepth 1 -name 'game_*.json' | wc -l) games"
      tail -n 2 "$OUT"/*.log 2>/dev/null | tail -n 8
    } > "$OUT/progress.txt.tmp" && mv -f "$OUT/progress.txt.tmp" "$OUT/progress.txt"
}

box_init
[ -n "$STAGE" ] || box_finish "NO_STAGE: build the code stage (tools/stage_code.py) and pass STAGE" 1
box_bind_run_stage

# ---- the records of an earlier entry
restored=(rollouts.jsonl regrade_summary.json readout.json readout.md readout_refused.json readout.log regrade.log
          match.walls "$MATCH.fit.json" "timing_$MATCH.txt" "$MATCH.log")
[ -d "$OUT/games_$MATCH" ] || restored+=("games_$MATCH.tar.gz")
box_restore "${restored[@]}" || box_finish "RESTORE_FAILED (restore.log)" 1
if [ -f "$OUT/games_$MATCH.tar.gz" ]; then
    [ -d "$OUT/games_$MATCH" ] || timeout -k 30s 10m tar -xzf "$OUT/games_$MATCH.tar.gz" -C "$OUT" \
        || box_finish "MATCH_RESTORE_FAILED ($MATCH)" 1
    rm -f "$OUT/games_$MATCH.tar.gz"
    box_mark_landed "games_$MATCH.tar.gz" "$OUT/games_$MATCH" || echo "games_$MATCH will go up again (restore.log)"
fi
box_pip huggingface_hub psutil pytest scipy requests || echo "pip install failed (pip.log)"

# ---- the code and the Rust wheel
box_stage_code || box_finish "CODE_STAGING_FAILED (staging.log)" 1
cd "$BOX_REPO" || box_finish "CODE_STAGING_FAILED (no $BOX_REPO)" 1
box_build_wheel || box_finish "BUILD_FAILED (build.log)" 1
box_facts > "$OUT/box.txt.tmp" 2>&1
mv -f "$OUT/box.txt.tmp" "$OUT/box.txt"
timeout 2m python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || box_finish "NO_CUDA (box.txt)" 1
if [ -z "$WORKERS" ]; then
    WORKERS=$(( $(box_workers) - 3 ))
    (( WORKERS <= 48 )) || WORKERS=48
    (( WORKERS >= 4 )) || WORKERS=4
fi
echo "re-grade workers: $WORKERS" >> "$OUT/box.txt"
box_upload_async
box_monitor_start

# ---- the crash barrier: the re-grade's tests and the match path's, on this wheel
if ! box_marked_this_stage "$BOX_STATE/TESTED"; then
    : > "$OUT/tests.log"
    box_bounded tests "$TESTS_CUT_MIN" tests.log python -m pytest tests/test_rollout_regrade.py \
        tests/test_critic_bench.py tests/test_turn_bench_stats.py tests/test_lookahead_player.py \
        tests/test_game_core.py tests/test_vision.py tests/test_parity_integration.py tests/test_match_memory.py \
        tests/test_raw_player.py tests/test_reference_player.py tests/test_eval_workers.py \
        tests/test_eval_inference_server.py tests/test_eval_match_failures.py tests/test_game_record.py \
        tests/test_elo_collect.py -q -p no:cacheprovider -m ""
    tail -n 3 "$OUT/tests.log"
    [ "$BOX_RC" -eq 0 ] || box_finish "TESTS_FAILED rc=$BOX_RC (tests.log)" 1
    box_mark "$BOX_STATE/TESTED"
fi

# ---- the inputs
if ! box_marked_this_stage "$BOX_STATE/INPUTS_DONE"; then
    box_bounded inputs "$INPUTS_CUT_MIN" staging.log python - "$INPUTS" "$OWN2_HF" "$BENCH_TAR" "$BENCH_HF" \
        "$BENCH_CORPUS_TAR" <<'EOF' || box_finish "INPUTS_FAILED rc=$BOX_RC (staging.log)" 1
import pathlib, shutil, sys, tarfile
from huggingface_hub import hf_hub_download
REPO = "momom2/wesnoth-model-checkpoints"
inputs, own2, bench_tar, bench_hf, corpus_tar = sys.argv[1:]
inputs = pathlib.Path(inputs)
inputs.mkdir(parents=True, exist_ok=True)
shutil.copyfile(hf_hub_download(REPO, own2), inputs / "own2.pt")
with tarfile.open(hf_hub_download(REPO, bench_tar), "r:gz") as tf:
    member = tf.extractfile("bench/rebuild.jsonl")
    (inputs / "rebuild.jsonl").write_bytes(member.read())
shutil.copyfile(hf_hub_download(REPO, f"{bench_hf}/validation.json"), inputs / "validation.json")
shutil.rmtree(inputs / "replays_dataset_imitation", ignore_errors=True)
with tarfile.open(hf_hub_download(REPO, corpus_tar), "r:gz") as tf:
    tf.extractall(inputs)
assert (inputs / "replays_dataset_imitation" / "manifest.jsonl").is_file(), "no manifest in the benchmark's corpus"
print("inputs ready", flush=True)
EOF
    echo "$OWN2_SHA256  $OWN2_PT" | sha256sum -c --status || box_finish "OWN2_SHA256_MISMATCH ($OWN2_HF)" 1
    cmp -s "$INPUTS/rebuild.jsonl" "$STATIC" \
        || box_finish "STEP1_REBUILD_DIFFERS: $BENCH_TAR's rebuild.jsonl is not the committed $STATIC" 1
    box_mark "$BOX_STATE/INPUTS_DONE"
fi
box_bounded reference 15 staging.log python tools/reference_player.py --ensure \
    || box_finish "REFERENCE_MISSING rc=$BOX_RC (staging.log)" 1
read -r REF_LABEL REF_PT EO < <(timeout 1m python -c "import json; r = json.load(open('configs/reference_player.json')); \
print(r['label'], r['checkpoint_local'], r['decode']['raw_end_turn_offset'])") \
    || box_finish "REFERENCE_CONFIG_UNREADABLE (configs/reference_player.json)" 1
[ "$REF_LABEL" = own1 ] && [ "$EO" = -1.5 ] \
    || box_finish "REFERENCE_CHANGED: configs/reference_player.json names $REF_LABEL at $EO; this match measures own1 at -1.5" 1

# ---- the re-grade, resumed once when it leaves a candidate incomplete
regrade() {
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before the re-grade: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    box_bounded --stall "$OUT/regrade.log" "$REGRADE_STALL_MIN" regrade "$REGRADE_CUT_MIN" regrade.log \
        python tools/rollout_regrade.py --records "$INPUTS/validation.json" \
        --dataset "$INPUTS/replays_dataset_imitation" --checkpoint "$OWN2_PT" --step1 "$INPUTS/rebuild.jsonl" \
        --out "$OUT" --workers "$WORKERS" --device cuda
}
if ! grep -q "REGRADE_DONE" "$OUT/regrade.log" 2>/dev/null; then
    regrade
    [ "$BOX_RC" -eq 0 ] || regrade
    box_upload_async
fi
REGRADE_NOTE=$(grep -ho "REGRADE_[A-Z]*.*" "$OUT/regrade.log" 2>/dev/null | tail -1 | cut -c1-200)

# ---- the readout
if [ ! -f "$OUT/readout.json" ]; then
    : > "$OUT/readout.log"
    box_bounded readout 15 readout.log python tools/rollout_readout.py --rollouts "$OUT/rollouts.jsonl" \
        --static "$STATIC" --out "$OUT"
    box_upload_async
fi
READING=$(reading_of)

# ---- the offset match
mapfile -t REF_B < <(timeout 1m python tools/reference_player.py --flags b | tr ' ' '\n')
mapfile -t EO25_A < <(timeout 1m python tools/reference_player.py --flags a | tr ' ' '\n' \
    | awk 'prev == "--label-a" {$0 = "own1_eo25"} prev == "--raw-end-turn-offset-a" {$0 = "-2.5"} {print; prev = $0}')
if ! [[ " ${EO25_A[*]} " == *" --label-a own1_eo25 "* && " ${EO25_A[*]} " == *" --spec-a $REF_PT "* \
        && " ${EO25_A[*]} " == *" --memory-a 64 "* && " ${EO25_A[*]} " == *" --raw-end-turn-offset-a -2.5 "* \
        && " ${REF_B[*]} " == *" --label-b own1 "* && " ${REF_B[*]} " == *" --spec-b $REF_PT "* \
        && " ${REF_B[*]} " == *" --memory-b 64 "* && " ${REF_B[*]} " == *" --raw-end-turn-offset-b -1.5 "* ]]; then
    box_finish "MATCH_FLAGS_FAILED (tools/reference_player.py --flags); $REGRADE_NOTE; $READING" 1
fi
match() {                        # one attempt, resumed in its directory, then its fit
    local dir="$OUT/games_$MATCH" t0 f
    if [ -f "$OUT/$MATCH.fit.json" ]; then echo "match $MATCH done"; return 0; fi
    if [ ! -f "$OUT/timing_$MATCH.txt" ]; then
        box_gpu_ok || box_finish "GPU_UNRESPONSIVE before the match: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
        echo "$(box_stamp) $MATCH: seed base $MATCH_SEED_BASE, ${EO25_A[*]} ${REF_B[*]}" >> "$OUT/$MATCH.log"
        t0=$(date +%s)
        box_bounded "match $MATCH" "$MATCH_CUT_MIN" "$MATCH.log" \
            python tools/run_elo_batch.py "${EO25_A[@]}" "${REF_B[@]}" \
            --outdir "$dir" --games "$GAMES" --max-extra-games 1500 --seed-base "$MATCH_SEED_BASE" \
            --mcts-sims 0 --raw-temperature-a 0 --raw-temperature-b 0 \
            --persistent-workers --shared-inference --no-infer-compile --device cuda --jobs "$JOBS" \
            --time-budget-min 40
        echo "$MATCH: $(( $(date +%s) - t0 )) s, $(find "$dir" -maxdepth 1 -name 'game_*.json' 2>/dev/null | wc -l) games," \
             "$(decisive_results "$dir") decisive, rc=$BOX_RC $BOX_WHY" | tee "$OUT/timing_$MATCH.txt" | tee -a "$OUT/match.walls"
        for f in "$dir"/.inference_server_*.json; do
            [ -f "$f" ] && cp -f "$f" "$OUT/$MATCH.server_${f##*/.inference_server_}"
        done
    fi
    box_bounded "fit $MATCH" 10 "$MATCH.log" \
        python tools/elo_collect.py "$dir" --no-catalog --save-json "$OUT/$MATCH.fit.json"
}
box_upload_dir "games_$MATCH" "$OUT/games_$MATCH"
box_upload_hold "$MATCH.fit.json" "games_$MATCH.tar.gz"
box_upload_hold "timing_$MATCH.txt" "games_$MATCH.tar.gz"
match
decisive=$(decisive_results "$OUT/games_$MATCH")
if [[ $decisive =~ ^[0-9]+$ ]] && [ "$decisive" -lt "$GAMES" ]; then
    rm -f "$OUT/$MATCH.fit.json" "$OUT/timing_$MATCH.txt"
    box_clear "$MATCH.fit.json" "timing_$MATCH.txt" \
        || echo "the short attempt's fit may stay on HF until the second lands (upload.log)"
    match
    decisive=$(decisive_results "$OUT/games_$MATCH")
fi
box_on_round
if ! [[ $decisive =~ ^[0-9]+$ ]] || [ ! -f "$OUT/$MATCH.fit.json" ]; then
    box_finish "MATCH_FAILED ($MATCH.log); $REGRADE_NOTE; $READING" 1
fi
[ "$decisive" -ge "$GAMES" ] || box_finish "MATCH_CUT: $decisive of $GAMES decisive ($MATCH.log); $REGRADE_NOTE; $READING" 1
# The re-grade's reading is the run's result: without one the run failed, its match fitted or not.
[[ $READING == "READING "* ]] \
    || box_finish "NO_READING: ${READING:-the readout printed none} (readout.log); $REGRADE_NOTE; match fitted ($MATCH.fit.json)" 1
box_finish "ROLLOUT_REGRADE_DONE $REGRADE_NOTE; $READING; match fitted on $decisive decisive games ($MATCH.fit.json)"
