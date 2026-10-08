#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# Step 1 of the self-play program (docs/selfplay_program_20261008.md, "Step 1"):
# does a critic learned from games on hand rank, and select among, candidate
# turns better than the static HP margin? On one box:
#   the step's tests and the core's, on the wheel built from the stage;
#   the inputs: the reference (parity3), the nine match tarballs of
#     2026-10-01 to 2026-10-04 (M), the raw replays (H: the corpus rebuilt at
#     version 5), the turn-value benchmark's records and the corpus it was
#     played from;
#   the free readouts: parity3's prior gaps at its decisions in 100 games of
#     its match against parity2 (tools/prior_gaps.py);
#   the positions of M and H on all cores (tools/critic_positions.py), with
#     its crash barrier: under 1% of the games failed, both sources present;
#   the benchmark's states rebuilt and encoded (tools/critic_bench.py rebuild);
#   six critics (tools/critic_train.py): T25, T50, T100 (true state, 25, 50
#     and 100% of M), O100 (the mover's observation, 100% of M), TH (true
#     state, H), Tsmall (true state, 100% of M, a quarter of the width and
#     depth, from scratch); each ends by patience, its epochs or 40 minutes;
#   the readout (tools/critic_bench.py read, stats), whose reading the finish
#     reason names: Pass, Data-limited or Kill.
#
# Box: one RTX 4090, 32 or more cores, 64 GB, 120 GB of disk
# (docs/box_specs.md "Current box shape"). Expected wall about 3.6 hours:
# bring-up and tests 25 min, inputs 5, prior gaps 10, corpus 10, positions
# 15, benchmark rebuild 3, the critics about 145, the readout 5. The critics'
# share assumes about 3 ms per trained position on the full network (parity3's
# own passes cost 4.24 ms with its memory, policy losses and recomputation):
# T25 10 min, T50 20, T100 and O100 35 each, TH 40 (its bound), Tsmall 5.
# About $1.5-2.3 at $0.42-0.63/h; the pre-registration budgets about 3
# box-hours. BOX_MAX_H, the dead-man's switch, is 6 (1.7 times the estimate).
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md). Every
# step has a bound; the dead-man's switch finishes the entry after BOX_MAX_H
# hours whatever it is doing. Records go to HF $HF_DIR every 30 minutes and
# after each milestone: each critic's checkpoint and its holdout, step and
# signal rows as it trains, the positions' manifest and summary, the
# benchmark's directory as bench.tar.gz. Re-entry, on this machine or a new
# one, skips a critic whose summary says DONE (its checkpoint comes back from
# HF) and the readouts already on HF; a new machine rebuilds the corpus and
# the positions, which are deterministic. Every exit past the stage check
# uploads the records with ALL_DONE last and stops the instance.
# Never `set -x`: the HF token and the instance key are in the environment.
# box-needs: disk_gb=120 ram_gb=64 gpu_ram_gb=24 cores=32 gpu=4090
set -uo pipefail
WORKDIR=/workspace
OUT=$WORKDIR/criticstep1
STAGE="${STAGE:-}"
RAW_TAR="${RAW_TAR:-tier-b/corpus_v3/raw_corpus_20260929.tar}"
BENCH_HF="${BENCH_HF:-tier-b/turn_value_20260925}"
BENCH_CORPUS_TAR="${BENCH_CORPUS_TAR:-tier-b/replays_dataset_imitation_dedup_20260908.tar.gz}"
WORKERS="${WORKERS:-}"                           # default: box_workers (cores, memory-bounded), after box_init
export HF_DIR="${HF_DIR:-tier-b/critic_step1_20261008}"
# Bounds in minutes, from the estimates above.
TESTS_CUT_MIN="${TESTS_CUT_MIN:-40}"
INPUTS_CUT_MIN="${INPUTS_CUT_MIN:-30}"
GAPS_CUT_MIN="${GAPS_CUT_MIN:-45}"               # estimated 15
BUILD_CUT_MIN="${BUILD_CUT_MIN:-60}"             # estimated 10 on 32 cores
BUILD_STALL_MIN="${BUILD_STALL_MIN:-15}"
POSITIONS_CUT_MIN="${POSITIONS_CUT_MIN:-90}"     # estimated 20 on 32 cores
POSITIONS_STALL_MIN="${POSITIONS_STALL_MIN:-15}" # the builder logs every 200 games
CRITIC_MINUTES="${CRITIC_MINUTES:-40}"           # the trainer's own wall bound
CRITIC_CUT_MIN="${CRITIC_CUT_MIN:-60}"           # loading, the last holdout read and the save past it
CRITIC_STALL_MIN="${CRITIC_STALL_MIN:-15}"       # the trainer logs every 50 steps
BOX_MAX_H="${BOX_MAX_H:-6}"
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }

CORPUS=replays_dataset_imitation                 # the corpus at version 5, built in the staged repository
MATCHES_DIR=$WORKDIR/matches                     # one directory per match: <run>__<match>
BENCH_DIR=$WORKDIR/bench_inputs                  # validation.json and the corpus it was played from
POS=$WORKDIR/positions
REF=training/checkpoints/parity3.pt              # tools/reference_player.py --ensure puts it there
MATCH_TARS=(parity_memory_20261001/games_obs8a_vs_obs8b parity_memory_20261001/games_arm64_vs_obs8
            parity_memory_20261001/games_arm64_vs_arm0 parity_memory_20261001/games_arm16_vs_arm0
            parity_memory_pass2_20261002/games_obs8a_vs_obs8b parity_memory_pass2_20261002/games_arm64_vs_obs8
            parity_memory_pass2_20261002/games_arm64_vs_arm0 parity_memory_pass2_20261002/games_arm16_vs_arm0
            imitation_anneal_20261003/games_cand64_vs_ref)
# name source view fraction arch
CRITICS=("T25 M true 0.25 full" "T50 M true 0.5 full" "T100 M true 1.0 full" "O100 M obs 1.0 full"
         "TH H true 1.0 full" "Tsmall M true 1.0 small")

critic_done() {                  # critic_done NAME: its summary says the training ended
    grep -q '"DONE": true' "$OUT/$1.summary.json" 2>/dev/null
}
all_critics_done() {
    local spec
    for spec in "${CRITICS[@]}"; do
        critic_done "${spec%% *}" || return 1
    done
}
notes() {
    local spec state=""
    for spec in "${CRITICS[@]}"; do
        critic_done "${spec%% *}" && state="$state ${spec%% *}"
    done
    echo "critics done:${state:- none}"
}
# shellcheck disable=SC2317 # called by the library, in the reason of an unexpected exit
box_notes() { notes; }
box_on_round() {                 # progress.txt, before each upload round
    { date -u
      notes
      grep -h "games, .* positions" "$OUT/positions.log" 2>/dev/null | tail -1
      tail -n 2 "$OUT"/train_*.log 2>/dev/null | tail -n 6
    } > "$OUT/progress.txt.tmp" && mv -f "$OUT/progress.txt.tmp" "$OUT/progress.txt"
}

box_init
[ -n "$WORKERS" ] || WORKERS=$(box_workers)
[ -n "$STAGE" ] || box_finish "NO_STAGE: build the code stage (tools/stage_code.py) and pass STAGE" 1
box_bind_run_stage
restored=(prior_gaps.json prior_gaps.games.jsonl positions_summary.json positions_manifest.jsonl
          readout.json readout.md)
for spec in "${CRITICS[@]}"; do
    name=${spec%% *}
    restored+=("$name.summary.json")
    # A summary says DONE only once its checkpoint has landed.
    box_upload_hold "$name.summary.json" "$name.pt"
done
box_restore "${restored[@]}" || box_finish "RESTORE_FAILED (restore.log)" 1
for spec in "${CRITICS[@]}"; do
    name=${spec%% *}
    if critic_done "$name"; then
        box_restore "$name.pt" "$name.holdout.jsonl" "$name.steps.jsonl" "$name.signal.jsonl" "train_$name.log" \
            || box_finish "RESTORE_FAILED (restore.log)" 1
        [ -f "$OUT/$name.pt" ] || box_finish "CHECKPOINT_MISSING: $name.summary.json says DONE, $name.pt is not on HF" 1
    fi
done
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

# ---- the crash barrier: the step's tests and the core's, on this wheel
if ! box_marked_this_stage "$BOX_STATE/TESTED"; then
    : > "$OUT/tests.log"
    box_bounded tests "$TESTS_CUT_MIN" tests.log python -m pytest tests/test_critic_data.py tests/test_critic_train.py \
        tests/test_turn_bench_stats.py tests/test_prior_gaps.py tests/test_game_record.py tests/test_match_memory.py \
        tests/test_game_core.py -q -p no:cacheprovider -m ""
    tail -n 3 "$OUT/tests.log"
    [ "$BOX_RC" -eq 0 ] || box_finish "TESTS_FAILED rc=$BOX_RC (tests.log)" 1
    box_mark "$BOX_STATE/TESTED"
fi

# ---- the inputs
box_bounded reference 15 staging.log python tools/reference_player.py --ensure \
    || box_finish "REFERENCE_MISSING rc=$BOX_RC (staging.log)" 1
if ! box_marked_this_stage "$BOX_STATE/INPUTS_DONE"; then
    box_bounded inputs "$INPUTS_CUT_MIN" staging.log python - "$MATCHES_DIR" "$BENCH_DIR" "$BENCH_HF" \
        "$BENCH_CORPUS_TAR" "$RAW_TAR" "${MATCH_TARS[@]}" <<'EOF' || box_finish "INPUTS_FAILED rc=$BOX_RC (staging.log)" 1
import pathlib, shutil, sys, tarfile
from huggingface_hub import hf_hub_download
REPO = "momom2/wesnoth-model-checkpoints"
matches, bench, bench_hf, bench_corpus, raw_tar, *tars = sys.argv[1:]
matches, bench = pathlib.Path(matches), pathlib.Path(bench)
for spec in tars:                                # each match into <run>__<match>
    run, games = spec.split("/")
    dest = matches / f"{run}__{games[len('games_'):]}"
    tmp = matches / ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(dest, ignore_errors=True)
    with tarfile.open(hf_hub_download(REPO, f"tier-b/{spec}.tar.gz"), "r:gz") as tf:
        tf.extractall(tmp)
    shutil.move(str(tmp / games), str(dest))
    print(dest.name, len(list(dest.glob("*.game.jsonl.gz"))), "records", flush=True)
bench.mkdir(parents=True, exist_ok=True)
for name in ("validation.json", "verdict.json"):
    shutil.copyfile(hf_hub_download(REPO, f"{bench_hf}/{name}"), bench / name)
shutil.rmtree(bench / "replays_dataset_imitation", ignore_errors=True)
with tarfile.open(hf_hub_download(REPO, bench_corpus), "r:gz") as tf:
    tf.extractall(bench)
assert (bench / "replays_dataset_imitation" / "manifest.jsonl").is_file(), "no manifest in the benchmark's corpus"
with tarfile.open(hf_hub_download(REPO, raw_tar), "r") as tf:
    tf.extractall(".")
    print("raw replays", sum(1 for n in tf.getnames() if n.endswith(".bz2")), flush=True)
EOF
    box_mark "$BOX_STATE/INPUTS_DONE"
fi

# ---- the free readouts: parity3's prior gaps in 100 games of its match against parity2
if [ ! -f "$OUT/prior_gaps.json" ]; then
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before the prior gaps: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    rm -f "$OUT/prior_gaps.games.jsonl"
    box_bounded "prior gaps" "$GAPS_CUT_MIN" prior_gaps.log python tools/prior_gaps.py \
        --games "$MATCHES_DIR/imitation_anneal_20261003__cand64_vs_ref" --player cand64 --checkpoint "$REF" \
        --out "$OUT/prior_gaps.json" --device cuda \
        || box_finish "PRIOR_GAPS_${BOX_WHY^^} rc=$BOX_RC (prior_gaps.log)" 1
    box_upload_async
fi

# ---- the corpus at version 5 and the positions (only while a critic is left to train)
if ! all_critics_done; then
    if ! box_marked_this_stage "$BOX_STATE/CORPUS_DONE"; then
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
    if ! box_marked_this_stage "$POS/BUILT"; then
        box_marked_this_stage "$POS/STARTED" || { rm -rf "$POS"; box_mark "$POS/STARTED"; } \
            || box_finish "POSITIONS_UNWRITABLE ($POS)" 1
        from=$(box_size "$OUT/positions.log")
        box_bounded --stall "$OUT/positions.log" "$POSITIONS_STALL_MIN" positions "$POSITIONS_CUT_MIN" positions.log \
            python tools/critic_positions.py --out "$POS" --vocab-from "$REF" --matches "$MATCHES_DIR"/*__* \
            --corpus "$CORPUS" --bench configs/bench_states.json --workers "$WORKERS"
        tail -c "+$(( from + 1 ))" "$OUT/positions.log" | grep -q "BUILD_DONE" \
            || box_finish "POSITIONS_${BOX_WHY^^} rc=$BOX_RC (positions.log; it continues on re-entry)" 1
        cp -f "$POS/summary.json" "$OUT/positions_summary.json"
        cp -f "$POS/manifest.jsonl" "$OUT/positions_manifest.jsonl"
        box_bounded positions-check 5 positions.log python - "$POS/summary.json" <<'EOF' \
            || box_finish "POSITIONS_BARRIER rc=$BOX_RC (positions.log, positions_summary.json)" 1
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
counts = s["counts"]
print("positions", json.dumps(counts), "benchmark", s["bench_check"], flush=True)
games = s["games"]
errors = sum(c.get("games_error", 0) for c in counts.values())
assert s["done"], "the build did not finish"
assert errors < 0.01 * games, f"{errors} of {games} games failed"
for source in ("M", "H"):
    assert counts.get(source, {}).get("games_ok", 0) > 0, f"no {source} games"
assert s["bench_check"]["leaked"] == 0
EOF
        box_mark "$POS/BUILT"
        box_upload_async
    fi
fi

# ---- the benchmark's states: rebuilt from the corpus it was played from, encoded both ways
box_upload_dir bench "$OUT/bench"
if ! grep -q "REBUILD_DONE" "$OUT/bench_rebuild.log" 2>/dev/null || [ ! -f "$OUT/bench/states.pkl" ]; then
    box_bounded "bench rebuild" 30 bench_rebuild.log python tools/critic_bench.py rebuild \
        --records "$BENCH_DIR/validation.json" --dataset "$BENCH_DIR/replays_dataset_imitation" \
        --vocab-from "$REF" --out "$OUT/bench" --workers "$WORKERS" \
        || box_finish "BENCH_REBUILD_${BOX_WHY^^} rc=$BOX_RC (bench_rebuild.log)" 1
    cp -f "$OUT/bench/rebuild_summary.json" "$OUT/bench_rebuild_summary.json"
fi

# ---- the six critics
train_critic() {                 # train_critic NAME SOURCE VIEW FRACTION ARCH
    local name=$1 args from
    args=(--positions "$POS" --source "$2" --view "$3" --fraction "$4" --arch "$5" --out "$OUT/$name.pt"
          --device cuda --max-minutes "$CRITIC_MINUTES")
    [ "$5" = small ] || args+=(--init "$REF")
    # An attempt that did not finish starts over: its rows go.
    rm -f "$OUT/$name.pt" "$OUT/$name".{holdout,steps,signal}.jsonl "$OUT/$name.summary.json"
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before critic $name: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    from=$(box_size "$OUT/train_$name.log")
    box_bounded --stall "$OUT/train_$name.log" "$CRITIC_STALL_MIN" "critic $name" "$CRITIC_CUT_MIN" "train_$name.log" \
        python tools/critic_train.py "${args[@]}"
    if [ "$BOX_RC" -ne 0 ] || ! tail -c "+$(( from + 1 ))" "$OUT/train_$name.log" | grep -q "CRITIC_TRAIN_DONE" \
            || ! critic_done "$name"; then
        box_finish "CRITIC_${name}_${BOX_WHY^^} rc=$BOX_RC (train_$name.log) $(notes)" 1
    fi
    box_upload_async
}
for spec in "${CRITICS[@]}"; do
    read -r name source view fraction arch <<< "$spec"
    critic_done "$name" || train_critic "$name" "$source" "$view" "$fraction" "$arch"
done

# ---- the readout and its reading
read_args=()
for spec in "${CRITICS[@]}"; do
    name=${spec%% *}
    read_args+=(--critic "$name=$OUT/$name.pt")
done
box_bounded "bench read" 20 bench_read.log python tools/critic_bench.py read --out "$OUT/bench" \
    "${read_args[@]}" --device cuda || box_finish "BENCH_READ_${BOX_WHY^^} rc=$BOX_RC (bench_read.log)" 1
box_bounded "bench stats" 15 bench_stats.log python tools/critic_bench.py stats --out "$OUT/bench" \
    || box_finish "BENCH_STATS_${BOX_WHY^^} rc=$BOX_RC (bench_stats.log)" 1
cp -f "$OUT/bench/readout.json" "$OUT/readout.json"
cp -f "$OUT/bench/readout.md" "$OUT/readout.md"
reading=$(grep -o "READING [A-Za-z-]*" "$OUT/bench_stats.log" | tail -1)
box_finish "CRITIC_STEP1_DONE ${reading:-READING_UNKNOWN} (readout.md, prior_gaps.json)"
