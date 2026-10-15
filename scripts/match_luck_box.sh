#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# The dice's share of a match's result (tools/match_luck.py), on one CPU box:
#   the tool's tests and the game records' tests, on the wheel built from
#   the stage;
#   the recorded argmax matches from HF (MATCHES, NAME=HF path), fetched
#   into $WORKDIR/match_data, outside the records;
#   every game walked into rows.jsonl (a row per game, written as each game
#   finishes; a re-entry restores it from HF and walks only the games it
#   lacks), then the readout (readout.txt, readout.json).
#
# The default MATCHES: 23 matches, about 19,500 games, every one both sides
# at argmax (raw:t0, or the deterministic look-ahead player), recorded
# whole, 2026-10-08 to 2026-10-12; four of them A/A (one network, one
# decode, on both sides: own64_selfplay, gen_own1_selfplay, selfpin,
# kv00), about 3,000 games.
#
# Box: CPU only, 16 to 32 usable cores, 16 GB of memory (a walking worker
# holds about 230 MB), 20 GB of disk. Expected wall about 35 minutes:
# bring-up and the Rust build about 10, the tests 2, the downloads
# (about 130 MB) 2, the walk 10 to 15 (0.2 to 0.8 s a game after a 10 s
# warm-up per worker on the laptop, about 3 CPU-hours for the whole set,
# longer games for the parity3 baselines), the readout 1, the final upload
# a few minutes. Cost at $0.10-0.30 an hour: $0.06-0.18 expected;
# BOX_MAX_H 1.5 caps it at $0.15-0.45 plus the final round.
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md): the
# onstart of scripts/rent_box.py fetches the library of STAGE, then this
# script. Every step has a bound; the dead-man's switch finishes the entry
# after BOX_MAX_H hours whatever it is doing. Records go to HF $HF_DIR every
# 30 minutes and at the end. Every exit past the stage check, clean or not,
# uploads the records with ALL_DONE last and stops the instance.
# Never `set -x`: the HF token and the instance key are in the environment.
# box-needs: disk_gb=20 ram_gb=16 cores=16
set -uo pipefail
WORKDIR=/workspace
OUT=$WORKDIR/matchluck
DATA=$WORKDIR/match_data
STAGE="${STAGE:-}"
export HF_DIR="${HF_DIR:-tier-b/match_luck_20261015}"
JOBS="${JOBS:-}"                                 # empty: box_workers
# Bounds in minutes.
TESTS_CUT_MIN="${TESTS_CUT_MIN:-15}"             # the two files take about 20 s on the laptop
FETCH_CUT_MIN="${FETCH_CUT_MIN:-20}"
WALK_CUT_MIN="${WALK_CUT_MIN:-60}"               # 4x the expected walk
WALK_STALL_MIN="${WALK_STALL_MIN:-15}"           # no row for this long: the walk is cut
READOUT_CUT_MIN="${READOUT_CUT_MIN:-15}"
BOX_MAX_H="${BOX_MAX_H:-1.5}"
BOX_WORKER_GB="${BOX_WORKER_GB:-0.5}"
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }

DEFAULT_MATCHES="
own64_selfplay=tier-b/own_confirm_20261011/games_own64_selfplay.tar.gz
own64_vs_p3=tier-b/own_confirm_20261011/games_own64_vs_p3.tar.gz
own64_vs_own0=tier-b/own_confirm_20261011/games_own64_vs_own0.tar.gz
own0_vs_p3slots0=tier-b/own_confirm_20261011/games_own0_vs_p3slots0.tar.gz
gen_own1_selfplay=tier-b/memory_own_round2_20261011/games_gen_own1_selfplay.tar.gz
r1_own2_64_vs_own1_64=tier-b/memory_own_round2_20261011/games_r1_own2_64_vs_own1_64.tar.gz
r2_own2_0_vs_own1_0=tier-b/memory_own_round2_20261011/games_r2_own2_0_vs_own1_0.tar.gz
r3_own2_64_vs_own2_0=tier-b/memory_own_round2_20261011/games_r3_own2_64_vs_own2_0.tar.gz
m1_own64=tier-b/memory_own_trajectories_20261011/games_m1_own64.tar.gz
m2_own0=tier-b/memory_own_trajectories_20261011/games_m2_own0.tar.gz
confirm_kv15_vs_own1=tier-b/decode_knobs_20261012/games_confirm.tar.gz
kv00=tier-b/decode_knobs_20261012/games_kv00.tar.gz
kv10=tier-b/decode_knobs_20261012/games_kv10.tar.gz
kv15=tier-b/decode_knobs_20261012/games_kv15.tar.gz
la_exposure_vs_own1=tier-b/lookahead_exposure_gate_20261012/games_exposure.tar.gz
la_material_vs_parity3=tier-b/lookahead_material_gate_20261009/games_material.tar.gz
selfpin=tier-b/parity3_baselines_20261008/games_selfpin.tar.gz
slots64_vs_slots0=tier-b/parity3_baselines_20261008/games_slots64_vs_slots0.tar.gz
slots64_vs_slots16=tier-b/parity3_baselines_20261008/games_slots64_vs_slots16.tar.gz
a1_eo25=tier-b/memory_in_play_parity3_20261009/games_a1_eo25.tar.gz
a2_reset=tier-b/memory_in_play_parity3_20261009/games_a2_reset.tar.gz
a3_eo35=tier-b/memory_in_play_parity3_20261009/games_a3_eo35.tar.gz
own1_eo25_vs_own1=tier-b/rollout_regrade_20261012/games_own1_eo25_vs_own1.tar.gz
"
MATCHES="${MATCHES:-$DEFAULT_MATCHES}"
read -r -d '' -a MATCH_PAIRS <<< "$MATCHES" || true    # NAME=HF path, one per word

rows() {                         # rows: the rows walked so far
    if [ -f "$OUT/rows.jsonl" ]; then wc -l < "$OUT/rows.jsonl"; else echo 0; fi
}
notes() {
    local errors=0
    [ ! -f "$OUT/rows.jsonl" ] || errors=$(grep -c '"error"' "$OUT/rows.jsonl")
    echo "rows=$(rows) errors=$errors readout=$([ -s "$OUT/readout.json" ] && echo yes || echo no)"
}
# shellcheck disable=SC2317 # called by the library, in the reason of an unexpected exit
box_notes() { notes; }
box_on_round() {                 # progress.txt, before each upload round
    { date -u
      notes
      tail -n 3 "$OUT/walk.log" 2>/dev/null
    } > "$OUT/progress.txt.tmp" && mv -f "$OUT/progress.txt.tmp" "$OUT/progress.txt"
}

box_init
[ -n "$STAGE" ] || box_finish "NO_STAGE: build the code stage (tools/stage_code.py) and pass STAGE" 1
box_bind_run_stage
box_restore rows.jsonl || box_finish "RESTORE_FAILED (restore.log)" 1
box_pip huggingface_hub psutil pytest numpy || echo "pip install failed (pip.log)"

# ---- the code and the Rust wheel (the walk replays every game on the core)
box_stage_code || box_finish "CODE_STAGING_FAILED (staging.log)" 1
cd "$BOX_REPO" || box_finish "CODE_STAGING_FAILED (no $BOX_REPO)" 1
box_build_wheel || box_finish "BUILD_FAILED (build.log)" 1
box_facts > "$OUT/box.txt.tmp" 2>&1
mv -f "$OUT/box.txt.tmp" "$OUT/box.txt"
[ -n "$JOBS" ] || JOBS=$(box_workers)
echo "jobs $JOBS" >> "$OUT/box.txt"
box_upload_async
box_monitor_start

# ---- the crash barrier: the tool's tests and the records' on this wheel
if ! box_marked_this_stage "$BOX_STATE/TESTED"; then
    : > "$OUT/tests.log"
    box_bounded tests "$TESTS_CUT_MIN" tests.log python -m pytest tests/test_match_luck.py \
        tests/test_game_record.py -q -p no:cacheprovider -m ""
    tail -n 3 "$OUT/tests.log"
    [ "$BOX_RC" -eq 0 ] || box_finish "TESTS_FAILED rc=$BOX_RC (tests.log)" 1
    box_mark "$BOX_STATE/TESTED"
fi

# ---- the matches' games from HF: data.txt lists name, local path, bytes, HF path
mkdir -p "$DATA"
box_bounded fetch "$FETCH_CUT_MIN" fetch.log python - "$DATA" "$OUT/data.txt" "${MATCH_PAIRS[@]}" <<'EOF' \
    || box_finish "FETCH_FAILED rc=$BOX_RC (fetch.log)" 1
import os
import sys

from huggingface_hub import hf_hub_download

data, manifest, *pairs = sys.argv[1:]
lines = []
for pair in pairs:
    name, _, path = pair.partition("=")
    local = hf_hub_download("momom2/wesnoth-model-checkpoints", path, cache_dir=data)
    lines.append(f"{name}\t{local}\t{os.path.getsize(local)}\t{path}")
    print(lines[-1], flush=True)
with open(manifest + ".tmp", "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
os.replace(manifest + ".tmp", manifest)
EOF
WALK_ARGS=()
while IFS=$'\t' read -r name path _; do
    [ -z "$name" ] || WALK_ARGS+=(--match "$name=$path")
done < "$OUT/data.txt"
[ "${#WALK_ARGS[@]}" -eq $(( 2 * ${#MATCH_PAIRS[@]} )) ] \
    || box_finish "FETCH_INCOMPLETE: data.txt lists $(( ${#WALK_ARGS[@]} / 2 )) of ${#MATCH_PAIRS[@]} matches" 1

# ---- the walk: one row per game, the games already in rows.jsonl skipped
touch "$OUT/rows.jsonl"
box_bounded --stall "$OUT/rows.jsonl" "$WALK_STALL_MIN" walk "$WALK_CUT_MIN" walk.log \
    python tools/match_luck.py walk --out "$OUT/rows.jsonl" --jobs "$JOBS" "${WALK_ARGS[@]}"
WALK_RC=$BOX_RC WALK_WHY=$BOX_WHY
box_upload_async
[ "$WALK_RC" -eq 0 ] || box_finish "WALK_FAILED rc=$WALK_RC $WALK_WHY $(notes) (walk.log)" 1

# ---- the readout
rm -f "$OUT/readout.txt" "$OUT/readout.json"
box_bounded readout "$READOUT_CUT_MIN" readout.txt \
    python tools/match_luck.py readout "$OUT/rows.jsonl" --save-json "$OUT/readout.json"
[ "$BOX_RC" -eq 0 ] || box_finish "READOUT_FAILED rc=$BOX_RC $(notes) (readout.txt)" 1
box_on_round
box_finish "MATCH_LUCK_DONE $(notes)"
