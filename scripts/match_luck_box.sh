#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# The dice's share of a match's result (tools/match_luck.py,
# docs/match_luck_prereg_20261019.md), on one box, CPU only:
#   the tool's tests and the game records' tests, on the wheel built from
#   the stage;
#   the recorded argmax matches from HF (MATCHES: NAME=HF path=SHA-256),
#   each checked against its pin and fetched into $WORKDIR/match_data,
#   outside the records;
#   each match walked in a step of its own into rows.jsonl (a row per game,
#   written as each game finishes; a re-entry restores it from HF and walks
#   only the games it lacks), then the readout of the matches walked whole
#   (readout.txt, readout.json).
# A match that cannot be fetched, or whose walk fails, is left out
# (matches_failed.txt) and the others go on; a re-entry on the same machine
# keeps a failed walk out.
#
# The default MATCHES: 23 matches, about 19,500 games, every one both sides
# at argmax (raw:t0, or the deterministic look-ahead player), recorded
# whole, 2026-10-08 to 2026-10-12; four of them A/A (one network, one
# decode, on both sides: own64_selfplay, gen_own1_selfplay, selfpin,
# kv00), about 3,000 games.
#
# Box: CPU only (any GPU), 16 to 32 usable cores of a Ryzen or EPYC, at
# least 8 GB of memory (a walking worker holds about 230 MB), 20 GB of
# disk. Expected wall about 45 minutes: bring-up and the Rust build about
# 10, the tests 2, the downloads (123 MB) 2, the walk 10 to 25 (0.2 to
# 0.8 s a game after a 10 s warm-up per worker on the laptop, about 1.1 to
# 4.3 CPU-hours for the whole set, and a pool started per match), the
# readout 1, the final upload a few minutes. BOX_MAX_H 1.5.
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md): the
# onstart of scripts/rent_box.py fetches the library of STAGE, then this
# script. Every step has a bound; the dead-man's switch finishes the entry
# after BOX_MAX_H hours whatever it is doing. Records go to HF $HF_DIR every
# 30 minutes, after the walk and at the end. Every exit past the stage
# check, clean or not, uploads the records with ALL_DONE last and stops the
# instance.
# Never `set -x`: the HF token and the instance key are in the environment.
# box-needs: disk_gb=20 ram_gb=8 cores=16
set -uo pipefail
WORKDIR=/workspace
OUT=$WORKDIR/matchluck
DATA=$WORKDIR/match_data
STAGE="${STAGE:-}"
export HF_DIR="${HF_DIR:-tier-b/match_luck_20261019}"
JOBS="${JOBS:-}"                                 # empty: box_workers
# Bounds in minutes.
TESTS_CUT_MIN="${TESTS_CUT_MIN:-15}"             # the two files take about 20 s on the laptop
FETCH_CUT_MIN="${FETCH_CUT_MIN:-20}"
WALK_CUT_MIN="${WALK_CUT_MIN:-20}"               # one match: about 1,000 games, 1 to 3 minutes expected
WALK_STALL_MIN="${WALK_STALL_MIN:-10}"           # no row for this long: the match's walk is cut
READOUT_CUT_MIN="${READOUT_CUT_MIN:-15}"
BOX_MAX_H="${BOX_MAX_H:-1.5}"
BOX_WORKER_GB="${BOX_WORKER_GB:-0.5}"
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }

# NAME=HF path=SHA-256 of the tarball's content (read from HF 2026-10-19).
DEFAULT_MATCHES="
own64_selfplay=tier-b/own_confirm_20261011/games_own64_selfplay.tar.gz=ed8c54cc62c586e50a75646bb123c1c13f9503fb7a491797f3348efdd64e1449
own64_vs_p3=tier-b/own_confirm_20261011/games_own64_vs_p3.tar.gz=758c6ac06c52dfa60344eed6faf42ee9c206fb8dbdb07fd7a3af43468a25f634
own64_vs_own0=tier-b/own_confirm_20261011/games_own64_vs_own0.tar.gz=c0714ad3d170f370a629b856ed33002d4601256723f578aeb2365699b23d75ed
own0_vs_p3slots0=tier-b/own_confirm_20261011/games_own0_vs_p3slots0.tar.gz=ff8eba14822656b2022eb2dfa1dd7d081032455041e0e6413e239cd2c4e88811
gen_own1_selfplay=tier-b/memory_own_round2_20261011/games_gen_own1_selfplay.tar.gz=0f11ed16dda2302dbc36916db1a6b2a3f72245664267a1cfdbb6ea1c10e3c181
r1_own2_64_vs_own1_64=tier-b/memory_own_round2_20261011/games_r1_own2_64_vs_own1_64.tar.gz=ecde6901b24d765f26f7b0ce8fe87aee8a12c10e4c6155cb3485c994d58cc20f
r2_own2_0_vs_own1_0=tier-b/memory_own_round2_20261011/games_r2_own2_0_vs_own1_0.tar.gz=c1a03679773800c8b12a504176f50a1de7a502155f48087950eced3e7fc181bf
r3_own2_64_vs_own2_0=tier-b/memory_own_round2_20261011/games_r3_own2_64_vs_own2_0.tar.gz=a99719ffc63e7757b3a15ef2331f89412c18699a7dfd82197efec2c80b0fd4b0
m1_own64=tier-b/memory_own_trajectories_20261011/games_m1_own64.tar.gz=1155408497af9df62f7e97be0c3579b482ca8f31fa48111f0f2dd56d325427e9
m2_own0=tier-b/memory_own_trajectories_20261011/games_m2_own0.tar.gz=4106073a974f34888d93df32edade7c442a6435afb4c41cae6bef909725e36fd
confirm_kv15_vs_own1=tier-b/decode_knobs_20261012/games_confirm.tar.gz=51d77284b3c57c5e054366450f2c45bfcb2eee2adf6f5b984ae9103111134a22
kv00=tier-b/decode_knobs_20261012/games_kv00.tar.gz=e746814d76c26a7869065e407399aeb3fc21d0355e2bc6bbebe8066d06d94de2
kv10=tier-b/decode_knobs_20261012/games_kv10.tar.gz=562633ea6ba1c262fff417469bf0cfa093170b9a0eb4cf9b2c3a342d3d9541ef
kv15=tier-b/decode_knobs_20261012/games_kv15.tar.gz=e7475f4e07b30e86fb228700872e8590efe39a163c403dde76473957b7f45eba
la_exposure_vs_own1=tier-b/lookahead_exposure_gate_20261012/games_exposure.tar.gz=70b9ccf65ae4e0edf0dd593cad3843eea67d768a5bb1e8a04eed5bedb0c8f4c3
la_material_vs_parity3=tier-b/lookahead_material_gate_20261009/games_material.tar.gz=c63f50af568d4db3db431a3baa3a6494088afdcc9d33d4cfc22da214205ff87d
selfpin=tier-b/parity3_baselines_20261008/games_selfpin.tar.gz=b8bf7d1979061cb6d64f588dfda1b261bad711e8406bc4407102b1a794727f88
slots64_vs_slots0=tier-b/parity3_baselines_20261008/games_slots64_vs_slots0.tar.gz=9c9cf98735c78828343a0d088e645e07a98e4f9e302ee7fead51f39c8651b782
slots64_vs_slots16=tier-b/parity3_baselines_20261008/games_slots64_vs_slots16.tar.gz=750792cf7b855185fc2ab9ce562c27de43bcca0635859753aa0b4f45373f65d9
a1_eo25=tier-b/memory_in_play_parity3_20261009/games_a1_eo25.tar.gz=be9f72dbe05ef950ab9fbc11ff00f77dbecd8ec7a6ad82aa067a5e4c044a2a93
a2_reset=tier-b/memory_in_play_parity3_20261009/games_a2_reset.tar.gz=4bc6f56dee4c792125af26254febf4beb879e22d3bfbf10ba4c48bfe8be97204
a3_eo35=tier-b/memory_in_play_parity3_20261009/games_a3_eo35.tar.gz=cb27f6519cbe12956814c55c399b6a7d74d24e8aa175598c2cbca92a76516d62
own1_eo25_vs_own1=tier-b/rollout_regrade_20261012/games_own1_eo25_vs_own1.tar.gz=c8f1c980fa0e9e8edfe5e7681ff42bb60dee04a4e002f1abfd72b66063e153e8
"
MATCHES="${MATCHES:-$DEFAULT_MATCHES}"

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

# ---- the match list (begin): tests/test_match_luck_box.py runs this block and the next with the library stubbed
read -r -d '' -a MATCH_SPECS <<< "$MATCHES" || true    # NAME=HF path=SHA-256, one per word
MATCH_NAMES=()
for spec in ${MATCH_SPECS[@]+"${MATCH_SPECS[@]}"}; do
    [[ $spec =~ ^([A-Za-z0-9_]+)=[^=]+\.tar\.gz=[0-9a-f]{64}$ ]] \
        || box_finish "BAD_MATCHES: '$spec' is not NAME=HF_PATH.tar.gz=SHA256" 1
    case " ${MATCH_NAMES[*]} " in
        *" ${BASH_REMATCH[1]} "*) box_finish "BAD_MATCHES: ${BASH_REMATCH[1]} is named twice" 1 ;;
    esac
    MATCH_NAMES+=("${BASH_REMATCH[1]}")
done
[ "${#MATCH_NAMES[@]}" -gt 0 ] || box_finish "BAD_MATCHES: no match given" 1
# ---- the match list (end)

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

# ---- the matches (begin)
: > "$OUT/matches_failed.txt"
left_out() {                     # left_out NAME WHY: NAME is not read, and says why
    echo "LEFT_OUT $1: $2" | tee -a "$OUT/matches_failed.txt"
}
# The tarballs, each checked against its pin; data.txt lists those that
# landed (name, local path, bytes, HF path), written as each one lands.
rm -f "$OUT/data.txt"
box_bounded fetch "$FETCH_CUT_MIN" fetch.log python tools/match_luck.py fetch --data "$DATA" \
    --manifest "$OUT/data.txt" "${MATCH_SPECS[@]}"
FETCH_RC=$BOX_RC
FETCHED_NAMES=()
FETCHED_PATHS=()
if [ -f "$OUT/data.txt" ]; then
    while IFS=$'\t' read -r name path _; do
        [ -z "$name" ] || { FETCHED_NAMES+=("$name"); FETCHED_PATHS+=("$path"); }
    done < "$OUT/data.txt"
fi
fetched_path() {                 # fetched_path NAME: the local tarball of NAME; fails when it was not fetched
    local i
    for i in "${!FETCHED_NAMES[@]}"; do
        if [ "${FETCHED_NAMES[$i]}" = "$1" ]; then echo "${FETCHED_PATHS[$i]}"; return 0; fi
    done
    return 1
}
WALKED=()
walk_match() {                   # walk_match NAME: its rows, once; a walk that failed on this machine stays out
    local name=$1 path
    if box_marked_this_stage "$OUT/walk_$name.done"; then
        WALKED+=("$name")
        return 0
    fi
    if [ -f "$OUT/walk_$name.failed" ]; then
        left_out "$name" "its walk failed at an earlier entry, $(cat "$OUT/walk_$name.failed") (walk.log)"
        return 1
    fi
    path=$(fetched_path "$name") || { left_out "$name" "not fetched, fetch rc=$FETCH_RC (fetch.log)"; return 1; }
    box_bounded --stall "$OUT/rows.jsonl" "$WALK_STALL_MIN" "walk $name" "$WALK_CUT_MIN" walk.log \
        python tools/match_luck.py walk --out "$OUT/rows.jsonl" --jobs "$JOBS" --match "$name=$path"
    if [ "$BOX_RC" -ne 0 ]; then
        echo "rc=$BOX_RC $BOX_WHY" > "$OUT/walk_$name.failed"
        left_out "$name" "its walk failed, rc=$BOX_RC $BOX_WHY (walk.log)"
        return 1
    fi
    box_mark "$OUT/walk_$name.done"
    WALKED+=("$name")
}
touch "$OUT/rows.jsonl"
for name in "${MATCH_NAMES[@]}"; do
    walk_match "$name" || true
done
box_upload_async
[ "${#WALKED[@]}" -gt 0 ] || box_finish "NO_MATCH_WALKED $(notes)" 1

# The readout of the matches walked whole.
rm -f "$OUT/readout.txt" "$OUT/readout.json"
box_bounded readout "$READOUT_CUT_MIN" readout.txt python tools/match_luck.py readout "$OUT/rows.jsonl" \
    --matches "${WALKED[@]}" --save-json "$OUT/readout.json"
[ "$BOX_RC" -eq 0 ] || box_finish "READOUT_FAILED rc=$BOX_RC $(notes) (readout.txt)" 1
box_on_round
LEFT=$(sed -n 's/^LEFT_OUT \([^:]*\):.*/\1/p' "$OUT/matches_failed.txt" | sort -u | tr '\n' ' ')
LEFT=${LEFT% }
box_finish "MATCH_LUCK_DONE $(notes) walked ${#WALKED[@]} of ${#MATCH_NAMES[@]}; left out: ${LEFT:-none}"
# ---- the matches (end)
