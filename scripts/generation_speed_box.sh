#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# Self-play generation speed on one box (branch feature/generation-speed),
# round 2: production-configured matches of the reference against itself,
# the base code (main) and this stage, timed on the same box, with the
# same games, and how deep the games in flight can go with and without the
# graphed server.
#
# Two code stages: STAGE, this branch's (tools/stage_code.py --script
# scripts/generation_speed_box.sh), and BASE_STAGE, main at the commit the
# branch left (staged from a checkout of it with tools/stage_code.py --out
# ... --upload tier-b/staging/<name>.tar.gz). The base stage is unpacked
# beside this one and its match driver runs from there; both use this
# stage's Rust wheel (phase 32: main's adapter takes any phase from 31, and
# the phase adds an optional argument main never passes).
#
# Every arm is the round 2 generation match (training/metrics/
# memory_own_round2_20261011/: own1 against itself at 64 slots, the
# reference decode of configs/reference_player.json, bf16, the packed trunk
# and embed, game records on, --persistent-workers --shared-inference
# --device cuda) from SEED_BASE with no replacements: GAMES slots, or
# GAMES_PER_JOB per game in flight when that is more, so a deep arm is not
# mostly its tail; every arm plays the same first GAMES games (up to bf16
# near-tie flips: batch composition differs from run to run). The arms, in
# this order:
#   base_1, this_1, base_2, this_2: main and this stage's defaults (the
#     worker's masks and the view's units built by the core, the raw
#     player's decisions on the simulator's view, the stage timing), 20
#     games in flight, alternated;
#   the graphed check (tools/graphed_check.py, graphed_check.json): the
#     server's preflight at 120 games in flight (the largest bucket of its
#     caps captured into the graphs' shared pool), then the graphed server
#     against the eager one on own1 and this GPU up to batches of 120; a
#     crash barrier: anything but a pass leaves the graphed arms out;
#   inflight2, inflight3, inflight4, inflight6: 2, 3, 4 and 6 games per
#     worker process (--worker-games), 40, 60, 80 and 120 in flight;
#   graphed, graphed_inflight2, graphed_inflight3, graphed_inflight4,
#     graphed_inflight6: the same depths and 20 in flight on the graphed
#     server (--graphed-serve);
#   base_3: main again, after every flag arm;
#   best_repeat: the fastest arm of this stage (the server's decisions per
#     second) with no failed game, again.
# The session is valid when the three base runs lie within VALID_SPREAD of
# their mean on the server's decisions per second (requests over the
# server's wall); the base runs before and after, and the best arm and its
# repeat, bracket the drift. Every arm records its failed games (the
# driver's failed_*.json records); an arm with one is a failed arm in
# speed.json and in arms_failed.txt, and the run goes on. Per arm the
# records keep the match's wall and games, the server's stats (requests,
# batches, mean batch, infer, window, idle and reply seconds, the device
# span, failed batches and requests; the graphed summary with its preflight
# and peak reserved memory) and, for this stage, the workers' decision
# stages (.worker_stages.json); speed.json reads them: games per hour, the
# server's decisions per second and infer ms per batch, each arm against
# the base runs' mean. Game walls alone cannot resolve small wins on a
# shared host (identical repeats have differed 1.6-1.8x, the base runs of
# round 1 by 18%); the server's counters and the stage times are the
# primary readings.
#
# Expected wall on a 4090 with 31-32 usable cores (round 1's walls:
# training/metrics/generation_speed_20261013/walls.txt): bring-up, wheel
# and tests 15 minutes, the graphed check 5, the three base arms 7 to 8
# minutes each, this stage's two 6, the eight deep arms 3 to 5 each (the
# 120-in-flight ones play 600 games), the repeat 4: about 1.6 hours,
# $0.70-1.00 at $0.42-0.60 an hour. BOX_MAX_H 3 caps it at $1.30-1.80 plus
# the final round.
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md). No
# periodic upload rounds while the arms run (they would share the machine
# with the timing): the records go up after each arm and at the finish.
# Re-entry skips the arms that are done. Never `set -x`: the HF token and
# the instance key are in the environment.
# box-needs: disk_gb=60 ram_gb=64 gpu_ram_gb=24 cores=32 gpu=4090
set -uo pipefail
WORKDIR=/workspace
STAGE="${STAGE:-}"
BASE_STAGE="${BASE_STAGE:-}"
HF_DIR="${HF_DIR:-tier-b/generation_speed_r2}"
export HF_DIR
BOX_MAX_H="${BOX_MAX_H:-3}"
GAMES="${GAMES:-400}"
GAMES_PER_JOB="${GAMES_PER_JOB:-5}"
DEPTHS="${DEPTHS:-2 3 4 6}"                     # games per worker process of the deep arms, 20 workers each
# No committed result carries a seed from 130000 to 130999.
SEED_BASE="${SEED_BASE:-130000}"
VALID_SPREAD="${VALID_SPREAD:-0.05}"
TESTS_CUT_MIN="${TESTS_CUT_MIN:-40}"
ARM_CUT_MIN="${ARM_CUT_MIN:-30}"
MATCH_BUDGET_MIN="${MATCH_BUDGET_MIN:-22}"     # the match stops admitting games here; the arm's cut is ARM_CUT_MIN
OUT=$WORKDIR/generation_speed
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }

BASE_REPO=$WORKDIR/base_repo
ARMS="base_1 this_1 base_2 this_2"
for d in $DEPTHS; do ARMS="$ARMS inflight$d"; done
ARMS="$ARMS graphed"
for d in $DEPTHS; do ARMS="$ARMS graphed_inflight$d"; done
ARMS="$ARMS base_3 best_repeat"

notes() {                        # what is done, for the finish reason
    local name done_=()
    for name in $ARMS; do
        ! box_marked_this_stage "$OUT/arm_$name.done" || done_+=("$name")
    done
    echo "done: ${done_[*]:-nothing}"
}
# shellcheck disable=SC2317 # called by the library, in the reason of an unexpected exit
box_notes() { notes; }

box_init
[ -n "$STAGE" ] || box_finish "NO_STAGE: build the code stage (tools/stage_code.py) and pass STAGE" 1
[ -n "$BASE_STAGE" ] || box_finish "NO_BASE_STAGE: stage main's commit and pass BASE_STAGE" 1
box_bind_run_stage
box_restore speed.json graphed_check.json tests.log || box_finish "RESTORE_FAILED (restore.log)" 1
box_pip huggingface_hub psutil pytest scipy requests || echo "pip install failed (pip.log)"

# ---- the code, the base code and the Rust wheel
box_stage_code || box_finish "CODE_STAGING_FAILED (staging.log)" 1
cd "$BOX_REPO" || box_finish "CODE_STAGING_FAILED (no $BOX_REPO)" 1
box_build_wheel || box_finish "BUILD_FAILED (build.log)" 1
if [ "$(cat "$BASE_REPO/.staged_from" 2>/dev/null)" != "$BASE_STAGE" ]; then
    rm -rf "$BASE_REPO.tmp"
    box_bounded base-stage 15 staging.log python "$BOX_LIB/box_stage.py" fetch "$BASE_STAGE" --into "$BASE_REPO.tmp" \
        || box_finish "BASE_STAGING_FAILED (staging.log)" 1
    { printf '%s\n' "$BASE_STAGE" > "$BASE_REPO.tmp/.staged_from" && rm -rf "$BASE_REPO" \
        && mv "$BASE_REPO.tmp" "$BASE_REPO"; } || box_finish "BASE_STAGING_FAILED (unpacking)" 1
fi
[ -f "$BASE_REPO/tools/run_elo_batch.py" ] || box_finish "BASE_STAGING_FAILED (no match driver in $BASE_STAGE)" 1
box_facts > "$OUT/box.txt.tmp" 2>&1
mv -f "$OUT/box.txt.tmp" "$OUT/box.txt"
timeout 2m python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || box_finish "NO_CUDA (box.txt)" 1
box_upload

# ---- the tests of what this stage changed, on this wheel; the CUDA tests are recorded
# (test_cuda_seam_matches_reference_enumeration fails on main as well: its
# CPU reference and its CUDA forward differ by more than its tolerance)
if ! box_marked_this_stage "$BOX_STATE/TESTED"; then
    : > "$OUT/tests.log"
    box_bounded tests "$TESTS_CUT_MIN" tests.log python -m pytest tests/test_graphed_serve.py \
        tests/test_core_masks.py tests/test_core_view.py tests/test_decision_state.py tests/test_eval_workers.py \
        tests/test_eval_inference_server.py tests/test_stage_timing.py tests/test_match_memory.py \
        tests/test_server_priors.py tests/test_game_core.py tests/test_memory_model.py \
        tests/test_eval_match_failures.py tests/test_raw_player.py tests/test_raw_player_compact.py \
        tests/test_inference_seam.py tests/test_leaf_wire.py -q -p no:cacheprovider -m ""
    tail -n 3 "$OUT/tests.log"
    [ "$BOX_RC" -eq 0 ] || box_finish "TESTS_FAILED rc=$BOX_RC (tests.log)" 1
    : > "$OUT/cuda_tests.log"
    box_bounded cuda-tests 15 cuda_tests.log python -m pytest tests/test_server_priors_cuda.py \
        -q -p no:cacheprovider -m ""
    tail -n 3 "$OUT/cuda_tests.log"
    box_mark "$BOX_STATE/TESTED"
fi
CUDA_TESTS=$(grep -Eo "[0-9]+ (passed|failed|error|errors|skipped)" "$OUT/cuda_tests.log" 2>/dev/null | tr '\n' ' ')
box_upload

# ---- the reference: its checkpoint (SHA-256 checked) and the self-play sides' flags
box_bounded reference 15 staging.log python tools/reference_player.py --ensure \
    || box_finish "REFERENCE_MISSING rc=$BOX_RC (staging.log)" 1
REF_PT=$BOX_REPO/$(timeout 1m python -c "import json; print(json.load(open('configs/reference_player.json'))['checkpoint_local'])") \
    || box_finish "REFERENCE_CONFIG_UNREADABLE (configs/reference_player.json)" 1
[ -f "$REF_PT" ] || box_finish "REFERENCE_MISSING ($REF_PT)" 1
side_flags() {                   # side_flags SIDE LABEL: the reference's flags for SIDE, one per line, under LABEL, its checkpoint by absolute path
    local side=$1 label=$2 flags i
    mapfile -t flags < <(timeout 1m python tools/reference_player.py --flags "$side" | tr ' ' '\n')
    for (( i = 0; i + 1 < ${#flags[@]}; i++ )); do
        case ${flags[i]} in
            "--label-$side") flags[i + 1]=$label ;;
            "--spec-$side") flags[i + 1]=$REF_PT ;;
        esac
    done
    printf '%s\n' "${flags[@]}"
}
mapfile -t SIDE_A < <(side_flags a own1a)
mapfile -t SIDE_B < <(side_flags b own1b)
[[ " ${SIDE_A[*]} " == *" --spec-a $REF_PT "* && " ${SIDE_B[*]} " == *" --memory-b 64 "* ]] \
    || box_finish "REFERENCE_FLAGS_FAILED (tools/reference_player.py --flags): ${SIDE_A[*]} | ${SIDE_B[*]}" 1

# ---- the graphed server against the eager one on own1 and this GPU, at the deepest depth (a crash barrier)
MAX_JOBS=20
for d in $DEPTHS; do (( d * 20 > MAX_JOBS )) && MAX_JOBS=$(( d * 20 )); done
if [ ! -f "$OUT/graphed_check.json" ]; then
    rm -f "$OUT/graphed_check.json.pending"
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before the graphed check: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    box_bounded graphed-check 25 graphed_check.log python tools/graphed_check.py --checkpoint "$REF_PT" \
        --device cuda --max-batch "$MAX_JOBS" --batch-sizes "1,4,9,16,20,32,40,60,80,$MAX_JOBS" \
        --out "$OUT/graphed_check.json.pending"
    case $BOX_RC in
        0|8) mv -f "$OUT/graphed_check.json.pending" "$OUT/graphed_check.json" ;;
        *) printf '{"pass": false, "rc": %d, "why": "%s", "log": "graphed_check.log"}\n' "$BOX_RC" "$BOX_WHY" \
               > "$OUT/graphed_check.json" ;;
    esac
    box_upload
fi
GRAPHED_OK=$(timeout 1m python -c "import json, sys; print(int(json.load(open(sys.argv[1]))['pass']))" \
    "$OUT/graphed_check.json" 2>/dev/null)

# ---- the arms
arm() {                          # arm NAME REPO JOBS FLAGS...: one match, once; false when it failed or lost a game
    local name=$1 repo=$2 jobs=$3 dir="$WORKDIR/games_$1" t0 f games failed
    shift 3
    ! box_marked_this_stage "$OUT/arm_$name.done" || return 0
    box_gpu_ok || box_finish "GPU_UNRESPONSIVE before arm $name: rc=$BOX_RC $BOX_WHY (gpu.log)" 1
    rm -rf "$dir"
    games=$(( GAMES_PER_JOB * jobs > GAMES ? GAMES_PER_JOB * jobs : GAMES ))
    cd "$repo" || box_finish "NO_REPO $repo" 1
    echo "$(box_stamp) $name: $repo, jobs $jobs, games $games, $*" >> "$OUT/arm_$name.log"
    t0=$(box_now)
    box_bounded "arm $name" "$ARM_CUT_MIN" "arm_$name.log" python tools/run_elo_batch.py \
        "${SIDE_A[@]}" "${SIDE_B[@]}" --outdir "$dir" --games "$games" --max-extra-games 0 \
        --seed-base "$SEED_BASE" --mcts-sims 0 --raw-temperature-a 0 --raw-temperature-b 0 \
        --persistent-workers --shared-inference --no-infer-compile --device cuda --jobs "$jobs" \
        --time-budget-min "$MATCH_BUDGET_MIN" "$@"
    cd "$BOX_REPO" || box_finish "NO_REPO $BOX_REPO" 1
    failed=$(find "$dir" -maxdepth 1 -name 'failed_*.json' 2>/dev/null | wc -l)
    printf '{"arm": "%s", "repo": "%s", "jobs": %d, "flags": "%s", "slots": %d, "wall_s": %d, "games": %d, "failed_games": %d, "rc": %d, "why": "%s"}\n' \
        "$name" "$repo" "$jobs" "$*" "$games" "$(( $(box_now) - t0 ))" \
        "$(find "$dir" -maxdepth 1 -name 'game_*.json' 2>/dev/null | wc -l)" "$failed" "$BOX_RC" "$BOX_WHY" \
        > "$OUT/arm_$name.json"
    for f in "$dir"/.inference_server_*.json "$dir"/.worker_stages.json; do
        [ -f "$f" ] && cp -f "$f" "$OUT/arm_${name}.${f##*/.}"
    done
    box_mark "$OUT/arm_$name.done"
    box_upload
    # run_elo_batch exits 3 or 4 when its games end without enough decisive
    # results (no replacements here): the timing is whole. A failed game, or
    # any other exit, fails the arm.
    case $BOX_RC in
        0|3|4) ;;
        *) echo "ARM_FAILED $name rc=$BOX_RC $BOX_WHY (arm_$name.log)" | tee -a "$OUT/arms_failed.txt"
           return 1 ;;
    esac
    if [ "$failed" -gt 0 ]; then
        echo "ARM_FAILED $name: $failed failed games (arm_$name.log, games_$name/failed_*.json)" \
            | tee -a "$OUT/arms_failed.txt"
        return 1
    fi
}
for name in base_1 this_1 base_2 this_2; do
    repo=$BOX_REPO
    [ "${name%_*}" = this ] || repo=$BASE_REPO
    arm "$name" "$repo" 20 || box_finish "ARM_FAILED $name (arm_$name.log) $(notes)" 1
done
for d in $DEPTHS; do
    arm "inflight$d" "$BOX_REPO" $(( 20 * d )) --worker-games "$d" || true
done
if [ "$GRAPHED_OK" = 1 ]; then
    arm graphed "$BOX_REPO" 20 --graphed-serve || true
    for d in $DEPTHS; do
        arm "graphed_inflight$d" "$BOX_REPO" $(( 20 * d )) --worker-games "$d" --graphed-serve || true
    done
else
    echo "graphed arms left out: the graphed check did not pass (graphed_check.json)" | tee -a "$OUT/arms_failed.txt"
fi
arm base_3 "$BASE_REPO" 20 || echo "the last base arm failed: the drift is not bracketed" >> "$OUT/arms_failed.txt"
# The fastest arm of this stage with no failed game, again.
BEST=$(timeout 1m python - "$OUT" <<'EOF'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
best = None
for p in sorted(out.glob("arm_*.json")):
    if p.name.count(".") > 1:
        continue
    a = json.loads(p.read_text(encoding="utf-8"))
    s = out / f"arm_{a['arm']}.inference_server_0.json"
    if a["arm"].startswith("base") or a["arm"] == "best_repeat" or a["failed_games"] or a["rc"] not in (0, 3, 4) \
            or not s.exists():
        continue
    srv = json.loads(s.read_text(encoding="utf-8"))
    rate = srv["requests"] / srv["wall_s"] if srv.get("wall_s") else 0.0
    if best is None or rate > best[0]:
        best = (rate, a)
if best is not None:
    print(best[1]["arm"], best[1]["jobs"], best[1]["flags"])
EOF
)
if [ -n "$BEST" ]; then
    read -r BEST_ARM BEST_JOBS BEST_FLAGS <<< "$BEST"
    echo "$BEST_ARM" > "$OUT/best.txt"
    # shellcheck disable=SC2086 # BEST_FLAGS is a list of flags
    arm best_repeat "$BOX_REPO" "$BEST_JOBS" $BEST_FLAGS \
        || echo "the best arm's repeat failed: its drift is not bracketed" >> "$OUT/arms_failed.txt"
else
    echo "no arm of this stage to repeat" >> "$OUT/arms_failed.txt"
fi

# ---- speed.json
box_bounded summary 5 summary.log python - "$OUT" "$VALID_SPREAD" <<'EOF'
import json, pathlib, sys
out, spread = pathlib.Path(sys.argv[1]), float(sys.argv[2])
arms = {}
for p in sorted(out.glob("arm_*.json")):
    if p.name.count(".") > 1:
        continue
    a = json.loads(p.read_text(encoding="utf-8"))
    name = a["arm"]
    servers = [json.loads(s.read_text(encoding="utf-8")) for s in sorted(out.glob(f"arm_{name}.inference_server_*.json"))]
    row = {"jobs": a["jobs"], "flags": a["flags"], "rc": a["rc"], "slots": a.get("slots"), "games": a["games"],
           "failed_games": a.get("failed_games", 0), "wall_s": a["wall_s"],
           "games_per_hour": round(3600.0 * a["games"] / a["wall_s"], 1) if a["wall_s"] else None}
    if servers:
        s = servers[0]
        row["server"] = {k: s.get(k) for k in ("requests", "batches", "mean_batch", "infer_s", "window_s",
                                                "idle_s", "reply_s", "gpu_ms", "wall_s", "graphed",
                                                "connections", "max_batch", "failed_batches", "failed_requests",
                                                "fatal", "graphed_serve")}
        row["decisions_per_s"] = round(s["requests"] / s["wall_s"], 2) if s.get("wall_s") else None
        row["infer_ms_per_batch"] = round(1e3 * s["infer_s"] / s["batches"], 3) if s.get("batches") else None
        row["device_ms_per_batch"] = round(s["gpu_ms"] / s["batches"], 3) if s.get("batches") else None
    row["failed"] = bool(row["failed_games"] or a["rc"] not in (0, 3, 4)
                         or (servers and (servers[0].get("failed_requests") or servers[0].get("fatal"))))
    stages = out / f"arm_{name}.worker_stages.json"
    if stages.exists():
        w = json.loads(stages.read_text(encoding="utf-8"))
        row["worker"] = {k: w[k] for k in ("decisions", "decision_wall_ms", "decision_cpu_ms", "own_wall_ms",
                                           "stage_ms_per_decision")}
    arms[name] = row


def rate(name, key="decisions_per_s"):
    return arms.get(name, {}).get(key)


def ratio(a, b, key="decisions_per_s"):
    x, y = rate(a, key), rate(b, key)
    return round(x / y, 4) if x and y else None


bases = [n for n in ("base_1", "base_2", "base_3") if rate(n)]
mean = sum(rate(n) for n in bases) / len(bases) if bases else None
best = (out / "best.txt").read_text(encoding="utf-8").strip() if (out / "best.txt").exists() else None
base_rate = {k: sum(arms[n][k] for n in bases) / len(bases) for k in ("decisions_per_s", "games_per_hour")
             if bases and all(arms[n].get(k) for n in bases)}
summary = {
    "arms": arms,
    # main against this stage holds when the base runs, before and after
    # the flag arms, agree on the server's decisions per second.
    "valid": len(bases) >= 2 and all(abs(rate(n) / mean - 1.0) <= spread for n in bases),
    "base_spread": {n: round(rate(n) / mean, 4) for n in bases} if mean else {},
    "best": best,
    "best_vs_repeat": ratio("best_repeat", best) if best else None,
    "failed_arms": sorted(n for n, r in arms.items() if r["failed"]),
    "against_base": {n: {k: round(r[k] / base_rate[k], 4) for k in base_rate if r.get(k)}
                     for n, r in arms.items()},
    "against_this_1": {n: {k: ratio(n, "this_1", k) for k in ("decisions_per_s", "games_per_hour",
                                                                "infer_ms_per_batch")}
                       for n in arms if not n.startswith("base")},
}
(out / "speed.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print(json.dumps({"valid": summary["valid"], "base_spread": summary["base_spread"], "best": best,
                  "best_vs_repeat": summary["best_vs_repeat"], "failed_arms": summary["failed_arms"],
                  "decisions_per_s": {n: r.get("decisions_per_s") for n, r in arms.items()}}), flush=True)
EOF
SUMMARY=$(tail -n 1 "$OUT/summary.log" | cut -c1-600)
FAILED_ARMS=$(awk '$1 == "ARM_FAILED" {print $2}' "$OUT/arms_failed.txt" 2>/dev/null | tr -d ':' | sort -u | tr '\n' ' ')
box_finish "GENERATION_SPEED_DONE $SUMMARY; cuda tests: ${CUDA_TESTS:-none}; failed: ${FAILED_ARMS:-none}"
