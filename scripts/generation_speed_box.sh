#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# Self-play generation speed on one box, round 5 (branch
# feature/generation-speed-5): the server's device work. Production-
# configured matches of the reference against itself, timed on one box with
# the same games: the base code (main), round 4's fastest configuration on
# this branch (the reference arm), then this branch's device levers one at a
# time and together, at 120 and 160 games in flight with six games per
# worker process, and the server's device work profiled.
#
# Two code stages: STAGE, this branch's (tools/stage_code.py --script
# scripts/generation_speed_box.sh), and BASE_STAGE, main as the lead staged
# it (tools/stage_code.py --out ... --upload tier-b/staging/<name>.tar.gz),
# unpacked beside this one, its match driver run from there on this stage's
# Rust wheel (phase 34; no Rust change this round).
#
# Round 4 on this box type (training/metrics/generation_speed_r4_20261014/):
# all_p120x6 1,612 decisions a second, the server's device busy 0.78 of its
# wall, so the GPU is the bound; about 0.48 ms of device time a decision.
# The levers, each with its switch on run_elo_batch:
#   priors:  --priors-only, the servers compute the policy's priors alone (no
#            value head, no belief head): the same priors, the same games;
#   compile: --compile-packed, the trunk's layer loop compiled by inductor
#            (its elementwise work fused): a rounding change;
#   bf16res: --bf16-residual, the trunk's residual stream in bf16 (half its
#            elementwise traffic, no casts between the layers): a rounding
#            change.
# A rounding change plays other games (2026-09-11: the compiled loop made
# games 25% longer on relset), so every arm records its decisions per game;
# the games record both switches (infer_compile, infer_bf16_residual), which
# are for self-play generation only, never a strength match.
#
# The arms, in this order (pJxG: J games in flight, G per worker process):
#   base_1: main, 20 in flight, one game per worker process;
#   ref_p120x6: this branch at round 4's fastest configuration (the Rust
#     stage, the memory kept on the server, the incremental view);
#   profile_ref: the reference arm with PROFILE_BATCHES of its server's
#     batches under torch.profiler after PROFILE_AFTER (tools/
#     server_profile.py: the device time by kernel class and by region, the
#     heaviest kernels, the trunk's achieved TFLOP/s); not a timing arm;
#   priors_p120x6, compile_p120x6, bf16res_p120x6, compile_bf16res_p120x6;
#   base_2;
#   ref_p160x6, compile_p160x6, bf16res_p160x6;
#   profile_best: the fastest timing arm, profiled;
#   base_3;
#   best_repeat: the fastest timing arm with no failed game, again.
# The session is valid when the three base runs lie within VALID_SPREAD of
# their mean on the server's decisions per second (requests over the
# server's wall); the base runs before, between and after, and the best arm
# and its repeat, bracket the drift. Each arm fails alone: a match that
# crashes or loses a game is a failed arm (arms_failed.txt, speed.json) and
# the next arm runs. Per arm the records keep the match's wall and games,
# each server's stats (requests, batches, mean batch, idle, window, read,
# infer and reply seconds, the device span, the host seconds by stage,
# failed batches and requests, the memory states kept and released, the
# profile) and the workers' decision stages (.worker_stages.json).
# speed.json is rewritten after every arm: games per hour, the servers'
# decisions per second, decisions per game, infer and device ms per batch,
# host ms per batch by stage, the profile, each arm against the base runs'
# mean and against the reference arm. Game walls alone cannot resolve small
# wins on a shared host (identical repeats have differed 1.6-1.8x); the
# server counters and the stage times are the primary readings.
#
# Expected wall on a 4090 with 31-32 usable cores (round 4's walls):
# bring-up, wheel and tests 20-25 minutes (the CUDA tests compile the loop),
# the three base arms 6 to 7 minutes each, the twelve other arms 3 to 6 each
# (600 or 800 games each; a compiled arm's server compiles for about a
# minute first): about 1.8 hours, $0.75-1.10 at $0.42-0.60 an hour.
# BOX_MAX_H 3 caps it at $1.30-1.80 plus the final round.
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md). No
# periodic upload rounds while the arms run (they would share the machine
# with the timing): the records go up after each arm and at the finish,
# which uploads before it stops the instance. Re-entry skips the arms that
# are done. Never `set -x`: the HF token and the instance key are in the
# environment.
# box-needs: disk_gb=60 ram_gb=64 gpu_ram_gb=24 cores=32 gpu=4090
set -uo pipefail
WORKDIR=/workspace
STAGE="${STAGE:-}"
BASE_STAGE="${BASE_STAGE:-}"
HF_DIR="${HF_DIR:-tier-b/generation_speed_r5}"
export HF_DIR
BOX_MAX_H="${BOX_MAX_H:-3}"
GAMES="${GAMES:-400}"
GAMES_PER_JOB="${GAMES_PER_JOB:-5}"
PROFILE_BATCHES="${PROFILE_BATCHES:-40}"       # a profiled arm's batches under torch.profiler
PROFILE_AFTER="${PROFILE_AFTER:-400}"           # the batches each profiled server serves first
# No committed result carries a seed from 130000 to 130999 but the
# generation speed runs', which play the same games.
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
# ---- the arm list (begin): tests/test_generation_speed_box.py reads it
# This branch's device levers (the reference arm runs none of them).
PRIORS="--priors-only"
COMPILE="--compile-packed"
BF16RES="--bf16-residual"
PROFILE="--inference-profile-batches $PROFILE_BATCHES --inference-profile-after $PROFILE_AFTER"
# The arms of this branch: name, games in flight, then flags. Their memory
# floor is 300 MB a game in flight (the driver's default, 700, asks 112 GB
# at 160).
DEEP_120=(
    "ref_p120x6 120"
    "profile_ref 120 $PROFILE"
    "priors_p120x6 120 $PRIORS"
    "compile_p120x6 120 $COMPILE"
    "bf16res_p120x6 120 $BF16RES"
    "compile_bf16res_p120x6 120 $COMPILE $BF16RES"
)
DEEP_160=(
    "ref_p160x6 160"
    "compile_p160x6 160 $COMPILE"
    "bf16res_p160x6 160 $BF16RES"
)
ARMS="base_1"
for spec in "${DEEP_120[@]}"; do ARMS="$ARMS ${spec%% *}"; done
ARMS="$ARMS base_2"
for spec in "${DEEP_160[@]}"; do ARMS="$ARMS ${spec%% *}"; done
ARMS="$ARMS profile_best base_3 best_repeat"
# ---- the arm list (end)

# shellcheck disable=SC2317 # called through box_notes, by the library
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
box_restore speed.json tests.log || box_finish "RESTORE_FAILED (restore.log)" 1
box_pip huggingface_hub psutil pytest scipy requests || echo "pip install failed (pip.log)"

# ---- the code, the base code and the Rust wheel
box_stage_code || box_finish "CODE_STAGING_FAILED (staging.log)" 1
cd "$BOX_REPO" || box_finish "CODE_STAGING_FAILED (no $BOX_REPO)" 1
box_build_wheel || box_finish "BUILD_FAILED (build.log)" 1
fetch_stage() {                  # fetch_stage NAME STAGE DIR: another code stage unpacked into DIR, once
    local name=$1 stage=$2 dir=$3
    if [ "$(cat "$dir/.staged_from" 2>/dev/null)" != "$stage" ]; then
        rm -rf "$dir.tmp"
        box_bounded "$name-stage" 15 staging.log python "$BOX_LIB/box_stage.py" fetch "$stage" --into "$dir.tmp" \
            || box_finish "${name^^}_STAGING_FAILED (staging.log)" 1
        { printf '%s\n' "$stage" > "$dir.tmp/.staged_from" && rm -rf "$dir" \
            && mv "$dir.tmp" "$dir"; } || box_finish "${name^^}_STAGING_FAILED (unpacking)" 1
    fi
    [ -f "$dir/tools/run_elo_batch.py" ] || box_finish "${name^^}_STAGING_FAILED (no match driver in $stage)" 1
}
fetch_stage base "$BASE_STAGE" "$BASE_REPO"
box_facts > "$OUT/box.txt.tmp" 2>&1
mv -f "$OUT/box.txt.tmp" "$OUT/box.txt"
timeout 2m python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || box_finish "NO_CUDA (box.txt)" 1
box_upload

# ---- the tests of what this branch changed, on this wheel; the CUDA tests are recorded
if ! box_marked_this_stage "$BOX_STATE/TESTED"; then
    : > "$OUT/tests.log"
    box_bounded tests "$TESTS_CUT_MIN" tests.log python -m pytest tests/test_generation_identity.py \
        tests/test_serve_stage.py tests/test_server_profile.py tests/test_packed_compile.py \
        tests/test_view_cache.py tests/test_graphed_serve.py tests/test_core_masks.py \
        tests/test_core_view.py tests/test_decision_state.py tests/test_eval_workers.py \
        tests/test_eval_inference_server.py tests/test_stage_timing.py tests/test_match_memory.py \
        tests/test_server_priors.py tests/test_game_core.py tests/test_memory_model.py \
        tests/test_eval_match_failures.py tests/test_raw_player.py tests/test_raw_player_compact.py \
        tests/test_inference_seam.py tests/test_leaf_wire.py -q -p no:cacheprovider -m ""
    tail -n 3 "$OUT/tests.log"
    [ "$BOX_RC" -eq 0 ] || box_finish "TESTS_FAILED rc=$BOX_RC (tests.log)" 1
    : > "$OUT/cuda_tests.log"
    box_bounded cuda-tests 25 cuda_tests.log python -m pytest tests/test_server_priors_cuda.py \
        tests/test_packed_compile_cuda.py \
        tests/test_match_memory.py::test_the_served_memory_batch_is_the_slot_memorys_padding \
        "tests/test_serve_stage.py::test_on_cuda_the_rust_stage_answers_as_the_python_path" \
        "tests/test_serve_stage.py::test_on_cuda_the_kept_memory_plays_as_the_wire" \
        "tests/test_serve_stage.py::test_on_cuda_priors_only_serves_the_same_priors" \
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

# ---- the arms (begin): tests/test_generation_speed_box.py runs this block with the library stubbed
# Each arm fails alone: a match that crashes or loses a game is a failed arm
# (arms_failed.txt, speed.json) and the next arm runs.
# ---- speed.json: the arms so far, rewritten after every arm
summarize() {
    box_bounded summary 5 summary.log python - "$OUT" "$VALID_SPREAD" <<'EOF'
import json, pathlib, sys
out, spread = pathlib.Path(sys.argv[1]), float(sys.argv[2])


def servers_row(servers):
    """The arm's servers as one: counts and seconds summed over the
    servers, the wall the longest server's."""
    def total(k):
        return sum(float(s.get(k) or 0.0) for s in servers)
    stage = {}
    for s in servers:
        for k, v in (s.get("stage_s") or {}).items():
            stage[k] = round(stage.get(k, 0.0) + v, 3)
    batches, leaves = int(total("batches")), int(total("leaves"))
    return {"servers": len(servers), "requests": int(total("requests")), "batches": batches, "leaves": leaves,
            "mean_batch": leaves / batches if batches else 0.0,
            **{k: round(total(k), 3) for k in ("infer_s", "window_s", "recv_s", "idle_s", "reply_s", "gpu_ms")},
            "stage_s": stage, "wall_s": max(float(s.get("wall_s") or 0.0) for s in servers),
            "graphed": [s.get("graphed") for s in servers], "connections": int(total("connections")),
            "releases": int(total("releases")), "max_batch": servers[0].get("max_batch"),
            "rust_stage": [s.get("rust_stage") for s in servers],
            "server_memory": [s.get("server_memory") for s in servers],
            "memory_states": [s.get("memory_states") for s in servers],
            "failed_batches": int(total("failed_batches")),
            "failed_requests": int(total("failed_requests")), "fatal": [s.get("fatal") for s in servers if s.get("fatal")],
            "graphed_serve": [s.get("graphed_serve") for s in servers],
            "priors_only": [s.get("priors_only") for s in servers],
            "compile_packed": [s.get("compile_packed") for s in servers],
            "bf16_residual": [s.get("bf16_residual") for s in servers],
            "profile": [s.get("profile") for s in servers if s.get("profile")]}


arms = {}
for p in sorted(out.glob("arm_*.json")):
    if p.name.count(".") > 1:
        continue
    a = json.loads(p.read_text(encoding="utf-8"))
    name = a["arm"]
    servers = [json.loads(s.read_text(encoding="utf-8")) for s in sorted(out.glob(f"arm_{name}.inference_server_*.json"))]
    row = {"jobs": a["jobs"], "flags": a["flags"], "repo": a["repo"], "rc": a["rc"], "slots": a.get("slots"),
           "games": a["games"], "failed_games": a.get("failed_games", 0), "wall_s": a["wall_s"],
           "games_per_hour": round(3600.0 * a["games"] / a["wall_s"], 1) if a["wall_s"] else None}
    if servers:
        s = servers_row(servers)
        row["server"] = s
        row["decisions_per_s"] = round(s["requests"] / s["wall_s"], 2) if s["wall_s"] else None
        # A rounding change plays other games: their length shows here.
        row["decisions_per_game"] = round(s["requests"] / a["games"], 1) if a["games"] else None
        row["infer_ms_per_batch"] = round(1e3 * s["infer_s"] / s["batches"], 3) if s["batches"] else None
        row["device_ms_per_batch"] = round(s["gpu_ms"] / s["batches"], 3) if s["batches"] else None
        row["host_ms_per_batch"] = {k: round(1e3 * v / s["batches"], 3) for k, v in s["stage_s"].items()} \
            if s["batches"] else {}
    memory_refused = sum((m or {}).get("refused", 0) for m in (row.get("server") or {}).get("memory_states", []))
    row["failed"] = bool(row["failed_games"] or a["rc"] not in (0, 3, 4) or memory_refused
                         or (servers and (row["server"]["failed_requests"] or row["server"]["fatal"])))
    stages = out / f"arm_{name}.worker_stages.json"
    if stages.exists():
        w = json.loads(stages.read_text(encoding="utf-8"))
        row["worker"] = {k: w[k] for k in ("workers", "decisions", "decision_wall_ms", "decision_cpu_ms",
                                           "own_wall_ms", "stage_ms_per_decision") if k in w}
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
    # The arms against each other hold when the base runs, before, between
    # and after the others, agree on the server's decisions per second.
    "valid": len(bases) >= 2 and all(abs(rate(n) / mean - 1.0) <= spread for n in bases),
    "base_spread": {n: round(rate(n) / mean, 4) for n in bases} if mean else {},
    "best": best,
    "best_vs_repeat": ratio("best_repeat", best) if best else None,
    "failed_arms": sorted(n for n, r in arms.items() if r["failed"]),
    "against_base": {n: {k: round(r[k] / base_rate[k], 4) for k in base_rate if r.get(k)}
                     for n, r in arms.items()},
    "against_ref": {n: {k: ratio(n, "ref_p160x6" if r["jobs"] == 160 else "ref_p120x6", k)
                        for k in ("decisions_per_s", "games_per_hour", "infer_ms_per_batch",
                                  "device_ms_per_batch", "decisions_per_game")}
                    for n, r in arms.items() if not n.startswith("base")},
    "profiles": {n: r["server"]["profile"] for n, r in arms.items() if (r.get("server") or {}).get("profile")},
}
tmp = out / "speed.json.tmp"
tmp.write_text(json.dumps(summary, indent=1), encoding="utf-8")
tmp.replace(out / "speed.json")
print(json.dumps({"valid": summary["valid"], "base_spread": summary["base_spread"], "best": best,
                  "best_vs_repeat": summary["best_vs_repeat"], "failed_arms": summary["failed_arms"],
                  "decisions_per_s": {n: r.get("decisions_per_s") for n, r in arms.items()}}), flush=True)
EOF
}

# ---- the arms
arm() {                          # arm NAME REPO JOBS FLAGS...: one match, once; false when it failed or lost a game
    local name=$1 repo=$2 jobs=$3 dir="$WORKDIR/games_$1" t0 f games failed rc why
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
    rc=$BOX_RC why=$BOX_WHY              # the summary's own step resets BOX_RC
    cd "$BOX_REPO" || box_finish "NO_REPO $BOX_REPO" 1
    failed=$(find "$dir" -maxdepth 1 -name 'failed_*.json' 2>/dev/null | wc -l)
    printf '{"arm": "%s", "repo": "%s", "jobs": %d, "flags": "%s", "slots": %d, "wall_s": %d, "games": %d, "failed_games": %d, "rc": %d, "why": "%s"}\n' \
        "$name" "$repo" "$jobs" "$*" "$games" "$(( $(box_now) - t0 ))" \
        "$(find "$dir" -maxdepth 1 -name 'game_*.json' 2>/dev/null | wc -l)" "$failed" "$rc" "$why" \
        > "$OUT/arm_$name.json"
    for f in "$dir"/.inference_server_*.json "$dir"/.worker_stages.json; do
        [ -f "$f" ] && cp -f "$f" "$OUT/arm_${name}.${f##*/.}"
    done
    box_mark "$OUT/arm_$name.done"
    summarize
    box_upload
    # run_elo_batch exits 3 or 4 when its games end without enough decisive
    # results (no replacements here): the timing is whole. A failed game, or
    # any other exit, fails the arm.
    case $rc in
        0|3|4) ;;
        *) echo "ARM_FAILED $name rc=$rc $why (arm_$name.log)" | tee -a "$OUT/arms_failed.txt"
           return 1 ;;
    esac
    if [ "$failed" -gt 0 ]; then
        echo "ARM_FAILED $name: $failed failed games (arm_$name.log, games_$name/failed_*.json)" \
            | tee -a "$OUT/arms_failed.txt"
        return 1
    fi
}
fastest() {                      # fastest: "NAME JOBS FLAGS" of this branch's fastest timing arm with no failure
    timeout 1m python - "$OUT" <<'EOF'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
best = None
for p in sorted(out.glob("arm_*.json")):
    if p.name.count(".") > 1:
        continue
    a = json.loads(p.read_text(encoding="utf-8"))
    name = a["arm"]
    servers = [json.loads(s.read_text(encoding="utf-8")) for s in sorted(out.glob(f"arm_{name}.inference_server_*.json"))]
    if (name.startswith(("base", "profile")) or name == "best_repeat"
            or a["failed_games"] or a["rc"] not in (0, 3, 4) or not servers
            or any(s.get("failed_requests") or s.get("fatal") for s in servers)):
        continue
    wall = max(s.get("wall_s") or 0.0 for s in servers)
    rate = sum(s["requests"] for s in servers) / wall if wall else 0.0
    if best is None or rate > best[0]:
        best = (rate, a)
if best is not None:
    print(best[1]["arm"], best[1]["jobs"], best[1]["flags"])
EOF
}
summarize
arm base_1 "$BASE_REPO" 20 || echo "the first base arm failed: the drift is not bracketed" >> "$OUT/arms_failed.txt"
for spec in "${DEEP_120[@]}"; do
    read -r name jobs flags <<< "$spec"
    # shellcheck disable=SC2086 # flags is a list of flags
    arm "$name" "$BOX_REPO" "$jobs" --worker-games 6 --min-free-mb 300 $flags || true
done
arm base_2 "$BASE_REPO" 20 || echo "the second base arm failed: the drift is not bracketed" >> "$OUT/arms_failed.txt"
for spec in "${DEEP_160[@]}"; do
    read -r name jobs flags <<< "$spec"
    # shellcheck disable=SC2086 # flags is a list of flags
    arm "$name" "$BOX_REPO" "$jobs" --worker-games 6 --min-free-mb 300 $flags || true
done
# The fastest timing arm, profiled, then (after the last base run) again.
BEST=$(fastest)
if [ -n "$BEST" ]; then
    read -r BEST_ARM BEST_JOBS BEST_FLAGS <<< "$BEST"
    # shellcheck disable=SC2086 # BEST_FLAGS is a list of flags
    arm profile_best "$BOX_REPO" "$BEST_JOBS" $BEST_FLAGS $PROFILE || true
else
    echo "arm profile_best left out: no arm of this branch passed" | tee -a "$OUT/arms_failed.txt"
fi
arm base_3 "$BASE_REPO" 20 || echo "the last base arm failed: the drift is not bracketed" >> "$OUT/arms_failed.txt"
if [ -n "$BEST" ]; then
    echo "$BEST_ARM" > "$OUT/best.txt"
    # shellcheck disable=SC2086 # BEST_FLAGS is a list of flags
    arm best_repeat "$BOX_REPO" "$BEST_JOBS" $BEST_FLAGS \
        || echo "the best arm's repeat failed: its drift is not bracketed" >> "$OUT/arms_failed.txt"
else
    echo "no arm of this branch to repeat" >> "$OUT/arms_failed.txt"
fi

summarize
box_upload
SUMMARY=$(tail -n 1 "$OUT/summary.log" | cut -c1-600)
FAILED_ARMS=$(awk '$1 == "ARM_FAILED" {print $2}' "$OUT/arms_failed.txt" 2>/dev/null | tr -d ':' | sort -u | tr '\n' ' ')
# ---- the arms (end)
box_finish "GENERATION_SPEED_DONE $SUMMARY; cuda tests: ${CUDA_TESTS:-none}; failed: ${FAILED_ARMS:-none}"
