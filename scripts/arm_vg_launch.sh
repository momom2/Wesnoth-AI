#!/bin/bash
# Arm VG launch (2026-09-01): value grounding on consulted states.
# Recipe = arm T (TCS teacher, mover frame, project reval, policy
# anchor, K-tripwire) MINUS value-memory PLUS --value-ground, on the
# code with the aux/moves-left detach and per-decade telemetry.
# Hand-driven like the teacher arms; run on the box from
# /workspace/wai after setup (see docs/arm_vg_leg_20260901.md).
#
# Stages: full test suite -> anchor build -> fork-guard smoke ->
# supervised training loop + pin/probe/escrow/watchdog daemons.
set -u
cd /workspace/wai
PY=python
WORKDIR=/workspace
# ARM_TAG names the campaign file and the HF escrow folder
# (vg = 2026-09-01 arm, vg2 = principled-mixture arm, ...).
ARM_TAG="${ARM_TAG:-vg}"
CAMPAIGN_FILE="tier_b_${ARM_TAG}.pt"
CAMPAIGN="training/checkpoints/${CAMPAIGN_FILE}"
HF_PREFIX="${HF_PREFIX:-tier-b/arm_${ARM_TAG}_$(date -u +%Y%m%d)/}"
SEED_CKPT=training/checkpoints/seed_imit_tierb_start.pt
# Actor pool sized from the cgroup CPU quota (nproc is HOST-wide on
# Vast; same derivation as vast_onstart.sh): quota - 4, min 8.
_CORES=$("$PY" - <<'PYEOF'
import os
def cores():
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()
        if q != "max":
            return max(1, int(int(q) / int(p)))
    except OSError:
        pass
    return os.cpu_count() or 8
print(cores())
PYEOF
)
ACTOR_POOL="${ACTOR_POOL:-$(( _CORES - 4 ))}"
[ "$ACTOR_POOL" -lt 8 ] && ACTOR_POOL=8
echo "[armVG] actor pool: $ACTOR_POOL (quota $_CORES cores)"
# Signal profiling is OPT-IN (user ruling 2026-09-02): the per-
# iteration gradient-norm telemetry (SIGNAL_TELEMETRY=1) and the
# per-pin deep profile (PROFILE_PINS=1) both default OFF so they
# never accumulate overhead unnoticed in a future run.
SIG_FLAG=""
[ "${SIGNAL_TELEMETRY:-0}" = 1 ] && SIG_FLAG="--signal-telemetry"
PROFILE_PINS="${PROFILE_PINS:-0}"
echo "[armVG] signal telemetry: ${SIGNAL_TELEMETRY:-0}; per-pin profile: $PROFILE_PINS"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65536 2>/dev/null || true

# Terminal-failure handler (user ruling 2026-09-02): final escrow,
# then STOP the box so a tripwire saves credit as well as the model.
# Needs $WORKDIR/.vast_api_key + .instance_id (written at provision).
fatal_stop() {
    echo "[armVG] terminal failure ($1) -- escrow + stop box"
    WORKDIR="$WORKDIR" REPO_ROOT=/workspace/wai \
        CAMPAIGN_FILE="$CAMPAIGN_FILE" HF_PREFIX="$HF_PREFIX" \
        "$PY" scripts/box_stop_on_abort.py >> "$WORKDIR/train.log" 2>&1
}

stage="${1:-all}"

if [ "$stage" = "tests" ] || [ "$stage" = "all" ]; then
    echo "[armVG] FULL test suite (slow tier included)..."
    "$PY" -m pytest -m "" -q > "$WORKDIR/pytest_full.log" 2>&1
    rc=$?
    tail -3 "$WORKDIR/pytest_full.log"
    if [ $rc -ne 0 ]; then
        echo "[armVG] FATAL: test suite failed (rc=$rc)"
        touch "$WORKDIR/ABORTED_tests"
        fatal_stop tests
        exit 1
    fi
fi

if [ "$stage" = "rust" ] || [ "$stage" = "all" ]; then
    # Rust reach/enumeration kernels are the DEFAULT path (user
    # ruling 2026-09-02); a box must build the wheel or fail loudly
    # rather than train 4.5x slower on the Python fallback.
    if ! "$PY" -c "import wesnoth_core" 2>/dev/null; then
        echo "[armVG] building wesnoth_core (rustup minimal + maturin)..."
        # Rust build scripts need a C linker; the pytorch runtime
        # image ships none (2026-09-02: "linker cc not found").
        if ! command -v cc >/dev/null 2>&1; then
            (apt-get update -qq && apt-get install -y -qq gcc) \
                > "$WORKDIR/apt_gcc.log" 2>&1 || true
        fi
        if ! command -v cargo >/dev/null 2>&1; then
            curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
                > "$WORKDIR/rustup.log" 2>&1
        fi
        export PATH="$HOME/.cargo/bin:$PATH"
        "$PY" -m pip install -q maturin 2>&1 | grep -v WARNING | tail -1
        "$PY" -m pip install -q rust/wesnoth_core > "$WORKDIR/rust_build.log" 2>&1
        if ! "$PY" -c "import wesnoth_core" 2>/dev/null; then
            echo "[armVG] FATAL: wesnoth_core wheel failed (see rust_build.log)"
            touch "$WORKDIR/ABORTED_rust"; fatal_stop rust; exit 1
        fi
    fi
    echo "[armVG] wesnoth_core: $("$PY" -c "import wesnoth_core, sys; print('ok', getattr(wesnoth_core, '__file__', ''))")"
fi

if [ "$stage" = "anchor" ] || [ "$stage" = "all" ]; then
    if [ ! -f replays_dataset_imitation/policy_anchor.npz ]; then
        echo "[armVG] building policy anchor (500 games)..."
        "$PY" tools/policy_anchor.py \
            --dataset-dir replays_dataset_imitation \
            --out replays_dataset_imitation/policy_anchor.npz \
            --games 500 --seed 20260901 --log-level INFO \
            > "$WORKDIR/anchor_build.log" 2>&1 || {
            echo "[armVG] FATAL: anchor build failed"
            touch "$WORKDIR/ABORTED_anchor"; fatal_stop anchor; exit 1; }
    fi
fi

if [ "$stage" = "smoke" ] || [ "$stage" = "all" ]; then
    echo "[armVG] fork-guard smoke (guards armed, VG path on)..."
    SIM_FORK_GUARD=1 "$PY" tools/sim_self_play.py \
        --mcts --mcts-sims 8 --device cpu \
        --d-model 384 --num-layers 8 --num-heads 12 --d-ff 1536 \
        --iterations 1 --games-per-iter 1 --max-turns 8 \
        --ladder-ratio 1.0 --midgame-ratio 0 --mini-ratio 0 \
        --fogless-ratio 0 \
        --game-log-dir "" --validate-export-every 0 \
        --trainer-history-csv "$WORKDIR/smoke_history.csv" \
        --turn-boundary-frame mover --turn-project reval \
        --value-ground \
        --checkpoint-in "$SEED_CKPT" \
        --checkpoint-out "$WORKDIR/fork_guard_smoke.pt" \
        --save-every 1000 --log-level INFO \
        > "$WORKDIR/smoke.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[armVG] FATAL: smoke rc=$rc (see smoke.log)"
        touch "$WORKDIR/ABORTED_smoke"
        fatal_stop smoke
        exit 1
    fi
    rm -f "$WORKDIR"/fork_guard_smoke.pt*
    echo "[armVG] smoke PASSED"
fi

case "$stage" in
    tests|anchor|smoke) exit 0 ;;
esac

echo "[armVG] launching daemons + training..."
mkdir -p "$WORKDIR/pins" "$WORKDIR/probes" "$WORKDIR/profiles"
# The leg's CSV must start clean: the slow-tier e2e tests and any
# earlier smoke write 1-game rows into the same repo path
# (2026-09-02: three such rows masqueraded as "iteration 0").
_CSV=training/logs/trainer_history_local.csv
if [ -s "$_CSV" ] && [ ! -f "$CAMPAIGN" ]; then
    mv "$_CSV" "training/logs/trainer_history_pre_leg_$(date -u +%H%M%S).csv"
    echo "[armVG] rotated pre-leg CSV rows aside"
fi

# Escrow: rolling checkpoint + CSV every 30 min; probes/profiles/
# logs on a second loop.
CAMPAIGN_FILE="$CAMPAIGN_FILE" HF_PREFIX="$HF_PREFIX" \
    WORKDIR="$WORKDIR" setsid nohup "$PY" scripts/hf_upload_loop.py \
    > "$WORKDIR/upload.log" 2>&1 < /dev/null &
HF_PREFIX="$HF_PREFIX" WORKDIR="$WORKDIR" \
    setsid nohup "$PY" scripts/probe_escrow_loop.py \
    > "$WORKDIR/probe_escrow.log" 2>&1 < /dev/null &

# Stall watchdog.
WORKDIR="$WORKDIR" setsid nohup "$PY" scripts/stall_watchdog.py \
    > "$WORKDIR/watchdog.log" 2>&1 < /dev/null &

# Pin + probe loop: snapshot the rolling file on decision_step
# advance; 24-game probe vs the seed, both sides MCTS-32 raw frame
# (--no-turn-search is implied by run_elo_batch's game runner).
setsid nohup bash -c '
cd /workspace/wai
last=0
while true; do
    sleep 600
    [ -f '"$CAMPAIGN"' ] || continue
    step=$('"$PY"' -c "
import torch
try:
    print(int(torch.load(\"'"$CAMPAIGN"'\", map_location=\"cpu\",
                          weights_only=False).get(\"decision_step\", 0)))
except Exception:
    print(0)")
    if [ "$step" -gt 0 ] && [ $((step - last)) -ge 27000 ]; then
        last=$step
        pin=/workspace/pins/pin_$step.pt
        cp '"$CAMPAIGN"' "$pin"
        echo "$(date -u +%FT%TZ) pin $step" >> /workspace/pins.log
        '"$PY"' tools/run_elo_batch.py \
            --label-a "pin_$step" --spec-a "$pin" \
            --label-b seed --spec-b '"$SEED_CKPT"' \
            --games 24 --mcts-sims 32 --no-turn-search \
            --device cuda \
            --outdir /workspace/probes/pin_$step \
            --time-budget-min 120 --min-free-mb 500 \
            >> /workspace/probes/probe.log 2>&1
        '"$PY"' tools/elo_collect.py /workspace/probes/pin_$step \
            --no-catalog >> /workspace/pins.log 2>&1 || true
        # Deep signal profile per pin (user ruling 2026-09-02: keep
        # checking the mixture parameters): provenance-split
        # gradient tree + post-Adam update tree + held-out
        # consultation probe, 4 games (~20 min alongside training).
        # OPT-IN via PROFILE_PINS=1.
        [ "'"$PROFILE_PINS"'" = 1 ] && \
        '"$PY"' signal_profiler/run_profile_v2.py --vg \
            --checkpoint "$pin" --games '"${PROFILE_GAMES:-4}"' \
            --consult-cap 200 --seed 31337 --device cuda \
            --out /workspace/profiles/pin_$step.json \
            > /workspace/profiles/pin_$step.out 2>&1 || true
        '"$PY"' - "$step" <<PYEOF >> /workspace/pins.log 2>&1 || true
import json, sys
d = json.load(open(f"/workspace/profiles/pin_{sys.argv[1]}.json"))
t = d["terms"]; u = d["update_tree"]["terms"]
row = " ".join(f"{k}={t[k]['norm']:.2f}/{t[k]['proj_frac']:+.2f}"
               for k in ("value_game", "value_ground", "value_consist",
                         "policy_distill") if k in t)
dv = u["value_consist"]["value_movement"]["consult"].get("mean_abs", float("nan")) if "value_consist" in u else float("nan")
print(f"  profile pin_{sys.argv[1]}: {row} | consist step dv_consult={dv:.4f} linres={d['linearity_residual_frac']:.3f}")
PYEOF
    fi
done' > "$WORKDIR/pinloop.log" 2>&1 < /dev/null &

# Training, supervised (10 tries).
tries=0
while [ $tries -lt 10 ]; do
    CKPT_IN=$([ -f "$CAMPAIGN" ] && echo "$CAMPAIGN" || echo "$SEED_CKPT")
    "$PY" tools/sim_self_play.py --device cuda \
        --mcts --mcts-sims 32 \
        --d-model 384 --num-layers 8 --num-heads 12 --d-ff 1536 \
        --replay-buffer --replay-updates 16 --value-coef 1.0 \
        --replay-minibatch 128 --replay-capacity 24000 \
        --train-batch-size 32 --mcts-batch-size 16 \
        --mini-ratio 0 --midgame-ratio 0.2 --fogless-ratio 0.2 \
        --ladder-ratio 0.6 \
        --max-turns-min 60 \
        --mcts-aux-score \
        --validate-export-every 1 \
        --value-label-smoothing 0.02 \
        --holdout-size 512 --holdout-per-game-cap 64 \
        --human-anchor-policy-file \
            replays_dataset_imitation/policy_anchor.npz \
        --abort-decisive-rate 0.35 --abort-window 20 \
        --abort-holdout-stall 60 \
        --abort-k-median 10 \
        --turn-boundary-frame mover \
        --turn-project reval \
        --value-ground \
        $SIG_FLAG \
        --actor-pool "$ACTOR_POOL" --actor-max-batch 16 \
        --games-per-iter 24 \
        --checkpoint-in "$CKPT_IN" \
        --checkpoint-out "$CAMPAIGN" \
        --iterations 100000 --save-every 1 --log-level INFO \
        >> "$WORKDIR/train.log" 2>&1
    rc=$?
    echo "[armVG] training exited rc=$rc at $(date -u +%FT%TZ)" \
        >> "$WORKDIR/train.log"
    [ $rc -eq 0 ] && break
    if [ $rc -ge 128 ]; then
        if [ -f "$WORKDIR/WATCHDOG_STALL" ]; then
            rm -f "$WORKDIR/WATCHDOG_STALL"
            echo "[armVG] watchdog kill; relaunching" \
                >> "$WORKDIR/train.log"
        else
            echo "[armVG] signal exit; standing down" \
                >> "$WORKDIR/train.log"
            break
        fi
    fi
    if [ $rc -ge 3 ] && [ $rc -le 9 ]; then
        touch "$WORKDIR/ABORTED_$rc"
        fatal_stop "tripwire rc=$rc"
        break
    fi
    tries=$((tries + 1))
    sleep 60
done
if [ $tries -ge 10 ]; then
    touch "$WORKDIR/ABORTED_relaunch_cap"
    fatal_stop "relaunch cap"
fi
