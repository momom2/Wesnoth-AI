#!/bin/bash
# Minimal self-play loop launch (docs/az_minimal_spec.md), 2026-09-03.
# Run ON the box from /workspace/wai after box_provision_arm.sh.
# Stages: full test suite -> wesnoth_core wheel -> smoke (one tiny
# iteration through the real loop) -> daemons + supervised loop.
set -u
cd /workspace/wai
PY=python
WORKDIR=/workspace
ARM_TAG="${ARM_TAG:-az}"
CAMPAIGN_FILE="tier_b_${ARM_TAG}.pt"
CAMPAIGN="training/checkpoints/${CAMPAIGN_FILE}"
HF_PREFIX="${HF_PREFIX:-tier-b/arm_${ARM_TAG}_$(date -u +%Y%m%d)/}"
SEED_CKPT="${SEED_CKPT:-training/checkpoints/seed_imit_tierb_start.pt}"
_CORES=$("$PY" - <<'PYEOF'
import os
try:
    q, p = open("/sys/fs/cgroup/cpu.max").read().split()
    print(max(1, int(int(q) / int(p))) if q != "max" else (os.cpu_count() or 8))
except OSError:
    print(os.cpu_count() or 8)
PYEOF
)
ACTORS="${ACTORS:-$(( _CORES - 4 ))}"
[ "$ACTORS" -lt 4 ] && ACTORS=4
echo "[az] actors: $ACTORS (quota $_CORES cores)"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65536 2>/dev/null || true

fatal_stop() {
    echo "[az] terminal failure ($1) -- escrow + stop box"
    WORKDIR="$WORKDIR" REPO_ROOT=/workspace/wai \
        CAMPAIGN_FILE="$CAMPAIGN_FILE" HF_PREFIX="$HF_PREFIX" \
        "$PY" scripts/box_stop_on_abort.py >> "$WORKDIR/train.log" 2>&1
}

stage="${1:-all}"

if [ "${LAUNCH_SKIP_TESTS:-0}" = "1" ] && [ "$stage" = "all" ]; then
    echo "[az] test suite SKIPPED (LAUNCH_SKIP_TESTS=1: resume of an already-tested leg)"
elif [ "$stage" = "tests" ] || [ "$stage" = "all" ]; then
    echo "[az] FULL test suite..."
    "$PY" -m pytest -m "" -q > "$WORKDIR/pytest_full.log" 2>&1
    rc=$?; tail -1 "$WORKDIR/pytest_full.log"
    if [ $rc -ne 0 ]; then
        touch "$WORKDIR/ABORTED_tests"; fatal_stop tests; exit 1
    fi
fi

if [ "$stage" = "rust" ] || [ "$stage" = "all" ]; then
    if ! "$PY" -c "import wesnoth_core" 2>/dev/null; then
        echo "[az] building wesnoth_core..."
        command -v cc >/dev/null 2>&1 || \
            (apt-get update -qq && apt-get install -y -qq gcc) > "$WORKDIR/apt_gcc.log" 2>&1 || true
        command -v cargo >/dev/null 2>&1 || \
            curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal > "$WORKDIR/rustup.log" 2>&1
        export PATH="$HOME/.cargo/bin:$PATH"
        "$PY" -m pip install -q maturin 2>&1 | grep -v WARNING | tail -1
        "$PY" -m pip install -q rust/wesnoth_core > "$WORKDIR/rust_build.log" 2>&1
        if ! "$PY" -c "import wesnoth_core" 2>/dev/null; then
            touch "$WORKDIR/ABORTED_rust"; fatal_stop rust; exit 1
        fi
    fi
    echo "[az] wesnoth_core ok"
fi

if [ "$stage" = "smoke" ] || [ "$stage" = "all" ]; then
    echo "[az] smoke: one tiny iteration through the real loop (cpu)..."
    rm -rf "$WORKDIR/smoke_az"
    "$PY" tools/az_loop.py --seed-checkpoint "$SEED_CKPT" \
        --campaign "$WORKDIR/smoke_az/campaign.pt" \
        --workdir "$WORKDIR/smoke_az" --iterations 1 --games-per-iter 2 \
        --actors 2 --sims 4 --max-turns 8 --pin-every 1000 \
        --device cpu --log-level INFO > "$WORKDIR/smoke.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[az] FATAL: smoke rc=$rc (smoke.log)"
        touch "$WORKDIR/ABORTED_smoke"; fatal_stop smoke; exit 1
    fi
    echo "[az] smoke PASSED: $(tail -1 "$WORKDIR/smoke_az/az_history.csv" | cut -c1-80)"
fi

case "$stage" in tests|rust|smoke) exit 0 ;; esac

echo "[az] launching daemons + loop..."
mkdir -p "$WORKDIR/pins" "$WORKDIR/probes" "$WORKDIR/profiles"
CAMPAIGN_FILE="$CAMPAIGN_FILE" HF_PREFIX="$HF_PREFIX" WORKDIR="$WORKDIR" \
    setsid nohup "$PY" scripts/hf_upload_loop.py > "$WORKDIR/upload.log" 2>&1 < /dev/null &
HF_PREFIX="$HF_PREFIX" WORKDIR="$WORKDIR" \
    setsid nohup "$PY" scripts/probe_escrow_loop.py > "$WORKDIR/probe_escrow.log" 2>&1 < /dev/null &
WORKDIR="$WORKDIR" setsid nohup "$PY" scripts/stall_watchdog.py > "$WORKDIR/watchdog.log" 2>&1 < /dev/null &

tries=0
while [ $tries -lt 10 ]; do
    "$PY" tools/az_loop.py --seed-checkpoint "$SEED_CKPT" \
        --campaign "$CAMPAIGN" --workdir "$WORKDIR" \
        --iterations "${ITERATIONS:-60}" --games-per-iter "${GAMES_PER_ITER:-24}" \
        --actors "$ACTORS" --sims "${SIMS:-32}" --value-coef "${VALUE_COEF:-1.0}" \
        --lr "${LR:-1e-4}" --max-turns "${MAX_TURNS:-60}" \
        --pin-every "${PIN_EVERY:-10}" --probe-games "${PROBE_GAMES:-40}" \
        --start-iter "${START_ITER:-0}" \
        --device cuda --log-level INFO >> "$WORKDIR/train.log" 2>&1
    rc=$?
    echo "[az] loop exited rc=$rc at $(date -u +%FT%TZ)" >> "$WORKDIR/train.log"
    [ $rc -eq 0 ] && { echo "[az] budget complete"; fatal_stop "budget complete"; break; }
    if [ $rc -ge 128 ]; then
        if [ -f "$WORKDIR/WATCHDOG_STALL" ]; then rm -f "$WORKDIR/WATCHDOG_STALL"; else break; fi
    fi
    if [ $rc -ge 3 ] && [ $rc -le 9 ]; then
        touch "$WORKDIR/ABORTED_$rc"; fatal_stop "tripwire rc=$rc"; break
    fi
    tries=$((tries + 1)); sleep 60
done
if [ $tries -ge 10 ]; then touch "$WORKDIR/ABORTED_relaunch_cap"; fatal_stop "relaunch cap"; fi
