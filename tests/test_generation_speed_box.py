"""scripts/generation_speed_box.sh's arms, run by bash with the box library
stubbed (no box, GPU or HF): every arm runs in its order, an arm whose
match crashes or loses a game fails alone and the next arm runs, the
graphed arm takes the fastest passing arm's flags (and is left out when the
graphed check fails), the repeat runs the fastest passing arm, and
speed.json holds every arm after each one."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from helpers.posix_bash import bash_path, find_bash

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "generation_speed_box.sh"
BASH = find_bash()
pytestmark = [pytest.mark.slow, pytest.mark.skipif(BASH is None, reason="no POSIX bash")]

# The library's functions the block calls, and box_bounded answering for the
# steps: STUB_<step> = crash (exit 1, no record) or lose (a failed game);
# RATE_<step>: the decisions a second its server records (default 1000);
# STUB_graphed_check = fail.
STUBS = r"""
box_marked_this_stage() { [ -f "$1" ]; }
box_mark() { mkdir -p "$(dirname "$1")" && echo "stage=$STAGE" > "$1"; }
box_upload() { :; }
box_gpu_ok() { :; }
box_now() { date +%s; }
box_stamp() { date; }
box_finish() { printf '%s\n' "$1" > "$OUT/FINISH"; exit "${2:-0}"; }
notes() { echo notes; }
arg_of() {                       # arg_of FLAG ARGS...: the value after the first FLAG
    local flag=$1
    shift
    while [ "$#" -gt 0 ]; do
        [ "$1" = "$flag" ] && { echo "$2"; return; }
        shift
    done
}
box_bounded() {                  # box_bounded NAME CUT_MIN LOG CMD...
    local name=$1 log=$3 key rate dir out
    shift 3
    echo "$name $*" >> "$OUT/steps.txt"
    key=STUB_${name//[ -]/_}
    rate=RATE_${name//[ -]/_}
    BOX_RC=0 BOX_WHY=ok
    case $name in
        arm\ *)
            dir=$(arg_of --outdir "$@")
            mkdir -p "$dir"
            if [ "${!key:-}" != crash ]; then
                echo '{}' > "$dir/game_A_B_s1_1.json"
                echo '{}' > "$dir/game_A_B_s2_2.json"
                printf '{"requests": %d, "wall_s": 10.0, "batches": 100, "leaves": %d, "infer_s": 1.0, "gpu_ms": 500.0, "stage_s": {"t_parse": 0.1}, "max_batch": 120}\n' \
                    "$(( ${!rate:-1000} * 10 ))" "$(( ${!rate:-1000} * 10 ))" > "$dir/.inference_server_0.json"
                [ "${!key:-}" != lose ] || echo '{}' > "$dir/failed_A_B_s1_3.json"
            fi
            [ "${!key:-}" != crash ] || { BOX_RC=1 BOX_WHY=failed; } ;;
        graphed-check)
            out=$(arg_of --out "$@")
            if [ "${!key:-}" = fail ]; then echo '{"pass": false}' > "$out"; else echo '{"pass": true}' > "$out"; fi ;;
        summary) "$@" >> "$OUT/$log" 2>&1 || BOX_RC=$? ;;
    esac
    return "$BOX_RC"
}
"""


def _region(begin: str, end: str) -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    return text[text.index(begin):text.index(end)]


def _run(tmp_path: Path, **stubs) -> dict:
    out, work = tmp_path / "out", tmp_path / "work"
    out.mkdir()
    work.mkdir()
    for d in ("repo", "base", "ref"):
        (tmp_path / d).mkdir()
    env = "\n".join(f"{k}={v}" for k, v in stubs.items())
    script = f"""set -uo pipefail
export PATH="{bash_path(Path(sys.executable).parent)}:/usr/bin:$PATH"
OUT="{bash_path(out)}" WORKDIR="{bash_path(work)}" BOX_REPO="{bash_path(tmp_path / 'repo')}"
BASE_REPO="{bash_path(tmp_path / 'base')}" REF_REPO="{bash_path(tmp_path / 'ref')}" STAGE=stage_test
GAMES=400 GAMES_PER_JOB=5 SEED_BASE=130000 ARM_CUT_MIN=30 MATCH_BUDGET_MIN=22 VALID_SPREAD=0.05
GRAPHED_MAX_BATCH=160 REF_PT=own1.pt CUDA_TESTS="2 passed"
SIDE_A=(--label-a own1a) SIDE_B=(--label-b own1b --memory-b 64)
BOX_RC=0 BOX_WHY=ok
{env}
{STUBS}
{_region("# ---- the arm list (begin)", "# ---- the arm list (end)")}
{_region("# ---- the arms (begin)", "# ---- the arms (end)")}
box_finish "GENERATION_SPEED_DONE"
"""
    path = tmp_path / "driver.sh"
    path.write_bytes(script.encode())
    done = subprocess.run([BASH, bash_path(path)], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    steps = (out / "steps.txt").read_text().splitlines() if (out / "steps.txt").exists() else []
    speed = json.loads((out / "speed.json").read_text()) if (out / "speed.json").exists() else {}
    return {"rc": done.returncode, "stderr": done.stderr, "out": out,
            "finish": (out / "FINISH").read_text() if (out / "FINISH").exists() else "",
            "arms": [s.split()[1] for s in steps if s.startswith("arm ")],
            "steps": steps, "speed": speed,
            "failed": (out / "arms_failed.txt").read_text() if (out / "arms_failed.txt").exists() else ""}


ORDER = ["base_1", "ref_p120x6", "off_p120x6", "rust_p120x6", "memory_p120x6", "view_p120x6", "all_p120x6",
         "base_2", "rust_p160x6", "memory_p160x6", "view_p160x6", "all_p160x6", "graphed_best", "base_3",
         "best_repeat"]


def _arm_line(got, name):
    return next(s for s in got["steps"] if s.startswith(f"arm {name} "))


def test_every_arm_runs_and_the_graphed_and_repeat_arms_take_the_fastest(tmp_path):
    got = _run(tmp_path, RATE_arm_all_p160x6=3000, RATE_arm_graphed_best=3500)
    assert got["rc"] == 0 and "GENERATION_SPEED_DONE" in got["finish"], (got["finish"], got["stderr"][-2000:])
    assert got["arms"] == ORDER and not got["failed"]
    graphed = _arm_line(got, "graphed_best")
    assert "--graphed-serve" in graphed and "--jobs 160" in graphed
    assert "--rust-stage --server-memory --view-refresh incremental" in graphed
    repeat = _arm_line(got, "best_repeat")
    assert "--graphed-serve" in repeat and "--jobs 160" in repeat
    speed = got["speed"]
    assert sorted(speed["arms"]) == sorted(ORDER) and speed["valid"]
    assert speed["best"] == "graphed_best" and speed["against_ref"]["graphed_best"]["decisions_per_s"] == 3.5
    assert speed["arms"]["ref_p120x6"]["repo"].endswith("/ref") and speed["arms"]["base_2"]["repo"].endswith("/base")
    ref = _arm_line(got, "ref_p120x6")
    assert "--worker-games 6" in ref and "--rust-stage" not in ref


def test_failing_arms_fail_alone(tmp_path):
    """A base arm that crashes, the fastest arm crashing, an arm that loses
    a game: each is a failed arm, every other arm runs, and the graphed and
    repeat arms take the fastest arm that passed."""
    got = _run(tmp_path, STUB_arm_base_1="crash", STUB_arm_all_p160x6="crash", RATE_arm_all_p160x6=9000,
               STUB_arm_memory_p120x6="lose", RATE_arm_memory_p120x6=8000, RATE_arm_rust_p160x6=2000)
    assert got["rc"] == 0 and "GENERATION_SPEED_DONE" in got["finish"], (got["finish"], got["stderr"][-2000:])
    assert got["arms"] == ORDER
    assert "ARM_FAILED all_p160x6" in got["failed"] and "ARM_FAILED memory_p120x6" in got["failed"]
    assert "ARM_FAILED base_1" in got["failed"]
    graphed = _arm_line(got, "graphed_best")
    assert "--jobs 160" in graphed and "--rust-stage --no-server-memory --view-refresh full" in graphed
    speed = got["speed"]
    assert {"all_p160x6", "memory_p120x6", "base_1"} <= set(speed["failed_arms"])
    assert speed["arms"]["view_p160x6"]["decisions_per_s"] == 1000.0


def test_a_failed_graphed_check_leaves_out_the_graphed_arm(tmp_path):
    got = _run(tmp_path, STUB_graphed_check="fail", RATE_arm_view_p120x6=2500)
    assert got["rc"] == 0 and "GENERATION_SPEED_DONE" in got["finish"], (got["finish"], got["stderr"][-2000:])
    assert "graphed_best" not in got["arms"] and "arm graphed_best left out" in got["failed"]
    repeat = _arm_line(got, "best_repeat")
    assert "--jobs 120" in repeat and "--no-rust-stage --no-server-memory --view-refresh incremental" in repeat
    assert got["speed"]["best"] == "view_p120x6"
