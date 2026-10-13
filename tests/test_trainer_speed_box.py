"""scripts/trainer_speed_box.sh's arms, run by bash with the box library
stubbed (no box, GPU or HF): the base-against-this arms come first and a
failure after them costs only what depends on it. A gradient check that
crashes leaves the packed arms out, an arm that fails after writing its
total keeps its timed windows, and a last arm of this stage's defaults
brackets the flag arms."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from helpers.posix_bash import bash_path, find_bash

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "trainer_speed_box.sh"
BASH = find_bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="no POSIX bash")

# The library's functions the block calls, and box_bounded answering for
# the steps: STUB_<step> = crash (exit 1 before any record) or late (exit 1
# after the run's total row).
STUBS = r"""
box_marked_this_stage() { [ -f "$1" ]; }
box_mark() { mkdir -p "$(dirname "$1")" && echo "stage=$STAGE" > "$1"; }
box_upload() { :; }
box_gpu_ok() { :; }
box_finish() { printf '%s\n' "$1" > "$OUT/FINISH"; exit "${2:-0}"; }
notes() { echo notes; }
out_arg() {                      # the first --out of a command line
    while [ "$#" -gt 0 ]; do
        [ "$1" = --out ] && { echo "$2"; return; }
        shift
    done
}
arm_records() {                  # a run's records as tools/bench_sequence_step.py writes them
    {
        echo '{"kind": "config"}'
        for i in 0 1; do
            echo "{\"kind\": \"window\", \"i\": $i, \"seconds\": 2.5, \"positions\": 512, \"peak_gb\": 12.0, \"kept_steps\": 2, \"kept_tokens\": 40000, \"kept_gb\": 0.5}"
        done
        echo '{"kind": "total", "windows": 2, "positions": 1024, "seconds": 5.0, "positions_per_second": 204.8}'
    } > "$1"
}
box_bounded() {                  # box_bounded NAME CUT_MIN LOG CMD...
    local name=$1 log=$3 key out
    shift 3
    echo "$name $*" >> "$OUT/steps.txt"
    key=STUB_${name// /_}
    out=$(out_arg "$@")
    BOX_RC=0 BOX_WHY=ok
    case $name in
        arm\ *) [ "${!key:-}" = crash ] || arm_records "$out" ;;
        gradcheck) [ "${!key:-}" = crash ] || echo '{"pass": true}' > "$out" ;;
        compare\ *) echo '{"speedup_total": 1.01, "speedup_median": 1.0, "speedup_ci95": [0.99, 1.02]}' > "$out" ;;
        summary) "$@" >> "$OUT/$log" 2>&1 || BOX_RC=$? ;;
    esac
    case ${!key:-} in crash|late) BOX_RC=1 BOX_WHY=failed ;; esac
    return "$BOX_RC"
}
"""


def _block() -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    return text[text.index("# ---- the arms (begin)"):text.index("# ---- the arms (end)")]


def _step_name(line: str) -> str:
    """A bounded step's name: `arm NAME`, `compare A B`, or one word."""
    words = line.split()
    return " ".join(words[:{"arm": 2, "compare": 3}.get(words[0], 1)])


def _run(tmp_path: Path, **stubs) -> dict:
    out, repo, base = tmp_path / "out", tmp_path / "repo", tmp_path / "base"
    for d in (out, repo, base):
        d.mkdir()
    env = "\n".join(f"{k}={v}" for k, v in stubs.items())
    script = f"""set -uo pipefail
export PATH="{bash_path(Path(sys.executable).parent)}:/usr/bin:$PATH"
OUT="{bash_path(out)}" BOX_REPO="{bash_path(repo)}" BASE_REPO="{bash_path(base)}" STAGE=stage_test
SEQ=seq CORPUS=corpus OWN=own INIT_PT=own1.pt RUN_SEED=1 LR=2.8e-5 DECAY_FROM=0.5 PASS_POSITIONS=2000000
PROBE_EVERY=1200000 WARMUP=1 WINDOWS=2 PROFILE_WINDOWS=1 CAL_WARMUP=1 CAL_WINDOWS=2 CAL_PADDED_GB=4
CAL_PACKED_GB=2.5 PEAK_GB=22 MARGIN_GB=1 VALID_SPREAD=0.03 ARM_CUT_MIN=25 CUDA_TESTS="2 passed"
BOX_RC=0 BOX_WHY=ok
{env}
{STUBS}
{_block()}
"""
    path = tmp_path / "driver.sh"
    path.write_bytes(script.encode())
    done = subprocess.run([BASH, bash_path(path)], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    steps = (out / "steps.txt").read_text().splitlines() if (out / "steps.txt").exists() else []
    return {"rc": done.returncode, "stderr": done.stderr, "out": out,
            "finish": (out / "FINISH").read_text() if (out / "FINISH").exists() else "",
            "steps": steps,
            "names": [_step_name(s) for s in steps],
            "failed": (out / "arms_failed.txt").read_text() if (out / "arms_failed.txt").exists() else ""}


def test_a_crashed_gradient_check_leaves_out_only_the_packed_arms(tmp_path):
    got = _run(tmp_path, STUB_gradcheck="crash")
    assert got["rc"] == 0 and "TRAINER_SPEED_DONE" in got["finish"], (got["finish"], got["stderr"][-2000:])
    names = got["names"]
    first = ["arm base_1", "arm this_1", "arm base_2", "arm this_2"]
    assert names[:5] == [*first, "gradcheck"]
    assert not any(n in names for n in ("arm packed", "arm cal_packed", "arm packed_budget"))
    assert {"arm budget", "arm preload", "arm this_3"} <= set(names)
    check = json.loads((got["out"] / "gradcheck.json").read_text())
    assert check["pass"] is False and check["rc"] == 1
    assert "GRADCHECK_FAILED gradcheck rc=1" in got["failed"]
    speed = json.loads((got["out"] / "speed.json").read_text())
    assert speed["gradcheck"] is False and speed["valid"] and speed["flags_valid"]
    assert "P4" in speed["not_measured"]


def test_an_arm_that_fails_after_its_total_keeps_its_windows(tmp_path):
    """A base arm whose profile failed is timed; a packed arm that crashed
    is left out, its budget arm with it, and this_3 brackets the rest."""
    got = _run(tmp_path, STUB_arm_base_1="late", STUB_arm_packed="crash")
    assert got["rc"] == 0 and "TRAINER_SPEED_DONE" in got["finish"], (got["finish"], got["stderr"][-2000:])
    assert (got["out"] / "arm_base_1.done").exists() and not (got["out"] / "arm_packed.done").exists()
    assert "AFTER_TOTAL_FAILED base_1" in got["failed"] and "ARM_FAILED packed" in got["failed"]
    names = got["names"]
    assert "arm cal_packed" not in names and names.index("arm this_3") > names.index("arm preload")
    assert "compare this_1 this_3" in names
    speed = json.loads((got["out"] / "speed.json").read_text())
    assert "base_1" in speed["arms"] and "this_1_vs_this_3" in speed["comparisons"]


def test_with_every_step_passing_the_preload_arm_takes_the_fastest_flags(tmp_path):
    got = _run(tmp_path)
    assert got["rc"] == 0 and "TRAINER_SPEED_DONE" in got["finish"], (got["finish"], got["stderr"][-2000:])
    names = got["names"]
    assert names.index("gradcheck") < names.index("arm packed") < names.index("arm packed_budget")
    preload = next(s for s in got["steps"] if s.startswith("arm preload"))
    budget = next(s for s in got["steps"] if s.startswith("arm packed_budget"))
    assert "--preload" in preload and "--packed-trunk" in preload and "--activation-budget-gb" in budget
    assert not got["failed"]
