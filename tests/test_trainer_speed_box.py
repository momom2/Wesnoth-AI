"""scripts/trainer_speed_box.sh's arms, run by bash with the box library
stubbed (no box, GPU or HF): the gradient check reads every candidate
against main's fastest flags, a candidate that fails it (or a check that
does not finish) is left out of the timing, the timing interleaves ref, the
passing candidates and ref2, and a timing that fails after writing rows is
summarized from them; CANDIDATES picks the candidates; the streams sweep
times each count of SWEEP_STREAMS with half its streams own, outside
production's arguments, and is skipped when the list is empty."""
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
# the steps: STUB_<step> = crash (exit 1, nothing written) or late (exit 1
# after the rows); FAIL_CANDIDATES: the candidates the gradient check fails.
STUBS = r"""
box_marked_this_stage() { [ -f "$1" ]; }
box_mark() { mkdir -p "$(dirname "$1")" && echo "stage=$STAGE" > "$1"; }
box_upload() { :; }
box_gpu_ok() { :; }
box_finish() { printf '%s\n' "$1" > "$OUT/FINISH"; exit "${2:-0}"; }
notes() { echo notes; }
box_bounded() {                  # box_bounded NAME CUT_MIN LOG CMD...
    local name=$1 key
    shift 3
    echo "$name $*" >> "$OUT/steps.txt"
    key=STUB_${name// /_}
    key=${key//-/_}
    BOX_RC=0 BOX_WHY=ok
    if [ "${!key:-}" != crash ]; then
        python "$FAKE" "$name" "$@" || BOX_RC=$?
    fi
    case ${!key:-} in crash|late) BOX_RC=1 BOX_WHY=failed ;; esac
    return "$BOX_RC"
}
"""

# What the stubbed steps write: the gradient check's verdicts, the timing's
# rows and summary.
FAKE = r'''
import json, os, sys
name, args = sys.argv[1], sys.argv[2:]
def opt(flag):
    return args[args.index(flag) + 1]
if name == "gradcheck":
    fail = os.environ.get("FAIL_CANDIDATES", "").split()
    flags = [a.split("=", 1)[1] for a in args if a.startswith("--candidate=")]
    tags = {"--trunk-bf16": "bf16", "--fused-relu": "relu", "--compiled-trunk": "compiled", "--fp8-trunk": "fp8"}
    def name_of(f):
        return "_".join(tags[x] for x in f.split() if x in tags)
    cands = [{"candidate": f, "pass": name_of(f) not in fail, "windows": []} for f in flags]
    json.dump({"pass": all(c["pass"] for c in cands), "candidates": cands}, open(opt("--out"), "w"))
    sys.exit(0 if all(c["pass"] for c in cands) else 8)
if name == "timing":
    arms = [a.split("=", 2)[1] for a in args if a.startswith("--arm=")]
    with open(opt("--out"), "w") as f:
        f.write(json.dumps({"kind": "config", "arm": arms[0]}) + "\n")
    if os.environ.get("STUB_timing") != "late":
        summary = {"valid": True, "arms": {a: {"positions_per_second": 1000.0 + i,
                   "against_reference": None if a == "ref" else {"speedup_median": 1.0 + i / 10}}
                   for i, a in enumerate(arms)}}
        json.dump(summary, open(opt("--summary"), "w"))
if name == "timing-summary":
    json.dump({"valid": None, "arms": {"ref": {"positions_per_second": 999.0}}}, open(opt("--out"), "w"))
if name == "sweep":
    arms = [a.split("=", 2)[1] for a in args if a.startswith("--arm=")]
    with open(opt("--out"), "w") as f:
        f.write(json.dumps({"kind": "config", "arm": arms[0]}) + "\n")
    if os.environ.get("STUB_sweep") != "late":
        json.dump({"valid": None, "arms": {a: {"positions_per_second": 1200.0 * (i + 1), "peak_gb_max": 8.0 * (i + 1)}
                   for i, a in enumerate(arms)}}, open(opt("--summary"), "w"))
if name == "sweep-summary":
    json.dump({"valid": None, "arms": {"s32": {"positions_per_second": 1199.0, "peak_gb_max": 8.0}}},
              open(opt("--out"), "w"))
'''


def _block() -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    return text[text.index("# ---- the arms (begin)"):text.index("# ---- the arms (end)")]


def _run(tmp_path: Path, **env) -> dict:
    out = tmp_path / "out"
    out.mkdir(parents=True)
    fake = tmp_path / "fake.py"
    fake.write_text(FAKE, encoding="utf-8")
    exports = "\n".join(f"export {k}='{v}'" for k, v in env.items())
    script = f"""set -uo pipefail
export PATH="{bash_path(Path(sys.executable).parent)}:/usr/bin:$PATH"
OUT="{bash_path(out)}" STAGE=stage_test FAKE="{bash_path(fake)}"
SEQ=seq CORPUS=corpus OWN=own INIT_PT=own1.pt RUN_SEED=1 LR=2.8e-5 DECAY_FROM=0.5 PASS_POSITIONS=2000000
PROBE_EVERY=1200000 WARMUP=1 ROUNDS=2 PROFILE_WINDOWS=1 BUDGET_GB=19.18 BYTES_PER_TOKEN=102139
VALID_SPREAD=0.03 GRADCHECK_CUT_MIN=25 TIMING_CUT_MIN=45 CUDA_TESTS="2 passed"
CANDIDATES="bf16 relu compiled bf16_relu bf16_compiled fp8 compiled_fp8"
SWEEP_STREAMS="32 64 128" SWEEP_FLAGS="" SWEEP_WARMUP=1 SWEEP_ROUNDS=2 SWEEP_CUT_MIN=30
BOX_RC=0 BOX_WHY=ok
{exports}
{STUBS}
{_block()}
"""
    path = tmp_path / "driver.sh"
    path.write_bytes(script.encode())
    done = subprocess.run([BASH, bash_path(path)], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    steps = (out / "steps.txt").read_text().splitlines() if (out / "steps.txt").exists() else []
    return {"rc": done.returncode, "stderr": done.stderr, "out": out,
            "finish": (out / "FINISH").read_text() if (out / "FINISH").exists() else "",
            "steps": steps, "failed": (out / "arms_failed.txt").read_text() if (out / "arms_failed.txt").exists() else ""}


FASTEST = "--packed-trunk --activation-budget-gb 19.18 --activation-bytes-per-token 102139 --columns --window-batched"
CANDIDATES = ["bf16", "relu", "compiled", "bf16_relu", "bf16_compiled", "fp8", "compiled_fp8"]


def _arms(steps, step="timing"):
    timing = next(s for s in steps if s.startswith(step + " "))
    return [a.split("=", 1) for a in timing.split("--arm=")[1:]]


def test_every_candidate_is_checked_against_main_s_fastest_flags_and_timed_between_two_references(tmp_path):
    got = _run(tmp_path)
    assert got["rc"] == 0 and "TRAINER_SPEED_DONE" in got["finish"], (got["finish"], got["stderr"][-2000:])
    names = [s.split()[0] for s in got["steps"]]
    assert names == ["gradcheck", "timing", "sweep"]
    check = got["steps"][0]
    assert f"--reference={FASTEST}" in check
    assert check.count("--candidate=") == 7 and f"--candidate={FASTEST} --trunk-bf16 --compiled-trunk" in check
    assert f"--candidate={FASTEST} --compiled-trunk --fp8-trunk" in check
    arms = _arms(got["steps"])
    assert [a[0] for a in arms] == ["ref", *CANDIDATES, "ref2"]
    assert all(a[1].strip().startswith(FASTEST) for a in arms)
    assert arms[0][1].strip() == FASTEST and arms[-1][1].split(" --reference")[0].strip() == FASTEST
    assert "--reference ref" in got["steps"][1] and "--summary" in got["steps"][1]
    assert '"valid": true' in got["finish"] and "2 passed" in got["finish"] and not got["failed"]


def test_a_candidate_that_fails_its_check_is_left_out(tmp_path):
    got = _run(tmp_path, FAIL_CANDIDATES="relu bf16_compiled")
    assert got["rc"] == 0 and "TRAINER_SPEED_DONE" in got["finish"], (got["finish"], got["stderr"][-2000:])
    assert [a[0] for a in _arms(got["steps"])] == ["ref", "bf16", "compiled", "bf16_relu", "fp8", "compiled_fp8",
                                                   "ref2"]
    assert "LEFT_OUT relu" in got["failed"] and "LEFT_OUT bf16_compiled" in got["failed"]
    assert json.loads((got["out"] / "gradcheck.json").read_text())["pass"] is False


def test_a_check_that_does_not_finish_leaves_every_candidate_out(tmp_path):
    got = _run(tmp_path, STUB_gradcheck="crash")
    assert got["rc"] == 0 and "TRAINER_SPEED_DONE" in got["finish"], (got["finish"], got["stderr"][-2000:])
    assert [a[0] for a in _arms(got["steps"])] == ["ref", "ref2"]
    assert "GRADCHECK_FAILED rc=1" in got["failed"]
    assert json.loads((got["out"] / "gradcheck.json").read_text())["candidates"] == []


def test_a_timing_that_fails_after_its_rows_is_summarized_from_them(tmp_path):
    got = _run(tmp_path, STUB_timing="late")
    assert got["rc"] == 0 and "TRAINER_SPEED_DONE" in got["finish"], (got["finish"], got["stderr"][-2000:])
    assert [s.split()[0] for s in got["steps"]] == ["gradcheck", "timing", "timing-summary", "sweep"]
    assert "TIMING_FAILED rc=1" in got["failed"] and "999.0" in got["finish"]


def test_a_timing_that_writes_nothing_ends_the_run(tmp_path):
    got = _run(tmp_path, STUB_timing="crash")
    assert got["rc"] == 1 and got["finish"].startswith("TIMING_FAILED: no speed.json")


def test_candidates_picks_the_candidates_and_an_unknown_name_ends_the_run(tmp_path):
    got = _run(tmp_path / "a", CANDIDATES="fp8 compiled_fp8", SWEEP_STREAMS="")
    assert got["rc"] == 0 and "TRAINER_SPEED_DONE" in got["finish"], (got["finish"], got["stderr"][-2000:])
    assert got["steps"][0].count("--candidate=") == 2
    assert [a[0] for a in _arms(got["steps"])] == ["ref", "fp8", "compiled_fp8", "ref2"]
    none = _run(tmp_path / "b", CANDIDATES="", SWEEP_STREAMS="")
    assert none["rc"] == 0 and [s.split()[0] for s in none["steps"]] == ["timing"], none["stderr"][-2000:]
    assert [a[0] for a in _arms(none["steps"])] == ["ref", "ref2"]
    wrong = _run(tmp_path / "c", CANDIDATES="fp16")
    assert wrong["rc"] == 1 and wrong["finish"].startswith("UNKNOWN_CANDIDATE fp16") and not wrong["steps"]


def test_the_sweep_times_each_stream_count_with_half_its_streams_own(tmp_path):
    got = _run(tmp_path, SWEEP_FLAGS="--packed-trunk --columns --window-batched --compiled-trunk")
    assert got["rc"] == 0 and "TRAINER_SPEED_DONE" in got["finish"], (got["finish"], got["stderr"][-2000:])
    sweep = next(s for s in got["steps"] if s.startswith("sweep "))
    assert [a[0] for a in _arms(got["steps"], "sweep")] == ["s32", "s64", "s128"]
    for name, flags in _arms(got["steps"], "sweep"):
        n = int(name[1:])
        assert flags.split(" --reference")[0].strip().endswith(f"--compiled-trunk --streams {n} --own-streams {n // 2}")
    assert "--not-production" in sweep and "--reference s32" in sweep and "--seed 1" in sweep
    assert '"sweep": {"s32": [1200.0, 8.0], "s64": [2400.0, 16.0], "s128": [3600.0, 24.0]}' in got["finish"]
    default = _run(tmp_path / "default")
    assert all(a[1].strip().startswith(FASTEST + " --streams") for a in _arms(default["steps"], "sweep"))


def test_a_sweep_that_fails_after_its_rows_is_summarized_and_one_that_wrote_none_is_named(tmp_path):
    late = _run(tmp_path / "late", STUB_sweep="late")
    assert late["rc"] == 0 and [s.split()[0] for s in late["steps"]][-2:] == ["sweep", "sweep-summary"]
    assert "SWEEP_FAILED rc=1" in late["failed"] and '"s32": [1199.0, 8.0]' in late["finish"]
    crash = _run(tmp_path / "crash", STUB_sweep="crash")
    assert crash["rc"] == 0 and "SWEEP_FAILED rc=1" in crash["failed"] and "no rows" in crash["failed"]
    assert '"sweep"' not in crash["finish"]
