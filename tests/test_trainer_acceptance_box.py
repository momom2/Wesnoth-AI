"""scripts/trainer_acceptance_box.sh's passes, run by bash with the box library
stubbed (no box, GPU or HF): B's gradient check against A gates the passes,
the passes run A, B, A2 from own1 with their flags, the real readout reads
their probes, and a pass that fails stops the run and resumes from its
checkpoint on re-entry while the passes done stay done."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from helpers.posix_bash import bash_path, find_bash

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "trainer_acceptance_box.sh"
BASH = find_bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="no POSIX bash")

# box_bounded answering for the steps: STUB_<step> = crash (exit 1, nothing
# written); GRADCHECK_RC: the check's exit code; the readout runs for real.
STUBS = r"""
box_marked_this_stage() { [ -f "$1" ]; }
box_mark() { mkdir -p "$(dirname "$1")" && echo "stage=$STAGE" > "$1"; }
box_upload() { :; }
box_gpu_ok() { :; }
box_finish() { printf '%s\n' "$1" > "$OUT/FINISH"; exit "${2:-0}"; }
notes() { echo notes; }
box_bounded() {                  # box_bounded [--stall FILE MIN] NAME CUT_MIN LOG CMD...
    local name key log
    [ "$1" != --stall ] || shift 3
    name=$1 log=$3
    shift 3
    echo "$name $*" >> "$OUT/steps.txt"
    key=STUB_${name// /_}
    BOX_RC=0 BOX_WHY=ok
    if [ "$name" = readout ]; then
        (cd "$REPO" && "$@") > "$OUT/$log" 2>&1 || BOX_RC=$?
    elif [ "${!key:-}" != crash ]; then
        python "$FAKE" "$name" "$@" > "$OUT/$log" 2>&1 || BOX_RC=$?
    else
        BOX_RC=1 BOX_WHY=failed
    fi
    return "$BOX_RC"
}
"""

# The stubbed check and passes: a pass writes its checkpoint, its probe's
# per-game-side sums (B's cross-entropy 0.001 above A's) and its log line.
FAKE = r'''
import json, os, sys
name, args = sys.argv[1], sys.argv[2:]
def opt(flag):
    return args[args.index(flag) + 1]
if name == "gradcheck":
    json.dump({"pass": os.environ.get("GRADCHECK_RC", "0") == "0", "candidates": []}, open(opt("--out"), "w"))
    sys.exit(int(os.environ.get("GRADCHECK_RC", "0")))
if name.startswith("pass "):
    out = opt("--out")
    stem = out[:-3]
    open(out, "w").write("checkpoint")
    ce = 2.0 + (0.001 if name == "pass B" else 0.0)
    sides = [{"file": f"g{i}", "side": s, "k0": [10, 10 * (ce + 0.01 * i), 12, 1.2],
              "k64": [10, 10 * (ce - 0.1 + 0.01 * i), 12, 1.1]} for i in range(6) for s in (1, 2)]
    with open(stem + ".probe_sides.jsonl", "a") as f:
        f.write(json.dumps({"positions": 500000, "final": True, "sides": sides}) + "\n")
    print("SEQUENCE_TRAIN_DONE 500000 positions, 977 steps in 1000 s")
'''


def _block() -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    return text[text.index("# ---- the passes (begin)"):text.index("# ---- the passes (end)")]


def _run(tmp_path: Path, out: Path, **env) -> dict:
    out.mkdir(exist_ok=True)
    fake = tmp_path / "fake.py"
    fake.write_text(FAKE, encoding="utf-8")
    (out / "steps.txt").unlink(missing_ok=True)
    (out / "FINISH").unlink(missing_ok=True)
    exports = "\n".join(f"export {k}='{v}'" for k, v in env.items())
    script = f"""set -uo pipefail
export PATH="{bash_path(Path(sys.executable).parent)}:/usr/bin:$PATH"
OUT="{bash_path(out)}" STAGE=stage_test FAKE="{bash_path(fake)}" REPO="{bash_path(ROOT)}"
SEQ=seq CORPUS=corpus OWN=own INIT_PT=own1.pt RUN_SEED=1 LR=2.8e-5 DECAY_FROM=0.5 PASS_POSITIONS=500000
A_FLAGS="--columns --activation-budget-gb 15.97" B_FLAGS="--packed-trunk --window-batched --trunk-bf16"
GRADCHECK_CUT_MIN=20 PASS_CUT_MIN=70 PASS_STALL_MIN=15 PASSES="A B A2" CUDA_TESTS="3 passed"
BOX_RC=0 BOX_WHY=ok
{exports}
{STUBS}
{_block()}
"""
    path = tmp_path / "driver.sh"
    path.write_bytes(script.encode())
    done = subprocess.run([BASH, bash_path(path)], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    steps = (out / "steps.txt").read_text().splitlines() if (out / "steps.txt").exists() else []
    return {"rc": done.returncode, "stderr": done.stderr, "steps": steps,
            "names": [" ".join(s.split()[:2]) if s.startswith("pass ") else s.split()[0] for s in steps],
            "finish": (out / "FINISH").read_text() if (out / "FINISH").exists() else ""}


def test_the_check_gates_three_passes_and_the_readout_reads_them(tmp_path):
    out = tmp_path / "out"
    got = _run(tmp_path, out)
    assert got["rc"] == 0 and "TRAINER_ACCEPTANCE_DONE" in got["finish"], (got["finish"], got["stderr"][-2000:])
    assert got["names"] == ["gradcheck", "pass A", "pass B", "pass A2", "readout"]
    check = got["steps"][0]
    assert "--reference=--columns --activation-budget-gb 15.97" in check
    assert "--candidate=--packed-trunk --window-batched --trunk-bf16" in check and "--init-from own1.pt" in check
    passes = {s.split()[1]: s for s in got["steps"] if s.startswith("pass ")}
    assert passes["A"].endswith("--columns --activation-budget-gb 15.97")
    assert passes["B"].endswith("--packed-trunk --window-batched --trunk-bf16")
    assert all("--init-from own1.pt" in s and "--pass-positions 500000" in s and "--probe-ks 0,64" in s
               for s in passes.values())
    result = json.loads((out / "acceptance.json").read_text())
    assert result["readings"]["ce_all_k0"]["B_minus_A"]["diff"] == pytest.approx(0.001)
    assert result["pass"] is False                    # a shift with no noise and no repeat difference
    assert '"positions_per_second": {"A": 500.0, "B": 500.0, "A2": 500.0}' in got["finish"]
    assert (out / "acceptance.txt").read_text().startswith("reading")


def test_a_failed_check_ends_the_run_before_any_pass(tmp_path):
    got = _run(tmp_path, tmp_path / "out", GRADCHECK_RC="8")
    assert got["rc"] == 1 and got["finish"].startswith("B_GRADCHECK_FAILED")
    assert got["names"] == ["gradcheck"]


def test_a_failed_pass_stops_the_run_and_resumes_on_re_entry(tmp_path):
    out = tmp_path / "out"
    got = _run(tmp_path, out, STUB_pass_B="crash")
    assert got["rc"] == 1 and got["finish"].startswith("PASS_FAILED B"), got["finish"]
    assert got["names"] == ["gradcheck", "pass A", "pass B"]
    (out / "pass_B.pt").write_text("a checkpoint the pass wrote before it died")
    again = _run(tmp_path, out)
    assert again["rc"] == 0 and "TRAINER_ACCEPTANCE_DONE" in again["finish"], again["finish"]
    assert again["names"] == ["pass B", "pass A2", "readout"]
    resumed = again["steps"][0]
    assert "--resume" in resumed and "--init-from" not in resumed
