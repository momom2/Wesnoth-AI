"""scripts/trainer_acceptance_box.sh's passes, run by bash with the box library
stubbed (no box, GPU or HF): the gradient check of B and B8 against A gates
the passes, the passes run A, B, A2, B8, B64 and BHALF from own1 with their
flags and rates, the real readout reads every finished arm against A; B
failing its check ends the run and B8 failing its own leaves B8 out; A's
pass failing stops the run and resumes on re-entry, another's is left out
and stays out; ARMS picks the passes."""
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
# written); the readout runs for real.
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

# The stubbed check and passes: the check fails the candidates FAIL_CHECK
# names (B, B8) and exits 8 then, or exits GRADCHECK_RC; a pass writes its
# checkpoint, its probe's per-game-side sums (B's cross-entropy 0.001 above
# A's, B64's 0.003) and its log line.
FAKE = r'''
import json, os, sys
name, args = sys.argv[1], sys.argv[2:]
def opt(flag):
    return args[args.index(flag) + 1]
if name == "gradcheck":
    if os.environ.get("GRADCHECK_RC"):
        sys.exit(int(os.environ["GRADCHECK_RC"]))
    fail = os.environ.get("FAIL_CHECK", "").split()
    flags = [a.split("=", 1)[1] for a in args if a.startswith("--candidate=")]
    cands = [{"candidate": f, "pass": ("B8" if "--fp8-trunk" in f else "B") not in fail} for f in flags]
    json.dump({"pass": all(c["pass"] for c in cands), "candidates": cands}, open(opt("--out"), "w"))
    sys.exit(0 if all(c["pass"] for c in cands) else 8)
if name.startswith("pass "):
    out = opt("--out")
    stem = out[:-3]
    open(out, "w").write("checkpoint")
    ce = 2.0 + {"pass B": 0.001, "pass B64": 0.003}.get(name, 0.0)
    sides = [{"file": f"g{i}", "side": s, "k0": [10, 10 * (ce + 0.01 * i), 12, 1.2],
              "k64": [10, 10 * (ce - 0.1 + 0.01 * i), 12, 1.1]} for i in range(6) for s in (1, 2)]
    with open(stem + ".probe_sides.jsonl", "a") as f:
        f.write(json.dumps({"positions": 500000, "final": True, "sides": sides}) + "\n")
    print("SEQUENCE_TRAIN_DONE 500000 positions, 977 steps in 1000 s")
'''

A_FLAGS = "--columns --activation-budget-gb 15.97"
B_FLAGS = "--packed-trunk --window-batched --fused-relu"


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
A_FLAGS="{A_FLAGS}" B_FLAGS="{B_FLAGS}" B8_FLAGS="" BIG_STREAMS=64 SUBSET_EVERY=2
ARMS="A B A2 B8 B64 BHALF" GRADCHECK_CUT_MIN=20 PASS_CUT_MIN=70 PASS_STALL_MIN=15 CUDA_TESTS="3 passed"
BOX_RC=0 BOX_WHY=ok
{exports}
{STUBS}
{_block()}
"""
    path = tmp_path / "driver.sh"
    path.write_bytes(script.encode())
    done = subprocess.run([BASH, bash_path(path)], cwd=tmp_path, capture_output=True, text=True, timeout=180)
    steps = (out / "steps.txt").read_text().splitlines() if (out / "steps.txt").exists() else []
    failed = out / "arms_failed.txt"
    return {"rc": done.returncode, "stderr": done.stderr, "steps": steps,
            "names": [" ".join(s.split()[:2]) if s.startswith("pass ") else s.split()[0] for s in steps],
            "finish": (out / "FINISH").read_text() if (out / "FINISH").exists() else "",
            "failed": failed.read_text() if failed.exists() else ""}


def _passes(steps):
    return {s.split()[1]: s for s in steps if s.startswith("pass ")}


def test_the_check_gates_the_arms_and_the_readout_reads_every_one(tmp_path):
    out = tmp_path / "out"
    got = _run(tmp_path, out)
    assert got["rc"] == 0 and "TRAINER_ACCEPTANCE_DONE" in got["finish"], (got["finish"], got["stderr"][-2000:])
    assert got["names"] == ["gradcheck", "pass A", "pass B", "pass A2", "pass B8", "pass B64", "pass BHALF",
                            "readout"]
    check = got["steps"][0]
    assert f"--reference={A_FLAGS}" in check and "--init-from own1.pt" in check
    assert f"--candidate={B_FLAGS} " in check
    assert "--candidate=--packed-trunk --window-batched --fp8-trunk " in check     # fp8 takes no fused ReLU
    passes = _passes(got["steps"])
    assert passes["A"].endswith(f"--lr 2.8e-5 {A_FLAGS}") and passes["A2"].endswith(f"--lr 2.8e-5 {A_FLAGS}")
    assert passes["B"].endswith(f"--lr 2.8e-5 {B_FLAGS}")
    assert passes["B8"].endswith("--lr 2.8e-5 --packed-trunk --window-batched --fp8-trunk")
    assert passes["B64"].endswith(f"--lr 3.9598e-05 {B_FLAGS} --streams 64 --own-streams 32")
    assert passes["BHALF"].endswith(f"--lr 2.8e-5 {B_FLAGS} --train-every 2")
    assert all("--init-from own1.pt" in s and "--pass-positions 500000" in s and "--probe-ks 0,64" in s
               for s in passes.values())
    result = json.loads((out / "acceptance.json").read_text())
    assert list(result["arms"]) == ["B", "B8", "B64", "BHALF"]
    assert result["arms"]["B"]["readings"]["ce_all_k0"]["minus_A"]["diff"] == pytest.approx(0.001)
    assert result["arms"]["B64"]["readings"]["ce_all_k0"]["minus_B"]["diff"] == pytest.approx(0.002)
    assert result["arms"]["B"]["pass"] is False and result["arms"]["B8"]["pass"] is True
    assert '"B64": 500.0' in got["finish"] and '"BHALF": 500.0' in got["finish"] and not got["failed"]
    assert (out / "acceptance.txt").read_text().startswith("reading")


def test_b_failing_its_check_ends_the_run_and_b8_failing_its_own_leaves_b8_out(tmp_path):
    got = _run(tmp_path, tmp_path / "b", FAIL_CHECK="B")
    assert got["rc"] == 1 and got["finish"].startswith("B_GRADCHECK_FAILED") and got["names"] == ["gradcheck"]
    unfinished = _run(tmp_path, tmp_path / "rc", GRADCHECK_RC="1")
    assert unfinished["rc"] == 1 and unfinished["finish"].startswith("B_GRADCHECK_UNFINISHED")
    b8 = _run(tmp_path, tmp_path / "b8", FAIL_CHECK="B8")
    assert b8["rc"] == 0 and "TRAINER_ACCEPTANCE_DONE" in b8["finish"], (b8["finish"], b8["stderr"][-2000:])
    assert "pass B8" not in b8["names"] and "LEFT_OUT B8: its gradient check" in b8["failed"]
    assert list(json.loads((tmp_path / "b8" / "acceptance.json").read_text())["arms"]) == ["B", "B64", "BHALF"]


def test_a_failed_pass_of_a_stops_the_run_and_resumes_on_re_entry(tmp_path):
    out = tmp_path / "out"
    got = _run(tmp_path, out, STUB_pass_A="crash")
    assert got["rc"] == 1 and got["finish"].startswith("PASS_FAILED A"), got["finish"]
    assert got["names"] == ["gradcheck", "pass A"]
    (out / "pass_A.pt").write_text("a checkpoint the pass wrote before it died")
    again = _run(tmp_path, out)
    assert again["rc"] == 0 and "TRAINER_ACCEPTANCE_DONE" in again["finish"], again["finish"]
    assert again["names"][0] == "pass A" and "--resume" in again["steps"][0] and "--init-from" not in again["steps"][0]


def test_another_failed_pass_is_left_out_and_stays_out_on_re_entry(tmp_path):
    out = tmp_path / "out"
    got = _run(tmp_path, out, STUB_pass_B64="crash")
    assert got["rc"] == 0 and "TRAINER_ACCEPTANCE_DONE" in got["finish"], (got["finish"], got["stderr"][-2000:])
    assert got["names"][-2:] == ["pass BHALF", "readout"] and "LEFT_OUT B64: its pass failed" in got["failed"]
    assert list(json.loads((out / "acceptance.json").read_text())["arms"]) == ["B", "B8", "BHALF"]
    again = _run(tmp_path, out)
    assert again["names"] == ["readout"] and "TRAINER_ACCEPTANCE_DONE" in again["finish"]


def test_arms_picks_the_passes(tmp_path):
    got = _run(tmp_path, tmp_path / "few", ARMS="A B B64")
    assert got["rc"] == 0 and got["names"] == ["gradcheck", "pass A", "pass B", "pass B64", "readout"], got["stderr"]
    assert got["steps"][0].count("--candidate=") == 1
    result = json.loads((tmp_path / "few" / "acceptance.json").read_text())
    assert "A2" not in result["files"] and result["arms"]["B"]["pass"] is None
    no_a = _run(tmp_path, tmp_path / "no_a", ARMS="B A2")
    assert no_a["rc"] == 1 and no_a["finish"].startswith("NO_ARM_A") and not no_a["steps"]
    unknown = _run(tmp_path, tmp_path / "unknown", ARMS="A B9")
    assert unknown["rc"] == 1 and unknown["finish"].startswith("UNKNOWN_ARM B9") and not unknown["steps"]
