"""scripts/optimizer_bench_box.sh's arms, run by bash with the box library
stubbed (no box, GPU or HF): every arm trains from scratch over the same
positions with its optimizer and its grid's rate, the scheduled ones cooling
down and keeping a weight average, schedule-free with neither; the averages
of the best two arms that keep one are probed on their finished passes; the
real readout reads every finished arm against adamw_mid; an arm that fails
is left out, the others run, and a re-entry keeps it out, skips what is done
and resumes a pass from its checkpoint; ARMS refuses a name it cannot read."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from helpers.posix_bash import bash_path, find_bash

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "optimizer_bench_box.sh"
BASH = find_bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="no POSIX bash")

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

# A stubbed pass writes its checkpoint, its probes at about half the pass and
# at its end (per-game-side sums whose cross-entropy is 2 plus the arm's
# offset), its steps; an average's probe writes the final probe of the
# average, 0.004 below its arm's.
FAKE = r'''
import json, sys
name, args = sys.argv[1], sys.argv[2:]
OFFSET = {"schedulefree_mid": -0.05, "muon_mid": -0.03, "soap_mid": -0.02, "adamw_hi": -0.01}
def opt(flag):
    return args[args.index(flag) + 1]
def sides(ce):
    return [{"file": f"g{i}", "side": s, "k0": [10, 10 * (ce + 0.3 + 0.01 * i), 12, 1.2],
             "k64": [10, 10 * (ce + 0.01 * i), 12, 1.1], "k64_reset": [10, 10 * (ce + 0.02), 12, 1.3]}
            for i in range(6) for s in (1, 2)]
kind, arm = name.split()
stem = opt("--out")[:-3]
ce = 2.0 + OFFSET.get(arm, 0.0)
if kind == "pass":
    open(stem + ".pt", "w").write("checkpoint")
    for positions, final in ((501248, False), (1000448, True)):
        with open(stem + ".probe_sides.jsonl", "a") as f:
            f.write(json.dumps({"positions": positions, "final": final, "weights": "live", "sides": sides(ce + 0.1 * (not final))}) + "\n")
        with open(stem + ".probe.jsonl", "a") as f:
            f.write(json.dumps({"positions": positions, "final": final, "weights": "live", "k64": {"ce_all": ce},
                                "optimizer_state_gb": 0.25}) + "\n")
    with open(stem + ".steps.jsonl", "a") as f:
        for i in range(30):
            f.write(json.dumps({"step": i + 1, "positions": 512 * (i + 1), "seconds": 0.5, "peak_gb": 20.0}) + "\n")
    print("SEQUENCE_TRAIN_DONE 1000448 positions")
if kind == "ema":
    with open(stem + ".ema.probe_sides.jsonl", "a") as f:
        f.write(json.dumps({"positions": 1000448, "final": True, "weights": "ema", "sides": sides(ce - 0.004)}) + "\n")
'''

B_FLAGS = "--packed-trunk --columns --window-batched --compiled-trunk --activation-budget-gb auto"
ARMS = ["adamw_mid", "muon_mid", "soap_mid", "schedulefree_mid", "adamw_hygiene_mid", "adamw_lo", "adamw_hi",
        "muon_lo", "muon_hi", "soap_lo", "soap_hi", "schedulefree_lo", "schedulefree_hi", "adamw_hygiene_lo",
        "adamw_hygiene_hi", "adamw_mid2"]


def _block() -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    return text[text.index("# ---- the arms (begin)"):text.index("# ---- the arms (end)")]


def _defaults() -> str:
    """The script's own settings lines, from RUN_SEED to REPEAT."""
    text = SCRIPT.read_text(encoding="utf-8")
    return text[text.index('RUN_SEED="${RUN_SEED:-'):text.index("WORKERS=")]


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
SEQ=seq CORPUS=corpus CUDA_TESTS="4 passed" PASS_CUT_MIN=45 PASS_STALL_MIN=15 EMA_CUT_MIN=20
BOX_RC=0 BOX_WHY=ok
{exports}
{_defaults()}
{STUBS}
{_block()}
"""
    path = tmp_path / "driver.sh"
    path.write_bytes(script.encode())
    done = subprocess.run([BASH, bash_path(path)], cwd=tmp_path, capture_output=True, text=True, timeout=180)
    steps = (out / "steps.txt").read_text().splitlines() if (out / "steps.txt").exists() else []
    failed = out / "arms_failed.txt"
    return {"rc": done.returncode, "stderr": done.stderr, "steps": steps,
            "names": [" ".join(s.split()[:2]) if s.split()[0] in ("pass", "ema") else s.split()[0] for s in steps],
            "finish": (out / "FINISH").read_text() if (out / "FINISH").exists() else "",
            "failed": failed.read_text() if failed.exists() else ""}


def _steps(steps, kind):
    return {s.split()[1]: s for s in steps if s.startswith(kind + " ")}


def test_every_arm_trains_from_scratch_with_its_optimizer_and_rate_and_the_readout_reads_them(tmp_path):
    out = tmp_path / "out"
    got = _run(tmp_path, out)
    assert got["rc"] == 0 and "OPTIMIZER_BENCH_DONE" in got["finish"], (got["finish"], got["stderr"][-2000:])
    assert got["names"] == [f"pass {a}" for a in ARMS] + ["ema muon_mid", "ema soap_mid", "readout"]
    rates = {"lo": "1.4e-4", "mid": "2.8e-4", "mid2": "2.8e-4", "hi": "5.6e-4"}
    sf_rates = {"lo": "2.8e-4", "mid": "8.4e-4", "hi": "2.8e-3"}
    for arm, step in _steps(got["steps"], "pass").items():
        optimizer, rate = arm.rsplit("_", 1)
        assert "--pass-positions 1000000 --probe-every 501000 --barrier-positions 2000000" in step
        assert "--seed 20260929" in step and "--init-from" not in step and "--resume" not in step
        assert B_FLAGS in step and f"--out {bash_path(out)}/pass_{arm}.pt" in step
        if optimizer == "schedulefree":
            assert f"--optimizer schedulefree --lr {sf_rates[rate]} " in step
            assert "--decay-from" not in step and "--weight-ema" not in step
        else:
            assert f"--optimizer {optimizer} --lr {rates[rate]} " in step
            assert step.endswith("--decay-from 0.8 --weight-ema 0.998")
    passes = _steps(got["steps"], "pass")
    for arm, step in _steps(got["steps"], "ema").items():
        flags = passes[arm].split(f"pass_{arm}.pt", 1)[1]
        assert f"--resume --out {bash_path(out)}/pass_{arm}.pt{flags} --probe-weights ema" in step
    readout = got["steps"][-1]
    assert "--reference adamw_mid --repeat adamw_mid2 --arms " + " ".join(ARMS) in readout
    bench = json.loads((out / "bench.json").read_text())
    assert set(bench["arms"]) == set(ARMS) and not bench["refused"]
    muon = bench["arms"]["muon_mid"]
    assert muon["probes"]["final"]["ce_all_k64"]["minus"]["diff"] == pytest.approx(-0.03)
    assert muon["probes"]["mid"]["ce_all_k64"]["minus"]["diff"] == pytest.approx(-0.03)
    assert muon["ema"]["minus_live"]["ce_all_k64"]["minus"]["diff"] == pytest.approx(-0.004)
    assert muon["positions_per_second"] == pytest.approx(1024.0) and muon["peak_gb"] == 20.0
    assert "ema: muon_mid soap_mid" in got["finish"] and "4 passed" in got["finish"]


def test_an_arm_that_fails_is_left_out_and_a_re_entry_keeps_it_out(tmp_path):
    out = tmp_path / "out"
    got = _run(tmp_path, out, STUB_pass_muon_mid="crash", STUB_ema_soap_mid="crash")
    assert got["rc"] == 0 and "OPTIMIZER_BENCH_DONE" in got["finish"], (got["finish"], got["stderr"][-2000:])
    assert [n for n in got["names"] if n.startswith("pass ")] == [f"pass {a}" for a in ARMS]
    assert "LEFT_OUT muon_mid" in got["failed"] and "EMA_FAILED soap_mid" in got["failed"]
    assert [n for n in got["names"] if n.startswith("ema ")] == ["ema soap_mid", "ema adamw_hi"]
    bench = json.loads((out / "bench.json").read_text())
    assert "muon_mid" not in bench["arms"] and "ema" not in bench["arms"]["soap_mid"]
    (out / "pass_adamw_lo.done").unlink()
    again = _run(tmp_path, out)
    assert again["rc"] == 0 and "OPTIMIZER_BENCH_DONE" in again["finish"], again["stderr"][-2000:]
    assert again["names"] == ["pass adamw_lo", "ema soap_mid", "readout"], "only what is not done runs again"
    assert "--resume" in again["steps"][0], "a pass with a checkpoint continues it"


def test_a_pass_started_again_clears_an_earlier_attempt_s_rows(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "pass_adamw_mid.probe.jsonl").write_text('{"positions": 1, "final": false}\n')
    got = _run(tmp_path, out, ARMS="adamw_mid", EMA_ARMS="")
    assert got["rc"] == 0 and got["names"] == ["pass adamw_mid", "readout"], got["stderr"][-2000:]
    rows = (out / "pass_adamw_mid.probe.jsonl").read_text().splitlines()
    assert len(rows) == 2 and json.loads(rows[0])["positions"] == 501248


def test_arms_refuses_a_name_it_cannot_read(tmp_path):
    for arms in ("adamw_med", "lion_mid"):
        (tmp_path / arms).mkdir()
        got = _run(tmp_path / arms, tmp_path / arms / "out", ARMS=arms)
        assert got["rc"] == 1 and got["finish"].startswith(f"UNKNOWN_ARM {arms}") and not got["steps"]
