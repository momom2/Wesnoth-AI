"""scripts/stage_c_box.sh's plan and stages, run by bash with the box library
stubbed (no box, GPU, core or HF): stage 1's records restored, C0, the plan
from C0's rates, the gradient pass, the stage C driver, the planted readout
and the verdict. Each part's done marker waits for its files; the arms are
trimmed when the forwards run at the bound; a run that does not fit the
switch stops before the pass; a failed C0 or pass stops what needs it, a
failed stage C driver leaves the plant to run and a failed plant leaves stage
C's verdict; stage 1's failed gates stop the run; a re-entry runs nothing
done."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from helpers.posix_bash import bash_path, find_bash

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "stage_c_box.sh"
BASH = find_bash()
pytestmark = [pytest.mark.slow, pytest.mark.skipif(BASH is None, reason="no POSIX bash")]

STUBS = r"""
box_marked_this_stage() { [ -f "$1" ]; }
box_mark() { mkdir -p "$(dirname "$1")" && echo "stage=$STAGE" > "$1"; }
box_upload_async() { :; }
box_upload_hold() { echo "hold $*" >> "$OUT/steps.txt"; }
box_now() { cat "$STUB_DIR/clock"; }
box_stamp() { date; }
box_finish() { printf '%s\n' "$1" > "$OUT/FINISH"; exit "${2:-0}"; }
box_bounded() {                  # box_bounded [--stall FILE MIN] NAME CUT_MIN LOG CMD...
    local name cut log
    [ "$1" != --stall ] || shift 3
    name=$1 cut=$2 log=$3
    shift 3
    echo "$name [$cut] $*" >> "$OUT/steps.txt"
    BOX_RC=0 BOX_WHY=ok
    python "$FAKE" "$name" "$@" >> "$OUT/$log" 2>&1 || BOX_RC=$?
    [ "$BOX_RC" -eq 0 ] || BOX_WHY=failed
    return "$BOX_RC"
}
"""

# Each step writes what the real tool would leave for the next, then exits
# with STUB_RC's code for it (a JSON object: step name -> exit code). Stage 1's
# restore writes STUB_PLAN's plan (games, held out from) and labels, its
# readout with the gates of STUB_FAIL_GATES failed, and real game tarballs;
# STUB_RATES is "trainer forward decisions-a-game".
FAKE = r'''
import json, os, pathlib, shutil, sys, tarfile
name, args = sys.argv[1], sys.argv[2:]
def opt(flag):
    return args[args.index(flag) + 1]
def opts(flag):
    i, out = args.index(flag) + 1, []
    while i < len(args) and not args[i].startswith("--"):
        out.append(args[i])
        i += 1
    return out
rc = json.loads(os.environ.get("STUB_RC", "{}")).get(name, 0)
t, f, D = (float(x) for x in os.environ.get("STUB_RATES", "1000 3000 100").split())
if name == "restore stage1":
    out = pathlib.Path(opt("--out"))
    n, held = (int(x) for x in os.environ.get("STUB_PLAN", "40 30").split())
    for item in opts("--restore"):
        p = out / item
        if item == "plan.txt":
            p.write_text(f"{n} {held} 100\n")
        elif item == "labels.jsonl":
            p.write_text("".join(json.dumps({"game": f"g{i}", "side": s, "heldout": i >= held}) + "\n"
                                 for i in range(n) for s in (1, 2)))
        elif item == "readout.json":
            failed = os.environ.get("STUB_FAIL_GATES", "").split()
            p.write_text(json.dumps({k: {"pass": k not in failed}
                                     for k in ("i_sample", "legal_mass", "i_count", "dice_control", "i_v")}))
        elif item.endswith(".tar.gz"):
            d = out / item[:-len(".tar.gz")]
            d.mkdir(exist_ok=True)
            (d / "game_x.json").write_text("{}")
            with tarfile.open(p, "w:gz") as tf:
                tf.add(d, arcname=d.name)
            shutil.rmtree(d)
        else:
            p.write_text('{"game": "x"}\n')
elif rc in (0, 3) and name in ("c0 pass", "pass", "plant pass"):
    games = int(opt("--train-limit")) if "--train-limit" in args else 30
    out = pathlib.Path(opt("--out"))
    out = out if rc == 0 else out.with_name(out.stem + ".replay_failed.pt")
    out.write_text("a pass")
    out.with_suffix(".json").write_text(json.dumps({
        "decisions": games * D, "seconds": 2 * games * D / t, "games": games, "rate": {"positions_per_second": t},
        "replay": {"kl": 0.0, "limit": 1e-4}, "floor": {"kl": 1e-6, "se": 1e-7}}))
elif rc == 0 and name == "c0 forwards":
    pathlib.Path(opt("--out")).write_text(json.dumps({"positions_per_second": f, "games": int(opt("--held-games")),
                                                      "replay": {"kl": 0.0, "max_abs": 0.0}}))
elif name == "stage c":
    body = {"arms": {}} if rc else {"gate": {"verdict": os.environ.get("STUB_GATE", "resolved"), "reason": "stub"},
                                    "optimizer": {"winner": "reduced_adamw"}, "errors": {}}
    pathlib.Path(opt("--out")).write_text(json.dumps(body))
elif name == "plant readout" and rc == 0:
    pathlib.Path(opt("--out")).write_text(json.dumps({"verdict": os.environ.get("STUB_PLANT", "pass"), "seats": {},
                                                      "delta": 0.1}))
elif name.startswith("plant ") and rc in (0, 1):
    pathlib.Path(opt("--out")).write_text("{}\n")
sys.exit(rc)
'''


def _block(begin: str, end: str) -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    return text[text.index(begin):text.index(end)]


def _run(tmp_path: Path, deadline_min: float | None = None, **env) -> dict:
    """One entry; `deadline_min`: when the dead-man's switch fires, from now."""
    out, s1, checkout = tmp_path / "out", tmp_path / "stage1", tmp_path / "checkout"
    (out / "tmp").mkdir(parents=True, exist_ok=True)
    (checkout / "tools").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "tools" / "stage_c_plan.py", checkout / "tools" / "stage_c_plan.py")
    (out / "steps.txt").unlink(missing_ok=True)
    now = int(time.time())
    (tmp_path / "clock").write_text(str(now))
    if deadline_min is not None:
        (out / "tmp" / "deadman.at").write_text(str(int(now + deadline_min * 60)))
    fake = tmp_path / "fake.py"
    fake.write_text(FAKE, encoding="utf-8")
    settings = "\n".join(f"export {k}='{v}'" for k, v in env.items())
    script = f"""set -uo pipefail
export PATH="{bash_path(Path(sys.executable).parent)}:/usr/bin:$PATH"
cd "{bash_path(checkout)}" || exit 9
OUT="{bash_path(out)}" BOX_OUT="{bash_path(out)}" S1="{bash_path(s1)}" STAGE=stage_test BOX_LIB=/lib
FAKE="{bash_path(fake)}" CKPT=/ckpt/own2.pt WORKERS=3 SMALL_CUT_MIN=30 C0_CUT_MIN=40 RESTORE_CUT_MIN=60
export STUB_DIR="{bash_path(tmp_path)}"
{settings}
{_block("# ---- the configuration (begin)", "# ---- the configuration (end)")}
PARTS=(c0 pass stage_c plant)
RESTORE=()
STATE=()
{STUBS}
{_block("# ---- the plan (begin)", "# ---- the plan (end)")}
{_block("# ---- the stages (begin)", "# ---- the stages (end)")}
"""
    path = tmp_path / "driver.sh"
    path.write_bytes(script.encode())
    done = subprocess.run([BASH, bash_path(path)], cwd=tmp_path, capture_output=True, text=True, timeout=600)
    steps = (out / "steps.txt").read_text().splitlines() if (out / "steps.txt").exists() else []
    names = []
    for s in steps:
        words = s.split()
        names.append(" ".join(words[:2]) if words[0] in ("hold", "restore", "c0", "stage", "plant") else words[0])
    return {"rc": done.returncode, "stderr": done.stderr, "steps": steps, "names": names, "out": out,
            "finish": (out / "FINISH").read_text().strip() if (out / "FINISH").exists() else ""}


def _step(got: dict, name: str) -> str:
    return next(s for s in got["steps"] if s.startswith(name + " ["))


def _run_steps(got: dict) -> list:
    return [n for n in got["names"] if not n.startswith(("hold", "restore"))]


ALL = ["c0 pass", "c0 forwards", "pass", "stage c", "plant delta", "plant copy", "plant rows", "plant labels",
       "plant pass", "plant readout"]


def test_the_run_restores_measures_passes_reads_plants_and_decides(tmp_path):
    """Stage 1's 40 games (30 trained on) in 8 matches of 5: C0 at its 200
    games' limit, the pass on every training game with its cut from C0's
    trainer rate, the driver on every arm with both curve points (the
    forwards at 3x the trainer), the plant on the first 5,000 games, and the
    verdict; a re-entry runs nothing done."""
    got = _run(tmp_path, CHUNK_GAMES=5)
    assert got["rc"] == 0 and got["finish"].startswith("STAGE_C_DONE"), (got["finish"], got["stderr"][-3000:])
    assert _run_steps(got) == ALL, got["names"]
    out, s1 = bash_path(got["out"]), bash_path(tmp_path / "stage1")
    dirs = " ".join(f"{s1}/games_s{k:02d}" for k in range(8))
    assert all((tmp_path / "stage1" / f"games_s{k:02d}" / "game_x.json").is_file() for k in range(8))
    assert (tmp_path / "stage1" / "walk.jsonl").read_text().count("\n") == 8
    assert f"--games {dirs} --labels {s1}/labels.jsonl --train-limit 200 --out {out}/c0_pass.pt" in _step(got, "c0 pass")
    # 30 games x 100 decisions at 1,000 a second: 50 s, so the pass's cut is 15 minutes.
    assert _step(got, "pass").startswith("pass [15] python tools/pg_pass.py --checkpoint /ckpt/own2.pt")
    driver = _step(got, "stage c")
    assert "--arms sgd reduced_adamw muon heads_only probe_span --curve-splits 2 4 " in driver
    assert f"--factory-arg games={dirs.replace(' ', ',')} " in driver
    assert "--price-per-game 7.76e-5 --round-fixed-cost 2.42 " in driver
    assert "--train-games 5000 --class attack --readout-fraction 0.2 " in _step(got, "plant delta")
    assert "--heldout-from 30 " in _step(got, "plant labels")
    assert f"--no-walk --walk {out}/walk_planted.jsonl" in _step(got, "plant rows")
    for hold in ("hold c0.done c0_pass.json c0_pass.pt c0_forwards.json", "hold pass.done pass.pt",
                 "hold stage_c.done stage_c.json",
                 "hold plant.done plant_readout.json plant_grads.pt pass_planted.pt labels_planted.jsonl"):
        assert hold in got["steps"], hold
    verdict = json.loads((got["out"] / "verdict.json").read_text())
    assert verdict["gate"]["verdict"] == "resolved" and verdict["plant"]["verdict"] == "pass"
    # The plan reads the pass's rate past its warm-up (1,000), not its wall (500 a second).
    assert verdict["c0"]["rates"]["trainer"] == 1000 and verdict["c0"]["trim"] is False
    assert verdict["next"].startswith("P (the planted step played)")
    assert "gate=resolved plant=pass" in got["finish"]
    again = _run(tmp_path, CHUNK_GAMES=5)
    assert again["rc"] == 0 and _run_steps(again) == [], (again["steps"], again["stderr"][-2000:])


def test_forwards_at_the_bound_trim_the_arms_and_a_run_past_the_switch_stops_before_the_pass(tmp_path):
    got = _run(tmp_path / "a", STUB_RATES="1000 1200 100")
    assert got["rc"] == 0, (got["finish"], got["stderr"][-3000:])
    assert "--arms sgd reduced_adamw muon probe_span --curve-splits 4 " in _step(got, "stage c")
    slow = _run(tmp_path / "b", deadline_min=60, STUB_RATES="2 6 100")
    assert slow["rc"] == 1 and slow["finish"].startswith("C0_OVER_SWITCH"), slow["finish"]
    assert _run_steps(slow) == ["c0 pass", "c0 forwards"]


def test_a_failed_replay_stops_what_needs_it_and_stays_failed(tmp_path):
    got = _run(tmp_path, STUB_RC='{"c0 pass": 3}')
    assert got["rc"] == 1 and _run_steps(got) == ["c0 pass"]
    assert "c0: FAILED, the replay identity failed on 200 games" in got["finish"]
    again = _run(tmp_path)
    assert again["rc"] == 1 and _run_steps(again) == [] and "FAILED at an earlier entry" in again["finish"]
    full = _run(tmp_path / "pass", STUB_RC='{"pass": 3}')
    assert full["rc"] == 1 and _run_steps(full) == ["c0 pass", "c0 forwards", "pass"]
    assert "pass: FAILED, the replay identity failed on 30 games" in full["finish"]


def test_a_failed_driver_leaves_the_plant_and_a_failed_plant_leaves_the_verdict(tmp_path):
    got = _run(tmp_path / "c", STUB_RC='{"stage c": 1}')
    assert got["rc"] == 1 and _run_steps(got) == ALL, got["names"]
    assert "stage_c: FAILED, the driver ended rc=1" in got["finish"] and "plant: done" in got["finish"]
    assert "hold stage_c.done" not in " ".join(got["steps"])
    plant = _run(tmp_path / "p", STUB_RC='{"plant pass": 1, "plant labels": 1}', STUB_GATE="neither")
    assert plant["rc"] == 1 and _run_steps(plant) == ALL[:-1], plant["names"]
    assert "plant: FAILED, plant pass ended rc=1" in plant["finish"] and "stage_c: done" in plant["finish"]
    assert "gate=neither plant=None" in plant["finish"]
    missed = _run(tmp_path / "m", STUB_PLANT="fail")
    assert "the planted readout missed: debug before P and rounds" in missed["finish"]


def test_stage_1s_failed_gates_or_missing_games_stop_the_run(tmp_path):
    got = _run(tmp_path / "g", STUB_FAIL_GATES="i_count")
    assert got["rc"] == 1 and got["finish"].startswith("STAGE1_UNUSABLE: gates failed: i_count"), got["finish"]
    assert _run_steps(got) == []
