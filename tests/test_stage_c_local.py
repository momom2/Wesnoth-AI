"""scripts/stage_c_local.py: stage C's parts in the box script's order on a
local stage 1, with the RX 6600's fp16 flags; the box script's failure rules
(a failed C0 stops everything, a failed driver leaves the plant); a display
reset or a STOP file ends the run and a new start resumes; stage 1's failed
gates refuse it. The tools are stood in for by a fake runner that writes
what each step leaves."""
from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.stage_c_local as sc  # noqa: E402

ORDER = ["corrections", "c0 pass", "c0 forwards", "plan", "pass", "stage c", "plant delta", "plant copy",
         "plant rows", "plant labels", "plant pass", "plant readout"]


def _stage1(root: Path, gates_pass: bool = True) -> Path:
    s1 = root / "stage1"
    for c in range(6):
        (s1 / f"games_s{c:02d}").mkdir(parents=True, exist_ok=True)
        (s1 / f"walk_s{c:02d}.jsonl").write_text(json.dumps({"game": f"g{c}"}) + "\n", encoding="utf-8")
    (s1 / "labels.jsonl").write_text("", encoding="utf-8")
    (s1 / "fits.json").write_text("{}", encoding="utf-8")
    readout = {k: {"pass": gates_pass or k != "i_v"} for k in sc.GATES}
    (s1 / "readout.json").write_text(json.dumps(readout), encoding="utf-8")
    return s1


class FakeTools:
    """Each step's products, its command and interpreter recorded; `rc`
    sets a step's exit; `reset_at` names the step during which the display
    resets."""

    def __init__(self, rc=None, reset_at=None):
        self.calls, self.rc, self.reset_at = [], dict(rc or {}), reset_at
        self.current = None

    def __call__(self, out, name, cmd, log_name, cut, python, **_):
        self.calls.append((name, cmd, python))
        self.current = name
        rc = self.rc.get(name, 0)
        if rc:
            return rc
        product = {"corrections": ("corrections.json", {"alpha": 0.83, "gamma": -0.88, "beta": [0.0] * 7}),
                   "c0 pass": ("c0_pass.json", {"replay": {"limit": 1.1e-4}}),
                   "c0 forwards": ("c0_forwards.json", {"positions_per_second": 500}),
                   "plan": ("c0_plan.json", {"trim": False, "total_minutes": 600.0,
                                             "cuts": {"pass": 900, "stage_c": 1000, "plant": 500}}),
                   "pass": ("pass.pt", {}),
                   "stage c": ("stage_c.json", {"gate": {"verdict": "resolved"}, "optimizer": {}, "errors": {}}),
                   "plant delta": ("plant_grads.pt", {}), "plant copy": ("walk_planted.jsonl", {}),
                   "plant pass": ("pass_planted.pt", {}),
                   "plant readout": ("plant_readout.json", {"verdict": "pass", "seats": {}, "delta": 0.1})}.get(name)
        if product:
            (out / product[0]).write_text(json.dumps(product[1]), encoding="utf-8")
        return 0

    def resets(self, t0):
        return 1 if self.current == self.reset_at else 0

    def names(self):
        return [c[0] for c in self.calls]


def _run(tmp_path, tools, s1=None):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    args = Namespace(out=out, stage1=s1 or _stage1(tmp_path), checkpoint=Path("own2.pt"), python="gpu-py",
                     tools_python="cpu-py", workers=3)
    return sc.run(sc.Run(args, runner=tools, resets=tools.resets)), out


def test_the_parts_run_in_order_with_the_fp16_flags_and_a_new_start_runs_nothing(tmp_path):
    tools = FakeTools()
    rc, out = _run(tmp_path, tools)
    assert rc == 0 and (out / "ALL_DONE").exists()
    assert tools.names() == ORDER
    cmds = {name: cmd for name, cmd, _ in tools.calls}
    for name in ("c0 pass", "pass", "plant pass"):
        assert all(f in cmds[name] for f in ("--fp16", "--packed-trunk", "--floor-chunk", "--corrections")), name
    assert "--floor-sides" not in cmds["c0 pass"] and "--fresh" in cmds["c0 pass"]
    assert cmds["pass"][cmds["pass"].index("--floor-sides") + 1] == str(sc.FLOOR_SIDES)
    for name in ("c0 forwards", "plant delta", "plant readout"):
        assert "--fp16" in cmds[name], name
    assert "fp16=1" in cmds["stage c"] and "heads_only" in cmds["stage c"]
    assert cmds["c0 forwards"][cmds["c0 forwards"].index("--replay-kl-limit") + 1] == repr(1.1e-4)
    python = {name: py for name, _, py in tools.calls}
    assert {n for n, py in python.items() if py == "cpu-py"} == {"corrections", "plan", "plant copy", "plant rows",
                                                                         "plant labels"}
    assert (out / "walk.jsonl").read_text(encoding="utf-8").count("\n") == 6
    verdict = json.loads((out / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["next"].startswith("P (the planted step played)")
    again = FakeTools()
    assert _run(tmp_path, again)[0] == 0 and again.names() == []


def test_a_failed_c0_stops_everything_and_stays_failed(tmp_path):
    tools = FakeTools(rc={"c0 pass": 3})
    rc, out = _run(tmp_path, tools)
    assert rc == 1 and tools.names() == ["corrections", "c0 pass"]
    assert "replay identity failed" in (out / "c0.failed").read_text(encoding="utf-8")
    again = FakeTools()
    assert _run(tmp_path, again)[0] == 1 and again.names() == []


def test_a_failed_driver_leaves_the_plant_to_run(tmp_path):
    tools = FakeTools(rc={"stage c": 1})
    rc, out = _run(tmp_path, tools)
    assert rc == 1 and tools.names() == ORDER
    assert (out / "stage_c.failed").exists() and (out / "plant.done").exists()
    assert "did not read its gate" in json.loads((out / "verdict.json").read_text(encoding="utf-8"))["next"]


def test_a_display_reset_stops_the_run_and_a_new_start_resumes_past_what_landed(tmp_path):
    tools = FakeTools(reset_at="pass")
    rc, out = _run(tmp_path, tools)
    assert rc == 2 and tools.names() == ORDER[:5]
    assert "display-driver reset" in (out / "STOP").read_text(encoding="utf-8")
    blocked = FakeTools()
    assert _run(tmp_path, blocked)[0] == 2 and blocked.names() == []          # STOP holds until removed
    (out / "STOP").unlink()
    resumed = FakeTools()
    assert _run(tmp_path, resumed)[0] == 0
    assert resumed.names() == ORDER[4:]                                 # the pass again, nothing before it


def test_stage_1s_failed_gates_refuse_the_run(tmp_path):
    tools = FakeTools()
    rc, _ = _run(tmp_path, tools, s1=_stage1(tmp_path, gates_pass=False))
    assert rc == 1 and tools.names() == []
