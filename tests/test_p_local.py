"""scripts/p_local.py: the planted step's match and readout, then the link
match and its readout (only when stage 1's sharpening slope is resolved),
run only when stage C's verdict says P; a planted miss or a link miss stops
the rounds; a new start resumes; each match's time budget fits its cut,
and a match that ends short is played again once. The tools are stood in
for by a fake runner."""
from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.p_local as pl  # noqa: E402
from scripts.local_machine import MATCH_DRAIN_MIN  # noqa: E402

ORDER = ["p_match", "p_readout", "link_match", "link_readout"]


class FakeTools:
    """`rcs`: per step name, the exit statuses its first calls return (0
    after them)."""

    def __init__(self, planted="pass", link="holds", rcs=None):
        self.calls, self.planted, self.link = [], planted, link
        self.rcs = {k: list(v) for k, v in (rcs or {}).items()}

    def __call__(self, out, name, cmd, log_name, cut, python, **_):
        self.calls.append((name, cmd, python))
        target = Path(cmd[cmd.index("--out") + 1]) if "--out" in cmd else None
        if name == "p_readout":
            target.write_text(json.dumps({"verdict": self.planted, "seats": {}}), encoding="utf-8")
        elif name == "link_readout":
            target.write_text(json.dumps({"verdict": self.link, "measured": 0.1, "forecast": 0.106,
                                          "miss_se": -0.5}), encoding="utf-8")
        left = self.rcs.get(name.removesuffix(" again"), [])
        return left.pop(0) if left else 0

    def resets(self, t0):
        return 0

    def names(self):
        return [c[0] for c in self.calls]


def _run(tmp_path, tools, nxt="P (the planted step played), then rounds", resolved=True, spoil=None):
    sc = tmp_path / "stage_c"
    sc.mkdir(parents=True, exist_ok=True)
    (tmp_path / "stage1").mkdir(exist_ok=True)
    (tmp_path / "stage1" / "readout.json").write_text(json.dumps({"sharpening": {
        "slope": 2.64, "se": 0.08 if resolved else 1.5, "resolved": resolved}}), encoding="utf-8")
    (sc / "verdict.json").write_text(json.dumps({"next": nxt}), encoding="utf-8")
    (sc / "planted.pt").write_text("theta'", encoding="utf-8")
    (sc / "ALL_DONE").write_text("done\n", encoding="utf-8")
    if spoil:
        spoil(sc)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    args = Namespace(out=out, stage1=tmp_path / "stage1", stage_c=sc, checkpoint=Path("own2.pt"), e=0.02,
                     python="gpu-py", tools_python="cpu-py")
    return pl.run(pl.P(args, runner=tools, resets=tools.resets)), out


def test_p_and_the_link_run_in_order_and_a_new_start_runs_nothing(tmp_path):
    tools = FakeTools()
    rc, out = _run(tmp_path, tools)
    assert rc == 0 and tools.names() == ORDER
    cmds = {n: c for n, c, _ in tools.calls}
    p = cmds["p_match"]
    assert p[p.index("--spec-a") + 1].endswith("planted.pt") and "--log-decisions" in p
    assert p[p.index("--games") + 1] == str(pl.P_GAMES)
    link = cmds["link_match"]
    assert float(link[link.index("--raw-temperature-a") + 1]) == 1 / 1.02
    assert float(link[link.index("--raw-temperature-b") + 1]) == 1 / 0.98 and "--log-decisions" not in link
    assert {n for n, _, py in tools.calls if py == "cpu-py"} == {"p_readout", "link_readout"}
    again = FakeTools()
    assert _run(tmp_path, again)[0] == 0 and again.names() == []


def test_a_planted_miss_or_a_link_miss_stops_the_rounds(tmp_path):
    rc, out = _run(tmp_path / "p", FakeTools(planted="fail"))
    assert rc == 3 and "P_MISS" in (out / "STOP").read_text(encoding="utf-8")
    rc, out = _run(tmp_path / "l", FakeTools(link="misses: first-order labels do not price play"))
    assert rc == 3 and "LINK_MISS" in (out / "STOP").read_text(encoding="utf-8")


def test_p_waits_for_stage_cs_verdict(tmp_path):
    tools = FakeTools()
    rc, _ = _run(tmp_path, tools, nxt="no rounds: the user chooses the $1,000 case")
    assert rc == 3 and tools.names() == []
    sized = FakeTools()
    rc, _ = _run(tmp_path / "sized", sized,
                 nxt="P (the planted step played), then rounds at the size the gain curve names")
    assert rc == 0 and sized.names() == ORDER


def test_an_unresolved_slope_plays_no_link_match(tmp_path):
    tools = FakeTools()
    rc, out = _run(tmp_path, tools, resolved=False)
    assert rc == 0 and tools.names() == ORDER[:2]
    assert json.loads((out / "link.json").read_text(encoding="utf-8"))["verdict"] == "not run"


def test_p_waits_for_stage_c_to_finish(tmp_path):
    """Stage C writes its verdict even when it stops early: P starts only on
    a finished stage C (ALL_DONE, no STOP, no failed part)."""
    for name, spoil in (("unfinished", lambda sc: (sc / "ALL_DONE").unlink()),
                        ("stopped", lambda sc: (sc / "STOP").write_text("x", encoding="utf-8")),
                        ("failed", lambda sc: (sc / "plant.failed").write_text("x", encoding="utf-8"))):
        tools = FakeTools()
        rc, out = _run(tmp_path / name, tools, spoil=spoil)
        assert rc == 3 and tools.names() == [], name
        assert "P_NOT_DUE" in (out / "runner.log").read_text(encoding="utf-8"), name


def test_a_match_has_a_time_budget_within_its_cut_and_is_played_again_once_when_short(tmp_path):
    """run_elo_batch's default budget, 55 min, would stop P's 5,000 games
    short (exit 3) and fail P: each match's budget is its cut less
    MATCH_DRAIN_MIN, and a match that ends short is played again once; a
    second short end fails the part."""
    tools = FakeTools(rcs={"p_match": [3]})
    rc, out = _run(tmp_path / "again", tools)
    assert rc == 0 and tools.names() == ["p_match", "p_match again", *ORDER[1:]]
    for name, cmd, _ in tools.calls:
        if "match" in name:
            assert cmd[cmd.index("--time-budget-min") + 1] == f"{pl.MATCH_CUT_MIN - MATCH_DRAIN_MIN:g}", name
    assert "p_match: ended rc=3, short" in (out / "runner.log").read_text(encoding="utf-8")
    short = FakeTools(rcs={"link_match": [3, 1]})
    rc, out = _run(tmp_path / "short", short)
    assert rc == 1 and short.names() == [*ORDER[:3], "link_match again"]
    assert "rc=1" in (out / "link_match.failed").read_text(encoding="utf-8")
