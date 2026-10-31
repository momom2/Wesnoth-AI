"""scripts/rounds_local.py: stage C's gate deciding whether rounds run and
at what size, round 1 from stage C's decision, each later round's games,
corrections, pass and step in order, each played by the policy the previous
round left, a claim every third round, and design §6's two stops (the pause
on weak confirmed readings, the claims' transfer check, both reported when
both fire; the upper tail logged). The tools are stood in for by a fake
runner that writes what each step leaves."""
from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.rounds_local as rl  # noqa: E402


def _rung(estimate: float, se: float) -> dict:
    return {"halves": {"A": {"exact": {"estimate": estimate, "se": se}},
                       "B": {"exact": {"estimate": estimate, "se": se}}}}


class FakeTools:
    """Each step's products; `adopt` per round (round 1 included), `reading`
    each adopted rung's half-B ΔĴ and SE, `claim_elo` the claim's Elo."""

    def __init__(self, adopt=None, reading=(0.02, 0.004), claim_elo=-400.0):
        self.calls, self.adopt, self.reading, self.claim_elo = [], adopt or {}, reading, claim_elo

    def __call__(self, out, name, cmd, log_name, cut, python, **_):
        self.calls.append((name, cmd, python))
        arg = lambda flag: cmd[cmd.index(flag) + 1]                          # noqa: E731
        words = name.split()
        if name.endswith(" step"):
            k = int(words[1])
            adopted = self.adopt.get(k, True)
            dec = {"adopt": adopted, "s": 0.4 if adopted else 0.0, "eps": 1e-3, "next_eps": 2e-3 if adopted else 5e-4,
                   "chosen": 0, "reason": "read"}
            if k > 1:
                dec["rungs"] = [_rung(*self.reading)]
            Path(arg("--out")).write_text(json.dumps({"decision": dec, "candidate": {"path": arg("--candidate")}
                                                      if adopted else None}), encoding="utf-8")
            if adopted:
                Path(arg("--candidate")).write_text("pi", encoding="utf-8")
        elif name.endswith(" games"):
            play = Path(arg("--out"))
            play.mkdir(parents=True, exist_ok=True)
            (play / "labels.jsonl").write_text("", encoding="utf-8")
        elif name.endswith(" corrections"):
            Path(arg("--out")).write_text("{}", encoding="utf-8")
        elif name.endswith(" pass"):
            Path(arg("--out")).write_text("pass", encoding="utf-8")
        elif name.endswith(" claim fit"):
            label = f"pi{words[1]}t1"
            Path(arg("--save-json")).write_text(json.dumps({"tables": {"PURE (decisive only, primary)": {
                label: {"elo": self.claim_elo, "se": 5.0}, "own2": {"elo": 0.0, "se": 0.0}}}}), encoding="utf-8")
        return 0

    def resets(self, t0):
        return 0

    def names(self):
        return [c[0] for c in self.calls]

    def cmd(self, name):
        return next(c for n, c, _ in self.calls if n == name)


RESOLVED = {"verdict": "resolved", "reason": "read"}


def _setup(tmp_path: Path, gate=RESOLVED, arm="reduced_adamw"):
    s1, sc = tmp_path / "stage1", tmp_path / "stage_c"
    for d in (s1 / "games_s00", sc):
        d.mkdir(parents=True, exist_ok=True)
    (s1 / "labels.jsonl").write_text("", encoding="utf-8")
    (s1 / "d0.fit.json").write_text(json.dumps({"tables": {"PURE (decisive only, primary)": {
        "own2t1": {"elo": -415.2, "se": 5.5}, "own2": {"elo": 0.0, "se": 0.0}}}}), encoding="utf-8")
    (sc / "stage_c.json").write_text(json.dumps({"optimizer": {"winner": arm}, "gate": gate,
                                                 "ladder": {"rungs": [_rung(0.02, 0.004)]}}), encoding="utf-8")
    (sc / "ALL_DONE").write_text("done\n", encoding="utf-8")
    return s1, sc


def _run(tmp_path, tools, rounds=3, monkeypatch=None, p_verdicts=("pass", "holds"), no_p_gate=False,
         gate=RESOLVED, finished=True, arm="reduced_adamw", heads_only_pass=False, end_by=None, now=None):
    s1, sc = _setup(tmp_path, gate, arm)
    if not finished:
        (sc / "ALL_DONE").unlink()
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    p_out = None
    if p_verdicts is not None:
        p_out = tmp_path / "p"
        p_out.mkdir(exist_ok=True)
        for name, verdict in zip(("p_readout.json", "link.json"), p_verdicts):
            (p_out / name).write_text(json.dumps({"verdict": verdict}), encoding="utf-8")
    args = Namespace(out=out, stage1=s1, stage_c=sc, checkpoint=Path("own2.pt"), python="gpu-py",
                     tools_python="cpu-py", workers=3, p_out=p_out, no_p_gate=no_p_gate,
                     heads_only_pass=heads_only_pass, end_by=end_by)
    clock = {} if now is None else {"now": now}
    return rl.run(rl.Rounds(args, runner=tools, resets=tools.resets, **clock), rounds), out


def _reference(monkeypatch):
    monkeypatch.setattr(rl, "reference_flags", lambda side, python: [
        f"--label-{side}", "own2", f"--spec-{side}", "own2.pt", f"--memory-{side}", "64",
        f"--raw-end-turn-offset-{side}", "-1.5"])


ORDER = ["round 1 step", "round 2 games", "round 2 corrections", "round 2 pass", "round 2 step", "round 3 games",
         "round 3 corrections", "round 3 pass", "round 3 step", "round 3 claim", "round 3 claim fit"]


def test_the_rounds_run_in_order_each_played_by_the_last_adopted_policy(tmp_path, monkeypatch):
    _reference(monkeypatch)
    tools = FakeTools(adopt={2: False})                    # round 2 keeps pi_1
    rc, out = _run(tmp_path, tools)
    assert rc == 0 and tools.names() == ORDER
    first = tools.cmd("round 1 step")
    assert first[first.index("--adopt-from") + 1].endswith("stage_c.json")
    for k in (1, 2, 3):                                  # each ladder keeps its readings, so a restart resumes
        assert f"cache={out / f'round_{k:02d}' / 'eval_cache'}" in tools.cmd(f"round {k} step")
    pi_1 = str(out / "round_01" / "pi.pt")
    assert tools.cmd("round 2 games")[tools.cmd("round 2 games").index("--checkpoint") + 1] == pi_1
    assert tools.cmd("round 3 games")[tools.cmd("round 3 games").index("--checkpoint") + 1] == pi_1
    for k in (2, 3):
        cmd = tools.cmd(f"round {k} pass")
        assert all(f in cmd for f in ("--fp16", "--packed-trunk", "--corrections")) and cmd[cmd.index("--checkpoint") + 1] == pi_1
        step = tools.cmd(f"round {k} step")
        assert step[step.index("--eps-from") + 1] == str(out / f"round_{k - 1:02d}" / "round.json")
        assert "fp16=1" in step and step[step.index("--arm-from") + 1].endswith("stage_c.json")
    games = tools.cmd("round 2 games")
    assert games[games.index("--games") + 1] == str(rl.N_TRAIN + rl.N_HELD)
    assert games[games.index("--heldout-from") + 1] == str(rl.N_TRAIN)
    assert games[games.index("--seed-base") + 1] == str(rl.SEED_BASE + 2 * rl.SEED_STRIDE)
    claim = tools.cmd("round 3 claim")
    assert claim[claim.index("--seed-base") + 1] == str(rl.CLAIM_SEED_BASE + 3 * rl.CLAIM_SEED_STRIDE)
    assert claim[claim.index("--spec-a") + 1] == str(out / "round_03" / "pi.pt")
    assert claim[claim.index("--raw-temperature-a") + 1] == "1" and "--raw-end-turn-offset-a" not in claim
    python = {n: py for n, _, py in tools.calls}
    assert {n for n, py in python.items() if py == "cpu-py"} == {"round 2 corrections", "round 3 corrections",
                                                                    "round 3 claim fit"}
    again = FakeTools()
    assert _run(tmp_path, again)[0] == 0 and again.names() == []


def test_weak_confirmed_readings_pause_the_rounds(tmp_path, monkeypatch):
    _reference(monkeypatch)
    tools = FakeTools(reading=(0.001, 0.01))
    rc, out = _run(tmp_path, tools)
    assert rc == 3 and "PAUSE after round 3" in (out / "STOP").read_text(encoding="utf-8")


def test_a_claim_far_below_its_forecast_stops_the_rounds(tmp_path, monkeypatch):
    _reference(monkeypatch)
    tools = FakeTools(reading=(0.05, 0.004), claim_elo=-420.0)        # forecast +52 Elo over D0, claim -5
    rc, out = _run(tmp_path, tools)
    stop = (out / "STOP").read_text(encoding="utf-8")
    assert rc == 3 and "gains do not transfer" in stop and "PAUSE" not in stop


def test_both_stops_are_reported_when_both_fire(tmp_path, monkeypatch):
    _reference(monkeypatch)
    tools = FakeTools(reading=(0.001, 0.01), claim_elo=-500.0)        # a weak sum, and the claim 85 Elo under D0
    rc, out = _run(tmp_path, tools)
    stop = (out / "STOP").read_text(encoding="utf-8")
    assert rc == 3 and "PAUSE after round 3" in stop and "gains do not transfer" in stop


def test_the_claims_upper_tail_is_logged_and_the_rounds_go_on(tmp_path, monkeypatch):
    _reference(monkeypatch)
    tools = FakeTools(claim_elo=-300.0)                               # +115 Elo over D0 against +21 forecast
    rc, out = _run(tmp_path, tools)
    assert rc == 0 and not (out / "STOP").exists()
    assert "UPPER_TAIL at round 3" in (out / "runner.log").read_text(encoding="utf-8")


def test_stage_cs_gate_decides_whether_rounds_run_and_at_what_size(tmp_path, monkeypatch):
    _reference(monkeypatch)
    for name, gate in (("neither", {"verdict": "neither", "reason": "unresolved"}),
                       ("unread", {"verdict": "unread", "reason": "no driver readout"}), ("missing", None),
                       ("infinite", {"verdict": "rounds_at_size", "round_size": float("inf")})):
        tools = FakeTools()
        rc, out = _run(tmp_path / name, tools, gate=gate, no_p_gate=True)        # P's waiver is not stage C's
        assert rc == 3 and tools.names() == [], name
        assert "ROUNDS_WAIT before round 1" in (out / "runner.log").read_text(encoding="utf-8")
    unfinished = FakeTools()                                # a resolved gate in an unfinished stage C
    rc, out = _run(tmp_path / "unfinished", unfinished, finished=False, no_p_gate=True)
    assert rc == 3 and unfinished.names() == []
    assert "stage C has not finished" in (out / "runner.log").read_text(encoding="utf-8")
    sized = FakeTools()
    rc, _ = _run(tmp_path / "sized", sized, gate={"verdict": "rounds_at_size", "round_size": 7342.6})
    games = sized.cmd("round 2 games")
    assert rc == 0 and games[games.index("--games") + 1] == str(7343 + rl.N_HELD)
    assert games[games.index("--heldout-from") + 1] == "7343"


def test_a_failed_part_stops_the_chain_and_stays_failed(tmp_path, monkeypatch):
    _reference(monkeypatch)

    class Failing(FakeTools):
        def __call__(self, out, name, cmd, log_name, cut, python, **kw):
            if name == "round 2 pass":
                self.calls.append((name, cmd, python))
                return 3
            return super().__call__(out, name, cmd, log_name, cut, python, **kw)

    tools = Failing()
    rc, out = _run(tmp_path, tools)
    assert rc == 1 and tools.names() == ORDER[:4]
    assert "replay identity" in (out / "r02_pass.failed").read_text(encoding="utf-8")
    again = FakeTools()
    assert _run(tmp_path, again)[0] == 1 and again.names() == []


def test_round_2_waits_on_stage_p_unless_waived(tmp_path, monkeypatch):
    _reference(monkeypatch)
    tools = FakeTools()
    rc, out = _run(tmp_path / "miss", tools, p_verdicts=("fail", "holds"))
    assert rc == 3 and tools.names() == ["round 1 step"]
    waived = FakeTools()
    rc, _ = _run(tmp_path / "waived", waived, p_verdicts=None, no_p_gate=True)
    assert rc == 0 and waived.names() == ORDER
    unnamed = FakeTools()
    assert _run(tmp_path / "unnamed", unnamed, p_verdicts=None)[0] == 3 and unnamed.names() == ["round 1 step"]
    unlinked = FakeTools()                                   # the link not run: the slope under 3 SE
    rc, _ = _run(tmp_path / "unlinked", unlinked, p_verdicts=("pass", "not run"))
    assert rc == 0 and unlinked.names() == ORDER


def test_the_heads_only_pass_serves_only_an_arm_that_reads_the_heads_alone(tmp_path, monkeypatch):
    from wesnoth_ai.pg_step import HEADS_ONLY_ARMS
    assert rl.HEADS_ONLY_ARMS == HEADS_ONLY_ARMS
    _reference(monkeypatch)
    for arm, asked, heads in (("heads_only", True, True), ("probe_span", True, True),
                              ("reduced_adamw", True, False), ("heads_only", False, False)):
        tools = FakeTools()
        rc, _ = _run(tmp_path / f"{arm}_{asked}", tools, arm=arm, heads_only_pass=asked)
        assert rc == 0, (arm, asked)
        for k in (2, 3):
            assert ("--heads-only" in tools.cmd(f"round {k} pass")) is heads, (arm, asked, k)


class Clock:
    """The rounds' clock: each pass moves it on by `pass_h` hours, and each
    call of the runner goes through it."""

    def __init__(self, tools, start: str, pass_h: float):
        from datetime import datetime
        self.tools, self.t = tools, datetime.strptime(start, rl.END_BY_FORMAT).timestamp()
        self.pass_s = pass_h * 3600

    def now(self) -> float:
        return self.t

    def __call__(self, out, name, *a, **k):
        if name.endswith(" pass"):
            self.t += self.pass_s
        return self.tools(out, name, *a, **k)

    def resets(self, t0):
        return 0


def test_end_by_starts_a_round_only_when_the_longest_round_fits(tmp_path, monkeypatch):
    """Rounds of 4 h from Nov 2 00:00: round k >= 3 starts at 4 (k - 2) h
    and needs 1.2 x 4 = 4.8 h before --end-by."""
    _reference(monkeypatch)
    for end_by, last in (("2026-11-02 20:00", 5), ("2026-11-02 08:00", 2), ("2026-11-02 13:00", 4), (None, 6)):
        tools = FakeTools(claim_elo=-380.0)
        clock = Clock(tools, "2026-11-02 00:00", 4.0)
        rc, out = _run(tmp_path / str(end_by).replace(":", "").replace(" ", "_"), clock, rounds=6, end_by=end_by,
                       now=clock.now)
        assert rc == 0, end_by
        assert max(int(n.split()[1]) for n in tools.names()) == last, end_by
        log = (out / "runner.log").read_text(encoding="utf-8")
        assert ("past --end-by" in log) is (last < 6), end_by
