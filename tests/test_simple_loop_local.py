"""scripts/simple_loop_local.py, the simple training loop's runner: what a
CPU can check. Every tool is faked (it writes the files the runner reads);
the runner's own order, markers, pool, matches, kill criterion, warning
and pruning are real."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.local_machine as lm  # noqa: E402
import scripts.simple_loop_local as sl  # noqa: E402

GAMES, PAST = 16, 4


class FakeTools:
    """run_tool's stand-in: each tool writes what the runner reads of it.
    `elo(step)` gives a match's (Elo, standard error), `norm(round)` a
    pass's gradient norm, and `fail` maps a step's name to the exit status
    it ends with instead."""

    def __init__(self, elo=None, norm=None, fail=None):
        self.calls = []
        self.elo = elo or (lambda step: (0.0, 10.0))
        self.norm = norm or (lambda k: 1.0)
        self.fail = dict(fail or {})

    def tools(self):
        return [cmd[0] for _, cmd in self.calls]

    def __call__(self, out, name, cmd, log_name, timeout_min, python, **kw):
        self.calls.append((name, list(cmd)))
        if name in self.fail:
            return self.fail.pop(name)

        def arg(flag):
            return cmd[cmd.index(flag) + 1]
        tool = cmd[0]
        if tool == "tools/run_elo_batch.py":
            d = Path(arg("--outdir"))
            d.mkdir(parents=True, exist_ok=True)
            for i in range(int(arg("--games"))):
                (d / f"game_x_{int(arg('--seed-base')) + i}.json").write_text("{}", encoding="utf-8")
            return 4
        if tool == "tools/pg_game_pass.py":
            grad = Path(arg("--out"))
            k = int(grad.parent.name.split("_")[1])
            grad.write_bytes(b"grad")
            lam = float(arg("--lam"))
            grad.with_suffix(".json").write_text(json.dumps({
                "game_sides": 2 * GAMES - PAST, "skipped_sides": PAST, "decisions": 4000, "norm": self.norm(k),
                "replay": {"kl": 1e-6}, "value": {"cross_entropy": 0.4, "squared_error": 0.5},
                "weight": {"lam": lam, "mean_square": 0.8 * lam, "mean_square_at_1": 0.8},
                "rate": {"decisions_per_second": 80.0}, "peak_memory_mb": 2000.0,
                "loss_scale": {"overflows": 0}, "groups": [{"scale": 1024.0}]}), encoding="utf-8")
            return 0
        if tool == "tools/pg_adam_step.py":
            pi = Path(arg("--out"))
            pi.write_text(f"network after {pi.parent.name}", encoding="utf-8")
            Path(arg("--state-out")).write_bytes(b"adam")
            pi.with_name("pi.step.json").write_text(json.dumps({"update_norm": 0.1}), encoding="utf-8")
            return 0
        if tool == "scripts/match_local.py":
            d = Path(arg("--out"))
            d.mkdir(parents=True, exist_ok=True)
            elo, se = self.elo(int(d.name.split("_")[1]))
            (d / "result.json").write_text(json.dumps({"elo": elo, "se": se, "complete": True, "decisive": 800,
                                                       "wins": 400}), encoding="utf-8")
            return 0
        raise AssertionError(f"an unexpected tool: {cmd[:2]}")


def _args(tmp_path, *extra):
    start = tmp_path / "start.pt"
    if not start.exists():
        start.write_text("the start network", encoding="utf-8")
    args = sl.parse_args(["--out", str(tmp_path / "run"), "--start", str(start), "--lr", "1e-5",
                          "--games", str(GAMES), "--past-games", str(PAST), "--match-every", "2",
                          "--python", "gpu-python", "--tools-python", "cpu-python", *extra])
    args.out.mkdir(parents=True, exist_ok=True)
    return args


def _loop(tmp_path, fake, *extra):
    return sl.Loop(_args(tmp_path, *extra), runner=fake, resets=lambda t0: 0)


def _flag(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_a_round_plays_passes_and_steps_and_a_match_reads_every_second(tmp_path):
    """The first match reads the start network. A round plays its games
    against itself and against a past self, passes over both and steps from
    the network before it; the optimizer's state is carried from the second
    round on. A matched network joins the pool."""
    fake = FakeTools()
    loop = _loop(tmp_path, fake, "--max-rounds", "4")
    assert loop.run() == 0
    out = loop.out
    a_round = ["tools/run_elo_batch.py", "tools/run_elo_batch.py", "tools/pg_game_pass.py", "tools/pg_adam_step.py"]
    match = ["scripts/match_local.py"]
    assert fake.tools() == match + a_round * 2 + match + a_round * 2 + match
    assert [m["step"] for m in loop.pool()] == [0, 2, 4] and [m["step"] for m in loop.matches()] == [0, 2, 4]
    assert [p["round"] for p in loop.progress()] == [1, 2, 3, 4]

    by_name = dict(fake.calls)
    own, past = by_name["round 1 games (self)"], by_name["round 1 games (past)"]
    start = str(sl.checkpoint_of(out, 0))
    assert _flag(own, "--spec-a") == _flag(own, "--spec-b") == _flag(past, "--spec-a") == start
    assert (_flag(own, "--games"), _flag(past, "--games")) == (str(GAMES - PAST), str(PAST))
    assert _flag(own, "--label-a") == "s0000" and _flag(own, "--label-b") == "s0000b"
    assert _flag(past, "--label-b") == "p0000" and _flag(past, "--spec-b") == start
    for cmd in (own, past):
        line = " ".join(cmd)
        assert "--raw-temperature-a 1 --raw-temperature-b 1 --log-decisions" in line
        assert "--max-extra-games 0" in line and "--raw-end-turn-offset" not in line
        assert f"--memory-a {sl.SLOTS}" in line and f"--memory-b {sl.SLOTS}" in line
    seeds = [int(_flag(cmd, "--seed-base")) for name, cmd in fake.calls if "games" in name]
    assert len(set(seeds)) == len(seeds) == 8 and min(seeds) >= sl.SEED_BASE

    the_pass = by_name["round 3 pass"]
    assert _flag(the_pass, "--checkpoint") == str(sl.checkpoint_of(out, 2))
    games = the_pass[the_pass.index("--games") + 1:the_pass.index("--out")]
    assert games == [str(sl.round_dir(out, 3) / "games_self"), str(sl.round_dir(out, 3) / "games_past")]
    assert "--fp16" in the_pass and "--packed-trunk" in the_pass and _flag(the_pass, "--lam") == "1.0"
    assert [(p["lam"], p["weight_mean_square"]) for p in loop.progress()] == [(1.0, 0.8)] * 4
    assert "--state" not in by_name["round 1 step"]
    step = by_name["round 3 step"]
    assert _flag(step, "--state") == str(sl.round_dir(out, 2) / "adam.pt") and _flag(step, "--lr") == "1e-05"
    assert _flag(step, "--checkpoint") == str(sl.checkpoint_of(out, 2))
    a_match = by_name["match at step 2"]
    assert "--sampled-candidate" in a_match and _flag(a_match, "--checkpoint") == str(sl.checkpoint_of(out, 2))
    past_of_round_4 = json.loads((sl.round_dir(out, 4) / "past.json").read_text(encoding="utf-8"))
    assert past_of_round_4["step"] in (0, 2)


def test_the_weights_lambda_reaches_the_pass_and_a_round_records_the_one_its_pass_used(tmp_path):
    """--lam goes to the pass. A round's line carries the lambda its pass
    recorded: a pass run before a stop keeps the one it used when the step
    after it runs under another."""
    class StopAfterPass(FakeTools):
        def __call__(self, out, name, cmd, *rest, **kw):
            rc = super().__call__(out, name, cmd, *rest, **kw)
            if name == "round 2 pass":
                (Path(out) / "STOP").write_text("asked\n", encoding="utf-8")
            return rc
    fake = StopAfterPass()
    loop = _loop(tmp_path, fake, "--max-rounds", "3")
    with pytest.raises(lm.Stopped):
        loop.run()
    (loop.out / "STOP").unlink()
    again = FakeTools()
    assert _loop(tmp_path, again, "--max-rounds", "3", "--lam", "0.99").run() == 0
    assert [name for name, _ in again.calls][0] == "round 2 step"
    assert _flag(dict(again.calls)["round 3 pass"], "--lam") == "0.99"
    progress = loop.progress()
    assert [p["lam"] for p in progress] == [1.0, 1.0, 0.99]
    assert progress[2]["weight_mean_square"] == pytest.approx(0.8 * 0.99) and progress[2]["weight_mean_square_at_1"] == 0.8
    with pytest.raises(SystemExit):
        _args(tmp_path, "--lam", "1.5")


def test_a_failed_pass_stops_the_run_and_the_same_command_resumes_after_it(tmp_path):
    """A pass that an exact check refuses writes STOP and ends the run. With
    STOP gone, the run goes on at that pass: no game is played again and no
    finished round repeated."""
    fake = FakeTools(fail={"round 2 pass": sl.PASS_REFUSED})
    loop = _loop(tmp_path, fake, "--max-rounds", "3")
    with pytest.raises(sl.RunFailed, match="exact check"):
        loop.run()
    stop = loop.out / "STOP"
    assert "round 2's pass" in stop.read_text(encoding="utf-8")
    with pytest.raises(lm.Stopped):
        _loop(tmp_path, FakeTools(), "--max-rounds", "3").run()
    stop.unlink()
    again = FakeTools()
    assert _loop(tmp_path, again, "--max-rounds", "3").run() == 0
    names = [name for name, _ in again.calls]
    assert names[:3] == ["round 2 pass", "round 2 step", "match at step 2"]
    assert not any(n.startswith("round 1") or n == "match at step 0" for n in names)
    assert [p["round"] for p in _loop(tmp_path, FakeTools()).progress()] == [1, 2, 3]


def test_two_matches_in_a_row_far_below_the_first_end_the_run(tmp_path):
    """One low match does not stop the run; the second in a row does, with
    the reason in STOP."""
    readings = {0: (0.0, 10.0), 2: (-60.0, 10.0), 4: (-10.0, 10.0), 6: (-70.0, 10.0), 8: (-80.0, 10.0)}
    fake = FakeTools(elo=lambda step: readings[step])
    loop = _loop(tmp_path, fake, "--max-rounds", "20")
    with pytest.raises(SystemExit) as stopped:
        loop.run()
    assert stopped.value.code == sl.EXIT_KILL
    assert [m["step"] for m in loop.matches()] == [0, 2, 4, 6, 8]
    assert "kill criterion" in (loop.out / "STOP").read_text(encoding="utf-8")
    first = {"step": 0, "elo": 0.0, "se": 10.0}
    low, fine = {"step": 2, "elo": -43.0, "se": 10.0}, {"step": 4, "elo": -42.0, "se": 10.0}
    assert sl.below_first(low, first) and not sl.below_first(fine, first)       # 3 SE of the difference: 42.4
    assert sl.kill_reason([first, low]) is None and sl.kill_reason([first, low, fine]) is None
    assert sl.kill_reason([first, fine, low, low]) is not None


def test_an_early_match_reads_one_more_round(tmp_path):
    loop = _loop(tmp_path, FakeTools(), "--max-rounds", "4", "--early-match", "1")
    assert loop.run() == 0
    assert [m["step"] for m in loop.matches()] == [0, 1, 2, 4] == [m["step"] for m in loop.pool()]


def test_a_match_the_run_stopped_before_is_played_at_the_next_start(tmp_path):
    """A STOP written while round 2's step ran leaves its match unplayed;
    the next start plays it before round 3."""
    class StopAfter(FakeTools):
        def __call__(self, out, name, cmd, *rest, **kw):
            rc = super().__call__(out, name, cmd, *rest, **kw)
            if name == "round 2 step":
                (Path(out) / "STOP").write_text("asked\n", encoding="utf-8")
            return rc
    loop = _loop(tmp_path, StopAfter(), "--max-rounds", "3")
    with pytest.raises(lm.Stopped):
        loop.run()
    assert [m["step"] for m in loop.matches()] == [0]
    (loop.out / "STOP").unlink()
    again = FakeTools()
    assert _loop(tmp_path, again, "--max-rounds", "3").run() == 0
    assert [name for name, _ in again.calls][0] == "match at step 2"
    assert [m["step"] for m in loop.matches()] == [0, 2]


def test_a_blown_up_gradient_is_warned_of_and_stops_nothing(tmp_path):
    """From the sixth round on, a gradient norm past ten times the median of
    the rounds before it is logged in warnings.log; the run goes on."""
    fake = FakeTools(norm=lambda k: 25.0 if k in (3, 7) else 2.0)
    loop = _loop(tmp_path, fake, "--max-rounds", "8")
    assert loop.run() == 0
    warnings = (loop.out / "warnings.log").read_text(encoding="utf-8").splitlines()
    assert len(warnings) == 1 and "round 7" in warnings[0] and "12.5 times the median" in warnings[0]
    assert [bool(p["warning"]) for p in loop.progress()] == [False] * 6 + [True, False]
    assert sl.blowup(100.0, [1.0] * (sl.BLOWUP_MIN_ROUNDS - 1)) is None
    assert sl.blowup(10.0, [1.0] * sl.BLOWUP_MIN_ROUNDS) is None


def test_a_finished_round_keeps_the_pool_and_the_latest_checkpoints_only(tmp_path):
    """After round 5 with a match every second round: the pool's networks
    (0, 2, 4) and the latest two (4, 5) are kept, the others' checkpoints
    are gone, the optimizer's state is the latest round's alone and no
    gradient file is left."""
    loop = _loop(tmp_path, FakeTools(), "--max-rounds", "5")
    assert loop.run() == 0
    kept = sorted(int(p.parent.name.split("_")[1]) for p in loop.out.glob("round_*/pi.pt"))
    assert kept == [0, 2, 4, 5]
    assert [p.parent.name for p in loop.out.glob("round_*/adam.pt")] == ["round_0005"]
    assert not list(loop.out.glob("round_*/grad.pt")) and len(list(loop.out.glob("round_*/grad.json"))) == 5
    assert sl.prunable(5, [0, 2, 4]) == [1, 3] and sl.prunable(1, [0]) == []


def test_the_past_self_is_a_pure_draw_that_covers_the_pool():
    pool = [{"step": s, "checkpoint": f"c{s}", "sha256": ""} for s in (0, 24, 48)]
    draws = [sl.draw_past(pool, 7, k)["step"] for k in range(1, 200)]
    assert draws == [sl.draw_past(list(reversed(pool)), 7, k)["step"] for k in range(1, 200)]
    assert set(draws) == {0, 24, 48} and min(draws.count(s) for s in (0, 24, 48)) > 40
    assert draws != [sl.draw_past(pool, 8, k)["step"] for k in range(1, 200)]
    with pytest.raises(ValueError):
        sl.draw_past([], 7, 1)


def test_a_round_short_of_its_games_is_played_again_once_then_refused(tmp_path):
    """A set of games that ends short is played again once from the games on
    disk; more than LOST_MAX still missing ends the run."""
    class Short(FakeTools):
        def __call__(self, out, name, cmd, *rest, **kw):
            if cmd[0] == "tools/run_elo_batch.py" and "(self)" in name:
                self.calls.append((name, list(cmd)))
                Path(cmd[cmd.index("--outdir") + 1]).mkdir(parents=True, exist_ok=True)
                return 1
            return super().__call__(out, name, cmd, *rest, **kw)
    fake = Short()
    with pytest.raises(sl.RunFailed, match="games missing"):
        _loop(tmp_path, fake, "--max-rounds", "1").run()
    assert [n for n, _ in fake.calls if "games" in n] == ["round 1 games (self)", "round 1 games (self) again"]


def test_another_configuration_is_refused_and_the_learning_rate_and_lambda_may_change(tmp_path, monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(sl, "below_normal_priority", lambda: None)
    monkeypatch.setattr(sl, "Loop", lambda args: type("L", (), {"run": lambda self: 0})())
    start = tmp_path / "start.pt"
    start.write_text("the start network", encoding="utf-8")
    base = ["--out", str(tmp_path / "run"), "--start", str(start)]
    assert sl.main(base + ["--lr", "1e-5"]) == 0
    assert sl.main(base + ["--lr", "3e-5", "--max-rounds", "10"]) == 0
    assert sl.main(base + ["--lr", "1e-5", "--lam", "0.99"]) == 0
    with pytest.raises(SystemExit, match="CONFIG_DIFFERS.*past_games"):
        sl.main(base + ["--lr", "1e-5", "--past-games", "0"])
    (tmp_path / "run" / "STOP").write_text("asked\n", encoding="utf-8")
    assert sl.main(base + ["--lr", "1e-5"]) == sl.EXIT_STOP


def _declared(tool: str) -> set:
    src = (ROOT / tool).read_text(encoding="utf-8")
    flags = set(re.findall(r"add_argument\(\s*f?\"(--[a-z0-9-]+)\"", src))
    for side in ("a", "b"):
        flags |= {f.replace("{side}", side) for f in re.findall(r"add_argument\(\s*f\"(--[a-z0-9{}-]+)\"", src)}
    return flags


def test_every_flag_the_loop_passes_is_one_its_tool_declares(tmp_path):
    fake = FakeTools()
    assert _loop(tmp_path, fake, "--max-rounds", "2", "--yield-gpu", "20").run() == 0
    seen = set()
    for _, cmd in fake.calls:
        seen.add(cmd[0])
        declared = _declared(cmd[0])
        for tok in cmd[1:]:
            if tok.startswith("--"):
                assert tok in declared or (tok.startswith("--no-") and "--" + tok[5:] in declared), (cmd[0], tok)
    assert seen == {"tools/run_elo_batch.py", "tools/pg_game_pass.py", "tools/pg_adam_step.py",
                    "scripts/match_local.py"}


def test_the_driver_imports_no_torch():
    code = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); import scripts.simple_loop_local; "
            f"print('torch' in sys.modules)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip() == "False"
