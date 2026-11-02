"""scripts/chain_local.py: stage C's end chained to P and the rounds, on
fakes (no process is started, nothing sleeps)."""
import json
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

import scripts.chain_local as cl

HEAD = "a" * 40


def _args(tmp_path, **kw):
    sc = tmp_path / "stage_c"
    sc.mkdir(parents=True, exist_ok=True)
    base = dict(stage_c=sc, stage1=tmp_path / "stage1", p_out=tmp_path / "p", rounds_out=tmp_path / "rounds",
                e=0.02, rounds=3, checkpoint=Path("own2.pt"), checkpoint_sha256="cafe", expect_head=HEAD[:8],
                gpu_python="gpu-py", tools_python="cpu-py", hip_visible_devices="1", log=None, final_out=None,
                heads_only_pass=False, rounds_end_by=None, rounds_n_train=None,
                final_games=cl.FINAL_GAMES, final_seed_base=cl.FINAL_SEED_BASE, tail_out=None,
                tail_games=cl.TAIL_GAMES, tail_seed_base=cl.TAIL_SEED_BASE, continue_rounds=None,
                continue_end_by=None, continue_final_out=None, skip_p=False)
    base.update(kw)
    return Namespace(**base)


class Fakes:
    """The runner's liveness as a list of answers (then False), the steps'
    exit codes by script, and what ran."""

    def __init__(self, sc: Path, alive=(), finish_at=None, rc=None, head=(HEAD, False), sha="cafe",
                 rounds_writes=None, match_rcs=()):
        self.sc, self.answers, self.finish_at = sc, list(alive), finish_at
        self.rc, self.ran, self.looks, self.sleeps = dict(rc or {}), [], 0, 0
        self.head_v, self.sha_v, self.rounds_writes = head, sha, rounds_writes
        self.match_rcs = list(match_rcs)            # match_local's codes, call by call (then rc's)

    def alive(self) -> bool:
        self.looks += 1
        if self.finish_at is not None and self.looks == self.finish_at:
            (self.sc / "ALL_DONE").write_text("done\n", encoding="utf-8")
        return self.answers.pop(0) if self.answers else False

    def run(self, cmd, console, hip):
        script = cmd[2]
        self.ran.append((script, cmd, hip))
        if script == "scripts/rounds_local.py" and self.rounds_writes:
            self.rounds_writes(Path(cmd[cmd.index("--out") + 1]))
        if script == "scripts/match_local.py" and self.match_rcs:
            return self.match_rcs.pop(0)
        return self.rc.get(script, 0)

    def sleep(self, s):
        self.sleeps += 1

    def chain(self, args):
        return cl.Chain(args, alive=self.alive, run=self.run, sleep=self.sleep, head=lambda: self.head_v,
                        sha=lambda p: self.sha_v)


def _log(args) -> str:
    return (args.rounds_out.parent / "chain.log").read_text(encoding="utf-8")


def test_a_finished_stage_c_starts_p_then_the_rounds(tmp_path):
    args = _args(tmp_path)
    f = Fakes(args.stage_c, alive=[True, True, True, True], finish_at=3)   # ALL_DONE at the third look
    assert f.chain(args)() == 0
    assert [s for s, _, _ in f.ran] == ["scripts/p_local.py", "scripts/rounds_local.py"]
    p, rounds = f.ran[0][1], f.ran[1][1]
    assert p[p.index("--e") + 1] == "0.02" and p[p.index("--out") + 1] == str(args.p_out)
    assert rounds[rounds.index("--p-out") + 1] == str(args.p_out) and rounds[rounds.index("--rounds") + 1] == "3"
    for cmd in (p, rounds):
        assert cmd[0] == "gpu-py" and cmd[cmd.index("--tools-python") + 1] == "cpu-py" and "--keep-awake" in cmd
        assert cmd[cmd.index("--checkpoint") + 1] == "own2.pt"
    assert {hip for _, _, hip in f.ran} == {"1"} and "--heads-only-pass" not in rounds and "--n-train" not in rounds
    assert "CHAIN_P" in _log(args) and "CHAIN_ROUNDS" in _log(args)
    assert _log(args).index("CHAIN_VERDICT") < _log(args).index("CHAIN_P:")      # read again before P
    assert "did not read its gate" in json.loads((args.stage_c / "verdict.json").read_text(encoding="utf-8"))["next"]
    args = _args(tmp_path / "heads", heads_only_pass=True, rounds_end_by="2026-11-07 12:00", rounds_n_train=5000)
    (args.stage_c / "ALL_DONE").write_text("done\n", encoding="utf-8")
    f = Fakes(args.stage_c)
    assert f.chain(args)() == 0
    rounds = f.ran[1][1]
    assert "--heads-only-pass" in rounds
    assert rounds[rounds.index("--end-by") + 1] == "2026-11-07 12:00"
    assert rounds[rounds.index("--n-train") + 1] == "5000"


def test_a_runner_gone_without_all_done_refuses_after_two_looks(tmp_path):
    args = _args(tmp_path)
    (args.stage_c / "runner.log").write_text("x\n2026-11-01 STOPPED: STOP found before the plant\n", encoding="utf-8")
    f = Fakes(args.stage_c, alive=[True, False, True, False, False])
    assert f.chain(args)() == 1 and f.ran == []
    assert f.looks == 5                                        # one look of a gap is not an end
    assert "CHAIN_REFUSED" in _log(args) and "STOP found before the plant" in _log(args)


def test_an_unfinished_stage_c_a_wrong_head_or_checkpoint_refuses(tmp_path):
    cases = (("stop", {}, lambda sc: (sc / "STOP").write_text("x", encoding="utf-8"), "holds a STOP"),
             ("failed", {}, lambda sc: (sc / "plant.failed").write_text("x", encoding="utf-8"), "failed parts"),
             ("head", {"head": ("b" * 40, False)}, None, "not aaaaaaaa"),
             ("dirty", {"head": (HEAD, True)}, None, "tracked changes"),
             ("sha", {"sha": "beef"}, None, "SHA-256 is beef"))
    for name, fakes, spoil, why in cases:
        args = _args(tmp_path / name)
        (args.stage_c / "ALL_DONE").write_text("done\n", encoding="utf-8")
        if spoil:
            spoil(args.stage_c)
        f = Fakes(args.stage_c, **fakes)
        assert f.chain(args)() == 1 and f.ran == [], name
        assert why in _log(args), name


def test_a_p_that_does_not_pass_starts_no_round(tmp_path):
    args = _args(tmp_path)
    (args.stage_c / "ALL_DONE").write_text("done\n", encoding="utf-8")
    f = Fakes(args.stage_c, rc={"scripts/p_local.py": 3})
    assert f.chain(args)() == 1 and [s for s, _, _ in f.ran] == ["scripts/p_local.py"]
    assert "CHAIN_STOPPED: P returned 3" in _log(args)


def test_skip_p_runs_no_p_and_waives_the_rounds_p_gate(tmp_path):
    args = _args(tmp_path, skip_p=True, final_out=tmp_path / "final")
    (args.stage_c / "ALL_DONE").write_text("done\n", encoding="utf-8")
    f = Fakes(args.stage_c, rc={"scripts/p_local.py": 3}, rounds_writes=_rounds([1]))
    assert f.chain(args)() == 0
    assert [s for s, _, _ in f.ran] == ["scripts/rounds_local.py", "scripts/match_local.py"]
    rounds = f.ran[0][1]
    assert "--no-p-gate" in rounds and "--p-out" not in rounds
    log = _log(args)
    assert "no P (--skip-p)" in log and "CHAIN_P:" not in log and "P is skipped" in log
    assert log.index("CHAIN_VERDICT") < log.index("CHAIN_ROUNDS")
    assert cl.parse_args(ARGV + ["--skip-p"]).skip_p and not cl.parse_args(ARGV).skip_p


def test_a_runner_alive_long_after_all_done_refuses(tmp_path):
    args = _args(tmp_path)
    (args.stage_c / "ALL_DONE").write_text("done\n", encoding="utf-8")
    f = Fakes(args.stage_c, alive=[True] * (cl.EXIT_WAIT_LOOKS + 1))
    assert f.chain(args)() == 1 and f.ran == [] and f.sleeps == cl.EXIT_WAIT_LOOKS
    assert "still alive" in _log(args)


def _rounds(adopted):
    """A fake rounds_local writing round_<k>/round.json, a candidate for each
    k in `adopted`."""
    def write(out: Path):
        for k in range(1, 4):
            d = out / f"round_{k:02d}"
            d.mkdir(parents=True, exist_ok=True)
            cand = {"path": str(d / "pi.pt"), "sha256": "cafe"} if k in adopted else None
            (d / "round.json").write_text(json.dumps({"candidate": cand, "decision": {}}), encoding="utf-8")
    return write


def test_the_final_match_plays_the_last_adopted_round(tmp_path):
    args = _args(tmp_path, final_out=tmp_path / "final")
    (args.stage_c / "ALL_DONE").write_text("done\n", encoding="utf-8")
    for rounds_rc in (0, 3):
        f = Fakes(args.stage_c, rc={"scripts/rounds_local.py": rounds_rc}, rounds_writes=_rounds({1, 2}))
        assert f.chain(args)() == (0 if rounds_rc == 0 else 1)
        assert [s for s, _, _ in f.ran] == ["scripts/p_local.py", "scripts/rounds_local.py", "scripts/match_local.py"]
        m = f.ran[2][1]
        assert m[0] == "gpu-py" and m[m.index("--label") + 1] == "final_round2"
        assert m[m.index("--checkpoint") + 1] == str(args.rounds_out / "round_02" / "pi.pt")
        assert m[m.index("--out") + 1] == str(args.final_out) and m[m.index("--games") + 1] == "12800"
        assert m[m.index("--seed-base") + 1] == "900000" and m[m.index("--tools-python") + 1] == "cpu-py"
        assert "--keep-awake" in m and f.ran[2][2] == "1"
    assert "CHAIN_FINAL:" in _log(args) and "CHAIN_FINAL_DONE: match_local returned 0" in _log(args)


def test_no_final_match_without_a_candidate_a_good_sha_or_ended_rounds(tmp_path):
    cases = (("none", {}, set(), "no round adopted"),
             ("sha", {}, {1}, "not round 1's cafe"),
             ("failed", {"scripts/rounds_local.py": 1}, {1}, "the rounds returned 1"),
             ("stopped", {"scripts/rounds_local.py": 2}, {1}, "the rounds returned 2"))
    for name, rc, adopted, why in cases:
        args = _args(tmp_path / name, final_out=tmp_path / name / "final")
        (args.stage_c / "ALL_DONE").write_text("done\n", encoding="utf-8")
        f = Fakes(args.stage_c, rc=rc, rounds_writes=_rounds(adopted))
        if name == "sha":
            chain = f.chain(args)
            chain.sha = lambda p: "beef" if p.name == "pi.pt" else "cafe"
            assert chain() == 1
        else:
            assert f.chain(args)() == 1, name
        assert "scripts/match_local.py" not in [s for s, _, _ in f.ran], name
        assert why in _log(args), name


def _rounds_by_call(*adopted):
    """A fake rounds_local whose n-th call writes rounds 1-5, a candidate for
    each k in adopted[n]."""
    calls = iter(adopted)

    def write(out: Path):
        now = next(calls)
        for k in range(1, 6):
            d = out / f"round_{k:02d}"
            d.mkdir(parents=True, exist_ok=True)
            cand = {"path": str(d / "pi.pt"), "sha256": "cafe"} if k in now else None
            (d / "round.json").write_text(json.dumps({"candidate": cand, "decision": {}}), encoding="utf-8")
    return write


def _tail_args(tmp_path, **kw):
    return _args(tmp_path, final_out=tmp_path / "final", tail_out=tmp_path / "tail",
                 rounds_end_by="2026-11-07 10:00", **kw)


def test_the_tail_match_plays_the_final_candidate_at_t1_once_the_final_match_passes(tmp_path):
    args = _tail_args(tmp_path)
    (args.stage_c / "ALL_DONE").write_text("done\n", encoding="utf-8")
    f = Fakes(args.stage_c, rounds_writes=_rounds({1, 2}))
    assert f.chain(args)() == 0
    assert [s for s, _, _ in f.ran][2:] == ["scripts/match_local.py", "scripts/match_local.py"]
    final, tail = f.ran[2][1], f.ran[3][1]
    assert "--t1" not in final and final[final.index("--out") + 1] == str(args.final_out)
    assert tail[tail.index("--checkpoint") + 1] == final[final.index("--checkpoint") + 1] == \
        str(args.rounds_out / "round_02" / "pi.pt")
    assert "--t1" in tail and tail[tail.index("--label") + 1] == "final_round2t1"
    assert tail[tail.index("--out") + 1] == str(args.tail_out) and tail[tail.index("--games") + 1] == "12800"
    assert tail[tail.index("--seed-base") + 1] == "950000" and "--keep-awake" in tail and f.ran[3][2] == "1"
    assert "CHAIN_TAIL: the tail match (raw:t1) on round 2's" in _log(args)
    assert "CHAIN_TAIL_DONE: match_local returned 0" in _log(args)
    for name, rounds_rc, match_rcs, n_matches, rc in (("final failed", 0, [5], 1, 1), ("tail failed", 0, [0, 1], 2, 1),
                                                      ("design stop", 3, [], 2, 1)):
        args = _tail_args(tmp_path / name.replace(" ", "_"))
        (args.stage_c / "ALL_DONE").write_text("done\n", encoding="utf-8")
        f = Fakes(args.stage_c, rc={"scripts/rounds_local.py": rounds_rc}, rounds_writes=_rounds({1}),
                  match_rcs=match_rcs)
        assert f.chain(args)() == rc, name
        assert [s for s, _, _ in f.ran].count("scripts/match_local.py") == n_matches, name


def test_the_continuation_resumes_the_rounds_then_measures_a_newer_round(tmp_path):
    args = _tail_args(tmp_path, continue_rounds=12, continue_end_by="2026-11-08 12:00",
                      continue_final_out=tmp_path / "final_cont")
    (args.stage_c / "ALL_DONE").write_text("done\n", encoding="utf-8")
    f = Fakes(args.stage_c, rounds_writes=_rounds_by_call({1, 2}, {1, 2, 4}))
    assert f.chain(args)() == 0
    assert [s for s, _, _ in f.ran] == ["scripts/p_local.py", "scripts/rounds_local.py", "scripts/match_local.py",
                                        "scripts/match_local.py", "scripts/rounds_local.py", "scripts/match_local.py"]
    first, again = f.ran[1][1], f.ran[4][1]
    assert first[first.index("--rounds") + 1] == "3" and first[first.index("--end-by") + 1] == "2026-11-07 10:00"
    assert again[again.index("--rounds") + 1] == "12" and again[again.index("--end-by") + 1] == "2026-11-08 12:00"
    assert again[again.index("--out") + 1] == str(args.rounds_out) and again[again.index("--p-out") + 1] == str(args.p_out)
    m = f.ran[5][1]
    assert m[m.index("--label") + 1] == "final_round4" and "--t1" not in m
    assert m[m.index("--checkpoint") + 1] == str(args.rounds_out / "round_04" / "pi.pt")
    assert m[m.index("--out") + 1] == str(args.continue_final_out) and m[m.index("--seed-base") + 1] == "900000"
    log = _log(args)
    assert log.index("CHAIN_TAIL_DONE") < log.index("CHAIN_CONTINUE:") < log.index("CHAIN_CONTINUE_DONE")
    assert "CHAIN_CONTINUE_FINAL_DONE: match_local returned 0" in log
    # No round adopted after the measured one: the rounds went on, nothing more is measured.
    args = _tail_args(tmp_path / "same", continue_rounds=12, continue_end_by="2026-11-08 12:00",
                      continue_final_out=tmp_path / "same" / "final_cont")
    (args.stage_c / "ALL_DONE").write_text("done\n", encoding="utf-8")
    f = Fakes(args.stage_c, rounds_writes=_rounds_by_call({1, 2}, {1, 2}))
    assert f.chain(args)() == 0
    assert [s for s, _, _ in f.ran].count("scripts/rounds_local.py") == 2
    assert [s for s, _, _ in f.ran].count("scripts/match_local.py") == 2
    assert "CHAIN_NO_CONTINUE_FINAL: no round after round 2 adopted a candidate" in _log(args)


def test_no_continuation_after_a_failed_match_or_unfinished_rounds(tmp_path):
    for name, rounds_rc, match_rcs in (("tail", 0, [0, 5]), ("final", 0, [1]), ("stop", 3, [])):
        args = _tail_args(tmp_path / name, continue_rounds=12, continue_end_by="2026-11-08 12:00")
        (args.stage_c / "ALL_DONE").write_text("done\n", encoding="utf-8")
        f = Fakes(args.stage_c, rc={"scripts/rounds_local.py": rounds_rc}, rounds_writes=_rounds({1}),
                  match_rcs=match_rcs)
        assert f.chain(args)() == 1, name
        assert [s for s, _, _ in f.ran].count("scripts/rounds_local.py") == 1, name
        assert "CHAIN_CONTINUE" not in _log(args), name


ARGV = ["--stage-c", "sc", "--stage1", "s1", "--p-out", "p", "--rounds-out", "r", "--e", "0.02", "--rounds", "9",
        "--checkpoint", "own2.pt", "--checkpoint-sha256", "cafe", "--expect-head", "abc", "--gpu-python", "gpu-py"]


def test_the_command_line_holds_the_tail_and_the_continuation_together(capsys):
    args = cl.parse_args(ARGV + ["--final-out", "f", "--tail-out", "t", "--rounds-end-by", "2026-11-07 10:00",
                                 "--continue-rounds", "39", "--continue-end-by", "2026-11-08 12:00",
                                 "--continue-final-out", "fc"])
    assert (args.tail_games, args.tail_seed_base, args.continue_rounds) == (12800, 950000, 39)
    for extra, why in ((["--tail-out", "t"], "follow the final match"),
                       (["--final-out", "f", "--continue-rounds", "39"], "go together"),
                       (["--final-out", "f", "--continue-end-by", "2026-11-08 12:00"], "go together"),
                       (["--final-out", "f", "--continue-rounds", "9", "--continue-end-by", "2026-11-08 12:00"],
                        "past --rounds"),
                       (["--continue-rounds", "39", "--continue-end-by", "2026-11-08 12:00"], "past --rounds"),
                       (["--final-out", "f", "--rounds-end-by", "2026-11-07 10:00", "--continue-rounds", "39",
                         "--continue-end-by", "2026-11-07 09:00"], "after --rounds-end-by"),
                       (["--final-out", "f", "--continue-rounds", "39", "--continue-end-by", "Nov 8"], "not 'YYYY"),
                       (["--final-out", "f", "--tail-out", "f"], "must differ"),
                       (["--final-out", "f", "--continue-final-out", "fc"], "needs the continuation")):
        with pytest.raises(SystemExit):
            cl.parse_args(ARGV + extra)
        assert why in capsys.readouterr().err, extra


def test_an_unreadable_checkpoint_refuses_with_a_reason(tmp_path):
    args = _args(tmp_path, checkpoint=tmp_path / "missing.pt")
    (args.stage_c / "ALL_DONE").write_text("done\n", encoding="utf-8")
    f = Fakes(args.stage_c)
    chain = f.chain(args)
    chain.sha = cl.file_sha256
    assert chain() == 1 and f.ran == []
    assert "missing.pt cannot be read" in _log(args)


def test_runner_alive_reads_a_failed_or_empty_query_as_alive(monkeypatch):
    monkeypatch.setattr(cl.sys, "platform", "win32")
    for rc, out, alive in ((0, "0\n", False), (0, "2\n", True), (0, "", True), (1, "", True), (1, "0", True)):
        monkeypatch.setattr(cl.subprocess, "run",
                            lambda *a, rc=rc, out=out, **k: subprocess.CompletedProcess(a, rc, stdout=out, stderr=""))
        assert cl.runner_alive("x.py") is alive, (rc, out)

    def unstartable(*a, **k):
        raise OSError("powershell not found")
    monkeypatch.setattr(cl.subprocess, "run", unstartable)
    assert cl.runner_alive("x.py") is True                   # not started: nothing seen to end
