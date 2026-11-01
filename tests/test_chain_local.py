"""scripts/chain_local.py: stage C's end chained to P and the rounds, on
fakes (no process is started, nothing sleeps)."""
import json
import subprocess
from argparse import Namespace
from pathlib import Path

import scripts.chain_local as cl

HEAD = "a" * 40


def _args(tmp_path, **kw):
    sc = tmp_path / "stage_c"
    sc.mkdir(parents=True, exist_ok=True)
    base = dict(stage_c=sc, stage1=tmp_path / "stage1", p_out=tmp_path / "p", rounds_out=tmp_path / "rounds",
                e=0.02, rounds=3, checkpoint=Path("own2.pt"), checkpoint_sha256="cafe", expect_head=HEAD[:8],
                gpu_python="gpu-py", tools_python="cpu-py", hip_visible_devices="1", log=None, final_out=None,
                heads_only_pass=False, rounds_end_by=None,
                final_games=cl.FINAL_GAMES, final_seed_base=cl.FINAL_SEED_BASE)
    base.update(kw)
    return Namespace(**base)


class Fakes:
    """The runner's liveness as a list of answers (then False), the steps'
    exit codes by script, and what ran."""

    def __init__(self, sc: Path, alive=(), finish_at=None, rc=None, head=(HEAD, False), sha="cafe",
                 rounds_writes=None):
        self.sc, self.answers, self.finish_at = sc, list(alive), finish_at
        self.rc, self.ran, self.looks, self.sleeps = dict(rc or {}), [], 0, 0
        self.head_v, self.sha_v, self.rounds_writes = head, sha, rounds_writes

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
    assert {hip for _, _, hip in f.ran} == {"1"} and "--heads-only-pass" not in rounds
    assert "CHAIN_P" in _log(args) and "CHAIN_ROUNDS" in _log(args)
    assert _log(args).index("CHAIN_VERDICT") < _log(args).index("CHAIN_P:")      # read again before P
    assert "did not read its gate" in json.loads((args.stage_c / "verdict.json").read_text(encoding="utf-8"))["next"]
    args = _args(tmp_path / "heads", heads_only_pass=True, rounds_end_by="2026-11-07 12:00")
    (args.stage_c / "ALL_DONE").write_text("done\n", encoding="utf-8")
    f = Fakes(args.stage_c)
    assert f.chain(args)() == 0
    rounds = f.ran[1][1]
    assert "--heads-only-pass" in rounds
    assert rounds[rounds.index("--end-by") + 1] == "2026-11-07 12:00"


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
