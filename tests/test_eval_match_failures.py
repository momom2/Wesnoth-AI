"""What the match driver (tools/run_elo_batch.py) does when games do
not produce results: a failed game leaves a failure record, the
chunk's exit status says whether the match is done, and no game is
admitted once a shared inference server has died.

The games are played by a stub script in place of elo_eval_game (the
driver's GAME_SCRIPT), so the driver's own loop runs for real in a
second or two: one process per game, or persistent workers speaking
the same `__DONE__ <rc>` protocol."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wesnoth_ai.constants import OBSERVATION_EPOCH  # noqa: E402

STUB = r'''
import json
import sys
from pathlib import Path

PLAN = json.loads(Path(__file__).with_suffix(".json").read_text(encoding="utf-8"))


def play(argv):
    label_a, _spec_a, label_b, _spec_b, side_a, seed, outdir = argv[1:8]
    if int(seed) in PLAN["fail"]:
        print("stub: game failed on purpose", file=sys.stderr, flush=True)
        return 1
    rec = dict(PLAN["template"], label_a=label_a, label_b=label_b,
               side_a=int(side_a), seed=int(seed), outcome_a=PLAN["outcome"])
    out = Path(outdir) / f"game_{label_a}_{label_b}_s{side_a}_{seed}.json"
    out.write_text(json.dumps(rec), encoding="utf-8")
    return 0


if "--worker" in sys.argv[1:]:
    for line in sys.stdin:
        if line.strip():
            print(f"__DONE__ {play(json.loads(line))}", flush=True)
else:
    sys.exit(play(sys.argv))
'''

# What the driver's pre-scan checks on a resume, for 'dummy' against
# 'dummy' at sims 0 on the cpu.
TEMPLATE = {"procedure_a": "raw", "procedure_b": "raw", "max_turns": 200,
            "margin_a": 0.0, "combat_stream": "per_game",
            "observation_epoch": int(OBSERVATION_EPOCH)}


def _stub_games(tmp_path, monkeypatch, *, fail=(), outcome="win") -> Path:
    """Point the driver at the stub; the stub fails the seeds in `fail`
    and writes `outcome` for the others."""
    from tools import run_elo_batch
    script = tmp_path / "stub_game.py"
    script.write_text(STUB, encoding="utf-8")
    _plan(script, fail=fail, outcome=outcome)
    monkeypatch.setattr(run_elo_batch, "GAME_SCRIPT", script)
    monkeypatch.setattr(run_elo_batch, "POLL_S", 0.05)
    return script


def _plan(script: Path, *, fail=(), outcome="win") -> None:
    script.with_suffix(".json").write_text(json.dumps(
        {"fail": list(fail), "outcome": outcome, "template": TEMPLATE}), encoding="utf-8")


def _argv(out: Path, *extra: str):
    return ["x", "--label-a", "A", "--spec-a", "dummy", "--label-b", "B",
            "--spec-b", "dummy", "--outdir", str(out), "--games", "4",
            "--mcts-sims", "0", "--device", "cpu", "--jobs", "4",
            "--max-extra-games", "0", "--time-budget-min", "5",
            "--min-free-mb", "0", *extra]


def test_a_failed_game_is_recorded_and_the_chunk_exits_failed(tmp_path, monkeypatch):
    """Slot 1 (A on side 2, seed 10001) fails: it leaves a record naming
    the game and why, the chunk exits EXIT_FAILED, and a re-run replays
    exactly that slot and completes the match."""
    from tools import run_elo_batch as rb
    script = _stub_games(tmp_path, monkeypatch, fail=(10001,))
    out = tmp_path / "games"
    assert rb.main(_argv(out)) == rb.EXIT_FAILED
    assert sorted(p.name for p in out.glob("game_*.json")) == [
        "game_A_B_s1_10000.json", "game_A_B_s1_10002.json", "game_A_B_s2_10003.json"]
    rec = json.loads((out / "failed_A_B_s2_10001.json").read_text(encoding="utf-8"))
    assert (rec["label_a"], rec["label_b"], rec["side_a"], rec["seed"]) == ("A", "B", 2, 10001)
    assert rec["returncode"] == 1 and rec["failures"] == 1
    assert "stub: game failed on purpose" in rec["reason"]

    _plan(script)                                  # the cause is gone
    assert rb.main(_argv(out)) == rb.EXIT_COMPLETE
    assert len(list(out.glob("game_*.json"))) == 4
    # The record stays: the match had a failure on its way.
    assert (out / "failed_A_B_s2_10001.json").exists()


def test_a_chunk_cut_by_its_time_budget_exits_resumable(tmp_path):
    """A chunk whose budget ends before its games leaves them to the
    next chunk, and says so in its exit status."""
    from tools import run_elo_batch as rb
    out = tmp_path / "games"
    argv = _argv(out)
    argv[argv.index("--time-budget-min") + 1] = "0"
    assert rb.main(argv) == rb.EXIT_RESUMABLE
    assert not list(out.glob("game_*.json"))


def test_a_spent_replacement_guard_exits_short(tmp_path, monkeypatch):
    """Every game capped and no replacement left: the schedule is played
    out with no decisive result, and a re-run would add nothing."""
    from tools import run_elo_batch as rb
    _stub_games(tmp_path, monkeypatch, outcome="timeout")
    out = tmp_path / "games"
    assert rb.main(_argv(out)) == rb.EXIT_GUARD_SPENT
    assert len(list(out.glob("game_*.json"))) == 4
    assert rb.main(_argv(out)) == rb.EXIT_GUARD_SPENT


class _DyingServer:
    """An inference server handle whose process dies once `after` result
    files exist in `outdir`."""

    def __init__(self, outdir: Path, after: int, sha: str):
        self.address = "unused"
        self.info = {"infer_bf16": False, "packed_trunk": False, "relevant_set": False,
                     "terrain_multi_hot": False, "checkpoint_sha256": sha}
        self.proc = SimpleNamespace(returncode=3)
        self._outdir, self._after = outdir, after

    def alive(self) -> bool:
        return len(list(self._outdir.glob("game_*.json"))) < self._after

    def err_tail(self, n: int = 4096) -> str:
        return "CUDA error: an illegal memory access was encountered"

    def shutdown(self, timeout: float = 30.0):
        return None


def test_no_game_is_admitted_after_an_inference_server_died(tmp_path, monkeypatch):
    """One job at a time: the server dies after the second game, so the
    third and fourth are never launched and the chunk exits failed."""
    import hashlib
    from tools import eval_inference_server
    from tools import run_elo_batch as rb
    _stub_games(tmp_path, monkeypatch)
    spec = tmp_path / "net.pt"
    spec.write_bytes(b"weights")
    sha = hashlib.sha256(b"weights").hexdigest()
    out = tmp_path / "games"
    monkeypatch.setattr(eval_inference_server, "launch_inference_server",
                        lambda *a, **k: _DyingServer(out, 2, sha))
    argv = _argv(out, "--raw-temperature-a", "0", "--persistent-workers",
                 "--shared-inference")
    argv[argv.index("--spec-a") + 1] = str(spec)
    argv[argv.index("--jobs") + 1] = "1"
    assert rb.main(argv) == rb.EXIT_FAILED
    assert sorted(p.name for p in out.glob("game_*.json")) == [
        "game_A_B_s1_10000.json", "game_A_B_s2_10001.json"]
    assert not list(out.glob("failed_*.json")), "an unplayed slot is not a failed game"
