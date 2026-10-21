"""The decision log in the game record (tools/game_record.py `decisions`),
written by elo_eval_game and run_elo_batch under --log-decisions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from tools.decision_log import ENTRY_KEYS  # noqa: E402


def _checkpoint(tmp_path: Path) -> str:
    import torch
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(1)
    path = tmp_path / "net.pt"
    TransformerPolicy(device=torch.device("cpu"), d_model=32, num_layers=1, num_heads=2,
                      d_ff=64).save_checkpoint(path)
    return str(path)


def _game(spec: str, out: Path, *extra: str):
    """One temperature-1 self-play game of `spec` on both seats; (result,
    record)."""
    from tools.elo_eval_game import main
    from tools.game_record import read_records
    assert main(["x", "A", spec, "B", spec, "1", "7", str(out), "--mcts-sims", "0",
                 "--raw-temperature-a", "1", "--raw-temperature-b", "1", "--max-turns", "2",
                 "--device", "cpu", *extra]) == 0
    result = json.loads((out / "game_A_B_s1_7.json").read_text(encoding="utf-8"))
    (record,) = list(read_records(out / "game_A_B_s1_7.game.jsonl.gz"))
    return result, record


def test_the_record_carries_the_log_only_when_asked_and_the_game_is_the_same(tmp_path):
    spec = _checkpoint(tmp_path)
    r0, rec0 = _game(spec, tmp_path / "off")
    r1, rec1 = _game(spec, tmp_path / "on", "--log-decisions")
    assert "decisions" not in rec0 and r0["decisions_logged"] is False
    assert r1["decisions_logged"] is True
    assert rec1["commands"] == rec0["commands"] and rec1["final_digest"] == rec0["final_digest"]
    log = rec1["decisions"]
    assert log and all(tuple(e) == ENTRY_KEYS for e in log)
    assert {e["side"] for e in log} == {1, 2}
    live = [e["cmd"] for e in log if not e["refused"]]
    assert live == sorted(set(live)) and live[-1] < len(rec1["commands"])
    # The commands no decision starts are the simulator's side starts.
    assert {c[0] for k, c in enumerate(rec1["commands"]) if k not in live} == {"init_side"}
    # Every end_turn command of a player side is one logged end_turn decision.
    ends = [k for k, c in enumerate(rec1["commands"]) if c[0] == "end_turn"]
    assert set(ends) <= {e["cmd"] for e in log if e["action"][1] == 3}
    # A resume never mixes logged and unlogged games.
    from tools.elo_eval_game import main
    with pytest.raises(SystemExit, match="decisions_logged"):
        main(["x", "A", spec, "B", spec, "1", "7", str(tmp_path / "off"), "--mcts-sims", "0",
              "--raw-temperature-a", "1", "--raw-temperature-b", "1", "--max-turns", "2",
              "--device", "cpu", "--log-decisions"])


def test_the_log_is_refused_unless_both_sides_play_the_raw_player(tmp_path):
    from tools.elo_eval_game import main
    from tools.run_elo_batch import main as batch
    spec = _checkpoint(tmp_path)
    for extra in (["--raw-temperature-a", "1"], ["--raw-temperature-a", "1", "--mcts-sims-b", "4"]):
        with pytest.raises(SystemExit, match="log-decisions"):
            main(["x", "A", spec, "B", spec, "1", "7", str(tmp_path / "g"), "--mcts-sims", "0",
                  "--max-turns", "1", "--device", "cpu", "--log-decisions", *extra])
    with pytest.raises(SystemExit) as refused:
        batch(["x", "--label-a", "A", "--spec-a", spec, "--label-b", "B", "--spec-b", "dummy",
               "--outdir", str(tmp_path / "b"), "--games", "1", "--device", "cpu", "--mcts-sims", "0",
               "--raw-temperature-a", "1", "--log-decisions", "--time-budget-min", "0"])
    assert refused.value.code == 2                     # argparse's refusal


@pytest.mark.slow
def test_the_batch_driver_writes_logged_records(tmp_path):
    """run_elo_batch --log-decisions: every game's record carries its log,
    and a re-run without the flag refuses the outdir."""
    from tools.game_record import read_records
    from tools.run_elo_batch import main as batch
    spec = _checkpoint(tmp_path)
    out = tmp_path / "games"
    argv = ["x", "--label-a", "A", "--spec-a", spec, "--label-b", "B", "--spec-b", spec,
            "--outdir", str(out), "--games", "2", "--max-extra-games", "0", "--device", "cpu",
            "--jobs", "2", "--mcts-sims", "0", "--raw-temperature-a", "1", "--raw-temperature-b", "1",
            "--max-turns", "2", "--min-free-mb", "0"]
    assert batch(argv + ["--log-decisions"]) in (0, 4)
    records = sorted(out.glob("game_*.game.jsonl.gz"))
    assert len(records) == 2
    for path in records:
        (rec,) = list(read_records(path))
        assert rec["decisions"] and {e["side"] for e in rec["decisions"]} == {1, 2}
    with pytest.raises(SystemExit, match="decisions_logged"):
        batch(argv)
