"""The faction forced onto one side of every eval game
(scenario_pool.FORCED_FACTION, the Knalgan Alliance by the user's
choice) is an estimand: every result records it (the faction's name, or
"none"), a result from before the field reads as the Knalgan Alliance,
and two dirs played under different settings never pool -- not in one
outdir, not in one catalog fit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wesnoth_ai.constants import OBSERVATION_EPOCH  # noqa: E402


def _play(out: Path, seed: int):
    """One real 2-turn game, dummy against dummy; its result and the
    setup its game record kept."""
    from tools.elo_eval_game import main
    from tools.game_record import read_records
    assert main(["x", "A", "dummy", "B", "dummy", "1", str(seed), str(out),
                 "--mcts-sims", "0", "--max-turns", "2", "--device", "cpu"]) == 0
    stem = f"game_A_B_s1_{seed}"
    result = json.loads((out / f"{stem}.json").read_text(encoding="utf-8"))
    (record,) = read_records(out / f"{stem}.game.jsonl.gz")
    return result, record["setup"]


def test_the_forced_faction_is_recorded_as_played(tmp_path, monkeypatch):
    from wesnoth_ai.rules import scenario_pool
    result, setup = _play(tmp_path / "knalgan", 7)
    assert result["forced_faction"] == "Knalgan Alliance"
    assert "Knalgan Alliance" in (setup["faction1"], setup["faction2"])
    monkeypatch.setattr(scenario_pool, "FORCED_FACTION", None)
    result, _setup = _play(tmp_path / "uniform", 7)
    assert result["forced_faction"] == "none"


def _decisive(seed: int, **over) -> dict:
    rec = {"label_a": "A", "label_b": "B", "outcome_a": "win" if seed % 2 else "loss",
           "margin_a": 0.5, "side_a": 1 + seed % 2, "seed": seed,
           "procedure_a": "raw:t0", "procedure_b": "raw:t0", "max_turns": 200,
           "combat_stream": "per_game", "observation_epoch": int(OBSERVATION_EPOCH)}
    rec.update(over)
    return rec


def _write_dir(d: Path, seeds, **over) -> Path:
    d.mkdir()
    for s in seeds:
        (d / f"game_A_B_s{1 + s % 2}_{s}.json").write_text(json.dumps(_decisive(s, **over)),
                                                           encoding="utf-8")
    return d


def test_dirs_under_different_forced_factions_do_not_pool(tmp_path):
    """A legacy dir (no field) and an explicit Knalgan dir pool; a dir
    with no forced faction is refused against them."""
    from tools import elo_collect
    cat = tmp_path / "cat.json"
    legacy = _write_dir(tmp_path / "legacy", range(10000, 10010))
    knalgan = _write_dir(tmp_path / "knalgan", range(20000, 20010),
                         forced_faction="Knalgan Alliance")
    uniform = _write_dir(tmp_path / "uniform", range(30000, 30010), forced_faction="none")
    assert elo_collect.main(["x", str(legacy), "--catalog-path", str(cat)]) == 0
    assert elo_collect.main(["x", str(knalgan), "--catalog-path", str(cat)]) == 0
    with pytest.raises(SystemExit, match="forced_faction"):
        elo_collect.main(["x", str(uniform), "--catalog-path", str(cat)])


def test_one_outdir_holds_one_forced_faction(tmp_path):
    """The batch pre-scan and elo_collect refuse a dir that would mix."""
    from tools import elo_collect
    from tools.run_elo_batch import main
    out = _write_dir(tmp_path / "games", (10000,), forced_faction="none")
    with pytest.raises(SystemExit, match="forced"):
        main(["x", "--label-a", "A", "--spec-a", "dummy", "--label-b", "B",
              "--spec-b", "dummy", "--outdir", str(out), "--games", "1",
              "--mcts-sims", "0", "--raw-temperature-a", "0", "--raw-temperature-b", "0",
              "--device", "cpu", "--jobs", "1", "--max-extra-games", "0",
              "--time-budget-min", "5", "--min-free-mb", "0"])
    (out / "game_A_B_s2_10001.json").write_text(json.dumps(_decisive(10001)),
                                                encoding="utf-8")
    with pytest.raises(SystemExit, match="mixed forced_faction"):
        elo_collect.main(["x", str(out), "--no-catalog"])
