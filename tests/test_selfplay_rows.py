"""Per-game-side rows of a self-play run (tools/selfplay_rows.py): each
game's two sides in their own frames, capped games kept at z = 0."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from tools.match_luck import COLUMNS  # noqa: E402
from tools.selfplay_rows import load_rows, side_rows  # noqa: E402


def _walk_row(outcome_a="win", side_a=2, seed=105):
    return {"match": "selfplay", "game": f"game_A_B_s{side_a}_{seed}", "seed": seed, "side_a": side_a,
            "scenario_id": "m", "faction_a": "Drakes", "faction_b": "Undead", "label_a": "A",
            "label_b": "B", "outcome_a": outcome_a, "turns": 9, "ended_by": "leader",
            "S": {c: float(i + 1) for i, c in enumerate(COLUMNS)}}


def test_each_side_reads_the_game_in_its_own_frame():
    one, two = side_rows(_walk_row(), seed_base=100)
    assert (one["side"], two["side"]) == (1, 2) and one["index"] == two["index"] == 5
    # Side A sits on seat 2 and won.
    assert (two["z"], one["z"]) == (1, -1) and not one["capped"]
    assert two["S"] == [float(i + 1) for i in range(len(COLUMNS))]
    assert one["S"] == [-x for x in two["S"]]
    assert two["cell"] == "m|2|Drakes|Undead" and one["cell"] == "m|1|Undead|Drakes"
    assert (one["label"], two["label"]) == ("B", "A")
    capped = side_rows(_walk_row(outcome_a="timeout"), seed_base=100)
    assert [r["z"] for r in capped] == [0, 0] and all(r["capped"] for r in capped)
    broken = side_rows({**_walk_row(), "error": "RecordMismatch: x"}, seed_base=100)
    assert all("S" not in r and r["error"] for r in broken)
    with pytest.raises(ValueError, match="seed base"):
        side_rows(_walk_row(), seed_base=200)


def test_a_run_directory_is_walked_into_side_rows(tmp_path):
    """Two short games whose turn cap ends them: four rows, z = 0, the
    luck sums opposite between the two sides of a game."""
    from tests.helpers.selfplay_games import selfplay_game, tiny_checkpoint
    from tools.selfplay_rows import main
    spec = tiny_checkpoint(tmp_path / "net.pt")
    games = tmp_path / "games"
    selfplay_game(spec, games, 40, 1)
    selfplay_game(spec, games, 41, 2)
    assert main(["--games", str(games), "--walk", str(tmp_path / "walk.jsonl"),
                 "--out", str(tmp_path / "rows.jsonl"), "--seed-base", "40"]) == 0
    rows = load_rows(tmp_path / "rows.jsonl")
    assert sorted((r["index"], r["side"]) for r in rows) == [(0, 1), (0, 2), (1, 1), (1, 2)]
    assert all(r["capped"] and r["z"] == 0 for r in rows)
    by_game = {}
    for r in rows:
        by_game.setdefault(r["game"], []).append(r)
    for one, two in by_game.values():
        assert one["S"] == pytest.approx([-x for x in two["S"]])
