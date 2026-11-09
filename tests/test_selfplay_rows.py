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
    """Three short games whose turn cap ends them, one record lost (as a
    game killed on its timeout leaves it): the walk fails past
    --max-errors, and within it the lost game's two rows carry the error,
    the others z = 0 and opposite luck sums; the labels leave both seats
    of the lost game out."""
    from tests.helpers.selfplay_games import selfplay_game, tiny_checkpoint
    from tools.pg_labels import side_labels
    from tools.selfplay_rows import main
    spec = tiny_checkpoint(tmp_path / "net.pt")
    games = tmp_path / "games"
    for i in range(3):
        selfplay_game(spec, games, 40 + i, 1 + i % 2)
    (games / "game_A_B_s1_40.game.jsonl.gz").unlink()
    argv = ["--games", str(games), "--walk", str(tmp_path / "walk.jsonl"), "--out", str(tmp_path / "rows.jsonl"),
            "--seed-base", "40"]
    assert main(argv) == 1
    assert main(argv + ["--max-errors", "1"]) == 0
    rows = load_rows(tmp_path / "rows.jsonl")
    assert sorted((r["index"], r["side"]) for r in rows) == [(i, s) for i in range(3) for s in (1, 2)]
    lost = [r for r in rows if r["index"] == 0]
    assert all(r["error"] == "no record" and "S" not in r for r in lost)
    walked = [r for r in rows if r["index"] != 0]
    assert all(r["capped"] and r["z"] == 0 for r in walked)
    by_game = {}
    for r in walked:
        by_game.setdefault(r["game"], []).append(r)
    for one, two in by_game.values():
        assert one["S"] == pytest.approx([-x for x in two["S"]])
    labels, info = side_labels(rows, heldout_from=0)
    assert info["left_out"] == ["game_A_B_s1_40"]
    assert sorted((lab["game"], lab["side"]) for lab in labels) == sorted((r["game"], r["side"]) for r in walked)
