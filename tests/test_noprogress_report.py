"""Unit tests for tools/noprogress_report.py (F2, 2026-08-10).

Pins the would-fire semantics against wesnoth_sim.noprogress_summary's
contract: `max_quiet` is the longest quiet streak incl. the tail,
`tail_quiet` is the terminal streak (never resumed), and
`resumed_streaks` lists quiet streaks (>=3) that ended with fighting
resuming -- the false-fire candidates.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.noprogress_report import analyze, iter_games  # noqa: E402


def _g(turns, max_quiet, tail_quiet, resumed):
    return {"turns": turns,
            "noprogress": {"max_quiet": max_quiet,
                           "tail_quiet": tail_quiet,
                           "resumed_streaks": resumed}}


def test_analyze_fire_classes():
    games = [
        # Quiet tail of 10: true fire at K<=10, saves (10-K) turns.
        _g(40, 10, 10, []),
        # Mid-game 6-quiet streak that RESUMED; tail active.
        _g(50, 6, 0, [6]),
        # Always busy.
        _g(30, 1, 0, []),
    ]
    rep = analyze(games, [4, 8, 12])
    assert rep["n_games"] == 3
    assert rep["total_turns"] == 120

    k4 = rep["per_k"][4]
    assert k4["fired"] == 2          # games 1 and 2
    assert k4["false"] == 1          # game 2's resumed streak >= 4
    assert k4["true"] == 1           # game 1's tail >= 4
    assert k4["turns_saved"] == 6    # 10 - 4

    k8 = rep["per_k"][8]
    assert k8["fired"] == 1 and k8["false"] == 0 and k8["true"] == 1
    assert k8["turns_saved"] == 2

    k12 = rep["per_k"][12]
    assert k12 == {"fired": 0, "false": 0, "true": 0, "turns_saved": 0}


def test_iter_games_filters_short_and_missing(tmp_path):
    d = tmp_path / "iter_000000"
    d.mkdir()
    rows = [
        _g(40, 5, 5, []),                       # kept
        _g(3, 5, 5, []),                        # too short
        {"turns": 40},                          # no noprogress readout
        "not json at all",
    ]
    with (d / "games.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write((r if isinstance(r, str) else json.dumps(r)) + "\n")
    kept = list(iter_games([tmp_path], min_turns=10))
    assert len(kept) == 1
    assert kept[0]["turns"] == 40
