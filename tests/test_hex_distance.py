"""`tools.abilities.hex_distance` on Wesnoth's odd-q hex grid."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.abilities import hex_distance, hex_neighbors  # noqa: E402


def test_neighbours_are_one_step_away_on_both_column_parities():
    for x, y in ((4, 4), (5, 4)):
        assert hex_distance(x, y, x, y) == 0
        for nx, ny in hex_neighbors(x, y):
            assert hex_distance(x, y, nx, ny) == 1, (x, y, nx, ny)
            assert hex_distance(nx, ny, x, y) == 1, (nx, ny, x, y)


def test_distance_is_symmetric_and_counts_straight_lines():
    for ax, ay, bx, by in [(0, 0, 5, 5), (3, 7, 11, 2), (4, 4, 4, 8), (1, 0, 2, 1), (5, 3, 6, 4)]:
        assert hex_distance(ax, ay, bx, by) == hex_distance(bx, by, ax, ay)
    assert hex_distance(0, 0, 10, 0) == 10
    assert hex_distance(5, 0, 5, 8) == 8
