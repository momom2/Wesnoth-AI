#!/usr/bin/env python3
"""Elo collector: the material-sign draw convention must reassign
drawn games by final material margin (dead zone -> stays a draw)
while the pure convention keeps them as draws — both fits from the
SAME game records (locked 2026-07-04)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

import glob
import pytest

from tools.elo_collect import build_pairs
from tools.elo_ladder import fit_elo


def _g(a, b, outcome_a, margin_a=0.0, side_a=1):
    return {"label_a": a, "label_b": b, "outcome_a": outcome_a,
            "margin_a": margin_a, "side_a": side_a}


def test_material_reassigns_draws_pure_keeps_them():
    games = [
        _g("new", "old", "win"),                    # decisive
        _g("new", "old", "draw", margin_a=+0.4),    # new ahead at cap
        _g("old", "new", "timeout", margin_a=-0.3), # A=old behind -> new
        _g("new", "old", "draw", margin_a=+0.01),   # dead zone
    ]
    labels, pure, mat = build_pairs(games, eps=0.02)
    i, j = 0, 1                     # labels sorted: ["new", "old"]
    assert labels == ["new", "old"]
    p, m = pure[(i, j)], mat[(i, j)]
    assert (p.wins_i, p.draws, p.wins_j) == (1, 3, 0)
    assert (m.wins_i, m.draws, m.wins_j) == (3, 1, 0), (
        "both material-ahead draws must become wins for 'new'; the "
        "dead-zone draw must remain a draw")


def test_material_fit_separates_where_pure_cannot():
    # All games drawn, but 'new' finishes ahead every time: the pure
    # fit sees perfect symmetry; the material fit must rank new > old.
    games = [_g("new", "old", "draw", margin_a=0.5) for _ in range(10)]
    labels, pure, mat = build_pairs(games, eps=0.02)
    elo_p, _ = fit_elo(2, pure, 1, 0.0, 1.0, 0.5)
    elo_m, _ = fit_elo(2, mat, 1, 0.0, 1.0, 0.5)
    assert abs(elo_p[0] - elo_p[1]) < 1.0, "pure: all-draws -> level"
    assert elo_m[0] > elo_m[1] + 100, "material: must separate clearly"
