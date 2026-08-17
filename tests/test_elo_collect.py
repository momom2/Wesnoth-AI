#!/usr/bin/env python3
"""Elo collector conventions, from the SAME game records:
PURE counts decisive games only — a capped/stalled game is a
no-result absence, not a draw (user ruling 2026-08-17, revising the
2026-07-11 draws-are-draws convention); MATERIAL-SIGN (diagnostic)
still adjudicates absences by final material margin (dead zone ->
stays a draw)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


from tools.elo_collect import build_pairs
from tools.elo_ladder import fit_elo


def _g(a, b, outcome_a, margin_a=0.0, side_a=1):
    return {"label_a": a, "label_b": b, "outcome_a": outcome_a,
            "margin_a": margin_a, "side_a": side_a}


def test_material_adjudicates_absences_pure_excludes_them():
    games = [
        _g("new", "old", "win"),                    # decisive
        _g("new", "old", "draw", margin_a=+0.4),    # new ahead at cap
        _g("old", "new", "timeout", margin_a=-0.3), # A=old behind -> new
        _g("new", "old", "draw", margin_a=+0.01),   # dead zone
    ]
    labels, pure, mat, nores = build_pairs(games, eps=0.02)
    i, j = 0, 1                     # labels sorted: ["new", "old"]
    assert labels == ["new", "old"]
    p, m = pure[(i, j)], mat[(i, j)]
    assert (p.wins_i, p.draws, p.wins_j) == (1, 0, 0), (
        "PURE counts only the decisive game; the three capped games "
        "are absences, not draws")
    assert nores[(i, j)] == 3
    assert (m.wins_i, m.draws, m.wins_j) == (3, 1, 0), (
        "both material-ahead absences must become wins for 'new'; "
        "the dead-zone one must remain a draw")


def test_material_fit_separates_where_pure_has_no_data():
    # All games capped, but 'new' finishes ahead every time: PURE has
    # ZERO rating information (all absences -> prior keeps both at
    # the anchor); the material diagnostic must rank new > old.
    games = [_g("new", "old", "draw", margin_a=0.5) for _ in range(10)]
    labels, pure, mat, nores = build_pairs(games, eps=0.02)
    assert nores[(0, 1)] == 10
    elo_p, _ = fit_elo(2, pure, 1, 0.0, 1.0, 0.5)
    elo_m, _ = fit_elo(2, mat, 1, 0.0, 1.0, 0.5)
    assert abs(elo_p[0] - elo_p[1]) < 1.0, "pure: no data -> level"
    assert elo_m[0] > elo_m[1] + 100, "material: must separate clearly"
