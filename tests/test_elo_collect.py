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


# ---------------------------------------------------------------------
# Estimands the collector must carry to the catalog (2026-09-13): the
# fields below change the players or the procedure and were guarded
# only INSIDE a games dir, so two dirs measuring different things
# pooled into one Bradley-Terry fit in silence.
# ---------------------------------------------------------------------
def _result(**over):
    rec = {"label_a": "ref", "label_b": "chal", "outcome_a": "win",
           "margin_a": 0.5, "side_a": 1, "seed": 10_000,
           "basis_a": "relset", "basis_b": "relset",
           "mcts_batch": 1, "infer_bf16": True, "infer_compile": False,
           "shared_inference": True, "infer_packed_trunk": True,
           "combat_stream": "per_game",
           "value_center_a": None, "value_center_b": None,
           "moves_left_utility": None}
    rec.update(over)
    return rec


def test_dir_estimands_reports_every_travelling_field():
    from tools.elo_collect import dir_estimands
    est = dir_estimands([_result(), _result(seed=10_001, side_a=2)])
    assert est == {"basis_a": "relset", "basis_b": "relset",
                   "mcts_batch": 1, "infer_bf16": True,
                   "infer_compile": False, "shared_inference": True,
                   "infer_packed_trunk": True,
                   "combat_stream": "per_game"}, (
        "None-valued fields must drop out (they constrain nothing); "
        "every other field must travel")


def test_dir_estimands_refuses_a_mixed_dir():
    import pytest
    from tools.elo_collect import dir_estimands
    for field, other in (("basis_b", "full"),
                         ("mcts_batch", 4),
                         ("infer_bf16", False),
                         ("combat_stream", "shared"),
                         ("value_center_a", 0.3)):
        with pytest.raises(SystemExit, match=field):
            dir_estimands([_result(),
                           _result(seed=10_001, **{field: other})])


def test_legacy_files_default_to_the_old_regime():
    """A result file from before these fields existed: full board,
    B=1, fp32 eager, per-process, and the SHARED combat stream."""
    from tools.elo_collect import dir_estimands
    est = dir_estimands([{"label_a": "a", "label_b": "b",
                          "outcome_a": "win"}])
    assert est["basis_a"] == "full" and est["basis_b"] == "full"
    assert est["combat_stream"] == "shared"
    assert est["mcts_batch"] == 1
    assert est["infer_bf16"] is False


def test_estimands_round_trip_through_a_result_file(tmp_path, monkeypatch):
    """End to end on the production CLI: elo_collect reads a dir of
    result files, and the catalog edge carries the estimands and the
    (side, seed) slots the games were played on."""
    import json
    from tools import elo_collect
    from tools.elo_catalog import decode_game_ids, load_catalog
    games_dir = tmp_path / "pinX"
    games_dir.mkdir()
    for i in range(8):
        rec = _result(seed=10_000 + i, side_a=1 if i % 2 == 0 else 2,
                      outcome_a="win" if i % 2 == 0 else "loss",
                      procedure_a="raw:t0", procedure_b="raw:t0",
                      max_turns=200)
        (games_dir / f"game_{i}.json").write_text(json.dumps(rec),
                                                  encoding="utf-8")
    cat_path = tmp_path / "cat.json"
    rc = elo_collect.main(["elo_collect", str(games_dir),
                           "--catalog-path", str(cat_path)])
    assert rc == 0
    edge, = load_catalog(cat_path)["edges"].values()
    est = edge["protocol"]["estimands"]
    assert est["basis_a"] == "relset"
    assert est["combat_stream"] == "per_game"
    assert est["infer_bf16"] is True
    assert decode_game_ids(edge["games"]) == {
        (1 if i % 2 == 0 else 2, 10_000 + i) for i in range(8)}
