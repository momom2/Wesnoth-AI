"""Tests for the material draw tiebreaker (tools/draw_tiebreak.py)
and its integration into MCTS terminal values and trainer z targets.

The scoring function only reads `gs.map.units[*].{side,cost}`,
`gs.sides[i].current_gold`, and `gs.global_info._village_owner`,
so the tests use lightweight SimpleNamespace state stubs.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

from tools.draw_tiebreak import (  # noqa: E402
    DrawTiebreakConfig, draw_tiebreak_z,
)
from tools.mcts import _terminal_value  # noqa: E402


def _gs(*, gold=(100, 100), villages=(0, 0), unit_costs=((), ())):
    """Two-side state stub. `villages[i]` = how many villages side
    i+1 owns; `unit_costs[i]` = recruit costs of side i+1's units."""
    owner = {}
    pos = 0
    for side_i, n in enumerate(villages):
        for _ in range(n):
            owner[(pos, 0)] = side_i + 1
            pos += 1
    units = [
        SimpleNamespace(side=side_i + 1, cost=c)
        for side_i, costs in enumerate(unit_costs)
        for c in costs
    ]
    return SimpleNamespace(
        map=SimpleNamespace(units=units),
        sides=[SimpleNamespace(current_gold=gold[0]),
               SimpleNamespace(current_gold=gold[1])],
        global_info=SimpleNamespace(_village_owner=owner),
    )


CFG = DrawTiebreakConfig()


def test_equal_material_scores_zero():
    gs = _gs(gold=(100, 100), villages=(3, 3),
             unit_costs=((14, 20), (14, 20)))
    assert draw_tiebreak_z(gs, 1, CFG) == 0.0
    assert draw_tiebreak_z(gs, 2, CFG) == 0.0


def test_each_component_moves_the_score():
    base = _gs()
    assert draw_tiebreak_z(base, 1, CFG) == 0.0
    assert draw_tiebreak_z(_gs(villages=(2, 0)), 1, CFG) > 0.0
    assert draw_tiebreak_z(_gs(gold=(140, 100)), 1, CFG) > 0.0
    assert draw_tiebreak_z(_gs(unit_costs=((20, 20), ())), 1, CFG) > 0.0


def test_antisymmetric_between_sides():
    gs = _gs(gold=(150, 90), villages=(4, 1),
             unit_costs=((14, 14, 20), (17,)))
    z1 = draw_tiebreak_z(gs, 1, CFG)
    z2 = draw_tiebreak_z(gs, 2, CFG)
    assert z1 > 0.0
    assert abs(z1 + z2) < 1e-12


def test_cap_bounds_runaway_differential():
    gs = _gs(gold=(10_000, 0), villages=(20, 0),
             unit_costs=(tuple([20] * 30), ()))
    z = draw_tiebreak_z(gs, 1, CFG)
    # tanh saturates at float precision, so the bound is closed.
    assert 0.0 < z <= CFG.cap < 1.0
    # A material-crushed draw must still beat a LOSS by a wide margin.
    assert draw_tiebreak_z(gs, 2, CFG) >= -CFG.cap > -1.0


def test_terminal_value_draw_uses_tiebreak():
    gs = _gs(villages=(3, 0))
    sim = SimpleNamespace(winner=0, gs=gs,
                          done=True)
    assert _terminal_value(sim, 1) == 0.0                  # legacy
    z = _terminal_value(sim, 1, CFG)
    assert 0.0 < z < CFG.cap
    assert _terminal_value(sim, 2, CFG) == -z


def test_terminal_value_decisive_outcomes_unaffected():
    gs = _gs(villages=(0, 5))   # winner is material-behind: irrelevant
    sim = SimpleNamespace(winner=1, gs=gs, done=True)
    assert _terminal_value(sim, 1, CFG) == 1.0
    assert _terminal_value(sim, 2, CFG) == -1.0


def test_config_from_json(tmp_path):
    p = tmp_path / "tb.json"
    p.write_text('{"_about": "doc", "cap": 0.5, "weight_gold": 0.1}',
                 encoding="utf-8")
    cfg = DrawTiebreakConfig.from_json(p)
    assert cfg.cap == 0.5
    assert cfg.weight_gold == 0.1
    # Untouched fields keep defaults.
    assert cfg.weight_village == DrawTiebreakConfig().weight_village


def test_shipped_config_matches_code_defaults():
    """configs/draw_tiebreak.json documents the defaults; if the
    dataclass defaults drift, the doc must move with them."""
    shipped = DrawTiebreakConfig.from_json(
        Path(__file__).parent / "configs" / "draw_tiebreak.json")
    assert shipped == DrawTiebreakConfig()


# ---------------------------------------------------------------------
# Training-label decoupling (2026-07-10): draws train z=0 by default;
# the tiebreak is search-only unless --train-draw-tiebreak opts back in
# ---------------------------------------------------------------------

def _finalize_gs(**kw):
    """_gs() plus the global_info fields finalize_game touches
    (AUGMENT global_info -- replacing it would clobber the
    _village_owner map that material scoring reads)."""
    g = _gs(**kw)
    g.global_info.turn_number = 20
    g.global_info.current_side = 1
    return g


def test_finalize_draw_trains_honest_zero_by_default():
    import torch
    from tools.mcts import MCTSConfig
    from tools.mcts_policy import MCTSPolicy, _PendingMCTSState
    from transformer_policy import TransformerPolicy

    base = TransformerPolicy(device=torch.device("cpu"), d_model=32,
                             num_layers=1, num_heads=4, d_ff=64)
    final = _finalize_gs(villages=(3, 0))   # imbalanced draw
    mp = MCTSPolicy(base, MCTSConfig(draw_tiebreak=CFG))
    mp._pending["g"] = [
        _PendingMCTSState(gs=_finalize_gs(), visit_counts=[], side=s)
        for s in (1, 2)]
    mp.finalize_game("g", winner=0, final_gs=final)
    assert [e.z for e in mp._queue] == [0.0, 0.0], \
        "default: drawn games train the value head toward z=0"
    # aux target still carries material (that is the aux head's job)
    assert mp._queue[0].aux_target is not None
    assert mp._queue[0].aux_target > 0.0


def test_finalize_draw_legacy_tiebreak_optin():
    import torch
    from tools.mcts import MCTSConfig
    from tools.mcts_policy import MCTSPolicy, _PendingMCTSState
    from transformer_policy import TransformerPolicy

    base = TransformerPolicy(device=torch.device("cpu"), d_model=32,
                             num_layers=1, num_heads=4, d_ff=64)
    final = _finalize_gs(villages=(3, 0))
    mp = MCTSPolicy(base, MCTSConfig(draw_tiebreak=CFG),
                    train_draw_tiebreak=True)
    mp._pending["g"] = [
        _PendingMCTSState(gs=_finalize_gs(), visit_counts=[], side=s)
        for s in (1, 2)]
    mp.finalize_game("g", winner=0, final_gs=final)
    zs = [e.z for e in mp._queue]
    assert 0.0 < zs[0] <= CFG.cap and zs[1] == -zs[0], \
        "legacy opt-in restores antisymmetric material-z draw labels"


def test_z_composition_stats():
    from tools.mcts_policy import MCTSPolicy
    from trainer import MCTSExperience, TrainStats

    class _E:
        def __init__(self, z): self.z = z
    st = TrainStats()
    MCTSPolicy._attach_z_composition(
        st, [_E(1.0), _E(1.0), _E(-1.0), _E(0.0), _E(0.12), _E(-0.2)])
    assert abs(st.z_win_frac - 2 / 6) < 1e-9
    assert abs(st.z_loss_frac - 1 / 6) < 1e-9
    assert abs(st.z_draw_frac - 3 / 6) < 1e-9
