"""Detector advisor (propose + dispose): a Tier-1 finding is reconstructed
into played vs proposed orderings and judged by a value function -> delta_v.

Uses a stub value_fn (negated enemy HP) so the direction is known: the
proposed ordering activates a backstab, lowering enemy HP, so delta_v > 0.
The real pipeline swaps in the model's C51 value head."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from tools.detector_advisor import (                              # noqa: E402
    advice_signals, delta_v_for_finding, _reorder_before,
)
from tools.swap_detector import (                                 # noqa: E402
    SideTurn, hex_neighbors, opposite_hex, backstab_setup_findings,
)
from tools.combat_outcomes import choose_counter_weapon           # noqa: E402
from sim_test_helpers import fresh_scenario_sim                   # noqa: E402
from tools.replay_dataset import _build_recruit_unit              # noqa: E402


def _backstab_side_turn():
    """A played side-turn [attack, move] where the thief attacks WITHOUT
    backstab, then a flanker moves onto the opposite hex -- so reordering
    (move first) activates the backstab. Defender beefed so the fight never
    kills (no advancement bail). Returns (SideTurn, thief, flanker, target)."""
    sim = fresh_scenario_sim(seed=5, max_turns=10,
                             scenario_id="multiplayer_The_Freelands")
    gs = sim.gs
    gs.map.units.clear()
    xpmod = int(getattr(gs.global_info, "_experience_modifier", 100) or 100)

    def inb(h):
        return 0 <= h[0] < gs.map.size_x and 0 <= h[1] < gs.map.size_y

    dx, dy = 12, 12
    A = next(h for h in hex_neighbors(dx, dy) if inb(h))
    opp = opposite_hex((dx, dy), A)
    assert opp is not None and inb(opp)
    S = next(h for h in hex_neighbors(*opp)
             if inb(h) and h not in {(dx, dy), A, opp})

    tgt = _build_recruit_unit("Orcish Grunt", side=2, x=dx, y=dy, next_uid=1,
                              game_id="t", trait_seed_hex="00000001",
                              exp_modifier=xpmod)
    tgt.current_hp = 200
    tgt.max_hp = 200
    thief = _build_recruit_unit("Thief", side=1, x=A[0], y=A[1], next_uid=2,
                                game_id="t", trait_seed_hex="00000002",
                                exp_modifier=xpmod)
    flk = _build_recruit_unit("Thief", side=1, x=S[0], y=S[1], next_uid=3,
                              game_id="t", trait_seed_hex="00000003",
                              exp_modifier=xpmod)
    for u in (tgt, thief, flk):
        gs.map.units.add(u)

    dw = choose_counter_weapon(gs, thief, tgt, 0)
    attack_cmd = ["attack", A[0], A[1], dx, dy, 0, dw, "deadbeef"]
    move_cmd = ["move", [S[0], opp[0]], [S[1], opp[1]]]
    st = SideTurn("t", 1, 1, gs, [attack_cmd, move_cmd])   # attack FIRST
    return st, thief, flk, tgt


def _enemy_hp_value(side: int):
    """Stub value: higher when the enemy (not `side`) has less total HP."""
    def f(gs):
        return -sum(u.current_hp for u in gs.map.units if u.side != side)
    return f


def test_reorder_before_bubbles_move_ahead_of_attack():
    acts = [["attack"], ["x"], ["move"]]
    assert _reorder_before(acts, move_idx=2, attack_idx=0) == [
        ["move"], ["attack"], ["x"]]


def test_backstab_finding_carries_reorder_indices():
    st, *_ = _backstab_side_turn()
    findings, _inc = backstab_setup_findings(st)
    assert len(findings) == 1, findings
    f = findings[0]
    assert f.attack_idx == 0 and f.move_idx == 1


def test_advisor_delta_v_positive_for_backstab_setup():
    st, _thief, flk, tgt = _backstab_side_turn()
    value_fn = _enemy_hp_value(side=1)
    sigs = advice_signals(st, value_fn)
    assert len(sigs) == 1, sigs
    s = sigs[0]
    assert s.motif == "backstab_setup"
    assert s.proposed_action[0] == "move"
    assert s.divergence_action[0] == "attack"
    # activating the backstab lowers enemy HP -> higher value -> delta_v > 0.
    assert s.delta_v is not None
    assert s.delta_v > 0.0, s.delta_v


def test_advisor_delta_v_none_when_no_reorder_indices():
    st, *_ = _backstab_side_turn()
    findings, _ = backstab_setup_findings(st)
    f = findings[0]
    f.attack_idx = None                       # simulate an unindexed finding
    assert delta_v_for_finding(st, f, _enemy_hp_value(1)) is None
