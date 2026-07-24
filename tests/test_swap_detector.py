"""Swap detector verifier -- gate 2: the thief-backstab theorem.

A backstab doubles per-hit damage, so against a defender that survives
an unbuffed hit the backstab attack's outcome distribution strictly
dominates the no-backstab one (enemy HP stochastically lower, P(kill)
higher, own HP >= via less retaliation). The distributional verifier
must return STRICTLY_BETTER -- computed from the two exact
OutcomeDistributions, not sampled rolls.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from tools.abilities import hex_neighbors, opposite_hex           # noqa: E402
from tools.combat_outcomes import enumerate_attack_outcomes       # noqa: E402
from tools.replay_dataset import _build_recruit_unit, _stats_for  # noqa: E402
from tools.scenario_pool import (                                 # noqa: E402
    random_setup, build_scenario_gamestate, load_factions,
)
from tools.swap_detector import (                                 # noqa: E402
    compare_distributions, Verdict, ATTACK_DIMS,
)
from tools.wesnoth_sim import WesnothSim                          # noqa: E402
from wesnoth_ai.classes import Position                           # noqa: E402


def _thief_vs_leader(with_flanker: bool):
    """A side-`s` Thief adjacent to the enemy leader; optionally a same-
    side flanker on the hex OPPOSITE the Thief (activating backstab).
    Returns (gs, attack_action) or skips if the geometry can't be set."""
    load_factions()
    setup = random_setup(random.Random(3), forced_faction=None)
    gs = build_scenario_gamestate(setup)
    sim = WesnothSim(gs, scenario_id=setup.scenario_id, max_turns=30)
    gs = sim.gs
    xpmod = int(getattr(gs.global_info, "_experience_modifier", 100) or 100)

    side = gs.global_info.current_side
    dfd = next((u for u in gs.map.units if u.side != side and u.is_leader), None)
    if dfd is None:
        pytest.skip("no enemy leader")
    occ = {(u.position.x, u.position.y) for u in gs.map.units}

    a_hex = opp_hex = None
    for (ax, ay) in hex_neighbors(dfd.position.x, dfd.position.y):
        if (ax, ay) in occ or not (0 <= ax < gs.map.size_x
                                   and 0 <= ay < gs.map.size_y):
            continue
        opp = opposite_hex((dfd.position.x, dfd.position.y), (ax, ay))
        if opp is None:
            continue
        ox, oy = opp
        if ((ox, oy) in occ or not (0 <= ox < gs.map.size_x
                                    and 0 <= oy < gs.map.size_y)):
            continue
        a_hex, opp_hex = (ax, ay), (ox, oy)
        break
    if a_hex is None:
        pytest.skip("no free adjacent+opposite hex pair around the leader")

    assert "backstab" in _stats_for("Thief")["attacks"][0].get("specials", []), \
        "test premise: Thief's first attack has the backstab special"
    gs.map.units.add(_build_recruit_unit(
        "Thief", side=side, x=a_hex[0], y=a_hex[1], next_uid=8001,
        game_id="t", trait_seed_hex="12345678", exp_modifier=xpmod))
    if with_flanker:
        gs.map.units.add(_build_recruit_unit(
            "Thief", side=side, x=opp_hex[0], y=opp_hex[1], next_uid=8002,
            game_id="t", trait_seed_hex="12345678", exp_modifier=xpmod))
    action = {"type": "attack",
              "start_hex": Position(a_hex[0], a_hex[1]),
              "target_hex": dfd.position, "attack_index": 0}
    return gs, action


def test_thief_backstab_is_strictly_better():
    gs_base, action = _thief_vs_leader(with_flanker=False)
    gs_cand, _ = _thief_vs_leader(with_flanker=True)
    d_base = enumerate_attack_outcomes(gs_base, action, advancement_choice="uniform")
    d_cand = enumerate_attack_outcomes(gs_cand, action, advancement_choice="uniform")
    assert d_base is not None and d_cand is not None

    # Sanity: backstab really is off in baseline / on in candidate --
    # the enemy-HP distributions must differ (candidate hits harder).
    from tools.swap_detector import _marginal, ATTACK_DIMS as _D
    enemy_hp = next(d for d in _D if d.name == "enemy_hp")
    assert _marginal(d_base, enemy_hp.value) != _marginal(d_cand, enemy_hp.value)

    cmp = compare_distributions(d_base, d_cand)
    assert cmp.verdict is Verdict.STRICTLY_BETTER, (
        f"backstab must dominate; got {cmp.verdict} / {cmp.vector}")
    # enemy-HP strictly lower, and nothing is worse for us.
    assert cmp.vector["enemy_hp"] == ">"
    assert "<" not in cmp.vector.values()


def test_identical_distributions_are_equal():
    gs, action = _thief_vs_leader(with_flanker=False)
    d = enumerate_attack_outcomes(gs, action, advancement_choice="uniform")
    assert d is not None
    cmp = compare_distributions(d, d)
    assert cmp.verdict is Verdict.EQUAL
    assert set(cmp.vector.values()) == {"="}


def test_none_distribution_is_inconclusive():
    gs, action = _thief_vs_leader(with_flanker=False)
    d = enumerate_attack_outcomes(gs, action, advancement_choice="uniform")
    assert compare_distributions(None, d).verdict is Verdict.INCONCLUSIVE
    assert compare_distributions(d, None).verdict is Verdict.INCONCLUSIVE


def _spearman_with_optional_leader(with_leader: bool):
    """A side-`s` Spearman (L1) adjacent to the enemy leader; optionally a
    same-side Lieutenant (L2, leadership) on a free hex adjacent to the
    Spearman -- which grants +25% leadership damage."""
    load_factions()
    setup = random_setup(random.Random(3), forced_faction=None)
    gs = build_scenario_gamestate(setup)
    sim = WesnothSim(gs, scenario_id=setup.scenario_id, max_turns=30)
    gs = sim.gs
    xpmod = int(getattr(gs.global_info, "_experience_modifier", 100) or 100)
    side = gs.global_info.current_side
    dfd = next((u for u in gs.map.units if u.side != side and u.is_leader), None)
    if dfd is None:
        pytest.skip("no enemy leader")
    occ = {(u.position.x, u.position.y) for u in gs.map.units}
    a_hex = next(((ax, ay) for (ax, ay) in hex_neighbors(dfd.position.x, dfd.position.y)
                  if (ax, ay) not in occ and 0 <= ax < gs.map.size_x
                  and 0 <= ay < gs.map.size_y), None)
    if a_hex is None:
        pytest.skip("no free hex adjacent to the enemy leader")
    occ.add(a_hex)
    l_hex = next(((lx, ly) for (lx, ly) in hex_neighbors(a_hex[0], a_hex[1])
                  if (lx, ly) not in occ and (lx, ly) != (dfd.position.x, dfd.position.y)
                  and 0 <= lx < gs.map.size_x and 0 <= ly < gs.map.size_y), None)
    if l_hex is None:
        pytest.skip("no free hex adjacent to the attacker for a leader")

    from tools.swap_detector import _has_leadership, _unit_level
    assert _has_leadership("Lieutenant") and _unit_level("Lieutenant") > _unit_level("Spearman")
    gs.map.units.add(_build_recruit_unit(
        "Spearman", side=side, x=a_hex[0], y=a_hex[1], next_uid=8100,
        game_id="t", trait_seed_hex="12345678", exp_modifier=xpmod))
    if with_leader:
        gs.map.units.add(_build_recruit_unit(
            "Lieutenant", side=side, x=l_hex[0], y=l_hex[1], next_uid=8101,
            game_id="t", trait_seed_hex="12345678", exp_modifier=xpmod))
    action = {"type": "attack", "start_hex": Position(a_hex[0], a_hex[1]),
              "target_hex": dfd.position, "attack_index": 0}
    return gs, action


def test_leadership_setup_is_strictly_better():
    gs_base, action = _spearman_with_optional_leader(with_leader=False)
    gs_cand, _ = _spearman_with_optional_leader(with_leader=True)
    d_base = enumerate_attack_outcomes(gs_base, action, advancement_choice="uniform")
    d_cand = enumerate_attack_outcomes(gs_cand, action, advancement_choice="uniform")
    assert d_base is not None and d_cand is not None
    cmp = compare_distributions(d_base, d_cand)
    assert cmp.verdict is Verdict.STRICTLY_BETTER, (cmp.verdict, cmp.vector)
    assert cmp.vector["enemy_hp"] == ">"
    assert "<" not in cmp.vector.values()


def test_pos_mp_dominance_criterion():
    """(position, MP) dominance: a unit at X with m MP dominates (Y, n)
    iff it can ACTUALLY reach Y (terrain/ZoC) landing with >= n MP."""
    from tools.swap_detector import pos_mp_dominates, _reach
    from sim_test_helpers import fresh_scenario_sim
    from tools.replay_dataset import _build_recruit_unit
    sim = fresh_scenario_sim(seed=7, max_turns=10,
                             scenario_id="multiplayer_The_Freelands")
    sim.gs.map.units.clear()
    xpmod = int(getattr(sim.gs.global_info, "_experience_modifier", 100) or 100)
    u = _build_recruit_unit("Spearman", side=1, x=10, y=10, next_uid=1,
                            game_id="t", trait_seed_hex="12345678",
                            exp_modifier=xpmod)
    sim.gs.map.units.add(u)
    m = int(u.current_moves)
    # same hex: dominates (X, n) for n <= m, not for n > m.
    assert pos_mp_dominates(sim.gs, u, (10, 10), m)
    assert not pos_mp_dominates(sim.gs, u, (10, 10), m + 1)
    # a reachable hex Y: dominates (Y, mp[Y]) exactly, not (Y, mp[Y]+1).
    r = _reach(sim.gs, u)
    Y = next((h for h in r.landable if h != (10, 10)), None)
    assert Y is not None, "spearman should reach some hex"
    n = r.mp[Y]
    assert pos_mp_dominates(sim.gs, u, Y, n)
    assert not pos_mp_dominates(sim.gs, u, Y, n + 1)
