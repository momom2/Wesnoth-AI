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

from sim_test_helpers import commit_view                          # noqa: E402
from tools.abilities import hex_neighbors, opposite_hex           # noqa: E402
from tools.combat_outcomes import enumerate_attack_outcomes       # noqa: E402
from tools.replay_dataset import _stats_for                       # noqa: E402
from wesnoth_ai.game_core import build_recruit_unit               # noqa: E402
from wesnoth_ai.rules.scenario_pool import (                                 # noqa: E402
    random_setup, build_scenario_gamestate, load_factions,
)
from tools.swap_detector import (                                 # noqa: E402
    compare_distributions, Verdict,
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
    gs.map.units.add(build_recruit_unit(
        "Thief", side=side, x=a_hex[0], y=a_hex[1], next_uid=8001,
        game_id="t", trait_seed_hex="12345678", exp_modifier=xpmod))
    if with_flanker:
        gs.map.units.add(build_recruit_unit(
            "Thief", side=side, x=opp_hex[0], y=opp_hex[1], next_uid=8002,
            game_id="t", trait_seed_hex="12345678", exp_modifier=xpmod))
    commit_view(sim)
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
    gs.map.units.add(build_recruit_unit(
        "Spearman", side=side, x=a_hex[0], y=a_hex[1], next_uid=8100,
        game_id="t", trait_seed_hex="12345678", exp_modifier=xpmod))
    if with_leader:
        gs.map.units.add(build_recruit_unit(
            "Lieutenant", side=side, x=l_hex[0], y=l_hex[1], next_uid=8101,
            game_id="t", trait_seed_hex="12345678", exp_modifier=xpmod))
    commit_view(sim)
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


def test_enumerate_children_via_sim_matches_dp():
    """The sim-driven outcome enumerator (the core's fight with each
    hit/miss sequence forced) must reproduce the exact DP distribution --
    proving its materialization is bit-faithful without re-implementing
    any post-combat bookkeeping."""
    from tools.swap_detector import (
        enumerate_children_via_sim, hex_neighbors)
    from tools.combat_outcomes import (
        enumerate_attack_outcomes, outcome_key_for_child,
        choose_counter_weapon)
    from sim_test_helpers import fresh_scenario_sim
    sim = fresh_scenario_sim(seed=11, max_turns=10,
                             scenario_id="multiplayer_The_Freelands")
    gs = sim.gs
    gs.map.units.clear()
    xpmod = int(getattr(gs.global_info, "_experience_modifier", 100) or 100)
    ax, ay = 10, 10
    dx, dy = next((h for h in hex_neighbors(ax, ay)
                   if 0 <= h[0] < gs.map.size_x and 0 <= h[1] < gs.map.size_y))
    att = build_recruit_unit("Spearman", side=1, x=ax, y=ay, next_uid=1,
                              game_id="t", trait_seed_hex="00000001",
                              exp_modifier=xpmod)
    dfd = build_recruit_unit("Orcish Grunt", side=2, x=dx, y=dy, next_uid=2,
                              game_id="t", trait_seed_hex="00000002",
                              exp_modifier=xpmod)
    gs.map.units.add(att)
    gs.map.units.add(dfd)
    commit_view(sim)
    d_weapon = choose_counter_weapon(gs, att, dfd, 0)
    attack_cmd = ["attack", ax, ay, dx, dy, 0, d_weapon, "deadbeef"]
    action = {"type": "attack", "start_hex": Position(ax, ay),
              "target_hex": Position(dx, dy), "attack_index": 0}

    dp = enumerate_attack_outcomes(gs, action, advancement_choice=None)
    assert dp is not None
    children = enumerate_children_via_sim(gs, attack_cmd)
    assert children is not None and len(children) > 1

    agg = {}
    for child, p in children:
        key = outcome_key_for_child(child, att.id, dfd.id)
        agg[key] = agg.get(key, 0.0) + p
    assert abs(sum(agg.values()) - 1.0) < 1e-9
    for key in set(agg) | set(dp.probs):
        assert abs(agg.get(key, 0.0) - dp.probs.get(key, 0.0)) < 1e-6, (
            key, agg.get(key, 0.0), dp.probs.get(key, 0.0))


def test_enumerate_children_via_sim_matches_dp_with_advancement():
    """With advancement_choice='uniform' the sim-driven enumerator must
    still reproduce the exact DP distribution, INCLUDING the uniform spread
    over advancement targets. A Spearman one XP short of levelling (2
    targets: Swordsman / Pikeman) attacks a weak defender it can kill; the
    enumerator forces each advancement choice (1/2) and must match
    enumerate_attack_outcomes(advancement_choice='uniform')."""
    from tools.swap_detector import (
        enumerate_children_via_sim, hex_neighbors, _advance_targets)
    from tools.combat_outcomes import (
        enumerate_attack_outcomes, outcome_key_for_child,
        choose_counter_weapon)
    from sim_test_helpers import fresh_scenario_sim
    assert len(_advance_targets("Spearman")) > 1, "test premise: 2 advances"

    sim = fresh_scenario_sim(seed=13, max_turns=10,
                             scenario_id="multiplayer_The_Freelands")
    gs = sim.gs
    gs.map.units.clear()
    xpmod = int(getattr(gs.global_info, "_experience_modifier", 100) or 100)
    ax, ay = 10, 10
    dx, dy = next((h for h in hex_neighbors(ax, ay)
                   if 0 <= h[0] < gs.map.size_x and 0 <= h[1] < gs.map.size_y))
    att = build_recruit_unit("Spearman", side=1, x=ax, y=ay, next_uid=1,
                              game_id="t", trait_seed_hex="00000001",
                              exp_modifier=xpmod)
    att.current_exp = att.max_exp - 1          # any XP levels it
    dfd = build_recruit_unit("Walking Corpse", side=2, x=dx, y=dy, next_uid=2,
                              game_id="t", trait_seed_hex="00000002",
                              exp_modifier=xpmod)
    gs.map.units.add(att)
    gs.map.units.add(dfd)
    commit_view(sim)
    d_weapon = choose_counter_weapon(gs, att, dfd, 0)
    attack_cmd = ["attack", ax, ay, dx, dy, 0, d_weapon, "deadbeef"]
    action = {"type": "attack", "start_hex": Position(ax, ay),
              "target_hex": Position(dx, dy), "attack_index": 0}

    dp = enumerate_attack_outcomes(gs, action, advancement_choice="uniform")
    assert dp is not None
    children = enumerate_children_via_sim(
        gs, attack_cmd, advancement_choice="uniform")
    assert children is not None
    # the Spearman must actually advance in some outcome (2 keyed types)
    adv_types = {outcome_key_for_child(c, att.id, dfd.id)[8] for c, _ in children}
    assert {"Swordsman", "Pikeman"} & adv_types, adv_types

    agg = {}
    for child, p in children:
        key = outcome_key_for_child(child, att.id, dfd.id)
        agg[key] = agg.get(key, 0.0) + p
    assert abs(sum(agg.values()) - 1.0) < 1e-9
    for key in set(agg) | set(dp.probs):
        assert abs(agg.get(key, 0.0) - dp.probs.get(key, 0.0)) < 1e-6, (
            key, agg.get(key, 0.0), dp.probs.get(key, 0.0))


def test_strong_attacker_first_flags_weaker_lead():
    """Two own units attack the same target; the recorded order leads with
    the LOWER solo-kill-probability unit (a Thief that can't kill an 18-HP
    Walking Corpse) ahead of the higher one (an Orcish Grunt that can), so
    reordering to lead with the Grunt banks the Thief on the kill branch.
    strong_attacker_first must fire, naming the Grunt as the better lead."""
    from tools.swap_detector import (
        strong_attacker_first_findings, SideTurn, hex_neighbors)
    from tools.combat_outcomes import choose_counter_weapon
    from sim_test_helpers import fresh_scenario_sim

    sim = fresh_scenario_sim(seed=9, max_turns=10,
                             scenario_id="multiplayer_The_Freelands")
    gs = sim.gs
    gs.map.units.clear()
    xpmod = int(getattr(gs.global_info, "_experience_modifier", 100) or 100)

    def inb(h):
        return 0 <= h[0] < gs.map.size_x and 0 <= h[1] < gs.map.size_y

    dx, dy = 12, 12
    nb = [h for h in hex_neighbors(dx, dy) if inb(h)]
    sx, syx = nb[0], nb[1]
    tgt = build_recruit_unit("Walking Corpse", side=2, x=dx, y=dy, next_uid=1,
                              game_id="t", trait_seed_hex="00000001",
                              exp_modifier=xpmod)
    thief = build_recruit_unit("Thief", side=1, x=sx[0], y=sx[1], next_uid=2,
                                game_id="t", trait_seed_hex="00000002",
                                exp_modifier=xpmod)
    grunt = build_recruit_unit("Orcish Grunt", side=1, x=syx[0], y=syx[1],
                                next_uid=3, game_id="t",
                                trait_seed_hex="00000003", exp_modifier=xpmod)
    for u in (tgt, thief, grunt):
        gs.map.units.add(u)
    commit_view(sim)

    dw_t = choose_counter_weapon(gs, thief, tgt, 0)
    dw_g = choose_counter_weapon(gs, grunt, tgt, 0)
    # recorded order leads with the WEAKER attacker (the Thief).
    actions = [
        ["attack", sx[0], sx[1], dx, dy, 0, dw_t, "deadbeef"],
        ["attack", syx[0], syx[1], dx, dy, 0, dw_g, "deadbeef"],
    ]
    st = SideTurn("t", 1, 1, gs, actions)
    findings, inc = strong_attacker_first_findings(st)
    assert inc == 0
    assert len(findings) == 1, findings
    f = findings[0]
    assert f.motif == "strong_attacker_first"
    assert f.vector["recorded_lead"] == "Thief"
    assert f.vector["better_lead"] == "Orcish Grunt"
    assert float(f.vector["p_kill_better"]) > float(f.vector["p_kill_lead"])
