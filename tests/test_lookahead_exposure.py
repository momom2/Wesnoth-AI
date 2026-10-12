"""The look-ahead's material-and-exposure evaluator
(wesnoth_ai/lookahead_exposure.py, GameCore.threats): a recruit costs
nothing, a unit left where the opponent can hit it reads worse, each attack
is valued by the exact outcome enumeration, the attacks are assigned
greedily, and the observed world's hidden units take no part."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers.parity_games import record, state_of  # noqa: E402
from wesnoth_ai import game_core as gc  # noqa: E402
from wesnoth_ai.classes import Position  # noqa: E402
from wesnoth_ai.lookahead_exposure import (  # noqa: E402
    Threat, assign_greedily, material, threats, unit_value, valuation)

needs_core = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")
CASTLE = {(2, 3): "Kh", (2, 2): "Ch", (3, 3): "Ch", (1, 3): "Ch", (18, 3): "Kh"}


def _sim(units, fog: bool = False, villages=None):
    from tools.wesnoth_sim import WesnothSim
    data = record([("Lieutenant", 1, 2, 3, True), ("Lieutenant", 2, 18, 3, True), *units],
                  special={**CASTLE, **{hex_: "Gg^Vh" for owned in (villages or {}).values() for hex_ in owned}},
                  recruits={1: ["Spearman", "Bowman"], 2: ["Spearman"]}, fog=fog, villages=villages)
    sim = WesnothSim(state_of(data), scenario_id="", apply_scenario_events=False, max_turns=3)
    sim._seed_salt = "exposure-test"
    return sim


def _world(sim, side: int = 1):
    from wesnoth_ai.lookahead_world import build_world
    return build_world(sim, side, "observed", "exposure-test")


def _one(world, action):
    from wesnoth_ai.lookahead_world import expand
    (out,) = expand(world, action)
    return out


def _units(core):
    return {u["id"]: u for u in core.units_export()}


@needs_core
def test_a_recruit_moves_gold_into_a_unit_at_its_cost():
    """Out of every enemy's reach, a recruit changes neither the material
    nor M: the gold it costs is the unit's value at full hit points, both
    the unit database's cost (no rounding enters, so the tolerance is the
    float sum's)."""
    world = _world(_sim([]))
    before = valuation(world.sim.core.core, 1)
    after_state = _one(world, {"type": "recruit", "unit_type": "Spearman", "target_hex": Position(2, 2)})
    core = after_state.core.core
    recruit = next(u for u in core.units_export() if u["name"] == "Spearman")
    sides_before, sides_after = world.sim.core.core.sides_export(), core.sides_export()
    assert sides_before[0][2] - sides_after[0][2] == recruit["cost"] > 0, "the recruit was paid for"
    after = valuation(core, 1)
    assert before.exposure == after.exposure == 0.0
    assert after.material == pytest.approx(before.material, abs=1e-9)
    assert after.m == pytest.approx(before.m, abs=1e-9)


@needs_core
def test_a_unit_moved_next_to_two_strong_enemies_reads_worse_than_one_kept_out_of_reach():
    """The same Spearman moved next to two Dwarvish Fighters or back out of
    their four hexes of movement: the material is the same, the exposed
    move carries both fighters' attacks, and the evaluator values it
    lower."""
    from wesnoth_ai.lookahead_evaluators import MaterialExposureEvaluator
    world = _world(_sim([("Spearman", 1, 8, 3, False), ("Dwarvish Fighter", 2, 13, 2, False),
                         ("Dwarvish Fighter", 2, 13, 3, False)]))
    exposed = _one(world, {"type": "move", "start_hex": Position(8, 3), "target_hex": Position(12, 3)})
    safe = _one(world, {"type": "move", "start_hex": Position(8, 3), "target_hex": Position(5, 3)})
    v_exposed, v_safe = valuation(exposed.core.core, 1), valuation(safe.core.core, 1)
    assert v_exposed.material == v_safe.material
    assert v_safe.exposure == 0.0 and v_safe.threats == 0
    units = _units(exposed.core.core)
    attackers = {units[t.attacker]["name"] for t in threats(exposed.core.core, 1, list(units.values()))[0]
                 if units[t.target]["name"] == "Spearman"}
    assert "Dwarvish Fighter" in attackers and v_exposed.threats >= 2 and v_exposed.exposure > 0
    values = MaterialExposureEvaluator(20.0).values([exposed, safe], 1)
    assert values[0] < values[1]


@needs_core
def test_the_end_turn_candidate_is_read_on_the_board_before_the_opponents_turn_start():
    """In turn 2, side 2 holds a village with a wounded Spearman on it: its
    turn start pays it income and heals the Spearman. The end_turn
    candidate's outcome state carries both, and the evaluator reads it as
    the board stands at the decision: M is the current board's."""
    from wesnoth_ai.lookahead_evaluators import MaterialExposureEvaluator
    from wesnoth_ai.lookahead_world import expand
    sim = _sim([("Spearman", 2, 16, 6, False, {"hp": 5})], villages={2: [(16, 6)]})
    sim.step({"type": "end_turn"})
    sim.step({"type": "end_turn"})
    assert (sim.turn_number, int(sim.current_side)) == (2, 1)
    world = _world(sim)
    (out,) = expand(world, {"type": "end_turn"})
    now, then = world.sim.core.core, out.core.core
    spear = next(u["id"] for u in now.units_export() if u["name"] == "Spearman")
    assert _units(then)[spear]["current_hp"] > _units(now)[spear]["current_hp"], "the turn start heals"
    assert then.sides_export()[1][2] > now.sides_export()[1][2], "the turn start pays income"
    root = valuation(now, 1)
    assert valuation(then, 1).m < root.m
    ev = MaterialExposureEvaluator(20.0)
    ev.m_values = []
    ev.values([out], 1)
    assert ev.m_values == [root.m]


def _expected_loss(dist, hp: int) -> float:
    """The target's expected hit-point loss over an outcome distribution
    keyed (a_hp, d_hp, ...): its hit points on a death, else what it lost."""
    return sum(p * (hp if key[1] <= 0 else max(0, hp - key[1])) for key, p in dist.probs.items())


@needs_core
def test_an_attacks_expected_loss_is_the_enumerators_from_its_hex():
    """A Spearman two hexes from a Bowman: its one threat names a hex next
    to the Bowman, the spear (the dialog's pick: the javelin's answer is
    the bow), the Bowman's sword as the counter; its loss is the
    enumerator's expected loss with the Spearman standing on that hex,
    priced at the Bowman's cost over its hit points. The reverse threat
    takes the bow against the javelin."""
    from tools.combat_outcomes import choose_counter_weapon, enumerate_attack_outcomes
    world = _world(_sim([("Bowman", 1, 8, 3, False), ("Spearman", 2, 11, 3, False)]))
    core = world.sim.core.core
    rows, fallbacks, unvalued = core.threats(2, 1, 0.5)
    assert (fallbacks, unvalued) == (0, 0)
    ((attacker, target, x, y, weapon, counter, hp_loss, death),) = rows
    units = _units(core)
    assert units[attacker]["name"] == "Spearman" and units[target]["name"] == "Bowman"
    assert (weapon, counter) == (0, 0)
    there = state_of(record([("Lieutenant", 1, 2, 3, True), ("Lieutenant", 2, 18, 3, True),
                             ("Bowman", 1, 8, 3, False), ("Spearman", 2, x, y, False)], special=CASTLE, fog=False))
    spear, bow = (next(u for u in there.map.units if u.name == n) for n in ("Spearman", "Bowman"))
    assert choose_counter_weapon(there, spear, bow, weapon) == counter
    dist = enumerate_attack_outcomes(there, {"type": "attack", "start_hex": Position(x, y),
                                             "target_hex": Position(8, 3), "attack_index": weapon},
                                     advancement_choice="uniform")
    bowman = units[target]
    assert hp_loss == pytest.approx(_expected_loss(dist, bowman["current_hp"]), abs=1e-12) and hp_loss > 0
    assert death == pytest.approx(sum(p for k, p in dist.probs.items() if k[1] <= 0), abs=1e-12)
    v = valuation(core, 1)
    assert v.exposure == pytest.approx(hp_loss * bowman["cost"] / bowman["max_hp"], abs=1e-12)
    ((_, _, _, _, weapon_back, counter_back, _, _),) = core.threats(1, 2, 0.5)[0]
    assert (weapon_back, counter_back) == (1, 1)


def _threat(attacker: str, target: str, loss: float) -> Threat:
    return Threat(attacker, target, (0, 0), 0, -1, loss, 0.0)


def test_the_greedy_assignment_uses_each_attacker_once_and_caps_a_target_at_its_value():
    """Two attackers, two targets: A takes its best (10 on T1); B's 9 on T1
    is capped at the 2 left of T1's 12, so B takes its 5 on T2 instead (a
    static sort by loss would count 10 + 2). An attacker counts once, and
    two attacks on one target count at most its value."""
    total, taken = assign_greedily([_threat("A", "T1", 10), _threat("B", "T1", 9), _threat("B", "T2", 5),
                                    _threat("A", "T2", 4)], {"T1": 12.0, "T2": 20.0})
    assert total == 15
    assert [(t.attacker, t.target, gain) for t, gain in taken] == [("A", "T1", 10), ("B", "T2", 5)]
    total, taken = assign_greedily([_threat("A", "T1", 10), _threat("B", "T1", 9)], {"T1": 12.0})
    assert total == 12 and [gain for _, gain in taken] == [10, 2]
    total, _ = assign_greedily([_threat("A", "T1", 10), _threat("A", "T2", 8)], {"T1": 30.0, "T2": 30.0})
    assert total == 10


@needs_core
def test_no_hidden_unit_contributes_in_the_observed_world():
    """A Wolf Rider in side 1's fog can reach its Spearman: on the true
    board it is a threat; in side 1's observed world it is no threat, and
    no unit side 1 cannot see (the Wolf Rider, the enemy leader) counts in
    the material."""
    sim = _sim([("Spearman", 1, 6, 3, False), ("Wolf Rider", 2, 13, 3, False)], fog=True)
    true_core = sim.core.core
    true_units = true_core.units_export()
    wolf = next(u for u in true_units if u["name"] == "Wolf Rider")
    seen = set(true_core.visible_ids(1))
    assert wolf["id"] not in seen, "the board must hide the Wolf Rider from side 1"
    true_rows = true_core.threats(2, 1, 0.5)[0]
    assert [r[0] for r in true_rows] == [wolf["id"]]
    observed = _world(sim).sim.core.core
    assert not set(_units(observed)) - seen
    v = valuation(observed, 1)
    assert v.threats == 0 and v.exposure == 0.0
    hidden = sum(unit_value(u) for u in true_units if u["side"] == 2 and u["id"] not in seen)
    assert v.material == pytest.approx(material(true_units, observed.sides_export(), 1) + hidden, abs=1e-9)


@needs_core
def test_the_evaluator_gives_the_same_values_on_the_same_states():
    """An attack's outcome states valued twice, and again in a rebuilt
    world, give the same values to the last bit."""
    from wesnoth_ai.lookahead_evaluators import MaterialExposureEvaluator
    from wesnoth_ai.lookahead_world import expand
    sim = _sim([("Bowman", 1, 8, 3, False), ("Spearman", 2, 9, 3, False), ("Spearman", 2, 10, 4, False)])
    attack = {"type": "attack", "start_hex": Position(8, 3), "target_hex": Position(9, 3), "attack_index": 1}
    outs = expand(_world(sim), attack)
    assert len(outs) > 2
    ev = MaterialExposureEvaluator(20.0)
    first = ev.values(outs, 1)
    again = ev.values(outs, 1)
    rebuilt = ev.values(expand(_world(sim), attack), 1)
    assert np.array_equal(first, again) and np.array_equal(first, rebuilt)
    assert ev.counts == {"exposure_weapon_fallbacks": 0, "exposure_unvalued_attacks": 0}
