"""Exact counter-weapon chooser (tools/combat_outcomes, the core's
port of battle_context::choose_defender_weapon + better_combat +
calculate_probability_of_debuff, 1.18.4). The tests pin
engine-derivable choices on real units -- including the Giant Scorpion
case where the exact rating DIFFERS from the old v1 max-damage
heuristic (poison term favors the sting).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from sim_test_helpers import commit_view, fresh_scenario_sim   # noqa: E402
from tools.combat_outcomes import choose_counter_weapon   # noqa: E402
from wesnoth_ai.game_core import build_unit   # noqa: E402


def _surgical_matchup(sim, att_type: str, dfd_type: str):
    """Repurpose two scenario units as a (attacker, defender) pair of
    the given DB types: stats/weapons/level/alignment all key off
    Unit.name, so renaming redirects every lookup. The live attacks
    list is the type's, as the core builds it. HP raised so no death
    branch muddies the hand-derived expectations."""
    gs = sim.gs
    side = gs.global_info.current_side
    att = next(u for u in gs.map.units if u.side == side and u.attacks)
    dfd = next(u for u in gs.map.units if u.side != side and u.attacks
               and u.side in (1, 2)
               and "petrified" not in (u.statuses or set()))
    for u, name in ((att, att_type), (dfd, dfd_type)):
        u.name = name
        u.attacks = build_unit({"uid": 0, "type": name}).attacks
        u.current_hp = 60
        u.max_hp = 60
        u.statuses.discard("poisoned")
        u.statuses.discard("slowed")
        if hasattr(u, "_defense_table"):
            del u._defense_table     # stale pre-rename stash
    commit_view(sim)
    return gs, att, dfd


def test_clasher_counters_with_the_spear():
    """Spearman (spear 7x3 melee) attacks a Drake Clasher (war talon
    5x4 blade vs spear 6x4 pierce+firststrike, BOTH melee). Neither
    side can die (damage caps far below 60 hp), so the primary
    kill-probability comparison ties at 0 and the choice falls to
    average damage dealt: the Clasher spear's 6/strike strictly
    beats the talon's 5 (Spearman blade/pierce resists are both
    neutral, cth identical) -- the engine picks the spear, index 1."""
    sim = fresh_scenario_sim(seed=17, max_turns=10, mini=True)
    gs, att, dfd = _surgical_matchup(sim, "Spearman", "Drake Clasher")
    idx = choose_counter_weapon(gs, att, dfd, 0)
    assert idx == 1, f"expected Clasher spear (1), got {idx}"
    assert choose_counter_weapon(gs, att, dfd, 0) == idx


def test_scorpion_counters_with_the_poison_sting():
    """Spearman attacks a Giant Scorpion (sting 9x1 POISON vs
    pincers 4x4, both melee). No kill is possible, so the choice is
    the average-damage band: per attacker-cth c, the sting expects
    9c damage + (c - 0) * POISON_AMOUNT(8) = 17c of rating versus
    the pincers' 16c -- the engine prefers the sting (index 0).
    The old v1 heuristic (damage x strikes: 9 vs 16) picked the
    pincers; this is the case that proves the exact port differs."""
    sim = fresh_scenario_sim(seed=18, max_turns=10, mini=True)
    gs, att, dfd = _surgical_matchup(sim, "Spearman", "Giant Scorpion")
    idx = choose_counter_weapon(gs, att, dfd, 0)
    assert idx == 0, f"expected Scorpion sting (0), got {idx}"


def test_no_matching_range_means_no_counter():
    """A melee-only defender attacked at range can't retaliate:
    chooser returns -1 (and only then). Drake Clasher has no ranged
    weapon; Spearman's javelin (index 1) is ranged."""
    sim = fresh_scenario_sim(seed=19, max_turns=10, mini=True)
    gs, att, dfd = _surgical_matchup(sim, "Spearman", "Drake Clasher")
    idx = choose_counter_weapon(gs, att, dfd, 1)   # javelin, ranged
    assert idx == -1


def test_an_attacker_the_fight_levels_is_scored_at_full_hp():
    """Corpus game 0223d226cc2f (replays_dataset), command 333: a
    Skeleton at 33/34 HP and 26/27 XP attacks a Dwarvish Fighter at
    22/38 HP, poisoned, and the engine recorded the axe (index 0).
    Fighting a level-1 unit gives 1 XP, so the Skeleton levels
    whatever happens and `combatant::fight` scores every outcome it
    survives at full HP (forced_levelup). Neither side can die here, so
    both counters leave it at 34, tie, and `better_combat`'s last
    tie-break keeps the first candidate. Scored at its real HP, the
    hammer's 10x2 against the Skeleton's impact weakness beats the
    axe's 4x3 and wins instead."""
    sim = fresh_scenario_sim(seed=20, max_turns=10, mini=True)
    gs, att, dfd = _surgical_matchup(sim, "Skeleton", "Dwarvish Fighter")
    att.current_hp, att.max_hp, att.current_exp, att.max_exp = 33, 34, 26, 27
    dfd.current_hp, dfd.max_hp, dfd.current_exp, dfd.max_exp = 22, 38, 3, 32
    dfd.statuses.add("poisoned")
    commit_view(sim)
    assert choose_counter_weapon(gs, att, dfd, 0) == 0
