"""Exact counter-weapon chooser (tools/combat_outcomes).

Port of battle_context::choose_defender_weapon + better_combat +
calculate_probability_of_debuff (1.18.4). The unit tests pin the
ported comparison math on hand-crafted marginals; the integration
tests pin engine-derivable choices on real units -- including the
Giant Scorpion case where the exact rating DIFFERS from the old v1
max-damage heuristic (poison term favors the sting).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from sim_test_helpers import fresh_scenario_sim   # noqa: E402
from tools.combat_outcomes import (   # noqa: E402
    _CombatantMarginals as M,
    _better_combat,
    _engine_marginals,
    _probability_of_debuff,
    _strike_dp,
    choose_counter_weapon,
)
from wesnoth_ai.sim import combat as cb   # noqa: E402


def test_better_combat_kill_probability_dominates():
    """Primary comparison: P(we kill them) - P(they kill us), 0.01
    threshold band."""
    us = M(death=0.10, avg_hp=20.0, poisoned=0.0)
    them_a = M(death=0.50, avg_hp=10.0, poisoned=0.0)
    them_b = M(death=0.30, avg_hp=10.0, poisoned=0.0)
    assert _better_combat(us, them_a, us, them_b, 1.0) is True
    assert _better_combat(us, them_b, us, them_a, 1.0) is False


def test_better_combat_poison_breaks_kill_tie():
    """Secondary comparison: poisoning THEM is worth
    (poisoned - death) * POISON_AMOUNT of average hp."""
    us = M(death=0.0, avg_hp=20.0, poisoned=0.0)
    them_pois = M(death=0.0, avg_hp=15.0, poisoned=0.9)
    them_clean = M(death=0.0, avg_hp=15.0, poisoned=0.0)
    assert _better_combat(us, them_pois, us, them_clean, 1.0) is True
    assert _better_combat(us, them_clean, us, them_pois, 1.0) is False


def test_better_combat_final_tiebreak_most_damage():
    """Within both 0.01 bands, the lower opponent average hp wins."""
    us = M(death=0.0, avg_hp=20.0, poisoned=0.0)
    them_low = M(death=0.0, avg_hp=12.0, poisoned=0.0)
    them_high = M(death=0.0, avg_hp=12.005, poisoned=0.0)
    assert _better_combat(us, them_low, us, them_high, 1.0) is True
    assert _better_combat(us, them_high, us, them_low, 1.0) is False


def test_debuff_formula_edges():
    f = _probability_of_debuff
    # Poisoner always touches us, no level-up cure: certain poison.
    assert f(0.0, True, 1.0, 1.0, False, 0.0) == 1.0
    # Enemy doesn't poison: healthy stays healthy.
    assert f(0.0, False, 1.0, 1.0, False, 0.0) == 0.0
    # Already poisoned, never touched, no cure: stays poisoned.
    assert f(1.0, False, 0.0, 1.0, False, 0.0) == 1.0
    # Guaranteed kill that levels us up: cured for sure.
    assert f(1.0, False, 0.0, 1.0, True, 1.0) == 0.0
    # Partial touch probability flows straight through.
    assert abs(f(0.0, True, 0.6, 1.0, False, 0.0) - 0.6) < 1e-12


def _surgical_matchup(sim, att_type: str, dfd_type: str):
    """Repurpose two scenario units as a (attacker, defender) pair of
    the given DB types: stats/weapons/level/alignment all key off
    Unit.name, so renaming redirects every lookup. The live attacks
    list is rebuilt from the DB too (the chooser's index guard reads
    it). HP raised so no death branch muddies the hand-derived
    expectations."""
    from tools.replay_dataset import _attacks_from_stats, _stats_for
    gs = sim.gs
    side = gs.global_info.current_side
    att = next(u for u in gs.map.units if u.side == side and u.attacks)
    dfd = next(u for u in gs.map.units if u.side != side and u.attacks
               and u.side in (1, 2)
               and "petrified" not in (u.statuses or set()))
    for u, name in ((att, att_type), (dfd, dfd_type)):
        u.name = name
        u.attacks = _attacks_from_stats(_stats_for(name))
        u.current_hp = 60
        u.max_hp = 60
        u.statuses.discard("poisoned")
        u.statuses.discard("slowed")
        if hasattr(u, "_defense_table"):
            del u._defense_table     # stale pre-rename stash
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
    assert choose_counter_weapon(gs, att, dfd, 0) == 0


def _battle_stats(damage: int, strikes: int) -> cb.BattleStats:
    """50% to hit, no special."""
    return cb.BattleStats(
        cth=50, damage=damage, slow_damage=damage // 2, n_attacks=strikes,
        orig_attacks=strikes, rounds=1, firststrike=False, drains=False,
        drain_constant=0, drain_percent=0, plague=False, plague_type="",
        poisons=False, slows=False, petrifies=False, backstab=False,
        swarm=False, is_attacker=True)


def _combat_unit(hp: int, max_hp: int, xp: int, max_xp: int) -> cb.CombatUnit:
    return cb.CombatUnit(
        side=1, hp=hp, max_hp=max_hp, level=1, experience=xp,
        max_experience=max_xp, alignment=cb.Alignment.NEUTRAL, weapons=[],
        resistance={}, defense_pct=50)


def _attacker_average_hp(attacker, defender, a_stats, d_stats) -> float:
    states = _strike_dp(a_stats, d_stats, attacker, defender, track_touched=True)
    return _engine_marginals(states, a_stats, d_stats, attacker, defender)[0].avg_hp


def test_levelup_scoring_follows_the_engine_fight_paths():
    """A (10/20 HP) strikes first, once, for 5 at 50%: it kills B
    (5 HP) half the time. Numbers worked by hand from
    attack_prediction.cpp.

    At 2/9 XP only a kill's 8 XP levels A. One strike each takes
    `one_strike_fight`, whose conditional_levelup scales A's surviving
    HP by 1 - P(kill) / P(survive) = 0.5 and adds P(kill) at full HP:
    A is 10 w.p. 0.75 and 6 w.p. 0.25 before, so 0.5 * 9 + 0.5 * 20 =
    14.5. When B strikes twice for 2 the fight goes through the matrix,
    which moves exactly the kill outcomes to full HP:
    0.5 * 20 + 0.5 * (0.25 * 10 + 0.5 * 8 + 0.25 * 6) = 14.0.
    At 8/9 XP the fight's 1 XP alone levels A: every survivor at 20."""
    b = _combat_unit(5, 30, 0, 100)
    on_kill = _combat_unit(10, 20, 2, 9)
    assert _attacker_average_hp(on_kill, b, _battle_stats(5, 1), _battle_stats(4, 1)) == 14.5
    assert _attacker_average_hp(on_kill, b, _battle_stats(5, 1), _battle_stats(2, 2)) == 14.0
    forced = _combat_unit(10, 20, 8, 9)
    assert _attacker_average_hp(forced, b, _battle_stats(5, 1), _battle_stats(4, 1)) == 20.0
