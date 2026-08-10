#!/usr/bin/env python3
"""Reward-shaping unit tests.

`rewards.compute_delta` and `WeightedReward.__call__` are the
trainer's signal source. They've accreted comments about specific
anti-patterns (the "0.22 flat-return plateau", the 985-game overnight
that climbed via 400+ invalid recruits per game) -- evidence that
nobody had unit-tested them before. These tests cover every
`StepDelta` field via hand-built (prev, new) GameState pairs so the
shaping math is locked in.

Why hand-built states (not via WesnothSim): we want each test to
isolate ONE field at a time. Going through a real sim would couple
multiple deltas together (e.g., a kill changes HP AND gold AND
villages-controlled if the dying unit was on a village). Hand-built
states let us verify, e.g., "when only our_gold_lost changes, only
that term contributes to the reward".
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from wesnoth_ai.classes import (
    Alignment, GameState, GlobalInfo, Map, Position, SideInfo, Unit,
)
from wesnoth_ai.rewards import (
    OUTCOME_DRAW, OUTCOME_LOSS, OUTCOME_ONGOING, OUTCOME_TIMEOUT, OUTCOME_WIN,
    StepDelta, WeightedReward, _action_had_visible_effect,
    _min_enemy_distance, compute_delta, hex_distance,
)


# ---------------------------------------------------------------------
# Builders -- minimal valid GameState pieces
# ---------------------------------------------------------------------

def _u(uid: str, side: int, x: int, y: int,
       *, hp=40, max_hp=40, mp=5, max_mp=5, xp=0, max_xp=32,
       cost=14, is_leader=False, name="Spearman",
       has_attacked=False) -> Unit:
    """Minimal Unit. Fills resistances / defenses / movement_costs with
    benign default lengths; tests don't read them."""
    return Unit(
        id=uid, name=name, name_id=0, side=side,
        is_leader=is_leader, position=Position(x, y),
        max_hp=max_hp, max_moves=max_mp, max_exp=max_xp,
        cost=cost, alignment=Alignment.NEUTRAL, levelup_names=[],
        current_hp=hp, current_moves=mp, current_exp=xp,
        has_attacked=has_attacked,
        attacks=[], resistances=[1.0]*6, defenses=[0.5]*14,
        movement_costs=[1]*14, abilities=set(), traits=set(),
        statuses=set(),
    )


def _gs(units: List[Unit],
        *, current_side=1, turn=1,
        side_villages: Optional[List[int]] = None,
        side_gold: Optional[List[int]] = None) -> GameState:
    """Build a minimal GameState. Two SideInfo entries by default with
    100g, 0 villages each (override via the kwargs)."""
    if side_villages is None:
        side_villages = [0, 0]
    if side_gold is None:
        side_gold = [100, 100]
    sides = [
        SideInfo(player=f"Side {i+1}", recruits=[],
                 current_gold=side_gold[i], base_income=2,
                 nb_villages_controlled=side_villages[i])
        for i in range(2)
    ]
    return GameState(
        game_id="test",
        map=Map(size_x=20, size_y=20, mask=set(), fog=set(),
                hexes=set(), units=set(units)),
        global_info=GlobalInfo(
            current_side=current_side, turn_number=turn,
            time_of_day="day", village_gold=2,
            village_upkeep=1, base_income=2,
        ),
        sides=sides,
    )


# ---------------------------------------------------------------------
# compute_delta: per-field coverage
# ---------------------------------------------------------------------

def test_prev_state_none_returns_zero_deltas():
    """First step of an episode: prev_state is None. All numeric
    deltas zero except possibly min_enemy_distance / outcome."""
    new = _gs([_u("a", 1, 5, 5), _u("b", 2, 10, 10)])
    delta = compute_delta(None, new, "move")
    assert delta.enemy_hp_lost == 0
    assert delta.enemy_gold_lost == 0
    assert delta.our_hp_lost == 0
    assert delta.our_gold_lost == 0
    assert delta.villages_gained == 0
    assert delta.villages_lost == 0
    assert delta.unit_recruited_cost == 0
    assert delta.leader_moved is False
    assert delta.invalid_action is False
    assert delta.outcome == OUTCOME_ONGOING


def test_enemy_hp_lost_when_enemy_takes_damage():
    """Enemy unit dropped 5 HP; enemy_hp_lost = 5, no other fields fire."""
    prev = _gs([_u("a", 1, 5, 5), _u("b", 2, 6, 5, hp=40)])
    new  = _gs([_u("a", 1, 5, 5), _u("b", 2, 6, 5, hp=35)])
    delta = compute_delta(prev, new, "attack")
    assert delta.enemy_hp_lost == 5
    assert delta.our_hp_lost == 0
    assert delta.enemy_gold_lost == 0
    assert delta.our_gold_lost == 0


def test_our_hp_lost_when_we_take_damage():
    """Counter-attack: our unit loses HP. our_hp_lost is recorded
    (the default WeightedReward doesn't read it, but it's available
    for custom reward fns)."""
    prev = _gs([_u("a", 1, 5, 5, hp=40), _u("b", 2, 6, 5)])
    new  = _gs([_u("a", 1, 5, 5, hp=33), _u("b", 2, 6, 5)])
    delta = compute_delta(prev, new, "attack")
    assert delta.our_hp_lost == 7
    assert delta.enemy_hp_lost == 0


def test_enemy_gold_lost_on_kill():
    """Enemy unit gone from new_state -> enemy_gold_lost += cost,
    enemy_hp_lost += unit's pre-death HP."""
    prev = _gs([_u("a", 1, 5, 5),
                _u("b", 2, 6, 5, hp=12, cost=20)])
    new  = _gs([_u("a", 1, 5, 5)])
    delta = compute_delta(prev, new, "attack")
    assert delta.enemy_gold_lost == 20
    assert delta.enemy_hp_lost == 12   # the dying unit's last HP counts as "damage dealt"
    assert delta.our_gold_lost == 0


def test_our_gold_lost_on_our_unit_dying():
    """Our unit gone -> our_gold_lost += cost. enemy_hp_lost NOT
    incremented (we lost HP, not them)."""
    prev = _gs([_u("a", 1, 5, 5, cost=17, hp=8),
                _u("b", 2, 6, 5)])
    new  = _gs([_u("b", 2, 6, 5)])
    delta = compute_delta(prev, new, "end_turn")  # death-on-counter
    assert delta.our_gold_lost == 17
    assert delta.enemy_hp_lost == 0
    assert delta.enemy_gold_lost == 0


def test_villages_gained_increments_correctly():
    """side 1 captured a village -> villages_gained = 1."""
    prev = _gs([_u("a", 1, 5, 5)],
               side_villages=[2, 1])
    new  = _gs([_u("a", 1, 5, 5)],
               side_villages=[3, 1])
    delta = compute_delta(prev, new, "move")
    assert delta.villages_gained == 1
    assert delta.villages_lost == 0


def test_villages_lost():
    """side 1 lost a village (the enemy captured one of ours)."""
    prev = _gs([_u("a", 1, 5, 5)],
               side_villages=[3, 0])
    new  = _gs([_u("a", 1, 5, 5)],
               side_villages=[2, 1])
    delta = compute_delta(prev, new, "end_turn")
    assert delta.villages_lost == 1
    assert delta.villages_gained == 0


def test_unit_recruited_cost_credited_only_on_success():
    """Recruit action + new unit on our side appears -> credit cost.
    Recruit action without a new unit (Wesnoth rejected) -> 0."""
    prev = _gs([_u("a", 1, 5, 5, is_leader=True)])
    new_success = _gs([_u("a", 1, 5, 5, is_leader=True),
                       _u("r", 1, 6, 5, name="Spearman")])
    delta = compute_delta(prev, new_success, "recruit",
                          recruit_cost=14)
    assert delta.unit_recruited_cost == 14

    # Same prev, no new unit -> 0 credit.
    new_failure = _gs([_u("a", 1, 5, 5, is_leader=True)])
    delta_fail = compute_delta(prev, new_failure, "recruit",
                               recruit_cost=14)
    assert delta_fail.unit_recruited_cost == 0
    # ... AND the failure case is also flagged as invalid_action because
    # nothing changed.
    assert delta_fail.invalid_action is True


def test_recruit_with_zero_cost_arg_does_not_credit():
    """If recruit_cost==0 the credit path is skipped even if a unit
    appeared. Defends against the lookup-fallback bug where an unknown
    unit type silently gets cost=0."""
    prev = _gs([_u("a", 1, 5, 5, is_leader=True)])
    new  = _gs([_u("a", 1, 5, 5, is_leader=True),
                _u("r", 1, 6, 5)])
    delta = compute_delta(prev, new, "recruit", recruit_cost=0)
    assert delta.unit_recruited_cost == 0


def test_leader_moved_flag():
    """Our leader changed hexes between snapshots."""
    prev = _gs([_u("ldr", 1, 5, 5, is_leader=True)])
    new  = _gs([_u("ldr", 1, 6, 5, is_leader=True)])
    delta = compute_delta(prev, new, "move")
    assert delta.leader_moved is True


def test_leader_not_moved_when_static():
    prev = _gs([_u("ldr", 1, 5, 5, is_leader=True),
                _u("g",   1, 6, 5)])
    new  = _gs([_u("ldr", 1, 5, 5, is_leader=True),
                _u("g",   1, 7, 5)])  # the GRUNT moved, not the leader
    delta = compute_delta(prev, new, "move")
    assert delta.leader_moved is False


def test_leader_dying_does_not_count_as_moved():
    """Leader gone in new_state -> moved flag stays False (the death
    is captured by terminal outcome / our_gold_lost, not by the
    leader-move term)."""
    prev = _gs([_u("ldr", 1, 5, 5, is_leader=True)])
    new  = _gs([])  # leader killed
    delta = compute_delta(prev, new, "end_turn")
    assert delta.leader_moved is False
    assert delta.our_gold_lost > 0


def test_invalid_action_when_state_unchanged():
    """`invalid_action` fires when the (prev, new) pair is structurally
    indistinguishable -- the action was rejected by Wesnoth and nothing
    moved."""
    prev = _gs([_u("a", 1, 5, 5)])
    new  = _gs([_u("a", 1, 5, 5)])
    delta = compute_delta(prev, new, "move")
    assert delta.invalid_action is True


def test_invalid_action_clears_when_side_swaps():
    """End-of-turn side swap is enough state change to NOT be invalid,
    even if no other field moved."""
    prev = _gs([_u("a", 1, 5, 5)], current_side=1)
    new  = _gs([_u("a", 1, 5, 5)], current_side=2)
    delta = compute_delta(prev, new, "end_turn")
    assert delta.invalid_action is False


# ---------------------------------------------------------------------
# min_enemy_distance: leader-fallback gradient
# ---------------------------------------------------------------------

def test_min_enemy_distance_uses_non_leader_when_present():
    """When a non-leader friendly is closer to the enemy than the
    leader, the non-leader's distance wins."""
    state = _gs([
        _u("ldr",   1, 5, 5, is_leader=True),
        _u("grunt", 1, 9, 5),
        _u("enemy", 2, 11, 5),
    ])
    # grunt (9,5) <-> enemy (11,5) hex distance = 2.
    # leader (5,5) <-> enemy (11,5) hex distance = 6.
    assert _min_enemy_distance(state, our_side=1) == 2


def test_min_enemy_distance_doubles_when_only_leader():
    """No non-leader allies -> falls back to leader's distance × 2.
    This keeps "recruit and send forward" strictly better than
    "leader walks to enemy" at every distance (see the rewards.py
    docstring for the rationale)."""
    state = _gs([
        _u("ldr",   1, 5, 5, is_leader=True),
        _u("enemy", 2, 11, 5),
    ])
    # leader-only distance = 6; doubled = 12.
    assert _min_enemy_distance(state, our_side=1) == 12


def test_min_enemy_distance_zero_when_no_enemies():
    """Fallback edge case: no enemy units visible."""
    state = _gs([_u("ldr", 1, 5, 5, is_leader=True)])
    assert _min_enemy_distance(state, our_side=1) == 0


# ---------------------------------------------------------------------
# WeightedReward summation
# ---------------------------------------------------------------------

def test_weighted_reward_outcome_terms():
    """terminal_win/loss/draw/timeout each contribute their full
    weight when outcome matches."""
    rf = WeightedReward(
        terminal_win=+1.0, terminal_loss=-1.0,
        terminal_draw=0.0, terminal_timeout=-0.1,
        # zero out the per-step shaping so we read the terminal alone.
        gold_killed_delta=0, village_delta=0, damage_dealt=0,
        unit_recruited_cost=0, per_turn_penalty=0,
        leader_move_penalty=0, invalid_action_penalty=0,
        min_enemy_distance_penalty=0,
    )
    base = StepDelta(side=1, turn=1, action_type="end_turn")
    assert rf(StepDelta(**{**base.__dict__, "outcome": OUTCOME_WIN})) == pytest.approx(+1.0)
    assert rf(StepDelta(**{**base.__dict__, "outcome": OUTCOME_LOSS})) == pytest.approx(-1.0)
    assert rf(StepDelta(**{**base.__dict__, "outcome": OUTCOME_DRAW})) == pytest.approx(0.0)
    assert rf(StepDelta(**{**base.__dict__, "outcome": OUTCOME_TIMEOUT})) == pytest.approx(-0.1)
    assert rf(StepDelta(**{**base.__dict__, "outcome": OUTCOME_ONGOING})) == pytest.approx(0.0)


def test_weighted_reward_gold_killed_delta_uses_signed_diff():
    """gold_killed_delta scales (enemy_gold_lost - our_gold_lost),
    so a 14g kill while losing a 17g ally nets a NEGATIVE shaping
    even though both fields are non-negative on their own."""
    rf = WeightedReward(
        gold_killed_delta=0.01,
        terminal_win=0, terminal_loss=0, terminal_draw=0,
        terminal_timeout=0, village_delta=0, damage_dealt=0,
        unit_recruited_cost=0, per_turn_penalty=0,
        leader_move_penalty=0, invalid_action_penalty=0,
        min_enemy_distance_penalty=0,
    )
    delta = StepDelta(side=1, turn=1, action_type="attack",
                      enemy_gold_lost=14, our_gold_lost=17)
    assert rf(delta) == pytest.approx(0.01 * (14 - 17))


def test_weighted_reward_leader_move_penalty_is_charged_once():
    """Sign convention (post-2026-04-29): the configured value IS
    the contribution sign-and-all. Negative value -> negative
    contribution, fires once per leader_moved=True step.
    """
    rf = WeightedReward(
        gold_killed_delta=0.01, leader_move_penalty=-0.05,
        terminal_win=0, terminal_loss=0, terminal_draw=0,
        terminal_timeout=0, village_delta=0, damage_dealt=0,
        unit_recruited_cost=0, per_turn_penalty=0,
        invalid_action_penalty=0, min_enemy_distance_penalty=0,
    )
    delta = StepDelta(
        side=1, turn=1, action_type="move",
        enemy_gold_lost=14, leader_moved=True)
    # 0.01 * 14 + (-0.05) = 0.14 - 0.05 = 0.09
    assert rf(delta) == pytest.approx(0.01 * 14 + (-0.05))


def test_leader_move_penalty_turn_range_excludes_outside_window():
    """When `leader_move_penalty_turn_range=(5, 15)` is set, the
    penalty fires only for turns 5-15 inclusive. Earlier and later
    leader moves contribute nothing (only the gold-killed term)."""
    rf = WeightedReward(
        gold_killed_delta=0.01, leader_move_penalty=-0.05,
        leader_move_penalty_turn_range=(5, 15),
        terminal_win=0, terminal_loss=0, terminal_draw=0,
        terminal_timeout=0, village_delta=0, damage_dealt=0,
        unit_recruited_cost=0, per_turn_penalty=0,
        invalid_action_penalty=0, min_enemy_distance_penalty=0,
    )
    common = dict(side=1, action_type="move", leader_moved=True)
    # Before window: penalty skipped.
    d_pre = StepDelta(turn=4, **common)
    assert rf(d_pre) == pytest.approx(0.0)
    # In window (low end).
    d_lo = StepDelta(turn=5, **common)
    assert rf(d_lo) == pytest.approx(-0.05)
    # In window (mid).
    d_mid = StepDelta(turn=10, **common)
    assert rf(d_mid) == pytest.approx(-0.05)
    # In window (high end).
    d_hi = StepDelta(turn=15, **common)
    assert rf(d_hi) == pytest.approx(-0.05)
    # After window: skipped.
    d_post = StepDelta(turn=16, **common)
    assert rf(d_post) == pytest.approx(0.0)


def test_leader_move_penalty_turn_range_open_ended():
    """`(None, 10)` = no lower bound; `(10, None)` = no upper bound."""
    rf_early = WeightedReward(
        leader_move_penalty=-0.05,
        leader_move_penalty_turn_range=(None, 10),
        terminal_win=0, terminal_loss=0, terminal_draw=0,
        terminal_timeout=0, gold_killed_delta=0, village_delta=0,
        damage_dealt=0, unit_recruited_cost=0, per_turn_penalty=0,
        invalid_action_penalty=0, min_enemy_distance_penalty=0,
    )
    rf_late = WeightedReward(
        leader_move_penalty=-0.05,
        leader_move_penalty_turn_range=(10, None),
        terminal_win=0, terminal_loss=0, terminal_draw=0,
        terminal_timeout=0, gold_killed_delta=0, village_delta=0,
        damage_dealt=0, unit_recruited_cost=0, per_turn_penalty=0,
        invalid_action_penalty=0, min_enemy_distance_penalty=0,
    )
    common = dict(side=1, action_type="move", leader_moved=True)
    # Early fires for turn 1, skips for turn 11.
    assert rf_early(StepDelta(turn=1, **common)) == pytest.approx(-0.05)
    assert rf_early(StepDelta(turn=11, **common)) == pytest.approx(0.0)
    # Late skips for turn 9, fires for turn 11.
    assert rf_late(StepDelta(turn=9, **common)) == pytest.approx(0.0)
    assert rf_late(StepDelta(turn=11, **common)) == pytest.approx(-0.05)


def test_weighted_reward_invalid_action_penalty():
    """Default `invalid_action_penalty=-0.001` fires when
    invalid_action is True. With 500 invalid actions we get -0.5 --
    the exact cap the rewards docstring promises."""
    rf = WeightedReward()  # default weights
    inv = StepDelta(side=1, turn=1, action_type="recruit",
                    invalid_action=True)
    val = rf(inv)
    # Default min_enemy_distance is 0 and ongoing outcome contributes
    # nothing -> only the invalid penalty (-0.001) applies.
    assert val == pytest.approx(-0.001)


def test_weighted_reward_min_enemy_distance_penalty_scales():
    """Negative coefficient × distance: at distance 12 with default
    -0.0001 we get -0.0012/step."""
    rf = WeightedReward()  # defaults (negative penalty fields)
    delta = StepDelta(side=1, turn=1, action_type="end_turn",
                      min_enemy_distance=12)
    val = rf(delta)
    assert val == pytest.approx(-0.0001 * 12)


# ---------------------------------------------------------------------
# hex_distance: parity + symmetry
# ---------------------------------------------------------------------

def test_hex_distance_identity_zero():
    """d(a, a) == 0 for several samples."""
    for x in range(0, 10):
        for y in range(0, 10):
            assert hex_distance(x, y, x, y) == 0


def test_hex_distance_symmetry():
    """d(a, b) == d(b, a) -- standard metric property."""
    for ax, ay, bx, by in [
        (0, 0, 5, 5), (3, 7, 11, 2), (4, 4, 4, 8),
        (0, 0, 1, 0), (1, 0, 2, 1), (5, 3, 6, 4),
    ]:
        d_ab = hex_distance(ax, ay, bx, by)
        d_ba = hex_distance(bx, by, ax, ay)
        assert d_ab == d_ba, f"asymmetric: ({ax},{ay})↔({bx},{by}): {d_ab} != {d_ba}"


def test_hex_distance_adjacent_hexes_are_one():
    """Wesnoth's six-neighbour adjacency rule: in odd-q offset, the
    six neighbours of (x, y) are at distance 1. Even/odd column has
    different y-offsets."""
    # Even column (x=4): neighbours are (4,3), (4,5), (3,3), (3,4),
    #                                   (5,3), (5,4) (above + lateral).
    for nx, ny in [(4, 3), (4, 5), (3, 3), (3, 4), (5, 3), (5, 4)]:
        assert hex_distance(4, 4, nx, ny) == 1, f"({nx},{ny}) should be adjacent to (4,4)"
    # Odd column (x=5): neighbours are (5,3), (5,5), (4,4), (4,5),
    #                                  (6,4), (6,5).
    for nx, ny in [(5, 3), (5, 5), (4, 4), (4, 5), (6, 4), (6, 5)]:
        assert hex_distance(5, 4, nx, ny) == 1, f"({nx},{ny}) should be adjacent to (5,4)"


def test_hex_distance_long_range():
    """A handful of fixed distances cross-checked against the formula."""
    # Same row, far apart: pure horizontal.
    assert hex_distance(0, 0, 10, 0) == 10
    # Same column.
    assert hex_distance(5, 0, 5, 8) == 8


# ---------------------------------------------------------------------
# _action_had_visible_effect (tracks invalid_action gate)
# ---------------------------------------------------------------------

def test_visible_effect_picks_up_position_change():
    prev = _gs([_u("a", 1, 5, 5)])
    new  = _gs([_u("a", 1, 6, 5)])
    assert _action_had_visible_effect(prev, new)


def test_visible_effect_picks_up_hp_change():
    prev = _gs([_u("a", 1, 5, 5, hp=20)])
    new  = _gs([_u("a", 1, 5, 5, hp=15)])
    assert _action_had_visible_effect(prev, new)


def test_visible_effect_picks_up_has_attacked_flip():
    prev = _gs([_u("a", 1, 5, 5, has_attacked=False)])
    new  = _gs([_u("a", 1, 5, 5, has_attacked=True)])
    assert _action_had_visible_effect(prev, new)


def test_visible_effect_negative_when_truly_static():
    prev = _gs([_u("a", 1, 5, 5)])
    new  = _gs([_u("a", 1, 5, 5)])
    assert not _action_had_visible_effect(prev, new)


# ---------------------------------------------------------------------
# Integration: combined fields produce expected reward
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Customizability hooks: per-unit-type and turn-conditional bonuses
# ---------------------------------------------------------------------

def test_units_recruited_populated_on_success():
    """compute_delta should populate `units_recruited` with the unit
    name of any unit that newly appeared on our side this step."""
    prev = _gs([_u("a", 1, 5, 5, is_leader=True)])
    new  = _gs([_u("a", 1, 5, 5, is_leader=True),
                _u("r", 1, 6, 5, name="Wose")])
    delta = compute_delta(prev, new, "recruit", recruit_cost=17)
    assert delta.units_recruited == ("Wose",)


def test_units_recruited_empty_on_recruit_fail():
    """No unit appeared -> empty tuple. (Caller can't tell from the
    cost field alone whether the recruit succeeded.)"""
    prev = _gs([_u("a", 1, 5, 5, is_leader=True)])
    new  = _gs([_u("a", 1, 5, 5, is_leader=True)])  # no new unit
    delta = compute_delta(prev, new, "recruit", recruit_cost=17)
    assert delta.units_recruited == ()


def test_unit_type_bonus_fires_on_match():
    """A UnitTypeBonus for 'Wose' should add weight=0.5 to the reward
    of a step that recruited a Wose. Other unit types are unaffected."""
    from wesnoth_ai.rewards import UnitTypeBonus
    rf = WeightedReward(
        unit_type_bonuses=[UnitTypeBonus("Wose", weight=0.5)],
        # zero everything else for a clean read.
        gold_killed_delta=0, village_delta=0, damage_dealt=0,
        unit_recruited_cost=0, per_turn_penalty=0,
        leader_move_penalty=0, invalid_action_penalty=0,
        min_enemy_distance_penalty=0,
        terminal_win=0, terminal_loss=0, terminal_draw=0,
        terminal_timeout=0,
    )
    delta = StepDelta(side=1, turn=1, action_type="recruit",
                      units_recruited=("Wose",))
    assert rf(delta) == pytest.approx(0.5)
    # Different unit -> no bonus
    delta2 = StepDelta(side=1, turn=1, action_type="recruit",
                       units_recruited=("Elvish Fighter",))
    assert rf(delta2) == pytest.approx(0.0)


def test_turn_conditional_bonus_fires_in_window():
    """Predicate fires within turn_range[lo, hi]; bonus is awarded."""
    from wesnoth_ai.rewards import TurnConditionalBonus
    rf = WeightedReward(
        turn_conditional_bonuses=[
            TurnConditionalBonus(
                name="early_village",
                turn_range=(1, 3),
                predicate=lambda st, side: True,    # always-true
                weight=1.0, once=False,
            ),
        ],
        gold_killed_delta=0, village_delta=0, damage_dealt=0,
        unit_recruited_cost=0, per_turn_penalty=0,
        leader_move_penalty=0, invalid_action_penalty=0,
        min_enemy_distance_penalty=0,
        terminal_win=0, terminal_loss=0, terminal_draw=0,
        terminal_timeout=0,
    )
    # Need post_state attached (else the predicate is silently
    # skipped). Build a dummy.
    state = _gs([_u("a", 1, 5, 5)])
    delta = StepDelta(side=1, turn=2, action_type="move",
                      post_state=state, game_label="g")
    assert rf(delta) == pytest.approx(1.0)


def test_turn_conditional_bonus_skipped_outside_window():
    """turn=5 but window is (1, 3) -> no bonus."""
    from wesnoth_ai.rewards import TurnConditionalBonus
    rf = WeightedReward(
        turn_conditional_bonuses=[
            TurnConditionalBonus(
                name="x", turn_range=(1, 3),
                predicate=lambda st, side: True,
                weight=1.0, once=False),
        ],
        gold_killed_delta=0, village_delta=0, damage_dealt=0,
        unit_recruited_cost=0, per_turn_penalty=0,
        leader_move_penalty=0, invalid_action_penalty=0,
        min_enemy_distance_penalty=0,
        terminal_win=0, terminal_loss=0, terminal_draw=0,
        terminal_timeout=0,
    )
    state = _gs([_u("a", 1, 5, 5)])
    delta = StepDelta(side=1, turn=5, action_type="move",
                      post_state=state, game_label="g")
    assert rf(delta) == pytest.approx(0.0)


def test_turn_conditional_bonus_once_per_game():
    """With once=True, the bonus fires exactly ONCE per (game, side).
    Subsequent same-game/side calls are skipped even when the
    predicate keeps returning True."""
    from wesnoth_ai.rewards import TurnConditionalBonus
    rf = WeightedReward(
        turn_conditional_bonuses=[
            TurnConditionalBonus(
                name="x", turn_range=(1, 10),
                predicate=lambda st, side: True,
                weight=1.0, once=True),
        ],
        gold_killed_delta=0, village_delta=0, damage_dealt=0,
        unit_recruited_cost=0, per_turn_penalty=0,
        leader_move_penalty=0, invalid_action_penalty=0,
        min_enemy_distance_penalty=0,
        terminal_win=0, terminal_loss=0, terminal_draw=0,
        terminal_timeout=0,
    )
    state = _gs([_u("a", 1, 5, 5)])
    d1 = StepDelta(side=1, turn=1, action_type="move",
                   post_state=state, game_label="g1")
    d2 = StepDelta(side=1, turn=2, action_type="move",
                   post_state=state, game_label="g1")
    assert rf(d1) == pytest.approx(1.0)
    assert rf(d2) == pytest.approx(0.0)
    # Different side: still gets the bonus on its own first fire.
    d3 = StepDelta(side=2, turn=1, action_type="move",
                   post_state=state, game_label="g1")
    assert rf(d3) == pytest.approx(1.0)
    # Different game label: side=1 starts fresh.
    d4 = StepDelta(side=1, turn=1, action_type="move",
                   post_state=state, game_label="g2")
    assert rf(d4) == pytest.approx(1.0)


def test_turn_conditional_bonus_reset_game_state():
    """`reset_game_state(game_label)` clears the fired-set for that
    game, letting `once=True` fire again."""
    from wesnoth_ai.rewards import TurnConditionalBonus
    rf = WeightedReward(
        turn_conditional_bonuses=[
            TurnConditionalBonus(
                name="x", turn_range=(1, 10),
                predicate=lambda st, side: True,
                weight=1.0, once=True),
        ],
        gold_killed_delta=0, village_delta=0, damage_dealt=0,
        unit_recruited_cost=0, per_turn_penalty=0,
        leader_move_penalty=0, invalid_action_penalty=0,
        min_enemy_distance_penalty=0,
        terminal_win=0, terminal_loss=0, terminal_draw=0,
        terminal_timeout=0,
    )
    state = _gs([_u("a", 1, 5, 5)])
    d1 = StepDelta(side=1, turn=1, action_type="move",
                   post_state=state, game_label="g")
    assert rf(d1) == pytest.approx(1.0)
    assert rf(d1) == pytest.approx(0.0)        # already fired
    rf.reset_game_state("g")
    assert rf(d1) == pytest.approx(1.0)        # fires again


def test_turn_conditional_bonus_silent_without_post_state():
    """Bonus is silently inert if post_state is None -- compute_delta
    didn't attach it. Defensive default: don't crash training when
    a caller forgets attach_post_state=True."""
    from wesnoth_ai.rewards import TurnConditionalBonus
    rf = WeightedReward(
        turn_conditional_bonuses=[
            TurnConditionalBonus(
                name="x", turn_range=(1, 10),
                predicate=lambda st, side: 1/0,    # would explode
                weight=1.0, once=False),
        ],
        gold_killed_delta=0, village_delta=0, damage_dealt=0,
        unit_recruited_cost=0, per_turn_penalty=0,
        leader_move_penalty=0, invalid_action_penalty=0,
        min_enemy_distance_penalty=0,
        terminal_win=0, terminal_loss=0, terminal_draw=0,
        terminal_timeout=0,
    )
    delta = StepDelta(side=1, turn=2, action_type="move",
                      post_state=None)  # not attached
    # No crash, no bonus.
    assert rf(delta) == pytest.approx(0.0)


def test_turn_conditional_bonus_predicate_exception_caught():
    """If the predicate raises, we treat as not-fired and continue
    -- a bad predicate shouldn't crash training."""
    from wesnoth_ai.rewards import TurnConditionalBonus
    def boom(st, side):
        raise ValueError("oops")
    rf = WeightedReward(
        turn_conditional_bonuses=[
            TurnConditionalBonus(
                name="x", turn_range=(1, 10),
                predicate=boom, weight=1.0, once=False),
        ],
        gold_killed_delta=0, village_delta=0, damage_dealt=0,
        unit_recruited_cost=0, per_turn_penalty=0,
        leader_move_penalty=0, invalid_action_penalty=0,
        min_enemy_distance_penalty=0,
        terminal_win=0, terminal_loss=0, terminal_draw=0,
        terminal_timeout=0,
    )
    state = _gs([_u("a", 1, 5, 5)])
    delta = StepDelta(side=1, turn=2, action_type="move",
                      post_state=state)
    assert rf(delta) == pytest.approx(0.0)


def test_compute_delta_attach_post_state_flag():
    """attach_post_state=True puts new_state on the StepDelta;
    default False leaves it None to avoid retention overhead."""
    prev = _gs([_u("a", 1, 5, 5)])
    new  = _gs([_u("a", 1, 5, 5)])

    # Default: no attachment.
    d_off = compute_delta(prev, new, "move")
    assert d_off.post_state is None

    # Opt in: new_state attached by reference.
    d_on = compute_delta(prev, new, "move", attach_post_state=True)
    assert d_on.post_state is new


def test_compute_delta_propagates_game_label():
    """game_label kwarg should land on the StepDelta."""
    prev = _gs([_u("a", 1, 5, 5)])
    new  = _gs([_u("a", 1, 5, 5)])
    delta = compute_delta(prev, new, "move", game_label="iter3_g7")
    assert delta.game_label == "iter3_g7"


# ---------------------------------------------------------------------
# Predicate registry + JSON/YAML config loader
# ---------------------------------------------------------------------

def test_predicate_registry_has_builtins():
    """The four built-in predicates we ship register on import."""
    from wesnoth_ai.rewards import available_predicates, get_predicate
    names = available_predicates()
    for n in ("leader_on_village", "leader_on_keep",
              "controls_majority_villages", "no_units_lost"):
        assert n in names, f"missing built-in predicate: {n}"
    # get_predicate returns a callable
    pred = get_predicate("leader_on_keep")
    assert callable(pred)


def test_predicate_unknown_raises():
    from wesnoth_ai.rewards import get_predicate
    with pytest.raises(KeyError, match="Unknown predicate"):
        get_predicate("not_a_real_predicate")


def test_register_predicate_overrides_existing():
    """Last registration wins. Custom predicates can shadow built-ins
    for one-off experiments."""
    from wesnoth_ai.rewards import register_predicate, get_predicate
    sentinel_calls = {"count": 0}
    def custom(state, side):
        sentinel_calls["count"] += 1
        return True
    register_predicate("__test_override__", custom)
    pred = get_predicate("__test_override__")
    state = _gs([_u("a", 1, 5, 5)])
    assert pred(state, 1) is True
    assert sentinel_calls["count"] == 1


def test_load_reward_config_json(tmp_path):
    """Round-trip a JSON config: scalars override defaults, lists
    populate, predicates resolve through the registry."""
    import json
    from wesnoth_ai.rewards import load_reward_config, UnitTypeBonus, TurnConditionalBonus

    cfg = {
        "gold_killed_delta": 0.05,         # override
        "leader_move_penalty": 0.0,        # override (off)
        "unit_type_bonuses": [
            {"unit_type": "Wose", "weight": 0.5},
            {"unit_type": "Elvish Fighter", "weight": 0.05},
        ],
        "turn_conditional_bonuses": [
            {
                "name": "early_keep",
                "turn_range": [1, 3],
                "predicate": "leader_on_keep",
                "weight": 0.2,
                "once": True,
            },
        ],
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))

    rf = load_reward_config(p)
    assert rf.gold_killed_delta == pytest.approx(0.05)
    assert rf.leader_move_penalty == pytest.approx(0.0)
    # WeightedReward defaults preserved for fields the config didn't touch
    assert rf.terminal_win == pytest.approx(1.0)
    # Bonus lists populated correctly
    assert len(rf.unit_type_bonuses) == 2
    assert isinstance(rf.unit_type_bonuses[0], UnitTypeBonus)
    assert rf.unit_type_bonuses[0].unit_type == "Wose"
    assert rf.unit_type_bonuses[0].weight == pytest.approx(0.5)
    assert len(rf.turn_conditional_bonuses) == 1
    tcb = rf.turn_conditional_bonuses[0]
    assert isinstance(tcb, TurnConditionalBonus)
    assert tcb.turn_range == (1, 3)
    assert tcb.weight == pytest.approx(0.2)
    assert callable(tcb.predicate)


def test_load_reward_config_unknown_predicate_raises(tmp_path):
    import json
    from wesnoth_ai.rewards import load_reward_config

    cfg = {
        "turn_conditional_bonuses": [
            {"name": "x", "turn_range": [1, 3],
             "predicate": "no_such_predicate", "weight": 1.0},
        ],
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    with pytest.raises(KeyError, match="Unknown predicate"):
        load_reward_config(p)


def test_load_reward_config_unknown_scalar_key_raises(tmp_path):
    """A typo'd scalar key (e.g. 'gold_kileld_delta') is rejected at
    load time -- prevents silent misconfiguration."""
    import json
    from wesnoth_ai.rewards import load_reward_config

    cfg = {"gold_kileld_delta": 0.1}    # typo
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="unknown key"):
        load_reward_config(p)


def test_load_reward_config_bad_extension_raises(tmp_path):
    from wesnoth_ai.rewards import load_reward_config
    p = tmp_path / "cfg.txt"
    p.write_text("x")
    with pytest.raises(ValueError, match="unsupported"):
        load_reward_config(p)


def test_load_reward_config_round_trip_predicate_fires(tmp_path):
    """End-to-end: config loaded predicate actually fires in a
    realistic step. Uses the leader_on_keep built-in -- our test
    GameState builder doesn't make villages/keeps in the hex set, so
    we register a custom always-true predicate to keep the test
    self-contained."""
    import json
    from wesnoth_ai.rewards import (
        load_reward_config, register_predicate,
    )
    register_predicate("__always_true__", lambda st, side: True)

    cfg = {
        # Zero out all defaults so we read the bonus alone.
        "gold_killed_delta": 0.0,
        "village_delta": 0.0,
        "damage_dealt": 0.0,
        "unit_recruited_cost": 0.0,
        "leader_move_penalty": 0.0,
        "invalid_action_penalty": 0.0,
        "min_enemy_distance_penalty": 0.0,
        "terminal_win": 0.0,
        "terminal_loss": 0.0,
        "terminal_draw": 0.0,
        "terminal_timeout": 0.0,
        "turn_conditional_bonuses": [
            {
                "name": "always_fires",
                "turn_range": [1, 100],
                "predicate": "__always_true__",
                "weight": 0.7,
                "once": False,
            },
        ],
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    rf = load_reward_config(p)

    state = _gs([_u("a", 1, 5, 5)])
    delta = StepDelta(side=1, turn=2, action_type="move",
                      post_state=state, game_label="g")
    assert rf(delta) == pytest.approx(0.7)


def test_combined_kill_plus_village_gain():
    """A move that ends on a village AND somehow killed an enemy
    earlier in the same step (rare but happens via plague/leadership
    chains): both shaping terms fire additively."""
    prev = _gs([
        _u("a", 1, 5, 5),
        _u("v", 2, 6, 5, cost=20, hp=15),
    ], side_villages=[1, 2])
    new  = _gs([
        _u("a", 1, 6, 5),  # we moved AND captured the dying enemy's hex
    ], side_villages=[2, 1])  # we gained a village; they lost one
    delta = compute_delta(prev, new, "move")
    assert delta.enemy_gold_lost == 20
    assert delta.enemy_hp_lost == 15  # dying unit's last HP counted
    assert delta.villages_gained == 1
    rf = WeightedReward()
    val = rf(delta)
    # Just verify the sign is positive and non-trivial (combat + village
    # both contribute), not the exact arithmetic -- WeightedReward's
    # other terms (min_enemy_distance, etc.) fluctuate.
    assert val > 0.05


# ---------------------------------------------------------------------
# --reward-config x --mcts refusal (F5 ruling, 2026-08-10)
# ---------------------------------------------------------------------

def test_reward_config_refused_under_mcts(tmp_path, capsys):
    """Shaping rewards are structurally inert under --mcts
    (MCTSPolicy.observe is a no-op), so a non-default --reward-config
    there must be REFUSED at arg validation -- not silently ignored
    (the trap that hid the weight_gold=0 non-fix) and not merely
    warned about. --mcts is the default; --reinforce + the same
    config must still parse past this gate."""
    from tools.sim_self_play import main as ssp_main

    cfg = tmp_path / "custom_rewards.json"
    cfg.write_text('{"weight_gold_killed": 0.5}', encoding="utf-8")

    with pytest.raises(SystemExit) as ei:
        ssp_main(["prog", "--reward-config", str(cfg)])
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "no effect under --mcts" in err
    assert "--reinforce" in err
