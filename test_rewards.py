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

import pytest

from classes import (
    Alignment, GameState, GlobalInfo, Hex, Map, Position, SideInfo, Unit,
)
from rewards import (
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
    """Even with a per-step delta that ALSO has positive shaping, the
    leader_move_penalty subtracts once when leader_moved is True."""
    rf = WeightedReward(
        gold_killed_delta=0.01, leader_move_penalty=0.05,
        terminal_win=0, terminal_loss=0, terminal_draw=0,
        terminal_timeout=0, village_delta=0, damage_dealt=0,
        unit_recruited_cost=0, per_turn_penalty=0,
        invalid_action_penalty=0, min_enemy_distance_penalty=0,
    )
    delta = StepDelta(
        side=1, turn=1, action_type="move",
        enemy_gold_lost=14, leader_moved=True)
    assert rf(delta) == pytest.approx(0.01 * 14 - 0.05)


def test_weighted_reward_invalid_action_penalty():
    """Default invalid_action_penalty=0.001 fires when invalid_action
    is True. With 500 invalid actions we get -0.5 -- the exact cap
    the rewards docstring promises."""
    rf = WeightedReward()  # default weights
    inv = StepDelta(side=1, turn=1, action_type="recruit",
                    invalid_action=True)
    val = rf(inv)
    # Default min_enemy_distance is 0 and ongoing outcome contributes
    # nothing -> only the invalid penalty applies.
    assert val == pytest.approx(-0.001)


def test_weighted_reward_min_enemy_distance_penalty_scales():
    """Penalty proportional to distance: at distance 12 (typical
    starting separation), default 0.0001 weight gives -0.0012/step."""
    rf = WeightedReward()  # defaults
    # Strip the invalid_action flag and any other contributors.
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
    from rewards import UnitTypeBonus
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


def test_unit_type_bonus_stacks():
    """Multiple bonuses entries with different unit types each fire
    independently; same-type entries sum."""
    from rewards import UnitTypeBonus
    rf = WeightedReward(
        unit_type_bonuses=[
            UnitTypeBonus("Wose", weight=0.5),
            UnitTypeBonus("Wose", weight=0.1),       # stacks with above
            UnitTypeBonus("Elvish Fighter", weight=0.05),
        ],
        gold_killed_delta=0, village_delta=0, damage_dealt=0,
        unit_recruited_cost=0, per_turn_penalty=0,
        leader_move_penalty=0, invalid_action_penalty=0,
        min_enemy_distance_penalty=0,
        terminal_win=0, terminal_loss=0, terminal_draw=0,
        terminal_timeout=0,
    )
    # Recruited a Wose: 0.5 + 0.1 = 0.6.
    delta = StepDelta(side=1, turn=1, action_type="recruit",
                      units_recruited=("Wose",))
    assert rf(delta) == pytest.approx(0.6)


def test_turn_conditional_bonus_fires_in_window():
    """Predicate fires within turn_range[lo, hi]; bonus is awarded."""
    from rewards import TurnConditionalBonus
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
    from rewards import TurnConditionalBonus
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
    from rewards import TurnConditionalBonus
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
    from rewards import TurnConditionalBonus
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
    from rewards import TurnConditionalBonus
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
    from rewards import TurnConditionalBonus
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
# OpenerPolicy wrapper
# ---------------------------------------------------------------------

def test_opener_policy_fires_scripted_moves_then_falls_through():
    """The opener's two scripted moves fire on calls 1 and 2; call 3
    delegates to the base policy."""
    from tools.openers import Opener, OpenerPolicy, end_turn

    class _StubBase:
        def __init__(self):
            self.calls = 0
        def select_action(self, state, *, game_label="default"):
            self.calls += 1
            return {"type": "move", "_from_base": True}

    base = _StubBase()
    opener = Opener(name="t", moves=[end_turn(), end_turn()],
                    sides=(1, 2))
    policy = OpenerPolicy(base=base, opener=opener)
    state = _gs([_u("a", 1, 5, 5)])

    a1 = policy.select_action(state, game_label="g")
    a2 = policy.select_action(state, game_label="g")
    a3 = policy.select_action(state, game_label="g")
    assert a1 == {"type": "end_turn"}
    assert a2 == {"type": "end_turn"}
    assert a3.get("_from_base") is True
    assert base.calls == 1   # only the third call reached base


def test_opener_policy_falls_through_when_move_returns_none():
    """A move returning None doesn't advance the cursor: we delegate
    to base AND retry the same opener-step on the next decision."""
    from tools.openers import Opener, OpenerPolicy

    class _StubBase:
        def __init__(self):
            self.calls = 0
        def select_action(self, state, *, game_label="default"):
            self.calls += 1
            return {"type": "end_turn", "_from_base": True}

    fired_once = {"flag": False}
    def conditional_move(state, side):
        if not fired_once["flag"]:
            fired_once["flag"] = True
            return None             # first call: gate fails
        return {"type": "move", "_from_opener": True}  # second: gate passes

    base = _StubBase()
    opener = Opener(name="t", moves=[conditional_move], sides=(1,))
    policy = OpenerPolicy(base=base, opener=opener)
    state = _gs([_u("a", 1, 5, 5)])

    # First call: opener returns None -> fall through to base.
    a1 = policy.select_action(state, game_label="g")
    assert a1.get("_from_base") is True
    assert base.calls == 1
    # Second call: opener fires (cursor still at 0).
    a2 = policy.select_action(state, game_label="g")
    assert a2.get("_from_opener") is True
    # Third call: cursor advanced past end -> base.
    a3 = policy.select_action(state, game_label="g")
    assert a3.get("_from_base") is True


def test_opener_policy_per_side_filtering():
    """sides=(1,) means side 2 always goes to base."""
    from tools.openers import Opener, OpenerPolicy, end_turn

    class _StubBase:
        def select_action(self, state, *, game_label="default"):
            return {"type": "move", "_from_base": True}

    base = _StubBase()
    opener = Opener(name="t", moves=[end_turn()], sides=(1,))
    policy = OpenerPolicy(base=base, opener=opener)

    # Side 1 -> opener fires.
    s1 = _gs([_u("a", 1, 5, 5)], current_side=1)
    a1 = policy.select_action(s1, game_label="g")
    assert a1 == {"type": "end_turn"}

    # Side 2 -> bypassed, base fires.
    s2 = _gs([_u("a", 2, 5, 5)], current_side=2)
    a2 = policy.select_action(s2, game_label="g")
    assert a2.get("_from_base") is True


def test_opener_policy_reset_game():
    """reset_game(label) clears the cursor for that game so the next
    run starts opener fresh."""
    from tools.openers import Opener, OpenerPolicy, end_turn

    class _StubBase:
        def select_action(self, state, *, game_label="default"):
            return {"type": "move", "_from_base": True}

    base = _StubBase()
    opener = Opener(name="t", moves=[end_turn()], sides=(1,))
    policy = OpenerPolicy(base=base, opener=opener)
    state = _gs([_u("a", 1, 5, 5)])

    # Burn the opener.
    policy.select_action(state, game_label="g")
    a2 = policy.select_action(state, game_label="g")
    assert a2.get("_from_base") is True
    # Reset and re-fire.
    policy.reset_game("g")
    a3 = policy.select_action(state, game_label="g")
    assert a3 == {"type": "end_turn"}


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
