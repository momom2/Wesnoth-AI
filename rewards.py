"""Reward function for the Wesnoth AI trainer.

Design goals (from the user, Phase 3 planning):

  - Customizable. Experimenting with "incentivize unorthodox strategies"
    should mean editing a dataclass field, not touching the trainer.
  - Maintainable. No scope creep: one weighted sum, all terms local,
    no plumbing needed anywhere else.
  - Dense enough to train on. Pure terminal ±1 is too sparse with
    MAX_ACTIONS_PER_GAME=200 and an untrained policy — leader kills
    will be a faraway tactical dream early on. The main per-turn
    signal is **gold-killed-delta**: gold-value of enemy units we
    killed since last step, minus gold-value of our units lost. This
    is a better proxy than raw HP damage because it accounts for
    hit-points, resistances, and trait rolls (a 40-HP Fighter lost is
    worse than a 30-HP Guardsman even if the HP numbers look similar).
  - Also small positive terms for village capture, damage dealt, and
    the cost of newly-recruited units (encourage army-building and
    engagement in early training). These sum to << 1 relative to the
    terminal reward so they don't dominate.

The module exposes:
  - `WeightedReward` dataclass — the weighted-sum reward function.
  - `StepDelta` — the observable change between two game states that
    the trainer computes and feeds to reward fns.
  - `RewardFn` Protocol — any `StepDelta → float` callable is a valid
    reward fn; WeightedReward is the default.

The trainer computes StepDeltas itself (comparing prev_state and
new_state snapshots it already has); reward fns just consume them.
That keeps deltas as a shared interchange so custom reward fns don't
reinvent state diffing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from classes import GameState


# Outcome of a game as seen from ONE side's perspective.
OUTCOME_ONGOING = "ongoing"
OUTCOME_WIN     = "win"
OUTCOME_LOSS    = "loss"
OUTCOME_DRAW    = "draw"
OUTCOME_TIMEOUT = "timeout"   # hit MAX_ACTIONS_PER_GAME without a victor


@dataclass
class StepDelta:
    """Everything a reward function might want about ONE Python-side
    decision step, from the perspective of the side that just acted.

    The trainer populates this by diffing the pre-action and post-action
    GameState snapshots. Fields are named from *our* (the actor's) view:
    `enemy_gold_lost` is good for us, `our_gold_lost` is bad.
    """

    # Step metadata.
    side: int                        # 1 or 2, the side acting
    turn: int                        # game-turn number at this step
    action_type: str                 # "move", "attack", "recruit", "end_turn"

    # Per-step numeric deltas (all non-negative; signs come from weights).
    enemy_hp_lost:   int = 0         # total HP our enemies lost this step
    our_hp_lost:     int = 0         # total HP we lost this step
    enemy_gold_lost: int = 0         # sum(cost) of enemy units that died
    our_gold_lost:   int = 0         # sum(cost) of our units that died
    villages_gained: int = 0         # villages we captured this step
    villages_lost:   int = 0         # villages we lost this step
    unit_recruited_cost: int = 0     # cost of a unit we just recruited (0 otherwise)
    leader_moved:    bool = False    # our leader changed hexes this step

    # Invalid-action flag: True when the action produced NO observable
    # state change (no unit moved, no HP changed, no unit appeared, no
    # village changed, no side swap). Used to penalize recruit-spam /
    # invalid-move attempts that the custom AI stage silently ignores.
    invalid_action:  bool = False

    # Smallest hex distance between any friendly unit and any enemy unit
    # visible to `side`. Used by the reward to apply an ongoing penalty
    # proportional to distance, pushing the policy to at least put ONE
    # unit in contact with an enemy. Zero if either side has no visible
    # units (no meaningful distance to compute).
    min_enemy_distance: int = 0

    # Terminal flag.
    outcome: str = OUTCOME_ONGOING   # one of the OUTCOME_* constants


class RewardFn(Protocol):
    """Any callable mapping a StepDelta to a scalar reward."""

    def __call__(self, delta: StepDelta) -> float: ...


@dataclass
class WeightedReward:
    """Default reward: weighted sum of the StepDelta fields.

    All weights default to 0 except the ones we want on for Phase 3.
    Override by constructing with different values:

        reward_fn = WeightedReward(village_delta=0.2, damage_dealt=0.005)

    Scale guidance for the defaults (assumes MAX_ACTIONS_PER_GAME=200,
    100g starting gold, avg unit cost ~17):

      - terminal_win/loss ±1.0 is the anchor: a won game earns ~1 total.
      - gold_killed_delta 0.01 means a clean 14g fighter kill earns
        0.14 and a 20g Drake Clasher kill earns 0.2. Across a
        competitive match, total should be O(0.5), comparable to the
        terminal signal.
      - village_delta 0.05 per village: a 2-village map with typical
        back-and-forth ≈ 0.1 total, small relative to combat.
      - damage_dealt 0.0005 per HP: a full Drake Clasher health bar
        (43 HP) knocked to zero earns 0.02 before the kill bonus —
        noise level, just to distinguish "engaged" from "idled".
      - unit_recruited_cost 0.001 per gold: 14g fighter earns 0.014.
        Tiny; just biases toward spending gold rather than hoarding.
      - per_turn_penalty 0.0: enable only if we observe policies
        learning to run out the clock.
    """

    # Terminal rewards.
    terminal_win:     float = +1.0
    terminal_loss:    float = -1.0
    terminal_draw:    float =  0.0
    terminal_timeout: float =  0.0

    # Per-step shaping. See docstring for scale guidance.
    gold_killed_delta:    float = 0.01     # (enemy_gold_lost - our_gold_lost)
    village_delta:        float = 0.05     # (villages_gained - villages_lost)
    damage_dealt:         float = 0.0005   # enemy_hp_lost only
    unit_recruited_cost:  float = 0.001
    per_turn_penalty:     float = 0.0      # applied once per Python-side step

    # Penalty subtracted whenever our leader changed hexes this step.
    # Purpose: break the "every game returns ~0.22" uniformity that
    # prevents policy-gradient updates. Random policies that happen to
    # recruit more (leader stays on the keep) now get measurably
    # better reward than those that wander the leader around — which
    # creates the inter-episode variance the policy gradient needs.
    # Scaled so a game with ~20 leader moves has ~0.2 total penalty,
    # comparable to the ~0.1 recruit/village shaping, so the gradient
    # nudges the policy toward "recruit more, move leader less".
    leader_move_penalty:  float = 0.01

    # Flat penalty for actions Wesnoth silently rejected (state didn't
    # change). Added after the overnight run showed the policy spamming
    # ~500 invalid recruits per game — our custom AI stage doesn't
    # blacklist-on-failure, so there's no engine-side back-pressure on
    # garbage actions. With 500-action games a 0.001 per-invalid
    # penalty caps at -0.5 for a fully-wasted game, strong enough to
    # be the dominant shaping term for a stuck-spamming policy but not
    # enough to drown the terminal ±1 for a game that actually plays.
    invalid_action_penalty: float = 0.001

    # Per-step penalty proportional to the minimum hex distance between
    # any friendly unit and any enemy unit. Purpose: random exploration
    # basically never stumbles into combat on a 16×20 map with ~12 hex
    # starting distance. Keeping a unit close costs nothing; leaving
    # them all far away accumulates a steady negative reward, so the
    # policy gets gradient toward "send at least ONE unit to the
    # enemy's zone". 0.0001 × 12 × ~500 actions ≈ -0.6 for a game
    # that never closes — roughly the payoff of a small kill, so
    # closing is clearly worth it.
    min_enemy_distance_penalty: float = 0.0001

    def __call__(self, delta: StepDelta) -> float:
        r  = self.gold_killed_delta * (delta.enemy_gold_lost - delta.our_gold_lost)
        r += self.village_delta     * (delta.villages_gained - delta.villages_lost)
        r += self.damage_dealt      * delta.enemy_hp_lost
        r += self.unit_recruited_cost * delta.unit_recruited_cost
        r -= self.per_turn_penalty
        if delta.leader_moved:
            r -= self.leader_move_penalty
        if delta.invalid_action:
            r -= self.invalid_action_penalty
        r -= self.min_enemy_distance_penalty * delta.min_enemy_distance

        if delta.outcome == OUTCOME_WIN:
            r += self.terminal_win
        elif delta.outcome == OUTCOME_LOSS:
            r += self.terminal_loss
        elif delta.outcome == OUTCOME_DRAW:
            r += self.terminal_draw
        elif delta.outcome == OUTCOME_TIMEOUT:
            r += self.terminal_timeout
        # OUTCOME_ONGOING: no terminal contribution

        return r


def hex_distance(a_x: int, a_y: int, b_x: int, b_y: int) -> int:
    """Wesnoth hex distance (odd-q offset coordinates).

    Matches the formula in Wesnoth's C++ `map_location::distance_between`:
    horizontal distance plus a half-step vertical component, with a
    +1 penalty for the zig-zag between even- and odd-x columns.
    """
    hd = abs(a_x - b_x)
    a_even = (a_x & 1) == 0
    b_even = (b_x & 1) == 0
    vpenalty = 0
    if (a_even and not b_even and a_y <= b_y) or \
       (b_even and not a_even and b_y <= a_y):
        vpenalty = 1
    return max(hd, abs(a_y - b_y) + hd // 2 + vpenalty)


def _action_had_visible_effect(
    prev_state: GameState, new_state: GameState,
) -> bool:
    """True iff the transition shows SOMETHING changed.

    Used to detect invalid actions: the custom AI stage silently loops
    when Wesnoth rejects an action, so Python sees state_t == state_t+1.
    Changes we consider 'visible': side swap (end_turn), unit count
    change (recruit/death), position change (move), HP change (combat),
    current_moves change (move with no position change — shouldn't
    happen but handled), has_attacked flip (attempted attack even on
    full-absorb), village count change. Healing between turns also
    flips HP, but that's tied to side swap which we catch separately.
    """
    if prev_state.global_info.current_side != new_state.global_info.current_side:
        return True
    prev_units = {u.id: u for u in prev_state.map.units}
    new_units = {u.id: u for u in new_state.map.units}
    if set(prev_units.keys()) != set(new_units.keys()):
        return True
    for uid, u in prev_units.items():
        nu = new_units[uid]
        if (u.position.x, u.position.y) != (nu.position.x, nu.position.y):
            return True
        if u.current_hp != nu.current_hp:
            return True
        if u.current_moves != nu.current_moves:
            return True
        if u.has_attacked != nu.has_attacked:
            return True
    if prev_state.sides and new_state.sides:
        for idx in range(min(len(prev_state.sides), len(new_state.sides))):
            if (prev_state.sides[idx].nb_villages_controlled
                    != new_state.sides[idx].nb_villages_controlled):
                return True
    return False


def _min_enemy_distance(state: GameState, our_side: int) -> int:
    """Smallest hex distance between any our_side NON-LEADER unit and
    any enemy unit. Falls back to 2× the leader's distance if we have
    no non-leader units yet.

    Why non-leader: if the leader counts, the cheapest way to minimize
    the distance penalty is to walk the leader straight at the enemy
    — which we already punish via leader_move_penalty AND which breaks
    recruiting (leader off keep = no recruits). Requiring a NON-leader
    to close creates the intended pressure: recruit → send the recruit
    forward → make contact.
    Why 2× fallback (not equal, not zero): zero makes "never recruit"
    a penalty-free local optimum. Equal to leader distance lets the
    policy substitute leader-walking for recruiting. 2× makes
    "recruit and then rely on the leader to close" strictly worse than
    "recruit and move the recruit" at every distance — the penalty
    from having only a leader at distance D is 2×D×k, while a single
    recruited fighter at the same spot would give D×k. A zero-recruit
    policy always pays strictly more per step than a recruit-and-move
    policy.
    Returns 0 only if there's no enemy visible (nothing to close to).
    """
    enemies = [u for u in state.map.units if u.side != our_side]
    if not enemies:
        return 0
    non_leaders = [u for u in state.map.units
                   if u.side == our_side and not u.is_leader]
    fallback_multiplier = 1
    if non_leaders:
        sources = non_leaders
    else:
        sources = [u for u in state.map.units
                   if u.side == our_side and u.is_leader]
        fallback_multiplier = 2
    if not sources:
        return 0
    best = None
    for u in sources:
        for e in enemies:
            d = hex_distance(u.position.x, u.position.y,
                             e.position.x, e.position.y)
            if best is None or d < best:
                best = d
    return (best * fallback_multiplier) if best is not None else 0


def compute_delta(
    prev_state: Optional[GameState],
    new_state: GameState,
    action_type: str,
    *,
    recruit_cost: int = 0,
    outcome: str = OUTCOME_ONGOING,
) -> StepDelta:
    """Diff two game states into a StepDelta for `new_state.current_side`.

    Called by the trainer between action-step pairs. Pre-recruit info
    that isn't derivable from the state (e.g., the cost of the unit
    just recruited, because its stats are already in new_state) comes
    in as keyword args.

    Convention: the 'side' of the delta is the side that *acted* to
    transition prev_state → new_state. That is usually
    `prev_state.global_info.current_side`; on a fresh episode where
    prev_state is None, we use new_state's current side.

    If prev_state is None (first step of an episode), all deltas are
    zero except any terminal outcome; this avoids spurious "the enemy
    just appeared" signals on the initial observation.
    """
    acting_side = (
        prev_state.global_info.current_side if prev_state is not None
        else new_state.global_info.current_side
    )
    delta = StepDelta(
        side=acting_side,
        turn=new_state.global_info.turn_number,
        action_type=action_type,
        # unit_recruited_cost filled in below only if a unit actually
        # appeared — the in-arg is the INTENDED cost, but our custom AI
        # stage doesn't tell Python whether Wesnoth accepted the recruit,
        # and rejected recruits must not be rewarded (otherwise the
        # policy learns to spam invalid recruits forever; see the
        # 985-game overnight run that climbed to mean_return=1.28 while
        # attempting 400+ recruits per game).
        unit_recruited_cost=0,
        outcome=outcome,
    )

    # Distance to closest enemy is always reported (when possible).
    # Computed on new_state so it reflects the CURRENT state the policy
    # just transitioned into — a penalty on "still far away" after this
    # action.
    delta.min_enemy_distance = _min_enemy_distance(new_state, acting_side)

    if prev_state is None:
        return delta

    # Invalid-action detection. If the full state has not changed
    # relative to prev_state, the action Python sent did nothing —
    # rejected by Wesnoth (invalid move target, no gold, etc.). Flagged
    # here; the reward function decides what to charge for it.
    delta.invalid_action = not _action_had_visible_effect(prev_state, new_state)

    # Unit-keyed lookups. unit.id is a stable per-unit identifier
    # assigned by Wesnoth (e.g., "knalgan_leader" or an auto-generated
    # "Dwarvish Fighter-42").
    prev_units = {u.id: u for u in prev_state.map.units}
    new_units  = {u.id: u for u in new_state.map.units}

    # Recruit-success check: credit unit_recruited_cost only if a unit
    # of our side actually appeared in new_state that wasn't in
    # prev_state. Same-turn death-of-ally is vanishingly rare on a
    # recruit step so we don't bother netting it out.
    if action_type == 'recruit' and recruit_cost > 0:
        appeared = any(
            u.side == acting_side and u.id not in prev_units
            for u in new_state.map.units
        )
        if appeared:
            delta.unit_recruited_cost = recruit_cost

    # Damage / deaths.
    for uid, u_prev in prev_units.items():
        u_new = new_units.get(uid)
        if u_new is None:
            # Unit gone: died (or left fog, but we assume same-side fog
            # so same-side disappearance == death; cross-side is a best
            # effort).
            if u_prev.side == acting_side:
                delta.our_gold_lost += u_prev.cost
            else:
                delta.enemy_gold_lost += u_prev.cost
                delta.enemy_hp_lost   += u_prev.current_hp
            continue
        hp_drop = max(u_prev.current_hp - u_new.current_hp, 0)
        if hp_drop > 0 and u_new.side != acting_side:
            delta.enemy_hp_lost += hp_drop
        elif hp_drop > 0 and u_new.side == acting_side:
            delta.our_hp_lost += hp_drop

    # Village counts (from sides info; side indices are 1-based).
    if prev_state.sides and new_state.sides:
        idx = acting_side - 1
        if 0 <= idx < len(prev_state.sides) and idx < len(new_state.sides):
            prev_v = prev_state.sides[idx].nb_villages_controlled
            new_v  = new_state.sides[idx].nb_villages_controlled
            if new_v > prev_v:
                delta.villages_gained = new_v - prev_v
            elif new_v < prev_v:
                delta.villages_lost = prev_v - new_v

    # Leader movement: did OUR (acting-side) leader change hex this step?
    # Implementation note: we look for a unit on acting_side with
    # is_leader=True in both snapshots; if their position differs, count
    # it. Death handles itself (leader gone → not "moved" → no penalty,
    # terminal reward fires separately).
    prev_leader_pos = _our_leader_pos(prev_state, acting_side)
    new_leader_pos  = _our_leader_pos(new_state,  acting_side)
    if prev_leader_pos is not None and new_leader_pos is not None:
        if (prev_leader_pos.x, prev_leader_pos.y) != (new_leader_pos.x, new_leader_pos.y):
            delta.leader_moved = True

    return delta


def _our_leader_pos(state: GameState, side: int):
    """Return the Position of `side`'s leader in `state`, or None if
    no leader of that side is visible."""
    for u in state.map.units:
        if u.side == side and u.is_leader:
            return u.position
    return None
