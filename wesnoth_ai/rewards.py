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

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Set, Tuple

from wesnoth_ai.classes import GameState


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

    # Fraction of the map currently visible to `side` after this
    # step's action lands. Computed in `compute_delta` from a fresh
    # per-side visibility scan; range [0, 1]. Used by
    # `WeightedReward.fog_reveal_weight` to pay a continuous
    # per-step shaping reward proportional to how much of the map
    # the side can see right now.
    #
    # Why "currently visible" rather than "newly revealed": continuous
    # payment with the trainer's gamma discount handles all the
    # right incentives automatically. Reveal-and-hold contributes
    # the full target weight over the game's discounted horizon;
    # reveal-then-lose-sight-then-re-reveal loses exactly
    # `(1-gamma)*gamma^n` of the weight share per hex per missed
    # step (the user's stated intent, 2026-05-17). No explicit
    # anti-oscillation bookkeeping is needed: a hex only pays
    # when it's visible right now, and the discount factor on the
    # missed-step payments is the natural penalty.
    visible_fraction: float = 0.0

    # Positive per-step delta in our minimum MP-cost to reach a hex
    # adjacent to the enemy leader (i.e., an attack-from position).
    # Specifically: max(0, prev_min_mp - new_min_mp). A delta of +3
    # means our closest non-leader unit closed by 3 MP this step;
    # since terrain costs are 1 for flat, 2 for forest, 3 for hills,
    # etc., the MP delta accounts for "going around an obstacle"
    # being not actually closer in path-cost terms even if hex
    # distance shrinks.
    #
    # Negative deltas (we retreated) clamp to 0 -- this term ONLY
    # rewards closing. Pair with `approach_enemy_leader_per_mp` in
    # WeightedReward. Computed via terrain-aware Dijkstra from
    # adjacent-to-leader hexes outward; see
    # `_min_mp_to_enemy_leader`.
    #
    # Renamed 2026-05-20 from `closing_to_enemy_leader` (hex-distance
    # delta) per user observation that going around passable
    # obstacles wasn't credited under the old metric. The new
    # metric correctly rewards routing around a mountain when that
    # path is actually cheaper.
    closing_to_enemy_leader_mp: int = 0

    # Fraction of total non-leader MP that went unused at end_turn.
    # 0.0 means "the acting side used every MP available across all
    # non-leader units"; 1.0 means "no non-leader unit moved at all
    # this turn." Fires only on end_turn for the acting side; 0 for
    # every other action.
    #
    # Why non-leader only: the leader has its own incentive to stay
    # on the keep (recruiting), so we don't want to pressure it to
    # walk every turn. The metric uses a FRACTION (not absolute
    # count) so recruiting more units doesn't increase the
    # per-step penalty -- a side with 10 non-leaders that uses 50%
    # of MP and a side with 5 non-leaders that uses 50% of MP pay
    # the same penalty per end_turn. No anti-recruit pressure.
    #
    # Why end_turn only: the penalty incentivizes USING MP within
    # the turn. Once end_turn fires, leftover MP is forfeited; the
    # signal needs to land then, not on intermediate actions where
    # the side may still spend more MP before ending the turn.
    #
    # Pair with `unused_mp_penalty` in WeightedReward (negative
    # value). User intent (2026-05-20): incentivize the policy to
    # actually move units forward rather than letting them sit on
    # half-walked routes.
    unused_mp_fraction: float = 0.0

    # Unit types newly appeared on `side` this step (i.e. successful
    # recruits). Empty tuple unless action_type=='recruit' and Wesnoth
    # accepted the recruit. Used by `WeightedReward.unit_type_bonuses`
    # to award per-type bonuses for "incentivize unorthodox openings"
    # style training (e.g. +0.5 reward per Wose recruited). Multiple
    # entries are possible only if the trainer ever batches recruits;
    # today it doesn't, so this is len 0 or 1.
    units_recruited: Tuple[str, ...] = ()

    # Optional post-action snapshot (used by `TurnConditionalBonus`
    # predicates -- see WeightedReward). compute_delta sets this iff
    # turn-conditional bonuses are configured (signalled via the
    # `attach_post_state=True` kwarg) so we don't pay the reference
    # retention cost in the common case. None means "no predicate
    # access available; turn-conditional bonuses are skipped".
    post_state: Optional[GameState] = None

    # Game label for keyed once-per-game bookkeeping in stateful
    # reward components (e.g. TurnConditionalBonus.once=True). Empty
    # string treats every step as a fresh game -- safe but defeats
    # `once`. Pass game_label through `compute_delta` to enable proper
    # per-game tracking.
    game_label: str = ""

    # Terminal flag.
    outcome: str = OUTCOME_ONGOING   # one of the OUTCOME_* constants


class RewardFn(Protocol):
    """Any callable mapping a StepDelta to a scalar reward."""

    def __call__(self, delta: StepDelta) -> float: ...


# ---------------------------------------------------------------------
# Customizability hooks: per-unit-type and turn-conditional bonuses
# ---------------------------------------------------------------------
# The user explicitly wants behavior gated by config/data/reward
# shaping rather than weights. These dataclasses are the building
# blocks for "+0.5 per Wose recruited", "+1.0 if leader on village
# by turn 3", and similar opener-incentive levers without touching
# the trainer.

@dataclass
class UnitTypeBonus:
    """Flat reward added each time a unit of `unit_type` is recruited
    on our side. Stackable: multiple entries with the same unit_type
    sum their weights.

    Example -- bias toward Wose openers in Knalgan vs Drakes::

        WeightedReward(unit_type_bonuses=[
            UnitTypeBonus("Wose", weight=0.5),
            UnitTypeBonus("Elvish Fighter", weight=0.05),
        ])

    `unit_type` matches `Unit.name` (i.e. the Wesnoth type id, NOT
    the localized name). For variations (e.g. Walking Corpse:mounted)
    use the composite key. Stateless: no per-game tracking.
    """
    unit_type: str
    weight:    float


# ---------------------------------------------------------------------
# Predicate registry for TurnConditionalBonus
# ---------------------------------------------------------------------
# JSON / YAML config files can't carry callable references, so we
# resolve predicate names through a string -> function registry. New
# predicates: register them via `register_predicate("name", fn)` --
# typically near the bottom of this module for built-ins, or in user
# code before loading a config.
#
# Predicate signature: `(post_state: GameState, side: int) -> bool`.

_Predicate = Callable[[GameState, int], bool]
_PREDICATE_REGISTRY: Dict[str, _Predicate] = {}


def register_predicate(name: str, fn: _Predicate) -> None:
    """Associate a predicate name (used in reward config files) with
    a callable. Later calls win."""
    _PREDICATE_REGISTRY[name] = fn


def get_predicate(name: str) -> _Predicate:
    """Resolve a registered predicate name. Raises KeyError with the
    available list if `name` is unknown."""
    if name not in _PREDICATE_REGISTRY:
        raise KeyError(
            f"Unknown predicate {name!r}. Available: "
            f"{sorted(_PREDICATE_REGISTRY)}")
    return _PREDICATE_REGISTRY[name]


def available_predicates() -> List[str]:
    """Sorted list of registered predicate names."""
    return sorted(_PREDICATE_REGISTRY)


@dataclass
class TurnConditionalBonus:
    """Conditional reward: when `predicate(post_state, side)` returns
    True during turns in `turn_range` (inclusive), award `weight`.
    With `once=True` (default) the bonus fires at most once per
    (game_label, side) -- subsequent step-evaluations on the same
    game/side are skipped.

    Example -- "+1.0 if our leader is on a village by turn 3"::

        from wesnoth_ai.classes import TerrainModifiers
        def leader_on_village(state, side):
            for u in state.map.units:
                if u.side == side and u.is_leader:
                    for h in state.map.hexes:
                        if (h.position.x, h.position.y) == (
                            u.position.x, u.position.y):
                            return TerrainModifiers.VILLAGE in h.modifiers
            return False
        WeightedReward(turn_conditional_bonuses=[
            TurnConditionalBonus(
                name="early_village",
                turn_range=(1, 3),
                predicate=leader_on_village,
                weight=1.0,
            ),
        ])

    Predicate gets the POST-action state and the acting side number.
    Evaluated only when `delta.post_state` is set -- otherwise the
    bonus is silently inert (a defensive default; explicit predicate
    failures shouldn't break training).
    """
    name:       str
    turn_range: Tuple[int, int]
    predicate:  Callable[[GameState, int], bool]
    weight:     float
    once:       bool = True


@dataclass
class WeightedReward:
    """Default reward: weighted sum of the StepDelta fields.

    SIGN CONVENTION (post-2026-04-29): every field is the SIGNED
    REWARD CONTRIBUTION. Positive = reward; negative = penalty.
    The `__call__` method sums fields additively with no per-field
    sign-flipping. So a config of `leader_move_penalty: -0.01`
    contributes exactly -0.01 every leader-move step. Field names
    ending `_penalty` are kept for readability; the sign is in the
    value, not in the name.

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
      - per_turn_penalty 0.0: enable (typically NEGATIVE) only if we
        observe policies learning to run out the clock.
      - leader_move_penalty -0.01: typically -0.2 across a 20-move
        game, comparable to recruit/village shaping.
      - invalid_action_penalty -0.001: caps at -0.5 over 500
        invalid attempts; back-pressure on garbage actions.
      - min_enemy_distance_penalty -0.0001: ~-0.6 across a 500-step
        game where units never close, comparable to a small kill --
        so closing dominates camping.
    """

    # Terminal rewards.
    terminal_win:     float = +1.0
    terminal_loss:    float = -1.0
    terminal_draw:    float =  0.0
    terminal_timeout: float =  0.0

    # Per-step shaping. See docstring for scale guidance.
    # Per the sign convention below, all of these are reward
    # contributions: gold_killed_delta etc. are typically positive
    # (good things to reward); per_turn_penalty defaults to 0 and
    # is typically NEGATIVE when configured (a per-step cost).
    gold_killed_delta:    float = 0.01     # (enemy_gold_lost - our_gold_lost)
    village_delta:        float = 0.05     # (villages_gained - villages_lost)
    damage_dealt:         float = 0.0005   # enemy_hp_lost only
    unit_recruited_cost:  float = 0.001
    per_turn_penalty:     float = 0.0      # applied once per Python-side step

    # SIGN CONVENTION (changed 2026-04-29): every field's value is the
    # SIGNED REWARD CONTRIBUTION. Positive = reward; negative = penalty.
    # The `__call__` method sums each field's contribution with no
    # sign-flipping, so a config of `leader_move_penalty: -0.01`
    # produces a reward contribution of -0.01 every leader-move step.
    # Field names ending in `_penalty` are kept for readability but
    # are NOT semantically penalty-only -- a positive value would
    # be a reward. Defaults below are negative for the disciplinary
    # fields (matches the prior "subtract" semantics with the new
    # config-as-truth convention).
    #
    # Per-step contribution when our leader changed hexes. Negative
    # default discourages wandering the leader away from the keep --
    # the original fix for "every game returns ~0.22 with no inter-
    # episode variance" because random policies that happened to
    # recruit more (leader stays on keep) couldn't be distinguished
    # from those that wandered. Scaled so a game with ~20 leader
    # moves contributes ~-0.2.
    leader_move_penalty:  float = -0.01

    # Optional (turn_lo, turn_hi) inclusive range that restricts
    # `leader_move_penalty` to specific turns. None (default) means
    # ALWAYS apply -- backwards-compatible. A typical mid-game config
    # might use `(1, 15)` to discourage early-game leader wandering
    # but stop punishing endgame leader maneuvers (when the keep is
    # secure and the leader needs to participate in attacks). The
    # comparison is `lo <= delta.turn <= hi`; either bound may be
    # `None` to leave that side open (e.g. `(None, 15)` for "only
    # turns 1-15", `(20, None)` for "turn 20 onward").
    leader_move_penalty_turn_range: Optional[Tuple[Optional[int], Optional[int]]] = None

    # Flat contribution for actions Wesnoth/sim silently rejected
    # (state didn't change). Negative default added after the
    # overnight run showed the policy spamming ~500 invalid
    # recruits per game; with 500-action games this caps at -0.5.
    invalid_action_penalty: float = -0.001

    # Per-step contribution proportional to the minimum hex
    # distance between any friendly unit and any enemy unit.
    # SUPERSEDED by `approach_enemy_leader_per_hex` (delta form)
    # in 2026-05-13's tuning pass. The absolute-distance penalty
    # fires every step at full magnitude and accumulates over the
    # whole game (200+ actions); aggressive values saturate the
    # C51 value support [-1, +1] and collapse mean_return to the
    # clip, killing the policy gradient. Default left at the
    # near-zero pre-2026-05-12 magnitude for backwards
    # compatibility; recommended setting in JSON configs is 0.0.
    min_enemy_distance_penalty: float = -0.0001

    # Fog-reveal shaping reward, continuous-payment form.
    #
    # Semantics: `fog_reveal_weight` is the TOTAL discounted return
    # contribution earned by exploring the entire map immediately
    # and keeping all hexes visible for the rest of the game. The
    # per-step payment is `(1 - fog_reveal_gamma) * weight *
    # visible_fraction`, and the geometric series of those
    # payments at full sustained visibility sums to `weight` over
    # an infinite-horizon game (approaches `weight` for long
    # finite games; ~99% of weight by step 460 at gamma=0.99).
    #
    # Anti-oscillation is automatic: a hex only contributes reward
    # while it's currently visible, so toggling visibility just
    # forfeits the missed-step shares. A hex visible at every step
    # except step n loses exactly `(1-gamma) * weight * (1/H) *
    # gamma^n` of the target weight -- matches the user's stated
    # contract (2026-05-17).
    #
    # `fog_reveal_gamma` MUST match the trainer's gamma for the
    # "weight = total" identity to hold. Default 0.99 matches
    # `TrainerConfig.gamma`. If you ever tune the trainer's
    # gamma, mirror it here.
    #
    # Off by default (0.0). Recommended starting value when
    # enabling: 0.3 (comparable to a string of kills, well below
    # the ±1.0 terminal signal). Bump if exploration isn't
    # emerging; cut if the policy explores instead of fighting.
    #
    # Intent: incentivize early-game exploration on maps where
    # the enemy starts beyond sight (ladder), so the policy
    # learns to spread units instead of camping the keep.
    # Mini-maps barely contribute (vision saturates in ~2 turns)
    # so this term is near-zero there -- it specifically targets
    # the long-march regime mini-maps don't exercise.
    fog_reveal_weight: float = 0.0
    fog_reveal_gamma:  float = 0.99

    # Per-MP closing-cost reward: weight × max(0, prev_mp - new_mp),
    # where `mp` is the minimum MP cost for any of our non-leaders
    # to walk to a hex adjacent to the enemy leader. Terrain-aware:
    # crossing a forest hex pays 2 MP, mountain 3 MP, etc., so this
    # metric credits "routed around an obstacle" the same as "walked
    # straight" when the path cost was actually equivalent. Old
    # `approach_enemy_leader_per_hex` (hex distance) penalized
    # routing around passable obstacles even when MP-equivalent.
    #
    # ONLY positive (closing) is rewarded; retreating is 0, never a
    # penalty (avoids odd gradient signs when the policy retreats
    # to set up a flank).
    #
    # BOUNDED total contribution: per game, max sum is the MP
    # equivalent of the starting separation (typically 20-50 MP on
    # ladder maps). At weight 0.01, cumulative ceiling is ~0.2-0.5,
    # well inside C51 [-1, +1].
    #
    # See `_min_mp_to_enemy_leader` for the metric. The enemy
    # leader's position is god-view (deliberate fog-of-war breach,
    # documented at the helper).
    approach_enemy_leader_per_mp: float = 0.0

    # Per-end_turn penalty proportional to unused non-leader MP
    # (StepDelta.unused_mp_fraction). Negative by convention; a
    # value of -0.05 means "a turn where 100% of non-leader MP was
    # wasted contributes -0.05 to that step's reward." Reward
    # contribution = unused_mp_penalty * unused_mp_fraction; only
    # fires on end_turn steps.
    #
    # Magnitude guidance: with 200 turns/game and 50% unused MP
    # average, a weight of -0.005 yields ~-0.5 cumulative per side
    # per game. Match against terminal_win = +1.0 means "fully
    # wasting MP for a whole game is half as bad as losing."
    # Default -0.002 to start conservatively.
    #
    # Off by default (0.0) so existing reward configs are unchanged.
    # Intent: pressure the policy to ACTUALLY MOVE units toward
    # action rather than leaving them at half-walked routes
    # (observed 2026-05-20: closest_approach drifts upward, units
    # don't push toward the enemy keep). Excludes the leader so
    # this doesn't fight the keep-stay-for-recruit dynamic.
    unused_mp_penalty: float = 0.0

    # When True, `gold_killed_delta` rewards ONLY the gold-value of
    # enemy units we killed -- ignoring our own losses. Default False
    # preserves the original "net P&L" semantics
    # (enemy_gold_lost - our_gold_lost).
    #
    # Why this knob exists (added 2026-05-14): the 17839 run showed
    # the policy initially learned to attack (peaked at 8% on iter
    # 99), then REGRESSED to 0% attacks by iter 210 because at low
    # skill our retaliation losses dominate kills, making attacks
    # net-negative under symmetric gold_killed. The asymmetric form
    # ("only credit kills, ignore losses") removes the disincentive
    # to engage; terminal_win/loss still encodes "we got crushed"
    # at game end, so the overall reward landscape isn't pathological.
    gold_killed_one_sided: bool = False

    # Flat reward per attack action attempted (regardless of
    # damage outcome). Encourages exploration of engagement even
    # when individual attacks miss or get killed in retaliation.
    # Bounded: ~30 attacks/game at 10% attack rate × this weight =
    # ~0.06 cumulative at the default 0.002.
    #
    # Skipped when delta.invalid_action is True (the sim rejected
    # the attack -- usually a fog edge case). We don't reward
    # "tried to attack a thing that doesn't exist."
    #
    # Why this knob exists: gradient-wise, the gradient signal
    # for "attack" only fires if attack appears at all in a
    # trajectory. Once the policy converges to attack% ~= 0, the
    # signal goes to zero and the policy can't recover. A flat
    # per-attempt bonus keeps the "attack" branch alive in the
    # gradient even when the network's attack-prob is low.
    attack_attempt_bonus: float = 0.0

    # Customizability hooks (default empty -- keep behavior identical
    # to pre-2026-04-28 unless the user opts in).
    #
    # Per-unit-type recruit bonuses: stackable, stateless. See
    # UnitTypeBonus.
    unit_type_bonuses: List[UnitTypeBonus] = field(default_factory=list)
    # Turn-conditional bonuses with optional once-per-game gating.
    # See TurnConditionalBonus. Internally we maintain a fired-set
    # keyed by (game_label, side, bonus_name) so `once=True` works
    # correctly across multi-game runs without leaking across games.
    turn_conditional_bonuses: List[TurnConditionalBonus] = field(default_factory=list)

    # Internal: tracks which (game_label, side, bonus_name) triples
    # have already fired their once-per-game bonus. Init in
    # __post_init__ so dataclass copies don't share the dict.
    _fired_once: Dict[Tuple[str, int, str], bool] = field(
        default_factory=dict, repr=False, compare=False)

    # Optional per-component reward accumulator. When the trainer
    # sets this to a dict (per side), `__call__` adds each
    # component's contribution into it under a stable key
    # (e.g., "gold_killed", "approach_mp", "fog_reveal"). At the
    # end of an iter, the trainer reads these out, writes to
    # trainer_history.csv, and zeros the dict.
    #
    # Default None = no accumulation = no overhead. Cost when
    # set: one dict-update per non-zero component (~5-10 updates
    # per call, ~1 µs each on cpython) ≈ ~10 µs per call. Over
    # 4000 reward calls per iter that's ~40 ms -- well under
    # 0.1% of an iter's cost.
    _component_acc: Optional[Dict[str, float]] = field(
        default=None, repr=False, compare=False)

    def reset_game_state(self, game_label: str = "") -> None:
        """Clear once-per-game bookkeeping for `game_label`. Call this
        between games when re-using the same WeightedReward instance.
        With game_label='' (default), clears ALL games' state -- safe
        but coarse."""
        if not game_label:
            self._fired_once.clear()
            return
        stale = [k for k in self._fired_once if k[0] == game_label]
        for k in stale:
            del self._fired_once[k]

    def __call__(self, delta: StepDelta) -> float:
        # Sign convention: every field is a SIGNED reward contribution.
        # Positive = reward, negative = penalty. The arithmetic is
        # purely additive with no per-field sign-flipping, so a config
        # of `leader_move_penalty: -0.01` contributes exactly -0.01
        # every step the leader moved. This makes the JSON config
        # match the effective contribution one-to-one (audited
        # 2026-04-29 against earlier "subtract penalties" convention,
        # which inverted JSON sign vs effect and was confusing).
        #
        # Per-component accumulator (optional): when self._component_acc
        # is set (trainer-controlled), each contribution is also
        # added under a stable key. Keeps the running totals per
        # iter so trainer_history.csv can report what's actually
        # firing (debugging recruit-underuse / atk%-collapse).
        acc = self._component_acc

        # gold_killed contribution. Symmetric mode (default) credits
        # (enemy_kills - our_losses) -- accurate P&L but at low policy
        # skill the our_losses term dominates and disincentivizes
        # engagement. One-sided mode rewards only enemy kills; terminal
        # win/loss still encodes "we got crushed" at game end so the
        # overall landscape stays correct.
        if self.gold_killed_one_sided:
            c = self.gold_killed_delta * delta.enemy_gold_lost
        else:
            c = self.gold_killed_delta * (delta.enemy_gold_lost - delta.our_gold_lost)
        r = c
        if acc is not None and c:
            acc["gold_killed"] = acc.get("gold_killed", 0.0) + c

        c = self.village_delta * (delta.villages_gained - delta.villages_lost)
        r += c
        if acc is not None and c:
            acc["village_delta"] = acc.get("village_delta", 0.0) + c

        c = self.damage_dealt * delta.enemy_hp_lost
        r += c
        if acc is not None and c:
            acc["damage_dealt"] = acc.get("damage_dealt", 0.0) + c

        c = self.unit_recruited_cost * delta.unit_recruited_cost
        r += c
        if acc is not None and c:
            acc["unit_recruited_cost"] = acc.get("unit_recruited_cost", 0.0) + c

        c = self.per_turn_penalty
        r += c
        if acc is not None and c:
            acc["per_turn_penalty"] = acc.get("per_turn_penalty", 0.0) + c

        if delta.leader_moved:
            tr = self.leader_move_penalty_turn_range
            apply_penalty = True
            if tr is not None:
                lo, hi = tr
                if lo is not None and delta.turn < lo:
                    apply_penalty = False
                if hi is not None and delta.turn > hi:
                    apply_penalty = False
            if apply_penalty:
                c = self.leader_move_penalty
                r += c
                if acc is not None and c:
                    acc["leader_move_penalty"] = acc.get("leader_move_penalty", 0.0) + c

        if delta.invalid_action:
            c = self.invalid_action_penalty
            r += c
            if acc is not None and c:
                acc["invalid_action"] = acc.get("invalid_action", 0.0) + c

        c = self.min_enemy_distance_penalty * delta.min_enemy_distance
        r += c
        if acc is not None and c:
            acc["min_enemy_distance"] = acc.get("min_enemy_distance", 0.0) + c

        # Closing-distance reward. Bounded: a per-step cap of "moved
        # how much closer this step" times the (small) weight, summed
        # only when delta is positive. Total contribution per game is
        # bounded by the map diameter ~30-40 hexes × weight.
        if delta.closing_to_enemy_leader_mp > 0:
            c = (self.approach_enemy_leader_per_mp
                 * delta.closing_to_enemy_leader_mp)
            r += c
            if acc is not None and c:
                acc["approach_mp"] = acc.get("approach_mp", 0.0) + c

        # Unused-MP penalty (end_turn only). delta.unused_mp_fraction
        # is computed in compute_delta and is non-zero only on
        # end_turn steps. The product is the per-step contribution.
        if self.unused_mp_penalty and delta.unused_mp_fraction:
            c = self.unused_mp_penalty * delta.unused_mp_fraction
            r += c
            if acc is not None and c:
                acc["unused_mp"] = acc.get("unused_mp", 0.0) + c
        # Fog-reveal: continuous per-step payment proportional to
        # current visible fraction. Geometric series of these
        # payments at full sustained visibility approaches
        # `fog_reveal_weight` (the total target). See WeightedReward
        # docstring for the math.
        if self.fog_reveal_weight and delta.visible_fraction:
            c = ((1.0 - self.fog_reveal_gamma)
                 * self.fog_reveal_weight
                 * delta.visible_fraction)
            r += c
            if acc is not None and c:
                acc["fog_reveal"] = acc.get("fog_reveal", 0.0) + c
        # Per-attempt attack bonus. Fires on every VALID attack
        # action. Keeps the "attack" gradient signal alive even when
        # the policy's attack-prob has shrunk to ~0 -- gives the
        # network a small constant pull back toward engagement.
        # Invalid attacks (sim rejected, fog edge case) don't count.
        if (delta.action_type == "attack" and not delta.invalid_action
                and self.attack_attempt_bonus):
            c = self.attack_attempt_bonus
            r += c
            if acc is not None and c:
                acc["attack_attempt"] = acc.get("attack_attempt", 0.0) + c

        # Per-unit-type recruit bonuses. Iterate the list (small in
        # practice; <10 typical configurations) and accumulate
        # contributions. units_recruited may have multiple entries if
        # a future trainer batches recruits, so we count occurrences
        # via the list intersection rather than a single bool.
        if self.unit_type_bonuses and delta.units_recruited:
            ut_total = 0.0
            for bonus in self.unit_type_bonuses:
                count = sum(1 for u in delta.units_recruited
                            if u == bonus.unit_type)
                if count:
                    ut_total += bonus.weight * count
            if ut_total:
                r += ut_total
                if acc is not None:
                    acc["unit_type_bonus"] = acc.get("unit_type_bonus", 0.0) + ut_total

        # Turn-conditional bonuses. Need post_state for the predicate;
        # if absent (e.g. test-builder StepDelta or compute_delta
        # called without attach_post_state), skip silently.
        if self.turn_conditional_bonuses and delta.post_state is not None:
            for bonus in self.turn_conditional_bonuses:
                lo, hi = bonus.turn_range
                if not (lo <= delta.turn <= hi):
                    continue
                if bonus.once:
                    key = (delta.game_label, delta.side, bonus.name)
                    if self._fired_once.get(key):
                        continue
                # Predicate exceptions shouldn't crash a training run.
                # Catch + log; the bonus stays unawarded for this step.
                try:
                    fired = bool(bonus.predicate(delta.post_state, delta.side))
                except Exception:
                    fired = False
                if fired:
                    r += bonus.weight
                    if acc is not None:
                        acc["turn_cond_bonus"] = acc.get("turn_cond_bonus", 0.0) + bonus.weight
                    if bonus.once:
                        self._fired_once[(delta.game_label,
                                          delta.side, bonus.name)] = True

        if delta.outcome == OUTCOME_WIN:
            r += self.terminal_win
            if acc is not None:
                acc["terminal"] = acc.get("terminal", 0.0) + self.terminal_win
        elif delta.outcome == OUTCOME_LOSS:
            r += self.terminal_loss
            if acc is not None:
                acc["terminal"] = acc.get("terminal", 0.0) + self.terminal_loss
        elif delta.outcome == OUTCOME_DRAW:
            r += self.terminal_draw
            if acc is not None and self.terminal_draw:
                acc["terminal"] = acc.get("terminal", 0.0) + self.terminal_draw
        elif delta.outcome == OUTCOME_TIMEOUT:
            r += self.terminal_timeout
            if acc is not None and self.terminal_timeout:
                acc["terminal"] = acc.get("terminal", 0.0) + self.terminal_timeout
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


# Visibility helpers extracted to `visibility.py` so encoder,
# action_sampler, and rewards share one implementation of the
# fog-of-war contract. Local module-level aliases preserve the
# (now thin) namespace consumers used to import.
from wesnoth_ai.visibility import (visible_fraction_for as _visible_fraction,
                        visible_hexes_for as _compute_visible_hexes,
                        sight_radius_for as _sight_radius_for)
__all__ = (_visible_fraction, _compute_visible_hexes, _sight_radius_for)  # quiet linters


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


def _min_dist_to_enemy_leader(state: GameState, our_side: int) -> Optional[int]:
    """Min hex distance from any of `our_side`'s NON-LEADER units to
    any OTHER side's leader. Returns None when the metric is
    undefined: either we have no non-leader units (can't close
    without an army), or there's no opposing leader (already won/lost).

    ─────────────────────────────────────────────────────────────────
    DELIBERATE FOG-OF-WAR BREACH (audited 2026-05-20)
    ─────────────────────────────────────────────────────────────────
    This function reads the enemy leader's position from
    `state.map.units`, which is god-view in the sim. The encoder
    + action_sampler are fog-filtered (visibility.py), so the policy
    never sees enemy unit tokens it shouldn't — but the SCALAR REWARD
    derived from this position IS exposed to the policy via
    `policy.observe(reward)`. The policy can correlate "action a in
    state s produced reward r" across episodes to infer rough enemy
    leader direction.

    This is intentional: without a directional gradient, exploration
    rewards alone don't bias the policy toward enemy encounter (see
    BACKLOG / 2026-05-20 log analysis). The trade-off is accepted.

    Note for future work: if we ever decide to remove the breach,
    candidates are (a) only fire the closing reward when at least
    one our-side unit has the enemy leader's hex inside its sight
    disc (turns the reward off in fog, but kills the directional
    gradient too); (b) replace with a distance-to-enemy-KEEP metric
    (the keep's terrain is genuinely observable, no breach); (c)
    remove the reward entirely and rely on combat / village
    deltas to provide gradient once units close enough to interact.
    The "no breach" rewrite isn't free — option (b) gives the same
    directional signal at the cost of slightly weaker grading (the
    keep position is fixed; the enemy leader may roam, which we'd
    no longer pressure against).
    ─────────────────────────────────────────────────────────────────

    Why non-leader: same reasoning as `_min_enemy_distance`. We don't
    want to reward walking our OWN leader at the enemy — the leader
    needs to stay near the keep to recruit. The signal we want is
    "send recruits to attack the enemy leader," which is exactly
    "non-leader of mine -> their leader."

    Why None (not 0): a delta of `prev - new` is only meaningful when
    BOTH samples are real distances. If either side is missing units
    needed for the metric, the delta is meaningless and we want to
    skip the reward for this step, not award `prev - 0 = prev`.
    """
    # Enemy leaders: any side != our_side that has an is_leader unit.
    # In 2p ladder play this is always exactly one unit (side 1 vs 2);
    # the loop tolerates >2-side scenery maps just in case.
    enemy_leaders = [u for u in state.map.units
                     if u.side != our_side and u.is_leader]
    if not enemy_leaders:
        return None
    non_leaders = [u for u in state.map.units
                   if u.side == our_side and not u.is_leader]
    if not non_leaders:
        return None
    best = None
    for u in non_leaders:
        for e in enemy_leaders:
            d = hex_distance(u.position.x, u.position.y,
                             e.position.x, e.position.y)
            if best is None or d < best:
                best = d
    return best


# ---------------------------------------------------------------------
# MP-cost metric (Dijkstra with terrain-aware costs).
#
# Replaces the hex-distance metric for the closing-to-enemy-leader
# reward. The user observation (2026-05-20): a unit may need to go
# around a passable obstacle (e.g., a mountain range with one pass)
# to actually approach the enemy leader; in that case hex distance
# overstates the true closing because some intermediate hexes cost
# more MP. The MP-cost metric uses Wesnoth's actual terrain costs
# via _move_cost_at_hex, so "1 MP closer" always means "the unit
# really did spend ~1 MP and is genuinely 1 step nearer."
#
# Inherits the same deliberate fog-of-war breach as
# _min_dist_to_enemy_leader (see that function's docstring).
# ---------------------------------------------------------------------

# Cache for the per-(terrain, unit-type, target) Dijkstra dist field.
# Within a game, the map terrain is stable, the enemy leader's hex
# is mostly stable (it doesn't move every step), and we typically
# have 1-3 unique non-leader unit types. So once a state's dist field
# is computed, the next ~hundreds of states reuse it. ~99% cache hit
# rate observed in profiling; without the cache, this adds ~16s per
# training iter; with it, ~0.5s.
#
# Key: (id(state.global_info._terrain_codes), unit_name, target_xy).
# `id()` of the terrain_codes dict is a stable per-game proxy --
# different games re-build the dict, so cache entries from a prior
# game naturally become unreachable. The cache is bounded to prevent
# unbounded growth across many games.
_MP_DIST_CACHE: Dict[Tuple, Dict[Tuple[int, int], int]] = {}
_MP_DIST_CACHE_MAX = 256


def _dijkstra_mp_field_to_adjacent(
    state: GameState, template_unit, target_xy: Tuple[int, int],
) -> Optional[Dict[Tuple[int, int], int]]:
    """Compute (or fetch from cache) the dict mapping each hex coord
    (x, y) on the map to the minimum MP cost for a unit with
    `template_unit`'s movement profile, starting at (x, y), to reach
    a hex adjacent to `target_xy`. The unit STOPS adjacent to the
    target; it doesn't enter the target's hex (matching Wesnoth
    attack mechanics).

    Returns None if the target's hex isn't on the map. Hexes the
    template cannot reach are simply absent from the returned dict
    (caller treats absence as "unreachable, no reward").

    Algorithm: multi-source Dijkstra-from-outward. Sources are all
    hexes adjacent to `target_xy` (the attack-from positions).
    Edge relaxation: when processing hex u and relaxing to neighbor
    v, the edge weight is the MP cost for the template to ENTER u
    (because in the actual path v→u→...→adjacent(T), the unit pays
    cost(u) when stepping into u from v). The target_xy itself is
    excluded from the propagation graph.

    Cached on (terrain identity, unit name, target). Subsequent
    calls within the same game hit the cache cleanly.
    """
    import heapq
    from tools.wesnoth_sim import _move_cost_at_hex
    from tools.abilities import hex_neighbors

    terrain = getattr(state.global_info, "_terrain_codes", None)
    terrain_id = id(terrain) if terrain else id(state.map.hexes)
    cache_key = (terrain_id, template_unit.name, target_xy)
    cached = _MP_DIST_CACHE.get(cache_key)
    if cached is not None:
        return cached

    on_map = {(h.position.x, h.position.y) for h in state.map.hexes}
    if target_xy not in on_map:
        return None

    # Sources: hexes adjacent to target_xy that are on the map.
    sources = [(nx, ny) for nx, ny in hex_neighbors(*target_xy)
               if (nx, ny) in on_map]
    if not sources:
        return None

    UNREACHABLE = 99
    dist: Dict[Tuple[int, int], int] = {s: 0 for s in sources}
    pq: List[Tuple[int, int, int]] = [(0, sx, sy) for sx, sy in sources]
    heapq.heapify(pq)
    # Pre-fetch into a local for tight-loop speed.
    move_cost = _move_cost_at_hex
    while pq:
        cost, x, y = heapq.heappop(pq)
        if cost > dist[(x, y)]:
            continue
        # Cost to ENTER (x, y) for a unit walking from a neighbor
        # toward target. This is the edge weight when relaxing
        # (x, y) -> any neighbor in the outward search.
        step_out = move_cost(template_unit, state, x, y)
        if step_out >= UNREACHABLE:
            # Can't step OUT of (x, y) toward target (i.e., can't
            # enter (x, y) from a neighbor in the forward direction).
            # Don't relax outgoing edges. (x, y) itself remains
            # reachable with the cost we already recorded for it.
            continue
        for nx, ny in hex_neighbors(x, y):
            if (nx, ny) not in on_map:
                continue
            if (nx, ny) == target_xy:
                # The unit STOPS adjacent to T; doesn't path through it.
                continue
            new_cost = cost + step_out
            if new_cost < dist.get((nx, ny), UNREACHABLE):
                dist[(nx, ny)] = new_cost
                heapq.heappush(pq, (new_cost, nx, ny))

    # Crude size-bounded LRU: clear-on-fill. Acceptable because the
    # cache is dominated by within-game hits; a clear is cheap and
    # only fires when a new game has built up enough entries to
    # exceed the cap, which means the previous game's entries are
    # already stale.
    if len(_MP_DIST_CACHE) >= _MP_DIST_CACHE_MAX:
        _MP_DIST_CACHE.clear()
    _MP_DIST_CACHE[cache_key] = dist
    return dist


def _min_mp_to_enemy_leader(
    state: GameState, our_side: int,
) -> Optional[int]:
    """Min MP cost for any of our_side's non-leader units to walk
    to a hex adjacent to the enemy leader. Terrain-aware via
    `tools.wesnoth_sim._move_cost_at_hex`.

    The same fog-of-war breach as `_min_dist_to_enemy_leader`
    applies: this function reads the enemy leader's position from
    god-view. See that helper's docstring for the rationale and
    the audit trail.

    Returns None when the metric is undefined (no enemy leader, no
    own non-leaders, or every non-leader's path to the leader is
    blocked).

    Non-leader-only for the same reason as the hex-distance
    metric: we want to incentivize "send your army at theirs,"
    not "walk the leader off the keep." See `_min_dist_to_enemy_leader`
    for the full argument.
    """
    enemy_leaders = [u for u in state.map.units
                     if u.side != our_side and u.is_leader]
    if not enemy_leaders:
        return None
    non_leaders = [u for u in state.map.units
                   if u.side == our_side and not u.is_leader]
    if not non_leaders:
        return None

    # Group non-leaders by unit name (== movement type for our
    # purposes: same Wesnoth unit type means same movement_costs).
    # We compute one Dijkstra per (unit-type, enemy-leader-position)
    # pair, then look up each non-leader's current hex.
    by_name: Dict[str, list] = {}
    for u in non_leaders:
        by_name.setdefault(u.name, []).append(u)

    best: Optional[int] = None
    for e in enemy_leaders:
        target_xy = (e.position.x, e.position.y)
        for u_name, group in by_name.items():
            template = group[0]
            dist_field = _dijkstra_mp_field_to_adjacent(
                state, template, target_xy)
            if dist_field is None:
                continue
            for u in group:
                d = dist_field.get((u.position.x, u.position.y))
                if d is None:
                    continue
                if best is None or d < best:
                    best = d
    return best


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
    game_label: str = "",
    attach_post_state: bool = False,
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

    `game_label`: opaque identifier propagated to the StepDelta so
    stateful reward components (TurnConditionalBonus.once=True)
    can scope their fired-set per game.

    `attach_post_state`: when True, store a reference to `new_state`
    on the StepDelta. Required for TurnConditionalBonus predicates to
    fire. Default False so callers that don't use predicate-bonuses
    don't pay the (small) reference-retention cost.
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
        game_label=game_label,
        post_state=(new_state if attach_post_state else None),
    )

    # Distance to closest enemy is always reported (when possible).
    # Computed on new_state so it reflects the CURRENT state the policy
    # just transitioned into — a penalty on "still far away" after this
    # action.
    delta.min_enemy_distance = _min_enemy_distance(new_state, acting_side)

    # Current-visibility fraction for the continuous fog-reveal
    # shaping reward. Always computed (~µs); the reward consumes
    # it via `WeightedReward.fog_reveal_weight × (1-gamma) ×
    # visible_fraction`. No bookkeeping needed on global_info --
    # each step's visibility is a pure function of the current
    # unit positions, recomputed fresh.
    delta.visible_fraction = _visible_fraction(new_state, acting_side)

    # Closing-MP delta. Compute the min-MP-cost-to-reach-the-enemy-
    # leader from ANY of our non-leader units BEFORE and AFTER this
    # step; the delta is `max(0, prev_mp - new_mp)` (only credit
    # closing). `_min_mp_to_enemy_leader` returns None when the
    # metric is undefined (no enemy leader, no own non-leaders, or
    # all paths blocked) -- we treat those steps as 0-delta. The
    # prev_state==None branch below handles game start.
    if prev_state is not None:
        prev_mp = _min_mp_to_enemy_leader(prev_state, acting_side)
        new_mp = _min_mp_to_enemy_leader(new_state, acting_side)
        if prev_mp is not None and new_mp is not None:
            closing = prev_mp - new_mp
            if closing > 0:
                delta.closing_to_enemy_leader_mp = closing

    # Unused-MP fraction at end_turn. Computed from prev_state (the
    # acting side's MPs just before they ended their turn -- the
    # new_state has already swapped to the next side and reset MPs
    # via init_side). Only non-leader units count; the leader has
    # its own keep-stay incentive and shouldn't contribute to this
    # penalty.
    #
    # Fraction = sum(non_leader.current_moves) / sum(non_leader.max_moves).
    # Bounded [0, 1]. 0 means "we used every MP," 1 means "no
    # non-leader moved at all this turn." When there are no
    # non-leaders, the metric is undefined; we leave it at 0 (no
    # penalty when there's nobody to move).
    if (action_type == "end_turn"
            and prev_state is not None):
        total_max = 0
        total_cur = 0
        for u in prev_state.map.units:
            if u.side != acting_side or u.is_leader:
                continue
            mm = int(getattr(u, "max_moves", 0) or 0)
            if mm <= 0:
                continue
            total_max += mm
            total_cur += int(getattr(u, "current_moves", 0) or 0)
        if total_max > 0:
            # Clamp into [0, 1] defensively in case current_moves
            # > max_moves from a healing-effect / scenario event.
            frac = max(0.0, min(1.0, total_cur / total_max))
            delta.unused_mp_fraction = frac

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
    if action_type == 'recruit':
        appeared_units = [
            u for u in new_state.map.units
            if u.side == acting_side and u.id not in prev_units
        ]
        if appeared_units:
            if recruit_cost > 0:
                delta.unit_recruited_cost = recruit_cost
            # Record unit type names for the per-type bonus path.
            # WeightedReward.unit_type_bonuses sums weights across
            # this tuple, so the type appears here regardless of
            # whether recruit_cost was passed (custom-era unit
            # type that lookup can't price still gets typed bonus).
            delta.units_recruited = tuple(u.name for u in appeared_units)

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


# ---------------------------------------------------------------------
# Built-in predicates for TurnConditionalBonus
# ---------------------------------------------------------------------
# Names match what reward-config JSON/YAML files reference.

def _pred_leader_on_village(state: GameState, side: int) -> bool:
    """Our leader is currently standing on a village hex."""
    from wesnoth_ai.classes import TerrainModifiers
    leader = next((u for u in state.map.units
                   if u.side == side and u.is_leader), None)
    if leader is None:
        return False
    for h in state.map.hexes:
        if (h.position.x, h.position.y) == (leader.position.x,
                                            leader.position.y):
            return TerrainModifiers.VILLAGE in h.modifiers
    return False


def _pred_leader_on_keep(state: GameState, side: int) -> bool:
    """Our leader is currently standing on a keep hex (where it can
    recruit). Useful as a 'don't wander the leader off' bonus that's
    less harsh than `leader_move_penalty` -- only credits while the
    leader CAN recruit, vs. always-on regardless of position."""
    from wesnoth_ai.classes import TerrainModifiers
    leader = next((u for u in state.map.units
                   if u.side == side and u.is_leader), None)
    if leader is None:
        return False
    for h in state.map.hexes:
        if (h.position.x, h.position.y) == (leader.position.x,
                                            leader.position.y):
            return TerrainModifiers.KEEP in h.modifiers
    return False


def _pred_controls_majority_villages(state: GameState, side: int) -> bool:
    """We control strictly more than half the visible villages.
    Triggers a positional-dominance bonus that's silent when the map
    has zero villages (avoids spurious credit on village-free maps)."""
    if not state.sides:
        return False
    total = 0
    for s in state.sides:
        total += int(s.nb_villages_controlled)
    if total == 0:
        return False
    side_idx = side - 1
    if not (0 <= side_idx < len(state.sides)):
        return False
    ours = int(state.sides[side_idx].nb_villages_controlled)
    return ours * 2 > total


def _pred_no_units_lost(state: GameState, side: int) -> bool:
    """Hard to compute from post-state alone (we'd need to compare
    against game start). Instead: True iff WE still have at least
    as many non-leader units as the opponent does. Useful as a
    'preserve army' bonus."""
    if not state.sides:
        return False
    our_count = sum(1 for u in state.map.units
                    if u.side == side and not u.is_leader)
    enemy_count = sum(1 for u in state.map.units
                      if u.side != side and not u.is_leader)
    return our_count >= enemy_count


register_predicate("leader_on_village",        _pred_leader_on_village)
register_predicate("leader_on_keep",           _pred_leader_on_keep)
register_predicate("controls_majority_villages", _pred_controls_majority_villages)
register_predicate("no_units_lost",            _pred_no_units_lost)


# ---------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------
# Builds a `WeightedReward` from a JSON or YAML file, including the
# customizability lists (unit_type_bonuses, turn_conditional_bonuses).
# Predicates referenced by name resolve through `_PREDICATE_REGISTRY`.

def load_reward_config(path) -> "WeightedReward":
    """Construct a `WeightedReward` from a JSON or YAML config file.

    Sign convention (see WeightedReward docstring): each field's
    value is the signed reward contribution. Negative penalty
    fields like `leader_move_penalty: -0.01` contribute -0.01 per
    fire; positive shaping fields like `village_delta: 0.05`
    contribute +0.05 per village swing.

    File format -- all keys optional; missing keys keep WeightedReward
    defaults::

        {
          "gold_killed_delta": 0.01,
          "village_delta": 0.05,
          "leader_move_penalty": -0.01,
          "unit_type_bonuses": [
            {"unit_type": "Wose", "weight": 0.5},
            {"unit_type": "Elvish Fighter", "weight": 0.05}
          ],
          "turn_conditional_bonuses": [
            {
              "name": "early_village",
              "turn_range": [1, 3],
              "predicate": "leader_on_village",
              "weight": 1.0,
              "once": true
            }
          ]
        }

    Format dispatch by extension: `.json` -> `json.load`, `.yaml` /
    `.yml` -> `yaml.safe_load` (requires PyYAML). Predicate names
    resolve through `get_predicate`; unknown names raise KeyError
    with the available list.
    """
    from pathlib import Path
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".json":
        import json
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                f"loading {p} requires PyYAML (`pip install pyyaml`); "
                f"or convert the config to .json"
            ) from e
        with p.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError(
            f"unsupported reward-config extension {suffix!r}; "
            f"use .json, .yaml, or .yml")
    if not isinstance(data, dict):
        raise ValueError(
            f"reward-config root must be a dict; got {type(data).__name__}")

    # Pull out the structured lists -- WeightedReward's other fields
    # are scalar-only, so they pass through as-is.
    raw_unit_bonuses = data.pop("unit_type_bonuses", []) or []
    raw_turn_bonuses = data.pop("turn_conditional_bonuses", []) or []

    unit_type_bonuses: List[UnitTypeBonus] = []
    for entry in raw_unit_bonuses:
        if not isinstance(entry, dict):
            raise ValueError(
                f"unit_type_bonuses entry must be a dict; got {entry!r}")
        unit_type_bonuses.append(UnitTypeBonus(
            unit_type=str(entry["unit_type"]),
            weight=float(entry["weight"]),
        ))

    turn_conditional_bonuses: List[TurnConditionalBonus] = []
    for entry in raw_turn_bonuses:
        if not isinstance(entry, dict):
            raise ValueError(
                f"turn_conditional_bonuses entry must be a dict; got {entry!r}")
        pred_name = str(entry["predicate"])
        predicate = get_predicate(pred_name)   # KeyError on miss
        turn_range = entry["turn_range"]
        if not (isinstance(turn_range, (list, tuple)) and len(turn_range) == 2):
            raise ValueError(
                f"turn_range must be [lo, hi]; got {turn_range!r}")
        turn_conditional_bonuses.append(TurnConditionalBonus(
            name=str(entry["name"]),
            turn_range=(int(turn_range[0]), int(turn_range[1])),
            predicate=predicate,
            weight=float(entry["weight"]),
            once=bool(entry.get("once", True)),
        ))

    # Validate the remaining (scalar) keys against WeightedReward
    # fields so a typo'd key is caught at load time, not silently
    # ignored. Keys that start with `_` are treated as inert
    # documentation (e.g. `_about`, `_generated_at`); they pass
    # through unchallenged but don't end up on the WeightedReward
    # instance.
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(WeightedReward)}
    valid_fields -= {"unit_type_bonuses", "turn_conditional_bonuses",
                     "_fired_once"}
    doc_keys = [k for k in data if k.startswith("_")]
    for k in doc_keys:
        del data[k]
    for k in data:
        if k not in valid_fields:
            raise ValueError(
                f"reward-config has unknown key {k!r}; valid scalar "
                f"keys: {sorted(valid_fields)}")

    return WeightedReward(
        unit_type_bonuses=unit_type_bonuses,
        turn_conditional_bonuses=turn_conditional_bonuses,
        **data,
    )
