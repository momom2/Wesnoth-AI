"""Self-play training using the in-process Wesnoth simulator.

Why this exists: subprocess-based self-play (the old `game_manager.py`
plan) requires Wesnoth + IPC + Lua, none of which work reliably on
Windows or on the cluster. The Python simulator (`tools/wesnoth_sim.py`)
runs the same game logic in-process at ~1000x the speed and is
trivially cluster-portable. This module is the glue that connects
the existing TransformerPolicy (model + encoder + trainer) to
WesnothSim, drives self-play games, and applies REINFORCE+baseline
gradient updates.

The loop, per iteration:

  1. Sample N initial states from `replays_dataset/*.json.gz`.
  2. For each, build a WesnothSim and roll out the model against
     itself, calling `policy.select_action` for every action and
     `policy.observe` for the per-step shaping reward.
  3. On terminal, emit per-side terminal rewards (WIN/LOSS/TIMEOUT)
     so the trajectory closes cleanly.
  4. Call `policy.train_step()` to apply one gradient update across
     all queued trajectories (one per side per game).

Reward function: the existing `WeightedReward` from rewards.py. It
diffs pre/post step GameStates -- we deepcopy the GameState before
each step so the diff is well-defined (the sim mutates in place,
swapping the units set, the sides list, and global_info attributes).
deepcopy is ~0.5ms per call; an 80-turn game with ~200 actions/side
adds ~200ms of overhead, negligible against any backward pass.

Usage:
    python tools/sim_self_play.py
        --checkpoint-in training/checkpoints/supervised_epoch3.pt
        --replay-pool   replays_dataset
        --iterations    50
        --games-per-iter 8
        --max-turns     40
        --save-every    10
        --checkpoint-out training/checkpoints/sim_selfplay.pt

A "game" produces TWO trajectories (one per side); --games-per-iter
controls game count, not trajectory count. Set --max-turns
conservatively early on -- the dummy gets 20 turns done in <1s but a
trained policy might play full-length games.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Make project root importable when run as a script.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

from classes import GameState, Unit
from tools.scenario_pool import LADDER_SCENARIO_IDS
from tools.scenario_pool import classify_scenario as _classify_scenario
from rewards import (
    OUTCOME_DRAW, OUTCOME_LOSS, OUTCOME_ONGOING, OUTCOME_TIMEOUT, OUTCOME_WIN,
    StepDelta, WeightedReward, compute_delta, hex_distance, load_reward_config,
)
from transformer_policy import TransformerPolicy
from wesnoth_sim import PvPDefaults, SimResult, WesnothSim
import openers as openers_mod
from openers import Opener, OpenerPolicy


log = logging.getLogger("sim_self_play")


# ---------------------------------------------------------------------
# Reward bookkeeping
# ---------------------------------------------------------------------

@dataclass
class GameOutcome:
    """Per-game summary printed in the iteration log."""
    game_label:     str
    winner:         int          # 0 = draw / timeout, else side index
    ended_by:       str
    turns:          int
    side1_actions:  int
    side2_actions:  int
    side1_reward:   float        # cumulative shaping + terminal for side 1
    side2_reward:   float
    # Action-type tallies across both sides; keys are the strings
    # action_sampler emits ("recruit", "move", "attack", "end_turn").
    # Used by run_iteration() to print an action histogram so the
    # operator can see at a glance what the policy is actually doing
    # (e.g. "97% recruit + end_turn, 0% attack" = the no-kills signal).
    action_counts:  Dict[str, int]
    # Living-unit counts at game end, per side. A useful "did the
    # game actually decide something" metric -- a draw with 20+ units
    # alive on each side is a different failure mode from a draw
    # with both armies wiped out.
    side1_units_end: int
    side2_units_end: int
    # Min hex distance any of THIS side's units ever got to the
    # OPPOSING side's leader across the whole game. None when the
    # opposing leader was never present (rare; usually means the
    # game ended on a leader-kill before this side acted). The
    # smaller, the more threatening: 1 = adjacent (attack range);
    # >10 = the army never closed on the leader. This is the
    # headline no-kills signal in the trainer log.
    side1_closest_approach: Optional[int]
    side2_closest_approach: Optional[int]
    # Econ diagnostics (added 2026-05-20 for recruit-underuse
    # investigation). end_gold_* = gold each side had when the game
    # ended -- if persistently high, the policy could afford more
    # units but isn't choosing them. n_recruits_* = total successful
    # recruits; n_recruit_attempts_* = total recruit picks
    # (including bounces). Bounces = attempts - successes.
    side1_end_gold: int = 0
    side2_end_gold: int = 0
    n_recruits_s1: int = 0
    n_recruits_s2: int = 0
    n_recruit_attempts_s1: int = 0
    n_recruit_attempts_s2: int = 0
    # Which map class produced this game. The AGGREGATE decisive rate
    # over a mixed curriculum is MISLEADING (proven 2026-07-03: the
    # trainer log read ~50% decisive while ladder maps were 0/8
    # decisive — every kill came from the mini half, amplified by the
    # pool deadline abandoning slow ladder draws). "ladder" | "mini" |
    # "drill" | "" (unknown / legacy producer).
    map_class: str = ""
    # Accepted actions per SIDE-TURN (one side's decision sequence
    # within one turn), pooled across the game. MCTS depth
    # calibration input: at ~A actions per side-turn, S sims explore
    # roughly S/A of one turn plan ahead (2026-07-03 user request).
    turn_action_counts: List[int] = field(default_factory=list)


def _outcome_for(winner: int, ended_by: str, side: int) -> str:
    """Map WesnothSim's (winner, ended_by) into per-side OUTCOME_* keys
    used by `WeightedReward`."""
    if ended_by in ("max_turns", "max_actions"):
        return OUTCOME_TIMEOUT
    if winner == 0:
        return OUTCOME_DRAW
    return OUTCOME_WIN if winner == side else OUTCOME_LOSS


# Cost lookup for recruit shaping. Re-uses the same unit_stats.json
# the sim already loads, so we don't drift between sim and rewards.
def _recruit_cost_lookup() -> Dict[str, int]:
    path = Path(__file__).resolve().parent.parent / "unit_stats.json"
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        log.warning("unit_stats.json not found; recruit cost defaults to 14")
        return {}
    return {n: int(u.get("cost", 14)) for n, u in data.get("units", {}).items()}


# ---------------------------------------------------------------------
# One game's rollout
# ---------------------------------------------------------------------

def _leader_of(gs: GameState, side: int) -> Optional[Unit]:
    """First is_leader=True unit for `side`, or None if the leader
    is dead / hasn't been placed yet. Shared with
    `tools/diagnose_selfplay.py` (which imports this)."""
    for u in gs.map.units:
        if u.side == side and u.is_leader:
            return u
    return None


def _update_closest_approach(
    gs: GameState, current: Dict[int, Optional[int]],
) -> None:
    """For sides 1 and 2 present on the board, compute min hex
    distance from any of that side's units to the OPPOSING side's
    leader. Update `current[side]` to the running minimum.

    Symmetric: both sides are sampled every state regardless of
    whose turn it is, so the metric captures "did this side's
    army ever get close to the enemy leader" across the whole
    game, not just on its turns.

    O(N_units) per call; with ~10-30 units per state and one call
    per action (~200 actions/game) this is ~3% overhead on a
    typical iter -- worth it for a metric that's the headline
    diagnostic for the no-kills problem.
    """
    # Cache leader positions for sides 1 and 2 up front.
    leader_pos: Dict[int, Tuple[int, int]] = {}
    for s in (1, 2):
        leader = _leader_of(gs, s)
        if leader is not None:
            leader_pos[s] = (leader.position.x, leader.position.y)
    for my_side in (1, 2):
        # Find the enemy leader's position (the OTHER player-side).
        other = 2 if my_side == 1 else 1
        if other not in leader_pos:
            continue
        ex, ey = leader_pos[other]
        my_units = [u for u in gs.map.units if u.side == my_side]
        if not my_units:
            continue
        local_min = min(
            hex_distance(u.position.x, u.position.y, ex, ey)
            for u in my_units
        )
        prev = current.get(my_side)
        if prev is None or local_min < prev:
            current[my_side] = local_min


def _would_recruit_bounce(action: Dict, gs: "GameState") -> bool:
    """True if `action` is a recruit on a hex the SIM (god-view)
    knows is occupied. The action sampler's mask only knows visible
    units, so the model can pick a fog-hidden castle hex -- the sim
    has ground truth and would reject. We use this to detect
    bounces in the harness BEFORE calling sim.step, so the side's
    turn isn't consumed by a no-op.

    Cheap: linear over gs.map.units (~10-30 entries on a typical
    mid-game state); fires only on recruit actions.
    """
    if action.get("type") != "recruit":
        return False
    tgt = action.get("target_hex")
    if tgt is None:
        return False
    for u in gs.map.units:
        if u.position.x == tgt.x and u.position.y == tgt.y:
            return True
    return False


def _would_move_bounce_on_fog(action: Dict, gs: "GameState") -> bool:
    """True if `action` is a MOVE onto a hex the sim's god-view
    knows is occupied by a unit invisible to the acting side.

    Why this case is special: with the fog-of-war filter
    (visibility.py + encoder.py), enemy units in fog don't appear
    in the sampler's enemy_mask -- the target hex looks empty to
    the policy. The sim still rejects the move (Wesnoth never
    allows stacking) and, without this pre-check, the rejection
    would be treated as an illegal-shaped action and consume the
    side's turn via the end_turn fallback (wesnoth_sim.py step()).

    Fair-information principle: the policy picked a hex that
    LOOKED legal given everything it could see. Punishing it for
    information it didn't have would violate the legality-mask
    contract in CLAUDE.md. Instead, the harness anticipates the
    bounce, marks the hex in `_move_rejected_hexes` (mirrored
    into hex_dynamic_flags bit 1 by the encoder), and re-decides
    -- same shape as the recruit-rejection retry.

    NOT a bounce-on-fog (these are policy bugs that should still
    cost the turn or trip the invalid_action signal):
      * Move onto a hex occupied by a VISIBLE enemy or friendly.
        Visible occupancy already maps to enemy_mask=1 or
        occupancy=1 in the legality mask, so the policy
        shouldn't have picked it. If it did, that's the model
        being wrong on visible state -- not fair information.
      * Move onto a non-neighbour hex / from a unit that has no
        MP / etc. Those checks happen earlier in
        _action_to_command and return None outright.

    Cheap: linear over gs.map.units, fires only on move actions.
    """
    if action.get("type") != "move":
        return False
    tgt = action.get("target_hex")
    if tgt is None:
        return False
    current_side = gs.global_info.current_side
    # Find the unit at the target (if any). If it's invisible to
    # the acting side, the bounce qualifies as fair-information.
    occupant = None
    for u in gs.map.units:
        if u.position.x == tgt.x and u.position.y == tgt.y:
            occupant = u
            break
    if occupant is None:
        return False
    if occupant.side == current_side:
        # Friendly already-occupied. The policy can see its own
        # units -- not a fog bounce.
        return False
    # Enemy unit. Is it visible to the acting side?
    from visibility import units_visible_to
    visible_ids = {id(u) for u in units_visible_to(gs, current_side)}
    return id(occupant) not in visible_ids


def play_one_game(
    sim:         WesnothSim,
    policy:      TransformerPolicy,
    reward_fn,
    *,
    game_label:  str,
    cost_lookup: Dict[str, int],
) -> GameOutcome:
    """Drive `sim` to completion, calling policy.select_action +
    policy.observe at each step. Returns the per-game summary."""
    side1_reward = 0.0
    side2_reward = 0.0
    # Action-type histogram across both sides. We tally on the
    # pre-bounce-retry action (the one sim.step accepted), so a
    # rejected-recruit-then-different-pick lands as a single entry
    # under the picked type -- the histogram reflects what the sim
    # actually did, not what the model first guessed.
    action_counts: Dict[str, int] = {}
    # Recruit diagnostics for the per-iter recruit-underuse audit.
    # n_recruits_per_side counts SUCCESSFUL recruits (state changed,
    # unit appeared); n_recruit_attempts counts every recruit action
    # the policy attempted (including bounced ones). Difference =
    # bounces (god-view occupied hex). Compared against gold-at-
    # end-of-game to answer "could the policy have afforded more
    # recruits but chose not to?".
    n_recruits_per_side: Dict[int, int] = {1: 0, 2: 0}
    n_recruit_attempts_per_side: Dict[int, int] = {1: 0, 2: 0}
    # (turn_number, side) -> accepted actions in that side-turn.
    turn_action_tally: Dict[tuple, int] = {}
    # Per-side closest-approach tracker, updated after every sim.step
    # (and seeded from the initial state below). Surfaces as the
    # headline no-kills metric in the iter log.
    closest_approach: Dict[int, Optional[int]] = {}
    # Seed with the starting position so a game that ends before any
    # side acts (rare) still has a measurement.
    _update_closest_approach(sim.gs, closest_approach)
    # Populated lazily as each side acts. We don't pre-seed (1, 2)
    # because the sim handles N-side replays (some have 3+ declared
    # sides) and we want every side that called select_action to
    # get its terminal observe at game end -- otherwise its
    # `_pending` entries leak.
    last_acting_side: Dict[int, bool] = {}

    # Optimization #6: skip per-step shaping when the policy ignores
    # it (MCTS). Read the flag once. Default True keeps any unknown
    # policy on the REINFORCE behavior.
    uses_step_rewards = getattr(policy, "uses_step_rewards", True)

    while not sim.done:
        acting_side = sim.gs.global_info.current_side
        # Snapshot the state BEFORE the step. Two reasons it has to
        # be a deepcopy and not just `sim.gs`:
        #   1) policy.select_action stores the state ref in a
        #      Transition; the trainer reforwards on it later. If we
        #      passed sim.gs directly, sim.step would mutate the
        #      very state the trainer is reforwarding -- the stored
        #      action index then points at a slot now masked out by
        #      the legality mask in the post-step state, log_prob
        #      collapses to -inf, and the policy_loss explodes.
        #   2) compute_delta needs (pre, post) to diff. sim.step
        #      replaces sim.gs.map.units / sides / global_info, so a
        #      saved reference IS pre; the live sim.gs IS post.
        pre_state = copy.deepcopy(sim.gs)
        action = policy.select_action(pre_state, game_label=game_label, sim=sim)

        # Recruit-rejection retry loop. Per the legality-mask
        # contract (CLAUDE.md): a recruit attempt on a hex that the
        # MODEL thinks is empty (visible state) but is actually
        # occupied (god-view, e.g. by a fog-hidden enemy) is
        # rejected. We re-decide WITHOUT consuming the side's turn
        # -- the model sees the new rejection state via the per-hex
        # feature + mask and tries again. Loop bounds: each
        # rejection adds one hex to the set; eventually the
        # recruit-mask exhausts (no legal hexes) and the policy
        # picks a different action type. No K-cap needed: the mask
        # shrinks monotonically within a turn.
        #
        # We pre-check occupancy here (rather than letting sim.step
        # do it) so sim.step's contract stays simple: every step()
        # call advances the game by exactly one accepted action.
        # The pre-check is cheap: a single pass over gs.map.units.
        while _would_recruit_bounce(action, sim.gs):
            tgt = action["target_hex"]
            rejected = (
                getattr(sim.gs.global_info,
                        "_recruit_rejected_hexes", None) or set()
            )
            rejected.add((tgt.x, tgt.y))
            setattr(sim.gs.global_info,
                    "_recruit_rejected_hexes", rejected)
            log.debug(
                f"recruit rejected: {action.get('unit_type')!r} on "
                f"({tgt.x},{tgt.y}) (god-view occupied); "
                f"re-deciding with hex blacklisted")
            # Undo the rejected select_action's recorded target. MCTSPolicy
            # exposes drop_last_pending (pops the pending MCTSExperience tail
            # AND rolls back the decision_step increment) -- without it the
            # bounced decision would be trained on with terminal z and
            # over-advance the combat-oracle anneal. Policies without it
            # (REINFORCE) fall back to observe(reward=0), which lands the
            # rejected pick in the trajectory with a neutral signal.
            drop = getattr(policy, "drop_last_pending", None)
            if not (callable(drop) and drop(game_label)):
                policy.observe(game_label, acting_side, 0.0, done=False)
            pre_state = copy.deepcopy(sim.gs)
            action = policy.select_action(pre_state, game_label=game_label, sim=sim)

        # Same mechanic for MOVE onto a fog-hidden enemy hex.
        # Fair-information contract: the hex looked empty to the
        # policy (the enemy unit was filtered out of the encoder's
        # tokens by visibility.units_visible_to), so the bounce
        # isn't punishable. Mark the hex in `_move_rejected_hexes`,
        # observe(reward=0) to drop the rejected Transition tail,
        # and re-decide. Loop bound: each bounce adds one hex to
        # the per-turn rejection set; the legality mask blacklists
        # it; the move-target choice space shrinks monotonically
        # within the turn. (Worst case the policy exhausts its
        # legal moves and picks attack / end_turn instead.)
        while _would_move_bounce_on_fog(action, sim.gs):
            tgt = action["target_hex"]
            move_rejected = (
                getattr(sim.gs.global_info,
                        "_move_rejected_hexes", None) or set()
            )
            move_rejected.add((tgt.x, tgt.y))
            setattr(sim.gs.global_info,
                    "_move_rejected_hexes", move_rejected)
            log.debug(
                f"move rejected: ({tgt.x},{tgt.y}) (god-view "
                f"occupied by fog-hidden enemy); re-deciding "
                f"with hex blacklisted")
            drop = getattr(policy, "drop_last_pending", None)
            if not (callable(drop) and drop(game_label)):
                policy.observe(game_label, acting_side, 0.0, done=False)
            pre_state = copy.deepcopy(sim.gs)
            action = policy.select_action(pre_state, game_label=game_label, sim=sim)
        atype = action.get("type", "end_turn")
        if atype == "recruit":
            unit_type = action.get("unit_type", "")
            recruit_cost = cost_lookup.get(unit_type)
            if recruit_cost is None:
                # Unknown unit type (custom era / out-of-date scrape /
                # typo). Fall back to 14 -- the smallfoot/orcishfoot
                # baseline -- not 0. Zero would silently zero the
                # gold-spent shaping for this recruit, biasing the
                # policy toward repeatedly picking the unknown type
                # (high reward / no apparent gold cost). Log once so
                # we notice the missing entry without spamming.
                if not getattr(_recruit_cost_lookup, "_warned",
                               set()).__contains__(unit_type):
                    log.warning(
                        f"recruit cost: unit type {unit_type!r} not in "
                        f"unit_stats.json; falling back to 14. "
                        f"Re-run tools/scrape_unit_stats.py if this "
                        f"unit is from a freshly-installed era.")
                    if not hasattr(_recruit_cost_lookup, "_warned"):
                        _recruit_cost_lookup._warned = set()
                    _recruit_cost_lookup._warned.add(unit_type)
                recruit_cost = 14
        else:
            recruit_cost = 0

        sim.step(action)
        action_counts[atype] = action_counts.get(atype, 0) + 1
        # Per-side-turn action tally: how many decisions one side
        # makes within one turn. This is the MCTS depth calibration
        # input — at ~A actions per side-turn, a search of S sims
        # only looks ~S/A "turns" ahead within its own turn plan.
        # Keyed on the PRE-step turn/side (end_turn advances them).
        _tk = (pre_state.global_info.turn_number, acting_side)
        turn_action_tally[_tk] = turn_action_tally.get(_tk, 0) + 1
        _update_closest_approach(sim.gs, closest_approach)

        # Recruit diagnostics: did this recruit attempt actually
        # produce a new unit? `delta.units_recruited` populates
        # below from compute_delta. For now, count the attempt
        # (any recruit-typed action that reached sim.step). A
        # bounced fog/occupied recruit went through the retry
        # loop above and is NOT in this branch -- so attempts
        # here are non-bounced; successes get counted when the
        # delta is computed.
        if atype == "recruit":
            n_recruit_attempts_per_side[acting_side] = (
                n_recruit_attempts_per_side.get(acting_side, 0) + 1)

        # Per-step shaping reward only -- no terminal contribution
        # here. Terminal reward is added per side after the loop so
        # each side's terminal payoff is attached to its OWN last
        # transition, even when the game ended on the other side's
        # killing move.
        #
        # `attach_post_state` is opt-in based on whether the reward
        # function has turn-conditional bonuses configured; doing it
        # unconditionally would retain a deepcopy-equivalent
        # reference per Transition.
        if uses_step_rewards:
            attach_post = bool(getattr(reward_fn,
                                       "turn_conditional_bonuses", None))
            delta = compute_delta(
                pre_state, sim.gs, atype,
                recruit_cost=recruit_cost,
                outcome=OUTCOME_ONGOING,
                game_label=game_label,
                attach_post_state=attach_post,
            )
            step_r = reward_fn(delta)
            # Count successful recruits (state delta confirms a unit
            # appeared). units_recruited is a tuple of names; empty
            # if the recruit didn't actually land.
            recruited = (len(delta.units_recruited)
                         if atype == "recruit" else 0)
        else:
            # Optimization #6: MCTS discards per-step shaping (observe
            # is a no-op; z comes from the winner), so skip the
            # compute_delta Dijkstra entirely. Recover ONLY the
            # recruit-success diagnostic via a cheap pre/post
            # unit-count diff (a recruit adds exactly one unit, or
            # zero if it bounced) -- benchmarked equal to
            # delta.units_recruited.
            step_r = 0.0
            recruited = 0
            if atype == "recruit":
                pre_n = sum(1 for u in pre_state.map.units
                            if u.side == acting_side)
                post_n = sum(1 for u in sim.gs.map.units
                             if u.side == acting_side)
                recruited = max(0, post_n - pre_n)
        if atype == "recruit" and recruited:
            n_recruits_per_side[acting_side] = (
                n_recruits_per_side.get(acting_side, 0) + recruited)
        policy.observe(game_label, acting_side, step_r, done=False)
        last_acting_side[acting_side] = True
        if acting_side == 1:
            side1_reward += step_r
        else:
            side2_reward += step_r

    # Game over. Emit terminal reward to every side that ACTED so
    # each trajectory the policy started actually gets a terminal
    # observe(done=True). Iterating over `last_acting_side` (rather
    # than a hardcoded (1, 2) tuple) handles replays declaring more
    # than 2 sides: WesnothSim's `_apply_command(end_turn)` cycles
    # `current_side` modulo `len(gs.sides)`, so a 3- or 4-side
    # replay's `select_action` keys land on sides > 2. Without this
    # they leak forever in `_pending`.
    final_turn = sim.gs.global_info.turn_number
    for side, acted in list(last_acting_side.items()):
        if not acted:
            # Side never acted (e.g. game ended before they got a
            # chance). Nothing to attach the reward to.
            continue
        outcome = _outcome_for(sim.winner, sim.ended_by, side)
        term_delta = StepDelta(
            side=side,
            turn=final_turn,
            action_type="terminal",
            outcome=outcome,
            game_label=game_label,
        )
        terminal_r = reward_fn(term_delta)
        policy.observe(game_label, side, terminal_r, done=True)
        if side == 1:
            side1_reward += terminal_r
        elif side == 2:
            side2_reward += terminal_r

    # MCTS-mode hook: REINFORCE policy is a no-op here; the MCTS
    # wrapper (tools.mcts_policy.MCTSPolicy) drains its per-game
    # `_pending` into the trainer queue with the terminal z derived
    # from `winner`. Defined as a no-op on TransformerPolicy so the
    # call is unconditional.
    policy.finalize_game(game_label, sim.winner, final_gs=sim.gs)

    # Living-unit counts at game end. Counts every unit on the
    # final-state board belonging to each side -- includes leaders.
    # A side-1 win normally leaves side2_units_end at >0 (the leader
    # died, surviving units don't matter) but useful: a draw with
    # 30 units alive on each side is "two armies sat around" while
    # a draw with 2 units alive is "they wiped each other out".
    s1_units_end = 0
    s2_units_end = 0
    for u in sim.gs.map.units:
        if u.side == 1:
            s1_units_end += 1
        elif u.side == 2:
            s2_units_end += 1

    # End-of-game gold per side. Defensive against scenarios with
    # missing sides[] entries (test fixtures).
    s1_gold = (int(sim.gs.sides[0].current_gold)
               if sim.gs.sides else 0)
    s2_gold = (int(sim.gs.sides[1].current_gold)
               if sim.gs.sides and len(sim.gs.sides) >= 2 else 0)
    return GameOutcome(
        game_label=game_label,
        winner=sim.winner,
        ended_by=sim.ended_by,
        turns=final_turn,
        side1_actions=sim._actions_by_side.get(1, 0),
        side2_actions=sim._actions_by_side.get(2, 0),
        side1_reward=side1_reward,
        side2_reward=side2_reward,
        action_counts=action_counts,
        side1_units_end=s1_units_end,
        side2_units_end=s2_units_end,
        side1_closest_approach=closest_approach.get(1),
        side2_closest_approach=closest_approach.get(2),
        side1_end_gold=s1_gold,
        side2_end_gold=s2_gold,
        n_recruits_s1=n_recruits_per_side.get(1, 0),
        n_recruits_s2=n_recruits_per_side.get(2, 0),
        n_recruit_attempts_s1=n_recruit_attempts_per_side.get(1, 0),
        n_recruit_attempts_s2=n_recruit_attempts_per_side.get(2, 0),
        map_class=_classify_scenario(getattr(sim, "scenario_id", "")
                                     or ""),
        turn_action_counts=sorted(turn_action_tally.values()),
    )


# ---------------------------------------------------------------------
# Initial-state pool
# ---------------------------------------------------------------------

# Self-play seed pool: union of the Ladder Era's three official map
# packs (Competitive + Classic + Adventurous, 21 scenarios total).
# Pulled from the single source-of-truth list in scenario_pool so
# the same canonical set drives both scenario sampling
# (`random_setup`) and replay-corpus filtering (`_is_ladder_map`).
# Wesnoth-side canonical IDs are verified case-exact against
# `wesnoth_src/data/multiplayer/scenarios/2p_*.cfg`.
_LADDER_MAP_SCENARIO_IDS: frozenset = frozenset(LADDER_SCENARIO_IDS)


def _is_ladder_map(scenario_id: str) -> bool:
    """True if `scenario_id` matches a Ladder Era map (any pack).
    Tolerates `_Ladder_Random` and `_Ladder` suffixes the Ladder Era
    add-on appends when a game is launched via its random-pool
    picker (e.g. `multiplayer_Basilisk_Ladder_Random` ->
    `multiplayer_Basilisk`)."""
    s = scenario_id
    for suffix in ("_Ladder_Random", "_Ladder"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s in _LADDER_MAP_SCENARIO_IDS


def _gather_replay_pool(replay_pool: Path) -> List[Path]:
    """Return .json.gz files under `replay_pool` filtered to the
    Ladder Era's 21-map PvP pool (Competitive + Classic +
    Adventurous packs).

    Why this filter: replays in the corpus include

      - **Co-op survival** (e.g. multiplayer_2p_Dark_Forecast):
        sides 1 and 2 are AI-controlled enemy waves; humans play
        sides 3 and 4. The sim runs select_action for sides 1 and
        2 only -> useless for PvP training.
      - **Custom maps / niche scenarios:** scenarios that aren't on
        any official PvP map list, with potentially weird stat
        balance, custom WML events, or deviating starting golds.
        Risk: noise.
      - **Real PvP on the Ladder Era's 21 maps:** the standard
        sanctioned PvP pool. KEEP.

    The Ladder Era add-on is the authoritative source for the
    "PvP-quality" map list, since the human ladder community
    explicitly curates it. Whitelisting against that list gives
    self-play a clean, focused state distribution and matches what
    we want the model to be good at.

    Sanity check: also reject replays whose sides-1/2 factions are
    `Custom` (survival mode marker) -- defense in depth in case a
    scenario_id matches the whitelist but the game was somehow
    re-routed.

    Without index.jsonl, we fall back to using all .json.gz
    unfiltered and warn loudly.
    """
    pool = Path(replay_pool)
    files = sorted(pool.glob("*.json.gz"))
    if not files:
        raise RuntimeError(f"No .json.gz files in {pool}")
    idx_path = pool / "index.jsonl"
    if idx_path.exists():
        keep_names: set = set()
        per_map: dict = {}
        n_seen = 0
        n_dropped_scenario = 0
        n_dropped_factions = 0
        with idx_path.open() as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_seen += 1
                sid = e.get("scenario_id", "")
                if not _is_ladder_map(sid):
                    n_dropped_scenario += 1
                    continue
                facs = e.get("factions") or []
                if (len(facs) < 2
                        or facs[0] in ("Custom", "")
                        or facs[1] in ("Custom", "")):
                    n_dropped_factions += 1
                    continue
                keep_names.add(e.get("file", ""))
                # Strip suffix for the per-map count so Ladder
                # Random and plain forms aggregate.
                base = sid
                for suf in ("_Ladder_Random", "_Ladder"):
                    if base.endswith(suf):
                        base = base[: -len(suf)]
                        break
                per_map[base] = per_map.get(base, 0) + 1
        if keep_names:
            files = [f for f in files if f.name in keep_names]
            if not files:
                raise RuntimeError(
                    "index.jsonl filtered out every replay -- check "
                    "the ladder filter logic")
            log.info(
                f"replay pool: kept {len(files)}/{n_seen} ladder-map PvP "
                f"replays (dropped {n_dropped_scenario} non-ladder "
                f"scenarios, {n_dropped_factions} co-op / AI sides 1-2)"
            )
            log.info(
                f"  per-map: " + ", ".join(
                    f"{name.replace('multiplayer_', '')}={count}"
                    for name, count in sorted(
                        per_map.items(), key=lambda kv: -kv[1])
                )
            )
            return files
    log.warning(
        f"replay pool: no index.jsonl in {pool}; using all "
        f"{len(files)} .json.gz files unfiltered (will likely "
        f"include non-ladder scenarios -- generate index.jsonl "
        f"via tools/replay_extract.py for proper filtering)"
    )
    return files


# ---------------------------------------------------------------------
# Iteration loop
# ---------------------------------------------------------------------

def _play_one_game_safe(
    *, setup, max_turns, pvp_defaults, policy, reward_fn,
    cost_lookup, game_label,
) -> Optional[GameOutcome]:
    """Run one game end-to-end from a `ScenarioSetup` (random
    scenario + faction + leader picks). Catches exceptions, drops
    pending transitions on crash, returns None on failure.

    Pre-pivot this used `WesnothSim.from_replay(<replay_path>)`.
    Post-pivot (2026-04-30) it builds the GameState directly from
    scenario .cfg + map + faction data via
    `tools.scenario_pool.build_scenario_gamestate`. No replay
    file involved. Scenario events fire in `WesnothSim.__init__`
    (CoB neutrals, Aethermaw morph, etc.).
    """
    from tools.scenario_pool import build_scenario_gamestate
    # Map pvp_defaults onto build_scenario_gamestate kwargs.
    sg = (pvp_defaults.starting_gold if pvp_defaults else 100)
    bi = (pvp_defaults.base_income   if pvp_defaults else 2)
    vg = (pvp_defaults.village_gold  if pvp_defaults else 2)
    vu = (pvp_defaults.village_support if pvp_defaults else 1)
    em = (pvp_defaults.experience_modifier if pvp_defaults else 70)
    try:
        gs = build_scenario_gamestate(
            setup,
            starting_gold=sg, base_income=bi,
            village_gold=vg, village_upkeep=vu,
            experience_modifier=em,
        )
        sim = WesnothSim(gs, scenario_id=setup.scenario_id,
                         max_turns=max_turns)
    except Exception as e:
        log.warning(f"skipping {setup.label()}: {e}")
        return None
    if hasattr(policy, "reset_game"):
        policy.reset_game(game_label)
    if hasattr(reward_fn, "reset_game_state"):
        reward_fn.reset_game_state(game_label)
    try:
        return play_one_game(
            sim, policy, reward_fn,
            game_label=game_label, cost_lookup=cost_lookup,
        )
    except Exception as e:
        log.exception(f"game {game_label} crashed: {e}")
        policy.drop_pending(game_label)
        return None


def _worker_loop(
    *, worker_id, policy, reward_fn, cost_lookup,
    max_turns, pvp_defaults, worker_rng, shared,
    forced_faction=...,
    mini_maps=False,
    mini_ratio: float = 0.0,
    drill_ratio: float = 0.0,
):
    """Per-thread rollout loop. Each worker pulls a game index from
    `shared.next_game` (atomic under the master lock), assigns
    itself a unique game_label, runs one game, appends the outcome
    to `shared.outcomes`. Stops when `next_game` would exceed
    target_games. Uses scenario_pool.random_setup for the seed --
    no replay pool involved."""
    from tools.scenario_pool import random_setup
    while True:
        with shared["lock"]:
            if shared["next_game"] >= shared["target_games"]:
                return
            g_idx = shared["next_game"]
            shared["next_game"] += 1
        setup = random_setup(worker_rng, forced_faction=forced_faction,
                             mini_maps=mini_maps, mini_ratio=mini_ratio,
                             drill_ratio=drill_ratio)
        game_label = (f"iter{shared['iter_idx']}_"
                      f"w{worker_id}_g{g_idx}")
        outcome = _play_one_game_safe(
            setup=setup, max_turns=max_turns,
            pvp_defaults=pvp_defaults, policy=policy,
            reward_fn=reward_fn, cost_lookup=cost_lookup,
            game_label=game_label,
        )
        if outcome is not None:
            with shared["lock"]:
                shared["outcomes"].append(outcome)


class SpoolWorkers:
    """Spawn + supervise N independent self-play worker PROCESSES
    (tools/selfplay_worker.py) and collect their spooled games.

    The winning architecture from the 2026-07 measurements: each
    worker plays whole games in-process with its own GPU forwards (no
    inference server, no IPC — the central-server pool capped at
    ~200 req/s with the GPU idle; independent processes saturated
    it). The spool directory is the only seam: workers atomically
    write one pickle per game; the learner consumes, trains, and
    saves checkpoints that workers hot-reload between games."""

    def __init__(self, n: int, spool_dir: Path, checkpoint: Path,
                 args, log_level: str):
        import subprocess
        self._n = n
        self._dir = spool_dir / "games"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint = checkpoint
        self._cmd_tail = [
            "--checkpoint", str(checkpoint),
            "--spool-dir", str(spool_dir),
            "--mcts-sims", str(args.mcts_sims),
            "--mini-ratio", str(max(0.0, min(1.0, args.mini_ratio))),
            "--drill-ratio", str(max(0.0, min(1.0, args.drill_ratio))),
            "--max-turns", str(args.max_turns),
            "--draw-tiebreak-cap", str(max(0.0, args.draw_tiebreak_cap)),
            "--moves-left-utility", str(args.mcts_moves_left_utility),
            "--log-level", log_level,
        ]
        self._seed0 = args.seed * 1_000_003 + 7
        self._subprocess = subprocess
        self._procs: List = [None] * n
        for i in range(n):
            self._spawn(i)

    def _spawn(self, i: int) -> None:
        worker = Path(__file__).resolve().parent / "selfplay_worker.py"
        self._procs[i] = self._subprocess.Popen(
            [sys.executable, str(worker), "--worker-id", str(i),
             "--seed", str(self._seed0 + i)] + self._cmd_tail,
            stdout=self._subprocess.DEVNULL,
            stderr=self._subprocess.DEVNULL,
        )

    def ensure_alive(self) -> None:
        """Respawn crashed workers (called once per iteration)."""
        for i, p in enumerate(self._procs):
            if p is None or p.poll() is not None:
                rc = None if p is None else p.poll()
                log.warning(f"spool worker {i} died (rc={rc}); respawning")
                self._spawn(i)

    def collect(self, policy, want: int,
                timeout_s: float = 3600.0) -> List["GameOutcome"]:
        """Block until `want` spooled games are consumed (or timeout).
        Per game file: advance the global decision counter, offer the
        WHOLE game to the learner-side holdout, else queue it for
        training; delete the file."""
        import pickle as _pickle
        outcomes: List[GameOutcome] = []
        deadline = time.perf_counter() + timeout_s
        while len(outcomes) < want:
            files = sorted(self._dir.glob("game_*.pkl"))
            if not files:
                if time.perf_counter() > deadline:
                    log.error(
                        f"spool collect timed out with "
                        f"{len(outcomes)}/{want} games; check worker "
                        f"logs / respawn next iteration.")
                    break
                time.sleep(5.0)
                continue
            for f in files[:want - len(outcomes)]:
                try:
                    payload = _pickle.loads(f.read_bytes())
                except Exception as e:                  # noqa: BLE001
                    log.warning(f"unreadable spool file {f.name}: {e}")
                    f.unlink(missing_ok=True)
                    continue
                base = getattr(policy, "_base", policy)
                with base._lock:
                    base._decision_step += int(payload["n_decisions"])
                exps = payload["experiences"]
                offer = getattr(policy, "offer_holdout_game", None)
                if offer is None or not offer(exps):
                    with policy._lock:
                        policy._queue.extend(exps)
                outcomes.append(payload["outcome"])
                f.unlink(missing_ok=True)
        return outcomes

    def shutdown(self) -> None:
        for p in self._procs:
            if p is not None and p.poll() is None:
                p.terminate()


def _gpu_mem_mb():
    """(allocated_MB, reserved_MB) of the trainer process, or
    (None, None) off-CUDA. Reserved > allocated = allocator cache;
    reserved growth with flat allocated = fragmentation/ratchet."""
    import torch
    if not torch.cuda.is_available():
        return (None, None)
    return (round(torch.cuda.memory_allocated() / 2**20),
            round(torch.cuda.memory_reserved() / 2**20))


def run_iteration(
    policy:        TransformerPolicy,
    pool_files:    Optional[List[Path]],   # legacy; ignored post-pivot
    reward_fn,
    cost_lookup:   Dict[str, int],
    *,
    iter_idx:      int,
    games_per_iter: int,
    max_turns:     int,
    rng:           random.Random,
    pvp_defaults:  Optional[PvPDefaults] = None,
    workers:       int = 0,
    train_at_end:  bool = True,
    forced_faction: Optional[str] = ...,
    mini_maps:     bool = False,
    mini_ratio:    float = 0.0,
    drill_ratio:   float = 0.0,
    snapshot_sink: Optional[Callable[[Dict], None]] = None,
    actor_pool=None,
    spool=None,
) -> List[GameOutcome]:
    """Roll out `games_per_iter` games and call `train_step` once at
    the end. Returns the per-game outcomes for logging.

    `train_at_end`: when False, skip the post-rollout `train_step`
    call and leave queued trajectories on `policy._queue` for the
    caller to inspect or drain. Tests use this to assert queue/
    pending invariants directly; production callers leave it True.

    `pvp_defaults`: forwarded to `WesnothSim.from_replay`. When set,
    each game starts with standard 2p ladder economy/experience
    rather than whatever the source replay's host had configured.
    Self-play wants this -- it ensures the policy learns a single
    consistent ruleset rather than per-host quirks.

    `workers`: 0 = serial (existing behavior; runs on the main
    thread, simplest path). >= 1 = spawn N worker threads each
    pulling games from a shared counter and feeding trajectories
    to the policy concurrently. Builds on the snapshot+lock design
    in TransformerPolicy: workers' select_action / observe calls
    are thread-safe; the main thread fires train_step after all
    workers finish.

    Throughput: on CPU, the GIL serializes most of the forward
    work, so workers buy mostly the encode-while-other-thread-is-
    forwarding overlap (~30% speedup at workers=4). On GPU,
    multiple workers dispatching forwards in parallel keep the
    GPU saturated while the main thread also runs gradient compute
    (the snapshot design makes this safe).
    """
    outcomes: List[GameOutcome] = []
    t0 = time.perf_counter()

    # Reset the per-component reward accumulator at the start of the
    # iter. WeightedReward (if that's the reward_fn) accumulates per
    # component into _component_acc; we read it out at iter end and
    # log to trainer_history. No-op for non-WeightedReward callers
    # (custom reward fns are free to not expose this attribute).
    if hasattr(reward_fn, "_component_acc"):
        reward_fn._component_acc = {}

    if spool is not None:
        # Spool path (MCTS only): fully independent worker processes
        # play games with their own in-process GPU forwards and drop
        # one pickle per game; collect() advances the anneal counter,
        # routes whole games to the learner-side holdout, and queues
        # the rest for the shared train_step below.
        spool.ensure_alive()
        outcomes.extend(spool.collect(policy, games_per_iter))
    elif actor_pool is not None:
        # Actor-pool path (MCTS only): self-play runs in weightless
        # actor PROCESSES feeding this process's central batched-
        # inference server (see tools/actor_pool). The actors ship back
        # completed MCTSExperiences; we drain them into the learner's
        # queue here, and the shared train_step below applies the
        # gradient update exactly as in the in-process paths.
        base_seed = rng.randint(0, 2**31 - 1)
        pool_outcomes, pool_exps = actor_pool.run_iteration(
            iter_idx, games_per_iter, base_seed)
        outcomes.extend(pool_outcomes)
        with policy._lock:
            policy._queue.extend(pool_exps)
    elif workers <= 0:
        # Serial path -- simplest, used for tests and smoke runs.
        from tools.scenario_pool import random_setup
        for g_idx in range(games_per_iter):
            setup = random_setup(rng, forced_faction=forced_faction,
                                 mini_maps=mini_maps,
                                 mini_ratio=mini_ratio,
                                 drill_ratio=drill_ratio)
            game_label = f"iter{iter_idx}_g{g_idx}"
            outcome = _play_one_game_safe(
                setup=setup, max_turns=max_turns,
                pvp_defaults=pvp_defaults, policy=policy,
                reward_fn=reward_fn, cost_lookup=cost_lookup,
                game_label=game_label,
            )
            if outcome is not None:
                outcomes.append(outcome)
    else:
        # Parallel path -- N worker threads share the policy +
        # reward_fn + replay pool. Each worker has its own RNG
        # (seeded from the master rng) so games stay deterministic
        # given the master seed even with worker scheduling jitter.
        import threading
        shared = {
            "lock":           threading.Lock(),
            "next_game":      0,
            "target_games":   games_per_iter,
            "iter_idx":       iter_idx,
            "outcomes":       outcomes,
        }
        threads = []
        for w in range(workers):
            worker_rng = random.Random(rng.randint(0, 2**32 - 1))
            t = threading.Thread(
                target=_worker_loop,
                kwargs=dict(
                    worker_id=w, policy=policy,
                    reward_fn=reward_fn, cost_lookup=cost_lookup,
                    max_turns=max_turns, pvp_defaults=pvp_defaults,
                    worker_rng=worker_rng, shared=shared,
                    forced_faction=forced_faction,
                    mini_maps=mini_maps,
                    mini_ratio=mini_ratio,
                    drill_ratio=drill_ratio,
                ),
                daemon=True,
                name=f"selfplay-w{w}",
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    rollout_dt = time.perf_counter() - t0
    n_actions = sum(o.side1_actions + o.side2_actions for o in outcomes)
    n_turns = sum(o.turns for o in outcomes)
    log.info(
        f"iter {iter_idx}: rolled {len(outcomes)} games in {rollout_dt:.1f}s "
        f"({n_actions} actions, {n_turns} turns; "
        f"{n_actions/max(rollout_dt, 1e-9):.0f} actions/s)"
    )

    # Aggregate-side win-rate. winner==1 means side 1 won; we flip per
    # side so both contribute to a single "did the better policy win"
    # statistic. Currently both sides share the same policy, so we
    # just report side-1 wins / draws / losses.
    s1_wins = sum(1 for o in outcomes if o.winner == 1)
    s2_wins = sum(1 for o in outcomes if o.winner == 2)
    draws   = sum(1 for o in outcomes if o.winner == 0)
    avg_r1  = (sum(o.side1_reward for o in outcomes) / len(outcomes)) if outcomes else 0.0
    avg_r2  = (sum(o.side2_reward for o in outcomes) / len(outcomes)) if outcomes else 0.0
    log.info(
        f"iter {iter_idx}: outcomes s1_wins={s1_wins} s2_wins={s2_wins} "
        f"draws/timeouts={draws}; mean_reward s1={avg_r1:+.3f} s2={avg_r2:+.3f}"
    )

    # Behavioral diagnostics: action histogram + mean turns/game +
    # mean living units per side at game end. These three numbers
    # are the headline indicator of the no-kills problem: if the
    # histogram shows recruit + end_turn dominate and units_end is
    # high for both sides, the policy is "build forever, never
    # engage". A healthy training trajectory should see attack% rise
    # over iterations, mean_turns drop (decisive games end sooner),
    # and units_end skew toward the winner.
    # Computed below; declared up front so the train_at_end block can
    # write a CSV row regardless of whether outcomes were produced.
    action_pcts: Dict[str, float] = {k: 0.0 for k in
                                     ("recruit", "move", "attack",
                                      "end_turn", "other")}
    mean_turns: float = 0.0
    mean_u1: float = 0.0
    mean_u2: float = 0.0
    mean_ca1: Optional[float] = None
    mean_ca2: Optional[float] = None
    if outcomes:
        # Sum per-type tallies across games.
        totals: Dict[str, int] = {}
        for o in outcomes:
            for k, v in o.action_counts.items():
                totals[k] = totals.get(k, 0) + v
        total_actions = max(sum(totals.values()), 1)
        # Stable order so it's grep-able across runs; "other" sweeps
        # up anything that isn't one of the four known categories.
        known = ["recruit", "move", "attack", "end_turn"]
        for k in known:
            action_pcts[k] = 100.0 * totals.get(k, 0) / total_actions
        other = sum(v for k, v in totals.items() if k not in known)
        action_pcts["other"] = 100.0 * other / total_actions
        parts = [f"{k}={action_pcts[k]:.0f}%" for k in known]
        if other:
            parts.append(f"other={action_pcts['other']:.0f}%")
        mean_turns = n_turns / len(outcomes)
        mean_u1 = sum(o.side1_units_end for o in outcomes) / len(outcomes)
        mean_u2 = sum(o.side2_units_end for o in outcomes) / len(outcomes)
        # Closest-approach: average over games that have a value
        # (a game where this side's leader never met any opposing
        # unit returns None and is excluded). Smaller is more
        # threatening; high values across iterations = no-kills.
        ca1_vals = [o.side1_closest_approach for o in outcomes
                    if o.side1_closest_approach is not None]
        ca2_vals = [o.side2_closest_approach for o in outcomes
                    if o.side2_closest_approach is not None]
        if ca1_vals: mean_ca1 = sum(ca1_vals) / len(ca1_vals)
        if ca2_vals: mean_ca2 = sum(ca2_vals) / len(ca2_vals)
        ca_str = (f"  closest_approach s1="
                  f"{f'{mean_ca1:.1f}' if mean_ca1 is not None else 'n/a'}"
                  f" s2="
                  f"{f'{mean_ca2:.1f}' if mean_ca2 is not None else 'n/a'}")
        log.info(
            f"iter {iter_idx}: actions[{', '.join(parts)}] "
            f"mean_turns={mean_turns:.1f} "
            f"mean_units_end s1={mean_u1:.1f} s2={mean_u2:.1f}"
            f"{ca_str}"
        )

    # Per-map-class decisive split. The AGGREGATE decisive rate is
    # misleading over a mixed curriculum (2026-07-03: ~50% aggregate,
    # 0/8 decisive on ladder maps — all kills came from minis, and
    # the pool deadline abandoning slow ladder draws inflated it
    # further). Ladder decisiveness is the number that matters for
    # full-game strength; log both.
    ladder_n = sum(1 for o in outcomes if o.map_class == "ladder")
    ladder_dec = sum(1 for o in outcomes
                     if o.map_class == "ladder" and o.winner != 0)
    other_n = len(outcomes) - ladder_n
    other_dec = sum(1 for o in outcomes
                    if o.map_class != "ladder" and o.winner != 0)
    # Per-class SIDE split (2026-07-07: a 1-7 side-2 iteration was
    # only visible as an anecdote; global s1/s2 hides which map class
    # carries an asymmetry).
    ladder_s1 = sum(1 for o in outcomes
                    if o.map_class == "ladder" and o.winner == 1)
    ladder_s2 = ladder_dec - ladder_s1
    other_s1 = sum(1 for o in outcomes
                   if o.map_class != "ladder" and o.winner == 1)
    other_s2 = other_dec - other_s1
    if outcomes:
        log.info(
            f"iter {iter_idx}: decisive split -- ladder "
            f"{ladder_dec}/{ladder_n} (s1 {ladder_s1}, s2 {ladder_s2}), "
            f"mini/drill {other_dec}/{other_n} "
            f"(s1 {other_s1}, s2 {other_s2})")
    # Actions-per-side-turn distribution, pooled across the iter's
    # games (MCTS depth calibration: S sims / A actions-per-side-turn
    # ≈ how much of one turn plan the search can look ahead).
    pooled_apt = sorted(
        c for o in outcomes for c in (o.turn_action_counts or []))
    apt_mean = (sum(pooled_apt) / len(pooled_apt)) if pooled_apt else None
    apt_median = pooled_apt[len(pooled_apt) // 2] if pooled_apt else None
    if pooled_apt:
        log.info(
            f"iter {iter_idx}: actions/side-turn mean={apt_mean:.1f} "
            f"median={apt_median} max={pooled_apt[-1]} "
            f"(n={len(pooled_apt)} side-turns)")

    train_stats = None
    if train_at_end:
        # One gradient step over all queued trajectories.
        train_t0 = time.perf_counter()
        train_stats = policy.train_step()
        train_dt = time.perf_counter() - train_t0
        # getattr-guarded: non-trainable/stub policies (openers, dummy)
        # may return a minimal stats object without the aux field.
        _aux = getattr(train_stats, "aux_loss", 0.0)
        aux_str = f" aux={_aux:.4f}" if _aux else ""
        _ml = getattr(train_stats, "moves_left_loss", 0.0)
        aux_str += f" moves_left={_ml:.4f}" if _ml else ""
        _fce = getattr(train_stats, "fresh_value_ce", float("nan"))
        if _fce == _fce:                       # not NaN
            aux_str += f" fresh_value_ce={_fce:.4f}"
            _fent = getattr(train_stats, "fresh_pred_entropy",
                            float("nan"))
            _ffloor = getattr(train_stats, "fresh_ce_floor",
                              float("nan"))
            if _fent == _fent:
                aux_str += f" fresh_pred_entropy={_fent:.4f}"
            if _ffloor == _ffloor:
                aux_str += f" fresh_ce_floor={_ffloor:.4f}"
        log.info(
            f"iter {iter_idx}: train_step in {train_dt:.1f}s "
            f"trajectories={train_stats.n_trajectories} transitions={train_stats.n_transitions} "
            f"loss={train_stats.total_loss:.4f} policy={train_stats.policy_loss:.4f} "
            f"value={train_stats.value_loss:.4f}{aux_str} entropy={train_stats.entropy:.4f} "
            f"mean_return={train_stats.mean_return:+.3f} grad_norm={train_stats.grad_norm:.3f}"
        )

    # Held-out value CE with the post-update net (--holdout-size;
    # MCTSPolicy only -- getattr-guarded so every other policy type is
    # untouched). This is the generalization probe: the train value
    # loss above is measured on replay samples the net already fit.
    holdout_loss = holdout_n = None
    holdout_fn = getattr(policy, "holdout_metrics", None)
    if train_at_end and holdout_fn is not None:
        hm = holdout_fn()
        if hm is not None:
            holdout_loss, holdout_n = hm
            log.info(
                f"iter {iter_idx}: holdout value CE={holdout_loss:.4f} "
                f"on {holdout_n} held-out states (never trained on)")

    # Optional snapshot sink: the main loop passes a callable that
    # appends a CSV row to disk. Tests don't pass one, so this is
    # a no-op there. Keeping the return type a plain List[GameOutcome]
    # avoids touching every test that asserts on it.
    if snapshot_sink is not None:
        # Econ aggregates across all games this iter.
        mean_end_gold_s1 = (
            sum(o.side1_end_gold for o in outcomes) / len(outcomes)
            if outcomes else 0.0)
        mean_end_gold_s2 = (
            sum(o.side2_end_gold for o in outcomes) / len(outcomes)
            if outcomes else 0.0)
        n_recruits_s1 = sum(o.n_recruits_s1 for o in outcomes)
        n_recruits_s2 = sum(o.n_recruits_s2 for o in outcomes)
        n_recruit_attempts_s1 = sum(o.n_recruit_attempts_s1 for o in outcomes)
        n_recruit_attempts_s2 = sum(o.n_recruit_attempts_s2 for o in outcomes)
        # Per-component reward sums (read+snapshot, then leave on
        # reward_fn for any other consumer; the next iter resets).
        reward_components = dict(
            getattr(reward_fn, "_component_acc", None) or {})
        log.info(
            f"iter {iter_idx}: econ "
            f"mean_end_gold s1={mean_end_gold_s1:.1f} s2={mean_end_gold_s2:.1f}; "
            f"recruits s1={n_recruits_s1}(of {n_recruit_attempts_s1}) "
            f"s2={n_recruits_s2}(of {n_recruit_attempts_s2})"
        )
        if reward_components:
            comp_str = ", ".join(
                f"{k}={v:+.3f}" for k, v in sorted(
                    reward_components.items(), key=lambda kv: -abs(kv[1])))
            log.info(f"iter {iter_idx}: reward components -- {comp_str}")
        snapshot_sink({
            "iter":                iter_idx,
            "n_games":             len(outcomes),
            "rollout_seconds":     rollout_dt,
            "n_actions":           n_actions,
            "mean_turns":          mean_turns,
            "s1_wins":             s1_wins,
            "s2_wins":             s2_wins,
            "draws":               draws,
            "action_recruit_pct":  action_pcts["recruit"],
            "action_move_pct":     action_pcts["move"],
            "action_attack_pct":   action_pcts["attack"],
            "action_end_turn_pct": action_pcts["end_turn"],
            "action_other_pct":    action_pcts["other"],
            "mean_units_end_s1":   mean_u1,
            "mean_units_end_s2":   mean_u2,
            "closest_approach_s1": mean_ca1,
            "closest_approach_s2": mean_ca2,
            "train_stats":         train_stats,
            # Per-component reward + econ diagnostics.
            "reward_components":   reward_components,
            "mean_end_gold_s1":    mean_end_gold_s1,
            "mean_end_gold_s2":    mean_end_gold_s2,
            "n_recruits_s1":       n_recruits_s1,
            "n_recruits_s2":       n_recruits_s2,
            "n_recruit_attempts_s1": n_recruit_attempts_s1,
            "n_recruit_attempts_s2": n_recruit_attempts_s2,
            "holdout_value_loss":  holdout_loss,
            "holdout_n":           holdout_n,
            "fresh_value_ce":      (getattr(train_stats, "fresh_value_ce",
                                            None) if train_stats else None),
            "fresh_pred_entropy":  (getattr(train_stats,
                                            "fresh_pred_entropy",
                                            None) if train_stats else None),
            "fresh_ce_floor":      (getattr(train_stats, "fresh_ce_floor",
                                            None) if train_stats else None),
            "ladder_s1_wins":      ladder_s1,
            "ladder_s2_wins":      ladder_s2,
            "other_s1_wins":       other_s1,
            "other_s2_wins":       other_s2,
            # Cumulative training-unit progress, so cost-per-unit is
            # computable from the CSV alone (2026-07-07 reporting
            # rule: runway in training units, not wall-clock).
            "decision_step":       getattr(
                getattr(policy, "_base", policy), "_decision_step",
                None),
            # Trainer-process GPU memory (MB). The undiagnosed creep
            # (2026-07-10: OOM'd a 16GB card in ~3h) becomes a curve:
            # linear slope = leak-like, staircase = allocator
            # high-water ratchet on variable-length batches.
            "gpu_mem_alloc_mb":    _gpu_mem_mb()[0],
            "gpu_mem_reserved_mb": _gpu_mem_mb()[1],
            "ladder_games":        ladder_n,
            "ladder_decisive":     ladder_dec,
            "other_games":         other_n,
            "other_decisive":      other_dec,
            "actions_per_turn_mean":   apt_mean,
            "actions_per_turn_median": apt_median,
        })
    return outcomes


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

class _TrainerHistoryCSV:
    """Lazy CSV writer for per-iter training stats. Writes the header
    on the first row so the file is self-describing.

    Columns are stable across iters (every per-game / per-iter
    diagnostic the operator might want to plot over time). When a
    field is None (e.g. closest-approach with no living units to
    measure) we emit an empty cell so pandas reads it as NaN.

    Atomic-ish: each row is one append, line-buffered. A crash
    mid-row would truncate to a partial line that pandas / awk
    would skip cleanly.
    """

    # Stable column order. Adding new columns at the end is
    # backwards-compatible (older pandas reads will just see fewer
    # columns); reordering or removing columns is a breaking change.
    # Stable per-component reward column names. Match the keys
    # WeightedReward.__call__ writes into its `_component_acc`.
    # Aggregated across all games in the iter and across both sides.
    # Empty cells (= 0) mean that component didn't fire this iter.
    _REWARD_COMPONENT_KEYS = (
        "gold_killed", "village_delta", "damage_dealt",
        "unit_recruited_cost", "per_turn_penalty",
        "leader_move_penalty", "invalid_action",
        "min_enemy_distance", "approach_mp", "unused_mp",
        "fog_reveal", "attack_attempt", "unit_type_bonus",
        "turn_cond_bonus", "terminal",
    )

    # Per-side mean gold / recruits at end of game, for diagnosing
    # recruit-underuse. mean_end_gold_s1 = "side 1 ended its game
    # with X gold left on average" -- if high, the policy is not
    # spending. n_recruits_s1 = "total successful recruits by side 1
    # across all games this iter."
    _ECON_KEYS = (
        "mean_end_gold_s1", "mean_end_gold_s2",
        "n_recruits_s1", "n_recruits_s2",
        "n_recruit_attempts_s1", "n_recruit_attempts_s2",
    )

    COLUMNS = [
        "iter", "timestamp", "n_games", "rollout_seconds",
        "n_actions", "mean_turns",
        "s1_wins", "s2_wins", "draws",
        "action_recruit_pct", "action_move_pct",
        "action_attack_pct", "action_end_turn_pct",
        "action_other_pct",
        "mean_units_end_s1", "mean_units_end_s2",
        "closest_approach_s1", "closest_approach_s2",
        "train_n_trajectories", "train_n_transitions",
        "train_loss", "train_policy_loss", "train_value_loss",
        "train_entropy", "train_mean_return", "train_grad_norm",
    ] + [f"r_{k}" for k in _REWARD_COMPONENT_KEYS] + list(_ECON_KEYS) + [
        # Held-out generalization probe (--holdout-size; MCTS only).
        # Appended LAST so rows appended to a pre-existing CSV stay
        # column-compatible with its older header.
        "holdout_value_loss", "holdout_n",
        # Pre-update value CE on this iter's fresh games (2026-07-07;
        # distribution-matched generalization, no training data lost)
        # + prediction entropy (overconfidence curve) + the state-
        # blind marginal floor (outcome-mix predictability cap).
        "fresh_value_ce", "fresh_pred_entropy", "fresh_ce_floor",
        # Per-map-class decisive split (2026-07-03; aggregate decisive
        # over a mixed curriculum is misleading) + per-class SIDE
        # split (2026-07-07; asymmetries were anecdotes before).
        "ladder_games", "ladder_decisive",
        "other_games", "other_decisive",
        "ladder_s1_wins", "ladder_s2_wins",
        "other_s1_wins", "other_s2_wins",
        # Actions-per-side-turn distribution (MCTS depth calibration).
        "actions_per_turn_mean", "actions_per_turn_median",
        # Cumulative decision counter (training-unit progress; makes
        # cost-per-unit computable from the CSV alone).
        "decision_step",
        # Trainer-process CUDA memory (creep tracking, 2026-07-10).
        "gpu_mem_alloc_mb", "gpu_mem_reserved_mb",
    ]

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Appending rows with MORE columns than the file's existing
        # header silently orphans the extra cells (DictReader maps
        # them to None) -- exactly what hid the holdout column on the
        # 2026-07-06 Vast run, whose clone shipped an old-header CSV.
        # On mismatch, rotate the old file aside and start fresh.
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("r", encoding="utf-8", newline="") as f:
                existing = f.readline().rstrip("\r\n").split(",")
            if existing != self.COLUMNS:
                rotated = self.path.with_suffix(
                    self.path.suffix + ".oldschema")
                self.path.replace(rotated)
                logging.getLogger("sim_self_play").warning(
                    f"trainer-history CSV header mismatch; rotated "
                    f"old file to {rotated.name}")
        # Open in append mode; write header only if file is empty
        # (or doesn't exist). Line buffering (buffering=1) so each
        # row is flushed; cluster jobs that get walltime-killed
        # leave the partial CSV recoverable.
        write_header = (not self.path.exists()
                        or self.path.stat().st_size == 0)
        self._f = self.path.open("a", encoding="utf-8",
                                 newline="", buffering=1)
        import csv as _csv
        self._writer = _csv.DictWriter(self._f, fieldnames=self.COLUMNS,
                                       extrasaction="ignore")
        if write_header:
            self._writer.writeheader()

    def append(self, snapshot: Dict) -> None:
        import datetime as _dt
        row = dict(snapshot)
        row["timestamp"] = _dt.datetime.now().isoformat(timespec="seconds")
        ts = snapshot.get("train_stats")
        if ts is not None:
            row["train_n_trajectories"] = ts.n_trajectories
            row["train_n_transitions"]  = ts.n_transitions
            row["train_loss"]           = ts.total_loss
            row["train_policy_loss"]    = ts.policy_loss
            row["train_value_loss"]     = ts.value_loss
            row["train_entropy"]        = ts.entropy
            row["train_mean_return"]    = ts.mean_return
            row["train_grad_norm"]      = ts.grad_norm
        # Per-component reward sums (from WeightedReward._component_acc).
        acc = snapshot.get("reward_components") or {}
        for k in self._REWARD_COMPONENT_KEYS:
            v = acc.get(k)
            if v is not None:
                row[f"r_{k}"] = v
        # Econ diagnostics for recruit-underuse investigation.
        for k in self._ECON_KEYS:
            v = snapshot.get(k)
            if v is not None:
                row[k] = v
        # `extrasaction="ignore"` drops the `train_stats` object
        # itself; csv would otherwise stringify it as the repr.
        self._writer.writerow(row)

    def close(self) -> None:
        try:
            self._f.close()
        except OSError:
            pass


def _default_history_csv() -> Path:
    """Default path for the per-iter CSV. SLURM jobs get a unique
    file per job so chain links don't overwrite each other; local
    runs share one rolling `trainer_history_local.csv` so iterating
    locally accumulates."""
    import os as _os
    jobid = _os.environ.get("SLURM_JOB_ID")
    name = (f"trainer_history_{jobid}.csv" if jobid
            else "trainer_history_local.csv")
    return Path("training/logs") / name


def _parse_time_budget(spec: Optional[str]) -> Optional[int]:
    """Parse a wall-time budget string into seconds. Accepts:

      - None / empty -> None (no budget)
      - raw integer seconds:        "13800"
      - MM:SS:                      "50:00"      (50 minutes)
      - HH:MM:SS:                   "03:50:00"   (3h 50m)

    Returns the budget in integer seconds, or None when the input is
    None / empty. Raises ValueError on a malformed spec so the
    caller can fail loudly rather than silently disabling the
    budget.
    """
    if not spec:
        return None
    s = spec.strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 2:
            mm, ss = parts
            return int(mm) * 60 + int(ss)
        if len(parts) == 3:
            hh, mm, ss = parts
            return int(hh) * 3600 + int(mm) * 60 + int(ss)
        raise ValueError(
            f"--time-budget {spec!r}: expected 'HH:MM:SS', 'MM:SS', "
            f"or integer seconds"
        )
    return int(s)


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay-pool", type=Path,
                    default=Path("replays_dataset"),
                    help="Directory of .json.gz replays to sample initial states from.")
    ap.add_argument("--checkpoint-in", type=Path, default=None,
                    help="Optional starting checkpoint. Default: random init.")
    ap.add_argument("--checkpoint-out", type=Path,
                    default=Path("training/checkpoints/sim_selfplay.pt"),
                    help="Where to write checkpoints.")
    ap.add_argument("--reset-decision-step", action="store_true",
                    help="After loading --checkpoint-in, reset the "
                         "training-progress counter (decision_step) to 0. "
                         "Use when warm-starting a checkpoint as WEIGHTS "
                         "ONLY for a fresh training campaign: the combat-"
                         "oracle anneal (combat_alphas_at) then runs from "
                         "full strength over ~COMBAT_ANNEAL_HORIZON "
                         "decisions instead of inheriting a counter that "
                         "may already be past the anneal floor. Do NOT use "
                         "when resuming an in-progress run -- it would "
                         "restart the anneal mid-training.")
    ap.add_argument("--iterations", type=int, default=10,
                    help="Hard ceiling on train_step iterations. When "
                         "`--time-budget` is set, this is mostly a "
                         "safety cap -- the time budget will normally "
                         "exit first.")
    ap.add_argument("--trainer-history-csv", type=Path, default=None,
                    help="Path to append one CSV row per iter with "
                         "rollout + train stats. Default: "
                         "training/logs/trainer_history_<SLURM_JOB_ID>.csv "
                         "on cluster, training/logs/trainer_history_"
                         "local.csv otherwise. Pass an empty string "
                         "(--trainer-history-csv='') to disable.")
    ap.add_argument("--time-budget", type=str, default=None,
                    help="Stop after this much elapsed wall time, "
                         "saving a final checkpoint before exit. "
                         "Accepts 'HH:MM:SS', 'MM:SS', or a raw "
                         "integer of seconds (e.g. '03:50:00' = "
                         "13800 = 3h50m). On cluster jobs this is "
                         "the right way to bound runtime -- pair with "
                         "a large --iterations ceiling so the time "
                         "budget is the practical exit. Leave None "
                         "(default) to run until --iterations runs "
                         "out or the operator kills the process.")
    ap.add_argument("--games-per-iter", type=int, default=4,
                    help="Self-play games rolled out per train_step.")
    ap.add_argument("--max-turns", type=int, default=200,
                    help="Per-game turn cap. Default 200 -- effectively "
                         "no limit for normal PvP (most games end in "
                         "20-40 turns by leader-kill). The sim's "
                         "max_actions_per_side is the real safety net.")
    ap.add_argument("--save-every", type=int, default=10,
                    help="Save checkpoint every N iterations.")
    ap.add_argument("--holdout-size", type=int, default=0,
                    help="MCTS only: divert whole games into a frozen "
                         "held-out set until it reaches N experiences, "
                         "then log the net's value CE on it each iter "
                         "(generalization probe -- the train value "
                         "loss is measured on replay samples the net "
                         "already fit). 0 = off. Not persisted across "
                         "resumes: a resumed run re-collects, "
                         "restarting the curve's baseline.")
    ap.add_argument("--holdout-per-game-cap", type=int, default=64,
                    help="Max states RANDOMLY SAMPLED into the holdout "
                         "from each diverted game (default 64), so a "
                         "512-state holdout spans ~8 games instead of "
                         "~2 whole ones (2026-07-07: a 2-game holdout "
                         "measured those games' idiosyncrasies, not "
                         "generalization). Diverted games' remaining "
                         "states are discarded, not trained on.")
    ap.add_argument("--value-label-smoothing", type=float, default=0.0,
                    help="Mix this much uniform mass into the C51 "
                         "value TRAIN target (eval CE stays "
                         "unsmoothed). Counters extreme-atom collapse "
                         "under many replay updates on hard terminal "
                         "targets (2026-07-07 diagnosis: Z entropy "
                         "1.86->1.13 in 46 iters while holdout CE "
                         "diverged). Try 0.02. 0 = off (default).")
    ap.add_argument("--abort-decisive-rate", type=float, default=None,
                    help="Abort tripwire: once the trailing "
                         "--abort-window iterations are full, stop if "
                         "the fraction of decisive (non-draw) games "
                         "falls below this value. Saves a final "
                         "checkpoint and exits with code 4 so a "
                         "wrapper can distinguish 'tripwire' from "
                         "'done'. Guards paid GPU runs against the "
                         "known all-draws failure shape (see "
                         "tier_a_runbook.md). Off by default.")
    ap.add_argument("--abort-window", type=int, default=20,
                    help="Trailing iteration count for "
                         "--abort-decisive-rate (default 20). The "
                         "tripwire only arms once this many "
                         "iterations have completed, so it doubles "
                         "as the burn-in period.")
    ap.add_argument("--abort-holdout-stall", type=int, default=None,
                    help="Memorization tripwire (needs --holdout-size): "
                         "stop if the held-out value CE has not made a "
                         "new best (by --abort-holdout-min-delta) for "
                         "this many consecutive iterations. Saves a "
                         "final checkpoint and exits with code 5. The "
                         "2026-07-02 Kaggle data motivates it: train "
                         "value loss fell 3.8->1.15 while holdout CE "
                         "sat flat at ~3.1 -- pure buffer memorization. "
                         "Pick a GENEROUS window (hours of compute): "
                         "the frozen holdout goes stale as the policy "
                         "improves, so short windows false-trip on "
                         "long runs.")
    ap.add_argument("--abort-holdout-min-delta", type=float,
                    default=0.01,
                    help="Improvement below this doesn't count as a "
                         "new holdout best (default 0.01 CE nats; "
                         "guards against noise resetting the stall "
                         "counter).")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for replay sampling.")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING"])
    ap.add_argument("--no-map-settings", dest="use_map_settings",
                    action="store_false",
                    help="Do NOT override the source replay's economy "
                         "settings with PvP defaults. Use this when "
                         "you want the sim to track the source's "
                         "actual settings (e.g. for replay-recon "
                         "smoke tests). Default: ON -- self-play uses "
                         "standard 2p ladder defaults regardless of "
                         "source-replay quirks.")
    ap.set_defaults(use_map_settings=True)
    ap.add_argument("--starting-gold", type=int, default=100,
                    help="PvP defaults: starting gold per side.")
    ap.add_argument("--village-gold", type=int, default=2,
                    help="PvP defaults: gold per village per turn.")
    ap.add_argument("--village-support", type=int, default=1,
                    help="PvP defaults: upkeep absorbed per village.")
    ap.add_argument("--value-coef", type=float, default=None,
                    help="Override the trainer's value-loss weight "
                         "(TrainerConfig default 0.5). Higher weights "
                         "value learning harder -- relevant because "
                         "the value head is the diagnosed training "
                         "bottleneck (2026-06-15). Default None = leave "
                         "at 0.5.")
    ap.add_argument("--exp-modifier", type=int, default=70,
                    help="PvP defaults: experience required modifier (%%).")

    # Reward + opener customization. Both default to "off" (existing
    # WeightedReward defaults / no opener) so omitting these flags
    # reproduces the prior behavior bit-for-bit.
    ap.add_argument("--reward-config", type=Path, default=None,
                    help="Path to a JSON or YAML reward-shaping config "
                         "(see rewards.load_reward_config). Lets you "
                         "tune weights, add per-unit-type recruit "
                         "bonuses, and define turn-conditional "
                         "bonuses without code edits. Default: use "
                         "WeightedReward() defaults.")
    ap.add_argument("--opener-spec", type=str, default=None,
                    help=f"Name of a registered opener to wrap the "
                         f"policy with (see tools/openers.py). "
                         f"Available: {', '.join(openers_mod.available()) or '(none)'}. "
                         f"Default: no opener; the model controls "
                         f"every decision from turn 1.")
    ap.add_argument("--torch-threads", type=int, default=None,
                    help="Cap torch's intra-op CPU thread pool "
                         "(torch.set_num_threads). CPU-ONLY tuning: by "
                         "default this is applied only on a CPU device "
                         "(to ~4, or $WAI_TORCH_THREADS) -- on the tiny "
                         "per-leaf states the all-cores pool loses to "
                         "oversubscription (measured ~1.3-2.3x on CPU). "
                         "On a CUDA/DML device it is NOT capped by "
                         "default (the CPU-side MCTS/encoding wants all "
                         "cores while the GPU does forwards). Pass an "
                         "explicit value to force a cap on ANY device; "
                         "pass 0 to force torch's default. Only affects "
                         "float reduction order (~1e-7); combat RNG / "
                         "state_key / mask are integer + untouched.")
    ap.add_argument("--train-batch-size", type=int, default=None,
                    help="forward_batch chunk size for the trainer "
                         "(TrainerConfig.train_batch_size). THE key GPU "
                         "knob: default None is device-aware -- 1 on "
                         "CPU (batching these shapes doesn't help), 2 on "
                         "DML (AMD-driver-stable), 128 on CUDA so the "
                         "replay minibatch forwards as ONE batched call "
                         "instead of 128 batch-1 calls (near-worst GPU "
                         "utilization otherwise). Raise on a large GPU; "
                         "lower if you hit OOM.")
    # --- fresh-init architecture (model scaling, plan §3.2) ----------
    # Default None = "not specified" -> fall back to the checkpoint's
    # saved arch (warm-start) or TransformerPolicy's ctor default
    # (fresh). Provide these to scale the net up for a campaign; the
    # 0.47M weights WON'T load into a wider/deeper net, so passing a
    # size that differs from --checkpoint-in starts FRESH at the
    # requested size (logged). See docs/superhuman_training_plan.md
    # §3.2 for param targets (Tier-a ~3-10M = 384/6/8/1536, etc.).
    ap.add_argument("--d-model", type=int, default=None,
                    help="Transformer width for FRESH init (model "
                         "scaling). Default None -> checkpoint arch if "
                         "warm-starting, else TransformerPolicy default. "
                         "Differing from --checkpoint-in starts fresh.")
    ap.add_argument("--num-layers", type=int, default=None,
                    help="Transformer depth for fresh init (see "
                         "--d-model).")
    ap.add_argument("--num-heads", type=int, default=None,
                    help="Attention heads for fresh init (see "
                         "--d-model). Must divide --d-model.")
    ap.add_argument("--d-ff", type=int, default=None,
                    help="Feed-forward hidden dim for fresh init (see "
                         "--d-model).")
    ap.add_argument("--device", default=None,
                    help="Torch device for the policy. Accepts "
                         "'cuda' (cluster), 'cpu', 'dml' (local "
                         "AMD/Intel GPU via DirectML -- ~4x CPU "
                         "speed on the RX 6600 since the scatter "
                         "backward fix), or 'auto' (DML > CUDA > "
                         "CPU). Default None lets TransformerPolicy "
                         "pick (CPU).")
    ap.add_argument("--workers", type=int, default=0,
                    help="Rollout worker threads per iteration. 0 "
                         "(default) = serial on the main thread. "
                         ">= 1 = spawn N workers feeding trajectories "
                         "to the policy concurrently. Safe via the "
                         "policy's snapshot+lock design (see "
                         "transformer_policy.TransformerPolicy._lock). "
                         "On CPU, ~30%% speedup at workers=4 (GIL "
                         "limits gains); on GPU, larger speedup as "
                         "forwards dispatch in parallel.")
    ap.add_argument("--actor-pool", type=int, default=0,
                    help="MCTS only. >0 = run self-play in N "
                         "WEIGHTLESS actor PROCESSES feeding a central "
                         "batched-inference server (this main process "
                         "owns the GPU and batches forwards across "
                         "actors -- SEED-RL pattern, tools/actor_pool). "
                         "Escapes the GIL that caps --workers threads "
                         "and keeps a GPU fed on a CPU-rich host. "
                         "Mutually exclusive with --workers. Training "
                         "is no longer bit-deterministic (dynamic "
                         "cross-actor batching).")
    ap.add_argument("--actor-max-batch", type=int, default=0,
                    help="Max inference batch the actor-pool server "
                         "fuses per GPU forward. 0 (default) = "
                         "max(8, --actor-pool).")
    ap.add_argument("--forced-faction", default=None,
                    help="If set, every game has at least one side "
                         "playing this faction. Pass 'none' to "
                         "explicitly disable the module default "
                         "(currently 'Knalgan Alliance'). Pass any "
                         "default-era faction name (e.g. 'Drakes') "
                         "to lock that faction instead.")
    ap.add_argument("--mini-maps", action="store_true",
                    help="Restrict scenario sampling to the smallest "
                         "5 Ladder maps (Sablestone Delta, Weldyn "
                         "Channel, Den of Onis, Swamp of Dread, "
                         "Hamlets). Cells per map: 690-870 vs the "
                         "full pool's 690-2352. Used for the "
                         "engagement-curriculum phase: leaders "
                         "start ~12-15 hexes apart, so the policy "
                         "can discover engagement before the "
                         "long-march cost dominates. See "
                         "scenario_pool.MINI_MAP_SCENARIO_IDS.")
    ap.add_argument("--mini-ratio", type=float, default=0.0,
                    help="Fraction of games per iter that sample from "
                         "the mini-maps pool (rest go to ladder). "
                         "Range [0, 1]. Used to MIX mini and ladder "
                         "in one run -- e.g. 0.3 = ~30%% mini for "
                         "engagement gradient, 70%% ladder for the "
                         "production distribution. Ignored when "
                         "--mini-maps is also set (already 100%% "
                         "mini). Default 0.0 = pure ladder.")
    ap.add_argument("--drill-ratio", type=float, default=0.0,
                    help="Fraction of games per iter that sample from "
                         "the capability-drill pool (drill_duel / "
                         "drill_village_rush / drill_chokepoint -- "
                         "see scenario_pool.DRILL_SCENARIO_IDS). "
                         "Combines with --mini-ratio: one roll "
                         "splits ladder/mini/drill, so the two "
                         "ratios must sum to <= 1. Default 0.0.")
    # MCTS-mode flags. Default OFF: the existing REINFORCE path runs.
    # When --mcts is set, action selection runs an AlphaZero-style
    # tree search and the trainer minimizes CE against visit-count
    # distributions instead of policy gradient. Per-step shaping
    # rewards are silently ignored in MCTS mode (AlphaZero distills
    # the terminal z onto every visited state); --reward-config is
    # still parsed so REINFORCE / MCTS configs can share JSON files.
    ap.add_argument("--mcts", action="store_true",
                    help="Use MCTS for action selection instead of "
                         "raw policy sampling. Adds N_sim model "
                         "forwards per move (much slower per game) "
                         "in exchange for better policy targets.")
    ap.add_argument("--mcts-sims", type=int, default=50,
                    help="Number of MCTS simulations per move "
                         "(--mcts only). 50 is a reasonable "
                         "starting budget; AlphaZero used 800 for "
                         "chess, but Wesnoth's branching factor and "
                         "model-forward latency push us to fewer "
                         "sims early in development.")
    ap.add_argument("--mcts-c-puct", type=float, default=1.5,
                    help="PUCT exploration constant (--mcts only).")
    ap.add_argument("--mcts-batch-size", type=int, default=None,
                    help="Batched leaf evaluation (--mcts only). Default is "
                         "device-aware: B=1 on CPU, B=16 on CUDA (a batched "
                         "leaf forward amortizes per-forward kernel-launch + "
                         "host-sync overhead — the per-leaf forward is the "
                         "dominant GPU cost of an --mcts run, so leaving it "
                         "at 1 on CUDA starves the device). Pass an explicit "
                         "value to override; profile on the GPU node to pick "
                         "B (8-32). Applies to BOTH the default Gumbel root "
                         "(each sequential-halving phase evaluates its leaves "
                         "through one forward_batch with virtual loss) and "
                         "the classic root. Falls back to serial when "
                         "--mcts-outcome-buckets is on. Composes with "
                         "--workers (cross-game batching).")
    ap.add_argument("--mcts-fpu-reduction", type=float, default=0.25,
                    help="First-play urgency: unvisited edges score "
                         "as (parent value - this) instead of 0, so "
                         "small sim budgets deepen instead of "
                         "sweeping every legal action once. "
                         "Negative value disables (legacy Q=0).")
    ap.add_argument("--mcts-temperature", type=float, default=1.0,
                    help="Root sampling temperature for the first "
                         "--mcts-temperature-decisions of each game "
                         "(AlphaZero tau). <=0 = always argmax.")
    ap.add_argument("--mcts-temperature-decisions", type=int,
                    default=30,
                    help="How many decisions per game are sampled "
                         "proportional to visits^(1/tau) before "
                         "switching to argmax-visits.")
    ap.add_argument("--mcts-classic-root", action="store_true",
                    help="Use classic AlphaZero root handling "
                         "(Dirichlet noise + visit-count temperature "
                         "+ visit-count targets) instead of the "
                         "default Gumbel root (Gumbel-Top-k "
                         "candidates + sequential halving + "
                         "completed-Q targets; Danihelka 2022).")
    ap.add_argument("--mcts-gumbel-m", type=int, default=16,
                    help="Gumbel root: number of candidate actions "
                         "sampled without replacement for "
                         "sequential halving.")
    ap.add_argument("--mcts-no-exact-outcomes", action="store_true",
                    help="Disable exact combat-outcome enumeration "
                         "at chance nodes (tools/combat_outcomes "
                         "prob-matrix DP); falls back to pure "
                         "sampled outcomes.")
    ap.add_argument("--mcts-no-tree-reuse", action="store_true",
                    help="Disable state-key-checked subtree reuse "
                         "across consecutive decisions (on by "
                         "default; reuse only fires when the live "
                         "successor state exactly matches the "
                         "searched child, so combat RNG divergence "
                         "auto-rebuilds).")
    ap.add_argument("--mcts-outcome-buckets", action="store_true",
                    help="Enable Tier-2 adaptive outcome bucketing at "
                         "chance nodes (Gumbel/serial path only): "
                         "same-event-class combat outcomes share one "
                         "network forward (copy-at-expansion), then "
                         "split adaptively when within-bucket value "
                         "heterogeneity becomes significant (PARSS "
                         "backbone + OGA significance trigger). Default "
                         "OFF; complements root batching by cutting "
                         "redundant per-outcome forwards.")
    ap.add_argument("--mcts-bucket-v-min", type=int, default=16,
                    help="Min bucket visits before a split is "
                         "considered (--mcts-outcome-buckets).")
    ap.add_argument("--mcts-bucket-z-sig", type=float, default=2.0,
                    help="Significance threshold (in SEs of the half-"
                         "mean difference) for splitting a bucket "
                         "(--mcts-outcome-buckets).")
    ap.add_argument("--mcts-playout-cap", action="store_true",
                    help="Playout-cap randomization (KataGo): only a "
                         "random fraction of self-play moves "
                         "(--mcts-playout-cap-prob) run the full sim "
                         "budget AND record a policy target; the rest "
                         "run a cheap budget (--mcts-playout-cap-fast-"
                         "sims) and record nothing. ~3-10x more games "
                         "per GPU-hour; value targets still attach to "
                         "every recorded full move.")
    ap.add_argument("--mcts-playout-cap-prob", type=float, default=0.25,
                    help="P(full-budget, recorded move) under "
                         "--mcts-playout-cap (default 0.25).")
    ap.add_argument("--mcts-playout-cap-fast-sims", type=int, default=0,
                    help="Sim budget for fast (unrecorded) moves under "
                         "--mcts-playout-cap. 0 = max(1, --mcts-sims//4).")
    ap.add_argument("--mcts-aux-score", action="store_true",
                    help="Add the auxiliary margin head (KataGo §3.5): "
                         "the model predicts the final MATERIAL margin "
                         "(tanh, denser than win/loss z), trained with "
                         "an MSE term (--mcts-aux-coef) against "
                         "draw_tiebreak.material_margin. Changes the "
                         "model arch (adds a head) -> fresh init unless "
                         "warm-starting an aux-on checkpoint. Targets "
                         "are produced only in MCTS mode.")
    ap.add_argument("--mcts-aux-coef", type=float, default=0.15,
                    help="Weight of the auxiliary margin MSE loss "
                         "(--mcts-aux-score). KataGo uses ~0.15.")
    ap.add_argument("--mcts-moves-left", action="store_true",
                    help="Add the Lc0-style moves-left head: the model "
                         "predicts the fraction of the turn budget "
                         "still to be played (sigmoid), trained with an "
                         "MSE term (--mcts-moves-left-coef) against the "
                         "game's actual remaining turns. A dense TEMPO "
                         "signal the sparse z can't provide (2026-07-04 "
                         "action-spam diagnosis). Adds a head -> "
                         "partial-load on aux-off checkpoints. The "
                         "search-side utility consumer is separate and "
                         "default-off pending calibration.")
    ap.add_argument("--mcts-moves-left-coef", type=float, default=0.1,
                    help="Weight of the moves-left MSE loss "
                         "(--mcts-moves-left).")
    ap.add_argument("--mcts-moves-left-utility", type=float, default=0.0,
                    help="Lc0-style search utility: winning lines are "
                         "nudged toward FEWER expected remaining moves "
                         "(and losing lines toward more) by this weight "
                         "in PUCT selection. 0 = off (default). Needs "
                         "the moves-left head (--mcts-moves-left) to "
                         "carry any signal; start ~0.2.")
    ap.add_argument("--spool-workers", type=int, default=0,
                    help="MCTS only: N INDEPENDENT self-play worker "
                         "processes, each playing whole games with its "
                         "own in-process GPU forwards and spooling one "
                         "pickle per game (tools/selfplay_worker.py); "
                         "the learner consumes, trains, and saves "
                         "checkpoints the workers hot-reload. The "
                         "measured replacement for --actor-pool (whose "
                         "central server capped at ~200 req/s with the "
                         "GPU idle). Size to ~min(cores, 24) on a 24GB "
                         "card (each worker holds a CUDA context).")
    ap.add_argument("--spool-dir", type=Path,
                    default=Path("training/spool"),
                    help="Directory for spooled game files "
                         "(--spool-workers).")
    ap.add_argument("--replay-buffer", action="store_true",
                    help="Enable AlphaZero-style experience replay + "
                         "multi-epoch training (MCTS mode). Default OFF "
                         "= legacy one-gradient-step-per-fresh-batch-"
                         "then-discard. Diagnosis 2026-06-15: one-pass "
                         "is severely sample-inefficient -- the value "
                         "head needs ~80-100 steps to converge but got "
                         "1 per shifting batch, so it stalled at the "
                         "~uniform floor and the policy plateaued. "
                         "Replay retains recent experiences and takes "
                         "several minibatch steps per iter.")
    ap.add_argument("--replay-updates", type=int, default=8,
                    help="Gradient steps per iteration when "
                         "--replay-buffer is on (default 8).")
    ap.add_argument("--replay-minibatch", type=int, default=128,
                    help="Experiences per gradient step under "
                         "--replay-buffer (default 128).")
    ap.add_argument("--replay-capacity", type=int, default=4000,
                    help="Max experiences retained in the replay "
                         "buffer (default 4000). Each holds a "
                         "deepcopied game state -- watch memory on "
                         "modest hardware.")
    ap.add_argument("--replay-min-size", type=int, default=512,
                    help="Warm up with legacy one-pass training until "
                         "the replay buffer holds this many "
                         "experiences (default 512).")
    ap.add_argument("--draw-tiebreak-cap", type=float, default=0.3,
                    help="MCTS draws score by material differential "
                         "(villages + gold + unit value) in "
                         "(-cap, +cap) instead of a flat z=0, both "
                         "at search turn-cap terminals and in the "
                         "trainer's z target. <=0 disables. See "
                         "configs/draw_tiebreak.json.")
    ap.add_argument("--draw-tiebreak-config", type=Path, default=None,
                    help="JSON overriding the draw-tiebreak weights "
                         "(takes precedence over --draw-tiebreak-cap).")
    args = ap.parse_args(argv[1:])

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve the device FIRST so device-specific tuning below (thread
    # cap, train batch size) can key off it. (Was previously chosen
    # later; moved up for cluster-readiness, 2026-06-15.)
    import os as _os
    import torch
    if args.device in ("auto", "dml", "directml"):
        # Route through the helper so DML's `privateuseone:N` device
        # object (not a valid torch.device string) is resolved.
        from tools.device_select import select_inference_device
        device = select_inference_device(args.device)
    else:
        device = torch.device(args.device) if args.device else None
    _is_cpu = (device is None) or str(device).startswith("cpu")

    # Optimization #1 (CPU-ONLY): cap torch's intra-op thread pool
    # before the first forward. On CPU the tiny per-leaf states
    # oversubscribe the all-cores pool; capping to ~4 measured
    # ~1.3-2.3x. On a GPU device, leave the default so CPU-side
    # MCTS/encoding can use all cores while the GPU does forwards --
    # capping there would throttle the host pipeline. An explicit
    # --torch-threads forces a cap on ANY device. See --torch-threads.
    if args.torch_threads is not None:
        _n_threads = int(args.torch_threads)           # explicit, any device
    elif _is_cpu:
        _n_threads = int(_os.environ.get("WAI_TORCH_THREADS", "4"))
    else:
        _n_threads = 0                                 # GPU: torch default
    if _n_threads > 0:
        _was = torch.get_num_threads()
        torch.set_num_threads(_n_threads)
        log.info(f"torch intra-op threads {_was} -> {_n_threads} "
                 f"(CPU-only tuning; device={device})")

    rng = random.Random(args.seed)
    # Self-play seeds are now scenarios + random factions/leaders
    # (tools.scenario_pool), not replay starting states. The
    # `--replay-pool` flag is retained for legacy callers but
    # ignored. pool_files=None makes that explicit downstream.
    pool_files = None
    cost_lookup = _recruit_cost_lookup()
    # Eagerly load factions to surface any setup issue NOW rather
    # than on the first worker thread.
    from tools.scenario_pool import (load_factions, LADDER_SCENARIO_IDS,
                                     MINI_MAP_SCENARIO_IDS)
    factions = load_factions()
    active_pool = (MINI_MAP_SCENARIO_IDS if args.mini_maps
                   else LADDER_SCENARIO_IDS)
    pool_label = "mini-maps" if args.mini_maps else "ladder"
    log.info(f"scenario pool: {len(active_pool)} {pool_label} maps "
             f"x {len(factions)} factions = "
             f"{len(active_pool) * len(factions) ** 2} "
             f"setup combinations (faction matchups with replacement)")
    if args.mini_maps:
        log.info(f"  mini-maps active: {', '.join(active_pool)}")

    # (device already resolved above, before the thread-cap.)
    # When warm-starting from a checkpoint with non-default arch
    # (e.g. supervised_epoch3.pt at d_model=128 while
    # TransformerPolicy's default is d_model=512), build the policy
    # at the SAVED arch so load_checkpoint can resume weights
    # rather than discarding everything and starting from random
    # init. The cluster job's chain logic relies on this -- losing
    # the warm-start every iteration would burn cluster time.
    arch_kwargs: Dict[str, int] = {}
    ckpt_aux_score = False
    ckpt_moves_left = False
    if args.checkpoint_in and args.checkpoint_in.exists():
        # Resolve to a LOADABLE checkpoint: prefer the primary, but if it's
        # unreadable (truncated by a kill mid-write on a preemptible node),
        # fall back to the rolling `.bak` that save_checkpoint keeps. Doing
        # this here — before load_checkpoint below — means both the arch
        # peek and the weight load use the same good file, so a spot
        # preemption costs at most the last save interval, not the whole run.
        candidates = [
            args.checkpoint_in,
            args.checkpoint_in.with_suffix(args.checkpoint_in.suffix + ".bak"),
        ]
        raw = None
        for cand in candidates:
            if not cand.exists():
                continue
            try:
                raw = torch.load(cand, map_location="cpu",
                                 weights_only=False)
                if cand != args.checkpoint_in:
                    log.warning(
                        f"primary checkpoint {args.checkpoint_in} unreadable;"
                        f" resuming from backup {cand}")
                    args.checkpoint_in = cand
                break
            except Exception as e:
                log.warning(f"checkpoint {cand} unreadable: {e!r}")
        if raw is not None:
            saved_arch = raw.get("arch", {}) or {}
            for k in ("d_model", "num_layers", "num_heads", "d_ff"):
                if k in saved_arch:
                    arch_kwargs[k] = int(saved_arch[k])
            ckpt_aux_score = bool(raw.get("aux_score", False))
            ckpt_moves_left = bool(raw.get("moves_left", False))
            if arch_kwargs:
                log.info(f"warm-start arch from checkpoint: {arch_kwargs}"
                         f"{' +aux_score' if ckpt_aux_score else ''}"
                         f"{' +moves_left' if ckpt_moves_left else ''}")
        else:
            log.warning(
                "no loadable checkpoint (primary or .bak); falling back to "
                "TransformerPolicy defaults / random init"
            )

    # --- model scaling (plan §3.2): explicit CLI arch flags WIN over
    # the checkpoint's saved arch. If they differ from a warm-start
    # checkpoint, build at the requested size and let load_checkpoint
    # below hit "arch mismatch" -> discard weights -> fresh init (the
    # plan's intended "scaling means fresh init"). Logged loudly so a
    # warm-start isn't silently dropped.
    cli_arch = {k: v for k, v in (
        ("d_model", args.d_model), ("num_layers", args.num_layers),
        ("num_heads", args.num_heads), ("d_ff", args.d_ff),
    ) if v is not None}
    if cli_arch:
        conflicts = {k: (arch_kwargs[k], cli_arch[k]) for k in cli_arch
                     if k in arch_kwargs and arch_kwargs[k] != cli_arch[k]}
        arch_kwargs.update(cli_arch)
        log.info(f"arch from CLI flags: {cli_arch}")
        if conflicts:
            log.warning(
                "CLI arch flags differ from --checkpoint-in arch "
                f"({conflicts} as checkpoint->cli); the warm-start "
                "will be DISCARDED and training starts fresh at the "
                "requested size (model scaling).")
    if "num_heads" in arch_kwargs and "d_model" in arch_kwargs:
        if arch_kwargs["d_model"] % arch_kwargs["num_heads"] != 0:
            raise SystemExit(
                f"--num-heads ({arch_kwargs['num_heads']}) must divide "
                f"--d-model ({arch_kwargs['d_model']}).")

    # Auxiliary margin head (plan §3.5): on if requested OR if warm-
    # starting an aux-on checkpoint (so the head isn't dropped on
    # resume). It's a model-arch change, so it must be set at
    # construction.
    aux_score_flag = bool(args.mcts_aux_score) or ckpt_aux_score
    moves_left_flag = bool(args.mcts_moves_left) or ckpt_moves_left
    policy = TransformerPolicy(device=device, aux_score=aux_score_flag,
                               moves_left=moves_left_flag,
                               **arch_kwargs)
    if aux_score_flag:
        policy._trainer.config.aux_coef = float(args.mcts_aux_coef)
        log.info(f"auxiliary margin head ON (aux_coef="
                 f"{args.mcts_aux_coef}; KataGo §3.5)")
    if moves_left_flag:
        policy._trainer.config.moves_left_coef = float(
            args.mcts_moves_left_coef)
        log.info(f"moves-left head ON (moves_left_coef="
                 f"{args.mcts_moves_left_coef}; Lc0-style tempo signal)")
    # train_batch_size: forward_batch chunk size. THE key GPU knob --
    # TransformerPolicy defaults to 1 (CPU) / 2 (DML) and leaves CUDA
    # at 1, so a CUDA run would forward the replay minibatch as
    # batch-1 calls. Override to a batched size on CUDA. See
    # --train-batch-size. (Read per-chunk as max(1, config.
    # train_batch_size), so a post-construction set takes effect.)
    if args.train_batch_size is not None:
        _tbs = max(1, int(args.train_batch_size))
    elif (device is not None) and str(device).startswith("cuda"):
        _tbs = 128
    else:
        _tbs = policy._trainer.config.train_batch_size  # keep 1/2 default
    policy._trainer.config.train_batch_size = _tbs
    log.info(f"train_batch_size = {_tbs} (forward_batch chunk; "
             f"device={device})")
    if args.value_coef is not None:
        # Override the value-loss weight (TrainerConfig default 0.5).
        # The trainer reads self.config.value_coef each step, so a
        # post-construction set takes effect; used by the replay
        # sweep to test whether weighting value learning harder helps
        # (the value head is the diagnosed bottleneck).
        policy._trainer.config.value_coef = float(args.value_coef)
        log.info(f"value_coef override -> {args.value_coef}")
    if args.value_label_smoothing:
        policy._trainer.config.value_label_smoothing = float(
            args.value_label_smoothing)
        log.info(f"value label smoothing -> "
                 f"{args.value_label_smoothing} (train loss only)")
    if args.checkpoint_in and args.checkpoint_in.exists():
        log.info(f"loading checkpoint {args.checkpoint_in}")
        try:
            policy.load_checkpoint(args.checkpoint_in)
        except RuntimeError as e:
            # Arch mismatch (e.g., default size bumped from 0.5M to
            # 26M without re-warmstarting). load_checkpoint raises a
            # RuntimeError with "arch mismatch" prefix; treat that as
            # "start fresh" rather than aborting the job — chain
            # links survive across model-size changes that way.
            if "arch mismatch" in str(e).lower():
                log.warning(
                    f"checkpoint {args.checkpoint_in} has incompatible "
                    f"arch ({e}); discarding and training from random "
                    f"init."
                )
            else:
                raise
    else:
        log.warning("no input checkpoint -- training from random init")

    # Weights-only warm-start: forget the loaded training-progress
    # counter so the combat-oracle anneal restarts from full strength.
    if args.reset_decision_step:
        prev = policy._decision_step
        policy._decision_step = 0
        log.info(f"--reset-decision-step: decision_step {prev} -> 0 "
                 f"(combat-oracle anneal restarts at full strength)")

    # Vocab stays dynamically growable (config-driven customization: a
    # new id appends a fresh learnable embedding row; existing ids never
    # shift -- see encoder.register_names). Once the base roster is in
    # place (a warm-started checkpoint already carries it), arm growth
    # logging so any NEW unit type appearing during self-play is a
    # visible breadcrumb rather than a silent change on a long run. Skip
    # for a fresh init, whose first iteration legitimately populates the
    # whole roster.
    if len(policy._encoder.unit_type_to_id) > 0:
        policy.watch_vocab_growth()

    # Optional MCTS wrapper: replaces raw policy sampling with an
    # AlphaZero-style tree search. Same duck-typed
    # `select_action` / `observe` / `finalize_game` / `train_step`
    # interface as the underlying TransformerPolicy, so the rollout
    # loop doesn't branch.
    if args.mcts:
        from tools.draw_tiebreak import DrawTiebreakConfig
        from tools.mcts import MCTSConfig
        from tools.mcts_policy import MCTSPolicy, ReplayConfig
        if args.draw_tiebreak_config is not None:
            tiebreak = DrawTiebreakConfig.from_json(
                args.draw_tiebreak_config)
        elif args.draw_tiebreak_cap > 0:
            tiebreak = DrawTiebreakConfig(cap=args.draw_tiebreak_cap)
        else:
            tiebreak = None
        # mcts_batch_size: device-aware default (mirrors train_batch_size).
        # The per-leaf model forward is the dominant cost of an --mcts run;
        # leaving B=1 on CUDA issues one un-batched forward per simulation
        # and starves the GPU. Auto-bump to 16 on CUDA when the flag is
        # unset; an explicit --mcts-batch-size always wins. Gated off when
        # --mcts-outcome-buckets is on (the batched path falls back to
        # serial there anyway; see MCTSConfig / mcts._run_sim_batch).
        if args.mcts_batch_size is not None:
            _mbs = max(1, int(args.mcts_batch_size))
        elif ((device is not None) and str(device).startswith("cuda")
              and not args.mcts_outcome_buckets):
            _mbs = 16
        else:
            _mbs = 1
        if ((device is not None) and str(device).startswith("cuda")
                and _mbs == 1 and not args.mcts_outcome_buckets):
            log.warning(
                "MCTS leaf batching is B=1 on CUDA: each simulation runs an "
                "un-batched leaf forward, which starves the GPU. Set "
                "--mcts-batch-size 8-32 (profile to pick B).")
        log.info(f"mcts_batch_size = {_mbs} (leaf forward_batch; "
                 f"device={device})")
        mcts_cfg = MCTSConfig(
            n_simulations=args.mcts_sims,
            moves_left_utility=args.mcts_moves_left_utility,
            c_puct=args.mcts_c_puct,
            batch_size=_mbs,
            fpu_reduction=(None if args.mcts_fpu_reduction < 0
                           else args.mcts_fpu_reduction),
            temperature=args.mcts_temperature,
            temperature_decisions=args.mcts_temperature_decisions,
            draw_tiebreak=tiebreak,
            tree_reuse=not args.mcts_no_tree_reuse,
            gumbel_root=not args.mcts_classic_root,
            gumbel_m=args.mcts_gumbel_m,
            exact_outcome_enumeration=not args.mcts_no_exact_outcomes,
            outcome_buckets=args.mcts_outcome_buckets,
            bucket_v_min=args.mcts_bucket_v_min,
            bucket_z_sig=args.mcts_bucket_z_sig,
            playout_cap_randomization=args.mcts_playout_cap,
            playout_cap_prob=args.mcts_playout_cap_prob,
            playout_cap_fast_sims=args.mcts_playout_cap_fast_sims,
        )
        if mcts_cfg.outcome_buckets and not mcts_cfg.gumbel_root:
            # Bucketing rides the serial _run_one_sim path the Gumbel
            # root uses; the classic batched loop leaves it off (v1).
            # Don't let the flag silently no-op.
            log.warning(
                "--mcts-outcome-buckets has no effect with "
                "--mcts-classic-root (bucketing is implemented only on "
                "the Gumbel/serial path in v1); disabling it.")
            mcts_cfg.outcome_buckets = False
        root_desc = (f"gumbel(m={mcts_cfg.gumbel_m})"
                     if mcts_cfg.gumbel_root else
                     f"classic(tau={mcts_cfg.temperature}"
                     f"x{mcts_cfg.temperature_decisions})")
        log.info(
            f"MCTS mode enabled: sims={mcts_cfg.n_simulations} "
            f"c_puct={mcts_cfg.c_puct} batch_size={mcts_cfg.batch_size} "
            f"fpu={mcts_cfg.fpu_reduction} root={root_desc} "
            f"tree_reuse={mcts_cfg.tree_reuse} "
            f"outcome_buckets="
            f"{'on(v_min=%d,z=%.1f)' % (mcts_cfg.bucket_v_min, mcts_cfg.bucket_z_sig) if mcts_cfg.outcome_buckets else 'off'} "
            f"playout_cap="
            f"{'on(p=%.2f,fast=%d)' % (mcts_cfg.playout_cap_prob, mcts_cfg.playout_cap_fast_sims or max(1, mcts_cfg.n_simulations // 4)) if mcts_cfg.playout_cap_randomization else 'off'} "
            f"draw_tiebreak_cap="
            f"{tiebreak.cap if tiebreak else 'off'}. "
            f"--reward-config is ignored in MCTS mode (AlphaZero "
            f"distills terminal z, not shaping rewards)."
        )
        replay_cfg = ReplayConfig(
            enabled=args.replay_buffer,
            capacity=args.replay_capacity,
            updates_per_iter=args.replay_updates,
            minibatch=args.replay_minibatch,
            min_size=args.replay_min_size,
        )
        if replay_cfg.enabled:
            log.info(
                f"replay buffer ON: capacity={replay_cfg.capacity} "
                f"updates/iter={replay_cfg.updates_per_iter} "
                f"minibatch={replay_cfg.minibatch} "
                f"warmup>={replay_cfg.min_size} (multi-epoch training; "
                f"value head gets {replay_cfg.updates_per_iter}x the "
                f"gradient steps per iter vs legacy one-pass)")
        policy = MCTSPolicy(policy, mcts_cfg, replay_config=replay_cfg,
                            holdout_size=args.holdout_size,
                            holdout_per_game_cap=args.holdout_per_game_cap)
        if args.holdout_size > 0:
            # Works on both the in-process path (finalize_game) and
            # --actor-pool (the pool drain offers each per-game
            # _R_EXPS payload to offer_holdout_game).
            log.info(
                f"holdout probe ON: first ~{args.holdout_size} "
                f"experiences (whole games) are held out of "
                f"training and scored each iter.")

    # Optional opener wrapper: scripts the first K decisions per
    # game-side, then delegates to the learned policy. Forwarding to
    # `policy.observe` / `train_step` / `save_checkpoint` is duck-typed
    # in OpenerPolicy so the trainer keeps working unchanged.
    if args.opener_spec:
        try:
            opener = openers_mod.get_opener(args.opener_spec)
        except KeyError as e:
            log.error(str(e))
            return 2
        policy = OpenerPolicy(base=policy, opener=opener)
        log.info(f"opener: {args.opener_spec!r} "
                 f"({len(opener.moves)} moves, sides={opener.sides})")

    if args.reward_config is not None:
        try:
            reward_fn = load_reward_config(args.reward_config)
        except (KeyError, ValueError, ImportError) as e:
            log.error(f"failed to load reward config: {e}")
            return 2
        log.info(
            f"reward config: {args.reward_config} "
            f"({len(reward_fn.unit_type_bonuses)} unit-type bonuses, "
            f"{len(reward_fn.turn_conditional_bonuses)} "
            f"turn-conditional bonuses)")
    else:
        reward_fn = WeightedReward()

    pvp_defaults: Optional[PvPDefaults] = None
    if args.use_map_settings:
        pvp_defaults = PvPDefaults(
            starting_gold=args.starting_gold,
            village_gold=args.village_gold,
            village_support=args.village_support,
            experience_modifier=args.exp_modifier,
        )
        log.info(f"using PvP defaults: {pvp_defaults}")
    else:
        log.info("--no-map-settings: keeping source-replay settings")

    # Translate the --forced-faction CLI string into the right value
    # for random_setup: ... = "use module default", None = disabled,
    # str = lock to that faction.
    forced_faction_arg: object = ...
    if args.forced_faction is not None:
        if args.forced_faction.lower() == "none":
            forced_faction_arg = None
        else:
            forced_faction_arg = args.forced_faction

    # Parse --time-budget once. Wall-time exit lets cluster jobs run
    # for "as much as SLURM will give me, minus a couple minutes of
    # headroom to save the checkpoint" rather than guessing how many
    # iterations fit in a walltime window.
    time_budget_s = _parse_time_budget(args.time_budget)
    t_start = time.perf_counter()
    if time_budget_s is not None:
        log.info(
            f"time-budget: {time_budget_s} s "
            f"({time_budget_s // 3600:02d}:"
            f"{(time_budget_s % 3600) // 60:02d}:"
            f"{time_budget_s % 60:02d} HH:MM:SS); will save + exit "
            f"after the first iteration that finishes past this "
            f"mark. --iterations is a ceiling only."
        )

    # Trainer history CSV. Empty string disables; None falls back
    # to the default per-job path. The writer is line-buffered so a
    # walltime-killed job leaves a recoverable file.
    history_csv: Optional[_TrainerHistoryCSV] = None
    csv_arg = args.trainer_history_csv
    if csv_arg is None:
        csv_path = _default_history_csv()
    elif str(csv_arg) == "":
        csv_path = None
    else:
        csv_path = csv_arg
    if csv_path is not None:
        try:
            history_csv = _TrainerHistoryCSV(csv_path)
            log.info(f"trainer history CSV: {csv_path}")
        except OSError as e:
            log.warning(f"couldn't open trainer history CSV "
                        f"{csv_path}: {e}; continuing without")

    # Trip if N iterations in a row produce zero games. The most
    # common cause is missing data files (terrain_db.json,
    # unit_stats.json) — every scenario gets skipped silently and
    # the trainer happily burns walltime on empty iters with
    # `loss=0.0000`. Observed 2026-05-09 in selfplay job 17834
    # which "completed" 999 iters of 0 games. Fail fast instead.
    # Graceful-cancel sentinel: fixed canonical path so writer (GUI)
    # and reader (here) agree regardless of where args.checkpoint_out
    # points. The GUI's "Stop training (save)" button writes a file
    # at this path; the loop below polls for it after each iter.
    # Living at training/checkpoints/ because that's where every
    # other training-state file lives.
    _CANCEL_SENTINEL = (Path("training") / "checkpoints"
                        / ".cancel_local")
    # Clear any leftover sentinel from a previous run that didn't
    # clean up (e.g. force-killed after detecting but before
    # removing). Without this, a fresh training would detect the
    # stale sentinel at iter 0 and exit immediately, leaving the
    # operator confused about why their run "finished" with no
    # progress.
    if _CANCEL_SENTINEL.exists():
        try:
            _CANCEL_SENTINEL.unlink()
            log.info(f"cleared stale graceful-cancel sentinel "
                     f"({_CANCEL_SENTINEL})")
        except OSError:
            pass

    # Optional multiprocess actor pool (MCTS only). Built here so all
    # scenario/device/pvp settings are resolved; started before the
    # loop and torn down via atexit (daemon actors also die with the
    # parent). Mutually exclusive with --workers / --opener-spec.
    actor_pool = None
    if args.actor_pool > 0:
        if not args.mcts:
            log.error("--actor-pool requires --mcts"); return 2
        if args.workers > 0:
            log.error("--actor-pool is mutually exclusive with --workers")
            return 2
        if args.opener_spec:
            log.error("--actor-pool does not support --opener-spec "
                      "(actors do not apply openers)"); return 2
        import atexit
        from tools.actor_pool import ActorPool
        scenario_opts = dict(
            forced_faction=forced_faction_arg,
            mini_maps=args.mini_maps,
            mini_ratio=max(0.0, min(1.0, float(args.mini_ratio))),
            drill_ratio=max(0.0, min(1.0, float(args.drill_ratio))),
        )
        actor_pool = ActorPool(
            policy, args.actor_pool, mcts_cfg,
            scenario_opts=scenario_opts, max_turns=args.max_turns,
            pvp_defaults=pvp_defaults, device=device,
            max_batch=(args.actor_max_batch or None),
            log_level=logging.getLogger().level)
        actor_pool.start()
        atexit.register(actor_pool.shutdown)
        log.info(f"actor pool: {args.actor_pool} processes "
                 f"(GIL-free; --workers thread-path disabled)")

    # Spool workers: independent per-game processes, the measured
    # replacement for the actor pool (see SpoolWorkers docstring).
    spool = None
    if args.spool_workers > 0:
        if not args.mcts:
            log.error("--spool-workers requires --mcts"); return 2
        if args.actor_pool > 0 or args.workers > 0:
            log.error("--spool-workers is mutually exclusive with "
                      "--actor-pool / --workers"); return 2
        if args.opener_spec:
            log.error("--spool-workers does not support --opener-spec")
            return 2
        import atexit
        # Workers boot from the checkpoint file: guarantee it exists
        # (a fresh run hasn't saved yet).
        if not args.checkpoint_out.exists():
            policy.save_checkpoint(args.checkpoint_out)
        spool = SpoolWorkers(args.spool_workers, args.spool_dir,
                             args.checkpoint_out, args,
                             log_level=args.log_level)
        atexit.register(spool.shutdown)
        log.info(f"spool workers: {args.spool_workers} independent "
                 f"processes -> {args.spool_dir} (in-process GPU "
                 f"forwards; no inference server)")

    # Decisive-rate abort tripwire state (--abort-decisive-rate).
    # Trailing window of (decisive_games, total_games) per iteration;
    # armed once the window is full (= burn-in of --abort-window iters).
    from collections import deque as _deque
    abort_rate = args.abort_decisive_rate
    abort_hist: "_deque[Tuple[int, int]]" = _deque(
        maxlen=max(1, args.abort_window))
    if abort_rate is not None:
        log.info(
            f"abort tripwire ON: stop if decisive-game rate over the "
            f"trailing {args.abort_window} iters drops below "
            f"{abort_rate:.0%} (armed after iter {args.abort_window}).")
    if args.holdout_size > 0 and not args.mcts:
        log.warning("--holdout-size requires --mcts; probe inactive.")
    # Holdout-stall (memorization) tripwire state. Tracks the running
    # best holdout CE; an iteration only resets the stall counter when
    # it beats the best by at least min_delta.
    holdout_stall_limit = args.abort_holdout_stall
    if holdout_stall_limit is not None and args.holdout_size <= 0:
        log.warning("--abort-holdout-stall needs --holdout-size; "
                    "tripwire inactive.")
        holdout_stall_limit = None
    holdout_best: Optional[float] = None
    holdout_stall = 0
    if holdout_stall_limit is not None:
        log.info(
            f"holdout-stall tripwire ON: stop if holdout value CE "
            f"makes no new best (min delta "
            f"{args.abort_holdout_min_delta}) for "
            f"{holdout_stall_limit} consecutive iters.")

    DEAD_ITER_LIMIT = 5
    consecutive_dead = 0
    for it in range(args.iterations):
        outcomes = run_iteration(
            policy, pool_files, reward_fn, cost_lookup,
            iter_idx=it,
            games_per_iter=args.games_per_iter,
            max_turns=args.max_turns,
            rng=rng,
            pvp_defaults=pvp_defaults,
            workers=args.workers,
            forced_faction=forced_faction_arg,
            mini_maps=args.mini_maps,
            mini_ratio=max(0.0, min(1.0, float(args.mini_ratio))),
            drill_ratio=max(0.0, min(1.0, float(args.drill_ratio))),
            snapshot_sink=(history_csv.append if history_csv else None),
            actor_pool=actor_pool,
            spool=spool,
        )
        if not outcomes:
            consecutive_dead += 1
            if consecutive_dead >= DEAD_ITER_LIMIT:
                log.error(
                    f"{DEAD_ITER_LIMIT} consecutive iterations rolled "
                    f"zero games. Likely cause: missing scrape file "
                    f"(terrain_db.json / unit_stats.json) or scenario "
                    f"setup error. Aborting; check earlier WARNING "
                    f"logs for the underlying skip reason."
                )
                return 3
        else:
            consecutive_dead = 0
        # Decisive-rate tripwire: on a paid GPU node, a policy that
        # draws essentially every game generates almost no win/loss
        # signal -- the known failure shape (the 2026-05 iter-168
        # baseline had ZERO leaderkills on full maps). Predefined
        # abort > deciding at hour two with money burning. State is
        # saved before exiting (checkpoint + line-buffered CSV), so
        # nothing is lost: diagnose, adjust, resume from the same
        # checkpoint (WITHOUT --reset-decision-step).
        if abort_rate is not None and outcomes:
            abort_hist.append(
                (sum(1 for o in outcomes if o.winner != 0),
                 len(outcomes)))
            if len(abort_hist) == abort_hist.maxlen:
                dec = sum(d for d, _ in abort_hist)
                tot = sum(t for _, t in abort_hist)
                rate = (dec / tot) if tot else 0.0
                if rate < abort_rate:
                    policy.save_checkpoint(args.checkpoint_out)
                    log.error(
                        f"ABORT TRIPWIRE: decisive-game rate "
                        f"{rate:.1%} ({dec}/{tot} over the last "
                        f"{len(abort_hist)} iters) is below the "
                        f"--abort-decisive-rate {abort_rate:.0%} "
                        f"threshold after iter {it + 1}. Final "
                        f"checkpoint saved to {args.checkpoint_out}; "
                        f"trainer-history CSV is flushed per row. "
                        f"Diagnose before re-launching: check "
                        f"closest_approach / attack%% trends in the "
                        f"CSV, consider --mini-ratio (engagement "
                        f"curriculum), a higher --draw-tiebreak-cap, "
                        f"or a longer --max-turns. Exit code 4."
                    )
                    if history_csv is not None:
                        history_csv.close()
                    return 4
        # Holdout-stall (memorization) tripwire: the holdout CE is the
        # only metric that distinguishes value learning from replay-
        # buffer fitting (measured 2026-07-02: train value loss fell
        # 3.8->1.15 while holdout CE sat flat at ~3.1). If it makes no
        # new best for the configured stretch, training is either
        # memorizing or stalled -- either way, on paid compute, stop
        # and diagnose. State is saved (checkpoint + flushed CSV).
        if holdout_stall_limit is not None:
            hl = getattr(policy, "last_holdout_loss", None)
            if hl is not None:
                if (holdout_best is None
                        or hl < holdout_best - args.abort_holdout_min_delta):
                    holdout_best = hl
                    holdout_stall = 0
                else:
                    holdout_stall += 1
                    if holdout_stall >= holdout_stall_limit:
                        policy.save_checkpoint(args.checkpoint_out)
                        log.error(
                            f"ABORT TRIPWIRE (holdout stall): holdout "
                            f"value CE has not beaten its best "
                            f"({holdout_best:.4f}) by "
                            f"{args.abort_holdout_min_delta} for "
                            f"{holdout_stall} consecutive iters "
                            f"(latest {hl:.4f}) after iter {it + 1}, "
                            f"while training continued -- the "
                            f"memorization signature. Final "
                            f"checkpoint saved to "
                            f"{args.checkpoint_out}; CSV flushed. "
                            f"Compare train_value_loss vs "
                            f"holdout_value_loss columns before "
                            f"re-launching (capacity? signal? stale "
                            f"holdout on a long run?). Exit code 5."
                        )
                        if history_csv is not None:
                            history_csv.close()
                        return 5
        # Time-budget early exit. Check AFTER the iteration finishes
        # rather than mid-iteration: a partial iteration's gradient
        # update wouldn't have happened yet, so cutting mid-iter
        # would waste the rollout work AND the loop's invariant
        # (every emitted "iter K done" message reflects a completed
        # train_step) would break. Save a checkpoint at the natural
        # save-every cadence below, then break.
        elapsed = time.perf_counter() - t_start
        time_budget_exceeded = (
            time_budget_s is not None and elapsed >= time_budget_s
        )
        # Graceful-cancel sentinel: the GUI's "Stop training (save)"
        # button writes this file. We detect it between iters,
        # save, delete the sentinel (so a re-launch doesn't
        # immediately exit), and break out of the loop. Worst-case
        # latency = one iteration of rollout. Symmetric with the
        # time-budget exit -- both want "save what you've got and
        # stop cleanly" semantics.
        cancel_requested = _CANCEL_SENTINEL.exists()
        if (it + 1) % args.save_every == 0 \
                or (it + 1) == args.iterations \
                or time_budget_exceeded \
                or cancel_requested:
            policy.save_checkpoint(args.checkpoint_out)
        if cancel_requested:
            log.info(
                f"graceful-cancel sentinel detected at "
                f"{_CANCEL_SENTINEL}; checkpoint saved at iter "
                f"{it + 1}, exiting cleanly.")
            try:
                _CANCEL_SENTINEL.unlink()
            except OSError as e:
                log.warning(f"couldn't remove sentinel "
                            f"{_CANCEL_SENTINEL}: {e}; next "
                            f"training run will exit immediately "
                            f"unless you delete it manually.")
            break
        if time_budget_exceeded:
            log.info(
                f"time-budget exhausted ({elapsed:.0f}s >= "
                f"{time_budget_s}s) after iter {it + 1}; "
                f"checkpoint saved, exiting cleanly so the next "
                f"chain link can pick up where this one left off."
            )
            break
    if history_csv is not None:
        history_csv.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
