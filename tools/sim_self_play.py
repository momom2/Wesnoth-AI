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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Make project root importable when run as a script.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

from classes import GameState, Unit
from tools.scenario_pool import LADDER_SCENARIO_IDS
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
            # Drop the pending Transition the policy recorded for
            # the rejected select_action so the trainer doesn't
            # re-forward on it. observe() with reward=0 keeps the
            # trajectory shape consistent: the rejected pick lands
            # in the trajectory with a neutral signal -- not great,
            # not awful. A future refactor could expose a "drop
            # just the tail" API on the policy.
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
        if atype == "recruit" and delta.units_recruited:
            n_recruits_per_side[acting_side] = (
                n_recruits_per_side.get(acting_side, 0)
                + len(delta.units_recruited))
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
                             mini_maps=mini_maps, mini_ratio=mini_ratio)
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
    snapshot_sink: Optional[Callable[[Dict], None]] = None,
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

    if workers <= 0:
        # Serial path -- simplest, used for tests and smoke runs.
        from tools.scenario_pool import random_setup
        for g_idx in range(games_per_iter):
            setup = random_setup(rng, forced_faction=forced_faction,
                                 mini_maps=mini_maps,
                                 mini_ratio=mini_ratio)
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

    train_stats = None
    if train_at_end:
        # One gradient step over all queued trajectories.
        train_t0 = time.perf_counter()
        train_stats = policy.train_step()
        train_dt = time.perf_counter() - train_t0
        log.info(
            f"iter {iter_idx}: train_step in {train_dt:.1f}s "
            f"trajectories={train_stats.n_trajectories} transitions={train_stats.n_transitions} "
            f"loss={train_stats.total_loss:.4f} policy={train_stats.policy_loss:.4f} "
            f"value={train_stats.value_loss:.4f} entropy={train_stats.entropy:.4f} "
            f"mean_return={train_stats.mean_return:+.3f} grad_norm={train_stats.grad_norm:.3f}"
        )

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
    ] + [f"r_{k}" for k in _REWARD_COMPONENT_KEYS] + list(_ECON_KEYS)

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
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
    ap.add_argument("--mcts-batch-size", type=int, default=1,
                    help="Batched leaf evaluation (--mcts only). "
                         "B=1 on CPU; B=8-32 amortizes kernel "
                         "launch on GPU but our forward already "
                         "dominates so the gain is modest.")
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

    import torch
    if args.device in ("auto", "dml", "directml"):
        # Route through the helper so DML's `privateuseone:N` device
        # object (not a valid torch.device string) is resolved.
        from tools.device_select import select_inference_device
        device = select_inference_device(args.device)
    else:
        device = torch.device(args.device) if args.device else None

    # When warm-starting from a checkpoint with non-default arch
    # (e.g. supervised_epoch3.pt at d_model=128 while
    # TransformerPolicy's default is d_model=512), build the policy
    # at the SAVED arch so load_checkpoint can resume weights
    # rather than discarding everything and starting from random
    # init. The cluster job's chain logic relies on this -- losing
    # the warm-start every iteration would burn cluster time.
    arch_kwargs: Dict[str, int] = {}
    if args.checkpoint_in and args.checkpoint_in.exists():
        try:
            raw = torch.load(args.checkpoint_in, map_location="cpu",
                             weights_only=False)
            saved_arch = raw.get("arch", {}) or {}
            for k in ("d_model", "num_layers", "num_heads", "d_ff"):
                if k in saved_arch:
                    arch_kwargs[k] = int(saved_arch[k])
            if arch_kwargs:
                log.info(f"warm-start arch from checkpoint: {arch_kwargs}")
        except Exception as e:
            log.warning(
                f"couldn't peek arch from {args.checkpoint_in}: {e!r}; "
                f"falling back to TransformerPolicy defaults"
            )

    policy = TransformerPolicy(device=device, **arch_kwargs)
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

    # Optional MCTS wrapper: replaces raw policy sampling with an
    # AlphaZero-style tree search. Same duck-typed
    # `select_action` / `observe` / `finalize_game` / `train_step`
    # interface as the underlying TransformerPolicy, so the rollout
    # loop doesn't branch.
    if args.mcts:
        from tools.draw_tiebreak import DrawTiebreakConfig
        from tools.mcts import MCTSConfig
        from tools.mcts_policy import MCTSPolicy
        if args.draw_tiebreak_config is not None:
            tiebreak = DrawTiebreakConfig.from_json(
                args.draw_tiebreak_config)
        elif args.draw_tiebreak_cap > 0:
            tiebreak = DrawTiebreakConfig(cap=args.draw_tiebreak_cap)
        else:
            tiebreak = None
        mcts_cfg = MCTSConfig(
            n_simulations=args.mcts_sims,
            c_puct=args.mcts_c_puct,
            batch_size=args.mcts_batch_size,
            fpu_reduction=(None if args.mcts_fpu_reduction < 0
                           else args.mcts_fpu_reduction),
            temperature=args.mcts_temperature,
            temperature_decisions=args.mcts_temperature_decisions,
            draw_tiebreak=tiebreak,
            tree_reuse=not args.mcts_no_tree_reuse,
            gumbel_root=not args.mcts_classic_root,
            gumbel_m=args.mcts_gumbel_m,
            exact_outcome_enumeration=not args.mcts_no_exact_outcomes,
        )
        root_desc = (f"gumbel(m={mcts_cfg.gumbel_m})"
                     if mcts_cfg.gumbel_root else
                     f"classic(tau={mcts_cfg.temperature}"
                     f"x{mcts_cfg.temperature_decisions})")
        log.info(
            f"MCTS mode enabled: sims={mcts_cfg.n_simulations} "
            f"c_puct={mcts_cfg.c_puct} batch_size={mcts_cfg.batch_size} "
            f"fpu={mcts_cfg.fpu_reduction} root={root_desc} "
            f"tree_reuse={mcts_cfg.tree_reuse} "
            f"draw_tiebreak_cap="
            f"{tiebreak.cap if tiebreak else 'off'}. "
            f"--reward-config is ignored in MCTS mode (AlphaZero "
            f"distills terminal z, not shaping rewards)."
        )
        policy = MCTSPolicy(policy, mcts_cfg)

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
            snapshot_sink=(history_csv.append if history_csv else None),
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
