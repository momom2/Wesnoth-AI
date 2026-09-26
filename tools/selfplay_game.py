"""One self-play game through the simulator, and its summary.

`play_one_game` drives a WesnothSim to its end with one policy playing
every side: a deepcopy of the state per decision, the recruit-bounce
retry, the per-step shaping reward and the terminal one, and the
per-game summary (`GameOutcome`). `_play_one_game_safe` builds the game
from a scenario setup or a mid-game start, plays it and records it;
`_worker_loop` is the in-process thread that plays an iteration's
games. The actor pool's actors (tools/actor_worker.py), the legacy
trainer (tools/sim_self_play.py) and the probes play their games
through here.
"""

from __future__ import annotations

import copy
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from wesnoth_ai.classes import GameState, Unit
from wesnoth_ai.paths import UNIT_STATS_PATH
from wesnoth_ai.rules.scenario_pool import classify_scenario as _classify_scenario
from wesnoth_ai.rewards import (
    OUTCOME_DRAW, OUTCOME_LOSS, OUTCOME_ONGOING, OUTCOME_TIMEOUT, OUTCOME_WIN,
    StepDelta, compute_delta, hex_distance,
)
from wesnoth_ai.transformer_policy import TransformerPolicy
from tools.wesnoth_sim import WesnothSim
from tools.game_record import note_search_outcomes, record_game


log = logging.getLogger("selfplay_game")

# Strict-sync validation exporter (tools/validation_exports). Set by
# tools/sim_self_play.py's main() from --validate-export-every; when
# set, play_one_game exports every Nth finished game per category
# (mini / ladder / ladder_fogless / midgame) as a Wesnoth-loadable
# replay for local strict-sync verification. Counters are per process.
VALIDATION_EXPORTER = None


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
    # "" (unknown / legacy producer).
    map_class: str = ""
    # Accepted actions per SIDE-TURN (one side's decision sequence
    # within one turn), pooled across the game. MCTS depth
    # calibration input: at ~A actions per side-turn, S sims explore
    # roughly S/A of one turn plan ahead (2026-07-03 user request).
    turn_action_counts: List[int] = field(default_factory=list)
    # Fog condition + village-ownership diagnostics (2026-07-11,
    # fogless-mixing experiment): `fogless` = this game ran with fog
    # of war off (ladder pool only). villages_mean_* = time-average
    # of villages owned per TURN (sampled once at each turn
    # boundary; the "is anyone actually capturing?" curve);
    # villages_end_* = final ownership count. Consumers getattr()
    # these with defaults: an outcome pickled by an older code
    # version lacks them.
    fogless: bool = False
    # Game began from a human-corpus mid-game position (2026-07-12
    # backward-curriculum starts) rather than a fresh scenario.
    midgame: bool = False
    # Full per-game engagement telemetry (tools/engagement_stats.py,
    # user spec 2026-07-12): attacks (attempted / Wesnoth-invalid /
    # sim-rejected), damage, kills by unit type + value, healing by
    # source, advancements, first contact, scouting, unused MP,
    # villages fraction, material, search diagnostics, map constants.
    # None on outcomes from pre-telemetry producers (getattr-guard
    # all consumers).
    engagement: Optional[Dict] = None
    # No-progress tracker readout (wesnoth_sim.noprogress_summary):
    # max_quiet / tail_quiet / resumed_streaks -- collected on EVERY
    # game so candidate stalemate-K values can be evaluated offline
    # before enforcement (2026-07-21).
    noprogress: Optional[Dict] = None
    # The game's ACTUAL turn cap (jittered per game since 2026-07-20;
    # logged since 2026-07-21 -- capped-game analysis needed it).
    max_turns: Optional[int] = None
    villages_mean_s1: float = 0.0
    villages_mean_s2: float = 0.0
    villages_end_s1: int = 0
    villages_end_s2: int = 0


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
    try:
        with UNIT_STATS_PATH.open(encoding="utf-8") as f:
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
    # Villages owned per side, sampled once per TURN boundary
    # (fogless-mixing observability, 2026-07-11).
    village_turn_samples: List[tuple] = []
    _last_village_turn = 0
    # Engagement telemetry (2026-07-12): per-game accumulator on the
    # REAL sim only (fork() never carries it -> MCTS pays nothing).
    eng = (sim.enable_engagement_stats()
           if hasattr(sim, "enable_engagement_stats") else None)
    # Count via TERRAIN (map-build truth): the VILLAGE *modifier* is
    # only stamped on owned villages at capture time
    # (replay_dataset._parse_hex_code vs _capture_village).
    from wesnoth_ai.classes import Terrain as _T
    map_total_villages = sum(
        1 for h in sim.gs.map.hexes if _T.VILLAGE in h.terrain_types)
    start_gold = {i + 1: int(sd.current_gold)
                  for i, sd in enumerate(sim.gs.sides[:2])}
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
        from tools.mcts import fork_guard
        _note = getattr(policy, "note_observation", None)

        def _decide():
            # ONE path for every decision, initial or bounce retry
            # (project round-3 C1: the retry skipped the GBC
            # observation note, so its recorded state had no stream
            # anchor and lost its labels). The observation is a
            # no-op unless gbc labels are on; a duplicate
            # observation diffs to zero events.
            pre = copy.deepcopy(sim.gs)
            if _note is not None:
                _note(game_label, pre)
            with fork_guard(sim):
                return pre, policy.select_action(
                    pre, game_label=game_label, sim=sim)

        pre_state, action = _decide()

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
            sim.reject_recruit_hex(tgt.x, tgt.y)
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
            pre_state, action = _decide()

        # Moves onto fog-hidden enemy hexes are NOT pre-bounced here
        # anymore (2026-07-17): the sim resolves them Wesnoth-
        # faithfully inside step() -- the unit walks the planned
        # route and stops per the engine's blocked/ambush rules
        # (tools/pathfind_sim.walk_move_path), revealing the hidden
        # unit. A real partial move, real information gained; no
        # re-decide loop needed. (The recruit bounce above stays: a
        # recruit has no partial-execution semantics to fall back
        # on.)
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

        commands_before = len(sim.command_history)
        sim.step(action)
        note_search_outcomes(sim, policy, game_label, commands_before)
        action_counts[atype] = action_counts.get(atype, 0) + 1
        # Per-side-turn action tally: how many decisions one side
        # makes within one turn. This is the MCTS depth calibration
        # input — at ~A actions per side-turn, a search of S sims
        # only looks ~S/A "turns" ahead within its own turn plan.
        # Keyed on the PRE-step turn/side (end_turn advances them).
        _tk = (pre_state.global_info.turn_number, acting_side)
        turn_action_tally[_tk] = turn_action_tally.get(_tk, 0) + 1
        _update_closest_approach(sim.gs, closest_approach)
        _tn_now = sim.gs.global_info.turn_number
        if _tn_now != _last_village_turn:
            _last_village_turn = _tn_now
            _vown = getattr(sim.gs.global_info,
                            "_village_owner", None) or {}
            village_turn_samples.append(
                (sum(1 for v in _vown.values() if v == 1),
                 sum(1 for v in _vown.values() if v == 2)))

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
    # observe(done=True), and no side the policy never played gets
    # one.
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
    policy.finalize_game(game_label, sim.winner, final_gs=sim.gs,
                         midgame=getattr(sim, "_midgame_start", False))

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
    engagement = None
    if eng is not None:
        engagement = eng.to_dict()
        engagement["map_total_villages"] = map_total_villages
        engagement["start_gold"] = start_gold
        # Average owned-villages fraction of the map total, per side.
        if map_total_villages and village_turn_samples:
            n_s = len(village_turn_samples)
            engagement["villages_frac_avg"] = {
                1: sum(v[0] for v in village_turn_samples)
                   / n_s / map_total_villages,
                2: sum(v[1] for v in village_turn_samples)
                   / n_s / map_total_villages,
            }
        else:
            engagement["villages_frac_avg"] = {1: None, 2: None}
        # End-of-game material = bank + summed unit cost.
        from tools.draw_tiebreak import _side_material
        engagement["material_end"] = {
            s: _side_material(sim.gs, s)[1] + _side_material(sim.gs, s)[2]
            for s in (1, 2)}
        if hasattr(policy, "pop_search_diag"):
            engagement["search"] = policy.pop_search_diag(game_label)
    if VALIDATION_EXPORTER is not None:
        VALIDATION_EXPORTER.maybe_export(sim, game_label=game_label)
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
        fogless=not getattr(sim.gs.global_info, "_fog", True),
        midgame=getattr(sim, "_midgame_start", False),
        engagement=engagement,
        noprogress=(sim.noprogress_summary()
                    if hasattr(sim, "noprogress_summary") else None),
        max_turns=getattr(sim, "max_turns", None),
        villages_mean_s1=(sum(v[0] for v in village_turn_samples)
                          / len(village_turn_samples)
                          if village_turn_samples else 0.0),
        villages_mean_s2=(sum(v[1] for v in village_turn_samples)
                          / len(village_turn_samples)
                          if village_turn_samples else 0.0),
        villages_end_s1=sum(
            1 for v in (getattr(sim.gs.global_info, "_village_owner",
                                None) or {}).values() if v == 1),
        villages_end_s2=sum(
            1 for v in (getattr(sim.gs.global_info, "_village_owner",
                                None) or {}).values() if v == 2),
    )

# ---------------------------------------------------------------------
# One game from a setup; the in-process worker threads
# ---------------------------------------------------------------------

def _play_one_game_safe(
    *, setup, max_turns, pvp_defaults, policy, reward_fn,
    cost_lookup, game_label, no_progress_turns: int = 0,
    seed_salt: str = "", record_extra: Optional[dict] = None,
) -> Optional[GameOutcome]:
    """Run one game end-to-end from a `ScenarioSetup` (random
    scenario + faction + leader picks). Catches exceptions, drops
    pending transitions on crash, returns None on failure.

    A finished game is written to the process's game-record sink
    (`tools.game_record.configure`), with `record_extra` beside it.

    `seed_salt` is the game's own combat-luck stream (`WesnothSim.
    _seed_salt`): without one every self-play game replays the unsalted
    `request_seed(k)` stream, so the k-th combat roll is the same in
    every game of a run -- the correlation eval games lost on
    2026-09-13. Pass something unique per game; exports carry the
    salted seeds as recorded.

    Pre-pivot this used `WesnothSim.from_replay(<replay_path>)`.
    Post-pivot (2026-04-30) it builds the GameState directly from
    scenario .cfg + map + faction data via
    `wesnoth_ai.rules.scenario_pool.build_scenario_gamestate`. No replay
    file involved. Scenario events fire in `WesnothSim.__init__`
    (CoB neutrals, Aethermaw morph, etc.).
    """
    from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate
    # Mid-game start: `setup` is ("__midgame__", gs, scenario_id,
    # cut_turn, begin_side, provenance) from sample_midgame_start
    # (see _worker_loop).
    if isinstance(setup, tuple) and setup and setup[0] == "__midgame__":
        _, gs, scen_id, _cut, begin_side, mg_prov = setup
        try:
            sim = WesnothSim(gs, scenario_id=scen_id,
                             max_turns=max_turns,
                             apply_scenario_events=False,
                             begin_side=begin_side,
                             no_progress_turns=no_progress_turns)
            sim._seed_salt = seed_salt
            sim.enable_uniform_advancement()
            sim._midgame_start = True
            sim._midgame_provenance = mg_prov
        except Exception as e:                        # noqa: BLE001
            log.warning(f"skipping midgame start ({scen_id}): {e}")
            return None
        if hasattr(policy, "reset_game"):
            policy.reset_game(game_label)
        if hasattr(reward_fn, "reset_game_state"):
            reward_fn.reset_game_state(game_label)
        try:
            outcome = play_one_game(
                sim, policy, reward_fn,
                game_label=game_label, cost_lookup=cost_lookup,
            )
        except Exception as e:                        # noqa: BLE001
            log.exception(f"midgame game {game_label} crashed: {e}")
            policy.drop_pending(game_label)
            return None
        record_game(sim, setup, game_label=game_label,
                    players={"policy": type(policy).__name__}, extra=record_extra)
        return outcome
    # Map pvp_defaults onto build_scenario_gamestate kwargs.
    # starting_gold is NOT mapped (bugfix 2026-07-21): passing the
    # PvP default overrode every scenario's own [side] gold= --
    # minis designed for 50g trained on 100, Arcanclave's 175 got
    # cut (and the since-deleted gold=0 drills silently gained
    # recruiting). Scenario settings are ground truth (user
    # ruling); None = scenario value, 100 fallback for the many
    # maps that specify none. The eval path (elo_ladder) was
    # already scenario-first -- this also closes a train/eval gap.
    # The village economy and the experience modifier are NOT mapped
    # either, for the same reason and by the same ruling (2026-09-21):
    # `PvPDefaults` carries the multiplayer defaults, and passing them
    # here overrode `village_gold=3` on five of the seven mini
    # scenarios, so every mini self-play game paid a third less
    # village income than its map specifies. None = the scenario's
    # value. `PvPDefaults` still governs the midgame-splice path
    # (`WesnothSim.from_replay`), which has no scenario cfg to read.
    sg = vg = vu = em = None
    bi = (pvp_defaults.base_income   if pvp_defaults else 2)
    try:
        gs = build_scenario_gamestate(
            setup,
            starting_gold=sg, base_income=bi,
            village_gold=vg, village_upkeep=vu,
            experience_modifier=em,
        )
        sim = WesnothSim(gs, scenario_id=setup.scenario_id,
                         max_turns=max_turns,
                         no_progress_turns=no_progress_turns)
        sim._seed_salt = seed_salt
        sim.enable_uniform_advancement()
    except Exception as e:
        log.warning(f"skipping {setup.label()}: {e}")
        return None
    if hasattr(policy, "reset_game"):
        policy.reset_game(game_label)
    if hasattr(reward_fn, "reset_game_state"):
        reward_fn.reset_game_state(game_label)
    try:
        outcome = play_one_game(
            sim, policy, reward_fn,
            game_label=game_label, cost_lookup=cost_lookup,
        )
    except Exception as e:
        log.exception(f"game {game_label} crashed: {e}")
        policy.drop_pending(game_label)
        return None
    record_game(sim, setup, game_label=game_label, build={"base_income": bi},
                players={"policy": type(policy).__name__}, extra=record_extra)
    return outcome


def _worker_loop(
    *, worker_id, policy, reward_fn, cost_lookup,
    max_turns, pvp_defaults, shared,
    forced_faction=...,
    mini_maps=False,
    mini_ratio: float = 0.0,
    fogless_ratio: float = 0.0,
    midgame_ratio: float = 0.0,
    ladder_ratio: float = 1.0,
    midgame_dataset: Optional[Path] = None,
    max_turns_min: Optional[int] = None,
    no_progress_turns: int = 0,
):
    """Per-thread rollout loop. Each worker pulls a game index from
    `shared.next_game` (atomic under the master lock), runs that game,
    appends the outcome to `shared.outcomes`. Stops when `next_game`
    would exceed target_games. A game's draws (its category, setup,
    turn cap and combat luck) come from `shared.game_seed` and its
    index only, whichever worker plays it: a per-worker generator made
    the game depend on which thread won the race for the index, so a
    seeded run did not repeat (2026-09-18)."""
    from wesnoth_ai.rules.scenario_pool import random_setup, roll_mix
    while True:
        with shared["lock"]:
            if shared["next_game"] >= shared["target_games"]:
                return
            g_idx = shared["next_game"]
            shared["next_game"] += 1
        game_rng = random.Random(shared["game_seed"] + g_idx * 1_000_003)
        cat = roll_mix(game_rng, midgame=midgame_ratio,
                       mini=mini_ratio,
                       fogless=fogless_ratio, ladder=ladder_ratio)
        setup = None
        if cat == "midgame":
            from tools.midgame_starts import sample_midgame_start
            mg = sample_midgame_start(
                game_rng, midgame_dataset or Path("replays_dataset"))
            if mg is not None:
                setup = ("__midgame__",) + mg
            else:
                cat = "ladder"  # degraded sample -> regular game
        if setup is None:
            setup = random_setup(game_rng, forced_faction=forced_faction,
                                 mini_maps=mini_maps, category=cat)
        game_label = f"iter{shared['iter_idx']}_g{g_idx}"
        outcome = _play_one_game_safe(
            setup=setup,
            max_turns=_roll_max_turns(game_rng, max_turns,
                                      max_turns_min),
            pvp_defaults=pvp_defaults, policy=policy,
            reward_fn=reward_fn, cost_lookup=cost_lookup,
            game_label=game_label, no_progress_turns=no_progress_turns,
            seed_salt=f"pool:{game_label}",
        )
        if outcome is not None:
            with shared["lock"]:
                shared["outcomes"].append(outcome)


def _roll_max_turns(rng, max_turns: int, max_turns_min=None) -> int:
    """Per-game turn cap. With a min set, uniform in [min, max]:
    anti-horizon-gaming (2026-07-20 user directive) -- a FIXED cap
    let the policy learn to bank gold until a known last turn; a
    jittered cap makes end-hoarding unreliable so material must be
    converted as it accrues. Training paths only (eval/demo keep
    fixed caps; the eval contract is unchanged)."""
    if not max_turns_min or max_turns_min >= max_turns:
        return max_turns
    return rng.randint(max_turns_min, max_turns)


def k_median_of(outcomes) -> Optional[float]:
    """Median actions-per-side-turn pooled over an iteration's games
    (the actions_per_turn_median CSV statistic; consumed by the
    --abort-k-median K-collapse tripwire)."""
    pooled = sorted(c for o in outcomes
                    for c in (o.turn_action_counts or []))
    return pooled[len(pooled) // 2] if pooled else None
