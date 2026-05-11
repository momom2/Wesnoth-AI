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
from typing import Dict, List, Optional, Tuple

# Make project root importable when run as a script.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

from classes import GameState
from tools.scenario_pool import LADDER_SCENARIO_IDS
from rewards import (
    OUTCOME_DRAW, OUTCOME_LOSS, OUTCOME_ONGOING, OUTCOME_TIMEOUT, OUTCOME_WIN,
    StepDelta, WeightedReward, compute_delta, load_reward_config,
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
    policy.finalize_game(game_label, sim.winner)

    return GameOutcome(
        game_label=game_label,
        winner=sim.winner,
        ended_by=sim.ended_by,
        turns=final_turn,
        side1_actions=sim._actions_by_side.get(1, 0),
        side2_actions=sim._actions_by_side.get(2, 0),
        side1_reward=side1_reward,
        side2_reward=side2_reward,
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
        setup = random_setup(worker_rng, forced_faction=forced_faction)
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

    if workers <= 0:
        # Serial path -- simplest, used for tests and smoke runs.
        from tools.scenario_pool import random_setup
        for g_idx in range(games_per_iter):
            setup = random_setup(rng, forced_faction=forced_faction)
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

    if train_at_end:
        # One gradient step over all queued trajectories.
        train_t0 = time.perf_counter()
        stats = policy.train_step()
        train_dt = time.perf_counter() - train_t0
        log.info(
            f"iter {iter_idx}: train_step in {train_dt:.1f}s "
            f"trajectories={stats.n_trajectories} transitions={stats.n_transitions} "
            f"loss={stats.total_loss:.4f} policy={stats.policy_loss:.4f} "
            f"value={stats.value_loss:.4f} entropy={stats.entropy:.4f} "
            f"mean_return={stats.mean_return:+.3f} grad_norm={stats.grad_norm:.3f}"
        )
    return outcomes


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

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
                    help="How many train_step iterations to run.")
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
                    help="Torch device for the policy (e.g. 'cuda', "
                         "'cuda:0', 'cpu'). Default: TransformerPolicy "
                         "picks CPU. Pass 'cuda' on the cluster.")
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
    from tools.scenario_pool import load_factions, LADDER_SCENARIO_IDS
    factions = load_factions()
    log.info(f"scenario pool: {len(LADDER_SCENARIO_IDS)} ladder maps "
             f"x {len(factions)} factions = "
             f"{len(LADDER_SCENARIO_IDS) * len(factions) ** 2} "
             f"setup combinations (faction matchups with replacement)")

    import torch
    device = torch.device(args.device) if args.device else None
    policy = TransformerPolicy(device=device)
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
        from tools.mcts import MCTSConfig
        from tools.mcts_policy import MCTSPolicy
        mcts_cfg = MCTSConfig(
            n_simulations=args.mcts_sims,
            c_puct=args.mcts_c_puct,
            batch_size=args.mcts_batch_size,
        )
        log.info(
            f"MCTS mode enabled: sims={mcts_cfg.n_simulations} "
            f"c_puct={mcts_cfg.c_puct} batch_size={mcts_cfg.batch_size}. "
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

    # Trip if N iterations in a row produce zero games. The most
    # common cause is missing data files (terrain_db.json,
    # unit_stats.json) — every scenario gets skipped silently and
    # the trainer happily burns walltime on empty iters with
    # `loss=0.0000`. Observed 2026-05-09 in selfplay job 17834
    # which "completed" 999 iters of 0 games. Fail fast instead.
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
        if (it + 1) % args.save_every == 0 or (it + 1) == args.iterations:
            policy.save_checkpoint(args.checkpoint_out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
