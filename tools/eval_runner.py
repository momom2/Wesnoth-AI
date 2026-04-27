"""Per-game eval loop for `tools/eval_vs_builtin.py`.

Thin re-implementation of game_manager.py's per-turn loop, scoped to
the eval task: launch a Wesnoth process on a specific eval scenario,
drive our side via the trained policy, watch for terminal state,
return the outcome.

Differences vs game_manager.py:
  - No reward bookkeeping, no per-step delta computation, no
    train_step. Just rollout under torch.no_grad().
  - The opposing side runs Wesnoth's default RCA AI inside Wesnoth
    itself; the Python loop only ever sees state frames from OUR
    side's turns.
  - Per-game timeout: bail with outcome="timeout" if the game runs
    past `max_actions` Python-side decisions.
  - Returns a `GameResult` rather than mutating shared trainer state.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

# Make project root + tools/ importable when run as a script.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

from action_sampler import sample_action
from classes import GameState
from constants import MAX_ACTIONS_PER_GAME, SCENARIOS_PATH
from encoder import GameStateEncoder
from model import WesnothModel
from state_converter import StateConverter
from wesnoth_interface import WesnothGame


log = logging.getLogger("eval_runner")


@dataclass
class GameResult:
    """Outcome of one eval game.

    `winner`:
       0 = draw / timeout / endless
       1 = side 1 won
       2 = side 2 won
    `our_side`: which side our policy played (1 or 2). Combine with
    `winner` to bucket as win / loss / draw for our model.
    """
    scenario_id:    str
    our_side:       int            # 1 or 2
    winner:         int            # 0 / 1 / 2
    outcome:        str            # 'win' | 'loss' | 'draw' | 'timeout' | 'errored'
    turns:          int            # final turn number
    our_actions:    int            # number of actions our policy emitted
    wall_seconds:   float
    error:          Optional[str] = None


def _outcome_from(winner: int, our_side: int, hit_timeout: bool) -> str:
    if hit_timeout:
        return "timeout"
    if winner == 0:
        return "draw"
    return "win" if winner == our_side else "loss"


async def play_one(
    scenario_id:    str,
    our_side:       int,
    encoder:        GameStateEncoder,
    model:          WesnothModel,
    converter:      StateConverter,
    label:          str   = "eval",
    max_actions:    int   = MAX_ACTIONS_PER_GAME,
    state_timeout:  float = 90.0,
    launch_lock:    Optional[asyncio.Lock] = None,
) -> GameResult:
    """Run one eval game to completion (or timeout). Returns the outcome.

    Concurrency: the function is async only because read_state /
    send_action use threading internally (via asyncio.to_thread). To
    run multiple games in parallel, gather() several `play_one` calls.
    """
    t0 = time.perf_counter()
    game = WesnothGame(
        label=label,
        scenario_path=SCENARIOS_PATH,
        scenario_id=scenario_id,
    )
    error: Optional[str] = None
    winner = 0
    turns = 0
    our_actions = 0
    hit_timeout = False
    last_gs: Optional[GameState] = None  # for terminal-state fallback

    try:
        # Launch under the shared lock when one's provided -- so that
        # this WesnothGame's pre_launch_logs snapshot includes all the
        # previous parallel games' logs, AND we lock in our own log
        # path before the next game's snapshot is taken. Without this,
        # parallel launches all snapshot the same set, then all see
        # the same N "new" .out.logs and pick the wrong one (cross-talk
        # of state frames between unrelated games -- exactly the
        # "state frame for side X but we're side Y" bug the unsafe
        # parallel eval hit).
        if launch_lock is not None:
            async with launch_lock:
                game.start_wesnoth()
                # Wait up to ~5s for Wesnoth to create its .out.log
                # file, then lock the path in so later reads don't
                # re-probe (re-probing would include OTHER games'
                # subsequently-created logs in the "new" set).
                for _ in range(50):
                    candidate = game._find_out_log()
                    if candidate is not None:
                        game._log_path = candidate
                        break
                    await asyncio.sleep(0.1)
        else:
            game.start_wesnoth()
        if not game.is_running:
            return GameResult(
                scenario_id=scenario_id, our_side=our_side, winner=0,
                outcome="errored", turns=0, our_actions=0,
                wall_seconds=time.perf_counter() - t0,
                error="wesnoth failed to start",
            )

        while True:
            if our_actions >= max_actions:
                hit_timeout = True
                break

            payload = await asyncio.to_thread(game.read_state, state_timeout)
            if not payload:
                # No state arrived within timeout. Treat as timeout
                # (game effectively stalled) and move on.
                hit_timeout = True
                break

            try:
                gs: GameState = converter.convert_payload_to_game_state(payload)
            except Exception as e:
                error = f"state parse: {e}"
                break
            last_gs = gs

            if game.game_id is None:
                game.adopt_game_id(gs.game_id)

            turns = gs.global_info.turn_number
            if gs.game_over:
                winner = gs.winner or 0
                break

            # Sanity: only OUR side should be feeding us state frames
            # (the opposing side runs default RCA inside Wesnoth and
            # never invokes our turn_stage.lua). If we somehow see the
            # wrong side, just no-op end_turn -- the protocol doesn't
            # let us play out-of-turn anyway.
            if gs.global_info.current_side != our_side:
                log.warning(
                    f"[{label}] state frame for side "
                    f"{gs.global_info.current_side} but we're side "
                    f"{our_side}; ending turn"
                )
                action = {"type": "end_turn"}
            else:
                with torch.no_grad():
                    encoded = encoder.encode(gs)
                    output = model(encoded)
                    sampled = sample_action(encoded, output, gs)
                action = sampled.action

            wire = converter.convert_action_to_json(action)
            sent = await asyncio.to_thread(game.send_action, wire)
            if not sent:
                # Wesnoth's Lua side has a 30s action-wait timeout; if
                # send fails, the side ends its turn naturally. Eat
                # the loss of one action and keep going.
                log.debug(f"[{label}] action send failed; turn will time out")
            our_actions += 1
    except Exception as e:
        error = f"game loop: {e}"
        log.exception(f"[{label}] error during eval game")
    finally:
        try:
            game.terminate()
        except Exception:
            pass

    # Terminal-state fallback. If we hit `hit_timeout` (state didn't
    # arrive within state_timeout, OR Wesnoth process exited mid-wait)
    # but the LAST observed state had only one side's leader alive,
    # the missing-state was caused by the game ending on the opponent's
    # turn -- our turn_stage never got to emit a `game_over=true` frame.
    # Use the last state's leader-alive count to infer the winner.
    # If leaders on BOTH sides are still alive in the last state, this
    # is a genuine stall -- keep `outcome="timeout"`.
    if hit_timeout and last_gs is not None and not error:
        s1_alive = any(u.is_leader and u.side == 1 for u in last_gs.map.units)
        s2_alive = any(u.is_leader and u.side == 2 for u in last_gs.map.units)
        if s1_alive and not s2_alive:
            winner = 1
            hit_timeout = False
        elif s2_alive and not s1_alive:
            winner = 2
            hit_timeout = False
        elif not s1_alive and not s2_alive:
            winner = 0   # mutual elim, treat as draw
            hit_timeout = False
        # else: both alive, genuine timeout

    return GameResult(
        scenario_id=scenario_id,
        our_side=our_side,
        winner=winner,
        outcome=("errored" if error else
                 _outcome_from(winner, our_side, hit_timeout)),
        turns=turns,
        our_actions=our_actions,
        wall_seconds=time.perf_counter() - t0,
        error=error,
    )


async def play_many(
    matchups:      List[Dict],          # list of {scenario_id, our_side}
    encoder:       GameStateEncoder,
    model:         WesnothModel,
    converter:     StateConverter,
    parallel:      int   = 4,
    max_actions:   int   = MAX_ACTIONS_PER_GAME,
    state_timeout: float = 90.0,
    progress_cb=None,                   # callable(GameResult) -> None
) -> List[GameResult]:
    """Run a batch of eval games with up to `parallel` Wesnoth processes
    in flight at once. Returns results in MATCHUP order (NOT completion
    order) so the caller can join with their input metadata.
    """
    results: List[Optional[GameResult]] = [None] * len(matchups)
    sem = asyncio.Semaphore(parallel)
    # Serializes Wesnoth launches across the pool so each game's
    # pre_launch_logs snapshot sees the previously-launched games'
    # .out.logs as "existing" and only its own as "new". Held briefly
    # -- just long enough for Wesnoth to create its log file (up to
    # ~5s in the worst case). The semaphore above still gates total
    # in-flight games at `parallel`.
    launch_lock = asyncio.Lock()
    next_label_id = 0

    async def _runner(idx: int, m: Dict) -> None:
        nonlocal next_label_id
        async with sem:
            label = f"eval{idx:03d}"
            log.info(
                f"[{label}] starting: {m['scenario_id']} "
                f"(our_side={m['our_side']})"
            )
            res = await play_one(
                scenario_id=m["scenario_id"],
                our_side=m["our_side"],
                encoder=encoder, model=model, converter=converter,
                label=label, max_actions=max_actions,
                state_timeout=state_timeout,
                launch_lock=launch_lock,
            )
            results[idx] = res
            if progress_cb is not None:
                try:
                    progress_cb(res)
                except Exception:
                    log.exception("progress_cb failed")

    await asyncio.gather(*[
        _runner(i, m) for i, m in enumerate(matchups)
    ])
    # All slots filled by gather; for typing.
    return [r for r in results if r is not None]
