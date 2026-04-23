"""Coordinates Wesnoth subprocesses + policy + per-step reward feeding.

Per game (per side's turn-local view):

    [read state_t] → [attribute reward to prev action] → [policy picks
     action_t (= internally records a pending Transition for side_t)]
     → [send action_t to Lua] → loop

At end of game we finalize: both sides' last pending transitions get a
terminal reward (win/loss/draw) with `done=True`, which flushes them
into the policy's training queue.

Reward plumbing is only active when the policy is trainable (exposes
an ``observe`` method). Scripted policies (DummyPolicy) don't care.

After a batch of games completes, if the policy is trainable and has
`train_step`, we call it. It drains the queue and runs one gradient
update. Checkpoints are saved on a schedule (CHECKPOINT_FREQUENCY) and
delegated to the policy's own save_checkpoint method.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

from classes import GameState
from constants import (
    CHECKPOINT_FREQUENCY,
    CHECKPOINTS_PATH,
    LOG_FREQUENCY,
    LOGS_PATH,
    MAX_ACTIONS_PER_GAME,
    NUM_PARALLEL_GAMES,
    REPLAYS_PATH,
    SCENARIOS_PATH,
)
from policy import Policy, get_policy
from profiling import Profiler
from rewards import (
    OUTCOME_DRAW,
    OUTCOME_LOSS,
    OUTCOME_ONGOING,
    OUTCOME_TIMEOUT,
    OUTCOME_WIN,
    StepDelta,
    WeightedReward,
    compute_delta,
)
from state_converter import StateConverter
from wesnoth_interface import WesnothGame


# Per-label state we carry between turns to compute StepDeltas. Tuple:
#   (prev_state, prev_action_type, prev_acting_side, prev_recruit_cost)
_PrevEntry = Tuple[GameState, str, int, int]


class GameManager:
    """Coordinates a batch of Wesnoth subprocesses sharing a single policy."""

    def __init__(
        self,
        num_games: int = NUM_PARALLEL_GAMES,
        policy: Optional[Policy] = None,
        reward_fn=None,
    ):
        self.num_games = num_games
        self.games: Dict[str, WesnothGame] = {}

        # Shared state converter — one mapping from unit-type-name to an
        # integer ID, used consistently across games.
        self.global_converter = StateConverter()

        # Serializes Wesnoth launches so each WesnothGame gets a clean
        # pre-launch-logs snapshot; otherwise concurrent launches all
        # see the same "new" log files and adopt the wrong one. Held
        # briefly — just long enough for Wesnoth to start writing its
        # log so the next launch's snapshot can see it as "existing".
        self._launch_lock = asyncio.Lock()

        self.logger = logging.getLogger("game_manager")
        self._setup_logging()

        self.policy: Policy = policy if policy is not None else get_policy("dummy")
        self.logger.info(f"Policy: {type(self.policy).__name__}")

        # Default reward fn is WeightedReward with its sensible defaults
        # (gold_killed_delta at 0.01, village_delta at 0.05, small
        # damage/recruit bonuses, ±1 terminal). Callers can pass any
        # RewardFn-compatible callable to override — the Phase 3 user
        # ask for "incentivize unorthodox strategies" flows through
        # construction with different weights, no other code touched.
        self._reward_fn = reward_fn if reward_fn is not None else WeightedReward()

        # Per-(game_label) state needed to build StepDeltas between
        # consecutive frames. See _PrevEntry type alias.
        self._prev: Dict[str, _PrevEntry] = {}

        # Rolling-pool bookkeeping. These track when we last fired a
        # train_step / checkpoint so the pool's event loop can fire
        # them at a schedule independent of batch boundaries.
        self._last_trained_at = 0
        self._last_checkpoint_at = 0

        # Profiling. handle_game_turn wraps its stages in
        # `with self._profiler.time("..."): ...` and the periodic
        # stats log emits self._profiler.report().
        self._profiler = Profiler()

        self.stats = {
            'games_completed': 0,
            'total_actions': 0,
            'wins_side1': 0,
            'wins_side2': 0,
            'draws': 0,
            'timeouts': 0,
            'state_conversion_errors': 0,
            'action_send_errors': 0,
        }

        for p in (CHECKPOINTS_PATH, REPLAYS_PATH, LOGS_PATH):
            p.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> None:
        log_file = LOGS_PATH / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

    # ------------------------------------------------------------------
    # Game lifecycle
    # ------------------------------------------------------------------

    async def create_game(self, label: str) -> WesnothGame:
        """Launch a Wesnoth process for this game.

        Takes self._launch_lock so the pre-launch-logs snapshot taken
        inside start_wesnoth doesn't overlap with another launch. We
        also wait briefly for the new .out.log to appear before
        releasing the lock, so the NEXT game's snapshot sees our log
        as "existing".
        """
        self.logger.info(f"Creating game {label}")
        scenario_path = SCENARIOS_PATH / "training_scenario.cfg"
        game = WesnothGame(label, scenario_path)

        async with self._launch_lock:
            game.start_wesnoth()
            # Wait up to ~5s for Wesnoth to create its .out.log file,
            # then lock in the path so later reads don't re-probe
            # (re-probing would include OTHER games' subsequently-
            # created logs in the "new" set and assign the wrong one).
            for _ in range(50):
                candidate = game._find_out_log()
                if candidate is not None:
                    game._log_path = candidate
                    self.logger.debug(
                        f"Game {label}: .out.log = {candidate.name}")
                    break
                await asyncio.sleep(0.1)
        return game

    async def handle_game_turn(self, game: WesnothGame) -> bool:
        """One state→action exchange. Returns False when the game ends."""
        label = game.label
        try:
            # read_state spins in a polling loop with time.sleep, which
            # blocks the asyncio event loop. Push it to a thread so
            # parallel games can actually run in parallel — otherwise
            # asyncio.gather over N game tasks just serializes them.
            with self._profiler.time("read_state"):
                state_payload = await asyncio.to_thread(game.read_state)
            if not state_payload:
                self.logger.warning(f"Game {label}: no state, ending game")
                return False

            try:
                with self._profiler.time("parse_state"):
                    game_state = self.global_converter.convert_payload_to_game_state(state_payload)
            except Exception as e:
                self.logger.error(
                    f"Game {label}: state parse failed: {e}", exc_info=True)
                self.stats['state_conversion_errors'] += 1
                preview = state_payload[:400] if state_payload else "<empty>"
                self.logger.debug(f"Payload preview: {preview}")
                return False

            if game.game_id is None:
                game.adopt_game_id(game_state.game_id)

            # Attribute an intermediate reward to the previous action,
            # if any. Terminal rewards are applied in _finalize_game.
            with self._profiler.time("reward_delta"):
                self._reward_previous_step(label, game_state)

            if game_state.game_over:
                winner = game_state.winner or 0
                self.logger.info(f"Game {label}: over, winner={winner}")
                self._finalize_game(label, game_state, winner, OUTCOME_WIN)
                self._record_game_result(game, winner)
                return False

            # Policy forward + action send.
            try:
                with self._profiler.time("policy_select"):
                    internal_action = self.policy.select_action(
                        game_state, game_label=label
                    )
            except Exception as e:
                self.logger.error(f"Policy error: {e}", exc_info=True)
                internal_action = {'type': 'end_turn'}

            wire_action = self.global_converter.convert_action_to_json(internal_action)

            # Stash what we need to compute this step's delta next time
            # we see a state frame.
            self._prev[label] = (
                game_state,
                internal_action.get('type', ''),
                game_state.global_info.current_side,
                _recruit_cost(internal_action, game_state),
            )

            with self._profiler.time("send_action"):
                sent = game.send_action(wire_action)
            if not sent:
                # Non-fatal: Lua's 30s action timeout will end the side's
                # turn and the game proceeds. We lose one action of
                # learning signal but not the whole game.
                self.logger.warning(f"Game {label}: action send failed (turn will time out)")
                self.stats['action_send_errors'] += 1
            else:
                self.stats['total_actions'] += 1
            return True

        except Exception as e:
            self.logger.error(f"Game {label}: turn error: {e}", exc_info=True)
            return False

    async def run_game(self, label: str) -> None:
        game: Optional[WesnothGame] = None
        try:
            game = await self.create_game(label)
            self.games[label] = game
            self.logger.info(f"Game {label}: started")

            actions = 0
            while actions < MAX_ACTIONS_PER_GAME:
                if not await self.handle_game_turn(game):
                    break
                actions += 1
                if game.check_game_over():
                    self._record_game_result(game, game.winner)
                    break

            if actions >= MAX_ACTIONS_PER_GAME:
                self.logger.warning(
                    f"Game {label}: timeout after {actions} actions")
                self.stats['timeouts'] += 1
                # Seal both sides' trajectories with OUTCOME_TIMEOUT
                # so their last pending transitions still reach the
                # training queue (no terminal bonus by default).
                prev = self._prev.get(label)
                if prev is not None:
                    self._finalize_game(
                        label, prev[0], winner=0, default_outcome=OUTCOME_TIMEOUT,
                    )
                else:
                    self._drop_trajectories(label)

        except Exception as e:
            self.logger.error(f"Game {label}: error: {e}", exc_info=True)
            self._drop_trajectories(label)
        finally:
            if label in self.games:
                self.games[label].terminate()
                del self.games[label]
            self._prev.pop(label, None)
            self.logger.info(f"Game {label}: finished")

    # ------------------------------------------------------------------
    # Reward plumbing
    # ------------------------------------------------------------------

    def _reward_previous_step(
        self, label: str, game_state: GameState,
    ) -> None:
        """If there is a pending previous action for this game, compute
        its intermediate (non-terminal) delta and hand the reward to the
        policy's observe hook (if any)."""
        prev = self._prev.get(label)
        if prev is None:
            return
        prev_state, prev_action_type, prev_side, prev_recruit_cost = prev
        delta = compute_delta(
            prev_state, game_state, prev_action_type,
            recruit_cost=prev_recruit_cost, outcome=OUTCOME_ONGOING,
        )
        reward = self._reward_fn(delta)
        self._observe_policy(label, prev_side, reward, done=False)

    def _finalize_game(
        self,
        label: str,
        game_state: GameState,
        winner: int,
        default_outcome: str = OUTCOME_WIN,  # unused — see body
    ) -> None:
        """Seal both sides' trajectories with the appropriate terminal
        outcome. `winner` is 0 for draw/timeout/endless, 1/2 otherwise.
        `default_outcome` kept for signature symmetry; outcome per side
        is always derived from `winner` here (OUTCOME_TIMEOUT if
        default_outcome == OUTCOME_TIMEOUT)."""
        for side in (1, 2):
            if default_outcome == OUTCOME_TIMEOUT:
                outcome = OUTCOME_TIMEOUT
            elif winner == 0:
                outcome = OUTCOME_DRAW
            elif winner == side:
                outcome = OUTCOME_WIN
            else:
                outcome = OUTCOME_LOSS
            terminal = StepDelta(
                side=side,
                turn=game_state.global_info.turn_number,
                action_type='terminal',
                outcome=outcome,
            )
            reward = self._reward_fn(terminal)
            self._observe_policy(label, side, reward, done=True)

    def _drop_trajectories(self, label: str) -> None:
        """Error path: discard pending transitions for this game."""
        dropper = getattr(self.policy, 'drop_pending', None)
        if dropper is not None:
            dropper(label)

    def _observe_policy(
        self, label: str, side: int, reward: float, done: bool,
    ) -> None:
        observer = getattr(self.policy, 'observe', None)
        if observer is None:
            return
        try:
            observer(label, side, reward, done=done)
        except Exception as e:
            self.logger.error(f"policy.observe error: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Stats + training loop
    # ------------------------------------------------------------------

    def _record_game_result(self, game: WesnothGame, winner: Optional[int]) -> None:
        self.stats['games_completed'] += 1
        if winner == 1:
            self.stats['wins_side1'] += 1
        elif winner == 2:
            self.stats['wins_side2'] += 1
        else:
            self.stats['draws'] += 1

        if self.stats['games_completed'] % LOG_FREQUENCY == 0:
            self._log_stats()

    def _log_stats(self) -> None:
        total = self.stats['games_completed']
        if total == 0:
            return
        self.logger.info("=" * 70)
        self.logger.info(f"Stats @ {total} games:")
        self.logger.info(
            f"  wins S1={self.stats['wins_side1']} "
            f"S2={self.stats['wins_side2']} "
            f"draws={self.stats['draws']} "
            f"timeouts={self.stats['timeouts']}")
        self.logger.info(f"  total actions: {self.stats['total_actions']}")
        self.logger.info(
            f"  errors: parse={self.stats['state_conversion_errors']} "
            f"send={self.stats['action_send_errors']}")
        self.logger.info("  " + self._profiler.throughput(total))
        self.logger.info("  per-stage:\n" + self._profiler.report())
        self.logger.info("=" * 70)

    async def run_training(self) -> None:
        """Rolling pool of N in-flight games.

        Keeps self.num_games tasks running at all times. Whenever one
        finishes, immediately spawn a replacement — so the slowest game
        in a set doesn't pause the others.

        train_step and checkpoints fire on game-count schedules rather
        than at batch boundaries (there are no batches anymore).
        """
        trainable = bool(getattr(self.policy, 'trainable', False))
        train_every = self.num_games        # same cadence as old batched version
        ckpt_every = CHECKPOINT_FREQUENCY

        self.logger.info(
            f"Rolling pool: {self.num_games} in-flight games. "
            f"train_step every {train_every} games. "
            f"checkpoint every {ckpt_every} games."
        )

        # Map of label → Task, for cancellation on shutdown.
        active: Dict[str, asyncio.Task] = {}
        counter = 0

        def _spawn() -> None:
            nonlocal counter
            label = f"game_{counter}"
            counter += 1
            active[label] = asyncio.create_task(
                self.run_game(label), name=label,
            )

        try:
            for _ in range(self.num_games):
                _spawn()

            while True:
                done, _ = await asyncio.wait(
                    list(active.values()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                finished_labels = [
                    label for label, task in active.items() if task in done
                ]
                # Drain exceptions so they're logged, not silently swallowed.
                for label in finished_labels:
                    task = active.pop(label)
                    if task.exception() is not None:
                        self.logger.error(
                            f"Game {label} raised: {task.exception()}",
                            exc_info=task.exception(),
                        )
                    _spawn()

                # Fire training and checkpointing on game-count schedules.
                self._maybe_train(train_every, trainable)
                self._maybe_checkpoint(ckpt_every, trainable)

        except asyncio.CancelledError:
            raise
        except KeyboardInterrupt:
            self.logger.info("Interrupted — cancelling in-flight games")
            for task in active.values():
                task.cancel()
            await asyncio.gather(*active.values(), return_exceptions=True)
            if trainable:
                self._save_checkpoint()
        finally:
            for game in list(self.games.values()):
                game.terminate()

    def _maybe_train(self, every: int, trainable: bool) -> None:
        n = self.stats['games_completed']
        if not trainable or n == 0 or n < self._last_trained_at + every:
            return
        try:
            self.policy.train_step()  # type: ignore[attr-defined]
        except Exception as e:
            self.logger.error(f"Training error: {e}", exc_info=True)
        self._last_trained_at = n

    def _maybe_checkpoint(self, every: int, trainable: bool) -> None:
        n = self.stats['games_completed']
        if not trainable or n == 0 or n < self._last_checkpoint_at + every:
            return
        self._save_checkpoint()
        self._last_checkpoint_at = n

    def _save_checkpoint(self) -> None:
        saver = getattr(self.policy, 'save_checkpoint', None)
        if saver is None:
            return
        path = CHECKPOINTS_PATH / f"checkpoint_{self.stats['games_completed']}.pt"
        try:
            saver(path)
        except Exception as e:
            self.logger.error(f"Checkpoint save failed: {e}", exc_info=True)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _recruit_cost(action: Dict, game_state: GameState) -> int:
    """Best-effort cost lookup for a recruit action. We look up a unit
    of the same name in the current game state; if none exists, return
    0 (the reward-shaping term contributes 0 then, which is fine)."""
    if action.get('type') != 'recruit':
        return 0
    name = action.get('unit_type', '')
    for u in game_state.map.units:
        if u.name == name:
            return u.cost
    return 0


async def main():
    manager = GameManager(num_games=NUM_PARALLEL_GAMES)
    await manager.run_training()


if __name__ == "__main__":
    asyncio.run(main())
