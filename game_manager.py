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
            state_payload = await asyncio.to_thread(game.read_state)
            if not state_payload:
                self.logger.warning(f"Game {label}: no state, ending game")
                return False

            try:
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
            self._reward_previous_step(label, game_state)

            if game_state.game_over:
                winner = game_state.winner or 0
                self.logger.info(f"Game {label}: over, winner={winner}")
                self._finalize_game(label, game_state, winner, OUTCOME_WIN)
                self._record_game_result(game, winner)
                return False

            # Policy forward + action send.
            try:
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

            if not game.send_action(wire_action):
                self.logger.warning(f"Game {label}: action send failed")
                self.stats['action_send_errors'] += 1
                return False

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
        self.logger.info("=" * 70)

    async def run_training(self) -> None:
        self.logger.info(f"Running {self.num_games} parallel games per batch")
        try:
            while True:
                tasks = [
                    self.run_game(f"game_{self.stats['games_completed'] + i}")
                    for i in range(self.num_games)
                ]
                await asyncio.gather(*tasks)

                if getattr(self.policy, 'trainable', False):
                    try:
                        self.policy.train_step()  # type: ignore[attr-defined]
                    except Exception as e:
                        self.logger.error(f"Training error: {e}", exc_info=True)

                if (self.stats['games_completed']
                        and self.stats['games_completed'] % CHECKPOINT_FREQUENCY == 0
                        and getattr(self.policy, 'trainable', False)):
                    self._save_checkpoint()

                self.logger.info(
                    f"Batch done. Total games: {self.stats['games_completed']}")
        except KeyboardInterrupt:
            self.logger.info("Interrupted")
            if getattr(self.policy, 'trainable', False):
                self._save_checkpoint()
        finally:
            for game in self.games.values():
                game.terminate()

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
