"""Manages parallel Wesnoth games driven by a pluggable policy.

Scope in Phase 1: pure orchestration around the file/log IPC. We launch
Wesnoth, shuttle state in, ask the policy for an action, shuttle the
action back out. No model, no training, no replay buffer — that stuff
was keeping company with a broken transformer and is kept out of the
way until Phase 2 reintroduces a trainable policy.

The `policy` attribute is duck-typed: any object with a
`select_action(game_state) -> dict` method works. See `dummy_policy.py`
for the scripted Phase 1 implementation. Trainable policies (Phase 2)
will add `train_step(...)` and `trainable = True`; the training-loop
hook here looks for that.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional

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
from state_converter import StateConverter
from wesnoth_interface import WesnothGame


class GameManager:
    """Coordinates a batch of Wesnoth subprocesses sharing a single policy."""

    def __init__(
        self,
        num_games: int = NUM_PARALLEL_GAMES,
        policy: Optional[Policy] = None,
    ):
        self.num_games = num_games
        self.games: Dict[str, WesnothGame] = {}

        # Shared state converter — one mapping from unit-type-name to an
        # integer ID, used consistently across games.
        self.global_converter = StateConverter()

        self.logger = logging.getLogger("game_manager")
        self._setup_logging()

        # Default policy: scripted, deterministic, just enough to make
        # the IPC path observable. main.py's --policy flag can pick
        # anything registered in policy.py's _REGISTRY.
        self.policy: Policy = policy if policy is not None else get_policy("dummy")
        self.logger.info(f"Policy: {type(self.policy).__name__}")

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

    async def create_game(self, label: str) -> WesnothGame:
        """`label` is a Python-side tag (e.g., "game_0"). The actual
        Wesnoth-side game_id comes from the scenario's preload and is
        learned when the first state frame arrives."""
        self.logger.info(f"Creating game {label}")
        scenario_path = SCENARIOS_PATH / "training_scenario.cfg"
        game = WesnothGame(label, scenario_path)
        game.start_wesnoth()
        return game

    def get_ai_action(self, game_state: GameState) -> Dict:
        """Ask the policy for an action; translate to wire format."""
        try:
            action = self.policy.select_action(game_state)
            # Lua expects flat 1-indexed fields (start_x/start_y/...),
            # not our internal Position objects. The converter handles
            # both the shape and the coordinate system.
            return self.global_converter.convert_action_to_json(action)
        except Exception as e:
            self.logger.error(f"Policy error: {e}", exc_info=True)
            return {'type': 'end_turn'}

    async def handle_game_turn(self, game: WesnothGame) -> bool:
        """Run one state→action exchange. Return False to end the game."""
        label = game.label
        try:
            state_payload = game.read_state()
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

            # Adopt the Wesnoth-side game_id on the first frame. All
            # subsequent send_action writes will go to the right dir.
            if game.game_id is None:
                game.adopt_game_id(game_state.game_id)

            if game_state.game_over:
                self.logger.info(
                    f"Game {label}: game over, winner={game_state.winner}")
                self._record_game_result(game, game_state.winner)
                return False

            action = self.get_ai_action(game_state)

            if not game.send_action(action):
                self.logger.warning(f"Game {label}: action send failed")
                self.stats['action_send_errors'] += 1
                return False

            self.stats['total_actions'] += 1
            return True

        except Exception as e:
            self.logger.error(f"Game {label}: turn error: {e}", exc_info=True)
            return False

    async def run_game(self, label: str) -> None:
        """`label` is a human-readable per-batch tag; Wesnoth's own
        game_id is learned from the first state frame."""
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

        except Exception as e:
            self.logger.error(f"Game {label}: error: {e}", exc_info=True)
        finally:
            if label in self.games:
                self.games[label].terminate()
                del self.games[label]
            self.logger.info(f"Game {label}: finished")

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
        self.logger.info("=== Stats ===")
        self.logger.info(f"Games: {total}")
        self.logger.info(f"Side 1 wins: {self.stats['wins_side1']}")
        self.logger.info(f"Side 2 wins: {self.stats['wins_side2']}")
        self.logger.info(f"Draws: {self.stats['draws']}")
        self.logger.info(f"Timeouts: {self.stats['timeouts']}")
        self.logger.info(f"Total actions: {self.stats['total_actions']}")
        self.logger.info(f"Parse errors: {self.stats['state_conversion_errors']}")
        self.logger.info(f"Send errors: {self.stats['action_send_errors']}")
        self.logger.info("=" * 70)

    async def run_training(self) -> None:
        """Main loop. Training hook fires when the policy is trainable."""
        self.logger.info(f"Running {self.num_games} parallel games per batch")
        try:
            while True:
                tasks = [
                    self.run_game(f"game_{self.stats['games_completed'] + i}")
                    for i in range(self.num_games)
                ]
                await asyncio.gather(*tasks)

                # Phase 2: trainable policies will implement train_step.
                # For now this is a no-op.
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
        """Delegated to the policy. No-op if the policy isn't trainable."""
        saver = getattr(self.policy, 'save_checkpoint', None)
        if saver is None:
            return
        path = CHECKPOINTS_PATH / f"checkpoint_{self.stats['games_completed']}.pt"
        try:
            saver(path)
            self.logger.info(f"Saved checkpoint to {path}")
        except Exception as e:
            self.logger.error(f"Checkpoint save failed: {e}", exc_info=True)


async def main():
    manager = GameManager(num_games=NUM_PARALLEL_GAMES)
    await manager.run_training()


if __name__ == "__main__":
    asyncio.run(main())
