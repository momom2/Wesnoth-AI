#!/usr/bin/env python3
# test_mock_game.py
# Test the training manager with a mock Wesnoth game (simulates file-based communication)

import asyncio
import json
import random
from pathlib import Path
import tempfile

from classes import GameConfig


class MockWesnothGame:
    """Simulates a Wesnoth game by writing mock state files."""

    def __init__(self, config: GameConfig):
        self.config = config
        self.turn = 1
        self.running = True
        self.actions_received = []

    async def run(self):
        """Main game loop - simulates Wesnoth writing states and reading actions."""
        print(f"[{self.config.game_id}] Mock game started")

        while self.running and self.turn <= 5:  # Run for 5 turns
            # Write game state
            state = self._generate_mock_state()
            with open(self.config.state_file, 'w') as f:
                json.dump(state, f, indent=2)

            # Create signal
            self.config.signal_file.touch()

            print(f"[{self.config.game_id}] Turn {self.turn}: Waiting for action...")

            # Wait for action (with timeout)
            action = await self._wait_for_action(timeout=5.0)

            if action:
                print(f"[{self.config.game_id}] Received action: {action['type']}")
                self.actions_received.append(action)

                if action['type'] == 'end_turn':
                    self.turn += 1
            else:
                print(f"[{self.config.game_id}] Timeout waiting for action")
                break

        # Game over
        final_state = self._generate_mock_state(game_over=True)
        with open(self.config.state_file, 'w') as f:
            json.dump(final_state, f, indent=2)
        self.config.signal_file.touch()

        print(f"[{self.config.game_id}] Game ended after {self.turn} turns")

    def _generate_mock_state(self, game_over=False):
        """Generate a mock game state."""
        return {
            "game_id": self.config.game_id,
            "turn": self.turn,
            "side": 1,
            "gold": 100 - (self.turn * 10),
            "game_over": game_over,
            "winner": 1 if game_over and random.random() > 0.5 else (2 if game_over else None),
            "map": {
                "width": 30,
                "height": 20,
                "mask": [],
                "fog": [],
                "hexes": [
                    {
                        "x": x,
                        "y": y,
                        "terrain_types": ["FLAT"],
                        "modifiers": []
                    }
                    for x in range(10, 15)
                    for y in range(10, 15)
                ],
                "units": [
                    {
                        "name": "Elvish Archer",
                        "side": 1,
                        "is_leader": False,
                        "x": 12,
                        "y": 12,
                        "max_hp": 29,
                        "max_moves": 6,
                        "max_exp": 36,
                        "cost": 14,
                        "alignment": "NEUTRAL",
                        "levelup_names": ["Elvish Ranger"],
                        "current_hp": 29 - self.turn,
                        "current_moves": 6,
                        "current_exp": self.turn * 3,
                        "has_attacked": False,
                        "attacks": [
                            {
                                "name": "bow",
                                "type": "PIERCE",
                                "damage": 5,
                                "strikes": 4,
                                "is_ranged": True,
                                "specials": []
                            }
                        ],
                        "resistances": {
                            "blade": 0,
                            "pierce": 0,
                            "impact": 0,
                            "fire": 0,
                            "cold": 0,
                            "arcane": 0
                        },
                        "defenses": {},
                        "movement_costs": {},
                        "abilities": [],
                        "traits": ["DEXTROUS"],
                        "statuses": {
                            "poisoned": False,
                            "slowed": False,
                            "petrified": False
                        }
                    }
                ]
            },
            "recruits": [
                {
                    "name": "Elvish Fighter",
                    "hp": 33,
                    "moves": 5,
                    "exp": 32,
                    "cost": 14,
                    "alignment": "NEUTRAL",
                    "levelup_names": ["Elvish Captain"],
                    "attacks": [],
                    "resistances": {},
                    "defenses": {},
                    "movement_costs": {},
                    "abilities": [],
                    "traits": []
                }
            ]
        }

    async def _wait_for_action(self, timeout=5.0):
        """Wait for AI to write action file."""
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            if self.config.action_file.exists():
                try:
                    with open(self.config.action_file, 'r') as f:
                        action = json.load(f)
                    self.config.action_file.unlink()
                    return action
                except (FileNotFoundError, json.JSONDecodeError):
                    pass

            await asyncio.sleep(0.1)

        return None


class MockAI:
    """Simulates the AI side - reads states and writes actions."""

    def __init__(self, config: GameConfig):
        self.config = config
        self.running = True
        self.states_processed = 0

    async def run(self):
        """Monitor for state updates and respond with actions."""
        print(f"[AI-{self.config.game_id}] Mock AI started")

        while self.running:
            # Wait for signal
            if self.config.signal_file.exists():
                try:
                    # Read state
                    with open(self.config.state_file, 'r') as f:
                        state = json.load(f)

                    # Remove signal
                    self.config.signal_file.unlink()

                    print(f"[AI-{self.config.game_id}] Processing turn {state['turn']}")

                    # Generate mock action
                    action = self._generate_mock_action(state)

                    # Write action
                    with open(self.config.action_file, 'w') as f:
                        json.dump(action, f)

                    self.states_processed += 1

                    # Check if game is over
                    if state.get('game_over'):
                        print(f"[AI-{self.config.game_id}] Game over detected")
                        break

                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"[AI-{self.config.game_id}] Error: {e}")

            await asyncio.sleep(0.1)

        print(f"[AI-{self.config.game_id}] Processed {self.states_processed} states")

    def _generate_mock_action(self, state):
        """Generate a random valid action based on state."""
        # Simple logic: just end turn for now
        return {"type": "end_turn"}


async def test_single_game():
    """Test a single mock game interaction."""
    print("\n" + "=" * 60)
    print("Testing Single Game Communication")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "game_0"
        base_path.mkdir(parents=True)

        config = GameConfig(
            game_id="game_0",
            map_name="test_map",
            faction1="Rebels",
            faction2="Northerners",
            state_file=base_path / "state.json",
            action_file=base_path / "action.json",
            signal_file=base_path / "signal"
        )

        # Create mock game and AI
        game = MockWesnothGame(config)
        ai = MockAI(config)

        # Run both concurrently
        await asyncio.gather(
            game.run(),
            ai.run()
        )

        print(f"\n[OK] Game completed {game.turn} turns")
        print(f"[OK] AI processed {ai.states_processed} states")
        print(f"[OK] Game received {len(game.actions_received)} actions")

        assert game.turn >= 5, "Game should complete at least 5 turns"
        assert ai.states_processed >= 5, "AI should process at least 5 states"
        assert len(game.actions_received) >= 5, "Game should receive at least 5 actions"

    print("[OK] Single game test passed!")


async def test_parallel_games():
    """Test multiple mock games running in parallel."""
    print("\n" + "=" * 60)
    print("Testing Parallel Games Communication")
    print("=" * 60)

    num_games = 3

    with tempfile.TemporaryDirectory() as tmpdir:
        games = []
        ais = []

        # Create multiple game instances
        for i in range(num_games):
            base_path = Path(tmpdir) / f"game_{i}"
            base_path.mkdir(parents=True)

            config = GameConfig(
                game_id=f"game_{i}",
                map_name="test_map",
                faction1="Rebels",
                faction2="Northerners",
                state_file=base_path / "state.json",
                action_file=base_path / "action.json",
                signal_file=base_path / "signal"
            )

            games.append(MockWesnothGame(config))
            ais.append(MockAI(config))

        # Run all games and AIs concurrently
        tasks = []
        for game, ai in zip(games, ais):
            tasks.append(game.run())
            tasks.append(ai.run())

        await asyncio.gather(*tasks)

        print(f"\n[OK] All {num_games} games completed")
        for i, (game, ai) in enumerate(zip(games, ais)):
            print(f"  Game {i}: {game.turn} turns, {ai.states_processed} states processed")

    print("[OK] Parallel games test passed!")


async def main():
    """Run all mock game tests."""
    try:
        await test_single_game()
        await test_parallel_games()

        print("\n" + "=" * 60)
        print("All Mock Game Tests Passed!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
