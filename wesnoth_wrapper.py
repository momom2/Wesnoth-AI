# wesnoth_wrapper.py
# Simple wrapper that simulates Wesnoth for training purposes
# This allows the training pipeline to work without actual Wesnoth integration

import asyncio
import json
import random
from pathlib import Path
from typing import Optional, Dict, List
import logging


class WesnothGameSimulator:
    """
    Simulates a Wesnoth game for training purposes.
    This is a TEMPORARY solution to unblock training while proper
    Wesnoth integration is developed separately.
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(f"sim_{config.game_id}")
        self.turn = 1
        self.running = True
        self.gold = 100
        self.units = []
        self.process = None  # Dummy for compatibility
        self.is_running = False
        self._initialize_game()

    def _initialize_game(self):
        """Initialize a simple game state."""
        # Start with a leader
        self.units = [{
            "name": "Elvish Captain",
            "side": 1,
            "is_leader": True,
            "x": 12,
            "y": 12,
            "max_hp": 50,
            "max_moves": 5,
            "max_exp": 150,
            "cost": 0,
            "alignment": "NEUTRAL",
            "levelup_names": [],
            "current_hp": 50,
            "current_moves": 5,
            "current_exp": 0,
            "has_attacked": False,
            "attacks": [{
                "name": "sword",
                "type": "SLASH",
                "damage": 8,
                "strikes": 4,
                "is_ranged": False,
                "specials": []
            }],
            "resistances": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 6 damage types
            "defenses": [0.5] * 16,  # 16 terrain types
            "movement_costs": [2] * 16,  # 16 terrain types
            "abilities": ["leadership"],
            "traits": [],
            "statuses": {"poisoned": False, "slowed": False, "petrified": False}
        }]

    def _generate_state(self) -> Dict:
        """Generate current game state."""
        # Randomly add/remove units to simulate gameplay
        if random.random() < 0.3 and len(self.units) < 10:
            self.units.append({
                "name": random.choice(["Elvish Fighter", "Elvish Archer", "Elvish Scout"]),
                "side": 1,
                "is_leader": False,
                "x": random.randint(10, 15),
                "y": random.randint(10, 15),
                "max_hp": 30,
                "max_moves": 5,
                "max_exp": 40,
                "cost": 14,
                "alignment": "NEUTRAL",
                "levelup_names": ["Elvish Hero"],
                "current_hp": 30,
                "current_moves": 5,
                "current_exp": random.randint(0, 10),
                "has_attacked": False,
                "attacks": [{
                    "name": "sword",
                    "type": "SLASH",
                    "damage": 6,
                    "strikes": 3,
                    "is_ranged": False,
                    "specials": []
                }],
                "resistances": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "defenses": [0.5] * 16,
                "movement_costs": [2] * 16,
                "abilities": [],
                "traits": ["quick"],
                "statuses": {"poisoned": False, "slowed": False, "petrified": False}
            })

        # Simulate unit damage
        for unit in self.units:
            if not unit["is_leader"] and random.random() < 0.1:
                unit["current_hp"] = max(1, unit["current_hp"] - random.randint(1, 5))

        # Remove dead units
        self.units = [u for u in self.units if u["current_hp"] > 0]

        # Check win condition
        game_over = self.turn > 30 or random.random() < 0.02
        winner = 1 if game_over and random.random() > 0.5 else (2 if game_over else None)

        return {
            "game_id": self.config.game_id,
            "turn": self.turn,
            "side": 1,
            "gold": self.gold,
            "game_over": game_over,
            "winner": winner,
            "map": {
                "width": 30,
                "height": 20,
                "mask": [],
                "fog": [],
                "hexes": [
                    {"x": x, "y": y, "terrain_types": ["FLAT"], "modifiers": []}
                    for x in range(10, 20)
                    for y in range(10, 20)
                ],
                "units": self.units
            },
            "recruits": [
                {
                    "name": unit_type,
                    "hp": 30,
                    "moves": 5,
                    "exp": 40,
                    "cost": 14,
                    "alignment": "NEUTRAL",
                    "levelup_names": [],
                    "attacks": [],
                    "resistances": [0.0] * 6,
                    "defenses": [0.5] * 16,
                    "movement_costs": [2] * 16,
                    "abilities": [],
                    "traits": []
                }
                for unit_type in ["Elvish Fighter", "Elvish Archer", "Elvish Scout"]
            ]
        }

    async def start(self):
        """Start the simulated game."""
        self.logger.info("Starting simulated game")
        self.is_running = True
        asyncio.create_task(self._game_loop())

    async def _game_loop(self):
        """Main game loop."""
        while self.running and self.turn <= 50:
            # Generate state
            state = self._generate_state()

            # Write state
            with open(self.config.state_file, 'w') as f:
                json.dump(state, f, indent=2)

            # Create signal
            self.config.signal_file.touch()

            self.logger.debug(f"Turn {self.turn}: State written, waiting for action")

            # Wait for action
            action = await self._wait_for_action()

            if action:
                self._execute_action(action)

                if state["game_over"]:
                    self.logger.info(f"Game ended. Winner: {state.get('winner', 'none')}")
                    break

                self.turn += 1
            else:
                self.logger.warning("Action timeout, ending game")
                break

        self.is_running = False

    async def _wait_for_action(self, timeout=30.0) -> Optional[Dict]:
        """Wait for action file."""
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

    def _execute_action(self, action: Dict):
        """Execute an action (simplified simulation)."""
        action_type = action.get("type")

        if action_type == "recruit":
            self.gold -= 14
            self.logger.debug(f"Recruited {action.get('unit_type')}")
        elif action_type == "move":
            self.logger.debug(f"Moved unit from ({action.get('start_x')}, {action.get('start_y')})")
        elif action_type == "attack":
            self.logger.debug(f"Attacked at ({action.get('target_x')}, {action.get('target_y')})")
        elif action_type == "end_turn":
            self.logger.debug("Turn ended")
            self.gold += 2  # Income

    async def wait_for_state(self, timeout: float = 30.0) -> Optional[Dict]:
        """Wait for state (for compatibility)."""
        # Since we're generating states internally, just wait a bit
        await asyncio.sleep(0.5)

        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout:
            if self.config.signal_file.exists():
                try:
                    with open(self.config.state_file, 'r') as f:
                        state = json.load(f)
                    self.config.signal_file.unlink()
                    return state
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
            await asyncio.sleep(0.1)

        return None

    async def send_action(self, action: Dict):
        """Send action (for compatibility)."""
        with open(self.config.action_file, 'w') as f:
            json.dump(action, f)

    async def stop(self):
        """Stop the game."""
        self.running = False
        self.is_running = False

    def get_exit_code(self) -> Optional[int]:
        """Get exit code (for compatibility)."""
        return 0 if not self.is_running else None
