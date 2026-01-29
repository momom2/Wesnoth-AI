# local_game_launcher.py

import asyncio
import subprocess
import json
import logging
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass

from assumptions import WESNOTH_STATE_TIMEOUT, GAME_STARTUP_TIMEOUT
from classes import GameConfig

# USE_SIMULATOR: Set to True to use Python game simulator instead of real Wesnoth
USE_SIMULATOR = True

@dataclass
class WesnothConfig:
    """Configuration for Wesnoth executable and addon paths."""
    wesnoth_exe: Path
    userdata_dir: Path
    addon_dir: Path


class LocalGameInstance:
    """Manages a single Wesnoth game process running locally."""

    def __init__(self, config: GameConfig, wesnoth_config: WesnothConfig):
        self.config = config
        self.wesnoth_config = wesnoth_config
        self.process: Optional[subprocess.Popen] = None
        self.logger = logging.getLogger(f"wesnoth_{config.game_id}")
        self.is_running = False

    def _generate_scenario_file(self) -> Path:
        """Generate a scenario file for this specific game instance."""
        template_path = Path("wesnoth_plugin/training_scenario.cfg")
        scenario_path = self.config.state_file.parent / "scenario.cfg"

        # Read template
        with open(template_path, 'r') as f:
            template = f.read()

        # TODO: Map selection logic - for now using a default map
        map_file = "multiplayer/1v1_maps/Hornshark_Island.map"

        # TODO: Faction selection logic - for now random
        # Replace placeholders
        scenario = template.replace("{MAP_FILE}", map_file)
        scenario = scenario.replace("{STARTING_GOLD}", "100")
        scenario = scenario.replace("{LEADER_1_TYPE}", "Elvish Captain")
        scenario = scenario.replace("{FACTION_1}", "Rebels")
        scenario = scenario.replace("{RECRUIT_LIST_1}", "Elvish Fighter,Elvish Archer,Elvish Scout")
        scenario = scenario.replace("{LEADER_2_TYPE}", "Orcish Warrior")
        scenario = scenario.replace("{FACTION_2}", "Northerners")
        scenario = scenario.replace("{RECRUIT_LIST_2}", "Orcish Grunt,Orcish Archer,Wolf Rider")

        # Write scenario file
        with open(scenario_path, 'w') as f:
            f.write(scenario)

        return scenario_path

    def _build_wesnoth_command(self, scenario_file: Path) -> list:
        """Build command line to launch Wesnoth."""
        cmd = [
            str(self.wesnoth_config.wesnoth_exe),
            "--nogui",  # No graphical interface
            "--noaddons",  # Don't load other addons (only our AI plugin)
            "--data-dir", str(self.wesnoth_config.userdata_dir),
            "--scenario", str(scenario_file),
            # Pass game_id and base_path to Lua AI
            "--preprocess-defines",
            f"EXTERNAL_AI_GAME_ID={self.config.game_id}",
            f"EXTERNAL_AI_PATH={self.config.state_file.parent}"
        ]
        return cmd

    async def start(self):
        """Start the Wesnoth game process."""
        try:
            # Ensure game directory exists
            self.config.state_file.parent.mkdir(parents=True, exist_ok=True)

            # Generate scenario file
            scenario_file = self._generate_scenario_file()

            # Build command
            cmd = self._build_wesnoth_command(scenario_file)

            self.logger.info(f"Starting Wesnoth: {' '.join(cmd)}")

            # Start process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            self.is_running = True
            self.logger.info(f"Wesnoth process started with PID {self.process.pid}")

        except Exception as e:
            self.logger.error(f"Failed to start Wesnoth: {e}")
            self.is_running = False

    async def wait_for_state(self, timeout: float = WESNOTH_STATE_TIMEOUT) -> Optional[Dict]:
        """Wait for game to write state file."""
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            if self.config.signal_file.exists():
                try:
                    # Read state
                    with open(self.config.state_file, 'r') as f:
                        state = json.load(f)

                    # Remove signal file
                    self.config.signal_file.unlink()

                    return state

                except (FileNotFoundError, json.JSONDecodeError) as e:
                    self.logger.warning(f"Error reading state: {e}")

            # Check if process is still running
            if self.process and self.process.poll() is not None:
                self.logger.warning("Wesnoth process exited unexpectedly")
                self.is_running = False
                return None

            await asyncio.sleep(0.1)

        self.logger.error("Timeout waiting for game state")
        return None

    async def send_action(self, action: Dict):
        """Send action to Wesnoth via action file."""
        try:
            with open(self.config.action_file, 'w') as f:
                json.dump(action, f)
            self.logger.debug(f"Sent action: {action}")
        except Exception as e:
            self.logger.error(f"Failed to send action: {e}")

    async def stop(self):
        """Stop the Wesnoth process."""
        if self.process:
            self.logger.info("Stopping Wesnoth process")
            self.process.terminate()
            try:
                await asyncio.wait_for(
                    asyncio.create_task(self._wait_for_process()),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                self.logger.warning("Wesnoth didn't terminate, killing")
                self.process.kill()

            self.is_running = False

    async def _wait_for_process(self):
        """Wait for process to exit."""
        while self.process.poll() is None:
            await asyncio.sleep(0.1)

    def get_exit_code(self) -> Optional[int]:
        """Get exit code if process has exited."""
        if self.process:
            return self.process.poll()
        return None


class LocalGameManager:
    """Manages multiple local Wesnoth game instances."""

    def __init__(self, wesnoth_config: WesnothConfig):
        self.wesnoth_config = wesnoth_config
        self.games: Dict[str, LocalGameInstance] = {}
        self.logger = logging.getLogger("local_game_manager")

        # Ensure addon is installed
        self._setup_addon()

    def _setup_addon(self):
        """Ensure our AI addon is in Wesnoth's addon directory."""
        addon_target = self.wesnoth_config.addon_dir / "External_AI"
        addon_source = Path("wesnoth_plugin")

        if addon_target.exists():
            self.logger.info(f"AI addon already exists at {addon_target}")
        else:
            self.logger.info(f"Installing AI addon to {addon_target}")
            addon_target.mkdir(parents=True, exist_ok=True)

            # Copy files
            import shutil
            for file in addon_source.glob("*"):
                if file.is_file():
                    shutil.copy(file, addon_target / file.name)

    async def create_game(self, config: GameConfig):
        """Create and start a new game instance."""
        if USE_SIMULATOR:
            from wesnoth_wrapper import WesnothGameSimulator
            game = WesnothGameSimulator(config)
            await game.start()
            self.games[config.game_id] = game
            self.logger.info(f"Started simulator for {config.game_id}")
        else:
            game = LocalGameInstance(config, self.wesnoth_config)
            await game.start()
            self.games[config.game_id] = game
            self.logger.info(f"Started real Wesnoth for {config.game_id}")
        return game

    async def stop_game(self, game_id: str):
        """Stop a specific game."""
        if game_id in self.games:
            await self.games[game_id].stop()
            del self.games[game_id]

    async def stop_all(self):
        """Stop all running games."""
        for game_id in list(self.games.keys()):
            await self.stop_game(game_id)
