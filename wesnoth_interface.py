"""Per-game Wesnoth process wrapper and file-based IPC.

Design: Wesnoth on Windows is a GUI-subsystem executable, so its stdout
never reaches a piped subprocess. We ignore stdout entirely and do all
IPC via files in the game's per-process directory. Lua and Python both
see the same directory thanks to the add-on directory junction created
by main.install_addon().

Per-turn protocol:
  1. Lua CA `ca_state_sender` writes state.wml.tmp, renames to state.wml.
  2. Python polls for state.wml, reads it, deletes it.
  3. Python decides an action, writes action.lua.tmp, renames to action.lua.
  4. Lua CA `ca_action_executor` reads action.lua, deletes it, executes.

File names are constants in constants.py so Lua and Python agree.
"""

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

from constants import (
    ACTION_FILE_NAME,
    ACTION_TIMEOUT_SECONDS,
    GAMES_PATH,
    STATE_FILE_NAME,
    STATE_POLL_INTERVAL,
    STATE_TIMEOUT_SECONDS,
    WESNOTH_PATH,
)


class WesnothGame:
    """One Wesnoth subprocess, one game, one per-game IPC directory."""

    def __init__(self, game_id: str, scenario_path: Path):
        self.game_id = game_id
        self.scenario_path = scenario_path
        self.logger = logging.getLogger(f"wesnoth_{game_id}")

        self.game_dir = GAMES_PATH / game_id
        self.game_dir.mkdir(parents=True, exist_ok=True)

        self.state_path = self.game_dir / STATE_FILE_NAME
        self.action_path = self.game_dir / ACTION_FILE_NAME
        self.game_id_path = self.game_dir / "game_id.txt"

        # Tell the scenario's preload event which game this is.
        self.game_id_path.write_text(game_id)

        # Start clean — remove any stale IPC files from prior runs.
        for stale in [self.state_path, self.action_path,
                      self.game_dir / (STATE_FILE_NAME + ".tmp"),
                      self.game_dir / (ACTION_FILE_NAME + ".tmp")]:
            stale.unlink(missing_ok=True)

        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        self.game_over = False
        self.winner: Optional[int] = None

    def start_wesnoth(self) -> None:
        """Launch the Wesnoth process. Its stdout is unusable on Windows;
        we rely on the file-based IPC and Wesnoth's own log files under
        <userdata>/logs/ for observability."""
        cmd = [str(WESNOTH_PATH), "--test", "ai_training"]
        self.logger.info(f"Starting Wesnoth: {' '.join(cmd)}")

        # DEVNULL for stdout/stderr: on Windows they're disconnected anyway,
        # and on POSIX we don't want Wesnoth's verbose logs flooding ours.
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.is_running = True
        self.logger.info(f"Wesnoth started (PID {self.process.pid})")

    def read_state(self, timeout: float = STATE_TIMEOUT_SECONDS) -> Optional[str]:
        """Poll for a fresh state file, return its contents, delete it.

        Returns the raw WML state string (the Python-side parser in
        state_converter.py consumes it), or None on timeout/process death.
        """
        deadline = time.time() + timeout

        while time.time() < deadline:
            if self.state_path.exists():
                try:
                    content = self.state_path.read_text(encoding="utf-8")
                except OSError as e:
                    # File may have been mid-rename; try again.
                    self.logger.debug(f"Transient read error: {e}")
                    time.sleep(STATE_POLL_INTERVAL)
                    continue

                try:
                    self.state_path.unlink()
                except OSError:
                    pass  # already gone; harmless

                self.logger.debug(f"Read {len(content)} bytes of state")
                return content

            if self.process is not None and self.process.poll() is not None:
                self.logger.warning(
                    f"Wesnoth exited (returncode={self.process.returncode}) "
                    f"while waiting for state"
                )
                return None

            time.sleep(STATE_POLL_INTERVAL)

        self.logger.warning(f"Timeout after {timeout:.1f}s waiting for state")
        return None

    def send_action(self, action: Dict, timeout: float = ACTION_TIMEOUT_SECONDS) -> bool:
        """Write the action file atomically. Returns True if the file
        landed on disk; doesn't wait for Lua to consume it."""
        lua_code = "return " + self._dict_to_lua(action) + "\n"
        tmp = self.game_dir / (ACTION_FILE_NAME + ".tmp")

        try:
            tmp.write_text(lua_code, encoding="utf-8")
            os.replace(tmp, self.action_path)  # atomic overwrite cross-platform
        except OSError as e:
            self.logger.error(f"Failed to write action: {e}")
            return False

        self.logger.debug(f"Sent action: {action.get('type', '?')}")
        return True

    def _dict_to_lua(self, d: Dict) -> str:
        """Serialize a Python dict to a Lua table literal.

        Supports: str, int, float, bool, nested dicts, lists of same, and
        Position-like objects (anything with .x and .y attributes).
        """
        parts = []
        for key, value in d.items():
            parts.append(f"{key} = {self._lua_value(value)}")
        return "{" + ", ".join(parts) + "}"

    def _lua_value(self, value) -> str:
        if isinstance(value, str):
            # Use double-quoted string with backslash-escapes on the two
            # characters Lua treats specially.
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return repr(value)
        if hasattr(value, "x") and hasattr(value, "y"):  # Position
            return f"{{x = {value.x}, y = {value.y}}}"
        if isinstance(value, dict):
            return self._dict_to_lua(value)
        if isinstance(value, (list, tuple)):
            items = [self._lua_value(v) for v in value]
            return "{" + ", ".join(items) + "}"
        # Fall back to repr — caller-beware if it contains anything weird.
        return repr(value)

    def check_game_over(self) -> bool:
        """True if the game-over marker file exists. The scenario writes
        this via the Lua 'endlevel' handler; we look for it opportunistically
        but the authoritative signal is the `game_over` flag embedded in
        the state WML (see state_converter)."""
        # Placeholder: with file IPC, game_over flows through the state
        # payload itself, so this method is kept only for interface compat.
        return self.game_over

    def terminate(self) -> None:
        """Kill the Wesnoth process if still running. Safe to call twice."""
        if self.process and self.process.poll() is None:
            self.logger.info(f"Terminating Wesnoth (PID {self.process.pid})")
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except OSError as e:
                self.logger.error(f"Termination error: {e}")
        self.is_running = False

    def __del__(self):
        self.terminate()
