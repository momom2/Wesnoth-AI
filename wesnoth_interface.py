"""Per-game Wesnoth process wrapper and mixed-transport IPC.

Why this is mixed: Wesnoth 1.18's Lua sandbox excludes `io` entirely,
so Lua cannot write files directly. Meanwhile, `wesnoth.exe` on Windows
is a GUI-subsystem binary whose stdout never reaches a piped subprocess.
The only writable outbound channel available to Lua is `std_print()`,
which Wesnoth routes to `<userdata>/logs/wesnoth-*.out.log`. We tail that
file.

Inbound (Python → Lua) is simpler: Python writes `action.lua` atomically;
Lua reads it via `wesnoth.read_file`. Lua can't delete files either, so
each action includes a monotonic `seq` field; Lua tracks the
last-executed seq in `wml.variables` and ignores stale ones.

Per-turn protocol:
  1. Lua CA `ca_state_sender` std_print()s state wrapped in marker lines.
  2. Python tails the .out.log, extracts the framed block, parses WML.
  3. Python writes action.lua with seq = self._next_seq and increments.
  4. Lua CA `ca_action_executor` reads action.lua, checks seq freshness,
     executes, stores new seq in wml.variables.last_action_seq.
"""

import itertools
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
    STATE_POLL_INTERVAL,
    STATE_TIMEOUT_SECONDS,
    WESNOTH_LOGS_PATH,
    WESNOTH_PATH,
)

# Must match the constants in ca_state_sender.lua
FRAME_BEGIN = "===WESNOTH_AI_STATE_BEGIN==="
FRAME_END = "===WESNOTH_AI_STATE_END==="

# Monotonic action-sequence counter shared across all WesnothGame instances.
# Using a module-level counter avoids races when parallel games share a Lua
# environment; each game still has its own action.lua file so they don't
# collide.
_seq_source = itertools.count(start=1)


class WesnothGame:
    """One Wesnoth subprocess, one game, one per-game IPC directory.

    The Wesnoth-side game_id (the subdir name under add-ons/wesnoth_ai/
    games/ that Lua reads action.lua from) is chosen by the scenario's
    preload event as a random string. Python doesn't know the id until
    it parses the first state frame; `adopt_game_id(...)` is called
    then. Until adoption, send_action fails — but the game loop always
    reads state before sending, so that's the natural ordering.

    `label` here is a short human-readable tag used in log-file names
    and in game_manager's stats; it does NOT need to match the
    Wesnoth-side game_id.
    """

    def __init__(self, label: str, scenario_path: Path):
        self.label = label
        self.scenario_path = scenario_path
        self.logger = logging.getLogger(f"wesnoth_{label}")

        # Assigned by adopt_game_id() when the first state frame lands.
        self.game_id: Optional[str] = None
        self.game_dir: Optional[Path] = None
        self.action_path: Optional[Path] = None

        # Log-tailing state.
        self._log_path: Optional[Path] = None
        self._log_offset = 0
        self._log_buffer = ""

        # Per-process launch timestamp (used to pick the right .out.log).
        self._launch_time: Optional[float] = None

        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        self.game_over = False
        self.winner: Optional[int] = None

    def adopt_game_id(self, game_id: str) -> None:
        """Called once, after the first state frame, to lock in the
        id Lua generated and create the per-game IPC directory."""
        if self.game_id is not None:
            # Already adopted. If the id changed, that's a protocol
            # error — ignore subsequent changes and log.
            if self.game_id != game_id:
                self.logger.warning(
                    f"game_id changed mid-game: {self.game_id!r} → "
                    f"{game_id!r}; keeping the first one")
            return
        self.game_id = game_id
        self.game_dir = GAMES_PATH / game_id
        self.game_dir.mkdir(parents=True, exist_ok=True)
        self.action_path = self.game_dir / ACTION_FILE_NAME

        # Clear any stale action file (shouldn't exist — id is random).
        for stale in (self.action_path,
                      self.game_dir / (ACTION_FILE_NAME + ".tmp")):
            stale.unlink(missing_ok=True)

        self.logger.info(f"adopted game_id={game_id}, ipc dir {self.game_dir}")

    def start_wesnoth(self) -> None:
        """Launch Wesnoth on our training scenario.

        Using `--test ai_training`, which opens a GUI window. Headless
        (`--nogui --multiplayer --scenario=...`) turned out to be
        unreliable on this Windows install — see
        memory/reference_wesnoth_headless_attempt.md for findings.
        When headless works, latency can drop substantially; until
        then, the GUI overhead is tolerable for single-game runs, and
        Phase 3.2 will hide most of it behind parallel processes.

        Stdout/stderr are discarded — on Windows they're not connected
        to a console anyway. Lua-originated output reaches us via
        <userdata>/logs/wesnoth-*.out.log.
        """
        cmd = [str(WESNOTH_PATH), "--test", "ai_training"]

        # Small slack so we don't mistake a log file that was created a
        # hair before Popen for ours. Wesnoth typically writes its log
        # filename within ~100ms of launch.
        self._launch_time = time.time() - 0.5

        self.logger.info(f"Starting Wesnoth: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.is_running = True
        self.logger.info(f"Wesnoth started (PID {self.process.pid})")

    def _find_out_log(self) -> Optional[Path]:
        """Find the .out.log file Wesnoth opened for this process.

        Wesnoth names logs wesnoth-<UTC-date>-<UTC-time>-<rand>.out.log.
        We pick the newest .out.log whose mtime is >= our launch time.
        """
        if not WESNOTH_LOGS_PATH.exists():
            return None
        if self._launch_time is None:
            return None

        candidates = [
            p for p in WESNOTH_LOGS_PATH.glob("wesnoth-*.out.log")
            if p.stat().st_mtime >= self._launch_time
        ]
        if not candidates:
            return None
        # Newest first.
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    def _read_new_log_content(self) -> str:
        """Pull any new bytes from the .out.log since we last read."""
        if self._log_path is None:
            self._log_path = self._find_out_log()
            if self._log_path is None:
                return ""

        try:
            with self._log_path.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(self._log_offset)
                new = f.read()
                self._log_offset = f.tell()
                return new
        except OSError as e:
            self.logger.debug(f"log read error: {e}")
            return ""

    def _extract_state_frame(self) -> Optional[str]:
        """If our buffer contains a complete BEGIN..END frame, consume it
        and return just the WML payload (without markers or meta header).
        """
        begin = self._log_buffer.find(FRAME_BEGIN)
        if begin < 0:
            # Discard anything before a BEGIN marker so the buffer doesn't
            # grow unboundedly with unrelated engine output.
            if len(self._log_buffer) > 64 * 1024:
                self._log_buffer = self._log_buffer[-1024:]
            return None

        end = self._log_buffer.find(FRAME_END, begin + len(FRAME_BEGIN))
        if end < 0:
            return None  # frame still being written

        frame = self._log_buffer[begin + len(FRAME_BEGIN):end]
        self._log_buffer = self._log_buffer[end + len(FRAME_END):]

        # Strip the opening newline and the "meta:" header line that
        # ca_state_sender emits right after BEGIN.
        lines = frame.splitlines()
        while lines and (not lines[0].strip() or lines[0].startswith("meta:")):
            lines.pop(0)
        return "\n".join(lines)

    def read_state(self, timeout: float = STATE_TIMEOUT_SECONDS) -> Optional[str]:
        """Block until the Lua side emits a fresh state frame, then return
        the raw WML payload. None on timeout or process death.
        """
        deadline = time.time() + timeout

        while time.time() < deadline:
            self._log_buffer += self._read_new_log_content()

            frame = self._extract_state_frame()
            if frame is not None:
                self.logger.debug(f"Got state frame ({len(frame)} chars)")
                return frame

            if self.process is not None and self.process.poll() is not None:
                self.logger.warning(
                    f"Wesnoth exited (returncode={self.process.returncode}) "
                    f"before state arrived"
                )
                # Drain any last bytes in case the frame landed just before exit.
                self._log_buffer += self._read_new_log_content()
                return self._extract_state_frame()

            time.sleep(STATE_POLL_INTERVAL)

        self.logger.warning(f"Timeout after {timeout:.1f}s waiting for state")
        return None

    def send_action(self, action: Dict,
                    timeout: float = ACTION_TIMEOUT_SECONDS) -> bool:
        """Write the action file atomically with a fresh sequence number.
        Returns True if the file landed on disk. Doesn't wait for Lua to
        consume it — the CA evaluator polls and handles that itself.

        Requires adopt_game_id() to have been called (i.e., at least
        one state frame observed). In practice the turn-loop always
        reads state before sending, so this is a non-issue.
        """
        if self.action_path is None or self.game_dir is None:
            self.logger.error("send_action before adopt_game_id()")
            return False

        action_with_seq = dict(action)
        action_with_seq["seq"] = next(_seq_source)

        lua_code = "return " + self._dict_to_lua(action_with_seq) + "\n"
        tmp = self.game_dir / (ACTION_FILE_NAME + ".tmp")

        try:
            tmp.write_text(lua_code, encoding="utf-8")
            os.replace(tmp, self.action_path)
        except OSError as e:
            self.logger.error(f"Failed to write action: {e}")
            return False

        self.logger.debug(
            f"Sent action seq={action_with_seq['seq']} type={action.get('type', '?')}")
        return True

    def _dict_to_lua(self, d: Dict) -> str:
        parts = [f"{k} = {self._lua_value(v)}" for k, v in d.items()]
        return "{" + ", ".join(parts) + "}"

    def _lua_value(self, value) -> str:
        if isinstance(value, str):
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
        return repr(value)

    def check_game_over(self) -> bool:
        """Compatibility shim. Game-over state now flows through the state
        payload (game_over / winner fields), not a separate check."""
        return self.game_over

    def terminate(self) -> None:
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
