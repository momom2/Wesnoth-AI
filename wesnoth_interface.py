# wesnoth_interface.py
# Interface to Wesnoth game process via file-based IPC
# FIXED: Changed from JSON to WML format parsing

import subprocess
import logging
import time
from pathlib import Path
from typing import Optional, Dict
import threading
import queue

from constants import WESNOTH_PATH, BASE_PATH

class WesnothGame:
    """Manages a single Wesnoth game process and communication via files."""
    
    def __init__(self, game_id: str, scenario_path: Path):
        self.game_id = game_id
        self.scenario_path = scenario_path
        self.logger = logging.getLogger(f"wesnoth_{game_id}")
        
        # Create game directory
        self.game_dir = BASE_PATH / "add-ons" / "wesnoth_ai" / "games" / game_id
        self.game_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.action_file = self.game_dir / "action_input.lua"
        self.game_id_file = self.game_dir / "game_id.txt"
        
        # Write game_id to file so Lua can read it
        self.game_id_file.write_text(game_id)
        
        # Initialize action file with default
        self.action_file.write_text("return {type = 'end_turn'}\n")
        
        # Process handle
        self.process = None
        
        # Stdout reader thread
        self.stdout_queue = queue.Queue()
        self.stdout_thread = None
        
        # Game state
        self.is_running = False
        self.game_over = False
        self.winner = None
        
    def _stdout_reader(self):
        """Thread function to read stdout line by line."""
        try:
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    self.stdout_queue.put(line)
                if self.process.poll() is not None:
                    break
        except Exception as e:
            self.logger.error(f"Error in stdout reader: {e}")
    
    def start_wesnoth(self):
        """Launch Wesnoth process with the training scenario."""
        try:
            # Build command line
            # Use --test instead of --scenario, and --nogui with --multiplayer
            cmd = [
                str(WESNOTH_PATH),
                "--test", "ai_training",  # Run as test scenario
                "--log-info=all"
            ]
            
            self.logger.info(f"Starting Wesnoth: {' '.join(cmd)}")
            
            # Start process with stdout capture
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=str(BASE_PATH)
            )
            
            # Start stdout reader thread
            self.stdout_thread = threading.Thread(
                target=self._stdout_reader,
                daemon=True
            )
            self.stdout_thread.start()
            
            self.is_running = True
            self.logger.info(f"Wesnoth process started (PID: {self.process.pid})")
            
        except Exception as e:
            self.logger.error(f"Failed to start Wesnoth: {e}")
            raise
    
    def read_state(self, timeout_seconds: float = 30.0) -> Optional[str]:
        """
        Read game state from Wesnoth via stdout.
        Returns WML string instead of JSON dict.
        """
        start_time = time.time()
        
        try:
            while time.time() - start_time < timeout_seconds:
                try:
                    # Get line from queue with timeout
                    line = self.stdout_queue.get(timeout=0.1)
                    
                    # Look for WML state marker (changed from JSON)
                    if line.strip().startswith("===WML_STATE_BEGIN==="):
                        # Read WML lines until end marker
                        wml_lines = []
                        while True:
                            wml_line = self.stdout_queue.get(timeout=5.0)
                            if wml_line.strip().startswith("===WML_STATE_END==="):
                                break
                            wml_lines.append(wml_line)
                        
                        # Return WML string (not parsed)
                        wml_str = ''.join(wml_lines)
                        
                        # Log first 200 chars for debugging
                        preview = wml_str[:200] if len(wml_str) > 200 else wml_str
                        self.logger.debug(f"Received WML state (preview): {preview}...")
                        
                        return wml_str
                    
                except queue.Empty:
                    # Check if process died
                    if self.process.poll() is not None:
                        self.logger.warning(f"Wesnoth process terminated")
                        return None
                    continue
            
            self.logger.warning(f"Timeout waiting for state")
            return None
            
        except Exception as e:
            self.logger.error(f"Error reading state: {e}", exc_info=True)
            return None
    
    def send_action(self, action: Dict, timeout_seconds: float = 30.0) -> bool:
        """Send action to Wesnoth via Lua file."""
        try:
            # Convert action dict to Lua table format
            lua_code = self._dict_to_lua(action)
            lua_content = f"return {lua_code}\n"
            
            # Write to action file
            self.action_file.write_text(lua_content)
            
            self.logger.debug(f"Sent action: {action.get('type', 'unknown')}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending action: {e}", exc_info=True)
            return False
    
    def _dict_to_lua(self, d: Dict) -> str:
        """Convert Python dict to Lua table string."""
        parts = []
        for key, value in d.items():
            if isinstance(value, str):
                # Escape single quotes in strings
                escaped_value = value.replace("'", "\\'")
                parts.append(f"{key} = '{escaped_value}'")
            elif isinstance(value, bool):
                parts.append(f"{key} = {str(value).lower()}")
            elif isinstance(value, (int, float)):
                parts.append(f"{key} = {value}")
            elif hasattr(value, 'x') and hasattr(value, 'y'):  # Position object
                parts.append(f"{key} = {{x = {value.x}, y = {value.y}}}")
            elif isinstance(value, dict):
                parts.append(f"{key} = {self._dict_to_lua(value)}")
            elif isinstance(value, list):
                # Convert list to Lua table
                list_items = []
                for item in value:
                    if isinstance(item, str):
                        list_items.append(f"'{item}'")
                    elif isinstance(item, (int, float)):
                        list_items.append(str(item))
                    elif isinstance(item, dict):
                        list_items.append(self._dict_to_lua(item))
                    else:
                        list_items.append(str(item))
                parts.append(f"{key} = {{{', '.join(list_items)}}}")
            else:
                parts.append(f"{key} = {value}")
        
        return "{" + ", ".join(parts) + "}"
    
    def check_game_over(self) -> bool:
        """Check if game over by looking for marker in stdout."""
        try:
            while not self.stdout_queue.empty():
                line = self.stdout_queue.get_nowait()
                if "===GAME_OVER===" in line:
                    # Next line should have winner
                    try:
                        winner_line = self.stdout_queue.get(timeout=1.0)
                        self.winner = int(winner_line.strip())
                    except:
                        self.winner = None
                    self.game_over = True
                    self.logger.info(f"Game over detected. Winner: {self.winner}")
                    return True
        except queue.Empty:
            pass
        except Exception as e:
            self.logger.error(f"Error checking game over: {e}")
        
        return False
    
    def terminate(self):
        """Terminate the Wesnoth process and clean up resources."""
        self.logger.info(f"Terminating game {self.game_id}")
        
        # Terminate process
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception as e:
                self.logger.error(f"Error terminating process: {e}")
        
        self.is_running = False
        
    def __del__(self):
        """Cleanup on deletion."""
        self.terminate()
