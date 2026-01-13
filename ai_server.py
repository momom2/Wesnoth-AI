"""
AI Server - Receives game states from Wesnoth Lua AI and returns actions from the transformer model
"""

import socket
import json
import logging
import threading
from typing import Dict, Optional
from logging.handlers import RotatingFileHandler
import torch

from assumptions import HOST, PORT
from transformer import WesnothTransformer
from classes import Input, Memory
from game_manager import TrainingManager
from action_selector import ActionSelector

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('ai_server.log', maxBytes=1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)

class AIServerConnection:
    """Handles a single connection from a Wesnoth game instance."""

    def __init__(self, socket: socket.socket, address: tuple, server: 'AIServer'):
        self.socket = socket
        self.address = address
        self.server = server
        self.game_id: Optional[str] = None
        self.logger = logging.getLogger(f"connection_{address}")

    def handle(self):
        """Main connection handling loop."""
        try:
            self.logger.info(f"New connection from {self.address}")

            while True:
                # Receive JSON message (newline-delimited)
                data = self._receive_line()
                if not data:
                    break

                # Parse JSON
                try:
                    message = json.loads(data)
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON: {e}")
                    continue

                # Process message
                response = self._process_message(message)

                # Send response
                if response:
                    self._send_json(response)

        except Exception as e:
            self.logger.error(f"Connection error: {e}")
        finally:
            self._cleanup()

    def _receive_line(self) -> Optional[str]:
        """Receive a newline-terminated message."""
        buffer = b""
        try:
            while True:
                chunk = self.socket.recv(1)
                if not chunk:
                    return None
                if chunk == b"\n":
                    return buffer.decode('utf-8')
                buffer += chunk
        except socket.timeout:
            return None
        except Exception as e:
            self.logger.error(f"Error receiving: {e}")
            return None

    def _send_json(self, obj: dict) -> bool:
        """Send JSON object as newline-terminated message."""
        try:
            message = json.dumps(obj) + "\n"
            self.socket.sendall(message.encode('utf-8'))
            return True
        except Exception as e:
            self.logger.error(f"Error sending: {e}")
            return False

    def _process_message(self, message: dict) -> Optional[dict]:
        """Process received message and return response."""
        msg_type = message.get('type')

        if msg_type == 'state':
            # Received game state, return action
            return self._handle_state(message)
        elif msg_type == 'game_over':
            # Game ended
            return self._handle_game_over(message)
        else:
            self.logger.warning(f"Unknown message type: {msg_type}")
            return {'type': 'error', 'message': f'Unknown message type: {msg_type}'}

    def _handle_state(self, message: dict) -> dict:
        """Handle game state and return action."""
        self.game_id = message.get('game_id', 'unknown')

        # Convert to AI input format
        try:
            ai_input = self.server.training_manager.convert_state_to_ai_input(message)
        except Exception as e:
            self.logger.error(f"Error converting state: {e}")
            return {'type': 'action', 'action_type': 'end_turn'}

        # Get action from AI
        try:
            with torch.no_grad():
                start_logits, target_logits, attack_logits, recruit_logits, value = \
                    self.server.ai_model(ai_input.map, ai_input.recruits, ai_input.memory)

                action = self._select_action(
                    message,
                    start_logits,
                    target_logits,
                    attack_logits,
                    recruit_logits
                )
        except Exception as e:
            self.logger.error(f"Error getting AI action: {e}")
            return {'type': 'action', 'action_type': 'end_turn'}

        self.logger.info(f"Game {self.game_id}: AI selected action {action['action_type']}")
        return action

    def _select_action(self, state: dict, start_logits, target_logits, attack_logits, recruit_logits) -> dict:
        """Select action from model outputs."""
        # Use ActionSelector to convert model outputs to actions
        return self.server.action_selector.select_action(
            state,
            start_logits,
            target_logits,
            attack_logits,
            recruit_logits,
            training=True  # Add exploration during training
        )

    def _handle_game_over(self, message: dict) -> dict:
        """Handle game over notification."""
        winner = message.get('winner')
        self.logger.info(f"Game {self.game_id} ended, winner: {winner}")

        # TODO: Update replay buffer with rewards

        return {'type': 'ack'}

    def _cleanup(self):
        """Clean up connection resources."""
        self.logger.info(f"Connection closed: {self.address}")
        try:
            self.socket.close()
        except:
            pass


class AIServer:
    """Main AI server that interfaces with Wesnoth Lua AI."""

    def __init__(self, host: str = HOST, port: int = PORT + 1):  # Use different port
        self.host = host
        self.port = port
        self.running = True

        # Initialize AI model
        self.logger = logging.getLogger("ai_server")
        self.logger.info("Loading AI model...")

        self.ai_model = WesnothTransformer()
        self.training_manager = TrainingManager(num_parallel_games=0)  # Don't auto-start games
        self.action_selector = ActionSelector(temperature=1.0, exploration_factor=0.1)

        # Load checkpoint if available
        self._load_checkpoint()

        # Create server socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((host, port))

        self.logger.info(f"AI Server initialized on {host}:{port}")

    def _load_checkpoint(self):
        """Load latest model checkpoint if available."""
        from pathlib import Path

        checkpoint_dir = Path("./training/checkpoints")
        if not checkpoint_dir.exists():
            self.logger.info("No checkpoint directory found, starting with fresh model")
            return

        # Find all checkpoint files
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            self.logger.info("No checkpoints found, starting with fresh model")
            return

        # Load the latest checkpoint
        latest_checkpoint = checkpoints[-1]
        self.logger.info(f"Loading checkpoint: {latest_checkpoint}")

        try:
            checkpoint = torch.load(latest_checkpoint)

            # Load model state
            self.ai_model.load_state_dict(checkpoint['model_state'])

            # Load training stats if available
            if 'training_stats' in checkpoint:
                self.training_manager.training_stats = checkpoint['training_stats']
                self.logger.info(f"Loaded training stats: {checkpoint['training_stats']}")

            self.logger.info(f"Successfully loaded checkpoint from {latest_checkpoint}")

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            self.logger.info("Starting with fresh model")

    def save_checkpoint(self, games_completed: int = 0):
        """Save current model checkpoint."""
        from pathlib import Path

        checkpoint_dir = Path("./training/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state': self.ai_model.state_dict(),
            'training_stats': self.training_manager.training_stats,
            'timestamp': __import__('datetime').datetime.now().isoformat()
        }

        checkpoint_path = checkpoint_dir / f"checkpoint_{games_completed}.pt"

        try:
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")

            # Keep only last 5 checkpoints to save space
            checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
            if len(checkpoints) > 5:
                for old_checkpoint in checkpoints[:-5]:
                    old_checkpoint.unlink()
                    self.logger.info(f"Deleted old checkpoint: {old_checkpoint}")

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

    def start(self):
        """Start the server."""
        self.socket.listen(10)
        self.socket.settimeout(1.0)

        self.logger.info(f"AI Server listening on {self.host}:{self.port}")
        self.logger.info("Press Ctrl+C to exit")

        while self.running:
            try:
                client_socket, address = self.socket.accept()
                self._handle_new_connection(client_socket, address)
            except socket.timeout:
                continue
            except KeyboardInterrupt:
                self.logger.info("Shutting down...")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")

        self._cleanup()

    def _handle_new_connection(self, client_socket: socket.socket, address: tuple):
        """Handle new connection in a separate thread."""
        connection = AIServerConnection(client_socket, address, self)
        thread = threading.Thread(target=connection.handle)
        thread.daemon = True
        thread.start()

    def _cleanup(self):
        """Clean up server resources."""
        self.running = False
        try:
            self.socket.close()
        except:
            pass
        self.logger.info("AI Server shut down")


if __name__ == '__main__':
    # Use port 15001 for AI server (15000 is for WML server)
    server = AIServer(host='localhost', port=15001)
    try:
        server.start()
    except KeyboardInterrupt:
        pass
