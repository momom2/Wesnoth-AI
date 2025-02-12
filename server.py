import socket
import struct
import gzip
import logging
import threading
import msvcrt
from dataclasses import dataclass, field
from typing import Optional, Dict, Set, Any
from datetime import datetime
import re
import random
from collections import defaultdict
from logging.handlers import RotatingFileHandler
from assumptions import HOST, PORT, TIMEOUT, LOG_FILE, HANDSHAKE_SIZE, CHUNK_SIZE

# Pre-compile all regex patterns for efficiency
USERNAME_PATTERN = re.compile(r'username="([^"]+)"')
GAME_NAME_PATTERN = re.compile(r'name="([^"]+)"')

# Pre-generate common responses once
VERSION_RESPONSE = gzip.compress(b'[version]\n[/version]\n')
MUSTLOGIN_RESPONSE = gzip.compress(b'[mustlogin]\n[/mustlogin]\n')
JOIN_LOBBY_RESPONSE = gzip.compress(b'[join_lobby]\nis_moderator=no\n[/join_lobby]\n')

@dataclass
class Game:
    """Represents a game instance in the server."""
    id: int
    name: str
    host: str
    created: datetime
    scenario: str = ""
    era: str = ""
    turn: int = 0
    sides: Dict[int, str] = field(default_factory=dict)
    active: bool = True
    
    def to_wml(self) -> str:
        """Generates WML representation of the game state."""
        parts = [
            '[game]',
            f'id={self.id}',
            f'name="{self.name}"'
        ]
        if self.scenario:
            parts.append(f'scenario="{self.scenario}"')
        if self.era:
            parts.append(f'era="{self.era}"')
        if self.turn:
            parts.append(f'turn={self.turn}')
        
        # Add sides information
        for side_num, player in self.sides.items():
            parts.append(f'[side]\nside={side_num}\nplayer="{player}"\n[/side]')
            
        parts.append('[/game]')
        return '\n'.join(parts)

class ClientConnection:
    """Manages a single client connection."""
    def __init__(self, socket: socket.socket, address: str, server: 'WesnothTrainingServer'):
        self.socket = socket
        self.address = address
        self.server = server
        self.username: Optional[str] = None
        self.game_id: Optional[int] = None
        self.buffer = bytearray()
        self.logger = logging.getLogger(f"client_{address}")
        
    def handle(self):
        """Main client handling loop."""
        try:
            self._handle_handshake()
            while True:
                wml = self._receive_wml()
                if not wml:
                    break
                self._process_wml(wml)
        except (socket.error, ConnectionError) as e:
            self.logger.error(f"Connection error: {e}")
        finally:
            self._cleanup()
    
    def _handle_request_choice(self, wml: str):
        # Extract request_id if present
        match = re.search(r'request_id=(\d+)', wml)
        if match:
            request_id = match.group(1)
                    
            # Check if this is a random_seed request
            if '[random_seed]' in wml:
                # Generate random seed response
                #seed = random.randint(0, 9999) # Testing with a fixed seed for now
                seed = 0
                response = f'[choice]\nrequest_id={request_id}\n[random_seed]\nseed={seed}\n[/random_seed]\n[/choice]\n'
            else:
                # For other types of choices, send back an empty choice
                response = f'[choice]\nrequest_id={request_id}\n[/choice]\n'
                        
            if not self.send_wml(response, log=False):
                self.logger.debug(f"Failed to send choice response: {response}")
            self.logger.debug(f"Sent choice response: {response}")

    def _handle_handshake(self):
        """Handles initial client handshake."""
        try:
            data = self.socket.recv(HANDSHAKE_SIZE)
            if not data:
                raise ConnectionError("Empty handshake")
            
            if data == b'\x00\x00\x00\x01':
                self.logger.info("Client requested TLS (not supported)")
                raise ConnectionError("TLS not supported")
                
            self.socket.send(struct.pack('!I', 1))
            self._send_raw(VERSION_RESPONSE)
            
        except socket.timeout:
            raise ConnectionError("Handshake timeout")
            
    def _receive_wml(self) -> Optional[str]:
        """Receives and decodes a WML message with partial message handling."""
        try:
            # First get the message size
            while len(self.buffer) < HANDSHAKE_SIZE:
                chunk = self.socket.recv(HANDSHAKE_SIZE - len(self.buffer))
                if not chunk:
                    return None
                self.buffer.extend(chunk)
            
            size = struct.unpack('!I', self.buffer[:HANDSHAKE_SIZE])[0]
            self.buffer = self.buffer[HANDSHAKE_SIZE:]
            
            # Then get the complete message
            while len(self.buffer) < size:
                remaining = size - len(self.buffer)
                chunk = self.socket.recv(min(remaining, CHUNK_SIZE))
                if not chunk:
                    return None
                self.buffer.extend(chunk)
            
            # Extract the complete message
            message = self.buffer[:size]
            self.buffer = self.buffer[size:]
            
            # Decompress and decode
            decompressed = gzip.decompress(message)
            wml = decompressed.decode('utf-8')
            
            # Log the received WML
            self.logger.debug(f"Received WML: {wml}")
            
            return wml
            
        except socket.timeout:
            self.logger.warning("Socket timeout during receive")
            return None
        except Exception as e:
            self.logger.error(f"Error receiving WML: {e}")
            return None

    def _process_wml(self, wml: str):
        """Processes received WML messages."""
        try:
            if '[version]' in wml:
                self._send_raw(MUSTLOGIN_RESPONSE)
                
            elif '[login]' in wml:
                match = USERNAME_PATTERN.search(wml)
                if match:
                    username = match.group(1)
                    if self.server.add_user(username, self):
                        self.username = username
                        self._send_raw(JOIN_LOBBY_RESPONSE)
                        self.server.send_game_list(self)
                        
            elif '[create_game]' in wml:
                if self.username:
                    match = GAME_NAME_PATTERN.search(wml)
                    if match:
                        game_name = match.group(1)
                        game_id = self.server.create_game(game_name, self.username)
                        if game_id is not None:
                            self.game_id = game_id
                            
            elif '[leave_game]' in wml and self.username:
                if self.game_id is not None:
                    self.server.remove_game(self.game_id)
                    self.game_id = None
                    
            elif '[request_choice]' in wml:
                self._handle_request_choice(wml)
                    
        except Exception as e:
            self.logger.error(f"Error processing WML: {e}")

    def send_wml(self, wml: str, log=True) -> bool:
        """Sends a WML message to the client."""
        try:
            if log:
                self.logger.debug(f"Sending WML: {wml}")
            compressed = gzip.compress(wml.encode('utf-8'))
            size = struct.pack('!I', len(compressed))
            self.socket.sendall(size + compressed)
            return True
        except Exception as e:
            if log:
                self.logger.error(f"Error sending WML: {e}")
            return False

    def _send_raw(self, compressed_data: bytes) -> bool:
        """Sends pre-compressed data to the client."""
        try:
            # Log the decompressed data for debugging
            try:
                decompressed = gzip.decompress(compressed_data).decode('utf-8')
                self.logger.debug(f"Sending raw WML: {decompressed}")
            except Exception as e:
                self.logger.debug(f"Could not decode raw WML for logging: {e}")
                
            size = struct.pack('!I', len(compressed_data))
            self.socket.sendall(size + compressed_data)
            return True
        except Exception as e:
            self.logger.error(f"Error sending pre-compressed WML: {e}")
            return False

    def _cleanup(self):
        """Cleans up client resources."""
        try:
            if self.game_id is not None:
                self.server.remove_game(self.game_id)
            if self.username is not None:
                self.server.remove_user(self.username)
            self.socket.close()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

class WesnothTrainingServer:
    """Main server class managing multiple games and clients."""
    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = port
        self.running = True
        
        # Server socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((host, port))
        
        # State management
        self.games: Dict[int, Game] = {}
        self.clients: Dict[str, ClientConnection] = {}
        self.next_game_id = 1
        
        # Configure logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Configures logging with both file and console output."""
        # Clear any existing handlers
        logging.getLogger().handlers = []
        
        # Set root logger to DEBUG
        logging.getLogger().setLevel(logging.DEBUG)
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=1024*1024, backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)  # Ensure file gets everything
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)  # Changed from INFO to DEBUG
        console_handler.setFormatter(console_formatter)
        
        # Add handlers to root logger
        logging.getLogger().addHandler(file_handler)
        logging.getLogger().addHandler(console_handler)
        
        # Get our logger
        self.logger = logging.getLogger("wesnoth_server")

    def start(self):
        """Starts the server."""
        self.socket.listen(5)
        self.socket.settimeout(TIMEOUT)
        self.logger.info(f"Server listening on {self.host}:{self.port}")
        
        # Start keyboard listener thread
        keyboard_thread = threading.Thread(target=self._keyboard_listener)
        keyboard_thread.daemon = True
        keyboard_thread.start()
        
        # Main server loop
        while self.running:
            try:
                client_socket, address = self.socket.accept()
                self._handle_new_client(client_socket, address)
            except socket.timeout:
                continue
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
        
        self._cleanup()

    def _handle_new_client(self, client_socket: socket.socket, address: Any):
        """Handles a new client connection."""
        self.logger.info(f"New connection from {address}")
        client = ClientConnection(client_socket, address, self)
        client_thread = threading.Thread(target=client.handle)
        client_thread.daemon = True
        client_thread.start()

    def _keyboard_listener(self):
        """Listens for keyboard commands."""
        self.logger.info("Press 'k' to display current games, 'q' to exit")
        while self.running:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8').lower()
                if key == 'k':
                    self._display_games()
                elif key == 'q':
                    self.logger.info("Shutting down server...")
                    self.running = False
                    break

    def _display_games(self):
        """Displays information about all current games."""
        print("\n=== Current Games ===")
        if not self.games:
            print("No active games")
        else:
            for game in self.games.values():
                print(f"\n{game}")
        print("===================\n")

    def add_user(self, username: str, client: ClientConnection) -> bool:
        """Adds a new user to the server."""
        if username in self.clients:
            return False
        self.clients[username] = client
        self.logger.info(f"User '{username}' logged in")
        return True

    def remove_user(self, username: str):
        """Removes a user from the server."""
        if username in self.clients:
            del self.clients[username]
            self.logger.info(f"User '{username}' logged out")

    def create_game(self, name: str, host: str) -> Optional[int]:
        """Creates a new game."""
        game_id = self.next_game_id
        self.next_game_id += 1
        
        game = Game(
            id=game_id,
            name=name,
            host=host,
            created=datetime.now()
        )
        
        self.games[game_id] = game
        self.logger.info(f"Created game {game_id}: {name}")
        self._broadcast_game_list()
        return game_id

    def remove_game(self, game_id: int):
        """Removes a game from the server."""
        if game_id in self.games:
            game = self.games[game_id]
            game.active = False
            del self.games[game_id]
            self.logger.info(f"Removed game {game_id}")
            self._broadcast_game_list()

    def send_game_list(self, client: ClientConnection):
        """Sends the game list to a specific client."""
        wml = self._generate_game_list()
        client.send_wml(wml)

    def _broadcast_game_list(self):
        """Broadcasts the game list to all clients."""
        wml = self._generate_game_list()
        compressed = gzip.compress(wml.encode('utf-8'))
        size = struct.pack('!I', len(compressed))
        message = size + compressed
        
        # Send to all connected clients
        for client in list(self.clients.values()):
            try:
                client.socket.sendall(message)
            except Exception as e:
                self.logger.error(f"Error broadcasting to {client.username}: {e}")

    def _generate_game_list(self) -> str:
        """Generates the WML for the game list."""
        if not self.games:
            return '[gamelist]\n[/gamelist]\n'
        
        parts = ['[gamelist]']
        parts.extend(game.to_wml() for game in self.games.values())
        parts.append('[/gamelist]')
        return '\n'.join(parts)

    def _cleanup(self):
        """Cleans up server resources."""
        self.running = False
        
        # Close all client connections
        for client in list(self.clients.values()):
            try:
                client.socket.close()
            except:
                pass
        
        # Close server socket
        try:
            self.socket.close()
        except:
            pass
            
        self.logger.info("Server shut down")

if __name__ == '__main__':
    server = WesnothTrainingServer()
    try:
        server.start()
    except KeyboardInterrupt:
        pass