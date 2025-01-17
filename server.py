import socket
import struct
import gzip
import threading
import logging
from dataclasses import dataclass
from typing import Dict, Optional
import xml.etree.ElementTree as ET

# TODO: Determine appropriate logging format and level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Client:
    socket: socket.socket
    username: Optional[str] = None
    game_id: Optional[int] = None
    # TODO: Add fields for:
    # - side number in game
    # - player state (lobby/game/observing)
    # - connection state (handshake/lobby/game)
    
@dataclass
class Game:
    id: int
    name: str
    host: Client
    clients: Dict[str, Client]
    observers: Dict[str, Client]
    started: bool = False
    # TODO: Add fields for:
    # - game state (setup/playing/ended)
    # - current turn
    # - scenario data
    # - replay data
    # - game settings (era, modifications, etc.)

class WesnothServer:
    def __init__(self, host='localhost', port=15000):
        """
        Initialize the Wesnoth server.
        
        UNCERTAIN: Is port 15000 a good default? The official server seems to use 15000
        but documentation doesn't specify if this is standard.
        """
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        self.clients: Dict[socket.socket, Client] = {}
        self.games: Dict[int, Game] = {}
        self.next_game_id = 1
        
    def start(self):
        """Start the server"""
        self.socket.bind((self.host, self.port))
        # UNCERTAIN: Is 5 an appropriate backlog value?
        self.socket.listen(5)
        logger.info(f"Server started on {self.host}:{self.port}")
        
        while True:
            client_socket, address = self.socket.accept()
            logger.info(f"New connection from {address}")
            client = Client(socket=client_socket)
            self.clients[client_socket] = client
            
            # Handle each client in a separate thread
            thread = threading.Thread(target=self.handle_client, args=(client,))
            thread.daemon = True
            thread.start()
    
    def handle_client(self, client: Client):
        """Handle individual client connections"""
        try:
            # Handle handshake
            if not self.handle_handshake(client):
                return
                
            # Main client loop
            while True:
                wml = self.receive_wml(client.socket)
                if not wml:
                    break
                    
                self.process_wml(client, wml)
                
        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            self.disconnect_client(client)

    def handle_handshake(self, client: Client) -> bool:
        """
        Handle the initial handshake with a client
        
        UNCERTAIN: The documentation mentions protocol changes in 1.13+ and 1.15+
        Need to verify this implementation works with current Wesnoth version.
        """
        try:
            # Receive client's 4 bytes
            data = client.socket.recv(4)
            if len(data) != 4:
                return False
                
            # Check if client wants TLS (we don't support it yet)
            if data == b'\x00\x00\x00\x01':
                logger.warning("Client requested TLS which is not supported")
                return False
                
            # Send back connection number (always 1 in modern versions)
            client.socket.send(struct.pack('!I', 1))
            
            # Send version request
            self.send_wml(client.socket, self.create_version_request())
            
            return True
            
        except Exception as e:
            logger.error(f"Handshake failed: {e}")
            return False
            
    def receive_wml(self, sock: socket.socket) -> Optional[str]:
        """
        Receive and decode a WML message
        
        UNCERTAIN: Current implementation assumes messages are complete.
        Need to handle partial messages and message boundaries properly.
        """
        try:
            # Read message size (4 bytes, big endian)
            size_data = sock.recv(4)
            if not size_data:
                return None
            size = struct.unpack('!I', size_data)[0]
            
            # Read the full message
            data = b''
            while len(data) < size:
                chunk = sock.recv(min(size - len(data), 4096))
                if not chunk:
                    return None
                data += chunk
                
            # Decompress and decode
            decompressed = gzip.decompress(data)
            return decompressed.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error receiving WML: {e}")
            return None
            
    def send_wml(self, sock: socket.socket, wml: str) -> bool:
        """Encode and send a WML message"""
        try:
            # Compress the WML
            compressed = gzip.compress(wml.encode('utf-8'))
            
            # Send size followed by data
            size = struct.pack('!I', len(compressed))
            sock.send(size)
            sock.send(compressed)
            return True
            
        except Exception as e:
            logger.error(f"Error sending WML: {e}")
            return False
            
    def create_version_request(self) -> str:
        """Create a version request WML message"""
        # TODO: Implement proper version negotiation
        return '[version]\n[/version]\n'
        
    def process_wml(self, client: Client, wml: str):
        """
        Process received WML messages
        
        UNCERTAIN: Using XML parsing for WML might not handle all WML features correctly.
        Need to verify this works with complex WML or implement proper WML parser.
        """
        try:
            # Parse WML into XML-like structure for easier processing
            root = ET.fromstring(wml.replace('[', '<').replace(']', '>'))
            
            if root.tag == 'version':
                # TODO: Implement handle_version
                self.handle_version(client, root)
            elif root.tag == 'login':
                # TODO: Implement handle_login
                self.handle_login(client, root)
            elif root.tag == 'create_game':
                # TODO: Implement handle_create_game
                self.handle_create_game(client, root)
            elif root.tag == 'join':
                # TODO: Implement handle_join_game
                self.handle_join_game(client, root)
            # TODO: Implement handlers for:
            # - scenario_diff (game setup changes)
            # - start_game
            # - leave_game
            # - turn (game actions)
            # - chat messages
            else:
                logger.warning(f"Unhandled WML tag: {root.tag}")
                
        except Exception as e:
            logger.error(f"Error processing WML: {e}")

    def disconnect_client(self, client: Client):
        """Clean up when a client disconnects"""
        try:
            if client.game_id and client.game_id in self.games:
                game = self.games[client.game_id]
                if client.username in game.clients:
                    del game.clients[client.username]
                if client.username in game.observers:
                    del game.observers[client.username]
                    
                # If host disconnects, end the game
                if game.host == client:
                    self.end_game(game)
                    
            if client.socket in self.clients:
                del self.clients[client.socket]
                
            client.socket.close()
            
        except Exception as e:
            logger.error(f"Error disconnecting client: {e}")

    def end_game(self, game: Game):
        """End a game and notify all participants"""
        try:
            # Notify all participants
            for participant in list(game.clients.values()) + list(game.observers.values()):
                self.send_wml(participant.socket, '[leave_game]\n[/leave_game]\n')
                participant.game_id = None
                
            # Remove the game
            if game.id in self.games:
                del self.games[game.id]
                
        except Exception as e:
            logger.error(f"Error ending game: {e}")

    # TODO: Implement methods for:
    # - handle_version(self, client: Client, data: ET.Element)
    # - handle_login(self, client: Client, data: ET.Element)
    # - handle_create_game(self, client: Client, data: ET.Element)
    # - handle_join_game(self, client: Client, data: ET.Element)
    # - handle_game_setup(self, game: Game, data: ET.Element)
    # - handle_start_game(self, game: Game)
    # - handle_turn(self, game: Game, client: Client, data: ET.Element)
    # - broadcast_to_game(self, game: Game, wml: str)
    # - validate_game_action(self, game: Game, client: Client, action: str) -> bool

if __name__ == '__main__':
    server = WesnothServer()
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Server shutting down")