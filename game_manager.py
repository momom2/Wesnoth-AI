# game_manager.py

import asyncio
import json
import logging
import datetime
import random
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
import torch

from assumptions import (
    MAX_ACTIONS_ALLOWED, ACTION_PENALTY, TIMEOUT_PENALTY,
    MEMORY_STATE_SIZE, REPLAY_BUFFER_SIZE, REPLAY_BATCH_SIZE,
    CHECKPOINT_FREQUENCY, REPLAY_SAVE_FREQUENCY
)
from classes import (
    GameConfig, Map, Unit, Attack, Position, Hex, Memory,
    Input, PartialUnit, Alignment, UnitAbility, UnitTrait,
    DamageType, Terrain, TerrainModifiers, AttackSpecial
)
from transformer import WesnothTransformer

@dataclass
class Experience:
    """Single training experience with metadata."""
    game_id: str
    state: Input
    action: dict
    value: float
    reward: Optional[float]  # None until game ends
    timestamp: datetime.datetime
    action_number: int
    game_length: Optional[int]  # None until game ends

class GameInstance:
    """Manages a single game of Wesnoth."""
    def __init__(self, config: GameConfig):
        self.config = config
        self.logger = logging.getLogger(f"game_{config.game_id}")
        self.is_active = True
        self.last_state = None
        self.action_count = 0
        self.experiences: List[Experience] = []
        
    async def watch_for_state_update(self) -> Optional[dict]:
        """Monitor state file for updates from Wesnoth."""
        while self.is_active:
            try:
                if self.config.signal_file.exists():
                    # Read new state
                    state_data = json.loads(self.config.state_file.read_text())
                    self.config.signal_file.unlink()  # Clear signal
                    self.last_state = state_data
                    return state_data
            except Exception as e:
                self.logger.error(f"Error reading game state: {e}")
                await asyncio.sleep(0.1)
        return None
    
    async def send_action(self, action: dict):
        """Send an action back to Wesnoth."""
        try:
            self.config.action_file.write_text(json.dumps(action))
            self.action_count += 1
        except Exception as e:
            self.logger.error(f"Error sending action: {e}")

class TrainingManager:
    """Manages multiple concurrent games and AI training."""
    def __init__(self, num_parallel_games: int = 16):
        self.num_games = num_parallel_games
        self.games: Dict[str, GameInstance] = {}
        self.game_memories: Dict[str, Memory] = {}
        self.logger = logging.getLogger("training_manager")
        
        # Training statistics
        self.training_stats = {
            'games_completed': 0,
            'wins': 0,
            'timeouts': 0,
            'average_game_length': 0,
            'average_reward': 0.0,
            'loss_policy': 0.0,
            'loss_value': 0.0,
            'loss_consistency': 0.0
        }
        
        # Experience replay buffer
        self.replay_buffer: List[Experience] = []
        
        # Initialize AI model and optimizer
        self.ai = WesnothTransformer()
        self.optimizer = torch.optim.Adam(self.ai.parameters())
        
        # Set up save directories
        self.save_dir = Path("./training")
        self.save_dir.mkdir(exist_ok=True)
        (self.save_dir / "checkpoints").mkdir(exist_ok=True)
        (self.save_dir / "replays").mkdir(exist_ok=True)
    
    async def run_training_loop(self):
        """Main training loop managing multiple games and AI updates."""
        try:
            while True:
                # Ensure we have enough active games
                await self.maintain_game_count()
                
                # Collect states from all active games
                game_states = await self.collect_game_states()
                
                if game_states:
                    # Get AI decisions for all games that need them
                    actions = await self.process_states_with_ai(game_states)
                    
                    # Send actions back to games
                    await self.send_actions(actions)
                    
                    # Update AI if we have enough experiences
                    await self.maybe_update_ai()
                
                # Brief sleep to prevent tight loop
                await asyncio.sleep(0.01)
                
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            await self.cleanup()
    
    async def maintain_game_count(self):
        """Ensures we maintain the desired number of parallel games."""
        current_count = len([g for g in self.games.values() if g.is_active])
        games_needed = self.num_games - current_count
        
        if games_needed > 0:
            for _ in range(games_needed):
                game_id = f"game_{len(self.games)}"
                config = self.create_game_config(game_id)
                self.games[game_id] = GameInstance(config)
    
    async def collect_game_states(self) -> Dict[str, dict]:
        """Collects new states from all active games."""
        states = {}
        for game_id, game in self.games.items():
            if game.is_active:
                state = await game.watch_for_state_update()
                if state:
                    states[game_id] = state
                    
                    # Check for game completion
                    if state.get('game_over') or game.action_count >= MAX_ACTIONS_ALLOWED:
                        await self.handle_game_completion(game_id)
        return states
    
    def convert_state_to_ai_input(self, state: dict) -> Input:
        """Convert game state from Wesnoth format to AI input format."""
        # Convert map information
        map_state = state['map']
        game_map = Map(
            size_x=map_state['width'],
            size_y=map_state['height'],
            mask=set(Position(x=p['x'], y=p['y']) for p in map_state['mask']),
            fog=set(Position(x=p['x'], y=p['y']) for p in map_state['fog']),
            hexes=set(self.convert_hex(h) for h in map_state['hexes']),
            units=set(self.convert_unit(u) for u in map_state['units'])
        )
        
        # Convert recruitment options
        recruits = [self.convert_partial_unit(u) for u in state['recruits']]
        
        # Get or initialize memory for this game
        game_id = state['game_id']
        if game_id not in self.game_memories:
            self.game_memories[game_id] = Memory(state=[0.0] * MEMORY_STATE_SIZE)
        
        return Input(
            map=game_map,
            recruits=recruits,
            memory=self.game_memories[game_id]
        )
    
    def convert_hex(self, hex_data: dict) -> Hex:
        """Convert hex data from Wesnoth format to internal format."""
        return Hex(
            position=Position(x=hex_data['x'], y=hex_data['y']),
            terrain_types=set(Terrain[t.upper()] for t in hex_data['terrain_types']),
            modifiers=set(TerrainModifiers[m.upper()] for m in hex_data.get('modifiers', []))
        )
    
    def convert_unit(self, unit_data: dict) -> Unit:
        """Convert unit data from Wesnoth format to internal format."""
        return Unit(
            name=unit_data['name'],
            side=unit_data['side'],
            is_leader=unit_data['is_leader'],
            position=Position(x=unit_data['x'], y=unit_data['y']),
            max_hp=unit_data['max_hp'],
            max_moves=unit_data['max_moves'],
            max_exp=unit_data['max_exp'],
            cost=unit_data['cost'],
            alignment=Alignment[unit_data['alignment'].upper()],
            levelup_names=unit_data['levelup_names'],
            current_hp=unit_data['current_hp'],
            current_moves=unit_data['current_moves'],
            current_exp=unit_data['current_exp'],
            has_attacked=unit_data['has_attacked'],
            attacks=[self.convert_attack(a) for a in unit_data['attacks']],
            resistances=unit_data['resistances'],
            defenses=unit_data['defenses'],
            movement_costs=unit_data['movement_costs'],
            abilities=set(UnitAbility[a.upper()] for a in unit_data['abilities']),
            traits=set(UnitTrait[t.upper()] for t in unit_data['traits'])
        )
    
    def convert_attack(self, attack_data: dict) -> Attack:
        """Convert attack data from Wesnoth format to internal format."""
        return Attack(
            type_id=DamageType[attack_data['type'].upper()],
            number_strikes=attack_data['strikes'],
            damage_per_strike=attack_data['damage'],
            is_ranged=attack_data['is_ranged'],
            weapon_specials=set(AttackSpecial[s.upper()] for s in attack_data['specials'])
        )
    
    def convert_partial_unit(self, unit_data: dict) -> PartialUnit:
        """Convert recruitment option from Wesnoth format to internal format."""
        return PartialUnit(
            name=unit_data['name'],
            hp=unit_data['hp'],
            moves=unit_data['moves'],
            exp=unit_data['exp'],
            cost=unit_data['cost'],
            alignment=Alignment[unit_data['alignment'].upper()],
            levelup_names=unit_data['levelup_names'],
            attacks=[self.convert_attack(a) for a in unit_data['attacks']],
            resistances=unit_data['resistances'],
            defenses=unit_data['defenses'],
            movement_costs=unit_data['movement_costs'],
            abilities=set(UnitAbility[a.upper()] for a in unit_data['abilities']),
            traits=set(UnitTrait[t.upper()] for t in unit_data['traits'])
        )
    
    async def process_states_with_ai(self, game_states: Dict[str, dict]) -> Dict[str, dict]:
        """Get AI decisions for all active games that need them."""
        actions = {}
        for game_id, state in game_states.items():
            # Convert game state to AI input format
            ai_input = self.convert_state_to_ai_input(state)
            
            # Get AI action
            with torch.no_grad():
                start_logits, target_logits, attack_logits, recruit_logits, value = self.ai(
                    ai_input.map, ai_input.recruits, ai_input.memory
                )
                action = self.select_action(start_logits, target_logits, attack_logits, recruit_logits)
            
            actions[game_id] = action
            
            # Store experience
            exp = Experience(
                game_id=game_id,
                state=ai_input,
                action=action,
                value=value.item(),
                reward=None,  # Will be set when game ends
                timestamp=datetime.datetime.now(),
                action_number=self.games[game_id].action_count,
                game_length=None  # Will be set when game ends
            )
            self.games[game_id].experiences.append(exp)
        
        return actions
    
    def create_game_config(self, game_id: str) -> GameConfig:
        """Create configuration for a new game."""
        base_path = Path(f"./games/{game_id}")
        base_path.mkdir(parents=True, exist_ok=True)
        
        return GameConfig(
            game_id=game_id,
            map_name="standard_1v1",  # We'll need a list of maps
            faction1="random",        # We'll need faction selection logic
            faction2="random",
            state_file=base_path / "state.json",
            action_file=base_path / "action.json",
            signal_file=base_path / "signal"
        )
    
    async def handle_game_completion(self, game_id: str):
        """Handle end of game, update experiences with final rewards."""
        game = self.games[game_id]
        game.is_active = False
        
        # Calculate final reward
        final_reward = self.calculate_reward(game)
        
        # Update all experiences from this game with reward and game length
        for exp in game.experiences:
            exp.reward = final_reward
            exp.game_length = game.action_count
            
            # Add to replay buffer with length-based discounting
            discounted_reward = final_reward / math.sqrt(game.action_count)
            exp.reward = discounted_reward
            
            if len(self.replay_buffer) >= REPLAY_BUFFER_SIZE:
                self.replay_buffer.pop(0)
            self.replay_buffer.append(exp)
        
        # Update training statistics
        self.update_training_stats(game, final_reward)
        
        # Save replay if needed
        if self.training_stats['games_completed'] % REPLAY_SAVE_FREQUENCY == 0:
            self.save_replay(game)
    
    def calculate_reward(self, game: GameInstance) -> float:
        """Calculate reward for a game outcome."""
        base_reward = 1.0 if game.last_state['winner'] == 1 else -1.0
        action_penalties = game.action_count * ACTION_PENALTY
        
        # Apply timeout penalty if game reached action limit
        if game.action_count >= MAX_ACTIONS_ALLOWED:
            base_reward -= TIMEOUT_PENALTY
            self.training_stats['timeouts'] += 1
        
        return base_reward - action_penalties
    
    def update_training_stats(self, game: GameInstance, final_reward: float):
        """Update training statistics after game completion."""
        self.training_stats['games_completed'] += 1
        if game.last_state['winner'] == 1:
            self.training_stats['wins'] += 1
        
        # Update moving averages
        alpha = 0.01  # Small value for smooth averaging
        self.training_stats['average_game_length'] = (
            (1 - alpha) * self.training_stats['average_game_length'] +
            alpha * game.action_count
        )
        self.training_stats['average_reward'] = (
            (1 - alpha) * self.training_stats['average_reward'] +
            alpha * final_reward
        )
        
        # Log progress periodically
        if self.training_stats['games_completed'] % 100 == 0:
            self.logger.info(f"Training stats: {self.training_stats}")
    
    def save_replay(self, game: GameInstance):
        """Save detailed game replay for analysis."""
        replay_path = self.save_dir / "replays" / f"game_{self.training_stats['games_completed']}.json"
        replay_data = {
            'game_id': game.config.game_id,
            'actions': game.action_count,
            'winner': game.last_state['winner'],
            'experiences': [
                {
                    'action_number': exp.action_number,
                    'action': exp.action,
                    'value': exp.value,
                    'reward': exp.reward
                }
                for exp in game.experiences
            ]
        }
        replay_path.write_text(json.dumps(replay_data, indent=2))
    
    async def maybe_update_ai(self):
        """Perform AI updates if we have enough experiences."""
        if len(self.replay_buffer) >= REPLAY_BATCH_SIZE:
            # Sample experiences, applying off-policy correction
            batch = random.sample(self.replay_buffer, REPLAY_BATCH_SIZE)
            
            # Sort by timestamp for consistency loss calculation
            batch.sort(key=lambda x: x.timestamp)
            
            # We'll implement the actual training step next
            # This will include:
            # 1. Policy and value losses
            # 2. Self-supervised consistency loss
            # 3. Value prefix loss
            # TODO: Implement training step
            
            # Save checkpoint periodically
            if self.training_stats['games_completed'] % CHECKPOINT_FREQUENCY == 0:
                self.save_checkpoint()
    
    def save_checkpoint(self):
        """Save model checkpoint and training stats."""
        checkpoint = {
            'model_state': self.ai.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'timestamp': datetime.datetime.now().isoformat()
        }
        checkpoint_path = (
            self.save_dir / "checkpoints" / 
            f"checkpoint_{self.training_stats['games_completed']}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Keep only last 5 checkpoints to save space
        checkpoint_files = sorted(list((self.save_dir / "checkpoints").glob("*.pt")))
        if len(checkpoint_files) > 5:
            checkpoint_files[0].unlink()
    
    async def cleanup(self):
        """Clean up resources and save final progress."""
        self.logger.info("Saving final checkpoint...")
        self.save_checkpoint()
        
        for game in self.games.values():
            if game.is_active:
                game.is_active = False
        
        self.logger.info("Training session complete. Final stats:")
        self.logger.info(json.dumps(self.training_stats, indent=2))