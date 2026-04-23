# game_manager.py
# Main training manager coordinating multiple Wesnoth games
# FIXED: Updated to handle WML format instead of JSON, improved error handling

import asyncio
import logging
import time
import torch
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from wesnoth_interface import WesnothGame
from state_converter import StateConverter
from classes import GameState, Experience
from state_encodings import GameStateEncoder
from transformer import WesnothTransformer
from action_selector import ActionSelector
from training import Trainer
from constants import (
    NUM_PARALLEL_GAMES, MAX_ACTIONS_PER_GAME, SCENARIOS_PATH,
    CHECKPOINTS_PATH, REPLAYS_PATH, LOGS_PATH, ACTION_PENALTY,
    TIMEOUT_PENALTY, CHECKPOINT_FREQUENCY, LOG_FREQUENCY,
    REPLAY_BUFFER_SIZE, TRANSFORMER_MEMORY_SIZE
)

class GameManager:
    """Manages multiple concurrent Wesnoth games for training."""
    
    def __init__(self, num_games: int = NUM_PARALLEL_GAMES):
        self.num_games = num_games
        self.games: Dict[str, WesnothGame] = {}
        self.converters: Dict[str, StateConverter] = {}
        
        # Create shared state converter for consistent unit type mapping
        self.global_converter = StateConverter()
        
        self.logger = logging.getLogger("game_manager")
        self._setup_logging()
        
        # Initialize model components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        self.encoder = GameStateEncoder().to(self.device)
        self.model = WesnothTransformer().to(self.device)
        self.action_selector = ActionSelector()
        self.trainer = Trainer(self.model)
        
        # Memory states for each game
        self.game_memories: Dict[str, torch.Tensor] = {}
        
        # Training statistics
        self.stats = {
            'games_completed': 0,
            'total_actions': 0,
            'wins_side1': 0,
            'wins_side2': 0,
            'draws': 0,
            'timeouts': 0,
            'state_conversion_errors': 0,
            'action_send_errors': 0
        }
        
        # Experience replay buffer
        self.replay_buffer: List[Experience] = []
        
        # Per-game experience tracking
        self.game_experiences: Dict[str, List[Experience]] = {}
        
        # Create directories
        CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)
        REPLAYS_PATH.mkdir(parents=True, exist_ok=True)
        LOGS_PATH.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Configure logging."""
        log_file = LOGS_PATH / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    async def create_game(self, game_id: str) -> WesnothGame:
        """Create and initialize a new game."""
        self.logger.info(f"Creating game {game_id}")
        
        # Create game instance
        scenario_path = SCENARIOS_PATH / "training_scenario.cfg"
        game = WesnothGame(game_id, scenario_path)
        
        # Use global converter for consistent unit type mapping
        self.converters[game_id] = self.global_converter
        
        # Initialize memory (batch_size=1, memory_size, d_model)
        self.game_memories[game_id] = torch.zeros(
            1, TRANSFORMER_MEMORY_SIZE, 256, device=self.device
        )
        
        # Initialize experience list for this game
        self.game_experiences[game_id] = []
        
        # Start Wesnoth process
        game.start_wesnoth()
        
        return game
    
    def get_ai_action(self, game_state: GameState, game_id: str) -> Dict:
        """Get AI action from the model."""
        try:
            # Encode game state
            map_rep, global_feat, fog_mask = self.encoder.encode_game_state(game_state)
            
            # Move to device
            map_rep = map_rep.unsqueeze(0).to(self.device)
            global_feat = global_feat.unsqueeze(0).to(self.device)
            fog_mask = fog_mask.unsqueeze(0).to(self.device)
            
            # Get memory for this game
            memory = self.game_memories[game_id]
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(map_rep, global_feat, memory, fog_mask)
            
            # Unpack outputs
            (start_x_logits, start_y_logits, target_x_logits, target_y_logits,
             attack_logits, recruit_logits, end_turn_logit, value, new_memory) = outputs
            
            # Update memory
            self.game_memories[game_id] = new_memory.detach()
            
            # Select action
            action = self.action_selector.select_action(
                start_x_logits[0], start_y_logits[0],
                target_x_logits[0], target_y_logits[0],
                attack_logits[0], recruit_logits[0],
                end_turn_logit[0],
                game_state,
                temperature=1.0
            )
            
            # Store experience (reward will be assigned later)
            experience = Experience(
                game_id=game_id,
                state=game_state,
                action=action,
                value=value[0].item(),
                reward=None,  # Will be assigned at game end
                turn_number=game_state.global_info.turn_number,
                action_number=len(self.game_experiences[game_id])
            )
            self.game_experiences[game_id].append(experience)
            
            return action
            
        except Exception as e:
            self.logger.error(f"Error in get_ai_action: {e}", exc_info=True)
            # Return safe default action
            return {'type': 'end_turn'}
    
    async def handle_game_turn(self, game: WesnothGame) -> bool:
        """Handle one turn of a game. Returns False if game should end."""
        game_id = game.game_id
        
        try:
            # Read state from Wesnoth (now returns WML string)
            state_wml = game.read_state()
            if not state_wml:
                self.logger.warning(f"Game {game_id}: No state received, ending game")
                return False
            
            # Convert WML to internal format
            try:
                game_state = self.converters[game_id].convert_wml_to_game_state(state_wml)
            except Exception as e:
                self.logger.error(f"Game {game_id}: Error converting WML state: {e}", exc_info=True)
                self.stats['state_conversion_errors'] += 1
                
                # Log the problematic WML for debugging
                self.logger.debug(f"Problematic WML (first 500 chars): {state_wml[:500]}")
                
                # Try to continue or end game gracefully
                return False
            
            # Check if game is over
            if game_state.game_over:
                self.logger.info(f"Game {game_id}: Game over, winner = {game_state.winner}")
                self._finalize_game_experiences(game_id, game_state.winner)
                self._record_game_result(game, game_state.winner)
                return False
            
            # Get AI decision
            action = self.get_ai_action(game_state, game_id)
            
            # Convert action to format for Lua file (already in correct format from action_selector)
            action_for_wesnoth = action
            
            # Send action to Wesnoth
            success = game.send_action(action_for_wesnoth)
            if not success:
                self.logger.warning(f"Game {game_id}: Failed to send action")
                self.stats['action_send_errors'] += 1
                return False
            
            self.stats['total_actions'] += 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"Game {game_id}: Error in handle_game_turn: {e}", exc_info=True)
            return False
    
    def _finalize_game_experiences(self, game_id: str, winner: Optional[int]):
        """Assign rewards to all experiences for a completed game."""
        experiences = self.game_experiences.get(game_id, [])
        if not experiences:
            return
        
        # Determine base reward based on outcome
        for exp in experiences:
            current_side = exp.state.global_info.current_side
            
            if winner is None:
                # Draw/timeout
                base_reward = -TIMEOUT_PENALTY
            elif winner == current_side:
                # Win
                base_reward = 1.0
            else:
                # Loss
                base_reward = -1.0
            
            # Apply temporal discounting and action penalty
            gamma = 0.99
            num_actions = len(experiences)
            action_idx = exp.action_number
            
            # Discount from end of game
            discount = gamma ** (num_actions - action_idx - 1)
            
            # Action penalty
            penalty = ACTION_PENALTY * action_idx
            
            exp.reward = base_reward * discount - penalty
        
        # Add to replay buffer
        self.replay_buffer.extend(experiences)
        
        # Limit buffer size
        if len(self.replay_buffer) > REPLAY_BUFFER_SIZE:
            self.replay_buffer = self.replay_buffer[-REPLAY_BUFFER_SIZE:]
        
        self.logger.info(f"Game {game_id}: Finalized {len(experiences)} experiences. "
                        f"Replay buffer size: {len(self.replay_buffer)}")
    
    async def run_game(self, game_id: str):
        """Run a single game to completion."""
        try:
            game = await self.create_game(game_id)
            self.games[game_id] = game
            
            self.logger.info(f"Game {game_id}: Started")
            
            # Game loop
            action_count = 0
            while action_count < MAX_ACTIONS_PER_GAME:
                should_continue = await self.handle_game_turn(game)
                if not should_continue:
                    break
                
                action_count += 1
                
                # Check for game over signal from Wesnoth
                if game.check_game_over():
                    winner = game.winner
                    self._finalize_game_experiences(game_id, winner)
                    self._record_game_result(game, winner)
                    break
            
            if action_count >= MAX_ACTIONS_PER_GAME:
                self.logger.warning(f"Game {game_id}: Timeout after {action_count} actions")
                self._finalize_game_experiences(game_id, None)
                self.stats['timeouts'] += 1
            
        except Exception as e:
            self.logger.error(f"Game {game_id}: Error: {e}", exc_info=True)
        finally:
            # Clean up
            if game_id in self.games:
                self.games[game_id].terminate()
                del self.games[game_id]
            
            if game_id in self.game_memories:
                del self.game_memories[game_id]
            
            if game_id in self.game_experiences:
                del self.game_experiences[game_id]
            
            self.logger.info(f"Game {game_id}: Finished")
    
    def _record_game_result(self, game: WesnothGame, winner: Optional[int]):
        """Record game result in statistics."""
        self.stats['games_completed'] += 1
        
        if winner == 1:
            self.stats['wins_side1'] += 1
        elif winner == 2:
            self.stats['wins_side2'] += 1
        else:
            self.stats['draws'] += 1
        
        # Log statistics
        if self.stats['games_completed'] % LOG_FREQUENCY == 0:
            self._log_training_stats()
    
    def _log_training_stats(self):
        """Log comprehensive training statistics."""
        total_games = self.stats['games_completed']
        if total_games == 0:
            return
        
        win_rate_s1 = (self.stats['wins_side1'] / total_games) * 100
        win_rate_s2 = (self.stats['wins_side2'] / total_games) * 100
        
        self.logger.info("=" * 70)
        self.logger.info("=== Training Statistics ===")
        self.logger.info(f"Games completed: {total_games}")
        self.logger.info(f"Win rate (Side 1): {win_rate_s1:.2f}%")
        self.logger.info(f"Win rate (Side 2): {win_rate_s2:.2f}%")
        self.logger.info(f"Draws/Timeouts: {self.stats['draws'] + self.stats['timeouts']}")
        self.logger.info(f"Total actions: {self.stats['total_actions']}")
        self.logger.info(f"State conversion errors: {self.stats['state_conversion_errors']}")
        self.logger.info(f"Action send errors: {self.stats['action_send_errors']}")
        self.logger.info(f"Replay buffer size: {len(self.replay_buffer)}")
        self.logger.info(f"Training updates: {self.trainer.training_stats['updates']}")
        
        if self.trainer.training_stats['updates'] > 0:
            self.logger.info(f"Recent losses:")
            self.logger.info(f"  Policy: {self.trainer.training_stats['policy_loss']:.4f}")
            self.logger.info(f"  Value: {self.trainer.training_stats['value_loss']:.4f}")
            self.logger.info(f"  Total: {self.trainer.training_stats['total_loss']:.4f}")
        
        self.logger.info("=" * 70)
    
    def _save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint_path = CHECKPOINTS_PATH / f"checkpoint_{self.stats['games_completed']}.pt"
        
        torch.save({
            'games_completed': self.stats['games_completed'],
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'training_stats': self.trainer.training_stats,
            'game_stats': self.stats,
            'unit_type_mapping': self.global_converter.unit_type_to_id
        }, checkpoint_path)
        
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    async def run_training(self):
        """Main training loop."""
        self.logger.info(f"Starting training with {self.num_games} parallel games")
        
        try:
            while True:
                # Create batch of games
                tasks = []
                for i in range(self.num_games):
                    game_id = f"game_{self.stats['games_completed'] + i}"
                    tasks.append(self.run_game(game_id))
                
                # Run games in parallel
                await asyncio.gather(*tasks)
                
                # Train model if we have enough experiences
                if len(self.replay_buffer) >= 100:
                    self.logger.info("Training model...")
                    try:
                        training_stats = self.trainer.train_step(
                            self.replay_buffer,
                            self.encoder
                        )
                        self.logger.info(f"Training: {training_stats}")
                    except Exception as e:
                        self.logger.error(f"Error during training step: {e}", exc_info=True)
                
                # Save checkpoint periodically
                if self.stats['games_completed'] % CHECKPOINT_FREQUENCY == 0:
                    self._save_checkpoint()
                
                self.logger.info(f"Completed batch. Total games: {self.stats['games_completed']}")
                
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            self._save_checkpoint()
        finally:
            # Clean up all games
            for game in self.games.values():
                game.terminate()

async def main():
    """Entry point for training."""
    manager = GameManager(num_games=NUM_PARALLEL_GAMES)
    await manager.run_training()

if __name__ == "__main__":
    asyncio.run(main())
