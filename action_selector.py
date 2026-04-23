# action_selector.py
# Select actions from model output with proper validation

import torch
import torch.nn.functional as F
import random
from typing import Dict, List, Optional

from classes import GameState, Position, Unit

class ActionSelector:
    """Selects actions from transformer output."""
    
    def __init__(self):
        pass
    
    def select_action(
        self,
        start_x_logits: torch.Tensor,
        start_y_logits: torch.Tensor,
        target_x_logits: torch.Tensor,
        target_y_logits: torch.Tensor,
        attack_logits: torch.Tensor,
        recruit_logits: torch.Tensor,
        end_turn_logit: torch.Tensor,
        game_state: GameState,
        temperature: float = 1.0
    ) -> Dict:
        """
        Select action from model output with legal move filtering.
        
        NOTE: All coordinates are 0-indexed internally.
        
        Args:
            *_logits: Model outputs
            game_state: Current game state for validation
            temperature: Sampling temperature (1.0 = normal, lower = more deterministic)
        
        Returns:
            Action dictionary with 0-indexed coordinates
        """
        
        # Check end turn probability
        end_turn_prob = torch.sigmoid(end_turn_logit).item()
        if end_turn_prob > 0.7 or random.random() < end_turn_prob:
            return {'type': 'end_turn'}
        
        # Get map dimensions
        map_width = game_state.map.size_x
        map_height = game_state.map.size_y
        
        # Clamp logits to valid map dimensions
        start_x_logits = start_x_logits[:map_width]
        start_y_logits = start_y_logits[:map_height]
        target_x_logits = target_x_logits[:map_width]
        target_y_logits = target_y_logits[:map_height]
        
        # Apply temperature to logits
        start_x_logits = start_x_logits / temperature
        start_y_logits = start_y_logits / temperature
        target_x_logits = target_x_logits / temperature
        target_y_logits = target_y_logits / temperature
        
        # Sample coordinates
        start_x_probs = F.softmax(start_x_logits, dim=0)
        start_y_probs = F.softmax(start_y_logits, dim=0)
        
        start_x = torch.multinomial(start_x_probs, 1).item()
        start_y = torch.multinomial(start_y_probs, 1).item()
        
        # Validate coordinates are within bounds
        start_x = min(start_x, map_width - 1)
        start_y = min(start_y, map_height - 1)
        
        start_pos = Position(start_x, start_y)
        
        # Check what's at start position
        unit_at_start = self._get_unit_at_position(game_state, start_pos)
        
        if unit_at_start and unit_at_start.side == game_state.global_info.current_side:
            # We have our own unit - move or attack
            return self._select_unit_action(
                unit_at_start,
                target_x_logits,
                target_y_logits,
                attack_logits,
                game_state,
                map_width,
                map_height
            )
        else:
            # Try to recruit
            return self._select_recruit_action(
                start_pos,
                recruit_logits,
                game_state
            )
    
    def _get_unit_at_position(self, game_state: GameState, pos: Position) -> Optional[Unit]:
        """Get unit at position (0-indexed coordinates)."""
        for unit in game_state.map.units:
            if unit.position.x == pos.x and unit.position.y == pos.y:
                return unit
        return None
    
    def _select_unit_action(
        self,
        unit: Unit,
        target_x_logits: torch.Tensor,
        target_y_logits: torch.Tensor,
        attack_logits: torch.Tensor,
        game_state: GameState,
        map_width: int,
        map_height: int
    ) -> Dict:
        """Select action for a unit (move or attack)."""
        
        # Clamp target logits to map dimensions
        target_x_logits = target_x_logits[:map_width]
        target_y_logits = target_y_logits[:map_height]
        
        # Sample target coordinates
        target_x_probs = F.softmax(target_x_logits, dim=0)
        target_y_probs = F.softmax(target_y_logits, dim=0)
        
        target_x = torch.multinomial(target_x_probs, 1).item()
        target_y = torch.multinomial(target_y_probs, 1).item()
        
        # Validate coordinates
        target_x = min(target_x, map_width - 1)
        target_y = min(target_y, map_height - 1)
        
        target_pos = Position(target_x, target_y)
        
        # Check what's at target
        target_unit = self._get_unit_at_position(game_state, target_pos)
        
        if target_unit and target_unit.side != unit.side:
            # Enemy unit - attack
            weapon_probs = F.softmax(attack_logits, dim=0)
            weapon_idx = torch.multinomial(weapon_probs, 1).item()
            
            # Clamp to valid range
            weapon_idx = min(weapon_idx, len(unit.attacks) - 1) if unit.attacks else 0
            
            return {
                'type': 'attack',
                'start_hex': unit.position,
                'target_hex': target_pos,
                'attack_index': weapon_idx
            }
        else:
            # Empty or friendly - move
            return {
                'type': 'move',
                'start_hex': unit.position,
                'target_hex': target_pos
            }
    
    def _select_recruit_action(
        self,
        position: Position,
        recruit_logits: torch.Tensor,
        game_state: GameState
    ) -> Dict:
        """Select recruit action."""
        
        current_side = game_state.global_info.current_side
        recruits = game_state.sides[current_side - 1].recruits
        
        if not recruits:
            # No recruits available, end turn
            return {'type': 'end_turn'}
        
        # Sample recruit index
        recruit_probs = F.softmax(recruit_logits, dim=0)
        recruit_idx = torch.multinomial(recruit_probs, 1).item()
        
        # Clamp to valid range
        recruit_idx = min(recruit_idx, len(recruits) - 1)
        unit_type = recruits[recruit_idx]
        
        return {
            'type': 'recruit',
            'unit_type': unit_type,
            'target_hex': position
        }
