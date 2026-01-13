"""
Action Selection Logic - Converts transformer outputs to Wesnoth actions
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging


class ActionSelector:
    """Selects actions from transformer model outputs."""

    def __init__(self, temperature: float = 1.0, exploration_factor: float = 0.1):
        """
        Args:
            temperature: Temperature for sampling (higher = more random)
            exploration_factor: Probability of taking random action
        """
        self.temperature = temperature
        self.exploration_factor = exploration_factor
        self.logger = logging.getLogger("action_selector")

    def select_action(
        self,
        game_state: dict,
        start_logits: torch.Tensor,
        target_logits: torch.Tensor,
        attack_logits: torch.Tensor,
        recruit_logits: torch.Tensor,
        training: bool = False
    ) -> dict:
        """
        Select action from model outputs.

        Args:
            game_state: Current game state from Wesnoth
            start_logits: [batch, num_units] - which unit to move
            target_logits: [batch, map_height, map_width] - where to move
            attack_logits: [batch, 2] - attack vs move decision
            recruit_logits: [batch, num_recruit_types] - which unit to recruit
            training: If True, add exploration noise

        Returns:
            Action dictionary to send back to Wesnoth
        """
        # Extract units from game state
        my_side = game_state['side']
        all_units = game_state['map']['units']
        my_units = [u for u in all_units if u['side'] == my_side]
        enemy_units = [u for u in all_units if u['side'] != my_side]

        if not my_units:
            return self._end_turn_action()

        # Decide on action type: recruit, move, or attack
        action_type = self._select_action_type(
            game_state,
            recruit_logits,
            my_units
        )

        if action_type == 'recruit':
            return self._select_recruit_action(game_state, recruit_logits)
        elif action_type == 'move_or_attack':
            return self._select_unit_action(
                game_state,
                my_units,
                enemy_units,
                start_logits,
                target_logits,
                attack_logits,
                training
            )
        else:
            return self._end_turn_action()

    def _select_action_type(
        self,
        game_state: dict,
        recruit_logits: torch.Tensor,
        my_units: List[dict]
    ) -> str:
        """Decide whether to recruit or move units."""
        gold = game_state.get('gold', 0)
        recruits = game_state.get('recruits', [])

        # Can we afford to recruit?
        if not recruits:
            return 'move_or_attack'

        min_recruit_cost = min(r['cost'] for r in recruits)
        if gold < min_recruit_cost:
            return 'move_or_attack'

        # Do we have units that can still move?
        movable_units = [u for u in my_units if u['current_moves'] > 0]
        if not movable_units:
            # Try recruiting if we can't move anything
            return 'recruit'

        # Sample from recruit logits to decide
        recruit_probs = F.softmax(recruit_logits[0] / self.temperature, dim=0)

        # Simple heuristic: recruit if probability is high enough
        max_recruit_prob = recruit_probs.max().item()
        if max_recruit_prob > 0.7:  # High confidence in recruiting
            return 'recruit'

        return 'move_or_attack'

    def _select_recruit_action(
        self,
        game_state: dict,
        recruit_logits: torch.Tensor
    ) -> dict:
        """Select which unit type to recruit."""
        recruits = game_state['recruits']
        gold = game_state['gold']

        if not recruits:
            return self._end_turn_action()

        # Get probabilities for each recruit type
        recruit_probs = F.softmax(recruit_logits[0] / self.temperature, dim=0)

        # Filter affordable units
        affordable_indices = [
            i for i, r in enumerate(recruits)
            if i < len(recruit_probs) and r['cost'] <= gold
        ]

        if not affordable_indices:
            return self._end_turn_action()

        # Sample from affordable units
        affordable_probs = recruit_probs[affordable_indices]
        affordable_probs = affordable_probs / affordable_probs.sum()

        recruit_idx = affordable_indices[
            torch.multinomial(affordable_probs, 1).item()
        ]

        unit_type = recruits[recruit_idx]['name']

        self.logger.info(f"Selected recruit: {unit_type}")

        return {
            'type': 'action',
            'action_type': 'recruit',
            'unit_type': unit_type
        }

    def _select_unit_action(
        self,
        game_state: dict,
        my_units: List[dict],
        enemy_units: List[dict],
        start_logits: torch.Tensor,
        target_logits: torch.Tensor,
        attack_logits: torch.Tensor,
        training: bool
    ) -> dict:
        """Select a unit and its action (move or attack)."""
        # Filter units that can still act
        movable_units = [u for u in my_units if u['current_moves'] > 0]

        if not movable_units:
            return self._end_turn_action()

        # Get probabilities for each unit
        unit_probs = F.softmax(start_logits[0] / self.temperature, dim=0)

        # Only consider movable units
        movable_indices = []
        for i, unit in enumerate(my_units):
            if unit['current_moves'] > 0 and i < len(unit_probs):
                movable_indices.append(i)

        if not movable_indices:
            return self._end_turn_action()

        # Sample unit to move
        movable_probs = unit_probs[movable_indices]
        movable_probs = movable_probs / movable_probs.sum()

        selected_idx = movable_indices[
            torch.multinomial(movable_probs, 1).item()
        ]
        selected_unit = my_units[selected_idx]

        # Decide whether to attack or just move
        attack_probs = F.softmax(attack_logits[0] / self.temperature, dim=0)
        should_attack = attack_probs[1] > attack_probs[0]  # Index 1 = attack

        if should_attack and not selected_unit['has_attacked']:
            # Try to find an enemy to attack
            action = self._select_attack_target(
                selected_unit,
                enemy_units,
                target_logits
            )
            if action:
                return action

        # Move action
        return self._select_move_target(
            selected_unit,
            target_logits,
            game_state['map']
        )

    def _select_attack_target(
        self,
        attacker: dict,
        enemies: List[dict],
        target_logits: torch.Tensor
    ) -> Optional[dict]:
        """Select an enemy to attack."""
        if not enemies or not attacker['attacks']:
            return None

        # Get map dimensions from target_logits shape
        # target_logits shape is [batch, height, width]
        map_height = target_logits.shape[1]
        map_width = target_logits.shape[2]

        # Get target probabilities
        target_probs = F.softmax(
            target_logits[0].flatten() / self.temperature,
            dim=0
        )

        # Find enemies in attack range
        attacker_x, attacker_y = attacker['x'], attacker['y']
        attackable_enemies = []

        for enemy in enemies:
            # Check if adjacent (simplified - should use proper hex distance)
            dx = abs(enemy['x'] - attacker_x)
            dy = abs(enemy['y'] - attacker_y)
            if dx + dy <= 1:  # Adjacent
                attackable_enemies.append(enemy)

        if not attackable_enemies:
            return None

        # Pick highest probability enemy
        best_enemy = None
        best_prob = -1

        for enemy in attackable_enemies:
            # Convert position to target logits index using actual map width
            # Ensure coordinates are within bounds
            if 0 <= enemy['y'] < map_height and 0 <= enemy['x'] < map_width:
                idx = enemy['y'] * map_width + enemy['x']
                prob = target_probs[idx].item()
                if prob > best_prob:
                    best_prob = prob
                    best_enemy = enemy

        if best_enemy:
            self.logger.info(
                f"Selected attack: {attacker.get('id', 'unknown')} -> "
                f"enemy at ({best_enemy['x']}, {best_enemy['y']})"
            )

            return {
                'type': 'action',
                'action_type': 'attack',
                'attacker_id': attacker.get('id'),
                'target_x': best_enemy['x'],
                'target_y': best_enemy['y'],
                'weapon_index': 0  # Use first attack
            }

        return None

    def _select_move_target(
        self,
        unit: dict,
        target_logits: torch.Tensor,
        game_map: dict
    ) -> dict:
        """Select where to move the unit."""
        # Get target probabilities
        map_height = game_map['height']
        map_width = game_map['width']

        target_probs = F.softmax(
            target_logits[0, :map_height, :map_width].flatten() / self.temperature,
            dim=0
        )

        # Sample target position
        target_idx = torch.multinomial(target_probs, 1).item()
        target_y = target_idx // map_width
        target_x = target_idx % map_width

        # Simple validity check: not too far from current position
        # In reality, should check movement points and pathfinding
        unit_x, unit_y = unit['x'], unit['y']
        max_dist = unit['current_moves']

        # If sampled position is too far, pick something closer
        dist = abs(target_x - unit_x) + abs(target_y - unit_y)
        if dist > max_dist:
            # Just move toward the target
            if target_x > unit_x:
                target_x = min(unit_x + max_dist, target_x)
            elif target_x < unit_x:
                target_x = max(unit_x - max_dist, target_x)

            if target_y > unit_y:
                target_y = min(unit_y + max_dist, target_y)
            elif target_y < unit_y:
                target_y = max(unit_y - max_dist, target_y)

        self.logger.info(
            f"Selected move: {unit.get('id', 'unknown')} "
            f"({unit_x},{unit_y}) -> ({target_x},{target_y})"
        )

        return {
            'type': 'action',
            'action_type': 'move',
            'unit_id': unit.get('id'),
            'target_x': int(target_x),
            'target_y': int(target_y)
        }

    def _end_turn_action(self) -> dict:
        """Return an end turn action."""
        self.logger.info("Selected: end turn")
        return {
            'type': 'action',
            'action_type': 'end_turn'
        }
