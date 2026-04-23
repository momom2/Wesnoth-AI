# training.py
# Training logic for Wesnoth AI

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import random

from classes import Experience, Position
from constants import (
    REPLAY_BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    POLICY_LOSS_WEIGHT, VALUE_LOSS_WEIGHT, CONSISTENCY_LOSS_WEIGHT
)

class Trainer:
    """Handles model training."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        self.training_stats = {
            'updates': 0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'consistency_loss': 0.0,
            'total_loss': 0.0
        }
        
        # Build unit type to index mapping for recruit actions
        self.unit_type_to_idx = {}
        self.next_unit_idx = 0
    
    def get_unit_type_idx(self, unit_type: str) -> int:
        """Get or create index for unit type."""
        if unit_type not in self.unit_type_to_idx:
            self.unit_type_to_idx[unit_type] = self.next_unit_idx
            self.next_unit_idx += 1
        return self.unit_type_to_idx[unit_type]
    
    def compute_policy_loss(
        self,
        start_x_logits: torch.Tensor,
        start_y_logits: torch.Tensor,
        target_x_logits: torch.Tensor,
        target_y_logits: torch.Tensor,
        attack_logits: torch.Tensor,
        recruit_logits: torch.Tensor,
        end_turn_logit: torch.Tensor,
        actions: List[Dict]
    ) -> torch.Tensor:
        """Compute policy loss."""
        losses = []
        
        for i, action in enumerate(actions):
            action_type = action['type']
            
            if action_type == 'end_turn':
                # Binary classification for end_turn
                target = torch.tensor([1.0], device=end_turn_logit.device)
                loss = F.binary_cross_entropy_with_logits(
                    end_turn_logit[i:i+1],
                    target
                )
                losses.append(loss)
                
            else:
                # Not end turn - penalize end turn logit
                target = torch.tensor([0.0], device=end_turn_logit.device)
                loss_end = F.binary_cross_entropy_with_logits(
                    end_turn_logit[i:i+1],
                    target
                )
                
                if action_type in ['move', 'attack']:
                    # Coordinate loss (0-indexed)
                    start_pos = action['start_hex']
                    target_pos = action['target_hex']
                    
                    # Clamp coordinates to logit dimensions
                    start_x = min(start_pos.x, start_x_logits.size(1) - 1)
                    start_y = min(start_pos.y, start_y_logits.size(1) - 1)
                    target_x = min(target_pos.x, target_x_logits.size(1) - 1)
                    target_y = min(target_pos.y, target_y_logits.size(1) - 1)
                    
                    loss_sx = F.cross_entropy(
                        start_x_logits[i:i+1],
                        torch.tensor([start_x], device=start_x_logits.device)
                    )
                    loss_sy = F.cross_entropy(
                        start_y_logits[i:i+1],
                        torch.tensor([start_y], device=start_y_logits.device)
                    )
                    loss_tx = F.cross_entropy(
                        target_x_logits[i:i+1],
                        torch.tensor([target_x], device=target_x_logits.device)
                    )
                    loss_ty = F.cross_entropy(
                        target_y_logits[i:i+1],
                        torch.tensor([target_y], device=target_y_logits.device)
                    )
                    
                    coord_loss = loss_sx + loss_sy + loss_tx + loss_ty
                    
                    if action_type == 'attack':
                        # Attack weapon loss
                        weapon_idx = action.get('attack_index', 0)
                        weapon_idx = min(weapon_idx, attack_logits.size(1) - 1)
                        
                        loss_weapon = F.cross_entropy(
                            attack_logits[i:i+1],
                            torch.tensor([weapon_idx], device=attack_logits.device)
                        )
                        coord_loss += loss_weapon
                    
                    losses.append(loss_end + coord_loss)
                
                elif action_type == 'recruit':
                    # Recruit loss
                    unit_type = action.get('unit_type', '')
                    unit_idx = self.get_unit_type_idx(unit_type)
                    unit_idx = min(unit_idx, recruit_logits.size(1) - 1)
                    
                    loss_recruit = F.cross_entropy(
                        recruit_logits[i:i+1],
                        torch.tensor([unit_idx], device=recruit_logits.device)
                    )
                    
                    losses.append(loss_end + loss_recruit)
                else:
                    # Unknown action type, just penalize end_turn
                    losses.append(loss_end)
        
        if losses:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, requires_grad=True)
    
    def compute_value_loss(
        self,
        predicted_values: torch.Tensor,
        actual_rewards: torch.Tensor
    ) -> torch.Tensor:
        """Compute value prediction loss."""
        return F.mse_loss(predicted_values, actual_rewards)
    
    def compute_consistency_loss(
        self,
        batch: List[Experience],
        encoder,
        device: torch.device
    ) -> torch.Tensor:
        """
        Self-supervised consistency loss.
        States close in time should have similar value predictions.
        Fixed to allow gradient flow.
        """
        if len(batch) < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Group experiences by game
        game_experiences = {}
        for exp in batch:
            if exp.game_id not in game_experiences:
                game_experiences[exp.game_id] = []
            game_experiences[exp.game_id].append(exp)
        
        # Sort each game's experiences by turn/action number
        for game_id in game_experiences:
            game_experiences[game_id].sort(key=lambda x: (x.turn_number, x.action_number))
        
        # Compare adjacent experiences within each game
        losses = []
        gamma = 0.99
        
        for game_id, experiences in game_experiences.items():
            for i in range(len(experiences) - 1):
                exp1 = experiences[i]
                exp2 = experiences[i + 1]
                
                # Only compare if close in time
                if exp2.turn_number - exp1.turn_number > 2:
                    continue
                
                # Encode both states (with gradients enabled)
                map1, global1, fog1 = encoder.encode_game_state(exp1.state)
                map2, global2, fog2 = encoder.encode_game_state(exp2.state)
                
                # Move to device
                map1 = map1.unsqueeze(0).to(device)
                global1 = global1.unsqueeze(0).to(device)
                fog1 = fog1.unsqueeze(0).to(device)
                map2 = map2.unsqueeze(0).to(device)
                global2 = global2.unsqueeze(0).to(device)
                fog2 = fog2.unsqueeze(0).to(device)
                
                # Create dummy memory
                memory = torch.zeros(1, 128, 256, device=device)
                
                # Get values WITH gradients
                outputs1 = self.model(map1, global1, memory, fog1)
                value1 = outputs1[7]  # Value is the 8th output
                
                outputs2 = self.model(map2, global2, memory, fog2)
                value2 = outputs2[7]
                
                # Values should be similar (with discount)
                target_value2 = value1 * gamma
                loss = F.mse_loss(value2, target_value2.detach())
                losses.append(loss)
        
        if losses:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)
    
    def train_step(
        self,
        experiences: List[Experience],
        encoder
    ) -> dict:
        """Perform one training step."""
        
        device = next(self.model.parameters()).device
        
        # Sample batch
        if len(experiences) < REPLAY_BATCH_SIZE:
            batch = experiences
        else:
            batch = random.sample(experiences, REPLAY_BATCH_SIZE)
        
        # Filter out experiences without rewards yet
        batch = [exp for exp in batch if exp.reward is not None]
        if not batch:
            return self.training_stats
        
        # Prepare data
        states = []
        actions = []
        rewards = []
        
        for exp in batch:
            map_rep, global_feat, fog_mask = encoder.encode_game_state(exp.state)
            states.append((map_rep, global_feat, fog_mask))
            
            # Convert action dict to proper format
            action_dict = exp.action if isinstance(exp.action, dict) else {
                'type': 'end_turn'
            }
            actions.append(action_dict)
            rewards.append(exp.reward)
        
        # Create dummy memory (will be replaced with proper memory later)
        batch_size = len(batch)
        memory = torch.zeros(batch_size, 128, 256, device=device)
        
        # Stack inputs
        map_reps = torch.stack([s[0] for s in states]).to(device)
        global_feats = torch.stack([s[1] for s in states]).to(device)
        fog_masks = torch.stack([s[2] for s in states]).to(device)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        
        # Forward pass
        self.optimizer.zero_grad()
        
        (start_x_logits, start_y_logits, target_x_logits, target_y_logits,
         attack_logits, recruit_logits, end_turn_logit, values, new_memory) = self.model(
            map_reps, global_feats, memory, fog_masks
        )
        
        # Compute losses
        policy_loss = self.compute_policy_loss(
            start_x_logits, start_y_logits, target_x_logits, target_y_logits,
            attack_logits, recruit_logits, end_turn_logit, actions
        )
        
        value_loss = self.compute_value_loss(values, reward_tensor)
        
        consistency_loss = self.compute_consistency_loss(batch, encoder, device)
        
        # Total loss
        total_loss = (
            POLICY_LOSS_WEIGHT * policy_loss +
            VALUE_LOSS_WEIGHT * value_loss +
            CONSISTENCY_LOSS_WEIGHT * consistency_loss
        )
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Update stats
        self.training_stats['updates'] += 1
        self.training_stats['policy_loss'] = policy_loss.item()
        self.training_stats['value_loss'] = value_loss.item()
        self.training_stats['consistency_loss'] = consistency_loss.item()
        self.training_stats['total_loss'] = total_loss.item()
        
        return self.training_stats
