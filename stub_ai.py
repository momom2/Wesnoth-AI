# stub_ai.py
# Temporary stub AI that returns random actions
# This allows the training loop to run while transformer is being implemented

import torch
import torch.nn as nn
from typing import Tuple


class StubAI(nn.Module):
    """
    Stub AI that returns random actions.
    Replace with actual WesnothTransformer once implemented.
    """

    def __init__(self):
        super().__init__()
        # Dummy parameter so optimizer doesn't complain
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, ai_input, fog_mask=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns random logits for actions.

        Returns:
            start_logits: Shape (num_units,) - which unit to act with
            target_logits: Shape (num_hexes,) - where to move/attack
            attack_logits: Shape (max_attacks,) - which attack to use
            recruit_logits: Shape (num_recruits,) - which unit to recruit
            value: Shape (1,) - estimated win probability
        """
        # Get map dimensions
        num_hexes = len(ai_input.map.hexes) if ai_input.map.hexes else 100
        num_units = len(ai_input.map.units) if ai_input.map.units else 1
        num_recruits = len(ai_input.recruits) if ai_input.recruits else 3

        # Random logits (will be softmaxed later)
        start_logits = torch.randn(num_units)
        target_logits = torch.randn(num_hexes)
        attack_logits = torch.randn(4)  # Assume max 4 attacks
        recruit_logits = torch.randn(num_recruits)
        value = torch.randn(1) * 0.5  # Between -0.5 and 0.5

        return start_logits, target_logits, attack_logits, recruit_logits, value
