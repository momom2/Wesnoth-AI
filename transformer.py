# transformer.py
# Complete transformer-based AI model for Wesnoth

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from assumptions import MEMORY_STATE_SIZE, MAX_ATTACKS, MAX_RECRUITS


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)

        # Linear projections and reshape for multi-head
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, V)

        # Concatenate heads and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)

        return output


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feed-forward."""

    def __init__(self, d_model: int, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection and normalization
        attended = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attended))

        # Feed-forward with residual connection and normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class WesnothTransformer(nn.Module):
    """
    Transformer-based AI for Wesnoth.

    Architecture:
    1. Encode map hexes, units, and global features
    2. Process through transformer layers
    3. Decode to action logits and value estimate
    """

    def __init__(self,
                 d_model: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 d_ff: int = 1024,
                 dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model

        # Input encoders
        self.hex_encoder = nn.Sequential(
            nn.Linear(3, 64),  # terrain_type + 2 modifiers (simplified)
            nn.ReLU(),
            nn.Linear(64, d_model)
        )

        self.unit_encoder = nn.Sequential(
            nn.Linear(129, 128),  # From UNIT_ENCODING_DIM
            nn.ReLU(),
            nn.Linear(128, d_model)
        )

        self.global_encoder = nn.Sequential(
            nn.Linear(10, 64),  # gold, turn, side, etc.
            nn.ReLU(),
            nn.Linear(64, d_model)
        )

        self.memory_encoder = nn.Linear(MEMORY_STATE_SIZE, d_model)

        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model) * 0.02)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Action heads
        self.unit_selector = nn.Linear(d_model, 1)  # Select which unit to act with
        self.target_selector = nn.Linear(d_model, 1)  # Select target hex
        self.attack_head = nn.Linear(d_model, MAX_ATTACKS)  # Which attack
        self.recruit_head = nn.Linear(d_model, MAX_RECRUITS)  # Which unit to recruit

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()  # Value between -1 and 1
        )

    def forward(self, ai_input, fog_mask: Optional[torch.Tensor] = None):
        """
        Forward pass through the model.

        Args:
            ai_input: Input dataclass with map, recruits, memory
            fog_mask: Optional mask for fogged hexes

        Returns:
            start_logits: Logits for unit selection
            target_logits: Logits for target hex selection
            attack_logits: Logits for attack selection
            recruit_logits: Logits for recruit selection
            value: Estimated win probability
        """
        # Extract components
        game_map = ai_input.map
        recruits = ai_input.recruits
        memory = ai_input.memory

        # Encode hexes (simplified - using first terrain type only)
        hex_features = []
        for hex in game_map.hexes:
            # Simplified: one-hot first terrain type + binary modifiers
            terrain_id = list(hex.terrain_types)[0].value if hex.terrain_types else 0
            has_shadow = any(m.value == 2 for m in hex.modifiers) if hex.modifiers else 0
            has_illuminate = any(m.value == 3 for m in hex.modifiers) if hex.modifiers else 0
            hex_features.append([float(terrain_id), float(has_shadow), float(has_illuminate)])

        if not hex_features:
            hex_features = [[0.0, 0.0, 0.0]]  # Dummy hex

        hex_tensor = torch.tensor(hex_features, dtype=torch.float32).unsqueeze(0)  # [1, num_hexes, 3]
        hex_encoded = self.hex_encoder(hex_tensor)  # [1, num_hexes, d_model]

        # Encode units (simplified - using available features)
        unit_features = []
        for unit in game_map.units:
            # Simplified encoding: just basic stats
            feat = [
                float(unit.side),
                float(unit.is_leader),
                unit.current_hp / max(unit.max_hp, 1),
                unit.current_moves / max(unit.max_moves, 1),
                unit.current_exp / max(unit.max_exp, 1),
                float(unit.alignment.value),
            ] + [0.0] * 123  # Pad to 129
            unit_features.append(feat[:129])

        if not unit_features:
            unit_features = [[0.0] * 129]  # Dummy unit

        unit_tensor = torch.tensor(unit_features, dtype=torch.float32).unsqueeze(0)  # [1, num_units, 129]
        unit_encoded = self.unit_encoder(unit_tensor)  # [1, num_units, d_model]

        # Encode global features
        global_feat = [
            0.0,  # gold (simplified)
            0.0,  # turn
            1.0,  # side
        ] + [0.0] * 7  # Pad to 10
        global_tensor = torch.tensor([global_feat], dtype=torch.float32)  # [1, 10]
        global_encoded = self.global_encoder(global_tensor).unsqueeze(1)  # [1, 1, d_model]

        # Encode memory
        memory_tensor = torch.tensor([memory.state], dtype=torch.float32)  # [1, MEMORY_STATE_SIZE]
        memory_encoded = self.memory_encoder(memory_tensor).unsqueeze(1)  # [1, 1, d_model]

        # Concatenate all tokens
        tokens = torch.cat([global_encoded, memory_encoded, hex_encoded, unit_encoded], dim=1)

        # Add positional encoding
        seq_len = tokens.size(1)
        tokens = tokens + self.pos_encoding[:, :seq_len, :]

        # Process through transformer
        for layer in self.transformer_layers:
            tokens = layer(tokens, mask=fog_mask)

        # Extract different token types
        num_global = 2
        num_hexes = hex_encoded.size(1)
        num_units = unit_encoded.size(1)

        global_tokens = tokens[:, :num_global, :]
        hex_tokens = tokens[:, num_global:num_global+num_hexes, :]
        unit_tokens = tokens[:, num_global+num_hexes:, :]

        # Generate action logits
        # Unit selection: which unit to move/attack with
        unit_logits = self.unit_selector(unit_tokens).squeeze(-1)  # [1, num_units]

        # Target hex selection: where to move/attack
        target_logits = self.target_selector(hex_tokens).squeeze(-1)  # [1, num_hexes]

        # Attack selection: which attack to use
        pooled = torch.mean(global_tokens, dim=1)  # [1, d_model]
        attack_logits = self.attack_head(pooled)  # [1, MAX_ATTACKS]

        # Recruit selection: which unit to recruit
        recruit_logits = self.recruit_head(pooled)  # [1, MAX_RECRUITS]

        # Value estimate
        value = self.value_head(pooled)  # [1, 1]

        # Remove batch dimension and return
        return (
            unit_logits.squeeze(0),      # [num_units]
            target_logits.squeeze(0),    # [num_hexes]
            attack_logits.squeeze(0),    # [MAX_ATTACKS]
            recruit_logits.squeeze(0),   # [MAX_RECRUITS]
            value.squeeze(0)             # [1]
        )
