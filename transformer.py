# transformer.py
# Transformer model for Wesnoth AI

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from constants import (
    TRANSFORMER_D_MODEL, TRANSFORMER_NUM_LAYERS, TRANSFORMER_NUM_HEADS,
    TRANSFORMER_D_FF, TRANSFORMER_DROPOUT, TRANSFORMER_MEMORY_SIZE,
    MAX_MAP_WIDTH, MAX_MAP_HEIGHT, MAX_ATTACKS, MAX_RECRUITS
)

class MultiHeadAttention(nn.Module):
    """Multi-head attention with masking."""
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Project and reshape
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        
        # Combine heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)

class TransformerBlock(nn.Module):
    """Single transformer block."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attended = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attended))
        
        # Feed-forward
        fed_forward = self.ff(x)
        return self.norm2(x + self.dropout(fed_forward))

class MemoryModule(nn.Module):
    """Memory module for storing and retrieving game history."""
    
    def __init__(self, d_model: int, memory_size: int, num_heads: int, dropout: float):
        super().__init__()
        
        self.d_model = d_model
        self.memory_size = memory_size
        
        self.memory_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        self.update_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(
        self,
        query_state: torch.Tensor,
        memory_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_state: [batch, seq_len, d_model]
            memory_state: [batch, memory_size, d_model]
        
        Returns:
            conditioned_features: [batch, seq_len, d_model]
            new_memory: [batch, memory_size, d_model]
        """
        # Read from memory
        memory_output, _ = self.memory_attention(
            query=query_state,
            key=memory_state,
            value=memory_state,
            need_weights=False
        )
        conditioned = self.norm1(query_state + memory_output)
        
        # Update memory
        memory_updates, _ = self.memory_attention(
            query=memory_state,
            key=query_state,
            value=query_state,
            need_weights=False
        )
        
        # Gated update
        gate_input = torch.cat([memory_state, memory_updates], dim=-1)
        gate = self.update_gate(gate_input)
        new_memory = gate * memory_updates + (1 - gate) * memory_state
        new_memory = self.norm2(new_memory)
        
        return conditioned, new_memory

class WesnothTransformer(nn.Module):
    """Main transformer for Wesnoth AI."""
    
    def __init__(
        self,
        d_model: int = TRANSFORMER_D_MODEL,
        num_layers: int = TRANSFORMER_NUM_LAYERS,
        num_heads: int = TRANSFORMER_NUM_HEADS,
        d_ff: int = TRANSFORMER_D_FF,
        dropout: float = TRANSFORMER_DROPOUT,
        memory_size: int = TRANSFORMER_MEMORY_SIZE,
        max_width: int = MAX_MAP_WIDTH,
        max_height: int = MAX_MAP_HEIGHT
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_width = max_width
        self.max_height = max_height
        
        # Input projection (hex features -> d_model)
        # Hex feature size: 150 (from encodings.py)
        self.hex_projection = nn.Linear(150, d_model)
        self.global_projection = nn.Linear(7, d_model)
        
        # Memory module
        self.memory_module = MemoryModule(d_model, memory_size, num_heads, dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Pooling for global decisions
        self.pooling_head = nn.Linear(d_model, 1)
        
        # Output heads
        self.start_x_head = nn.Linear(d_model, max_width)
        self.start_y_head = nn.Linear(d_model, max_height)
        self.target_x_head = nn.Linear(d_model, max_width)
        self.target_y_head = nn.Linear(d_model, max_height)
        
        self.attack_head = nn.Linear(d_model, MAX_ATTACKS)
        self.recruit_head = nn.Linear(d_model, MAX_RECRUITS)
        self.end_turn_head = nn.Linear(d_model, 1)
        self.value_head = nn.Linear(d_model, 1)
        
    def forward(
        self,
        map_representation: torch.Tensor,
        global_features: torch.Tensor,
        memory: torch.Tensor,
        fog_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            map_representation: [batch, height, width, features]
            global_features: [batch, 7]
            memory: [batch, memory_size, d_model]
            fog_mask: [batch, height, width]
        
        Returns:
            start_x_logits, start_y_logits, target_x_logits, target_y_logits,
            attack_logits, recruit_logits, end_turn_logit, value, new_memory
        """
        batch_size, height, width, _ = map_representation.shape
        
        # Project inputs
        hex_features = self.hex_projection(map_representation)  # [batch, height, width, d_model]
        global_feat = self.global_projection(global_features)  # [batch, d_model]
        
        # Broadcast global features
        global_feat = global_feat.unsqueeze(1).unsqueeze(1).expand(-1, height, width, -1)
        combined = hex_features + global_feat
        
        # Reshape to sequence
        sequence = combined.view(batch_size, height * width, self.d_model)
        
        # Create attention mask from fog
        attention_mask = fog_mask.view(batch_size, height * width).unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.expand(-1, -1, height * width, -1)
        
        # Process with memory
        sequence, new_memory = self.memory_module(sequence, memory)
        
        # Process through transformer
        for layer in self.layers:
            sequence = layer(sequence, attention_mask)
        
        # Reshape back
        processed = sequence.view(batch_size, height, width, self.d_model)
        
        # Attention-pool over the hex grid to one global feature vector.
        # F.softmax doesn't accept tuple dims (only a single int), so
        # flatten the spatial axes, softmax across them, reshape back.
        scores = self.pooling_head(processed).squeeze(-1)      # [B, H, W]
        flat = scores.view(batch_size, height * width)
        pooling_weights = F.softmax(flat, dim=1).view(
            batch_size, height, width, 1)
        pooled = (processed * pooling_weights).sum(dim=(1, 2))  # [B, d_model]

        # All heads project from the pooled global vector. Output shapes:
        #   start_x/target_x: [B, max_width]      — padded beyond actual W
        #   start_y/target_y: [B, max_height]     — padded beyond actual H
        #   attack_logits:    [B, MAX_ATTACKS]
        #   recruit_logits:   [B, MAX_RECRUITS]
        #   end_turn_logit:   [B, 1]
        #   value:            [B, 1]
        # action_selector clamps coord heads to the actual map size, so
        # the padding positions don't need to be masked here.
        # TODO(phase-2): revisit architecture; per-hex coord heads would
        # be more informative than projecting from a single pooled vector.
        start_x_logits = self.start_x_head(pooled)
        start_y_logits = self.start_y_head(pooled)
        target_x_logits = self.target_x_head(pooled)
        target_y_logits = self.target_y_head(pooled)

        attack_logits   = self.attack_head(pooled)
        recruit_logits  = self.recruit_head(pooled)
        end_turn_logit  = self.end_turn_head(pooled)
        value           = self.value_head(pooled)
        
        return (
            start_x_logits,
            start_y_logits,
            target_x_logits,
            target_y_logits,
            attack_logits,
            recruit_logits,
            end_turn_logit,
            value,
            new_memory
        )
