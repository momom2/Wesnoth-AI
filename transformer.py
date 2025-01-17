# transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from assumptions import MAX_ATTACKS, MAX_RECRUITS

class MultiHeadAttention(nn.Module):
    """
    Custom multi-head attention that handles our hex-based spatial structure.
    This lets the model look at different aspects of the game state simultaneously,
    like one head focusing on combat opportunities while another watches for threats.
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Create learnable projections for queries, keys, and values
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
        
        # Create queries, keys, and values for all heads
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k))
        
        # Apply mask if provided (for fog of war or invalid actions)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Get attention weights and apply to values
        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        
        # Combine heads and project back to original dimension
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)

class TransformerBlock(nn.Module):
    """
    A single transformer block that processes the game state.
    Each block lets the model refine its understanding of the tactical situation.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network for processing attention outputs
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection and normalization
        attended = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attended))
        
        # Feed-forward with residual connection and normalization
        fed_forward = self.ff(x)
        return self.norm2(x + self.dropout(fed_forward))

class WesnothTransformer(nn.Module):
    """
    The main transformer architecture for our Wesnoth AI.
    Processes the game state and outputs action probabilities and value estimates.
    """
    def __init__(self,
                 d_model: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 d_ff: int = 1024,
                 dropout: float = 0.1,
                 max_memory_size: int = 128,
                 max_recruits: int = MAX_RECRUITS,
                 max_attacks: int = MAX_ATTACKS):
        super().__init__()
        
        # Embedding projections
        self.hex_projection = nn.Linear(101, d_model)  # Project hex features to model dimension
        self.global_projection = nn.Linear(7, d_model)  # Project global features
        self.memory_size = max_memory_size
        self.memory_projection = nn.Linear(max_memory_size, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output heads
        self.start_hex_head = nn.Linear(d_model, 1)  # Probability of selecting each hex
        self.target_hex_head = nn.Linear(d_model, 1)  # Probability of targeting each hex
        self.attack_head = nn.Linear(d_model, max_attacks)  # Probabilities for the attacks
        self.recruit_head = nn.Linear(d_model, max_recruits)  # Probabilities for recruitment options
        self.value_head = nn.Linear(d_model, 1)  # Position evaluation
        
    def forward(self, 
                map_representation: torch.Tensor,
                global_features: torch.Tensor,
                memory: torch.Tensor,
                fog_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, height, width, _ = map_representation.shape
        
        # Project all inputs to model dimension
        hex_features = self.hex_projection(map_representation)
        global_features = self.global_projection(global_features)
        memory_features = self.memory_projection(memory)
        
        # Combine features into sequence
        # Reshape hex features to sequence
        hex_seq = hex_features.view(batch_size, height * width, -1)
        
        # Add global and memory features to sequence
        sequence = torch.cat([
            hex_seq,
            global_features.unsqueeze(1),
            memory_features.unsqueeze(1)
        ], dim=1)
        
        # Process through transformer layers
        for layer in self.layers:
            sequence = layer(sequence, fog_mask)
        
        # Split processed features
        hex_features = sequence[:, :height * width].view(batch_size, height, width, -1)
        
        # Generate outputs
        start_logits = self.start_hex_head(hex_features).squeeze(-1)
        target_logits = self.target_hex_head(hex_features).squeeze(-1)
        attack_logits = self.attack_head(hex_features.mean(dim=[1, 2]))
        recruit_logits = self.recruit_head(hex_features.mean(dim=[1, 2]))
        value = self.value_head(hex_features.mean(dim=[1, 2]))
        
        return start_logits, target_logits, attack_logits, recruit_logits, value