"""Policy-and-value network for the Wesnoth AI.

Consumes an `EncodedState` (five token streams) and emits:

- **actor logits**: per possible actor token (unit, recruit, end_turn),
  a scalar "should I act as this".
- **target logits**: per actor × per hex, a pointer-network score —
  "given I'm acting as actor *a*, how much do I want to target hex *h*".
- **weapon logits**: per actor × per attack slot (MAX_ATTACKS=4).
  Meaningful only when actor is a unit and target contains an enemy.
  Not conditioned on target yet — a Phase 3.2 refinement.
- **value**: scalar state value (for policy-gradient baseline).

The sampler (action_sampler.py) consumes this to build an action dict.

Scale: Caves of the Basilisk is ~1700 hexes + ~30 units + ~14 recruits =
~1750 tokens. At d_model=128, 3 layers, 4 heads, one forward pass is
~50ms on the RX 6600 via DirectML (see memory/user_gpu_setup.md).

Phase 3.1 deliberately keeps this small. When training plateaus we
can grow d_model, add layers, add unit-attack/resistance features, and
condition the weapon head on the target. All changes localized here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from encoder import EncodedState


# How many attack slots per unit the weapon head predicts. Wesnoth
# units have 1–4 attacks typically; MAX_ATTACKS=4 covers them.
MAX_ATTACKS = 4


# Token-kind tags added to every stream so the transformer can tell
# "this is a hex" from "this is a recruit" during self-attention.
class TokenKind:
    HEX      = 0
    UNIT     = 1
    RECRUIT  = 2
    GLOBAL   = 3
    END_TURN = 4
    COUNT    = 5


# Mutually-exclusive categorization for ACTOR tokens. Sampler uses this
# to decide what to sample next (target hex? end the turn? weapon?).
class ActorKind:
    UNIT     = 0
    RECRUIT  = 1
    END_TURN = 2
    COUNT    = 3


@dataclass
class ModelOutput:
    """Everything the sampler needs from one forward pass.

    Shapes (batch dim = 1 in Phase 3.1):
      A = num_units + num_recruits + 1   — # of actor slots, incl. end_turn
      H = # hex tokens
    """
    actor_logits:  torch.Tensor  # [1, A]            pick an actor
    actor_kind:    torch.Tensor  # [1, A] long       UNIT / RECRUIT / END_TURN
    target_logits: torch.Tensor  # [1, A, H]         pick a hex per actor
    weapon_logits: torch.Tensor  # [1, A, MAX_ATTACKS]
    value:         torch.Tensor  # [1, 1]
    num_units:     int
    num_recruits:  int


class WesnothModel(nn.Module):
    """Transformer over all token streams + four heads."""

    def __init__(
        self,
        d_model:     int = 128,
        num_layers:  int = 3,
        num_heads:   int = 4,
        d_ff:        int = 256,
        dropout:     float = 0.0,
        max_attacks: int = MAX_ATTACKS,
    ):
        super().__init__()
        self.d_model     = d_model
        self.max_attacks = max_attacks

        # Distinguish streams at attention time.
        self.token_kind_embed = nn.Embedding(TokenKind.COUNT, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=d_ff, dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        # Heads.
        self.actor_head     = nn.Linear(d_model, 1)
        # Pointer-network projections: query from actor, key from hex.
        self.target_q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.target_k_proj  = nn.Linear(d_model, d_model, bias=False)
        self.weapon_head    = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, max_attacks),
        )
        self.value_head     = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, encoded: EncodedState) -> ModelOutput:
        device = encoded.hex_tokens.device
        d      = self.d_model

        H = encoded.hex_tokens.size(1)
        U = encoded.unit_tokens.size(1)
        R = encoded.recruit_tokens.size(1)

        # Apply token-kind embedding to each stream.
        def with_kind(tokens, kind):
            if tokens.size(1) == 0:
                return tokens
            kind_vec = self.token_kind_embed.weight[kind]  # [d]
            return tokens + kind_vec  # broadcast over seq & batch

        hex_in      = with_kind(encoded.hex_tokens,      TokenKind.HEX)
        unit_in     = with_kind(encoded.unit_tokens,     TokenKind.UNIT)
        recruit_in  = with_kind(encoded.recruit_tokens,  TokenKind.RECRUIT)
        global_in   = with_kind(encoded.global_token,    TokenKind.GLOBAL)
        end_turn_in = with_kind(encoded.end_turn_token,  TokenKind.END_TURN)

        # Concatenate. Order fixed: hex, unit, recruit, global, end_turn.
        x = torch.cat([hex_in, unit_in, recruit_in, global_in, end_turn_in], dim=1)
        x = self.encoder(x)  # [1, H+U+R+2, d]

        # Split contextualized embeddings back out.
        hex_ctx      = x[:, :H]
        unit_ctx     = x[:, H : H + U]
        recruit_ctx  = x[:, H + U : H + U + R]
        # global_ctx  = x[:, H + U + R : H + U + R + 1]   # unused currently
        end_turn_ctx = x[:, H + U + R + 1 : H + U + R + 2]

        # Actor-slot tokens, same order used by actor_kind.
        actor_ctx = torch.cat([unit_ctx, recruit_ctx, end_turn_ctx], dim=1)
        # Shape: [1, A, d] where A = U + R + 1.

        actor_kind = torch.tensor(
            [ActorKind.UNIT] * U
            + [ActorKind.RECRUIT] * R
            + [ActorKind.END_TURN],
            device=device, dtype=torch.long,
        ).unsqueeze(0)  # [1, A]

        actor_logits = self.actor_head(actor_ctx).squeeze(-1)  # [1, A]

        # Target logits: pointer attention from each actor to each hex.
        # For empty hex_ctx we emit a [1, A, 0] tensor gracefully.
        if H == 0:
            target_logits = torch.zeros(
                actor_ctx.size(0), actor_ctx.size(1), 0,
                device=device, dtype=actor_ctx.dtype,
            )
        else:
            q = self.target_q_proj(actor_ctx)      # [1, A, d]
            k = self.target_k_proj(hex_ctx)        # [1, H, d]
            target_logits = (q @ k.transpose(-1, -2)) / (d ** 0.5)  # [1, A, H]

        weapon_logits = self.weapon_head(actor_ctx)  # [1, A, max_attacks]

        value = self.value_head(x.mean(dim=1))  # [1, 1]

        return ModelOutput(
            actor_logits=actor_logits,
            actor_kind=actor_kind,
            target_logits=target_logits,
            weapon_logits=weapon_logits,
            value=value,
            num_units=U,
            num_recruits=R,
        )
