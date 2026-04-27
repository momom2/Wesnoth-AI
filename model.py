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
        # dropout=1e-4 (not 0) to disable PyTorch's TransformerEncoderLayer
        # "better-transformer" fast path. That path invokes the fused op
        # `_transformer_encoder_layer_fwd`, which is NOT implemented on
        # torch-directml — DML silently falls back to CPU and shuttles
        # every activation over PCI-e on each layer. Any dropout > 0 in
        # the layer's config gates the fast path off (see PyTorch
        # nn.TransformerEncoderLayer source); 1e-4 is effectively no
        # noise (expectation shift well under float32 precision) but
        # keeps every forward on the GPU.
        dropout:     float = 1e-4,
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
        # enable_nested_tensor=False is required for CPU throughput with
        # src_key_padding_mask: PyTorch's nested-tensor path is a prototype
        # that's ~2× SLOWER than dense on CPU in 1.18-era torch. Our
        # padding is small (most samples in a batch are similar sizes),
        # so eating a bit of wasted compute on pad positions is much
        # cheaper than going through the nested path.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False,
        )

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

    # ------------------------------------------------------------------
    # Batched forward — one transformer pass for many transitions at
    # once. Used by the trainer to amortize per-forward PyTorch overhead.
    # Rollout still goes through `forward` (single sample).
    # ------------------------------------------------------------------
    def forward_batch(self, encoded_list):
        """Run one padded transformer forward over B encoded states.

        Returns a list of B per-sample ModelOutput objects with the
        EXACT shapes the single-sample path produces, so downstream code
        (sampler, reforward_logprob_entropy, legality masking) needs no
        changes. The point of batching is to amortize Python-side
        per-op overhead across B samples; the transformer FLOP count is
        similar either way, but one big Linear/MHA invocation beats B
        small ones on CPU by ~2-4×.
        """
        B = len(encoded_list)
        if B == 0:
            return []
        if B == 1:
            return [self.forward(encoded_list[0])]

        d = self.d_model
        device = encoded_list[0].hex_tokens.device
        dtype  = encoded_list[0].hex_tokens.dtype

        Us = [e.unit_tokens.size(1)    for e in encoded_list]
        Rs = [e.recruit_tokens.size(1) for e in encoded_list]
        Hs = [e.hex_tokens.size(1)     for e in encoded_list]
        U_max = max(Us); R_max = max(Rs); H_max = max(Hs)

        def _pad(t, L_target, L_cur):
            if L_cur == L_target:
                return t
            if L_cur == 0:
                return torch.zeros(1, L_target, d, device=device, dtype=dtype)
            pad = torch.zeros(1, L_target - L_cur, d, device=device, dtype=dtype)
            return torch.cat([t, pad], dim=1)

        hex_pads     = [_pad(e.hex_tokens,     H_max, Hs[i]) for i, e in enumerate(encoded_list)]
        unit_pads    = [_pad(e.unit_tokens,    U_max, Us[i]) for i, e in enumerate(encoded_list)]
        recruit_pads = [_pad(e.recruit_tokens, R_max, Rs[i]) for i, e in enumerate(encoded_list)]

        hex_batch     = torch.cat(hex_pads,     dim=0)   # [B, H_max, d]
        unit_batch    = torch.cat(unit_pads,    dim=0)   # [B, U_max, d]
        recruit_batch = torch.cat(recruit_pads, dim=0)   # [B, R_max, d]
        global_batch  = torch.cat([e.global_token   for e in encoded_list], dim=0)  # [B, 1, d]
        end_turn_batch= torch.cat([e.end_turn_token for e in encoded_list], dim=0)  # [B, 1, d]

        # Token-kind additive embeddings.
        kk = self.token_kind_embed.weight
        if H_max > 0: hex_batch      = hex_batch     + kk[TokenKind.HEX]
        if U_max > 0: unit_batch     = unit_batch    + kk[TokenKind.UNIT]
        if R_max > 0: recruit_batch  = recruit_batch + kk[TokenKind.RECRUIT]
        global_batch   = global_batch   + kk[TokenKind.GLOBAL]
        end_turn_batch = end_turn_batch + kk[TokenKind.END_TURN]

        x = torch.cat([hex_batch, unit_batch, recruit_batch,
                       global_batch, end_turn_batch], dim=1)
        seq_len = x.size(1)

        # Key-padding mask: True at positions the attention should IGNORE.
        # For each sample, the pad slots in each block are marked True.
        # Global + end_turn are always real (1 token each).
        pad_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
        for b in range(B):
            if Hs[b] < H_max:
                pad_mask[b, Hs[b]:H_max] = True
            if Us[b] < U_max:
                pad_mask[b, H_max + Us[b]:H_max + U_max] = True
            if Rs[b] < R_max:
                pad_mask[b, H_max + U_max + Rs[b]:H_max + U_max + R_max] = True

        x = self.encoder(x, src_key_padding_mask=pad_mask)  # [B, seq_len, d]

        # Split the batched context back into blocks.
        hex_ctx_b      = x[:, :H_max]                                 # [B, H_max, d]
        unit_ctx_b     = x[:, H_max : H_max + U_max]                  # [B, U_max, d]
        recruit_ctx_b  = x[:, H_max + U_max : H_max + U_max + R_max]  # [B, R_max, d]
        end_turn_ctx_b = x[:, H_max + U_max + R_max + 1 :
                              H_max + U_max + R_max + 2]              # [B, 1, d]

        # Value head uses the mean over valid (non-pad) positions only.
        # Using the full mean would dilute every sample's value by its
        # own pad count, which is a different amount of dilution per
        # sample — breaks the value estimate across batch.
        valid = (~pad_mask).to(x.dtype).unsqueeze(-1)   # [B, seq_len, 1]
        x_sum   = (x * valid).sum(dim=1)                # [B, d]
        x_count = valid.sum(dim=1).clamp(min=1.0)       # [B, 1]
        x_mean  = x_sum / x_count                       # [B, d]
        value_b = self.value_head(x_mean)               # [B, 1]

        # Heads applied to the padded streams once each — replaces the
        # old per-sample loop that called actor_head / target_q_proj /
        # target_k_proj / weapon_head separately for every sample (4×B
        # small Linear launches). On GPU each Linear pays a fixed
        # ~30 µs launch overhead, so amortizing them with B=32 saves
        # ~4 ms / batch on the heads alone.
        unit_actor_b    = self.actor_head(unit_ctx_b).squeeze(-1)     # [B, U_max]
        recruit_actor_b = self.actor_head(recruit_ctx_b).squeeze(-1)  # [B, R_max]
        end_actor_b     = self.actor_head(end_turn_ctx_b).squeeze(-1) # [B, 1]

        unit_q_b    = self.target_q_proj(unit_ctx_b)                  # [B, U_max, d]
        recruit_q_b = self.target_q_proj(recruit_ctx_b)               # [B, R_max, d]
        end_q_b     = self.target_q_proj(end_turn_ctx_b)              # [B, 1, d]
        hex_k_b     = self.target_k_proj(hex_ctx_b)                   # [B, H_max, d]

        unit_weapon_b    = self.weapon_head(unit_ctx_b)               # [B, U_max, MAX_ATTACKS]
        recruit_weapon_b = self.weapon_head(recruit_ctx_b)            # [B, R_max, MAX_ATTACKS]
        end_weapon_b     = self.weapon_head(end_turn_ctx_b)           # [B, 1, MAX_ATTACKS]

        scale = d ** 0.5
        outputs = []
        for b in range(B):
            U_b, R_b, H_b = Us[b], Rs[b], Hs[b]

            # Per-sample shapes are produced by slicing the padded
            # heads down to the real (non-pad) positions and cat'ing
            # the three actor streams in canonical order:
            #   units (U_b) | recruits (R_b) | end_turn (1)
            # All operations here are view/index/cat — no fresh
            # heavyweight kernel launches.
            actor_logits = torch.cat([
                unit_actor_b[b:b+1, :U_b],
                recruit_actor_b[b:b+1, :R_b],
                end_actor_b[b:b+1],
            ], dim=1)  # [1, A_b]

            actor_kind = torch.tensor(
                [ActorKind.UNIT] * U_b
                + [ActorKind.RECRUIT] * R_b
                + [ActorKind.END_TURN],
                device=device, dtype=torch.long,
            ).unsqueeze(0)  # [1, A_b]

            if H_b == 0:
                A_b = U_b + R_b + 1
                target_logits = torch.zeros(
                    1, A_b, 0, device=device, dtype=dtype,
                )
            else:
                q_b = torch.cat([
                    unit_q_b[b:b+1, :U_b],
                    recruit_q_b[b:b+1, :R_b],
                    end_q_b[b:b+1],
                ], dim=1)  # [1, A_b, d]
                k_b = hex_k_b[b:b+1, :H_b]  # [1, H_b, d]
                target_logits = (q_b @ k_b.transpose(-1, -2)) / scale  # [1, A_b, H_b]

            weapon_logits = torch.cat([
                unit_weapon_b[b:b+1, :U_b],
                recruit_weapon_b[b:b+1, :R_b],
                end_weapon_b[b:b+1],
            ], dim=1)  # [1, A_b, MAX_ATTACKS]

            value_sample = value_b[b:b+1]  # [1, 1]

            outputs.append(ModelOutput(
                actor_logits=actor_logits,
                actor_kind=actor_kind,
                target_logits=target_logits,
                weapon_logits=weapon_logits,
                value=value_sample,
                num_units=U_b,
                num_recruits=R_b,
            ))
        return outputs
