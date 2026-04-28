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


# Action sub-type for UNIT actors. Recruit / end_turn actors don't
# need this sub-decision -- their action type is implicit in the
# actor kind. The type head only fires for UNIT actors and chooses
# between attacking an enemy or moving to an empty hex (HOLD is not
# modeled in v1; see CLAUDE.md design discussion).
class UnitActionType:
    ATTACK = 0
    MOVE   = 1
    COUNT  = 2


def _compute_marginal_type_logits(
    actor_logits: torch.Tensor,
    type_logits:  torch.Tensor,
    U: int, R: int,
    device,
) -> torch.Tensor:
    """Aggregate per-leaf-action-type probabilities by marginalizing
    over the actor distribution. Returns log-probabilities (for
    consistency with the rest of the model output). Layout
    [1, T+2] = [ATTACK, MOVE, RECRUIT, END_TURN].

    Done WITHOUT the legality mask -- the sampler's `_masked_*`
    helpers compose the legality on top. This is a debug / coarse-
    prior signal, not an exact distribution.
    """
    # Softmax over actors (joint actor probability).
    actor_probs = torch.softmax(actor_logits, dim=-1)   # [1, A]
    # Softmax over type axis (per-actor type prior).
    type_probs = torch.softmax(type_logits, dim=-1)     # [1, A, T]
    # Layout-dependent slicing: actor index 0..U-1 = UNIT,
    # U..U+R-1 = RECRUIT, U+R = END_TURN.
    if U > 0:
        unit_actor_p = actor_probs[:, :U]                   # [1, U]
        unit_type_p  = type_probs[:, :U, :]                  # [1, U, T]
        # Sum P(actor) * P(type | actor) over UNIT actors.
        attack_mass = (unit_actor_p * unit_type_p[:, :, UnitActionType.ATTACK]).sum(dim=-1)
        move_mass   = (unit_actor_p * unit_type_p[:, :, UnitActionType.MOVE]).sum(dim=-1)
    else:
        attack_mass = torch.zeros(1, device=device)
        move_mass   = torch.zeros(1, device=device)
    if R > 0:
        recruit_mass = actor_probs[:, U:U + R].sum(dim=-1)   # [1]
    else:
        recruit_mass = torch.zeros(1, device=device)
    end_turn_mass = actor_probs[:, U + R]                    # [1]

    # Stack and convert to log-probs. Clamp the inner sum to avoid
    # log(0) when an entire actor-class has zero mass.
    masses = torch.stack([attack_mass, move_mass, recruit_mass, end_turn_mass],
                         dim=-1)                              # [1, 4]
    return torch.log(masses.clamp_min(1e-10))


@dataclass
class ModelOutput:
    """Everything the sampler needs from one forward pass.

    Shapes (batch dim = 1 in Phase 3.1):
      A = num_units + num_recruits + 1   — # of actor slots, incl. end_turn
      H = # hex tokens
      T = UnitActionType.COUNT (= 2: ATTACK, MOVE)
    """
    actor_logits:  torch.Tensor  # [1, A]            pick an actor
    actor_kind:    torch.Tensor  # [1, A] long       UNIT / RECRUIT / END_TURN
    type_logits:   torch.Tensor  # [1, A, T]         per-actor ATTACK/MOVE
                                 #                   prior. Meaningful only
                                 #                   for UNIT slots; sampler
                                 #                   ignores T-axis for
                                 #                   non-UNIT actors.
    target_logits: torch.Tensor  # [1, A, H]         pick a hex per actor
    weapon_logits: torch.Tensor  # [1, A, MAX_ATTACKS]
    value:         torch.Tensor  # [1, 1]
    num_units:     int
    num_recruits:  int

    # Diagnostic: marginal-over-actors probability of each action
    # type. Useful for "what % of decisions did the policy attack?"
    # logging and as a coarse PUCT prior at the type level.
    # Computed by the model after softmaxing actor_logits and
    # type_logits, summing P(actor) * P(type | actor) over UNIT
    # actors (with -inf masking elsewhere). Shape [1, T+2] where
    # the extra two indices capture aggregated RECRUIT and END_TURN
    # mass (so the sum across the four leaf-action-types equals 1
    # given a legal action distribution). Layout:
    #   0: ATTACK   (sum_unit_actors P(unit) * P(attack | unit))
    #   1: MOVE     (sum_unit_actors P(unit) * P(move   | unit))
    #   2: RECRUIT  (sum_recruit_actors P(recruit_slot))
    #   3: END_TURN (P(end_turn_slot))
    marginal_type_logits: torch.Tensor  # [1, T+2]


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
        # Per-actor sub-type head. Reads each actor token's
        # contextualized embedding and predicts P(ATTACK / MOVE | actor).
        # Only meaningful for UNIT slots; for RECRUIT and END_TURN
        # the sampler ignores it (their action type is fixed by
        # actor kind). Old checkpoints lack this Linear and
        # initialize fresh under load_checkpoint(strict=False).
        self.type_head      = nn.Linear(d_model, UnitActionType.COUNT)
        # Pointer-network projections: query from actor, key from hex.
        self.target_q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.target_k_proj  = nn.Linear(d_model, d_model, bias=False)
        self.weapon_head    = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, max_attacks),
        )
        # Value head: reads ONLY the global [CLS] token. Earlier
        # versions mean-pooled over all ~1750 token positions, which
        # diluted any global signal by 1700× and forced the network
        # to learn "ignore most of your input" via gradient pressure
        # alone. The global token is built specifically to carry
        # state-summary signal (turn number, gold, ToD), so reading
        # just it gives the value head the cleanest input.
        # Output is `tanh`-bounded to [-1, +1] so it stays well-scaled
        # against AlphaZero-style ±1 terminal returns regardless of
        # which shaping weights the reward function happens to have
        # this run. Trainer normalizes returns to the same range.
        self.value_head     = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Tanh(),
        )

    def forward(self, encoded: "EncodedState") -> ModelOutput:
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
        global_ctx   = x[:, H + U + R : H + U + R + 1]   # [1, 1, d]
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

        # Per-actor sub-type logits (ATTACK / MOVE). Computed for ALL
        # actor slots; sampler / reforward consult only the UNIT
        # slots' values (R and END_TURN slots have meaningless type
        # logits). Shape [1, A, T].
        type_logits = self.type_head(actor_ctx)

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

        # Read the contextualized global token (it has attended over
        # every other token via the encoder; carries the cleanest
        # global-state summary). Squeeze the seq dim so the head sees
        # [1, d].
        value = self.value_head(global_ctx.squeeze(1))  # [1, 1]

        # Diagnostic marginal: aggregated probability of each leaf-
        # action-type across the whole actor distribution. Useful for
        # logging "what % of decisions did the policy attack?" and
        # as a coarse PUCT prior at the leaf-type level.
        marginal_type_logits = _compute_marginal_type_logits(
            actor_logits, type_logits, U, R, device,
        )

        return ModelOutput(
            actor_logits=actor_logits,
            actor_kind=actor_kind,
            type_logits=type_logits,
            target_logits=target_logits,
            weapon_logits=weapon_logits,
            value=value,
            num_units=U,
            num_recruits=R,
            marginal_type_logits=marginal_type_logits,
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
        global_ctx_b   = x[:, H_max + U_max + R_max :
                              H_max + U_max + R_max + 1]              # [B, 1, d]
        end_turn_ctx_b = x[:, H_max + U_max + R_max + 1 :
                              H_max + U_max + R_max + 2]              # [B, 1, d]

        # Value head reads the contextualized global token (one per
        # sample). Same head as the single-sample path.
        value_b = self.value_head(global_ctx_b.squeeze(1))            # [B, 1]

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

        # Per-actor sub-type head, batched over actors.
        unit_type_b    = self.type_head(unit_ctx_b)                   # [B, U_max, T]
        recruit_type_b = self.type_head(recruit_ctx_b)                # [B, R_max, T]
        end_type_b     = self.type_head(end_turn_ctx_b)               # [B, 1, T]

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

            type_logits = torch.cat([
                unit_type_b[b:b+1, :U_b],
                recruit_type_b[b:b+1, :R_b],
                end_type_b[b:b+1],
            ], dim=1)  # [1, A_b, T]

            value_sample = value_b[b:b+1]  # [1, 1]

            marginal_type_logits = _compute_marginal_type_logits(
                actor_logits, type_logits, U_b, R_b, device,
            )

            outputs.append(ModelOutput(
                actor_logits=actor_logits,
                actor_kind=actor_kind,
                type_logits=type_logits,
                target_logits=target_logits,
                weapon_logits=weapon_logits,
                value=value_sample,
                num_units=U_b,
                num_recruits=R_b,
                marginal_type_logits=marginal_type_logits,
            ))
        return outputs
