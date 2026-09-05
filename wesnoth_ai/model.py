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
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wesnoth_ai.encoder import EncodedState
from wesnoth_ai.packed_trunk import (
    CompiledPackedTrunk, EmbeddedStreams, PackedTrunkWeights, build_packed_layout,
    check_packed_trunk_supported, flash_varlen_applies, packed_trunk, padded_gather_index,
)


# How many attack slots per unit the weapon head predicts. Wesnoth
# units have 1–4 attacks typically; MAX_ATTACKS=4 covers them.
MAX_ATTACKS = 4

# Distributional value head (C51 / categorical-K). The head emits a
# softmax over K bins on a fixed support [V_MIN, V_MAX]; the scalar
# value the rest of the codebase reads is the distribution's mean,
# and `cliffness = std(Z(s))` is the same distribution's spread.
# Both come "for free" from one head with one categorical-CE loss
# (training is implicit: noisy returns → wider distribution →
# higher std). See trainer.py::_project_returns_to_atoms for the
# loss-side projection.
#
# The support is fixed to match the trainer's existing `value_clip`
# range. If you raise value_clip above 1.0, widen V_MIN/V_MAX
# accordingly or returns clip to the support edges and lose
# resolution.
#
# K=51 is the Bellemare et al. (2017) default. With our [-1, +1]
# range that's 0.04 / bin — plenty of resolution.
VALUE_N_ATOMS = 51
VALUE_V_MIN   = -1.0
VALUE_V_MAX   = +1.0

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
    value:         torch.Tensor  # [1, 1]            mean of value distribution
    # Distributional value head outputs (categorical over K atoms).
    # Trainer's value loss reads `value_logits`; rollout / MCTS read
    # `value` (the mean). `cliffness` is the std of the predicted
    # distribution at this state -- a heteroscedastic uncertainty
    # estimate that comes for free from the categorical head and
    # marks states where small perturbations imply big value swings.
    value_logits:  torch.Tensor  # [1, K]            raw softmax logits
    cliffness:     torch.Tensor  # [1, 1]            std(Z(s))
    num_units:     int
    num_recruits:  int
    # Optional auxiliary prediction head (KataGo §3.5): the predicted
    # final MATERIAL margin from the acting side's perspective, tanh-
    # bounded to (-1, +1). `None` when the model was built without the
    # aux head (default). A denser training signal than win/loss z;
    # see trainer aux_coef + draw_tiebreak.material_margin.
    aux_score:     Optional[torch.Tensor] = None  # [1, 1] or None
    # Optional moves-left head (Lc0-style, 2026-07-04): predicted
    # FRACTION of the turn budget still to be played from this state,
    # sigmoid-bounded to (0, 1) (fraction of MOVES_LEFT_NORM_TURNS).
    # Trains as a dense regression alongside z; intended search-side
    # consumer (prefer shorter wins / longer losses) is wired
    # SEPARATELY and default-off pending calibration. `None` when the
    # model was built without the head (default).
    moves_left:    Optional[torch.Tensor] = None  # [1, 1] or None

    # GBC event-supervision tap (2026-08-14, docs/archive/gbc_spec.md):
    # contextualized token slices, populated ONLY when the model was
    # built with `gbc=True` and only on the single-state forward
    # path (the trainer's loss path). References to tensors already
    # computed in the forward -- zero extra compute; None by default
    # so eval/search paths pay nothing.
    unit_ctx:      Optional[torch.Tensor] = None  # [1, U, d] or None
    hex_ctx:       Optional[torch.Tensor] = None  # [1, H, d] or None
    global_ctx:    Optional[torch.Tensor] = None  # [1, 1, d] or None
    # Server-side priors (wesnoth_ai/server_priors.py): when set, the
    # legal actions with priors were computed on the inference server
    # from actor-shipped masks; enumerate_legal_actions_with_priors
    # unpacks them instead of reading the logits (which the server
    # then ships as placeholders).
    legal_compact: Optional[object] = None

    # Diagnostic: marginal-over-actors probability of each action
    # type. Layout [1, T+2]:
    #   0: ATTACK   (sum_unit_actors P(unit) * P(attack | unit))
    #   1: MOVE     (sum_unit_actors P(unit) * P(move   | unit))
    #   2: RECRUIT  (sum_recruit_actors P(recruit_slot))
    #   3: END_TURN (P(end_turn_slot))
    #
    # LAZY (optimization #2, 2026-06-14): this was a per-forward field
    # costing ~5-9% of every forward (two extra softmaxes + reductions),
    # but the ONLY reader is test_action_type_head.py -- self-play and
    # the trainer never touch it. It is now computed on demand from the
    # logits already stored on this dataclass, so the hot path pays
    # nothing. Recompute-on-access (no cache) is fine: the sole reader
    # touches it a handful of times.
    @property
    def marginal_type_logits(self) -> torch.Tensor:  # [1, T+2]
        return _compute_marginal_type_logits(
            self.actor_logits, self.type_logits,
            self.num_units, self.num_recruits,
            self.actor_logits.device,
        )


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
        aux_score:   bool = False,
        moves_left:  bool = False,
        gbc:         bool = False,
    ):
        super().__init__()
        self.d_model     = d_model
        self.max_attacks = max_attacks
        self.has_aux_score = bool(aux_score)
        self.has_moves_left = bool(moves_left)
        self.has_gbc = bool(gbc)
        # GBC event-prediction heads (docs/archive/gbc_spec.md, value-head
        # repair role): built only when `gbc=True`, so the default
        # model is byte-identical. Params ride model.parameters()
        # (optimizer) and state_dict (checkpoint) automatically.
        if self.has_gbc:
            from wesnoth_ai.gbc import GBCHeads
            self.gbc_heads = GBCHeads(d_model)
        else:
            self.gbc_heads = None

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
        # Packed varlen trunk for batched inference (docs/
        # gpu_forward_design_20260904.md section 5.3; wesnoth_ai/
        # packed_trunk.py). Off until measured on the box. When on, it
        # serves only the calls the flash varlen kernel can take (CUDA,
        # bf16/fp16; see _packed_trunk_applies); every other call keeps
        # the padded trunk.
        self.infer_packed_trunk = False
        # Compiled variant of the packed trunk (design note section 13;
        # packed_trunk.CompiledPackedTrunk): the layer loop as one
        # inductor graph over a native-dtype (bf16 on CUDA) copy of the
        # encoder's weights. Requires infer_packed_trunk. Set through
        # configure_packed_compile (backend, mode) or directly;
        # warmup_packed_compile compiles before serving;
        # packed_compile_active says whether compiled code serves.
        self.infer_compile_packed = False
        self._weights_version = 0        # bumped by load_state_dict; the copy follows it
        self._packed_weights: Optional[PackedTrunkWeights] = None
        self._packed_compile: Optional[CompiledPackedTrunk] = None

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
        # Distributional value head (C51-style categorical). Reads
        # the contextualized [CLS] global token (built specifically
        # to carry state-summary signal); earlier mean-pooled
        # variants diluted by 1700× over all token positions.
        #
        # The head outputs raw logits over K bins on fixed support
        # [V_MIN, V_MAX]; softmax in `forward` produces a probability
        # distribution Z(s). The scalar value the rest of the
        # codebase reads is `E[Z(s)]`. Cliffness — a free
        # heteroscedastic uncertainty estimate — is `std(Z(s))`.
        #
        # No `tanh`: the bin support is bounded by construction;
        # softmax can't put mass outside [V_MIN, V_MAX]. Categorical
        # CE in the trainer (vs. MSE on a tanh scalar) trains both
        # the distribution mean AND its spread — wider returns at a
        # state push the network toward a wider predicted
        # distribution.
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, VALUE_N_ATOMS),
        )
        # Bin support, registered as a buffer so it (a) saves with
        # the checkpoint and (b) moves with `.to(device)`.
        self.register_buffer(
            "_value_atoms",
            torch.linspace(VALUE_V_MIN, VALUE_V_MAX, VALUE_N_ATOMS),
        )
        # Optional auxiliary head (KataGo §3.5): predicts the final
        # material margin from the global token, tanh-bounded to
        # (-1, +1). Built only when `aux_score` is on, so the default
        # arch (and existing checkpoints) are byte-unchanged.
        self.aux_score_head = (
            nn.Linear(d_model, 1) if self.has_aux_score else None)
        # Optional moves-left head (Lc0-style): predicts the fraction
        # of the turn budget remaining from the global token, sigmoid-
        # bounded to (0, 1). Built only when `moves_left` is on, so
        # the default arch (and existing checkpoints) are unchanged.
        self.moves_left_head = (
            nn.Linear(d_model, 1) if self.has_moves_left else None)

    def forward(self, encoded: "EncodedState") -> ModelOutput:
        # Opt-in bf16 inference autocast (2026-08-05 throughput
        # program; the 15M profile put forward at 53% of rollout at
        # 20.7ms/leaf batch-1). Set `model.infer_autocast_bf16 = True`
        # (CLI --infer-bf16) to run the trunk+heads in bf16 on CUDA;
        # OUTPUTS are cast back to float32 in _finalize_output so
        # numpy consumers (action_sampler) never see bf16. Inference
        # only: `self.training` forwards keep full fp32 -- the
        # trainer's loss math is untouched.
        if (getattr(self, "infer_autocast_bf16", False)
                and not self.training
                and encoded.hex_tokens.device.type == "cuda"):
            import dataclasses
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self._forward_impl(encoded)
            return dataclasses.replace(out, **{
                f.name: v.float()
                for f in dataclasses.fields(out)
                if isinstance((v := getattr(out, f.name)), torch.Tensor)
                and v.dtype == torch.bfloat16})
        return self._forward_impl(encoded)

    def _forward_impl(self, encoded: "EncodedState") -> ModelOutput:
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

        # Distributional value head: read the contextualized global
        # token, emit K logits over the bin support, derive scalar
        # mean (`value`) and std (`cliffness`) from the resulting
        # distribution. Trainer uses `value_logits` directly for the
        # categorical-CE loss; rollout/MCTS read `value` and
        # `cliffness`.
        value_logits = self.value_head(global_ctx.squeeze(1))     # [B, K]
        value_probs  = F.softmax(value_logits, dim=-1)            # [B, K]
        atoms = self._value_atoms                                 # [K]
        value = (value_probs * atoms).sum(dim=-1, keepdim=True)   # [B, 1]
        var_v = ((value_probs * atoms.pow(2)).sum(dim=-1, keepdim=True)
                 - value.pow(2)).clamp_min(0)
        cliffness = var_v.sqrt()                                  # [B, 1]

        # Aux + moves-left heads read a DETACHED global token (user
        # ruling 2026-09-01: telemetry-only — the heads train, the
        # trunk receives no gradient from them). Before the detach
        # the aux MSE contributed ~5-6% of the total update
        # direction through the trunk (signal-profiler round 5).
        aux_score = None
        if self.aux_score_head is not None:
            aux_score = torch.tanh(
                self.aux_score_head(
                    global_ctx.squeeze(1).detach()))              # [B, 1]
        moves_left = None
        if self.moves_left_head is not None:
            moves_left = torch.sigmoid(
                self.moves_left_head(
                    global_ctx.squeeze(1).detach()))              # [B, 1]

        # marginal_type_logits is now a lazy property on ModelOutput
        # (optimization #2) -- not computed here; the sole reader is a
        # test, and self-play/training never touch it.
        return ModelOutput(
            actor_logits=actor_logits,
            actor_kind=actor_kind,
            type_logits=type_logits,
            target_logits=target_logits,
            weapon_logits=weapon_logits,
            value=value,
            value_logits=value_logits,
            cliffness=cliffness,
            num_units=U,
            num_recruits=R,
            aux_score=aux_score,
            moves_left=moves_left,
            # GBC tap: references only, populated only when the gbc
            # heads exist (trainer loss path; eval models stay None).
            unit_ctx=unit_ctx if self.has_gbc else None,
            hex_ctx=hex_ctx if self.has_gbc else None,
            global_ctx=global_ctx if self.has_gbc else None,
        )

    # ------------------------------------------------------------------
    # Batched forward — one transformer pass for many states at once,
    # with NO per-sample kernel launches (2026-09-04 rewrite: the
    # per-sample cat/matmul loop of the old version cost ~10 launches
    # per leaf and capped the inference server near 600 leaves/s on a
    # 4090; docs/box_specs.md "Pipeline baseline"). Per-sample
    # ModelOutputs are VIEWS into batched tensors.
    # ------------------------------------------------------------------

    def forward_batch(self, encoded_list, autocast_bf16: Optional[bool] = None,
                      packed: Optional[bool] = None):
        """Run one padded transformer forward over B encoded states.

        Returns a list of B per-sample ModelOutput objects with the
        EXACT shapes the single-sample path produces (views into the
        batched head outputs), so downstream code needs no changes.
        `autocast_bf16` overrides the model's `infer_autocast_bf16`
        for this call (the inference server's own switch); `packed`
        overrides the packed-trunk selection (see forward_streams).
        """
        B = len(encoded_list)
        if B == 0:
            return []
        if B == 1 and autocast_bf16 is None and packed is None:
            return [self.forward(encoded_list[0])]
        padded = self.forward_padded(encoded_list, autocast_bf16=autocast_bf16, packed=packed)
        return padded.samples()

    def forward_padded(self, encoded_list, autocast_bf16: Optional[bool] = None,
                       packed: Optional[bool] = None) -> "PaddedOutput":
        """The batched computation behind forward_batch: every head is
        applied once to padded [B, ...] tensors; actor slots are laid
        out per sample in the canonical compact order (units |
        recruits | end_turn) by one gather, so a per-sample output is
        a view. Launch count is independent of B.

        Runs under the same bf16 autocast as `forward` when
        `infer_autocast_bf16` is set (2026-09-04: the batched path ran
        fp32 eager, 1.5 ms per 714-token sample on a 4090, flat from
        batch 4 to 64); outputs are cast back to float32."""
        device = encoded_list[0].hex_tokens.device
        use_bf16 = (getattr(self, "infer_autocast_bf16", False)
                    if autocast_bf16 is None else bool(autocast_bf16))
        if use_bf16 and not self.training and device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self._forward_padded_impl(encoded_list, packed)
            return out.float32()
        return self._forward_padded_impl(encoded_list, packed)

    def _forward_padded_impl(self, encoded_list, packed: Optional[bool] = None) -> "PaddedOutput":
        d = self.d_model
        device = encoded_list[0].hex_tokens.device
        dtype = encoded_list[0].hex_tokens.dtype
        B = len(encoded_list)
        Us = [e.unit_tokens.size(1) for e in encoded_list]
        Rs = [e.recruit_tokens.size(1) for e in encoded_list]
        Hs = [e.hex_tokens.size(1) for e in encoded_list]

        def _padded(tokens, L_max):
            if L_max == 0:
                return torch.zeros(B, 0, d, device=device, dtype=dtype)
            return torch.nn.utils.rnn.pad_sequence([t.squeeze(0) for t in tokens],
                                                   batch_first=True)

        return self.forward_streams(
            _padded([e.hex_tokens for e in encoded_list], max(Hs)),
            _padded([e.unit_tokens for e in encoded_list], max(Us)),
            _padded([e.recruit_tokens for e in encoded_list], max(Rs)),
            torch.cat([e.global_token for e in encoded_list], dim=0),
            torch.cat([e.end_turn_token for e in encoded_list], dim=0),
            list(zip(Us, Rs, Hs)), packed=packed)

    def forward_streams(self, hex_batch, unit_batch, recruit_batch, global_batch,
                        end_turn_batch, sizes, packed: Optional[bool] = None) -> "PaddedOutput":
        """Batched forward over already-padded streams ([B, L_max, d]
        each, WITHOUT token-kind embeddings) and per-sample sizes
        (U_b, R_b, H_b). The inference server feeds this straight from
        encoder.encode_from_raw_padded; forward_padded feeds it from
        EncodedStates. Same bf16 policy as forward_padded when called
        through it; callers that come here directly autocast themselves.

        `packed`: None selects the packed trunk where `infer_packed_trunk`
        and the flash varlen kernel apply (_packed_trunk_applies); True
        forces the packed code path (per-segment SDPA off the kernel's
        domain, for tests); False forces the padded trunk."""
        if self.infer_compile_packed and not self.infer_packed_trunk:
            raise ValueError("infer_compile_packed requires infer_packed_trunk")
        if packed is None:
            packed = self._packed_trunk_applies(hex_batch)
        if packed:
            return self._forward_streams_packed(hex_batch, unit_batch, recruit_batch,
                                                global_batch, end_turn_batch, sizes)
        d = self.d_model
        device = hex_batch.device
        B = len(sizes)
        Us = [s[0] for s in sizes]
        Rs = [s[1] for s in sizes]
        Hs = [s[2] for s in sizes]
        U_max, R_max, H_max = unit_batch.size(1), recruit_batch.size(1), hex_batch.size(1)
        kk = self.token_kind_embed.weight
        if H_max:
            hex_batch = hex_batch + kk[TokenKind.HEX]
        if U_max:
            unit_batch = unit_batch + kk[TokenKind.UNIT]
        if R_max:
            recruit_batch = recruit_batch + kk[TokenKind.RECRUIT]
        global_batch = global_batch + kk[TokenKind.GLOBAL]
        end_turn_batch = end_turn_batch + kk[TokenKind.END_TURN]
        x = torch.cat([hex_batch, unit_batch, recruit_batch, global_batch, end_turn_batch], dim=1)
        seq_len = x.size(1)

        # Key-padding mask and the compact actor gather index, both
        # built host-side in numpy and moved once. Global + end_turn
        # are never masked, so every row keeps >= 1 real position.
        pad_np = np.zeros((B, seq_len), dtype=bool)
        A_max = U_max + R_max + 1
        idx_np = np.full((B, A_max), H_max + U_max + R_max + 1, dtype=np.int64)
        kind_np = np.full((B, A_max), ActorKind.END_TURN, dtype=np.int64)
        for b in range(B):
            U_b, R_b, H_b = Us[b], Rs[b], Hs[b]
            pad_np[b, H_b:H_max] = True
            pad_np[b, H_max + U_b:H_max + U_max] = True
            pad_np[b, H_max + U_max + R_b:H_max + U_max + R_max] = True
            idx_np[b, :U_b] = np.arange(H_max, H_max + U_b)
            idx_np[b, U_b:U_b + R_b] = np.arange(H_max + U_max, H_max + U_max + R_b)
            kind_np[b, :U_b] = ActorKind.UNIT
            kind_np[b, U_b:U_b + R_b] = ActorKind.RECRUIT
        pad_mask = torch.from_numpy(pad_np).to(device)
        actor_idx = torch.from_numpy(idx_np).to(device)
        actor_kind = torch.from_numpy(kind_np)          # stays on CPU

        x = self.encoder(x, src_key_padding_mask=pad_mask)   # [B, seq_len, d]
        hex_ctx = x[:, :H_max]
        global_ctx = x[:, H_max + U_max + R_max:H_max + U_max + R_max + 1]  # [B, 1, d]
        actor_ctx = torch.gather(x, 1, actor_idx.unsqueeze(-1).expand(-1, -1, d))  # [B, A_max, d]
        return self._heads(actor_ctx, hex_ctx, global_ctx, actor_kind, sizes,
                           unit_ctx=x[:, H_max:H_max + U_max] if self.has_gbc else None)

    def _packed_trunk_applies(self, x: torch.Tensor) -> bool:
        """The packed trunk serves a call when it is switched on, the
        model is in eval mode, and the in-projections will produce a
        dtype the flash varlen kernel takes on CUDA: bf16/fp16 inputs, or
        fp32 inputs under a bf16/fp16 autocast (torch.is_autocast_enabled
        takes the device type: torch 2.5.1 torch/csrc/autograd/init.cpp
        :559-580). Everything else keeps the padded trunk."""
        if not (getattr(self, "infer_packed_trunk", False) and not self.training
                and x.device.type == "cuda"):
            return False
        dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
        return flash_varlen_applies(x.device, dtype)

    def _forward_streams_packed(self, hex_batch, unit_batch, recruit_batch, global_batch,
                                end_turn_batch, sizes) -> "PaddedOutput":
        """forward_streams on the packed layout (design note section 5.3;
        the index arrays follow section 4.3). One gather packs the real
        tokens of the padded streams into [total, d] and one embedding
        lookup adds the token kinds; _packed_trunk_heads does the rest.
        Every index array is built host-side and shipped in one pinned
        non-blocking copy: nothing here synchronizes with the host."""
        d = self.d_model
        B = len(sizes)
        H_max, U_max, R_max = hex_batch.size(1), unit_batch.size(1), recruit_batch.size(1)
        layout = build_packed_layout(sizes, H_max, U_max, R_max, TokenKind, ActorKind)
        index = layout.to_device(hex_batch.device)
        padded = torch.cat([hex_batch, unit_batch, recruit_batch, global_batch, end_turn_batch],
                           dim=1).reshape(B * (H_max + U_max + R_max + 2), d)
        x = padded.index_select(0, index.src) + self.token_kind_embed(index.kind)   # [total, d]
        return self._packed_trunk_heads(x, index, layout, sizes, H_max, U_max, R_max)

    def forward_embedded(self, streams: EmbeddedStreams,
                         packed: Optional[bool] = None) -> "PaddedOutput":
        """Batched forward over stream-ordered token embeddings
        (encoder.encode_from_raw_embedded; the server's packed-embed
        path). With the packed trunk, one gather orders the rows into
        the packed layout and no padded tensor is built at all. With the
        padded trunk, one gather lays them out as the padded streams
        (zeros at the pads, as pad_sequence fills them) and
        forward_streams runs unchanged. Same PaddedOutput as
        forward_streams on encode_from_raw_padded's streams; `packed` as
        in forward_streams."""
        if self.infer_compile_packed and not self.infer_packed_trunk:
            raise ValueError("infer_compile_packed requires infer_packed_trunk")
        tokens, sizes = streams.tokens, streams.sizes
        B, d = len(sizes), self.d_model
        U_max, R_max, H_max = (max(s[i] for s in sizes) for i in (0, 1, 2))
        if packed is None:
            packed = self._packed_trunk_applies(tokens)
        if packed:
            layout = build_packed_layout(sizes, H_max, U_max, R_max, TokenKind, ActorKind,
                                         source="streams")
            index = layout.to_device(tokens.device)
            x = tokens.index_select(0, index.src) + self.token_kind_embed(index.kind)
            return self._packed_trunk_heads(x, index, layout, sizes, H_max, U_max, R_max)
        L = H_max + U_max + R_max + 2
        idx = torch.from_numpy(padded_gather_index(sizes, H_max, U_max, R_max))
        if tokens.device.type == "cuda":
            idx = idx.pin_memory().to(tokens.device, non_blocking=True)
        elif tokens.device.type != "cpu":
            idx = idx.to(tokens.device)
        rows = torch.cat([tokens, tokens.new_zeros(1, d)]).index_select(0, idx).view(B, L, d)
        o = H_max + U_max + R_max
        return self.forward_streams(rows[:, :H_max], rows[:, H_max:H_max + U_max],
                                    rows[:, H_max + U_max:o], rows[:, o:o + 1], rows[:, o + 1:],
                                    sizes, packed=False)

    def _packed_trunk_heads(self, x, index, layout, sizes, H_max, U_max, R_max) -> "PaddedOutput":
        """The trunk on packed tokens x [total, d] (token kinds added),
        then the heads on the actor / hex / global (and unit, for GBC)
        contexts laid out in the padded shapes by index_select, so the
        heads and PaddedOutput are the padded path's."""
        if not getattr(self, "_packed_trunk_checked", False):
            check_packed_trunk_supported(self.encoder)
            self._packed_trunk_checked = True
        d = self.d_model
        B = len(sizes)
        A_max = U_max + R_max + 1
        if self.infer_compile_packed:
            x = self._run_compiled_packed_trunk(x, index)
        else:
            x = packed_trunk(self.encoder, x, index)
        actor_ctx = x.index_select(0, index.actor).view(B, A_max, d)
        hex_ctx = x.index_select(0, index.hex).view(B, H_max, d)
        global_ctx = x.index_select(0, index.glob).view(B, 1, d)
        unit_ctx = x.index_select(0, index.unit).view(B, U_max, d) if self.has_gbc else None
        return self._heads(actor_ctx, hex_ctx, global_ctx, torch.from_numpy(layout.actor_kind),
                           sizes, unit_ctx)

    # ------------------------------------------------------------------
    # Compiled packed trunk (design note section 13)
    # ------------------------------------------------------------------

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Every weight publication goes through here (the policy's
        inference snapshot, checkpoint loads); the version tells the
        compiled packed trunk to refresh its native-dtype copy."""
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._weights_version += 1
        return result

    def configure_packed_compile(self, *, backend="inductor",
                                 mode: Optional[str] = None) -> CompiledPackedTrunk:
        """Turns infer_compile_packed on with these torch.compile options
        (mode None is inductor's default, no CUDA graphs;
        "max-autotune-no-cudagraphs" adds the GEMM templates)."""
        if not self.infer_packed_trunk:
            raise ValueError("infer_compile_packed requires infer_packed_trunk")
        check_packed_trunk_supported(self.encoder)
        layer = self.encoder.layers[0]
        self._packed_compile = CompiledPackedTrunk(layer.activation, layer.norm1.eps,
                                                   backend=backend, mode=mode)
        self._packed_weights = None
        self.infer_compile_packed = True
        return self._packed_compile

    def warmup_packed_compile(self, shapes=None) -> dict:
        """Compiles the packed trunk on synthetic batches before serving.
        `shapes`: (B, hexes, units, recruits) per batch; the defaults
        bracket production on CUDA (under the server's bf16 autocast)
        and stay small on CPU. Returns packed_compile_stats()."""
        if self._packed_compile is None:
            self.configure_packed_compile()
        device = self._value_atoms.device
        cuda = device.type == "cuda"
        if shapes is None:
            shapes = ((16, 1300, 30, 7), (8, 700, 2, 0)) if cuda else ((4, 40, 3, 2), (2, 25, 2, 0))
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=cuda):
            for B, H, U, R in shapes:
                sizes = [(max(1, U - b % 3), max(0, R - b % 2), max(1, H - 7 * b))
                         for b in range(B)]
                self.forward_streams(*random_padded_streams(sizes, self.d_model, device),
                                     packed=True)
        return self.packed_compile_stats()

    @property
    def packed_compile_active(self) -> bool:
        return self._packed_compile is not None and self._packed_compile.active

    def packed_compile_stats(self) -> dict:
        if self._packed_compile is None:
            return {"active": False}
        stats = self._packed_compile.stats()
        stats["weights_dtype"] = (str(self._packed_weights.dtype).replace("torch.", "")
                                  if self._packed_weights is not None else None)
        return stats

    def _run_compiled_packed_trunk(self, x, index):
        """The compiled loop over the native-dtype weight copy: the
        autocast dtype where one is active (bf16 on the server), the
        input's otherwise (fp32 on CPU). The copy is (re)built for a new
        dtype or device and refreshed when the weight version moved."""
        dev = x.device.type
        dtype = torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else x.dtype
        if self._packed_compile is None:
            self.configure_packed_compile()
        w = self._packed_weights
        if w is None or w.dtype != dtype or w.device != x.device:
            w = self._packed_weights = PackedTrunkWeights.build(
                self.encoder, dtype, x.device, self._weights_version)
        elif w.version != self._weights_version:
            w.refresh(self.encoder, self._weights_version)
        return self._packed_compile.run(x, index, w)

    def _heads(self, actor_ctx, hex_ctx, global_ctx, actor_kind, sizes,
               unit_ctx) -> "PaddedOutput":
        """The four heads on contextualized actor [B, A_max, d], hex
        [B, H_max, d] and global [B, 1, d] rows, whichever trunk produced
        them."""
        d = self.d_model
        device, dtype = actor_ctx.device, actor_ctx.dtype
        B, A_max, _ = actor_ctx.shape
        H_max = hex_ctx.size(1)
        actor_logits = self.actor_head(actor_ctx).squeeze(-1)           # [B, A_max]
        type_logits = self.type_head(actor_ctx)                          # [B, A_max, T]
        weapon_logits = self.weapon_head(actor_ctx)                      # [B, A_max, W]
        if H_max == 0:
            target_logits = torch.zeros(B, A_max, 0, device=device, dtype=dtype)
        else:
            q = self.target_q_proj(actor_ctx)                            # [B, A_max, d]
            k = self.target_k_proj(hex_ctx)                              # [B, H_max, d]
            target_logits = torch.bmm(q, k.transpose(1, 2)) / (d ** 0.5)  # [B, A_max, H_max]

        g = global_ctx.squeeze(1)
        value_logits = self.value_head(g)                                # [B, K]
        value_probs = F.softmax(value_logits, dim=-1)
        atoms = self._value_atoms
        value = (value_probs * atoms).sum(dim=-1, keepdim=True)          # [B, 1]
        var_v = ((value_probs * atoms.pow(2)).sum(dim=-1, keepdim=True)
                 - value.pow(2)).clamp_min(0)
        cliffness = var_v.sqrt()
        aux_score = (torch.tanh(self.aux_score_head(g.detach()))
                     if self.aux_score_head is not None else None)
        moves_left = (torch.sigmoid(self.moves_left_head(g.detach()))
                      if self.moves_left_head is not None else None)
        return PaddedOutput(
            actor_logits=actor_logits, actor_kind=actor_kind, type_logits=type_logits,
            target_logits=target_logits, weapon_logits=weapon_logits, value=value,
            value_logits=value_logits, cliffness=cliffness, aux_score=aux_score,
            moves_left=moves_left, sizes=[tuple(s) for s in sizes],
            unit_ctx=unit_ctx,
            hex_ctx=hex_ctx if self.has_gbc else None,
            global_ctx=global_ctx if self.has_gbc else None)


@dataclass
class PaddedOutput:
    """Batched head outputs ([B, ...], padded) plus per-sample sizes.
    `samples()` yields the per-sample ModelOutputs as views; `to_cpu()`
    moves each batched field ONCE (one transfer per field per batch)
    and yields CPU views -- the inference server's path."""
    actor_logits: torch.Tensor          # [B, A_max]
    actor_kind: torch.Tensor            # [B, A_max] long, CPU
    type_logits: torch.Tensor           # [B, A_max, T]
    target_logits: torch.Tensor         # [B, A_max, H_max]
    weapon_logits: torch.Tensor         # [B, A_max, W]
    value: torch.Tensor                 # [B, 1]
    value_logits: torch.Tensor          # [B, K]
    cliffness: torch.Tensor             # [B, 1]
    aux_score: Optional[torch.Tensor]
    moves_left: Optional[torch.Tensor]
    sizes: List[Tuple[int, int, int]]   # (U_b, R_b, H_b)
    unit_ctx: Optional[torch.Tensor] = None
    hex_ctx: Optional[torch.Tensor] = None
    global_ctx: Optional[torch.Tensor] = None

    _TENSOR_FIELDS = ("actor_logits", "type_logits", "target_logits", "weapon_logits",
                      "value", "value_logits", "cliffness", "aux_score", "moves_left",
                      "unit_ctx", "hex_ctx", "global_ctx")

    def to_cpu(self) -> "PaddedOutput":
        kw = {f: getattr(self, f) for f in ("actor_kind", "sizes")}
        for f in self._TENSOR_FIELDS:
            v = getattr(self, f)
            kw[f] = v.cpu() if v is not None else None
        return PaddedOutput(**kw)

    def float32(self) -> "PaddedOutput":
        """Cast bf16 autocast outputs back to float32 (numpy consumers
        never see bf16, as in `forward`)."""
        kw = {f: getattr(self, f) for f in ("actor_kind", "sizes")}
        for f in self._TENSOR_FIELDS:
            v = getattr(self, f)
            kw[f] = (v.float() if v is not None and v.dtype == torch.bfloat16 else v)
        return PaddedOutput(**kw)

    def sample(self, b: int) -> ModelOutput:
        U_b, R_b, H_b = self.sizes[b]
        A_b = U_b + R_b + 1
        return ModelOutput(
            actor_logits=self.actor_logits[b:b + 1, :A_b],
            actor_kind=self.actor_kind[b:b + 1, :A_b].to(self.actor_logits.device),
            type_logits=self.type_logits[b:b + 1, :A_b],
            target_logits=self.target_logits[b:b + 1, :A_b, :H_b],
            weapon_logits=self.weapon_logits[b:b + 1, :A_b],
            value=self.value[b:b + 1],
            value_logits=self.value_logits[b:b + 1],
            cliffness=self.cliffness[b:b + 1],
            num_units=U_b, num_recruits=R_b,
            aux_score=self.aux_score[b:b + 1] if self.aux_score is not None else None,
            moves_left=self.moves_left[b:b + 1] if self.moves_left is not None else None,
            unit_ctx=self.unit_ctx[b:b + 1, :U_b] if self.unit_ctx is not None else None,
            hex_ctx=self.hex_ctx[b:b + 1, :H_b] if self.hex_ctx is not None else None,
            global_ctx=self.global_ctx[b:b + 1] if self.global_ctx is not None else None)

    def samples(self) -> List[ModelOutput]:
        return [self.sample(b) for b in range(len(self.sizes))]


def random_padded_streams(sizes, d_model: int, device, seed: int = 0):
    """Random padded streams for per-sample sizes (U, R, H), in
    forward_streams' argument order. Pad positions hold random values,
    so a trunk that read them would show. Used by warmup_packed_compile
    and the packed-trunk tests."""
    g = torch.Generator(device=device).manual_seed(seed)
    B = len(sizes)
    H_max, U_max, R_max = (max(s[i] for s in sizes) for i in (2, 0, 1))

    def stream(n):
        return torch.randn(B, n, d_model, generator=g, device=device)
    return stream(H_max), stream(U_max), stream(R_max), stream(1), stream(1), list(sizes)
