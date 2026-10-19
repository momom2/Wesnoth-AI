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

The output containers and the fixed vocabulary (ModelOutput,
PaddedOutput, TokenKind, ActorKind, UnitActionType, MAX_ATTACKS, the
value support) live in wesnoth_ai/model_output.py; the padded-stream
helpers in wesnoth_ai/padded_streams.py. Both are re-exported here.

The parity-memory recipe (docs/parity_memory_design_20260929.md, the
"Model interface" section lists the calls): `observation_parity` adds
the sighting stream's token kind and the belief head (one logit per hex
token); `memory_slots` adds the learned memory (wesnoth_ai/memory.py),
whose active slots enter the trunk as tokens and are written back after
it. Sightings and memory slots sit between the recruits and the global
token and are never actors or targets. With both off the network is
obs8's, parameter for parameter.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from wesnoth_ai.encoder import EncodedState
from wesnoth_ai.memory import MemoryBatch, MemoryState, SlotMemory, stream_row_index
from wesnoth_ai.model_output import (
    MAX_ATTACKS, VALUE_N_ATOMS, VALUE_V_MAX, VALUE_V_MIN, ActorKind, ModelOutput,
    PaddedOutput, TokenKind, UnitActionType,
)
from wesnoth_ai.packed_trunk import (
    CompiledPackedTrunk, EmbeddedStreams, PackedEmbeddedIndex, PackedTrunkWeights, WindowHeadIndex,
    batch_counts,
    build_packed_layout, check_packed_trunk_supported, flash_varlen_applies, host_to_device,
    packed_trunk, padded_gather_index,
)
from wesnoth_ai.material import MATERIAL_SCALE
from wesnoth_ai.profile_regions import region
from wesnoth_ai.padded_streams import (ExtraStreams, PaddedIndex, check_sighting_stream,
                                       extra_streams, material_batch, memory_batch,
                                       pad_encoded_streams, pad_sighting_streams, padded_index,
                                       padded_trunk_index, random_padded_streams, stream_widths)
from wesnoth_ai.trunk_modes import CompiledBlock, TrunkMode, check_trunk_mode_supported, run_trunk
from wesnoth_ai.fp8_inference import Fp8TrunkWeights, check_fp8_inference_supported, make_fp8_trunk_layers

__all__ = [
    "WesnothModel",
    # Re-exports: importers and pickles address these through this module.
    "ModelOutput", "PaddedOutput", "TokenKind", "ActorKind", "UnitActionType",
    "MAX_ATTACKS", "VALUE_N_ATOMS", "VALUE_V_MIN", "VALUE_V_MAX", "random_padded_streams",
]


def _check_packed_index(prebuilt: PackedEmbeddedIndex, sizes, sightings, mem) -> None:
    """A prebuilt packed index was built for this batch's per-row counts:
    its segments and memory rows are laid out row by row."""
    counts = batch_counts(sizes, sightings, mem.counts if mem is not None else None)
    if counts != prebuilt.counts:
        raise ValueError(f"a packed trunk index built for another batch: counts {prebuilt.counts}, "
                         f"the batch's {counts}")


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
        value_material: bool = False,
        # The parity-memory recipe (module docstring): the sighting
        # stream and the belief head; the learned memory's slot count
        # (0 = none).
        observation_parity: bool = False,
        memory_slots: int = 0,
    ):
        super().__init__()
        self.d_model     = d_model
        self.max_attacks = max_attacks
        self.has_aux_score = bool(aux_score)
        self.has_moves_left = bool(moves_left)
        self.has_gbc = bool(gbc)
        self.has_value_material = bool(value_material)
        self.observation_parity = bool(observation_parity)
        self.memory_slots = int(memory_slots)
        if self.memory_slots < 0:
            raise ValueError(f"memory_slots must be >= 0, got {memory_slots}")
        # GBC event-prediction heads (docs/archive/gbc_spec.md, value-head
        # repair role): built only when `gbc=True`, so the default
        # model is byte-identical. Params ride model.parameters()
        # (optimizer) and state_dict (checkpoint) automatically.
        if self.has_gbc:
            from wesnoth_ai.gbc import GBCHeads
            self.gbc_heads = GBCHeads(d_model)
        else:
            self.gbc_heads = None

        # Distinguish streams at attention time. The sighting and memory
        # rows exist only in a model that reads those streams, so obs8's
        # table keeps its shape.
        self.token_kind_embed = nn.Embedding(
            TokenKind.COUNT_EXTENDED if self.extended_streams else TokenKind.COUNT, d_model)

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
        # Inference that reads the policy alone (a raw player at self-play
        # generation, tools/eval_inference_server.py --priors-only): the
        # heads skip the value, its distribution, the optional heads and the
        # belief head, whose outputs come back None. The policy heads run as
        # they do otherwise, so their outputs are the same bits. Ignored in
        # training mode.
        self.infer_priors_only = False
        # The trainer's trunk modes at inference (wesnoth_ai/trunk_modes.py),
        # under a bf16 autocast, over a bf16 copy of the weights
        # (_weight_copy): the residual stream in bf16, layer norms included,
        # without the autocast's float32 stream and its two casts per layer
        # (the server's --bf16-residual); the first feed-forward linear, its
        # bias and the ReLU in one cuBLASLt call (--fused-relu). Rounding
        # changes: the bf16 stream's outputs differ from the float32
        # stream's in their last bits, and the fused ReLU's on CUDA wherever
        # cuBLASLt takes another algorithm for the RELU_BIAS epilogue, so a
        # game played under either is another game. Ignored in training
        # mode and without a bf16 autocast.
        self.infer_bf16_residual = False
        self.infer_fused_relu = False
        # The packed trunk's linear layers on e4m3 operands over the bf16
        # residual stream (wesnoth_ai/fp8_inference.py; the server's
        # --fp8-inference): another rounding, for generation only. Under a
        # bf16 autocast in eval mode only; compiled with infer_compile_packed.
        self.infer_fp8 = False
        self._fp8_weights: Optional[Fp8TrunkWeights] = None
        self._fp8_compile: Optional[CompiledPackedTrunk] = None
        self._weights_version = 0        # bumped by load_state_dict; the copies follow it
        # The frozen weight copies by (linear dtype, norm dtype).
        self._weight_copies: Dict[Tuple[torch.dtype, torch.dtype], PackedTrunkWeights] = {}
        self._packed_compile: Optional[CompiledPackedTrunk] = None
        self._compiled_weights: Optional[PackedTrunkWeights] = None   # the copy the compiled loop last read
        # The packed trunk's training variants (wesnoth_ai/trunk_modes.py),
        # set by the sequence trainer through set_trunk_mode; None runs
        # packed_trunk as before.
        self.trunk_mode: Optional[TrunkMode] = None
        self._trunk_block: Optional[CompiledBlock] = None

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
        # Material as an input of the value head (2026-09-08 study,
        # docs/value_head_study_20260907.md): a zero-initialized
        # projection of the scalar added to the global context before
        # the head, so a model warm-started from a checkpoint without
        # it starts exactly at that checkpoint. Built only when
        # `value_material` is on; the policy heads never see it.
        self.material_proj = nn.Linear(1, d_model) if self.has_value_material else None
        if self.material_proj is not None:
            nn.init.zeros_(self.material_proj.weight)
            nn.init.zeros_(self.material_proj.bias)
        # The belief head (design "The belief head"): per hex token, the
        # logit that an enemy unit the side cannot see stands there.
        self.belief_head = nn.Linear(d_model, 1) if self.observation_parity else None
        # The learned memory (wesnoth_ai/memory.py). Built last, so the
        # parameters before it draw from the generator as without it.
        self.slot_memory = SlotMemory(d_model, self.memory_slots) if self.memory_slots else None

    @property
    def extended_streams(self) -> bool:
        """The model reads the sighting or the memory stream."""
        return self.observation_parity or self.memory_slots > 0

    def initial_memory(self, k: int) -> torch.Tensor:
        """float32 [k, d]: a game-side's memory before its first decision,
        k in 0..memory_slots active slots (wesnoth_ai/memory.py)."""
        if self.slot_memory is None:
            raise ValueError("this model has no memory (memory_slots=0)")
        return self.slot_memory.initial_state(k)

    def _value_input(self, g: torch.Tensor, material: Optional[torch.Tensor]) -> torch.Tensor:
        """The value head's input: the global context, plus the
        material projection when the model carries one."""
        if self.material_proj is None:
            return g
        if material is None:
            raise ValueError("a value_material model needs EncodedState.material "
                             "(the packed embed and the server priors paths do "
                             "not carry it)")
        return g + self.material_proj(material.to(g.dtype) / MATERIAL_SCALE)

    def forward(self, encoded: "EncodedState",
                memory: Optional[object] = None) -> ModelOutput:
        """One state's outputs. `memory`: the side's memory state, float32
        [k, d], or a `MemoryState` (a player's: None state for the
        game-side's first decision), required by a model with memory
        slots; the output's `memory` is the state after this decision. A
        model reading the parity-memory streams runs the batched path with
        one sample."""
        if isinstance(memory, MemoryState):
            memory = self.initial_memory(memory.k) if memory.state is None else memory.state
        if self.extended_streams:
            states = None if memory is None else [memory]
            return self.forward_padded([encoded], memory=states).sample(0)
        if memory is not None:
            raise ValueError("memory state passed to a model without memory slots")
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
        value_logits = self.value_head(
            self._value_input(global_ctx.squeeze(1), encoded.material))  # [B, K]
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
                      packed: Optional[bool] = None,
                      memory: Optional[Sequence[torch.Tensor]] = None):
        """Run one padded transformer forward over B encoded states.

        Returns a list of B per-sample ModelOutput objects with the
        EXACT shapes the single-sample path produces (views into the
        batched head outputs), so downstream code needs no changes.
        `autocast_bf16` overrides the model's `infer_autocast_bf16`
        for this call (the inference server's own switch); `packed`
        overrides the packed-trunk selection (see forward_streams);
        `memory` holds each sample's memory state (forward_streams).
        """
        B = len(encoded_list)
        if B == 0:
            return []
        if memory is not None and len(memory) != B:
            raise ValueError(f"{len(memory)} memory states for {B} samples")
        if memory is not None:
            # A player's `MemoryState` (None state: the game-side's first
            # decision) or the state itself, as `forward` takes them.
            memory = [m if not isinstance(m, MemoryState)
                      else self.initial_memory(m.k) if m.state is None else m.state
                      for m in memory]
        if B == 1 and autocast_bf16 is None and packed is None:
            return [self.forward(encoded_list[0], memory=None if memory is None else memory[0])]
        padded = self.forward_padded(encoded_list, autocast_bf16=autocast_bf16, packed=packed,
                                     memory=memory)
        return padded.samples()

    def forward_padded(self, encoded_list, autocast_bf16: Optional[bool] = None,
                       packed: Optional[bool] = None,
                       memory: Optional[Sequence[torch.Tensor]] = None) -> "PaddedOutput":
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
                out = self._forward_padded_impl(encoded_list, packed, memory)
            return out.float32()
        return self._forward_padded_impl(encoded_list, packed, memory)

    def _forward_padded_impl(self, encoded_list, packed: Optional[bool] = None,
                             memory: Optional[Sequence[torch.Tensor]] = None) -> "PaddedOutput":
        sighting, counts = pad_sighting_streams(encoded_list, self.d_model)
        return self.forward_streams(*pad_encoded_streams(encoded_list, self.d_model),
                                    packed=packed, material=material_batch(encoded_list),
                                    sighting_batch=sighting, sighting_counts=counts,
                                    memory=memory)

    def forward_streams(self, hex_batch, unit_batch, recruit_batch, global_batch,
                        end_turn_batch, sizes, packed: Optional[bool] = None,
                        material: Optional[torch.Tensor] = None, *,
                        sighting_batch: Optional[torch.Tensor] = None,
                        sighting_counts: Optional[Sequence[int]] = None,
                        memory: Optional[Sequence[torch.Tensor]] = None,
                        index: Optional[PaddedIndex] = None) -> "PaddedOutput":
        """Batched forward over already-padded streams ([B, L_max, d]
        each, WITHOUT token-kind embeddings) and per-sample sizes
        (U_b, R_b, H_b). The inference server feeds this straight from
        encoder.encode_from_raw_padded; forward_padded feeds it from
        EncodedStates. Same bf16 policy as forward_padded when called
        through it; callers that come here directly autocast themselves.

        `packed`: None selects the packed trunk where `infer_packed_trunk`
        and the flash varlen kernel apply (_packed_trunk_applies); True
        forces the packed code path (per-segment SDPA off the kernel's
        domain, for tests); False forces the padded trunk.

        The parity-memory streams: `sighting_batch` [B, S_max, d] with
        `sighting_counts` (required with `observation_parity`), and
        `memory`, each sample's state, float32 [k_b, d] (required with
        memory slots). The output then carries `belief_logits` and the
        new states in `memory`.

        `index`: the padded trunk's index arrays already on the device
        (`trunk_index`); built here otherwise."""
        if self.infer_compile_packed and not self.infer_packed_trunk:
            raise ValueError("infer_compile_packed requires infer_packed_trunk")
        extras = extra_streams(self, len(sizes), hex_batch.device, sighting_batch,
                               sighting_counts, memory)
        if packed is None:
            packed = self._packed_trunk_applies(hex_batch)
        if packed:
            if index is not None:
                raise ValueError("a padded trunk index reached the packed trunk")
            return self._forward_streams_packed(hex_batch, unit_batch, recruit_batch,
                                                global_batch, end_turn_batch, sizes,
                                                material=material, extras=extras)
        d = self.d_model
        device = hex_batch.device
        U_max, R_max, H_max = unit_batch.size(1), recruit_batch.size(1), hex_batch.size(1)
        S_max, K_max = (extras.S_max, extras.K_max) if extras is not None else (0, 0)
        kk = self.token_kind_embed.weight
        if H_max:
            hex_batch = hex_batch + kk[TokenKind.HEX]
        if U_max:
            unit_batch = unit_batch + kk[TokenKind.UNIT]
        if R_max:
            recruit_batch = recruit_batch + kk[TokenKind.RECRUIT]
        global_batch = global_batch + kk[TokenKind.GLOBAL]
        end_turn_batch = end_turn_batch + kk[TokenKind.END_TURN]
        blocks = [hex_batch, unit_batch, recruit_batch]
        if S_max:
            blocks.append(extras.sighting + kk[TokenKind.SIGHTING])
        if K_max:
            blocks.append(self.slot_memory.tokens(extras.memory) + kk[TokenKind.MEMORY])
        x = torch.cat(blocks + [global_batch, end_turn_batch], dim=1)

        # Key-padding mask and the compact actor gather index, built
        # host-side in numpy (padded_streams.padded_trunk_index) and
        # moved once, or given already on the device.
        if index is None:
            pad_np, idx_np, kind_np = padded_trunk_index(
                sizes, H_max, U_max, R_max,
                sightings=extras.sighting_counts if extras is not None else None, S_max=S_max,
                memory=extras.memory_counts if extras is not None else None, K_max=K_max)
            pad_mask = torch.from_numpy(pad_np).to(device)
            actor_idx = torch.from_numpy(idx_np).to(device)
            actor_kind = torch.from_numpy(kind_np)          # stays on CPU
        else:
            index.check((H_max, U_max, R_max, S_max, K_max),
                        batch_counts(sizes, extras.sighting_counts if extras is not None else None,
                                     extras.memory_counts if extras is not None else None))
            pad_mask, actor_idx, actor_kind = index.pad_mask, index.actor, index.actor_kind

        x = self.encoder(x, src_key_padding_mask=pad_mask)   # [B, seq_len, d]
        o = H_max + U_max + R_max + S_max + K_max           # the global token's slot
        hex_ctx = x[:, :H_max]
        global_ctx = x[:, o:o + 1]                           # [B, 1, d]
        actor_ctx = torch.gather(x, 1, actor_idx.unsqueeze(-1).expand(-1, -1, d))  # [B, A_max, d]
        new_memory = None
        if extras is not None and extras.memory is not None:
            new_memory = self.slot_memory.write(x[:, o - K_max:o], extras.memory)
        return self._heads(actor_ctx, hex_ctx, global_ctx, actor_kind, sizes,
                           unit_ctx=x[:, H_max:H_max + U_max] if self.has_gbc else None,
                           material=material, memory_padded=new_memory,
                           memory_counts=extras.memory_counts if extras is not None else None)

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
                                end_turn_batch, sizes,
                                material: Optional[torch.Tensor] = None,
                                extras: Optional[ExtraStreams] = None) -> "PaddedOutput":
        """forward_streams on the packed layout (design note section 5.3;
        the index arrays follow section 4.3). One gather packs the real
        tokens of the padded streams into [total, d] and one embedding
        lookup adds the token kinds; _packed_trunk_heads does the rest.
        Every index array is built host-side and shipped in one pinned
        non-blocking copy: nothing here synchronizes with the host."""
        d = self.d_model
        B = len(sizes)
        H_max, U_max, R_max = hex_batch.size(1), unit_batch.size(1), recruit_batch.size(1)
        S_max, K_max = (extras.S_max, extras.K_max) if extras is not None else (0, 0)
        layout = build_packed_layout(
            sizes, H_max, U_max, R_max, TokenKind, ActorKind,
            sightings=extras.sighting_counts if extras is not None else None,
            memory=extras.memory_counts if extras is not None else None,
            S_max=S_max, K_max=K_max)
        index = layout.to_device(hex_batch.device)
        blocks = [hex_batch, unit_batch, recruit_batch]
        if S_max:
            blocks.append(extras.sighting)
        if K_max:
            blocks.append(self.slot_memory.tokens(extras.memory))
        L = H_max + U_max + R_max + S_max + K_max + 2
        padded = torch.cat(blocks + [global_batch, end_turn_batch], dim=1).reshape(B * L, d)
        x = padded.index_select(0, index.src) + self.token_kind_embed(index.kind)   # [total, d]
        return self._packed_trunk_heads(x, index, layout, sizes, H_max, U_max, R_max,
                                        material=material, extras=extras)

    def forward_embedded(self, streams: EmbeddedStreams,
                         packed: Optional[bool] = None,
                         material: Optional[torch.Tensor] = None,
                         memory: Optional[Sequence[torch.Tensor]] = None,
                         index=None) -> "PaddedOutput":
        """Batched forward over stream-ordered token embeddings
        (encoder.encode_from_raw_embedded; the server's packed-embed
        path and the imitation trainer's). With the packed trunk, one
        gather orders the rows into the packed layout (the memory rows
        inserted before the globals) and no padded tensor is built at
        all. With the padded trunk, one gather lays them out as the
        padded streams (zeros at the pads, as pad_sequence fills them)
        and forward_streams runs unchanged. Same PaddedOutput as
        forward_streams on the same streams; `packed`, `material` and
        `memory` as in forward_streams, the sighting counts from
        `streams`. `index`: the batch's `trunk_index` for this trunk,
        already on the device (a caller that runs the forward twice builds
        it once); built here otherwise."""
        if self.infer_compile_packed and not self.infer_packed_trunk:
            raise ValueError("infer_compile_packed requires infer_packed_trunk")
        tokens, sizes = streams.tokens, streams.sizes
        sight_counts = streams.sighting_counts
        B, d = len(sizes), self.d_model
        H_max, U_max, R_max, S_max, _ = stream_widths(sizes, sight_counts)
        check_sighting_stream(self, sight_counts, B)
        if packed is None:
            packed = self._packed_trunk_applies(tokens)
        if packed:
            if index is not None and not isinstance(index, PackedEmbeddedIndex):
                raise ValueError("a padded trunk index reached the packed trunk")
            return self._forward_embedded_packed(streams, H_max, U_max, R_max, S_max,
                                                 material, memory, prebuilt=index)
        if index is not None and not isinstance(index, PaddedIndex):
            raise ValueError("a packed trunk index reached the padded trunk")
        L = H_max + U_max + R_max + S_max + 2
        if index is None:
            idx = host_to_device(padded_gather_index(sizes, H_max, U_max, R_max,
                                                     sightings=sight_counts, S_max=S_max),
                                 tokens.device)
        else:
            idx = index.gather
        rows = torch.cat([tokens, tokens.new_zeros(1, d)]).index_select(0, idx).view(B, L, d)
        o = H_max + U_max + R_max
        g = o + S_max                                        # the global token's slot
        return self.forward_streams(rows[:, :H_max], rows[:, H_max:H_max + U_max],
                                    rows[:, H_max + U_max:o], rows[:, g:g + 1], rows[:, g + 1:],
                                    sizes, packed=False, material=material,
                                    sighting_batch=rows[:, o:g] if sight_counts is not None else None,
                                    sighting_counts=sight_counts, memory=memory, index=index)

    def trunk_index(self, sizes, sighting_counts: Optional[Sequence[int]],
                    memory_counts: Optional[Sequence[int]], device: torch.device,
                    packed: bool):
        """The index arrays of a forward_embedded call on a batch of these
        sizes (U_b, R_b, H_b), sighting counts and memory slot counts (None
        for a stream the model does not read), on `device`, built host-side
        and copied without a host synchronization: a `PaddedIndex` for the
        padded trunk, a `PackedEmbeddedIndex` for the packed one."""
        if not packed:
            return padded_index(sizes, sightings=sighting_counts, memory=memory_counts, device=device)
        H_max, U_max, R_max, S_max, K_max = stream_widths(sizes, sighting_counts, memory_counts)
        layout = build_packed_layout(sizes, H_max, U_max, R_max, TokenKind, ActorKind,
                                     source="streams", sightings=sighting_counts,
                                     memory=memory_counts, S_max=S_max, K_max=K_max)
        rows = (host_to_device(stream_row_index(memory_counts, K_max), device)
                if memory_counts is not None and K_max else None)
        return PackedEmbeddedIndex(layout=layout, index=layout.to_device(device), memory_rows=rows,
                                   counts=batch_counts(sizes, sighting_counts, memory_counts))

    def _forward_embedded_packed(self, streams: EmbeddedStreams, H_max: int, U_max: int,
                                 R_max: int, S_max: int, material, memory,
                                 prebuilt: Optional[PackedEmbeddedIndex] = None) -> "PaddedOutput":
        """forward_embedded on the packed trunk: the memory tokens join the
        stream rows before the globals, one gather orders every row into
        the packed layout."""
        tokens, sizes = streams.tokens, streams.sizes
        mem = memory_batch(self, memory, len(sizes), tokens.device)
        n_before_globals = sum(u + r + h for u, r, h in sizes) + sum(streams.sighting_counts or ())
        if prebuilt is not None:
            _check_packed_index(prebuilt, sizes, streams.sighting_counts, mem)
        with region("order"):
            if mem is not None and mem.K_max:
                rows = None if prebuilt is None else prebuilt.memory_rows
                tokens = torch.cat([tokens[:n_before_globals], self.slot_memory.stream_rows(mem, rows),
                                    tokens[n_before_globals:]])
        if prebuilt is None:
            layout = build_packed_layout(sizes, H_max, U_max, R_max, TokenKind, ActorKind,
                                         source="streams", sightings=streams.sighting_counts,
                                         memory=mem.counts if mem is not None else None,
                                         S_max=S_max, K_max=mem.K_max if mem is not None else 0)
            index = layout.to_device(tokens.device)
        else:
            layout, index = prebuilt.layout, prebuilt.index
        with region("order"):
            x = tokens.index_select(0, index.src) + self.token_kind_embed(index.kind)
        extras = (ExtraStreams(sighting=None, sighting_counts=streams.sighting_counts, memory=mem)
                  if streams.sighting_counts is not None or mem is not None else None)
        return self._packed_trunk_heads(x, index, layout, sizes, H_max, U_max, R_max,
                                        material=material, extras=extras)

    def _packed_trunk_heads(self, x, index, layout, sizes, H_max, U_max, R_max,
                            material: Optional[torch.Tensor] = None,
                            extras: Optional[ExtraStreams] = None) -> "PaddedOutput":
        """The trunk on packed tokens x [total, d] (token kinds added),
        then the heads on the actor / hex / global (and unit, for GBC)
        contexts laid out in the padded shapes by index_select, so the
        heads and PaddedOutput are the padded path's; the memory write
        on the memory slots' outputs."""
        if not getattr(self, "_packed_trunk_checked", False):
            check_packed_trunk_supported(self.encoder)
            self._packed_trunk_checked = True
        d = self.d_model
        B = len(sizes)
        A_max = U_max + R_max + 1
        if self.infer_compile_packed and self.training:
            raise ValueError("the compiled packed trunk (infer_compile_packed) serves inference only: its "
                             "attention op has no backward and its weights are a frozen copy")
        with region("trunk"):
            x = self._inference_layers(x, index)
        with region("gather"):
            actor_ctx = x.index_select(0, index.actor).view(B, A_max, d)
            hex_ctx = x.index_select(0, index.hex).view(B, H_max, d)
            global_ctx = x.index_select(0, index.glob).view(B, 1, d)
            unit_ctx = x.index_select(0, index.unit).view(B, U_max, d) if self.has_gbc else None
        new_memory = None
        if extras is not None and extras.memory is not None:
            with region("memory_write"):
                h_memory = x.index_select(0, index.memory).view(B, extras.K_max, d)
                new_memory = self.slot_memory.write(h_memory, extras.memory)
        with region("heads"):
            return self._heads(actor_ctx, hex_ctx, global_ctx, torch.from_numpy(layout.actor_kind),
                               sizes, unit_ctx, material=material, memory_padded=new_memory,
                               memory_counts=extras.memory_counts if extras is not None else None)

    def packed_step_trunk(self, before: Sequence[torch.Tensor], after: Sequence[torch.Tensor],
                          memory: MemoryBatch, index: PackedEmbeddedIndex
                          ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """One time step of the window-batched sequence trainer
        (tools/sequence_window_batched.py): the packed trunk on the step's
        stream tokens, `before` the memory (hex, unit, recruit and sighting
        rows) and `after` it (the globals and the end_turn row), with the
        memory's tokens between them, and the memory's write. Returns the
        trunk's output [total, d] in the packed layout of `index` and the new
        memory [B, K_max, d]; the heads run later, over a window's outputs
        (`window_heads`)."""
        if not getattr(self, "_packed_trunk_checked", False):
            check_packed_trunk_supported(self.encoder)
            self._packed_trunk_checked = True
        if self.infer_compile_packed or self.has_gbc:
            raise ValueError("the window-batched trunk runs the eager packed trunk of a model without GBC heads")
        rows = [self.slot_memory.stream_rows(memory, index.memory_rows)] if memory.K_max else []
        tokens = torch.cat(list(before) + rows + list(after))
        x = tokens.index_select(0, index.index.src) + self.token_kind_embed(index.index.kind)
        x = self._packed_layers(x, index.index)
        new_memory = None
        if memory.K_max:
            h_memory = x.index_select(0, index.index.memory).view(len(memory.counts), memory.K_max, self.d_model)
            new_memory = self.slot_memory.write(h_memory, memory)
        return x, new_memory

    def set_trunk_mode(self, mode: Optional[TrunkMode]) -> None:
        """The packed trunk's training variants from here on (None or an
        inactive mode: packed_trunk as before); a compiled mode builds its
        CompiledBlock, which compiles at its first call."""
        if mode is not None and mode.active:
            check_packed_trunk_supported(self.encoder)
            check_trunk_mode_supported(self.encoder, mode)
        self.trunk_mode = mode if mode is not None and mode.active else None
        self._trunk_block = (CompiledBlock(mode.compile_backend)
                             if self.trunk_mode is not None and mode.compile_backend is not None else None)

    def trunk_block_stats(self) -> Optional[dict]:
        return None if self._trunk_block is None else self._trunk_block.stats()

    def _packed_layers(self, x: torch.Tensor, index) -> torch.Tensor:
        """The packed trunk's layers on x [total, d]: packed_trunk, or its
        training variant when one is set (set_trunk_mode)."""
        if self.trunk_mode is None:
            return packed_trunk(self.encoder, x, index)
        return run_trunk(self.encoder, x, index, self.trunk_mode, self._trunk_block)

    def _inference_layers(self, x: torch.Tensor, index) -> torch.Tensor:
        """The packed trunk's layers on x [total, d] for _packed_trunk_heads:
        the fp8 loop or the inference trunk modes where they apply, the
        compiled loop with infer_compile_packed, _packed_layers otherwise."""
        if self.infer_fp8 and self._bf16_inference(x):
            return self._run_fp8_trunk(x, index)
        mode = self._inference_trunk_mode(x)
        if mode is not None:
            return self._run_inference_trunk_mode(x, index, mode)
        if self.infer_compile_packed:
            return self._run_compiled_packed_trunk(x, index)
        return self._packed_layers(x, index)

    def window_heads(self, x: torch.Tensor, index: WindowHeadIndex) -> "PaddedOutput":
        """The heads over a window's trunk outputs `x` (each time step's
        `packed_step_trunk` output, concatenated): every row of every step
        laid out at the window's widths (`packed_trunk.window_head_index`),
        the padded slots pointing at their row's end_turn token as in one
        batch's layout."""
        d = self.d_model
        n = len(index.sizes)
        actor_ctx = x.index_select(0, index.actor).view(n, index.A_max, d)
        hex_ctx = x.index_select(0, index.hex).view(n, index.H_max, d)
        global_ctx = x.index_select(0, index.glob).view(n, 1, d)
        return self._heads(actor_ctx, hex_ctx, global_ctx, index.actor_kind, index.sizes, None)

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
        self._weight_copies.clear()
        self._compiled_weights = None
        self._fp8_compile = None             # rebuilt from these options at its first call
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
                                     packed=True, **self._warmup_extras(B, device))
        return self.packed_compile_stats()

    def _warmup_extras(self, B: int, device: torch.device) -> dict:
        """forward_streams' parity-memory arguments for a warmup batch: a
        few sightings, and every slot active."""
        kw = {}
        if self.observation_parity:
            kw["sighting_counts"] = [b % 3 for b in range(B)]
            kw["sighting_batch"] = torch.randn(B, 2, self.d_model, device=device)
        if self.slot_memory is not None:
            kw["memory"] = [self.initial_memory(self.memory_slots).detach() for _ in range(B)]
        return kw

    @property
    def packed_compile_active(self) -> bool:
        return self._packed_compile is not None and self._packed_compile.active

    def packed_compile_stats(self) -> dict:
        """The compiled loop's stats: the fp8 loop's under infer_fp8."""
        if self.infer_fp8:
            if self._fp8_compile is None:
                return {"active": False, "fp8": True}
            return {**self._fp8_compile.stats(), "fp8": True, "weights_dtype": "float8_e4m3fn"}
        if self._packed_compile is None:
            return {"active": False}
        stats = self._packed_compile.stats()
        w = self._compiled_weights
        stats["weights_dtype"] = str(w.dtype).replace("torch.", "") if w is not None else None
        return stats

    def _bf16_inference(self, x: torch.Tensor) -> bool:
        """Eval mode under a bf16 autocast on x's device: where the
        inference trunk switches apply."""
        dev = x.device.type
        return (not self.training and torch.is_autocast_enabled(dev)
                and torch.get_autocast_dtype(dev) == torch.bfloat16)

    def _inference_trunk_mode(self, x: torch.Tensor) -> Optional[TrunkMode]:
        """infer_bf16_residual and infer_fused_relu as a trainer's trunk mode,
        or None where they do not apply: both off, training mode, or no
        bf16 autocast."""
        if not (self.infer_bf16_residual or self.infer_fused_relu) or not self._bf16_inference(x):
            return None
        return TrunkMode(bf16_residual=self.infer_bf16_residual, fused_relu=self.infer_fused_relu)

    def fp8_weights(self, device: torch.device) -> Fp8TrunkWeights:
        """The fp8 loop's weight copy (wesnoth_ai/fp8_inference.py): built on
        first use and for a new device, refreshed in place when the weight
        version moved."""
        w = self._fp8_weights
        if w is None or w.device != device:
            check_fp8_inference_supported(self.encoder, device)
            w = self._fp8_weights = Fp8TrunkWeights.build(self.encoder, device, self._weights_version)
        elif w.version != self._weights_version:
            w.refresh(self.encoder, self._weights_version)
        return w

    def _run_fp8_trunk(self, x: torch.Tensor, index) -> torch.Tensor:
        """The fp8 layer loop on the bf16 stream, eager, or compiled with
        infer_compile_packed (its own CompiledPackedTrunk, of the bf16
        loop's backend and mode). Returns bf16 rows; the heads read them
        under the caller's autocast."""
        w = self.fp8_weights(x.device)
        with torch.autocast(x.device.type, enabled=False):
            x = x.to(torch.bfloat16)
            if not self.infer_compile_packed:
                eps = self.encoder.layers[0].norm1.eps
                return make_fp8_trunk_layers(eps)(x, index.cu_seqlens, index.max_len, w.layers, w.final_norm)
            if self._fp8_compile is None:
                if self._packed_compile is None:
                    self.configure_packed_compile()
                layer = self.encoder.layers[0]
                self._fp8_compile = CompiledPackedTrunk(
                    layer.activation, layer.norm1.eps, backend=self._packed_compile.backend,
                    mode=self._packed_compile.mode, loop=make_fp8_trunk_layers(layer.norm1.eps))
            return self._fp8_compile.run(x, index, w)

    def _run_inference_trunk_mode(self, x: torch.Tensor, index, mode: TrunkMode) -> torch.Tensor:
        """`mode` over the bf16 weight copy, its layer norms in the residual
        stream's dtype: run_trunk eager, or the compiled loop with
        infer_compile_packed for the bf16 stream (inductor fuses the ReLU
        itself, so the fused ReLU is not compiled). Returns rows in the
        residual stream's dtype; the heads read them under the caller's
        autocast, the memory's write in float32."""
        stream = torch.bfloat16 if mode.bf16_residual else torch.float32
        w = self._weight_copy(torch.bfloat16, stream, x.device)
        if not self.infer_compile_packed:
            return run_trunk(self.encoder, x, index, mode, weights=w)
        if mode.fused_relu:
            raise ValueError("infer_fused_relu is an eager call: the compiled loop fuses the ReLU itself")
        with torch.autocast(x.device.type, enabled=False):
            return self._run_compiled(x.to(torch.bfloat16), index, w)

    def _run_compiled_packed_trunk(self, x, index):
        """The compiled loop over the native-dtype weight copy: the
        autocast dtype where one is active (bf16 on the server), the
        input's otherwise (fp32 on CPU), the layer norms in float32."""
        dev = x.device.type
        dtype = torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else x.dtype
        return self._run_compiled(x, index, self._weight_copy(dtype, torch.float32, x.device))

    def _run_compiled(self, x: torch.Tensor, index, w: PackedTrunkWeights) -> torch.Tensor:
        if self._packed_compile is None:
            self.configure_packed_compile()
        self._compiled_weights = w
        return self._packed_compile.run(x, index, w)

    def _weight_copy(self, dtype: torch.dtype, norm_dtype: torch.dtype,
                     device: torch.device) -> PackedTrunkWeights:
        """The frozen copy of the trunk's weights, linears in `dtype` and
        layer norms in `norm_dtype`: built on first use and for a new
        device, refreshed in place when the weight version moved."""
        key = (dtype, norm_dtype)
        w = self._weight_copies.get(key)
        if w is None or w.device != device:
            w = self._weight_copies[key] = PackedTrunkWeights.build(
                self.encoder, dtype, device, self._weights_version, norm_dtype=norm_dtype)
        elif w.version != self._weights_version:
            w.refresh(self.encoder, self._weights_version)
        return w

    def _heads(self, actor_ctx, hex_ctx, global_ctx, actor_kind, sizes,
               unit_ctx, material: Optional[torch.Tensor] = None,
               memory_padded: Optional[torch.Tensor] = None,
               memory_counts: Optional[List[int]] = None) -> "PaddedOutput":
        """The four heads on contextualized actor [B, A_max, d], hex
        [B, H_max, d] and global [B, 1, d] rows, whichever trunk produced
        them; the belief head on the hex rows when the model has one.
        The new memory states ride along."""
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

        if self.infer_priors_only and not self.training:
            value_logits = value = cliffness = aux_score = moves_left = belief_logits = None
        else:
            g = global_ctx.squeeze(1)
            value_logits = self.value_head(self._value_input(g, material))   # [B, K]
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
            belief_logits = (self.belief_head(hex_ctx).squeeze(-1)             # [B, H_max]
                             if self.belief_head is not None else None)
        return PaddedOutput(
            actor_logits=actor_logits, actor_kind=actor_kind, type_logits=type_logits,
            target_logits=target_logits, weapon_logits=weapon_logits, value=value,
            value_logits=value_logits, cliffness=cliffness, aux_score=aux_score,
            moves_left=moves_left, sizes=[tuple(s) for s in sizes],
            unit_ctx=unit_ctx,
            hex_ctx=hex_ctx if self.has_gbc else None,
            global_ctx=global_ctx if self.has_gbc else None,
            belief_logits=belief_logits, memory_padded=memory_padded,
            memory_counts=None if memory_counts is None else list(memory_counts))
