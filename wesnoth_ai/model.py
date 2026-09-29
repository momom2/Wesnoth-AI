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

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from wesnoth_ai.encoder import EncodedState
from wesnoth_ai.memory import MemoryBatch, SlotMemory
from wesnoth_ai.model_output import (
    MAX_ATTACKS, VALUE_N_ATOMS, VALUE_V_MAX, VALUE_V_MIN, ActorKind, ModelOutput,
    PaddedOutput, TokenKind, UnitActionType,
)
from wesnoth_ai.packed_trunk import (
    CompiledPackedTrunk, EmbeddedStreams, PackedTrunkWeights, build_packed_layout,
    check_packed_trunk_supported, flash_varlen_applies, packed_trunk, padded_gather_index,
)
from wesnoth_ai.material import MATERIAL_SCALE
from wesnoth_ai.padded_streams import (ExtraStreams, material_batch, pad_encoded_streams,
                                       pad_sighting_streams, padded_trunk_index,
                                       random_padded_streams)

__all__ = [
    "WesnothModel",
    # Re-exports: importers and pickles address these through this module.
    "ModelOutput", "PaddedOutput", "TokenKind", "ActorKind", "UnitActionType",
    "MAX_ATTACKS", "VALUE_N_ATOMS", "VALUE_V_MIN", "VALUE_V_MAX", "random_padded_streams",
]


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

    def _memory_batch(self, memory: Optional[Sequence[torch.Tensor]], B: int,
                      device: torch.device) -> Optional[MemoryBatch]:
        """The batch's memory states, checked against the model: required
        by a model with memory slots, refused by one without."""
        if self.slot_memory is None:
            if memory is not None:
                raise ValueError("memory states passed to a model without memory slots")
            return None
        if memory is None:
            raise ValueError(
                f"this model has memory_slots={self.memory_slots}: every forward takes each "
                f"sample's memory state (initial_memory(k) at a game-side's first decision, "
                f"the returned memory after it; [0, d] for k = 0)")
        if len(memory) != B:
            raise ValueError(f"{len(memory)} memory states for {B} samples")
        return self.slot_memory.batch(memory, device)

    def _check_sighting_stream(self, counts: Optional[Sequence[int]], B: int) -> None:
        """The sighting stream is present exactly when the model reads the
        parity observation (a parity encoder always builds it, empty or
        not)."""
        if counts is None:
            if self.observation_parity:
                raise ValueError("a model built with observation_parity needs the sighting "
                                 "stream: encode with an observation_parity encoder")
            return
        if not self.observation_parity:
            raise ValueError("a sighting stream reached a model built without observation_parity")
        if len(counts) != B:
            raise ValueError(f"{len(counts)} sighting counts for {B} samples")

    def _extra_streams(self, B: int, device: torch.device,
                       sighting_batch: Optional[torch.Tensor],
                       sighting_counts: Optional[Sequence[int]],
                       memory: Optional[Sequence[torch.Tensor]]) -> Optional[ExtraStreams]:
        """The parity-memory streams of a forward on padded streams, or
        None for obs8's model."""
        if (sighting_batch is None) != (sighting_counts is None):
            raise ValueError("the sighting stream needs both its tokens and its counts")
        self._check_sighting_stream(sighting_counts, B)
        mem = self._memory_batch(memory, B, device)
        if sighting_batch is not None and sighting_batch.size(1) < max(sighting_counts, default=0):
            raise ValueError("sighting counts exceed the sighting stream's width")
        if sighting_batch is None and mem is None:
            return None
        return ExtraStreams(sighting=sighting_batch,
                            sighting_counts=None if sighting_counts is None else list(sighting_counts),
                            memory=mem)

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
                memory: Optional[torch.Tensor] = None) -> ModelOutput:
        """One state's outputs. `memory`: the side's memory state, float32
        [k, d], required by a model with memory slots; the output's
        `memory` is the state after this decision. A model reading the
        parity-memory streams runs the batched path with one sample."""
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
                        memory: Optional[Sequence[torch.Tensor]] = None) -> "PaddedOutput":
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
        new states in `memory`."""
        if self.infer_compile_packed and not self.infer_packed_trunk:
            raise ValueError("infer_compile_packed requires infer_packed_trunk")
        extras = self._extra_streams(len(sizes), hex_batch.device, sighting_batch,
                                     sighting_counts, memory)
        if packed is None:
            packed = self._packed_trunk_applies(hex_batch)
        if packed:
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
        # moved once.
        pad_np, idx_np, kind_np = padded_trunk_index(
            sizes, H_max, U_max, R_max,
            sightings=extras.sighting_counts if extras is not None else None, S_max=S_max,
            memory=extras.memory_counts if extras is not None else None, K_max=K_max)
        pad_mask = torch.from_numpy(pad_np).to(device)
        actor_idx = torch.from_numpy(idx_np).to(device)
        actor_kind = torch.from_numpy(kind_np)          # stays on CPU

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
                           material=material, memory=new_memory)

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
                         memory: Optional[Sequence[torch.Tensor]] = None) -> "PaddedOutput":
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
        `streams`."""
        if self.infer_compile_packed and not self.infer_packed_trunk:
            raise ValueError("infer_compile_packed requires infer_packed_trunk")
        tokens, sizes = streams.tokens, streams.sizes
        sight_counts = streams.sighting_counts
        B, d = len(sizes), self.d_model
        U_max, R_max, H_max = (max(s[i] for s in sizes) for i in (0, 1, 2))
        S_max = max(sight_counts, default=0) if sight_counts is not None else 0
        self._check_sighting_stream(sight_counts, B)
        if packed is None:
            packed = self._packed_trunk_applies(tokens)
        if packed:
            return self._forward_embedded_packed(streams, H_max, U_max, R_max, S_max,
                                                 material, memory)
        L = H_max + U_max + R_max + S_max + 2
        idx = torch.from_numpy(padded_gather_index(sizes, H_max, U_max, R_max,
                                                   sightings=sight_counts, S_max=S_max))
        if tokens.device.type == "cuda":
            idx = idx.pin_memory().to(tokens.device, non_blocking=True)
        elif tokens.device.type != "cpu":
            idx = idx.to(tokens.device)
        rows = torch.cat([tokens, tokens.new_zeros(1, d)]).index_select(0, idx).view(B, L, d)
        o = H_max + U_max + R_max
        g = o + S_max                                        # the global token's slot
        return self.forward_streams(rows[:, :H_max], rows[:, H_max:H_max + U_max],
                                    rows[:, H_max + U_max:o], rows[:, g:g + 1], rows[:, g + 1:],
                                    sizes, packed=False, material=material,
                                    sighting_batch=rows[:, o:g] if sight_counts is not None else None,
                                    sighting_counts=sight_counts, memory=memory)

    def _forward_embedded_packed(self, streams: EmbeddedStreams, H_max: int, U_max: int,
                                 R_max: int, S_max: int, material, memory) -> "PaddedOutput":
        """forward_embedded on the packed trunk: the memory tokens join the
        stream rows before the globals, one gather orders every row into
        the packed layout."""
        tokens, sizes = streams.tokens, streams.sizes
        mem = self._memory_batch(memory, len(sizes), tokens.device)
        if mem is not None and mem.K_max:
            n_before_globals = sum(u + r + h for u, r, h in sizes) + sum(streams.sighting_counts or ())
            tokens = torch.cat([tokens[:n_before_globals], self.slot_memory.stream_rows(mem),
                                tokens[n_before_globals:]])
        layout = build_packed_layout(sizes, H_max, U_max, R_max, TokenKind, ActorKind,
                                     source="streams", sightings=streams.sighting_counts,
                                     memory=mem.counts if mem is not None else None,
                                     S_max=S_max, K_max=mem.K_max if mem is not None else 0)
        index = layout.to_device(tokens.device)
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
        if self.infer_compile_packed:
            x = self._run_compiled_packed_trunk(x, index)
        else:
            x = packed_trunk(self.encoder, x, index)
        actor_ctx = x.index_select(0, index.actor).view(B, A_max, d)
        hex_ctx = x.index_select(0, index.hex).view(B, H_max, d)
        global_ctx = x.index_select(0, index.glob).view(B, 1, d)
        unit_ctx = x.index_select(0, index.unit).view(B, U_max, d) if self.has_gbc else None
        new_memory = None
        if extras is not None and extras.memory is not None:
            h_memory = x.index_select(0, index.memory).view(B, extras.K_max, d)
            new_memory = self.slot_memory.write(h_memory, extras.memory)
        return self._heads(actor_ctx, hex_ctx, global_ctx, torch.from_numpy(layout.actor_kind),
                           sizes, unit_ctx, material=material, memory=new_memory)

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
               unit_ctx, material: Optional[torch.Tensor] = None,
               memory: Optional[List[torch.Tensor]] = None) -> "PaddedOutput":
        """The four heads on contextualized actor [B, A_max, d], hex
        [B, H_max, d] and global [B, 1, d] rows, whichever trunk produced
        them; the belief head on the hex rows when the model has one.
        `memory`, the new states, rides along."""
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
            belief_logits=belief_logits, memory=memory)
