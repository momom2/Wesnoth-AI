"""Padded-stream plumbing around WesnothModel's batched forward.

forward_streams takes five padded streams ([B, L_max, d] each, no
token-kind embeddings) plus per-sample sizes (U_b, R_b, H_b). This
module builds them and the padded trunk's index arrays:

- pad_encoded_streams: EncodedStates -> the padded streams
  (forward_padded's input).
- pad_sighting_streams: the parity observation's sighting stream of
  EncodedStates, padded, with its per-sample counts.
- extra_streams (with check_sighting_stream and memory_batch): the
  parity-memory recipe's sighting and memory streams of one forward,
  checked against the model's flags.
- padded_trunk_index: the key-padding mask, the compact actor gather
  index and the actor kinds of the padded trunk, host-side numpy.
- random_padded_streams: random streams for given sizes
  (warmup_packed_compile and the packed-trunk tests).

The packed layout's counterpart is wesnoth_ai/packed_trunk.py
(build_packed_layout, padded_gather_index). wesnoth_ai/model.py
re-exports random_padded_streams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch

from wesnoth_ai.memory import MemoryBatch
from wesnoth_ai.model_output import ActorKind


@dataclass
class ExtraStreams:
    """The parity-memory recipe's two streams of one forward, between the
    recruits and the global token: the sighting tokens ([B, S_max, d]
    padded, None on the packed-embed path, where they are stream rows)
    with their per-sample counts, and the memory states. None fields:
    the model has no such stream."""
    sighting: Optional[torch.Tensor]
    sighting_counts: Optional[List[int]]
    memory: Optional[MemoryBatch]

    @property
    def S_max(self) -> int:
        if self.sighting is not None:
            return int(self.sighting.size(1))
        return max(self.sighting_counts, default=0) if self.sighting_counts else 0

    @property
    def K_max(self) -> int:
        return self.memory.K_max if self.memory is not None else 0

    @property
    def memory_counts(self) -> Optional[List[int]]:
        return self.memory.counts if self.memory is not None else None


def check_sighting_stream(model, counts: Optional[Sequence[int]], B: int) -> None:
    """The sighting stream is present exactly when `model` reads the
    parity observation (a parity encoder always builds it, empty or
    not)."""
    if counts is None:
        if model.observation_parity:
            raise ValueError("a model built with observation_parity needs the sighting "
                             "stream: encode with an observation_parity encoder")
        return
    if not model.observation_parity:
        raise ValueError("a sighting stream reached a model built without observation_parity")
    if len(counts) != B:
        raise ValueError(f"{len(counts)} sighting counts for {B} samples")


def memory_batch(model, memory: Optional[Sequence[torch.Tensor]], B: int,
                 device: torch.device) -> Optional[MemoryBatch]:
    """A batch's memory states padded for the trunk, checked against
    `model`: required by a model with memory slots, refused by one
    without."""
    if model.slot_memory is None:
        if memory is not None:
            raise ValueError("memory states passed to a model without memory slots")
        return None
    if memory is None:
        raise ValueError(
            f"this model has memory_slots={model.memory_slots}: every forward takes each "
            f"sample's memory state (initial_memory(k) at a game-side's first decision, "
            f"the returned memory after it; [0, d] for k = 0)")
    if len(memory) != B:
        raise ValueError(f"{len(memory)} memory states for {B} samples")
    return model.slot_memory.batch(memory, device)


def extra_streams(model, B: int, device: torch.device, sighting_batch: Optional[torch.Tensor],
                  sighting_counts: Optional[Sequence[int]],
                  memory: Optional[Sequence[torch.Tensor]]) -> Optional[ExtraStreams]:
    """The parity-memory streams of a forward on padded streams, checked
    against `model`; None for obs8's model."""
    if (sighting_batch is None) != (sighting_counts is None):
        raise ValueError("the sighting stream needs both its tokens and its counts")
    check_sighting_stream(model, sighting_counts, B)
    mem = memory_batch(model, memory, B, device)
    if sighting_batch is not None and sighting_batch.size(1) < max(sighting_counts, default=0):
        raise ValueError("sighting counts exceed the sighting stream's width")
    if sighting_batch is None and mem is None:
        return None
    return ExtraStreams(sighting=sighting_batch,
                        sighting_counts=None if sighting_counts is None else list(sighting_counts),
                        memory=mem)


def material_batch(encoded_list):
    """[B, 1] material of the batch, or None when any state lacks it
    (a `value_material` model then refuses the forward)."""
    if any(getattr(e, "material", None) is None for e in encoded_list):
        return None
    return torch.cat([e.material for e in encoded_list], dim=0)


def pad_encoded_streams(encoded_list, d_model: int):
    """The five padded streams of forward_streams' signature plus the
    per-sample sizes, from EncodedStates (batch dim 1 each). Pad
    positions are zeros (pad_sequence's fill)."""
    device = encoded_list[0].hex_tokens.device
    dtype = encoded_list[0].hex_tokens.dtype
    B = len(encoded_list)
    Us = [e.unit_tokens.size(1) for e in encoded_list]
    Rs = [e.recruit_tokens.size(1) for e in encoded_list]
    Hs = [e.hex_tokens.size(1) for e in encoded_list]

    def _padded(tokens, L_max):
        if L_max == 0:
            return torch.zeros(B, 0, d_model, device=device, dtype=dtype)
        return torch.nn.utils.rnn.pad_sequence([t.squeeze(0) for t in tokens],
                                               batch_first=True)

    return (_padded([e.hex_tokens for e in encoded_list], max(Hs)),
            _padded([e.unit_tokens for e in encoded_list], max(Us)),
            _padded([e.recruit_tokens for e in encoded_list], max(Rs)),
            torch.cat([e.global_token for e in encoded_list], dim=0),
            torch.cat([e.end_turn_token for e in encoded_list], dim=0),
            list(zip(Us, Rs, Hs)))


def pad_sighting_streams(encoded_list, d_model: int):
    """The padded sighting stream ([B, S_max, d], zeros at the pads) and
    the per-sample counts S_b of states encoded with the parity
    observation; (None, None) when no state carries sightings. A batch
    that mixes the two encodings is refused."""
    tokens = [getattr(e, "sighting_tokens", None) for e in encoded_list]
    if all(t is None for t in tokens):
        return None, None
    if any(t is None for t in tokens):
        raise ValueError("a batch mixes states with and without the sighting stream "
                         "(encoders with and without observation_parity)")
    counts = [t.size(1) for t in tokens]
    if max(counts) == 0:
        first = tokens[0]
        return first.new_zeros(len(tokens), 0, d_model), counts
    return torch.nn.utils.rnn.pad_sequence([t.squeeze(0) for t in tokens],
                                           batch_first=True), counts


def padded_trunk_index(sizes, H_max: int, U_max: int, R_max: int, *,
                       sightings=None, S_max: int = 0, memory=None, K_max: int = 0):
    """Host-side index arrays of the padded trunk for per-sample sizes
    (U_b, R_b, H_b) over the sequence [hex | unit | recruit | sighting |
    memory | global | end_turn] padded to H_max, U_max, R_max, S_max,
    K_max (the sighting and memory blocks are the parity-memory
    recipe's; `sightings` and `memory` give their per-sample counts): the
    key-padding mask [B, seq_len] (global and end_turn are never masked,
    so every row keeps >= 1 real position), the compact actor gather
    index [B, A_max] laying each sample's actors out as units |
    recruits | end_turn (pad slots point at the end_turn row), and the
    actor kinds [B, A_max]."""
    B = len(sizes)
    Us = [s[0] for s in sizes]
    Rs = [s[1] for s in sizes]
    Hs = [s[2] for s in sizes]
    Ss = sightings if sightings is not None else [0] * B
    Ks = memory if memory is not None else [0] * B
    seq_len = H_max + U_max + R_max + S_max + K_max + 2
    A_max = U_max + R_max + 1
    pad_np = np.zeros((B, seq_len), dtype=bool)
    idx_np = np.full((B, A_max), seq_len - 1, dtype=np.int64)
    kind_np = np.full((B, A_max), ActorKind.END_TURN, dtype=np.int64)
    s0 = H_max + U_max + R_max              # the first sighting slot
    k0 = s0 + S_max                         # the first memory slot
    for b in range(B):
        U_b, R_b, H_b = Us[b], Rs[b], Hs[b]
        pad_np[b, H_b:H_max] = True
        pad_np[b, H_max + U_b:H_max + U_max] = True
        pad_np[b, H_max + U_max + R_b:s0] = True
        pad_np[b, s0 + Ss[b]:k0] = True
        pad_np[b, k0 + Ks[b]:k0 + K_max] = True
        idx_np[b, :U_b] = np.arange(H_max, H_max + U_b)
        idx_np[b, U_b:U_b + R_b] = np.arange(H_max + U_max, H_max + U_max + R_b)
        kind_np[b, :U_b] = ActorKind.UNIT
        kind_np[b, U_b:U_b + R_b] = ActorKind.RECRUIT
    return pad_np, idx_np, kind_np


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
