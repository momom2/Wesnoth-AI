"""Padded-stream plumbing around WesnothModel's batched forward.

forward_streams takes five padded streams ([B, L_max, d] each, no
token-kind embeddings) plus per-sample sizes (U_b, R_b, H_b). This
module builds them and the padded trunk's index arrays:

- pad_encoded_streams: EncodedStates -> the padded streams
  (forward_padded's input).
- padded_trunk_index: the key-padding mask, the compact actor gather
  index and the actor kinds of the padded trunk, host-side numpy.
- random_padded_streams: random streams for given sizes
  (warmup_packed_compile and the packed-trunk tests).

The packed layout's counterpart is wesnoth_ai/packed_trunk.py
(build_packed_layout, padded_gather_index). wesnoth_ai/model.py
re-exports random_padded_streams.
"""

from __future__ import annotations

import numpy as np
import torch

from wesnoth_ai.model_output import ActorKind


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


def padded_trunk_index(sizes, H_max: int, U_max: int, R_max: int):
    """Host-side index arrays of the padded trunk for per-sample sizes
    (U_b, R_b, H_b) over the sequence [hex | unit | recruit | global |
    end_turn] padded to H_max, U_max, R_max: the key-padding mask
    [B, seq_len] (global and end_turn are never masked, so every row
    keeps >= 1 real position), the compact actor gather index
    [B, A_max] laying each sample's actors out as units | recruits |
    end_turn (pad slots point at the end_turn row), and the actor
    kinds [B, A_max]."""
    B = len(sizes)
    Us = [s[0] for s in sizes]
    Rs = [s[1] for s in sizes]
    Hs = [s[2] for s in sizes]
    seq_len = H_max + U_max + R_max + 2
    A_max = U_max + R_max + 1
    pad_np = np.zeros((B, seq_len), dtype=bool)
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
