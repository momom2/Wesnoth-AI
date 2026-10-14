"""Inference requests and replies as raw bytes, no pickling.

The eval inference server (tools/eval_inference_server.py) and its
workers exchanged pickled objects: a request was a `PackedRequest`
(one numpy buffer and a tuple of `LeafHeader` dataclasses), a reply a
tuple of about ten numpy arrays and a `CompactActions` dataclass per
leaf. Pickling and unpickling those cost about 0.3 ms per leaf on each
side (0.19 to pickle a request, 0.11 to unpickle it, 0.18 to pickle a
reply, measured on own1's leaves on the laptop, 2026-10-14), most of it
the per-object overhead of the dataclasses and the small arrays. Here
a message is one buffer: an int64 header (per leaf, its scalars and,
per field, the dtype and shape) and the arrays' bytes, each array on an
8-byte boundary. The receiver builds the arrays as views into the
buffer it received, no copies.

The values are the same bytes either way: a request carries the fields
of `wesnoth_ai.leaf_wire` cast to their wire dtypes as `pack_request`
does, a reply the arrays of `tools.inference_seam.output_to_wire`'s
priors tuple with their own dtypes.

A message may start with `prefix` int64 words, the caller's own framing
(the server's request kind and id).
"""
from __future__ import annotations

import struct
from typing import List, Optional, Sequence, Tuple

import numpy as np

from wesnoth_ai.encoder import RawEncoded
from wesnoth_ai.leaf_wire import ALL_FIELDS, MASK_FIELDS, PARITY_FIELDS, RAW_FIELDS, _leaf_arrays
from wesnoth_ai.memory import MemoryState
from wesnoth_ai.server_priors import CompactActions, PackedMasks

# The dtypes a field may have on the wire, by code.
DTYPES: Tuple[np.dtype, ...] = tuple(np.dtype(t) for t in (
    np.bool_, np.int8, np.uint8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64))
_CODE = {dt: i for i, dt in enumerate(DTYPES)}
MAX_NDIM = 3
# Per field: dtype code (-1 for an absent field), ndim, then MAX_NDIM dims.
FIELD_WORDS = 2 + MAX_NDIM
_ABSENT = (-1,) + (0,) * (FIELD_WORDS - 1)
_ZERO_DIMS = tuple((0,) * (MAX_NDIM - nd) for nd in range(MAX_NDIM + 1))
_ALIGN = 8
_PAD = tuple(bytes(n) for n in range(_ALIGN))

# A request leaf's scalars, in header order.
REQUEST_SCALARS = ("memory_k", "our_faction_id", "their_faction_id", "hex_subset",
                   "end_turn_bias", "n_units", "n_recruits", "n_hexes")
# A reply leaf: its two scalars, then the priors tuple's arrays.
REPLY_SCALARS = ("num_units", "num_recruits")
REPLY_FIELDS = ("value", "value_logits", "cliffness", "aux_score", "moves_left", "actor_kind",
                "actor", "kind", "target", "weapon", "prior", "memory")


def _float_bits(x: float) -> int:
    return struct.unpack("<q", struct.pack("<d", x))[0]


def _bits_float(w: int) -> float:
    return struct.unpack("<d", struct.pack("<q", w))[0]


def encode_bundle(leaves: Sequence[Tuple[Sequence[int], Sequence[Optional[np.ndarray]]]],
                  prefix: Sequence[int] = ()) -> bytes:
    """One message: the `prefix` words (the caller's framing), int64 n,
    n header rows (each leaf's scalars, then per array its dtype code,
    ndim and dims), the arrays' bytes in leaf order, each on an 8-byte
    boundary. Every leaf has the same number of scalars and of arrays
    (None: absent)."""
    words: List[int] = list(prefix)
    words.append(len(leaves))
    chunks: List[object] = [None]
    for scalars, fields in leaves:
        words.extend(scalars)
        for a in fields:
            if a is None:
                words.extend(_ABSENT)
                continue
            if not a.flags.c_contiguous:
                a = np.ascontiguousarray(a)
            nd = a.ndim
            code = _CODE.get(a.dtype)
            if code is None or nd > MAX_NDIM:
                raise ValueError(f"no wire form for an array of {a.dtype} {a.shape}")
            words.append(code)
            words.append(nd)
            words.extend(a.shape)
            words.extend(_ZERO_DIMS[nd])
            chunks.append(a)
            pad = -a.nbytes % _ALIGN
            if pad:
                chunks.append(_PAD[pad])
    chunks[0] = struct.pack(f"<{len(words)}q", *words)
    return b"".join(chunks)


def decode_bundle(buf, n_scalars: int, n_fields: int, offset: int = 0
                  ) -> List[Tuple[List[int], List[Optional[np.ndarray]]]]:
    """Inverse of `encode_bundle` from byte `offset` of `buf` (bytes,
    bytearray or a uint8 array), past the prefix: per leaf its scalars
    and its arrays, views into `buf` (read-only when `buf` is)."""
    whole = np.frombuffer(buf, dtype=np.uint8)
    n = int(whole[offset:offset + 8].view(np.int64)[0])
    width = n_scalars + FIELD_WORDS * n_fields
    words = whole[offset + 8:offset + 8 * (1 + n * width)].view(np.int64).tolist()
    off = offset + 8 * (1 + n * width)
    out = []
    for i in range(n):
        row = words[i * width:(i + 1) * width]
        fields: List[Optional[np.ndarray]] = []
        j = n_scalars
        for _ in range(n_fields):
            code = row[j]
            if code < 0:
                fields.append(None)
            else:
                dt = DTYPES[code]
                shape = tuple(row[j + 2:j + 2 + row[j + 1]])
                nbytes = dt.itemsize
                for d in shape:
                    nbytes *= d
                fields.append(whole[off:off + nbytes].view(dt).reshape(shape))
                off += nbytes + (-nbytes % _ALIGN)
            j += FIELD_WORDS
        out.append((row[:n_scalars], fields))
    return out


# ---------------------------------------------------------------------
# Requests: (RawEncoded, PackedMasks[, MemoryState]) per leaf
# ---------------------------------------------------------------------

def encode_request(items: Sequence[Tuple], prefix: Sequence[int] = ()) -> bytes:
    """Actor side: B (RawEncoded, PackedMasks) pairs, or triples with the
    player's MemoryState, as one buffer (the fields `leaf_wire.
    pack_request` carries, in its dtypes)."""
    leaves = []
    for item in items:
        raw, masks = item[0], item[1]
        memory = item[2] if len(item) > 2 else None
        scalars = (-1 if memory is None else int(memory.k), int(raw.our_faction_id),
                   int(raw.their_faction_id), int(bool(raw.hex_subset)),
                   _float_bits(float(masks.end_turn_bias)), int(masks.n_units),
                   int(masks.n_recruits), int(masks.n_hexes))
        leaves.append((scalars, list(_leaf_arrays(raw, masks, memory))))
    return encode_bundle(leaves, prefix)


_RAW_NAMES = tuple(f for f, _ in RAW_FIELDS)
_MASK_NAMES = tuple(f for f, _ in MASK_FIELDS)
_PARITY_NAMES = tuple(f for f, _ in PARITY_FIELDS)


def decode_request(buf, offset: int = 0) -> List[Tuple]:
    """Server side: what `leaf_wire.unpack_request` returns, from a
    buffer `encode_request` wrote: per leaf (RawEncoded, PackedMasks),
    or with a MemoryState for a leaf with a memory (its state None at a
    game-side's first decision). The RawEncoded carries only the
    numeric streams."""
    n_raw, n_mask, n_parity = len(RAW_FIELDS), len(MASK_FIELDS), len(PARITY_FIELDS)
    out: List[Tuple] = []
    for scalars, vals in decode_bundle(buf, len(REQUEST_SCALARS), len(ALL_FIELDS), offset):
        memory_k, our_f, their_f, subset, bias, n_units, n_recruits, n_hexes = scalars
        raw = RawEncoded(
            hex_positions=None, unit_positions=None, unit_ids=None, recruit_types=None,
            our_faction_id=our_f, their_faction_id=their_f, hex_subset=bool(subset),
            **dict(zip(_RAW_NAMES, vals[:n_raw])),
            **dict(zip(_PARITY_NAMES, vals[n_raw + n_mask:n_raw + n_mask + n_parity])))
        masks = PackedMasks(end_turn_bias=_bits_float(bias), n_units=n_units,
                            n_recruits=n_recruits, n_hexes=n_hexes,
                            **dict(zip(_MASK_NAMES, vals[n_raw:n_raw + n_mask])))
        if memory_k < 0:
            out.append((raw, masks))
        else:
            out.append((raw, masks, MemoryState(memory_k, vals[-1])))
    return out


# ---------------------------------------------------------------------
# Replies: the priors tuple of tools.inference_seam.output_to_wire
# ---------------------------------------------------------------------

def encode_replies(wires: Sequence[tuple], prefix: Sequence[int] = ()) -> bytes:
    """Server side: priors replies, each the tuple
    ("c", U, R, value, value_logits, cliffness, aux_score, moves_left,
    actor_kind (int8), CompactActions, memory), as one buffer; every
    array keeps its dtype."""
    leaves = []
    for w in wires:
        if not (isinstance(w, tuple) and w and w[0] == "c"):
            raise ValueError("the byte wire carries priors replies only")
        _, U, R, value, value_logits, cliffness, aux, ml, kinds, c, memory = w
        leaves.append(((int(U), int(R)),
                       (value, value_logits, cliffness, aux, ml, kinds,
                        c.actor, c.kind, c.target, c.weapon, c.prior, memory)))
    return encode_bundle(leaves, prefix)


def decode_replies(buf, offset: int = 0) -> List[tuple]:
    """Actor side: the priors tuples `encode_replies` wrote, for
    `output_from_wire`."""
    out = []
    for (U, R), f in decode_bundle(buf, len(REPLY_SCALARS), len(REPLY_FIELDS), offset):
        value, value_logits, cliffness, aux, ml, kinds, actor, kind, target, weapon, prior, memory = f
        out.append(("c", U, R, value, value_logits, cliffness, aux, ml, kinds,
                    CompactActions(actor=actor, kind=kind, target=target, weapon=weapon, prior=prior),
                    memory))
    return out
