"""One contiguous buffer per inference request.

The actor pool's request path (tools/actor_pool.py, server-priors
protocol) used to pickle each leaf's RawEncoded and PackedMasks as
they are: about forty numpy objects per leaf plus the hex-position
list the server never reads. Unpickling that inside `Queue.get` was
54% of a serve thread's GIL time on 2026-09-04 (docs/box_specs.md,
server profile). Here a request of B leaves becomes ONE uint8 array
plus a tuple of per-leaf headers (shapes and scalars); the server
reconstructs the arrays as views into the buffer, no copies.

Only what the server's padded encode and batched priors read is
carried: the numeric streams of RawEncoded (positions, ids, type
strings are left as None) and every PackedMasks field. A parity-
observation leaf adds the faction posterior and the sighting stream; a
leaf of a player with a memory adds its slot count and, after the
game-side's first decision, the state its previous decision wrote
(`wesnoth_ai.memory.MemoryState`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from wesnoth_ai.encoder import RawEncoded
from wesnoth_ai.memory import MemoryState
from wesnoth_ai.server_priors import PackedMasks

# (field, dtype) in buffer order. Optional mask biases are None when
# absent, which the header records as a None shape.
RAW_FIELDS: Tuple[Tuple[str, np.dtype], ...] = (
    ("hex_xs", np.dtype(np.int64)), ("hex_ys", np.dtype(np.int64)),
    ("hex_terrain_ids", np.dtype(np.int64)),
    ("hex_modifier_flags", np.dtype(np.float32)),
    ("hex_dynamic_flags", np.dtype(np.float32)),
    ("unit_is_ours", np.dtype(np.float32)), ("unit_type_ids", np.dtype(np.int64)),
    ("unit_side_ids", np.dtype(np.int64)), ("unit_xs", np.dtype(np.int64)),
    ("unit_ys", np.dtype(np.int64)), ("unit_feats", np.dtype(np.float32)),
    ("recruit_is_ours", np.dtype(np.float32)), ("recruit_type_ids", np.dtype(np.int64)),
    ("recruit_side_ids", np.dtype(np.int64)), ("recruit_xs", np.dtype(np.int64)),
    ("recruit_ys", np.dtype(np.int64)), ("recruit_feats", np.dtype(np.float32)),
    ("global_feats", np.dtype(np.float32)),
)
MASK_FIELDS: Tuple[Tuple[str, np.dtype], ...] = (
    ("actor_mask", np.dtype(np.uint8)), ("type_valid", np.dtype(np.uint8)),
    ("attack_valid", np.dtype(np.uint8)), ("move_valid", np.dtype(np.uint8)),
    ("union_valid", np.dtype(np.uint8)), ("n_attacks", np.dtype(np.int8)),
    ("type_bias", np.dtype(np.float32)), ("attack_bias", np.dtype(np.float32)),
)
# The parity observation's streams, absent (a None shape) on obs8's leaves.
PARITY_FIELDS: Tuple[Tuple[str, np.dtype], ...] = (
    ("their_faction_probs", np.dtype(np.float32)), ("sight_type_ids", np.dtype(np.int64)),
    ("sight_xs", np.dtype(np.int64)), ("sight_ys", np.dtype(np.int64)),
    ("sight_feats", np.dtype(np.float32)),
)
# The memory state a leaf carries, absent at a game-side's first decision.
MEMORY_FIELDS: Tuple[Tuple[str, np.dtype], ...] = (("memory_state", np.dtype(np.float32)),)
ALL_FIELDS = RAW_FIELDS + MASK_FIELDS + PARITY_FIELDS + MEMORY_FIELDS
_ALIGN = 8


@dataclass(slots=True)
class LeafHeader:
    shapes: Tuple[Optional[Tuple[int, ...]], ...]   # ALL_FIELDS, in order
    memory_k: int                   # the player's active slots; -1 without a memory
    our_faction_id: int
    their_faction_id: int
    hex_subset: bool
    end_turn_bias: float
    n_units: int
    n_recruits: int
    n_hexes: int


@dataclass(slots=True)
class PackedRequest:
    buf: np.ndarray                 # uint8 [total]
    headers: Tuple[LeafHeader, ...]

    def __len__(self) -> int:
        return len(self.headers)


def _aligned(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


def _leaf_arrays(raw: RawEncoded, masks: PackedMasks, memory: Optional[MemoryState]):
    for name, dt in RAW_FIELDS:
        yield np.ascontiguousarray(getattr(raw, name), dtype=dt)
    for name, dt in MASK_FIELDS:
        a = getattr(masks, name)
        yield None if a is None else np.ascontiguousarray(a, dtype=dt)
    for name, dt in PARITY_FIELDS:
        a = getattr(raw, name, None)
        yield None if a is None else np.ascontiguousarray(a, dtype=dt)
    state = None if memory is None else memory.state
    if state is not None and hasattr(state, "detach"):
        state = state.detach().cpu().numpy()
    yield None if state is None else np.ascontiguousarray(state, dtype=np.float32)


def pack_request(items: Sequence[Tuple]) -> PackedRequest:
    """Actor side: B (RawEncoded, PackedMasks) pairs, or (RawEncoded,
    PackedMasks, MemoryState) triples for a player with a memory -> one
    buffer."""
    arrays: List[List[Optional[np.ndarray]]] = []
    headers: List[LeafHeader] = []
    total = 0
    for item in items:
        raw, masks = item[0], item[1]
        memory = item[2] if len(item) > 2 else None
        leaf = list(_leaf_arrays(raw, masks, memory))
        for a in leaf:
            if a is not None:
                total = _aligned(total) + a.nbytes
        arrays.append(leaf)
        headers.append(LeafHeader(
            shapes=tuple(None if a is None else a.shape for a in leaf),
            memory_k=-1 if memory is None else int(memory.k),
            our_faction_id=int(raw.our_faction_id),
            their_faction_id=int(raw.their_faction_id),
            hex_subset=bool(raw.hex_subset),
            end_turn_bias=float(masks.end_turn_bias),
            n_units=int(masks.n_units), n_recruits=int(masks.n_recruits),
            n_hexes=int(masks.n_hexes)))
    buf = np.empty(total, dtype=np.uint8)
    off = 0
    for leaf in arrays:
        for a in leaf:
            if a is None:
                continue
            off = _aligned(off)
            n = a.nbytes
            if n:
                buf[off:off + n] = a.reshape(-1).view(np.uint8)
            off += n
    return PackedRequest(buf=buf, headers=tuple(headers))


def unpack_request(req: PackedRequest) -> List[Tuple]:
    """Server side: views into the buffer, one (RawEncoded, PackedMasks)
    per leaf, or (RawEncoded, PackedMasks, MemoryState) for a leaf with a
    memory (its state a numpy view, None at a game-side's first decision).
    The RawEncoded carries only the numeric streams."""
    buf = req.buf
    out: List[Tuple] = []
    off = 0
    fields = ALL_FIELDS
    for h in req.headers:
        vals: List[Optional[np.ndarray]] = []
        for (name, dt), shape in zip(fields, h.shapes):
            if shape is None:
                vals.append(None)
                continue
            off = _aligned(off)
            # A plain product: np.prod on a 1-2 element tuple costs a
            # couple of microseconds, and the server pays it once per
            # field per leaf (26 x 16 per pool batch).
            count = 1
            for _d in shape:
                count *= _d
            n = count * dt.itemsize
            a = buf[off:off + n].view(dt).reshape(shape) if n else np.empty(shape, dtype=dt)
            vals.append(a)
            off += n
        n_raw, n_mask, n_parity = len(RAW_FIELDS), len(MASK_FIELDS), len(PARITY_FIELDS)
        r = dict(zip((f for f, _ in RAW_FIELDS), vals[:n_raw]))
        m = dict(zip((f for f, _ in MASK_FIELDS), vals[n_raw:n_raw + n_mask]))
        parity = dict(zip((f for f, _ in PARITY_FIELDS), vals[n_raw + n_mask:n_raw + n_mask + n_parity]))
        raw = RawEncoded(
            hex_positions=None, unit_positions=None, unit_ids=None, recruit_types=None,
            our_faction_id=h.our_faction_id, their_faction_id=h.their_faction_id,
            hex_subset=h.hex_subset, **r, **parity)
        masks = PackedMasks(end_turn_bias=h.end_turn_bias, n_units=h.n_units,
                            n_recruits=h.n_recruits, n_hexes=h.n_hexes, **m)
        if h.memory_k < 0:
            out.append((raw, masks))
        else:
            out.append((raw, masks, MemoryState(h.memory_k, vals[-1])))
    return out
