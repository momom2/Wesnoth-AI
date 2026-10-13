"""The static-shape serve path: one CUDA graph per bucket.

What the inference server pays per batch is host time launching
kernels, not device time: the pool's serve thread logs 16 ms of host
work per 16-leaf batch (encode 2.7, forward 8.2, priors 5.3 ms) and
0.14 ms waiting for the device (docs/box_specs.md "The post-review
box run"), and the eval path's stream span is the same 14-16 ms at a
mean batch of 6-8 as at 17. A CUDA graph replays a batch's whole
kernel sequence from one launch, which needs every shape fixed. Here
the shapes are fixed by BUCKETS: the batch is padded to `b_cap`
segments, the tokens to a bucket of rows, the actor and hex slots to
caps, and the compaction to a fixed capacity. The design note
(docs/gpu_forward_design_20260904.md section 4) priced this option
last when the GPU was the binding cost at 1,270 tokens per leaf; the
relevant-set basis cut the tokens to ~300 and left the host launches
as the cost.

Per batch, outside the graph: numpy only -- the batch's RawEncoded
fields concatenated into the pinned embed buffer, the memory states
into the pinned memory buffer, the packed rows' sources and kinds, the
segment offsets and head slots, and the masks, into two more pinned
buffers. Inside the graph: the host->device copies, the trained
embeddings of every buffer row, the gather of the real tokens into the
packed rows with their kind term, the packed trunk over a bf16 copy of
the weights, the heads on the capped slots, the memory write,
`server_priors.priors_outputs` at the fixed capacity, and the one
device->host copy of the results. Padding is inert by construction:
pad segments hold one token each and attend only to themselves (flash
varlen), rows past the last segment are never read by a head (the
flash kernel leaves them untouched; the per-segment fallback zeroes
them), spare embed and memory rows hold stale or zero values that no
real token gathers, padded actor, hex and memory slots point at a real
row and carry zero masks or are cut from the reply, so the priors'
masked softmaxes and the compaction see exactly the real batch.
`GraphedServe(..., graphs=False)` runs the same static-shape body
eagerly (the CPU tests, and an A/B on a box).

The parity-memory recipe (docs/parity_memory_design_20260929.md): a
segment lists its hexes, units, recruits, sightings, memory slots, the
global token and end_turn, as `packed_trunk.build_packed_layout` lays
them out. The sighting rows are embedded from their own buffer rows,
the global row reads the faction posterior, and the memory rows are the
states sent with the batch plus their slot embedding; the new states,
written from the trunk's outputs at those rows, ride back with the
priors. A model with the material value input or the GBC context is
refused (the priors protocol carries neither).

Not graphed: the wire (`tools/inference_seam.output_to_wire`). A batch
that exceeds a cap (segments, actor slots, hex slots, tokens, a segment
longer than `max_len`, sightings past the actor cap, legal entries)
returns None from `infer` and the caller serves it on the eager path;
the counters say how often.
"""
from __future__ import annotations

import contextlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from wesnoth_ai.encoder import MAX_FACTIONS
from wesnoth_ai.memory import MemoryBatch
from wesnoth_ai.model import MAX_ATTACKS, ActorKind, TokenKind, UnitActionType
from wesnoth_ai.packed_trunk import (
    FlatLayout, PackedIndex, PackedTrunkWeights, _block_counts, _exclusive_cumsum,
    _segment_sources, check_packed_trunk_supported, make_packed_trunk_layers,
)
from wesnoth_ai.server_priors import (
    PackedMasks, PendingPriors, _staged_capacity, mask_layout, priors_outputs, write_masks,
)

log = logging.getLogger(__name__)

# Graph work is serialized per process. Two serve threads capturing at
# once trip the caching allocator's `captures_underway.empty()` assert
# (torch 2.5.1 c10/cuda/CUDACachingAllocator.cpp:2967), and a REPLAY on
# one thread while the other captures trips the default CUDA
# generator's "Offset increment outside graph capture encountered
# unexpectedly" (every capture registers that generator, and a replay
# advances its offset; both seen on the 2026-09-14 pool arms). The
# lock covers the capture and the replay launch, not the wait for the
# device, so the threads still overlap their staging and their waits.
_GRAPH_LOCK = threading.RLock()

# The block kinds of a segment, in `packed_trunk.build_packed_layout`'s order.
_BLOCK_KINDS = np.array([TokenKind.HEX, TokenKind.UNIT, TokenKind.RECRUIT, TokenKind.SIGHTING,
                         TokenKind.MEMORY, TokenKind.GLOBAL, TokenKind.END_TURN], dtype=np.int64)


@dataclass(frozen=True)
class Caps:
    """The bucket axes. Segments per graph from `b_caps` (the first
    that fits the batch; `b_cap` alone means one value, the server's
    max_batch); actor slots and hex slots bucketed to the first cap
    that fits (a leaf's sightings are held under its actor cap too);
    token rows to the first bucket that fits the real tokens plus one
    per pad segment; `max_len` the longest segment any graph accepts;
    `capacity` legal entries per batch."""
    b_cap: int = 16
    a_caps: Tuple[int, ...] = (64, 128)
    h_caps: Tuple[int, ...] = (512, 1024, 2048)
    t_caps: Tuple[int, ...] = (1024, 2048, 3072, 4096, 6144, 8192, 12288)
    max_len: int = 2048
    capacity: int = 65536
    b_caps: Tuple[int, ...] = ()

    @property
    def segment_caps(self) -> Tuple[int, ...]:
        return tuple(sorted(self.b_caps)) if self.b_caps else (self.b_cap,)


def pool_caps(max_batch: int) -> Caps:
    """The pool's caps: the picker fills a batch to `max_batch` leaves
    with whole requests, so batches run past it (17.4 leaves on average
    at max_batch 16, up to about twice), and a segment cap of twice
    max_batch keeps them on the graphed path, with token rows for that
    many of the largest leaves."""
    return Caps(b_cap=max_batch, b_caps=(max_batch, 2 * max_batch),
                t_caps=_token_caps(2 * max_batch))


# The token buckets: steps of 1,024 rows up to 6,144, then coarser. A
# batch replays at its bucket's rows, so a coarse step pads the trunk's
# dense layers (up to 60% at 5,000 tokens into 8,192); every bucket
# costs one capture.
_T_CAPS = (1024, 2048, 3072, 4096, 5120, 6144, 8192, 10240, 12288, 16384, 20480, 24576, 32768)
# Rows per leaf the largest bucket reserves: a relevant-set leaf runs
# 250-400 hex, unit and recruit tokens, plus a 64-slot memory.
_ROWS_PER_LEAF = 512


def _token_caps(leaves: int) -> Tuple[int, ...]:
    """The token buckets up to `leaves` of the largest leaves."""
    rows = -(-_ROWS_PER_LEAF * leaves // 1024) * 1024
    return tuple(t for t in _T_CAPS if t < rows) + (rows,)


def eval_caps(max_batch: int) -> Caps:
    """The eval server's caps: its batches run below max_batch (one
    decision in flight per game), so the segment axis is bucketed
    coarsely (every bucket costs one capture), and the token rows reach
    a full batch of the largest leaves."""
    b_caps = tuple(sorted({b for b in (8, 16, 24, 32, 48, 64) if b < max_batch} | {max_batch}))
    return Caps(b_cap=max_batch, b_caps=b_caps, t_caps=_token_caps(max_batch))


@dataclass(frozen=True)
class Bucket:
    b_cap: int
    a_cap: int
    h_cap: int
    t_cap: int


@dataclass
class StaticIndex:
    """The per-batch index arrays at static shapes (numpy)."""
    cu_seqlens: np.ndarray   # [b_cap + 1] int32
    actor: np.ndarray        # [b_cap * a_cap] int64
    hex: np.ndarray          # [b_cap * h_cap] int64
    glob: np.ndarray         # [b_cap] int64
    actor_kind: np.ndarray   # [b_cap, a_cap] int64
    memory: np.ndarray       # [b_cap * k_cap] int64
    counts: np.ndarray       # [B, 7] the real segments' block counts
    total: int               # the real segments' rows


def static_index(sizes: Sequence[Tuple[int, int, int]], b_cap: int, a_cap: int,
                 h_cap: int, t_cap: int, *, sightings: Optional[Sequence[int]] = None,
                 memory: Optional[Sequence[int]] = None, k_cap: int = 0) -> StaticIndex:
    """`packed_trunk.build_packed_layout`'s head arrays and offsets
    extended to the caps: rows b < B as the real layout has them
    (padded slots point at the row's end_turn token), then b_cap - B
    pad segments of one token each right after the real tokens, whose
    slots all point at that token. `sightings` and `memory` are the
    per-sample sighting and memory-slot counts (each slot count at
    most `k_cap`). Requires total + (b_cap - B) <= t_cap; the rows from
    the last pad token to t_cap belong to no segment."""
    counts = _block_counts(sizes, sightings, memory)
    B = len(counts)
    if memory is not None and B and int(counts[:, 4].max()) > k_cap:
        raise ValueError(f"static_index: a memory of {int(counts[:, 4].max())} slots for k_cap {k_cap}")
    full = np.zeros((b_cap, counts.shape[1]), dtype=np.int64)
    full[:B] = counts
    lengths = full.sum(axis=1)
    lengths[B:] = 1
    cu = _exclusive_cumsum(lengths)
    if int(cu[-1]) > t_cap:
        raise ValueError(f"static_index: {int(cu[-1])} rows for t_cap {t_cap}")
    Hs, Us, Rs, Ss = full[:, 0], full[:, 1], full[:, 2], full[:, 3]
    Ks = full[:, 4]
    seg0, end = cu[:-1, None], (cu[1:] - 1)[:, None]                  # [b_cap, 1]
    a = np.arange(a_cap, dtype=np.int64)[None, :]
    actor = np.where(a < (Us + Rs)[:, None], seg0 + Hs[:, None] + a, end)
    h = np.arange(h_cap, dtype=np.int64)[None, :]
    hexes = np.where(h < Hs[:, None], seg0 + h, end)
    k = np.arange(k_cap, dtype=np.int64)[None, :]
    mem = np.where(k < Ks[:, None], seg0 + (Hs + Us + Rs + Ss)[:, None] + k, end)
    glob = np.where(np.arange(b_cap) < B, end[:, 0] - 1, end[:, 0])
    kinds = np.full((b_cap, a_cap), ActorKind.END_TURN, dtype=np.int64)
    kinds[a < (Us + Rs)[:, None]] = ActorKind.RECRUIT
    kinds[a < Us[:, None]] = ActorKind.UNIT
    # Every slot addresses a row of its own segment, below t_cap; a
    # wrong index here would be a device-side assert inside a replay,
    # which poisons the whole CUDA context.
    rows = int(cu[-1])
    for name, arr in (("actor", actor), ("hex", hexes), ("glob", glob), ("memory", mem)):
        if arr.size and (int(arr.min()) < 0 or int(arr.max()) >= rows):
            raise ValueError(f"static_index: {name} slot outside the {rows} rows")
    return StaticIndex(cu_seqlens=cu.astype(np.int32), actor=actor.reshape(-1),
                       hex=hexes.reshape(-1), glob=glob, actor_kind=kinds,
                       memory=mem.reshape(-1), counts=counts, total=int(cu[B]))


def static_sources(counts: np.ndarray, bases: Sequence[int], k_cap: int,
                   end_row: int) -> Tuple[np.ndarray, np.ndarray]:
    """The source row in the token table and the token kind of every
    real packed row: a segment's blocks in order, each block's rows from
    `bases` (the table's first hex, unit, recruit and sighting row, its
    memory rows at sample * k_cap, its global rows, one per sample) and
    its one shared end_turn row."""
    B = len(counts)
    starts = np.empty((B, len(_BLOCK_KINDS)), dtype=np.int64)
    for j in range(4):
        starts[:, j] = bases[j] + _exclusive_cumsum(counts[:, j])[:-1]
    starts[:, 4] = bases[4] + np.arange(B, dtype=np.int64) * k_cap
    starts[:, 5] = bases[5] + np.arange(B, dtype=np.int64)
    starts[:, 6] = end_row
    src = _segment_sources(counts, starts)
    kind = np.repeat(np.tile(_BLOCK_KINDS, B), counts.reshape(-1))
    return src, kind


def _clear_masks(mv: Dict[str, np.ndarray], dirty: Tuple[int, int, int]) -> None:
    """Zero what the previous batch wrote into the static mask views
    (`write_masks` fills rows b < B, slots a < A_max and, for the
    attack bias, hexes h < H_max), so the buffer is clean at the cost
    of that batch's size, not the caps'."""
    B, A, H = dirty
    if B == 0:
        return
    for name, arr in mv.items():
        if name == "attack_bias":
            arr[:B, :A, :H] = 0
        else:
            arr[:B, :A] = 0


@dataclass
class GraphedBatch:
    """One batch served from a bucket: the compact actions per sample,
    the head outputs by name ([B, ...] host arrays), the actor kinds
    [B, A_max] (CPU long), the sizes, and the new memory states
    [B, k_cap, d] (float32; sample b's first k_b rows) or None."""
    compact: List
    extras: Dict[str, np.ndarray]
    actor_kind: torch.Tensor
    sizes: List[Tuple[int, int, int]]
    memory: Optional[np.ndarray] = None


@dataclass
class _State:
    """One bucket's static buffers and, once captured, its graph."""
    bucket: Bucket
    rows: Tuple[int, ...]               # the token table's hex, unit, recruit, sighting, memory, global rows
    embed_layout: FlatLayout            # the RawEncoded fields of the batch, capped
    embed_host: torch.Tensor            # pinned uint8
    index_layout: FlatLayout
    index_host: torch.Tensor            # pinned uint8
    mask_layout: FlatLayout
    mask_host: torch.Tensor             # pinned uint8
    memory_host: Optional[torch.Tensor] = None   # pinned float32 [b_cap, k_cap, d]
    out_layout: Optional[FlatLayout] = None
    out_host: Optional[torch.Tensor] = None
    graph: Optional[torch.cuda.CUDAGraph] = None
    actor_kind: Optional[torch.Tensor] = None   # CPU, set per call (eager) or at capture
    dirty: Tuple[int, int, int] = (0, 0, 0)     # (B, A_max, H_max) the mask views last held
    replays: int = 0
    capture_s: float = 0.0
    stats: Dict[str, int] = field(default_factory=dict)

    @property
    def bases(self) -> Tuple[int, ...]:
        """The first row of each block of the token table, then its end_turn row."""
        return tuple(int(x) for x in _exclusive_cumsum(np.asarray(self.rows, dtype=np.int64)))


class GraphedServe:
    """Serves (RawEncoded, PackedMasks) batches, with each leaf's memory
    state for a model with a memory, through per-bucket CUDA graphs; see
    the module docstring. One instance per serve thread
    (tools/inference_seam builds them from a factory), or one shared by
    several through its lock."""

    def __init__(self, model, encoder, device: torch.device, *, caps: Caps = Caps(),
                 graphs: bool = True, extras: Optional[Sequence[str]] = None):
        if getattr(model, "has_gbc", False):
            raise ValueError("graphed serve does not carry the GBC unit context")
        if getattr(model, "has_value_material", False):
            raise ValueError("graphed serve does not carry the material value input")
        if bool(getattr(encoder, "observation_parity", False)) != bool(getattr(model, "observation_parity", False)):
            raise ValueError("graphed serve: the encoder and the model disagree on the parity observation")
        check_packed_trunk_supported(model.encoder)
        self.model, self.encoder, self.device, self.caps = model, encoder, device, caps
        self.graphs = bool(graphs) and device.type == "cuda"
        if extras is None:
            extras = ["value", "value_logits", "cliffness"]
            extras += [n for n, on in (("aux_score", model.has_aux_score),
                                       ("moves_left", model.has_moves_left)) if on]
        self.extras = tuple(extras)
        self.parity = bool(getattr(encoder, "observation_parity", False))
        self.k_cap = int(getattr(model, "memory_slots", 0) or 0)
        self.T = int(UnitActionType.COUNT)
        self.W = int(MAX_ATTACKS)
        # The layer loop as a pure function of tensors over a copy of
        # the encoder's weights in the compute dtype (packed_trunk
        # section 13): no per-call weight casts, and the copy is
        # refreshed IN PLACE when the model publishes, so a captured
        # graph reads the new values through the same storage.
        layer = model.encoder.layers[0]
        self._layers = make_packed_trunk_layers(layer.activation, layer.norm1.eps)
        self._weights: Optional[PackedTrunkWeights] = None
        self._initial: Optional[Tuple[int, np.ndarray]] = None   # (weights version, initial memory)
        self._states: Dict[Bucket, _State] = {}
        # One private memory pool per graph: sharing one across
        # graphs is safe only when they replay in capture order, and
        # buckets replay in whatever order the batches arrive.
        self._lock = threading.Lock()
        self.fallbacks: Dict[str, int] = {}
        self.served = 0
        self.capture_s = 0.0

    # -- buckets ---------------------------------------------------------

    def _pick(self, B: int, A_max: int, H_max: int, S_max: int, total: int,
              longest: int) -> Tuple[Optional[Bucket], str]:
        """The first bucket the batch fits, or (None, the axis it does
        not fit: the fallback counter's key)."""
        c = self.caps
        if longest > c.max_len:
            return None, "segment_len"
        b_cap = next((b for b in c.segment_caps if b >= B), None)
        if b_cap is None:
            return None, "segments"
        a_cap = next((a for a in c.a_caps if a >= A_max), None)
        if a_cap is None:
            return None, "actors"
        if S_max > a_cap:
            return None, "sightings"
        h_cap = next((h for h in c.h_caps if h >= H_max), None)
        if h_cap is None:
            return None, "hexes"
        t_cap = next((t for t in c.t_caps if t >= total + (b_cap - B)), None)
        if t_cap is None:
            return None, "tokens"
        return Bucket(b_cap, a_cap, h_cap, t_cap), ""

    def _stream_caps(self, bucket: Bucket) -> Tuple[int, ...]:
        """The embed buffer's rows per stream: hexes fill the token rows;
        units, recruits and sightings are under a_cap per leaf."""
        per_leaf = bucket.b_cap * bucket.a_cap
        caps = (bucket.t_cap, per_leaf, per_leaf)
        return caps + ((per_leaf,) if self.parity else ())

    def _state(self, bucket: Bucket) -> _State:
        st = self._states.get(bucket)
        if st is not None:
            return st
        dev = self.device
        pin = dev.type == "cuda"
        caps = self._stream_caps(bucket)
        fields = [(name, dt, (cap,) + shape)
                  for (stream, spec), cap in zip(self.encoder._stream_fields(), caps)
                  for name, dt, shape in spec]
        their = (("their_faction_probs", torch.float32, (bucket.b_cap, MAX_FACTIONS)) if self.parity
                 else ("their_faction_id", torch.int64, (bucket.b_cap,)))
        fields += [("global_feats", torch.float32, (bucket.b_cap, self.encoder.widths.global_feats)),
                   ("our_faction_id", torch.int64, (bucket.b_cap,)), their]
        embed_layout = FlatLayout(fields)
        index_layout = FlatLayout([
            ("cu_seqlens", torch.int32, (bucket.b_cap + 1,)),
            ("src", torch.int64, (bucket.t_cap,)),
            ("kind", torch.int64, (bucket.t_cap,)),
            ("actor", torch.int64, (bucket.b_cap * bucket.a_cap,)),
            ("hex", torch.int64, (bucket.b_cap * bucket.h_cap,)),
            ("glob", torch.int64, (bucket.b_cap,)),
            ("memory", torch.int64, (bucket.b_cap * self.k_cap,)),
        ])
        ml = mask_layout(bucket.b_cap, bucket.a_cap, bucket.h_cap, self.T, self.W,
                         type_bias=True, attack_bias=True)
        sighting_rows = caps[3] if self.parity else 0
        st = _State(
            bucket=bucket,
            rows=caps[:3] + (sighting_rows, bucket.b_cap * self.k_cap, bucket.b_cap),
            embed_layout=embed_layout,
            embed_host=torch.zeros(embed_layout.nbytes, dtype=torch.uint8, pin_memory=pin),
            index_layout=index_layout,
            index_host=torch.zeros(index_layout.nbytes, dtype=torch.uint8, pin_memory=pin),
            mask_layout=ml,
            mask_host=torch.zeros(ml.nbytes, dtype=torch.uint8, pin_memory=pin),
            memory_host=(torch.zeros(bucket.b_cap, self.k_cap, int(self.model.d_model),
                                     dtype=torch.float32, pin_memory=pin) if self.k_cap else None))
        self._states[bucket] = st
        return st

    # -- the static body -------------------------------------------------

    def _current_weights(self) -> PackedTrunkWeights:
        """The trunk's weight copy at the model's current version
        (built on first use; refreshed in place after a publication).
        Called before every run, outside the graph."""
        model = self.model
        w = self._weights
        version = int(getattr(model, "_weights_version", 0))
        if w is None:
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
            w = self._weights = PackedTrunkWeights.build(model.encoder, dtype, self.device, version)
        elif w.version != version:
            w.refresh(model.encoder, version)
        return w

    def _initial_memory(self) -> np.ndarray:
        """The learned initial memory on the host, float32 [slots, d], at
        the model's current weights: what a game-side's first decision reads
        (`model.initial_memory(k)` is its first k rows)."""
        version = int(getattr(self.model, "_weights_version", 0))
        if self._initial is None or self._initial[0] != version:
            with torch.no_grad():
                init = self.model.initial_memory(self.k_cap).detach().float().cpu().numpy()
            self._initial = (version, init)
        return self._initial[1]

    def _body(self, st: _State) -> None:
        """Everything from the host->device copies to the one
        device->host copy, on the bucket's static buffers. Captured
        once per bucket on CUDA; run as is on CPU."""
        model, dev, c = self.model, self.device, self.caps
        bucket = st.bucket
        B, A, H, K = bucket.b_cap, bucket.a_cap, bucket.h_cap, self.k_cap
        d = int(model.d_model)
        bf16 = dev.type == "cuda"
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16) if bf16
                    else contextlib.nullcontext())
        with torch.no_grad(), autocast:
            iv = st.index_layout.torch_views(st.index_host.to(dev, non_blocking=True))
            mv = st.mask_layout.torch_views(st.mask_host.to(dev, non_blocking=True))
            index = PackedIndex(src=None, kind=None, actor=iv["actor"], hex=iv["hex"], unit=None,
                                glob=iv["glob"], cu_seqlens=iv["cu_seqlens"],
                                cu_host=st.index_layout.numpy_views(
                                    st.index_host.numpy())["cu_seqlens"].tolist(),
                                max_len=c.max_len)
            # The trained embeddings of every table row (the eager path
            # runs them outside its autocast, so they stay fp32 here),
            # the real tokens gathered into the packed rows with their
            # kind term.
            enc = self.encoder
            with torch.autocast("cuda", enabled=False) if bf16 else contextlib.nullcontext():
                ev = st.embed_layout.torch_views(st.embed_host.to(dev, non_blocking=True))
                their = ev["their_faction_probs"] if self.parity else ev["their_faction_id"]
                parts = [
                    enc._hex_embedding(ev["hex_xs"], ev["hex_ys"], ev["hex_terrain_ids"],
                                       ev["hex_modifier_flags"], ev["hex_dynamic_flags"]),
                    enc._unit_embedding(ev["unit_type_ids"], ev["unit_side_ids"],
                                        ev["unit_xs"], ev["unit_ys"], ev["unit_feats"]),
                    enc._unit_embedding(ev["recruit_type_ids"], ev["recruit_side_ids"],
                                        ev["recruit_xs"], ev["recruit_ys"], ev["recruit_feats"]),
                ]
                if self.parity:
                    parts.append(enc._sighting_embedding(ev["sight_type_ids"], ev["sight_xs"],
                                                         ev["sight_ys"], ev["sight_feats"]))
                memory = None
                if K:
                    memory = st.memory_host.to(dev, non_blocking=True)
                    parts.append(model.slot_memory.tokens(MemoryBatch(memory, [K] * B)).reshape(-1, d))
                parts += [enc._global_embedding(ev["global_feats"], ev["our_faction_id"], their),
                          enc.end_turn_token.view(1, -1)]
                tokens = torch.cat(parts, dim=0)
                x = tokens.index_select(0, iv["src"]) + model.token_kind_embed(iv["kind"])
            w = self._weights
            h = self._layers(x, index.cu_seqlens, index.max_len, w.layers, w.final_norm)
            actor_ctx = h.index_select(0, index.actor).view(B, A, d)
            hex_ctx = h.index_select(0, index.hex).view(B, H, d)
            global_ctx = h.index_select(0, index.glob).view(B, 1, d)
            sizes = [(0, 0, 0)] * B
            padded = model._heads(actor_ctx, hex_ctx, global_ctx, st.actor_kind, sizes, None)
            if bf16:
                padded = padded.float32()
            extras = [getattr(padded, n) for n in self.extras]
            if K:
                h_memory = h.index_select(0, iv["memory"]).view(B, K, d)
                extras.append(model.slot_memory.write(h_memory, MemoryBatch(memory, [K] * B)))
            out_layout, flat_out = priors_outputs(padded, mv, c.capacity, extras)
            if st.out_host is None:
                st.out_layout = out_layout
                st.out_host = torch.zeros(out_layout.nbytes, dtype=torch.uint8,
                                          pin_memory=dev.type == "cuda")
            st.out_host.copy_(flat_out, non_blocking=True)

    def _capture(self, st: _State) -> None:
        """Warm the body on a side stream, then record it once. The
        first batch of the bucket is what the warmups and the capture
        compute on, so its buffers already hold real data."""
        t0 = time.perf_counter()
        with _GRAPH_LOCK:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._body(st)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            # thread_local: the other serve thread keeps issuing its own
            # CUDA work (replays, the eager path) while this one records;
            # the default "global" mode makes those calls fail and the
            # capture with them ("operation failed due to a previous
            # error during capture", the 2026-09-14 pool arm).
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                self._body(st)
            torch.cuda.synchronize(self.device)
        st.graph = g
        st.capture_s = time.perf_counter() - t0
        self.capture_s += st.capture_s

    def _run(self, st: _State) -> None:
        if not self.graphs:
            self._body(st)
            return
        with _GRAPH_LOCK:
            if st.graph is None:
                try:
                    self._capture(st)
                except Exception as e:                   # noqa: BLE001 -- eager serves this bucket
                    log.warning("graphed serve: capture failed for %s (%s: %s); this bucket "
                                "runs its static body eagerly", st.bucket, type(e).__name__, e)
                    st.graph = False                     # type: ignore[assignment]
            if st.graph:
                st.graph.replay()
                st.replays += 1
                return
        self._body(st)

    # -- one batch ---------------------------------------------------------

    def _fallback(self, why: str) -> None:
        self.fallbacks[why] = self.fallbacks.get(why, 0) + 1

    def _stage_embed(self, st: _State, raws, counts: np.ndarray) -> None:
        """The batch's RawEncoded fields into the embed buffer's rows."""
        ev = st.embed_layout.numpy_views(st.embed_host.numpy())
        B = len(raws)
        totals = (counts[:, 0].sum(), counts[:, 1].sum(), counts[:, 2].sum(), counts[:, 3].sum())
        for (stream, spec), n_rows in zip(self.encoder._stream_fields(), totals):
            if n_rows:
                for name, _, _ in spec:
                    np.concatenate([getattr(r, name) for r in raws], out=ev[name][:int(n_rows)])
        np.stack([r.global_feats for r in raws], out=ev["global_feats"][:B])
        ev["our_faction_id"][:B] = [r.our_faction_id for r in raws]
        if self.parity:
            np.stack([r.their_faction_probs for r in raws], out=ev["their_faction_probs"][:B])
        else:
            ev["their_faction_id"][:B] = [r.their_faction_id for r in raws]

    def _stage_memory(self, st: _State, memories) -> List[int]:
        """Each leaf's memory state into the memory buffer's first k rows
        of its segment (the learned initial memory at a game-side's first
        decision); the slot counts."""
        buf = st.memory_host.numpy()
        counts = []
        initial = None
        for b, m in enumerate(memories):
            k = int(m.k)
            if m.state is None:
                if initial is None:
                    initial = self._initial_memory()
                buf[b, :k] = initial[:k]
            else:
                state = m.state.detach().cpu().numpy() if torch.is_tensor(m.state) else m.state
                if state.shape != (k, buf.shape[2]):
                    raise ValueError(f"a memory state of shape {state.shape} for {k} slots")
                buf[b, :k] = state
            counts.append(k)
        return counts

    def infer(self, raws, packs: Sequence[PackedMasks],
              memories: Optional[Sequence] = None) -> Optional[GraphedBatch]:
        """The batch served from its bucket, or None when it does not
        fit one, so the caller serves it eagerly. `memories`: one
        `MemoryState` per leaf for a model with a memory."""
        if bool(self.k_cap) != (memories is not None):
            raise ValueError(f"graphed serve: a model of {self.k_cap} memory slots "
                             f"{'needs' if self.k_cap else 'takes no'} memory states")
        c = self.caps
        sizes = [(r.unit_xs.shape[0], r.recruit_type_ids.shape[0], r.hex_xs.shape[0]) for r in raws]
        sightings = [r.sight_xs.shape[0] for r in raws] if self.parity else None
        slots = [int(m.k) for m in memories] if memories is not None else None
        B = len(sizes)
        counts = _block_counts(sizes, sightings, slots)
        lengths = counts.sum(axis=1)
        total = int(lengths.sum())
        A_max = int((counts[:, 1] + counts[:, 2]).max()) + 1
        H_max = int(counts[:, 0].max())
        S_max = int(counts[:, 3].max())
        bucket, why = self._pick(B, A_max, H_max, S_max, total, int(lengths.max()))
        if bucket is None:
            self._fallback(why)
            return None
        sidx = static_index(sizes, bucket.b_cap, bucket.a_cap, bucket.h_cap, bucket.t_cap,
                            sightings=sightings, memory=slots, k_cap=self.k_cap)
        with self._lock:
            st = self._state(bucket)
            self._stage_embed(st, raws, counts)
            if self.k_cap:
                self._stage_memory(st, memories)
            bases = st.bases
            end_row = bases[-1]
            src, kind = static_sources(counts, bases[:6], self.k_cap, end_row)
            iv = st.index_layout.numpy_views(st.index_host.numpy())
            iv["src"][:total] = src
            iv["src"][total:] = end_row
            iv["kind"][:total] = kind
            iv["kind"][total:] = TokenKind.END_TURN
            iv["cu_seqlens"][:] = sidx.cu_seqlens
            iv["actor"][:] = sidx.actor
            iv["hex"][:] = sidx.hex
            iv["glob"][:] = sidx.glob
            iv["memory"][:] = sidx.memory
            mv = st.mask_layout.numpy_views(st.mask_host.numpy())
            _clear_masks(mv, st.dirty)
            st.dirty = (B, A_max, H_max)
            write_masks(mv, packs)
            if _staged_capacity(mv, self.W) > c.capacity:
                self._fallback("capacity")
                return None
            st.actor_kind = torch.from_numpy(sidx.actor_kind)
            self._current_weights()
            self._run(st)
            n_extras = len(self.extras) + (1 if self.k_cap else 0)
            pending = PendingPriors(host=st.out_host, layout=st.out_layout, n_samples=B,
                                    capacity=c.capacity, n_extras=n_extras, device=self.device)
            compact, host = pending.finish()
            # Copies: the static output buffer is rewritten by the next batch.
            host = [np.array(a[:B], copy=True) for a in host]
            compact = [type(x)(**{k: np.array(getattr(x, k), copy=True)
                                  for k in ("actor", "kind", "target", "weapon", "prior")})
                       for x in compact]
        self.served += 1
        extras = dict(zip(self.extras, host))
        return GraphedBatch(compact=compact, extras=extras,
                            actor_kind=st.actor_kind[:B, :A_max], sizes=[tuple(s) for s in sizes],
                            memory=host[-1] if self.k_cap else None)

    def summary(self) -> Dict[str, object]:
        return {"graphs": self.graphs, "served": self.served, "fallbacks": dict(self.fallbacks),
                "capture_s": round(self.capture_s, 2),
                "buckets": {f"{b.b_cap}x{b.a_cap}x{b.h_cap}x{b.t_cap}": st.replays
                            for b, st in self._states.items()}}


__all__ = ["Caps", "Bucket", "StaticIndex", "GraphedBatch", "static_index", "static_sources",
           "GraphedServe", "pool_caps", "eval_caps"]
