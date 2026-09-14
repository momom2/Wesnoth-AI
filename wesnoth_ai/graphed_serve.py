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

Per batch, outside the graph: the packed embed (encoder
.encode_from_raw_embedded), the gather of the real tokens into the
static token buffer with their kind term, the static index and mask
arrays written into two pinned host buffers. Inside the graph: the two
host->device copies, the packed trunk over the static rows, the heads
on the capped slots, `server_priors.priors_outputs` at the fixed
capacity, and the one device->host copy of the results. Padding is
inert by construction: pad segments hold one token each and attend
only to themselves (flash varlen), rows past the last segment are
never read by a head (the flash kernel leaves them untouched; the
per-segment fallback zeroes them), padded actor and hex slots point at
a real row and carry zero masks, so the priors' masked softmaxes and
the compaction see exactly the real batch. `GraphedServe(...,
graphs=False)` runs the same static-shape body eagerly (the CPU tests,
and an A/B on a box).

Not graphed (yet): the embed stage and the wire. A batch that exceeds
a cap (segments, actor slots, hex slots, tokens, a segment longer than
`max_len`, legal entries) returns None from `infer` and the caller
serves it on the eager path; the counters say how often.
"""
from __future__ import annotations

import contextlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from wesnoth_ai.model import MAX_ATTACKS, ActorKind, TokenKind, UnitActionType
from wesnoth_ai.packed_trunk import (
    FlatLayout, PackedIndex, PackedTrunkWeights, build_packed_layout,
    check_packed_trunk_supported, make_packed_trunk_layers,
)
from wesnoth_ai.server_priors import (
    PackedMasks, PendingPriors, _staged_capacity, mask_layout, priors_outputs, write_masks,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Caps:
    """The bucket axes. `b_cap` segments per graph (the server's
    max_batch); actor slots and hex slots bucketed to the first cap
    that fits; token rows to the first bucket that fits the real
    tokens plus one per pad segment; `max_len` the longest segment any
    graph accepts; `capacity` legal entries per batch."""
    b_cap: int = 16
    a_caps: Tuple[int, ...] = (64, 128)
    h_caps: Tuple[int, ...] = (512, 1024)
    t_caps: Tuple[int, ...] = (1024, 2048, 3072, 4096, 6144, 8192, 12288)
    max_len: int = 2048
    capacity: int = 65536


@dataclass(frozen=True)
class Bucket:
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


def static_index(sizes: Sequence[Tuple[int, int, int]], b_cap: int, a_cap: int,
                 h_cap: int, t_cap: int) -> StaticIndex:
    """`packed_trunk.build_packed_layout`'s head arrays and offsets
    extended to the caps: rows b < B as the real layout has them
    (padded slots point at the row's end_turn token), then b_cap - B
    pad segments of one token each right after the real tokens, whose
    slots all point at that token. Requires total + (b_cap - B) <=
    t_cap; the rows from the last pad token to t_cap belong to no
    segment."""
    B = len(sizes)
    sz = np.asarray(sizes, dtype=np.int64).reshape(B, 3)
    Us = np.zeros(b_cap, dtype=np.int64)
    Rs = np.zeros(b_cap, dtype=np.int64)
    Hs = np.zeros(b_cap, dtype=np.int64)
    Us[:B], Rs[:B], Hs[:B] = sz[:, 0], sz[:, 1], sz[:, 2]
    lengths = Us + Rs + Hs + 2
    lengths[B:] = 1
    cu = np.zeros(b_cap + 1, dtype=np.int64)
    np.cumsum(lengths, out=cu[1:])
    if int(cu[-1]) > t_cap:
        raise ValueError(f"static_index: {int(cu[-1])} rows for t_cap {t_cap}")
    seg0, end = cu[:-1, None], (cu[1:] - 1)[:, None]                  # [b_cap, 1]
    a = np.arange(a_cap, dtype=np.int64)[None, :]
    actor = np.where(a < (Us + Rs)[:, None], seg0 + Hs[:, None] + a, end)
    h = np.arange(h_cap, dtype=np.int64)[None, :]
    hexes = np.where(h < Hs[:, None], seg0 + h, end)
    glob = np.where(np.arange(b_cap) < B, end[:, 0] - 1, end[:, 0])
    kinds = np.full((b_cap, a_cap), ActorKind.END_TURN, dtype=np.int64)
    kinds[a < (Us + Rs)[:, None]] = ActorKind.RECRUIT
    kinds[a < Us[:, None]] = ActorKind.UNIT
    # Every slot addresses a row of its own segment, below t_cap; a
    # wrong index here would be a device-side assert inside a replay,
    # which poisons the whole CUDA context.
    rows = int(cu[-1])
    for name, arr in (("actor", actor), ("hex", hexes), ("glob", glob)):
        if arr.size and (int(arr.min()) < 0 or int(arr.max()) >= rows):
            raise ValueError(f"static_index: {name} slot outside the {rows} rows")
    return StaticIndex(cu_seqlens=cu.astype(np.int32), actor=actor.reshape(-1),
                       hex=hexes.reshape(-1), glob=glob, actor_kind=kinds)


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
class _State:
    """One bucket's static buffers and, once captured, its graph."""
    bucket: Bucket
    x: torch.Tensor                     # [t_cap, d] fp32 on the device
    index_layout: FlatLayout
    index_host: torch.Tensor            # pinned uint8
    mask_layout: FlatLayout
    mask_host: torch.Tensor             # pinned uint8
    out_layout: Optional[FlatLayout] = None
    out_host: Optional[torch.Tensor] = None
    graph: Optional[torch.cuda.CUDAGraph] = None
    actor_kind: Optional[torch.Tensor] = None   # CPU, set per call (eager) or at capture
    dirty: Tuple[int, int, int] = (0, 0, 0)     # (B, A_max, H_max) the mask views last held
    replays: int = 0
    stats: Dict[str, int] = field(default_factory=dict)


class GraphedServe:
    """Serves (RawEncoded, PackedMasks) batches through per-bucket CUDA
    graphs; see the module docstring. One instance per server process,
    shared by its serve threads (the static buffers are behind a lock;
    the embed runs outside it)."""

    def __init__(self, model, encoder, device: torch.device, *, caps: Caps = Caps(),
                 graphs: bool = True, extras: Sequence[str] = ("value", "value_logits", "cliffness")):
        if getattr(model, "has_gbc", False):
            raise ValueError("graphed serve does not carry the GBC unit context")
        check_packed_trunk_supported(model.encoder)
        self.model, self.encoder, self.device, self.caps = model, encoder, device, caps
        self.graphs = bool(graphs) and device.type == "cuda"
        self.extras = tuple(extras)
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
        self._states: Dict[Bucket, _State] = {}
        # One private memory pool per graph: sharing one across
        # graphs is safe only when they replay in capture order, and
        # buckets replay in whatever order the batches arrive.
        self._lock = threading.Lock()
        self.fallbacks: Dict[str, int] = {}
        self.served = 0

    # -- buckets ---------------------------------------------------------

    def _pick(self, B: int, A_max: int, H_max: int, total: int,
              longest: int) -> Optional[Bucket]:
        c = self.caps
        if B > c.b_cap or longest > c.max_len:
            return None
        a_cap = next((a for a in c.a_caps if a >= A_max), None)
        h_cap = next((h for h in c.h_caps if h >= H_max), None)
        t_cap = next((t for t in c.t_caps if t >= total + (c.b_cap - B)), None)
        if a_cap is None or h_cap is None or t_cap is None:
            return None
        return Bucket(a_cap, h_cap, t_cap)

    def _state(self, bucket: Bucket) -> _State:
        st = self._states.get(bucket)
        if st is not None:
            return st
        c, dev = self.caps, self.device
        pin = dev.type == "cuda"
        d = int(self.model.d_model)
        index_layout = FlatLayout([
            ("cu_seqlens", torch.int32, (c.b_cap + 1,)),
            ("actor", torch.int64, (c.b_cap * bucket.a_cap,)),
            ("hex", torch.int64, (c.b_cap * bucket.h_cap,)),
            ("glob", torch.int64, (c.b_cap,)),
        ])
        ml = mask_layout(c.b_cap, bucket.a_cap, bucket.h_cap, self.T, self.W,
                         type_bias=True, attack_bias=True)
        st = _State(
            bucket=bucket,
            x=torch.zeros(bucket.t_cap, d, dtype=torch.float32, device=dev),
            index_layout=index_layout,
            index_host=torch.zeros(index_layout.nbytes, dtype=torch.uint8, pin_memory=pin),
            mask_layout=ml,
            mask_host=torch.zeros(ml.nbytes, dtype=torch.uint8, pin_memory=pin))
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

    def _body(self, st: _State) -> None:
        """Everything from the two host->device copies to the one
        device->host copy, on the bucket's static buffers. Captured
        once per bucket on CUDA; run as is on CPU."""
        model, dev, c = self.model, self.device, self.caps
        bucket = st.bucket
        B, A, H = c.b_cap, bucket.a_cap, bucket.h_cap
        d = st.x.shape[1]
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
            w = self._weights
            h = self._layers(st.x, index.cu_seqlens, index.max_len, w.layers, w.final_norm)
            actor_ctx = h.index_select(0, index.actor).view(B, A, d)
            hex_ctx = h.index_select(0, index.hex).view(B, H, d)
            global_ctx = h.index_select(0, index.glob).view(B, 1, d)
            sizes = [(0, 0, 0)] * B
            padded = model._heads(actor_ctx, hex_ctx, global_ctx, st.actor_kind, sizes, None)
            if bf16:
                padded = padded.float32()
            extras = [getattr(padded, n) for n in self.extras]
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
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._body(st)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._body(st)
        st.graph = g

    def _run(self, st: _State) -> None:
        if not self.graphs:
            self._body(st)
            return
        if st.graph is None:
            try:
                self._capture(st)
            except Exception as e:                       # noqa: BLE001 -- eager serves this bucket
                log.warning("graphed serve: capture failed for %s (%s: %s); this bucket runs "
                            "its static body eagerly", st.bucket, type(e).__name__, e)
                st.graph = False                         # type: ignore[assignment]
        if st.graph:
            st.graph.replay()
            st.replays += 1
        else:
            self._body(st)

    # -- one batch ---------------------------------------------------------

    def _fallback(self, why: str) -> None:
        self.fallbacks[why] = self.fallbacks.get(why, 0) + 1

    def infer(self, raws, packs: Sequence[PackedMasks]):
        """(compact actions per sample, the extras as host arrays [b_cap, ...],
        the actor kinds [B, A_max] (CPU long), the sizes) -- or None when
        the batch does not fit a bucket, so the caller serves it eagerly."""
        model, encoder, dev, c = self.model, self.encoder, self.device, self.caps
        with torch.no_grad():
            streams = encoder.encode_from_raw_embedded(raws, device=dev)
        sizes = streams.sizes
        B = len(sizes)
        U_max, R_max, H_max = (max(s[i] for s in sizes) for i in (0, 1, 2))
        A_max = U_max + R_max + 1
        lengths = [u + r + h + 2 for u, r, h in sizes]
        total = sum(lengths)
        bucket = self._pick(B, A_max, H_max, total, max(lengths))
        if bucket is None:
            self._fallback("shape")
            return None
        layout = build_packed_layout(sizes, H_max, U_max, R_max, TokenKind, ActorKind,
                                     source="streams")
        sidx = static_index(sizes, c.b_cap, bucket.a_cap, bucket.h_cap, bucket.t_cap)
        with self._lock:
            st = self._state(bucket)
            # The real tokens into the static rows, with their kind term.
            gather = torch.from_numpy(np.concatenate([layout.src, layout.kind]))
            if dev.type == "cuda":
                gather = gather.pin_memory().to(dev, non_blocking=True)
            src, kind = torch.split(gather, [len(layout.src), len(layout.kind)])
            with torch.no_grad():
                st.x[:total].copy_(streams.tokens.index_select(0, src)
                                   + model.token_kind_embed(kind))
            iv = st.index_layout.numpy_views(st.index_host.numpy())
            iv["cu_seqlens"][:] = sidx.cu_seqlens
            iv["actor"][:] = sidx.actor
            iv["hex"][:] = sidx.hex
            iv["glob"][:] = sidx.glob
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
            pending = PendingPriors(host=st.out_host, layout=st.out_layout, n_samples=B,
                                    capacity=c.capacity, n_extras=len(self.extras), device=dev)
            compact, host = pending.finish()
            # Copies: the static output buffer is rewritten by the next batch.
            host = [np.array(a[:B], copy=True) for a in host]
            compact = [type(x)(**{k: np.array(getattr(x, k), copy=True)
                                  for k in ("actor", "kind", "target", "weapon", "prior")})
                       for x in compact]
        self.served += 1
        return compact, host, st.actor_kind[:B, :A_max], [tuple(s) for s in sizes]

    def summary(self) -> Dict[str, object]:
        return {"graphs": self.graphs, "served": self.served, "fallbacks": dict(self.fallbacks),
                "buckets": {f"{b.a_cap}x{b.h_cap}x{b.t_cap}": st.replays
                            for b, st in self._states.items()}}


__all__ = ["Caps", "Bucket", "StaticIndex", "static_index", "GraphedServe"]
