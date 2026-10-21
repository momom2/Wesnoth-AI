"""The inference server's batch with its host work in Rust, and the memory
states it keeps between decisions.

`serve_requests` answers a batch of request messages: the Rust core
parses them in place (`wesnoth_core.ServeRequests`) and writes every
host buffer straight from them, the embedding stage in
`GameStateEncoder.stage_raws`'s layout, the packed trunk's index, the
masks with their compaction capacity and the memory states, then the
replies from the priors' output buffer; Python keeps the torch calls.
The buffers and the replies are what the Python path
(`tools/eval_inference_server.py` with `--no-rust-stage`) builds, byte for
byte (tests/test_serve_stage.py), so a game plays the same either way.
Two pinned buffers cross to the device per batch: the embedding stage,
and the masks with the index and the memory's.

`ServerMemory` keeps each game-side's memory state on the device between
its decisions, instead of the state crossing the wire both ways at every
decision (own1's 64 slots of width 384: 98 KB each way per leaf). A
leaf names its state by the key its client gave the game-side and the
version it read (`byte_wire.ServedMemory`); the books are the Rust
core's (`wesnoth_core.MemoryTable`): two device rows per game-side, its
last two states, so a decision re-decided after a refused recruit reads
the state before the refused one. A read of a version the books do not
hold is refused, never served from another state; the states of a
connection go when its client releases them (the end of each game, in a
`finally`) or when the connection closes (a worker that exits or is
killed), and a restarted server holds none.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from wesnoth_ai.byte_wire import DTYPES, REQUEST_SCALARS
from wesnoth_ai.encoder import MAX_FACTIONS, StagedBatch
from wesnoth_ai.leaf_wire import ALL_FIELDS, MASK_FIELDS, PARITY_FIELDS
from wesnoth_ai.memory import MemoryBatch
from wesnoth_ai.model import ActorKind, TokenKind, UnitActionType
from wesnoth_ai import profile_regions
from wesnoth_ai.packed_trunk import (FlatLayout, PackedEmbeddedIndex, PackedIndex, PackedLayout,
                                     batch_counts)
from wesnoth_ai.profile_regions import region
from wesnoth_ai.server_priors import mask_layout, priors_outputs

log = logging.getLogger(__name__)

FIELD_NAMES = tuple(name for name, _ in ALL_FIELDS)
FIELD = {name: i for i, name in enumerate(FIELD_NAMES)}
SCALAR = {name: i for i, name in enumerate(REQUEST_SCALARS)}
_CODE = {dt: i for i, dt in enumerate(DTYPES)}
FIELD_CODES = tuple(_CODE[dt] for _, dt in ALL_FIELDS)
# A leaf may leave out the parity streams and its memory state.
FIELD_OPTIONAL = tuple(name == "memory_state" or name in {n for n, _ in PARITY_FIELDS}
                       for name in FIELD_NAMES)
REQUEST_HEAD_WORDS = 3            # tools/serve_loop.py: kind, request id, leaves
_TOKEN_KINDS = (TokenKind.HEX, TokenKind.UNIT, TokenKind.RECRUIT, TokenKind.SIGHTING,
                TokenKind.MEMORY, TokenKind.GLOBAL, TokenKind.END_TURN)
_ACTOR_KINDS = (ActorKind.UNIT, ActorKind.RECRUIT, ActorKind.END_TURN)
_MASK_NAMES = ("actor_mask", "slot_kind", "n_attacks", "type_valid", "attack_bits", "move_bits",
               "union_bits")
# The head outputs a reply carries, in byte_wire.REPLY_FIELDS's order.
_REPLY_EXTRAS = ("value", "value_logits", "cliffness", "aux_score", "moves_left")
_TORCH_CODE = {torch.float32: _CODE[np.dtype(np.float32)], torch.float64: _CODE[np.dtype(np.float64)],
               torch.int32: _CODE[np.dtype(np.int32)], torch.int64: _CODE[np.dtype(np.int64)],
               torch.int8: _CODE[np.dtype(np.int8)], torch.uint8: _CODE[np.dtype(np.uint8)]}
assert tuple(n for n, _ in MASK_FIELDS)[:6] == ("actor_mask", "type_valid", "attack_valid", "move_valid",
                                               "union_valid", "n_attacks")


def parse_requests(messages: Sequence[bytes]):
    """The batch's request messages parsed in place (`ServeRequests`)."""
    import wesnoth_core
    return wesnoth_core.ServeRequests(list(messages), REQUEST_HEAD_WORDS, list(FIELD_NAMES),
                                      list(REQUEST_SCALARS), list(FIELD_CODES), list(FIELD_OPTIONAL))


# ---------------------------------------------------------------------
# The memory states kept on the server
# ---------------------------------------------------------------------

class ServerMemory:
    """Every game-side's last two memory states on the device, float32
    [rows, slots, d] (row 0 the learned initial memory), and their books
    (`wesnoth_core.MemoryTable`). Used by one serve thread."""

    def __init__(self, model, device: torch.device, *, max_keys: int = 64):
        import wesnoth_core
        self.slots = int(getattr(model, "memory_slots", 0) or 0)
        if not self.slots:
            raise ValueError("server-side memory for a model without a memory")
        self.model, self.device = model, device
        self.d = int(model.d_model)
        self.books = wesnoth_core.MemoryTable(int(max_keys))
        self.states = torch.zeros(8, self.slots, self.d, dtype=torch.float32, device=device)
        self._initial_version: Optional[int] = None
        self.refused = 0
        self.released = 0

    def plan(self, cids, keys, versions, ks):
        """Per leaf the row it reads, the row it writes and why it is
        refused (None when it is not); see `MemoryTable.plan`."""
        reads, writes, why = self.books.plan([int(c) for c in cids], [int(k) for k in keys],
                                             [int(v) for v in versions], [int(k) for k in ks])
        self.refused += sum(w is not None for w in why)
        return reads, writes, why

    def _ready(self) -> None:
        """Rows for every state the books hold, and row 0 at the model's
        current learned initial memory."""
        need = int(self.books.rows)
        if need > self.states.size(0):
            grown = torch.zeros(max(need, 2 * self.states.size(0)), self.slots, self.d,
                                dtype=torch.float32, device=self.device)
            grown[:self.states.size(0)] = self.states
            self.states = grown
        version = int(getattr(self.model, "_weights_version", 0))
        if self._initial_version != version:
            with torch.no_grad():
                self.states[0] = self.model.initial_memory(self.slots).to(self.device)
            self._initial_version = version

    def gather(self, reads: torch.Tensor, counts: torch.Tensor, host_counts: List[int]) -> MemoryBatch:
        """The batch's memory as the trunk reads it: each leaf's state from
        row `reads[b]`, zeros past its slot count `counts[b]`, as
        `InferenceServer._memory_batch` pads them. `reads` and `counts`
        are int64 [B] on the device."""
        self._ready()
        k_max = max(host_counts, default=0)
        rows = self.states.index_select(0, reads)[:, :k_max]
        keep = torch.arange(k_max, device=self.device).view(1, k_max) < counts.view(-1, 1)
        padded = torch.where(keep.unsqueeze(-1), rows, rows.new_zeros(()))
        return MemoryBatch(padded=padded, counts=list(host_counts))

    def scatter(self, writes: torch.Tensor, new_memory: torch.Tensor) -> None:
        """The batch's new states [B, K_max, d] into rows `writes`."""
        k_max = new_memory.size(1)
        self.states[:, :k_max].index_copy_(0, writes, new_memory.float())

    def commit(self) -> None:
        self.books.commit()

    def abort(self) -> None:
        self.books.abort()

    def release_connection(self, cid: int) -> None:
        self.released += int(self.books.release_connection(int(cid)))

    def summary(self) -> Dict[str, int]:
        return {"held": len(self.books), "rows": int(self.states.size(0)), "refused": self.refused,
                "released": self.released}


# ---------------------------------------------------------------------
# One batch
# ---------------------------------------------------------------------

def _packed_trunk_applies(model, device: torch.device, bf16: bool) -> bool:
    """`WesnothModel._packed_trunk_applies` for this server's forwards."""
    probe = torch.empty(0, device=device)
    if bf16 and device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return bool(model._packed_trunk_applies(probe))
    return bool(model._packed_trunk_applies(probe))


def _embed_layout(encoder, totals: Dict[str, int], B: int):
    """`GameStateEncoder.stage_raws`'s layout and its fill plan: per field
    (wire field or scalar, byte offset, bytes, exact)."""
    spec = encoder._stream_fields()
    their = (("their_faction_probs", torch.float32, (B, MAX_FACTIONS)) if encoder.observation_parity
             else ("their_faction_id", torch.int64, (B,)))
    fields = [(name, dt, (totals[stream],) + shape) for stream, stream_fields in spec
              for name, dt, shape in stream_fields]
    fields += [("global_feats", torch.float32, (B, encoder.widths.global_feats)),
               ("our_faction_id", torch.int64, (B,)), their]
    layout = FlatLayout(fields)
    plan = []
    for name, _dt, _shape, off, n in layout.fields:
        src = FIELD.get(name)
        if src is None:
            src = -1 - SCALAR[name]
        plan.append((src, off, n, True))
    return layout, plan, their[0]


def serve_requests(server, requests, stats: Optional[Dict[str, float]] = None, *,
                   memory: Optional[ServerMemory] = None, cids: Optional[Sequence[int]] = None
                   ) -> List[bytes]:
    """One reply message per request (`requests`: objects with `message`
    and `rid`; `cids`: each request's connection, with `memory`). A
    request the memory books refuse gets an error reply and the rest are
    served. `server`: the `InferenceServer` whose model, encoder, device
    and precision serve the batch."""
    t0 = time.perf_counter()
    parsed = parse_requests([r.message for r in requests])
    counts = parsed.memory_counts(_slots(server._model))
    if memory is None and (parsed.scalars(SCALAR["memory_key"]) >= 0).any():
        raise ValueError("a leaf names a memory state the server keeps, and this server keeps none")
    plan = None
    if memory is not None and counts is not None:
        keys = parsed.scalars(SCALAR["memory_key"])
        versions = parsed.scalars(SCALAR["memory_version"])
        leaf_cids = np.repeat(np.asarray(cids, dtype=np.int64), parsed.per_request())
        reads, writes, why = memory.plan(leaf_cids, keys, versions, counts)
        if any(w is not None for w in why):
            by_request = dict(zip((id(r) for r in requests), cids))
            return answer_refused(requests, parsed.per_request(), why, memory, lambda kept: serve_requests(
                server, kept, stats, memory=memory, cids=[by_request[id(r)] for r in kept]))
        plan = (reads, writes, [int(v) + 1 for v in versions])
    t1 = time.perf_counter()
    try:
        replies = _serve_parsed(server, parsed, counts, stats, memory, plan, t0, t1)
    except BaseException:
        if memory is not None:
            memory.abort()
        raise
    return replies


def answer_refused(requests, per_request: Sequence[int], why, memory: ServerMemory,
                   serve_kept: Callable[[list], List[bytes]]) -> List[bytes]:
    """The batch's replies when the memory books refused some of its
    leaves: each request with a refused leaf answered by the reason, the
    others served by `serve_kept` (their replies in order)."""
    from tools.serve_loop import error_message
    memory.abort()
    refused: Dict[int, str] = {}
    b = 0
    for r, n in enumerate(per_request):
        reasons = [w for w in why[b:b + n] if w is not None]
        if reasons:
            refused[r] = reasons[0]
        b += n
    for r, reason in refused.items():
        log.error("request %d refused: %s", requests[r].rid, reason)
    kept = [requests[r] for r in range(len(requests)) if r not in refused]
    served = iter(serve_kept(kept) if kept else [])
    return [error_message(requests[r].rid, f"memory refused: {refused[r]}") if r in refused
            else next(served) for r in range(len(requests))]


@dataclass
class HandlePlan:
    """The memory plan of a batch the Python path decoded: the states it
    reads on the device, the rows it writes, the versions it writes, and
    per leaf why it is refused."""
    why: List[Optional[str]]
    memory_batch: Optional[MemoryBatch] = None
    writes: Optional[torch.Tensor] = None
    versions: Optional[List[int]] = None

    @property
    def refused(self) -> bool:
        return any(w is not None for w in self.why)


def plan_handles(memory: ServerMemory, items, leaf_cids: Sequence[int]) -> HandlePlan:
    """`ServerMemory.plan` over decoded leaves (`byte_wire.decode_request`:
    each leaf's MemoryState carries a `ServedMemory`), and the states
    gathered when none is refused."""
    from wesnoth_ai.byte_wire import ServedMemory
    keys, versions, ks = [], [], []
    for it in items:
        m = it[2] if len(it) > 2 else None
        state = None if m is None else m.state
        if isinstance(state, ServedMemory):
            keys.append(state.key)
            versions.append(state.version)
        else:
            keys.append(-1)
            versions.append(0)
        ks.append(-1 if m is None else int(m.k))
    reads, writes, why = memory.plan(leaf_cids, keys, versions, ks)
    if any(w is not None for w in why):
        return HandlePlan(why=why)
    dev = memory.device
    return HandlePlan(why=why, memory_batch=memory.gather(torch.as_tensor(reads, device=dev),
                                                          torch.as_tensor(ks, device=dev), ks),
                      writes=torch.as_tensor(writes, device=dev), versions=[v + 1 for v in versions])


def _serve_graphed(server, gserve, parsed, counts, stats, memory, plan, t0: float, t1: float
                   ) -> Optional[List[bytes]]:
    """The batch from the graphed path (`GraphedServe.infer_parsed`), or
    None when it does not fit a bucket (the eager path serves it). A
    failure disables the graphed path for good, as `_serve_priors` does."""
    dev = server._device
    states = writes = None
    if plan is not None:
        index = torch.as_tensor(np.asarray([plan[0], plan[1], counts], dtype=np.int64)).to(dev)
        states = memory.gather(index[0], index[2], list(counts))
        writes = index[1]
    try:
        got = gserve.infer_parsed(parsed, counts, device_states=states)
    except Exception:                                    # noqa: BLE001 -- the eager path serves from here on
        log.exception("graphed serve failed on a batch of %d; serving eager from now on", len(parsed))
        server._graphed = server._graphed_factory = None
        server._graphed_by_thread.clear()
        return None
    if got is None:
        return None
    if plan is not None:
        memory.scatter(writes, got.memory)
        memory.commit()
    t2 = time.perf_counter()
    offsets = {name: (off, dt, shape) for name, dt, shape, off, _n in got.layout.fields}
    extras = []
    for n in _REPLY_EXTRAS:
        if n in got.extras:
            off, dt, shape = offsets[f"extra{got.extras.index(n)}"]
            extras.append((off, _TORCH_CODE[dt], list(shape[1:])))
        else:
            extras.append(None)
    memory_arg = None
    if got.wire_memory:
        off, _dt, shape = offsets[f"extra{len(got.extras)}"]
        memory_arg = (off, got.k_cap, int(shape[2]))
    replies = parsed.replies(got.out, [offsets[n][0] for n in ("ends", "actor", "kind", "target", "weapon",
                                                               "prior")],
                             got.capacity, extras, got.actor_kind, got.a_cap, memory_arg,
                             plan[2] if plan is not None else None)
    if stats is not None:
        for key, dt in (("t_parse", t1 - t0), ("t_graphed", t2 - t1), ("t_replies", time.perf_counter() - t2)):
            stats[key] = stats.get(key, 0.0) + dt
        stats["graphed_batches"] = stats.get("graphed_batches", 0.0) + 1
    return replies


def _serve_parsed(server, parsed, counts, stats, memory, plan, t0: float, t1: float) -> List[bytes]:
    gserve = server._graphed_serve()
    if gserve is not None and (memory is not None) == bool(getattr(gserve, "device_memory", False)):
        replies = _serve_graphed(server, gserve, parsed, counts, stats, memory, plan, t0, t1)
        if replies is not None:
            return replies
    model, encoder, dev = server._model, server._encoder, server._device
    cuda = dev.type == "cuda"
    bf16 = server._use_bf16()
    timing = stats is not None and cuda
    if timing:
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()
    B = len(parsed)
    sizes = parsed.sizes()
    parity = bool(encoder.observation_parity)
    if parsed.all_present(FIELD["their_faction_probs"]) != parity:
        raise ValueError(f"leaves {'without' if parity else 'with'} the parity observation's fields "
                         f"reached an encoder {'with' if parity else 'without'} observation_parity")
    sightings = parsed.sighting_counts() if parity else None
    if parity and sightings is None:
        raise ValueError("a parity leaf without its sighting stream")
    # The embedding stage, in stage_raws's layout.
    totals = {"hex": sum(s[2] for s in sizes), "unit": sum(s[0] for s in sizes),
              "recruit": sum(s[1] for s in sizes)}
    if parity:
        totals["sighting"] = sum(sightings)
    layout, fill_plan, their_field = _embed_layout(encoder, totals, B)
    embed_host = torch.empty(layout.nbytes, dtype=torch.uint8, pin_memory=cuda)
    parsed.fill_concat(embed_host.numpy(), fill_plan)
    # The masks, the packed index and the memory's arrays in one buffer.
    _total, A_max, H_max, _U_max, _R_max, _S_max, K_max, mem_rows, longest = parsed.index_dims(counts)
    T, W = int(UnitActionType.COUNT), int(model.max_attacks)
    packed = _packed_trunk_applies(model, dev, bf16)
    ml = mask_layout(B, A_max, H_max, T, W)
    aux_fields = [(name, dt, shape) for name, dt, shape, _o, _n in ml.fields]
    n_index = 0
    if packed:
        n_index = 2 * _total + B * (2 * A_max + H_max + _U_max + 1 + K_max) + B + 1 + mem_rows
        aux_fields.append(("index", torch.int64, (n_index,)))
    wire_memory = counts is not None and memory is None
    if counts is not None and memory is not None:
        aux_fields.append(("memory_index", torch.int64, (3, B)))
    aux = FlatLayout(aux_fields)
    aux_host = torch.empty(aux.nbytes, dtype=torch.uint8, pin_memory=cuda)
    aux_np = aux_host.numpy()
    offsets = {name: off for name, _dt, _shape, off, _n in aux.fields}
    HB = (H_max + 7) // 8
    capacity = parsed.fill_masks(aux_np, [offsets[n] for n in _MASK_NAMES], A_max, T, HB, H_max, W,
                                 (0, aux.nbytes))
    if packed:
        parsed.fill_packed_index(aux_np, offsets["index"], _TOKEN_KINDS, _ACTOR_KINDS, counts)
    initial = []
    if wire_memory:
        # Their own buffer: the mask buffer is cleared whole, these rows are
        # written once.
        d = int(model.d_model)
        mem_host = torch.empty(B * K_max * d * 4, dtype=torch.uint8, pin_memory=cuda)
        initial = parsed.fill_memory_states(mem_host.numpy(), 0, K_max, d, _slots(model))
    host_views = aux.numpy_views(aux_np)
    if plan is not None:
        mi = host_views["memory_index"]
        mi[0] = plan[0]
        mi[1] = plan[1]
        mi[2] = counts
    if profile_regions.active():
        lengths = [u + r + h + (sightings[b] if sightings else 0) + (counts[b] if counts else 0) + 2
                   for b, (u, r, h) in enumerate(sizes)]
        profile_regions.note_batch(lengths, A_max, H_max, K_max)
    with torch.no_grad():
        with region("h2d"):
            embed_dev = embed_host if not cuda else embed_host.to(dev, non_blocking=True)
            aux_dev = aux_host if not cuda else aux_host.to(dev, non_blocking=True)
        staged = StagedBatch(views=layout.torch_views(embed_dev), totals=totals,
                             sizes=[tuple(s) for s in sizes], sighting_counts=sightings,
                             their_field=their_field)
        v = aux.torch_views(aux_dev)
        prebuilt = _packed_index(v, host_views, parsed, sizes, sightings, counts, A_max, H_max, _U_max,
                                 K_max, _total, mem_rows, longest) if packed else None
        t_mem = time.perf_counter()
        states = None
        if wire_memory:
            mem_dev = mem_host if not cuda else mem_host.to(dev, non_blocking=True)
            padded_mem = mem_dev.view(torch.float32).view(B, K_max, d)
            for b in initial:
                if counts[b]:
                    padded_mem[b, :counts[b]] = model.initial_memory(counts[b]).to(dev)
            states = MemoryBatch(padded=padded_mem, counts=list(counts))
        elif counts is not None:
            with region("memory_gather"):
                states = memory.gather(v["memory_index"][0], v["memory_index"][2], list(counts))
        t2 = time.perf_counter()
        with region("embed"):
            streams = encoder.embed_staged(staged)
        with region("forward"):
            if bf16:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    padded = model.forward_embedded(streams, memory=states, index=prebuilt).float32()
            else:
                padded = model.forward_embedded(streams, memory=states, index=prebuilt)
        t3 = time.perf_counter()
        names = [n for n in _REPLY_EXTRAS if getattr(padded, n) is not None]
        ride = [getattr(padded, n) for n in names]
        if wire_memory:
            ride.append(padded.memory_padded.float())
        mask_views = {n: v[n] for n in _MASK_NAMES}
        with region("priors"):
            out_layout, flat_out = priors_outputs(padded, mask_views, capacity, ride)
            out_host = torch.empty(out_layout.nbytes, dtype=torch.uint8, pin_memory=cuda)
            out_host.copy_(flat_out, non_blocking=True)
        if plan is not None:
            with region("memory_scatter"):
                memory.scatter(v["memory_index"][1], padded.memory_padded)
        if timing:
            ev_end.record()
    t4 = time.perf_counter()
    if cuda:
        torch.cuda.current_stream(dev).synchronize()
    if plan is not None:
        memory.commit()
    t5 = time.perf_counter()
    out_offsets = {name: (off, dt, shape) for name, dt, shape, off, _n in out_layout.fields}
    extras = []
    for n in _REPLY_EXTRAS:
        if n in names:
            off, dt, shape = out_offsets[f"extra{names.index(n)}"]
            extras.append((off, _TORCH_CODE[dt], list(shape[1:])))
        else:
            extras.append(None)
    memory_arg = None
    if wire_memory:
        memory_arg = (out_offsets[f"extra{len(names)}"][0], K_max, int(model.d_model))
    actor_kind = np.ascontiguousarray(padded.actor_kind.numpy(), dtype=np.int64).reshape(-1)
    replies = parsed.replies(out_host.numpy(), [out_offsets[n][0] for n in ("ends", "actor", "kind", "target",
                                                                             "weapon", "prior")],
                             capacity, extras, actor_kind, A_max, memory_arg,
                             plan[2] if plan is not None else None)
    t6 = time.perf_counter()
    if timing:
        stats["gpu_ms"] = stats.get("gpu_ms", 0.0) + ev_start.elapsed_time(ev_end)
    if stats is not None:
        for key, dt in (("t_parse", t1 - t0), ("t_stage", t_mem - t1), ("t_memory", t2 - t_mem),
                        ("t_forward", t3 - t2),
                        ("t_priors", t4 - t3), ("t_finish", t5 - t4), ("t_replies", t6 - t5)):
            stats[key] = stats.get(key, 0.0) + dt
    return replies


def _slots(model) -> int:
    return int(getattr(model, "memory_slots", 0) or 0)


def _packed_index(v, host_views, parsed, sizes, sightings, counts, A_max, H_max, U_max, K_max, total,
                  mem_rows, longest) -> PackedEmbeddedIndex:
    """The `PackedEmbeddedIndex` of the filled index array: the layout on
    the host (numpy views), its device views, the memory's stream rows."""
    B = len(sizes)
    parts = [total, total, B * A_max, B * H_max, B * U_max, B, B * K_max, B + 1, B * A_max, mem_rows]
    dev_parts = torch.split(v["index"], parts)
    host = host_views["index"]
    bounds = np.cumsum([0] + parts)
    hp = [host[bounds[i]:bounds[i + 1]] for i in range(len(parts))]
    layout = PackedLayout(src=hp[0], kind=hp[1], actor=hp[2], hex=hp[3], unit=hp[4], glob=hp[5],
                          cu_seqlens=hp[7], actor_kind=hp[8].reshape(B, A_max), max_len=int(longest),
                          memory=hp[6])
    index = PackedIndex(src=dev_parts[0], kind=dev_parts[1], actor=dev_parts[2], hex=dev_parts[3],
                        unit=dev_parts[4], glob=dev_parts[5], cu_seqlens=dev_parts[7].to(torch.int32),
                        cu_host=hp[7].tolist(), max_len=int(longest), memory=dev_parts[6])
    rows = dev_parts[9] if counts is not None and K_max else None
    return PackedEmbeddedIndex(layout=layout, index=index, memory_rows=rows,
                               counts=batch_counts(sizes, sightings, counts))
