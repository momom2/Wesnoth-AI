"""Serving side of the actor pool (tools/actor_pool.py holds the design
overview and the manager).

- The manager <-> serve-process protocol (_SRV_* commands, _S_* replies).
- _BatchPicker (with _request_lengths, _Waiting): which queued requests
  share a batch, shared by the serve threads.
- _serve_loop: one serving thread, in the learner process or a serve
  process.
- _server_loop: the spawned serve-process body (ActorPool._spawn_servers'
  Process target; spawn pickles it by this module path).
- _merge_timelines, _best_window_rate, _picker_stats: the serve-stats
  helpers run_iteration merges with.

Logs under the pool's logger ("actor_pool").
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue as _queue
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from tools.actor_worker import _set_fd_safe_sharing

log = logging.getLogger("actor_pool")

# Serve-process control commands (main -> serve process).
_SRV_SYNC = "sync"        # (version, state bytes): load these weights
_SRV_SERVE = "serve"      # (iter_idx,): start the serve threads
_SRV_PAUSE = "pause"      # (): stop the serve threads, reply their stats
_SRV_PROBE = "probe"      # (payload,): one infer_batch outside serving
_SRV_STOP = "stop"
# Serve-process replies (serve process -> main), on the shared server queue.
_S_READY = "ready"        # model built
_S_SYNCED = "synced"      # payload: the version loaded
_S_STATS = "stats"        # payload: {"threads": [stats dicts], "picker": {...},
                          #           "packed_compile": model.packed_compile_stats()}
_S_PROBE = "probe"        # payload: wire outputs
_S_ERROR = "error"        # payload: traceback string


def _request_lengths(payload) -> List[int]:
    """Hex + unit tokens of every leaf of a request (a PackedRequest
    carries them in its headers; a legacy list carries RawEncodeds or
    (RawEncoded, PackedMasks) pairs): the sequence length a batch pads
    to. The packed trunk pads nothing, but the heads and the priors'
    mask kernels still run over the batch's longest leaf."""
    headers = getattr(payload, "headers", None)
    if headers is not None:
        return [h.n_hexes + h.n_units for h in headers]
    return [len(r[0].hex_xs) + len(r[0].unit_xs) if isinstance(r, tuple)
            else len(r.hex_xs) + len(r.unit_xs) for r in payload]


@dataclass(slots=True)
class _Waiting:
    item: tuple             # (actor id, request id, payload), as queued
    seq: int                # arrival order
    n_leaves: int
    lens: List[int]         # hex + unit tokens per leaf
    tokens: int             # the request's longest leaf
    skipped: int = 0        # batches formed while this request waited


class _BatchPicker:
    """Which queued requests share a batch (docs/gpu_forward_design_
    20260904.md section 7), shared by the serve threads. "fifo":
    arrival order until the batch holds max_batch leaves (the rule
    since 2026-07-22). "length": the same whenever everything waiting
    fits one batch; otherwise the oldest request anchors the batch and
    the requests nearest to it in token count fill it (one request is
    one tree on one map, so its token count is its longest leaf), and
    `gap` > 0 refuses any request further than that many tokens from
    the anchor. A request the length rule passes over although the
    fifo rule would have served it now (it was inside the fifo batch)
    is DISPLACED: it goes into the very next batch ahead of the anchor
    rule -- the displaced requests are a subset of one fifo batch, so
    they always fit -- and no request is displaced twice. The bound is
    per displacement: a request behind a long queue can still wait
    several batches while the rule prefers younger requests of its
    shape. `skipped` counts the displaced requests, each once."""

    def __init__(self, policy: str = "fifo", gap: int = 0):
        if policy not in ("fifo", "length"):
            raise ValueError(f"coalesce policy must be 'fifo' or 'length', got {policy!r}")
        self.policy = policy
        self.gap = int(gap)
        self._lock = threading.Lock()
        self._waiting: List[_Waiting] = []
        self._seq = 0
        # Telemetry: requests displaced by the length rule (each once),
        # and the waiting requests seen at each pick (queue depth).
        self.skipped = 0
        self.picks = 0
        self.depth = 0

    def take(self, queue, max_batch: int, timeout: float) -> List[_Waiting]:
        """Moves everything queued into the waiting list (blocking up to
        `timeout` for a first request only when nothing waits) and
        returns the next batch; empty when nothing arrived."""
        fresh = []
        with self._lock:
            idle = not self._waiting
        if idle:
            try:
                fresh.append(queue.get(timeout=timeout))
            except _queue.Empty:
                return []
        while True:
            try:
                fresh.append(queue.get_nowait())
            except _queue.Empty:
                break
        with self._lock:
            for item in fresh:
                self._waiting.append(self._wrap(item))
            return self._pick(max_batch)

    def flush(self) -> List[_Waiting]:
        """Every request still parked, removed: serving is stopping
        and nothing will pick them (the picker is rebuilt on the next
        start)."""
        with self._lock:
            parked, self._waiting = self._waiting, []
        return parked

    def _wrap(self, item) -> _Waiting:
        lens = _request_lengths(item[2])
        self._seq += 1
        return _Waiting(item=item, seq=self._seq, n_leaves=len(lens), lens=lens,
                        tokens=max(lens) if lens else 0)

    @staticmethod
    def _fifo_count(waiting: List[_Waiting], max_batch: int) -> int:
        """How many of the waiting requests, in arrival order, the fifo
        rule serves now: the batch fills until it holds max_batch
        leaves, the last request overshooting (the rule since
        2026-07-22)."""
        n = k = 0
        while k < len(waiting) and n < max_batch:
            n += waiting[k].n_leaves
            k += 1
        return k

    def _pick(self, max_batch: int) -> List[_Waiting]:
        waiting = self._waiting
        if not waiting:
            return []
        self.picks += 1
        self.depth += len(waiting)
        k = self._fifo_count(waiting, max_batch)
        if self.policy == "fifo":
            batch, self._waiting = waiting[:k], waiting[k:]
            return batch
        # Displaced last time: all of them, by age. They are a subset
        # of the fifo batch of that pick, so they fit one batch under
        # the same overshoot rule.
        batch = [w for w in waiting if w.skipped]
        n = sum(w.n_leaves for w in batch)
        rest = [w for w in waiting if not w.skipped]
        if rest and n < max_batch:
            anchor = max(w.tokens for w in batch) if batch else rest[0].tokens
            rest.sort(key=lambda w: (abs(w.tokens - anchor), w.seq))
            for w in rest:
                if n >= max_batch or (self.gap and abs(w.tokens - anchor) > self.gap):
                    break
                batch.append(w)
                n += w.n_leaves
        chosen = {w.seq for w in batch}
        # Passed over although the fifo rule would have served them
        # now: displaced, first in the next batch.
        for w in waiting[:k]:
            if w.seq not in chosen:
                w.skipped = 1
                self.skipped += 1
        self._waiting = [w for w in waiting if w.seq not in chosen]
        return batch


def _merge_timelines(per_thread: List[List[Tuple[float, int]]],
                     t_start: float) -> List[Tuple[float, int]]:
    """Total leaves served by time t (seconds since t_start), from the
    threads' (monotonic time, cumulative leaves) marks."""
    marks = sorted((t, i, n) for i, tl in enumerate(per_thread) for (t, n) in tl)
    latest = [0] * len(per_thread)
    out: List[Tuple[float, int]] = []
    for t, i, n in marks:
        latest[i] = n
        out.append((t - t_start, sum(latest)))
    return out


def _best_window_rate(timeline: List[Tuple[float, int]], window: float) -> Optional[float]:
    """Highest leaves/s over any span of at least `window` seconds
    between two marks; None when the timeline is shorter than that."""
    best = None
    j = 0
    for i, (t_i, n_i) in enumerate(timeline):
        while j < len(timeline) and timeline[j][0] < t_i + window:
            j += 1
        if j >= len(timeline):
            break
        t_j, n_j = timeline[j]
        rate = (n_j - n_i) / (t_j - t_i)
        best = rate if best is None or rate > best else best
    return best


def _picker_stats(picker: Optional[_BatchPicker]) -> Dict[str, int]:
    if picker is None:
        return {"skipped": 0, "picks": 0, "depth": 0}
    return {"skipped": picker.skipped, "picks": picker.picks, "depth": picker.depth}


def _serve_loop(server, picker: _BatchPicker, req_q, resp_qs, max_batch: int,
                serve_timeout: float, stop_ev, stats_out: List[Dict]) -> None:
    """One serving thread, in the learner process or a serve process:
    take a batch (the picker coalesces the queued requests) -> unpack
    -> encode+forward -> wire-serialize -> reply. Stage times are
    accumulated locally (no locks on the hot path) and appended to
    `stats_out` on exit; on CUDA the seam adds the host seconds of its
    own stages to the same dict."""
    from tools.inference_seam import output_to_wire
    from wesnoth_ai.leaf_wire import PackedRequest, unpack_request
    # (monotonic time, cumulative leaves) every ~10 s: the iteration
    # average hides the tail where most actors have finished
    # (2026-09-05 whole-pool profile: median game finish at 40% of
    # the wall); run_iteration derives the saturated rate from it.
    # time.monotonic is one system-wide clock, so the marks of every
    # process merge on the manager's time base.
    timeline: List[Tuple[float, int]] = []
    next_mark = time.monotonic()
    st = {"wait": 0.0, "unpack": 0.0, "infer": 0.0, "wire": 0.0, "put": 0.0, "gpu_ms": 0.0,
          "timeline": timeline,
          "leaves": 0, "batches": 0, "requests": 0, "tokens": 0, "padded": 0}
    while not stop_ev.is_set():
        t0 = time.monotonic()
        batch = picker.take(req_q, max_batch, serve_timeout)
        t1 = time.monotonic()
        st["wait"] += t1 - t0
        if not batch:
            continue
        flat = []
        for w in batch:
            payload = w.item[2]
            flat.extend(unpack_request(payload) if isinstance(payload, PackedRequest)
                        else payload)
        t2 = time.monotonic()
        try:
            outs = server.infer_batch(flat, stats=st)
            t3 = time.monotonic()
            wires = [output_to_wire(o) for o in outs]
        except Exception:                       # noqa: BLE001
            # A serve-thread death used to hang every actor waiting
            # on this batch (2026-09-04: the stats line below choked
            # on (raw, masks) items). Reply with a failure marker so
            # the actors raise instead.
            log.error("inference server failed on a batch of %d leaves:\n%s",
                      len(flat), traceback.format_exc())
            for w in batch:
                aid, rid, _payload = w.item
                resp_qs[aid].put((rid, None))
            continue
        t4 = time.monotonic()
        i = 0
        for w in batch:
            aid, rid, _payload = w.item
            resp_qs[aid].put((rid, wires[i:i + w.n_leaves]))
            i += w.n_leaves
        t5 = time.monotonic()
        st["unpack"] += t2 - t1
        st["infer"] += t3 - t2
        st["wire"] += t4 - t3
        st["put"] += t5 - t4
        st["leaves"] += len(flat)
        st["batches"] += 1
        st["requests"] += len(batch)
        if t5 >= next_mark:
            timeline.append((t5, st["leaves"]))
            next_mark = t5 + 10.0
        # Sequence lengths: hex tokens + unit tokens per leaf, and
        # what the batch pads to (its longest leaf).
        lens = [n for w in batch for n in w.lens]
        st["tokens"] += sum(lens)
        st["padded"] += len(lens) * max(lens)
    # Requests the picker lifted out of the queue but never batched:
    # the picker is rebuilt on the next start, so nothing would answer
    # them and their actors would block forever (2026-09-05 review;
    # the hard-deadline abandon reaches here with actors in flight).
    # Fail them so the actors raise. Requests still in the queue itself
    # stay there for the next start's threads.
    parked = picker.flush()
    if parked:
        log.error("serving stopped with %d request(s) parked in the picker (actors %s); "
                  "failing them", len(parked), sorted({w.item[0] for w in parked}))
        for w in parked:
            aid, rid, _payload = w.item
            resp_qs[aid].put((rid, None))
    stats_out.append(st)


def _server_loop(
    server_id: int, ctrl_q, server_q, req_q, resp_qs, blueprint, switches: Dict,
    device_str: str, serve_threads: int, max_batch: int, serve_timeout: float,
    coalesce: str, coalesce_gap: int, log_level: int, torch_threads: int,
) -> None:
    """Serve-process body: build the inference pair at the learner's
    architecture and switches, report READY, then answer the control
    queue (SYNC weights, SERVE / PAUSE the serve threads on this
    process's request queue, PROBE, STOP). Every failure is a reply on
    `server_q`, never a silent death; the process also exits on its
    own when the learner process is gone, so no CUDA context outlives
    the campaign."""
    logging.basicConfig(level=log_level,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    torch.set_num_threads(max(1, torch_threads))
    _set_fd_safe_sharing()
    from tools.inference_seam import (
        InferenceServer, build_inference_pair, load_inference_state, output_to_wire,
    )
    try:
        device = torch.device(device_str)
        model, encoder = build_inference_pair(blueprint, device)
        model.infer_autocast_bf16 = bool(switches["model_bf16"])
        model.infer_packed_trunk = bool(switches["packed_trunk"])
        if switches["compile_packed"]:
            model.configure_packed_compile(backend=switches["compile_backend"],
                                           mode=switches["compile_mode"])
            log.info("serve-%d packed compile warmup: %s", server_id,
                     model.warmup_packed_compile())
        server = InferenceServer(model, encoder, device=device,
                                 output_device=torch.device("cpu"),
                                 autocast_bf16=switches["autocast_bf16"],
                                 packed_embed=switches["packed_embed"])
    except Exception:                           # noqa: BLE001
        server_q.put((_S_ERROR, server_id, traceback.format_exc()))
        return
    server_q.put((_S_READY, server_id, None))
    parent = mp.parent_process()
    threads: List[threading.Thread] = []
    stop_ev = threading.Event()
    stats: List[Dict] = []
    picker: Optional[_BatchPicker] = None
    while True:
        try:
            cmd = ctrl_q.get(timeout=2.0)
        except _queue.Empty:
            if parent is not None and not parent.is_alive():
                log.error("serve-%d: the learner process is gone; exiting", server_id)
                break
            continue
        kind = cmd[0]
        try:
            if kind == _SRV_STOP:
                break
            if kind == _SRV_SYNC:
                if threads:
                    raise RuntimeError("SYNC while serving: weights change only "
                                       "between iterations")
                _, version, blob = cmd
                load_inference_state(blob, model, encoder, device)
                server_q.put((_S_SYNCED, server_id, int(version)))
            elif kind == _SRV_SERVE:
                if threads:
                    raise RuntimeError("SERVE while already serving")
                stop_ev = threading.Event()
                stats = []
                picker = _BatchPicker(coalesce, coalesce_gap)
                # `threads` holds started threads only: a start that
                # fails (the container's PID limit) is reported as an
                # error reply, the manager aborts the iteration, and
                # PAUSE joins what runs.
                for i in range(serve_threads):
                    th = threading.Thread(
                        target=_serve_loop,
                        args=(server, picker, req_q, resp_qs, max_batch, serve_timeout,
                              stop_ev, stats),
                        daemon=True, name=f"serve-{server_id}-{i}")
                    th.start()
                    threads.append(th)
            elif kind == _SRV_PAUSE:
                stop_ev.set()
                for th in threads:
                    th.join(timeout=10.0)
                threads = []
                server_q.put((_S_STATS, server_id,
                              {"threads": list(stats), "picker": _picker_stats(picker),
                               "packed_compile": model.packed_compile_stats()}))
            elif kind == _SRV_PROBE:
                outs = server.infer_batch(cmd[1])
                server_q.put((_S_PROBE, server_id, [output_to_wire(o) for o in outs]))
            else:
                raise RuntimeError(f"unknown serve-process command {kind!r}")
        except Exception:                       # noqa: BLE001
            server_q.put((_S_ERROR, server_id, traceback.format_exc()))
    stop_ev.set()
    for th in threads:
        th.join(timeout=10.0)

