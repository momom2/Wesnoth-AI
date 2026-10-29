"""A served batch past its launch (tools/serve_stage.py): its device work
queued, what its replies are written from once that work is done, and the
pinned host buffers its queued copies read, kept until then.

`serve_stage.serve_requests` completes a batch right after launching it;
`serve_stage.launch_requests` hands it to the pipelined serve loop
(tools/serve_pipelined.py), which completes it after it has launched the
next one. Either way the replies are the same bytes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from wesnoth_ai.byte_wire import DTYPES

_CODE = {dt: i for i, dt in enumerate(DTYPES)}
# The head outputs a reply carries, in byte_wire.REPLY_FIELDS's order.
REPLY_EXTRAS = ("value", "value_logits", "cliffness", "moves_left")
TORCH_CODE = {torch.float32: _CODE[np.dtype(np.float32)], torch.float64: _CODE[np.dtype(np.float64)],
              torch.int32: _CODE[np.dtype(np.int32)], torch.int64: _CODE[np.dtype(np.int64)],
              torch.int8: _CODE[np.dtype(np.int8)], torch.uint8: _CODE[np.dtype(np.uint8)]}


def merge_refused(requests, refused: Dict[int, str], served: List[bytes]) -> List[bytes]:
    """The batch's replies: each refused request answered by its reason,
    the others by `served`, in order."""
    from tools.serve_loop import error_message
    it = iter(served)
    return [error_message(requests[r].rid, f"memory refused: {refused[r]}") if r in refused
            else next(it) for r in range(len(requests))]


class LaunchedBatch:
    """A batch whose device work is queued. `wait()` returns once the
    device work is done, from any thread (the GIL released meanwhile);
    `complete()`, on the launching thread, waits too and returns the
    replies in order, or raises what the batch failed on."""

    def wait(self) -> None:
        raise NotImplementedError

    def complete(self) -> List[bytes]:
        raise NotImplementedError


class RefusedBatch(LaunchedBatch):
    """The memory books' refusals, and the batch's other requests launched."""

    def __init__(self, requests, refused: Dict[int, str], inner: Optional[LaunchedBatch]):
        self._requests, self._refused, self._inner = requests, refused, inner

    def wait(self) -> None:
        if self._inner is not None:
            self._inner.wait()

    def complete(self) -> List[bytes]:
        served = self._inner.complete() if self._inner is not None else []
        return merge_refused(self._requests, self._refused, served)


@dataclass
class EagerBatch(LaunchedBatch):
    """An eager batch past its launch (`serve_stage._launch_eager`)."""
    dev: torch.device
    done: Optional[Any]                  # a cuda event at the batch's end on the stream
    commit: Optional[Any]                # the ServerMemory whose books commit after the device ran
    stats: Optional[Dict[str, float]]
    events: Optional[tuple]              # the timing events (start, end)
    parsed: Any                          # the batch's wesnoth_core.ServeRequests
    out_layout: Any
    out_host: torch.Tensor
    names: List[str]
    memory_arg: Optional[tuple]          # (K_max, d) when the memory rides the wire
    capacity: int
    actor_kind: torch.Tensor
    a_max: int
    versions: Optional[List[int]]
    keep: tuple                          # the pinned host buffers the queued copies read

    def wait(self) -> None:
        if self.done is not None:
            self.done.synchronize()

    def complete(self) -> List[bytes]:
        t4 = time.perf_counter()
        if self.dev.type == "cuda":
            if self.done is not None:
                self.done.synchronize()
            else:
                torch.cuda.current_stream(self.dev).synchronize()
        if self.commit is not None:
            self.commit.commit()
        t5 = time.perf_counter()
        out_offsets = {name: (off, dt, shape) for name, dt, shape, off, _n in self.out_layout.fields}
        extras = []
        for n in REPLY_EXTRAS:
            if n in self.names:
                off, dt, shape = out_offsets[f"extra{self.names.index(n)}"]
                extras.append((off, TORCH_CODE[dt], list(shape[1:])))
            else:
                extras.append(None)
        memory_arg = None
        if self.memory_arg is not None:
            memory_arg = (out_offsets[f"extra{len(self.names)}"][0],) + self.memory_arg
        actor_kind = np.ascontiguousarray(self.actor_kind.numpy(), dtype=np.int64).reshape(-1)
        replies = self.parsed.replies(self.out_host.numpy(),
                                      [out_offsets[n][0] for n in ("ends", "actor", "kind", "target", "weapon",
                                                                   "prior")],
                                      self.capacity, extras, actor_kind, self.a_max, memory_arg, self.versions)
        t6 = time.perf_counter()
        self.keep = ()
        stats = self.stats
        if self.events is not None:
            stats["gpu_ms"] = stats.get("gpu_ms", 0.0) + self.events[0].elapsed_time(self.events[1])
        if stats is not None:
            for key, dt in (("t_finish", t5 - t4), ("t_replies", t6 - t5)):
                stats[key] = stats.get(key, 0.0) + dt
        return replies
