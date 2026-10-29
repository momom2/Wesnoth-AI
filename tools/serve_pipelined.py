"""The eval inference server's serve loop, pipelined (opt-in: the server's
--pipelined, run_elo_batch's --inference-pipelined).

The serial loop (tools/serve_loop.py `InferenceService`) collects a batch,
stages and launches it, waits for the device, writes the replies, and
only then collects the next: the device idles through all the host work
around its own. Here the serve thread launches batch N+1 (its host work
done, its device work queued behind N's) before it completes N (the
device wait, the replies), and collects N+1 while the device runs N:

    collect N+1 | launch N+1 | complete N, reply N | collect N+2 | launch N+2 | ...
    device:  ... N ......................| N+1 .......................| N+2 ...

A waiter thread waits on each launched batch's device work, in launch
order, and wakes the serve thread's wait when it is done, so the collect
of N+1 waits for a first request only while N runs, and N's end closes
its window (the device would idle). A collect that reads only a hello or a
release goes on collecting while N runs.

Why it is safe:
- A game in batch N sends its next request only after N's reply (the
  client is synchronous), so batch N+1 never holds a game of N.
- The memory states the server keeps (tools/serve_stage.py `ServerMemory`)
  live on the device and are read and written by the batches' device work
  in the order the batches are queued, on one stream. The books commit a
  batch's writes at its launch (`serve_stage.launch_requests` says why),
  so N+1's plan, made while N runs, sees them. A connection that closes
  while its request is in N frees its rows; a row N+1 takes for a new
  game-side is written after N's write of it, in stream order.
- A batch whose launch or completion raises is answered as a failed batch
  (the device's health probe, then each request alone through the serial
  `infer_fn`, but for one whose connection has closed: its memory states
  went at the close, and a retry would hold new ones nothing releases).
  A fault that fails the probe stops the service after the batch in
  flight is answered.
- stop() answers the batch in flight before the connections close.

The replies a client receives for a given batch are the serial loop's
bytes (`serve_stage.launch_requests`); which batch a request joins varies
with timing, as it does serially.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing.connection import Listener
from typing import Any, Callable, Dict, List, Optional

from tools.serve_loop import InferenceService, Request

log = logging.getLogger("serve_loop")


@dataclass
class _InFlight:
    """A batch between its launch and its replies."""
    batch: List[Request]
    gpu: Dict[str, float] = field(default_factory=dict)
    launched: Any = None              # launch_fn's batch; None once completed, or when answered at launch
    replies: Optional[list] = None
    errors: Optional[list] = None
    infer_s: float = 0.0              # the serve thread's seconds in the launch and the completion
    done: threading.Event = field(default_factory=threading.Event)   # the device work is done

    def wait_device(self) -> None:
        launched = self.launched
        if launched is not None:
            launched.wait()


class PipelinedInferenceService(InferenceService):
    """`InferenceService` whose batches overlap on the device (the module's
    docstring). `launch_fn(requests, stats=...) -> launched`: the batch's
    host work done and its device work queued; `launched.wait()` returns
    once the device work is done (called on the waiter thread, releasing
    the GIL meanwhile), `launched.complete()` returns `infer_fn`'s replies
    for the batch (on the serve thread). `infer_fn` serves a failed batch
    one request at a time."""

    pipelined = True

    def __init__(self, listener: Listener, infer_fn: Callable, launch_fn: Callable, hello: Dict, **kw):
        super().__init__(listener, infer_fn, hello, **kw)
        self._launch_fn = launch_fn
        self._waits: "queue.Queue[Optional[_InFlight]]" = queue.Queue()
        self._waiter: Optional[threading.Thread] = None

    def start(self) -> None:
        self._waiter = threading.Thread(target=self._wait_device, daemon=True, name="infer-waiter")
        self._waiter.start()
        super().start()

    def _serve(self) -> None:
        inflight: Optional[_InFlight] = None
        try:
            while not self._stop.is_set():
                self._admit()
                done = inflight.done if inflight is not None else None
                batch = self._collect(done)
                if not batch and done is not None and not done.is_set():
                    continue                       # a hello or a release alone; the device is busy
                # N+1 becomes the batch in flight before N is finished: should
                # finishing N raise, the `finally` answers N+1, not N again.
                previous, inflight = inflight, (self._launch(batch) if batch else None)
                if previous is not None:
                    self._finish(previous)
                if self.fatal is not None:
                    self._halt()
                    break
        finally:
            # A batch on the device is answered before the connections close.
            if inflight is not None:
                try:
                    self._finish(inflight)
                except Exception:                  # noqa: BLE001 -- the connections still close
                    log.exception("the batch in flight when serving ended was not answered")
            self._waits.put(None)
            if self._waiter is not None:
                self._waiter.join(5.0)
            self._close_all()

    def _launch(self, batch: List[Request]) -> _InFlight:
        """The batch launched, the waiter thread waiting on it; a launch
        that raises is answered now, as a failed batch."""
        item = _InFlight(batch)
        t0 = time.monotonic()
        try:
            item.launched = self._launch_fn(batch, stats=item.gpu)
        except Exception:                          # noqa: BLE001
            item.replies, item.errors = self._recover(batch, item.gpu, traceback.format_exc())
            item.done.set()
        item.infer_s = time.monotonic() - t0
        if item.launched is not None:
            self._waits.put(item)
        return item

    def _finish(self, item: _InFlight) -> None:
        """The batch's device wait and replies, counted and sent."""
        t0 = time.monotonic()
        if item.launched is not None:
            # The waiter's word that the device is done, rather than the
            # device's own wait inside complete(), which spins on HIP; a dead
            # waiter leaves the wait to complete().
            while not item.done.wait(0.05):
                if self._waiter is None or not self._waiter.is_alive():
                    break
            try:
                replies = list(item.launched.complete())
                if len(replies) != len(item.batch):
                    raise RuntimeError(f"a launched batch completed {len(replies)} replies "
                                       f"for {len(item.batch)} requests")
                item.replies, item.errors = replies, [None] * len(item.batch)
            except Exception:                      # noqa: BLE001
                item.replies, item.errors = self._recover(item.batch, item.gpu, traceback.format_exc())
            item.launched = None
        self._deliver(item.batch, item.replies, item.errors, item.gpu,
                      item.infer_s + time.monotonic() - t0)

    def _wait_device(self) -> None:
        """The waiter thread: each launched batch's device wait, in launch
        order, then the serve thread woken. A wait that raises marks the
        batch done too: its `complete()` raises the fault on the serve
        thread. A batch the serve thread completed first is done at once."""
        while True:
            item = self._waits.get()
            if item is None:
                return
            try:
                item.wait_device()
            except Exception:                      # noqa: BLE001 -- complete() reports it
                pass
            item.done.set()
            self._wake()
