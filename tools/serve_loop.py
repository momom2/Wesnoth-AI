"""The eval inference server's service: one thread reads every
connection, coalesces requests into batches and answers them.

Requests and replies are raw byte messages (`multiprocessing.connection`
send_bytes/recv_bytes): a request is three int64 words (kind, request id,
leaves) and its payload, a reply two (request id, status) and its
payload, the model's replies in wesnoth_ai/byte_wire.py's form, an
error's in UTF-8. The hello is the one pickled message: the client sends
the hello kind and the service answers with its hello dict. A release
(kind 3, no reply) gives up every memory state the server keeps for the
connection (tools/serve_stage.py `ServerMemory`); a connection that
closes gives them up too. Each connection has a number of its own,
never reused (`Request.cid`), which the memory states are kept under.

Why one thread. The service used to run a reader thread per connection
that unpickled each request and queued it for a serve thread. At 120
games in flight the 120 readers and the serve thread share one GIL, and
every torch call of the serve thread releases it during the call, so
each batch's hundreds of launches hand the GIL to readers and wait to
get it back: the box measured 28.5 ms of infer per batch of 41.6 leaves
in a match against 11.1 ms for a batch of 40 timed alone on the same
GPU (training/metrics/generation_speed_r2_20261013/). A CPU simulation
of the service with 60 connections and a fake forward of 300 small torch
calls (2026-10-14, laptop) read the forward at 14.6 ms per batch with
reader threads and pickled payloads, 9.5 with reader threads and raw
bytes, 2.4 with one thread reading every connection (2.7 alone); the
service answered 1,148, 1,542 and 2,433 leaves a second. Here the serve
thread reads the connections itself, between batches; an accept thread
only accepts.

Batching: the serve thread waits for the first ready connection, reads
every ready request, then reads what arrives within `window_s` up to
`max_batch` leaves, and runs one forward. Every worker game has one
decision in flight, so a batch never exceeds the games in flight.
"""
from __future__ import annotations

import logging
import queue
import selectors
import struct
import sys
import threading
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from multiprocessing.connection import Listener, Pipe, wait
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger("serve_loop")

KIND_HELLO, KIND_INFER, KIND_RELEASE = 1, 2, 3
STATUS_OK, STATUS_ERROR = 0, 1
_REQUEST_HEAD = struct.Struct("<qqq")        # kind, request id, leaves
_REPLY_HEAD = struct.Struct("<qq")           # request id, status
REQUEST_HEAD_BYTES = _REQUEST_HEAD.size
REPLY_HEAD_BYTES = _REPLY_HEAD.size
HELLO_MESSAGE = _REQUEST_HEAD.pack(KIND_HELLO, 0, 0)
RELEASE_MESSAGE = _REQUEST_HEAD.pack(KIND_RELEASE, 0, 0)


def reply_head(message) -> Tuple[int, int]:
    """(request id, status) of a reply message."""
    return _REPLY_HEAD.unpack_from(message, 0)


def ok_message(rid: int, payload: bytes = b"") -> bytes:
    """A reply carrying `payload`."""
    return _REPLY_HEAD.pack(rid, STATUS_OK) + payload


def error_message(rid: int, error: str) -> bytes:
    """A reply carrying an error's text."""
    return _REPLY_HEAD.pack(rid, STATUS_ERROR) + error.encode("utf-8", "replace")


def error_text(message) -> str:
    return bytes(message[REPLY_HEAD_BYTES:]).decode("utf-8", "replace")


@dataclass
class Request:
    conn: Any
    rid: int
    leaves: int
    message: bytes          # the whole request; its payload from REQUEST_HEAD_BYTES
    cid: int = 0            # the connection's number


@dataclass
class ServeStats:
    requests: int = 0
    batches: int = 0
    leaves: int = 0
    connections: int = 0
    releases: int = 0            # connections' memory states given up (a release or a close)
    failed_batches: int = 0      # batches whose forward raised
    failed_requests: int = 0     # requests that still failed alone
    batch_hist: Counter = field(default_factory=Counter)
    idle_s: float = 0.0      # waiting for a first request
    window_s: float = 0.0    # collecting the rest of a batch (its reads included)
    recv_s: float = 0.0      # reading requests off the connections
    infer_s: float = 0.0
    reply_s: float = 0.0
    gpu_ms: float = 0.0
    # The infer function's host seconds by stage (decode, encode, forward,
    # priors, finish, reply, wire), summed over batches.
    stage_s: Dict[str, float] = field(default_factory=dict)
    started: float = field(default_factory=time.monotonic)

    def add_batch(self, gpu: Dict[str, float]) -> None:
        for k, v in gpu.items():
            if k == "gpu_ms":
                self.gpu_ms += float(v)
            else:
                self.stage_s[k] = self.stage_s.get(k, 0.0) + float(v)

    def as_dict(self) -> Dict:
        return {
            "requests": self.requests, "batches": self.batches,
            "leaves": self.leaves, "connections": self.connections, "releases": self.releases,
            "failed_batches": self.failed_batches,
            "failed_requests": self.failed_requests,
            "mean_batch": (self.leaves / self.batches if self.batches else 0.0),
            "batch_hist": {str(k): v for k, v in sorted(self.batch_hist.items())},
            "idle_s": round(self.idle_s, 3), "window_s": round(self.window_s, 3),
            "recv_s": round(self.recv_s, 3),
            "infer_s": round(self.infer_s, 3), "reply_s": round(self.reply_s, 3),
            "gpu_ms": round(self.gpu_ms, 1),
            "stage_s": {k: round(v, 3) for k, v in sorted(self.stage_s.items())},
            "wall_s": round(time.monotonic() - self.started, 3),
        }

    def counters(self) -> Dict:
        """The running totals alone, read from another thread while the
        serve thread adds to them (no container is iterated)."""
        return {"wall_s": round(time.monotonic() - self.started, 3), "requests": self.requests,
                "batches": self.batches, "leaves": self.leaves, "gpu_ms": round(self.gpu_ms, 1),
                "idle_s": round(self.idle_s, 3), "infer_s": round(self.infer_s, 3)}


class Timeline:
    """A thread appending `stats.counters()` to `path` every `every_s`
    seconds, one JSON line each, flushed, and a last line at `stop`: the
    server's rate over any part of its run, read on the run
    (tools/box_probe.py and scripts/fp8_inference_box.sh read a steady
    window). A write that fails ends the timeline, never the serving or
    the server's exit, whose stats are the run's primary record."""

    def __init__(self, stats: ServeStats, path, every_s: float):
        self._stats, self._path, self._every = stats, path, float(every_s)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="serve-timeline", daemon=True)
        self.failed: Optional[str] = None

    def start(self) -> "Timeline":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)
        if self.failed is None:
            self._write()

    def _write(self) -> bool:
        import json
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(self._stats.counters()) + "\n")
                f.flush()
        except OSError as e:
            self.failed = f"{type(e).__name__}: {e}"
            log.warning("server timeline stopped: %s", self.failed)
            return False
        return True

    def _run(self) -> None:
        while not self._stop.wait(self._every):
            if not self._write():
                return


class _Ready:
    """The connections the serve thread reads, and a wait for the ones
    with a message: an epoll (or other) selector on POSIX, which costs
    nothing per connection per wait; `multiprocessing.connection.wait`
    on Windows, whose pipe handles no selector takes."""

    def __init__(self, on_close: Optional[Callable[[Any], None]] = None):
        self.conns: List[Any] = []
        self._sel = None if sys.platform == "win32" else selectors.DefaultSelector()
        self._on_close = on_close

    def add(self, conn) -> None:
        self.conns.append(conn)
        if self._sel is not None:
            self._sel.register(conn, selectors.EVENT_READ)

    def remove(self, conn) -> None:
        if conn in self.conns:
            self.conns.remove(conn)
            if self._on_close is not None:
                self._on_close(conn)
            if self._sel is not None:
                try:
                    self._sel.unregister(conn)
                except (KeyError, ValueError, OSError):
                    pass
        try:
            conn.close()
        except OSError:
            pass

    def wait(self, timeout: float) -> List[Any]:
        if self._sel is None:
            return wait(self.conns, timeout)
        return [key.fileobj for key, _ in self._sel.select(timeout)]


class InferenceService:
    """Accepts connections on `listener`, answers a hello with `hello`,
    coalesces infer requests and replies through `infer_fn`.

    `infer_fn(requests, stats=...) -> replies`: one whole reply message
    per `Request`, in order (`ok_message`, or a payload encoder writing
    `(rid, STATUS_OK)` as its prefix); `stats` is a dict the function
    adds device milliseconds ("gpu_ms") and host seconds by stage to.
    Generic over the payload so the batching is testable without torch.

    `release_fn(cid)` gives up the memory states kept for connection
    `cid`: on its release message, and when it closes. Called on the serve
    thread, between batches.

    `health_fn()` raises when the device can no longer run work; it is
    called after a batch fails. A sticky device fault fails every later
    call, so a service that kept answering would fail every game that
    reaches it while its process looked alive (2026-09-14: 40 of 40
    games). When the probe fails, the service answers the batch with the
    fault, stops, and keeps the fault in `fatal`."""

    def __init__(self, listener: Listener, infer_fn: Callable, hello: Dict, *,
                 window_s: float, max_batch: int,
                 health_fn: Optional[Callable[[], None]] = None,
                 release_fn: Optional[Callable[[int], None]] = None):
        self._listener = listener
        self._infer_fn = infer_fn
        self._hello = hello
        self._window = max(0.0, float(window_s))
        self._max_batch = max(1, int(max_batch))
        self._health_fn = health_fn
        self._release_fn = release_fn
        self._incoming: "queue.Queue[Any]" = queue.Queue()
        self._cids: Dict[Any, int] = {}
        self._next_cid = 1
        self._ready = _Ready(on_close=self._closed)
        # The accept thread wakes the serve thread's wait through this pipe
        # when it hands over a connection.
        self._wake_r, self._wake_w = Pipe(duplex=False)
        self._ready.add(self._wake_r)
        self._stop = threading.Event()
        self._serve_thread: Optional[threading.Thread] = None
        self.stats = ServeStats()
        self.fatal: Optional[str] = None

    def start(self) -> None:
        self._serve_thread = threading.Thread(target=self._serve, daemon=True,
                                              name="infer-serve")
        self._serve_thread.start()
        threading.Thread(target=self._accept, daemon=True, name="infer-accept").start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        try:
            self._listener.close()
        except OSError:
            pass
        if self._serve_thread is not None:
            self._serve_thread.join(timeout)

    def serve_until_closed(self, wait_for_eof: Callable[[], object],
                           poll_s: float = 0.5) -> Optional[str]:
        """Serve until `wait_for_eof()` returns (the driver closed the
        server's stdin) or a device fault stops the service; the fault,
        or None."""
        closed = threading.Event()

        def _wait() -> None:
            wait_for_eof()
            closed.set()

        threading.Thread(target=_wait, daemon=True, name="infer-stdin").start()
        while not closed.wait(poll_s):
            if self.fatal is not None:
                break
        return self.fatal

    # -- connections ---------------------------------------------------

    def _accept(self) -> None:
        while not self._stop.is_set():
            try:
                conn = self._listener.accept()
            except (OSError, EOFError):
                if self._stop.is_set():
                    return
                log.exception("accept failed")
                return
            self._incoming.put(conn)
            try:
                self._wake_w.send_bytes(b"")
            except OSError:
                return

    def _wait(self, timeout: float) -> List[Any]:
        """The connections with a message within `timeout`. A connection
        the accept thread hands over meanwhile joins the set and the wait
        goes on for what is left of it."""
        deadline = time.monotonic() + timeout
        while True:
            ready = self._ready.wait(max(0.0, timeout))
            if self._wake_r not in ready:
                return ready
            ready = [c for c in ready if c is not self._wake_r]
            while self._wake_r.poll():
                self._wake_r.recv_bytes()
            self._admit()
            timeout = deadline - time.monotonic()
            if ready or timeout <= 0:
                return ready

    def _admit(self) -> None:
        """The connections the accept thread took since the last batch."""
        while True:
            try:
                conn = self._incoming.get_nowait()
            except queue.Empty:
                return
            self.stats.connections += 1
            self._cids[conn] = self._next_cid
            self._next_cid += 1
            self._ready.add(conn)

    def _release(self, cid: int) -> None:
        self.stats.releases += 1
        if self._release_fn is not None:
            self._release_fn(cid)

    def _closed(self, conn) -> None:
        cid = self._cids.pop(conn, None)
        if cid is not None:
            self._release(cid)

    def _read(self, conns, batch: List[Request], leaves: int) -> int:
        """Read one message from each of `conns` (ready ones) until the
        batch holds `max_batch` leaves: an infer request joins the batch,
        a hello is answered, a closed connection is dropped. Returns the
        batch's leaves."""
        t0 = time.monotonic()
        for conn in conns:
            if leaves >= self._max_batch:
                break
            try:
                message = conn.recv_bytes()
            except (EOFError, OSError):
                self._ready.remove(conn)
                continue
            if len(message) < REQUEST_HEAD_BYTES:
                self._send(conn, error_message(-1, f"a request of {len(message)} bytes"))
                continue
            kind, rid, n = _REQUEST_HEAD.unpack_from(message, 0)
            if kind == KIND_INFER:
                batch.append(Request(conn, rid, n, message, self._cids.get(conn, 0)))
                leaves += n
            elif kind == KIND_RELEASE:
                self._release(self._cids.get(conn, 0))
            elif kind == KIND_HELLO:
                try:
                    conn.send(self._hello)
                except (OSError, EOFError, ValueError):
                    self._ready.remove(conn)
            else:
                self._send(conn, error_message(rid, f"unknown message kind {kind}"))
        self.stats.recv_s += time.monotonic() - t0
        return leaves

    def _send(self, conn, message) -> None:
        try:
            conn.send_bytes(message)
        except (OSError, EOFError, ValueError):
            self._ready.remove(conn)              # the client went away

    # -- batches -------------------------------------------------------

    def _collect(self) -> List[Request]:
        """The next batch: every request ready now (waiting up to 50 ms
        for a first one, so stop is noticed), then what arrives within
        the window, up to max_batch leaves."""
        st = self.stats
        t0 = time.monotonic()
        ready = self._wait(0.05)
        t1 = time.monotonic()
        st.idle_s += t1 - t0
        if not ready:
            return []
        batch: List[Request] = []
        leaves = self._read(ready, batch, 0)
        deadline = t1 + self._window
        while leaves < self._max_batch:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            ready = self._wait(remaining)
            if not ready:
                break
            leaves = self._read(ready, batch, leaves)
        st.window_s += time.monotonic() - t1
        return batch

    def _infer(self, batch: List[Request], gpu: Dict[str, float]) -> list:
        replies = self._infer_fn(batch, stats=gpu)
        if len(replies) != len(batch):
            raise RuntimeError(f"infer_fn returned {len(replies)} replies "
                               f"for {len(batch)} requests")
        return list(replies)

    def _answer(self, batch: List[Request], gpu: Dict[str, float]) -> Tuple[list, list]:
        """(replies, errors) for the batch, one of each per request.
        One forward for the whole batch; when it raises, each request
        alone, so only the offenders fail: a batch holds one decision
        from each of up to `jobs` games, and a fault of one position
        must not cost the other games their play."""
        try:
            return self._infer(batch, gpu), [None] * len(batch)
        except Exception:                          # noqa: BLE001
            error = traceback.format_exc()
        self.stats.failed_batches += 1
        probe = self._probe_failure()
        if probe is not None:
            self.fatal = f"{error}\nthen the device failed its health probe:\n{probe}"
            self.stats.failed_requests += len(batch)
            log.error("a failed batch left the device unable to serve; stopping:\n%s",
                      self.fatal)
            return [None] * len(batch), [self.fatal] * len(batch)
        if len(batch) == 1:
            replies, errors = [None], [error]
        else:
            log.error("batch of %d requests failed; retrying them one at a time:\n%s",
                      len(batch), error)
            replies, errors = [], []
            for p in batch:
                try:
                    replies.extend(self._infer([p], gpu))
                    errors.append(None)
                except Exception:                  # noqa: BLE001
                    replies.append(None)
                    errors.append(traceback.format_exc())
        failed = [p.rid for p, e in zip(batch, errors) if e is not None]
        self.stats.failed_requests += len(failed)
        if failed:
            log.error("%d of %d request(s) failed (rids %s):\n%s", len(failed), len(batch),
                      failed, next(e for e in errors if e is not None))
        else:
            log.warning("every request of the failed batch passed alone (a batch-level "
                        "fault, not a position's)")
        return replies, errors

    def _probe_failure(self) -> Optional[str]:
        """The health probe's traceback when the device can no longer
        serve; None when it can, or when there is no probe."""
        if self._health_fn is None:
            return None
        try:
            self._health_fn()
        except Exception:                          # noqa: BLE001
            return traceback.format_exc()
        return None

    def _serve(self) -> None:
        st = self.stats
        try:
            while not self._stop.is_set():
                self._admit()
                batch = self._collect()
                if not batch:
                    continue
                t0 = time.monotonic()
                gpu: Dict[str, float] = {}
                replies, errors = self._answer(batch, gpu)
                t1 = time.monotonic()
                # Counted before answering, so a client holding its reply
                # reads counters that include it.
                leaves = sum(p.leaves for p in batch)
                st.requests += len(batch)
                st.batches += 1
                st.leaves += leaves
                st.batch_hist[leaves] += 1
                st.infer_s += t1 - t0
                st.add_batch(gpu)
                for p, reply, error in zip(batch, replies, errors):
                    self._send(p.conn, reply if error is None else error_message(p.rid, error))
                st.reply_s += time.monotonic() - t1
                if self.fatal is not None:
                    # Accept nothing more; the process exits once the main
                    # thread sees the fault.
                    self._stop.set()
                    try:
                        self._listener.close()
                    except OSError:
                        pass
                    return
        finally:
            for conn in list(self._ready.conns):
                self._ready.remove(conn)
            try:
                self._wake_w.close()
            except OSError:
                pass
