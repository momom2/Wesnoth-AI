"""Teardown of spawned children and of the multiprocessing queues they
share with their manager (tools/actor_pool.ActorPool.shutdown).

Two properties of CPython's multiprocessing.Queue shape everything here
(Lib/multiprocessing/queues.py, 3.13; Python docs, multiprocessing,
"Joining processes that use queues"):

- `put()` only hands the object to the queue's feeder thread, which
  writes it into the pipe later. A process that has put anything waits
  at exit, in an untimed finalizer, until its feeder has written all of
  it. A pipe holds 64 KiB on Linux and 8 KiB on Windows, so a child that
  shipped more than that to a reader who stopped reading cannot exit
  until someone reads.
- A reader cannot bound the read of one message: `get(timeout)` waits
  for the first byte only, and the remainder of a message a killed
  writer left half written never comes.
"""

from __future__ import annotations

import contextlib
import logging
import queue as _queue
import threading
import time
from typing import Callable, Iterable, Optional, Sequence

log = logging.getLogger("actor_pool")

# Each read waits this long for a message before moving to the next
# queue; it also bounds how long the reader takes to notice its stop.
_READ_POLL = 0.05
# How long the caller waits for the reader to stop once the body is
# done. A reader past it is blocked on a half-written message.
_READER_STOP = 1.0


def close_queue(q, drain: bool) -> None:
    """Release one mp.Queue without ever waiting on its feeder thread.

    A feeder blocked in `send_bytes` on a pipe full of messages nobody
    will read (a wedged actor's unread PLAYs; measured: three 6.6 KB
    messages against Windows' 8 KiB pipe hang it about half the time,
    twenty reliably, Linux's 64 KiB pipe at eight to ten) never returns,
    and `join_thread()` is untimed. So `cancel_join_thread()` is what
    keeps this from hanging: after it the feeder is abandoned as a
    daemon and `join_thread()` is a no-op (CPython's Finalize.cancel
    clears its key). `drain` first reads out, with a bounded 50 ms wait
    per message, whatever is still buffered, so the pipe's bytes are
    released rather than left to the OS; pass it ONLY for a queue this
    process alone writes to, since a half-written message from a killed
    child would block a read for a remainder that never comes. Every
    child is joined before this runs, so nothing refills a queue behind
    us."""
    if q is None:
        return
    if drain:
        while True:
            try:
                q.get(timeout=0.05)
            except Exception:        # empty, or a queue already broken
                break
    try:
        q.cancel_join_thread()       # a plain queue.Queue has no feeder
    except Exception:
        pass
    try:
        q.close()
    except Exception:
        pass


@contextlib.contextmanager
def discarding(queues: Sequence, on_message: Optional[Callable] = None):
    """Read and discard `queues` on a daemon thread while the body runs,
    so that children whose output nobody else will read can exit.
    `on_message` sees every message read.

    The caller never waits on a read: the body runs on its own thread,
    and the reader gets `_READER_STOP` seconds to stop after it. A
    message a killed child left half written therefore costs a reader
    thread left blocked, never the caller."""
    stop = threading.Event()
    reader = threading.Thread(target=_discard_until, args=(list(queues), stop, on_message),
                              daemon=True, name="teardown-reader")
    reader.start()
    try:
        yield
    finally:
        stop.set()
        reader.join(timeout=_READER_STOP)
        if reader.is_alive():
            log.warning("teardown: a read of a child's queue has not returned in "
                        f"{_READER_STOP:.0f}s (a message a killed child left half "
                        "written?); the reader is left behind")


def _discard_until(queues: list, stop: threading.Event,
                   on_message: Optional[Callable]) -> None:
    n = 0
    while not stop.is_set():
        for q in queues:
            try:
                msg = q.get(timeout=_READ_POLL)
            except _queue.Empty:
                continue
            except Exception:        # noqa: BLE001 -- a message that would not unpickle
                stop.wait(_READ_POLL)
                continue
            n += 1
            if on_message is not None:
                on_message(msg)
    if n:
        log.info(f"teardown: discarded {n} message(s) the children sent while exiting")


def join_all(procs: Iterable, seconds: float) -> None:
    """Join `procs` against one deadline `seconds` from now, so the wait
    is `seconds` in all, not `seconds` per process."""
    deadline = time.monotonic() + seconds
    for p in procs:
        p.join(max(0.0, deadline - time.monotonic()))


def end_stragglers(procs: Iterable, grace: float = 5.0) -> None:
    """Terminate every process of `procs` still alive and join them
    against one deadline; then kill and join whatever survived."""
    alive = [p for p in procs if p.is_alive()]
    for p in alive:
        log.warning(f"terminating unresponsive process {p.name}")
        p.terminate()
    # terminate() only SIGNALS: without this join the process stays a
    # live child (and, on the box, a live CUDA context) while the
    # parent walks on.
    join_all(alive, grace)
    survivors = [p for p in alive if p.is_alive()]
    for p in survivors:
        log.error(f"process {p.name} survived terminate(); killing it")
        p.kill()
    join_all(survivors, grace)
