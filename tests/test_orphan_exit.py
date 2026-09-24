"""A pool child whose learner was killed exits, whatever it shipped
(tools/mp_teardown.run_child). The children are real processes that
import no torch (tests/queue_children.py): a process's exit waiting on
its queue's feeder happens only in a real process."""
from __future__ import annotations

import contextlib
import multiprocessing as mp
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).parent))

import queue_children  # noqa: E402


def _exited(proc: psutil.Process) -> bool:
    """It is not our child, so a zombie counts as exited: reaping it is
    its new parent's business."""
    try:
        return proc.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def _wait_for_exit(proc: psutil.Process, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while not _exited(proc):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def test_an_actor_exits_with_results_its_killed_learner_never_read():
    """A learner killed while its actors ship results (an OOM kill
    between two collect() calls of a stream) leaves each actor with a
    pipe full of them: one experience carries a whole game state, far
    more than a pipe holds (64 KiB on Linux, 8 KiB on Windows). The
    actor notices the kill and returns, but a process that has put on an
    mp.Queue waits at exit until everything it put is in the pipe, and
    nobody will read that pipe again. It never breaks either: the actor
    holds its read end itself. An actor that never exits holds the
    container's PID budget for the life of the box."""
    ctx = mp.get_context("spawn")
    shipped = ctx.Event()
    actor_pid = ctx.Value("i", 0)
    learner = ctx.Process(target=queue_children.learner_killed_while_its_actor_ships,
                          args=(1 << 20, actor_pid, shipped), name="learner")
    learner.start()
    actor = None
    try:
        assert shipped.wait(60.0), "the actor never shipped"
        actor = psutil.Process(actor_pid.value)
        learner.kill()
        learner.join(10.0)
        assert _wait_for_exit(actor, 10.0), (
            "the actor was still running 10 s after its learner was killed")
    finally:
        if learner.is_alive():
            learner.kill()
            learner.join(10.0)
        if actor is not None and not _exited(actor):
            with contextlib.suppress(psutil.NoSuchProcess):
                actor.kill()
