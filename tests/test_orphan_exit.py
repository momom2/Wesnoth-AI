"""A child whose parent was killed exits: an actor whatever it shipped
(tools/mp_teardown.run_child), an encode worker whatever it was waiting
on (tools/encode_worker.serve_files). The children are real processes
that import no torch (tests/queue_children.py): a process's exit waiting
on its queue's feeder happens only in a real process."""
from __future__ import annotations

import contextlib
import multiprocessing as mp
import sys
import time
from pathlib import Path

import psutil
import pytest

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


def _assert_child_exits_once_parent_is_killed(parent: mp.Process, child_pid, ready,
                                             child: str) -> None:
    """Start `parent`, wait until `ready`, kill it, and require the child
    whose pid it reports to exit within 10 s. Both are killed on the way
    out, so a failure leaves nothing running."""
    parent.start()
    proc = None
    try:
        assert ready.wait(60.0), f"the {child} never got ready"
        proc = psutil.Process(child_pid.value)
        parent.kill()
        parent.join(10.0)
        assert _wait_for_exit(proc, 10.0), (
            f"the {child} was still running 10 s after its parent was killed")
    finally:
        if parent.is_alive():
            parent.kill()
            parent.join(10.0)
        if proc is not None and not _exited(proc):
            with contextlib.suppress(psutil.NoSuchProcess):
                proc.kill()


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
    _assert_child_exits_once_parent_is_killed(learner, actor_pid, shipped, "actor")


@pytest.mark.parametrize("n_replays", [0, 3], ids=["waiting-for-a-replay", "output-queue-full"])
def test_an_encode_worker_exits_once_its_trainer_is_killed(n_replays):
    """`supervised_train --workers N` on a box, and the trainer is killed:
    an OOM kill, or the SL relaunch in scripts/vast_onstart.sh, whose
    pkill matches the trainer's command line and not its spawned
    workers'. A worker holds both ends of both its queues, so the pipes
    never break, and it waited forever for a replay or, its output queue
    full, for room; each relaunch left the previous run's workers
    running."""
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    worker_pid = ctx.Value("i", 0)
    trainer = ctx.Process(target=queue_children.trainer_killed_with_its_encode_worker,
                          args=(n_replays, worker_pid, ready), name="trainer")
    _assert_child_exits_once_parent_is_killed(trainer, worker_pid, ready, "encode worker")
