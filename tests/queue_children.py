"""Bodies of the short-lived child processes the pool's lifecycle tests
spawn (tests/test_actor_pool_lifecycle.py, tests/test_orphan_exit.py).
A spawned child imports the module of its target, so these live apart
from the test modules, which import torch: the child imports nothing
that imports torch."""


def ship_results_then_wait_for_stop(ctrl_q, result_q, shipped, nbytes: int) -> None:
    """An actor when an iteration or a stream fails: it has shipped
    `nbytes` of results the manager will not read, then it takes STOP
    and returns. Its exit then waits until the queue's feeder thread has
    pushed those bytes into the pipe."""
    result_q.put(b"x" * nbytes)
    shipped.set()
    ctrl_q.get(timeout=60.0)


def learner_killed_while_its_actor_ships(nbytes: int, actor_pid, shipped) -> None:
    """A learner that has started one actor the way the pool does and
    reads none of its results; it waits to be killed."""
    import multiprocessing as mp
    import time
    from tools.mp_teardown import start_child
    ctx = mp.get_context("spawn")
    ctrl_q, result_q = ctx.Queue(), ctx.Queue()
    start_child(ctx, ship_results_until_orphaned,
                (ctrl_q, result_q, actor_pid, shipped, nbytes), name="actor-0")
    time.sleep(600.0)


def ship_results_until_orphaned(ctrl_q, result_q, actor_pid, shipped, nbytes: int) -> None:
    """An actor with `nbytes` of results unread when its learner dies. It
    leaves the way the actor does once the learner is gone:
    tools.actor_worker._wait_for_command returns None."""
    import os
    from tools import actor_worker
    actor_worker._PARENT_POLL = 0.2         # notice the kill sooner than 2 s
    result_q.put(b"x" * nbytes)
    actor_pid.value = os.getpid()
    shipped.set()
    while actor_worker._wait_for_command(ctrl_q) is not None:
        pass
