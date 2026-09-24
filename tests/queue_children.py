"""Bodies of the short-lived child processes the pool's lifecycle tests
spawn (tests/test_actor_pool_lifecycle.py). A spawned child imports the
module of its target, so these live apart from the test modules, which
import torch: the child imports nothing but the standard library."""


def ship_results_then_wait_for_stop(ctrl_q, result_q, shipped, nbytes: int) -> None:
    """An actor when an iteration or a stream fails: it has shipped
    `nbytes` of results the manager will not read, then it takes STOP
    and returns. Its exit then waits until the queue's feeder thread has
    pushed those bytes into the pipe."""
    result_q.put(b"x" * nbytes)
    shipped.set()
    ctrl_q.get(timeout=60.0)
