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


def learner_killed_while_its_stopped_actor_flushes(actor_pid, stopped) -> None:
    """A learner killed during its own shutdown: it has sent STOP to an
    actor holding 1 MiB of results and reads none of them; it waits to
    be killed."""
    import multiprocessing as mp
    import time
    from tools.mp_teardown import start_child
    ctx = mp.get_context("spawn")
    ctrl_q, result_q = ctx.Queue(), ctx.Queue()
    start_child(ctx, ship_results_then_take_stop, (ctrl_q, result_q, actor_pid, stopped),
                name="actor-0")
    ctrl_q.put(("stop",))
    time.sleep(600.0)


def ship_results_then_take_stop(ctrl_q, result_q, actor_pid, stopped) -> None:
    """An actor that ships 1 MiB of results, far more than a pipe holds,
    then takes STOP and returns while its learner is alive."""
    import os
    result_q.put(b"x" * (1 << 20))
    ctrl_q.get(timeout=60.0)
    actor_pid.value = os.getpid()
    stopped.set()


def trainer_killed_with_its_encode_worker(n_replays: int, worker_pid, ready) -> None:
    """A trainer that has started one encode worker and handed it
    `n_replays` replays, reading none of its output; it waits to be
    killed. `ready` is set once the worker waits: for a replay when
    `n_replays` is 0, else for room on its full output queue."""
    import multiprocessing as mp
    import time
    from tools.mp_teardown import start_child
    ctx = mp.get_context("spawn")
    in_q = ctx.Queue(maxsize=n_replays + 1)
    out_q = ctx.Queue(maxsize=max(n_replays - 1, 1))
    started = ctx.Event()
    worker = start_child(ctx, encode_worker_noticing_sooner, (in_q, out_q, started),
                         name="encode-0")
    worker_pid.value = worker.pid
    for seq in range(n_replays):
        in_q.put((seq, f"replay_{seq}.json.gz"))
    started.wait(60.0)
    while n_replays and not out_q.full():
        time.sleep(0.01)
    ready.set()
    time.sleep(600.0)


def encode_worker_noticing_sooner(in_q, out_q, started) -> None:
    """tools.encode_worker's loop over replays that each encode to 1 MiB,
    far more than a pipe holds; it checks on its trainer every 0.2 s
    instead of every 2 s."""
    from tools import encode_worker
    encode_worker._PARENT_POLL = 0.2
    started.set()
    encode_worker.serve_files(in_q, out_q, _one_mebibyte_of_pairs)


def _one_mebibyte_of_pairs(gz_path: str) -> list:
    return [(b"x" * (1 << 20), None)]
