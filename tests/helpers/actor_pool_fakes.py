"""An ActorPool whose queues and processes are fakes: the real
run_iteration serve loop with no multiprocessing."""
import queue as _queue
from types import SimpleNamespace

from tools.actor_pool import ActorPool


class _FakeQ:
    def __init__(self, items=None):
        self._items = list(items or [])

    def put(self, x):
        self._items.append(x)

    def get_nowait(self):
        if not self._items:
            raise _queue.Empty
        return self._items.pop(0)

    def get(self, timeout=None):   # noqa: ARG002 - fake never blocks
        if not self._items:
            raise _queue.Empty
        return self._items.pop(0)


class _FakeProc:
    def __init__(self, alive, exitcode=None, name="actor"):
        self._alive = alive
        self.exitcode = exitcode
        self.name = name

    def is_alive(self):
        return self._alive


def _pool(procs, results, *, iteration_timeout=1800.0,
          drain_grace=1800.0, liveness_interval=0.0):
    pool = ActorPool.__new__(ActorPool)
    pool._n = len(procs)
    pool._started = True
    # A bare inference model: no weights version, no compile stats
    # (the pool reads both through getattr defaults).
    pool._policy = SimpleNamespace(
        _inference_model=SimpleNamespace(),
        _inference_encoder=SimpleNamespace(unit_type_to_id={}, faction_to_id={}))
    pool._iteration_timeout = iteration_timeout
    pool._drain_grace = drain_grace
    pool._liveness_interval = liveness_interval
    pool._max_batch = 8
    pool._serve_timeout = 0.0
    pool._serve_threads = 1
    pool._coalesce = "fifo"
    pool._coalesce_gap = 0
    pool.value_center = 0.0     # MCTSConfig.value_center broadcast (az4)
    pool.server_priors = False  # server-side priors flag (plan 1.3)
    pool._ctrl_qs = [_FakeQ() for _ in procs]
    pool._game_q = _FakeQ()
    pool._resp_qs = [_FakeQ() for _ in procs]
    pool._req_qs = [_FakeQ()]           # always empty -> idle path
    pool._server = None                 # never asked (no requests)
    pool._result_q = _FakeQ(results)
    pool._procs = list(procs)
    # No serve processes: the learner process is the only server.
    pool._serve_processes = 1
    pool._server_procs = []
    pool._server_ctrl_qs = []
    pool._server_versions = []
    pool._server_q = _FakeQ()
    pool._serving = False
    pool.last_serve_thread_errors = []
    pool.last_stuck_serve_threads = []
    pool._stuck_serve_threads = []
    pool._open_serving = None
    return pool
