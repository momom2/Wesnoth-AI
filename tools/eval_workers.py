"""Persistent evaluation workers for tools/run_elo_batch.py (plan
step 1.5).

One `elo_eval_game.py` process per game paid the checkpoint load,
CUDA init and torch.compile every time: of a 35 s raw game in the
2026-09-04 baseline, roughly 24 s was that. A worker is one
`elo_eval_game.py --worker` process that plays many games and keeps
the loaded policies across them, up to `slots` of them at once (each
in its own thread, `elo_eval_game.worker_loop`): with the shared
inference server a game spends most of its decision waiting for a
reply, and a second game in the same process works meanwhile.
Protocol: the driver writes one JSON object per line on the worker's
stdin, `{"job": N, "argv": [...]}`; the worker answers
`__DONE__ <N> <rc>` on stdout once that game's result file is on disk.
`JobHandle` mimics the subprocess.Popen surface the batch loop drives
(poll / wait / kill / returncode / pid), so the loop is unchanged; a
killed worker (per-game timeout) is replaced on the next submit, and
the other games it was playing are reported as `collateral`, for the
loop to play again.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

DONE_PREFIX = "__DONE__ "
# The return code of a game whose worker was killed for another game.
COLLATERAL_RC = -15


class _ErrShim:
    """Per-game view of a worker's stderr file: the batch loop tails it
    on failure and 'closes' it on success; neither may touch the
    worker's real file, which outlives the game."""

    def __init__(self, real, name: str):
        self._real = real
        self.name = name

    def fileno(self):
        return self._real.fileno()

    def seek(self, *a):
        return self._real.seek(*a)

    def read(self, *a):
        return self._real.read(*a)

    def close(self):
        pass


class Worker:
    def __init__(self, cmd: List[str], errf_path: Path, slots: int = 1):
        self.errf = open(errf_path, "w+b")
        self.slots = max(1, int(slots))
        self.proc = subprocess.Popen(cmd + ["--worker-games", str(self.slots)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.errf, text=True, bufsize=1)
        self._lock = threading.Lock()
        self._done: Dict[int, int] = {}
        self._next_job = 0
        self.in_flight = 0
        self.games = 0
        # The job a kill was for: the worker's other games are collateral.
        self.killed_for: Optional[int] = None
        self.pumped = threading.Event()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        try:
            for line in self.proc.stdout:
                if not line.startswith(DONE_PREFIX):
                    continue
                parts = line[len(DONE_PREFIX):].split()
                try:
                    job, rc = int(parts[0]), int(parts[1])
                except (IndexError, ValueError):
                    continue
                with self._lock:
                    self._done[job] = rc
        finally:
            self.pumped.set()

    @property
    def busy(self) -> bool:
        """Every slot holds a game."""
        return self.in_flight >= self.slots

    def alive(self) -> bool:
        return self.proc.poll() is None

    def submit(self, argv: List[str]) -> "JobHandle":
        with self._lock:
            job = self._next_job
            self._next_job += 1
            self.in_flight += 1
        self.proc.stdin.write(json.dumps({"job": job, "argv": argv}) + "\n")
        self.proc.stdin.flush()
        return JobHandle(self, job)

    def take_done(self, job: int) -> Optional[int]:
        with self._lock:
            return self._done.pop(job, None)

    def finished(self) -> None:
        with self._lock:
            self.in_flight = max(0, self.in_flight - 1)
            self.games += 1

    def kill(self, job: Optional[int] = None):
        if self.killed_for is None:
            self.killed_for = job
        if self.alive():
            self.proc.kill()
        self.proc.wait()

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:            # noqa: BLE001
            self.kill()
        try:
            self.errf.close()
        except OSError:
            pass


class JobHandle:
    def __init__(self, worker: Worker, job: int):
        self._w = worker
        self.job = job
        self.returncode: Optional[int] = None
        self.pid = worker.proc.pid
        # True when the worker died for ANOTHER game's kill: this game
        # did not fail, it was stopped, and the loop plays it again.
        self.collateral = False

    def poll(self) -> Optional[int]:
        if self.returncode is not None:
            return self.returncode
        rc = self._w.take_done(self.job)
        if rc is None:
            if self._w.alive():
                return None
            # The worker is gone: its last lines may still be in the pump.
            self._w.pumped.wait(timeout=5.0)
            rc = self._w.take_done(self.job)
            if rc is None:
                killed_for = self._w.killed_for
                if killed_for is not None and killed_for != self.job:
                    self.collateral = True
                    rc = COLLATERAL_RC
                else:
                    rc = self._w.proc.returncode or 1
        self.returncode = rc
        self._w.finished()
        return self.returncode

    def wait(self, timeout: Optional[float] = None) -> int:
        t0 = time.monotonic()
        while self.poll() is None:
            if timeout is not None and time.monotonic() - t0 > timeout:
                raise TimeoutError("worker job did not finish")
            time.sleep(0.05)
        return self.returncode

    def kill(self):
        self._w.kill(self.job)
        if self.returncode is None:
            self.returncode = -9
            self._w.finished()


class WorkerPool:
    """Up to `n` workers spawned lazily, each playing up to `slots`
    games at once; dead ones are replaced."""

    def __init__(self, cmd: List[str], n: int, errdir: Path, slots: int = 1):
        self._cmd = list(cmd)
        self._n = max(1, int(n))
        self._slots = max(1, int(slots))
        self._errdir = Path(errdir)
        self._workers: List[Worker] = []
        self._spawned = 0
        # Logs of workers that died; kept so a caller can read why, and
        # so `shutdown()` can say how many there were.
        self._dead_logs: List[Path] = []

    def _spawn(self) -> Worker:
        self._spawned += 1
        w = Worker(self._cmd, self._errdir / f".worker_{self._spawned}.log", self._slots)
        self._workers.append(w)
        return w

    def _evict_dead(self) -> None:
        """Drop workers that are gone, CLOSING them on the way out.

        The filter used to just rebuild the list, which dropped the
        only reference to a killed worker without calling `close()`:
        its stderr file object stayed open and its `.worker_N.log`
        stayed in the outdir until CPython happened to reclaim it, and
        `shutdown()` only closes what is still in the list. That is one
        leaked fd and one stale log per KILLED worker -- i.e. per
        timed-out game, and the leg-5 verdict saw 27 of 40 games time
        out. The discarded log is also the only record of why that
        worker died, so it is worth keeping rather than orphaning.
        A dead worker stays listed while one of its games is still to be
        polled.
        """
        keep, drop = [], []
        for w in self._workers:
            (keep if (w.alive() or w.in_flight) else drop).append(w)
        self._workers = keep
        for w in drop:
            try:
                w.close()
            except Exception:        # noqa: BLE001 -- a dead worker must not
                pass                 # block admitting the next game
            self._dead_logs.append(Path(w.errf.name))

    def submit(self, argv: List[str], game_tag: str = ""):
        """A (handle, errf) pair for the batch loop; the errf is a shim
        onto the worker's stderr whose `name` is a per-game path that
        does not exist (the loop unlinks it on success). A new game goes
        to the live worker with the fewest games in flight, or to a new
        worker when every live one is full."""
        self._evict_dead()
        live = [w for w in self._workers if w.alive() and w.killed_for is None]
        free = [w for w in live if not w.busy]
        if free:
            w = min(free, key=lambda w: w.in_flight)
        elif len(live) < self._n:
            w = self._spawn()
        else:
            raise RuntimeError("no free worker slot (the loop admits at most n * slots jobs)")
        handle = w.submit(argv)
        shim = _ErrShim(w.errf, str(self._errdir / f".stderr_{game_tag}.log"))
        return handle, shim

    @property
    def workers(self) -> List[Worker]:
        return list(self._workers)

    @property
    def dead_worker_logs(self) -> List[Path]:
        """Stderr logs of workers that died mid-match. Each one is the
        only record of why that worker went away."""
        return list(self._dead_logs)

    def shutdown(self):
        self._evict_dead()           # close whatever already died
        for w in self._workers:
            w.close()
        self._workers = []
