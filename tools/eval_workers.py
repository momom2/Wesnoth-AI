"""Persistent evaluation workers for tools/run_elo_batch.py (plan
step 1.5).

One `elo_eval_game.py` process per game paid the checkpoint load,
CUDA init and torch.compile every time: of a 35 s raw game in the
2026-09-04 baseline, roughly 24 s was that. A worker is one
`elo_eval_game.py --worker` process that plays many games and keeps
the loaded policies across them. Protocol: the driver writes one JSON
argv list per line on the worker's stdin; the worker answers
`__DONE__ <rc>` on stdout once the game's result file is on disk.
`JobHandle` mimics the subprocess.Popen surface the batch loop drives
(poll / wait / kill / returncode / pid), so the loop is unchanged; a
killed worker (per-game timeout) is replaced on the next submit.
"""
from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional

DONE_PREFIX = "__DONE__ "


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
    def __init__(self, cmd: List[str], errf_path: Path):
        self.errf = open(errf_path, "w+b")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.errf, text=True, bufsize=1)
        self._lines: "queue.Queue[Optional[str]]" = queue.Queue()
        self.busy = False
        self.games = 0
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        try:
            for line in self.proc.stdout:
                if line.startswith(DONE_PREFIX):
                    self._lines.put(line)
        finally:
            self._lines.put(None)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def submit(self, argv: List[str]) -> "JobHandle":
        self.busy = True
        self.proc.stdin.write(json.dumps(argv) + "\n")
        self.proc.stdin.flush()
        return JobHandle(self)

    def kill(self):
        if self.alive():
            self.proc.kill()
        self.proc.wait()
        self.busy = False

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
    def __init__(self, worker: Worker):
        self._w = worker
        self.returncode: Optional[int] = None
        self.pid = worker.proc.pid

    def poll(self) -> Optional[int]:
        if self.returncode is not None:
            return self.returncode
        try:
            line = self._w._lines.get_nowait()
        except queue.Empty:
            if not self._w.alive():
                self.returncode = self._w.proc.returncode or 1
                self._w.busy = False
            return self.returncode
        if line is None:
            self.returncode = self._w.proc.poll() or 1
        else:
            try:
                self.returncode = int(line[len(DONE_PREFIX):].strip())
            except ValueError:
                self.returncode = 1
        self._w.busy = False
        self._w.games += 1
        return self.returncode

    def wait(self, timeout: Optional[float] = None) -> int:
        t0 = time.monotonic()
        while self.poll() is None:
            if timeout is not None and time.monotonic() - t0 > timeout:
                raise TimeoutError("worker job did not finish")
            time.sleep(0.05)
        return self.returncode

    def kill(self):
        self._w.kill()
        if self.returncode is None:
            self.returncode = -9


class WorkerPool:
    """Up to `n` workers spawned lazily; dead ones are replaced."""

    def __init__(self, cmd: List[str], n: int, errdir: Path):
        self._cmd = list(cmd)
        self._n = max(1, int(n))
        self._errdir = Path(errdir)
        self._workers: List[Worker] = []
        self._spawned = 0
        # Logs of workers that died; kept so a caller can read why, and
        # so `shutdown()` can say how many there were.
        self._dead_logs: List[Path] = []

    def _spawn(self) -> Worker:
        self._spawned += 1
        w = Worker(self._cmd, self._errdir / f".worker_{self._spawned}.log")
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
        """
        keep, drop = [], []
        for w in self._workers:
            (keep if (w.alive() or w.busy) else drop).append(w)
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
        does not exist (the loop unlinks it on success)."""
        self._evict_dead()
        idle = [w for w in self._workers if w.alive() and not w.busy]
        if idle:
            w = idle[0]
        elif len([w for w in self._workers if w.alive()]) < self._n:
            w = self._spawn()
        else:
            raise RuntimeError("no idle worker (the loop admits at most n jobs)")
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
