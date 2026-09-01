"""Capture the states search consults during harvest (v2 stage).

The value head's training loss lives on REAL recorded states, but at
play time search reads the head on IMAGINED states — candidate turn
boundaries it materializes and grades. `capture_consultations` hooks
`batch_boundary_values` (the gate-candidate read path, the dominant
consultation volume) and reservoir-samples the non-terminal states
it evaluates. Holding the `gs` reference is enough THERE: the batch
reader frees each `boundary_sim` right after grading, so the object
is never mutated again. The `boundary_value` single-read path
(spines, projections) is NOT captured — its sims keep stepping after
the read, so a held reference would alias moving state.
"""
from __future__ import annotations

import random
from contextlib import contextmanager
from typing import List


class ConsultReservoir:
    """Fixed-size uniform reservoir over consulted states."""

    def __init__(self, cap: int, seed: int):
        self.cap = int(cap)
        self.states: List = []
        self.seen = 0
        self._rng = random.Random(seed)

    def offer(self, gs) -> None:
        self.seen += 1
        if len(self.states) < self.cap:
            self.states.append(gs)
        else:
            j = self._rng.randrange(self.seen)
            if j < self.cap:
                self.states[j] = gs


@contextmanager
def capture_consultations(cap: int = 400, seed: int = 0):
    """Patch the turn_search value-read paths for the duration of a
    harvest. Yields the reservoir; read `.states` afterwards."""
    from tools import turn_search

    res = ConsultReservoir(cap, seed)
    orig_batch = turn_search.batch_boundary_values

    def batch_hook(policy, mats, side, decision_step, *a, **kw):
        for m in mats:
            if not m.invalid and m.boundary_sim is not None \
                    and not m.boundary_sim.done:
                res.offer(m.boundary_sim.gs)
        return orig_batch(policy, mats, side, decision_step, *a, **kw)

    turn_search.batch_boundary_values = batch_hook
    try:
        yield res
    finally:
        turn_search.batch_boundary_values = orig_batch
