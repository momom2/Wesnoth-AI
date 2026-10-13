"""The memory each slot carries through a window of the sequence trainer
(tools/sequence_train.py), batched for the trunk in one indexing.

At a time step a row's memory is the initial memory (its game-side's first
decision), the memory the slot wrote at the previous time step of the
window, or the memory it carried into the window (detached). The step's
padded batch [B, K_max, d] is one `index_select` from a pool of rows: the
previous step's padded output, then a fresh copy of the initial memory for
each starting row and each carried memory, then a zero row for the pads.
Per-row slices and a padding copy per row, a few hundred small kernels a
time step forward and backward, become one gather and its scatter.

The values are the ones `SlotMemory.batch` pads (copies of the same rows,
zeros at the pads), and so are the gradients: each pool row is read at most
once, so the scatter adds every row's gradient to zero once, as the sum of
the per-row slices' zero-padded gradients did; the initial memory's copies
are made per starting row, in row order, as before.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from wesnoth_ai.memory import MemoryBatch
from wesnoth_ai.packed_trunk import host_to_device
from wesnoth_ai.sequence_streams import Step


class MemoryRouter:
    """The slots' memories along one window. `carried`: each slot's memory
    at the window's start (detached, [k, d])."""

    def __init__(self, model, device: torch.device, carried: Dict[int, torch.Tensor]):
        self.model, self.device = model, device
        self.carried = carried
        self._previous: Optional[torch.Tensor] = None          # the last step's padded output
        self._previous_rows: Dict[int, int] = {}               # slot -> its row in it
        self._latest: Dict[int, Tuple[torch.Tensor, int, int]] = {}   # slot -> (padded output, row, k)
        # What the last batch's rows read: ("initial" | "carried", state, None)
        # or ("previous", the previous step's output, its row).
        self.read: List[Tuple[str, torch.Tensor, Optional[int]]] = []

    def batch(self, steps: Sequence[Step]) -> MemoryBatch:
        """The step's memories, padded: what `SlotMemory.batch` gives for
        the same states."""
        d = self.model.d_model
        parts, n_rows, starts = [], 0, []
        self.read = []
        # The previous output joins the pool only when a row reads it: a pool
        # that held it unread would send it a zero gradient, which gives the
        # memory's write a zero gradient where it has none, and the optimizer
        # then decays the weights the window did not train.
        if any(not s.starts and s.slot in self._previous_rows for s in steps):
            K = self._previous.size(1)
            parts.append(self._previous.reshape(-1, d))
            n_rows = parts[0].size(0)
        for s in steps:
            if not s.starts and s.slot in self._previous_rows:
                starts.append(self._previous_rows[s.slot] * K)
                self.read.append(("previous", self._previous, self._previous_rows[s.slot]))
                continue
            state = self.model.initial_memory(s.k).to(self.device) if s.starts else self.carried[s.slot]
            self.read.append(("initial" if s.starts else "carried", state, None))
            if state.shape != (s.k, d):
                raise ValueError(f"slot {s.slot} carries a memory of shape {tuple(state.shape)}, its "
                                 f"game-side runs at {s.k} slots of width {d}")
            parts.append(state)
            starts.append(n_rows)
            n_rows += s.k
        counts = [s.k for s in steps]
        K_max = max(counts, default=0)
        if K_max == 0:
            return MemoryBatch(padded=torch.zeros(len(steps), 0, d, device=self.device), counts=counts)
        parts.append(torch.zeros(1, d, device=self.device))
        index = np.full((len(steps), K_max), n_rows, dtype=np.int64)
        for b, (start, k) in enumerate(zip(starts, counts)):
            index[b, :k] = np.arange(start, start + k)
        pool = torch.cat(parts)
        padded = pool.index_select(0, host_to_device(index.reshape(-1), self.device))
        return MemoryBatch(padded=padded.view(len(steps), K_max, d), counts=counts)

    def wrote(self, steps: Sequence[Step], padded: torch.Tensor) -> None:
        """The step's new memories, [B, K_max, d] in the step's row order."""
        self._previous = padded
        self._previous_rows = {s.slot: b for b, s in enumerate(steps)}
        for b, s in enumerate(steps):
            self._latest[s.slot] = (padded, b, s.k)

    def memories(self) -> Dict[int, torch.Tensor]:
        """Each slot's latest memory ([k, d], in the graph when the window
        wrote it): what the slot carries out of the window."""
        out = dict(self.carried)
        for slot, (padded, b, k) in self._latest.items():
            out[slot] = padded[b, :k]
        return out
