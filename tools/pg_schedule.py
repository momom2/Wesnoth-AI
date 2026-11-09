"""The policy-gradient pass's walk over its game-sides
(docs/training_design_20261019/current_design.md §1 step 3, §2 "The round
size").

The pass keeps eight split accumulators, a game's split being its index
modulo 8 (its label row's `split`). Streams interleave game-sides within a
window, so a window that mixed splits would mix their gradients: each split
runs its own schedule (`wesnoth_ai.sequence_streams.StreamSchedule`), one
after the other, and its windows hold its game-sides only. Within a split
the games are shuffled by the seed and a game's two sides follow each other,
so the two sides of one replayed game are dealt to slots together.

`PGMemoryRouter` is the trainer's memory router (tools/sequence_memory.py)
for rows that may be shadows: a refused decision reads the memory its side
holds and writes nothing, so the side's next decision reads the same memory
(in play the refused decision's write was undone,
`RawPolicyPlayer.drop_last_pending`). A step's padded memory batch is one
gather from a pool of the states its rows read, as the trainer's is; a state
two rows read (a shadow row and the decision after it) is gathered twice,
and its gradient sums both.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from wesnoth_ai.memory import MemoryBatch
from wesnoth_ai.packed_trunk import host_to_device
from wesnoth_ai.pg_records import N_SPLITS
from wesnoth_ai.sequence_streams import GameSide, Step, StreamSchedule


def game_order(games: Sequence[str], seed: int) -> List[GameSide]:
    """The games sorted, shuffled by the seed, each game's side 1 then
    side 2."""
    order = sorted(games)
    random.Random(seed).shuffle(order)
    return [GameSide(g, s) for g in order for s in (1, 2)]


def split_schedules(splits: Dict[str, int], lengths: Dict[GameSide, int], n_streams: int, seed: int,
                    k: int) -> List[StreamSchedule]:
    """One schedule per split, in split order: its games' sides on
    `n_streams` slots, every game-side at `k` memory slots."""
    by_split: Dict[int, List[str]] = {i: [] for i in range(N_SPLITS)}
    for game, split in splits.items():
        by_split[int(split)].append(game)
    return [StreamSchedule(game_order(by_split[i], seed + i), lengths, n_streams, seed, k=k, name=f"split {i}")
            for i in range(N_SPLITS)]


# A state a row reads: a padded output [B, K, d] and the row's index in it,
# or a state [k, d] and None.
_Source = Tuple[torch.Tensor, Optional[int]]


class PGMemoryRouter:
    """The slots' memories along one window, shadow rows honoured.
    `carried`: each slot's memory at the window's start ([k, d],
    detached)."""

    def __init__(self, model, device: torch.device, carried: Dict[int, torch.Tensor]):
        self.model, self.device = model, device
        self.carried = carried
        self._latest: Dict[int, Tuple[_Source, int]] = {}      # slot -> (what its next row reads, k)
        self._read: List[_Source] = []

    def _source(self, s: Step) -> _Source:
        if s.starts:
            return self.model.initial_memory(s.k).to(self.device), None
        latest = self._latest.get(s.slot)
        if latest is not None:
            return latest[0]
        return self.carried[s.slot], None

    def batch(self, steps: Sequence[Step]) -> MemoryBatch:
        """The step's memories, padded [B, K_max, d] (zeros past a row's
        slot count): each row's initial memory, the state its slot's last
        writing row wrote, or the memory its slot carried in."""
        d = self.model.d_model
        self._read = [self._source(s) for s in steps]
        counts = [s.k for s in steps]
        K_max = max(counts, default=0)
        if K_max == 0:
            return MemoryBatch(padded=torch.zeros(len(steps), 0, d, device=self.device), counts=counts)
        parts, offsets, starts, n_rows = [], {}, [], 0
        for (state, row), s in zip(self._read, steps):
            if id(state) not in offsets:
                offsets[id(state)] = n_rows
                parts.append(state.reshape(-1, d))
                n_rows += parts[-1].shape[0]
            start = offsets[id(state)] + (0 if row is None else row * state.shape[1])
            width = state.shape[0] if row is None else state.shape[1]
            if s.k > width:
                raise ValueError(f"slot {s.slot} reads a memory of {width} slots, its game-side runs at {s.k}")
            starts.append(start)
        parts.append(torch.zeros(1, d, device=self.device))
        index = np.full((len(steps), K_max), n_rows, dtype=np.int64)
        for b, (start, k) in enumerate(zip(starts, counts)):
            index[b, :k] = np.arange(start, start + k)
        pool = torch.cat(parts)
        padded = pool.index_select(0, host_to_device(index.reshape(-1), self.device))
        return MemoryBatch(padded=padded.view(len(steps), K_max, d), counts=counts)

    def wrote(self, steps: Sequence[Step], padded: torch.Tensor, shadow: Sequence[bool]) -> None:
        """The step's new memories [B, K_max, d], in the step's row order;
        a shadow row's slot keeps what the row read."""
        for b, s in enumerate(steps):
            self._latest[s.slot] = (self._read[b] if shadow[b] else (padded, b), s.k)

    def memories(self) -> Dict[int, torch.Tensor]:
        """Each slot's memory at the window's end ([k, d], in the graph when
        the window wrote it): what the slot carries out."""
        out = dict(self.carried)
        for slot, ((state, row), k) in self._latest.items():
            out[slot] = state[:k] if row is None else state[row, :k]
        return out
