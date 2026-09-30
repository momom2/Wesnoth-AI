"""The sequence trainer's streams (docs/parity_memory_design_20260929.md
"Training").

Each game gives two game-sides, one per player side. The pass deals them in
a seeded order to `n_streams` slots; each slot walks its game-side's
decisions in order and takes the next game-side of the order when one ends,
starting it from the learned initial memory. A window is `T` consecutive
time steps of every slot; the trainer back-propagates through the memory
across a window and carries it into the next without gradient.

What a game-side needs besides its decisions is a pure function of the run's
seed and the game-side: its memory size `k` (nested dropout, drawn from
`K_CHOICES` with `K_WEIGHTS`) and which of its positions carry the value
loss. So the schedule's state is a handful of integers, and a resumed run
continues exactly.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

# The memory sizes a game-side trains at and their weights: 1/8 each for
# 0, 8, 16 and 32 slots, 1/2 for all 64 (the design's "Sizes").
K_CHOICES = (0, 8, 16, 32, 64)
K_WEIGHTS = (1, 1, 1, 1, 4)


@dataclass(frozen=True, order=True)
class GameSide:
    file: str
    side: int


@dataclass(frozen=True)
class Step:
    """One position of one slot at one time step of a window."""
    slot: int
    game_side: GameSide
    offset: int                  # the decision's index in its game-side
    k: int                       # the game-side's memory size
    starts: bool                 # the game-side's first decision: memory from its initial state


def unit_draw(*parts) -> float:
    """A uniform draw in [0, 1) that is a pure function of `parts`."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2.0 ** 64


def memory_size(seed: int, game_side: GameSide) -> int:
    """The game-side's memory size, drawn once from `K_CHOICES`."""
    u = unit_draw(seed, "k", game_side.file, game_side.side) * sum(K_WEIGHTS)
    for k, w in zip(K_CHOICES, K_WEIGHTS):
        if u < w:
            return k
        u -= w
    return K_CHOICES[-1]


def value_selected(seed: int, game_side: GameSide, offset: int, p: float) -> bool:
    """Whether the position carries the value loss: each position of a game
    with n commands is kept with probability min(1, states per game / n)."""
    return unit_draw(seed, "v", game_side.file, game_side.side, offset) < p


def epoch_order(game_sides: Sequence[GameSide], seed: int) -> List[GameSide]:
    """The pass's order: the game-sides sorted, then shuffled by the seed."""
    order = sorted(game_sides)
    random.Random(seed).shuffle(order)
    return order


class StreamSchedule:
    """The slots' walk over the order. `lengths` holds each game-side's
    number of decisions; a game-side without any is passed over."""

    def __init__(self, order: Sequence[GameSide], lengths: Dict[GameSide, int], n_streams: int,
                 seed: int):
        if n_streams < 1:
            raise ValueError(f"n_streams must be at least 1, got {n_streams}")
        self.order = [g for g in order if lengths.get(g, 0) > 0]
        self.lengths = {g: int(lengths[g]) for g in self.order}
        self.seed = int(seed)
        self.n_streams = int(n_streams)
        self.next_index = 0
        self.slots: List[Tuple[int, int]] = [(-1, 0)] * self.n_streams   # (order index, offset)

    @property
    def total_positions(self) -> int:
        return sum(self.lengths.values())

    def exhausted(self) -> bool:
        return self.next_index >= len(self.order) and all(i < 0 for i, _ in self.slots)

    def _refill(self, s: int) -> None:
        if self.slots[s][0] < 0 and self.next_index < len(self.order):
            self.slots[s] = (self.next_index, 0)
            self.next_index += 1

    def window(self, T: int) -> List[List[Step]]:
        """The next `T` time steps: per step, the slots that hold a position,
        in slot order. Fewer than `T` lists when the pass runs out."""
        out: List[List[Step]] = []
        for _ in range(T):
            steps: List[Step] = []
            for s in range(self.n_streams):
                self._refill(s)
                index, offset = self.slots[s]
                if index < 0:
                    continue
                g = self.order[index]
                steps.append(Step(slot=s, game_side=g, offset=offset,
                                  k=memory_size(self.seed, g), starts=offset == 0))
                offset += 1
                self.slots[s] = (-1, 0) if offset >= self.lengths[g] else (index, offset)
            if not steps:
                break
            out.append(steps)
        return out

    def upcoming(self, n: int) -> List[GameSide]:
        """The game-sides the slots hold now and the next `n` of the order,
        for a loader to fetch ahead."""
        held = [self.order[i] for i, _ in self.slots if i >= 0]
        return held + self.order[self.next_index:self.next_index + n]

    def state_dict(self) -> Dict:
        return {"next_index": self.next_index, "slots": [list(s) for s in self.slots],
                "n_streams": self.n_streams, "seed": self.seed, "n_order": len(self.order)}

    def load_state_dict(self, state: Dict) -> None:
        if int(state["n_streams"]) != self.n_streams or int(state["seed"]) != self.seed \
                or int(state["n_order"]) != len(self.order):
            raise ValueError(f"the saved schedule ({state['n_streams']} streams, seed {state['seed']}, "
                             f"{state['n_order']} game-sides) is not this run's ({self.n_streams}, "
                             f"{self.seed}, {len(self.order)})")
        self.next_index = int(state["next_index"])
        self.slots = [(int(i), int(o)) for i, o in state["slots"]]
