"""The learned memory of the parity-memory recipe
(docs/parity_memory_design_20260929.md, "The memory").

A side's memory is k float32 rows of the model's width, k fixed for a
game-side, 0 <= k <= the model's `memory_slots`. At each of its decisions
the side's network reads the memory it wrote at its previous decision as k
tokens of the MEMORY kind, each with a learned slot embedding added, beside
the other tokens, and writes a new one from the trunk's outputs at those
tokens through a gate per slot and channel (GTrXL: Parisotto et al.,
"Stabilizing Transformers for Reinforcement Learning", ICML 2020):

    z  = sigmoid(W_z [h; m] + b_z)
    m' = (1 - z) * m + z * tanh(W_c h)

with h the trunk's output at the slot and m the slot's state. A game-side
starts from the learned initial memory. The state and the write are
float32 even when the trunk runs under bfloat16 autocast: the trunk reads a
cast of the state, and a bfloat16 state would lose a small update to
rounding at every step of a game of 100 to 300 steps. A run with k slots
uses the first k rows of the initial memory and of the slot embedding and
nothing past them, so one checkpoint trained at nested sizes plays at any
k (nested dropout: Rippel, Gelbart and Adams, ICML 2014).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn

# b_z's initial value: sigmoid(-2) = 0.12, so a fresh network keeps about
# 88% of each slot at each step (the design's "most of its memory").
MEMORY_GATE_BIAS_INIT = -2.0
# The learned initial memory starts small, inside the (-1, 1) range the
# write keeps the state in (a convex mix of the state and a tanh).
INITIAL_MEMORY_STD = 0.02


@dataclass
class MemoryBatch:
    """A batch's memory states as the trunk reads them: [B, K_max, d]
    float32, zeros past each sample's slot count."""
    padded: torch.Tensor
    counts: List[int]

    @property
    def K_max(self) -> int:
        return int(self.padded.size(1))


class SlotMemory(nn.Module):
    """The learned initial memory [slots, d], the slot embedding [slots, d]
    and the gated write."""

    def __init__(self, d_model: int, slots: int):
        super().__init__()
        if slots < 1:
            raise ValueError(f"SlotMemory needs at least one slot, got {slots}")
        self.d_model = int(d_model)
        self.slots = int(slots)
        self.initial = nn.Parameter(torch.randn(slots, d_model) * INITIAL_MEMORY_STD)
        self.slot_embed = nn.Embedding(slots, d_model)
        self.gate = nn.Linear(2 * d_model, d_model)
        self.candidate = nn.Linear(d_model, d_model)
        with torch.no_grad():
            self.gate.bias.fill_(MEMORY_GATE_BIAS_INIT)

    def initial_state(self, k: int) -> torch.Tensor:
        """float32 [k, d]: a game-side's memory before its first
        decision, a copy of the first k rows of the learned initial
        memory. The copy keeps the autograd link (in training the
        gradient reaches the parameter) but not the storage, so a weight
        publication never changes a state a player holds."""
        if not 0 <= int(k) <= self.slots:
            raise ValueError(f"{k} active slots asked of a memory of {self.slots}")
        return self.initial[:int(k)].float().clone()

    def batch(self, states: Sequence[torch.Tensor], device: torch.device) -> MemoryBatch:
        """The per-sample states ([k_b, d] float32 each) padded for the trunk."""
        for s in states:
            self._check_state(s)
        counts = [int(s.size(0)) for s in states]
        if max(counts, default=0) == 0:
            padded = torch.zeros(len(states), 0, self.d_model, device=device)
        else:
            padded = nn.utils.rnn.pad_sequence([s.to(device) for s in states], batch_first=True)
        return MemoryBatch(padded=padded, counts=counts)

    def tokens(self, batch: MemoryBatch) -> torch.Tensor:
        """[B, K_max, d]: each slot's state plus its slot embedding, the
        memory tokens before the token-kind term. Pad slots hold their
        slot embedding; the key-padding mask hides them."""
        return batch.padded + self.slot_embed.weight[:batch.K_max]

    def stream_rows(self, batch: MemoryBatch) -> torch.Tensor:
        """[sum k_b, d]: the memory tokens of every sample's active slots,
        sample after sample (the stream order of the packed layout)."""
        rows = np.concatenate([b * batch.K_max + np.arange(k, dtype=np.int64)
                               for b, k in enumerate(batch.counts)] or [np.zeros(0, np.int64)])
        tokens = self.tokens(batch).reshape(-1, self.d_model)
        index = torch.from_numpy(rows)
        if tokens.device.type == "cuda":
            index = index.pin_memory().to(tokens.device, non_blocking=True)
        elif tokens.device.type != "cpu":
            index = index.to(tokens.device)
        return tokens.index_select(0, index)

    def write(self, h: torch.Tensor, batch: MemoryBatch) -> torch.Tensor:
        """The new states, float32 [B, K_max, d] padded like `batch` (rows
        past a sample's count hold no state), from the trunk's outputs h
        [B, K_max, d] at the memory tokens. Runs in float32 whatever
        autocast is active."""
        with torch.autocast(h.device.type, enabled=False):
            m = batch.padded
            h = h.float()
            z = torch.sigmoid(self.gate(torch.cat([h, m], dim=-1)))
            return (1.0 - z) * m + z * torch.tanh(self.candidate(h))

    def _check_state(self, s: torch.Tensor) -> None:
        if s.dim() != 2 or s.size(1) != self.d_model:
            raise ValueError(f"a memory state is [k, {self.d_model}], got {tuple(s.shape)}")
        if s.size(0) > self.slots:
            raise ValueError(f"a memory state of {s.size(0)} slots for a memory of {self.slots}")
        if s.dtype != torch.float32:
            raise ValueError(f"a memory state is float32, got {s.dtype}")


def refuse_memory_model(model, consumer: str) -> None:
    """Raise when `model` carries a memory: `consumer` does not keep each
    side's memory state from one decision to the next yet
    (docs/parity_memory_design_20260929.md, "Serving and play")."""
    slots = int(getattr(model, "memory_slots", 0) or 0)
    if slots > 0:
        raise ValueError(
            f"{consumer} cannot play a model with a memory (memory_slots={slots}): it does not "
            f"carry each side's memory state from one decision to the next yet "
            f"(docs/parity_memory_design_20260929.md, \"Serving and play\")")
