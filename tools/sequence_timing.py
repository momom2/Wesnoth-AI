"""The sequence trainer's timing stream and its activation budget
(tools/sequence_train.py).

Every row of <out>.steps.jsonl carries the window's timing
(`WindowClock`): its wall time on the host, the trunk's tokens (the real
ones and the padded batch's), the device memory's peak, the time garbage
collection paused the process, the time the training thread waited for a
game record still being read, and the CPU time the loader's threads spent
decoding records. With an activation budget, the row also says how many of
the window's last time steps kept their activations (`plan_kept_steps`)
and, on CUDA, the memory those activations measured.

Memory figures are in GiB (2**30 bytes), the unit `--activation-budget-gb`
takes.
"""
from __future__ import annotations

import gc
import time
from typing import Dict, Optional, Sequence, Tuple

import torch

GIB = float(2 ** 30)

# The activations one token leaves for the backward pass in one trunk
# layer, at the recipe's width (d_model 384, d_ff 1536) under bfloat16
# autocast: docs/design_constants.md "Activation bytes per token". The
# padded trunk's count is per padded token, the packed trunk's per real
# token. Measured figures replace them through --activation-bytes-per-token.
ACTIVATION_BYTES_PER_TOKEN_LAYER = {"padded": 17_000, "packed": 10_800}
RECIPE_D_MODEL = 384


class GcClock:
    """The time garbage collection has paused the process, from
    `gc.callbacks` (one clock per process: `gc_clock`)."""

    def __init__(self):
        self.total = 0.0
        self._start: Optional[float] = None

    def __call__(self, phase: str, info) -> None:
        if phase == "start":
            self._start = time.perf_counter()
        elif self._start is not None:
            self.total += time.perf_counter() - self._start
            self._start = None


_GC_CLOCK: Optional[GcClock] = None


def gc_clock() -> GcClock:
    """The process's GC clock, installed on first use."""
    global _GC_CLOCK
    if _GC_CLOCK is None:
        _GC_CLOCK = GcClock()
        gc.callbacks.append(_GC_CLOCK)
    return _GC_CLOCK


def trunk_tokens(sizes: Sequence[Tuple[int, int, int]], sightings: Optional[Sequence[int]],
                 memory: Sequence[int]) -> Tuple[int, int]:
    """(real, padded) trunk tokens of one time step: each row holds its
    hexes, units, recruits, sightings, memory slots, the global and the
    end_turn token; the padded batch holds every row at the widest of each
    block."""
    S = list(sightings) if sightings is not None else [0] * len(sizes)
    real = sum(u + r + h + s + k + 2 for (u, r, h), s, k in zip(sizes, S, memory))
    widest = (max(h for _, _, h in sizes) + max(u for u, _, _ in sizes) + max(r for _, r, _ in sizes)
              + max(S, default=0) + max(memory, default=0) + 2)
    return real, len(sizes) * widest


def position_sizes(raws) -> Tuple[list, list]:
    """The per-row (U, R, H) and sighting counts `encoder.stage_raws` gives
    the same records, read off their arrays without staging them."""
    sizes = [(r.unit_type_ids.shape[0], r.recruit_type_ids.shape[0], r.hex_xs.shape[0]) for r in raws]
    return sizes, [0 if r.sight_type_ids is None else r.sight_type_ids.shape[0] for r in raws]


def bytes_per_token(trunk: str, arch: Dict[str, int], measured: Optional[float] = None) -> float:
    """The activations a token of `trunk` keeps across the whole trunk:
    `measured` when given, else the per-layer constant scaled to the
    architecture (activations grow with the width, d_ff being 4 x d_model
    in the recipe)."""
    if measured is not None:
        return float(measured)
    return ACTIVATION_BYTES_PER_TOKEN_LAYER[trunk] * arch["num_layers"] * arch["d_model"] / RECIPE_D_MODEL


def plan_kept_steps(step_bytes: Sequence[float], budget_bytes: float) -> int:
    """How many of the window's last time steps keep their activations: the
    longest suffix whose estimated bytes fit the budget together. The
    backward pass consumes the last time steps first, so their memory is
    freed before the earlier steps are recomputed."""
    kept, used = 0, 0.0
    for b in reversed(step_bytes):
        if used + b > budget_bytes:
            break
        used += b
        kept += 1
    return kept


class WindowClock:
    """One window's timing fields (module docstring)."""

    def __init__(self, device: torch.device, loader):
        self.device, self.loader = device, loader
        self.gc = gc_clock()
        self._t0 = 0.0
        self._marks = (0.0, 0.0, 0.0)

    def start(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._marks = (self.gc.total, self.loader.wait_seconds, self.loader.decode_seconds)
        self._t0 = time.perf_counter()

    def memory_allocated(self) -> int:
        """The device's allocated bytes now (no synchronization); 0 off CUDA."""
        return torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else 0

    def finish(self, tokens_real: int, tokens_padded: int, kept_steps: int, kept_tokens: int,
               kept_bytes: int) -> Dict:
        seconds = time.perf_counter() - self._t0
        gc0, wait0, decode0 = self._marks
        cuda = self.device.type == "cuda"
        return {"seconds": round(seconds, 4), "tokens_real": int(tokens_real),
                "tokens_padded": int(tokens_padded),
                "peak_gb": round(torch.cuda.max_memory_allocated(self.device) / GIB, 3) if cuda else None,
                "gc_seconds": round(self.gc.total - gc0, 4),
                "loader_wait_seconds": round(self.loader.wait_seconds - wait0, 4),
                "decode_seconds": round(self.loader.decode_seconds - decode0, 4),
                "kept_steps": int(kept_steps), "kept_tokens": int(kept_tokens),
                "kept_gb": round(kept_bytes / GIB, 3) if cuda else None}
