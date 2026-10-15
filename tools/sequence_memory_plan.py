"""The activation budget sized from the device's memory, measured in the run
(`--activation-budget-gb auto`; tools/sequence_train.py, docs/trainer_speed_20261013.md
"The activation budget under the compiled block").

A window's peak is its kept activations (the time steps whose forward is
not recomputed in the backward pass, tools/sequence_timing.plan_kept_steps)
plus the rest of the window: the embedding of its positions, the heads and
losses over them and their gradients, the trunk outputs' gradients, the
weights, the optimizer's state, a recomputed step. A fixed budget for the
kept activations alone leaves that rest to luck: the compiled packed trunk
under a 19.18 GiB budget kept 16-17 GiB as estimated (93,000 bytes a kept
token against the 102,139 assumed), and its rest, 1.8-3.4 GiB growing with
the window's tokens, with 3.2 GiB of the allocator's fragmentation, ran a
24 GiB card out of memory.

Here the budget is what the device leaves. Before each trained window the
kept activations may take

    capacity - outside - margin - rest_per_token * tokens(window)

where capacity is the device's memory, outside the most the process held
outside PyTorch's allocator so far (the CUDA context, the libraries),
margin `--memory-margin-gb`, and rest_per_token the largest rest a real
trunk token has needed so far (the window's peak allocation less its kept
bytes, over its real tokens), 32,000 bytes before any window (about twice
the largest measured, the compiled trunk's first window: 18,195). The
kept steps are then planned at the largest kept bytes a token measured so
far (the configured estimate before). Both figures only grow, so a window
like one seen fits as it did, and a larger window gets a smaller budget.
Fragmentation is the allocator's: expandable segments, the trainer's
default (tools/sequence_setup.default_cuda_allocator), keep the reserved
memory near the allocated, and the margin covers the rest.

Keeping a step or recomputing it computes the same numbers, so the plan
changes the speed and the memory, never the training.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict

import torch

GIB = float(2 ** 30)
# The rest of a window per real trunk token before any window is measured
# (module docstring).
PRIOR_REST_BYTES_PER_TOKEN = 32_000.0


@dataclass
class DeviceMemory:
    """What the planner reads of the device: its capacity, the bytes the
    process holds outside PyTorch's allocator, and the allocator's peak
    since the window's start (WindowClock resets it)."""
    capacity: Callable[[], int]
    outside: Callable[[], int]
    peak_allocated: Callable[[], int]


def device_memory(device: torch.device) -> DeviceMemory:
    """The CUDA device's memory; SystemExit elsewhere (nothing to plan against)."""
    if device.type != "cuda":
        raise SystemExit("--activation-budget-gb auto sizes the budget from a CUDA device's memory")

    def outside() -> int:
        free, total = torch.cuda.mem_get_info(device)
        return int(total - free - torch.cuda.memory_reserved(device))
    return DeviceMemory(capacity=lambda: int(torch.cuda.get_device_properties(device).total_memory),
                        outside=outside, peak_allocated=lambda: int(torch.cuda.max_memory_allocated(device)))


class MemoryPlanner:
    """The window's activation budget from the device's memory (module docstring)."""

    def __init__(self, memory: DeviceMemory, margin_bytes: float, kept_bytes_per_token: float):
        self.memory = memory
        self.margin = float(margin_bytes)
        self.capacity = float(memory.capacity())
        self.outside = float(memory.outside())
        self.kept_per_token = float(kept_bytes_per_token)
        self.rest_per_token = PRIOR_REST_BYTES_PER_TOKEN
        self._kept_measured = False
        self._rest_measured = False

    def budget(self, tokens_real: int) -> float:
        """The bytes the kept activations of a window of `tokens_real` real
        trunk tokens may take; 0 when the rest alone fills the device."""
        return max(0.0, self.capacity - self.outside - self.margin - self.rest_per_token * tokens_real)

    def observe(self, tokens_real: int, kept_tokens: int, kept_bytes: int) -> Dict:
        """A trained window's measurements; the figures the next plans use."""
        peak = float(self.memory.peak_allocated())
        self.outside = max(self.outside, float(self.memory.outside()))
        if kept_tokens > 0 and kept_bytes > 0:
            per = kept_bytes / kept_tokens
            self.kept_per_token = per if not self._kept_measured else max(self.kept_per_token, per)
            self._kept_measured = True
        rest = max(0.0, peak - kept_bytes) / max(1, tokens_real)
        self.rest_per_token = rest if not self._rest_measured else max(self.rest_per_token, rest)
        self._rest_measured = True
        return self.state()

    def state(self) -> Dict:
        return {"plan_capacity_gb": round(self.capacity / GIB, 3), "plan_outside_gb": round(self.outside / GIB, 3),
                "plan_margin_gb": round(self.margin / GIB, 3), "plan_kept_bytes_per_token": round(self.kept_per_token),
                "plan_rest_bytes_per_token": round(self.rest_per_token)}


# The CUDA caching allocator's configuration the trainer runs under unless
# the environment sets one: expandable segments, which grow a segment in
# place instead of reserving new fixed-size ones, so freed memory serves
# requests of other sizes (the fragmentation that left 3.2 GiB reserved and
# unallocated when the compiled trunk ran out of memory).
ALLOC_CONF_VARIABLE = "PYTORCH_CUDA_ALLOC_CONF"
DEFAULT_ALLOC_CONF = "expandable_segments:True"


def default_cuda_allocator(device: torch.device) -> Dict:
    """Set PYTORCH_CUDA_ALLOC_CONF to DEFAULT_ALLOC_CONF for a CUDA run when
    the environment sets none; what the run's allocator reads, and from
    where, for the training meta. The allocator reads the variable when CUDA
    first allocates: set after that, it would not apply, and the record says
    so."""
    current = os.environ.get(ALLOC_CONF_VARIABLE)
    if device.type != "cuda":
        return {"alloc_conf": current, "source": "not CUDA"}
    if current is not None:
        return {"alloc_conf": current, "source": "trainer default" if current == _SET_BY_TRAINER.get("value")
                else "environment"}
    if torch.cuda.is_initialized():
        return {"alloc_conf": None, "source": "CUDA initialized before the trainer: the allocator's own default"}
    os.environ[ALLOC_CONF_VARIABLE] = DEFAULT_ALLOC_CONF
    _SET_BY_TRAINER["value"] = DEFAULT_ALLOC_CONF
    return {"alloc_conf": DEFAULT_ALLOC_CONF, "source": "trainer default"}


# The value this process set, so a second trainer of the process (a bench's
# arms) records where it came from.
_SET_BY_TRAINER: Dict[str, str] = {}
