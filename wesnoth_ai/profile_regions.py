"""Named regions of the inference server's batch for torch.profiler
(tools/server_profile.py): `region(name)` is a `record_function` range
while a profile runs and a shared no-op context otherwise, so the
production path pays one function call per region. The device time of
the kernels launched inside a region is attributed to it, the innermost
region winning.

`note_batch` records a profiled batch's shape (leaves, tokens, segment
lengths, padded widths) for the profile's arithmetic per class.
"""
from __future__ import annotations

import contextlib
from typing import Dict, List, Sequence

_ACTIVE = False
_NULL = contextlib.nullcontext()
# The shapes of the batches served while a profile runs.
BATCHES: List[Dict[str, int]] = []


def region(name: str):
    if not _ACTIVE:
        return _NULL
    import torch
    return torch.profiler.record_function(name)


def active() -> bool:
    return _ACTIVE


def start() -> None:
    global _ACTIVE
    _ACTIVE = True
    BATCHES.clear()


def stop() -> List[Dict[str, int]]:
    global _ACTIVE
    _ACTIVE = False
    out = list(BATCHES)
    BATCHES.clear()
    return out


def note_batch(lengths: Sequence[int], a_max: int, h_max: int, k_max: int) -> None:
    """A profiled batch: each segment's tokens and the padded widths the
    heads run over."""
    if _ACTIVE:
        BATCHES.append({"leaves": len(lengths), "tokens": int(sum(lengths)),
                        "tokens_sq": int(sum(n * n for n in lengths)), "a_max": int(a_max),
                        "h_max": int(h_max), "k_max": int(k_max)})
