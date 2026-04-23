"""Pick the best available torch device.

On this machine:
- torch_directml.device(1) = AMD Radeon RX 6600 (discrete, what we want).
- torch_directml.device(0) = integrated iGPU, slower than CPU — AVOID.
- cpu fallback for environments without torch-directml installed.

The picker returns a `torch.device`-compatible object. Phase-3 code
should call `select_device()` once at startup and pass the result
around; do NOT assume CUDA is ever available, and do NOT call
`torch_directml.device()` without an explicit index.
"""

from __future__ import annotations

import os
from typing import Optional

import torch


def select_device(prefer: str = "auto"):
    """Return a torch device for model/training ops.

    Args:
        prefer: "auto" to auto-select; "cpu" to force CPU; "dml:N" to
                force DirectML device index N (0 = iGPU here, 1 = dGPU).
                Env var WESNOTH_AI_DEVICE overrides if set.

    Returns:
        A torch.device or an equivalent DirectML device handle.
    """
    override = os.environ.get("WESNOTH_AI_DEVICE", "").strip()
    if override:
        prefer = override

    if prefer == "cpu":
        return torch.device("cpu")

    if prefer.startswith("dml:"):
        idx = int(prefer.split(":", 1)[1])
        return _dml_device(idx)

    # auto: try the discrete DirectML device, fall back to CPU.
    dml = _dml_device(1)
    return dml if dml is not None else torch.device("cpu")


def _dml_device(index: int):
    """DirectML device at `index`, or None if torch-directml isn't usable."""
    try:
        import torch_directml
    except ImportError:
        return None

    try:
        count = torch_directml.device_count()
    except Exception:
        return None
    if index < 0 or index >= count:
        return None

    try:
        return torch_directml.device(index)
    except Exception:
        return None


def describe(device) -> str:
    """Human-readable name for logging."""
    try:
        import torch_directml
        # torch_directml devices stringify as "privateuseone:N".
        s = str(device)
        if s.startswith("privateuseone:"):
            idx = int(s.split(":", 1)[1])
            return f"{s} ({torch_directml.device_name(idx)})"
    except ImportError:
        pass
    return str(device)
