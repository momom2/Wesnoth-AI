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


def is_dml(device) -> bool:
    """True iff `device` is a DirectML (torch-directml) device. DML
    devices stringify as `privateuseone:N`; the underlying
    `torch.device.type` is `"privateuseone"`. Used by the training
    path to apply DML-specific mitigations (sync transfers,
    command-list flushes)."""
    try:
        # torch.device-like: has a `.type` attribute.
        return getattr(device, "type", "") == "privateuseone"
    except Exception:
        return False


def dml_sync(device) -> None:
    """Drain DirectML's pending command list by forcing a CPU
    round-trip.

    torch-directml 0.2.5 doesn't expose a public `synchronize()` or
    `empty_cache()` API. But any `.item()` / `.cpu()` on a DML tensor
    blocks until every queued op for that device has completed (the
    destination value must be populated). We exploit that by sending
    a tiny scalar through the round-trip explicitly.

    Why we need this: on the AMD RX 6600 + torch-directml 0.2.5
    combo, our training loop dies with "GPU device instance has been
    suspended" partway through the first `train_step`. The crash
    signature (suspended mid-chunk, partway through `_cat_to_dev`)
    is consistent with command-list overgrowth from the rollout
    phase (6 worker threads, thousands of tiny `.to()` + forwards)
    spilling into train_step without being flushed. Forcing a sync:
      * between rollout and train_step,
      * periodically inside the chunk loops,
    bounds the queue depth and keeps the driver alive.

    No-op on non-DML devices (CPU has no async queue; CUDA's
    semantics make this unnecessary -- next-kernel-uses-the-tensor
    forces the wait organically).

    Safe to call from any thread: the underlying `.item()` is a
    blocking call that doesn't interfere with concurrent forwards
    on the same device.
    """
    if not is_dml(device):
        return
    try:
        # The .item() call forces a device->host transfer, which
        # blocks until all queued device-side work has flushed.
        # The tensor itself is throwaway -- we don't care about
        # the value (always 0.0).
        torch.zeros(1, device=device).item()
    except RuntimeError:
        # If the device is already suspended this raises; let the
        # caller see the original exception rather than masking
        # it -- recovery isn't possible from this side.
        raise
