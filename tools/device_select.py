"""Torch device selection for inference AND training paths.

Centralizes the "pick the best available device" logic so every
tool (eval, diagnose, demo, self-play training) handles
`--device auto` the same way.

Priority order (when `prefer="auto"`):

  1. **DirectML on the discrete GPU.** When `torch-directml` is
     installed AND `torch_directml.device_count() >= 2`, we pick
     `device(1)` -- the discrete card. `device(0)` is the iGPU
     which on the user's box is slower than CPU (see
     memory/user_gpu_setup.md; benchmarked at 150 ms/iter vs
     CPU's 41 on a 2048² matmul). Picking the wrong device is
     worse than not using DML at all.
  2. **CUDA** if available (this is the cluster path).
  3. **CPU** as the universal fallback.

DML is preferred over CUDA only when both are present (i.e., on the
user's Windows box; the cluster has CUDA and no DML so it picks
CUDA correctly). The order matters for the local-dev workflow: the
user's RX 6600 via DML beats CPU by ~4× on the full training loop
(measured 2026-05-13: 3 iters × 2 games × 12 max-turns went from
185s CPU to 47s DML, both rollout and train_step ~4× faster).

**Training-safe as of 2026-05-13.** The scatter-backward bug
documented in memory/user_gpu_setup.md no longer trips (despite
torch-directml still being 0.2.5.dev240914 -- the surrounding
codebase changed enough since April that the offending code path
is gone). `aten::std.correction` still falls back to CPU silently
during advantage-normalization, but that's a slowdown, not a
crash. Re-evaluate if any new training-path op surfaces a hard
DML error.

Explicit `--device <spec>` overrides everything (returns
`torch.device(spec)` directly), so existing call sites that pass
`cuda` / `cpu` / `dml` / `cuda:0` keep working unchanged.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch


log = logging.getLogger("device_select")


def select_inference_device(prefer: Optional[str] = "auto"):
    """Resolve `prefer` to a `torch.device`. See module docstring for
    the auto-pick priority. Returns a `torch.device` (or
    DML-equivalent object that quacks like one).

    `prefer` values:
      - `"auto"` or `None`: DML (discrete) > CUDA > CPU.
      - `"dml"` or `"directml"`: force DirectML discrete card.
        Raises RuntimeError if torch-directml isn't installed.
      - `"cpu"`, `"cuda"`, `"cuda:0"`, etc.: pass-through to
        `torch.device(prefer)`. No magic.
    """
    # Explicit override: pass through to torch.device exactly.
    # Don't second-guess. The one exception: "dml" / "directml" is
    # not a built-in torch device string; route it through the same
    # auto path that resolves to torch_directml.device(1).
    if prefer and prefer not in ("auto", "dml", "directml"):
        return torch.device(prefer)

    force_dml = prefer in ("dml", "directml")

    # Try DirectML first (or only, if forced).
    try:
        import torch_directml  # type: ignore
        n = torch_directml.device_count()
        if n >= 2:
            # Discrete card is device(1); device(0) is the iGPU
            # (slower than CPU). See memory/user_gpu_setup.md.
            dev = torch_directml.device(1)
            log.info(f"selected DirectML device 1 "
                     f"(discrete; {n} DML devices visible)")
            return dev
        elif n >= 1:
            # Single DML device: probably an iGPU-only system.
            # Skip it -- CPU is faster.
            log.info(f"only iGPU DML device found ({n} visible); "
                     f"skipping (CPU is faster on the iGPU side)")
        else:
            log.info("torch_directml installed but no devices "
                     "visible; falling through")
    except ImportError:
        if force_dml:
            raise RuntimeError(
                "--device dml requested but torch_directml not "
                "installed. Install with: pip install torch-directml")
        log.debug("torch_directml not installed; checking CUDA")
    except Exception as e:
        if force_dml:
            raise RuntimeError(
                f"--device dml requested but DML init failed: {e}")
        log.debug(f"DML probe failed: {e}; falling through")

    if force_dml:
        raise RuntimeError(
            "--device dml requested but no usable DML device found")

    # CUDA path (cluster).
    if torch.cuda.is_available():
        log.info(f"selected CUDA device: "
                 f"{torch.cuda.get_device_name(0)}")
        return torch.device("cuda")

    # CPU fallback.
    log.info("selected CPU (no DML or CUDA available)")
    return torch.device("cpu")


def describe_device(dev) -> str:
    """One-line summary of a resolved device. Useful for the per-
    op startup log -- tells the operator which device the heavy
    work is about to run on."""
    s = str(dev)
    # torch_directml's privateuseone:N format -> human-friendly.
    if "privateuseone" in s:
        try:
            import torch_directml  # type: ignore
            idx = int(s.split(":")[-1])
            name = torch_directml.device_name(idx)
            return f"DirectML[{idx}] {name}"
        except Exception:
            return f"DirectML ({s})"
    if s.startswith("cuda"):
        try:
            return f"CUDA {torch.cuda.get_device_name(0)}"
        except Exception:
            return s
    return s
