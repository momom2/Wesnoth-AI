"""Device-level knobs for the TRAINING paths only.

Two settings that cost nothing to enable and were never set anywhere in
this project (audited 2026-09-13: no `allow_tf32` or
`set_float32_matmul_precision` call outside the virtualenv):

`enable_tf32`  lets fp32 matmuls run on the tensor cores. On Ampere and
               later that is the difference between the fp32 and the
               TF32 rate for every matmul the trainer does outside an
               autocast block, which on the imitation trainer's default
               path is all of them.
`adamw`        asks for the fused optimizer kernel when every parameter
               is on CUDA, which folds the whole step into one launch
               instead of one per tensor.

WHY TRAINING ONLY: TF32 rounds the mantissa to 10 bits. That is far
below the gaps this project reads off training curves (holdout cross
entropy differences of 0.3), but it is NOT acceptable on the paths that
must stay bit-exact or comparable across runs: the simulator, the
replay corpus sweeps, and eval matches whose results are the only
strength verdict. Those paths never call this module; the entry points
that do are the trainers.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional

import torch

log = logging.getLogger("train_perf")

_LOGGED = False


def enable_tf32(enable: bool = True, *, quiet: bool = False) -> Dict[str, bool]:
    """Turn TF32 on (or off) for matmul and cuDNN. Returns the settings
    as they were before, so a caller can restore them. A no-op without
    CUDA."""
    global _LOGGED
    before = {
        "matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn": bool(torch.backends.cudnn.allow_tf32),
    }
    if not torch.cuda.is_available():
        return before
    torch.backends.cuda.matmul.allow_tf32 = bool(enable)
    torch.backends.cudnn.allow_tf32 = bool(enable)
    if enable and not _LOGGED and not quiet:
        _LOGGED = True
        log.info("TF32 on for fp32 matmuls (training path only; "
                 "the sim, the corpus sweeps and eval are untouched)")
    return before


def adamw(params: Iterable[torch.nn.Parameter], *, lr: float,
          weight_decay: float = 0.0, fused: Optional[bool] = None,
          **kw) -> torch.optim.AdamW:
    """`torch.optim.AdamW` with the fused kernel when it applies: every
    parameter on CUDA and floating point. `fused=False` forces the
    reference path; `fused=True` asks for it regardless (and raises the
    way torch would). A torch too old for the argument falls back."""
    params = list(params)
    if fused is None:
        fused = bool(params) and all(
            p.is_cuda and p.is_floating_point() for p in params)
    if not fused:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kw)
    try:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay,
                                 fused=True, **kw)
    except (TypeError, RuntimeError) as exc:      # old torch, or an unsupported dtype
        log.debug("fused AdamW unavailable (%s); using the reference step", exc)
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kw)


__all__ = ["enable_tf32", "adamw"]
