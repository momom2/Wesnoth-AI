"""The fp16 gradient pass's loss scale (tools/pg_pass.py `--fp16`).

Under an fp16 autocast the backward's activation gradients are fp16, whose
normal range stops at 6.1e-5 and whose subnormals at 6.0e-8: a window's
gradient flowing through the trunk can underflow there. The pass multiplies
each window's loss by a power of two before its backward and divides the
parameter gradients by it after, exactly (Micikevicius et al., "Mixed
Precision Training", 2018). The scale is dynamic, by torch.amp.GradScaler's
rule (torch/amp/grad_scaler.py: halve on a non-finite gradient, double
after `growth_interval` finite ones), counted in windows. Where GradScaler
skips an optimizer step, the pass has none: every window is part of the
sum, so a window whose scaled gradient overflowed is run again at half the
scale (`GradientPass._window_gradient`), its readings undone first. A
window that still overflows at MIN_SCALE overflows in its forward or at
fp16's range itself: the pass stops there (`FloatingPointError`).

The start: a window's loss sums up to 512 decisions' log-probabilities
(32 streams of 16) weighted by R of order 1, not a mean, so its gradients
are larger than a mean loss's, and each overflow costs the window's forward
and backward again. INITIAL_SCALE is set where it was measured to hold: on
16 of the 2026-10-28 pilot's games, 27 windows ran at 2**10 without an
overflow (2026-10-29, training/metrics/hip_backward_20261029/README.md).
GradScaler's 2**16 start was not tried.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

INITIAL_SCALE = 2.0 ** 10
GROWTH_INTERVAL = 200        # windows: about a ninth of a 20,000-game pass's split (2,500 games, ~1,850 windows)
MIN_SCALE = 2.0 ** -8


@dataclass
class LossScale:
    scale: float = INITIAL_SCALE
    growth_interval: int = GROWTH_INTERVAL
    min_scale: float = MIN_SCALE
    finite_run: int = 0          # finite windows since the last change
    overflows: int = 0           # windows run again
    growths: int = 0
    smallest: Optional[float] = None     # the smallest scale a window ran at (None: the starting scale)

    def __post_init__(self) -> None:
        if self.smallest is None:
            self.smallest = self.scale

    def overflowed(self) -> None:
        """A window's scaled gradient was not finite: halve the scale (the
        window is then run again), or stop at the floor."""
        if self.scale <= self.min_scale:
            raise FloatingPointError(f"a window's gradient is not finite at the loss scale {self.scale:g}: "
                                     f"the fp16 forward or backward overflows (no scale avoids it)")
        self.scale /= 2.0
        self.smallest = min(self.smallest, self.scale)
        self.finite_run = 0
        self.overflows += 1

    def finite(self) -> None:
        """A window's gradient was finite: double the scale after
        `growth_interval` of them in a row."""
        self.finite_run += 1
        if self.finite_run >= self.growth_interval:
            self.scale *= 2.0
            self.finite_run = 0
            self.growths += 1

    def summary(self) -> dict:
        return asdict(self)
