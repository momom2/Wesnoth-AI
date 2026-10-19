"""An exponential moving average of a network's trained weights (Polyak
averaging in its exponential form), read in place of the weights for an
evaluation (tools/sequence_train.py `--weight-ema`).

After each applied optimizer step t (0 first), every parameter's average
moves toward the parameter by 1 - d_t, with

    d_t = min(decay, (1 + t) / (10 + t))

the warm-up of TensorFlow's `tf.train.ExponentialMovingAverage` with
`num_updates`: the first steps weigh more, so the average does not carry
the initialization (at decay 0.998 over 2,000 steps, a plain average would
keep 2% of it). The average is float32 and changes no weight: `applied()`
puts it into the parameters for the length of a block and restores the
training weights exactly after it.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Sequence, Tuple

import torch


class WeightEMA:
    """The average of `named` (namespaced names and parameters, the
    trainer's order)."""

    def __init__(self, named: Sequence[Tuple[str, torch.nn.Parameter]], decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"an average's decay is in (0, 1), got {decay}")
        self.decay = float(decay)
        self.names = [n for n, _ in named]
        self.params = [p for _, p in named]
        self.updates = 0
        self.average = [p.detach().float().clone() for p in self.params]

    def decay_at(self, t: int) -> float:
        return min(self.decay, (1.0 + t) / (10.0 + t))

    @torch.no_grad()
    def update(self) -> None:
        """After an applied optimizer step."""
        torch._foreach_lerp_(self.average, [p.detach().float() for p in self.params],
                             1.0 - self.decay_at(self.updates))
        self.updates += 1

    @torch.no_grad()
    def reset(self) -> None:
        """Start again from the current weights (a new pass from another's)."""
        self.average = [p.detach().float().clone() for p in self.params]
        self.updates = 0

    @contextmanager
    def applied(self):
        """The average in the parameters inside the block, the training
        weights restored exactly after it."""
        with torch.no_grad():
            kept = [p.detach().clone() for p in self.params]
            for p, a in zip(self.params, self.average):
                p.copy_(a)
        try:
            yield
        finally:
            with torch.no_grad():
                for p, k in zip(self.params, kept):
                    p.copy_(k)

    def module_states(self, modules: Dict[str, torch.nn.Module]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Each module's state dict with the average in place of its trained
        parameters (its other entries as they are), by namespace
        ({"model": model, "encoder": encoder}): loadable as weights."""
        by_name = dict(zip(self.names, self.average))
        out = {}
        for space, module in modules.items():
            state = dict(module.state_dict())
            for key in state:
                if f"{space}.{key}" in by_name:
                    state[key] = by_name[f"{space}.{key}"]
            out[space] = state
        return out

    @torch.no_grad()
    def load_module_states(self, states: Dict[str, Dict[str, torch.Tensor]], updates: int) -> None:
        """The average saved by `module_states`, and its step count."""
        for i, name in enumerate(self.names):
            space, key = name.split(".", 1)
            self.average[i].copy_(states[space][key])
        self.updates = int(updates)
