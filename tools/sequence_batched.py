"""The sequence trainer's forward over a window (tools/sequence_train.py):
time step by time step, or batched over the window (`--window-batched`).

A time step's forward is a few hundred small kernels and as many Python
dispatches, and on the packed trunk the step is host-bound (the GPU idle 87
to 90% of a window, docs/trainer_speed_20261013.md). Only the trunk and the
memory's write must run step by step: the memory a step reads is the one the
previous step wrote. `--window-batched` stages and embeds the window's 512
positions at once, runs each step's packed trunk and memory write on its
rows of that embedding, and runs the heads and the losses once over every
step's trunk outputs: the embedding's and the heads' and losses' kernels
are launched once a window instead of once a time step.

Other arithmetic than the stepped forward: the embedding's projections and
the heads' are matrix products over more rows (cuBLAS may tile them
otherwise), each embedding table's gradient is summed over the window in
one kernel instead of step by step, the softmaxes run over rows padded to
the window's widths, and the loss is summed in another order. Rounding,
nothing more: tests/test_sequence_launches.py checks the losses and
gradients against the stepped forward's on the CPU, and the box's gradient
check does on the GPU. The packed trunk only (the production trunk).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
from torch.utils.checkpoint import checkpoint

from tools.sequence_timing import trunk_tokens
from wesnoth_ai.imitation_loss import imitation_loss_parts
from wesnoth_ai.model_output import ActorKind
from wesnoth_ai.own_loss import own_policy_loss
from wesnoth_ai.packed_trunk import window_head_index
from wesnoth_ai.sequence_loss import belief_loss

# The streams before the memory's tokens in a packed segment, in order.
_BEFORE_MEMORY = ("hex", "unit", "recruit", "sighting")


@dataclass
class WindowForward:
    """A window's forward: its summed loss and what the read-back and the
    log take of it (tools/sequence_window.py), and its timing counts."""
    total: torch.Tensor
    logs: List = field(default_factory=list)          # (targets, [5, B] loss terms, belief losses) per batch
    own_sums: List = field(default_factory=list)
    own_counts: List[int] = field(default_factory=list)
    n_positions: int = 0
    tokens_real: int = 0
    tokens_padded: int = 0
    n_kept: int = 0
    kept_tokens: int = 0
    kept_bytes: int = 0


class WindowForwards:
    """The two forwards (a part of `sequence_train.Trainer`)."""

    def _planned_kept(self, window) -> int:
        """The time steps whose activations the window keeps; none in a
        forward-only window."""
        return self._kept_steps(window) if torch.is_grad_enabled() else 0

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=self.autocast, enabled=self.autocast is not None)

    def _keep(self, fw: WindowForward, t: int, n_steps: int, fn, inputs, tokens: int):
        """`fn(*inputs)` with its activations kept when step `t` is among the
        window's last `fw.n_kept`, else recomputed in the backward pass; run
        as it is in a forward-only window (`--train-every`), which keeps
        nothing."""
        if not torch.is_grad_enabled():
            return fn(*inputs)
        if t >= n_steps - fw.n_kept:
            before = self.clock.memory_allocated()
            outputs = fn(*inputs)
            fw.kept_bytes += self.clock.memory_allocated() - before
            fw.kept_tokens += tokens
            return outputs
        return checkpoint(fn, *inputs, use_reentrant=False)

    def _stepped_forward(self, window, router) -> WindowForward:
        """Each time step's forward and losses in turn."""
        fw = WindowForward(total=torch.zeros((), device=self.device), n_kept=self._planned_kept(window))
        for t, steps in enumerate(window):
            mems = router.batch(steps)
            x = self.inputs.stage(steps)
            staged, targets, b_target, b_mask, own = x.staged, x.targets, x.belief_target, x.belief_mask, x.own
            index = self._trunk_index(steps, staged)
            real, padded = trunk_tokens(staged.sizes, staged.sighting_counts, [s.k for s in steps])
            fw.tokens_real, fw.tokens_padded = fw.tokens_real + real, fw.tokens_padded + padded
            fn, inputs = ((self._step_forward, (staged, mems, targets, b_target, b_mask, index)) if own is None else
                          (self._step_forward_own, (staged, mems, targets, b_target, b_mask, own, index)))
            outputs = self._keep(fw, t, len(window), fn, inputs, real if self.packed else padded)
            pv, bsum, mem_padded, log_t, bl = outputs[:5]
            if own is not None:
                fw.own_sums.append(outputs[5].sum())
                fw.own_counts.append(outputs[5].numel())
            fw.total = fw.total + pv + self.args.belief_weight * bsum
            router.wrote(steps, mem_padded)
            fw.logs.append((targets, log_t, bl))
            fw.n_positions += len(steps)
        return fw

    def _step_trunk(self, before, after, mems, index):
        """One time step of the batched forward: its packed trunk and its
        memory's write, on its rows of the window's embedding."""
        with self._autocast():
            return self.model.packed_step_trunk(before, after, mems, index)

    def _batched_forward(self, window, router) -> WindowForward:
        """The window staged and embedded at once, the trunk and the memory
        step by step, the heads and the losses at once (module docstring)."""
        fw = WindowForward(total=torch.zeros((), device=self.device), n_kept=self._planned_kept(window))
        rows = [s for steps in window for s in steps]
        x = self.inputs.stage(rows)
        sizes, sightings = x.staged.sizes, x.staged.sighting_counts
        bounds, b0 = [], 0
        for steps in window:
            bounds.append((b0, b0 + len(steps)))
            b0 += len(steps)
        per_step = {"hex": lambda u, r, h, s: h, "unit": lambda u, r, h, s: u, "recruit": lambda u, r, h, s: r,
                    "sighting": lambda u, r, h, s: s}
        with self._autocast():
            embedded = self.encoder.embed_streams(x.staged)
        S = sightings if sightings is not None else [0] * len(sizes)
        split = {name: torch.split(embedded[name], [sum(per_step[name](*sizes[b], S[b]) for b in range(lo, hi))
                                                    for lo, hi in bounds])
                 for name in _BEFORE_MEMORY if name in embedded}
        split["global"] = torch.split(embedded["global"], [hi - lo for lo, hi in bounds])
        indices, outputs = [], []
        for (lo, hi), steps in zip(bounds, window):
            indices.append(self.model.trunk_index(sizes[lo:hi], None if sightings is None else sightings[lo:hi],
                                                  [s.k for s in steps], self.device, packed=True))
        heads = window_head_index([ix.layout for ix in indices], sizes, ActorKind.END_TURN, self.device)
        for t, ((lo, hi), steps) in enumerate(zip(bounds, window)):
            mems = router.batch(steps)
            real, padded = trunk_tokens(sizes[lo:hi], None if sightings is None else sightings[lo:hi],
                                        [s.k for s in steps])
            fw.tokens_real, fw.tokens_padded = fw.tokens_real + real, fw.tokens_padded + padded
            before = [split[name][t] for name in _BEFORE_MEMORY if name in split]
            after = [split["global"][t], embedded["end_turn"]]
            out_t, mem_padded = self._keep(fw, t, len(window), self._step_trunk,
                                           (before, after, mems, indices[t]), real)
            router.wrote(steps, mem_padded)
            outputs.append(out_t)
            fw.n_positions += len(steps)
        with self._autocast():
            out = self.model.window_heads(torch.cat(outputs), heads)
        out = out.float32()
        parts = imitation_loss_parts(out, x.targets)
        bl = belief_loss(out.belief_logits, x.belief_target, x.belief_mask)
        pv = parts.total
        if x.own is not None:
            soft = own_policy_loss(out, x.own)
            pv = pv + self.args.own_policy_weight * soft.sum()
            fw.own_sums.append(soft.detach().sum())
            fw.own_counts.append(soft.numel())
        fw.total = pv + self.args.belief_weight * bl.sum()
        fw.logs.append((x.targets, parts.log_tensor(), bl.detach()))
        return fw
