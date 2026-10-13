"""What the sequence trainer (tools/sequence_train.py) reads back of a
window: one transfer from the device after the backward pass.

The forward loop keeps everything it will log on the device: each time
step's per-sample loss terms and belief losses, and each own step's
soft-target losses. After the backward pass and the clip, `read_window`
copies them to the host with the step's loss, gradient norms and the
finiteness of the loss, the norm and every carried memory, all in one
buffer: the only wait for the device in a window. The host then decides
the update and sums the log exactly as it did when it read each value on
its own (`window_sums`): the same float32 values reach the same host
operations in the same order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch


@dataclass
class WindowRead:
    """A window's values on the host."""
    finite: bool                        # the loss and the gradient norm are finite
    loss: float
    norm: float
    group_norms: Optional[List[float]]  # GradientGroups.groups order; None without gradients
    memory_finite: List[bool]           # per carried memory, in the order given
    logs: List[torch.Tensor]            # per time step: its [5, B] loss terms (ImitationLossParts.log_tensor)
    belief: List[float]                 # per time step: its summed belief loss
    own: List[float]                    # per own time step: its summed soft-target loss


def read_window(loss: torch.Tensor, norm: torch.Tensor, group_norms: Optional[torch.Tensor],
                memories: Sequence[torch.Tensor], logs: Sequence[Tuple[object, torch.Tensor, torch.Tensor]],
                own_sums: Sequence[torch.Tensor], device: torch.device) -> WindowRead:
    """The window's values in one device-to-host copy (module docstring).
    `logs` holds per time step (targets, log tensor, belief losses);
    `own_sums` each own time step's summed soft-target loss, on the device."""
    parts = [torch.isfinite(loss.detach()).reshape(1), torch.isfinite(norm).reshape(1)]
    parts += [torch.isfinite(m).all().reshape(1) for m in memories]
    parts += [loss.detach().reshape(1), norm.reshape(1)]
    if group_norms is not None:
        parts.append(group_norms.reshape(-1))
    parts += [log_t.reshape(-1) for _, log_t, _ in logs]
    parts += [bl.sum().reshape(1) for _, _, bl in logs]
    parts += [s.reshape(1) for s in own_sums]
    flat = torch.cat([p.to(device=device, dtype=torch.float32) for p in parts]).cpu()
    n_groups = 0 if group_norms is None else group_norms.numel()
    sizes = ([2, len(memories), 1, 1, n_groups] + [log_t.numel() for _, log_t, _ in logs]
             + [1] * len(logs) + [1] * len(own_sums))
    chunks = list(torch.split(flat, sizes))
    flags, mem_ok, loss_v, norm_v, groups = chunks[:5]
    rest = chunks[5:]
    log_chunks, rest = rest[:len(logs)], rest[len(logs):]
    belief, own = rest[:len(logs)], rest[len(logs):]
    return WindowRead(
        finite=bool(flags[0]) and bool(flags[1]), loss=float(loss_v[0]), norm=float(norm_v[0]),
        group_norms=None if group_norms is None else groups.tolist(),
        memory_finite=[bool(x) for x in mem_ok],
        logs=[c.view(log_t.shape).clone() for c, (_, log_t, _) in zip(log_chunks, logs)],
        belief=[float(b[0]) for b in belief], own=[float(o[0]) for o in own])


def window_sums(read: WindowRead, logs: Sequence[Tuple[object, torch.Tensor, torch.Tensor]],
                own_counts: Sequence[int], with_own: bool) -> Dict:
    """The window's logged sums: per head the summed loss over the
    positions it trains and their count (the policy over the positions of
    positive policy weight, the value over the value positions, the belief
    over every position, the own loss over the own rows)."""
    sums = {"policy": 0.0, "policy_n": 0, "value": 0.0, "value_n": 0, "belief": 0.0, "belief_n": 0}
    if with_own:
        sums.update(own=0.0, own_n=0)
        for value, n in zip(read.own, own_counts):
            sums["own"] += value
            sums["own_n"] += int(n)
    for (targets, _, bl), vals, belief in zip(logs, read.logs, read.belief):
        ok_policy = torch.from_numpy(targets.policy_w) > 0
        sums["policy"] += float(vals[:4].sum(0)[ok_policy].sum())
        sums["policy_n"] += int(ok_policy.sum())
        ok_value = torch.from_numpy(targets.ok["value"])
        sums["value"] += float(vals[4][ok_value].sum())
        sums["value_n"] += int(ok_value.sum())
        sums["belief"] += belief
        sums["belief_n"] += int(bl.numel())
    return sums
