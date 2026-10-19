"""Muon: momentum orthogonalized by Newton-Schulz, for a network's hidden
2-D weight matrices (K. Jordan, Y. Jin, V. Boza, J. You, F. Cesista,
L. Newhouse and J. Bernstein, "Muon: An optimizer for hidden layers in neural
networks", 2024, https://kellerjordan.github.io/posts/muon/), in the form of
J. Liu et al., "Muon is Scalable for LLM Training" (Moonshot AI, 2025,
arXiv 2502.16982), whose update reuses AdamW's learning rate and weight
decay:

    M_t = mu M_{t-1} + G_t
    O_t = NewtonSchulz5(mu M_t + G_t)                      (Nesterov)
    W_t = W_{t-1} - lr (0.2 sqrt(max(A, B)) O_t + lambda W_{t-1})

for an A x B matrix. 0.2 sqrt(max(A, B)) gives the update the RMS of an
AdamW update (Liu et al.: "usually around 0.2 to 0.4"), and the decoupled
weight decay is theirs (Jordan's original has none). mu = 0.95 is both
papers' value. Newton-Schulz runs Jordan's quintic iteration, 5 steps in
bfloat16 with his coefficients, which leaves the singular values in a band
around 1 (about 0.7 to 1.3) instead of exactly 1.

A parameter group's `blocks` splits each of its matrices into that many
blocks of rows, each orthogonalized and scaled as a matrix of its own: an
attention in-projection is the query, key and value matrices stacked, and
Muon works better on them separately (Jordan, ibid.). Matrices of one block
shape are orthogonalized together, in one batched iteration.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Tuple

import torch

# Jordan's quintic coefficients (a, b, c) and step count.
NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
NS_STEPS = 5
MOMENTUM = 0.95
# Liu et al.'s factor: an update of RMS 0.2 per unit of sqrt(max(A, B)).
RMS_MATCH = 0.2


def newton_schulz(G: torch.Tensor, steps: int = NS_STEPS) -> torch.Tensor:
    """[..., A, B] in bfloat16: each matrix's orthogonalization, U V^T up to
    singular values in a band around 1 (Jordan's quintic iteration). The
    iteration runs on the wide orientation (A <= B), normalized to a
    Frobenius norm of 1 so the spectral norm starts at most 1."""
    a, b, c = NS_COEFFICIENTS
    X = G.bfloat16()
    tall = G.size(-2) > G.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.mT if tall else X


class Muon(torch.optim.Optimizer):
    """Muon over 2-D parameters (module docstring). Group options: `lr`,
    `weight_decay`, `momentum`, `nesterov`, `ns_steps` and `blocks` (the
    row blocks each matrix splits into)."""

    def __init__(self, params, lr: float, weight_decay: float = 0.0, momentum: float = MOMENTUM,
                 nesterov: bool = True, ns_steps: int = NS_STEPS, blocks: int = 1):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, blocks=blocks))
        for group in self.param_groups:
            for p in group["params"]:
                if p.dim() != 2 or p.size(0) % group["blocks"]:
                    raise ValueError(f"Muon takes matrices whose rows split into {group['blocks']} blocks, "
                                     f"got {tuple(p.shape)}")

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("Muon.step takes no closure")
        pending: Dict[Tuple, List] = defaultdict(list)
        for group in self.param_groups:
            mu = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(p.grad)
                update = p.grad.add(buf, alpha=mu) if group["nesterov"] else buf
                rows = p.size(0) // group["blocks"]
                key = (rows, p.size(1), group["ns_steps"], p.device)
                pending[key].append((p, group, update.reshape(group["blocks"], rows, p.size(1))))
        for (rows, cols, steps, _), items in pending.items():
            ortho = newton_schulz(torch.cat([u for _, _, u in items]), steps)
            scale = RMS_MATCH * math.sqrt(max(rows, cols))
            at = 0
            for p, group, u in items:
                o = ortho[at:at + u.size(0)].reshape(p.shape)
                at += u.size(0)
                p.mul_(1.0 - group["lr"] * group["weight_decay"])
                p.add_(o.to(p.dtype), alpha=-group["lr"] * scale)
        return None
