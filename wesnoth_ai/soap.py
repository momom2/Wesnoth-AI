"""SOAP: Adam in the eigenbasis of Shampoo's preconditioner, for 2-D
weight matrices (N. Vyas, D. Morwani, R. Zhao, M. Kwun, I. Shapira,
D. Brandfonbrener, L. Janson and S. Kakade, "SOAP: Improving and Stabilizing
Shampoo using Adam", 2024, arXiv 2409.11321), as the authors' reference
implementation (github.com/nikhilvyas/SOAP, soap.py) runs it on a matrix W
of A x B with gradient G:

    L <- lerp(L, G G^T, 1 - beta_s)      R <- lerp(R, G^T G, 1 - beta_s)
    P  = Q_L^T G Q_R                      (the gradient in the eigenbasis)
    M <- beta1 M + (1 - beta1) P          V <- beta2 V + (1 - beta2) P*P
    W <- W - lr sqrt(1 - beta2^t) / (1 - beta1^t) * Q_L (M / (sqrt(V) + eps)) Q_R^T
    W <- W - lr lambda W                  (decoupled weight decay, after the step)

M and V live in the eigenbasis. The first gradient only builds L, R and
their eigenbases (eigh) and moves no weight, so no gradient is projected on a
basis it built. Every `precondition_every` steps the bases are refreshed by
one power iteration and a QR decomposition, the estimated eigenvalues
sorted first (V's rows and columns follow the sort); M is carried into the
new basis through the original one. The reference code rotates M out of and
back into the unchanged basis at every other step, an identity up to
rounding, which this one skips. beta_s defaults to beta2 (the reference's
`shampoo_beta=-1`). Defaults are the reference's: betas (0.95, 0.95),
eps 1e-8, a refresh every 10 steps.

Cost of an A x B matrix: state A^2 + B^2 for L and R, as much again for
their bases, and 2 A B for M and V, all float32; per step about
3 (A^2 B + A B^2) multiply-adds for the two projections and the
statistics, and every `precondition_every` steps two A x A and B x B QR
decompositions and their power iterations. On the recipe's trunk (8 layers
of a 1152 x 384 in-projection, a 384 x 384 out-projection and the 1536 x 384
and 384 x 1536 feed-forward matrices) that is 137M floats of state, 0.55 GB,
where AdamW holds 0.12 GB for the whole network, and 147 GFLOP a step.
"""
from __future__ import annotations

import torch

BETAS = (0.95, 0.95)
EPS = 1e-8
PRECONDITION_EVERY = 10


def _eigenbasis(S: torch.Tensor) -> torch.Tensor:
    """The eigenvectors of the symmetric S, by decreasing eigenvalue
    (float32, float64 when the float32 decomposition fails)."""
    eye = torch.eye(S.size(0), device=S.device, dtype=torch.float32)
    try:
        _, Q = torch.linalg.eigh(S.float() + 1e-30 * eye)
    except torch.linalg.LinAlgError:
        _, Q = torch.linalg.eigh(S.double() + 1e-30 * eye.double())
        Q = Q.float()
    return torch.flip(Q, [1]).to(S.dtype)


def _refreshed(S: torch.Tensor, Q: torch.Tensor):
    """One power iteration on the basis Q of S, its columns sorted first by
    the eigenvalue each estimates: the new basis and the sort."""
    estimates = torch.diag(Q.T @ S @ Q)
    order = torch.argsort(estimates, descending=True)
    Q_new, _ = torch.linalg.qr(S @ Q[:, order])
    return Q_new, order


class SOAP(torch.optim.Optimizer):
    """SOAP over 2-D parameters (module docstring). Group options: `lr`,
    `betas`, `eps`, `weight_decay`, `precondition_every`, `shampoo_beta`
    (None: betas[1])."""

    def __init__(self, params, lr: float, betas=BETAS, eps: float = EPS, weight_decay: float = 0.0,
                 precondition_every: int = PRECONDITION_EVERY, shampoo_beta=None):
        super().__init__(params, dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay,
                                      precondition_every=int(precondition_every), shampoo_beta=shampoo_beta))
        for group in self.param_groups:
            for p in group["params"]:
                if p.dim() != 2:
                    raise ValueError(f"SOAP here takes matrices, got {tuple(p.shape)}")

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("SOAP.step takes no closure")
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            beta_s = beta2 if group["shampoo_beta"] is None else group["shampoo_beta"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                G = p.grad
                if "step" not in state:
                    self._start(state, G, beta_s)
                    continue
                P = state["q_left"].T @ G @ state["q_right"]
                state["step"] += 1
                t = state["step"]
                state["exp_avg"].mul_(beta1).add_(P, alpha=1.0 - beta1)
                state["exp_avg_sq_eigen"].mul_(beta2).add_(P.square(), alpha=1.0 - beta2)
                denom = state["exp_avg_sq_eigen"].sqrt().add_(group["eps"])
                step_size = group["lr"] * (1.0 - beta2 ** t) ** 0.5 / (1.0 - beta1 ** t)
                update = state["q_left"] @ (state["exp_avg"] / denom) @ state["q_right"].T
                p.add_(update, alpha=-step_size)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])
                self._accumulate(state, G, beta_s)
                if t % group["precondition_every"] == 0:
                    self._refresh(state)
        return None

    @staticmethod
    def _start(state, G: torch.Tensor, beta_s: float) -> None:
        """The state at a parameter's first gradient: the statistics and
        their eigenbases built from it, no step taken."""
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(G)
        state["exp_avg_sq_eigen"] = torch.zeros_like(G)
        state["gg_left"] = torch.zeros(G.size(0), G.size(0), device=G.device, dtype=G.dtype)
        state["gg_right"] = torch.zeros(G.size(1), G.size(1), device=G.device, dtype=G.dtype)
        SOAP._accumulate(state, G, beta_s)
        state["q_left"] = _eigenbasis(state["gg_left"])
        state["q_right"] = _eigenbasis(state["gg_right"])

    @staticmethod
    def _accumulate(state, G: torch.Tensor, beta_s: float) -> None:
        state["gg_left"].lerp_(G @ G.T, 1.0 - beta_s)
        state["gg_right"].lerp_(G.T @ G, 1.0 - beta_s)

    @staticmethod
    def _refresh(state) -> None:
        """New eigenbases; V's rows and columns follow the sort of the old
        bases' columns, M is carried through the original basis."""
        original = state["q_left"] @ state["exp_avg"] @ state["q_right"].T
        q_left, rows = _refreshed(state["gg_left"].float(), state["q_left"].float())
        q_right, cols = _refreshed(state["gg_right"].float(), state["q_right"].float())
        state["exp_avg_sq_eigen"] = state["exp_avg_sq_eigen"][rows][:, cols].contiguous()
        state["q_left"], state["q_right"] = q_left.to(original.dtype), q_right.to(original.dtype)
        state["exp_avg"] = state["q_left"].T @ original @ state["q_right"]

