"""Policy-gradient trainer (REINFORCE + value baseline + entropy bonus).

On-policy, no replay buffer. A batch of full trajectories goes through
one gradient update and is discarded. Simplest thing that works;
scaffolding for heavier algorithms (PPO, etc.) can replace Trainer.step
without changing the policy or reward modules.

Pieces:
  - Transition: one action's records (log_prob, value, entropy),
    plus the reward Python assigns afterward and a terminal flag.
  - TrainerConfig: all hyperparameters. Override by constructing with
    different values — there's no global state to worry about.
  - Trainer: holds the optimizer and runs one gradient step per
    step() call.
  - TrainStats: metrics for logging.

Returns are computed backward through each trajectory:

    G_t = r_t + γ · G_{t+1}    (if transition t is non-terminal)
    G_T = r_T                  (terminal)

Advantages are (returns - V(s)) normalized within the batch. Policy
loss maximizes log π(a|s) · advantage.detach(); value loss MSEs V(s)
against returns; entropy bonus encourages exploration.

Each side's trajectory in a self-play game is a separate element of
the batch — they just happen to share weights. Their rewards are
adversarial (one side's gold_killed_delta positive = other's
our_gold_lost positive), so the trainer naturally balances them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Transition:
    """One action's grad-tracked quantities plus its delivered reward.

    The tensors come from sample_action() at rollout time. reward and
    done are set by the game-manager loop after the subsequent state
    arrives (reward is a scalar float; gradients don't flow through it).
    """
    log_prob: torch.Tensor    # scalar, grad-enabled
    value:    torch.Tensor    # scalar, grad-enabled
    entropy:  torch.Tensor    # scalar, grad-enabled (for bonus term)
    reward:   float = 0.0
    done:     bool  = False


@dataclass
class TrainerConfig:
    """Hyperparameters. Edit defaults here OR override per-run."""
    learning_rate:  float = 1e-4
    weight_decay:   float = 1e-4
    gamma:          float = 0.99     # discount factor
    value_coef:     float = 0.5      # weight of value loss in total loss
    entropy_coef:   float = 0.01     # weight of -entropy in total loss
    grad_clip:      float = 1.0      # max gradient norm
    # Normalize advantages within a batch. Usually helps stability
    # when batches are reasonably sized; set False for very small
    # batches.
    normalize_advantages: bool = True


@dataclass
class TrainStats:
    policy_loss:   float = 0.0
    value_loss:    float = 0.0
    entropy:       float = 0.0
    total_loss:    float = 0.0
    grad_norm:     float = 0.0
    mean_return:   float = 0.0
    n_transitions: int   = 0
    n_trajectories: int  = 0


class Trainer:
    """One-gradient-step-per-call REINFORCE with a value baseline."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig = None,
        device=None,
    ):
        self.model  = model
        self.config = config if config is not None else TrainerConfig()
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def step(self, trajectories: List[List[Transition]]) -> TrainStats:
        """Apply one gradient update from the given batch of trajectories.

        Returns metrics; a zero TrainStats on empty input (no crash).
        """
        n_traj = sum(1 for t in trajectories if t)
        if n_traj == 0:
            return TrainStats()

        log_probs_all: List[torch.Tensor] = []
        values_all:    List[torch.Tensor] = []
        entropies_all: List[torch.Tensor] = []
        returns_all:   List[float]        = []

        for traj in trajectories:
            if not traj:
                continue
            returns = _compute_returns(traj, self.config.gamma)
            for t, G in zip(traj, returns):
                log_probs_all.append(t.log_prob)
                values_all.append(t.value.squeeze())
                entropies_all.append(t.entropy)
                returns_all.append(G)

        if not log_probs_all:
            return TrainStats()

        # Collect device from any live tensor — model might be on DML/CPU.
        dev = log_probs_all[0].device

        log_probs = torch.stack(log_probs_all)                       # [N]
        values    = torch.stack(values_all)                          # [N]
        entropies = torch.stack(entropies_all)                       # [N]
        returns   = torch.tensor(returns_all, device=dev,
                                 dtype=values.dtype)                 # [N]

        advantages = returns - values.detach()
        if self.config.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss  = -(log_probs * advantages).mean()
        value_loss   = F.mse_loss(values, returns)
        entropy_term = entropies.mean()

        total_loss = (
            policy_loss
            + self.config.value_coef   * value_loss
            - self.config.entropy_coef * entropy_term
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip
        )
        self.optimizer.step()

        return TrainStats(
            policy_loss   = float(policy_loss.item()),
            value_loss    = float(value_loss.item()),
            entropy       = float(entropy_term.item()),
            total_loss    = float(total_loss.item()),
            grad_norm     = float(grad_norm) if isinstance(grad_norm, float)
                            else float(grad_norm.item()),
            mean_return   = float(returns.mean().item()),
            n_transitions = int(len(log_probs_all)),
            n_trajectories= int(n_traj),
        )


def _compute_returns(traj: List[Transition], gamma: float) -> List[float]:
    """Discounted returns, iterated backward. List-of-floats output."""
    G = 0.0
    out: List[float] = []
    for t in reversed(traj):
        # Terminal transition: zero out the propagated future return.
        continuation = 0.0 if t.done else 1.0
        G = t.reward + gamma * G * continuation
        out.append(G)
    out.reverse()
    return out
