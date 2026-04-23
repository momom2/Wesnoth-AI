"""Policy-gradient trainer (REINFORCE + value baseline + entropy bonus).

Important architectural note: this trainer RE-FORWARDS the model at
training time rather than holding on to forward graphs across an
entire rollout. The previous approach (store log_prob/value tensors
with retained graph at sample time, backward through all of them at
train time) broke down at scale: four parallel games × ~200 actions
per game × ~8MB of retained activations per forward = several GB of
RAM pinned, which on a 16GB-ish machine triggered swap and the whole
pipeline froze. Re-forwarding keeps peak memory at O(1 model) and
lets us chunk through arbitrarily long trajectories.

Cost: each action is forwarded twice — once during rollout (no-grad,
cheap), once during training (with grads, building only this one
transition's graph). Total compute roughly doubles, but RAM usage
drops from GB to MB, which was the actual bottleneck.

Pieces:
  - Transition: indices + a reference to the GameState we sampled
    from. No tensors, no grads.
  - TrainerConfig: dataclass of hyperparameters.
  - Trainer.step(trajectories): for each transition, re-forwards
    (encoder + model), computes log_prob / entropy / value under
    current weights, accumulates loss, backprops, optimizer.step().
  - TrainStats: metrics for logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from action_sampler import reforward_logprob_entropy
from classes import GameState


@dataclass
class Transition:
    """What the policy stashes per select_action call.

    No grad tensors — those are built fresh from game_state when the
    trainer re-forwards. Keeping the raw GameState is cheap (a tree of
    Python dataclasses); keeping activations is not.
    """
    game_state: GameState
    actor_idx:  int
    target_idx: Optional[int] = None
    weapon_idx: Optional[int] = None
    # Filled in by game_manager after the next state arrives:
    reward:     float = 0.0
    done:       bool  = False


@dataclass
class TrainerConfig:
    """Hyperparameters; override by constructing with different values."""
    learning_rate:        float = 1e-4
    weight_decay:         float = 1e-4
    gamma:                float = 0.99
    value_coef:           float = 0.5
    entropy_coef:         float = 0.01
    grad_clip:            float = 1.0
    normalize_advantages: bool  = True
    # Cap transitions processed per train_step. Nothing's dropped from
    # trajectories themselves; just trims the backward-graph size.
    # Tune up if freezes remain; down if training is too expensive.
    max_transitions_per_step: int = 512


@dataclass
class TrainStats:
    policy_loss:    float = 0.0
    value_loss:     float = 0.0
    entropy:        float = 0.0
    total_loss:     float = 0.0
    grad_norm:      float = 0.0
    mean_return:    float = 0.0
    n_transitions:  int   = 0
    n_trajectories: int   = 0


class Trainer:
    """One-gradient-step-per-call REINFORCE with a value baseline.

    Re-forwards each Transition's game_state through encoder+model to
    compute grad-tracked log_prob / value / entropy — nothing is kept
    alive from the rollout-time forward.
    """

    def __init__(
        self,
        model:   nn.Module,
        encoder: nn.Module,
        config:  Optional[TrainerConfig] = None,
        device   = None,
    ):
        self.model   = model
        self.encoder = encoder
        self.config  = config if config is not None else TrainerConfig()
        self.device  = device
        self.optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(encoder.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def step(self, trajectories: List[List[Transition]]) -> TrainStats:
        n_traj = sum(1 for t in trajectories if t)
        if n_traj == 0:
            return TrainStats()

        flat: List[Transition] = []
        returns_flat: List[float] = []
        for traj in trajectories:
            if not traj:
                continue
            returns = _compute_returns(traj, self.config.gamma)
            flat.extend(traj)
            returns_flat.extend(returns)

        if not flat:
            return TrainStats()

        # Cap the number of transitions we process — for overlong
        # training batches, subsample uniformly. Keeps peak compute
        # bounded.
        cap = self.config.max_transitions_per_step
        if len(flat) > cap:
            step = len(flat) / cap
            idxs = [int(i * step) for i in range(cap)]
            flat = [flat[i] for i in idxs]
            returns_flat = [returns_flat[i] for i in idxs]

        # Pass 1: re-forward each transition, build per-transition
        # loss contributions, stack at the end and backward once.
        log_probs_all: List[torch.Tensor] = []
        values_all:    List[torch.Tensor] = []
        entropies_all: List[torch.Tensor] = []

        for t in flat:
            encoded = self.encoder.encode(t.game_state)
            output = self.model(encoded)
            log_prob, entropy = reforward_logprob_entropy(
                encoded, output, t.game_state,
                actor_idx=t.actor_idx,
                target_idx=t.target_idx,
                weapon_idx=t.weapon_idx,
            )
            value = output.value.squeeze()
            log_probs_all.append(log_prob)
            values_all.append(value)
            entropies_all.append(entropy)

        dev = log_probs_all[0].device
        log_probs = torch.stack(log_probs_all)           # [N]
        values    = torch.stack(values_all)              # [N]
        entropies = torch.stack(entropies_all)           # [N]
        returns = torch.tensor(returns_flat, device=dev, dtype=values.dtype)

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
            list(self.model.parameters()) + list(self.encoder.parameters()),
            self.config.grad_clip,
        )
        self.optimizer.step()

        return TrainStats(
            policy_loss    = float(policy_loss.item()),
            value_loss     = float(value_loss.item()),
            entropy        = float(entropy_term.item()),
            total_loss     = float(total_loss.item()),
            grad_norm      = float(grad_norm) if isinstance(grad_norm, float)
                             else float(grad_norm.item()),
            mean_return    = float(returns.mean().item()),
            n_transitions  = int(len(flat)),
            n_trajectories = int(n_traj),
        )


def _compute_returns(traj: List[Transition], gamma: float) -> List[float]:
    G = 0.0
    out: List[float] = []
    for t in reversed(traj):
        continuation = 0.0 if t.done else 1.0
        G = t.reward + gamma * G * continuation
        out.append(G)
    out.reverse()
    return out
