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
    # Lowered from 0.01 after the first 22 train_steps held entropy
    # ~8.3 (near max). The bonus was dominating the tiny shaping
    # gradients and preventing the policy from ever committing to an
    # action. Still nonzero so exploration isn't killed entirely.
    entropy_coef:         float = 0.001
    grad_clip:            float = 1.0
    normalize_advantages: bool  = True
    # Cap transitions processed per train_step. Originally 512 as a
    # defensive memory guard; re-forward training (trainer.py intro)
    # removed the memory pressure, so we can afford much more of the
    # collected data. 4000 is ~50% of a typical 4-game queue; each
    # train_step runs ~3-4x longer (~100-200 s on CPU) but uses 8x
    # more data. If freezes recur, bring it back down.
    max_transitions_per_step: int = 4000

    # How many re-forwarded transitions to push through the model in
    # one batched call. On CPU with our current tensor sizes (~1600
    # hex tokens × 128 d_model × 3 layers), batching the transformer
    # ran 1.7–2.8× SLOWER than a sequence of single forwards — the
    # padded activations spill past L2/L3 and PyTorch's CPU attention
    # doesn't amortize gemm setup across batch for these shapes. So
    # we default to 1 (equivalent to the old path via forward_batch's
    # B==1 shortcut). Raise this on GPU, where batched forwards
    # actually win. Benchmarks and rationale in history for this file.
    train_batch_size: int = 1


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

        # Two-pass training to keep peak activation memory bounded to a
        # single chunk's forward graph — required on DML, where the
        # whole-training-graph retention of the old design OOMed even
        # at small batch sizes.
        #
        # Pass 1 (no_grad): forward every transition to obtain its value
        #   estimate. Activations are not retained; we record a Python
        #   float per transition and move on.
        # Pass 2 (grad):    with the advantages precomputed from Pass 1
        #   values, forward each chunk again, build the chunk's share of
        #   the loss, and call .backward() before moving to the next
        #   chunk. Per-chunk backward releases that chunk's activations;
        #   optimizer.step() fires only once at the end.
        #
        # Cost: 2× forwards per transition. With torch.no_grad the first
        # pass is cheap (~40-50% of a training forward). The batching
        # savings on DML offset this; on CPU we use B=1 anyway so the
        # extra pass is the same ~30 ms/transition we already eat.
        dev = self.device or next(self.model.parameters()).device
        N = len(flat)
        B = max(1, self.config.train_batch_size)

        returns_t = torch.tensor(returns_flat, device=dev, dtype=torch.float32)

        # --- Pass 1: values without grad --------------------------------
        values_np: List[float] = []
        self.model.eval(); self.encoder.eval()
        try:
            with torch.no_grad():
                for start in range(0, N, B):
                    chunk = flat[start:start + B]
                    encoded_chunk = [self.encoder.encode(t.game_state) for t in chunk]
                    outputs = self.model.forward_batch(encoded_chunk)
                    for output in outputs:
                        values_np.append(float(output.value.squeeze().item()))
        finally:
            self.model.train(); self.encoder.train()

        values_est = torch.tensor(values_np, device=dev, dtype=torch.float32)
        advantages = returns_t - values_est
        if self.config.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- Pass 2: gradient accumulation, per-chunk backward ----------
        self.optimizer.zero_grad()

        # Scalar accumulators for logging (summed over N, divided at the
        # end to match the old mean-based metrics).
        sum_policy_loss = 0.0
        sum_value_loss  = 0.0
        sum_entropy     = 0.0

        for start in range(0, N, B):
            chunk = flat[start:start + B]
            L = len(chunk)
            encoded_chunk = [self.encoder.encode(t.game_state) for t in chunk]
            outputs = self.model.forward_batch(encoded_chunk)

            chunk_log_probs: List[torch.Tensor] = []
            chunk_values:    List[torch.Tensor] = []
            chunk_entropies: List[torch.Tensor] = []
            for t, encoded, output in zip(chunk, encoded_chunk, outputs):
                lp, ent = reforward_logprob_entropy(
                    encoded, output, t.game_state,
                    actor_idx=t.actor_idx,
                    target_idx=t.target_idx,
                    weapon_idx=t.weapon_idx,
                )
                chunk_log_probs.append(lp)
                chunk_values.append(output.value.squeeze())
                chunk_entropies.append(ent)

            lp_t  = torch.stack(chunk_log_probs)
            val_t = torch.stack(chunk_values)
            ent_t = torch.stack(chunk_entropies)
            adv_t = advantages[start:start + L]
            ret_t = returns_t[start:start + L]

            # Mean-style losses, scaled by L/N so summing across chunks
            # matches the old .mean() over all N.
            policy_loss  = -(lp_t * adv_t).sum() / N
            value_loss   = F.mse_loss(val_t, ret_t, reduction='sum') / N
            entropy_term = ent_t.sum() / N

            chunk_loss = (
                policy_loss
                + self.config.value_coef   * value_loss
                - self.config.entropy_coef * entropy_term
            )
            chunk_loss.backward()

            sum_policy_loss += float(policy_loss.item())
            sum_value_loss  += float(value_loss.item())
            sum_entropy     += float(entropy_term.item())

            # Drop references before the next chunk so the previous
            # chunk's graph + activations can be reclaimed even if DML's
            # allocator holds onto buffers.
            del chunk_log_probs, chunk_values, chunk_entropies
            del lp_t, val_t, ent_t, adv_t, ret_t
            del encoded_chunk, outputs, chunk_loss

        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.encoder.parameters()),
            self.config.grad_clip,
        )
        self.optimizer.step()

        return TrainStats(
            policy_loss    = float(sum_policy_loss),
            value_loss     = float(sum_value_loss),
            entropy        = float(sum_entropy),
            total_loss     = float(
                sum_policy_loss
                + self.config.value_coef   * sum_value_loss
                - self.config.entropy_coef * sum_entropy
            ),
            grad_norm      = float(grad_norm) if isinstance(grad_norm, float)
                             else float(grad_norm.item()),
            mean_return    = float(returns_t.mean().item()),
            n_transitions  = int(N),
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
