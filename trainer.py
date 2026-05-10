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

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from action_sampler import (
    _build_legality_masks,
    _masked_actor_logits,
    _masked_target_logits,
    _masked_weapon_logits,
    _unit_by_id,
    reforward_logprob_entropy,
)
from classes import GameState
from encoder import encode_raw


@dataclass
class Transition:
    """What the policy stashes per select_action call.

    No grad tensors — those are built fresh from game_state when the
    trainer re-forwards. Keeping the raw GameState is cheap (a tree of
    Python dataclasses); keeping activations is not.

    `type_idx` is the UnitActionType (ATTACK / MOVE) for unit actors;
    None for recruit / end_turn / legacy transitions. Defaulted to None
    so checkpoints / pickled trajectories from before the type head
    landed still load.
    """
    game_state: GameState
    actor_idx:  int
    target_idx: Optional[int] = None
    weapon_idx: Optional[int] = None
    type_idx:   Optional[int] = None
    # Filled in by game_manager after the next state arrives:
    reward:     float = 0.0
    done:       bool  = False


@dataclass
class MCTSExperience:
    """One root-state, MCTS-distilled training sample (AlphaZero
    convention). The trainer's `step_mcts` consumes a list of these.

    `visit_counts` records how many MCTS rollouts selected each
    legal (actor, target, weapon, type) action FROM this state. It
    is NOT pre-normalized -- the loss divides by total visits
    internally so different states with different rollout budgets
    weight correctly when batched.

    Schema: tuples of (actor_idx, target_idx, weapon_idx, count, type_idx).
    `type_idx` is the new (C.4) sub-decision for unit actors --
    UnitActionType.ATTACK or UnitActionType.MOVE. None for recruit
    / end_turn. Old (pre-C.4) MCTSExperience pickles used 4-tuples
    without type; the loss path tolerates either length and
    falls back to type_idx=None when the tuple is short, so legacy
    serialized data still trains.

    `z` is the terminal outcome from the perspective of the
    side-to-move at `game_state` (Wesnoth has no draws in PvP, but
    timeouts / mutual elimination produce z=0). For mid-game states
    on a played-out trajectory, AlphaZero uses the GAME's terminal
    z (not a bootstrap from the value net) -- the network learns
    to predict the eventual outcome from each visited state.

    Indices match `LegalActionPrior` from action_sampler so the
    same (actor, target, weapon, type) tuple a node returned during
    expansion can be stored verbatim here.
    """
    game_state:  GameState
    visit_counts: List[Tuple]    # 4-tuple (legacy) or 5-tuple (new)
    z:            float


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
    # Clamp discounted returns to the value head's output range so the
    # MSE loss has a finite, well-conditioned target. The model's value
    # head is `tanh`-bounded to [-1, +1] (model.py value_head), so any
    # return outside that range is unreachable -- leaving the clamp off
    # has the loss chasing impossible targets and the value estimate
    # saturating at ±1 with infinite gradient pressure.
    #
    # 1.0 is the natural cap for AlphaZero-style win/loss/draw rewards.
    # Bump if you genuinely need shaping rewards larger than the
    # terminal signal -- but in that case the value head should grow
    # too (drop the tanh + scale it).
    value_clip:           float = 1.0
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
        # training batches, subsample to bound peak compute.
        #
        # Use a random subset (not uniform stride). Stride-N gives a
        # sample that's heavily correlated with episode position: with
        # `gamma=0.99` and 200-step trajectories, returns near terminal
        # have ~1.0× weight while step-0 returns have ~0.13× weight,
        # and stride sampling preferentially keeps near-terminal
        # transitions (since they appear later in `flat`). The policy
        # then trains on a biased return distribution. Random sampling
        # gives every transition the same selection probability so the
        # subsampled batch's return distribution matches the full
        # batch's in expectation.
        cap = self.config.max_transitions_per_step
        if len(flat) > cap:
            idxs = random.sample(range(len(flat)), cap)
            # Sort to preserve trajectory order for any future
            # logic that walks `flat` sequentially (e.g. value
            # bootstrapping across consecutive steps). The sort is
            # cheap relative to the forward passes.
            idxs.sort()
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
        # Clamp returns to the value head's output range. Without this,
        # tanh-bounded value can never match a return like +5 (e.g. from
        # leftover shaping weights), leaving the MSE term saturating at
        # 1.0 forever and the policy loss starving for advantage signal.
        if self.config.value_clip is not None:
            returns_t = returns_t.clamp_(
                min=-float(self.config.value_clip),
                max=+float(self.config.value_clip),
            )

        # Both passes run in EVAL MODE so the two value forwards
        # (Pass 1 baseline, Pass 2 MSE target) see identical activations.
        # Old code only set eval() for Pass 1 and let Pass 2 inherit
        # train() from the caller -- with dropout=1e-4 (model.py:96)
        # this introduced tiny noise between the value used for
        # advantages and the value being fit, dragging the value head's
        # learning signal. Dropout=1e-4 is functionally a no-op
        # statistically (see model.py comment), so disabling it via
        # eval mode for the whole train_step doesn't lose meaningful
        # regularization. The try/finally restores train() mode for any
        # downstream caller that re-uses the model post-step.
        prev_model_training   = self.model.training
        prev_encoder_training = self.encoder.training
        self.model.eval(); self.encoder.eval()
        try:
            # Build the raw-encoded cache ONCE for the whole train_step.
            # `encode()` = `register_names` + `encode_raw` + `encode_from_raw`.
            # `encode_raw` is the expensive Python-side state-building (state-
            # walking, terrain lookups, threat distance scans); `encode_from_raw`
            # is just embedding gathers + projections. The two passes used to
            # call `encode()` separately, redoing the expensive part. Splitting
            # them and caching the raw chunk halves the encoder cost (one
            # `encode_raw` per transition for the whole train_step instead of
            # two) -- ~30-50% wall-clock speedup at typical chunk sizes.
            #
            # Caching the raw rather than the EncodedState matters: encode_from_raw's
            # output tensors are bound to the encoder's CURRENT embedding-table
            # parameters via autograd. Pass 1 runs under torch.no_grad so its
            # tensors have no graph; reusing them in Pass 2 would produce
            # zero-gradient backward calls. Re-running encode_from_raw per pass
            # produces the right grad-tracked tensors for that pass.
            register_names = self.encoder.register_names
            for t in flat:
                register_names(t.game_state)
            type_to_id    = self.encoder.unit_type_to_id
            faction_to_id = self.encoder.faction_to_id
            raw_cache = [
                encode_raw(t.game_state,
                           type_to_id=type_to_id,
                           faction_to_id=faction_to_id)
                for t in flat
            ]

            # --- Pass 1: values without grad ----------------------------
            values_np: List[float] = []
            with torch.no_grad():
                for start in range(0, N, B):
                    raw_chunk = raw_cache[start:start + B]
                    encoded_chunk = [self.encoder.encode_from_raw(r)
                                     for r in raw_chunk]
                    outputs = self.model.forward_batch(encoded_chunk)
                    for output in outputs:
                        values_np.append(float(output.value.squeeze().item()))

            values_est = torch.tensor(values_np, device=dev, dtype=torch.float32)
            advantages = returns_t - values_est
            if self.config.normalize_advantages and advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # --- Pass 2: gradient accumulation, per-chunk backward ------
            self.optimizer.zero_grad()

            # Scalar accumulators for logging (summed over N, divided
            # at the end to match the old mean-based metrics).
            sum_policy_loss = 0.0
            sum_value_loss  = 0.0
            sum_entropy     = 0.0

            for start in range(0, N, B):
                chunk = flat[start:start + B]
                L = len(chunk)
                raw_chunk = raw_cache[start:start + B]
                encoded_chunk = [self.encoder.encode_from_raw(r)
                                 for r in raw_chunk]
                outputs = self.model.forward_batch(encoded_chunk)

                chunk_log_probs: List[torch.Tensor] = []
                chunk_values:    List[torch.Tensor] = []
                chunk_value_logits: List[torch.Tensor] = []
                chunk_entropies: List[torch.Tensor] = []
                for t, encoded, output in zip(chunk, encoded_chunk, outputs):
                    lp, ent = reforward_logprob_entropy(
                        encoded, output, t.game_state,
                        actor_idx=t.actor_idx,
                        target_idx=t.target_idx,
                        weapon_idx=t.weapon_idx,
                        type_idx=t.type_idx,
                    )
                    chunk_log_probs.append(lp)
                    chunk_values.append(output.value.squeeze())
                    # value_logits is [1, K]; squeeze the leading 1.
                    chunk_value_logits.append(output.value_logits.squeeze(0))
                    chunk_entropies.append(ent)

                lp_t  = torch.stack(chunk_log_probs)
                val_t = torch.stack(chunk_values)
                vl_t  = torch.stack(chunk_value_logits)   # [L, K]
                ent_t = torch.stack(chunk_entropies)
                adv_t = advantages[start:start + L]
                ret_t = returns_t[start:start + L]

                # Mean-style losses, scaled by L/N so summing across
                # chunks matches the old .mean() over all N.
                policy_loss  = -(lp_t * adv_t).sum() / N
                # Distributional value loss: categorical CE between
                # predicted Z(s) and the bin-projection of the
                # (clipped) MC return. `ret_t` was already clipped
                # by `value_clip` earlier in this method, so it
                # falls inside the [V_MIN, V_MAX] support.
                atoms = self.model._value_atoms
                value_loss = _categorical_value_loss(vl_t, ret_t, atoms) / N
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
                # chunk's graph + activations can be reclaimed even if
                # DML's allocator holds onto buffers.
                del chunk_log_probs, chunk_values, chunk_entropies, chunk_value_logits
                del lp_t, val_t, ent_t, adv_t, ret_t
                del encoded_chunk, outputs, chunk_loss

            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.encoder.parameters()),
                self.config.grad_clip,
            )
            self.optimizer.step()
        finally:
            # Restore the caller's train/eval mode -- the caller may
            # toggle modes around train_step (e.g. for inference paths).
            if prev_model_training:
                self.model.train()
            if prev_encoder_training:
                self.encoder.train()

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


def _project_returns_to_atoms(
    returns: torch.Tensor, atoms: torch.Tensor,
) -> torch.Tensor:
    """Project a batch of scalar return targets onto a categorical
    distribution over `atoms`. Returns shape `[B, K]`.

    Linear interpolation between adjacent bins, per the "categorical
    algorithm" of C51 (Bellemare et al. 2017, Algorithm 1). For
    return r at the boundary between atom_l and atom_(l+1), mass
    splits proportionally; at an exact atom, mass=1 sits on it.
    Returns outside [atom_0, atom_K-1] are clamped to the support
    edges (consistent with the trainer's existing `value_clip`).

    The trainer uses this to convert (clipped) MC returns into the
    target distribution for the categorical-CE loss against the
    distributional value head.
    """
    B = returns.shape[0]
    K = atoms.shape[0]
    delta = atoms[1] - atoms[0]
    # Clamp to [V_MIN, V_MAX] so b lands inside [0, K-1].
    r = returns.clamp(atoms[0].item(), atoms[-1].item())
    b = (r - atoms[0]) / delta            # [B], real-valued in [0, K-1]
    l = b.floor().long().clamp(0, K - 2)  # [B], lower-bin index, room above
    weight_l = (l.float() + 1) - b        # [B], mass on bin l
    weight_u = b - l.float()              # [B], mass on bin l+1
    target = torch.zeros(B, K, device=returns.device, dtype=atoms.dtype)
    target.scatter_add_(1, l.unsqueeze(1), weight_l.unsqueeze(1))
    target.scatter_add_(1, (l + 1).unsqueeze(1), weight_u.unsqueeze(1))
    return target


def _categorical_value_loss(
    value_logits: torch.Tensor,    # [B, K] raw head logits
    returns:      torch.Tensor,    # [B] scalar targets (already clipped)
    atoms:        torch.Tensor,    # [K] bin support
) -> torch.Tensor:
    """Cross-entropy between the predicted distribution Z(s) and the
    projected target distribution. Sum-reduced over the batch (the
    trainer's chunk loop divides by N to match the old .mean()
    semantics)."""
    target_dist = _project_returns_to_atoms(returns, atoms)         # [B, K]
    log_probs = torch.nn.functional.log_softmax(value_logits, dim=-1)
    return -(target_dist * log_probs).sum()


# =====================================================================
# AlphaZero-style soft-target trainer extension
# =====================================================================
# Bolted onto the existing Trainer rather than living in a separate
# class because (a) it shares the same model + encoder + optimizer
# (one set of weights, one set of momenta) and (b) MCTS+REINFORCE
# co-training is a real option later (DeepMind's AlphaTensor uses it).
# Method-injection at module bottom keeps the diff to the REINFORCE
# core small.

def _mcts_factored_policy_loss(
    encoded,
    output,
    game_state: GameState,
    visit_counts: List[Tuple],
) -> Tuple[torch.Tensor, float, float]:
    """Cross-entropy of the model's factored policy against MCTS
    visit counts. Returns (loss, total_visits, action_kl_proxy).

    The factored loss decomposes joint cross-entropy across four
    heads:
      log P(actor) + [log P(type | actor) for unit actors]
                   + log P(target | actor [, type]) + [log P(weapon | actor) for ATTACK]
    Mathematically identical to a flat joint CE; avoids
    materializing the A*T*H*MAX_ATTACKS joint.

    Caches per-actor target / weapon / type log-probs since
    multiple visits typically share an actor.

    Visit-count tuple schema: 4-tuple (actor, target, weapon, count)
    on legacy data, 5-tuple (actor, target, weapon, count, type) on
    new data. The unpacking below tolerates both.
    """
    from action_sampler import _masked_target_logits_from_row, _masked_type_logits
    from model import UnitActionType

    # Helper to unpack legacy 4-tuple OR new 5-tuple.
    def _unpack(t):
        if len(t) >= 5:
            actor, target, weapon, count, type_idx = t[0], t[1], t[2], t[3], t[4]
        else:
            actor, target, weapon, count = t[0], t[1], t[2], t[3]
            type_idx = None
        return actor, target, weapon, count, type_idx

    masks = _build_legality_masks(encoded, game_state)
    actor_logits = _masked_actor_logits(encoded, output, masks.actor_valid)
    actor_logp = F.log_softmax(actor_logits.squeeze(0), dim=-1)  # [A]

    # Per-actor caches.
    type_logp_cache:        Dict[int, torch.Tensor] = {}
    target_attack_logp_cache: Dict[int, torch.Tensor] = {}
    target_move_logp_cache:   Dict[int, torch.Tensor] = {}
    target_union_logp_cache:  Dict[int, torch.Tensor] = {}
    weapon_logp_cache: Dict[int, Tuple[torch.Tensor, int]] = {}

    total_visits = sum(_unpack(t)[3] for t in visit_counts)
    if total_visits <= 0:
        return (
            actor_logp.new_zeros(()),
            0.0,
            0.0,
        )

    # Accumulate negative log-prob weighted by visit count.
    nll = actor_logp.new_zeros(())
    # Diagnostic: empirical visit entropy vs. policy entropy proxy
    # via the negative-log-prob average. Caller logs `entropy` from
    # the original REINFORCE convention; we expose total visits and
    # mean -log p(a|s) so the train_step log line stays informative.
    sum_actor_nlp = 0.0

    for tup in visit_counts:
        actor_idx, target_idx, weapon_idx, count, type_idx = _unpack(tup)
        if count <= 0:
            continue
        nll = nll - count * actor_logp[actor_idx]
        sum_actor_nlp += count * float(-actor_logp[actor_idx].item())

        # Type term (only for unit actors with a type_idx).
        is_unit = actor_idx < output.num_units
        if is_unit and type_idx is not None:
            tylp = type_logp_cache.get(actor_idx)
            if tylp is None:
                tyl = _masked_type_logits(
                    output, masks.type_valid, actor_idx,
                    type_bias=masks.type_bias,
                )
                if tyl.numel() == 0:
                    type_logp_cache[actor_idx] = None  # type: ignore
                else:
                    tylp = F.log_softmax(tyl, dim=-1)
                    type_logp_cache[actor_idx] = tylp
            if tylp is not None:
                nll = nll - count * tylp[type_idx]

        if target_idx is not None:
            # Pick the type-conditional cache for unit actors with
            # a type_idx; legacy entries (type_idx=None) fall back
            # to the union mask (legacy chain rule).
            if is_unit and type_idx == UnitActionType.ATTACK:
                tlp = target_attack_logp_cache.get(actor_idx)
                if tlp is None:
                    tl = _masked_target_logits_from_row(
                        output, masks.target_valid_attack[actor_idx],
                        actor_idx, attack_bias=masks.attack_bias[actor_idx],
                    )
                    tlp = F.log_softmax(tl, dim=-1) if tl.numel() else None
                    target_attack_logp_cache[actor_idx] = tlp
            elif is_unit and type_idx == UnitActionType.MOVE:
                tlp = target_move_logp_cache.get(actor_idx)
                if tlp is None:
                    tl = _masked_target_logits_from_row(
                        output, masks.target_valid_move[actor_idx],
                        actor_idx, attack_bias=None,
                    )
                    tlp = F.log_softmax(tl, dim=-1) if tl.numel() else None
                    target_move_logp_cache[actor_idx] = tlp
            else:
                # Legacy / recruit / end_turn -- union mask.
                tlp = target_union_logp_cache.get(actor_idx)
                if tlp is None:
                    tl = _masked_target_logits(
                        output, masks.target_valid, actor_idx,
                        attack_bias=masks.attack_bias,
                    )
                    tlp = F.log_softmax(tl, dim=-1) if tl.numel() else None
                    target_union_logp_cache[actor_idx] = tlp
            if tlp is None:
                continue
            nll = nll - count * tlp[target_idx]

        if weapon_idx is not None:
            cached = weapon_logp_cache.get(actor_idx)
            if cached is None:
                # Weapon mask needs the unit's actual attack count.
                unit_id = encoded.unit_ids[actor_idx]
                attacker = _unit_by_id(game_state, unit_id)
                num_attacks = len(attacker.attacks) if attacker else 0
                if num_attacks <= 0:
                    weapon_logp_cache[actor_idx] = (None, 0)  # type: ignore
                    continue
                wl = _masked_weapon_logits(output, actor_idx, num_attacks)
                wlp = F.log_softmax(wl, dim=-1)
                weapon_logp_cache[actor_idx] = (wlp, num_attacks)
            else:
                wlp, num_attacks = cached
            if wlp is None:
                continue
            if weapon_idx >= num_attacks:
                # Stale visit-count slot (the unit's attack list has
                # fewer slots now than at MCTS time). Skip silently.
                continue
            nll = nll - count * wlp[weapon_idx]

    loss = nll / float(total_visits)
    mean_actor_nlp = sum_actor_nlp / float(total_visits)
    return loss, float(total_visits), float(mean_actor_nlp)


def _trainer_step_mcts(
    self,                                     # Trainer (method injected below)
    experiences: List[MCTSExperience],
) -> TrainStats:
    """One AlphaZero-style gradient step. Each experience contributes
    a factored cross-entropy term against the MCTS visit
    distribution, plus an MSE term against the terminal outcome z.

    Like REINFORCE `step`, processes experiences in chunks of
    `train_batch_size` and calls `.backward()` per chunk to bound
    peak activation memory. Final `optimizer.step()` once at the end.
    """
    if not experiences:
        return TrainStats()

    cap = self.config.max_transitions_per_step
    if len(experiences) > cap:
        # Subsample. Same uniform-stride logic the REINFORCE path
        # uses; the subsample bias documented in BACKLOG.md applies
        # here too.
        step = len(experiences) / cap
        experiences = [experiences[int(i * step)] for i in range(cap)]

    dev = self.device or next(self.model.parameters()).device
    N = len(experiences)
    B = max(1, self.config.train_batch_size)

    # Clamp z values to the value head's range (matches REINFORCE
    # path's value_clip handling).
    zs = torch.tensor(
        [e.z for e in experiences], device=dev, dtype=torch.float32,
    )
    if self.config.value_clip is not None:
        zs.clamp_(min=-float(self.config.value_clip),
                  max=+float(self.config.value_clip))

    self.optimizer.zero_grad()

    sum_policy_loss = 0.0
    sum_value_loss  = 0.0
    sum_total_visits = 0.0
    sum_actor_nlp_weighted = 0.0  # for "entropy"-style logging

    self.model.train(); self.encoder.train()

    # Pre-compute the raw-encoded cache (one encode_raw per experience)
    # so the policy-loss helper -- which already calls encode internally
    # via _masked_target_logits' codepath -- doesn't pay the Python-
    # side state-building cost twice. encode_from_raw runs per chunk
    # below to produce the grad-tracked tensors.
    register_names = self.encoder.register_names
    for e in experiences:
        register_names(e.game_state)
    type_to_id    = self.encoder.unit_type_to_id
    faction_to_id = self.encoder.faction_to_id
    raw_cache = [
        encode_raw(e.game_state,
                   type_to_id=type_to_id,
                   faction_to_id=faction_to_id)
        for e in experiences
    ]

    for start in range(0, N, B):
        chunk = experiences[start:start + B]
        L = len(chunk)
        raw_chunk = raw_cache[start:start + B]
        encoded_chunk = [self.encoder.encode_from_raw(r) for r in raw_chunk]
        outputs = self.model.forward_batch(encoded_chunk)

        chunk_policy_losses: List[torch.Tensor] = []
        chunk_values: List[torch.Tensor] = []
        chunk_value_logits: List[torch.Tensor] = []
        for e, encoded, output in zip(chunk, encoded_chunk, outputs):
            policy_loss, total_v, mean_actor_nlp = (
                _mcts_factored_policy_loss(
                    encoded, output, e.game_state, e.visit_counts,
                )
            )
            chunk_policy_losses.append(policy_loss)
            chunk_values.append(output.value.squeeze())
            chunk_value_logits.append(output.value_logits.squeeze(0))
            sum_total_visits += total_v
            sum_actor_nlp_weighted += mean_actor_nlp * total_v

        policy_loss_t = torch.stack(chunk_policy_losses).sum() / N
        val_t = torch.stack(chunk_values)
        vl_t  = torch.stack(chunk_value_logits)
        z_t   = zs[start:start + L]
        # Distributional value loss: categorical CE on the projected
        # terminal-z target (z ∈ {-1, 0, +1} for win/draw/loss),
        # consistent with the REINFORCE path. `zs` was already
        # clipped to [V_MIN, V_MAX] by the value_clip block above.
        atoms = self.model._value_atoms
        value_loss = _categorical_value_loss(vl_t, z_t, atoms) / N

        chunk_loss = (
            policy_loss_t
            + self.config.value_coef * value_loss
        )
        chunk_loss.backward()

        sum_policy_loss += float(policy_loss_t.item())
        sum_value_loss  += float(value_loss.item())

        del chunk_policy_losses, chunk_values, val_t, z_t, chunk_loss
        del encoded_chunk, outputs

    grad_norm = torch.nn.utils.clip_grad_norm_(
        list(self.model.parameters()) + list(self.encoder.parameters()),
        self.config.grad_clip,
    )
    self.optimizer.step()

    self.model.eval(); self.encoder.eval()

    # "Entropy" slot reports mean -log p(actor | s), weighted by
    # visits, for logging continuity with the REINFORCE path.
    mean_actor_nlp = (
        sum_actor_nlp_weighted / sum_total_visits
        if sum_total_visits > 0 else 0.0
    )

    return TrainStats(
        policy_loss    = float(sum_policy_loss),
        value_loss     = float(sum_value_loss),
        entropy        = float(mean_actor_nlp),  # see comment above
        total_loss     = float(
            sum_policy_loss
            + self.config.value_coef * sum_value_loss
        ),
        grad_norm      = float(grad_norm) if isinstance(grad_norm, float)
                         else float(grad_norm.item()),
        mean_return    = float(zs.mean().item()),
        n_transitions  = int(N),
        n_trajectories = int(N),  # one experience = one root state
    )


# Inject as a method on Trainer. Gives users `trainer.step_mcts(exps)`
# alongside the REINFORCE `trainer.step(trajectories)`.
Trainer.step_mcts = _trainer_step_mcts
