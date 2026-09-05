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

import contextlib
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wesnoth_ai.action_sampler import (
    _NEG_INF,
    prior_bias_end_turn as _prior_bias_end_turn,
    _build_legality_masks,
    _masked_actor_logits,
    _masked_target_logits,
    _masked_weapon_logits,
    _unit_by_id,
    reforward_logprob_entropy,
)
from wesnoth_ai.classes import GameState
from wesnoth_ai.device import dml_sync
from wesnoth_ai.encoder import RawEncoded, encode_raw
from wesnoth_ai.model import UnitActionType
from wesnoth_ai.packed_trunk import FlatLayout

import logging

log = logging.getLogger("trainer")

# Once-per-process tripwire for a GBC step that computes no loss
# despite labeled experiences (see step_mcts's gbc block).
_GBC_SILENT_WARNED = False

# Count of visit-count index terms skipped by the stored-index bounds
# guard of the factored policy loss (see _oob_index). Module-level so
# the log throttle survives across steps; a nonzero value on a leg
# means the search-time and train-time index bases diverged somewhere
# and the leg needs a root-cause before its gradients are trusted.
_OOB_INDEX_EVENTS = 0

# Stages of one step_mcts call, in execution order; the keys of the
# `timings` dict step_mcts fills (seconds, added to whatever the dict
# already holds). "value_loss" covers every value-side term (value,
# consistency, trust, aux, moves-left, GBC).
STEP_MCTS_STAGES = ("encode_raw", "encode", "forward", "policy_loss",
                    "value_loss", "backward", "clip", "optimizer")

_CPU = torch.device("cpu")


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
    # Combat-oracle anneal counter at SAMPLE time. The re-forward in
    # Trainer.step must rebuild the legality masks (which carry the
    # annealed combat-oracle bias) at the SAME alpha the sampler used,
    # or the reference logP/entropy the policy gradient is computed
    # against won't match the on-policy sampling distribution. Defaults
    # to 0 so pre-existing pickled trajectories still load (they get
    # full-strength bias, matching the old behavior).
    decision_step: int = 0
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
    # Optional auxiliary target (KataGo §3.5): the final MATERIAL
    # margin from this state's side perspective, in (-1, +1). `None`
    # when aux targets aren't being collected (the trainer then skips
    # the aux loss). A denser companion to `z`. See
    # draw_tiebreak.material_margin + MCTSPolicy.finalize_game.
    aux_target:  Optional[float] = None
    # Fraction of the turn budget that was STILL to be played from
    # this state (turns_remaining / MOVES_LEFT_NORM_TURNS, clipped to
    # [0, 1]). Lc0-style moves-left target (2026-07-04): a dense
    # tempo signal the sparse z cannot provide. None when the game's
    # end turn was unknown (legacy pickles) -- the loss then skips.
    moves_left_target: Optional[float] = None
    # VALUE-loss weight for this state (truncation ruling 2026-08-17:
    # "there are no draws in real Wesnoth"). finalize_game seals a
    # winnerless game's states with 0.0 -- they train the policy at
    # full weight but are CENSORED observations for the value head
    # (the z=0 flood that poisoned leg 3's grader). Legacy pickles
    # default to 1.0. Distinct from game_weight (which normalizes
    # per-game influence on EVERY loss term).
    value_weight: float = 1.0
    # POLICY-loss magnitude for this state (2026-08-26). Multiplies
    # the state's policy CE AFTER visit normalization and is NOT
    # renormalized away: a state at 0.5 trains the policy at exactly
    # half strength. This is the one sound channel for magnitude-
    # scaled targets (conservative-mixture / margin-scaled steps) --
    # scaling the visit COUNTS instead silently cancels, because
    # _mcts_factored_policy_loss divides by total visits (counts are
    # a distribution, not a magnitude). Legacy pickles default 1.0.
    policy_weight: float = 1.0
    # Per-game normalization weight (2026-07-12): 1/n_recorded_states
    # of the source game, so every GAME contributes equally to the
    # gradient regardless of length (a 190-turn ladder draw no longer
    # outweighs a 10-turn mini ~19:1 in state count). Consumed by
    # step_mcts in every loss term. Old pickles lack the attr; all
    # consumers getattr() with default 1.0.
    game_weight: float = 1.0
    # Training-progress counter at the time this state's MCTS search ran.
    # Threaded into the distillation loss so the reference legality
    # masks rebuild the combat-oracle bias at the SAME annealed alpha the
    # search used (`combat_alphas_at`) -- search priors and loss must
    # agree. Default 0 = full-strength oracle (also the legacy-pickle
    # fallback, matching the pre-anneal behavior of old serialized data).
    decision_step: int = 0
    # GBC event-supervision labels (2026-08-14, docs/archive/gbc_spec.md):
    # fog-censored hindsight rows ("u", id, pred, y1, y2) / ("v", x,
    # y, pred, y1, y2) built in finalize_game. None on legacy pickles
    # and when labeling is off; the loss skips absent labels per
    # experience, so mixed buffers train unchanged.
    gbc_labels: Optional[List] = None
    # Source game identity (2026-08-30, value-memory ruling): lets
    # the learner-side per-GAME outcome reservoir cap states per
    # game and count independent outcomes. "" on legacy pickles --
    # such experiences simply never enter the reservoir.
    game_id: str = ""
    # Label provenance (arm VG2, 2026-09-02): "game" = terminal
    # outcome of the recorded game; "roll" = raw-policy rollout
    # outcome from a search-consulted state (categorical, like
    # "game"); "consist" = the search's depth-H estimate of a
    # consulted state -- a SCALAR bootstrap, trained with the
    # bias-corrected Gaussian term, never the categorical loss.
    label_kind: str = "game"
    # For "consist" states that ALSO received a rollout label: that
    # outcome. Paired data -> the learner estimates the bootstrap's
    # bias and residual variance every iteration (no tuned weights).
    z_pair: Optional[float] = None
    # Variance of z_pair as the mean of k rollouts (sample variance
    # / k), MEASURED from repeated playouts; None with one rollout.
    # Lets the learner subtract the rollout label's noise from the
    # search-vs-rollout spread without the 1 - V^2 proxy.
    z_pair_var: Optional[float] = None
    # Trust region (arm VG2): the head's mean prediction on this
    # consulted state under the weights at the START of the
    # iteration; the proximal term bounds movement away from it.
    # None = state not in the trust region.
    v_anchor: Optional[float] = None


@dataclass
class TrainerConfig:
    """Hyperparameters; override by constructing with different values."""
    learning_rate:        float = 1e-4
    weight_decay:         float = 1e-4
    gamma:                float = 0.99
    value_coef:           float = 0.5
    # Auxiliary-target loss weight (KataGo §3.5). Only has an effect
    # when the model was built with the aux head (`aux_score=True`) AND
    # the experiences carry `aux_target`s; otherwise it's a no-op. A
    # small weight (KataGo uses ~0.15) so the dense margin signal
    # regularizes the shared trunk without overwhelming the policy/value
    # objectives. See draw_tiebreak.material_margin.
    aux_coef:             float = 0.15
    # GBC event-supervision coefficient (2026-08-14): BCE of the
    # dies/flips achievement heads vs hindsight labels — the dense
    # value-adjacent signal (KataGo-ownership pattern; approved as
    # the value-head repair after gbc_spec.md 0d). Applies when the
    # model has gbc heads AND experiences carry labels; 0 = off.
    gbc_coef:             float = 0.1
    # Moves-left loss weight (Lc0-style). Effective only when the
    # model has the head (`moves_left=True`) AND experiences carry
    # `moves_left_target`s. Small: it is a trunk regularizer / future
    # search-utility input, not an objective that should compete with
    # policy/value.
    moves_left_coef:      float = 0.1
    # Mix eps of uniform mass into the projected C51 value target
    # (TRAIN loss only; eval CE stays unsmoothed). 0 = off. Guards
    # against extreme-atom collapse under many replay updates on
    # hard terminal targets -- see _categorical_value_loss.
    value_label_smoothing: float = 0.0
    # Per-state weight of DRAWN games (|z| < 1) in the MCTS value
    # loss. 1.0 = legacy (draws train the head toward their z).
    # 0.0 = decisive-only value learning: draws still feed the aux
    # and moves-left heads, but stop flattening the value head
    # (2026-07-10: ~71% of incoming states are draws; even with
    # honest z=0 labels their gradient mass erodes win/loss
    # discrimination -- human-corpus late AUC 0.88 -> 0.64 in 51
    # iters WITH a 512-state/iter rehearsal anchor).
    # Winnerless-state value weight, applied ONCE in finalize_game
    # (project round-1 C3). Default 0.0 = the 2026-08-17 truncation
    # ruling: censored games carry no value label.
    draw_value_weight: float = 0.0
    # Lowered from 0.01 after the first 22 train_steps held entropy
    # ~8.3 (near max). The bonus was dominating the tiny shaping
    # gradients and preventing the policy from ever committing to an
    # action. Still nonzero so exploration isn't killed entirely.
    entropy_coef:         float = 0.001
    grad_clip:            float = 1.0
    # Value loss form (minimal loop, docs/archive/az_minimal_spec.md):
    # "c51" = categorical CE on the projected result (legacy);
    # "mse_mean" = squared error between the head's MEAN prediction
    # and the result. The categorical loss charges a confident head
    # ~-log(tiny) for a coin-flip label, which is why the value term
    # owned 99% of every update; squared error charges in proportion
    # to the miss. The head stays C51 -- only the loss changes.
    value_loss_form:      str = "c51"
    # Arm VG2 (2026-09-02, "every parameter becomes a measurement"):
    # the consistency (bootstrap) term is the Gaussian NLL of the
    # head's MEAN prediction against the bias-corrected search
    # estimate, (v - (z - bias))^2 / (2 sigma2). Both statistics are
    # ESTIMATED by MCTSPolicy from paired (search, rollout) labels
    # each iteration and written here before step_mcts; the defaults
    # only apply before the first estimate exists (sigma2=1 -> the
    # term starts at unit-Gaussian strength, bias 0).
    consist_bias:         float = 0.0
    consist_sigma2:       float = 1.0
    # Trust-region multiplier on consulted states: lambda * (v -
    # v_anchor)^2. Driven by dual ascent on the live dv_consult
    # against trust_delta (docs/design_constants.md: search's
    # decision resolution, 2 C51 atoms = 0.08). 0 = off.
    trust_lambda:         float = 0.0
    trust_delta:          float = 0.08
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
    # we default to 1. On a GPU the loop sets 16 (tools/az_loop.py;
    # docs/box_specs.md "Training path cost (2026-09-05)").
    train_batch_size: int = 1
    # bf16 autocast around step_mcts's network forward (the encoder's
    # projections, the trunk and the heads), cuda only. The outputs
    # are cast back to float32 before any loss, so the log-softmaxes,
    # the C51 projection and the squared errors run in fp32 and only
    # the matmuls are bf16; the master weights, the AdamW step and the
    # inference snapshot are untouched (no GradScaler: bf16 keeps
    # fp32's exponent range). On cpu the switch is a no-op. Batch 16
    # on a 4090: 45.0 against 57.1 ms per experience; on one batch of
    # 64 the loss is within 3e-4 of fp32, gradient cosine 0.9994, norm
    # within 0.3% (docs/box_specs.md "Training path cost
    # (2026-09-05)"). Off by default; az_loop --train-bf16.
    train_autocast_bf16: bool = False


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
    aux_loss:       float = 0.0   # auxiliary margin loss (KataGo §3.5); 0 when off
    gbc_loss:       float = 0.0   # GBC event-supervision BCE; 0 when off
    moves_left_loss: float = 0.0  # Lc0-style moves-left MSE; 0 when off
    # Boundary-consistency telemetry (T1-F, 2026-07-29): mean of
    # V(s_pre)+V(s_post) over sampled side-switch pairs of recorded
    # states (no_grad, post-update net). Zero-sum calibration
    # predicts ~0; the fogged-play WYSIATI bias measured +0.4..+0.65
    # on the 2026-07-28 lineage (fogless ~0). Attached by
    # MCTSPolicy._attach_boundary_sum; nan when <4 pairs collected.
    boundary_sum:     float = float("nan")
    boundary_pairs_n: int   = 0
    # Size of the boundary-pair POOL the reading was sampled from.
    # Reported separately because boundary_pairs_n saturates at the
    # sample cap, so on its own it cannot distinguish "only 16 pairs
    # exist" from "4000 exist and we sampled 16" -- i.e. it cannot
    # tell you whether the reading's noise is fixable by sampling
    # more.
    boundary_pool_n:  int   = 0
    # Value CE on THIS iteration's incoming games, measured BEFORE any
    # gradient step touched them (nan when unavailable). Distribution-
    # matched generalization signal: unlike the frozen holdout it
    # tracks the current self-play distribution, and unlike the train
    # value loss the net has never seen these states. The gap
    # (value_loss vs this) is the memorization measurement.
    fresh_value_ce: float = float("nan")
    # Mean entropy of the predicted Z(s) on the fresh probe (nats;
    # uniform = ln 51 ~ 3.93). Continuous overconfidence curve.
    fresh_pred_entropy: float = float("nan")
    # Target composition of the iteration's incoming experiences
    # (2026-07-10 draw-spike diagnosis): fraction of z=+1 / z=-1 /
    # everything else (draws; tiebreak values under legacy labels).
    z_win_frac: float = float("nan")
    z_loss_frac: float = float("nan")
    z_draw_frac: float = float("nan")
    # Fresh CE restricted to decisive (+-1) incoming states -- the
    # gate metric when draw_value_weight=0 (pooled fresh_value_ce is
    # then structurally inflated by z=0 states the head is
    # deliberately not trained on).
    fresh_decisive_ce: float = float("nan")
    # States whose weight actually feeds the value loss this step
    # (decisive + weighted draws; starvation watch for
    # draw_value_weight=0 runs). Summed across replay updates.
    value_signal_states: int = 0
    # CE of the best state-blind predictor on the fresh probe (the
    # batch's empirical projected-z mixture). fresh_value_ce should
    # sit BELOW this; a high floor means the games' outcomes are
    # inherently mixed and caps what any head can achieve.
    fresh_ce_floor: float = float("nan")
    # In-training signal telemetry (2026-09-01 ruling: always on):
    # per-source gradient norms on a 128-state subsample (unclipped,
    # optimizer stubbed) + mean |dv| the applied update caused on
    # this iteration's search-consulted states. The trend view of
    # signal_profiler's offline gradient tree.
    sig_policy_norm: float = float("nan")
    sig_value_game_norm: float = float("nan")
    sig_value_ground_norm: float = float("nan")
    sig_value_consist_norm: float = float("nan")
    sig_dv_consult_mean: float = float("nan")
    sig_dv_consult_n: float = 0.0
    # Arm VG2 principled-mixture telemetry: the paired-label
    # estimates the learner used this iteration, the two new loss
    # terms, and the trust-region multiplier after dual ascent.
    consist_loss:      float = 0.0
    trust_loss:        float = 0.0
    consist_bias_hat:  float = float("nan")
    consist_sigma2_hat: float = float("nan")
    consist_pair_n:    float = 0.0
    trust_lambda:      float = float("nan")
    # Parameter-correctness monitors (every iteration, from the
    # same paired states): sigma2's two components, the head's
    # error vs rollout truth, the corrected label's residual.
    consist_var_diff:  float = float("nan")
    consist_roll_noise: float = float("nan")
    consist_sigma2_point: float = float("nan")
    consist_sigma2_se: float = float("nan")
    consist_head_minus_truth: float = float("nan")
    consist_label_minus_truth: float = float("nan")
    # Per-turn-decade fresh-probe decomposition (2026-09-01):
    # {"d1_10": {"ce","floor","auc","n"}, ..., "d61p": ...}. The
    # pooled fresh_value_ce above stays the usual read; these
    # columns separate phases whose +-1 labels carry very
    # different information (early-game outcomes are largely
    # aleatoric). None when the probe didn't run.
    fresh_by_decade: Optional[Dict] = None


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
        # Default sink for step_mcts's per-stage seconds (keys
        # STEP_MCTS_STAGES): callers that reach step_mcts through
        # MCTSPolicy.train_step (az_loop) set a dict here and read it
        # back; direct callers pass `timings=` instead. None = no timing.
        self.stage_timings: Optional[Dict[str, float]] = None
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

        # DML mitigation #1: drain any command-list residue left over
        # from the rollout phase BEFORE the first train_step transfer.
        # Rollout's 6 worker threads can leave thousands of small
        # forward + .to() ops queued; without this flush the very
        # first encode_from_raw_batch transfer hits a queue that's
        # already too deep for the AMD driver, crashing with
        # "device suspended". `dml_sync` is a no-op on CPU/CUDA so
        # this is free everywhere except DML. See device.py.
        dml_sync(self.device or next(self.model.parameters()).device)

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
        self.model.eval()
        self.encoder.eval()
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
                           faction_to_id=faction_to_id,
                           relevant_set=getattr(self.encoder,
                                                "relevant_set_hexes",
                                                False))
                for t in flat
            ]

            # --- Pass 1: values without grad ----------------------------
            # DML mitigation #2: periodic command-list flush inside
            # the chunk loop. On DML the per-chunk `.item()` call at
            # the inner `values_np.append(...)` line already syncs
            # the queue once per chunk -- that's our flush, free of
            # charge. (On CUDA it forces a small overhead but we
            # only get here at B=1; the chunk count is N which is
            # the number of transitions, ~3000 on a typical iter.
            # If this is the bottleneck on CUDA, sub-batch into
            # bigger Bs in TrainerConfig.) Logging the natural
            # sync rather than adding a redundant one.
            values_np: List[float] = []
            with torch.no_grad():
                for start in range(0, N, B):
                    raw_chunk = raw_cache[start:start + B]
                    encoded_chunk = self.encoder.encode_from_raw_batch(
                        raw_chunk)
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
                encoded_chunk = self.encoder.encode_from_raw_batch(
                    raw_chunk)
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
                        decision_step=getattr(t, "decision_step", 0),
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
    # Clamp to [V_MIN, V_MAX] so b lands inside [0, K-1]. Use the
    # device-resident atom endpoints as tensor bounds (clamp broadcasts
    # 0-dim tensors) rather than .item()-ing them — the latter is two
    # D2H syncs per chunk on CUDA for a fixed, non-learned buffer.
    r = returns.clamp(atoms[0], atoms[-1])
    b = (r - atoms[0]) / delta            # [B], real-valued in [0, K-1]
    lo = b.floor().long().clamp(0, K - 2)  # [B], lower-bin index, room above
    weight_l = (lo.float() + 1) - b       # [B], mass on bin lo
    weight_u = b - lo.float()             # [B], mass on bin lo+1
    target = torch.zeros(B, K, device=returns.device, dtype=atoms.dtype)
    target.scatter_add_(1, lo.unsqueeze(1), weight_l.unsqueeze(1))
    target.scatter_add_(1, (lo + 1).unsqueeze(1), weight_u.unsqueeze(1))
    return target


def _categorical_value_loss(
    value_logits: torch.Tensor,    # [B, K] raw head logits
    returns:      torch.Tensor,    # [B] scalar targets (already clipped)
    atoms:        torch.Tensor,    # [K] bin support
    label_smoothing: float = 0.0,  # TRAIN-only; eval passes 0
    weights: "Optional[torch.Tensor]" = None,   # [B] per-state
) -> torch.Tensor:
    """Cross-entropy between the predicted distribution Z(s) and the
    projected target distribution. Sum-reduced over the batch (the
    trainer's chunk loop divides by N to match the old .mean()
    semantics).

    `label_smoothing` mixes eps of uniform mass into the projected
    target: with hard +-1 outcome targets and many replay updates the
    C51 head otherwise collapses toward extreme-atom spikes
    (measured 2026-07-07: Z entropy 1.86->1.13, max-atom p
    0.39->0.58 in 46 iters) and a confidently-wrong spike makes the
    held-out CE explode. Smoothing bounds the target away from a
    delta so the head keeps calibrated uncertainty. Applied only to
    the TRAINING loss -- eval CE stays unsmoothed for comparability
    across runs."""
    target_dist = _project_returns_to_atoms(returns, atoms)         # [B, K]
    if label_smoothing > 0.0:
        K = atoms.shape[0]
        target_dist = (
            (1.0 - label_smoothing) * target_dist
            + label_smoothing / K)
    log_probs = torch.nn.functional.log_softmax(value_logits, dim=-1)
    per_state = -(target_dist * log_probs).sum(dim=-1)              # [B]
    if weights is not None:
        per_state = per_state * weights
    return per_state.sum()


# =====================================================================
# AlphaZero-style soft-target trainer extension
# =====================================================================
# Bolted onto the existing Trainer rather than living in a separate
# class because (a) it shares the same model + encoder + optimizer
# (one set of weights, one set of momenta) and (b) MCTS+REINFORCE
# co-training is a real option later (DeepMind's AlphaTensor uses it).
# Method-injection at module bottom keeps the diff to the REINFORCE
# core small.

def _unpack_visit(t) -> Tuple:
    """(actor, target, weapon, count, type) from a legacy 4-tuple or a
    5-tuple visit-count entry."""
    if len(t) >= 5:
        return t[0], t[1], t[2], t[3], t[4]
    return t[0], t[1], t[2], t[3], None


def _oob_index(kind: str, idx: int, size: int, actor_idx: int,
               num_units: int, decision_step: int) -> bool:
    """Stored-index bounds guard (2026-08-12). A visit-count tuple's
    indices were resolved against the SEARCH-time encoding; the loss
    re-encodes and re-masks, and a divergence puts an out-of-range
    index into a CUDA gather -- a device-side assert that kills the
    whole process with an ASYNC, misattributed traceback (observed
    once on the F1 leg, iter 34: Indexing.cu `srcIndex <
    srcSelectDimSize`). True = out of range: the caller skips the
    term; the divergence is logged with enough context to root-cause
    it (throttled after the first ten)."""
    if 0 <= idx < size:
        return False
    global _OOB_INDEX_EVENTS
    _OOB_INDEX_EVENTS += 1
    if _OOB_INDEX_EVENTS <= 10 or _OOB_INDEX_EVENTS % 200 == 0:
        log.error(
            f"visit-count {kind} index out of range: idx={idx} "
            f"size={size} actor_idx={actor_idx} "
            f"num_units={num_units} "
            f"decision_step={decision_step} "
            f"(occurrence #{_OOB_INDEX_EVENTS}; term skipped -- "
            f"search-time vs train-time index basis diverged; "
            f"root-cause before trusting this leg)")
    return True


def _mcts_factored_policy_loss_reference(
    encoded,
    output,
    game_state: GameState,
    visit_counts: List[Tuple],
    *,
    vectorized: bool = True,
    decision_step: int = 0,
) -> Tuple[torch.Tensor, float, float]:
    """One state's cross-entropy of the model's factored policy
    against MCTS visit counts, computed per state in Python: the
    REFERENCE for `_batched_factored_policy_loss` (which is what
    step_mcts runs; tests/test_batched_policy_loss.py pins the two
    together). Returns (loss, total_visits, mean -log p(actor)).

    The factored loss decomposes joint cross-entropy across four
    heads:
      log P(actor) + [log P(type | actor) for unit actors]
                   + log P(target | actor [, type]) + [log P(weapon | actor) for ATTACK]
    Mathematically identical to a flat joint CE; avoids
    materializing the A*T*H*MAX_ATTACKS joint.

    `vectorized=False` is the original per-tuple loop; True groups
    the terms per cached log-prob vector (same term set, float32
    summation reassociated).

    Visit-count tuple schema: 4-tuple (actor, target, weapon, count)
    on legacy data, 5-tuple (actor, target, weapon, count, type) on
    new data; both are accepted.
    """
    from wesnoth_ai.action_sampler import _masked_target_logits_from_row, _masked_type_logits

    _unpack = _unpack_visit

    masks = _build_legality_masks(encoded, game_state,
                                  decision_step=decision_step)

    def _oob(kind: str, idx: int, size: int, actor_idx: int) -> bool:
        return _oob_index(kind, idx, size, actor_idx, output.num_units,
                          decision_step)
    # Prior-bias symmetry: the trainer re-forward must apply the
    # SAME end_turn bias the rollout applied, or the CE would fight
    # a target the live priors never produced.
    actor_logits = _masked_actor_logits(
        encoded, output, masks.actor_valid,
        end_turn_bias=_prior_bias_end_turn(game_state))
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
    # Accumulated as a DETACHED tensor and read with a single .item()
    # before return, so neither path pays a per-visit D2H sync (300-900
    # per state on Gumbel roots) just to populate a log field.
    actor_nlp_t = actor_logp.new_zeros(())
    sum_actor_nlp = 0.0

    if not vectorized:
        # --- Original per-tuple accumulation (bit-exact reference) ---
        for tup in visit_counts:
            actor_idx, target_idx, weapon_idx, count, type_idx = _unpack(tup)
            if count <= 0:
                continue
            if _oob("actor", actor_idx, actor_logp.size(0), actor_idx):
                continue
            actor_term = count * actor_logp[actor_idx]
            nll = nll - actor_term
            actor_nlp_t = actor_nlp_t - actor_term.detach()

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
                if tylp is not None and not _oob(
                        "type", type_idx, tylp.size(0), actor_idx):
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
                if _oob("target", target_idx, tlp.size(0), actor_idx):
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
    else:
        # --- Optimization #5: vectorized accumulation ---------------
        # Pass 1 mirrors the loop above EXACTLY (same cache
        # population, same continue/skip decisions) but, instead of
        # building one autograd node per term, buckets (index, count)
        # pairs per cached log-prob vector. Pass 2 reduces each bucket
        # with a single index_select+sum -> O(unique vectors) graph
        # nodes. Identical term SET; only the float summation order
        # differs (gated; tolerance-tested).
        a_idx: List[int] = []
        a_cnt: List[float] = []
        type_b: Dict[int, Tuple[list, list]] = {}
        tgt_b: Dict[tuple, list] = {}   # (kind, actor) -> [vec, idxs, cnts]
        wpn_b: Dict[int, list] = {}     # actor -> [vec, idxs, cnts]
        for tup in visit_counts:
            actor_idx, target_idx, weapon_idx, count, type_idx = _unpack(tup)
            if count <= 0:
                continue
            if _oob("actor", actor_idx, actor_logp.size(0), actor_idx):
                continue
            a_idx.append(actor_idx)
            a_cnt.append(count)

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
                if tylp is not None and not _oob(
                        "type", type_idx, tylp.size(0), actor_idx):
                    tb = type_b.setdefault(actor_idx, ([], []))
                    tb[0].append(type_idx)
                    tb[1].append(count)

            if target_idx is not None:
                if is_unit and type_idx == UnitActionType.ATTACK:
                    kind = "a"
                    tlp = target_attack_logp_cache.get(actor_idx)
                    if tlp is None:
                        tl = _masked_target_logits_from_row(
                            output, masks.target_valid_attack[actor_idx],
                            actor_idx, attack_bias=masks.attack_bias[actor_idx],
                        )
                        tlp = F.log_softmax(tl, dim=-1) if tl.numel() else None
                        target_attack_logp_cache[actor_idx] = tlp
                elif is_unit and type_idx == UnitActionType.MOVE:
                    kind = "m"
                    tlp = target_move_logp_cache.get(actor_idx)
                    if tlp is None:
                        tl = _masked_target_logits_from_row(
                            output, masks.target_valid_move[actor_idx],
                            actor_idx, attack_bias=None,
                        )
                        tlp = F.log_softmax(tl, dim=-1) if tl.numel() else None
                        target_move_logp_cache[actor_idx] = tlp
                else:
                    kind = "u"
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
                if _oob("target", target_idx, tlp.size(0), actor_idx):
                    continue
                eb = tgt_b.setdefault((kind, actor_idx), [tlp, [], []])
                eb[1].append(target_idx)
                eb[2].append(count)

            if weapon_idx is not None:
                cached = weapon_logp_cache.get(actor_idx)
                if cached is None:
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
                    continue
                wb = wpn_b.setdefault(actor_idx, [wlp, [], []])
                wb[1].append(weapon_idx)
                wb[2].append(count)

        # Pass 2: one index_select + weighted sum per cached vector.
        _dev = actor_logp.device
        _dt = actor_logp.dtype

        def _term(vec, idxs, cnts):
            it = torch.as_tensor(idxs, dtype=torch.long, device=_dev)
            ct = torch.as_tensor(cnts, dtype=_dt, device=_dev)
            return (ct * vec.index_select(0, it)).sum()

        if a_idx:
            actor_term = _term(actor_logp, a_idx, a_cnt)
            nll = nll - actor_term
            actor_nlp_t = actor_nlp_t - actor_term.detach()
        for aidx, (idxs, cnts) in type_b.items():
            nll = nll - _term(type_logp_cache[aidx], idxs, cnts)
        for _key, (vec, idxs, cnts) in tgt_b.items():
            nll = nll - _term(vec, idxs, cnts)
        for aidx, (vec, idxs, cnts) in wpn_b.items():
            nll = nll - _term(vec, idxs, cnts)

    loss = nll / float(total_visits)
    # Single D2H sync per state for the diagnostic (was per visit tuple).
    sum_actor_nlp = float(actor_nlp_t.item())
    mean_actor_nlp = sum_actor_nlp / float(total_visits)
    return loss, float(total_visits), float(mean_actor_nlp)


# ---------------------------------------------------------------------
# Batched factored policy loss (2026-09-05)
# ---------------------------------------------------------------------
# The loss above, computed once per chunk instead of once per state:
# the legality masks are built on the host from each experience's
# RawEncoded, the index of every visit term and the mask rows those
# terms need go to the device in ONE flat buffer, and the device runs
# one masked log-softmax per head -- over the padded batch for the
# actor and type heads, over the gathered rows for the target and
# weapon heads -- followed by one weighted gather-sum. The per-state
# version cost 32-34 ms per experience in every configuration, the
# largest item of the training path (docs/box_specs.md "Training path
# cost (2026-09-05)").

_KIND_ATTACK, _KIND_MOVE, _KIND_UNION = 0, 1, 2


@dataclass
class _PolicyTargets:
    """One chunk's visit terms laid out for `_batched_factored_policy_loss`."""
    layout: FlatLayout
    host: torch.Tensor          # flat uint8 buffer holding every field
    n_rows: int                 # target rows: unique (sample, actor, kind)
    n_wrows: int                # weapon rows: unique (sample, actor)
    visits: float               # sum of total visits over the chunk


def _host_legality_masks(raw: RawEncoded, game_state: GameState,
                         decision_step: int):
    """One experience's legality masks as CPU tensors, built from its
    RawEncoded the way the search-time actors build them (a pure
    function of the observable state), so no per-experience device
    round trip is needed to stage them."""
    # wesnoth_ai does not import tools/ at module load.
    from tools.inference_seam import build_light_encoded
    light = build_light_encoded(raw, _CPU)
    light.hex_subset = bool(raw.hex_subset)
    return _build_legality_masks(light, game_state,
                                 decision_step=decision_step)


def _stage_policy_targets(
    chunk: Sequence[MCTSExperience],
    raws: Sequence[RawEncoded],
    sizes: Sequence[Tuple[int, int, int]],
    *,
    A_max: int, H_max: int, T: int, W: int,
    coef: Sequence[float],
    pin: bool,
) -> _PolicyTargets:
    """Host side. `coef[b]` is experience b's weight on the chunk loss
    (game weight x policy weight / total game weight); a term's
    coefficient is count x coef / the experience's total visits, so
    the device only sums coefficient x log-probability. Term
    selection and skipping mirror the reference exactly, including
    the out-of-range guard and the "stale weapon slot" skip."""
    B = len(chunk)
    actor_mask = np.zeros((B, A_max), dtype=np.uint8)
    actor_bias = np.zeros((B, A_max), dtype=np.float32)
    type_valid = np.zeros((B, A_max, T), dtype=np.uint8)
    type_bias = np.zeros((B, A_max, T), dtype=np.float32)
    # Flat index (into the head's flattened log-probabilities) and
    # coefficient per term; actor terms also keep the raw count for
    # the "entropy" log field.
    a_idx: List[int] = []
    a_coef: List[float] = []
    a_cnt: List[float] = []
    t_idx: List[int] = []
    t_coef: List[float] = []
    g_row: List[int] = []
    g_hex: List[int] = []
    g_coef: List[float] = []
    w_row: List[int] = []
    w_slot: List[int] = []
    w_coef: List[float] = []
    rows: Dict[Tuple[int, int, int], int] = {}      # (b, a, kind) -> row
    row_src: List[Tuple[np.ndarray, Optional[np.ndarray], int]] = []
    wrows: Dict[Tuple[int, int], int] = {}          # (b, a) -> weapon row
    wrow_natt: List[int] = []
    visits = 0.0
    for b, (e, raw) in enumerate(zip(chunk, raws)):
        U, R, H = sizes[b]
        A = U + R + 1
        vc = e.visit_counts
        total = float(sum(_unpack_visit(t)[3] for t in vc))
        if total <= 0.0:
            continue
        visits += total
        ds = int(getattr(e, "decision_step", 0))
        gs = e.game_state
        masks = _host_legality_masks(raw, gs, ds)
        ownership = np.concatenate([raw.unit_is_ours, raw.recruit_is_ours,
                                    np.ones(1, dtype=np.float32)])
        actor_mask[b, :A] = ((ownership != 0.0)
                             & (masks.actor_valid[0].numpy() != 0.0))
        actor_bias[b, A - 1] = _prior_bias_end_turn(gs)
        type_valid[b, :A] = masks.type_valid[0].numpy() != 0.0
        type_bias[b, :A] = masks.type_bias[0].numpy()
        attack_valid = masks.target_valid_attack.numpy()
        move_valid = masks.target_valid_move.numpy()
        union_valid = masks.target_valid.numpy()
        attack_bias = masks.attack_bias.numpy()
        by_id = None
        c = coef[b] / total
        for tup in vc:
            a, h, w, count, ty = _unpack_visit(tup)
            if count <= 0:
                continue
            if _oob_index("actor", a, A, a, U, ds):
                continue
            a_idx.append(b * A_max + a)
            a_coef.append(count * c)
            a_cnt.append(count)
            is_unit = a < U
            if is_unit and ty is not None and not _oob_index("type", ty, T, a, U, ds):
                t_idx.append((b * A_max + a) * T + ty)
                t_coef.append(count * c)
            if h is not None:
                if is_unit and ty == UnitActionType.ATTACK:
                    kind, valid, bias = _KIND_ATTACK, attack_valid, attack_bias
                elif is_unit and ty == UnitActionType.MOVE:
                    kind, valid, bias = _KIND_MOVE, move_valid, None
                else:
                    kind, valid, bias = _KIND_UNION, union_valid, attack_bias
                if H == 0:
                    continue
                if _oob_index("target", h, H, a, U, ds):
                    continue    # like the reference, the weapon term goes with it
                r = rows.get((b, a, kind))
                if r is None:
                    r = rows[(b, a, kind)] = len(row_src)
                    row_src.append((valid[a], None if bias is None else bias[a], H))
                g_row.append(r)
                g_hex.append(h)
                g_coef.append(count * c)
            if w is not None:
                if by_id is None:
                    by_id = {u.id: u for u in gs.map.units}
                unit = by_id.get(raw.unit_ids[a]) if is_unit else None
                n_att = len(unit.attacks) if unit is not None else 0
                if n_att <= 0 or w >= n_att:
                    continue    # stale visit-count slot: skipped silently
                if _oob_index("weapon", w, W, a, U, ds):
                    continue
                wr = wrows.get((b, a))
                if wr is None:
                    wr = wrows[(b, a)] = len(wrow_natt)
                    wrow_natt.append(n_att)
                w_row.append(wr)
                w_slot.append(w)
                w_coef.append(count * c)

    n_rows, n_wrows = len(row_src), len(wrow_natt)
    n_a, n_t, n_g, n_w = len(a_idx), len(t_idx), len(g_row), len(w_row)
    layout = FlatLayout([
        ("actor_idx", torch.int64, (n_a,)), ("type_idx", torch.int64, (n_t,)),
        ("tgt_idx", torch.int64, (n_g,)), ("wpn_idx", torch.int64, (n_w,)),
        ("row_b", torch.int64, (n_rows,)), ("row_a", torch.int64, (n_rows,)),
        ("wrow_b", torch.int64, (n_wrows,)), ("wrow_a", torch.int64, (n_wrows,)),
        ("actor_coef", torch.float32, (n_a,)), ("actor_cnt", torch.float32, (n_a,)),
        ("type_coef", torch.float32, (n_t,)), ("tgt_coef", torch.float32, (n_g,)),
        ("wpn_coef", torch.float32, (n_w,)),
        ("actor_bias", torch.float32, (B, A_max)),
        ("type_bias", torch.float32, (B, A_max, T)),
        ("row_bias", torch.float32, (n_rows, H_max)),
        ("row_hcount", torch.int32, (n_rows,)), ("wrow_natt", torch.int32, (n_wrows,)),
        ("actor_mask", torch.uint8, (B, A_max)),
        ("type_valid", torch.uint8, (B, A_max, T)),
        ("row_valid", torch.uint8, (n_rows, H_max)),
    ])
    host = torch.zeros(layout.nbytes, dtype=torch.uint8, pin_memory=pin)
    v = layout.numpy_views(host.numpy())
    v["actor_idx"][:] = a_idx
    v["actor_coef"][:] = a_coef
    v["actor_cnt"][:] = a_cnt
    v["type_idx"][:] = t_idx
    v["type_coef"][:] = t_coef
    v["tgt_idx"][:] = np.asarray(g_row, dtype=np.int64) * H_max + np.asarray(g_hex, dtype=np.int64)
    v["tgt_coef"][:] = g_coef
    v["wpn_idx"][:] = np.asarray(w_row, dtype=np.int64) * W + np.asarray(w_slot, dtype=np.int64)
    v["wpn_coef"][:] = w_coef
    v["row_b"][:] = [k[0] for k in rows]
    v["row_a"][:] = [k[1] for k in rows]
    for r, (valid_row, bias_row, Hb) in enumerate(row_src):
        v["row_valid"][r, :Hb] = valid_row != 0.0
        if bias_row is not None:
            v["row_bias"][r, :Hb] = bias_row
        v["row_hcount"][r] = Hb
    v["wrow_b"][:] = [k[0] for k in wrows]
    v["wrow_a"][:] = [k[1] for k in wrows]
    v["wrow_natt"][:] = wrow_natt
    v["actor_mask"][:] = actor_mask
    v["actor_bias"][:] = actor_bias
    v["type_valid"][:] = type_valid
    v["type_bias"][:] = type_bias
    return _PolicyTargets(layout=layout, host=host, n_rows=n_rows,
                          n_wrows=n_wrows, visits=visits)


def _batched_factored_policy_loss(
    padded, targets: _PolicyTargets,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Device side. `padded` is the chunk's model.PaddedOutput. Returns
    (the chunk's policy loss, already weighted and normalized; the
    detached sum of count x -log p(actor) for the "entropy" log
    field). A row whose mask has no legal entry is left unmasked
    within its state, as the reference does."""
    dev = padded.actor_logits.device
    v = targets.layout.torch_views(targets.host.to(dev, non_blocking=True))
    H_max = padded.target_logits.shape[2]
    W = padded.weapon_logits.shape[2]

    al = (padded.actor_logits + v["actor_bias"]).masked_fill(
        v["actor_mask"] == 0, _NEG_INF)
    a_vals = F.log_softmax(al, dim=-1).reshape(-1)[v["actor_idx"]]
    logp_sum = (v["actor_coef"] * a_vals).sum()
    actor_nlp = -(v["actor_cnt"] * a_vals.detach()).sum()

    if v["type_idx"].numel():
        tv = v["type_valid"] != 0
        ok = tv | ~tv.any(dim=-1, keepdim=True)
        tl = (padded.type_logits + v["type_bias"]).masked_fill(~ok, _NEG_INF)
        t_vals = F.log_softmax(tl, dim=-1).reshape(-1)[v["type_idx"]]
        logp_sum = logp_sum + (v["type_coef"] * t_vals).sum()

    if targets.n_rows:
        rows = padded.target_logits[v["row_b"], v["row_a"]] + v["row_bias"]
        rv = v["row_valid"] != 0
        in_state = (torch.arange(H_max, device=dev).unsqueeze(0)
                    < v["row_hcount"].unsqueeze(1))
        ok = torch.where(rv.any(dim=-1, keepdim=True), rv, in_state)
        g_vals = F.log_softmax(rows.masked_fill(~ok, _NEG_INF),
                               dim=-1).reshape(-1)[v["tgt_idx"]]
        logp_sum = logp_sum + (v["tgt_coef"] * g_vals).sum()

    if targets.n_wrows:
        wrows = padded.weapon_logits[v["wrow_b"], v["wrow_a"]]
        ok = (torch.arange(W, device=dev).unsqueeze(0)
              < v["wrow_natt"].unsqueeze(1))
        w_vals = F.log_softmax(wrows.masked_fill(~ok, _NEG_INF),
                               dim=-1).reshape(-1)[v["wpn_idx"]]
        logp_sum = logp_sum + (v["wpn_coef"] * w_vals).sum()

    return -logp_sum, actor_nlp


class _StageTimer:
    """Seconds per stage of one step_mcts call, added into `sink` at
    `flush` (keys STEP_MCTS_STAGES). On cuda every stage is a pair of
    events on the current stream, resolved after one synchronize --
    a host-bound stage is charged the stream's wait for it, a stage
    that overlaps queued device work only its non-overlapped part; on
    cpu perf_counter. No sink: every call is a no-op."""

    def __init__(self, device: torch.device, sink: Optional[Dict[str, float]]):
        self.sink = sink
        self.cuda = sink is not None and device.type == "cuda"
        self.seconds: Dict[str, float] = {}
        self._events: List[Tuple[str, object, object]] = []

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if self.sink is None:
            yield
            return
        if self.cuda:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                self._events.append((name, start, end))
        else:
            t0 = time.perf_counter()
            try:
                yield
            finally:
                self.seconds[name] = (self.seconds.get(name, 0.0)
                                      + time.perf_counter() - t0)

    def flush(self) -> None:
        if self.sink is None:
            return
        if self.cuda:
            torch.cuda.synchronize()
            for name, start, end in self._events:
                self.seconds[name] = (self.seconds.get(name, 0.0)
                                      + start.elapsed_time(end) / 1000.0)
            self._events.clear()
        for name, s in self.seconds.items():
            self.sink[name] = self.sink.get(name, 0.0) + s


def _training_autocast(enabled: bool):
    """The bf16 region of step_mcts's forward (TrainerConfig.
    train_autocast_bf16, cuda only); a null context when off."""
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16)


def _trainer_step_mcts(
    self,                                     # Trainer (method injected below)
    experiences: List[MCTSExperience],
    timings: Optional[Dict[str, float]] = None,
) -> TrainStats:
    """One AlphaZero-style gradient step. Each experience contributes
    a factored cross-entropy term against the MCTS visit
    distribution, plus a value term against the terminal outcome z.

    Like REINFORCE `step`, processes experiences in chunks of
    `train_batch_size` and calls `.backward()` per chunk to bound
    peak activation memory. Final `optimizer.step()` once at the end.

    `config.train_autocast_bf16` (cuda only) runs each chunk's encode
    and forward under bf16 autocast; the losses, the backward's
    parameter gradients, the clip and the optimizer step stay fp32.

    `timings`: seconds per stage (keys STEP_MCTS_STAGES) are ADDED to
    this dict; defaults to `self.stage_timings`. None = no timing.
    """
    if not experiences:
        return TrainStats()

    cap = self.config.max_transitions_per_step
    if len(experiences) > cap:
        # Subsample to a RANDOM subset, matching the REINFORCE path
        # (`step`, which moved off uniform stride deliberately).
        # Uniform stride correlates the kept set with episode position:
        # MCTS experiences are appended per-game in finalize_game order,
        # so stride-N would preferentially keep a fixed game-phase
        # subset and skew the value/policy target distribution. A random
        # subset keeps the kept batch's distribution matched to the full
        # set. (Usually inert under the documented replay-buffer recipe,
        # where minibatch <= cap; bites large non-replay iterations.)
        idxs = random.sample(range(len(experiences)), cap)
        experiences = [experiences[i] for i in idxs]

    dev = self.device or next(self.model.parameters()).device
    N = len(experiences)
    B = max(1, self.config.train_batch_size)
    # cuda only: the cpu path stays the fp32 reference (TrainerConfig).
    autocast_bf16 = bool(self.config.train_autocast_bf16) and dev.type == "cuda"

    # Clamp z values to the value head's range (matches REINFORCE
    # path's value_clip handling).
    zs = torch.tensor(
        [e.z for e in experiences], device=dev, dtype=torch.float32,
    )
    if self.config.value_clip is not None:
        zs.clamp_(min=-float(self.config.value_clip),
                  max=+float(self.config.value_clip))

    # Per-game normalization (see MCTSExperience.game_weight): every
    # loss term below is a weighted mean over states with these
    # weights, so each GAME sums to equal influence.
    gws = torch.tensor(
        [float(getattr(e, "game_weight", 1.0)) for e in experiences],
        device=dev, dtype=torch.float32)
    total_gw = max(float(gws.sum().item()), 1e-9)
    # Per-experience VALUE weight (truncation ruling 2026-08-17:
    # "there are no draws in real Wesnoth" -- a capped/stalled game
    # is a censored observation; finalize_game seals its states with
    # value_weight 0 so they feed the policy loss at full weight but
    # the value head not at all). Multiplies into BOTH value-side
    # weight builds below; legacy pickles default to 1.0.
    vws = torch.tensor(
        [float(getattr(e, "value_weight", 1.0)) for e in experiences],
        device=dev, dtype=torch.float32)
    # Arm VG2 label provenance: "consist" states are scalar
    # bootstraps -- they leave the categorical loss entirely and
    # train through the Gaussian term below; states with a
    # v_anchor enter the trust region.
    consist_mask = torch.tensor(
        [getattr(e, "label_kind", "game") == "consist"
         for e in experiences], device=dev)
    # value_weight is THE per-experience off-switch for every value-
    # side term (the profiler's and the in-loop telemetry's term
    # surgery zero it): the Gaussian consistency term and the trust
    # region honour it too, else a "policy-only" variant silently
    # carries them (VG3 iteration 0: sig_policy_norm == consist norm).
    vws_on = vws > 0
    consist_mask = consist_mask & vws_on
    vws = torch.where(consist_mask, torch.zeros_like(vws), vws)
    anchor_vals = [getattr(e, "v_anchor", None) for e in experiences]
    anchor_mask = torch.tensor([a is not None for a in anchor_vals],
                               device=dev) & vws_on
    anchor_t = torch.tensor([0.0 if a is None else float(a)
                             for a in anchor_vals],
                            device=dev, dtype=torch.float32)
    n_anchor = max(int(anchor_mask.sum().item()), 1)
    n_consist = max(int(consist_mask.sum().item()), 1)
    # Per-experience POLICY magnitude (see MCTSExperience.policy_weight):
    # multiplies the normalized per-state policy CE; deliberately not
    # in the denominator, so fractional weights shrink the update
    # instead of renormalizing back to full strength.
    pws = torch.tensor(
        [float(getattr(e, "policy_weight", 1.0)) for e in experiences],
        device=dev, dtype=torch.float32)

    # Auxiliary margin target (KataGo §3.5). Active only when the model
    # has the aux head, the weight is positive, AND every experience
    # carries a margin target -- otherwise the head/term is skipped
    # entirely (mixed or aux-off data trains exactly as before).
    aux_on = (
        self.config.aux_coef > 0
        and getattr(self.model, "has_aux_score", False)
        and all(getattr(e, "aux_target", None) is not None
                for e in experiences)
    )
    aux_t_full = (
        torch.tensor([e.aux_target for e in experiences],
                     device=dev, dtype=torch.float32)
        if aux_on else None
    )

    # Moves-left head (Lc0-style, 2026-07-04): same gating story as
    # the aux head -- head present + weight positive + every
    # experience carries a target, else the term vanishes entirely.
    ml_on = (
        self.config.moves_left_coef > 0
        and getattr(self.model, "has_moves_left", False)
        and all(getattr(e, "moves_left_target", None) is not None
                for e in experiences)
    )
    ml_t_full = (
        torch.tensor([e.moves_left_target for e in experiences],
                     device=dev, dtype=torch.float32)
        if ml_on else None
    )

    # GBC event supervision (2026-08-14, docs/archive/gbc_spec.md): per-
    # experience gate (labels may be absent on legacy/mixed data —
    # unlike aux, absence skips the EXPERIENCE, not the whole term).
    gbc_on = (
        self.config.gbc_coef > 0
        and getattr(self.model, "has_gbc", False)
        and any(getattr(e, "gbc_labels", None) for e in experiences)
    )

    self.optimizer.zero_grad()

    sum_policy_loss = 0.0
    sum_value_loss  = 0.0
    sum_aux_loss    = 0.0
    sum_ml_loss     = 0.0
    sum_gbc_loss    = 0.0
    sum_consist_loss = 0.0
    sum_trust_loss  = 0.0
    sum_total_visits = 0.0
    # Full-batch value-weight normalizer + "how many states actually
    # feed the value head" (dashboard starvation watch).
    # vws already carries the winnerless weight -- finalize_game is
    # the ONE authority (project round-1 C3: this second gate
    # multiplied it by draw_value_weight AGAIN, so the knob and the
    # tiebreak z were both nullified whenever finalize sealed 0).
    _w_full = gws * vws
    total_value_w = float(_w_full.sum().item())
    n_value_signal = int((_w_full > 0).sum().item())
    sum_actor_nlp_weighted = 0.0  # for "entropy"-style logging

    # Optimization #7 (2026-06-14): run the training forwards in eval()
    # (not train()), mirroring the REINFORCE step(). The model's 9
    # Dropout(p=1e-4) layers exist only to disable a torch fast path
    # (see model.py) and are a statistical no-op, so eval() loses no
    # meaningful regularization while skipping the dropout ops and
    # making the value-head training forward deterministic. The
    # unconditional eval() at function end already leaves the model in
    # eval() for the caller, so the post-condition is unchanged.
    self.model.eval()
    self.encoder.eval()

    timer = _StageTimer(dev, timings if timings is not None
                        else self.stage_timings)

    # Pre-compute the raw-encoded cache (one encode_raw per experience):
    # encode_from_raw_batch runs per chunk below to produce the grad-
    # tracked tensors, and the policy targets are staged from the same
    # RawEncoded (no second state walk).
    with timer.stage("encode_raw"):
        register_names = self.encoder.register_names
        for e in experiences:
            register_names(e.game_state)
        type_to_id    = self.encoder.unit_type_to_id
        faction_to_id = self.encoder.faction_to_id
        raw_cache = [
            encode_raw(e.game_state,
                       type_to_id=type_to_id,
                       faction_to_id=faction_to_id,
                       relevant_set=getattr(self.encoder,
                                            "relevant_set_hexes", False))
            for e in experiences
        ]
    # Each experience's weight on the policy loss (see
    # MCTSExperience.game_weight / policy_weight).
    policy_coef = (gws * pws / total_gw).tolist()

    for start in range(0, N, B):
        chunk = experiences[start:start + B]
        L = len(chunk)
        raw_chunk = raw_cache[start:start + B]
        # The bf16 region (config.train_autocast_bf16): the encoder's
        # projections, then the trunk and the heads. The model's own
        # inference switches (infer_autocast_bf16, the packed trunk)
        # do not apply to the training forward: this switch is the
        # only authority over its precision, and the padded trunk is
        # the one the backward runs through. Every output the losses
        # read is cast back to float32 as the region's last op, so
        # the loss stages below run in fp32 whatever the forward ran
        # in; a cast's backward runs in the dtype its forward used, so
        # the backward needs no context.
        with _training_autocast(autocast_bf16):
            with timer.stage("encode"):
                encoded_chunk = self.encoder.encode_from_raw_batch(raw_chunk)
            with timer.stage("forward"):
                padded = self.model.forward_padded(
                    encoded_chunk, autocast_bf16=False, packed=False)
                if autocast_bf16:
                    padded = padded.float32()
        # Staged on the host while the device runs the forward.
        with timer.stage("policy_loss"):
            targets = _stage_policy_targets(
                chunk, raw_chunk, padded.sizes,
                A_max=padded.actor_logits.shape[1],
                H_max=padded.target_logits.shape[2],
                T=padded.type_logits.shape[2],
                W=padded.weapon_logits.shape[2],
                coef=policy_coef[start:start + L],
                pin=dev.type == "cuda")
            policy_loss_t, actor_nlp_t = _batched_factored_policy_loss(
                padded, targets)
        sum_total_visits += targets.visits

        with timer.stage("value_loss"):
            chunk_loss, value_loss, chunk_sums = self._value_side_losses(
                padded, chunk, encoded_chunk, start, L,
                zs=zs, gws=gws, vws=vws, pws=pws, consist_mask=consist_mask,
                anchor_mask=anchor_mask, anchor_t=anchor_t,
                n_consist=n_consist, n_anchor=n_anchor,
                total_gw=total_gw, total_value_w=total_value_w,
                aux_on=aux_on, aux_t_full=aux_t_full,
                ml_on=ml_on, ml_t_full=ml_t_full, gbc_on=gbc_on)
            chunk_loss = policy_loss_t + chunk_loss
        sum_consist_loss += chunk_sums["consist"]
        sum_trust_loss += chunk_sums["trust"]
        sum_aux_loss += chunk_sums["aux"]
        sum_ml_loss += chunk_sums["ml"]
        sum_gbc_loss += chunk_sums["gbc"]

        with timer.stage("backward"):
            chunk_loss.backward()

        sum_policy_loss += float(policy_loss_t.item())
        sum_value_loss  += float(value_loss.item())
        sum_actor_nlp_weighted += float(actor_nlp_t.item())

        del chunk_loss, policy_loss_t, value_loss, targets
        del encoded_chunk, padded

    with timer.stage("clip"):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.encoder.parameters()),
            self.config.grad_clip,
        )
    with timer.stage("optimizer"):
        self.optimizer.step()
    timer.flush()

    self.model.eval()
    self.encoder.eval()

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
            + self.config.aux_coef   * sum_aux_loss
            + self.config.moves_left_coef * sum_ml_loss
            + self.config.gbc_coef   * sum_gbc_loss
        ),
        grad_norm      = float(grad_norm) if isinstance(grad_norm, float)
                         else float(grad_norm.item()),
        mean_return    = float(zs.mean().item()),
        n_transitions  = int(N),
        n_trajectories = int(N),  # one experience = one root state
        aux_loss       = float(sum_aux_loss),
        consist_loss   = float(sum_consist_loss),
        trust_loss     = float(sum_trust_loss),
        gbc_loss       = float(sum_gbc_loss),
        moves_left_loss = float(sum_ml_loss),
        value_signal_states = n_value_signal,
    )


def _trainer_value_side_losses(
    self, padded, chunk, encoded_chunk, start: int, L: int, *,
    zs, gws, vws, pws, consist_mask, anchor_mask, anchor_t,
    n_consist: int, n_anchor: int, total_gw: float, total_value_w: float,
    aux_on: bool, aux_t_full, ml_on: bool, ml_t_full, gbc_on: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Every value-side term of one chunk of step_mcts (value,
    consistency, trust region, aux margin, moves-left, GBC), summed
    with their coefficients. Returns (chunk sum, the plain value
    loss, the per-term floats for the step's running sums)."""
    gw_chunk = gws[start:start + L]
    val_t = padded.value.squeeze(-1)                 # [L]
    vl_t  = padded.value_logits                      # [L, K]
    z_t   = zs[start:start + L]
    sums = {"consist": 0.0, "trust": 0.0, "aux": 0.0, "ml": 0.0, "gbc": 0.0}
    # Distributional value loss: categorical CE on the projected
    # terminal-z target (z ∈ {-1, 0, +1} for win/draw/loss),
    # consistent with the REINFORCE path. `zs` was already
    # clipped to [V_MIN, V_MAX] by the value_clip block above.
    # Draws are down-weighted by config.draw_value_weight and the
    # loss normalizes by TOTAL WEIGHT (not N), so decisive states
    # keep full-strength gradient regardless of the batch's draw
    # share; an all-draw batch at weight 0 contributes no value
    # gradient at all.
    atoms = self.model._value_atoms
    w_t = gw_chunk * vws[start:start + L]
    if self.config.value_loss_form == "mse_mean":
        value_loss = ((val_t - z_t).pow(2) * w_t).sum() \
            / max(float(total_value_w), 1e-9)
    else:
        value_loss = _categorical_value_loss(
            vl_t, z_t, atoms,
            label_smoothing=self.config.value_label_smoothing,
            weights=w_t) / max(float(total_value_w), 1e-9)

    chunk_loss = self.config.value_coef * value_loss
    # Arm VG2 consistency term: Gaussian NLL of the head's MEAN
    # prediction against the bias-corrected search estimate,
    # (v - (z - b))^2 / (2 sigma2), b and sigma2 ESTIMATED from
    # paired labels by the policy each iteration. Gradient is
    # linear in the gap (no categorical blow-up), and the
    # term's strength IS the measured precision -- no weight.
    # Game-weight normalized like every other term.
    cm = consist_mask[start:start + L]
    if bool(cm.any()):
        tgt = z_t - float(self.config.consist_bias)
        sq = (val_t - tgt).pow(2) * gw_chunk * cm.float()
        consist_loss = sq.sum() / (
            2.0 * max(float(self.config.consist_sigma2), 1e-4)
            * n_consist)
        chunk_loss = chunk_loss + consist_loss
        sums["consist"] = float(consist_loss.item())
    # Trust region on consulted states: lambda * (v - v_anchor)^2,
    # lambda driven by dual ascent on the live dv_consult against
    # trust_delta (docs/design_constants.md).
    am = anchor_mask[start:start + L]
    if self.config.trust_lambda > 0 and bool(am.any()):
        tr = ((val_t - anchor_t[start:start + L]).pow(2)
              * am.float()).sum() / n_anchor
        trust_loss = self.config.trust_lambda * tr
        chunk_loss = chunk_loss + trust_loss
        sums["trust"] = float(trust_loss.item())
    # Auxiliary margin loss (KataGo §3.5): MSE of the predicted vs
    # final material margin, summed over the chunk and normalized by
    # N (matching the value-loss normalization), weighted by
    # aux_coef. Regularizes the shared trunk with a denser signal.
    if aux_on:
        aux_pred_t = padded.aux_score.squeeze(-1)
        aux_tgt_t  = aux_t_full[start:start + L]
        aux_loss = ((aux_pred_t - aux_tgt_t) ** 2
                    * gw_chunk).sum() / total_gw
        chunk_loss = chunk_loss + self.config.aux_coef * aux_loss
        sums["aux"] = float(aux_loss.item())
    # Moves-left loss (Lc0-style): MSE of the predicted vs actual
    # remaining-turn fraction; same normalization/weighting shape
    # as the aux term.
    if ml_on:
        ml_pred_t = padded.moves_left.squeeze(-1)
        ml_tgt_t  = ml_t_full[start:start + L]
        ml_loss = ((ml_pred_t - ml_tgt_t) ** 2
                   * gw_chunk).sum() / total_gw
        chunk_loss = chunk_loss + self.config.moves_left_coef * ml_loss
        sums["ml"] = float(ml_loss.item())
    # GBC event-supervision loss (2026-08-14, docs/archive/gbc_spec.md):
    # per-experience BCE of the dies/flips heads vs hindsight
    # labels, weighted by game_weight and normalized by total_gw
    # like every other term.
    if gbc_on:
        from wesnoth_ai.gbc import gbc_loss_for_output
        chunk_gbc: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for ei, e in enumerate(chunk):
            rows = getattr(e, "gbc_labels", None)
            if rows:
                gl = gbc_loss_for_output(
                    self.model, encoded_chunk[ei], padded.sample(ei), rows)
                if gl is not None:
                    chunk_gbc.append((gl, gws[start + ei]))
        if chunk_gbc:
            gbc_loss = sum(gl * w for gl, w in chunk_gbc) / total_gw
            chunk_loss = chunk_loss + self.config.gbc_coef * gbc_loss
            sums["gbc"] = float(gbc_loss.item())
        elif any(getattr(e, "gbc_labels", None) for e in chunk):
            # THIS CHUNK's labels are present but nothing computed a
            # loss -- the 2026-08-15 failure shape (ctx tap absent on
            # one forward path made GBC a silent no-op for hours).
            # Loud once per process; silence is the enemy here.
            # Chunks with NO labels at all are expected under value
            # grounding (label-free value-only experiences) and stay
            # quiet.
            global _GBC_SILENT_WARNED
            if not _GBC_SILENT_WARNED:
                _GBC_SILENT_WARNED = True
                log.warning(
                    "gbc_on but no GBC loss computed for this chunk "
                    "(ctx tap None? entities unresolvable?) -- the "
                    "aux signal is NOT training; investigate")
    return chunk_loss, value_loss, sums


def _trainer_step_value_from_raw(
    self,                                     # Trainer (method injected below)
    raws: list,                               # List[RawEncoded]
    zs: List[float],
    mls: Optional[List[Optional[float]]] = None,
) -> Dict[str, float]:
    """One gradient step training VALUE (+ moves-left) from
    pre-encoded RawEncoded inputs — no game_state, no policy loss.

    This is the human-corpus value fine-tune's train step: workers
    reconstruct + encode_raw games in parallel, the main process
    calls this with a batch of RawEncoded. The trunk gets gradient
    (full unfreeze), so it can learn win-predictive features rather
    than only re-mapping frozen ones. Policy heads receive no
    gradient (no policy term here), matching the visit-count-free
    human data.
    """
    import torch.nn.functional as F
    self.model.train()
    self.encoder.train()
    dev = self.device or next(self.model.parameters()).device
    N = len(raws)
    z_all = torch.tensor(zs, device=dev, dtype=torch.float32)
    if self.config.value_clip is not None:
        z_all.clamp_(min=-float(self.config.value_clip),
                     max=+float(self.config.value_clip))
    atoms = self.model._value_atoms
    ml_on = (getattr(self.model, "has_moves_left", False)
             and mls is not None and all(m is not None for m in mls))
    # Chunk the forward/backward by train_batch_size: each state is a
    # full-map token sequence (~1e3 tokens), so attention is O(S^2) and
    # a whole 256-batch at once OOMs even a 24GB GPU. Backward per
    # chunk accumulates grads; one optimizer.step() at the end == one
    # gradient step over the full batch (losses are /N).
    B = max(1, self.config.train_batch_size)
    self.optimizer.zero_grad()
    v_total = 0.0
    ml_total = 0.0
    for start in range(0, N, B):
        chunk = raws[start:start + B]
        encoded = self.encoder.encode_from_raw_batch(chunk, device=dev)
        outputs = self.model.forward_batch(encoded)
        vl_t = torch.stack([o.value_logits.squeeze(0) for o in outputs])
        z_t = z_all[start:start + len(chunk)]
        value_loss = _categorical_value_loss(
            vl_t, z_t, atoms,
            label_smoothing=self.config.value_label_smoothing) / N
        loss = self.config.value_coef * value_loss
        v_total += float(value_loss.item())
        if ml_on:
            ml_pred = torch.stack([o.moves_left.squeeze(-1)
                                   for o in outputs]).reshape(-1)
            ml_t = torch.tensor(mls[start:start + len(chunk)],
                                device=dev, dtype=torch.float32)
            ml_loss = F.mse_loss(ml_pred, ml_t, reduction="sum") / N
            loss = loss + self.config.moves_left_coef * ml_loss
            ml_total += float(ml_loss.item())
        loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(
        list(self.model.parameters())
        + list(self.encoder.parameters()),
        self.config.grad_clip))
    self.optimizer.step()
    return {"value_loss": v_total, "moves_left_loss": ml_total,
            "grad_norm": grad_norm}


def _trainer_values_from_raw(
    self,                                     # Trainer (method injected below)
    raws: list,
) -> List[float]:
    """No-grad E[V] per pre-encoded RawEncoded state (expectation of
    the C51 head over its atom support). Diagnostic helper."""
    if not raws:
        return []
    dev = self.device or next(self.model.parameters()).device
    B = max(1, self.config.train_batch_size)
    self.model.eval()
    self.encoder.eval()
    atoms = self.model._value_atoms
    out: List[float] = []
    with torch.no_grad():
        for start in range(0, len(raws), B):
            chunk = raws[start:start + B]
            encoded = self.encoder.encode_from_raw_batch(chunk, device=dev)
            outputs = self.model.forward_batch(encoded)
            vl_t = torch.stack([o.value_logits.squeeze(0) for o in outputs])
            probs = torch.softmax(vl_t, dim=-1)
            out.extend((probs * atoms).sum(dim=-1).tolist())
    return out


def _trainer_eval_value_metrics_from_raw(
    self,                                     # Trainer (method injected below)
    raws: list, zs: List[float],
) -> Dict[str, float]:
    """No-grad value diagnostics from pre-encoded RawEncoded inputs —
    the raw-fed twin of `eval_value_metrics` (same ce / pred_entropy /
    marginal_ce_floor definitions)."""
    nan = float("nan")
    if not raws:
        return {"ce": nan, "pred_entropy": nan, "marginal_ce_floor": nan}
    dev = self.device or next(self.model.parameters()).device
    N = len(raws)
    B = max(1, self.config.train_batch_size)
    z_all = torch.tensor(zs, device=dev, dtype=torch.float32)
    if self.config.value_clip is not None:
        z_all.clamp_(min=-float(self.config.value_clip),
                     max=+float(self.config.value_clip))
    self.model.eval()
    self.encoder.eval()
    atoms = self.model._value_atoms
    total = entropy_sum = 0.0
    with torch.no_grad():
        for start in range(0, N, B):
            chunk = raws[start:start + B]
            encoded = self.encoder.encode_from_raw_batch(chunk, device=dev)
            outputs = self.model.forward_batch(encoded)
            vl_t = torch.stack([o.value_logits.squeeze(0) for o in outputs])
            z_t = z_all[start:start + len(chunk)]
            total += float(_categorical_value_loss(vl_t, z_t, atoms).item()) / N
            logp = torch.log_softmax(vl_t, dim=-1)
            entropy_sum += float(-(logp.exp() * logp).sum().item())
        marginal = _project_returns_to_atoms(z_all, atoms).mean(dim=0)
        floor = float(-(marginal * marginal.clamp_min(1e-9).log()).sum().item())
    return {"ce": total, "pred_entropy": entropy_sum / N,
            "marginal_ce_floor": floor}


def _trainer_eval_value_metrics(
    self,                                     # Trainer (method injected below)
    experiences: List[MCTSExperience],
) -> Dict[str, float]:
    """No-grad value diagnostics of the CURRENT model on a fixed
    experience set, one forward pass:

    - "ce": categorical value CE as a GAME-WEIGHTED mean (same
      game_weight normalization as `step_mcts`, so mixed-length
      batches compare; NOTE step_mcts's value term additionally
      applies draw_value_weight, which this probe deliberately does
      NOT -- an all-draw holdout must still produce a number).
    - "pred_entropy": mean entropy of the predicted Z(s) distribution
      (nats; uniform over K=51 atoms = ln 51 ~ 3.93). The continuous
      overconfidence curve -- the 2026-07-07 diagnosis needed offline
      checkpoint probes for this.
    - "marginal_ce_floor": CE of the best STATE-BLIND predictor (the
      batch's empirical projected-z mixture) = entropy of that
      mixture. A learned head should score BELOW this; ce >> floor
      means worse-than-marginal, while a high floor says the games'
      outcomes are inherently mixed (e.g. coin-flip self-play) and
      caps what any head can achieve on this batch.

    Purpose: held-out generalization tracking. `step_mcts`'s logged
    value loss is measured on replay-buffer samples the net has
    already taken many gradient steps on, so it conflates "learning
    the value function" with "fitting the buffer's specific states".
    Evaluating on states that never entered training separates the
    two (see MCTSPolicy holdout diversion + fresh-probe).
    """
    nan = float("nan")
    if not experiences:
        return {"ce": nan, "ce_std": nan, "pred_entropy": nan,
                "marginal_ce_floor": nan, "value_auc": nan,
                "n_decisive": 0, "by_decade": {}}
    dev = self.device or next(self.model.parameters()).device
    N = len(experiences)
    B = max(1, self.config.train_batch_size)
    zs = torch.tensor(
        [e.z for e in experiences], device=dev, dtype=torch.float32,
    )
    if self.config.value_clip is not None:
        zs.clamp_(min=-float(self.config.value_clip),
                  max=+float(self.config.value_clip))
    self.model.eval()
    self.encoder.eval()
    register_names = self.encoder.register_names
    for e in experiences:
        register_names(e.game_state)
    type_to_id    = self.encoder.unit_type_to_id
    faction_to_id = self.encoder.faction_to_id
    atoms = self.model._value_atoms
    gws_e = torch.tensor(
        [float(getattr(e, "game_weight", 1.0)) for e in experiences],
        device=dev, dtype=torch.float32)
    total_gw_e = max(float(gws_e.sum().item()), 1e-9)
    total = 0.0
    entropy_sum = 0.0
    # Per-state CE values, for the weighted spread ("ce_std").
    # fresh_value_ce is the DEFAULT success metric (user 2026-07-22);
    # the std says whether an iteration-to-iteration move is signal
    # or probe noise (~256-state sample).
    ce_states: List[torch.Tensor] = []
    ev_states: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, N, B):
            chunk = experiences[start:start + B]
            raw_chunk = [
                encode_raw(e.game_state,
                           type_to_id=type_to_id,
                           faction_to_id=faction_to_id,
                           relevant_set=getattr(self.encoder,
                                                "relevant_set_hexes",
                                                False))
                for e in chunk
            ]
            encoded_chunk = self.encoder.encode_from_raw_batch(raw_chunk)
            outputs = self.model.forward_batch(encoded_chunk)
            vl_t = torch.stack(
                [o.value_logits.squeeze(0) for o in outputs])
            z_t = zs[start:start + len(chunk)]
            gw_c = gws_e[start:start + len(chunk)]
            total += float(_categorical_value_loss(
                vl_t, z_t, atoms, weights=gw_c).item()) / total_gw_e
            logp = torch.log_softmax(vl_t, dim=-1)
            entropy_sum += float(-(logp.exp() * logp).sum().item())
            ce_states.append(
                -(_project_returns_to_atoms(z_t, atoms) * logp)
                .sum(dim=-1))
            ev_states.append((logp.exp() * atoms).sum(dim=-1))
        marginal = _project_returns_to_atoms(zs, atoms).mean(dim=0)
        floor = float(
            -(marginal * marginal.clamp_min(1e-9).log()).sum().item())
        ce_all = torch.cat(ce_states)
        # Per-turn-decade decomposition (user ruling 2026-09-01):
        # the pooled CE mixes phases whose labels have very
        # different information content (early-game +-1 outcomes
        # are substantially aleatoric). Per decade: weighted CE,
        # the decade's own state-blind floor, outcome AUC, and n.
        by_decade: Dict[str, Dict] = {}
        turn_nos = torch.tensor(
            [int(e.game_state.global_info.turn_number)
             for e in experiences])
        dec_idx = ((turn_nos - 1) // 10).clamp(0, 6)
        ce_cpu = ce_all.cpu()
        ev_cpu = torch.cat(ev_states).cpu()
        zs_cpu = zs.cpu()
        gw_cpu = gws_e.cpu()
        atoms_cpu = atoms.cpu()
        for d in range(7):
            m = dec_idx == d
            nd = int(m.sum().item())
            if nd == 0:
                continue
            key = f"d{d * 10 + 1}_{d * 10 + 10}" if d < 6 else "d61p"
            gw_d = gw_cpu[m]
            tot_d = max(float(gw_d.sum().item()), 1e-9)
            ce_d = float((ce_cpu[m] * gw_d).sum().item()) / tot_d
            marg_d = _project_returns_to_atoms(
                zs_cpu[m], atoms_cpu).mean(dim=0)
            floor_d = float(
                -(marg_d * marg_d.clamp_min(1e-9).log()).sum().item())
            pos_d = ev_cpu[m & (zs_cpu > 0)]
            neg_d = ev_cpu[m & (zs_cpu < 0)]
            if len(pos_d) and len(neg_d):
                gt_d = (pos_d.unsqueeze(1)
                        > neg_d.unsqueeze(0)).float().sum()
                eq_d = (pos_d.unsqueeze(1)
                        == neg_d.unsqueeze(0)).float().sum()
                auc_d = float((gt_d + 0.5 * eq_d).item()) \
                    / (len(pos_d) * len(neg_d))
            else:
                auc_d = nan
            by_decade[key] = {"ce": ce_d, "floor": floor_d,
                              "auc": auc_d, "n": nd}
        # gw-weighted spread around the gw-weighted mean (matches
        # how "ce" itself is normalized).
        mean_w = float((ce_all * gws_e).sum().item()) / total_gw_e
        var_w = float(((ce_all - mean_w).pow(2) * gws_e).sum().item()) \
            / total_gw_e
        ce_std = var_w ** 0.5
        # Outcome AUC on decisive states (A1/A3 gate metric,
        # 2026-08-17): P(E[V] of a random win-state > E[V] of a
        # random loss-state), ties at 0.5 -- the level-discrimination
        # statistic the launch gate and the value-seed acceptance
        # test read. NaN when either class is absent.
        ev_all = torch.cat(ev_states)
        pos = ev_all[zs > 0]
        neg = ev_all[zs < 0]
        if len(pos) and len(neg):
            gt = (pos.unsqueeze(1) > neg.unsqueeze(0)).float().sum()
            eq = (pos.unsqueeze(1) == neg.unsqueeze(0)).float().sum()
            auc = float((gt + 0.5 * eq).item()) / (len(pos) * len(neg))
        else:
            auc = nan
    return {"ce": total, "ce_std": ce_std,
            "pred_entropy": entropy_sum / N,
            "marginal_ce_floor": floor,
            "value_auc": auc,
            "n_decisive": int((zs != 0).sum().item()),
            "by_decade": by_decade}


def _trainer_eval_value_loss(
    self,
    experiences: List[MCTSExperience],
) -> float:
    """Back-compat scalar wrapper (holdout probe + tripwire callers):
    just the "ce" of eval_value_metrics."""
    return _trainer_eval_value_metrics(self, experiences)["ce"]


# Inject as a method on Trainer. Gives users `trainer.step_mcts(exps)`
# alongside the REINFORCE `trainer.step(trajectories)`.
Trainer.step_mcts = _trainer_step_mcts
Trainer._value_side_losses = _trainer_value_side_losses
Trainer.eval_value_loss = _trainer_eval_value_loss
Trainer.eval_value_metrics = _trainer_eval_value_metrics
Trainer.values_from_raw = _trainer_values_from_raw
Trainer.step_value_from_raw = _trainer_step_value_from_raw
Trainer.eval_value_metrics_from_raw = _trainer_eval_value_metrics_from_raw
