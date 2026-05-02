"""Sample an action from a model's output, and re-compute the same
action's log-prob / entropy under updated weights at training time.

Two public entry points for the two phases of a typical on-policy RL
loop:

- ``sample_action(encoded, output, game_state)`` — stochastic sampling
  during rollout. Returns a ``SampledAction`` carrying the action dict
  AND the slot indices we chose (actor_idx, target_idx, weapon_idx).
  Intended to be called under ``torch.no_grad()`` — doesn't build a
  retained graph.

- ``reforward_logprob_entropy(encoded, output, game_state, actor_idx,
  target_idx, weapon_idx)`` — at training time, given stored indices,
  build grad-tracked log_prob and entropy tensors for the action we
  took, using the CURRENT model logits. This is the crucial trick
  that lets the trainer re-do the forward pass (with grads) without
  keeping thousands of forward graphs in memory across rollouts.

Masking rules (shared between the two paths):
  - Actor must be our side's (unit or recruit) or end_turn.
  - Actors with NO valid targets (e.g. a unit with 0 moves and
    has_attacked, or a recruit when leader isn't on a keep) are
    masked out. This is required for correctness — sampling an
    all-invalid actor would make every action a no-op.
  - Target hex mask per actor:
      * Unit actor: target must be the unit's own hex excluded,
        friendly-occupied excluded; empty hex requires
        `hex_distance(unit, hex) <= current_moves`; enemy-occupied
        hex requires `!has_attacked` and
        `hex_distance(unit, enemy) <= current_moves + 1` (can reach
        a neighbor to attack from).
      * Recruit actor: BFS from leader's keep through connected
        castle/keep hexes; targets are empty hexes in that network.
        Leader must be on a keep for any recruit to be valid.
      * End-turn: no target sampled.
    The mask is a CHEAP-UPPER-BOUND filter: anything we mark invalid
    is truly invalid, but some things we leave valid (e.g. moves
    into ZOC, impassable terrain) will still be rejected by Wesnoth.
    Residual invalidity is acceptable.
  - Weapon index is capped to the attacker's actual attack count.

Computing the masks is deterministic in game_state, so both paths
(sampling and re-forward) rebuild them rather than storing them in
the Transition. Cost is ~0.1-1ms per decision with the vectorized
numpy helper below — see profile for current numbers.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from classes import GameState, Position, TerrainModifiers, Unit
from combat_oracle import expected_attack_net_damage
from encoder import EncodedState
from model import ActorKind, ModelOutput
from rewards import hex_distance


log = logging.getLogger("action_sampler")


# Combat-oracle alpha schedule. See constants.py for full
# explanation. The TARGET alpha shifts attack-target logits
# (current behavior); the TYPE alpha shifts P(ATTACK | actor) on
# the type head (new in C.3). Both anneal as a multiplicative
# fraction of their configured value over a horizon of training
# decisions, with a floor (default 0.1×) so the bias persists at
# small strength forever.
from constants import (
    COMBAT_TARGET_ALPHA,
    COMBAT_TYPE_ALPHA,
    COMBAT_ANNEAL_HORIZON,
    COMBAT_ANNEAL_FLOOR_FRACTION,
)


def combat_alphas_at(decision_step: int) -> Tuple[float, float]:
    """Return (target_alpha, type_alpha) at training step
    `decision_step`. Linear decay from the configured value at
    decision 0 to `FLOOR × configured` at HORIZON; flat at FLOOR
    afterward. With HORIZON=0 the schedule degenerates to the
    configured values.

    Decoupling the schedule from the constants lets the trainer
    (or any caller that holds a decision counter) anneal without
    monkey-patching globals. Per-decision-step granularity matches
    the unit the sampler operates on.
    """
    if COMBAT_ANNEAL_HORIZON <= 0:
        return COMBAT_TARGET_ALPHA, COMBAT_TYPE_ALPHA
    floor = COMBAT_ANNEAL_FLOOR_FRACTION
    if decision_step <= 0:
        frac = 1.0
    elif decision_step >= COMBAT_ANNEAL_HORIZON:
        frac = floor
    else:
        # Linear from 1.0 at step 0 to `floor` at horizon.
        progress = decision_step / COMBAT_ANNEAL_HORIZON
        frac = 1.0 - progress * (1.0 - floor)
    return COMBAT_TARGET_ALPHA * frac, COMBAT_TYPE_ALPHA * frac


# Legacy alias kept for any external caller that imported the old
# name. New code should call `combat_alphas_at(step)`.
_COMBAT_LOGIT_ALPHA = COMBAT_TARGET_ALPHA


# Safer-than-torch.inf fill for "this slot is invalid". -inf sometimes
# produces NaNs in downstream softmax under pathological conditions.
_NEG_INF = -1e9


def _sample_from_logits(logits: torch.Tensor) -> int:
    """Draw one sample from a 1-D logits tensor via Gumbel-max.

    Avoids torch.distributions.Categorical.sample() because some
    backends (notably torch-directml's RX-6600 path) don't implement
    the multinomial/scatter ops Categorical uses. Gumbel-max is
    `argmax(logits + Gumbel(0,1))` and lands on the same distribution
    using only basic elementwise ops + argmax.
    """
    # Gumbel(0,1): -log(-log(U)) where U ~ Uniform(0,1).
    u = torch.rand_like(logits).clamp_(min=1e-9, max=1.0 - 1e-9)
    gumbel = -torch.log(-torch.log(u))
    return int((logits + gumbel).argmax().item())


def _logprob_entropy_1d(logits: torch.Tensor, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (log_prob, entropy) for a categorical over a 1-D logits
    vector, using only log_softmax + gather + elementwise ops. No
    scatter, no multinomial — DirectML-friendly."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    idx_t = torch.tensor(idx, device=logits.device, dtype=torch.long)
    log_prob = log_probs.gather(0, idx_t.unsqueeze(0)).squeeze(0)
    entropy = -(probs * log_probs).sum()
    return log_prob, entropy


@dataclass
class ActionPriors:
    """Legality-masked, normalized action priors for MCTS / PUCT.

    Shapes (batch dim = 1):
      A = num_units + num_recruits + 1       (actor slots)
      H = num_hex_tokens
      T = UnitActionType.COUNT (= 2: ATTACK, MOVE)
      W = MAX_ATTACKS

    Each conditional sums to 1 over its legal support, with mass on
    illegal actions exactly 0. For non-UNIT actors, `type_`,
    `target_attack`, and `target_move` are all-zero (those dims
    are undefined for RECRUIT / END_TURN). For non-RECRUIT actors,
    `target_recruit` is all-zero. For non-UNIT actors, `weapon` is
    all-zero. Degenerate rows (no legal options for a conditional)
    are also zero rather than NaN.

    PUCT consumes these as a factored joint:
      P(end_turn)              = actor[0, A-1]
      P(unit u attacks h, w)   = actor[0, u] * type_[0, u, ATTACK]
                                 * target_attack[0, u, h]
                                 * weapon[0, u, w]
      P(unit u moves to h)     = actor[0, u] * type_[0, u, MOVE]
                                 * target_move[0, u, h]
      P(recruit slot r at h)   = actor[0, U+r] * target_recruit[0, U+r, h]

    `masks` is the underlying LegalityMasks; carries decision-step-
    dependent combat-oracle biases that the soft-target trainer
    needs to reproduce these priors against the same logits.
    """
    actor:          torch.Tensor   # [1, A]
    type_:          torch.Tensor   # [1, A, T]
    target_attack:  torch.Tensor   # [1, A, H]
    target_move:    torch.Tensor   # [1, A, H]
    target_recruit: torch.Tensor   # [1, A, H]
    weapon:         torch.Tensor   # [1, A, W]
    value:          torch.Tensor   # [1, 1]
    masks:          "LegalityMasks"


@dataclass
class SampledAction:
    """What sample_action returns.

    `action` is the internal-format dict (0-indexed Positions) that
    game_manager hands to Lua. The `*_idx` fields record which slot
    of each distribution we drew from — the trainer replays these to
    compute the grad-enabled log_prob at training time.
    """
    action:      Dict
    actor_idx:   int
    type_idx:    Optional[int]   # None unless actor is UNIT (then ATTACK/MOVE)
    target_idx:  Optional[int]   # None iff action.type == 'end_turn'
    weapon_idx:  Optional[int]   # None unless action.type == 'attack'
    value_est:   float           # pooled from output.value; debug/logging


# ---------------------------------------------------------------------
# Sampling (no grads expected)
# ---------------------------------------------------------------------

def sample_action(
    encoded:    EncodedState,
    output:     ModelOutput,
    game_state: GameState,
    *,
    decision_step: int = 0,
) -> SampledAction:
    """Stochastic sampling. Caller should be in torch.no_grad context
    to avoid retaining the forward graph — this function only reads
    logits and samples indices.

    Chain (for UNIT actors): actor -> type (ATTACK/MOVE) -> target.
    For RECRUIT: actor -> target (recruit hex). For END_TURN: actor.

    The type sample uses `output.type_logits[actor]` masked by
    `masks.type_valid[actor]`; the target sample uses the
    type-conditional mask (`target_valid_attack` or
    `target_valid_move`). This separates "should I attack at all?"
    from "if I'm attacking, where?", giving cleaner PUCT priors and
    cleaner supervised CE.
    """
    from model import UnitActionType

    device = output.actor_logits.device
    value_est = float(output.value.squeeze().item())

    masks = _build_legality_masks(encoded, game_state,
                                  decision_step=decision_step)

    actor_logits = _masked_actor_logits(encoded, output, masks.actor_valid)
    actor_idx = _sample_from_logits(actor_logits.squeeze(0))

    kind = int(output.actor_kind[0, actor_idx].item())

    if kind == ActorKind.END_TURN:
        return SampledAction(
            action={'type': 'end_turn'},
            actor_idx=actor_idx, type_idx=None,
            target_idx=None, weapon_idx=None,
            value_est=value_est,
        )

    if kind == ActorKind.RECRUIT:
        # Recruit: no type sample (type is implicit). Sample target
        # from the union mask (recruit-hex network).
        target_logits = _masked_target_logits(
            output, masks.target_valid, actor_idx,
            attack_bias=masks.attack_bias,
        )
        if target_logits.numel() == 0:
            return SampledAction(
                action={'type': 'end_turn'},
                actor_idx=actor_idx, type_idx=None,
                target_idx=None, weapon_idx=None,
                value_est=value_est,
            )
        target_idx = _sample_from_logits(target_logits)
        target_pos = encoded.hex_positions[target_idx]
        recruit_type = encoded.recruit_types[actor_idx - output.num_units]
        return SampledAction(
            action={
                'type':       'recruit',
                'unit_type':  recruit_type,
                'target_hex': target_pos,
            },
            actor_idx=actor_idx, type_idx=None,
            target_idx=target_idx, weapon_idx=None,
            value_est=value_est,
        )

    # ActorKind.UNIT: type -> target (with type-conditional mask)
    # -> weapon (only for ATTACK).
    type_logits = _masked_type_logits(
        output, masks.type_valid, actor_idx, type_bias=masks.type_bias,
    )
    if type_logits.numel() == 0 or torch.isinf(type_logits).all():
        # No legal type for this unit (shouldn't happen since
        # actor_valid already gates on that, but guard against
        # empty distributions). Fall through to end_turn.
        return SampledAction(
            action={'type': 'end_turn'},
            actor_idx=actor_idx, type_idx=None,
            target_idx=None, weapon_idx=None,
            value_est=value_est,
        )
    type_idx = _sample_from_logits(type_logits)

    # Pick the type-conditional target mask.
    if type_idx == UnitActionType.ATTACK:
        type_target_row = masks.target_valid_attack[actor_idx]
    else:
        type_target_row = masks.target_valid_move[actor_idx]
    target_logits = _masked_target_logits_from_row(
        output, type_target_row, actor_idx,
        attack_bias=(masks.attack_bias[actor_idx]
                     if type_idx == UnitActionType.ATTACK else None),
    )
    if target_logits.numel() == 0:
        return SampledAction(
            action={'type': 'end_turn'},
            actor_idx=actor_idx, type_idx=type_idx,
            target_idx=None, weapon_idx=None,
            value_est=value_est,
        )
    target_idx = _sample_from_logits(target_logits)
    target_pos = encoded.hex_positions[target_idx]
    unit_pos = encoded.unit_positions[actor_idx]

    if type_idx == UnitActionType.MOVE:
        return SampledAction(
            action={'type': 'move',
                    'start_hex': unit_pos,
                    'target_hex': target_pos},
            actor_idx=actor_idx, type_idx=type_idx,
            target_idx=target_idx, weapon_idx=None,
            value_est=value_est,
        )

    # ATTACK: pick a weapon.
    unit_id = encoded.unit_ids[actor_idx]
    attacker = _unit_by_id(game_state, unit_id)
    num_attacks = len(attacker.attacks) if attacker else 0
    if num_attacks == 0:
        # Shouldn't happen (mask gated on attackability via
        # has_attacked + adjacency), but degrade safely to end_turn
        # rather than emit an attack with no weapon.
        return SampledAction(
            action={'type': 'end_turn'},
            actor_idx=actor_idx, type_idx=type_idx,
            target_idx=None, weapon_idx=None,
            value_est=value_est,
        )
    weapon_logits = _masked_weapon_logits(output, actor_idx, num_attacks)
    weapon_idx = _sample_from_logits(weapon_logits)
    return SampledAction(
        action={
            'type':         'attack',
            'start_hex':    unit_pos,
            'target_hex':   target_pos,
            'attack_index': weapon_idx,
        },
        actor_idx=actor_idx, type_idx=type_idx,
        target_idx=target_idx, weapon_idx=weapon_idx,
        value_est=value_est,
    )


# ---------------------------------------------------------------------
# Re-forward: grad-enabled log_prob / entropy for a stored action
# ---------------------------------------------------------------------

def reforward_logprob_entropy(
    encoded:    EncodedState,
    output:     ModelOutput,
    game_state: GameState,
    *,
    actor_idx:  int,
    target_idx: Optional[int],
    weapon_idx: Optional[int],
    type_idx:   Optional[int] = None,
    decision_step: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Given stored action indices, compute (log_prob, entropy) as
    scalar grad-enabled tensors. The trainer sums these across a
    batch to form the policy-gradient loss.

    Chain (must mirror `sample_action`):
        actor -> [type if UNIT] -> target -> [weapon if ATTACK]

    `type_idx` is None for non-UNIT actors (recruit / end_turn) and
    for legacy callers that pre-date the type head; in both cases
    we fall back to the old chain-rule (actor -> target -> weapon)
    so transitions stored before this change keep training without
    re-collection. Once new transitions land, type_idx is non-None
    for unit actions.

    Must be called with grads enabled (normal mode, NOT under no_grad).
    """
    from model import UnitActionType

    masks = _build_legality_masks(encoded, game_state,
                                  decision_step=decision_step)

    actor_logits = _masked_actor_logits(encoded, output, masks.actor_valid)
    log_prob, entropy = _logprob_entropy_1d(actor_logits.squeeze(0), actor_idx)

    if target_idx is None:
        return log_prob, entropy

    # Type term -- only for UNIT actors with a stored type_idx.
    is_unit_actor = actor_idx < output.num_units
    if is_unit_actor and type_idx is not None:
        type_logits = _masked_type_logits(
            output, masks.type_valid, actor_idx, type_bias=masks.type_bias,
        )
        lp_type, ent_type = _logprob_entropy_1d(type_logits, type_idx)
        log_prob = log_prob + lp_type
        entropy  = entropy  + ent_type

        # Target term: type-conditional mask.
        if type_idx == UnitActionType.ATTACK:
            row = masks.target_valid_attack[actor_idx]
            attack_bias_row = masks.attack_bias[actor_idx]
        else:
            row = masks.target_valid_move[actor_idx]
            attack_bias_row = None
        target_logits = _masked_target_logits_from_row(
            output, row, actor_idx, attack_bias=attack_bias_row,
        )
    else:
        # Recruit / end_turn / legacy transitions: use the union mask.
        target_logits = _masked_target_logits(
            output, masks.target_valid, actor_idx,
            attack_bias=masks.attack_bias,
        )
    lp_t, ent_t = _logprob_entropy_1d(target_logits, target_idx)
    log_prob = log_prob + lp_t
    entropy = entropy + ent_t

    if weapon_idx is None:
        return log_prob, entropy

    # Defensive bounds check: `encoded.unit_ids` indexes only the on-
    # board unit slots (length = output.num_units). Recruit slots use
    # `actor_idx >= num_units` and `encoded.recruit_types[actor_idx -
    # num_units]` instead. Today the `weapon_idx is None` short-circuit
    # above catches recruit actions (they have no weapon), but a future
    # change that gives recruits a weapon idx would IndexError here.
    # Bail safely with the partial log_prob/entropy we already have so
    # the trainer's loss accumulator stays well-defined.
    if actor_idx >= len(encoded.unit_ids):
        log.debug(
            f"reforward: actor_idx={actor_idx} is a recruit slot but "
            f"weapon_idx={weapon_idx} is set; skipping weapon term")
        return log_prob, entropy

    # Weapon mask needs the attacker's actual attack count — same as
    # at sampling time. We look up the unit by its slot index (which
    # indexes into encoded.unit_ids).
    unit_id = encoded.unit_ids[actor_idx]
    attacker = _unit_by_id(game_state, unit_id)
    num_attacks = len(attacker.attacks) if attacker else 0
    if num_attacks == 0:
        # Shouldn't happen if collection was consistent with state;
        # bail with current log_prob/entropy rather than failing loudly.
        return log_prob, entropy

    weapon_logits = _masked_weapon_logits(output, actor_idx, num_attacks)
    lp_w, ent_w = _logprob_entropy_1d(weapon_logits, weapon_idx)
    log_prob = log_prob + lp_w
    entropy = entropy + ent_w
    return log_prob, entropy


# ---------------------------------------------------------------------
# MCTS expansion: enumerate every legal action with its prior
# ---------------------------------------------------------------------
# The single-action sampler (`sample_action`) is fine for REINFORCE,
# but MCTS-PUCT needs the FULL prior distribution over the legal
# action set: per node expansion, `argmax(Q + cpuct*P*sqrt(N)/(1+n))`
# scores every legal child, and visit-count training distills `P`
# back from the search's empirical distribution.
#
# The factorization is:
#   P(action) = P(actor) * P(target | actor) [* P(weapon | actor) for attacks]
# All three factors come from masked softmaxes (same masks the
# sampler uses, so legality is consistent between sampling and MCTS
# expansion).

@dataclass
class LegalActionPrior:
    """One element of the MCTS expansion list. `prior` is the joint
    probability under the current policy; the trainer's distillation
    target replaces it with MCTS visit counts at training time.

    `actor_idx / type_idx / target_idx / weapon_idx` are the same
    indices the REINFORCE trainer uses via
    `reforward_logprob_entropy`, so MCTS + REINFORCE can share
    trainer code without re-deriving them. `type_idx` is the
    UnitActionType (ATTACK / MOVE) for unit actors; None for
    recruit / end_turn."""
    action:     Dict
    prior:      float
    actor_idx:  int
    target_idx: Optional[int]
    weapon_idx: Optional[int]
    type_idx:   Optional[int] = None


def enumerate_legal_actions_with_priors(
    encoded:    EncodedState,
    output:     ModelOutput,
    game_state: GameState,
    *,
    decision_step: int = 0,
) -> List[LegalActionPrior]:
    """Enumerate every legal action with its joint prior under the
    model's chain rule:

      P(action) = P(actor)
                * P(type | actor)        # only for UNIT actors
                * P(target | actor, type)
                * P(weapon | actor)      # only for ATTACK

    For RECRUIT actors: P(actor) * P(target | actor) (no type, no weapon).
    For END_TURN: just P(actor).

    Caller MUST be in `torch.no_grad()` (inference-only path).

    Priors over the returned list sum to ~1; MCTS treats them as a
    normalized probability distribution.
    """
    from model import UnitActionType

    masks = _build_legality_masks(encoded, game_state,
                                  decision_step=decision_step)
    actor_logits = _masked_actor_logits(encoded, output, masks.actor_valid)
    actor_p = F.softmax(actor_logits.squeeze(0), dim=-1)  # [A]
    A = actor_p.shape[0]

    actor_kind = output.actor_kind[0]      # [A] long
    num_units = output.num_units
    actor_valid_row = masks.actor_valid[0]   # [A] float

    out: List[LegalActionPrior] = []
    for actor_idx in range(A):
        # Skip actors the legality mask filtered out -- their masked
        # logit was -inf, so their softmax probability is 0 too. The
        # explicit skip is just to avoid the per-target inner loop.
        if float(actor_valid_row[actor_idx].item()) == 0.0:
            continue
        p_actor = float(actor_p[actor_idx].item())
        if p_actor <= 0.0:
            continue
        kind = int(actor_kind[actor_idx].item())

        if kind == ActorKind.END_TURN:
            out.append(LegalActionPrior(
                action={"type": "end_turn"},
                prior=p_actor,
                actor_idx=actor_idx, target_idx=None, weapon_idx=None,
                type_idx=None,
            ))
            continue

        if kind == ActorKind.RECRUIT:
            target_logits = _masked_target_logits(
                output, masks.target_valid, actor_idx,
                attack_bias=masks.attack_bias,
            )
            if target_logits.numel() == 0:
                continue
            target_p = F.softmax(target_logits, dim=-1)  # [H]
            recruit_type = encoded.recruit_types[actor_idx - num_units]
            for target_idx in range(target_p.shape[0]):
                p_t = float(target_p[target_idx].item())
                if p_t <= 0.0:
                    continue
                target_pos = encoded.hex_positions[target_idx]
                out.append(LegalActionPrior(
                    action={
                        "type": "recruit",
                        "unit_type": recruit_type,
                        "target_hex": target_pos,
                    },
                    prior=p_actor * p_t,
                    actor_idx=actor_idx, target_idx=target_idx,
                    weapon_idx=None, type_idx=None,
                ))
            continue

        # ActorKind.UNIT: chain over (type, target, [weapon]).
        unit_pos = encoded.unit_positions[actor_idx]
        unit_id  = encoded.unit_ids[actor_idx]
        attacker = _unit_by_id(game_state, unit_id)
        num_attacks = len(attacker.attacks) if attacker else 0

        # Type distribution.
        type_logits = _masked_type_logits(
            output, masks.type_valid, actor_idx, type_bias=masks.type_bias,
        )
        if type_logits.numel() == 0:
            continue
        type_p = F.softmax(type_logits, dim=-1)  # [T]

        # Pre-compute weapon distribution once for this unit (used
        # for all ATTACK targets).
        weapon_p = None
        if num_attacks > 0:
            weapon_logits = _masked_weapon_logits(output, actor_idx, num_attacks)
            weapon_p = F.softmax(weapon_logits, dim=-1)

        # ATTACK branch.
        p_attack = float(type_p[UnitActionType.ATTACK].item())
        if p_attack > 0.0 and num_attacks > 0:
            attack_target_logits = _masked_target_logits_from_row(
                output, masks.target_valid_attack[actor_idx], actor_idx,
                attack_bias=masks.attack_bias[actor_idx],
            )
            if attack_target_logits.numel() > 0:
                attack_target_p = F.softmax(attack_target_logits, dim=-1)
                for target_idx in range(attack_target_p.shape[0]):
                    p_t = float(attack_target_p[target_idx].item())
                    if p_t <= 0.0:
                        continue
                    target_pos = encoded.hex_positions[target_idx]
                    joint_atype = p_actor * p_attack * p_t
                    for weapon_idx in range(num_attacks):
                        p_w = float(weapon_p[weapon_idx].item())
                        if p_w <= 0.0:
                            continue
                        out.append(LegalActionPrior(
                            action={
                                "type":         "attack",
                                "start_hex":    unit_pos,
                                "target_hex":   target_pos,
                                "attack_index": weapon_idx,
                            },
                            prior=joint_atype * p_w,
                            actor_idx=actor_idx,
                            target_idx=target_idx,
                            weapon_idx=weapon_idx,
                            type_idx=UnitActionType.ATTACK,
                        ))

        # MOVE branch.
        p_move = float(type_p[UnitActionType.MOVE].item())
        if p_move > 0.0:
            move_target_logits = _masked_target_logits_from_row(
                output, masks.target_valid_move[actor_idx], actor_idx,
                attack_bias=None,
            )
            if move_target_logits.numel() > 0:
                move_target_p = F.softmax(move_target_logits, dim=-1)
                for target_idx in range(move_target_p.shape[0]):
                    p_t = float(move_target_p[target_idx].item())
                    if p_t <= 0.0:
                        continue
                    target_pos = encoded.hex_positions[target_idx]
                    out.append(LegalActionPrior(
                        action={
                            "type":       "move",
                            "start_hex":  unit_pos,
                            "target_hex": target_pos,
                        },
                        prior=p_actor * p_move * p_t,
                        actor_idx=actor_idx,
                        target_idx=target_idx, weapon_idx=None,
                        type_idx=UnitActionType.MOVE,
                    ))
    return out


# ---------------------------------------------------------------------
# Helpers (shared by sample + reforward)
# ---------------------------------------------------------------------

def _masked_actor_logits(
    encoded: EncodedState,
    output: ModelOutput,
    actor_valid: torch.Tensor,
) -> torch.Tensor:
    """Mask actor logits by both side-ownership and legality.

    `actor_valid` is [1, A] with 1 = this actor can produce at least one
    valid action. Combined with the ownership mask so only OUR actors
    with a valid target survive. End_turn is always both.
    """
    device = output.actor_logits.device
    ownership = torch.cat([
        encoded.unit_is_ours,     # [1, U]
        encoded.recruit_is_ours,  # [1, R]
        torch.ones(1, 1, device=device),  # end_turn always ours
    ], dim=1)
    mask = ownership * actor_valid
    return output.actor_logits.masked_fill(mask == 0, _NEG_INF)


def _masked_type_logits(
    output: ModelOutput,
    type_valid: torch.Tensor,    # [1, A, T] float
    actor_idx: int,
    type_bias: Optional[torch.Tensor] = None,    # [1, A, T] or None
) -> torch.Tensor:
    """Pull `output.type_logits[actor]` and mask types whose
    legality bit is zero. Returns [T] tensor; if every type is
    masked off, returns the unmasked row (caller should treat
    that as a degenerate state and degrade gracefully).

    `type_bias`: combat-oracle prior on type selection (currently
    only ATTACK gets a non-zero bias). Added to the row BEFORE
    masking, mirroring how `attack_bias` is composed into target
    logits."""
    row = output.type_logits[0, actor_idx]   # [T]
    if row.numel() == 0:
        return row
    if type_bias is not None:
        row = row + type_bias[0, actor_idx]
    mask = type_valid[0, actor_idx]          # [T]
    if mask.sum().item() == 0:
        return row
    return row.masked_fill(mask == 0, _NEG_INF)


def _masked_target_logits_from_row(
    output: ModelOutput,
    target_valid_row: torch.Tensor,    # [H] float for ONE actor
    actor_idx: int,
    attack_bias: Optional[torch.Tensor] = None,    # [H] or None
) -> torch.Tensor:
    """Variant of `_masked_target_logits` that takes a pre-sliced
    [H] row instead of the full [A, H] tensor. Used by the new
    type-conditional sampling path where we choose between
    `target_valid_attack[actor]` and `target_valid_move[actor]`
    BEFORE the masked logit comes out."""
    row = output.target_logits[0, actor_idx]  # [H]
    if row.numel() == 0:
        return row
    if attack_bias is not None:
        row = row + attack_bias
    if target_valid_row.sum().item() == 0:
        return row
    return row.masked_fill(target_valid_row == 0, _NEG_INF)


def _masked_target_logits(
    output: ModelOutput,
    target_valid: torch.Tensor,
    actor_idx: int,
    attack_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pull the target-logit row for `actor_idx`, mask invalid hexes,
    and (optionally) add the combat-oracle attack bias.

    target_valid is [A, H]. If the row is all-zero (shouldn't happen
    because _masked_actor_logits already filtered out such actors),
    we return the unmasked logits to keep sampling well-defined.
    """
    row = output.target_logits[0, actor_idx]  # [H]
    if row.numel() == 0:
        return row
    # Fold in the combat oracle BEFORE masking, so the bias doesn't
    # leak through on illegal positions. attack_bias is zero on non-
    # attack-candidate hexes, so it's a no-op for moves/recruits.
    if attack_bias is not None:
        row = row + attack_bias[actor_idx]
    mask = target_valid[actor_idx]  # [H] float
    if mask.sum().item() == 0:
        # No valid targets for this actor — shouldn't reach here because
        # actor masking would have zeroed the actor's probability. Falling
        # back to unmasked avoids a zero-probability Categorical.
        return row
    return row.masked_fill(mask == 0, _NEG_INF)


def _masked_weapon_logits(
    output: ModelOutput, actor_idx: int, num_attacks: int,
) -> torch.Tensor:
    weapons = output.weapon_logits[0, actor_idx].clone()
    if num_attacks < weapons.shape[0]:
        weapons[num_attacks:] = _NEG_INF
    return weapons


def _safe_softmax(logits: torch.Tensor, dim: int) -> torch.Tensor:
    """Softmax that returns 0 for slices with no legal entries.

    Without this:
      - All entries == -inf -> softmax NaN
      - All entries == _NEG_INF (-1e9) -> softmax uniform (= wrong:
        every entry is "equally illegal", but the right answer is
        zero mass everywhere)

    Detect the all-masked case by checking if the max of the slice
    is below a threshold real logits won't reach (-1e8 << -1e9 won't
    happen organically) and zero those slices' outputs. Partially-
    masked slices (some entries at _NEG_INF, others real) work
    correctly under plain softmax because exp(-1e9) ≈ 0."""
    max_v = logits.max(dim=dim, keepdim=True).values
    has_legal = (max_v > -1e8).to(logits.dtype)
    p = F.softmax(logits, dim=dim)
    p = torch.where(torch.isnan(p), torch.zeros_like(p), p)
    return p * has_legal


def _weapon_count_mask(
    encoded: EncodedState,
    game_state: GameState,
    *,
    A: int,
    W: int,
    device,
) -> torch.Tensor:
    """[1, A, W] float. 1 = weapon slot is within the actor's actual
    attack count (UNIT actor), 0 elsewhere (slots beyond an actor's
    weapons, OR non-UNIT actors -- their weapon distribution is
    undefined). Mirrors the per-decision masking sample_action does
    via `_masked_weapon_logits`. Without this, weapon slots beyond
    the actor's attack count keep their uninitialized noise (no
    gradient flow in supervised since reforward_logprob_entropy
    masks them) and leak into the prior."""
    mask = np.zeros((A, W), dtype=np.float32)
    for u_idx, uid in enumerate(encoded.unit_ids):
        unit = _unit_by_id(game_state, uid)
        if unit is None:
            continue
        n = min(len(unit.attacks), W)
        if n > 0:
            mask[u_idx, :n] = 1.0
    return torch.from_numpy(mask).to(device).unsqueeze(0)


def predict_priors(
    output:     ModelOutput,
    encoded:    EncodedState,
    game_state: GameState,
    *,
    decision_step: int = 0,
    masks:         Optional["LegalityMasks"] = None,
) -> ActionPriors:
    """Compute legality-masked, normalized action priors for MCTS.

    Mirrors `sample_action`'s decision chain but exposes the
    *distributions* rather than sampling one. Output is a factored
    prior (actor, type|actor, target|actor,type, weapon|actor) --
    PUCT multiplies factors as needed when expanding child actions.

    Doesn't call torch.no_grad(); the caller decides. The trainer's
    soft-target loss path needs a grad-tracked variant, so leaving
    the gradient choice to the caller keeps one code path.

    Re-uses an existing `LegalityMasks` if passed -- handy for MCTS
    nodes that cache masks alongside priors so re-visits don't
    rebuild them.
    """
    from model import ActorKind  # local: avoid model<->sampler cycle

    if masks is None:
        masks = _build_legality_masks(encoded, game_state,
                                      decision_step=decision_step)

    device = output.actor_logits.device

    # 1) Actor distribution.
    actor_logits = _masked_actor_logits(encoded, output, masks.actor_valid)
    actor_p = _safe_softmax(actor_logits, dim=-1)            # [1, A]

    A = actor_logits.size(-1)
    H = output.target_logits.size(-1)
    W = output.weapon_logits.size(-1)

    # 2) Type | actor (UNIT actors only). type_valid is 0 for
    # non-UNIT rows, so masking yields all-(-inf) → safe_softmax → 0.
    type_logits = output.type_logits + masks.type_bias
    type_logits = type_logits.masked_fill(masks.type_valid == 0, _NEG_INF)
    type_p = _safe_softmax(type_logits, dim=-1)              # [1, A, T]

    # 3) Target | actor, type. UNIT/ATTACK and UNIT/MOVE use the
    # type-conditional masks; the attack bias adds to ATTACK-target
    # logits only.
    target_attack_logits = (output.target_logits
                            + masks.attack_bias.unsqueeze(0))
    target_attack_logits = target_attack_logits.masked_fill(
        masks.target_valid_attack.unsqueeze(0) == 0, _NEG_INF,
    )
    target_attack_p = _safe_softmax(target_attack_logits, dim=-1)

    target_move_logits = output.target_logits.masked_fill(
        masks.target_valid_move.unsqueeze(0) == 0, _NEG_INF,
    )
    target_move_p = _safe_softmax(target_move_logits, dim=-1)

    # 4) Target | recruit_actor. RECRUIT actors store the recruit
    # hex set in masks.target_valid (the union mask); UNIT rows of
    # target_valid are the union of attack/move and we don't want
    # to mix those into the recruit prior, so gate by actor_kind.
    actor_kind = output.actor_kind                            # [1, A]
    is_recruit = (actor_kind == ActorKind.RECRUIT).float()    # [1, A]
    recruit_mask = (masks.target_valid.unsqueeze(0)
                    * is_recruit.unsqueeze(-1))               # [1, A, H]
    target_recruit_logits = output.target_logits.masked_fill(
        recruit_mask == 0, _NEG_INF,
    )
    target_recruit_p = _safe_softmax(target_recruit_logits, dim=-1)

    # 5) Weapon | actor. Mask slots beyond each unit's attack count.
    weapon_mask = _weapon_count_mask(encoded, game_state, A=A, W=W,
                                     device=device)
    weapon_logits = output.weapon_logits.masked_fill(
        weapon_mask == 0, _NEG_INF,
    )
    weapon_p = _safe_softmax(weapon_logits, dim=-1)           # [1, A, W]

    return ActionPriors(
        actor=actor_p,
        type_=type_p,
        target_attack=target_attack_p,
        target_move=target_move_p,
        target_recruit=target_recruit_p,
        weapon=weapon_p,
        value=output.value,
        masks=masks,
    )


def _enemy_unit_at(gs: GameState, pos: Position) -> Optional[Unit]:
    current_side = gs.global_info.current_side
    for u in gs.map.units:
        if (u.position.x == pos.x and u.position.y == pos.y
                and u.side != current_side):
            return u
    return None


def _unit_by_id(gs: GameState, unit_id: str) -> Optional[Unit]:
    for u in gs.map.units:
        if u.id == unit_id:
            return u
    return None


# ---------------------------------------------------------------------
# Legality masks — produced once per decision
# ---------------------------------------------------------------------

@dataclass
class LegalityMasks:
    """Per-decision legality: which actor slots can act, which target
    hexes are valid for each actor. Both are float tensors for
    straightforward multiplication with logits.

    `attack_bias` is an optional additive prior in logit-space: for
    each (actor, enemy-hex) the combat oracle's expected-net-damage,
    scaled by `_COMBAT_LOGIT_ALPHA`. Zero everywhere else (moves,
    recruits, non-attack pairs). Added to target_logits at sample
    time so learned logits + combat prior jointly drive attack
    targeting.
    """
    actor_valid:  torch.Tensor  # [1, A] float — 1 = actor has ≥1 valid target
    target_valid: torch.Tensor  # [A, H] float — 1 = hex is a valid target
                                #                  (union of attack ∪ move
                                #                  for unit actors; recruit-
                                #                  hex set for recruit
                                #                  actors). Kept for
                                #                  legacy callers; new
                                #                  code should consult
                                #                  the type-conditional
                                #                  pair below.
    target_valid_attack: torch.Tensor  # [A, H] -- enemy-attack legal
    target_valid_move:   torch.Tensor  # [A, H] -- empty-move legal
    type_valid: torch.Tensor    # [1, A, T] float -- 1 = action type
                                #                  legal for actor.
                                #                  Computed from the
                                #                  per-type target masks:
                                #                  ATTACK legal iff
                                #                  target_valid_attack
                                #                  has any 1; MOVE
                                #                  similarly. Always 0
                                #                  for non-UNIT actors.
    type_bias:    torch.Tensor  # [1, A, T] float -- combat-oracle prior
                                # ON THE TYPE distribution: raises
                                # P(ATTACK | actor) when at least one
                                # reachable enemy gives positive
                                # expected net damage. type_bias[..., MOVE]
                                # is always 0. Scaled by the annealed
                                # COMBAT_TYPE_ALPHA at compute time.
    attack_bias:  torch.Tensor  # [A, H] float — combat-oracle logit prior
                                # ON THE TARGET distribution (within
                                # ATTACK type). Scaled by the annealed
                                # COMBAT_TARGET_ALPHA at compute time.


def _build_legality_masks(
    encoded: EncodedState, game_state: GameState,
    *, decision_step: int = 0,
) -> LegalityMasks:
    """Assemble actor + target validity masks from the game state.

    Uses numpy internally to vectorize the hex-distance and occupancy
    checks. Transfers to the model's device at the end. Cost budget
    is ~1ms per decision on Caves-of-the-Basilisk scale; if that
    grows, the per-unit loop is the thing to flatten further.

    `decision_step`: per-Python-decision counter the trainer / policy
    threads through. Drives the combat-oracle alpha schedule (see
    `combat_alphas_at`). Default 0 = full-strength oracle.
    """
    target_alpha, type_alpha = combat_alphas_at(decision_step)
    device = encoded.unit_is_ours.device
    U = encoded.unit_tokens.size(1)
    R = encoded.recruit_tokens.size(1)
    H = encoded.hex_tokens.size(1)
    A = U + R + 1  # +1 for end_turn
    current_side = game_state.global_info.current_side

    from model import UnitActionType
    T = UnitActionType.COUNT
    actor_valid_np  = np.zeros(A, dtype=np.float32)
    target_valid_np = np.zeros((A, H), dtype=np.float32)
    target_attack_np = np.zeros((A, H), dtype=np.float32)
    target_move_np   = np.zeros((A, H), dtype=np.float32)
    type_valid_np   = np.zeros((A, T), dtype=np.float32)
    type_bias_np    = np.zeros((A, T), dtype=np.float32)
    attack_bias_np  = np.zeros((A, H), dtype=np.float32)

    # End_turn actor is the LAST slot; always valid, no target.
    actor_valid_np[A - 1] = 1.0

    if H == 0:
        return LegalityMasks(
            actor_valid  = torch.from_numpy(actor_valid_np).to(device).unsqueeze(0),
            target_valid = torch.from_numpy(target_valid_np).to(device),
            target_valid_attack = torch.from_numpy(target_attack_np).to(device),
            target_valid_move   = torch.from_numpy(target_move_np).to(device),
            type_valid   = torch.from_numpy(type_valid_np).to(device).unsqueeze(0),
            type_bias    = torch.from_numpy(type_bias_np).to(device).unsqueeze(0),
            attack_bias  = torch.from_numpy(attack_bias_np).to(device),
        )

    # Per-hex arrays used by every subsequent check.
    hex_xs = np.array([p.x for p in encoded.hex_positions], dtype=np.int32)
    hex_ys = np.array([p.y for p in encoded.hex_positions], dtype=np.int32)

    # Occupancy per hex: 0 empty, 1 friendly, 2 enemy. Lookups off the
    # hex-position list make sure we only mark hexes the encoder emits.
    # `pos_to_hex` is now cached on EncodedState (encoder builds it
    # once per decision instead of here per call). Saves rebuild
    # work when reforward_logprob_entropy or
    # enumerate_legal_actions_with_priors hits the same encoded
    # state multiple times.
    pos_to_hex = encoded.pos_to_hex
    # `occupancy`: 0=empty, 1=friendly, 2=attackable enemy, 3=inert
    # (occupies a hex for movement/recruit purposes but cannot be
    # attacked). Inert covers petrified scenery units (statues on
    # Thousand Stings Garrison / Caves of the Basilisk) -- Wesnoth's
    # mouse_events.cpp:753 sets `target_eligible &= !target_unit->
    # incapacitated();` so a petrified unit is NOT a legal click-to-
    # attack target. We mirror that here so the policy never picks a
    # statue as an attack target.
    occupancy = np.zeros(H, dtype=np.int8)
    unit_at: Dict[Tuple[int, int], Unit] = {}
    for u in game_state.map.units:
        key = (u.position.x, u.position.y)
        unit_at[key] = u
        j = pos_to_hex.get(key)
        if j is None:
            continue
        if u.side == current_side:
            occupancy[j] = 1
        elif "petrified" in (u.statuses or set()):
            occupancy[j] = 3
        else:
            occupancy[j] = 2

    friendly_mask = occupancy == 1
    enemy_mask    = occupancy == 2
    empty_mask    = occupancy == 0

    unit_id_to_obj = {u.id: u for u in game_state.map.units}

    # ----- Unit actors (slots 0..U-1) -----
    for i in range(U):
        uid = encoded.unit_ids[i]
        u = unit_id_to_obj.get(uid)
        if u is None or u.side != current_side:
            continue
        can_move   = u.current_moves > 0
        can_attack = not u.has_attacked
        if not (can_move or can_attack):
            continue
        ux, uy = u.position.x, u.position.y
        moves = u.current_moves

        dist = _hex_distance_vec(ux, uy, hex_xs, hex_ys)
        self_mask = (hex_xs == ux) & (hex_ys == uy)

        # Type-conditional target masks. UNIT actors split their
        # legal targets across ATTACK (enemy-occupied within
        # move+1 hexes) and MOVE (empty-and-reachable, excluding
        # our own current hex).
        move_row   = np.zeros(H, dtype=bool)
        attack_row = np.zeros(H, dtype=bool)
        if can_move:
            move_row = empty_mask & ~self_mask & (dist <= moves)
        if can_attack:
            attack_row = enemy_mask & (dist <= moves + 1)
            # Combat-oracle priors: for each valid attack target, score
            # the expected net damage using the unit's best weapon.
            #   - Per-target bias: feed scaled net into attack_bias_np
            #     so the target softmax leans toward favorable trades.
            #   - Per-actor type bias: aggregate (max over targets)
            #     the same scores into type_bias_np[ATTACK] so the
            #     type softmax leans toward "attack at all" when at
            #     least one favorable trade exists.
            if attack_row.any():
                best_score = float("-inf")
                for j in np.where(attack_row)[0]:
                    ex, ey = int(hex_xs[j]), int(hex_ys[j])
                    enemy_u = unit_at.get((ex, ey))
                    if enemy_u is None:
                        continue
                    try:
                        net = expected_attack_net_damage(u, enemy_u)
                    except Exception:
                        net = 0.0
                    attack_bias_np[i, j] = target_alpha * net
                    if net > best_score:
                        best_score = net
                # Type bias: only positive scores motivate the "attack"
                # nudge. A reachable enemy with NEGATIVE expected
                # net damage shouldn't bias the policy toward
                # attacking; pinning to max(0, best) keeps the bias
                # one-sided.
                if best_score > 0.0 and best_score != float("-inf"):
                    type_bias_np[i, UnitActionType.ATTACK] = (
                        type_alpha * best_score
                    )

        # Per-type legality + union into legacy target_valid_np.
        if attack_row.any():
            target_attack_np[i] = attack_row.astype(np.float32)
            type_valid_np[i, UnitActionType.ATTACK] = 1.0
        if move_row.any():
            target_move_np[i] = move_row.astype(np.float32)
            type_valid_np[i, UnitActionType.MOVE] = 1.0
        union_row = attack_row | move_row
        if union_row.any():
            target_valid_np[i] = union_row.astype(np.float32)
            actor_valid_np[i] = 1.0

    # ----- Recruit actors (slots U..U+R-1) -----
    if R > 0:
        leader = next(
            (u for u in game_state.map.units
             if u.side == current_side and u.is_leader),
            None,
        )
        leader_on_keep = False
        if leader is not None:
            lhex_idx = pos_to_hex.get((leader.position.x, leader.position.y))
            if lhex_idx is not None:
                # Pull the actual Hex to inspect modifiers.
                lhex = next(
                    (h for h in game_state.map.hexes
                     if h.position.x == leader.position.x
                     and h.position.y == leader.position.y),
                    None,
                )
                if lhex is not None and TerrainModifiers.KEEP in lhex.modifiers:
                    leader_on_keep = True

        if leader_on_keep and leader is not None:
            # Side gold for affordability gating. Slots whose unit
            # type costs more than this side's current gold get
            # actor_valid=0; the policy never wastes a decision on an
            # unaffordable recruit.
            side_gold = 0
            side_idx = current_side - 1
            if 0 <= side_idx < len(game_state.sides):
                side_gold = int(game_state.sides[side_idx].current_gold)
            # Per-turn rejection history (per the CLAUDE.md
            # legality-mask contract): hexes a previous recruit
            # attempt bounced on this turn. Subtracted from the
            # recruit hex mask so the policy can't re-attempt the
            # same fog-occupied hex within the turn. Cleared at
            # init_side, so next turn the hex is available again.
            rejected = (
                getattr(game_state.global_info,
                        "_recruit_rejected_hexes", None) or set()
            )
            recruit_hex_row = _recruit_hex_mask(
                game_state, pos_to_hex, unit_at, leader, H,
                rejected_hexes=rejected,
            )
            if recruit_hex_row.any():
                recruit_is_ours_np = encoded.recruit_is_ours.detach().cpu().numpy()[0]
                # Lazy import to avoid a circular dep at module-load
                # time (action_sampler is imported from many places
                # and tools/wesnoth_sim ultimately imports
                # action_sampler too).
                from tools.wesnoth_sim import _recruit_cost_for
                for r_off in range(R):
                    if recruit_is_ours_np[r_off] == 0:
                        continue
                    unit_type = encoded.recruit_types[r_off]
                    cost = _recruit_cost_for(unit_type)
                    if cost > side_gold:
                        # Unaffordable -- mask off entirely.
                        # actor_valid_np[a] stays 0 and target row
                        # all zeros.
                        continue
                    a = U + r_off
                    target_valid_np[a] = recruit_hex_row.astype(np.float32)
                    actor_valid_np[a] = 1.0

    return LegalityMasks(
        actor_valid  = torch.from_numpy(actor_valid_np).to(device).unsqueeze(0),
        target_valid = torch.from_numpy(target_valid_np).to(device),
        target_valid_attack = torch.from_numpy(target_attack_np).to(device),
        target_valid_move   = torch.from_numpy(target_move_np).to(device),
        type_valid   = torch.from_numpy(type_valid_np).to(device).unsqueeze(0),
        type_bias    = torch.from_numpy(type_bias_np).to(device).unsqueeze(0),
        attack_bias  = torch.from_numpy(attack_bias_np).to(device),
    )


def _hex_distance_vec(ax: int, ay: int,
                      bxs: np.ndarray, bys: np.ndarray) -> np.ndarray:
    """Vectorized odd-q hex distance — matches rewards.hex_distance.

    Kept local (rather than promoting to a shared util) to avoid a
    round-trip import cycle and because the scalar version already
    lives next to the reward code that needs it.
    """
    hd = np.abs(ax - bxs)
    a_even = (ax & 1) == 0
    b_even = (bxs & 1) == 0
    if a_even:
        cond1 = (~b_even) & (ay <= bys)
        cond2 = np.zeros_like(bxs, dtype=bool)
    else:
        cond1 = np.zeros_like(bxs, dtype=bool)
        cond2 = b_even & (bys <= ay)
    vpenalty = np.where(cond1 | cond2, 1, 0)
    return np.maximum(hd, np.abs(ay - bys) + (hd // 2) + vpenalty)


def _hex_neighbors(x: int, y: int) -> List[Tuple[int, int]]:
    """Six hex neighbors under Wesnoth's odd-q offset."""
    if x % 2 == 0:
        return [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
                (x - 1, y),                  (x + 1, y),
                                  (x, y + 1)]
    return [(x - 1, y),     (x, y - 1), (x + 1, y),
            (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]


def _recruit_hex_mask(
    game_state: GameState,
    pos_to_hex: Dict[Tuple[int, int], int],
    unit_at:    Dict[Tuple[int, int], Unit],
    leader:     Unit,
    H:          int,
    *,
    rejected_hexes: Optional[set] = None,
) -> np.ndarray:
    """Compute [H] bool mask of hexes that form the leader's castle
    network AND are visibly empty AND haven't been rejected this turn.
    BFS through CASTLE/KEEP modifiers starting at the leader's keep.

    Per the legality-mask contract (CLAUDE.md): a hex is "legal" iff
    it can be VALIDLY ATTEMPTED given the policy's observable state.
    Visible occupancy excludes hexes our units stand on. Rejection
    history (`rejected_hexes`, scoped to current turn) excludes
    hexes a prior attempt bounced -- prevents within-turn looping
    on the same fog-hidden enemy. Both clear at the right boundary:
    visible occupancy when our unit moves; rejection history at
    init_side.

    Fog hexes ARE legal (the model can attempt; bounce-on-fog is
    handled by the harness retry loop, not the mask).
    """
    rejected_hexes = rejected_hexes or set()
    mods_by_pos: Dict[Tuple[int, int], set] = {
        (h.position.x, h.position.y): h.modifiers for h in game_state.map.hexes
    }
    start = (leader.position.x, leader.position.y)
    visited = {start}
    q: deque = deque([start])
    valid: set = set()

    while q:
        x, y = q.popleft()
        for nx, ny in _hex_neighbors(x, y):
            if (nx, ny) in visited:
                continue
            nmods = mods_by_pos.get((nx, ny))
            if nmods is None:
                continue
            if TerrainModifiers.CASTLE in nmods or TerrainModifiers.KEEP in nmods:
                visited.add((nx, ny))
                q.append((nx, ny))
                if (nx, ny) in unit_at:
                    continue
                if (nx, ny) in rejected_hexes:
                    continue
                valid.add((nx, ny))

    mask = np.zeros(H, dtype=bool)
    for (x, y) in valid:
        j = pos_to_hex.get((x, y))
        if j is not None:
            mask[j] = True
    return mask
