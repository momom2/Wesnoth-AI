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
  - Weapon index is capped to the attacker's actual attack count;
    end-of-list weapon logits are masked out.
  - No mask on target hex — legality is enforced by Wesnoth rejecting
    invalid moves, not by us.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from classes import GameState, Position, Unit
from encoder import EncodedState
from model import ActorKind, ModelOutput


# Safer-than-torch.inf fill for "this slot is invalid". -inf sometimes
# produces NaNs in downstream softmax under pathological conditions.
_NEG_INF = -1e9


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
) -> SampledAction:
    """Stochastic sampling. Caller should be in torch.no_grad context
    to avoid retaining the forward graph — this function only reads
    logits and samples indices."""
    device = output.actor_logits.device
    value_est = float(output.value.squeeze().item())

    actor_logits = _masked_actor_logits(encoded, output)
    actor_dist = torch.distributions.Categorical(logits=actor_logits.squeeze(0))
    actor_idx = int(actor_dist.sample().item())

    kind = int(output.actor_kind[0, actor_idx].item())

    if kind == ActorKind.END_TURN:
        return SampledAction(
            action={'type': 'end_turn'},
            actor_idx=actor_idx, target_idx=None, weapon_idx=None,
            value_est=value_est,
        )

    target_logits = output.target_logits[0, actor_idx]  # [H]
    if target_logits.numel() == 0:
        return SampledAction(
            action={'type': 'end_turn'},
            actor_idx=actor_idx, target_idx=None, weapon_idx=None,
            value_est=value_est,
        )

    target_dist = torch.distributions.Categorical(logits=target_logits)
    target_idx = int(target_dist.sample().item())
    target_pos = encoded.hex_positions[target_idx]

    if kind == ActorKind.UNIT:
        unit_pos = encoded.unit_positions[actor_idx]
        unit_id  = encoded.unit_ids[actor_idx]
        enemy    = _enemy_unit_at(game_state, target_pos)

        if enemy is None:
            return SampledAction(
                action={'type': 'move', 'start_hex': unit_pos, 'target_hex': target_pos},
                actor_idx=actor_idx, target_idx=target_idx, weapon_idx=None,
                value_est=value_est,
            )

        attacker = _unit_by_id(game_state, unit_id)
        num_attacks = len(attacker.attacks) if attacker else 0
        if num_attacks == 0:
            return SampledAction(
                action={'type': 'move', 'start_hex': unit_pos, 'target_hex': target_pos},
                actor_idx=actor_idx, target_idx=target_idx, weapon_idx=None,
                value_est=value_est,
            )

        weapon_logits = _masked_weapon_logits(output, actor_idx, num_attacks)
        weapon_dist = torch.distributions.Categorical(logits=weapon_logits)
        weapon_idx = int(weapon_dist.sample().item())
        return SampledAction(
            action={
                'type':         'attack',
                'start_hex':    unit_pos,
                'target_hex':   target_pos,
                'attack_index': weapon_idx,
            },
            actor_idx=actor_idx, target_idx=target_idx, weapon_idx=weapon_idx,
            value_est=value_est,
        )

    # ActorKind.RECRUIT
    recruit_type = encoded.recruit_types[actor_idx - output.num_units]
    return SampledAction(
        action={
            'type':       'recruit',
            'unit_type':  recruit_type,
            'target_hex': target_pos,
        },
        actor_idx=actor_idx, target_idx=target_idx, weapon_idx=None,
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Given stored action indices, compute (log_prob, entropy) as
    scalar grad-enabled tensors. The trainer sums these across a
    batch to form the policy-gradient loss.

    Must be called with grads enabled (normal mode, NOT under no_grad).
    """
    device = output.actor_logits.device

    actor_logits = _masked_actor_logits(encoded, output)
    actor_dist = torch.distributions.Categorical(logits=actor_logits.squeeze(0))

    log_prob = actor_dist.log_prob(
        torch.tensor(actor_idx, device=device, dtype=torch.long)
    )
    entropy = actor_dist.entropy()

    if target_idx is None:
        return log_prob, entropy

    target_logits = output.target_logits[0, actor_idx]
    target_dist = torch.distributions.Categorical(logits=target_logits)
    log_prob = log_prob + target_dist.log_prob(
        torch.tensor(target_idx, device=device, dtype=torch.long)
    )
    entropy = entropy + target_dist.entropy()

    if weapon_idx is None:
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
    weapon_dist = torch.distributions.Categorical(logits=weapon_logits)
    log_prob = log_prob + weapon_dist.log_prob(
        torch.tensor(weapon_idx, device=device, dtype=torch.long)
    )
    entropy = entropy + weapon_dist.entropy()
    return log_prob, entropy


# ---------------------------------------------------------------------
# Helpers (shared by sample + reforward)
# ---------------------------------------------------------------------

def _masked_actor_logits(encoded: EncodedState, output: ModelOutput) -> torch.Tensor:
    device = output.actor_logits.device
    mask = torch.cat([
        encoded.unit_is_ours,     # [1, U]
        encoded.recruit_is_ours,  # [1, R]
        torch.ones(1, 1, device=device),  # end_turn always valid
    ], dim=1)
    return output.actor_logits.masked_fill(mask == 0, _NEG_INF)


def _masked_weapon_logits(
    output: ModelOutput, actor_idx: int, num_attacks: int,
) -> torch.Tensor:
    weapons = output.weapon_logits[0, actor_idx].clone()
    if num_attacks < weapons.shape[0]:
        weapons[num_attacks:] = _NEG_INF
    return weapons


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
