"""Sample an action from the model's output distributions.

Coordinates three inputs into one internal-format action dict:

- **EncodedState** — parallel lists mapping model-slot indices back to
  real game entities (hex positions, unit ids, recruit type names).
- **ModelOutput** — actor_logits / target_logits / weapon_logits / value.
- **GameState**   — used for legality masks (whose side? whose turn?).

Returns a `SampledAction` carrying both the action dict and the summed
log-probability of the sampled choices, so Phase 3.2's trainer can
compute policy gradients without re-running the model.

Masking rules applied here:
  - Actor must belong to the current side (either a unit we control or
    a recruit offered to our side). `end_turn` is always valid.
  - If action is an attack, weapon index is capped to the attacker's
    actual number of attacks.

NOT masked here:
  - Move legality (reachable?), recruit placement legality (castle-
    connected empty hex?), etc. Wesnoth rejects invalid actions and
    the Lua turn_stage stays in the loop asking for a new action, so
    it's fine for the policy to sometimes pick an illegal target —
    it'll just get rejected and pick again next iteration. During
    early training this is the intended exploration story.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from classes import GameState, Position, Unit
from encoder import EncodedState
from model import ActorKind, ModelOutput


# Finite "minus infinity" for masking. -inf in logits turns into NaN
# under some pathological softmax conditions; a large negative is
# numerically safer.
_NEG_INF = -1e9


@dataclass
class SampledAction:
    action:   Dict            # internal-format action dict (0-indexed Positions)
    log_prob: torch.Tensor    # scalar, sum of log-probs of sampled choices
    entropy:  torch.Tensor    # scalar, sum of entropies of the sampled dists
    value:    torch.Tensor    # scalar, V(s) from the model


def sample_action(
    encoded:    EncodedState,
    output:     ModelOutput,
    game_state: GameState,
) -> SampledAction:
    device = output.actor_logits.device
    value  = output.value.squeeze()  # scalar

    U = output.num_units
    R = output.num_recruits
    # actor slots: [unit_0 .. unit_{U-1}, recruit_0 .. recruit_{R-1}, end_turn]

    # --- actor mask ----------------------------------------------------
    actor_mask = torch.cat([
        encoded.unit_is_ours,     # [1, U]
        encoded.recruit_is_ours,  # [1, R]
        torch.ones(1, 1, device=device),
    ], dim=1)

    # end_turn is always valid, so the mask has at least one nonzero
    # column; no defensive fallback needed.
    actor_logits = output.actor_logits.masked_fill(actor_mask == 0, _NEG_INF)

    # --- sample actor --------------------------------------------------
    actor_dist = torch.distributions.Categorical(logits=actor_logits.squeeze(0))
    actor_idx_t = actor_dist.sample()
    actor_idx = actor_idx_t.item()
    log_prob = actor_dist.log_prob(actor_idx_t)
    entropy = actor_dist.entropy()

    kind = output.actor_kind[0, actor_idx].item()

    if kind == ActorKind.END_TURN:
        return SampledAction(
            action={'type': 'end_turn'},
            log_prob=log_prob,
            entropy=entropy,
            value=value,
        )

    # --- sample target hex --------------------------------------------
    target_row = output.target_logits[0, actor_idx]  # [H]
    if target_row.numel() == 0:
        # Pathological: no hexes in the state. Degrade to end_turn.
        return SampledAction(
            action={'type': 'end_turn'},
            log_prob=log_prob,
            entropy=entropy,
            value=value,
        )

    target_dist = torch.distributions.Categorical(logits=target_row)
    target_idx_t = target_dist.sample()
    target_idx = target_idx_t.item()
    log_prob = log_prob + target_dist.log_prob(target_idx_t)
    entropy = entropy + target_dist.entropy()
    target_pos = encoded.hex_positions[target_idx]

    # --- resolve action type based on state ----------------------------
    if kind == ActorKind.UNIT:
        unit_id  = encoded.unit_ids[actor_idx]
        unit_pos = encoded.unit_positions[actor_idx]
        return _build_unit_action(
            output=output,
            actor_idx=actor_idx,
            attacker_id=unit_id,
            attacker_pos=unit_pos,
            target_pos=target_pos,
            game_state=game_state,
            log_prob=log_prob,
            entropy=entropy,
            value=value,
        )

    # ActorKind.RECRUIT
    recruit_type = encoded.recruit_types[actor_idx - U]
    return SampledAction(
        action={
            'type':       'recruit',
            'unit_type':  recruit_type,
            'target_hex': target_pos,
        },
        log_prob=log_prob,
        entropy=entropy,
        value=value,
    )


def _build_unit_action(
    *,
    output:      ModelOutput,
    actor_idx:   int,
    attacker_id: str,
    attacker_pos: Position,
    target_pos:  Position,
    game_state:  GameState,
    log_prob:    torch.Tensor,
    entropy:     torch.Tensor,
    value:       torch.Tensor,
) -> SampledAction:
    """Move or attack, depending on what's at target_pos."""
    enemy = _enemy_unit_at(game_state, target_pos)

    if enemy is None:
        return SampledAction(
            action={
                'type':       'move',
                'start_hex':  attacker_pos,
                'target_hex': target_pos,
            },
            log_prob=log_prob,
            entropy=entropy,
            value=value,
        )

    # Attack. Pick a weapon from the attacker's actual attack list.
    attacker = _unit_by_id(game_state, attacker_id)
    num_attacks = len(attacker.attacks) if attacker else 0
    if num_attacks == 0:
        # Can't attack; emit a move. Wesnoth will reject (unit wants
        # to step onto an enemy), turn_stage keeps looping.
        return SampledAction(
            action={
                'type':       'move',
                'start_hex':  attacker_pos,
                'target_hex': target_pos,
            },
            log_prob=log_prob,
            entropy=entropy,
            value=value,
        )

    weapon_row = output.weapon_logits[0, actor_idx].clone()
    if num_attacks < weapon_row.shape[0]:
        weapon_row[num_attacks:] = _NEG_INF
    weapon_dist = torch.distributions.Categorical(logits=weapon_row)
    weapon_idx_t = weapon_dist.sample()
    weapon_idx = weapon_idx_t.item()
    log_prob = log_prob + weapon_dist.log_prob(weapon_idx_t)
    entropy = entropy + weapon_dist.entropy()

    return SampledAction(
        action={
            'type':         'attack',
            'start_hex':    attacker_pos,
            'target_hex':   target_pos,
            'attack_index': weapon_idx,
        },
        log_prob=log_prob,
        entropy=entropy,
        value=value,
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
