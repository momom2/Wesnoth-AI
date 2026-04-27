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


# How strongly the combat oracle nudges attack-target selection. Set
# as a fraction of the typical logit scale (~ ±3-5 after training).
# 0.1 × (net damage of ±40) = ±4 logit units, which moves the attack
# distribution noticeably but doesn't overwhelm learned preferences.
# Tune down if we see the policy obsessing over marginal attacks.
_COMBAT_LOGIT_ALPHA = 0.1


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

    masks = _build_legality_masks(encoded, game_state)

    actor_logits = _masked_actor_logits(encoded, output, masks.actor_valid)
    actor_idx = _sample_from_logits(actor_logits.squeeze(0))

    kind = int(output.actor_kind[0, actor_idx].item())

    if kind == ActorKind.END_TURN:
        return SampledAction(
            action={'type': 'end_turn'},
            actor_idx=actor_idx, target_idx=None, weapon_idx=None,
            value_est=value_est,
        )

    target_logits = _masked_target_logits(
        output, masks.target_valid, actor_idx,
        attack_bias=masks.attack_bias,
    )
    if target_logits.numel() == 0:
        return SampledAction(
            action={'type': 'end_turn'},
            actor_idx=actor_idx, target_idx=None, weapon_idx=None,
            value_est=value_est,
        )

    target_idx = _sample_from_logits(target_logits)
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
        weapon_idx = _sample_from_logits(weapon_logits)
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

    masks = _build_legality_masks(encoded, game_state)

    actor_logits = _masked_actor_logits(encoded, output, masks.actor_valid)
    log_prob, entropy = _logprob_entropy_1d(actor_logits.squeeze(0), actor_idx)

    if target_idx is None:
        return log_prob, entropy

    target_logits = _masked_target_logits(
        output, masks.target_valid, actor_idx,
        attack_bias=masks.attack_bias,
    )
    lp_t, ent_t = _logprob_entropy_1d(target_logits, target_idx)
    log_prob = log_prob + lp_t
    entropy = entropy + ent_t

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
    lp_w, ent_w = _logprob_entropy_1d(weapon_logits, weapon_idx)
    log_prob = log_prob + lp_w
    entropy = entropy + ent_w
    return log_prob, entropy


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
    attack_bias:  torch.Tensor  # [A, H] float — combat-oracle logit prior


def _build_legality_masks(
    encoded: EncodedState, game_state: GameState,
) -> LegalityMasks:
    """Assemble actor + target validity masks from the game state.

    Uses numpy internally to vectorize the hex-distance and occupancy
    checks. Transfers to the model's device at the end. Cost budget
    is ~1ms per decision on Caves-of-the-Basilisk scale; if that
    grows, the per-unit loop is the thing to flatten further.
    """
    device = encoded.unit_is_ours.device
    U = encoded.unit_tokens.size(1)
    R = encoded.recruit_tokens.size(1)
    H = encoded.hex_tokens.size(1)
    A = U + R + 1  # +1 for end_turn
    current_side = game_state.global_info.current_side

    actor_valid_np  = np.zeros(A, dtype=np.float32)
    target_valid_np = np.zeros((A, H), dtype=np.float32)
    attack_bias_np  = np.zeros((A, H), dtype=np.float32)

    # End_turn actor is the LAST slot; always valid, no target.
    actor_valid_np[A - 1] = 1.0

    if H == 0:
        return LegalityMasks(
            actor_valid  = torch.from_numpy(actor_valid_np).to(device).unsqueeze(0),
            target_valid = torch.from_numpy(target_valid_np).to(device),
            attack_bias  = torch.from_numpy(attack_bias_np).to(device),
        )

    # Per-hex arrays used by every subsequent check.
    hex_xs = np.array([p.x for p in encoded.hex_positions], dtype=np.int32)
    hex_ys = np.array([p.y for p in encoded.hex_positions], dtype=np.int32)

    # Occupancy per hex: 0 empty, 1 friendly, 2 enemy. Lookups off the
    # hex-position list make sure we only mark hexes the encoder emits.
    pos_to_hex = {(int(hex_xs[j]), int(hex_ys[j])): j for j in range(H)}
    occupancy = np.zeros(H, dtype=np.int8)
    unit_at: Dict[Tuple[int, int], Unit] = {}
    for u in game_state.map.units:
        key = (u.position.x, u.position.y)
        unit_at[key] = u
        j = pos_to_hex.get(key)
        if j is not None:
            occupancy[j] = 1 if u.side == current_side else 2

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

        row = np.zeros(H, dtype=bool)
        if can_move:
            # Empty and reachable. Exclude our own hex (a no-op move).
            row |= empty_mask & ~self_mask & (dist <= moves)
        if can_attack:
            # Enemy-occupied and close enough that SOME neighbor of the
            # enemy is reachable. Neighbor distance is distance ± 1, so
            # "enemy at <= moves + 1" is a tight bound for move-to-attack.
            # Distance == 1 already means we're adjacent (attack in place).
            row |= enemy_mask & (dist <= moves + 1)
            # Combat-oracle prior: for each valid attack target, score
            # the expected net damage using the unit's best weapon. We
            # feed this into attack_bias_np so sample_action can add
            # it to target_logits. Only apply to enemy hexes this
            # actor can actually reach-and-attack.
            reachable_enemies = enemy_mask & (dist <= moves + 1)
            if reachable_enemies.any():
                for j in np.where(reachable_enemies)[0]:
                    ex, ey = int(hex_xs[j]), int(hex_ys[j])
                    enemy_u = unit_at.get((ex, ey))
                    if enemy_u is None:
                        continue
                    try:
                        net = expected_attack_net_damage(u, enemy_u)
                    except Exception:
                        net = 0.0
                    attack_bias_np[i, j] = _COMBAT_LOGIT_ALPHA * net

        if row.any():
            target_valid_np[i] = row.astype(np.float32)
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
            recruit_hex_row = _recruit_hex_mask(
                game_state, pos_to_hex, unit_at, leader, H,
            )
            if recruit_hex_row.any():
                recruit_is_ours_np = encoded.recruit_is_ours.detach().cpu().numpy()[0]
                for r_off in range(R):
                    if recruit_is_ours_np[r_off] == 0:
                        continue
                    a = U + r_off
                    target_valid_np[a] = recruit_hex_row.astype(np.float32)
                    actor_valid_np[a] = 1.0

    return LegalityMasks(
        actor_valid  = torch.from_numpy(actor_valid_np).to(device).unsqueeze(0),
        target_valid = torch.from_numpy(target_valid_np).to(device),
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
) -> np.ndarray:
    """Compute [H] bool mask of hexes that form the leader's castle
    network AND are empty. BFS through CASTLE/KEEP modifiers starting
    at the leader's keep.
    """
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
                if (nx, ny) not in unit_at:
                    valid.add((nx, ny))

    mask = np.zeros(H, dtype=bool)
    for (x, y) in valid:
        j = pos_to_hex.get((x, y))
        if j is not None:
            mask[j] = True
    return mask
