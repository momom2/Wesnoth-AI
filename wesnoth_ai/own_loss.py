"""The policy loss at the memory player's own positions
(docs/memory_own_trajectories_prereg_20261011.md).

At an own position the target is a teacher's distribution over the legal
actions (`sequence_records.OwnTarget`: its most probable actions,
renormalised), and the loss is the cross-entropy of the student's raw
distribution over the legal actions to it:

    loss = - sum_a target(a) * log p(a)

with p the distribution the raw player decodes from (`server_priors`, before
any end_turn offset): the chain rule over the factored heads, each head's
softmax masked to the legal choices, log p(a) = log p(actor) + log p(type |
actor) + log p(target | actor, type) + log p(weapon | actor), the factors an
action does not have left out. The masks and their biases are the ones the
player decided under (`server_priors.PackedMasks`, staged as the server
stages them), so exp(log p) is the prior `server_priors.priors_outputs`
computes for the same action (tests/test_own_trajectories.py). For a
one-hot target the loss is the hard label's cross-entropy without label
smoothing.

`target_key` and `decode_key` read a choice off a target or a full prior
list at an end_turn logit offset, as the raw player decodes
(`tools/raw_player.shift_end_turn_prior`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from wesnoth_ai.sequence_records import OwnTarget
from wesnoth_ai.server_priors import (KIND_ATTACK, KIND_END_TURN, KIND_MOVE, KIND_RECRUIT,
                                      _unpack_bits, mask_layout, write_masks)

# The teacher's actions a target keeps (the pre-registration's top 16).
TOP_ACTIONS = 16
_NEG_INF = float("-inf")

# An action as the factored indices it is chosen by: (actor, kind, target, weapon).
ActionKey = Tuple[int, int, int, int]


def top_target(actor: np.ndarray, kind: np.ndarray, target: np.ndarray, weapon: np.ndarray,
               prior: np.ndarray, top: int = TOP_ACTIONS) -> OwnTarget:
    """The `top` most probable of a position's legal actions (compact arrays
    in the reference order, `server_priors.CompactActions`), most probable
    first (ties in the reference order), renormalised."""
    prior = np.asarray(prior, dtype=np.float64)
    order = np.argsort(-prior, kind="stable")[:top]
    kept = prior[order]
    mass = float(kept.sum())
    if not mass > 0.0:
        raise ValueError(f"a target of {len(prior)} legal actions holds no mass")
    is_end = np.asarray(kind) == KIND_END_TURN
    return OwnTarget(actor=np.asarray(actor, dtype=np.int32)[order],
                     kind=np.asarray(kind, dtype=np.int8)[order],
                     target=np.asarray(target, dtype=np.int32)[order],
                     weapon=np.asarray(weapon, dtype=np.int8)[order], prior=kept,
                     end_prior=float(prior[is_end].sum()))


def _shift_factors(end_prior: float, offset: float) -> Tuple[float, float]:
    """The factors the end_turn logit offset multiplies end_turn's prior
    and every other prior by (`raw_player.shift_end_turn_prior`)."""
    if not offset or not 0.0 < end_prior < 1.0:
        return 1.0, 1.0
    logit = np.log(end_prior) - np.log1p(-end_prior) + float(offset)
    p_new = 1.0 / (1.0 + np.exp(-logit))
    return p_new / end_prior, (1.0 - p_new) / (1.0 - end_prior)


def target_key(t: OwnTarget, offset: float, end_actor: int) -> ActionKey:
    """The action the raw player's argmax picks from the teacher's whole
    distribution at `offset`: end_turn or the most probable other action,
    both readable from the target (the most probable other action is its
    first that is not end_turn; end_turn's prior is kept whole)."""
    end_factor, other_factor = _shift_factors(t.end_prior, offset)
    others = np.flatnonzero(t.kind != KIND_END_TURN)
    if not len(others):
        return (int(end_actor), KIND_END_TURN, -1, -1)
    i = int(others[0])
    # The argmax keeps the first of equal priors in the reference order,
    # where end_turn comes last: it wins only when strictly above.
    if t.end_prior * end_factor > float(t.prior[i]) * other_factor:
        return (int(end_actor), KIND_END_TURN, -1, -1)
    return _key(t.actor[i], t.kind[i], t.target[i], t.weapon[i])


def decode_key(actor: np.ndarray, kind: np.ndarray, target: np.ndarray, weapon: np.ndarray,
               prior: np.ndarray, offset: float) -> Optional[ActionKey]:
    """The raw player's argmax over a full list of legal actions at
    `offset` (`raw_player.pick_index` on `shift_end_turn_prior`); None
    when nothing is legal."""
    from tools.raw_player import pick_index, shift_end_turn_prior
    if not len(prior):
        return None
    is_end = np.asarray(kind) == KIND_END_TURN
    i = pick_index(shift_end_turn_prior(np.asarray(prior, dtype=np.float64), is_end, offset), 0.0, None)
    return _key(actor[i], kind[i], target[i], weapon[i])


def _key(actor, kind, target, weapon) -> ActionKey:
    kind = int(kind)
    return (int(actor), kind, int(target) if kind != KIND_END_TURN else -1,
            int(weapon) if kind == KIND_ATTACK else -1)


def label_key(ai) -> Optional[ActionKey]:
    """A label (`ActionIndices`) as an action key; None for a label that
    names no action of the legal set (a timeout, a target off the hex
    tokens)."""
    kinds = {"attack": KIND_ATTACK, "move": KIND_MOVE, "recruit": KIND_RECRUIT, "end_turn": KIND_END_TURN}
    kind = kinds.get(ai.action_type)
    if kind is None or (kind != KIND_END_TURN and ai.target_idx is None):
        return None
    return _key(ai.actor_idx, kind, -1 if ai.target_idx is None else ai.target_idx,
                -1 if ai.weapon_idx is None else ai.weapon_idx)


# ---------------------------------------------------------------------------
# The loss
# ---------------------------------------------------------------------------

@dataclass
class OwnBatch:
    """The own rows of one time step's batch: their masks on the device,
    staged as `server_priors` stages them, and their targets as [n, K]
    index and probability tensors (pads hold probability 0)."""
    rows: torch.Tensor                  # long [n]: the rows of the batch
    views: Dict[str, torch.Tensor]
    actor: torch.Tensor                 # long [n, K]
    kind: torch.Tensor
    target: torch.Tensor
    weapon: torch.Tensor
    prob: torch.Tensor                  # float32 [n, K]


def stage_own_batch(rows: Sequence[int], packs: Sequence, targets: Sequence[OwnTarget],
                    A_max: int, H_max: int, n_types: int, n_weapons: int,
                    device: torch.device) -> OwnBatch:
    """The batch's own rows staged for `own_policy_loss`: `A_max` and
    `H_max` are the padded output's actor and hex widths (the whole
    batch's, own and human rows together)."""
    n = len(rows)
    layout = mask_layout(n, A_max, H_max, n_types, n_weapons,
                         type_bias=any(p.type_bias is not None for p in packs),
                         attack_bias=any(p.attack_bias is not None for p in packs))
    host = torch.zeros(layout.nbytes, dtype=torch.uint8, pin_memory=(device.type == "cuda"))
    write_masks(layout.numpy_views(host.numpy()), packs)
    views = layout.torch_views(host if device.type == "cpu" else host.to(device, non_blocking=True))
    K = max((len(t.prior) for t in targets), default=1)
    arrays = {name: np.zeros((n, K), dtype=np.int64) for name in ("actor", "kind", "target", "weapon")}
    arrays["kind"][:] = KIND_END_TURN
    prob = np.zeros((n, K), dtype=np.float32)
    for r, t in enumerate(targets):
        m = len(t.prior)
        arrays["actor"][r, :m] = t.actor
        arrays["kind"][r, :m] = t.kind
        arrays["target"][r, :m] = np.maximum(t.target, 0)
        arrays["weapon"][r, :m] = np.maximum(t.weapon, 0)
        prob[r, :m] = t.prob
    to = {name: torch.from_numpy(a).to(device) for name, a in arrays.items()}
    return OwnBatch(rows=torch.as_tensor(list(rows), dtype=torch.long, device=device), views=views,
                    prob=torch.from_numpy(prob).to(device), **to)


def _log_softmax_masked(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """log_softmax over the valid entries of each row (-inf elsewhere). A
    row with no valid entry is read whole: its values are never gathered,
    and an all -inf row would put NaN in the backward pass."""
    valid = valid | ~valid.any(dim=-1, keepdim=True)
    return F.log_softmax(logits.masked_fill(~valid, _NEG_INF), dim=-1)


def legal_log_probs(padded, batch: OwnBatch) -> torch.Tensor:
    """[n, K] float32: the student's log prior of each target action under
    the masks it was legal under (module docstring)."""
    v, rows = batch.views, batch.rows
    a, kind = batch.actor, batch.kind
    n, K = a.shape
    b = torch.arange(n, device=a.device).unsqueeze(1).expand(n, K)
    actor_logits = padded.actor_logits.float()[rows] + v["actor_bias"]
    log_actor = _log_softmax_masked(actor_logits, v["actor_mask"] != 0)
    lp = log_actor.gather(1, a)

    type_logits = padded.type_logits.float()[rows]
    if "type_bias" in v:
        type_logits = type_logits + v["type_bias"]
    log_type = _log_softmax_masked(type_logits[b, a], v["type_valid"][b, a] != 0)       # [n, K, T]
    unit_kind = kind <= KIND_MOVE
    lp = lp + torch.where(unit_kind, log_type.gather(2, kind.clamp(max=KIND_MOVE).unsqueeze(2)).squeeze(2),
                          torch.zeros_like(lp))

    H = padded.target_logits.shape[2]
    target_rows = padded.target_logits.float()[rows][b, a]                               # [n, K, H]
    attack = _unpack_bits(v["attack_bits"], H)[b, a]
    move = _unpack_bits(v["move_bits"], H)[b, a]
    union = _unpack_bits(v["union_bits"], H)[b, a]
    is_attack = (kind == KIND_ATTACK).unsqueeze(2)
    valid = torch.where(is_attack, attack, torch.where((kind == KIND_MOVE).unsqueeze(2), move, union))
    if "attack_bias" in v:
        target_rows = target_rows + torch.where(is_attack, v["attack_bias"][b, a],
                                                torch.zeros_like(target_rows))
    log_target = _log_softmax_masked(target_rows, valid)
    has_target = kind != KIND_END_TURN
    lp = lp + torch.where(has_target, log_target.gather(2, batch.target.unsqueeze(2)).squeeze(2),
                          torch.zeros_like(lp))

    W = padded.weapon_logits.shape[2]
    weapon_rows = padded.weapon_logits.float()[rows][b, a]                               # [n, K, W]
    w_ok = torch.arange(W, device=a.device).view(1, 1, W) < v["n_attacks"][b, a].long().unsqueeze(2)
    log_weapon = _log_softmax_masked(weapon_rows, w_ok)
    return lp + torch.where(kind == KIND_ATTACK, log_weapon.gather(2, batch.weapon.unsqueeze(2)).squeeze(2),
                            torch.zeros_like(lp))


def own_policy_loss(padded, batch: OwnBatch) -> torch.Tensor:
    """[n]: each own row's cross-entropy of the student's raw legal
    distribution to its target, in float32."""
    lp = legal_log_probs(padded, batch)
    live = batch.prob > 0
    return -(batch.prob * torch.where(live, lp, torch.zeros_like(lp))).sum(dim=1)


def keys_of(targets: Sequence[OwnTarget]) -> List[List[ActionKey]]:
    """Each target's actions as keys, in its order."""
    return [[_key(*e) for e in zip(t.actor, t.kind, t.target, t.weapon)] for t in targets]
