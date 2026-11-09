"""The policy-gradient loss and the readings taken from the same forward
(docs/training_design_20261019/current_design.md §1, §3.4).

Each row of a batch is one logged decision: the action the player sampled
(`pg_records.Decision.action`) under the masks it decided under, staged as
a one-action `own_loss.OwnBatch`. Its log probability is the network's
masked joint law (`own_loss.legal_log_probs`), and the pass's loss is

    pg_loss = - sum_rows w * log pi(a | h)

with w the row's signed weight R. The weight is applied here, outside
`own_loss.own_policy_loss`, whose `prob > 0` filter would drop every
negative weight and turn the update into imitation of the winners.

The joint law factors as actor, then for a unit its action type, then the
target hex, then for an attack its weapon, each a softmax masked to the
legal choices. From the same factors:

- `class_masses`: the law's mass on end_turn, on attacks and on recruits.
- `probe_offsets`: the forward with three scalars added to the end_turn
  actor's logit, the attack-type logit of every unit actor and every
  recruit actor's logit. At zero, d log pi(a)/d e is 1[a in c] - p(c) for
  end_turn and recruits, and 1[a is an attack] - p(attack | a's actor) for
  attacks (0 for an actor that is not a unit): probe P1.
- `head_scores`: d log pi(a) / d tau when a head's final linear map is
  scaled by (1 + tau): the logit of the chosen entry less the law's mean
  logit, per factor. The four heads' maps are `HEAD_PARAMETERS`.
- `legal_kl`: KL(p || q) between two networks' joint laws on the same
  positions, by the chain rule over the factors.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from wesnoth_ai.own_loss import OwnBatch, _log_softmax_masked, legal_log_probs, stage_own_batch
from wesnoth_ai.sequence_records import OwnTarget
from wesnoth_ai.server_priors import (_SLOT_END, _SLOT_RECRUIT, _SLOT_UNIT, KIND_ATTACK, KIND_END_TURN,
                                      KIND_MOVE, _unpack_bits)

CLASSES = ("end_turn", "attack", "recruit")
# The heads whose final linear map scales their logits, and that map's
# parameters (the target logits are q.k / sqrt(d), linear in the query's
# projection, which has no bias).
HEADS = ("actor", "type", "target", "weapon")
HEAD_PARAMETERS: Dict[str, Tuple[str, ...]] = {
    "actor": ("model.actor_head.weight", "model.actor_head.bias"),
    "type": ("model.type_head.weight", "model.type_head.bias"),
    "target": ("model.target_q_proj.weight",),
    "weapon": ("model.weapon_head.2.weight", "model.weapon_head.2.bias"),
}


def stage_actions(rows: Sequence[int], packs: Sequence, actions: Sequence[Tuple[int, int, int, int]],
                  A_max: int, H_max: int, n_types: int, n_weapons: int, device: torch.device) -> OwnBatch:
    """The rows' sampled actions under their masks, one action a row
    (column 0 of the batch, probability 1)."""
    targets = [OwnTarget(actor=np.array([a], dtype=np.int32), kind=np.array([k], dtype=np.int8),
                         target=np.array([t], dtype=np.int32), weapon=np.array([w], dtype=np.int8),
                         prior=np.ones(1), end_prior=float(k == KIND_END_TURN))
               for a, k, t, w in actions]
    return stage_own_batch(rows, packs, targets, A_max, H_max, n_types, n_weapons, device)


def action_log_probs(padded, batch: OwnBatch) -> torch.Tensor:
    """[n] float32: each row's log pi of its sampled action."""
    return legal_log_probs(padded, batch)[:, 0]


def pg_loss(padded, batch: OwnBatch, w: torch.Tensor) -> torch.Tensor:
    """- sum_rows w * log pi(a | h): the signed weights applied outside
    the imitation loss's `prob > 0` filter."""
    return -(w * action_log_probs(padded, batch)).sum()


# ---------------------------------------------------------------------------
# The factors
# ---------------------------------------------------------------------------

@dataclass
class FactorLaws:
    """Each factor's masked log law over a batch's rows, with its legal
    entries: the actor [n, A]; per actor the action type [n, A, T], the
    target among attacks, moves and recruit hexes [n, A, H] each, the
    weapon [n, A, W]. Entries outside a mask hold no law (read them
    through the masks)."""
    actor: torch.Tensor
    actor_ok: torch.Tensor
    type: torch.Tensor
    type_ok: torch.Tensor
    attack: torch.Tensor
    attack_ok: torch.Tensor
    move: torch.Tensor
    move_ok: torch.Tensor
    recruit: torch.Tensor
    recruit_ok: torch.Tensor
    weapon: torch.Tensor
    weapon_ok: torch.Tensor
    slot: torch.Tensor                    # [n, A] the actor slot kinds (server_priors._SLOT_*)


def factor_laws(padded, views: Dict[str, torch.Tensor], rows: torch.Tensor) -> FactorLaws:
    """The factors of the rows `rows` of `padded` under the staged mask
    views (one row of views per row of `rows`)."""
    H, W = padded.target_logits.shape[2], padded.weapon_logits.shape[2]
    actor_ok = views["actor_mask"] != 0
    type_ok = views["type_valid"] != 0
    attack_ok = _unpack_bits(views["attack_bits"], H)
    move_ok = _unpack_bits(views["move_bits"], H)
    recruit_ok = _unpack_bits(views["union_bits"], H)
    weapon_ok = torch.arange(W, device=actor_ok.device).view(1, 1, W) < views["n_attacks"].long().unsqueeze(2)
    target = padded.target_logits.float()[rows]
    return FactorLaws(
        actor=_log_softmax_masked(padded.actor_logits.float()[rows], actor_ok), actor_ok=actor_ok,
        type=_log_softmax_masked(padded.type_logits.float()[rows], type_ok), type_ok=type_ok,
        attack=_log_softmax_masked(target, attack_ok), attack_ok=attack_ok,
        move=_log_softmax_masked(target, move_ok), move_ok=move_ok,
        recruit=_log_softmax_masked(target, recruit_ok), recruit_ok=recruit_ok,
        weapon=_log_softmax_masked(padded.weapon_logits.float()[rows], weapon_ok), weapon_ok=weapon_ok,
        slot=views["slot_kind"].long())


def _law(log_law: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
    """The probabilities of a masked log law, exactly 0 off the mask."""
    return torch.where(ok, log_law.exp(), torch.zeros_like(log_law))


def class_masses(padded, batch: OwnBatch) -> torch.Tensor:
    """[n, 3] float32: the joint law's mass on end_turn, on attacks and on
    recruits (`CLASSES`) at each row, differentiable."""
    f = factor_laws(padded, batch.views, batch.rows)
    p_actor = _law(f.actor, f.actor_ok)
    p_attack = _law(f.type, f.type_ok)[:, :, KIND_ATTACK]
    end = (p_actor * (f.slot == _SLOT_END)).sum(dim=1)
    attack = (p_actor * p_attack * (f.slot == _SLOT_UNIT)).sum(dim=1)
    recruit = (p_actor * (f.slot == _SLOT_RECRUIT)).sum(dim=1)
    return torch.stack([end, attack, recruit], dim=1)


def probe_offsets(padded, batch: OwnBatch, offsets: torch.Tensor):
    """`padded` with the probe scalars `offsets` [3] (end_turn, attack,
    recruit, the order of `CLASSES`) added at the batch's rows: to the
    end_turn actor's logit, to every unit actor's attack-type logit and to
    every recruit actor's logit. The other fields are shared."""
    B, A = padded.actor_logits.shape
    slot = torch.zeros(B, A, dtype=torch.long, device=padded.actor_logits.device)
    slot[batch.rows] = batch.views["slot_kind"].long()
    end, attack, recruit = offsets[0], offsets[1], offsets[2]
    actor = padded.actor_logits.float() + end * (slot == _SLOT_END) + recruit * (slot == _SLOT_RECRUIT)
    type_logits = padded.type_logits.float()
    attack_column = torch.zeros(type_logits.shape[2], device=type_logits.device)
    attack_column[KIND_ATTACK] = 1.0
    type_logits = type_logits + attack * (slot == _SLOT_UNIT).unsqueeze(2) * attack_column
    return dataclasses.replace(padded, actor_logits=actor, type_logits=type_logits)


def _score(logits: torch.Tensor, log_law: torch.Tensor, ok: torch.Tensor, chosen: torch.Tensor) -> torch.Tensor:
    """d log softmax((1 + tau) l)[chosen] / d tau at 0: l[chosen] less the
    law's mean of l over the legal entries ([..., E] rows, chosen [...])."""
    mean = (_law(log_law, ok) * logits.masked_fill(~ok, 0.0)).sum(dim=-1)
    return logits.gather(-1, chosen.unsqueeze(-1)).squeeze(-1) - mean


def head_scores(padded, batch: OwnBatch) -> torch.Tensor:
    """[n, 4] float32: d log pi(a) / d tau for each head of `HEADS` scaled
    by (1 + tau), at each row's sampled action (column 0); 0 for a factor
    the action does not have."""
    rows = batch.rows
    f = factor_laws(padded, batch.views, rows)
    n = rows.shape[0]
    b = torch.arange(n, device=rows.device)
    a, kind = batch.actor[:, 0], batch.kind[:, 0]
    zero = torch.zeros(n, device=rows.device)
    actor = _score(padded.actor_logits.float()[rows], f.actor, f.actor_ok, a)
    unit = kind <= KIND_MOVE
    type_ = torch.where(unit, _score(padded.type_logits.float()[rows][b, a], f.type[b, a], f.type_ok[b, a],
                                     kind.clamp(max=KIND_MOVE)), zero)
    target_logits = padded.target_logits.float()[rows][b, a]
    is_attack, is_move = (kind == KIND_ATTACK).unsqueeze(1), (kind == KIND_MOVE).unsqueeze(1)
    log_target = torch.where(is_attack, f.attack[b, a], torch.where(is_move, f.move[b, a], f.recruit[b, a]))
    ok_target = torch.where(is_attack, f.attack_ok[b, a], torch.where(is_move, f.move_ok[b, a], f.recruit_ok[b, a]))
    target = torch.where(kind != KIND_END_TURN,
                         _score(target_logits, log_target, ok_target, batch.target[:, 0]), zero)
    weapon = torch.where(kind == KIND_ATTACK,
                         _score(padded.weapon_logits.float()[rows][b, a], f.weapon[b, a], f.weapon_ok[b, a],
                                batch.weapon[:, 0]), zero)
    return torch.stack([actor, type_, target, weapon], dim=1)


def head_direction(named_parameters: Dict[str, torch.Tensor], head: str) -> Dict[str, torch.Tensor]:
    """The weight-space direction that scales `head`'s logits: its final
    map's parameters, zero elsewhere (only the named entries are given)."""
    return {name: named_parameters[name].detach().clone() for name in HEAD_PARAMETERS[head]}


# ---------------------------------------------------------------------------
# The divergence between two networks
# ---------------------------------------------------------------------------

def _kl(lp: torch.Tensor, lq: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
    """KL of two masked log laws over their last dimension (0 for a row
    with no legal entry)."""
    terms = torch.where(ok, lp.exp() * (lp - lq), torch.zeros_like(lp))
    return terms.sum(dim=-1)


def legal_kl(p, q, views: Dict[str, torch.Tensor], rows: Optional[torch.Tensor] = None) -> torch.Tensor:
    """[n] float32: KL(p || q) of two networks' joint legal laws at the
    rows `rows` (default all) of their padded outputs `p` and `q`, under
    the staged mask views of those rows (`legal_kl_laws`)."""
    if rows is None:
        rows = torch.arange(p.actor_logits.shape[0], device=p.actor_logits.device)
    return legal_kl_laws(factor_laws(p, views, rows), factor_laws(q, views, rows))


# The factors of a FactorLaws, each with its mask `<name>_ok`.
LAW_KEYS = ("actor", "type", "attack", "move", "recruit", "weapon")


def compact_laws(laws: FactorLaws) -> Dict[str, torch.Tensor]:
    """Each factor's log law at its legal entries only, flat in the mask's
    order, on the CPU: a decision's share is about its legal actions, its
    actors and their types, not the padded actors x hexes."""
    return {k: getattr(laws, k)[getattr(laws, k + "_ok")].detach().float().cpu() for k in LAW_KEYS}


def expand_laws(compact: Dict[str, torch.Tensor], like: FactorLaws) -> FactorLaws:
    """`compact` laid back on the masks of `like` (the same positions'
    factors, another network's), 0 off them."""
    out = {}
    for k in LAW_KEYS:
        ok = getattr(like, k + "_ok")
        values = compact[k].to(ok.device)
        if values.numel() != int(ok.sum()):
            raise ValueError(f"{k}: {values.numel()} stored entries for {int(ok.sum())} legal ones: other positions")
        dense = torch.zeros_like(getattr(like, k))
        dense[ok] = values
        out[k] = dense
    return dataclasses.replace(like, **out)


def legal_kl_laws(P: FactorLaws, Q: FactorLaws) -> torch.Tensor:
    """[n] float32: KL(P || Q) of two joint legal laws given by their factors
    on the same masks. By the chain rule: the actor's KL, plus under each
    actor's probability its type's KL and, under each type's probability,
    its target's and (attacks) its weapon's."""
    p_actor = _law(P.actor, P.actor_ok)
    p_type = _law(P.type, P.type_ok)
    unit_terms = (_kl(P.type, Q.type, P.type_ok)
                  + p_type[:, :, KIND_ATTACK] * (_kl(P.attack, Q.attack, P.attack_ok)
                                                 + _kl(P.weapon, Q.weapon, P.weapon_ok))
                  + p_type[:, :, KIND_MOVE] * _kl(P.move, Q.move, P.move_ok))
    recruit_terms = _kl(P.recruit, Q.recruit, P.recruit_ok)
    zero = torch.zeros_like(unit_terms)
    per_actor = torch.where(P.slot == _SLOT_UNIT, unit_terms,
                            torch.where(P.slot == _SLOT_RECRUIT, recruit_terms, zero))
    return _kl(P.actor, Q.actor, P.actor_ok) + (p_actor * per_actor).sum(dim=1)
