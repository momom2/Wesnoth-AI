"""Server-side legal-action priors (docs/plan_20260904.md step 1.3).

The actor computes the legality masks (a pure function of the
observable state, no network) and ships them, bit-packed, with the
leaf's RawEncoded. The inference server evaluates the batch, computes
the masked softmax of every head ON THE DEVICE for the whole batch,
and replies with the compact legal-action arrays (actor, kind,
target, weapon, prior) plus value and cliffness: about 15 KB per leaf
instead of ~120 KB of raw target logits, and no per-leaf softmax work
on the actor. `unpack_compact` rebuilds the same LegalActionPrior list
that `enumerate_legal_actions_with_priors` produces from the raw
outputs (same actions, same order, same priors; differential test in
tests/test_server_priors.py).

Kinds in the compact arrays: 0 attack, 1 move, 2 recruit, 3 end_turn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from wesnoth_ai.action_sampler import (
    _NEG_INF, LegalActionPrior, _build_legality_masks, prior_bias_end_turn,
)
from wesnoth_ai.model import MAX_ATTACKS, ActorKind, UnitActionType

KIND_ATTACK, KIND_MOVE, KIND_RECRUIT, KIND_END_TURN = 0, 1, 2, 3


@dataclass(slots=True)
class PackedMasks:
    """One leaf's legality, packed for the wire (all numpy)."""
    actor_mask:   np.ndarray            # uint8 [A]: ownership * actor_valid
    type_valid:   np.ndarray            # uint8 [A, T]
    attack_valid: np.ndarray            # packed bits [A, ceil(H/8)]
    move_valid:   np.ndarray            # packed bits [A, ceil(H/8)]
    union_valid:  np.ndarray            # packed bits [A, ceil(H/8)] (recruit rows)
    n_attacks:    np.ndarray            # int8 [A]: weapon slots of unit actors
    end_turn_bias: float
    n_units:      int
    n_recruits:   int
    n_hexes:      int
    type_bias:    Optional[np.ndarray] = None   # float32 [A, T] when any nonzero
    attack_bias:  Optional[np.ndarray] = None   # float32 [A, H] when any nonzero


@dataclass(slots=True)
class CompactActions:
    """One leaf's legal actions with priors, in the reference order."""
    actor:  np.ndarray    # int32 [n]
    kind:   np.ndarray    # int8 [n]
    target: np.ndarray    # int32 [n], -1 for end_turn
    weapon: np.ndarray    # int8 [n], -1 when not an attack
    prior:  np.ndarray    # float64 [n]


def pack_masks(encoded, game_state, decision_step: int = 0) -> PackedMasks:
    """Actor side: build the legality masks for `game_state` and pack
    them. `encoded` may be a light EncodedState (no dense tokens)."""
    masks = _build_legality_masks(encoded, game_state, decision_step=decision_step)
    U = encoded.unit_tokens.size(1)
    R = encoded.recruit_tokens.size(1)
    H = encoded.hex_tokens.size(1)
    ownership = np.concatenate([
        encoded.unit_is_ours.detach().cpu().numpy().reshape(-1),
        encoded.recruit_is_ours.detach().cpu().numpy().reshape(-1),
        np.ones(1, dtype=np.float32)])
    actor_valid = masks.actor_valid[0].detach().cpu().numpy()
    by_id = {u.id: u for u in game_state.map.units}
    n_att = np.zeros(U + R + 1, dtype=np.int8)
    for a in range(U):
        u = by_id.get(encoded.unit_ids[a])
        n_att[a] = min(len(u.attacks), MAX_ATTACKS) if u is not None else 0

    def _bits(t):
        return np.packbits(t.detach().cpu().numpy().astype(bool), axis=1)

    type_bias = masks.type_bias[0].detach().cpu().numpy()
    attack_bias = masks.attack_bias.detach().cpu().numpy()
    return PackedMasks(
        actor_mask=((ownership != 0.0) & (actor_valid != 0.0)).astype(np.uint8),
        type_valid=masks.type_valid[0].detach().cpu().numpy().astype(np.uint8),
        attack_valid=_bits(masks.target_valid_attack),
        move_valid=_bits(masks.target_valid_move),
        union_valid=_bits(masks.target_valid),
        n_attacks=n_att,
        end_turn_bias=float(prior_bias_end_turn(game_state)),
        n_units=U, n_recruits=R, n_hexes=H,
        type_bias=type_bias.astype(np.float32) if np.any(type_bias) else None,
        attack_bias=attack_bias.astype(np.float32) if np.any(attack_bias) else None)


def _unpack_bits(packed: np.ndarray, H: int) -> np.ndarray:
    if H == 0:
        return np.zeros((packed.shape[0], 0), dtype=bool)
    return np.unpackbits(packed, axis=1, count=H).astype(bool)


def batched_priors(padded, packs: List[PackedMasks]) -> List[CompactActions]:
    """Server side: masked softmaxes of every head for the whole
    batch on the device; the legal entries are found with `nonzero`
    on the device and their factors gathered there, so the transfer
    is a few small arrays per batch instead of the full [B, A, H]
    tables. Priors are float64 products of the float32 factors, as
    in the reference; entries are ordered per sample as the reference
    emits them (actor-major; a unit actor's attacks (hex-major,
    weapon inner) before its moves; recruit actors; end_turn).
    `padded` is a model.PaddedOutput (device tensors)."""
    B = len(packs)
    device = padded.actor_logits.device
    A_max = padded.actor_logits.shape[1]
    H_max = padded.target_logits.shape[2]
    T = padded.type_logits.shape[2]
    W = padded.weapon_logits.shape[2]
    actor_m = np.zeros((B, A_max), dtype=bool)
    type_m = np.zeros((B, A_max, T), dtype=bool)
    atk_m = np.zeros((B, A_max, H_max), dtype=bool)
    mv_m = np.zeros((B, A_max, H_max), dtype=bool)
    uni_m = np.zeros((B, A_max, H_max), dtype=bool)
    n_att = np.zeros((B, A_max), dtype=np.int64)
    n_units = np.zeros(B, dtype=np.int64)
    n_rec = np.zeros(B, dtype=np.int64)
    et_bias = np.zeros(B, dtype=np.float32)
    any_tbias = any(p.type_bias is not None for p in packs)
    any_abias = any(p.attack_bias is not None for p in packs)
    t_bias = np.zeros((B, A_max, T), dtype=np.float32) if any_tbias else None
    a_bias = np.zeros((B, A_max, H_max), dtype=np.float32) if any_abias else None
    for b, p in enumerate(packs):
        A, H = p.n_units + p.n_recruits + 1, p.n_hexes
        actor_m[b, :A] = p.actor_mask != 0
        type_m[b, :A] = p.type_valid != 0
        atk_m[b, :A, :H] = _unpack_bits(p.attack_valid, H)
        mv_m[b, :A, :H] = _unpack_bits(p.move_valid, H)
        uni_m[b, :A, :H] = _unpack_bits(p.union_valid, H)
        n_att[b, :A] = p.n_attacks
        n_units[b] = p.n_units
        n_rec[b] = p.n_recruits
        et_bias[b] = p.end_turn_bias
        if p.type_bias is not None:
            t_bias[b, :A] = p.type_bias
        if p.attack_bias is not None:
            a_bias[b, :A, :H] = p.attack_bias

    def dev(a):
        return torch.from_numpy(a).to(device)

    with torch.no_grad():
        n_units_t = dev(n_units)
        n_rec_t = dev(n_rec)
        slotA = torch.arange(A_max, device=device).view(1, A_max)
        is_unit = slotA < n_units_t.view(B, 1)
        is_rec = (slotA >= n_units_t.view(B, 1)) & (slotA < (n_units_t + n_rec_t).view(B, 1))
        is_end = slotA == (n_units_t + n_rec_t).view(B, 1)
        actor_mt = dev(actor_m)
        al = padded.actor_logits.clone()
        al[is_end] += dev(et_bias)
        p_actor = F.softmax(al.masked_fill(~actor_mt, _NEG_INF), dim=-1)      # [B, A]
        live = actor_mt & (p_actor > 0)
        tl = padded.type_logits
        if t_bias is not None:
            tl = tl + dev(t_bias)
        p_type = F.softmax(tl.masked_fill(~dev(type_m), _NEG_INF), dim=-1)    # [B, A, T]
        n_att_t = dev(n_att)
        w_m = torch.arange(W, device=device).view(1, 1, W) < n_att_t.unsqueeze(-1)
        p_wpn = F.softmax(padded.weapon_logits.masked_fill(~w_m, _NEG_INF), dim=-1)
        tg = padded.target_logits
        tga = tg + dev(a_bias) if a_bias is not None else tg
        p_atk = F.softmax(tga.masked_fill(~dev(atk_m), _NEG_INF), dim=-1)     # [B, A, H]
        p_mv = F.softmax(tg.masked_fill(~dev(mv_m), _NEG_INF), dim=-1)
        p_rec = F.softmax(tg.masked_fill(~dev(uni_m), _NEG_INF), dim=-1)
        # Legal entries by FACTOR masks (a float32 product may underflow
        # where the reference's float64 product does not).
        unit_live = live & is_unit
        atk_ok = unit_live & (p_type[:, :, UnitActionType.ATTACK] > 0) & (n_att_t > 0)
        mv_ok = unit_live & (p_type[:, :, UnitActionType.MOVE] > 0)
        m_att = atk_ok.unsqueeze(-1).unsqueeze(-1) & (p_atk > 0).unsqueeze(-1) & (p_wpn > 0).unsqueeze(2)
        m_mv = mv_ok.unsqueeze(-1) & (p_mv > 0)
        m_rec = (live & is_rec).unsqueeze(-1) & (p_rec > 0)
        m_end = live & is_end
        ia = torch.nonzero(m_att)          # [Na, 4] (b, a, h, w)
        im = torch.nonzero(m_mv)           # [Nm, 3] (b, a, h)
        ir = torch.nonzero(m_rec)          # [Nr, 3]
        ie = torch.nonzero(m_end)          # [Ne, 2]
        fa = torch.stack([p_actor[ia[:, 0], ia[:, 1]],
                          p_type[ia[:, 0], ia[:, 1], UnitActionType.ATTACK],
                          p_atk[ia[:, 0], ia[:, 1], ia[:, 2]],
                          p_wpn[ia[:, 0], ia[:, 1], ia[:, 3]]], dim=1) if ia.numel() else torch.zeros(0, 4, device=device)
        fm = torch.stack([p_actor[im[:, 0], im[:, 1]],
                          p_type[im[:, 0], im[:, 1], UnitActionType.MOVE],
                          p_mv[im[:, 0], im[:, 1], im[:, 2]]], dim=1) if im.numel() else torch.zeros(0, 3, device=device)
        fr = torch.stack([p_actor[ir[:, 0], ir[:, 1]],
                          p_rec[ir[:, 0], ir[:, 1], ir[:, 2]]], dim=1) if ir.numel() else torch.zeros(0, 2, device=device)
        fe = p_actor[ie[:, 0], ie[:, 1]] if ie.numel() else torch.zeros(0, device=device)
        host = [t.cpu().numpy() for t in (ia, im, ir, ie, fa, fm, fr, fe)]
    ia, im, ir, ie, fa, fm, fr, fe = host
    fa, fm, fr, fe = (x.astype(np.float64) for x in (fa, fm, fr, fe))

    # One table of (b, a, kind, h, w, prior), sorted per sample in the
    # reference order.
    b_all = np.concatenate([ia[:, 0], im[:, 0], ir[:, 0], ie[:, 0]]).astype(np.int64)
    a_all = np.concatenate([ia[:, 1], im[:, 1], ir[:, 1], ie[:, 1]]).astype(np.int32)
    k_all = np.concatenate([np.full(len(ia), KIND_ATTACK), np.full(len(im), KIND_MOVE),
                            np.full(len(ir), KIND_RECRUIT), np.full(len(ie), KIND_END_TURN)]).astype(np.int8)
    h_all = np.concatenate([ia[:, 2], im[:, 2], ir[:, 2], np.full(len(ie), -1)]).astype(np.int32)
    w_all = np.concatenate([ia[:, 3], np.full(len(im), -1), np.full(len(ir), -1),
                            np.full(len(ie), -1)]).astype(np.int8)
    pr_all = np.concatenate([fa.prod(axis=1), fm.prod(axis=1), fr.prod(axis=1), fe])
    order = np.lexsort((w_all, h_all, k_all, a_all, b_all))
    b_all, a_all, k_all, h_all, w_all, pr_all = (x[order] for x in
                                                 (b_all, a_all, k_all, h_all, w_all, pr_all))
    bounds = np.searchsorted(b_all, np.arange(B + 1))
    out: List[CompactActions] = []
    for b in range(B):
        lo, hi = bounds[b], bounds[b + 1]
        out.append(CompactActions(actor=a_all[lo:hi].copy(), kind=k_all[lo:hi].copy(),
                                  target=h_all[lo:hi].copy(), weapon=w_all[lo:hi].copy(),
                                  prior=pr_all[lo:hi].copy()))
    return out


def unpack_compact(compact: CompactActions, encoded) -> List[LegalActionPrior]:
    """Actor side: the LegalActionPrior list from the compact arrays,
    with action dicts built from the encoded state's positions."""
    hexes = encoded.hex_positions
    upos = encoded.unit_positions
    rtypes = encoded.recruit_types
    U = encoded.unit_tokens.size(1)
    out: List[LegalActionPrior] = []
    for a, k, h, w, pr in zip(compact.actor.tolist(), compact.kind.tolist(),
                              compact.target.tolist(), compact.weapon.tolist(),
                              compact.prior.tolist()):
        if k == KIND_ATTACK:
            out.append(LegalActionPrior(
                action={"type": "attack", "start_hex": upos[a], "target_hex": hexes[h],
                        "attack_index": w},
                prior=pr, actor_idx=a, target_idx=h, weapon_idx=w,
                type_idx=UnitActionType.ATTACK))
        elif k == KIND_MOVE:
            out.append(LegalActionPrior(
                action={"type": "move", "start_hex": upos[a], "target_hex": hexes[h]},
                prior=pr, actor_idx=a, target_idx=h, weapon_idx=None,
                type_idx=UnitActionType.MOVE))
        elif k == KIND_RECRUIT:
            out.append(LegalActionPrior(
                action={"type": "recruit", "unit_type": rtypes[a - U], "target_hex": hexes[h]},
                prior=pr, actor_idx=a, target_idx=h, weapon_idx=None, type_idx=None))
        else:
            out.append(LegalActionPrior(
                action={"type": "end_turn"}, prior=pr, actor_idx=a,
                target_idx=None, weapon_idx=None, type_idx=None))
    return out


__all__ = ["PackedMasks", "CompactActions", "pack_masks", "batched_priors",
           "unpack_compact", "ActorKind"]
