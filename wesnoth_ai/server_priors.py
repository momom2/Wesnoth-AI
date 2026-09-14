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

Device traffic per batch (docs/gpu_forward_design_20260904.md §6.2):
every per-batch mask array is written into ONE pinned host buffer and
copied with one non-blocking transfer; the bit masks are unpacked on
the device; the legal entries are compacted on the device by a prefix
sum over a flat layout that IS the reference order, into a buffer
whose capacity is counted from the packed bits on the host (exact, so
no readback is needed to size it); the results and the ride-along
head outputs go back in ONE buffer with one wait at the end. Nothing
in between synchronizes with the host (no `nonzero`, no boolean-mask
indexing, no `.item()`), which tests/test_server_priors_cuda.py checks
under `torch.cuda.set_sync_debug_mode("error")`.

Kinds in the compact arrays: 0 attack, 1 move, 2 recruit, 3 end_turn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from wesnoth_ai.action_sampler import (
    _NEG_INF, LegalActionPrior, _build_legality_masks, prior_bias_end_turn,
)
from wesnoth_ai.model import MAX_ATTACKS, ActorKind, UnitActionType
from wesnoth_ai.packed_trunk import FlatLayout as _Layout

KIND_ATTACK, KIND_MOVE, KIND_RECRUIT, KIND_END_TURN = 0, 1, 2, 3
# Actor-slot kinds in the staging buffer; 0 marks a padded slot.
_SLOT_UNIT, _SLOT_RECRUIT, _SLOT_END = 1, 2, 3
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


def _popcount(a: np.ndarray) -> np.ndarray:
    """Bits set per byte of a uint8 array: numpy 2's bitwise_count, or
    the table (indexed by intp: a uint8 index array goes through a
    slower conversion path)."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(a)
    return _POPCOUNT[a.astype(np.intp)]


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


# ---------------------------------------------------------------------
# One flat buffer each way
# ---------------------------------------------------------------------

def _legal_capacity(p: PackedMasks, W: int) -> int:
    """Upper bound on one leaf's legal entries, from its masks. The
    device keeps an entry only where every factor's softmax is > 0,
    and a masked factor is exactly 0, so the bits bound the count;
    `_build_legality_masks` sets an actor/type bit only when the
    matching target row is non-empty, so no factor row is all-masked
    (an all-masked row would softmax to uniform > 0)."""
    U, R = p.n_units, p.n_recruits
    act = p.actor_mask != 0
    tv = p.type_valid != 0
    atk = _POPCOUNT[p.attack_valid].sum(axis=1)
    mv = _POPCOUNT[p.move_valid].sum(axis=1)
    uni = _POPCOUNT[p.union_valid].sum(axis=1)
    n_att = np.minimum(p.n_attacks.astype(np.int64), W)
    unit = act[:U]
    n = int((unit * tv[:U, UnitActionType.ATTACK] * atk[:U] * n_att[:U]).sum())
    n += int((unit * tv[:U, UnitActionType.MOVE] * mv[:U]).sum())
    n += int((act[U:U + R] * uni[U:U + R]).sum())
    n += int(act[U + R])
    return n


def _staged_capacity(hv: Dict[str, np.ndarray], W: int) -> int:
    """`_legal_capacity` summed over the batch, computed once on the
    staged views: pad slots hold actor_mask 0 and zero bits, so they add
    nothing, and the count equals the per-leaf sum exactly. About 15
    numpy calls per batch instead of per leaf; the per-leaf loop was the
    serve thread's largest host-side item in the priors (2.4-4.5 ms per
    16-leaf batch on the laptop, against 0.3 ms here)."""
    act = hv["actor_mask"] != 0                                            # [B, A]
    slot = hv["slot_kind"]
    tv = hv["type_valid"] != 0                                             # [B, A, T]
    atk = _popcount(hv["attack_bits"]).sum(axis=2, dtype=np.int64)         # [B, A]
    mv = _popcount(hv["move_bits"]).sum(axis=2, dtype=np.int64)
    uni = _popcount(hv["union_bits"]).sum(axis=2, dtype=np.int64)
    n_att = np.minimum(hv["n_attacks"].astype(np.int64), W)
    unit = act & (slot == _SLOT_UNIT)
    n = int((unit * tv[:, :, UnitActionType.ATTACK] * atk * n_att).sum())
    n += int((unit * tv[:, :, UnitActionType.MOVE] * mv).sum())
    n += int(((act & (slot == _SLOT_RECRUIT)) * uni).sum())
    n += int((act & (slot == _SLOT_END)).sum())
    return n


def mask_layout(B: int, A_max: int, H_max: int, T: int, W: int, *,
                type_bias: bool, attack_bias: bool) -> _Layout:
    """The flat host layout of a batch's masks. The two bias fields are
    laid out on request: the eager path asks for one when any pack
    carries it, the static-shape path (graphed_serve) always."""
    HB = (H_max + 7) // 8
    fields = [
        ("actor_bias", torch.float32, (B, A_max)),
        ("actor_mask", torch.uint8, (B, A_max)),
        ("slot_kind", torch.uint8, (B, A_max)),
        ("n_attacks", torch.int8, (B, A_max)),
        ("type_valid", torch.uint8, (B, A_max, T)),
        ("attack_bits", torch.uint8, (B, A_max, HB)),
        ("move_bits", torch.uint8, (B, A_max, HB)),
        ("union_bits", torch.uint8, (B, A_max, HB)),
    ]
    if type_bias:
        fields.append(("type_bias", torch.float32, (B, A_max, T)))
    if attack_bias:
        fields.append(("attack_bias", torch.float32, (B, A_max, H_max)))
    return _Layout(fields)


def write_masks(hv: Dict[str, np.ndarray], packs: Sequence[PackedMasks]) -> None:
    """Pack b's arrays into row b of the ZEROED views of a mask layout;
    a bias goes in where the layout has the field and the pack the
    array (a layout field no pack fills stays zero, which adds nothing)."""
    for b, p in enumerate(packs):
        U, R, H = p.n_units, p.n_recruits, p.n_hexes
        A = U + R + 1
        hb = p.attack_valid.shape[1]
        hv["actor_mask"][b, :A] = p.actor_mask
        hv["slot_kind"][b, :U] = _SLOT_UNIT
        hv["slot_kind"][b, U:U + R] = _SLOT_RECRUIT
        hv["slot_kind"][b, U + R] = _SLOT_END
        hv["actor_bias"][b, U + R] = p.end_turn_bias
        hv["n_attacks"][b, :A] = p.n_attacks
        hv["type_valid"][b, :A] = p.type_valid
        hv["attack_bits"][b, :A, :hb] = p.attack_valid
        hv["move_bits"][b, :A, :hb] = p.move_valid
        hv["union_bits"][b, :A, :hb] = p.union_valid
        if p.type_bias is not None and "type_bias" in hv:
            hv["type_bias"][b, :A] = p.type_bias
        if p.attack_bias is not None and "attack_bias" in hv:
            hv["attack_bias"][b, :A, :H] = p.attack_bias


def _stage_masks(packs: List[PackedMasks], A_max: int, H_max: int, T: int, W: int,
                 pin: bool) -> Tuple[_Layout, torch.Tensor, int]:
    """Host side: every per-batch mask array written into one flat
    (pinned) host buffer, plus the compaction capacity counted once
    over the staged views."""
    layout = mask_layout(len(packs), A_max, H_max, T, W,
                         type_bias=any(p.type_bias is not None for p in packs),
                         attack_bias=any(p.attack_bias is not None for p in packs))
    host = torch.empty(layout.nbytes, dtype=torch.uint8, pin_memory=pin)
    host.zero_()
    hv = layout.numpy_views(host.numpy())
    write_masks(hv, packs)
    return layout, host, _staged_capacity(hv, W)


def _as_bytes(t: torch.Tensor) -> torch.Tensor:
    """Flat uint8 view of a tensor's bytes; a copy when the flattened
    elements are not unit-stride (torch counts a size-1 dimension as
    contiguous at any stride, `view(dtype)` needs stride 1)."""
    x = t.reshape(-1)
    if x.numel() and x.stride(0) != 1:
        x = x.clone(memory_format=torch.contiguous_format)
    return x.view(torch.uint8)


def _unpack_bits(packed: torch.Tensor, H: int) -> torch.Tensor:
    """[..., ceil(H/8)] uint8 in numpy packbits order (first bit in
    the high position) -> [..., H] bool, on the packed tensor's device."""
    shifts = 7 - torch.arange(8, dtype=torch.uint8, device=packed.device)
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    return bits.reshape(*packed.shape[:-1], -1)[..., :H] != 0


@dataclass(slots=True)
class PendingPriors:
    """A batch whose device work and device->host copy are queued but
    not yet waited for; `finish` waits once and unpacks the buffer."""
    host: torch.Tensor          # uint8, pinned on CUDA
    layout: _Layout
    n_samples: int
    capacity: int
    n_extras: int
    device: torch.device

    def finish(self) -> Tuple[List[CompactActions], List[np.ndarray]]:
        """The compact actions per sample and the ride-along tensors
        as host arrays (views into the batch's buffer, in order)."""
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        hv = self.layout.numpy_views(self.host.numpy())
        ends = hv["ends"].astype(np.int64)
        total = int(ends[-1]) if self.n_samples else 0
        if total > self.capacity:
            raise RuntimeError(
                f"batched_priors: {total} legal entries exceed the mask-derived "
                f"capacity {self.capacity}; a legality mask admits fewer entries "
                f"than its factors")
        compact: List[CompactActions] = []
        lo = 0
        for b in range(self.n_samples):
            hi = int(ends[b])
            compact.append(CompactActions(
                actor=hv["actor"][lo:hi], kind=hv["kind"][lo:hi], target=hv["target"][lo:hi],
                weapon=hv["weapon"][lo:hi], prior=hv["prior"][lo:hi]))
            lo = hi
        return compact, [hv[f"extra{i}"] for i in range(self.n_extras)]


def start_priors(padded, packs: List[PackedMasks],
                 extras: Sequence[torch.Tensor] = ()) -> PendingPriors:
    """Server side, first half: stage the masks (one host->device
    copy), run the masked softmaxes and the compaction on the device,
    queue the one device->host copy of the results together with
    `extras` (device tensors the caller wants back in the same
    transfer, e.g. the value head). Nothing here waits for the
    device; `PendingPriors.finish` does, once.

    Priors are float64 products of the float32 factors in the
    reference's order of multiplication; entries per sample come out
    in the reference order (actor-major; a unit actor's attacks
    (hex-major, weapon inner) before its moves; recruit actors;
    end_turn) because the flat layout the prefix sum runs over is
    that order. `padded` is a model.PaddedOutput (device tensors)."""
    B = len(packs)
    device = padded.actor_logits.device
    pin = device.type == "cuda"
    A_max, T = padded.type_logits.shape[1], padded.type_logits.shape[2]
    H_max, W = padded.target_logits.shape[2], padded.weapon_logits.shape[2]
    layout, host, capacity = _stage_masks(packs, A_max, H_max, T, W, pin)
    with torch.no_grad():
        v = layout.torch_views(host.to(device, non_blocking=True))
        out_layout, flat_out = priors_outputs(padded, v, capacity, extras)
        host_out = torch.empty(out_layout.nbytes, dtype=torch.uint8, pin_memory=pin)
        host_out.copy_(flat_out, non_blocking=True)
    return PendingPriors(host=host_out, layout=out_layout, n_samples=B, capacity=capacity,
                         n_extras=len(extras), device=device)


def priors_outputs(padded, v: Dict[str, torch.Tensor], capacity: int,
                   extras: Sequence[torch.Tensor] = ()) -> Tuple[_Layout, torch.Tensor]:
    """The device side of `start_priors` on the staged mask views `v`:
    the masked softmaxes, the compaction of the first `capacity` legal
    entries in the reference order, and the one flat byte tensor the
    host reads back, with its layout. Shapes depend only on the padded
    sizes, `capacity` and the extras, so with those fixed the whole
    chain is static: graphed_serve captures it in a CUDA graph."""
    B, A_max, T = padded.type_logits.shape
    H_max, W = padded.target_logits.shape[2], padded.weapon_logits.shape[2]
    device = padded.actor_logits.device
    with torch.no_grad():
        actor_ok = v["actor_mask"] != 0
        slot = v["slot_kind"]
        al = (padded.actor_logits + v["actor_bias"]).masked_fill(~actor_ok, _NEG_INF)
        p_actor = F.softmax(al, dim=-1)                                       # [B, A]
        live = actor_ok & (p_actor > 0)
        tl = padded.type_logits
        if "type_bias" in v:
            tl = tl + v["type_bias"]
        p_type = F.softmax(tl.masked_fill(v["type_valid"] == 0, _NEG_INF), dim=-1)  # [B, A, T]
        n_att = v["n_attacks"]
        w_ok = torch.arange(W, device=device).view(1, 1, W) < n_att.unsqueeze(-1)
        p_wpn = F.softmax(padded.weapon_logits.masked_fill(~w_ok, _NEG_INF), dim=-1)
        tg = padded.target_logits
        tga = tg + v["attack_bias"] if "attack_bias" in v else tg
        p_atk = F.softmax(tga.masked_fill(~_unpack_bits(v["attack_bits"], H_max), _NEG_INF), dim=-1)
        p_mv = F.softmax(tg.masked_fill(~_unpack_bits(v["move_bits"], H_max), _NEG_INF), dim=-1)
        p_rec = F.softmax(tg.masked_fill(~_unpack_bits(v["union_bits"], H_max), _NEG_INF), dim=-1)
        # Legal entries by FACTOR masks (a float32 product may underflow
        # where the reference's float64 product does not).
        unit_live = live & (slot == _SLOT_UNIT)
        atk_ok = unit_live & (p_type[:, :, UnitActionType.ATTACK] > 0) & (n_att > 0)
        mv_ok = unit_live & (p_type[:, :, UnitActionType.MOVE] > 0)
        m_att = (atk_ok[:, :, None, None] & (p_atk > 0)[:, :, :, None]
                 & (p_wpn > 0)[:, :, None, :])                                # [B, A, H, W]
        m_mv = mv_ok[:, :, None] & (p_mv > 0)
        m_rec = (live & (slot == _SLOT_RECRUIT))[:, :, None] & (p_rec > 0)
        m_end = live & (slot == _SLOT_END)
        # Flat layout per (sample, actor): attacks [H, W], moves [H],
        # recruits [H], end_turn [1] -- the reference order -- so the
        # prefix sum ranks the legal entries directly.
        HW = H_max * W
        per_slot = HW + 2 * H_max + 1
        flat = torch.cat([m_att.reshape(B, A_max, HW), m_mv, m_rec, m_end[:, :, None]],
                         dim=2).reshape(-1)
        cnt = flat.cumsum(0, dtype=torch.int32)
        ends = cnt.view(B, A_max * per_slot)[:, -1]                           # [B] inclusive
        ks = torch.arange(1, capacity + 1, dtype=torch.int32, device=device)
        # Position of the k-th legal entry; slots past the count land
        # on the last position, a valid index that `finish` never reads.
        pos = torch.searchsorted(cnt, ks).clamp_(max=cnt.numel() - 1)
        ba = pos // per_slot
        rem = pos - ba * per_slot
        bb = ba // A_max
        aa = ba - bb * A_max
        is_atk = rem < HW
        is_mv = (rem >= HW) & (rem < HW + H_max)
        is_rec = (rem >= HW + H_max) & (rem < HW + 2 * H_max)
        Wd = max(W, 1)
        kind = torch.where(is_atk, KIND_ATTACK, torch.where(
            is_mv, KIND_MOVE, torch.where(is_rec, KIND_RECRUIT, KIND_END_TURN)))
        hh = torch.where(is_atk, rem // Wd, torch.where(
            is_mv, rem - HW, torch.where(is_rec, rem - HW - H_max, -1)))
        ww = torch.where(is_atk, rem - (rem // Wd) * Wd, -1)
        # Factors gathered per entry; a factor that does not apply is
        # 1.0, so one product expression serves every kind exactly.
        f_actor = p_actor[bb, aa]
        tsel = torch.where(is_atk, UnitActionType.ATTACK, UnitActionType.MOVE)
        f_type = torch.where(is_atk | is_mv, p_type[bb, aa, tsel], 1.0)
        if H_max:
            hi = hh.clamp_min(0)
            f_target = torch.where(is_atk, p_atk[bb, aa, hi], torch.where(
                is_mv, p_mv[bb, aa, hi], torch.where(is_rec, p_rec[bb, aa, hi], 1.0)))
        else:
            f_target = torch.ones_like(f_actor)
        f_weapon = (torch.where(is_atk, p_wpn[bb, aa, ww.clamp_min(0)], 1.0) if W
                    else torch.ones_like(f_actor))
        prior = ((f_actor.double() * f_type.double()) * f_target.double()) * f_weapon.double()
        outs: Dict[str, torch.Tensor] = {
            "ends": ends, "actor": aa.to(torch.int32), "kind": kind.to(torch.int8),
            "target": hh.to(torch.int32), "weapon": ww.to(torch.int8), "prior": prior}
        for i, t in enumerate(extras):
            outs[f"extra{i}"] = t
        out_layout = _Layout([(n, t.dtype, tuple(t.shape)) for n, t in outs.items()])
        flat_out = torch.cat([_as_bytes(outs[f[0]]) for f in out_layout.fields])
    return out_layout, flat_out


def batched_priors(padded, packs: List[PackedMasks]) -> List[CompactActions]:
    """Server side: the compact legal actions of a batch (see
    `start_priors`), waiting for the device here."""
    if not packs:
        return []
    return start_priors(padded, packs).finish()[0]


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


def compact_action(compact: CompactActions, i: int, encoded) -> Dict:
    """The action dict of element `i` alone: what `unpack_compact(...)
    [i].action` builds, without building the other elements. The raw
    player picks by prior on the compact arrays and materializes one
    action (2026-09-11 worker profile: unpacking every legal action
    was a quarter of the worker's Python per decision)."""
    a, k, h = int(compact.actor[i]), int(compact.kind[i]), int(compact.target[i])
    if k == KIND_ATTACK:
        return {"type": "attack", "start_hex": encoded.unit_positions[a],
                "target_hex": encoded.hex_positions[h], "attack_index": int(compact.weapon[i])}
    if k == KIND_MOVE:
        return {"type": "move", "start_hex": encoded.unit_positions[a],
                "target_hex": encoded.hex_positions[h]}
    if k == KIND_RECRUIT:
        U = encoded.unit_tokens.size(1)
        return {"type": "recruit", "unit_type": encoded.recruit_types[a - U],
                "target_hex": encoded.hex_positions[h]}
    return {"type": "end_turn"}


__all__ = ["PackedMasks", "CompactActions", "PendingPriors", "pack_masks",
           "start_priors", "priors_outputs", "mask_layout", "write_masks",
           "batched_priors", "unpack_compact", "compact_action", "ActorKind"]
