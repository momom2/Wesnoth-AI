"""The imitation loss over one padded batch.

`tools/supervised_train.py`'s batched flow used to finalize every pair
into its own EncodedState (about fifteen host-to-device copies and as
many embedding kernels per pair, four of the copies blocking) and to
compute every pair's cross-entropies one small kernel at a time. Here
the whole batch's label indices cross the bus in one pinned buffer and
each head's smoothed cross-entropy runs once over the PaddedOutput.

The numbers are the per-sample reference's (`_loss_parts_for_output`;
tests/test_imitation_flat_batch.py compares values and gradients).
torch's label-smoothed cross-entropy on one sample with mean reduction
is (1 - eps) * nll + eps / C * sum_c(-log p_c); with class weights (the
type head) the sum is over w_c * (-log p_c) / w_y
(aten/src/ATen/native/LossNLL.cpp, cross_entropy_loss_label_smoothing).
Pads are excluded from a row's C classes, as the per-sample views
exclude them.

Recipe note: the batched flow backpropagates the UNWEIGHTED actor
cross-entropy (the action-type weights scale only the per-pair CPU
flow's total); seed and seed2 trained this way, so `total` keeps it.
`actor_weighted` is reported for the record.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from wesnoth_ai.packed_trunk import FlatLayout

LABEL_SMOOTHING = 0.05
_NEG_INF = float("-inf")


@dataclass
class ImitationTargets:
    """One batch's labels as host arrays in one flat buffer (pinned on
    cuda). `*_ok` flags say which heads fire per sample, exactly the
    per-sample reference's conditions; absent indices are 0."""
    layout: FlatLayout
    host: torch.Tensor
    ok: Dict[str, np.ndarray]          # host copies of the flags, for the log
    policy_w: np.ndarray
    n: int

    def views(self, device: torch.device) -> Dict[str, torch.Tensor]:
        buf = self.host if device.type == "cpu" else self.host.to(device, non_blocking=True)
        return self.layout.torch_views(buf)


def build_imitation_targets(
    ais: Sequence,                                   # ActionIndices per sample
    zw: Sequence[Tuple[Optional[int], float, float]],   # (value z, value weight, policy weight)
    sizes: Sequence[Tuple[int, int, int]],           # (U, R, H) per sample
    *,
    n_types: int,
    n_weapons: int,
    n_atoms: int,
    type_loss_weights: Dict[str, float],
    device: torch.device,
) -> ImitationTargets:
    B = len(ais)
    i64 = np.zeros
    actor_idx, type_idx, target_idx, weapon_idx, value_edge = (i64(B, np.int64) for _ in range(5))
    a_len, h_len = np.zeros(B, np.int64), np.zeros(B, np.int64)
    actor_ok, type_ok, target_ok, weapon_ok, value_ok = (np.zeros(B, np.float32) for _ in range(5))
    actor_w, value_w, policy_w = (np.zeros(B, np.float32) for _ in range(3))
    for b, (ai, (z, vw, pw), (U, R, H)) in enumerate(zip(ais, zw, sizes)):
        A = U + R + 1
        a_len[b], h_len[b] = A, H
        policy_w[b] = pw
        actor_w[b] = type_loss_weights.get(ai.action_type, 1.0)
        if ai.actor_idx >= A:
            continue                                  # the pair contributes nothing
        actor_ok[b] = 1.0
        actor_idx[b] = ai.actor_idx
        if ai.type_idx is not None and 0 <= ai.type_idx < n_types:
            type_ok[b], type_idx[b] = 1.0, ai.type_idx
        if (ai.target_idx is not None and ai.action_type != "end_turn"
                and ai.target_idx < H):
            target_ok[b], target_idx[b] = 1.0, ai.target_idx
        if ai.weapon_idx is not None and ai.weapon_idx < n_weapons:
            weapon_ok[b], weapon_idx[b] = 1.0, ai.weapon_idx
        if z is not None and vw > 0.0:
            value_ok[b], value_w[b] = 1.0, vw
            value_edge[b] = n_atoms - 1 if z > 0 else 0
    class_w = np.array([type_loss_weights.get("attack", 1.0),
                        type_loss_weights.get("move", 1.0)], dtype=np.float32)
    if n_types != class_w.shape[0]:
        class_w = np.ones(n_types, dtype=np.float32)
    arrays = {"actor_idx": actor_idx, "type_idx": type_idx, "target_idx": target_idx,
              "weapon_idx": weapon_idx, "value_edge": value_edge, "a_len": a_len, "h_len": h_len,
              "actor_ok": actor_ok, "type_ok": type_ok, "target_ok": target_ok,
              "weapon_ok": weapon_ok, "value_ok": value_ok, "value_w": value_w,
              "policy_w": policy_w, "actor_w": actor_w, "class_w": class_w}
    layout = FlatLayout([(k, torch.from_numpy(v).dtype, tuple(v.shape)) for k, v in arrays.items()])
    host = torch.empty(layout.nbytes, dtype=torch.uint8, pin_memory=(device.type == "cuda"))
    views = layout.numpy_views(host.numpy())
    for k, v in arrays.items():
        views[k][...] = v
    ok = {"actor": actor_ok > 0, "type": type_ok > 0, "target": target_ok > 0,
          "weapon": weapon_ok > 0, "value": value_ok > 0}
    return ImitationTargets(layout, host, ok, policy_w, B)


@dataclass
class ImitationLossParts:
    """Per-sample [B] cross-entropies (zero where a head did not fire)
    and the batch's summed loss (the trainer divides by the batch
    size). `actor_raw`, `type`, `target`, `weapon`, `value_raw` are what
    the per-sample reference reports for the log."""
    total: torch.Tensor
    actor_raw: torch.Tensor
    actor_weighted: torch.Tensor
    type: torch.Tensor
    target: torch.Tensor
    weapon: torch.Tensor
    value_raw: torch.Tensor
    value: torch.Tensor

    def log_tensor(self) -> torch.Tensor:
        """[5, B] on the loss's own device: actor_raw, type, target,
        weapon, value_raw per sample. Kept on the device so the caller
        can finish the step before it pays for the transfer -- reading
        this straight after `backward()` is a hard sync that stops the
        host from queueing the optimizer step behind the backward."""
        return torch.stack([self.actor_raw, self.type, self.target,
                            self.weapon, self.value_raw]).detach()

    def log_values(self) -> List[List[float]]:
        """One transfer: [actor_raw, type, target, weapon, value_raw] per sample."""
        return self.log_tensor().cpu().tolist()


def _smoothed_ce(logits: torch.Tensor, idx: torch.Tensor, valid: Optional[torch.Tensor],
                 n_classes: torch.Tensor) -> torch.Tensor:
    """Per-row label-smoothed cross-entropy over the row's first
    n_classes entries (valid: [B, C] bool, None when every entry counts)."""
    if valid is not None:
        logits = logits.masked_fill(~valid, _NEG_INF)
    logp = F.log_softmax(logits, dim=-1)
    nll = -logp.gather(1, idx.unsqueeze(1)).squeeze(1)
    neg = -logp if valid is None else torch.where(valid, -logp, torch.zeros_like(logp))
    smooth = neg.sum(dim=1) / n_classes.to(logp.dtype)
    return (1.0 - LABEL_SMOOTHING) * nll + LABEL_SMOOTHING * smooth


def imitation_loss_parts(padded, targets: ImitationTargets,
                         eps: float = LABEL_SMOOTHING) -> ImitationLossParts:
    """The batch's imitation loss from a model.PaddedOutput."""
    dev = padded.actor_logits.device
    t = targets.views(dev)
    B, A_max = padded.actor_logits.shape
    bidx = torch.arange(B, device=dev)

    a_valid = torch.arange(A_max, device=dev).unsqueeze(0) < t["a_len"].unsqueeze(1)
    actor_raw = _smoothed_ce(padded.actor_logits, t["actor_idx"], a_valid, t["a_len"]) * t["actor_ok"]

    type_rows = padded.type_logits[bidx, t["actor_idx"]]                 # [B, T]
    T = type_rows.shape[1]
    logp_t = F.log_softmax(type_rows, dim=-1)
    nll_t = -logp_t.gather(1, t["type_idx"].unsqueeze(1)).squeeze(1)
    w_y = t["class_w"][t["type_idx"]]
    smooth_t = (-logp_t * t["class_w"].unsqueeze(0)).sum(dim=1) / (w_y * T)
    type_loss = ((1.0 - eps) * nll_t + eps * smooth_t) * t["type_ok"]

    target_rows = padded.target_logits[bidx, t["actor_idx"]]             # [B, H_max]
    h_valid = torch.arange(target_rows.shape[1], device=dev).unsqueeze(0) < t["h_len"].unsqueeze(1)
    target_loss = _smoothed_ce(target_rows, t["target_idx"], h_valid, t["h_len"]) * t["target_ok"]

    weapon_rows = padded.weapon_logits[bidx, t["actor_idx"]]             # [B, W]
    W = torch.full((B,), weapon_rows.shape[1], device=dev)
    weapon_loss = _smoothed_ce(weapon_rows, t["weapon_idx"], None, W) * t["weapon_ok"]

    logp_v = F.log_softmax(padded.value_logits, dim=-1)                 # [B, K]
    value_raw = -logp_v.gather(1, t["value_edge"].unsqueeze(1)).squeeze(1) * t["value_ok"] * t["actor_ok"]
    value = value_raw * t["value_w"]

    total = ((actor_raw + type_loss + target_loss + weapon_loss) * t["policy_w"]).sum() + value.sum()
    return ImitationLossParts(total, actor_raw, actor_raw * t["actor_w"], type_loss, target_loss,
                              weapon_loss, value_raw, value)
