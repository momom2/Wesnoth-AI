"""A time step's inputs for the sequence trainer (tools/sequence_train.py):
the staged batch, the imitation and belief targets and the own rows' batch.

`RecordInputs` stages them from the pickled records, each position a
`SequencePosition`. `ColumnInputs` stages the same bytes from columnar
records (wesnoth_ai/sequence_columns.py, written by
tools/columnize_sequences.py): each field of a time step's rows is a slice of
its game-side's array, copied into the pinned buffer by one concatenation per
field, the own rows' masks by one indexed write per field, and no Python
object exists per position (tests/test_sequence_columns.py compares the two
byte for byte).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from tools.sequence_timing import position_sizes
from wesnoth_ai.encoder import MAX_FACTIONS, StagedBatch
from wesnoth_ai.imitation_loss import ImitationTargets, build_imitation_targets
from wesnoth_ai.own_loss import _TARGETS_ALIGNMENT, OwnBatch
from wesnoth_ai.packed_trunk import FlatLayout
from wesnoth_ai.sequence_columns import ColumnarSide
from wesnoth_ai.sequence_loss import step_labels
from wesnoth_ai.sequence_streams import Step
from wesnoth_ai.server_priors import (KIND_END_TURN, _SLOT_END, _SLOT_RECRUIT, _SLOT_UNIT,
                                      mask_layout)


@dataclass
class StepInputs:
    """One time step's inputs but the memory, on the device."""
    staged: StagedBatch
    targets: ImitationTargets
    belief_target: torch.Tensor
    belief_mask: torch.Tensor
    own: Optional[OwnBatch]


def head_shapes(model) -> Tuple[int, int, int]:
    """(action types, weapons, value atoms): the widths of the heads' outputs."""
    return model.type_head.out_features, model.weapon_head[-1].out_features, model.value_head[-1].out_features


class _Inputs:
    """What both paths share: the run's facts and settings."""

    def __init__(self, loader, encoder, model, own, games, args, value_weight: float,
                 type_loss_weights: Dict[str, float], device: torch.device):
        self.loader, self.encoder, self.own, self.games = loader, encoder, own, games
        self.seed, self.value_states_per_game = args.seed, args.value_states_per_game
        self.value_weight, self.type_loss_weights, self.device = value_weight, type_loss_weights, device
        self.head_shapes = head_shapes(model)
        self.pin = device.type == "cuda"

    def _labelled(self, label_rows, steps, staged: StagedBatch) -> Tuple[ImitationTargets, torch.Tensor,
                                                                          torch.Tensor]:
        """The imitation targets and the belief target and mask of rows that
        carry `label`, `no_visible_unit` and `hidden_tokens`."""
        labels = step_labels(label_rows, steps, staged.sizes, self.games, seed=self.seed,
                             value_states_per_game=self.value_states_per_game, value_weight=self.value_weight,
                             pin=self.pin)
        n_types, n_weapons, n_atoms = self.head_shapes
        targets = build_imitation_targets(labels.ais, labels.zw, staged.sizes, n_types=n_types,
                                          n_weapons=n_weapons, n_atoms=n_atoms,
                                          type_loss_weights=self.type_loss_weights, device=self.device)
        b_target, b_mask = labels.belief_on(self.device)
        return targets, b_target, b_mask

    def own_rows(self, steps: Sequence[Step]) -> List[int]:
        return [] if self.own is None else [b for b, s in enumerate(steps) if s.game_side.file in self.own.files]


class RecordInputs(_Inputs):
    """The inputs from the pickled records (`SequenceLoader`)."""

    def positions(self, steps: Sequence[Step]) -> List:
        return [self.loader.sides(s.game_side)[s.offset] for s in steps]

    def sizes(self, steps: Sequence[Step]) -> Tuple[list, list]:
        """The rows' (U, R, H) and sighting counts, as staging gives them."""
        return position_sizes([p.raw for p in self.positions(steps)])

    def stage(self, steps: Sequence[Step]) -> StepInputs:
        positions = self.positions(steps)
        staged = self.encoder.stage_raws([p.raw for p in positions], device=self.device)
        targets, b_target, b_mask = self._labelled(positions, steps, staged)
        own = None
        if self.own is not None:
            n_types, n_weapons, _ = self.head_shapes
            own = self.own.batch(steps, positions, staged.sizes, n_types, n_weapons, self.device)
        return StepInputs(staged, targets, b_target, b_mask, own)


class _LabelRow:
    """What `step_labels` reads of a position, from its columns."""
    __slots__ = ("label", "no_visible_unit", "hidden_tokens")

    def __init__(self, side: ColumnarSide, i: int):
        self.label = side.label(i)
        lo, hi = side.span("hex", i)
        self.no_visible_unit = side.arrays["no_visible_unit"][lo:hi]
        lo, hi = side.span("hidden", i)
        self.hidden_tokens = side.arrays["hidden_tokens"][lo:hi]


class ColumnInputs(_Inputs):
    """The inputs from columnar records (`ColumnLoader`)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._checked = set()

    def rows(self, steps: Sequence[Step]) -> List[Tuple[ColumnarSide, int]]:
        rows = []
        for s in steps:
            side = self.loader.side(s.game_side)
            if s.game_side not in self._checked:
                # The widths this encoder reads, once per game-side (stage_raws checks every batch's first row).
                if len(side):
                    self.encoder._check_raw(side.position(0).raw)
                self._checked.add(s.game_side)
            rows.append((side, s.offset))
        return rows

    def sizes(self, steps: Sequence[Step]) -> Tuple[list, list]:
        rows = self.rows(steps)
        return ([(_count(side, "unit", i), _count(side, "recruit", i), _count(side, "hex", i)) for side, i in rows],
                [_count(side, "sighting", i) for side, i in rows])

    def stage(self, steps: Sequence[Step]) -> StepInputs:
        rows = self.rows(steps)
        staged = stage_columns(self.encoder, rows, self.device)
        targets, b_target, b_mask = self._labelled([_LabelRow(side, i) for side, i in rows], steps, staged)
        own = None
        own_rows = self.own_rows(steps)
        if own_rows:
            sizes = staged.sizes
            A_max = max(u for u, _, _ in sizes) + max(r for _, r, _ in sizes) + 1
            H_max = max(h for _, _, h in sizes)
            n_types, n_weapons, _ = self.head_shapes
            own = stage_own_columns(own_rows, [rows[b] for b in own_rows], A_max, H_max, n_types, n_weapons,
                                    self.device)
        return StepInputs(staged, targets, b_target, b_mask, own)


def _count(side: ColumnarSide, group: str, i: int) -> int:
    lo, hi = side.span(group, i)
    return hi - lo


def _ragged_arange(counts: np.ndarray) -> np.ndarray:
    """0 .. c-1 for every count c, concatenated."""
    starts = np.zeros(counts.size, dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    return np.arange(int(counts.sum()), dtype=np.int64) - np.repeat(starts, counts)


def _cat(rows, name: str, spans) -> np.ndarray:
    return np.concatenate([side.arrays[name][lo:hi] for (side, _), (lo, hi) in zip(rows, spans)])


# ---- the staged batch -------------------------------------------------------------

def stage_columns(encoder, rows: Sequence[Tuple[ColumnarSide, int]], device: torch.device) -> StagedBatch:
    """`GameStateEncoder.stage_raws` of the rows' positions, from their
    columns: the same layout, bytes and views."""
    B = len(rows)
    if B == 0:
        raise ValueError("stage_columns: empty batch")
    spec = encoder._stream_fields()
    spans = {stream: [side.span(stream, i) for side, i in rows] for stream, _ in spec}
    counts = {stream: [hi - lo for lo, hi in sp] for stream, sp in spans.items()}
    totals = {stream: sum(c) for stream, c in counts.items()}
    their = (("their_faction_probs", torch.float32, (B, MAX_FACTIONS))
             if encoder.observation_parity else ("their_faction_id", torch.int64, (B,)))
    fields = [(name, dt, (totals[stream],) + shape)
              for stream, stream_fields in spec for name, dt, shape in stream_fields]
    fields += [("global_feats", torch.float32, (B, encoder.widths.global_feats)),
               ("our_faction_id", torch.int64, (B,)), their]
    layout = FlatLayout(fields)
    host = torch.empty(layout.nbytes, dtype=torch.uint8, pin_memory=(device.type == "cuda"))
    hv = layout.numpy_views(host.numpy())
    for stream, stream_fields in spec:
        if totals[stream]:
            sp = spans[stream]
            for name, _, _ in stream_fields:
                np.concatenate([side.arrays[name][lo:hi] for (side, _), (lo, hi) in zip(rows, sp)], out=hv[name])
    np.stack([side.arrays["global_feats"][i] for side, i in rows], out=hv["global_feats"])
    hv["our_faction_id"][:] = [side.arrays["our_faction_id"][i] for side, i in rows]
    if encoder.observation_parity:
        np.stack([side.arrays["their_faction_probs"][i] for side, i in rows], out=hv["their_faction_probs"])
    else:
        hv["their_faction_id"][:] = [side.arrays["their_faction_id"][i] for side, i in rows]
    dev = host if device.type == "cpu" else host.to(device, non_blocking=True)
    return StagedBatch(views=layout.torch_views(dev), totals=totals,
                       sizes=list(zip(counts["unit"], counts["recruit"], counts["hex"])),
                       sighting_counts=counts.get("sighting"), their_field=their[0])


# ---- the own rows ----------------------------------------------------------------------

def stage_own_columns(rows: Sequence[int], own: Sequence[Tuple[ColumnarSide, int]], A_max: int, H_max: int,
                      n_types: int, n_weapons: int, device: torch.device) -> OwnBatch:
    """`own_loss.stage_own_batch` of the own rows `rows` of a batch, from
    their columns (`own`: each row's game-side and position): the same
    buffer, bytes and views."""
    n = len(rows)
    masks = mask_layout(n, A_max, H_max, n_types, n_weapons)
    target_spans = [side.span("target", i) for side, i in own]
    K = max((hi - lo for lo, hi in target_spans), default=1)
    layout = FlatLayout([("rows", torch.int64, (n,))]
                        + [(name, torch.int64, (n, K)) for name in ("actor", "kind", "target", "weapon")]
                        + [("prob", torch.float32, (n, K))])
    base = -(-masks.nbytes // _TARGETS_ALIGNMENT) * _TARGETS_ALIGNMENT
    host = torch.zeros(base + layout.nbytes, dtype=torch.uint8, pin_memory=(device.type == "cuda"))
    write_mask_columns(masks.numpy_views(host.numpy()[:masks.nbytes]), own)
    write_target_columns(layout.numpy_views(host.numpy()[base:]), rows, own, target_spans)
    buf = host if device.type == "cpu" else host.to(device, non_blocking=True)
    t = layout.torch_views(buf[base:])
    return OwnBatch(rows=t["rows"], views=masks.torch_views(buf[:masks.nbytes]), actor=t["actor"],
                    kind=t["kind"], target=t["target"], weapon=t["weapon"], prob=t["prob"])


def _scalars(own, name: str) -> np.ndarray:
    return np.array([side.arrays[name][i] for side, i in own])


def write_mask_columns(hv: Dict[str, np.ndarray], own: Sequence[Tuple[ColumnarSide, int]]) -> None:
    """`server_priors.write_masks` into ZEROED views, from columns: row b's
    masks at row b, one indexed write per field."""
    n = len(own)
    U, R = (_scalars(own, k).astype(np.int64) for k in ("mask_units", "mask_recruits"))
    A = U + R + 1
    actor_spans = [side.span("actor", i) for side, i in own]
    if [hi - lo for lo, hi in actor_spans] != A.tolist():
        raise ValueError("an own position's actor masks are not its units, recruits and end_turn")
    rows_a = np.repeat(np.arange(n), A)
    cols_a = _ragged_arange(A)
    hv["actor_mask"][rows_a, cols_a] = _cat(own, "actor_mask", actor_spans)
    hv["slot_kind"][rows_a, cols_a] = np.where(cols_a < U[rows_a], _SLOT_UNIT,
                                               np.where(cols_a < (U + R)[rows_a], _SLOT_RECRUIT, _SLOT_END))
    hv["n_attacks"][rows_a, cols_a] = _cat(own, "n_attacks", actor_spans)
    hv["type_valid"][rows_a, cols_a] = _cat(own, "type_valid", actor_spans)
    width = _scalars(own, "mask_bits_width").astype(np.int64)
    bit_spans = [side.span("bits", i) for side, i in own]
    if [hi - lo for lo, hi in bit_spans] != (A * width).tolist():
        raise ValueError("an own position's target bits are not one row per actor")
    rows_k, a_k, c_k = _blocks(A, width)
    for column, field in (("attack_valid", "attack_bits"), ("move_valid", "move_bits"),
                          ("union_valid", "union_bits")):
        hv[field][rows_k, a_k, c_k] = _cat(own, column, bit_spans)


def _blocks(rows_per: np.ndarray, width: np.ndarray):
    """(row, block row, block column) of every entry of per-row blocks of
    `rows_per` x `width`, row after row, each block row-major."""
    sizes = rows_per * width
    rows = np.repeat(np.arange(len(sizes)), sizes)
    k = _ragged_arange(sizes)
    w = np.repeat(width, sizes)
    return rows, k // np.maximum(w, 1), k % np.maximum(w, 1)


def write_target_columns(hv: Dict[str, np.ndarray], rows: Sequence[int], own: Sequence[Tuple[ColumnarSide, int]],
                         spans: Sequence[Tuple[int, int]]) -> None:
    """`own_loss._write_targets` into ZEROED views, from columns."""
    hv["rows"][:] = np.asarray(list(rows), dtype=np.int64)
    hv["kind"][:] = KIND_END_TURN
    m = np.array([hi - lo for lo, hi in spans], dtype=np.int64)
    rows_t = np.repeat(np.arange(len(own)), m)
    cols_t = _ragged_arange(m)
    hv["actor"][rows_t, cols_t] = _cat(own, "target_actor", spans)
    hv["kind"][rows_t, cols_t] = _cat(own, "target_kind", spans)
    hv["target"][rows_t, cols_t] = np.maximum(_cat(own, "target_target", spans), 0)
    hv["weapon"][rows_t, cols_t] = np.maximum(_cat(own, "target_weapon", spans), 0)
    # OwnTarget.prob, position by position: the kept priors over their own sum.
    priors = [side.arrays["target_prior"][lo:hi] for (side, _), (lo, hi) in zip(own, spans)]
    hv["prob"][rows_t, cols_t] = np.concatenate([(p / p.sum()).astype(np.float32) for p in priors])
