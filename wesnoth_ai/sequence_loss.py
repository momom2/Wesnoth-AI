"""The sequence trainer's labels and belief loss at one time step
(docs/parity_memory_design_20260929.md "Training", "The belief head").

The policy and value losses are `obs8`'s (`wesnoth_ai.imitation_loss`):
the policy on the winners' decisions at the game's weight, none at a
position whose turn ran out of time; the value on the positions the run's
seed selects, against the game's outcome for the side to move.

The belief loss of a position is the binary cross-entropy of its belief
logits against "an enemy unit the side cannot see stands on this hex
token", averaged over its hex tokens with no visible unit, weight 1 beside
the others.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from wesnoth_ai.imitation_loss import TIMEOUT_LABEL
from wesnoth_ai.packed_trunk import FlatLayout
from wesnoth_ai.sequence_streams import Step, value_selected


class SequenceLabelError(ValueError):
    """A position's label or belief targets do not fit its encoding."""


@dataclass
class StepLabels:
    """A time step's labels. The belief target and mask are host views of
    one buffer (pinned for a CUDA trainer), which `belief_on` copies to the
    device in one non-blocking transfer."""
    ais: List[object]                       # ActionIndices per sample
    zw: List[Tuple[int, float, float]]      # (outcome for the mover, value weight, policy weight)
    belief_host: torch.Tensor               # uint8: the buffer behind the two fields below
    belief_layout: FlatLayout

    @property
    def belief_target(self) -> torch.Tensor:
        """[B, H_max] float32: 1 where an enemy unit the side cannot see stands."""
        return self.belief_layout.torch_views(self.belief_host)["belief_target"]

    @property
    def belief_mask(self) -> torch.Tensor:
        """[B, H_max] float32: the tokens with no visible unit."""
        return self.belief_layout.torch_views(self.belief_host)["belief_mask"]

    def belief_on(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """(belief target, belief mask) on `device`, from one copy of the buffer."""
        buf = self.belief_host if device.type == "cpu" else self.belief_host.to(device, non_blocking=True)
        views = self.belief_layout.torch_views(buf)
        return views["belief_target"], views["belief_mask"]


def _ragged_rows(lengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(row, column) of every entry of rows of these lengths, row after row."""
    starts = np.zeros(lengths.size, dtype=np.int64)
    np.cumsum(lengths[:-1], out=starts[1:])
    rows = np.repeat(np.arange(lengths.size, dtype=np.int64), lengths)
    return rows, np.arange(int(lengths.sum()), dtype=np.int64) - np.repeat(starts, lengths)


@dataclass(frozen=True)
class GameFacts:
    """What the losses read of a game: its winning side (0 when it ended
    without one), its commands and its policy weight (the per-game weight
    of `obs8`'s recipe; 0 for the memory player's own games, whose policy
    target is the teacher's distribution, `wesnoth_ai.own_loss`)."""
    winner: int
    n_commands: int
    policy_weight: float


def step_labels(positions: Sequence, steps: Sequence[Step], sizes: Sequence[Tuple[int, int, int]],
                games: Dict[str, GameFacts], *, seed: int, value_states_per_game: int,
                value_weight: float, pin: bool = False) -> StepLabels:
    """The labels of one time step's positions, in batch order. A label
    whose actor is not one of its encoding's slots raises: the pre-encoding
    checked every label against its tokens, so a mismatch is a corrupt
    record, never something to train around. `pin`: the belief buffer in
    pinned memory, for a copy to CUDA that does not wait."""
    B = len(positions)
    H_max = max((h for _, _, h in sizes), default=0)
    ais, zw = [], []
    for b, (pos, step, (U, R, H)) in enumerate(zip(positions, steps, sizes)):
        ai = pos.label
        timeout = ai.action_type == TIMEOUT_LABEL
        if not timeout and not 0 <= ai.actor_idx < U + R + 1:
            raise SequenceLabelError(f"{step.game_side}, decision {step.offset}: actor {ai.actor_idx} "
                                     f"outside {U + R + 1} slots")
        if pos.no_visible_unit.shape[0] != H:
            raise SequenceLabelError(f"{step.game_side}, decision {step.offset}: belief targets for "
                                     f"{pos.no_visible_unit.shape[0]} hex tokens, the encoding has {H}")
        facts = games[step.game_side.file]
        won = step.game_side.side == facts.winner
        p = min(1.0, value_states_per_game / max(1, facts.n_commands))
        # A game that ended without a winner (a match game cut at its turn
        # cap) has no outcome to train the value on.
        decided = facts.winner != 0
        vw = value_weight if decided and value_selected(seed, step.game_side, step.offset, p) else 0.0
        pw = facts.policy_weight if won and not timeout else 0.0
        ais.append(ai)
        zw.append((1 if won else -1, vw, pw))
    layout = FlatLayout([("belief_target", torch.float32, (B, H_max)), ("belief_mask", torch.float32, (B, H_max))])
    host = torch.zeros(layout.nbytes, dtype=torch.uint8, pin_memory=pin)
    views = layout.numpy_views(host.numpy())
    if B:
        rows, cols = _ragged_rows(np.array([p.no_visible_unit.shape[0] for p in positions], dtype=np.int64))
        views["belief_mask"][rows, cols] = np.concatenate([p.no_visible_unit for p in positions])
        rows, _ = _ragged_rows(np.array([p.hidden_tokens.shape[0] for p in positions], dtype=np.int64))
        views["belief_target"][rows, np.concatenate([p.hidden_tokens for p in positions]).astype(np.int64)] = 1.0
    return StepLabels(ais=ais, zw=zw, belief_host=host, belief_layout=layout)


def belief_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[B]: each position's mean binary cross-entropy over its tokens with
    no visible unit (0 for a position that has none), in float32."""
    bce = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    n = mask.sum(dim=1)
    return (bce * mask).sum(dim=1) / n.clamp(min=1.0)
