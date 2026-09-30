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

import torch
import torch.nn.functional as F

from wesnoth_ai.imitation_loss import TIMEOUT_LABEL
from wesnoth_ai.sequence_streams import Step, value_selected


class SequenceLabelError(ValueError):
    """A position's label or belief targets do not fit its encoding."""


@dataclass
class StepLabels:
    ais: List[object]                       # ActionIndices per sample
    zw: List[Tuple[int, float, float]]      # (outcome for the mover, value weight, policy weight)
    belief_target: torch.Tensor             # [B, H_max] float32
    belief_mask: torch.Tensor               # [B, H_max] float32: the tokens with no visible unit


@dataclass(frozen=True)
class GameFacts:
    """What the losses read of a game: its winning side, its commands and
    its policy weight (the per-game weight of `obs8`'s recipe)."""
    winner: int
    n_commands: int
    policy_weight: float


def step_labels(positions: Sequence, steps: Sequence[Step], sizes: Sequence[Tuple[int, int, int]],
                games: Dict[str, GameFacts], *, seed: int, value_states_per_game: int,
                value_weight: float) -> StepLabels:
    """The labels of one time step's positions, in batch order. A label
    whose actor is not one of its encoding's slots raises: the pre-encoding
    checked every label against its tokens, so a mismatch is a corrupt
    record, never something to train around."""
    B = len(positions)
    H_max = max((h for _, _, h in sizes), default=0)
    target = torch.zeros(B, H_max, dtype=torch.float32)
    mask = torch.zeros(B, H_max, dtype=torch.float32)
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
        vw = value_weight if value_selected(seed, step.game_side, step.offset, p) else 0.0
        pw = facts.policy_weight if won and not timeout else 0.0
        ais.append(ai)
        zw.append((1 if won else -1, vw, pw))
        mask[b, :H] = torch.from_numpy(pos.no_visible_unit.astype("float32"))
        if pos.hidden_tokens.shape[0]:
            target[b, torch.from_numpy(pos.hidden_tokens)] = 1.0
    return StepLabels(ais=ais, zw=zw, belief_target=target, belief_mask=mask)


def belief_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[B]: each position's mean binary cross-entropy over its tokens with
    no visible unit (0 for a position that has none), in float32."""
    bce = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    n = mask.sum(dim=1)
    return (bce * mask).sum(dim=1) / n.clamp(min=1.0)
