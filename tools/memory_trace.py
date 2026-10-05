"""Every decision of a memory player's game, for the learner's memory
(docs/memory_everywhere_20261005.md, hole 6; the learner is
wesnoth_ai/memory_step.py).

A side's memory at a decision is what its previous decision wrote, so the
learner runs each game-side's decisions in order, the ones without a search
target included (playout-cap fast moves, the turn search's fast turns): a
gap would break the chain. The actor encodes each position here, where the
state is a view bound to its core (the parity observation is built by the
core, and the binding does not cross to the learner's process), and keeps
the belief head's targets, read from the observation's god view, which is
then dropped: the truth is a training target only.
"""
from __future__ import annotations

import dataclasses
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from wesnoth_ai.trainer import MCTSExperience


@dataclass
class TraceStep:
    side: int
    raw: object                         # RawEncoded, its observation dropped
    masks: object                       # PackedMasks or None
    hidden_tokens: Optional[np.ndarray]
    no_visible_unit: Optional[np.ndarray]
    recorded: bool                      # the decision left a training target


class MemoryTrace:
    """Per game label, a memory player's decisions in order."""

    def __init__(self, k: int, belief: bool):
        """`k`: the slots the player uses; `belief`: the encoder reads the
        parity observation, whose god view gives the belief targets."""
        self.k = int(k)
        self._belief = bool(belief)
        self._games: Dict[str, List[TraceStep]] = {}
        self._lock = threading.Lock()

    def note(self, game_label: str, encoder, game_state, masks, recorded: bool) -> None:
        """One decision at `game_state` (a view bound to the game's core)."""
        from wesnoth_ai.belief_targets import belief_targets
        side = int(game_state.global_info.current_side)
        raw = encoder.raw_of(game_state)
        hidden = free = None
        if self._belief:
            targets = belief_targets(raw, side)
            hidden, free = targets.hidden_tokens, targets.no_visible_unit
        step = TraceStep(side=side, raw=dataclasses.replace(raw, observation=None), masks=masks,
                         hidden_tokens=hidden, no_visible_unit=free, recorded=bool(recorded))
        with self._lock:
            self._games.setdefault(game_label, []).append(step)

    def drop_last(self, game_label: str) -> None:
        """The last decision bounced and is decided again: the engine
        records nothing for it, and the side's memory went back."""
        with self._lock:
            steps = self._games.get(game_label)
            if steps:
                steps.pop()

    def pop(self, game_label: str) -> List[TraceStep]:
        with self._lock:
            return self._games.pop(game_label, [])


def outcome_for(side: int, winner: int) -> float:
    return 0.0 if winner == 0 else (1.0 if winner == side else -1.0)


def sequence_experiences(steps: List[TraceStep], recorded: List[MCTSExperience], game_label: str,
                         k: int, winner: int) -> List[MCTSExperience]:
    """The game's experiences in decision order: a recorded decision takes
    the next of `recorded` (the game's experiences in the order their
    targets were recorded), any other becomes a carry position; each gets
    its place in its game-side, its encoding and its belief targets."""
    out: List[MCTSExperience] = []
    pending = iter(recorded)
    next_step: Dict[int, int] = {}
    for st in steps:
        if st.recorded:
            e = next(pending, None)
            if e is None:
                raise ValueError(f"{game_label}: the trace records more targets than the game holds")
            if int(e.game_state.global_info.current_side) != st.side:
                raise ValueError(f"{game_label}: a recorded target of side "
                                 f"{e.game_state.global_info.current_side} at side {st.side}'s decision")
        else:
            e = MCTSExperience(game_state=None, visit_counts=[], z=outcome_for(st.side, winner),
                               value_weight=0.0, policy_weight=0.0, game_weight=0.0,
                               label_kind="carry", game_id=str(game_label), masks=st.masks)
        e.raw, e.side, e.memory_k = st.raw, st.side, int(k)
        e.side_step = next_step.get(st.side, 0)
        next_step[st.side] = e.side_step + 1
        e.hidden_tokens, e.no_visible_unit = st.hidden_tokens, st.no_visible_unit
        out.append(e)
    if next(pending, None) is not None:
        raise ValueError(f"{game_label}: the game holds targets its trace does not record")
    return out


def is_carry(e) -> bool:
    """A decision kept for its side's memory chain only (no search target)."""
    return getattr(e, "label_kind", "game") == "carry"
