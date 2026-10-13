"""The records of the sequence pre-encoding (tools/preencode_sequences.py)
and of the memory player's own trajectories (tools/own_trajectories.py,
tools/own_targets.py), read by the sequence trainer (tools/sequence_train.py).

They live in an importable module because they are pickled by the
pre-encoder's worker processes: a class defined in a script run by path is
pickled under the worker's name for that script (`__mp_main__`), which no
other process can import.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class OwnTarget:
    """A teacher's distribution over the legal actions at a position: its
    most probable actions, in the trainer's factored indices (kind as in
    `server_priors`: 0 attack, 1 move, 2 recruit, 3 end_turn), most
    probable first, with their priors in the whole distribution; the
    target is them renormalised (`prob`)."""
    actor: np.ndarray               # int32 [n]
    kind: np.ndarray                # int8 [n]
    target: np.ndarray              # int32 [n], -1 for end_turn
    weapon: np.ndarray              # int8 [n], -1 unless an attack
    prior: np.ndarray               # float64 [n]: their priors in the whole distribution
    end_prior: float                # end_turn's prior in the whole distribution

    @property
    def mass(self) -> float:
        """The share of the whole distribution the kept actions hold."""
        return float(self.prior.sum())

    @property
    def prob(self) -> np.ndarray:
        """float32 [n]: the target, the kept priors renormalised."""
        return (self.prior / self.prior.sum()).astype(np.float32)


@dataclass
class SequencePosition:
    raw: object                     # RawEncoded, observation left out
    label: object                   # ActionIndices; action_type "timeout" names no action
    hidden_tokens: np.ndarray       # int64 [K]: hex tokens holding an enemy unit the side cannot see
    no_visible_unit: np.ndarray     # bool [H]: hex tokens with no visible unit
    turn: int = 0                   # the game's turn at the decision
    # Own positions only (the corpus's leave them None): the legality masks
    # the player decided under (`server_priors.PackedMasks`) and the
    # teacher's distribution there.
    masks: Optional[object] = None
    own_target: Optional[OwnTarget] = None


@dataclass
class GameSequence:
    file: str
    winner: int                     # the winning side; 0 for a game that ended without one
    n_commands: int
    sides: Dict[int, List[SequencePosition]] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class OwnCandidate:
    """A position before one of the memory player's recorded commands, not
    yet known to be a decision: a move followed by an attack from its end
    may be the approach the simulator made for one attack decision
    (`attack_label` is then that attack's label, and the attack's own
    candidate is no decision)."""
    position: SequencePosition
    attack_label: Optional[object] = None


@dataclass
class OwnCandidateSide:
    side: int
    candidates: List[OwnCandidate]
    forwards: Optional[int]         # the player's forwards in the game (the result's forwards_x)
    rejections: int                 # its recruits the engine refused, each one forward more
    offset: float                   # its decode's end_turn offset


@dataclass
class OwnCandidateGame:
    file: str                       # the own record's name
    winner: int                     # 0 for a game that ended without one
    n_commands: int
    sides: Dict[int, OwnCandidateSide] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
