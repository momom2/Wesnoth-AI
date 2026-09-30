"""The records of the sequence pre-encoding (tools/preencode_sequences.py),
read by the sequence trainer (tools/sequence_train.py).

They live in an importable module because they are pickled by the
pre-encoder's worker processes: a class defined in a script run by path is
pickled under the worker's name for that script (`__mp_main__`), which no
other process can import.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class SequencePosition:
    raw: object                     # RawEncoded, observation left out
    label: object                   # ActionIndices; action_type "timeout" names no action
    hidden_tokens: np.ndarray       # int64 [K]: hex tokens holding an enemy unit the side cannot see
    no_visible_unit: np.ndarray     # bool [H]: hex tokens with no visible unit
    turn: int = 0                   # the game's turn at the decision


@dataclass
class GameSequence:
    file: str
    winner: int                     # the winning side
    n_commands: int
    sides: Dict[int, List[SequencePosition]] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
