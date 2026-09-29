"""The belief head's targets (docs/parity_memory_design_20260929.md "The
belief head"): which hex tokens hold an enemy unit the side to move cannot
see, read from a pair's RawEncoded, whose observation keeps every unit's
hex and whether the side sees it. The truth is a training target only; it
never enters an input.

Every unit the side does not see is an enemy's: its own units and the
scenery are always visible (`visibility.units_visible_to`). Hexes and
tokens are both in the observation's map space (`tok_of_hex`), so no view's
hex order is involved.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BeliefTargets:
    hidden_tokens: np.ndarray      # int64 [K], ascending: hex tokens holding a hidden enemy unit
    n_untokened: int               # hidden enemy units whose hex has no token
    no_visible_unit: np.ndarray    # bool [H]: hex tokens with no visible unit (the loss's domain)


def belief_targets(raw, side: int) -> BeliefTargets:
    """The belief targets of `raw`, encoded for `side` (the mover)."""
    obs = raw.observation
    if obs is None or obs.tok_of_hex is None:
        raise ValueError("the RawEncoded carries no observation with its token index "
                         "(encode it through a core or the Rust kernels)")
    if int(obs.side) != int(side):
        raise ValueError(f"the observation is side {obs.side}'s, not side {side}'s")
    tok_of_hex = np.asarray(obs.tok_of_hex, dtype=np.int64)
    unit_hex = np.asarray(obs.unit_hex, dtype=np.int64)
    hidden = (np.asarray(obs.visible) == 0) & (unit_hex >= 0)
    tokens = tok_of_hex[unit_hex[hidden]]
    n_tokens = int(raw.hex_xs.shape[0])
    no_visible_unit = np.ones(n_tokens, dtype=bool)
    occupied = np.flatnonzero((np.asarray(obs.occupied) != 0) & (tok_of_hex >= 0))
    no_visible_unit[tok_of_hex[occupied]] = False
    return BeliefTargets(hidden_tokens=np.sort(tokens[tokens >= 0]).astype(np.int64),
                         n_untokened=int((tokens < 0).sum()),
                         no_visible_unit=no_visible_unit)
