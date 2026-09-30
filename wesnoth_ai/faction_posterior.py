"""The enemy's faction as a player can know it
(docs/parity_memory_design_20260929.md "The enemy's faction").

The posterior over the faction vocabulary's rows: the prior is one-hot on
the opponent's faction when its player chose it openly, and uniform over
the era's factions when it chose Random; a faction's likelihood is 1 when
it can field every unit type of the opponent the side has seen, and 0
otherwise. A faction fields its recruits, its leaders and random leaders,
every type they advance to, and the variations of those types (a plague
kill raises a Walking Corpse variation, which the Undead field). A faction
without a definition here ("Custom", "") cannot be checked and keeps its
likelihood of 1. A seen set that no candidate can field leaves the prior in
place and is counted (`posterior_counts`), for the pre-encoding's manifest.
"""
from __future__ import annotations

import functools
import json
from typing import Dict, FrozenSet, Iterable, Sequence

import numpy as np

_COUNTS = {"posteriors": 0, "inconsistent": 0}


@functools.lru_cache(maxsize=None)
def fieldable_types() -> Dict[str, FrozenSet[str]]:
    """Every unit type each faction can field, per faction of the
    `*-default.cfg` files (the default era's six and Dunefolk)."""
    from tools.unit_vocab import advancement_closure, variation_aliases
    from wesnoth_ai.paths import UNIT_STATS_PATH, WESNOTH_SRC_DIR
    from wesnoth_ai.rules.scenario_pool import _parse_faction_cfg
    units = json.loads(UNIT_STATS_PATH.read_text(encoding="utf-8"))["units"]
    out: Dict[str, FrozenSet[str]] = {}
    for cfg in sorted((WESNOTH_SRC_DIR / "data" / "multiplayer" / "factions").glob("*-default.cfg")):
        info = _parse_faction_cfg(cfg)
        if info is None:
            raise ValueError(f"{cfg}: no faction could be read from it")
        base = advancement_closure(set(info.recruit) | set(info.leader_pool) | set(info.random_leader_pool),
                                   units)
        out[info.name] = frozenset(base | set(variation_aliases(base, units)))
    return out


def consistent_factions(candidates: Sequence[str], seen_types: Iterable[str]) -> list:
    """The candidates that can field every seen type."""
    seen = frozenset(seen_types)
    fieldable = fieldable_types()
    return [f for f in candidates if f not in fieldable or seen <= fieldable[f]]


def faction_posterior(faction: str, chose_random: bool, era_factions: Sequence[str],
                      seen_types: Iterable[str], faction_to_id: Dict[str, int]) -> np.ndarray:
    """float32 [MAX_FACTIONS]: the posterior over the faction vocabulary's
    rows for an opponent playing `faction`, which it chose at random
    among `era_factions` or openly, given the unit types of it the side
    has seen. A faction without a row of its own takes the overflow row,
    as `their_faction_id` does."""
    from wesnoth_ai.encoder import MAX_FACTIONS, _lookup_id
    candidates = list(dict.fromkeys(era_factions)) if chose_random else []
    if not candidates:
        candidates = [faction]
    consistent = consistent_factions(candidates, seen_types)
    _COUNTS["posteriors"] += 1
    if not consistent:
        _COUNTS["inconsistent"] += 1
        consistent = candidates
    probs = np.zeros(MAX_FACTIONS, dtype=np.float64)
    for f in consistent:
        probs[_lookup_id(f, faction_to_id, MAX_FACTIONS)] += 1.0 / len(consistent)
    return probs.astype(np.float32)


def posterior_counts() -> Dict[str, int]:
    """The posteriors computed since the process started (or the last
    reset), and how many of them found no candidate faction able to
    field what the side had seen (each kept its prior)."""
    return dict(_COUNTS)


def reset_posterior_counts() -> None:
    for k in _COUNTS:
        _COUNTS[k] = 0
