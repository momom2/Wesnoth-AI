"""The enemy's faction as a player can know it
(docs/parity_memory_design_20260929.md "The enemy's faction").

The posterior over the faction vocabulary's rows. The prior is one-hot on
the opponent's faction when its player chose it openly, and uniform over
the factions a Random choice could draw when it chose Random: the era's
factions, less the side's own faction when the lobby's random faction mode
is "No Mirror" (a Random side avoids the other side's faction,
connect_engine.cpp:395-416, 1.18.4). A faction's likelihood is 1 when it
can field every informative unit type of the opponent the side has seen,
and 0 otherwise. A faction fields its recruits, its leaders and random
leaders, every type they advance to, and the variations of those types (a
plague kill raises a Walking Corpse variation, which the Undead field). A
type no faction of the era fields is not informative: a scenario can give
a side such units (Hornshark Island's Young Ogres, Sergeants and Ruffians),
and every faction would give it likelihood 0. A faction without a
definition here ("Custom", "") cannot be checked and keeps its likelihood
of 1. Two outcomes are counted (`posterior_counts`) for the pre-encoding's
manifest: a seen set no candidate can field (the prior is kept), and a
posterior that leaves out the opponent's true faction.
"""
from __future__ import annotations

import functools
import json
from typing import Dict, FrozenSet, Iterable, Sequence, Tuple

import numpy as np

_COUNTS = {"posteriors": 0, "inconsistent": 0, "excludes_truth": 0}

# The lobby's random faction modes under which a Random side cannot draw
# the other player's faction. "No Ally Mirror" avoids allies' factions
# only, and the two players of a 1v1 are enemies.
MIRROR_AVOIDING_MODES = frozenset({"No Mirror"})


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


@functools.lru_cache(maxsize=None)
def _era_fieldable(era_factions: Tuple[str, ...]) -> FrozenSet[str]:
    fieldable = fieldable_types()
    return frozenset().union(*(fieldable[f] for f in era_factions if f in fieldable))


def informative_types(era_factions: Sequence[str], seen_types: Iterable[str]) -> FrozenSet[str]:
    """The seen types some faction of the era can field."""
    return frozenset(seen_types) & _era_fieldable(tuple(era_factions))


def consistent_factions(candidates: Sequence[str], seen_types: Iterable[str]) -> list:
    """The candidates that can field every seen type."""
    seen = frozenset(seen_types)
    fieldable = fieldable_types()
    return [f for f in candidates if f not in fieldable or seen <= fieldable[f]]


def random_draws(chose_random: bool, era_factions: Sequence[str], own_faction: str,
                 random_faction_mode: str) -> list:
    """The factions the opponent's Random choice could have drawn, in era
    order; empty when it chose openly. The engine ignores the avoided
    factions when nothing else is left (flg_manager.cpp:206-209)."""
    if not chose_random:
        return []
    draws = list(dict.fromkeys(era_factions))
    if random_faction_mode in MIRROR_AVOIDING_MODES and own_faction in draws and len(draws) > 1:
        draws.remove(own_faction)
    return draws


def faction_posterior(faction: str, chose_random: bool, era_factions: Sequence[str],
                      seen_types: Iterable[str], faction_to_id: Dict[str, int],
                      own_faction: str = "", random_faction_mode: str = "Independent") -> np.ndarray:
    """float32 [MAX_FACTIONS]: the posterior over the faction vocabulary's
    rows for an opponent playing `faction`, which it chose at random or
    openly, given the unit types of it the side has seen. `own_faction` is
    the side's own faction and `random_faction_mode` the lobby's setting
    (`random_draws`). A faction without a row of its own takes the
    overflow row, as `their_faction_id` does."""
    from wesnoth_ai.encoder import MAX_FACTIONS, _lookup_id
    candidates = random_draws(chose_random, era_factions, own_faction, random_faction_mode) or [faction]
    consistent = consistent_factions(candidates, informative_types(era_factions, seen_types))
    _COUNTS["posteriors"] += 1
    if not consistent:
        _COUNTS["inconsistent"] += 1
        consistent = candidates
    elif faction in candidates and faction not in consistent:
        _COUNTS["excludes_truth"] += 1
    probs = np.zeros(MAX_FACTIONS, dtype=np.float64)
    for f in consistent:
        probs[_lookup_id(f, faction_to_id, MAX_FACTIONS)] += 1.0 / len(consistent)
    return probs.astype(np.float32)


def posterior_counts() -> Dict[str, int]:
    """The posteriors computed since the process started (or the last
    reset); how many found no candidate faction able to field what the
    side had seen (each kept its prior); and how many left out the
    opponent's true faction."""
    return dict(_COUNTS)


def reset_posterior_counts() -> None:
    for k in _COUNTS:
        _COUNTS[k] = 0
