"""Positions bound to the Rust core, harvested from games a random legal
player plays (tools/random_positions.py): what the parity observation and
the core's masks are encoded from."""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def core_positions():
    """Some sixty positions from four Ladder games (cached per process),
    one on Thousand Stings Garrison, whose statues are petrified units."""
    from tools.random_positions import harvest
    positions = []
    for seed in (17, 23, 31):
        positions += harvest(seed, decisions=600, every=25, max_turns=24)
    positions += harvest(101, decisions=200, every=25, max_turns=8)
    return tuple(positions)
