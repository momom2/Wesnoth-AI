"""Positions bound to the Rust core, harvested from games a random legal
player plays (tools/random_positions.py): what the parity observation and
the core's masks are encoded from."""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def core_positions():
    """Some fifty positions from three Ladder games (cached per process)."""
    from tools.random_positions import harvest
    positions = []
    for seed in (17, 23, 31):
        positions += harvest(seed, decisions=600, every=25, max_turns=24)
    return tuple(positions)
