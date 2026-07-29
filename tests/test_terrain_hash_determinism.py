"""Terrain enums must hash by VALUE, because the encoder tie-breaks on
set iteration order.

`GameStateEncoder` picks one terrain id per hex and, when a hex has
several terrain types and none is VILLAGE/CASTLE, falls back to
`next(iter(terrain_types)).value` (`encoder.py`, and `_first_terrain_id`).
That makes the encoding depend on set ITERATION ORDER, which depends on
member hashes.

Why that is load-bearing: the box runs ~76 self-play worker PROCESSES
whose experiences are re-encoded by a separate learner process. If
member hashes were address-derived (plain `Enum`) or string-derived
(`hash(name)`, which PYTHONHASHSEED randomizes), the same hex could
encode differently in different processes -- silently desynchronizing
the acting and training paths. That is the same failure class as the
2026-07-29 village-aliasing bug, where a search-imagined capture rewrote
the real game's encoder input.

MEASURED 2026-07-29: both enums are `IntEnum`, so `hash(member) ==
hash(member.value)` -- an int hash, identical in every process. A
cross-process nondeterminism was SUSPECTED here and this is what
refuted it. These tests exist so that a future change from `IntEnum` to
`Enum` (which looks harmless, and would silently reintroduce
address-derived hashing) fails loudly instead.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wesnoth_ai.classes import Terrain, TerrainModifiers   # noqa: E402


def test_terrain_enums_hash_by_value_not_identity():
    """`IntEnum` hashes as its int. Plain `Enum` would fall back to
    identity hashing, which varies per process."""
    for member in (Terrain.FLAT, Terrain.FOREST, Terrain.VILLAGE):
        assert hash(member) == hash(member.value)
        assert isinstance(member, int)
    for member in (TerrainModifiers.VILLAGE, TerrainModifiers.CASTLE):
        assert hash(member) == hash(member.value)


def test_terrain_set_iteration_order_is_stable_across_processes():
    """The property the encoder actually depends on, checked the only
    way that proves it: in fresh interpreters with DIFFERENT hash seeds.

    A string- or identity-derived hash would make this flap; an int
    hash cannot.
    """
    snippet = (
        "import sys; sys.path.insert(0, r'%s');"
        "from wesnoth_ai.classes import Terrain;"
        "s={Terrain.FOREST,Terrain.HILLS,Terrain.FLAT,"
        "Terrain.SHALLOWWATER};"
        "print(','.join(t.name for t in s))" % ROOT
    )
    orders = []
    for seed in ("1", "17", "424242"):
        out = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True, text=True,
            env={"PYTHONHASHSEED": seed, "PATH": ""},
        )
        assert out.returncode == 0, out.stderr
        orders.append(out.stdout.strip())
    assert len(set(orders)) == 1, (
        f"terrain set iteration order varies across processes: {orders} "
        f"-- the encoder's next(iter(terrain_types)) tie-break is no "
        f"longer deterministic, so workers and the learner can encode "
        f"the same hex differently")
