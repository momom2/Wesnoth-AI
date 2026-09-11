"""One observation per decision (docs/rust_port_plan.md phase 2c).

The encoder, the legality mask builder and the visibility module used
to rebuild the same facts about the side to move from the game state
on every decision: which hexes it sees, which units it sees, which
hexes its units may not cross (the reach context) and where it may
recruit. The 2026-09-11 worker profile put that repeated Python at
three quarters of an eval worker's own time. `observe(state, side)`
computes all of it in one call of the Rust kernel
(`wesnoth_core.observe_side`, rust/wesnoth_core/src/observe.rs) over
flat arrays: Python builds the per-unit arrays (O(units)) and keeps a
per-map geometry cache; everything else is arrays in MAP space, i.e.
hex index = position in `gs.map.hexes` (the pathfinder's order).

The Python originals stay the diff oracle (tests/test_rust_observe.py)
and the path is switched by `WESNOTH_RUST_OBSERVE` (default on when
the wheel carries the kernel; 0 forces the Python path).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from wesnoth_ai.classes import GameState, Unit

_ENABLED = os.environ.get("WESNOTH_RUST_OBSERVE", "1") != "0"
_KERNEL = None
_KERNEL_CHECKED = False


def kernel():
    """The Rust `observe_side`, or None (wheel absent, pre-2c wheel, or
    the switch off). Resolved once per process."""
    global _KERNEL, _KERNEL_CHECKED
    if not _KERNEL_CHECKED:
        _KERNEL_CHECKED = True
        if _ENABLED:
            try:
                import wesnoth_core
                _KERNEL = getattr(wesnoth_core, "observe_side", None)
            except ImportError:
                _KERNEL = None
    return _KERNEL


@dataclass
class MapGeometry:
    """Per-map arrays in map space, cached on the identity of the hex
    container (aliased across forks, replaced by terrain morphs)."""
    hexes: object
    keys: List[Tuple[int, int]]
    pos_index: Dict[Tuple[int, int], int]
    hx: np.ndarray
    hy: np.ndarray
    nbrs: np.ndarray                  # [H*6] i64, hex_neighbors order, -1 off-map
    castle_or_keep: np.ndarray        # [H] u8
    keep: np.ndarray                  # [H] u8


_GEOMETRY: Dict[int, MapGeometry] = {}


def map_geometry(state: GameState) -> MapGeometry:
    hexes = state.map.hexes
    hit = _GEOMETRY.get(id(hexes))
    if hit is not None and hit.hexes is hexes and len(hit.keys) == len(hexes):
        return hit
    from tools.abilities import hex_neighbors
    from wesnoth_ai.classes import TerrainModifiers
    keys = [(h.position.x, h.position.y) for h in hexes]
    pos_index = {p: i for i, p in enumerate(keys)}
    H = len(keys)
    hx = np.fromiter((p[0] for p in keys), dtype=np.int64, count=H)
    hy = np.fromiter((p[1] for p in keys), dtype=np.int64, count=H)
    nbrs = np.full(H * 6, -1, dtype=np.int64)
    castle_or_keep = np.zeros(H, dtype=np.uint8)
    keep = np.zeros(H, dtype=np.uint8)
    for i, (h, (x, y)) in enumerate(zip(hexes, keys)):
        for d, nb in enumerate(hex_neighbors(x, y)):
            j = pos_index.get(nb)
            if j is not None:
                nbrs[i * 6 + d] = j
        mods = h.modifiers
        if TerrainModifiers.KEEP in mods:
            keep[i] = 1
            castle_or_keep[i] = 1
        elif TerrainModifiers.CASTLE in mods:
            castle_or_keep[i] = 1
    geom = MapGeometry(hexes, keys, pos_index, hx, hy, nbrs, castle_or_keep, keep)
    if len(_GEOMETRY) >= 64:
        _GEOMETRY.clear()
    _GEOMETRY[id(hexes)] = geom
    return geom


@dataclass(eq=False)
class Observation:
    """What `side` observes in one state, arrays in map space. The
    unit arrays follow `gs.map.units` order at the time of the call.
    Equality is by content (the encoding differential tests compare
    whole records)."""

    def __eq__(self, other) -> bool:
        if not isinstance(other, Observation):
            return NotImplemented
        return (self.side == other.side and self.fog_on == other.fog_on
                and self.leader_on_keep == other.leader_on_keep
                and self.unit_ids == other.unit_ids
                and self.geometry.keys == other.geometry.keys
                and all(np.array_equal(getattr(self, k), getattr(other, k))
                        for k in ("unit_hex", "disc", "visible", "zoc", "enemy", "ally",
                                  "occupied", "inert", "recruit_row")))
    side: int
    fog_on: bool
    geometry: MapGeometry
    unit_ids: List
    unit_hex: np.ndarray              # [N] i64 map index or -1
    disc: np.ndarray                  # [H] u8
    visible: np.ndarray               # [N] u8
    zoc: np.ndarray                   # [H] u8
    enemy: np.ndarray                 # [H] u8
    ally: np.ndarray                  # [H] u8
    occupied: np.ndarray              # [H] u8
    inert: np.ndarray                 # [H] u8: visible scenery (occupied, never a target)
    recruit_row: np.ndarray           # [H] u8
    leader_on_keep: bool
    _units: Optional[List[Unit]] = field(default=None, repr=False)
    _disc_set: Optional[Set[Tuple[int, int]]] = field(default=None, repr=False)

    def disc_set(self) -> Set[Tuple[int, int]]:
        """The vision disc as the set of (x, y) the Python API returns."""
        if self._disc_set is None:
            keys = self.geometry.keys
            self._disc_set = set(map(keys.__getitem__, np.nonzero(self.disc)[0].tolist()))
        return self._disc_set

    def visible_units(self) -> List[Unit]:
        """`units_visible_to(state, side)`: the god-view list filtered,
        in `gs.map.units` order."""
        units = self._units
        if units is None:
            raise ValueError("observation detached from its units")
        return [u for u, v in zip(units, self.visible.tolist()) if v]

    def visible_ids(self) -> frozenset:
        return frozenset(uid for uid, v in zip(self.unit_ids, self.visible.tolist()) if v)

    def detached(self) -> "Observation":
        """A copy without the Unit references (picklable, for RawEncoded)."""
        return Observation(self.side, self.fog_on, self.geometry, self.unit_ids, self.unit_hex,
                           self.disc, self.visible, self.zoc, self.enemy, self.ally,
                           self.occupied, self.inert, self.recruit_row, self.leader_on_keep)


def _hider_hidden(state: GameState, u: Unit, uncovered) -> bool:
    from wesnoth_ai.visibility import _AMBUSH_ABILITIES, _hide_cover_active
    if not ((u.abilities or set()) & _AMBUSH_ABILITIES):
        return False
    return _hide_cover_active(state, u) and u.id not in uncovered


def observe(state: GameState, side: int) -> Optional[Observation]:
    """The side's observation through the Rust kernel, or None when
    the kernel is unavailable (callers fall back to the Python
    originals)."""
    fn = kernel()
    if fn is None:
        return None
    from tools.replay_dataset import _stats_for
    from wesnoth_ai.visibility import is_scenery_unit, sight_radius_for
    geom = map_geometry(state)
    units = list(state.map.units)
    n = len(units)
    gi = state.global_info
    uncovered = getattr(gi, "_uncovered_units", None) or set()
    fog_on = bool(getattr(gi, "_fog", True))
    ux = np.fromiter((u.position.x for u in units), dtype=np.int64, count=n)
    uy = np.fromiter((u.position.y for u in units), dtype=np.int64, count=n)
    pos_index = geom.pos_index
    uhex = np.fromiter((pos_index.get((u.position.x, u.position.y), -1) for u in units),
                       dtype=np.int64, count=n)
    uside = np.fromiter((u.side for u in units), dtype=np.int64, count=n)
    uradius = np.fromiter((sight_radius_for(u) for u in units), dtype=np.int64, count=n)
    uscenery = np.fromiter((is_scenery_unit(u) for u in units), dtype=np.uint8, count=n)
    upetrified = np.fromiter(("petrified" in (u.statuses or set()) for u in units),
                             dtype=np.uint8, count=n)
    uleader = np.fromiter((bool(u.is_leader) for u in units), dtype=np.uint8, count=n)
    uhider = np.fromiter((_hider_hidden(state, u, uncovered) for u in units),
                         dtype=np.uint8, count=n)
    uzoc = np.fromiter((int(_stats_for(u.name).get("level", 1)) >= 1 for u in units),
                       dtype=np.uint8, count=n)
    H = len(geom.keys)
    recruit_rej = np.zeros(H, dtype=np.uint8)
    for p in (getattr(gi, "_recruit_rejected_hexes", None) or ()):
        j = pos_index.get(p)
        if j is not None:
            recruit_rej[j] = 1
    disc, visible, zoc, enemy, ally, occupied, inert, recruit_row, on_keep = fn(
        geom.hx, geom.hy, geom.nbrs, geom.castle_or_keep, geom.keep, recruit_rej,
        ux, uy, uhex, uside, uradius, uscenery, upetrified, uleader, uhider, uzoc,
        int(side), fog_on)
    return Observation(int(side), fog_on, geom, [u.id for u in units], uhex, disc, visible,
                       zoc, enemy, ally, occupied, inert, recruit_row, bool(on_keep),
                       _units=units)
