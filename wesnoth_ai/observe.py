"""One observation per decision (docs/rust_port_plan.md phase 2c).

The encoder, the legality mask builder and the planner read the same
facts about the side to move: which units it sees, which hexes its units
may not cross (the reach context), where it may recruit and, in the
relevant-set basis, which hexes matter this decision. `observe(state,
side)` asks the Rust core that answers for the state
(`game_core.core_for`) for all of it in one call
(rust/wesnoth_core/src/core_observe.rs), and `observe(state, side,
reach=True)` adds every acting unit's landable row and the relevant hex
set, the union of those rows with the villages, the castles, the visible
units' hexes and the leader's castle network. The arrays are in MAP
space, in the core's hex order (`MapGeometry`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from wesnoth_ai.classes import GameState


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
    village: np.ndarray               # [H] u8: the village terrain type
    full_slot: np.ndarray             # [H] i64: the hex's slot in the row-major
                                      # (y, x) order, the full-board token order


_GEOMETRY: Dict[int, MapGeometry] = {}


def map_geometry(state: GameState) -> MapGeometry:
    hexes = state.map.hexes
    hit = _GEOMETRY.get(id(hexes))
    if hit is not None and hit.hexes is hexes and len(hit.keys) == len(hexes):
        return hit
    from tools.abilities import hex_neighbors
    from wesnoth_ai.classes import Terrain, TerrainModifiers
    keys = [(h.position.x, h.position.y) for h in hexes]
    pos_index = {p: i for i, p in enumerate(keys)}
    H = len(keys)
    hx = np.fromiter((p[0] for p in keys), dtype=np.int64, count=H)
    hy = np.fromiter((p[1] for p in keys), dtype=np.int64, count=H)
    nbrs = np.full(H * 6, -1, dtype=np.int64)
    castle_or_keep = np.zeros(H, dtype=np.uint8)
    keep = np.zeros(H, dtype=np.uint8)
    village = np.zeros(H, dtype=np.uint8)
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
        if Terrain.VILLAGE in h.terrain_types:
            village[i] = 1
    full_slot = np.empty(H, dtype=np.int64)
    full_slot[np.lexsort((hx, hy))] = np.arange(H, dtype=np.int64)
    geom = MapGeometry(hexes, keys, pos_index, hx, hy, nbrs, castle_or_keep, keep,
                       village, full_slot)
    if len(_GEOMETRY) >= 64:
        _GEOMETRY.clear()
    _GEOMETRY[id(hexes)] = geom
    return geom


_ARRAY_FIELDS = ("unit_hex", "seen", "visible", "zoc", "enemy", "ally", "occupied", "inert",
                 "recruit_row", "network", "acting", "unit_can_move", "unit_can_attack",
                 "landable", "relevant", "tok_of_hex")


@dataclass(eq=False)
class Observation:
    """What `side` observes in one state, arrays in map space. The
    unit arrays follow the core's unit order. Equality is by content."""

    def __eq__(self, other) -> bool:
        if not isinstance(other, Observation):
            return NotImplemented
        return (self.side == other.side and self.fog_on == other.fog_on
                and self.leader_on_keep == other.leader_on_keep
                and self.unit_ids == other.unit_ids
                and self.geometry.keys == other.geometry.keys
                and all(_arrays_equal(getattr(self, k), getattr(other, k))
                        for k in _ARRAY_FIELDS))
    side: int
    fog_on: bool
    geometry: MapGeometry
    unit_ids: List
    unit_hex: np.ndarray              # [N] i64 map index or -1
    seen: np.ndarray                  # [H] u8: the hexes the side sees
    visible: np.ndarray               # [N] u8
    zoc: np.ndarray                   # [H] u8
    enemy: np.ndarray                 # [H] u8
    ally: np.ndarray                  # [H] u8
    occupied: np.ndarray              # [H] u8
    inert: np.ndarray                 # [H] u8: visible scenery (occupied, never a target)
    recruit_row: np.ndarray           # [H] u8
    network: np.ndarray               # [H] u8: the leader's castle network
    leader_on_keep: bool
    # With reach=True: the acting units (own, not petrified, moves left
    # or an attack left), their move/attack flags and landable rows
    # (map space, zeros for the others), and the relevant hex set.
    acting: Optional[np.ndarray] = None          # [N] u8
    unit_can_move: Optional[np.ndarray] = None   # [N] u8
    unit_can_attack: Optional[np.ndarray] = None # [N] u8
    landable: Optional[np.ndarray] = None        # [N, H] u8
    relevant: Optional[np.ndarray] = None        # [H] u8
    # Map hex -> token slot of the encoded basis (-1 = no token); set
    # by the encoder, read by the mask builder.
    tok_of_hex: Optional[np.ndarray] = None      # [H] i64
    _seen_set: Optional[Set[Tuple[int, int]]] = field(default=None, repr=False)

    def seen_set(self) -> Set[Tuple[int, int]]:
        """The seen hexes as the set of (x, y) the Python API returns."""
        if self._seen_set is None:
            keys = self.geometry.keys
            self._seen_set = set(map(keys.__getitem__, np.nonzero(self.seen)[0].tolist()))
        return self._seen_set

    def visible_ids(self) -> frozenset:
        return frozenset(uid for uid, v in zip(self.unit_ids, self.visible.tolist()) if v)

    def relevant_set(self) -> Set[Tuple[int, int]]:
        """The relevant hex set as (x, y), `visibility.relevant_hex_positions`."""
        if self.relevant is None:
            raise ValueError("observation without reach")
        keys = self.geometry.keys
        return set(map(keys.__getitem__, np.nonzero(self.relevant)[0].tolist()))


def _arrays_equal(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return np.array_equal(a, b)


def observe(state: GameState, side: int, *, reach: bool = False) -> "Observation":
    """The side's observation from the core that answers for `state`
    (`game_core.core_for`). `reach` adds the landable rows and the
    relevant hex set."""
    from wesnoth_ai.game_core import core_for
    return core_for(state).observe(side, reach=reach)
