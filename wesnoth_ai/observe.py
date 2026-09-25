"""One observation per decision (docs/rust_port_plan.md phase 2c).

The encoder, the legality mask builder and the visibility module used
to rebuild the same facts about the side to move from the game state
on every decision: which units it sees, which hexes its units may not
cross (the reach context), where it may recruit and, in the
relevant-set basis, which hexes matter this decision. `observe(state,
side)` computes all of it in one call of the Rust kernel
(`wesnoth_core.observe_side`, rust/wesnoth_core/src/observe.rs) over
flat arrays and the hexes the side sees (`visibility.visible_hexes_for`,
the side's fog), and `observe(state, side, reach=True)`
adds every acting unit's landable row (`wesnoth_core.reach_rows`) and
the relevant hex set, the union of those rows with the villages, the
castles, the visible units' hexes and the leader's castle network
(`visibility.relevant_hex_positions`). Python builds the per-unit
arrays (O(units)) and keeps a per-map geometry cache; everything else
is arrays in MAP space, i.e. hex index = position in `gs.map.hexes`
(the pathfinder's order).

The Python originals stay the diff oracle (tests/test_rust_observe.py)
and the path is switched by `WESNOTH_RUST_OBSERVE` (default on when
the wheel carries the kernels; 0 forces the Python path).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from wesnoth_ai.classes import GameState, Unit

_ENABLED = os.environ.get("WESNOTH_RUST_OBSERVE", "1") != "0"
_KERNELS: Optional[Dict[str, object]] = None


def _kernels() -> Dict[str, object]:
    """The Rust kernels by name (empty when the wheel is absent, older
    than this module, or the switch is off). Resolved once per process."""
    global _KERNELS
    if _KERNELS is None:
        _KERNELS = {}
        if _ENABLED:
            try:
                import wesnoth_core
            except ImportError:
                wesnoth_core = None
            # Phase 12: observe_side takes the side's seen hexes.
            if wesnoth_core is not None and getattr(wesnoth_core, "__phase__", 0) >= 12:
                for name in ("observe_side", "reach_rows", "rows_from_reach"):
                    _KERNELS[name] = getattr(wesnoth_core, name)
    return _KERNELS


def kernel():
    """The Rust `observe_side`, or None."""
    return _kernels().get("observe_side")


def kernel_rows_from_reach():
    """The Rust `rows_from_reach`, or None."""
    return _kernels().get("rows_from_reach")


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
    _units: Optional[List[Unit]] = field(default=None, repr=False)
    _seen_set: Optional[Set[Tuple[int, int]]] = field(default=None, repr=False)

    def seen_set(self) -> Set[Tuple[int, int]]:
        """The seen hexes as the set of (x, y) the Python API returns."""
        if self._seen_set is None:
            keys = self.geometry.keys
            self._seen_set = set(map(keys.__getitem__, np.nonzero(self.seen)[0].tolist()))
        return self._seen_set

    def visible_units(self) -> List[Unit]:
        """`units_visible_to(state, side)`: the god-view list filtered,
        in `gs.map.units` order."""
        units = self._units
        if units is None:
            raise ValueError("observation detached from its units")
        return [u for u, v in zip(units, self.visible.tolist()) if v]

    def visible_ids(self) -> frozenset:
        return frozenset(uid for uid, v in zip(self.unit_ids, self.visible.tolist()) if v)

    def relevant_set(self) -> Set[Tuple[int, int]]:
        """The relevant hex set as (x, y), `visibility.relevant_hex_positions`."""
        if self.relevant is None:
            raise ValueError("observation without reach")
        keys = self.geometry.keys
        return set(map(keys.__getitem__, np.nonzero(self.relevant)[0].tolist()))

    def detached(self) -> "Observation":
        """A copy without the Unit references (picklable, for RawEncoded)."""
        return Observation(self.side, self.fog_on, self.geometry, self.unit_ids, self.unit_hex,
                           self.seen, self.visible, self.zoc, self.enemy, self.ally,
                           self.occupied, self.inert, self.recruit_row, self.network,
                           self.leader_on_keep, self.acting, self.unit_can_move,
                           self.unit_can_attack, self.landable, self.relevant, self.tok_of_hex)


def _arrays_equal(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return np.array_equal(a, b)


# The seen-hex array of a side's fog, by the identity of the frozenset
# `visibility.visible_hexes_for` returns (a tracked fog is one object
# until a command replaces it) and of the geometry.
_SEEN_ARRAYS: Dict[int, tuple] = {}


def seen_array(state: GameState, side: int, geom: MapGeometry) -> np.ndarray:
    """[H] u8 in map space: the hexes `side` sees."""
    from wesnoth_ai.visibility import visible_hexes_for
    seen = visible_hexes_for(state, side)
    hit = _SEEN_ARRAYS.get(id(seen))
    if hit is not None and hit[0] is seen and hit[1] is geom:
        return hit[2]
    arr = np.zeros(len(geom.keys), dtype=np.uint8)
    idx = [j for j in map(geom.pos_index.get, seen) if j is not None]
    arr[idx] = 1
    if len(_SEEN_ARRAYS) >= 256:
        _SEEN_ARRAYS.clear()
    _SEEN_ARRAYS[id(seen)] = (seen, geom, arr)
    return arr


def _hider_hidden(state: GameState, u: Unit, uncovered) -> bool:
    from wesnoth_ai.visibility import _AMBUSH_ABILITIES, _hide_cover_active
    if not ((u.abilities or set()) & _AMBUSH_ABILITIES):
        return False
    return _hide_cover_active(state, u) and u.id not in uncovered


def observe(state: GameState, side: int, *, reach: bool = False) -> Optional[Observation]:
    """The side's observation through the Rust kernels, or None when
    they are unavailable (callers fall back to the Python originals).
    `reach` adds the landable rows and the relevant hex set."""
    fns = _kernels()
    fn = fns.get("observe_side")
    if fn is None:
        return None
    from tools.pathfind_sim import emits_zoc
    from wesnoth_ai.visibility import is_scenery_unit
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
    uscenery = np.fromiter((is_scenery_unit(u) for u in units), dtype=np.uint8, count=n)
    upetrified = np.fromiter(("petrified" in (u.statuses or set()) for u in units),
                             dtype=np.uint8, count=n)
    uleader = np.fromiter((bool(u.is_leader) for u in units), dtype=np.uint8, count=n)
    uhider = np.fromiter((_hider_hidden(state, u, uncovered) for u in units),
                         dtype=np.uint8, count=n)
    # The one zone-of-control predicate. observe.rs still skips scenery
    # before reading this flag, which the engine does not do; the kernel
    # change is owed with the next Rust phase.
    uzoc = np.fromiter((emits_zoc(u) for u in units), dtype=np.uint8, count=n)
    H = len(geom.keys)
    recruit_rej = np.zeros(H, dtype=np.uint8)
    for p in (getattr(gi, "_recruit_rejected_hexes", None) or ()):
        j = pos_index.get(p)
        if j is not None:
            recruit_rej[j] = 1
    seen, visible, zoc, enemy, ally, occupied, inert, recruit_row, network, on_keep = fn(
        geom.nbrs, geom.castle_or_keep, geom.keep, recruit_rej, seen_array(state, side, geom),
        ux, uy, uhex, uside, uscenery, upetrified, uleader, uhider, uzoc,
        int(side), fog_on)
    obs = Observation(int(side), fog_on, geom, [u.id for u in units], uhex, seen, visible,
                      zoc, enemy, ally, occupied, inert, recruit_row, network, bool(on_keep),
                      _units=units)
    if reach:
        _add_reach(state, side, units, upetrified, uleader, obs, fns["reach_rows"])
    return obs


# The per-type terrain stacks the reach kernel consumes, cached by the
# identity of the pathfinder's cost lists (stable per map, unit type,
# slowed status and defense table).
_TYPE_ARRAYS: Dict[int, tuple] = {}
_STACKS: Dict[tuple, tuple] = {}


def _type_arrays(mcost, dsub) -> tuple:
    hit = _TYPE_ARRAYS.get(id(mcost))
    if hit is None or hit[0] is not mcost:
        hit = (mcost, np.asarray(mcost, dtype=np.int64), np.asarray(dsub, dtype=np.int64))
        if len(_TYPE_ARRAYS) > 1024:
            _TYPE_ARRAYS.clear()
        _TYPE_ARRAYS[id(mcost)] = hit
    return hit


def _stacks(rows: List[tuple]) -> Tuple[np.ndarray, np.ndarray]:
    """[T*H] stacks of the distinct type arrays, cached by their ids."""
    key = tuple(id(r[0]) for r in rows)
    hit = _STACKS.get(key)
    if hit is None or any(a is not b for a, (b, _, _) in zip(hit[0], rows)):
        tm = np.concatenate([r[1] for r in rows]) if rows else np.zeros(0, dtype=np.int64)
        td = np.concatenate([r[2] for r in rows]) if rows else np.zeros(0, dtype=np.int64)
        hit = (tuple(r[0] for r in rows), tm, td)
        if len(_STACKS) > 1024:
            _STACKS.clear()
        _STACKS[key] = hit
    return hit[1], hit[2]


def _add_reach(state: GameState, side: int, units: List[Unit], upetrified: np.ndarray,
               uleader: np.ndarray, obs: Observation, reach_fn) -> None:
    """The acting units' landable rows and the relevant hex set."""
    from tools.pathfind_sim import _terrain_arrays_for
    geom = obs.geometry
    n, H = len(units), len(geom.keys)
    acting = np.zeros(n, dtype=np.uint8)
    can_move = np.zeros(n, dtype=np.uint8)
    can_attack = np.zeros(n, dtype=np.uint8)
    unit_hexidx = np.full(n, -1, dtype=np.int64)
    unit_type = np.zeros(n, dtype=np.int64)
    unit_budget = np.zeros(n, dtype=np.int64)
    unit_skirm = np.zeros(n, dtype=np.uint8)
    rows: List[tuple] = []
    row_of: Dict[int, int] = {}
    for i, u in enumerate(units):
        if u.side != side or upetrified[i] or obs.unit_hex[i] < 0:
            continue
        moves = u.current_moves > 0
        if not (moves or not u.has_attacked):
            continue
        acting[i] = 1
        can_move[i] = 1 if moves else 0
        can_attack[i] = 0 if u.has_attacked else 1
        unit_hexidx[i] = obs.unit_hex[i]
        unit_budget[i] = int(u.current_moves)
        unit_skirm[i] = 1 if "skirmisher" in (u.abilities or set()) else 0
        if moves:
            _pos_to_idx, positions, _nbrs, mcost, dsub = _terrain_arrays_for(u, state)
            if len(positions) != H:
                raise ValueError("pathfinder map and geometry disagree")
            arrays = _type_arrays(mcost, dsub)
            row = row_of.get(id(mcost))
            if row is None:
                row = row_of[id(mcost)] = len(rows)
                rows.append(arrays)
            unit_type[i] = row
    tm, td = _stacks(rows)
    landable = reach_fn(geom.nbrs, tm, td, unit_hexidx, unit_type, unit_budget, unit_skirm,
                        can_move, obs.zoc, obs.enemy, obs.ally, obs.occupied)
    landable = np.asarray(landable, dtype=np.uint8).reshape(n, H)
    relevant = (geom.village | geom.castle_or_keep | obs.network | obs.occupied) != 0
    leader = next((i for i in range(n) if units[i].side == side and uleader[i]), None)
    if leader is not None and obs.unit_hex[leader] >= 0:
        relevant[obs.unit_hex[leader]] = True
    movers = can_move != 0
    if movers.any():
        relevant |= landable[movers].any(axis=0)
    obs.acting, obs.unit_can_move, obs.unit_can_attack = acting, can_move, can_attack
    obs.landable = landable
    obs.relevant = relevant.astype(np.uint8)
