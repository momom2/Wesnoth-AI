"""Per-side visibility for the fog-of-war contract.

The Wesnoth simulator (`tools/wesnoth_sim.py`) maintains god-view
internally -- it must, because combat resolution, recall lists,
victory checks, and the action-applier all need ground truth.
But when the POLICY observes state via the encoder + sampler, the
view has to be filtered so the model only sees what a real
Wesnoth client would render for that side.

This module is the single source of truth for that filter.
`encoder.py`, `action_sampler.py`, `observe.py` and `rewards.py` all
import from here, so the contract is consistent everywhere; the Rust
core mirrors it (rust/wesnoth_core/src/core_fog.rs, observe.rs).

Vision and fog (docs/wesnoth_rules.md "Vision and fog")
=======================================================

A unit sees every hex it could reach in one turn spending its maximum
movement at its movement costs (doubled when it is slowed), other
units ignored, plus every hex next to one of those (`unit_vision`).

A side sees its fog: the hexes it has cleared, kept per side on
`global_info._fog_cleared` ({side: frozenset of (x, y)}, replaced,
never mutated, so search forks share it safely). The command applier
(`tools.replay_dataset._apply_command`) keeps it as the engine does:

  refog(state, side)          the side's units' vision from where they
                              stand: at the side's turn start and end,
                              and for the defender after a fight that
                              killed, slowed or petrified it;
  clear_fog(state, u, hexes)  adds u's vision from each hex: every hex
                              a mover enters, a recruit's hex, an
                              advanced unit's hex;
  track_side(state, side)     starts tracking an untracked side before
                              a command changes its units.

A side with no tracked fog sees its units' vision from where they
stand (`side_vision`), which is what the engine clears for every side
when the game starts. Fog-off games track nothing.

Public API
==========

  visible_hexes_for(state, side) -> frozenset of (x, y)
      The hexes the side sees.
  visible_fraction_for(state, side) -> float
      Their share of the map (the fog-reveal shaping reward).
  units_visible_to(state, side) -> List[Unit]
      The god-view unit list filtered:
        * own-side units and scenery: always
        * enemy units hiding under an active hide-cover ability and
          neither uncovered nor discovered by adjacency: never
        * other enemy units: when fog is off or their hex is seen.

Shroud is not modelled: every hex's terrain is known, as in the
multiplayer ladder games we train on (fog on, shroud off).

Dependencies: classes (Unit, GameState), terrain_resolver (hides_cover),
  pathfind_sim (the movement cost arrays), replay_dataset
  (illuminated_lawful_bonus_at).
Dependents: rewards (visible_fraction_for), encoder and observe
  (visible_hexes_for, units_visible_to), action_sampler
  (units_visible_to), replay_dataset (the fog hooks).
"""

from __future__ import annotations

import logging
from heapq import heappop, heappush
from typing import AbstractSet, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from wesnoth_ai.classes import GameState, Unit

log = logging.getLogger("visibility")

Hex = Tuple[int, int]

# Cover abilities -- a unit with one of these CAN hide on the
# matching terrain / ToD (`_hide_cover_active` decides when). The
# Rust core keeps the same list as `HIDE_ABILITIES` in core_move.rs
# and core_observe.rs; tests/test_rust_constants.py compares them.
_AMBUSH_ABILITIES = frozenset({
    "ambush", "nightstalk", "concealment", "submerge",
})

# Unit types whose own cfg declares `vision=` or `[vision_costs]`
# (wesnoth_src/data/core/units, 1.18.7; pinned by
# tests/test_vision.py). Neither is modelled: they see with their
# movement. None is in the default era, the pool or the corpus.
OWN_VISION_TYPES = frozenset({
    "Dune Falconer", "Dune Sky Hunter", "Dragonfly", "Grand Dragonfly",
})
_WARNED_VISION_TYPES: Set[str] = set()

FOG_CLEARED = "_fog_cleared"


def vision_points(unit: Unit) -> int:
    """The unit's vision points: its maximum movement
    (`unit::vision()`, src/units/unit.hpp:1415-1418 at 1.18.4, for a
    type without `vision=`)."""
    if unit.name in OWN_VISION_TYPES and unit.name not in _WARNED_VISION_TYPES:
        _WARNED_VISION_TYPES.add(unit.name)
        log.warning("%s declares its own vision or vision costs; the simulator "
                    "sees with its movement (docs/wesnoth_rules.md, Vision and fog)", unit.name)
    return max(int(unit.max_moves), 0)


# One unit's vision area per (movement cost array, vision points,
# start hex). The cost arrays come from `pathfind_sim._terrain_arrays_for`,
# one per map, unit type, slowed status and defense table; the entry
# keeps the array itself, which pins its id and makes the identity
# check exact.
_VISION_CACHE: Dict[tuple, tuple] = {}
_VISION_CACHE_MAX = 8192


def _vision_area(nbrs, mcost, dsub, start: int, budget: int) -> Iterable[int]:
    """Hex indices reachable from `start` within `budget` (vertex costs),
    plus their neighbours: through the Rust reach kernel with an empty
    context when the wheel serves it (its reached set is the hexes whose
    cheapest route costs at most `budget`), else by the search below."""
    from tools import pathfind_sim
    kernel = pathfind_sim.reach_kernel()
    if kernel is not None:
        import numpy as np
        flat, mcost_a, dsub_a = pathfind_sim.rust_arrays(nbrs, mcost, dsub)
        empty = _empty_context(len(mcost))
        mp, _cost, _prev = kernel(flat, mcost_a, dsub_a, empty, empty, empty, start, budget, False)
        reached = np.nonzero(mp >= 0)[0]
        ring = flat.reshape(-1, 6)[reached].ravel()
        return np.union1d(reached, ring[ring >= 0]).tolist()
    return _vision_search(nbrs, mcost, start, budget)


_EMPTY_CONTEXT: Dict[int, object] = {}


def _empty_context(h: int):
    """A zero [H] u8 array: no zone of control, enemy or ally anywhere."""
    arr = _EMPTY_CONTEXT.get(h)
    if arr is None:
        import numpy as np
        arr = _EMPTY_CONTEXT[h] = np.zeros(h, dtype=np.uint8)
    return arr


def _vision_search(nbrs, mcost, start: int, budget: int) -> Set[int]:
    """`_vision_area` in Python: Dijkstra over vertex costs."""
    spent = {start: 0}
    frontier = [(0, start)]
    while frontier:
        cost, i = heappop(frontier)
        if cost > spent[i]:
            continue
        for j in nbrs[i]:
            if j < 0:
                continue
            nxt = cost + mcost[j]
            if nxt <= budget and nxt < spent.get(j, budget + 1):
                spent[j] = nxt
                heappush(frontier, (nxt, j))
    seen = set(spent)
    for i in spent:
        seen.update(j for j in nbrs[i] if j >= 0)
    return seen


def unit_vision(state: GameState, unit: Unit, at: Optional[Hex] = None) -> FrozenSet[Hex]:
    """Hexes `unit` sees from `at` (default: where it stands): every hex
    it could reach this turn spending its vision points at its vision
    costs, other units and zones of control ignored, plus every hex
    next to one of those (`pathfind::vision_path`,
    src/pathfind/pathfind.cpp:576-588, and the edges `find_routes`
    collects, :349-352 and :392-398, at 1.18.4). Vision costs are the
    movement costs, since no default-era type declares
    `[vision_costs]`, doubled when the unit is slowed
    (src/movetype.hpp:69-72, through `_move_cost_at_hex`)."""
    from tools.pathfind_sim import _terrain_arrays_for
    pos_to_idx, positions, nbrs, mcost, dsub = _terrain_arrays_for(unit, state)
    start = pos_to_idx.get(at if at is not None else (unit.position.x, unit.position.y))
    if start is None:
        return frozenset()
    budget = vision_points(unit)
    key = (id(mcost), budget, start)
    hit = _VISION_CACHE.get(key)
    if hit is not None and hit[0] is mcost:
        return hit[1]
    seen = frozenset(map(positions.__getitem__, _vision_area(nbrs, mcost, dsub, start, budget)))
    if len(_VISION_CACHE) >= _VISION_CACHE_MAX:
        _VISION_CACHE.clear()
    _VISION_CACHE[key] = (mcost, seen)
    return seen


def side_vision(state: GameState, side: int) -> FrozenSet[Hex]:
    """The union of the side's units' vision from where they stand."""
    areas = [unit_vision(state, u) for u in state.map.units if u.side == side]
    return frozenset().union(*areas)


def _fog_on(state: GameState) -> bool:
    return bool(getattr(state.global_info, "_fog", True))


def _set_cleared(state: GameState, side: int, hexes: FrozenSet[Hex]) -> None:
    """A new dict every time: search forks share the old one."""
    cleared = dict(getattr(state.global_info, FOG_CLEARED, None) or {})
    cleared[side] = hexes
    setattr(state.global_info, FOG_CLEARED, cleared)


def visible_hexes_for(state: GameState, side: int) -> AbstractSet[Hex]:
    """The hexes `side` sees: its cleared hexes when tracked, else its
    units' vision from where they stand. A frozenset."""
    tracked = (getattr(state.global_info, FOG_CLEARED, None) or {}).get(side)
    if tracked is not None:
        return tracked
    return side_vision(state, side)


def track_side(state: GameState, side: int) -> None:
    """Start tracking `side`'s fog from its units' vision, before a
    command moves, replaces or removes its units."""
    if not _fog_on(state):
        return
    if side in (getattr(state.global_info, FOG_CLEARED, None) or {}):
        return
    _set_cleared(state, side, side_vision(state, side))


def refog(state: GameState, side: int) -> None:
    """Recalculate `side`'s fog from where its units stand
    (`actions::recalculate_fog`, src/actions/vision.cpp:702-736)."""
    if _fog_on(state):
        _set_cleared(state, side, side_vision(state, side))


def clear_fog(state: GameState, unit: Unit, hexes: Iterable[Hex]) -> None:
    """Add `unit`'s vision from each of `hexes` to its side's fog
    (`shroud_clearer::clear_unit`, src/actions/vision.cpp:332-371)."""
    hexes = list(hexes)
    if not _fog_on(state) or not hexes:
        return
    base = visible_hexes_for(state, unit.side)
    _set_cleared(state, unit.side, frozenset(base).union(
        *(unit_vision(state, unit, at=h) for h in hexes)))


def visible_fraction_for(state: GameState, side: int) -> float:
    """Fraction of the map currently visible to `side`. Range
    [0, 1]. Returns 0 on an empty map.

    Consumed by the continuous-payment fog-reveal shaping
    reward (`rewards.WeightedReward.fog_reveal_weight`). The
    per-step contribution is `(1 - gamma) * weight * fraction`;
    over a fully-explored, sustained-visibility game the
    discounted sum approaches `weight` (see WeightedReward
    docstring).
    """
    hexes = state.map.hexes
    if not hexes:
        return 0.0
    # Fogless game: everything is effectively revealed, so the
    # fog-reveal shaping reward saturates rather than paying for
    # vision coverage that carries no information value.
    if not _fog_on(state):
        return 1.0
    return len(visible_hexes_for(state, side)) / len(hexes)


def _hide_cover_active(state: GameState, unit: Unit) -> bool:
    """True iff `unit` has a hide ability AND its current hex's
    terrain (or ToD, for nightstalk) satisfies the ability's
    cover condition.

    The covers are the engine's own `[hides] [filter_location]` filters
    (`wesnoth_src/data/core/macros/abilities.cfg`:280-382), which match
    the hex's terrain CODE, not what it defends like:

      - ambush:      terrain=*^F*,*^Qhhf,*^Qhuf
      - concealment: terrain=*^V*
      - submerge:    terrain=Wo*^*
      - nightstalk:  time_of_day=chaotic, i.e. lawful_bonus < 0

    Until 2026-09-13 the terrain covers were decided from the DEFENSE
    keys of a hand-rolled overlay table, which silently gave no cover
    on 30.4% of the Ladder pool's forest-overlay hexes and 27.7% of
    its village-overlay hexes, and gave cover on farmland, which is
    not a village. `terrain_resolver.hides_cover` matches the engine's
    globs instead; docs/wesnoth_rules.md has the census.

    SINGLE source of truth since 2026-07-18 (the sim's duplicate
    method was removed; walk_move_path and units_visible_to both
    consume this one).
    """
    abilities = unit.abilities or set()
    if not (abilities & _AMBUSH_ABILITIES):
        return False
    # Lazy import: replay_dataset is heavy (it pulls combat.py,
    # unit_stats.json, etc.). Importing at module load would slow
    # cold tests and cluster start. The lookup is per-unit-with-
    # hide-ability, which is a rare hot path.
    from tools.replay_dataset import illuminated_lawful_bonus_at
    from wesnoth_ai.rules.terrain_resolver import hides_cover
    codes = getattr(state.global_info, "_terrain_codes", None) or {}
    code = codes.get((unit.position.x, unit.position.y), "")
    for ability in ("ambush", "concealment", "submerge"):
        if ability in abilities and hides_cover(code, ability):
            return True
    if "nightstalk" in abilities:
        # `time_of_day=chaotic` on the ILLUMINATED time of day: an
        # [illuminates] unit on or next to the hex lifts the cover
        # (abilities.cpp:447-450, filter.cpp:268-273).
        if illuminated_lawful_bonus_at(state, unit, state.global_info.turn_number) < 0:
            return True
    return False


def leader_castle_network(state: GameState,
                          leader: Unit) -> Tuple[bool, Set[Tuple[int, int]]]:
    """(leader_on_keep, castle-network hex set) for recruit legality.

    The network is the BFS closure of CASTLE/KEEP-modifier hexes
    connected to the keep the leader stands on (Wesnoth:
    `can_recruit_on` walks castle tiles from the recruiting keep).
    Returns (False, empty set) when the leader is off-keep.

    SHARED CONTRACT: the legality mask (action_sampler) and the
    sim's recruit validation (wesnoth_sim._action_to_command) must
    both consume THIS function -- a mirror was how the sim ended up
    skipping connectivity entirely (audit 2026-07-17).
    """
    from collections import deque
    from wesnoth_ai.classes import TerrainModifiers

    mods_by_pos = {
        (h.position.x, h.position.y): h.modifiers
        for h in state.map.hexes
    }
    start = (leader.position.x, leader.position.y)
    if TerrainModifiers.KEEP not in (mods_by_pos.get(start) or set()):
        return False, set()
    from tools.abilities import hex_neighbors
    visited = {start}
    q = deque([start])
    network: Set[Tuple[int, int]] = set()
    while q:
        x, y = q.popleft()
        for nx, ny in hex_neighbors(x, y):
            if (nx, ny) in visited:
                continue
            nmods = mods_by_pos.get((nx, ny))
            if nmods is None:
                continue
            if (TerrainModifiers.CASTLE in nmods
                    or TerrainModifiers.KEEP in nmods):
                visited.add((nx, ny))
                q.append((nx, ny))
                network.add((nx, ny))
    return True, network


def _discovered_by_adjacency(state: GameState, hider: Unit,
                             observer_side: int) -> bool:
    """Wesnoth's `would_be_discovered` (display_context.cpp:29-49):
    a hidden unit is seen while ANY enemy of the hider stands on an
    adjacent tile (not incapacitated). `unit::invisible` is
    viewer-INDEPENDENT, so a discovery by a third party (e.g. an
    armed side-3 neutral adjacent to a side-2 hider) reveals the
    hider to every side — including `observer_side` (adversarial
    review 2026-07-18; previously only the observer's own units
    counted). The engine additionally requires the discoverer to be
    itself visible to the hider's team; we accept that reduction
    (documented sight-model simplification)."""
    from tools.abilities import hex_neighbors
    adj = set(hex_neighbors(hider.position.x, hider.position.y))
    for u in state.map.units:
        if u.side == hider.side:
            continue
        if is_scenery_unit(u):
            continue
        if "petrified" in (u.statuses or set()):
            continue
        if (u.position.x, u.position.y) in adj:
            return True
    return False


def is_scenery_unit(u) -> bool:
    """Board furniture vs combatant (single source of truth,
    2026-07-14; refines the 2026-07-11 scenery rule which treated ALL
    side>=3 units as scenery and made the Mini_Maps tentacles
    invulnerable blockers).

      scenery   = petrified (any side)  OR  attackless non-player
                  side unit (CoB/TSG statues, vortices, ToD fires):
                  always visible, unattackable, never an actor.
      combatant = everything else -- including ARMED non-petrified
                  side>=3 units (stationary tentacles): attackable,
                  fog-gated like any enemy, killable for XP.
    """
    return ("petrified" in (u.statuses or set())
            or (u.side not in (1, 2) and not u.attacks))


def enemy_villages_visible_to(state: GameState, side: int,
                              vis_set: Optional[Set[Tuple[int, int]]] = None) -> int:
    """How many villages held by `side`'s enemies the side can see.
    Wesnoth 1.18.4 never tells a player an enemy side's village
    count under fog or shroud (src/team.cpp:704-716 knows_about_team:
    "We don't know about enemies"; src/gui/dialogs/game_stats.cpp:139
    fills gold/villages/units only `if(known || see_all)`), so the
    count a player can form is over the villages on hexes it sees;
    with fog off every enemy village counts."""
    owner_map = getattr(state.global_info, "_village_owner", None) or {}
    fog_on = getattr(state.global_info, "_fog", True)
    if not fog_on:
        return sum(1 for o in owner_map.values() if o not in (0, side))
    if vis_set is None:
        vis_set = visible_hexes_for(state, side)
    return sum(1 for key, o in owner_map.items() if o not in (0, side) and key in vis_set)


def units_visible_to(
    state: GameState, side: int,
    vis_set: Optional[Set[Tuple[int, int]]] = None,
) -> List[Unit]:
    """Return the god-view unit list filtered to what `side` can
    see, per the Wesnoth fog-of-war contract.

    Rules:
      1. Own-side units: always included.
      2. Enemy units with an ACTIVE hide-cover ability that have
         not been uncovered this turn (i.e., NOT in the sim's
         `global_info._uncovered_units` set): EXCLUDED.
      3. Other enemy units: included iff their hex is in
         `visible_hexes_for(state, side)`.

    The side's seen hexes are read once per call and indexed for
    every enemy check.

    The legality contract in CLAUDE.md says hexes (not units) are
    always exposed to the encoder; we honor that by not filtering
    `state.map.hexes` -- only this function (which returns units,
    not hexes) is fog-restricted. Recruit phantoms for enemy
    sides need a separate filter at the encoder level (they're a
    distinct fog leak the simple unit filter doesn't cover).

    Callers that already hold the side's seen hexes (e.g. the
    encoder, which may have read them for the village-ownership
    fog gate) can pass them as `vis_set`; when omitted they are read
    lazily, at most once per call.

    Fog can be disabled per-game via `global_info._fog = False`
    (underscore attr so `GlobalInfo.__deepcopy__` carries it through
    MCTS state copies): the seen-hex gate is skipped and every
    non-hidden unit is visible. Hide-cover abilities still conceal
    (Wesnoth's ambush et al. work independently of fog).

    Returns a fresh list; callers may sort / reorder freely.
    """
    if not state.map.units:
        return []
    uncovered = getattr(state.global_info, "_uncovered_units", None) or set()
    fog_on = getattr(state.global_info, "_fog", True)
    out: List[Unit] = []
    for u in state.map.units:
        if u.side == side:
            out.append(u)
            continue
        # Scenery & statues are terrain-like: always visible, like
        # the map itself (fog hides UNITS' presence, not board
        # furniture). Armed side>=3 combatants (tentacles) are NOT
        # scenery -- they fall through to the enemy fog gates below.
        if is_scenery_unit(u):
            out.append(u)
            continue
        # Enemy unit. First gate: hide-cover ability (applies with
        # or without fog, as in Wesnoth). A hider is nonetheless
        # DISCOVERED while any non-incapacitated unit of the
        # observing side stands directly adjacent -- a LIVE
        # predicate, not a persistent reveal: move the adjacent unit
        # away and the hider re-hides. Only ambush-trigger / blocked
        # reveals and the hider's own attack set the persistent
        # UNCOVERED state (cleared at the hider's side's turn
        # start). `display_context.cpp:29-49 would_be_discovered`,
        # `unit.cpp:2596-2637 unit::invisible`,
        # `move.cpp:870` + `attack.cpp:1378` for the setters.
        if _hide_cover_active(state, u) and u.id not in uncovered:
            if not _discovered_by_adjacency(state, u, side):
                continue
        # Second gate: the side's seen hexes -- skipped entirely when
        # fog is off for this game. Read lazily (skip the work if
        # every enemy turns out to be hide-blocked).
        if not fog_on:
            out.append(u)
            continue
        if vis_set is None:
            vis_set = visible_hexes_for(state, side)
        if (u.position.x, u.position.y) not in vis_set:
            continue
        out.append(u)
    return out


# ---------------------------------------------------------------------
# Actor-slot contract (single source of truth, 2026-07-16)
# ---------------------------------------------------------------------
# The model's actor dimension is [visible units | own recruit
# phantoms | end_turn], and the TARGET dimension is the hex list.
# Every consumer that needs "slot i means X" MUST derive it from the
# three functions below -- the encoder builds its tokens from them
# and the behavior-cloning label builder resolves observed actions
# through them. History: these orderings used to be re-implemented
# independently ("mirrored"); when the encoder became fog-filtered
# (pre-recovery, ~2026-05) the dormant supervised-label mirror kept
# god-view enumeration and silently mislabeled 19%+ of pairs (found
# 2026-07-16 when SL was revived). Shared code, not mirrors.

def visible_units_in_slot_order(
    state: GameState, side: int,
    vis_set: Optional[Set[Tuple[int, int]]] = None,
) -> List[Unit]:
    """Unit slots 0..U-1: fog-visible units for `side`, sorted by
    (y, x, id)."""
    return sorted(
        units_visible_to(state, side, vis_set=vis_set),
        key=lambda u: (u.position.y, u.position.x, u.id),
    )


def own_recruit_types(state: GameState, side: int) -> List[str]:
    """Recruit slots U..U+R-1: the CURRENT side's recruit list, in
    side_info order (enemy lists are fog-hidden per Wesnoth's UI
    contract). Slot U+R is the end_turn sentinel."""
    if 0 < side <= len(state.sides):
        return list(state.sides[side - 1].recruits)
    return []


def hexes_in_slot_order(state: GameState) -> List:
    """Target slots: the hex list sorted row-major (y, x)."""
    return sorted(state.map.hexes,
                  key=lambda h: (h.position.y, h.position.x))


def relevant_hex_positions(state: GameState,
                           side: int) -> Set[Tuple[int, int]]:
    """The RELEVANT-SET hex positions for `side` (T2-B, 2026-07-29):
    union of
      a. own-unit single-turn reach (landable, shared planner on the
         side's OBSERVABLE context -- same primitive the legality
         mask consumes),
      b. visible-unit hexes (own + visible enemies + scenery; covers
         every legal attack-target hex),
      c. leader castle network incl. fog castle hexes + leader hex,
      d. all village hexes,   e. all castle/keep hexes.

    PURE FUNCTION OF OBSERVABLE STATE (legality-mask contract,
    CLAUDE.md #6): every component derives from terrain or the
    side's fog-filtered view; no god-view input. Measured 2026-07-29
    (T2-A): superset of every mask-offerable target hex on 1,840
    decisions x 10 ladder maps, zero violations; mean |set|/H ~0.30.

    Determinism: a set derived from deterministic components; ORDER
    is imposed by the caller filtering `hexes_in_slot_order` (see
    `relevant_hexes_in_slot_order`), so two calls on equal states
    yield identical slot orderings -- required because the trainer
    re-encodes stored states and replays target indices."""
    rel: Set[Tuple[int, int]] = set()
    from wesnoth_ai.classes import Terrain, TerrainModifiers
    for h in state.map.hexes:
        p = (h.position.x, h.position.y)
        if Terrain.VILLAGE in h.terrain_types:
            rel.add(p)
        mods = h.modifiers or set()
        if TerrainModifiers.CASTLE in mods or TerrainModifiers.KEEP in mods:
            rel.add(p)
    for u in units_visible_to(state, side):
        rel.add((u.position.x, u.position.y))
    leader = next((u for u in state.map.units
                   if u.side == side and u.is_leader), None)
    if leader is not None:
        _on_keep, net = leader_castle_network(state, leader)
        rel |= net
        rel.add((leader.position.x, leader.position.y))
    # Lazy import (codebase pattern: action_sampler does the same) --
    # visibility must not pull tools.* at module load.
    from tools.pathfind_sim import ReachContext, unit_reach
    ctx = ReachContext.for_side(state, side)
    for u in state.map.units:
        if u.side != side or "petrified" in (u.statuses or set()):
            continue
        if u.current_moves <= 0:
            continue
        rel |= set(unit_reach(u, state, ctx).landable)
    return rel


def relevant_hexes_in_slot_order(state: GameState) -> List:
    """Relevant-set variant of `hexes_in_slot_order` for the
    current side: the SAME row-major (y, x) ordering, filtered to
    `relevant_hex_positions`. Consumers (encoder / label builder)
    must choose one of the two functions per the relevant-set
    config flag -- never mix within a run (stored target indices
    are meaningless across the two hex spaces; flush buffers at
    the boundary)."""
    side = state.global_info.current_side
    rel = relevant_hex_positions(state, side)
    return [h for h in hexes_in_slot_order(state)
            if (h.position.x, h.position.y) in rel]
