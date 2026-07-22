#!/usr/bin/env python3
"""Wesnoth-1.18.4-faithful single-turn route planning + move execution.

Two layers, two knowledge levels (user contract 2026-07-17):

  PLANNING (`ReachContext` / `unit_reach` / `route_to`) runs on the
  ACTING SIDE'S OBSERVABLE STATE -- exactly what Wesnoth's own client
  uses when a player orders a move (`mouse_handler::get_route` builds
  `shortest_path_calculator` with the moving player's team;
  `get_visible_unit` / `enemy_zoc` consult only units visible to that
  team -- pathfind.cpp:742-820, 134-140). Hidden units neither block
  nor exert ZoC at this layer. The legality mask and the sim's
  order-to-path translation BOTH use this layer, so anything the mask
  offers, the sim can route.

  EXECUTION (`walk_move_path`) runs GOD-VIEW and resolves what
  actually happens when the planned path meets hidden reality,
  mirroring `unit_mover` (actions/move.cpp):
    - blocked: an invisible unit sits ON a path hex -> stop on the
      hex BEFORE it, remaining MP KEPT (post_move zeroes MP only for
      ambush / ZoC-final: move.cpp:1041-1043), blocker revealed.
    - ambush: entering a hex ADJACENT to a hidden `hides` enemy ->
      stop AT that hex, MP zeroed, ambushers revealed
      (check_for_ambushers, move.cpp:422-440; reveal_ambusher sets
      STATE_UNCOVERED, move.cpp:870).
    - village capture on the FINAL hex zeroes MP (move.cpp:1046-1053);
      passing THROUGH a village mid-path does NOT stop or capture
      (plot_turn has no village stop; capture fires on final_loc only).
    - ZoC landing: the planner already made ZoC hexes terminal, so a
      planned path never continues past one; if the final hex is in
      (visible-)enemy ZoC the walk reports mp_left=0
      (`final_loc == zoc_stop_` -> set_movement(0)).
  Ambush eligibility uses the PRE-MOVE visibility snapshot: the
  engine reads a cached `invisible()` during the walk
  (unit.cpp:2613-2618, cache cleared only in post_move), so a hider
  the mover was about to discover still ambushes.

Route preference is Wesnoth's cost model, not an explicit tie-break
(see docs/wesnoth_rules.md "Default route selection"): float cost =
MP + (defense_pct + ally_occupied)/10000 per entered hex, ZoC entry
charging all remaining MP. Residual exact ties are unspecified in the
engine (heap order) and broken deterministically here.

Sighted-move interrupts are deliberately NOT modelled: exported
[move] WML carries skip_sighted="all", which replay playback honours
(synced_commands.cpp:305-314), so sim, export and playback agree.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from heapq import heappush, heappop
from typing import Dict, List, Optional, Set, Tuple

log = logging.getLogger("pathfind_sim")

# Movement cost >= this is Wesnoth's UNREACHABLE sentinel
# (movetype.hpp: UNREACHABLE = 99).
UNREACHABLE = 99

# Wesnoth's tie-break scale: subcosts divide by 10000 so they can
# never outweigh a full movement point (pathfind.cpp:820).
_SUBCOST_SCALE = 1.0 / 10000.0

Coord = Tuple[int, int]


@dataclass
class ReachContext:
    """Per-decision, per-side observable-state snapshot shared by all
    of one side's `unit_reach` calls (occupancy / ZoC / playable sets
    are side-level, not unit-level).

    `god_view=True` builds the same structures from the full unit
    list -- used only by tests and god-view tooling, never by the
    mask or the order translation (CLAUDE.md: god-view masking is
    forbidden).
    """
    side: int
    playable: frozenset
    # Hexes holding a VISIBLE unit (any side, incl. own): cannot LAND.
    occupied_visible: Set[Coord] = field(default_factory=set)
    # Hexes holding a visible ENEMY of `side`: cannot ENTER.
    enemy_hexes: Set[Coord] = field(default_factory=set)
    # Hexes holding a visible NON-enemy unit: pass-through, +1 subcost
    # (pathfind.cpp:785).
    ally_hexes: Set[Coord] = field(default_factory=set)
    # Hexes covered by a visible enemy's ZoC.
    zoc_hexes: Set[Coord] = field(default_factory=set)

    @classmethod
    def for_side(cls, gs, side: int, *, god_view: bool = False,
                 exclude_unit=None) -> "ReachContext":
        from tools.abilities import hex_neighbors
        from replay_dataset import _stats_for
        from visibility import units_visible_to, is_scenery_unit

        playable = frozenset(
            (h.position.x, h.position.y) for h in gs.map.hexes)
        if god_view:
            units = list(gs.map.units)
        else:
            units = units_visible_to(gs, side)

        ctx = cls(side=side, playable=playable)
        for u in units:
            if exclude_unit is not None and u.id == exclude_unit.id:
                continue
            pos = (u.position.x, u.position.y)
            ctx.occupied_visible.add(pos)
            if u.side == side:
                ctx.ally_hexes.add(pos)
                continue
            # All non-own sides are enemies in our 2p (+hostile
            # neutrals) setting; scenery is inert set-dressing that
            # still occupies its hex but neither fights nor ZoCs.
            ctx.enemy_hexes.add(pos)
            if is_scenery_unit(u):
                continue
            # ZoC: level >= 1, not petrified (unit.hpp:1352-1355
            # `emit_zoc_ && !incapacitated()`).
            if "petrified" in (u.statuses or set()):
                continue
            if int(_stats_for(u.name).get("level", 1)) < 1:
                continue
            ctx.zoc_hexes.update(hex_neighbors(pos[0], pos[1]))
        return ctx


@dataclass
class UnitReach:
    """Single-turn reachability for one unit under one ReachContext.

    `mp[c]`   -- integer MP spent to stand on c (start: 0)
    `cost[c]` -- Wesnoth-comparable float cost (MP + subcosts) used
                 for route preference
    `prev[c]` -- predecessor hex on the preferred route
    `landable` -- hexes the unit may END a move order on: reachable,
                 not the start, and not visibly occupied (plot_turn
                 backtracks off visible-unit end hexes,
                 move.cpp:776-780; hidden occupants do NOT bar
                 landing here -- that resolves at execution).
    """
    start: Coord
    mp: Dict[Coord, int]
    cost: Dict[Coord, float]
    prev: Dict[Coord, Coord]
    landable: Set[Coord]


# Memoized (terrain_code, unit_name) -> defense pct. The resolver
# walks the terrain alias graph per call and defense_pct_at sits on
# the Dijkstra relax hot path (one call per edge per unit per
# decision) -- same rationale as wesnoth_sim's _MVT_RESOLVE_CACHE.
_DEF_PCT_CACHE: Dict[Tuple[str, str, int], int] = {}


def defense_pct_at(unit, gs, x: int, y: int) -> int:
    """Chance-to-be-hit percent for `unit` standing on (x, y) --
    Wesnoth's `defense_modifier` (higher = worse), used as the
    per-hex tie-break subcost (pathfind.cpp:815-820)."""
    from replay_dataset import _stats_for, _terrain_keys_at
    from tools.terrain_resolver import def_pct

    codes = getattr(gs.global_info, "_terrain_codes", {}) or {}
    code = codes.get((x, y))
    if code:
        c = code
        if c[:1].isdigit() and c[1:2] == " ":
            c = c[2:]
        defenses = getattr(unit, "_defense_table", None)
        if defenses is None:
            defenses = dict(_stats_for(unit.name).get("defense", {}))
        # Key includes the defense-table content, not just the unit
        # NAME: per-unit tables exist (feral cap via traits.py), and
        # a name-keyed cache would let the first-queried unit define
        # the value for all same-named units (review finding #5).
        # Tables are ~15 entries; the frozenset hash is ~1-2us.
        cache_key = (c, unit.name, hash(frozenset(defenses.items())))
        cached = _DEF_PCT_CACHE.get(cache_key)
        if cached is not None:
            return cached
        try:
            val = int(def_pct(c, defenses))
            _DEF_PCT_CACHE[cache_key] = val
            return val
        except Exception:  # noqa: BLE001 -- fall through to keys
            pass
    defenses = getattr(unit, "_defense_table", None)
    if defenses is None:
        defenses = dict(_stats_for(unit.name).get("defense", {}))
    keys = _terrain_keys_at(gs, x, y) or ["flat"]
    vals = [int(defenses.get(k, 50) or 50) for k in keys]
    # Best (min chance-to-be-hit) over the alias keys, matching the
    # defense-MIN rule (see terrain_resolver docstring).
    return min(vals) if vals else 50


# Per-(map-terrain, unit-name, slowed) precomputed (mvt_cost,
# defense_pct) per hex. The Dijkstra relaxes ~150 edges per unit per
# decision and MCTS rebuilds masks per node -- resolving terrain
# through the alias graph per edge was ~37us/edge (5.6ms/decision
# measured 2026-07-17); one full-map precompute per unit TYPE per
# scenario amortizes to dict lookups. Keyed by a live hash of the
# terrain-code dict so mid-game terrain morphs (Aethermaw)
# invalidate naturally.
_TERRAIN_MAPS_CACHE: Dict[Tuple[int, str, bool, int],
                          Dict[Coord, Tuple[int, int]]] = {}

# Monotonic terrain-epoch source. Map builders stamp
# `gs.global_info._terrain_epoch = next_terrain_epoch()` when the
# terrain-code dict is created, and terrain-morph events bump it --
# the int survives deepcopy (MCTS forks share the cache entry) while
# morphs invalidate it. States without a stamp (synthetic tests)
# fall back to content-hashing the dict.
#
# PROCESS-SALTED (adversarial-review HIGH finding, 2026-07-18):
# actor-pool workers each start their own counter; without a salt,
# two actors' FIRST maps both stamp epoch 1, and the trainer process
# -- which rebuilds masks from shipped states for the distillation
# loss -- would serve map A's cached movement/defense tables for map
# B's states. 62 random bits make cross-process collision
# negligible while staying a plain int (deepcopy/pickle-safe).
import itertools as _itertools
import secrets as _secrets
_TERRAIN_EPOCH = _itertools.count(_secrets.randbits(62))


def next_terrain_epoch() -> int:
    return next(_TERRAIN_EPOCH)


def _terrain_maps_for(unit, gs) -> Dict[Coord, Tuple[int, int]]:
    from tools.wesnoth_sim import _move_cost_at_hex
    codes = getattr(gs.global_info, "_terrain_codes", {}) or {}
    thash = getattr(gs.global_info, "_terrain_epoch", None)
    if thash is None and not codes:
        # Synthetic/test states (no terrain-code dict): the cache key
        # can't capture map shape, and all such states would collide
        # on one key (this DID pollute cross-test state, suite run
        # 2026-07-17) -- build fresh, uncached. Tiny grids only.
        return {
            (h.position.x, h.position.y): (
                _move_cost_at_hex(unit, gs, h.position.x, h.position.y),
                defense_pct_at(unit, gs, h.position.x, h.position.y))
            for h in gs.map.hexes
        }
    if thash is None:
        thash = hash(frozenset(codes.items()))
    slowed = "slowed" in (unit.statuses or set())
    _dt = getattr(unit, "_defense_table", None)
    _dt_hash = hash(frozenset(_dt.items())) if _dt is not None else 0
    key = (thash, unit.name, slowed, _dt_hash)
    m = _TERRAIN_MAPS_CACHE.get(key)
    if m is None:
        m = {}
        for h in gs.map.hexes:
            pos = (h.position.x, h.position.y)
            m[pos] = (_move_cost_at_hex(unit, gs, pos[0], pos[1]),
                      defense_pct_at(unit, gs, pos[0], pos[1]))
        _TERRAIN_MAPS_CACHE[key] = m
        if len(_TERRAIN_MAPS_CACHE) > 512:
            # Drop-all backstop against unbounded growth across many
            # scenarios in one process (each entry is one map x unit
            # type; 512 is far above a training run's working set).
            _TERRAIN_MAPS_CACHE.clear()
            _TERRAIN_MAPS_CACHE[key] = m
    return m


# Array-form terrain cache (2026-07-22 enumerate optimization): same
# key discipline as _TERRAIN_MAPS_CACHE, but flat numpy arrays indexed
# by hex id so unit_reach's inner loop does array reads instead of
# tuple-keyed dict.get (1.69M dict lookups per 240 enumerates on
# midgame states -- ~60% of the legality mask's cost).
_TERRAIN_ARRAYS_CACHE: Dict = {}


def _terrain_arrays_for(unit, gs):
    """(pos_to_idx, positions, nbr_idx[H,6], mcost[H], dsub[H]) for
    this (map, unit-type) pair. nbr_idx column order == hex_neighbors
    order (unit_reach's push order -- and therefore its heap
    tie-break behavior -- depends on it). -1 = off-map."""
    import numpy as np
    from tools.abilities import hex_neighbors

    codes = getattr(gs.global_info, "_terrain_codes", {}) or {}
    thash = getattr(gs.global_info, "_terrain_epoch", None)
    key = None
    if thash is not None or codes:
        if thash is None:
            thash = hash(frozenset(codes.items()))
        slowed = "slowed" in (unit.statuses or set())
        _dt = getattr(unit, "_defense_table", None)
        _dt_hash = hash(frozenset(_dt.items())) if _dt is not None else 0
        key = ("arr", thash, unit.name, slowed, _dt_hash)
        cached = _TERRAIN_ARRAYS_CACHE.get(key)
        if cached is not None:
            return cached

    tmaps = _terrain_maps_for(unit, gs)
    positions = [(h.position.x, h.position.y) for h in gs.map.hexes]
    pos_to_idx = {p: i for i, p in enumerate(positions)}
    # Plain Python lists, NOT numpy: the Dijkstra reads these one
    # scalar at a time, and numpy scalar indexing is several times
    # slower than list indexing (measured 2026-07-22: the numpy
    # variant was 1.8x SLOWER than the tuple-dict original).
    mcost = [0] * len(positions)
    dsub = [0] * len(positions)
    nbrs = []
    for i, p in enumerate(positions):
        c, d = tmaps[p]
        mcost[i] = int(c)
        dsub[i] = int(d)
        nbrs.append(tuple(pos_to_idx.get(npos, -1)
                          for npos in hex_neighbors(p[0], p[1])))
    out = (pos_to_idx, positions, nbrs, mcost, dsub)
    if key is not None:
        _TERRAIN_ARRAYS_CACHE[key] = out
        if len(_TERRAIN_ARRAYS_CACHE) > 512:
            _TERRAIN_ARRAYS_CACHE.clear()
            _TERRAIN_ARRAYS_CACHE[key] = out
    return out


def unit_reach(unit, gs, ctx: ReachContext,
               budget: Optional[int] = None) -> UnitReach:
    """Array-index port of `_unit_reach_reference` (2026-07-22):
    IDENTICAL algorithm, float composition, and heap/neighbor order
    -- so mp/cost/prev/landable are bit-exact (pinned by
    test_pathfind_parity) -- with hex-id arrays replacing the
    tuple-keyed dicts that dominated the profile.
    """
    start = (unit.position.x, unit.position.y)
    if budget is None:
        budget = int(unit.current_moves)
    skirmisher = "skirmisher" in (unit.abilities or set())
    pos_to_idx, positions, nbrs, mcost, dsub = \
        _terrain_arrays_for(unit, gs)
    H = len(positions)
    s_idx = pos_to_idx[start]

    # ctx sets -> byte flags (sets are small; H is a few hundred).
    zoc = bytearray(H)
    for p in ctx.zoc_hexes:
        i = pos_to_idx.get(p)
        if i is not None:
            zoc[i] = 1
    enemy = bytearray(H)
    for p in ctx.enemy_hexes:
        i = pos_to_idx.get(p)
        if i is not None:
            enemy[i] = 1
    ally = bytearray(H)
    for p in ctx.ally_hexes:
        i = pos_to_idx.get(p)
        if i is not None:
            ally[i] = 1

    INF = float("inf")
    mp_l = [-1] * H
    cost_l = [INF] * H
    prev_l = [-1] * H
    mp_l[s_idx] = 0
    cost_l[s_idx] = 0.0

    seq = 0
    heap: List[Tuple[float, int, int]] = [(0.0, 0, s_idx)]
    while heap:
        c, _, i = heappop(heap)
        if c > cost_l[i]:
            continue
        spent = mp_l[i]
        if i != s_idx and not skirmisher and zoc[i]:
            continue
        if spent >= budget:
            continue
        remaining = budget - spent
        for ni in nbrs[i]:
            if ni < 0 or enemy[ni]:
                continue
            terrain_cost = mcost[ni]
            if terrain_cost >= UNREACHABLE or terrain_cost > remaining:
                continue
            if not skirmisher and zoc[ni]:
                mp_charge = remaining          # pathfind.cpp:806
            else:
                mp_charge = terrain_cost
            subcost = dsub[ni]
            if ally[ni]:
                subcost += 1                   # pathfind.cpp:785
            ncost = c + mp_charge + subcost * _SUBCOST_SCALE
            if ncost < cost_l[ni]:
                cost_l[ni] = ncost
                mp_l[ni] = spent + mp_charge
                prev_l[ni] = i
                seq += 1
                heappush(heap, (ncost, seq, ni))

    mp: Dict[Coord, int] = {}
    cost: Dict[Coord, float] = {}
    prev: Dict[Coord, Coord] = {}
    for i in range(H):
        if mp_l[i] >= 0:
            p = positions[i]
            mp[p] = mp_l[i]
            cost[p] = cost_l[i]
            if prev_l[i] >= 0:
                prev[p] = positions[prev_l[i]]
    landable = {
        pos for pos in mp
        if pos != start and pos not in ctx.occupied_visible
    }
    return UnitReach(start=start, mp=mp, cost=cost, prev=prev,
                     landable=landable)


def _unit_reach_reference(unit, gs, ctx: ReachContext,
                          budget: Optional[int] = None) -> UnitReach:
    """Dijkstra over Wesnoth's single-turn cost model.

    Per entered hex (pathfind.cpp:742-820):
      - visible enemy on hex -> cannot enter (getNoPathValue);
      - terrain cost via _move_cost_at_hex; > remaining MP -> can't
        enter this turn (no multi-turn wrap: orders are per-turn);
      - entering a (visible-enemy) ZoC hex charges ALL remaining MP
        unless skirmisher (move_cost += remaining_movement,
        line 806) -- terminal for the turn;
      - float cost adds (defense_pct + ally_occupied)/10000 as the
        route-preference tie-break.

    The start hex is exempt from its own ZoC status (a unit may
    always leave a ZoC it starts in -- plot_turn only stops on
    ENTERED hexes, move.cpp:741-770).
    """
    from tools.abilities import hex_neighbors

    start = (unit.position.x, unit.position.y)
    if budget is None:
        budget = int(unit.current_moves)
    skirmisher = "skirmisher" in (unit.abilities or set())
    tmaps = _terrain_maps_for(unit, gs)

    mp: Dict[Coord, int] = {start: 0}
    cost: Dict[Coord, float] = {start: 0.0}
    prev: Dict[Coord, Coord] = {}
    # (float_cost, seq, x, y): seq keeps heap order deterministic.
    seq = 0
    heap: List[Tuple[float, int, int, int]] = [(0.0, 0, start[0], start[1])]

    while heap:
        c, _, x, y = heappop(heap)
        if c > cost.get((x, y), float("inf")):
            continue
        spent = mp[(x, y)]
        # Terminal hexes: ZoC (entered, non-skirmisher) or MP gone.
        if (x, y) != start and not skirmisher and (x, y) in ctx.zoc_hexes:
            continue
        if spent >= budget:
            continue
        remaining = budget - spent
        for npos in hex_neighbors(x, y):
            if npos in ctx.enemy_hexes:
                continue
            tm = tmaps.get(npos)
            if tm is None:                     # off-map
                continue
            terrain_cost, def_sub = tm
            if terrain_cost >= UNREACHABLE or terrain_cost > remaining:
                continue
            if not skirmisher and npos in ctx.zoc_hexes:
                mp_charge = remaining          # pathfind.cpp:806
            else:
                mp_charge = terrain_cost
            subcost = def_sub
            if npos in ctx.ally_hexes:
                subcost += 1                   # pathfind.cpp:785
            ncost = c + mp_charge + subcost * _SUBCOST_SCALE
            if ncost < cost.get(npos, float("inf")):
                cost[npos] = ncost
                mp[npos] = spent + mp_charge
                prev[npos] = (x, y)
                seq += 1
                heappush(heap, (ncost, seq, npos[0], npos[1]))

    landable = {
        pos for pos in mp
        if pos != start and pos not in ctx.occupied_visible
    }
    return UnitReach(start=start, mp=mp, cost=cost, prev=prev,
                     landable=landable)


def route_to(reach: UnitReach, target: Coord) -> Optional[List[Coord]]:
    """Preferred route start..target, or None if unreached."""
    if target not in reach.mp:
        return None
    path = [target]
    while path[-1] != reach.start:
        path.append(reach.prev[path[-1]])
    path.reverse()
    return path


@dataclass
class MoveOutcome:
    """Result of god-view execution of a planned path."""
    final_idx: int              # index into the path where the unit ends
    mp_left: int
    uncovered_ids: List[str]    # hider ids revealed (STATE_UNCOVERED)
    stop_reason: str            # "end" | "blocked" | "ambush"


def walk_move_path(gs, unit, xs: List[int], ys: List[int],
                   *, budget: Optional[int] = None,
                   enforce_budget: bool = True) -> MoveOutcome:
    """God-view execution of a move order along xs/ys (index 0 = the
    unit's current hex). Mirrors unit_mover's do_move walk; see
    module docstring for the blocked / ambush / village / ZoC rules
    and their move.cpp citations.

    `enforce_budget=False` is the RECONSTRUCTION mode: the engine
    already validated the recorded move when it was played, so a
    budget overrun (which can only mean our reconstructed MP
    drifted) must not truncate a human path -- MP just clamps to 0.
    The policy path keeps enforcement on (its orders are our own to
    validate).

    Does NOT mutate gs -- callers apply the outcome (position, MP,
    `_uncovered_units`, village capture) themselves, so the
    reconstruction path (replay_dataset._apply_command) and the
    policy path (WesnothSim.step) share one truncation semantics.
    """
    from tools.abilities import hex_neighbors
    from tools.wesnoth_sim import _move_cost_at_hex

    side = unit.side
    if budget is None:
        budget = int(unit.current_moves)
    skirmisher = "skirmisher" in (unit.abilities or set())

    unit_at: Dict[Coord, object] = {
        (u.position.x, u.position.y): u for u in gs.map.units
        if u.id != unit.id
    }
    uncovered: Set[str] = set(
        getattr(gs.global_info, "_uncovered_units", None) or set())

    # PRE-MOVE visibility snapshot for ambush eligibility: hiders
    # already discovered (ANY enemy of the hider adjacent,
    # would_be_discovered -- display_context.cpp:29-49; the check is
    # viewer-independent, so a third-party neutral adjacent to the
    # hider also breaks its hiding) or already uncovered don't
    # ambush; the snapshot is NOT updated as the mover walks (engine
    # reads the pre-move invisibility cache, unit.cpp:2613-2618).
    from visibility import _hide_cover_active as _cover_active
    from visibility import is_scenery_unit as _is_scenery
    def _is_hidden_hider(u) -> bool:
        if u.side == side or u.id in uncovered:
            return False
        if "petrified" in (u.statuses or set()):
            return False
        if not _cover_active(gs, u):
            return False
        # would_be_discovered: any adjacent enemy OF THE HIDER
        # (non-scenery, non-petrified) pre-move.
        hider_adj = set(hex_neighbors(u.position.x, u.position.y))
        for other in gs.map.units:
            if other.side == u.side or _is_scenery(other):
                continue
            if "petrified" in (other.statuses or set()):
                continue
            if (other.position.x, other.position.y) in hider_adj:
                return False
        return True

    hidden_hiders = [u for u in gs.map.units if _is_hidden_hider(u)]
    hider_adjacent: Set[Coord] = set()
    for hu in hidden_hiders:
        hider_adjacent.update(
            hex_neighbors(hu.position.x, hu.position.y))

    # ZoC at execution: plot_turn checks enemy_zoc with the CURRENT
    # TEAM's visibility (move.cpp:765-768), so hidden hiders exert
    # no ZoC even here (they stop movers via ambush instead).
    # Snapshot once per walk (route-start knowledge, like the
    # engine's plot).
    zoc_hexes: Set[Coord] = set()
    if not skirmisher:
        from replay_dataset import _stats_for
        from visibility import units_visible_to
        for u in units_visible_to(gs, side):
            if u.side == side:
                continue
            if "petrified" in (u.statuses or set()):
                continue
            if int(_stats_for(u.name).get("level", 1)) < 1:
                continue
            zoc_hexes.update(
                hex_neighbors(u.position.x, u.position.y))

    newly_uncovered: List[str] = []
    cum_cost = [0]                # cum_cost[j] = MP to stand at xs[j]
    final_idx = 0
    stop_reason = "end"

    for j in range(1, len(xs)):
        npos = (xs[j], ys[j])
        blocker = unit_at.get(npos)
        if blocker is not None and blocker.side != side:
            # Enemy occupant on a path hex. Planned paths never
            # cross VISIBLE enemies, so this occupant was hidden =
            # Wesnoth's "blocked": stop BEFORE it, reveal it, KEEP
            # remaining MP (move.cpp: post_move zeroes MP only for
            # ambush/ZoC-final).
            newly_uncovered.append(blocker.id)
            stop_reason = "blocked"
            break
        # Own-side occupant: pass-through (pathfind.cpp:777-786);
        # landing on it is prevented by the backtrack below.
        step_cost = _move_cost_at_hex(unit, gs, xs[j], ys[j])
        if enforce_budget and (step_cost >= UNREACHABLE
                               or cum_cost[-1] + step_cost > budget):
            break
        cum_cost.append(cum_cost[-1] + step_cost)
        final_idx = j
        # Ambush: entered a hex adjacent to a (pre-move-)hidden hides
        # enemy -> stop here, reveal ALL adjacent hidden hiders, MP
        # zeroed below.
        if npos in hider_adjacent:
            for hu in hidden_hiders:
                if npos in hex_neighbors(hu.position.x, hu.position.y):
                    newly_uncovered.append(hu.id)
            stop_reason = "ambush"
            break
        # ZoC: entering a visible enemy's ZoC ends movement mid-path
        # (mirror plot_turn's stop; recorded human paths normally
        # already end at ZoC, so this fires only on model drift).
        if not skirmisher and npos in zoc_hexes and j < len(xs) - 1:
            stop_reason = "zoc"
            break

    # Backtrack off occupied end hexes (plot_turn, move.cpp:776-780):
    # a pass-through ally can't be the LANDING hex. MP for the backed-
    # off steps is refunded (cum_cost indexes the shorter prefix).
    while final_idx > 0 and (xs[final_idx], ys[final_idx]) in unit_at:
        final_idx -= 1

    if final_idx == 0:
        return MoveOutcome(final_idx=0, mp_left=budget,
                           uncovered_ids=newly_uncovered,
                           stop_reason=stop_reason)

    fx, fy = xs[final_idx], ys[final_idx]
    mp_left = max(0, budget - cum_cost[final_idx])
    if stop_reason == "ambush":
        mp_left = 0                          # move.cpp:1041-1043
    elif not skirmisher and (fx, fy) in zoc_hexes:
        mp_left = 0                          # final_loc == zoc_stop_
    # Village capture on the FINAL hex zeroes MP (move.cpp:1046-1053).
    elif _is_capturable_village(gs, fx, fy, side):
        mp_left = 0

    return MoveOutcome(final_idx=final_idx, mp_left=mp_left,
                       uncovered_ids=newly_uncovered,
                       stop_reason=stop_reason)


def _is_capturable_village(gs, x: int, y: int, side: int) -> bool:
    from replay_dataset import _terrain_at
    if _terrain_at(gs, x, y) != "village":
        return False
    owners = getattr(gs.global_info, "_village_owner", {}) or {}
    return owners.get((x, y)) != side
