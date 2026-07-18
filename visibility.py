"""Per-side visibility for the fog-of-war contract.

The Wesnoth simulator (`tools/wesnoth_sim.py`) maintains god-view
internally -- it must, because combat resolution, recall lists,
victory checks, and the action-applier all need ground truth.
But when the POLICY observes state via the encoder + sampler, the
view has to be filtered so the model only sees what a real
Wesnoth client would render for that side.

This module is the single source of truth for that filter.
`encoder.py`, `action_sampler.py`, and `rewards.py` all import
from here, so the contract is consistent everywhere.

Public API
==========

  sight_radius_for(unit) -> int
      Per-Wesnoth-default sight range in hexes. Defaults to the
      unit's `max_moves` (the engine's fallback when `vision`
      isn't specified). Reads `max_moves` from our `Unit` dataclass.

  visible_hexes_for(state, side) -> Set[Tuple[int, int]]
      Hexes the side can SEE -- union of each of side's units'
      sight discs. Used for fog-clearing reward and for the
      sight component of unit-visibility.

  visible_fraction_for(state, side) -> float
      `len(visible_hexes_for) / total_map_hexes`. Range [0, 1].
      Used by the continuous-payment fog_reveal_weight shaping
      reward (see rewards.WeightedReward).

  units_visible_to(state, side) -> List[Unit]
      Side's god-view list filtered by the visibility rules:
        * own-side units: always
        * enemy units hiding under an active ambush ability AND
          not on the sim's `_uncovered_units` set: NEVER
        * other enemy units: only if in `visible_hexes_for(side)`
      The result is what the policy / sampler should iterate over
      when treating units as observations.

Visibility rules
================

Wesnoth's UI fog of war hides three things from a side:
  1. Enemy units outside the side's sight discs.
  2. Enemy units within sight discs that are hiding via an
     ability (ambush in forest, concealment in village,
     submerge in deep water, nightstalk at night) -- unless
     they have been "uncovered" this turn cycle.
  3. (Shroud only) terrain itself in unscouted hexes. We
     don't model shroud separately; the encoder already
     retains all hex tokens, per the legality-mask contract
     in CLAUDE.md.

We model rules 1 and 2 with this module. Rule 3 is a no-op
because our sim doesn't track shroud -- every hex is
considered terrain-visible. (Wesnoth scenarios used by our
training set are "fog of war on" with shroud off by default in
multiplayer ladder, so this matches the training data.)

The sim's `_uncovered_units` set (managed in
`WesnothSim._refresh_uncovered_state` and the ambush-trigger
path) is the authoritative record of which hiding units are
exposed. We read it directly; no copy.

Sight-radius simplification
===========================

Wesnoth's true vision computation respects terrain (some terrain
costs more to see through, some -- "vision-cost" -- can block
vision entirely). We approximate with a flat hex-distance disc of
radius `max_moves`. That's the engine's fallback when `vision`
isn't specified and is the default for every unit we encode
(per `tools/scrape_unit_stats.py`'s vision-field handling).

This loose disc occasionally credits visibility on hexes Wesnoth
itself wouldn't (e.g., across a vision-blocking mountain). The
error is small (< 5% of hexes typical) and conservative in the
right direction for training: the agent learns to fight with
slightly MORE optimistic vision than Wesnoth provides; on real
Wesnoth deploy the policy effectively over-estimates its
information, which is a benign failure mode (the sampler's
legality mask catches the actual moves Wesnoth would accept).

Dependencies: classes (Unit, GameState), replay_dataset
  (_terrain_keys_at, _lawful_bonus_at -- read-only).
Dependents: rewards (visible_fraction_for), encoder
  (units_visible_to), action_sampler (units_visible_to),
  tests/visibility/test_visibility.py.
"""

from __future__ import annotations

from typing import FrozenSet, Iterable, List, Optional, Set, Tuple

import numpy as np

from classes import GameState, Unit


# Cover abilities -- a unit with one of these CAN hide on the
# matching terrain / ToD. Match `WesnothSim._AMBUSH_ABILITIES`
# (kept verbatim so a future engine change touches one place
# only -- if these diverge, the sim's `_uncovered_units` set
# will get out of sync with our visibility check).
_AMBUSH_ABILITIES = frozenset({
    "ambush", "nightstalk", "concealment", "submerge",
})


def sight_radius_for(unit: Unit) -> int:
    """Wesnoth sight-disc radius for `unit`, in hexes.

    Defaults to `max_moves` (Wesnoth's engine fallback when the
    unit's `vision` attribute is unset). Our `Unit` dataclass
    doesn't carry a separate `vision` field -- `max_moves` is
    the proxy. Always >= 1 so degenerate `moves=0` units (boats
    of certain scenarios, etc.) still contribute a 1-hex disc.
    """
    return max(int(getattr(unit, "max_moves", 5)), 1)


def _hex_distance(ax: int, ay: int, bx: int, by: int) -> int:
    """Wesnoth hex distance (odd-q offset). Inlined from
    rewards.hex_distance so this module has no cyclic import
    risk; the formula is verbatim against `wesnoth_src/src/
    map_location.cpp::distance_between`.
    """
    hd = abs(ax - bx)
    a_even = (ax & 1) == 0
    b_even = (bx & 1) == 0
    vpenalty = 0
    if (a_even and not b_even and ay <= by) or \
       (b_even and not a_even and by <= ay):
        vpenalty = 1
    return max(hd, abs(ay - by) + hd // 2 + vpenalty)


def visible_hexes_for(state: GameState,
                      side: int) -> Set[Tuple[int, int]]:
    """Set of (x, y) hex coordinates `side` can see in `state`.

    Computed by iterating each unit on `side` and unioning the
    sight discs. Returns a fresh Set each call (callers may
    freeze if they want a long-lived snapshot).

    Cost: O(|units(side)| * |hexes|) per call. Hex iteration
    is over the side's BUFFER of hex positions which is small in
    practice (~1500 max on largest ladder maps); typical games
    have 6-15 units per side, so ~10k-25k distance calls per
    invocation. In the µs regime; safe to call per-step.

    If the map has no hexes or the side has no units, returns
    an empty set (the side sees nothing).
    """
    visible: Set[Tuple[int, int]] = set()
    if not state.map.units or not state.map.hexes:
        return visible
    our = [u for u in state.map.units if u.side == side]
    if not our:
        return visible
    # Optimization #4 (2026-06-14): vectorize the per-unit distance
    # disc over all hexes with numpy. The scalar _hex_distance loop
    # was O(units x hexes) Python calls -- ~4.4x slower on a 1175-hex
    # ladder map. This is a BIT-IDENTICAL transcription of
    # `_hex_distance` (odd-q offset; verified against it in
    # test_visibility); coords are emitted as python ints so set
    # membership matches the scalar path exactly.
    hex_coords = [(h.position.x, h.position.y) for h in state.map.hexes]
    hxs = np.fromiter((c[0] for c in hex_coords), dtype=np.int64,
                      count=len(hex_coords))
    hys = np.fromiter((c[1] for c in hex_coords), dtype=np.int64,
                      count=len(hex_coords))
    hx_even = (hxs & 1) == 0
    for u in our:
        r = sight_radius_for(u)
        ux, uy = u.position.x, u.position.y
        hd = np.abs(ux - hxs)
        # vpenalty mirrors _hex_distance's odd-q vertical penalty:
        #   (a_even & ~b_even & ay<=by) | (b_even & ~a_even & by<=ay)
        # with a=(ux,uy) the unit, b=(hx,hy) the hex.
        a_even = (ux & 1) == 0
        if a_even:
            vpen = (~hx_even) & (uy <= hys)
        else:
            vpen = hx_even & (hys <= uy)
        dist = np.maximum(hd, np.abs(uy - hys) + (hd >> 1) + vpen)
        for i in np.nonzero(dist <= r)[0]:
            visible.add(hex_coords[i])
    return visible


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
    # sight-disc coverage that carries no information value.
    if not getattr(state.global_info, "_fog", True):
        return 1.0
    return len(visible_hexes_for(state, side)) / len(hexes)


def _hide_cover_active(state: GameState, unit: Unit) -> bool:
    """True iff `unit` has a hide ability AND its current hex's
    terrain (or ToD, for nightstalk) satisfies the ability's
    cover condition.

    Verified abilities + their covers from
    `wesnoth_src/data/core/abilities.cfg`:

      - ambush:      forest terrain
      - concealment: village terrain
      - submerge:    deep_water terrain
      - nightstalk:  current ToD has lawful_bonus < 0
                     (night / second_watch)

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
    from tools.replay_dataset import _terrain_keys_at, _lawful_bonus_at
    keys = _terrain_keys_at(state, unit.position.x, unit.position.y)
    if "ambush" in abilities and "forest" in keys:
        return True
    if "concealment" in abilities and "village" in keys:
        return True
    if "submerge" in abilities and "deep_water" in keys:
        return True
    if "nightstalk" in abilities:
        bonus = _lawful_bonus_at(
            state, unit.position.x, unit.position.y,
            state.global_info.turn_number,
        )
        if bonus < 0:
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
    from classes import TerrainModifiers

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

    The visibility-disc computation is shared per call (the
    side's hex set is materialised once and indexed for every
    enemy check), so the cost stays in the µs regime even with
    many enemies.

    The legality contract in CLAUDE.md says hexes (not units) are
    always exposed to the encoder; we honor that by not filtering
    `state.map.hexes` -- only this function (which returns units,
    not hexes) is fog-restricted. Recruit phantoms for enemy
    sides need a separate filter at the encoder level (they're a
    distinct fog leak the simple unit filter doesn't cover).

    Callers that already hold the side's vision disc (e.g. the
    encoder, which may have computed it for the village-ownership
    fog gate) can pass it as `vis_set` to skip the recompute; when
    omitted it is computed lazily, at most once per call.

    Fog can be disabled per-game via `global_info._fog = False`
    (underscore attr so `GlobalInfo.__deepcopy__` carries it through
    MCTS state copies): the sight-disc gate is skipped and every
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
        # Second gate: sight disc -- skipped entirely when fog is
        # off for this game. Compute lazily (skip the work if every
        # enemy turns out to be hide-blocked).
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
