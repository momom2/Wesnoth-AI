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

    Mirrors `WesnothSim._hide_cover_active`; if the rules drift,
    this needs to update in lockstep with the sim.
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


def units_visible_to(state: GameState, side: int) -> List[Unit]:
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

    Returns a fresh list; callers may sort / reorder freely.
    """
    if not state.map.units:
        return []
    uncovered = getattr(state.global_info, "_uncovered_units", None) or set()
    # Compute the visibility disc ONCE, then index it for every
    # enemy. The alternative (one disc check per enemy via
    # _hex_distance from each own-unit) is the same work, but
    # the set-based form is faster on cpython and reusable.
    vis_set: Optional[Set[Tuple[int, int]]] = None
    out: List[Unit] = []
    for u in state.map.units:
        if u.side == side:
            out.append(u)
            continue
        # Enemy unit. First gate: hide-cover ability.
        if _hide_cover_active(state, u) and u.id not in uncovered:
            continue
        # Second gate: sight disc. Compute lazily (skip the work
        # entirely if every enemy turns out to be hide-blocked).
        if vis_set is None:
            vis_set = visible_hexes_for(state, side)
        if (u.position.x, u.position.y) not in vis_set:
            continue
        out.append(u)
    return out
