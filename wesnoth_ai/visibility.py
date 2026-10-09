"""Per-side visibility for the fog-of-war contract.

The simulator keeps god-view -- combat resolution, victory checks and
the command applier need ground truth -- but when the POLICY observes
the state through the encoder and the sampler, the view is filtered to
what a real Wesnoth client would render for that side. The Rust core
that answers for a state (`game_core.core_for`) computes that filter
(rust/wesnoth_core/src/core_fog.rs, core_observe.rs); this module is
the Python face of it, so `encoder.py`, `action_sampler.py` and
`rewards.py` read one contract.

Vision and fog (docs/wesnoth_rules.md "Vision and fog")
=======================================================

A unit sees every hex it could reach in one turn spending its maximum
movement at its movement costs (doubled when it is slowed), other
units ignored, plus every hex next to one of those. A side sees its
fog: the hexes it has cleared, which the core keeps as the engine does
(core_fog.rs: `refog` at the side's turn start and end and for the
defender after a fight that killed, slowed or petrified it;
`clear_fog_from` for each hex a mover enters, a recruit's hex and an
advanced unit's hex). A side with no tracked fog sees its units'
vision from where they stand, which is what the engine clears for every
side when the game starts. Fog-off games track nothing.

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

Dependencies: classes (Unit, GameState), game_core (the core).
Dependents: rewards (visible_fraction_for), encoder (the slot order),
  replay_dataset (the label builder's slot order), the simulator.
"""
from __future__ import annotations

from typing import AbstractSet, List, Optional, Set, Tuple

import numpy as np

from wesnoth_ai.classes import GameState, Unit

# Unit types whose own cfg declares `vision=` or `[vision_costs]`
# (wesnoth_src/data/core/units, 1.18.7; pinned by
# tests/test_vision.py). Neither is modelled: the core's units see with
# their movement. None is in the default era, the pool or the corpus.
OWN_VISION_TYPES = frozenset({
    "Dune Falconer", "Dune Sky Hunter", "Dragonfly", "Grand Dragonfly",
})


def _fog_on(state: GameState) -> bool:
    return bool(getattr(state.global_info, "_fog", True))


def visible_hexes_for(state: GameState, side: int) -> AbstractSet[Tuple[int, int]]:
    """The hexes `side` sees, from the core that answers for `state`. A
    frozenset."""
    from wesnoth_ai.game_core import core_for
    cs = core_for(state)
    keys = cs.geometry().keys
    return frozenset(keys[i] for i in np.flatnonzero(cs.core.seen_export(side)).tolist())


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
    if not _fog_on(state):
        return sum(1 for o in owner_map.values() if o not in (0, side))
    if vis_set is None:
        vis_set = visible_hexes_for(state, side)
    return sum(1 for key, o in owner_map.items() if o not in (0, side) and key in vis_set)


def units_visible_to(state: GameState, side: int) -> List[Unit]:
    """The god-view unit list filtered to what `side` can see, per the
    Wesnoth fog-of-war contract, by the core that answers for `state`
    (`game_core.core_for`), in the state's unit order:

      1. Own-side units and scenery: always included.
      2. Enemy units with an ACTIVE hide-cover ability, not uncovered
         this turn (`global_info._uncovered_units`) and not discovered
         by an adjacent enemy of theirs: EXCLUDED.
      3. Other enemy units: included iff fog is off or their hex is
         one the side sees.

    The legality contract in CLAUDE.md says hexes (not units) are
    always exposed to the encoder; only this function (which returns
    units, not hexes) is fog-restricted. Returns a fresh list."""
    if not state.map.units:
        return []
    from wesnoth_ai.game_core import core_for
    ids = set(core_for(state).core.visible_ids(side))
    return [u for u in state.map.units if u.id in ids]


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

def visible_units_in_slot_order(state: GameState, side: int) -> List[Unit]:
    """Unit slots 0..U-1: fog-visible units for `side`, sorted by
    (y, x, id)."""
    return sorted(
        units_visible_to(state, side),
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
    re-encodes stored states and replays target indices.

    Computed by the core that answers for `state`
    (`observe(reach=True)`, rust/wesnoth_core/src/core_observe.rs)."""
    from wesnoth_ai.observe import observe
    return observe(state, side, reach=True).relevant_set()


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
