#!/usr/bin/env python3
"""Terrain-overlay movement/defense resolution (2026-07-15).

Pins the movetype.cpp alias semantics (docs/wesnoth_rules.md
"mvt_alias resolution"): the alias walk defaults to BEST-of
(lowest movement cost / lowest chance-to-hit); a MINUS marker in
the alias list flips to WORST-of. Village overlays carry no
marker, so Village+Other is best-of movement -- user-verified in
real Wesnoth for swamp and mountain villages on Fallenstar Lake
(2026-07-15). Forest overlays DO carry the marker (mvt_alias=
-,_bas,Ft) and resolve worst-of.

Regression target: an earlier session claimed "tentacles cannot
move onto villages, even water ones" -- wrong. The resolver (and
the sim, which delegates via wesnoth_sim._move_cost_at_hex's memo)
prices water/swamp villages at the water/swamp cost for float
movetypes; only dry-base villages stay impassable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from tools.terrain_resolver import def_pct, mvt_cost

_DB = json.loads(
    (Path(__file__).parent.parent / "unit_stats.json").read_text(
        encoding="utf-8"))


def _unit(name):
    u = _DB["units"][name]
    return u["movement_costs"], u["defense"]


def test_village_overlays_are_best_of_movement_for_float():
    costs, defs = _unit("Tentacle of the Deep")
    # Water/swamp base + village overlay: village cost is 99 for
    # float, but best-of keeps the base terrain's cost.
    assert mvt_cost("Ww", costs) == 1
    assert mvt_cost("Ww^Vm", costs) == 1      # water village
    assert mvt_cost("Ss", costs) == 2
    assert mvt_cost("Ss^Vhs", costs) == 2     # swamp village
    # Dry-base villages: both components 99 -> still impassable.
    assert mvt_cost("Gg^Vh", costs) == 99
    assert mvt_cost("Mm^Vhh", costs) == 99
    # Defense follows the same best-of (lowest CTH) walk.
    assert def_pct("Ww^Vm", defs) == def_pct("Ww", defs) == 50


def test_village_overlays_are_best_of_movement_for_smallfoot():
    # The classic direction: a land unit entering a mountain or
    # swamp village pays the VILLAGE cost (1), not the harsh base.
    costs, _ = _unit("Spearman")
    assert mvt_cost("Mm", costs) > 1
    assert mvt_cost("Mm^Vhh", costs) == 1
    assert mvt_cost("Ss", costs) > 1
    assert mvt_cost("Ss^Vhs", costs) == 1


def test_forest_overlay_is_worst_of_movement():
    # ^Fp carries the MINUS marker (mvt_alias=-,_bas,Ft): a horse
    # unit on forested grass pays the forest cost, not grass.
    costs, _ = _unit("Horseman")
    assert mvt_cost("Gg", costs) == 1
    assert mvt_cost("Gg^Fp", costs) == max(
        mvt_cost("Gg", costs), mvt_cost("Gg^Fp", costs))
    assert mvt_cost("Gg^Fp", costs) > 1


def test_terrain_event_preserves_overlay_in_codes():
    """A scenario [terrain] event whose new code carries an overlay
    must store the FULL code in `_terrain_codes` -- the movement /
    defense resolvers walk the alias graph from that code, and the
    overlay can dominate it.

    Regression (found 2026-07-29 via the Aethermaw export census):
    `_terrain_action` stored the overlay-STRIPPED base ('Chw^Xo' ->
    'Chw'), so Aethermaw's turn-6 whirlpool walls (WML (22,19) /
    (28,22)) priced as walkable water-castles. Self-play moved units
    onto them, and the exported replays fail strict-sync in real
    Wesnoth ("found corrupt movement in replay" -- engine-verified
    on 2/2 such exports). Engine truth: ^Xo is the Impassable
    Overlay, mvt_alias=Xt (wesnoth_src/data/core/terrain.cfg:
    1743-1751), so the composite is impassable for every movetype.
    """
    from sim_test_helpers import fresh_scenario_sim
    from tools.replay_dataset import _fire_turn_events
    from tools.wesnoth_sim import _move_cost_at_hex

    sim = fresh_scenario_sim(0, scenario_id="multiplayer_Aethermaw")
    gs = sim.gs
    # Production event path, both sides' full morph schedule.
    for side, turn in [(1, 4), (2, 4), (1, 5), (2, 5), (1, 6), (2, 6)]:
        _fire_turn_events(gs, side, turn)

    codes = getattr(gs.global_info, "_terrain_codes")
    # The two wall hexes keep their overlay (WML (22,19)/(28,22) ->
    # python (21,18)/(27,21))...
    assert codes[(21, 18)] == "Chw^Xo"
    assert codes[(27, 21)] == "Chw^Xo"

    # ...and price impassable for a real unit from the scenario.
    u = next(x for x in gs.map.units if x.side == 1)
    assert _move_cost_at_hex(u, gs, 21, 18) >= 99
    assert _move_cost_at_hex(u, gs, 27, 21) >= 99

    # Control -- the plain-code morphs stay walkable water-castles
    # (don't over-block): WML (13,13) -> python (12,12).
    assert codes[(12, 12)] == "Chw"
    assert _move_cost_at_hex(u, gs, 12, 12) < 99
