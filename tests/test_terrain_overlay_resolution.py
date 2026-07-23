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
