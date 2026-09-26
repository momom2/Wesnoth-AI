#!/usr/bin/env python3
"""The hide-cover census over the Ladder pool, reproducibly.

For every playable hex of the 21 Ladder maps (`scenario_pool.
LADDER_SCENARIO_IDS`, the border ring stripped as `parse_map_data`
strips it): how many carry a forest overlay, a village overlay or a
deep-water base; how many of those give `ambush`, `concealment` and
`submerge` their cover under the engine's `[hides]` globs
(`terrain_resolver.hides_cover`); and, for the encoder's one-hot
finding, how many forest-overlay hexes the encoder's single terrain id
(`encoder._first_terrain_id`) labels FLAT, how many it labels
anything but FOREST, and how many the hex's alias-derived terrain set
(`Hex.terrain_mask`, the `terrain_multi_hot` view) leaves without the
FOREST bit.

    python tools/analysis/hide_cover_census.py [--json OUT]

The 2026-09-13 docs quoted this census from a reviewer's run with no
record; this is the tool and its output is the record
(training/metrics/bench_pipeline/hide_cover_20260913/census.json).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from wesnoth_ai.rules.scenario_pool import LADDER_SCENARIO_IDS, build_scenario_gamestate, random_setup  # noqa: E402
from wesnoth_ai.rules.terrain_resolver import hides_cover, strip_start_position  # noqa: E402
from wesnoth_ai.sim.classes import Terrain  # noqa: E402
from wesnoth_ai.encoder import _first_terrain_id  # noqa: E402


def census(scenario_ids: List[str]) -> Dict[str, object]:
    base = random_setup(random.Random(0))
    totals = {"maps": 0, "playable_hexes": 0, "forest_overlay": 0, "village_overlay": 0,
              "deep_water_base": 0, "ambush_cover": 0, "concealment_cover": 0, "submerge_cover": 0,
              "forest_overlay_encoded_flat": 0, "forest_overlay_not_encoded_forest": 0,
              "forest_overlay_mask_without_forest": 0}
    per_map: Dict[str, Dict[str, int]] = {}
    for sid in scenario_ids:
        gs = build_scenario_gamestate(dataclasses.replace(base, scenario_id=sid))
        codes = getattr(gs.global_info, "_terrain_codes", {}) or {}
        by_pos = {(h.position.x, h.position.y): h for h in gs.map.hexes}
        row = {k: 0 for k in totals if k != "maps"}
        for pos, h in by_pos.items():
            code = strip_start_position(codes.get(pos, "") or "")
            _base, _, overlay = code.partition("^")
            row["playable_hexes"] += 1
            if overlay.startswith("F"):
                row["forest_overlay"] += 1
                tid = _first_terrain_id(h.terrain_types)
                if tid == Terrain.FLAT.value:
                    row["forest_overlay_encoded_flat"] += 1
                if not (int(getattr(h, "terrain_mask", 0)) >> Terrain.FOREST.value) & 1:
                    row["forest_overlay_mask_without_forest"] += 1
                if tid != Terrain.FOREST.value:
                    row["forest_overlay_not_encoded_forest"] += 1
            if overlay.startswith("V"):
                row["village_overlay"] += 1
            if _base.startswith("Wo"):
                row["deep_water_base"] += 1
            for ability, key in (("ambush", "ambush_cover"), ("concealment", "concealment_cover"),
                                 ("submerge", "submerge_cover")):
                if hides_cover(code, ability):
                    row[key] += 1
        per_map[sid] = row
        totals["maps"] += 1
        for k, v in row.items():
            totals[k] += v
    return {"totals": totals, "per_map": per_map}


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, default=None, help="write the census here as JSON")
    args = ap.parse_args(argv)
    out = census(LADDER_SCENARIO_IDS)
    t = out["totals"]
    print(f"{t['maps']} maps, {t['playable_hexes']} playable hexes")
    print(f"forest overlay {t['forest_overlay']} (ambush cover {t['ambush_cover']}; "
          f"encoded FLAT {t['forest_overlay_encoded_flat']}, "
          f"not encoded FOREST {t['forest_overlay_not_encoded_forest']})")
    print(f"village overlay {t['village_overlay']} (concealment cover {t['concealment_cover']})")
    print(f"deep-water base {t['deep_water_base']} (submerge cover {t['submerge_cover']})")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print("wrote", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
