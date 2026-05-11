"""Build-time: extract the `[scenario]` block from one real Wesnoth save
per ladder scenario, strip the per-game `[side]` sub-blocks, and write
the result to `tools/templates/scenarios/<scenario_id>.wml`.

Why this step exists: Wesnoth scenarios use a heavy macro layer
(`{PLACE_IMAGE ...}`, `{FLASH_WHITE ...}`, `{QUAKE ...}`,
`{DEFAULT_SCHEDULE}`, etc.) defined under `wesnoth_src/data/core/macros/`.
The save-file format (which `tools/sim_to_replay.export_replay_from_scratch`
needs to emit) requires those macros to be FULLY EXPANDED. The
canonical way to expand them is to ask Wesnoth itself; one round-trip
through `wesnoth --multiplayer ...` produces a save file with all
macros pre-expanded. Replays in `replays_raw/` are exactly that --
real Wesnoth output, fully expanded.

This script scans `replays_raw/` once, picks the first bz2 matching
each ladder scenario id, extracts and parks the `[scenario]` block in
the repo so the runtime emitter has a known-good template per map.
The output goes under `tools/templates/scenarios/` and IS committed
to the repo -- runtime is then template-driven with no
`replays_raw/` dependency. Re-run this script after a wesnoth_src
version bump.

Strips: every top-level `[side]...[/side]` sub-block. The runtime
emitter renders fresh `[side]` blocks per game from sim state.

Keeps: `description`, `experience_modifier`, `id`, `map_data`,
`map_file`, `name`, `objectives`, `random_start_time`, `turns`,
plus all `[time]` / `[music]` / `[event]` / `[item]` /
`[default_faction]` etc. sub-blocks. These carry the scenario's
ToD schedule, music playlist, terrain-morph events, pre-placed
scenery, halos -- everything the from-scratch sim_to_replay path
needs to make Wesnoth playback faithfully reproduce the sim's
scenario.

Usage:
    python tools/extract_scenario_templates.py
    # writes tools/templates/scenarios/<scenario_id>.wml per ladder map
"""
from __future__ import annotations

import bz2
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.scenario_pool import LADDER_SCENARIO_IDS

log = logging.getLogger("extract_scenario_templates")

REPLAYS_RAW = _ROOT / "replays_raw"
OUT_DIR = _ROOT / "tools" / "templates" / "scenarios"


# Filename-token override for scenarios where the in-replay short name
# differs from the scenario_id's tail (Caves of the Basilisk's
# scenario_id is `multiplayer_Basilisk` but the filename uses
# `Caves_of_the_Basilisk`).
_FILENAME_TOKEN = {
    "multiplayer_Basilisk":                  "Caves_of_the_Basilisk",
    "multiplayer_elensefar_courtyard":       "Elensefar_Courtyard",
    "multiplayer_thousand_stings_garrison":  "Thousand_Stings_Garrison",
}


def _filename_token(scenario_id: str) -> str:
    return _FILENAME_TOKEN.get(scenario_id,
                               scenario_id.replace("multiplayer_", ""))


def _find_source_bz2(scenario_id: str) -> Optional[Path]:
    """Pick the first `.bz2` under `replays_raw/` whose filename
    contains the scenario's token AND whose contents confirm the
    `mp_scenario=<id>` / `id=<id>` tag. Returns None if no match."""
    token = _filename_token(scenario_id)
    for p in REPLAYS_RAW.rglob("*.bz2"):
        if token not in p.stem:
            continue
        try:
            with bz2.open(p, "rt", errors="replace") as f:
                head = f.read(4000)
        except OSError:
            continue
        if scenario_id in head:
            return p
    return None


def _extract_scenario_block(text: str) -> Optional[str]:
    """Pull the first `[scenario]...[/scenario]` block from `text`.
    Wesnoth saves have a single top-level `[scenario]`; we capture
    everything between (inclusive)."""
    start = text.find("[scenario]")
    if start < 0:
        return None
    # Need to match the corresponding `[/scenario]`. WML doesn't
    # nest `[scenario]` so the first `[/scenario]` AFTER `[scenario]`
    # is the right close.
    end_idx = text.find("[/scenario]", start)
    if end_idx < 0:
        return None
    return text[start:end_idx + len("[/scenario]")]


def _strip_player_side_blocks(scenario_block: str) -> str:
    """Remove `[side]...[/side]` sub-blocks for PLAYER sides (1 and 2)
    only. The runtime emitter inserts fresh per-game [side] blocks
    for sides 1+2 with sim-chosen factions/leaders/gold; replacing
    those is the whole point.

    SCENERY SIDES (side 3+) MUST be preserved. Several ladder maps
    use a neutral third side to hold pre-placed petrified-unit
    statues:
      - Caves of the Basilisk: 15 petrified victims on side 3
      - Thousand Stings Garrison: 66 frozen scorpions on side 3
      - Sullas Ruins: 5 stone-statue mages on side 3
    These units are scenery -- they don't move, attack, or
    participate in gameplay -- but they MUST be on the map for the
    scenario to render correctly. If we stripped them, the maps
    would load without their iconic decorations and (more
    importantly) the playback engine would let units walk through
    hex positions the statues are supposed to block.
    """
    def _is_player_side(m: "re.Match") -> bool:
        body = m.group(0)
        sn = re.search(r'^\s*side\s*=\s*"?(\d+)"?', body, re.MULTILINE)
        if sn is None:
            return False
        return int(sn.group(1)) in (1, 2)

    out: list = []
    last = 0
    for m in re.finditer(r"\[side\][\s\S]*?\[/side\]\s*", scenario_block):
        if _is_player_side(m):
            out.append(scenario_block[last:m.start()])
            last = m.end()
    out.append(scenario_block[last:])
    return "".join(out)


# Back-compat alias for any older test imports.
_strip_side_blocks = _strip_player_side_blocks


def _extract_one(scenario_id: str) -> Dict[str, str]:
    """Extract the [scenario] template for `scenario_id`. Returns a
    dict with `status`, `source`, and (on success) `template` fields."""
    src = _find_source_bz2(scenario_id)
    if src is None:
        return {"status": "no-source", "source": ""}
    try:
        with bz2.open(src, "rt", errors="replace") as f:
            text = f.read()
    except OSError as e:
        return {"status": f"read-error: {e}", "source": str(src)}
    block = _extract_scenario_block(text)
    if block is None:
        return {"status": "no-scenario", "source": src.name}
    stripped = _strip_player_side_blocks(block)
    return {"status": "ok", "source": src.name, "template": stripped}


def main(argv) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, str] = {}
    for sid in LADDER_SCENARIO_IDS:
        out = _extract_one(sid)
        if out["status"] != "ok":
            log.error(f"{sid}: {out['status']} ({out.get('source', '')})")
            summary[sid] = out["status"]
            continue
        out_path = OUT_DIR / f"{sid}.wml"
        # Prepend a brief header so the human reader knows where it
        # came from + how to regenerate.
        header = (
            "# Per-scenario [scenario] template for from-scratch save\n"
            "# WML emission. Extracted from a real Wesnoth save (after\n"
            "# the engine expanded macros + scenario events). All\n"
            f"# [side] sub-blocks stripped at extraction; the runtime\n"
            f"# emitter (tools/sim_to_replay.build_save_wml) inserts\n"
            f"# fresh [side] blocks per game from sim state.\n"
            f"#\n"
            f"# Source: replays_raw/.../{out['source']}\n"
            f"# Regenerate via: python tools/extract_scenario_templates.py\n"
        )
        out_path.write_text(header + out["template"], encoding="utf-8")
        log.info(f"{sid}: wrote {out_path.relative_to(_ROOT)} "
                 f"({len(out['template'])} chars)")
        summary[sid] = "ok"

    ok = sum(1 for v in summary.values() if v == "ok")
    log.info(f"summary: {ok}/{len(LADDER_SCENARIO_IDS)} extracted")
    return 0 if ok == len(LADDER_SCENARIO_IDS) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
