#!/usr/bin/env python3
"""The lobby era: the default era with each unit type's experience at a
hosted game's 70%, for the live games started from the command line.

A hosted game writes the host's experience modifier into the scenario; a
command-line start does not, so a scenario that declares no modifier plays
at 100% (docs/wesnoth_rules.md, "A command-line `--multiplayer` start
skips the lobby's parameter writes"). The modifier only scales each unit
type's base experience, `max(1, (base * modifier + 50) / 100)`
(src/units/types.cpp:577-589), and an era's `[modify_unit_type]
set_experience=` replaces a type's base for the game it is played in
(src/saved_game.cpp:367-370, src/play_controller.cpp:179-181,
src/units/types.cpp:1377-1382). So an era that sets each type's base to
its value at 70% plays, at the command line's 100%, a hosted game's
experience. The values come from the simulator's unit table; the live
tool checks every type against the engine's own base at a game's first
decision (`experience_defects`).

    python tools/lobby_era.py      # rewrites add-ons/wesnoth_ai/eras/lobby_era.cfg
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.replay_dataset import _scaled_max_exp  # noqa: E402
from wesnoth_ai.paths import UNIT_STATS_PATH  # noqa: E402
from wesnoth_ai.rules.wml_state import MP_EXPERIENCE_MODIFIER  # noqa: E402

ERA_ID = "wesnoth_ai_lobby_era"
ERA_PATH = ROOT / "add-ons" / "wesnoth_ai" / "eras" / "lobby_era.cfg"

HEADER = f"""\
# lobby_era.cfg -- written by tools/lobby_era.py from unit_stats.json; do not edit.
# The default era with each unit type's experience at {MP_EXPERIENCE_MODIFIER}% of its base,
# what a hosted game plays; tools/live_vs_rca.py starts its games in it
# (tools/lobby_era.py's docstring). A variation takes its parent's value
# (src/units/types.cpp:1402-1416).

#ifdef MULTIPLAYER
[era]
    id={ERA_ID}
    name="Default, lobby experience"
    description="The default era with each unit type's experience at {MP_EXPERIENCE_MODIFIER}% of its base."

    {{ERA_DEFAULT}}
"""

FOOTER = """\
[/era]
#endif
"""


def base_experiences() -> Dict[str, int]:
    """Each unit type's base experience in the simulator's table. A
    variation (`Type:variation`) is not a type of its own in the engine
    and shares its parent's base, so only parent types are listed."""
    units = json.loads(UNIT_STATS_PATH.read_text(encoding="utf-8"))["units"]
    return {name: int(stats["experience"]) for name, stats in units.items() if ":" not in name}


def render(modifier: int = MP_EXPERIENCE_MODIFIER) -> str:
    """The era file, each type's base set to its experience at `modifier`."""
    blocks = [
        f"    [modify_unit_type]\n"
        f"        type={unit_type}\n"
        f"        set_experience={_scaled_max_exp(base, modifier)}\n"
        f"    [/modify_unit_type]\n"
        for unit_type, base in sorted(base_experiences().items())
    ]
    return HEADER + "\n" + "".join(blocks) + FOOTER


def experience_defects(types: Sequence[dict], modifier: int) -> Tuple[List[str], List[str]]:
    """(defects, unknown) for a game's unit types as the engine reports
    them (`experience_types`, lua/board_report.lua): the table's types
    whose units need other than `modifier`% of the engine's own base, and
    the engine's types the table does not hold, which a default-era game
    cannot field (add-on units)."""
    known = base_experiences()
    defects, unknown = [], []
    for entry in types:
        if entry["type"] not in known:
            unknown.append(entry["type"])
            continue
        base = entry.get("base")
        if base is None:
            defects.append(f"{entry['type']}: the engine declares no base experience")
            continue
        want = _scaled_max_exp(int(base), modifier)
        if entry["applied"] != want:
            defects.append(f"{entry['type']}: needs {entry['applied']} where a hosted game at "
                           f"{modifier}% needs {want} (engine base {base}, table base {known[entry['type']]})")
    return defects, sorted(unknown)


def main() -> int:
    ERA_PATH.parent.mkdir(parents=True, exist_ok=True)
    ERA_PATH.write_bytes(render().encode("utf-8"))
    print(f"{ERA_PATH.relative_to(ROOT)}: {len(base_experiences())} unit types at {MP_EXPERIENCE_MODIFIER}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
