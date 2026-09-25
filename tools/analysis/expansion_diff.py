#!/usr/bin/env python3
"""Our macro expansion of a scenario against the game's own.

The generation path expands a scenario `.cfg` with our expander
(`tools/scenario_events.load_scenario_wml`), while the committed
templates under `tools/templates/scenarios/` hold the same scenarios
as Wesnoth's own preprocessor expanded them. Two renderings of one
file, and nothing compared them, which is how `{DEFAULT_SCHEDULE}`
came to be deleted by a hardcoded cosmetic list: our expansion emitted
zero `[time]` blocks where the game's emits six, and no test noticed.

This is the only check that covers the EXPAND stage. A census of the
expanded tree cannot see what the expander dropped, so a classifier
over our own output is blind exactly where the bugs are.

    python tools/analysis/expansion_diff.py [--out FILE] [--verbose]

Every difference is reported as a CLUSTER: a (kind, detail) pair with
the scenarios it affects, so a systematic gap reads as one line rather
than 28. The clusters are the artifact; `tests/test_expansion_diff.py`
fails when a new one appears.

What the comparison deliberately excludes, because the template is the
preprocessor's output PLUS our builder's edits
(`tools/build_scenario_templates.py`):

  * player sides 1 and 2, stripped at build;
  * the attributes the builder injects when the cfg omits them;
  * the two era events it appends;
  * presentation tags no consumer reads.

Everything else is compared, and a difference means our expansion and
the game's disagree about the game.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from wesnoth_ai.paths import REPO_ROOT, SCENARIO_TEMPLATES_DIR  # noqa: E402
from tools.replay_extract import parse_wml  # noqa: E402
from tools.scenario_events import load_scenario_wml  # noqa: E402
from tools.wml_state import (map_starting_positions, resolve_map_file,  # noqa: E402
                             split_map_grid)
from tools.scenario_pool import (LADDER_SCENARIO_IDS,  # noqa: E402
                                 MINI_MAP_SCENARIO_IDS)

TEMPLATES = SCENARIO_TEMPLATES_DIR
POOL = list(LADDER_SCENARIO_IDS) + list(MINI_MAP_SCENARIO_IDS)

# Presentation only: nothing downstream of the scenario reader looks at
# these, and the game's expansion is full of them.
COSMETIC_TAGS = frozenset({
    "music", "item", "terrain_graphics", "label", "image", "sound",
    "scroll", "screen_fade", "delay", "objectives", "message", "story",
    "part", "objective", "note", "color_palette", "color_range",
})
# Injected by tools/build_scenario_templates.py when the cfg omits
# them, so their presence in the template says nothing about the game.
INJECTED_ATTRS = frozenset({
    "turns", "experience_modifier", "has_mod_events", "loaded_resources",
    "objectives", "description", "name",
})
# Appended by the builder to mirror what the engine injects at load.
ERA_EVENT_NAMES = frozenset({"time over"})
# Compared as a resolved grid by `_map_grid`, never as text.
MAP_ATTRS = frozenset({"map_data", "map_file"})


def _side_number(node) -> Optional[int]:
    raw = (node.attrs.get("side", "") or "").strip().strip('"')
    return int(raw) if raw.isdigit() else None


def _scenario_block(root):
    if root is None:
        return None
    return root.first("multiplayer") or root.first("scenario")


def _events(block) -> Dict[str, int]:
    """Event names to their count, minus the era events the builder
    appends and the purely cosmetic ones (a prestart holding nothing
    but music)."""
    out: collections.Counter = collections.Counter()
    for event in block.all("event"):
        name = (event.attrs.get("name", "") or "?").strip().strip('"')
        if name in ERA_EVENT_NAMES:
            continue
        kinds = {c.tag for c in event.children} - COSMETIC_TAGS
        if not kinds:
            continue            # music-only prestart: presentation
        out[name] += 1
    return dict(out)


def _schedule(block) -> List[Tuple[str, str]]:
    return [((t.attrs.get("id", "?") or "?").strip().strip('"'),
             (t.attrs.get("lawful_bonus", "") or "0").strip().strip('"'))
            for t in block.all("time")]


def _time_areas(block) -> int:
    return len(block.all("time_area"))


def _cells(split) -> Tuple[List[List[str]], int]:
    """A grid as its consumers read it: rows of stripped cells. The
    inline form loses the last row's trailing spaces to the closing
    quote, so comparing raw row text reported three mini maps as
    differing when every cell was identical."""
    rows, border = split
    return [[c.strip() for c in row.split(",")] for row in rows], border


def _map_text(block) -> Optional[str]:
    """The map the scenario names, however it names it. A scenario
    points at a `.map` file and the engine's expansion inlines the grid,
    so the ATTRIBUTE always differs and the GRID is what has to match.
    Comparing the spelling would have reported a difference on 21 of 28
    scenarios and checked nothing."""
    data = (block.attrs.get("map_data", "") or "").strip()
    if data and not data.startswith("{"):
        return data
    path = resolve_map_file(REPO_ROOT, map_file=block.attrs.get("map_file", "") or "",
                            map_data=data)
    if path is None:
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def _scenery_sides(block) -> Dict[int, Dict[str, str]]:
    """Sides 3 and up, which both renderings keep. Player sides are
    stripped from the template, so they cannot be compared here; that
    is a gap this tool reports rather than hides."""
    out: Dict[int, Dict[str, str]] = {}
    for side in block.all("side"):
        number = _side_number(side)
        if number is None or number < 3:
            continue
        out[number] = {k: (v or "").strip().strip('"')
                       for k, v in side.attrs.items()
                       if k not in INJECTED_ATTRS}
        out[number]["#units"] = str(len(side.all("unit")))
        out[number]["#villages"] = str(len(side.all("village")))
    return out


def compare(scenario_id: str) -> List[Tuple[str, str]]:
    """(kind, detail) for every way our expansion differs from the
    game's on this scenario."""
    ours = _scenario_block(load_scenario_wml(scenario_id))
    template_path = TEMPLATES / f"{scenario_id}.wml"
    if ours is None:
        return [("unparsed", "our expander produced no scenario block")]
    if not template_path.is_file():
        return [("no template", str(template_path.name))]
    theirs = _scenario_block(
        parse_wml(template_path.read_text(encoding="utf-8", errors="replace")))
    if theirs is None:
        return [("unparsed", "the template has no scenario block")]

    out: List[Tuple[str, str]] = []
    ours_sched, theirs_sched = _schedule(ours), _schedule(theirs)
    if ours_sched != theirs_sched:
        out.append(("schedule",
                    f"ours {len(ours_sched)} [time] blocks, "
                    f"the game's {len(theirs_sched)}"))
    if _time_areas(ours) != _time_areas(theirs):
        out.append(("time areas",
                    f"ours {_time_areas(ours)}, the game's {_time_areas(theirs)}"))

    ours_ev, theirs_ev = _events(ours), _events(theirs)
    for name in sorted(set(ours_ev) | set(theirs_ev)):
        if ours_ev.get(name, 0) != theirs_ev.get(name, 0):
            out.append(("event count",
                        f"{name}: ours {ours_ev.get(name, 0)}, "
                        f"the game's {theirs_ev.get(name, 0)}"))

    ours_sides, theirs_sides = _scenery_sides(ours), _scenery_sides(theirs)
    if set(ours_sides) != set(theirs_sides):
        out.append(("scenery sides",
                    f"ours {sorted(ours_sides)}, the game's {sorted(theirs_sides)}"))
    for number in sorted(set(ours_sides) & set(theirs_sides)):
        a, b = ours_sides[number], theirs_sides[number]
        for key in sorted(set(a) | set(b)):
            if a.get(key) != b.get(key):
                out.append(("scenery side attr",
                            f"side {number} {key}: ours {a.get(key)!r}, "
                            f"the game's {b.get(key)!r}"))

    ours_map, theirs_map = _map_text(ours), _map_text(theirs)
    if ours_map is None or theirs_map is None:
        out.append(("map", f"unresolved: ours {ours_map is not None}, "
                           f"the game's {theirs_map is not None}"))
    else:
        ours_grid, ours_border = _cells(split_map_grid(ours_map))
        theirs_grid, theirs_border = _cells(split_map_grid(theirs_map))
        if ours_border != theirs_border:
            out.append(("map", f"border_size: ours {ours_border}, "
                               f"the game's {theirs_border}"))
        if ours_grid != theirs_grid:
            apart = sum(1 for a, b in zip(ours_grid, theirs_grid) if a != b)
            out.append(("map", f"the grids differ: {len(ours_grid)} rows "
                               f"against {len(theirs_grid)}, {apart} apart"))
        else:
            a = map_starting_positions(ours_map)
            b = map_starting_positions(theirs_map)
            if a != b:
                out.append(("map", f"starting positions: ours {a}, "
                                   f"the game's {b}"))

    for key in sorted((set(ours.attrs) | set(theirs.attrs))
                      - INJECTED_ATTRS - MAP_ATTRS):
        a = (ours.attrs.get(key, "") or "").strip().strip('"')
        b = (theirs.attrs.get(key, "") or "").strip().strip('"')
        if a != b:
            out.append(("scenario attr", f"{key}: ours {a!r}, the game's {b!r}"))
    return out


def clusters(scenarios: Sequence[str]) -> Dict[str, Dict]:
    """Differences grouped by (kind, detail), each with its scenarios."""
    grouped: Dict[Tuple[str, str], List[str]] = collections.defaultdict(list)
    for scenario_id in scenarios:
        for kind, detail in compare(scenario_id):
            grouped[(kind, detail)].append(scenario_id)
    return {f"{kind} | {detail}": {"kind": kind, "detail": detail,
                                   "scenarios": sorted(ids), "n": len(ids)}
            for (kind, detail), ids in sorted(grouped.items())}


def render(found: Dict[str, Dict], n_scenarios: int) -> str:
    lines = [f"{len(found)} divergence cluster(s) over {n_scenarios} scenarios"]
    for key, row in sorted(found.items(), key=lambda kv: (-kv[1]["n"], kv[0])):
        lines.append(f"\n  [{row['n']}/{n_scenarios}] {row['kind']}: {row['detail']}")
        shown = row["scenarios"][:4]
        lines.append("      " + ", ".join(shown)
                     + (" ..." if len(row["scenarios"]) > len(shown) else ""))
    return "\n".join(lines)


EXPECTED = REPO_ROOT / "tests" / "data" / "expansion_diff_expected.json"


def expected_clusters() -> Dict[str, Dict]:
    """The committed expectation: every accepted divergence with the
    reason it is accepted. `tests/test_expansion_diff.py` compares the
    live clusters against it."""
    return json.loads(EXPECTED.read_text(encoding="utf-8"))["clusters"]


def _write_expected(found: Dict[str, Dict]) -> None:
    """Refresh the expectation, carrying each surviving entry's
    classification over. A new entry is written UNCLASSIFIED so it
    cannot be committed without someone saying what it is."""
    doc = json.loads(EXPECTED.read_text(encoding="utf-8"))
    old = doc["clusters"]
    doc["clusters"] = {
        key: {**old.get(key, {"classification": "UNCLASSIFIED",
                              "stands_in": "", "why": ""}),
              "scenarios": row["scenarios"]}
        for key, row in found.items()}
    EXPECTED.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    fresh = [k for k in found if k not in old]
    print(f"\nwrote {EXPECTED}"
          + (f"\n  {len(fresh)} NEW, written UNCLASSIFIED:" if fresh else ""))
    for key in fresh:
        print(f"    {key}")


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--write-expected", action="store_true",
                    help="refresh tests/data/expansion_diff_expected.json")
    ap.add_argument("--scenario", action="append", default=None,
                    help="one scenario id; repeatable (default: the pool)")
    args = ap.parse_args(argv)
    scenarios = args.scenario or POOL
    found = clusters(scenarios)
    print(render(found, len(scenarios)))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"scenarios": list(scenarios), "clusters": found}, indent=1),
            encoding="utf-8")
        print(f"\nwrote {args.out}")
    if args.write_expected:
        if args.scenario:
            print("\nrefused: --write-expected needs the whole pool")
            return 2
        _write_expected(found)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
