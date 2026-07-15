"""Static validity checks for sim-exported Wesnoth replays.

Layer 1 of the replay-validation harness (2026-07-06, after OOS
errors — off-by-one unit positions — were observed in real-Wesnoth
playback of a self-play export; root cause: map parsers ignored the
`border_size=`/`usage=` header lines that add-on maps carry, shifting
the sim's whole coordinate frame vs Wesnoth's).

The checks replay the command stream against the embedded map's
GROUND TRUTH (keep markers, bounds, Wesnoth's 1-indexed floor) with a
lightweight unit-occupancy tracker — deliberately NOT our full sim,
whose coordinate conventions are exactly what's under test:

  1. Leaders spawn on the map's `<N> K*` keep markers; every
     [recruit]/[recall]'s [from] hex must BE the acting side's keep.
  2. Every [move]'s source hex must be occupied by a tracked unit
     (the first desync symptom is "move from an empty hex").
  3. Every [attack]'s [source] must be occupied.
  4. All coordinates are >= 1 (Wesnoth is 1-indexed) and in bounds.

Deaths/level-ups are not modeled, so only SOURCE-occupancy is
asserted (a stale corpse can't cause a false "source missing").

Usage:
    python tools/validate_replay.py REPLAY.bz2 [...]
or  from tools.validate_replay import validate_replay
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.replay_dataset import split_map_grid
from tools.replay_extract import parse_replay_file
from tools.sim_to_replay import _scrape_map_keep_positions


def _map_dims(map_data: str) -> Tuple[int, int]:
    """Playable (width, height) in WML coords, headers/border aware."""
    rows, border = split_map_grid(map_data)
    if not rows:
        return (0, 0)
    width = max(len(r.split(",")) for r in rows) - 2 * border
    height = len(rows) - 2 * border
    return (width, height)


def _ints(v: str) -> List[int]:
    out = []
    for part in str(v).split(","):
        part = part.strip()
        if part.lstrip("-").isdigit():
            out.append(int(part))
    return out


def validate_replay(path: Path) -> List[str]:
    """Return a list of violation strings; empty = valid."""
    problems: List[str] = []
    root = parse_replay_file(path)
    scn = root.first("scenario")
    if scn is None:
        return [f"{path.name}: no [scenario] block"]
    map_data = scn.attrs.get("map_data", "")
    if not map_data:
        return [f"{path.name}: no map_data in [scenario]"]
    width, height = _map_dims(map_data)
    keeps = _scrape_map_keep_positions(map_data)   # side -> WML (x,y)

    def in_bounds(x: int, y: int) -> bool:
        return 1 <= x <= width and 1 <= y <= height

    # Occupancy tracker: leaders spawn on keeps; pre-placed [unit]s
    # from the [side] blocks are added too.
    occupied: Set[Tuple[int, int]] = set(keeps.values())
    for side_node in scn.all("side"):
        for u in side_node.all("unit"):
            xs, ys = _ints(u.attrs.get("x", "")), _ints(u.attrs.get("y", ""))
            if xs and ys:
                occupied.add((xs[0], ys[0]))
    # Event-spawned units (Mini_Maps tentacles, Hornshark faction
    # starters): Wesnoth playback re-fires the scenario's events from
    # local game data, so those units exist at their spawn hexes even
    # though the composed save carries no [unit] blocks for them.
    # Mirror that here or side-3 attacks read "from empty hex"
    # (2026-07-14).
    scen_id = scn.attrs.get("id", "").strip().strip('"')
    if scen_id:
        try:
            from tools.scenario_events import load_scenario_wml
            root_wml = load_scenario_wml(scen_id)
            mp_node = (root_wml.first("multiplayer")
                       or root_wml.first("scenario")
                       if root_wml is not None else None)
            if mp_node is not None:
                for ev in mp_node.all("event"):
                    # RECURSIVE descent: Hornshark's faction starters
                    # spawn via [event]->[switch]->[case]->[unit], and
                    # WMLNode.all() only returns direct children --
                    # the non-recursive scan missed them and produced
                    # "attack from empty hex" false positives
                    # (independent review 2026-07-14, fixed
                    # 2026-07-15). Over-collection risk ([unit]
                    # inside [filter] etc.) is acceptable: occupancy
                    # is a permissive heuristic here.
                    stack = [ev]
                    while stack:
                        node = stack.pop()
                        for child in node.children:
                            if child.tag == "unit":
                                xs = _ints(child.attrs.get("x", ""))
                                ys = _ints(child.attrs.get("y", ""))
                                if xs and ys:
                                    occupied.add((xs[0], ys[0]))
                            stack.append(child)
        except Exception:                             # noqa: BLE001
            pass

    replay = root.children[-1]                     # trailing [replay]
    n_cmd = 0
    for cmd in replay.all("command"):
        n_cmd += 1
        raw_side = cmd.attrs.get("from_side", "0")
        side = int(raw_side) if str(raw_side).isdigit() else 0
        # from_side="server" marks engine-injected commands
        # ([end_turn] relays etc.) — no coordinates to check there.
        for ch in cmd.children:
            if ch.tag in ("recruit", "recall"):
                x, y = _ints(ch.attrs.get("x", "0"))[0], \
                       _ints(ch.attrs.get("y", "0"))[0]
                frm = ch.first("from")
                fx = _ints(frm.attrs.get("x", "0"))[0] if frm else 0
                fy = _ints(frm.attrs.get("y", "0"))[0] if frm else 0
                if side in keeps and (fx, fy) != keeps[side]:
                    problems.append(
                        f"cmd#{n_cmd} [{ch.tag}] side {side} from "
                        f"({fx},{fy}) but the map's keep is "
                        f"{keeps[side]} — coordinate-frame mismatch")
                if not in_bounds(x, y):
                    problems.append(
                        f"cmd#{n_cmd} [{ch.tag}] target ({x},{y}) out "
                        f"of bounds {width}x{height}")
                occupied.add((x, y))
            elif ch.tag == "move":
                xs, ys = _ints(ch.attrs.get("x", "")), \
                         _ints(ch.attrs.get("y", ""))
                if not xs or not ys:
                    continue
                src, dst = (xs[0], ys[0]), (xs[-1], ys[-1])
                if src not in occupied:
                    problems.append(
                        f"cmd#{n_cmd} [move] from empty hex {src} — "
                        f"the OOS signature (unit isn't where the "
                        f"replay thinks)")
                for x, y in zip(xs, ys):
                    if not in_bounds(x, y):
                        problems.append(
                            f"cmd#{n_cmd} [move] path hex ({x},{y}) "
                            f"out of bounds {width}x{height}")
                occupied.discard(src)
                occupied.add(dst)
            elif ch.tag == "attack":
                src_node = ch.first("source")
                if src_node is not None:
                    sx = _ints(src_node.attrs.get("x", "0"))[0]
                    sy = _ints(src_node.attrs.get("y", "0"))[0]
                    if (sx, sy) not in occupied:
                        problems.append(
                            f"cmd#{n_cmd} [attack] from empty hex "
                            f"({sx},{sy})")
            if len(problems) > 50:
                problems.append("... (truncated)")
                return problems
    return problems


def main(argv) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    bad = 0
    for arg in argv[1:]:
        p = Path(arg)
        problems = validate_replay(p)
        if problems:
            bad += 1
            print(f"INVALID {p.name}: {len(problems)} problem(s)")
            for pr in problems[:20]:
                print(f"  - {pr}")
        else:
            print(f"ok {p.name}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
