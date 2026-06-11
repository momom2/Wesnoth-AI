"""Dump our sim's per-unit state at every recruit and combat in a
replay, formatted for side-by-side comparison against Wesnoth's GUI.

The user can:
  1. Open the original .bz2 replay in Wesnoth's "Load Game -> Replays".
  2. Step through with the replay controls (Play / Next Side).
  3. Hover units to see their traits / max_hp / current_hp.
  4. Compare against the report this tool produces.

Where Wesnoth's hover tooltip differs from our report = bug.

Usage:
    python tools/dump_unit_states.py replays_dataset/<file>.json.gz
    python tools/dump_unit_states.py replays_dataset/<file>.json.gz --until 100

Example output (per recruit / combat):
    cmd[9] turn=1 RECRUIT side=2 Cavalryman at (38,5)
      seed=552d41fc -> traits={quick, strong} max_hp=33 max_xp=31
      [verify in Wesnoth: hover the just-recruited unit at (39,6)]

Coordinates are 1-indexed in the comparison line (Wesnoth's display).
The internal trace uses 0-indexed (matches our sim).

Dependencies: tools.replay_dataset.
Dependents: standalone CLI.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

from tools.replay_dataset import (
    _build_initial_gamestate, _apply_command, _setup_scenario_events,
    _stats_for,
)


def _fmt_traits(traits) -> str:
    return "{" + ", ".join(sorted(traits)) + "}"


def _to_wesnoth_xy(x: int, y: int) -> Tuple[int, int]:
    """Convert our 0-indexed to Wesnoth's 1-indexed display coords."""
    return x + 1, y + 1


def _find_unit_by_id(gs, uid: str):
    for u in gs.map.units:
        if u.id == uid:
            return u
    return None


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("replay", type=Path,
                    help="Extracted .json.gz from replay_extract.")
    ap.add_argument("--until", type=int, default=None,
                    help="Stop after this many commands.")
    args = ap.parse_args(argv[1:])

    with gzip.open(args.replay, "rt", encoding="utf-8") as f:
        data = json.load(f)
    gs = _build_initial_gamestate(data)
    _setup_scenario_events(gs, data.get("scenario_id", ""))

    print(f"=== {args.replay.name} ===")
    print(f"scenario: {data.get('scenario_id')}")
    print(f"factions: {data.get('factions')}")
    for s in data.get("starting_sides", []):
        print(f"  side {s['side']} ({s['faction']}) leader={s['leader_type']} "
              f"gold={s['gold']}")
    print()

    # Initial leaders.
    print("INITIAL UNITS (hover in Wesnoth at scenario start):")
    for u in sorted(gs.map.units, key=lambda x: (x.side, x.id)):
        wx, wy = _to_wesnoth_xy(u.position.x, u.position.y)
        print(f"  side {u.side} {u.name} u{u.id} at "
              f"({wx},{wy}) hp={u.current_hp}/{u.max_hp} "
              f"traits={_fmt_traits(u.traits)} xp={u.current_exp}/{u.max_exp}")
    print()

    cmds = data.get("commands", [])
    limit = args.until if args.until is not None else len(cmds)

    cur_turn = 1
    for i, cmd in enumerate(cmds[:limit]):
        kind = cmd[0]
        if kind == "init_side":
            side = cmd[1]
            print(f"--- cmd[{i}] init_side(side={side}) "
                  f"turn={gs.global_info.turn_number} ---")
        elif kind == "recruit":
            unit_type = cmd[1]
            tx, ty = cmd[2], cmd[3]
            seed = cmd[4] if len(cmd) > 4 else ""
            wx, wy = _to_wesnoth_xy(tx, ty)
            # Apply, then read the just-spawned unit.
            _apply_command(gs, cmd)
            spawned = None
            for u in gs.map.units:
                if u.position.x == tx and u.position.y == ty:
                    spawned = u
                    break
            print(f"cmd[{i}] turn={gs.global_info.turn_number} RECRUIT "
                  f"side={gs.global_info.current_side} {unit_type} "
                  f"at WML({wx},{wy}) seed={seed}")
            if spawned is not None:
                print(f"      -> u{spawned.id} hp={spawned.current_hp}/"
                      f"{spawned.max_hp} traits={_fmt_traits(spawned.traits)} "
                      f"xp=0/{spawned.max_exp}")
            continue
        elif kind == "attack":
            ax, ay, dx, dy = cmd[1], cmd[2], cmd[3], cmd[4]
            wax, way = _to_wesnoth_xy(ax, ay)
            wdx, wdy = _to_wesnoth_xy(dx, dy)
            # Pre-state.
            att_pre = None
            dfd_pre = None
            for u in gs.map.units:
                if u.position.x == ax and u.position.y == ay:
                    att_pre = (u.id, u.name, u.current_hp, u.max_hp,
                               sorted(u.traits))
                if u.position.x == dx and u.position.y == dy:
                    dfd_pre = (u.id, u.name, u.current_hp, u.max_hp,
                               sorted(u.traits))
            print(f"cmd[{i}] turn={gs.global_info.turn_number} ATTACK "
                  f"WML({wax},{way})->({wdx},{wdy}) weap={cmd[5]}/{cmd[6]} "
                  f"seed={cmd[7]}")
            if att_pre:
                print(f"  attacker u{att_pre[0]} {att_pre[1]} "
                      f"hp={att_pre[2]}/{att_pre[3]} traits={att_pre[4]}")
            if dfd_pre:
                print(f"  defender u{dfd_pre[0]} {dfd_pre[1]} "
                      f"hp={dfd_pre[2]}/{dfd_pre[3]} traits={dfd_pre[4]}")
            # Apply, then read post-state.
            _apply_command(gs, cmd)
            att_post = None
            dfd_post = None
            for u in gs.map.units:
                if att_pre and u.id == att_pre[0]:
                    att_post = (u.position.x, u.position.y, u.current_hp,
                                u.name)
                if dfd_pre and u.id == dfd_pre[0]:
                    dfd_post = (u.position.x, u.position.y, u.current_hp,
                                u.name)
            if att_pre:
                print(f"  -> attacker u{att_pre[0]}: " + (
                    f"hp={att_post[2]} (was {att_pre[2]}); "
                    f"name={att_post[3]}" if att_post
                    else "DEAD/REMOVED"))
            if dfd_pre:
                print(f"  -> defender u{dfd_pre[0]}: " + (
                    f"hp={dfd_post[2]} (was {dfd_pre[2]}); "
                    f"name={dfd_post[3]}" if dfd_post
                    else "DEAD/REMOVED"))
            continue
        elif kind == "move":
            xs, ys = cmd[1], cmd[2]
            sx, sy = xs[0], ys[0]
            tx, ty = xs[-1], ys[-1]
            wsx, wsy = _to_wesnoth_xy(sx, sy)
            wtx, wty = _to_wesnoth_xy(tx, ty)
            print(f"cmd[{i}] turn={gs.global_info.turn_number} MOVE "
                  f"WML({wsx},{wsy}) -> ({wtx},{wty}) side={cmd[3]}")
        elif kind == "end_turn":
            print(f"cmd[{i}] END_TURN turn={gs.global_info.turn_number}")
        _apply_command(gs, cmd)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
