#!/usr/bin/env python3
"""How far the simulator's vision disc is from the engine's vision rule,
measured on corpus positions.

At every decision of the acting side (each move, attack, recruit and
end_turn) in a sample of fogged corpus games, three sets of hexes:

- `disc`: what `visibility.visible_hexes_for` returns, a disc of radius
  max_moves around each own unit's current hex;
- `now`: the engine's rule (`visibility.unit_vision`) from the units'
  current hexes;
- `turn`: the engine's rule accumulated over the turn, as the engine
  keeps fog cleared during a side's turn: the units' vision at the start
  of the turn, from every hex a unit entered while moving, and around
  every recruit (src/actions/move.cpp:972-973, create.cpp:696-698,
  play_controller.cpp:525 and 589-590 at the 1.18.4 tag).

Reported per map, with denominators: hexes in each set difference per
decision, as counts and as a share of the map's hexes; enemy units on
the board and how many each rule shows (hiding abilities ignored, the
same for every rule).

    python tools/analysis/vision_rule_census.py [--per-map 3] [--out FILE.json]
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.replay_dataset import (_apply_command, _build_initial_gamestate,  # noqa: E402
                                  _setup_scenario_events)
from wesnoth_ai.visibility import unit_vision, visible_hexes_for  # noqa: E402

CORPUS = ROOT / "replays_dataset_imitation"
DECISIONS = ("move", "attack", "recruit", "end_turn")
Hex = Tuple[int, int]


def _map_key(file_name: str) -> str:
    """The map part of a corpus file name: `2024-03-22_2p__Caves_of_the_
    Basilisk_Turn_14_(335).json.gz` -> `2p__Caves_of_the_Basilisk`."""
    stem = file_name.split("_", 1)[1] if "_" in file_name else file_name
    return stem.split("_Turn_")[0]


def sample_games(per_map: int) -> List[Path]:
    """Fogged training-split games (the manifest's `fog` flag, never the
    holdout), the first `per_map` per map in file-name order."""
    rows = [json.loads(line) for line in (CORPUS / "manifest.jsonl").open(encoding="utf-8")]
    by_map: Dict[str, List[Path]] = collections.defaultdict(list)
    for row in sorted(rows, key=lambda r: r["file"]):
        if row.get("holdout") or not row.get("fog"):
            continue
        path = CORPUS / row["file"]
        key = _map_key(row["file"])
        if path.exists() and len(by_map[key]) < per_map:
            by_map[key].append(path)
    return [p for paths in by_map.values() for p in paths]


class Vision:
    """The engine rule per unit, cached by what it depends on."""

    def __init__(self, gs):
        self.gs = gs
        self.cache: Dict[tuple, Set[Hex]] = {}

    def of(self, unit, at: Hex = None) -> Set[Hex]:
        at = at if at is not None else (unit.position.x, unit.position.y)
        key = (unit.name, int(unit.max_moves), "slowed" in (unit.statuses or set()), at,
               getattr(self.gs.global_info, "_terrain_epoch", None))
        got = self.cache.get(key)
        if got is None:
            got = self.cache[key] = unit_vision(self.gs, unit, at=at)
        return got

    def side_now(self, side: int) -> Set[Hex]:
        out: Set[Hex] = set()
        for u in self.gs.map.units:
            if u.side == side:
                out |= self.of(u)
        return out


def walked_hexes(cmd: list, mover) -> List[Hex]:
    """The hexes a move entered, up to where the unit stopped."""
    path = list(zip(cmd[1], cmd[2]))
    end = (mover.position.x, mover.position.y)
    if end in path:
        path = path[:path.index(end) + 1]
    return path[1:]


def census_game(path: Path, per_map: Dict[str, collections.Counter]) -> None:
    data = json.load(gzip.open(str(path), "rt", encoding="utf-8"))
    sid = data.get("scenario_id", "?")
    gs = _build_initial_gamestate(data)
    _setup_scenario_events(gs, sid)
    vision = Vision(gs)
    board = len(gs.map.hexes)
    c = per_map[sid]
    c["games"] += 1
    c["board_hexes"] = board
    side, turn_seen = None, set()
    for cmd in data["commands"]:
        kind = cmd[0]
        if kind == "init_side":
            _apply_command(gs, cmd)
            side = cmd[1]
            turn_seen = vision.side_now(side)
            continue
        if kind in DECISIONS and side in (1, 2):
            disc = visible_hexes_for(gs, side)
            now = vision.side_now(side)
            enemies = [(u.position.x, u.position.y) for u in gs.map.units
                       if u.side not in (side,) and u.side in (1, 2)]
            c["decisions"] += 1
            c["disc"] += len(disc)
            c["now"] += len(now)
            c["turn"] += len(turn_seen)
            c["now_not_disc"] += len(now - disc)
            c["disc_not_now"] += len(disc - now)
            c["turn_not_disc"] += len(turn_seen - disc)
            c["disc_not_turn"] += len(disc - turn_seen)
            c["enemies"] += len(enemies)
            c["enemies_disc"] += sum(1 for e in enemies if e in disc)
            c["enemies_turn"] += sum(1 for e in enemies if e in turn_seen)
            c["enemies_turn_not_disc"] += sum(1 for e in enemies if e in turn_seen and e not in disc)
            c["enemies_disc_not_turn"] += sum(1 for e in enemies if e in disc and e not in turn_seen)
            c["decisions_differing"] += int(turn_seen != disc)
        if kind == "move":
            start = (cmd[1][0], cmd[2][0])
            mover = next((u for u in gs.map.units
                          if (u.position.x, u.position.y) == start), None)
            _apply_command(gs, cmd)
            if mover is not None and side is not None:
                moved = next((u for u in gs.map.units if u.id == mover.id), None)
                if moved is not None:
                    for h in walked_hexes(cmd, moved):
                        turn_seen |= vision.of(moved, at=h)
            continue
        if kind == "recruit":
            _apply_command(gs, cmd)
            x, y = cmd[2], cmd[3]
            new = next((u for u in gs.map.units if (u.position.x, u.position.y) == (x, y)), None)
            if new is not None:
                turn_seen |= vision.of(new)
            continue
        _apply_command(gs, cmd)


def report(per_map: Dict[str, collections.Counter]) -> str:
    lines = ["| map | games | decisions | board hexes | disc | engine, now | engine, turn | "
             "turn not disc | disc not turn | decisions differing | enemies on board | "
             "shown by disc | shown by turn | only turn | only disc |",
             "|" + "---|" * 15]
    total = collections.Counter()
    for sid, c in sorted(per_map.items()):
        total.update({k: v for k, v in c.items() if k != "board_hexes"})
        d = c["decisions"] or 1
        lines.append(
            f"| {sid.replace('multiplayer_', '')} | {c['games']} | {c['decisions']} | {c['board_hexes']} | "
            f"{c['disc'] / d:.1f} | {c['now'] / d:.1f} | {c['turn'] / d:.1f} | "
            f"{c['turn_not_disc'] / d:.1f} | {c['disc_not_turn'] / d:.1f} | "
            f"{c['decisions_differing']}/{c['decisions']} | {c['enemies'] / d:.2f} | "
            f"{c['enemies_disc'] / d:.2f} | {c['enemies_turn'] / d:.2f} | "
            f"{c['enemies_turn_not_disc'] / d:.2f} | {c['enemies_disc_not_turn'] / d:.2f} |")
    d = total["decisions"] or 1
    lines.append(
        f"| all | {total['games']} | {total['decisions']} | | {total['disc'] / d:.1f} | "
        f"{total['now'] / d:.1f} | {total['turn'] / d:.1f} | {total['turn_not_disc'] / d:.1f} | "
        f"{total['disc_not_turn'] / d:.1f} | {total['decisions_differing']}/{total['decisions']} | "
        f"{total['enemies'] / d:.2f} | {total['enemies_disc'] / d:.2f} | {total['enemies_turn'] / d:.2f} | "
        f"{total['enemies_turn_not_disc'] / d:.2f} | {total['enemies_disc_not_turn'] / d:.2f} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--per-map", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    t0 = time.time()
    games = sample_games(args.per_map)
    per_map: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for path in games:
        census_game(path, per_map)
    print(report(per_map))
    print(f"\n{len(games)} games, {time.time() - t0:.0f} s. Per-decision means; "
          "'enemies' counts side-1/2 enemy units on the board.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({k: dict(v) for k, v in per_map.items()}, indent=1),
                            encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
