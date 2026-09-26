#!/usr/bin/env python3
"""How much of the enemy the side to move cannot see at the benchmark
positions, where a search or a playout that starts from the true state
would see it all.

For each position of configs/bench_states.json (the turn-level gap
measured the first 60, the turn-value experiment all 200) the tool
rebuilds the side-turn boundary the way the measurements did
(`bench_states.reconstruct_boundary`) and counts, for the side to move:
the enemy's player units, those hidden from it under the current vision
rule (`visibility.units_visible_to`), their hit points, and whether the
enemy leader is among them. Findings: docs/hidden_information_20260926.md.

    python tools/analysis/hidden_at_boundary.py --out RECORD.jsonl
        [--first N] [--dataset DIR]
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bench_states import DEFAULT_DATASET, DEFAULT_MANIFEST, reconstruct_boundary
from wesnoth_ai.visibility import units_visible_to


def position_row(index: int, entry: dict, dataset: Path) -> dict:
    """The hidden-enemy counts at one benchmark position."""
    with gzip.open(dataset / entry["file"], "rt", encoding="utf-8") as f:
        data = json.load(f)
    gs, side = reconstruct_boundary(data, entry["cut_turn"])
    enemy = [u for u in gs.map.units if u.side in (1, 2) and u.side != side]
    visible = {u.id for u in units_visible_to(gs, side)}
    hidden = [u for u in enemy if u.id not in visible]
    return {"position": index, "file": entry["file"], "turn": gs.global_info.turn_number,
            "side": side, "fog": bool(getattr(gs.global_info, "_fog", True)),
            "enemy_units": len(enemy), "hidden_units": len(hidden),
            "enemy_hp": sum(u.current_hp for u in enemy),
            "hidden_hp": sum(u.current_hp for u in hidden),
            "leader_hidden": any(u.is_leader for u in hidden),
            "hidden_types": sorted(u.name for u in hidden)}


def summary(rows: list) -> dict:
    """Totals over a set of positions."""
    return {"positions": len(rows),
            "fog": sum(r["fog"] for r in rows),
            "enemy_units": sum(r["enemy_units"] for r in rows),
            "hidden_units": sum(r["hidden_units"] for r in rows),
            "positions_with_hidden": sum(r["hidden_units"] > 0 for r in rows),
            "leader_hidden": sum(r["leader_hidden"] for r in rows)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="one JSON row per position")
    ap.add_argument("--first", type=int, default=None, help="only the first N positions")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    states = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))["states"][:args.first]
    rows = [position_row(i, e, args.dataset) for i, e in enumerate(states)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print("first 60:", json.dumps(summary(rows[:60])))
    print(f"all {len(rows)}:", json.dumps(summary(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
