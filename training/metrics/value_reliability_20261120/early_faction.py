#!/usr/bin/env python3
"""Whether the value estimate's pessimism at the start of a side's early
turns follows what the side knows of the enemy's faction (a candidate
cause raised in docs/audit_fixes_prereg_20261125.md, "The value estimate
since the switch").

A sample of the simple loop's games is replayed through the simulator; no
network is run. At the first decision of each of a current-network side's
first own turns: the logged estimate v, the side's result z, and the
factions the enemy can still be for that side, from the core's own record
of the enemy unit types the side has seen (the record the faction input
is cut by, wesnoth_ai/faction_posterior.py). A game set up with open picks
(before round 609) tells the side the enemy's faction from the start; its
positions are split the same way, by whether the seen types alone would
name the faction, as the control.

    python training/metrics/value_reliability_20261120/early_faction.py \
        --run C:/Users/momom2/wesnoth_runs/simple_loop --rounds 673 704 --every 6 --out early_faction_673_704.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.game_record import read_records, start_core  # noqa: E402
from tools.pg_positions import decision_slots  # noqa: E402
from wesnoth_ai.constants import DEFAULT_ERA_FACTIONS  # noqa: E402
from wesnoth_ai.faction_posterior import consistent_factions, informative_types, random_draws  # noqa: E402
from wesnoth_ai.pg_records import decisions_of  # noqa: E402

RECORD_SUFFIX = ".game.jsonl.gz"
END_TURN = 3
TURNS = 10            # own turns read, from the first
COLUMNS = ("game", "turn", "v", "z", "told", "possible", "types", "past")


def rows_of(path: Path, res: dict, game: int, past: bool) -> List[tuple]:
    rec = next(iter(read_records(path)))
    decisions = decisions_of(rec)
    commands = rec["commands"]
    slots = decision_slots(decisions, [(int(k), int(x), int(y)) for k, x, y in rec.get("rejections", ())],
                           len(commands))
    side_a = int(res["side_a"])
    z_a = {"win": 1.0, "loss": -1.0}.get(res.get("outcome_a") or "", 0.0)
    current = (side_a,) if not str(res["label_b"]).endswith("b") else (1, 2)
    faction = {side_a: res["faction_a"], 3 - side_a: res["faction_b"]}
    cs = start_core(rec, verify=False)
    era = list(cs.statics.get("era_factions") or DEFAULT_ERA_FACTIONS)
    chose_random = cs.statics.get("chose_random") or ()
    mode = str(cs.statics.get("random_faction_mode") or "Independent")
    applied = 0
    opens_turn = {1: True, 2: True}
    turn = {1: 0, 2: 0}
    out = []
    for d, (k, _r) in zip(decisions, slots):
        while applied < k:
            cs.apply_command(list(commands[applied]))
            applied += 1
        if d.refused:
            continue
        if opens_turn[d.side] and d.side in current and turn[d.side] < TURNS and math.isfinite(d.v):
            them = 3 - d.side
            seen = informative_types(era, cs.core.seen_types(d.side, them))
            random_pick = bool(chose_random[them - 1]) if them - 1 < len(chose_random) else False
            # What the seen types alone leave possible, whatever the setup told the side.
            possible = len(consistent_factions(random_draws(True, era, faction[d.side], mode), seen)) or len(era)
            out.append((game, turn[d.side] + 1, d.v, z_a if d.side == side_a else -z_a, int(not random_pick),
                        possible, len(seen), int(past)))
        opens_turn[d.side] = d.action[1] == END_TURN
        if d.action[1] == END_TURN:
            turn[d.side] += 1
    return out


def gap(t: dict, mask: np.ndarray, games: int) -> dict:
    """v - z over the masked positions, its standard error over games."""
    n = int(mask.sum())
    if n < 30:
        return {"n": n}
    x = (t["v"] - t["z"])[mask]
    g = t["game"][mask].astype(int)
    m = float(x.mean())
    s = np.bincount(g, weights=x, minlength=games)
    c = np.bincount(g, minlength=games).astype(float)
    return {"n": n, "mean_v": float(t["v"][mask].mean()), "mean_z": float(t["z"][mask].mean()), "v_minus_z": m,
            "se": float(math.sqrt(((s - m * c) ** 2).sum()) / c.sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--rounds", type=int, nargs=2, required=True, metavar=("FIRST", "LAST"))
    ap.add_argument("--every", type=int, default=6, help="replay one game in this many")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    rows: List[tuple] = []
    seen_games = replayed = failed = 0
    for k in range(args.rounds[0], args.rounds[1] + 1):
        for kind_dir in ("games_self", "games_past"):
            for path in sorted((args.run / f"round_{k:04d}" / kind_dir).glob("*" + RECORD_SUFFIX)):
                seen_games += 1
                if seen_games % args.every:
                    continue
                result = path.with_name(path.name[:-len(RECORD_SUFFIX)] + ".json")
                if not result.is_file():
                    continue
                try:
                    rows += rows_of(path, json.loads(result.read_text(encoding="utf-8")), replayed,
                                    kind_dir == "games_past")
                    replayed += 1
                except Exception as e:                      # noqa: BLE001  a record that does not replay is counted
                    failed += 1
                    print(f"{path.name}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                if replayed % 100 == 0 and replayed:
                    print(f"{replayed} games replayed, {len(rows)} positions", flush=True)
    t = {name: np.array([r[i] for r in rows], dtype=np.float64) for i, name in enumerate(COLUMNS)}
    every = np.ones(t["v"].size, bool)
    named = t["possible"] == 1            # the seen types leave one faction
    res = {"rounds": list(args.rounds), "every": args.every, "games": replayed, "failed": failed,
           "positions": int(t["v"].size), "told_the_faction_share": float(t["told"].mean()),
           "all": gap(t, every, replayed), "by_turn": {}}
    print(f"rounds {args.rounds[0]}-{args.rounds[1]}: {replayed} games replayed ({failed} failed), "
          f"{t['v'].size} first decisions of a turn; told the enemy's faction by the setup: "
          f"{res['told_the_faction_share']:.2f}")
    print(f"{'own turn':9s} {'share named':>11s}   {'all':>22s}   {'types name the faction':>26s}   "
          f"{'they do not':>22s}")
    for turn in range(1, TURNS + 1):
        at = t["turn"] == turn
        cell = {"all": gap(t, at, replayed), "named_share": float(named[at].mean()) if at.any() else None,
                "named": gap(t, at & named, replayed), "not_named": gap(t, at & ~named, replayed),
                "possible_mean": float(t["possible"][at].mean()) if at.any() else None}
        res["by_turn"][str(turn)] = cell

        def show(g: dict) -> str:
            return f"{g['v_minus_z']:+.3f} ± {g['se']:.3f} (n={g['n']})" if "se" in g else f"n={g['n']}"
        print(f"{turn:9d} {cell['named_share']:11.2f}   {show(cell['all']):>22s}   {show(cell['named']):>26s}   "
              f"{show(cell['not_named']):>22s}")
    early = (t["turn"] >= 2) & (t["turn"] <= 6)
    res["turns_2_to_6"] = {"all": gap(t, early, replayed), "named": gap(t, early & named, replayed),
                           "not_named": gap(t, early & ~named, replayed)}
    print("own turns 2 to 6:", json.dumps({k: {a: round(b, 4) for a, b in v.items()}
                                           for k, v in res["turns_2_to_6"].items()}))
    args.out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
