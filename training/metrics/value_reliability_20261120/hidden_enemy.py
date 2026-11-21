#!/usr/bin/env python3
"""Whether the value estimate reads the visible board as the whole board
(cause C2 of docs/value_reliability_prereg_20261120.md's follow-up).

A sample of the simple loop's games is replayed through the simulator. At
the first decision of each turn of a current-network side: the logged
estimate v, the side's result z, and the enemy army the side sees and
does not see (units and recruit cost, the leader apart). No network is
run.

    python training/metrics/value_reliability_20261120/hidden_enemy.py \
        --run C:/Users/momom2/wesnoth_runs/simple_loop --rounds 177 208 --every 6 --out hidden_177_208.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.game_record import read_records, start_core  # noqa: E402
from tools.pg_positions import decision_slots  # noqa: E402
from wesnoth_ai.pg_records import decisions_of  # noqa: E402

RECORD_SUFFIX = ".game.jsonl.gz"
END_TURN = 3
COLUMNS = ("game", "side", "turn", "v", "z", "own_units", "own_cost", "seen_units", "seen_cost", "hidden_units",
           "hidden_cost", "leader_hidden", "faction", "enemy_faction", "fog", "hidden_seen_before",
           "hidden_never_seen")


def rows_of(path: Path, res: dict, game: int, cost: Dict[str, int], factions: Dict[str, int],
            track_seen: bool = False) -> List[tuple]:
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
    applied = 0
    opens_turn = {1: True, 2: True}
    turn = {1: 0, 2: 0}
    ever = {1: set(), 2: set()}          # the unit ids each side has seen at any of its own decisions so far
    out = []
    for d, (k, _r) in zip(decisions, slots):
        while applied < k:
            cs.apply_command(list(commands[applied]))
            applied += 1
        if d.refused:
            continue
        obs = cs.observe(d.side) if track_seen or opens_turn[d.side] else None
        if opens_turn[d.side] and d.side in current and math.isfinite(d.v):
            seen = set(obs.visible_ids())
            own_n = own_c = seen_n = seen_c = hid_n = hid_c = hid_before = 0
            leader_hidden = 0
            for u in cs.to_state().map.units:
                c = int(cost.get(u.name, 0))
                if u.side == d.side:
                    own_n += 1
                    own_c += c
                elif u.side == 3 - d.side:
                    if u.is_leader:
                        leader_hidden = int(u.id not in seen)
                    elif u.id in seen:
                        seen_n += 1
                        seen_c += c
                    else:
                        hid_n += 1
                        hid_c += c
                        hid_before += int(u.id in ever[d.side])
            out.append((game, d.side, turn[d.side], d.v, z_a if d.side == side_a else -z_a, own_n, own_c, seen_n,
                        seen_c, hid_n, hid_c, leader_hidden, factions.setdefault(faction[d.side], len(factions)),
                        factions.setdefault(faction[3 - d.side], len(factions)), int(obs.fog_on),
                        hid_before if track_seen else -1, hid_n - hid_before if track_seen else -1))
        if obs is not None and track_seen:
            ever[d.side] |= set(obs.visible_ids())
        opens_turn[d.side] = d.action[1] == END_TURN
        if d.action[1] == END_TURN:
            turn[d.side] += 1
    return out


def gap(t: Dict[str, np.ndarray], mask: np.ndarray, games: int) -> dict:
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
    ap.add_argument("--track-seen", action="store_true",
                    help="observe at every decision, to split the hidden enemy units into those the side "
                         "saw at an earlier decision of its own and those it never saw (about 15 times slower)")
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    cost = {name: int(u.get("cost") or 0) for name, u in
            json.loads((ROOT / "unit_stats.json").read_text(encoding="utf-8"))["units"].items()}
    factions: Dict[str, int] = {}
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
                    rows += rows_of(path, json.loads(result.read_text(encoding="utf-8")), replayed, cost, factions,
                                    args.track_seen)
                    replayed += 1
                except Exception as e:                      # noqa: BLE001  a record that does not replay is counted
                    failed += 1
                    print(f"{path.name}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                if replayed % 100 == 0 and replayed:
                    print(f"{replayed} games replayed, {len(rows)} positions", flush=True)
    t = {name: np.array([r[i] for r in rows], dtype=np.float64) for i, name in enumerate(COLUMNS)}
    names = sorted(factions, key=factions.get)
    enemy_cost = t["seen_cost"] + t["hidden_cost"]
    share = np.where(enemy_cost > 0, t["hidden_cost"] / np.maximum(enemy_cost, 1), 0.0)
    every = np.ones(t["v"].size, bool)
    res = {"rounds": list(args.rounds), "every": args.every, "games": replayed, "failed": failed,
           "positions": int(t["v"].size), "all": gap(t, every, replayed),
           "fog_on_share": float(t["fog"].mean()),
           "mean_hidden_units": float(t["hidden_units"].mean()), "mean_seen_units": float(t["seen_units"].mean()),
           "mean_own_units": float(t["own_units"].mean()),
           "by_hidden_units": {label: gap(t, m, replayed) for label, m in (
               ("0", t["hidden_units"] == 0), ("1", t["hidden_units"] == 1), ("2", t["hidden_units"] == 2),
               ("3-4", (t["hidden_units"] >= 3) & (t["hidden_units"] <= 4)),
               ("5-7", (t["hidden_units"] >= 5) & (t["hidden_units"] <= 7)), ("8+", t["hidden_units"] >= 8))},
           "by_hidden_share_of_enemy_cost": {label: gap(t, m, replayed) for label, m in (
               ("none hidden", t["hidden_units"] == 0), ("up to a quarter", (share > 0) & (share <= 0.25)),
               ("a quarter to a half", (share > 0.25) & (share <= 0.5)),
               ("a half to three quarters", (share > 0.5) & (share <= 0.75)),
               ("over three quarters", share > 0.75))},
           "by_turn_and_hidden": {}, "by_faction_none_hidden": {}, "by_faction_some_hidden": {}}
    for label, lo, hi in (("turns 1-4", 0, 3), ("turns 5-10", 4, 9), ("turns 11+", 10, 999)):
        in_turns = (t["turn"] >= lo) & (t["turn"] <= hi)
        res["by_turn_and_hidden"][label] = {
            "none hidden": gap(t, in_turns & (t["hidden_units"] == 0), replayed),
            "1-2 hidden": gap(t, in_turns & (t["hidden_units"] >= 1) & (t["hidden_units"] <= 2), replayed),
            "3+ hidden": gap(t, in_turns & (t["hidden_units"] >= 3), replayed)}
    for i, name in enumerate(names):
        res["by_faction_none_hidden"][name] = gap(t, (t["faction"] == i) & (t["hidden_units"] == 0), replayed)
        res["by_faction_some_hidden"][name] = gap(t, (t["faction"] == i) & (t["hidden_units"] > 0), replayed)
    # the estimate against the two armies it could be reading: the seen enemy, and the whole enemy
    d_seen = (t["own_cost"] - t["seen_cost"]) / 100.0
    d_true = (t["own_cost"] - enemy_cost) / 100.0
    hidden = t["hidden_cost"] / 100.0
    design = np.stack([np.ones(t["v"].size), d_seen, hidden], axis=1)
    res["regressions"] = {}
    for label, y in (("v", t["v"]), ("z", t["z"])):
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        res["regressions"][label] = {"intercept": float(beta[0]), "per_100_gold_own_less_seen_enemy": float(beta[1]),
                                     "per_100_gold_hidden_enemy": float(beta[2])}
    res["corr_v_with_own_less_seen"] = float(np.corrcoef(t["v"], d_seen)[0, 1])
    res["corr_v_with_own_less_whole"] = float(np.corrcoef(t["v"], d_true)[0, 1])
    res["corr_z_with_own_less_seen"] = float(np.corrcoef(t["z"], d_seen)[0, 1])
    res["corr_z_with_own_less_whole"] = float(np.corrcoef(t["z"], d_true)[0, 1])
    args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.save(args.out.with_suffix(".npy"), np.stack([t[c] for c in COLUMNS], axis=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
