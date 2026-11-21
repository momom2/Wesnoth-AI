#!/usr/bin/env python3
"""Where the value estimate's optimism sits (the follow-up of
docs/value_reliability_prereg_20261120.md), read from the decision logs of
the simple loop's games: by the decision's place in its side's turn, by
kind of decision, by side, by faction and map, and at a game's first
decision. No game is played and no network is run.

    python training/metrics/value_reliability_20261120/where_optimism.py \
        --run C:/Users/momom2/wesnoth_runs/simple_loop --rounds 177 208 --out where_177_208.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.game_record import read_records  # noqa: E402

RECORD_SUFFIX = ".game.jsonl.gz"
KINDS = {0: "attack", 1: "move", 2: "recruit", 3: "end_turn"}     # wesnoth_ai/server_priors.py
END_TURN = 3


def load(run: Path, first: int, last: int) -> Dict[str, np.ndarray]:
    """One row a decision of the current network's game-sides (both sides
    of a self game, side A of a past-self game), in play order within a
    side. `turn` is the side's own turn (0 first), `place` the decision's
    index in that turn, `turn_len` the turn's decisions, the end_turn one
    included."""
    cols: Dict[str, list] = {k: [] for k in ("game", "gside", "side", "past", "v", "z", "kind", "turn", "place",
                                             "turn_len", "index", "n", "faction", "map")}
    factions: Dict[str, int] = {}
    maps: Dict[str, int] = {}
    games = sides = 0
    for k in range(first, last + 1):
        for kind_dir in ("games_self", "games_past"):
            for path in sorted((run / f"round_{k:04d}" / kind_dir).glob("*" + RECORD_SUFFIX)):
                result = path.with_name(path.name[:-len(RECORD_SUFFIX)] + ".json")
                if not result.is_file():
                    continue
                res = json.loads(result.read_text(encoding="utf-8"))
                records = list(read_records(path))
                if len(records) != 1:
                    continue
                side_a = int(res["side_a"])
                z_a = {"win": 1.0, "loss": -1.0}.get(res.get("outcome_a") or "", 0.0)
                past = not str(res["label_b"]).endswith("b")
                for side in ((side_a,) if past else (side_a, 3 - side_a)):
                    ds = [d for d in records[0]["decisions"] if int(d["side"]) == side and not d["refused"]]
                    if not ds or not all(math.isfinite(float(d["v"])) for d in ds):
                        continue
                    kinds = np.array([int(d["action"][1]) for d in ds])
                    ends = np.flatnonzero(kinds == END_TURN)
                    turn = np.searchsorted(ends, np.arange(len(ds)), side="left")
                    starts = np.concatenate([[0], ends + 1])
                    place = np.arange(len(ds)) - starts[turn]
                    lengths = np.bincount(turn)
                    faction = res["faction_a"] if side == side_a else res["faction_b"]
                    n = len(ds)
                    cols["game"].append(np.full(n, games))
                    cols["gside"].append(np.full(n, sides))
                    cols["side"].append(np.full(n, side))
                    cols["past"].append(np.full(n, past))
                    cols["v"].append(np.array([float(d["v"]) for d in ds]))
                    cols["z"].append(np.full(n, z_a if side == side_a else -z_a))
                    cols["kind"].append(kinds)
                    cols["turn"].append(turn)
                    cols["place"].append(place)
                    cols["turn_len"].append(lengths[turn])
                    cols["index"].append(np.arange(n))
                    cols["n"].append(np.full(n, n))
                    cols["faction"].append(np.full(n, factions.setdefault(faction, len(factions))))
                    cols["map"].append(np.full(n, maps.setdefault(res["scenario_id"], len(maps))))
                    sides += 1
                games += 1
    data = {k: np.concatenate(x) for k, x in cols.items()}
    data["_factions"] = np.array(sorted(factions, key=factions.get))
    data["_maps"] = np.array(sorted(maps, key=maps.get))
    print(f"rounds {first}-{last}: {games} games, {sides} game-sides, {data['v'].size} decisions", flush=True)
    return data


def gap(d: Dict[str, np.ndarray], mask: np.ndarray) -> dict:
    """Over the decisions of `mask`: the mean estimate, the mean result,
    their difference and its standard error (games as the units: a ratio
    of sums over games, linearised)."""
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}
    v, z, g = d["v"][mask], d["z"][mask], d["game"][mask]
    diff = float((v - z).mean())
    size = int(d["game"].max()) + 1
    s = np.bincount(g, weights=v - z, minlength=size)
    c = np.bincount(g, minlength=size).astype(float)
    se = float(math.sqrt(((s - diff * c) ** 2).sum()) / c.sum())
    return {"n": n, "mean_v": float(v.mean()), "mean_z": float(z.mean()), "v_minus_z": diff, "se": se}


def slope(v: np.ndarray, z: np.ndarray) -> dict:
    vm, zm = v.mean(), z.mean()
    b = float(((v - vm) * (z - zm)).sum() / ((v - vm) ** 2).sum())
    resid = z - (zm + b * (v - vm))
    se = float(math.sqrt((resid ** 2).sum() / (v.size - 2) / ((v - vm) ** 2).sum()))
    return {"slope": b, "se": se, "sd_v": float(v.std()), "n": int(v.size)}


def read(d: Dict[str, np.ndarray]) -> dict:
    every = np.ones(d["v"].size, bool)
    res: dict = {"all": gap(d, every),
                 "by_side": {str(s): gap(d, d["side"] == s) for s in (1, 2)},
                 "self_games": gap(d, ~d["past"]), "past_games_current_side": gap(d, d["past"])}
    # C1: the place in the side's own turn
    last = d["kind"] == END_TURN
    first = d["place"] == 0
    long_turn = d["turn_len"] >= 4
    rel = np.where(d["turn_len"] > 1, d["place"] / np.maximum(d["turn_len"] - 1, 1), 0.0)
    res["by_place_in_turn"] = {
        "first_decision": gap(d, first & long_turn),
        "first_quarter": gap(d, long_turn & ~first & ~last & (rel <= 0.25)),
        "second_quarter": gap(d, long_turn & ~last & (rel > 0.25) & (rel <= 0.5)),
        "third_quarter": gap(d, long_turn & ~last & (rel > 0.5) & (rel <= 0.75)),
        "fourth_quarter": gap(d, long_turn & ~last & (rel > 0.75)),
        "end_turn_decision": gap(d, last & long_turn)}
    # the estimate's own movement: within a turn, and across the enemy's turn
    order_ok = np.r_[d["gside"][1:] == d["gside"][:-1], False]          # the next row is the same game-side
    nxt_v = np.r_[d["v"][1:], np.nan]
    within = order_ok & ~last                                            # next decision is in the same turn
    across = order_ok & last                                             # next decision opens the side's next turn
    size = int(d["game"].max()) + 1

    def mean_change(mask: np.ndarray) -> dict:
        x = (nxt_v - d["v"])[mask]
        g = d["game"][mask]
        m = float(x.mean())
        s = np.bincount(g, weights=x, minlength=size)
        c = np.bincount(g, minlength=size).astype(float)
        return {"n": int(mask.sum()), "mean_change": m, "se": float(math.sqrt(((s - m * c) ** 2).sum()) / c.sum())}

    res["changes"] = {"within_a_turn_per_decision": mean_change(within), "across_the_enemy_turn": mean_change(across)}
    turn_first = first & order_ok
    # a whole turn's rise: the end_turn decision's estimate less the turn's first
    rise, rise_game = [], []
    start_v = np.nan
    for i in range(d["v"].size):
        if d["place"][i] == 0:
            start_v = d["v"][i]
        if last[i]:
            rise.append(d["v"][i] - start_v)
            rise_game.append(d["game"][i])
    rise, rise_game = np.array(rise), np.array(rise_game)
    m = float(rise.mean())
    s = np.bincount(rise_game, weights=rise, minlength=size)
    c = np.bincount(rise_game, minlength=size).astype(float)
    res["changes"]["rise_over_a_whole_turn"] = {"n": int(rise.size), "mean_change": m,
                                                "se": float(math.sqrt(((s - m * c) ** 2).sum()) / c.sum())}
    res["changes"]["turns_with_a_first_decision"] = int(turn_first.sum())
    res["by_kind"] = {name: gap(d, d["kind"] == k) for k, name in KINDS.items()}
    # by the side's own turn number, at the turn's first decision
    res["first_decision_by_turn"] = {str(t + 1): gap(d, first & (d["turn"] == t)) for t in range(0, 20)}
    res["by_own_turn"] = {str(t + 1): gap(d, d["turn"] == t) for t in range(0, 20)}
    # C5: the game's first decision
    opening = d["index"] == 0
    res["first_decision_of_the_game"] = {**gap(d, opening), **slope(d["v"][opening], d["z"][opening])}
    for t in (1, 2, 4, 7):
        m_t = first & (d["turn"] == t)
        res[f"first_decision_of_own_turn_{t + 1}"] = slope(d["v"][m_t], d["z"][m_t])
    res["by_faction"] = {str(name): gap(d, d["faction"] == i) for i, name in enumerate(d["_factions"])}
    res["by_map"] = {str(name): gap(d, d["map"] == i) for i, name in enumerate(d["_maps"])}
    res["mean_decisions_in_a_turn"] = float(d["turn_len"][first].mean())
    wins = d["z"] > 0
    res["share_of_decisions_by_the_winner"] = float(wins.mean())
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--rounds", type=int, nargs=2, required=True, metavar=("FIRST", "LAST"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    res = read(load(args.run, *args.rounds))
    res["rounds"] = list(args.rounds)
    args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")

    def line(name: str, g: dict) -> str:
        if not g.get("n"):
            return f"{name:28s} none"
        return (f"{name:28s} n={g['n']:8d}  v {g['mean_v']:+.3f}  z {g['mean_z']:+.3f}  "
                f"v-z {g['v_minus_z']:+.3f} +- {g['se']:.3f}")

    print(line("all", res["all"]))
    for k, g in res["by_side"].items():
        print(line(f"side {k}", g))
    print(line("self games", res["self_games"]))
    print(line("past games, current side", res["past_games_current_side"]))
    print("--- by place in the side's turn (turns of 4 decisions or more)")
    for k, g in res["by_place_in_turn"].items():
        print(line(k, g))
    print("--- the estimate's own movement")
    for k, g in res["changes"].items():
        if isinstance(g, dict):
            print(f"{k:28s} n={g['n']:8d}  mean change {g['mean_change']:+.4f} +- {g['se']:.4f}")
    print("--- by kind")
    for k, g in res["by_kind"].items():
        print(line(k, g))
    print("--- at the turn's first decision, by own turn")
    for k, g in res["first_decision_by_turn"].items():
        print(line(f"turn {k}", g))
    print("--- the game's first decision:", {k: round(v, 3) for k, v in res["first_decision_of_the_game"].items()})
    for t in (2, 3, 5, 8):
        print(f"first decision of own turn {t}:", {k: round(v, 3) for k, v in res[f"first_decision_of_own_turn_{t}"].items()})
    print("--- by faction")
    for k, g in res["by_faction"].items():
        print(line(k, g))
    print("mean decisions in a turn: %.1f; share of decisions made by the winner: %.3f"
          % (res["mean_decisions_in_a_turn"], res["share_of_decisions_by_the_winner"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
