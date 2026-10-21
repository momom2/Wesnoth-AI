"""Round 10, invent stance: how far is own2's recruit composition from the human
mixture its network imitates?

Per mover faction, the share of each unit type among a side's whole-game
recruits (and among its turn-1 recruits): `own2` from S0(b)'s games
(training/metrics/own2_base_20261020/games_own2_vs_own1.tar.gz, own2's seat),
humans from every 4th file of replays_dataset_imitation on the same ladder
scenarios (a subsample, so the run stays near a minute). Reports the total
variation distance TV = (1/2) sum |share_own2 - share_human| and TV x the mean
recruits per side-game: the recruits per side-game a composition at the human
shares would change. That is a scale, under the assumption that the
network's cumulative recruit probabilities equal the human shares; the
assumption is unverified and nothing relies on the number.
Standard library and numpy only.
"""
from __future__ import annotations

import gzip
import json
import tarfile
from collections import defaultdict
from pathlib import Path

from round10_invent_composition import side_turn_recruits

ROOT = Path(__file__).resolve().parents[3]
GAMES = ROOT / "training" / "metrics" / "own2_base_20261020" / "games_own2_vs_own1.tar.gz"
HUMANS = ROOT / "replays_dataset_imitation"


def add(acc, fac, turns):
    allr = [u for t in turns for u in t]
    t1 = turns[0] if turns else []
    a = acc[fac]
    a["sides"] += 1
    a["n"] += len(allr)
    a["n1"] += len(t1)
    for u in allr:
        a["all"][u] += 1
    for u in t1:
        a["t1"][u] += 1


def new():
    return defaultdict(lambda: {"sides": 0, "n": 0, "n1": 0, "all": defaultdict(int), "t1": defaultdict(int)})


def tv(a, b, key):
    ta, tb = sum(a[key].values()), sum(b[key].values())
    units = set(a[key]) | set(b[key])
    return 0.5 * sum(abs(a[key].get(u, 0) / ta - b[key].get(u, 0) / tb) for u in units)


def main():
    own, scen = new(), set()
    with tarfile.open(GAMES, "r:gz") as tf:
        for m in tf:
            if not m.name.endswith(".game.jsonl.gz"):
                continue
            rec = json.loads(gzip.decompress(tf.extractfile(m).read()).decode())
            scen.add(rec["scenario_id"])
            for seat in ("a", "b"):
                p = rec["players"][seat]
                if p.get("label") != "own2":
                    continue
                side = int(p["side"])
                add(own, rec["setup"][f"faction{side}"], side_turn_recruits(rec["commands"], side))
    hum = new()
    for k, path in enumerate(sorted(HUMANS.glob("*.json.gz"))):
        if k % 4:
            continue
        d = json.loads(gzip.decompress(path.read_bytes()).decode())
        if d.get("scenario_id") not in scen:
            continue
        sides = {int(s["side"]): s["faction"] for s in d["starting_sides"]}
        if sorted(sides) != [1, 2]:
            continue
        for side in (1, 2):
            add(hum, sides[side], side_turn_recruits(d["commands"], side))
    print("faction             own2 sides  human sides  TV whole game  x own2 recruits/side-game  TV turn 1  x own2 turn-1 recruits")
    tot_change = tot_sides = 0.0
    for fac in sorted(own):
        a, b = own[fac], hum[fac]
        t_all, t_1 = tv(a, b, "all"), tv(a, b, "t1")
        n, n1 = a["n"] / a["sides"], a["n1"] / a["sides"]
        tot_change += t_all * n * a["sides"]
        tot_sides += a["sides"]
        print(f"{fac:18s} {a['sides']:10d} {b['sides']:12d}  {t_all:13.3f}  {t_all * n:25.2f}  {t_1:9.3f}  {t_1 * n1:21.2f}")
        top_o = sorted(a["all"].items(), key=lambda kv: -kv[1])[:3]
        top_h = sorted(b["all"].items(), key=lambda kv: -kv[1])[:3]
        print(f"   own2 top: {[(u, round(c / a['n'], 3)) for u, c in top_o]}")
        print(f"   human top: {[(u, round(c / b['n'], 3)) for u, c in top_h]}")
    print(f"pooled: {tot_change / tot_sides:.2f} recruits per side-game would change at the human shares")


if __name__ == "__main__":
    main()
