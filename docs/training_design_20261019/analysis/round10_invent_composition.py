"""Round 10, invent stance: does the argmax player collapse its recruit mixture?

Compares the recruit composition of `own1` at the reference decode (argmax,
end_turn offset -1.5, 64 memory slots, no band coins) with the human corpus
it imitates, on the ladder scenarios both share.

own1: the 6,400 games of S0(a) (training/metrics/near_tie_coins_20261020/
games_a.tar.gz). One seat per game carries the attack offset +1 (the game's
attack coin names it); only the other seat is read, so every side read is the
reference decode. Humans: replays_dataset_imitation (both sides of every game
whose scenario is one of the ladder scenarios own1 played), split by the
side's player rating (training/metrics/player_ratings/ratings.json, the
quantile-0.75 threshold that file states) where the player is rated.

Readings per side-game, by mover faction and pooled:
1. turn 1 (the side's first turn): recruits, distinct types, share of the
   commonest type, P(all recruits one type | >= 3 recruits), and P(a recruit
   repeats the previous recruit's type) within the turn;
2. whole game: recruits, distinct types, Herfindahl index of the type shares,
   P(a recruit repeats the previous recruit's type).
Means with standard errors over side-games. Standard library and numpy only.
"""
from __future__ import annotations

import gzip
import json
import math
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
GAMES_A = ROOT / "training" / "metrics" / "near_tie_coins_20261020" / "games_a.tar.gz"
HUMANS = ROOT / "replays_dataset_imitation"
RATINGS = ROOT / "training" / "metrics" / "player_ratings" / "ratings.json"
PLAYERS = ROOT / "training" / "metrics" / "player_ratings" / "players.jsonl"


def side_turn_recruits(commands, side):
    """The side's recruits per own turn: list of lists of unit types."""
    turns, current, mover = [], None, None
    for c in commands:
        if c[0] == "init_side":
            mover = int(c[1])
            if mover == side:
                current = []
                turns.append(current)
            continue
        if c[0] == "recruit" and mover == side and current is not None:
            current.append(c[1])
    return turns


def stats(turns):
    t1 = turns[0] if turns else []
    allr = [u for t in turns for u in t]
    out = {}
    if t1:
        cnt = defaultdict(int)
        for u in t1:
            cnt[u] += 1
        out["t1_n"] = len(t1)
        out["t1_distinct"] = len(cnt)
        out["t1_maxshare"] = max(cnt.values()) / len(t1)
        if len(t1) >= 3:
            out["t1_allsame_3plus"] = float(len(cnt) == 1)
        if len(t1) >= 2:
            out["t1_repeat"] = sum(t1[k] == t1[k - 1] for k in range(1, len(t1))) / (len(t1) - 1)
    if allr:
        cnt = defaultdict(int)
        for u in allr:
            cnt[u] += 1
        out["g_n"] = len(allr)
        out["g_distinct"] = len(cnt)
        out["g_hhi"] = sum((v / len(allr)) ** 2 for v in cnt.values())
        reps = [allr[k] == allr[k - 1] for k in range(1, len(allr))]
        if reps:
            out["g_repeat"] = float(np.mean(reps))
    return out


def own1_rows():
    rows, scenarios = [], set()
    with tarfile.open(GAMES_A, "r:gz") as tf:
        for m in tf:
            if not m.name.endswith(".game.jsonl.gz"):
                continue
            rec = json.loads(gzip.decompress(tf.extractfile(m).read()).decode())
            seat = rec["coins"]["attack"]["seat"]
            free = "b" if seat == "a" else "a"
            side = int(rec["players"][free]["side"])
            fac = rec["setup"][f"faction{side}"]
            scenarios.add(rec["scenario_id"])
            rows.append((fac, stats(side_turn_recruits(rec["commands"], side))))
    return rows, scenarios


def human_rows(scenarios):
    ratings = json.loads(RATINGS.read_text(encoding="utf-8"))
    rated = {p["id"]: p["rating_elo"] for p in ratings["players"] if p["games"] >= ratings["min_games"]}
    thr = ratings["threshold_elo"]
    names = {}
    with open(PLAYERS, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            names[r["file"]] = r["sides"]
    rows = []
    for path in sorted(HUMANS.glob("*.json.gz")):
        d = json.loads(gzip.decompress(path.read_bytes()).decode())
        if d.get("scenario_id") not in scenarios:
            continue
        sides = {int(s["side"]): s["faction"] for s in d["starting_sides"]}
        if sorted(sides) != [1, 2]:
            continue
        who = names.get(path.name, {})
        for side in (1, 2):
            pid = who.get(str(side))
            r = rated.get(pid)
            tier = "unrated" if r is None else ("top" if r >= thr else "rated_rest")
            rows.append((sides[side], tier, stats(side_turn_recruits(d["commands"], side))))
    return rows, thr


KEYS = ("t1_n", "t1_distinct", "t1_maxshare", "t1_allsame_3plus", "t1_repeat", "g_n", "g_distinct", "g_hhi", "g_repeat")


def summarize(label, dicts):
    parts = []
    for k in KEYS:
        v = np.array([d[k] for d in dicts if k in d], dtype=float)
        if len(v) < 2:
            parts.append(f"{k} -")
            continue
        parts.append(f"{k} {v.mean():.3f}({v.std(ddof=1) / math.sqrt(len(v)):.3f})")
    print(f"  {label:28s} n {len(dicts):6d}  " + "  ".join(parts))


def main():
    o_rows, scenarios = own1_rows()
    h_rows, thr = human_rows(scenarios)
    print(f"own1 reference-decode sides: {len(o_rows)} (S0(a), unhooked seat); ladder scenarios {len(scenarios)}")
    print(f"human sides on those scenarios: {len(h_rows)}; top tier = rating >= {thr:.1f} Elo (ratings.json quantile 0.75)")
    print("\n1-2. pooled")
    summarize("own1", [s for _, s in o_rows])
    summarize("humans, all", [s for _, _, s in h_rows])
    for tier in ("top", "rated_rest", "unrated"):
        summarize(f"humans, {tier}", [s for _, t, s in h_rows if t == tier])
    print("\nby faction (own1 / humans all / humans top)")
    facs = sorted({f for f, _ in o_rows})
    for fac in facs:
        print(f" {fac}")
        summarize("own1", [s for f, s in o_rows if f == fac])
        summarize("humans, all", [s for f, _, s in h_rows if f == fac])
        summarize("humans, top", [s for f, t, s in h_rows if f == fac and t == "top"])


if __name__ == "__main__":
    main()
