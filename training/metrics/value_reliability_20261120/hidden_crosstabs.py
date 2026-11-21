#!/usr/bin/env python3
"""Cross-tabulations of hidden_enemy.py's positions (its .npy): the estimate
less the result by how much enemy is hidden, holding the estimate's own
level fixed; by the enemy's faction; and what the estimate and the result
each make of seen and of hidden enemy units.

    python training/metrics/value_reliability_20261120/hidden_crosstabs.py hidden_177_208.npy
"""
from __future__ import annotations

import math
import sys

import numpy as np

COLUMNS = ("game", "side", "turn", "v", "z", "own_units", "own_cost", "seen_units", "seen_cost", "hidden_units",
           "hidden_cost", "leader_hidden", "faction", "enemy_faction", "fog")
FACTIONS = ("Undead", "Rebels", "Loyalists", "Drakes", "Knalgan Alliance", "Northerners")   # hidden_enemy.py's order


def main() -> int:
    a = np.load(sys.argv[1])
    t = {c: a[:, i] for i, c in enumerate(COLUMNS)}
    games = int(t["game"].max()) + 1

    def gap(mask: np.ndarray) -> str:
        n = int(mask.sum())
        if n < 50:
            return f"n={n:6d}  (too few)"
        x = (t["v"] - t["z"])[mask]
        g = t["game"][mask].astype(int)
        m = float(x.mean())
        s = np.bincount(g, weights=x, minlength=games)
        c = np.bincount(g, minlength=games).astype(float)
        se = math.sqrt(((s - m * c) ** 2).sum()) / c.sum()
        return f"n={n:6d}  v {t['v'][mask].mean():+.3f}  z {t['z'][mask].mean():+.3f}  v-z {m:+.3f} +- {se:.3f}"

    hid = t["hidden_units"]
    print("=== the estimate's level held fixed: v-z by hidden enemy units")
    for lo, hi in ((-1.01, -0.6), (-0.6, -0.2), (-0.2, 0.2), (0.2, 0.6), (0.6, 1.01)):
        band = (t["v"] > lo) & (t["v"] <= hi)
        print(f"v in ({lo:+.1f}, {hi:+.1f}]")
        for label, m in (("none hidden", hid == 0), ("1-2 hidden", (hid >= 1) & (hid <= 2)),
                         ("3-5 hidden", (hid >= 3) & (hid <= 5)), ("6+ hidden", hid >= 6)):
            print(f"   {label:12s} {gap(band & m)}")
    print("=== the same from turn 5 on (the opening, where all is hidden, left out)")
    late = t["turn"] >= 4
    for lo, hi in ((-0.6, -0.2), (-0.2, 0.2), (0.2, 0.6)):
        band = late & (t["v"] > lo) & (t["v"] <= hi)
        print(f"v in ({lo:+.1f}, {hi:+.1f}]")
        for label, m in (("none hidden", hid == 0), ("1-2 hidden", (hid >= 1) & (hid <= 2)),
                         ("3-5 hidden", (hid >= 3) & (hid <= 5)), ("6+ hidden", hid >= 6)):
            print(f"   {label:12s} {gap(band & m)}")
    print("=== by the deciding side's faction (rows) and the enemy's (columns): v-z")
    print(" " * 18 + "".join(f"{f[:9]:>11s}" for f in FACTIONS) + f"{'any enemy':>11s}")
    for i, f in enumerate(FACTIONS):
        cells = []
        for j in range(len(FACTIONS)):
            m = (t["faction"] == i) & (t["enemy_faction"] == j)
            cells.append(f"{(t['v'] - t['z'])[m].mean():+11.3f}" if m.sum() >= 100 else f"{'':>11s}")
        cells.append(f"{(t['v'] - t['z'])[t['faction'] == i].mean():+11.3f}")
        print(f"{f:18s}" + "".join(cells))
    print(f"{'any own faction':18s}" + "".join(f"{(t['v'] - t['z'])[t['enemy_faction'] == j].mean():+11.3f}"
                                                for j in range(len(FACTIONS))))
    print("=== by the enemy's faction")
    for j, f in enumerate(FACTIONS):
        print(f"   against {f:18s} {gap(t['enemy_faction'] == j)}")
    print("=== what a unit is worth to the estimate and to the result (least squares, turn 5 on, per unit)")
    design = np.stack([np.ones(late.sum()), t["own_units"][late], t["seen_units"][late], t["hidden_units"][late]], axis=1)
    for label in ("v", "z"):
        beta, *_ = np.linalg.lstsq(design, t[label][late], rcond=None)
        print(f"   {label}: intercept {beta[0]:+.3f}, an own unit {beta[1]:+.4f}, a seen enemy unit {beta[2]:+.4f}, "
              f"a hidden enemy unit {beta[3]:+.4f}")
    print("=== how much is hidden, by turn (mean hidden / mean enemy units)")
    for lo, hi in ((0, 1), (2, 3), (4, 6), (7, 9), (10, 13), (14, 30)):
        m = (t["turn"] >= lo) & (t["turn"] <= hi)
        enemy = (t["seen_units"] + t["hidden_units"])[m].mean()
        print(f"   turns {lo + 1}-{hi + 1}: hidden {hid[m].mean():.2f} of {enemy:.2f} enemy units; own {t['own_units'][m].mean():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
