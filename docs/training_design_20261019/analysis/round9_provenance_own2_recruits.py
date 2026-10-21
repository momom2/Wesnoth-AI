"""Round 9, provenance stance: does `own2` recruit like `own1` on the units where
batch 1's labels say `own1`'s near-tie recruit choices are wrong?

Reads training/metrics/own2_base_20261020/games_own2_vs_own1.tar.gz (6,402
games, `own2` against `own1`, both at the reference decode, 64 slots; the two
players share every game's map and draw, so their recruit counts compare on
the same setups) and counts each player's recruits per (faction, unit) per
side-game of that faction. Beside them, the additive head of
round9_provenance_recruit_cells.py fitted on all of batch 1 (ridge 2,000
labels): v(unit) in z per recruit, larger = the labels prefer it at a
near-tie. A unit the head prefers that `own2` already recruits more than
`own1` is a sign that `own2` moved toward the rule; nothing here measures
`own2`'s near-ties, so this is a reading, not a forecast.
Standard library and numpy only.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import math
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TAR = HERE.parents[2] / "training" / "metrics" / "own2_base_20261020" / "games_own2_vs_own1.tar.gz"

spec = importlib.util.spec_from_file_location("rc", HERE / "round9_provenance_recruit_cells.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)


def head_values(lam=2000.0):
    cells, a, _sg = rc.load()
    names = sorted({u for c in cells for u in rc.units_of(c)})
    at = {u: i for i, u in enumerate(names)}
    X = np.zeros((len(cells), len(names)))
    for i, c in enumerate(cells):
        u0, u1 = rc.units_of(c)
        X[i, at[u1]] += 1.0
        X[i, at[u0]] -= 1.0
    s, n = a["s_T"].sum(1), a["n"].sum(1)
    m = np.where(n > 0, s / np.maximum(n, 1), 0.0)
    beta = np.linalg.solve(X.T @ (X * n[:, None]) + lam * np.eye(len(names)), X.T @ (n * m))
    return {u: float(b) for u, b in zip(names, beta)}


def counts():
    per = {"own1": Counter(), "own2": Counter()}
    side_games = {"own1": Counter(), "own2": Counter()}
    with tarfile.open(TAR, "r:*") as tf:
        for mem in tf:
            name = mem.name.rsplit("/", 1)[-1]
            if not (mem.isfile() and name.endswith(".game.jsonl.gz")):
                continue
            rec = json.loads(gzip.decompress(tf.extractfile(mem).read()).decode("utf-8").splitlines()[0])
            setup = rec["setup"]
            side_of = {p["label"]: int(p["side"]) for p in rec["players"].values()}
            fac = {1: setup["faction1"], 2: setup["faction2"]}
            current = None
            for c in rec["commands"]:
                if c[0] == "init_side":
                    current = int(c[1])
                elif c[0] == "recruit" and current is not None:
                    for label, side in side_of.items():
                        if side == current:
                            per[label][(fac[side], c[1])] += 1
            for label, side in side_of.items():
                side_games[label][fac[side]] += 1
    return per, side_games


def main():
    v = head_values()
    per, sgs = counts()
    print(f"side-games per faction: own1 {dict(sgs['own1'])}; own2 {dict(sgs['own2'])}")
    print("faction | unit | head value v (z per recruit) | own1 recruits per side-game | own2 | own2 - own1 (SE, Poisson)")
    by_fac = defaultdict(list)
    for (fac, unit) in set(per["own1"]) | set(per["own2"]):
        by_fac[fac].append(unit)
    agree = disagree = 0
    for fac in sorted(by_fac):
        n1, n2 = sgs["own1"][fac], sgs["own2"][fac]
        rows = []
        for unit in by_fac[fac]:
            c1, c2 = per["own1"][(fac, unit)], per["own2"][(fac, unit)]
            r1, r2 = c1 / n1, c2 / n2
            se = math.sqrt(c1 / n1 ** 2 + c2 / n2 ** 2)
            hv = v.get(f"{fac}|{unit}")
            rows.append((unit, hv, r1, r2, r2 - r1, se))
        hv_mean = np.mean([r[1] for r in rows if r[1] is not None])
        for unit, hv, r1, r2, d, se in sorted(rows, key=lambda r: -(r[2])):
            if r1 < 0.05 and r2 < 0.05:
                continue
            tag = ""
            if hv is not None and abs(d) > 2 * se:
                pref = hv - hv_mean
                tag = "toward the head" if pref * d > 0 else "away from the head"
                agree += pref * d > 0
                disagree += pref * d <= 0
            hv_txt = f"{hv:+.4f}" if hv is not None else "  n/a  "
            print(f"  {fac:17s} {unit:20s} {hv_txt} | {r1:6.3f} | {r2:6.3f} | {d:+.3f} ({se:.3f}) {tag}")
    print(f"units whose recruit rate moved by more than 2 SE: {agree} toward the head's preference "
          f"(relative to the faction's mean value), {disagree} away")


if __name__ == "__main__":
    main()
