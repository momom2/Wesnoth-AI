"""Round 5 (invent): how much variance does the existing counter-keyed dice
coupling remove between two rollouts that start from slightly different
states?

Record: training/metrics/belief_worlds_20261012/rollouts_<world>.jsonl.
Each candidate turn of the 199-position benchmark has 8 rollouts per world
(both sides own2 at 0 slots, raw:t0.5+eo-1.5, to the fourth player-turn
start). Rollout r of a candidate uses the same seeds in every world
(docs/belief_worlds_prereg_20261012.md: "A rollout's dice come from the
re-grade's salt for that candidate and rollout in every arm"); the dice are
the sim's counter-keyed stream (WesnothSim._next_seed) and the sampler is
seeded from the same seeds. The worlds differ in the starting state:
  T  true state;
  TU same units, the mover's estimates for villages, gold and income, and
     the opponent's sighting record emptied (a small perturbation);
  O2 the units the mover cannot see removed (a larger perturbation).

For a read y (margin at read 2 or 4, value at read 2 or 4) and a world pair
(A, B), the coupled difference is y_A(r) - y_B(r) and the uncoupled one is
y_A(r) - y_B(r + 1 mod 8), both within one candidate, so the candidate's
mean difference is common to both and cancels in the ratio
    ratio = Var(coupled diff) / Var(uncoupled diff),
the share of the pair-difference variance the coupling leaves. Variances
are taken after removing each candidate's mean difference. Bootstrap over
positions (1,000 resamples) gives the interval.

Usage: python round5_invent_coupling_record.py  (numpy, standard library)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "training" / "metrics" / "belief_worlds_20261012"
WORLDS = {"T": "true", "TU": "true_units", "O2": "none"}
READS = [("margin", "2"), ("margin", "4"), ("value", "2"), ("value", "4")]
ROLLOUTS = 8


def load(world):
    out = {}
    for line in open(DATA / f"rollouts_{world}.jsonl"):
        x = json.loads(line)
        if "rollouts" not in x:
            continue
        rs = {r["r"]: r for r in x["rollouts"]}
        if len(rs) != ROLLOUTS:
            continue
        out[(x["index"], x["slot"])] = rs
    return out


def read_value(rollout, field, at):
    rd = rollout["reads"].get(at)
    return None if rd is None else rd[field]


def pair_rows(A, B, field, at):
    """Per candidate: arrays of coupled and uncoupled differences."""
    rows = []
    for key in sorted(set(A) & set(B)):
        a, b = A[key], B[key]
        if any(a[r]["seeds"] != b[r]["seeds"] for r in range(ROLLOUTS)):
            raise SystemExit(f"seeds differ at {key}")
        ya = [read_value(a[r], field, at) for r in range(ROLLOUTS)]
        yb = [read_value(b[r], field, at) for r in range(ROLLOUTS)]
        if None in ya or None in yb:
            continue
        ya, yb = np.array(ya, float), np.array(yb, float)
        coupled = ya - yb
        uncoupled = ya - np.roll(yb, -1)
        rows.append((key[0], coupled - coupled.mean(), uncoupled - uncoupled.mean()))
    return rows


def ratio(rows):
    c = np.concatenate([r[1] for r in rows])
    u = np.concatenate([r[2] for r in rows])
    return float(np.mean(c ** 2) / np.mean(u ** 2)), float(np.mean(c ** 2)), float(np.mean(u ** 2))


def bootstrap(rows, n=1000, seed=20261020):
    by_pos = defaultdict(list)
    for r in rows:
        by_pos[r[0]].append(r)
    positions = list(by_pos)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.choice(len(positions), len(positions), replace=True)
        rs = [r for i in pick for r in by_pos[positions[i]]]
        vals.append(ratio(rs)[0])
    return np.percentile(vals, [2.5, 97.5])


def main():
    data = {k: load(v) for k, v in WORLDS.items()}
    print("candidates per world:", {k: len(v) for k, v in data.items()})
    for a, b in (("T", "TU"), ("T", "O2"), ("TU", "O2")):
        for field, at in READS:
            rows = pair_rows(data[a], data[b], field, at)
            rt, vc, vu = ratio(rows)
            lo, hi = bootstrap(rows)
            print(f"{a}-{b} {field}@{at}: candidates {len(rows)}, "
                  f"coupled var {vc:.4g}, uncoupled var {vu:.4g}, "
                  f"ratio {rt:.3f} (95% {lo:.3f}-{hi:.3f})")
    # Identical rollouts (same reads and decision count) between T and TU.
    same = total = 0
    for key in set(data["T"]) & set(data["TU"]):
        for r in range(ROLLOUTS):
            total += 1
            x, y = data["T"][key][r], data["TU"][key][r]
            same += (x["reads"] == y["reads"] and x["decisions"] == y["decisions"])
    print(f"T-TU identical rollouts: {same} of {total}")


if __name__ == "__main__":
    sys.exit(main())
