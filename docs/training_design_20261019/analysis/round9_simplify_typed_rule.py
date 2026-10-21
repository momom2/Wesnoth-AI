"""Round 9, simplify stance: is a feature-free typed rule worth a match before the head is built?

A typed rule switches every band decision of the ordered pair types whose mean label is positive; it needs
no feature gather. Its value on batch 1 read in sample, sum over types of n_t max(m_t, 0) per side-game,
carries the selection of the positive types. Under a null of no typed effect, each type's estimate is
roughly N(0, se_t^2) (clustered SEs of readout_b_offline.json; types treated as independent), so the same
statistic has a null distribution; this script compares the two.
Input: training/metrics/near_tie_coins_20261020/readout_b_offline.json (60,000 games, 120,000 side-games).
numpy and the standard library only.
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
READOUT = ROOT / "training/metrics/near_tie_coins_20261020/readout_b_offline.json"
ELO = 800 / math.log(10)
RNG = np.random.default_rng(20261021)


def main():
    d = json.loads(READOUT.read_text())
    side_games = d["band"]["per_side_game"]["side_games"]
    pairs = d["labels"]["by_pair"]
    n = np.array([v["n"] / side_games for v in pairs.values()])
    m = np.array([v["mean"] for v in pairs.values()])
    se = np.array([v["se"] for v in pairs.values()])
    insample = float(np.sum(n * np.maximum(m, 0.0)))
    sims = np.sum(n * np.maximum(RNG.standard_normal((20000, len(n))) * se, 0.0), axis=1)
    p = float(np.mean(sims >= insample))
    print(f"{len(n)} ordered pair types, {n.sum():.1f} band decisions per side-game")
    print(f"typed rule, in-sample value: {insample:.4f} z per side-game = {ELO * insample:.1f} Elo")
    print(f"same statistic under a null of no typed effect: mean {ELO * sims.mean():.1f} Elo, "
          f"95th percentile {ELO * np.percentile(sims, 95):.1f}; P(null >= observed) = {p:.2f}")
    best = max(pairs, key=lambda k: pairs[k]["mean"] / pairs[k]["se"])
    v = pairs[best]
    k = v["n"] / side_games
    print(f"largest type by z: {best}, {v['mean']:+.4f} +- {v['se']:.4f} (z {v['mean'] / v['se']:.2f}), "
          f"{k:.2f} a side-game: always-switch value {ELO * k * v['mean']:+.1f} +- {ELO * k * v['se']:.1f} Elo, "
          f"selected as the largest of {len(n)}")


if __name__ == "__main__":
    main()
