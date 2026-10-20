"""Round 5 (provenance stance): do the replay's argmax disagreements sit at near-ties?

Reads training/metrics/critic_step1_20261009/prior_gaps.games.jsonl: 100 side-games of parity3
(64 slots, decode raw:t0+eo-1.5) replayed on another box; per game, the decisions, how many the
replay's argmax reproduced ("agree"), premoves, and per decision the top-1 minus top-k log-prob
gaps of the replay's reading.

If disagreements come from arithmetic noise of size eps in the margin, a game's disagreement count
should track its number of decisions with top-two gap below about eps. The script finds the eps
whose near-tie count matches the total disagreements and reports the per-game correlation between
the two counts, against the correlation with the game's decision count alone.
"""
import json
import sys
from pathlib import Path

import numpy as np

GAMES = Path(__file__).resolve().parents[3] / "training/metrics/critic_step1_20261009/prior_gaps.games.jsonl"


def main():
    games = [json.loads(line) for line in open(GAMES)]
    dis = np.array([g["decisions"] - g["agree"] - g.get("premoves", 0) for g in games], float)
    n_dec = np.array([g["decisions"] for g in games], float)
    gaps = [np.array([r["gaps"][0] for r in g["readings"] if r["gaps"]]) for g in games]
    all_gaps = np.concatenate(gaps)
    print(f"side-games {len(games)}, decisions {int(n_dec.sum())}, disagreements {int(dis.sum())} "
          f"({dis.sum() / n_dec.sum():.4f}), multi-legal readings {len(all_gaps)}")
    for q in (0.001, 0.003, 0.01, 0.03, 0.1):
        print(f"  share of readings with top-two gap below {q:<5}: {np.mean(all_gaps < q):.4f}")
    # eps that matches the total: the disagreement share as a gap quantile
    eps = np.quantile(all_gaps, dis.sum() / len(all_gaps))
    near = np.array([np.sum(g < eps) for g in gaps], float)
    r_near = np.corrcoef(near, dis)[0, 1]
    r_dec = np.corrcoef(n_dec, dis)[0, 1]
    # partial: near-ties given decision count
    def resid(y, x):
        X = np.vstack([np.ones_like(x), x]).T
        return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    r_partial = np.corrcoef(resid(near, n_dec), resid(dis, n_dec))[0, 1]
    print(f"eps matching the total: {eps:.4f} nats; per-game corr(near-ties, disagreements) {r_near:.3f}, "
          f"corr(decisions, disagreements) {r_dec:.3f}, partial corr given decisions {r_partial:.3f} "
          f"(n = {len(games)}, null SE about {1 / np.sqrt(len(games) - 3):.3f})")


if __name__ == "__main__":
    sys.exit(main())
