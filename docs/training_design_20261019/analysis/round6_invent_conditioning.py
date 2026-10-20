"""Round 6 (invent stance): how much does conditioning a fight's price on the parent state remove,
among the hand columns the luck walk already computed?

The luck walk (training/metrics/match_luck_20261019/rows.jsonl, tools/match_luck.py) stores, per
game, the sum S_c of each column c's exactly centred luck terms. Two columns differ only in how
the price of a fight's outcome depends on the parent state:
- material: cost x HP fraction of the two fighters, one global price (no conditioning);
- value_50 / value_100 / value_200: tanh(board material / scale), so the price of the same
  material change depends on the board's material margin before the fight (one scalar of
  conditioning), with a terminal outcome scored +1 or -1.
This script fits, by leave-one-match-out least squares, G = z - beta . S for each column set and
reads Var(G)/Var(z) pooled over the 23 decisive matches, and the paired difference between sets
with a game bootstrap (betas held at their leave-one-match-out fits, so the bootstrap reads only
the sampling of games).

It carries no decision: it shows whether the parent-conditioning that the dice-priced innovations
learn has any measured footing in the record. numpy and the standard library only.
"""
import json
from pathlib import Path

import numpy as np

ROWS = Path(__file__).resolve().parents[3] / "training/metrics/match_luck_20261019/rows.jsonl"
SETS = {
    "material": ["material"],
    "material_terminal": ["material_nonterminal", "terminal"],
    "value_50": ["value_50"],
    "value_100": ["value_100"],
    "value_50+100+200": ["value_50", "value_100", "value_200"],
    "joint7": ["hp", "material", "material_nonterminal", "terminal", "value_50", "value_100", "value_200"],
}
BOOT = 2000


def read():
    z, match, cols = [], [], {c: [] for s in SETS.values() for c in s}
    for line in open(ROWS):
        r = json.loads(line)
        if not r.get("decisive"):
            continue
        z.append(float(r["z"]))
        match.append(r["match"])
        for c in cols:
            cols[c].append(float(r["S"][c]))
    return np.array(z), np.array(match), {c: np.array(v) for c, v in cols.items()}


def corrected(z, match, X):
    """G = z - X beta_(-m) for each game of match m, beta fitted on the other matches."""
    g = np.empty_like(z)
    for m in np.unique(match):
        out = match == m
        beta, *_ = np.linalg.lstsq(X[~out], z[~out], rcond=None)
        g[out] = z[out] - X[out] @ beta
    return g


def main():
    z, match, cols = read()
    var_z = z.var()
    g = {}
    for name, cs in SETS.items():
        X = np.column_stack([cols[c] for c in cs])
        g[name] = corrected(z, match, X)
    print(f"decisive games {len(z)}, matches {len(np.unique(match))}, Var(z) {var_z:.4f}")
    rng = np.random.default_rng(20261020)
    idx = rng.integers(0, len(z), size=(BOOT, len(z)))
    ratios = {n: np.array([(g[n][i] ** 2).mean() / z[i].var() for i in idx]) for n in SETS}
    for n in SETS:
        r = (g[n] ** 2).mean() / var_z
        print(f"  {n:18s} kept {r:.4f} (boot sd {ratios[n].std():.4f}); explained {1 - r:.4f}")
    for a, b in (("material", "value_50"), ("material", "joint7"), ("value_50", "joint7"),
                 ("material_terminal", "value_50")):
        d = ratios[b] - ratios[a]
        point = (g[b] ** 2).mean() / var_z - (g[a] ** 2).mean() / var_z
        print(f"  {b} minus {a}: {point:+.4f} +- {d.std():.4f} (paired game bootstrap)")
    sub = match == "gen_own1_selfplay"
    for n in ("material", "value_50", "joint7"):
        print(f"  own1 self-play only, {n}: kept {(g[n][sub] ** 2).mean() / z[sub].var():.4f}")


if __name__ == "__main__":
    main()
