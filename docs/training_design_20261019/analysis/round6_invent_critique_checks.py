"""Round 6 (critique of the invent stance): three checks on numbers round6_invent.md rests on.

1. Is value_50's gain over material conditioning on the board, or its terminal term? Adds the set
   material + terminal (both fitted) to round6_invent_conditioning.py's leave-one-match-out fits.
2. The resolution of S0(d)'s gate: the paired SE of a kept-ratio difference on own1 self-play
   alone (800 games), where the gate is read, against the pooled 18,800-game SEs the stance cites.
3. S0(d)'s forward count with the decisive games' own mean turns (the stance multiplies the
   decisive count by the mean turns of all 19,466 walked games).

Reads training/metrics/match_luck_20261019/rows.jsonl. numpy and the standard library only.
"""
import json
from pathlib import Path

import numpy as np

ROWS = Path(__file__).resolve().parents[3] / "training/metrics/match_luck_20261019/rows.jsonl"
SETS = {
    "material": ["material"],
    "material+terminal": ["material", "terminal"],
    "value_50": ["value_50"],
    "joint7": ["hp", "material", "material_nonterminal", "terminal", "value_50", "value_100", "value_200"],
}
BOOT = 2000
OWN1 = "gen_own1_selfplay"


def read():
    rows = [json.loads(line) for line in open(ROWS)]
    dec = [r for r in rows if r.get("decisive")]
    z = np.array([float(r["z"]) for r in dec])
    match = np.array([r["match"] for r in dec])
    cols = {c: np.array([float(r["S"][c]) for r in dec]) for s in SETS.values() for c in s}
    turns_all = np.mean([r["turns"] for r in rows])
    turns_dec = np.mean([r["turns"] for r in dec])
    turns_own1 = np.mean([r["turns"] for r in dec if r["match"] == OWN1])
    return z, match, cols, (len(rows), turns_all, turns_dec, turns_own1)


def corrected(z, match, X):
    g = np.empty_like(z)
    for m in np.unique(match):
        out = match == m
        beta, *_ = np.linalg.lstsq(X[~out], z[~out], rcond=None)
        g[out] = z[out] - X[out] @ beta
    return g


def kept(g, z, idx):
    return np.array([(g[i] ** 2).mean() / z[i].var() for i in idx])


def main():
    z, match, cols, (n_all, t_all, t_dec, t_own1) = read()
    g = {n: corrected(z, match, np.column_stack([cols[c] for c in cs])) for n, cs in SETS.items()}
    rng = np.random.default_rng(20261020)
    idx = rng.integers(0, len(z), size=(BOOT, len(z)))
    sub = np.flatnonzero(match == OWN1)
    idx_own = sub[rng.integers(0, len(sub), size=(BOOT, len(sub)))]
    print("1. pooled kept ratios (leave-one-match-out), 18,800 games")
    pooled = {n: kept(g[n], z, idx) for n in SETS}
    for n in SETS:
        print(f"   {n:18s} {(g[n] ** 2).mean() / z.var():.4f} (boot sd {pooled[n].std():.4f})")
    for a, b in (("material", "material+terminal"), ("material+terminal", "value_50")):
        d = pooled[b] - pooled[a]
        pt = (g[b] ** 2).mean() / z.var() - (g[a] ** 2).mean() / z.var()
        print(f"   {b} minus {a}: {pt:+.4f} +- {d.std():.4f}")
    print(f"2. own1 self-play only ({len(sub)} games): kept ratios and paired SEs")
    own = {n: kept(g[n], z, idx_own) for n in SETS}
    for n in SETS:
        print(f"   {n:18s} {(g[n][sub] ** 2).mean() / z[sub].var():.4f} (boot sd {own[n].std():.4f})")
    for a, b in (("material", "value_50"), ("value_50", "joint7"), ("material", "joint7")):
        d = own[b] - own[a]
        print(f"   {b} minus {a}: paired SE {d.std():.4f} (pooled 18,800: {(pooled[b] - pooled[a]).std():.4f})")
    print("3. S0(d) forwards at 449.1 decisions per own1 game of 20.13 turns")
    for label, games, turns in (("stance: 18,800 x all-game turns", 18800, t_all),
                                ("decisive games, their turns", 18800, t_dec),
                                ("every walked game, all turns", n_all, t_all)):
        fw = games * 449.1 * turns / t_own1
        print(f"   {label:34s} mean turns {turns:.2f}: {fw / 1e6:.2f} M forwards, "
              f"{fw / 2440 / 3600:.2f} h, ${fw / 2440 / 3600 * 0.42:.2f}")


if __name__ == "__main__":
    main()
