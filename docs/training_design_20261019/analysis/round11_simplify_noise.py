"""Round 11, simplify stance: the noise scale of an outcome-driven policy
gradient on real self-play results, measured on the one direction the record
has confirmed in play (recruit preferences between two units, match L).

Reads round10_invent_recruit_rows_<match>_p0.json.gz (batch 1 of the near-tie
coins: own1 self-play at argmax with coins, 60,000 decisive games; every band
decision whose two candidates recruit different units, with its label
T = s * omega * R, omega = (D - P) / (P (1 - P)), R the luck- and
cell-corrected result).

For a logistic coin policy P = sigmoid(w . x), the REINFORCE term of one band
decision is s (D - P) x R = P (1 - P) T x; P is 1/4 at every decision, so
the per-game gradient is v_g = sum over the game's rows of T x up to one
constant, which cancels in every ratio below. Two feature sets:
  R43: the 43 (faction, unit) columns, x = e(u1) - e(u0) (the design's R);
  d:   the one count column, x = n1 - n0 (the side's earlier recruits of a1's
       type less a0's).
Readings (McCandlish et al. 2018, arXiv 1812.06162, eq. 2.9 and app. A):
  |G|^2 unbiased = (|sum v|^2 - sum |v|^2) / (n (n - 1)), n = all 60,000 games
                   (games without a row contribute v = 0);
  tr(Sigma)      = per-game covariance trace;
  B_simple       = tr(Sigma) / |G|^2, in games;
  the split-half dot product gA . gB at h games per half (random disjoint
  halves drawn from the 60,000, 200 draws each), its mean / sd: the z at which
  a batch of 2h games resolves the gradient from zero. The draws share one
  finite population, so the sd is understated by about sqrt(1 - 2h/n); the
  printed z_fpc divides the sd by that factor (h <= 5,000 keeps it near 1).
Jackknife SEs over 20 folds (crc32 of the game key). Numpy and the standard
library only.
"""
from __future__ import annotations

import gzip
import json
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MATCHES = ("b00", "b01", "b02", "b03", "b04", "b05", "b06")
FOLDS = 20
DRAWS = 200


def load():
    cols = defaultdict(list)
    n_games = 0
    for m in MATCHES:
        d = json.loads(gzip.decompress((HERE / f"round10_invent_recruit_rows_{m}_p0.json.gz").read_bytes()))
        n_games += int(d["games"])
        for k, v in d["rows"].items():
            cols[k].extend(v)
    return cols, n_games


def per_game_vectors(cols, n_games):
    games = np.array(cols["game"])
    keys, gid = np.unique(games, return_inverse=True)
    T = np.array(cols["T"], dtype=np.float64)
    names = sorted(set(f + "|" + u for f, u in zip(cols["fac"], cols["u0"])) |
                   set(f + "|" + u for f, u in zip(cols["fac"], cols["u1"])))
    at = {u: i for i, u in enumerate(names)}
    X = np.zeros((len(T), len(names)))
    rows = np.arange(len(T))
    X[rows, [at[f + "|" + u] for f, u in zip(cols["fac"], cols["u1"])]] += 1.0
    X[rows, [at[f + "|" + u] for f, u in zip(cols["fac"], cols["u0"])]] -= 1.0
    d = (np.array(cols["n1"]) - np.array(cols["n0"])).astype(np.float64)[:, None]
    out = {}
    for name, F in (("R43", X), ("d", d)):
        V = np.zeros((n_games, F.shape[1]))
        # games with rows take the first len(keys) slots; the rest are zero vectors
        np.add.at(V, gid, T[:, None] * F)
        out[name] = V
    folds = np.array([zlib.crc32(k.encode()) % FOLDS for k in keys] +
                     [i % FOLDS for i in range(n_games - len(keys))])
    return out, folds, len(keys), len(T)


def stats(V):
    n = V.shape[0]
    s = V.sum(0)
    sq = float((V * V).sum())
    g2 = (float(s @ s) - sq) / (n * (n - 1))
    mean = s / n
    trS = (sq - n * float(mean @ mean)) / (n - 1)
    return g2, trS


def jackknife(V, folds):
    full = stats(V)
    reps = [stats(V[folds != f]) for f in range(FOLDS)]
    out = []
    for j, val in enumerate(full):
        r = np.array([x[j] for x in reps])
        se = float(np.sqrt((FOLDS - 1) / FOLDS * ((r - r.mean()) ** 2).sum()))
        out.append((val, se))
    return out


def split_half(V, h, rng):
    n = V.shape[0]
    dots = np.empty(DRAWS)
    for k in range(DRAWS):
        idx = rng.choice(n, size=2 * h, replace=False)
        a = V[idx[:h]].mean(0)
        b = V[idx[h:]].mean(0)
        dots[k] = float(a @ b)
    return float(dots.mean()), float(dots.std())


def main():
    cols, n_games = load()
    vecs, folds, with_rows, n_rows = per_game_vectors(cols, n_games)
    print(f"games {n_games}; with a recruit-pair row {with_rows}; rows {n_rows}")
    T = np.array(cols["T"])
    print(f"mean T over rows {T.mean():+.5f}; mean T^2 {np.mean(T * T):.3f}")
    rng = np.random.default_rng(20261021)
    for name, V in vecs.items():
        (g2, g2_se), (trS, trS_se) = jackknife(V, folds)
        print(f"\n== {name}: {V.shape[1]} columns")
        print(f"  |G|^2 unbiased {g2:.4e} +- {g2_se:.4e} (z {g2 / g2_se:.2f})")
        print(f"  tr(Sigma) per game {trS:.4e} +- {trS_se:.4e}")
        if g2 > 0:
            b = trS / g2
            lo = trS / (g2 + 2 * g2_se)
            hi = trS / (g2 - 2 * g2_se) if g2 - 2 * g2_se > 0 else float("inf")
            print(f"  B_simple {b:,.0f} games (|G|^2 +- 2 SE: {lo:,.0f} to {hi:,.0f})")
        for h in (1_000, 2_500, 5_000):
            m, sd = split_half(V, h, rng)
            sd_fpc = sd / np.sqrt(1.0 - 2.0 * h / V.shape[0])
            print(f"  split halves of {h:>6,} games: mean gA.gB {m:.3e}, sd {sd:.3e}, z_fpc {m / sd_fpc:.2f}")


if __name__ == "__main__":
    main()
