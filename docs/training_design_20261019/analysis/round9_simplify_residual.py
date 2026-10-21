"""Round 9, simplify stance: the label residual when the pre-coin coefficients are fitted once and frozen.

The current design refits the setup-cell baseline b and the luck coefficients beta on batches 1..k-1
before batch k (current_design.md 3.2). This stance fits them once, on own1's 60,000 coin games of
batch 1 (training/metrics/near_tie_coins_20261020/rows_b0*.jsonl), and freezes them for every later
batch and round. Any coefficients fixed before a game is generated keep the label exact
(tools/coin_labels.py docstring, step 2), so the only question is the residual share
r = Var(z - b(cell) - beta.S) / Var(z) they leave on games they were not fitted on.

Reads the coin games' walk rows (decisive games; cell = scenario, side of A, faction of A, faction of B,
as round6_provenance_cells.py) and, for transfer to uncoined argmax games of other networks,
training/metrics/match_luck_20261019/rows.jsonl.

Reports out-of-sample residual shares, two folds:
  fold "batches": fit on b01-b03, test on b04-b06, and the reverse (b00 joins the second half);
  each with: beta of configs/luck_beta_20261019.json (what batch 1 used, b = 0); beta refit; beta refit
  plus raw cell means; plus an additive baseline (map x side + faction pair); plus cell means shrunk
  toward the additive baseline (one empirical-Bayes factor).
Then coefficients fitted on all 60,000 coin games, applied to the walk's own1 self-play (1,600 games),
to own2_64 vs own2_0 (805) and to all 23 matches (z centred by match, as round6_provenance_cells.py).
numpy and the standard library only.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
COIN = ROOT / "training/metrics/near_tie_coins_20261020"
WALK = ROOT / "training/metrics/match_luck_20261019/rows.jsonl"
BETA = ROOT / "configs/luck_beta_20261019.json"
RNG = np.random.default_rng(20261021)


def load(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    return [r for r in rows if r.get("decisive")]


def arrays(rows, columns):
    z = np.array([float(r["z"]) for r in rows])
    S = np.array([[float(r["S"][c]) for c in columns] for r in rows])
    cell = [(r["scenario_id"], int(r["side_a"]), r["faction_a"], r["faction_b"]) for r in rows]
    add_map = [(r["scenario_id"], int(r["side_a"])) for r in rows]
    add_fac = [(r["faction_a"], r["faction_b"]) for r in rows]
    return z, S, cell, add_map, add_fac


def fit_beta(z, S):
    X = np.column_stack([np.ones(len(z)), S])
    coef = np.linalg.lstsq(X, z, rcond=None)[0]
    return coef[0], coef[1:]


def group_means(keys, y):
    acc = defaultdict(lambda: [0.0, 0])
    for k, v in zip(keys, y):
        a = acc[k]
        a[0] += v
        a[1] += 1
    return {k: (s / n, n) for k, (s, n) in acc.items()}


def additive(train, y):
    """Two-way additive baseline (map x side, faction pair) by backfitting."""
    _, _, _, am, af = train
    a, f = defaultdict(float), defaultdict(float)
    for _ in range(20):
        ma = group_means(am, y - np.array([f[k] for k in af]))
        a = defaultdict(float, {k: m for k, (m, _) in ma.items()})
        mf = group_means(af, y - np.array([a[k] for k in am]))
        f = defaultdict(float, {k: m for k, (m, _) in mf.items()})
    return a, f


def predictors(train, beta_cfg):
    z, S, cell, am, af = train
    out = {}
    i0, b0 = beta_cfg
    out["beta_config"] = lambda t: i0 + t[1] @ b0
    i1, b1 = fit_beta(z, S)
    out["beta_refit"] = lambda t, i1=i1, b1=b1: i1 + t[1] @ b1
    y = z - i1 - S @ b1
    cm = group_means(cell, y)
    out["beta+cell"] = lambda t, cm=cm, i1=i1, b1=b1: i1 + t[1] @ b1 + np.array([cm.get(k, (0.0, 0))[0] for k in t[2]])
    a, f = additive(train, y)
    add_tr = np.array([a[k] for k in am]) + np.array([f[k] for k in af])
    out["beta+additive"] = lambda t, a=a, f=f, i1=i1, b1=b1: (i1 + t[1] @ b1 + np.array([a.get(k, 0.0) for k in t[3]])
                                                            + np.array([f.get(k, 0.0) for k in t[4]]))
    # cell deviation from the additive fit, shrunk by n / (n + k), k = within variance / between variance
    dev = y - add_tr
    dm = group_means(cell, dev)
    within = float(np.var(y - add_tr - np.array([dm[k][0] for k in cell])))
    ns = np.array([n for _, n in dm.values()])
    ms = np.array([m for m, _ in dm.values()])
    between = max(float(np.average(ms ** 2, weights=ns) - within * len(ns) / ns.sum()), 1e-9)
    kk = within / between
    shr = {c: m * n / (n + kk) for c, (m, n) in dm.items()}
    out["beta+shrunk_cell"] = lambda t, a=a, f=f, shr=shr, i1=i1, b1=b1: (
        i1 + t[1] @ b1 + np.array([a.get(k, 0.0) for k in t[3]]) + np.array([f.get(k, 0.0) for k in t[4]])
        + np.array([shr.get(k, 0.0) for k in t[2]]))
    return out, kk


def share(z, pred, var_ref):
    r = z - pred
    s = float(np.var(r)) / var_ref
    # SE of a variance ratio by bootstrap over games
    n = len(z)
    boots = []
    for _ in range(200):
        i = RNG.integers(0, n, n)
        boots.append(np.var(r[i]) / var_ref)
    return s, float(np.std(boots))


def subset(t, mask):
    z, S, cell, am, af = t
    idx = np.nonzero(mask)[0]
    return (z[idx], S[idx], [cell[i] for i in idx], [am[i] for i in idx], [af[i] for i in idx])


def main():
    cfg = json.loads(BETA.read_text())
    cols = cfg["columns"]
    beta_cfg = (cfg["intercept"], np.array(cfg["beta"]))
    names = [f"b0{k}" for k in range(7)]
    rows = {n: load([COIN / f"rows_{n}.jsonl"]) for n in names}
    print("decisive coin games per match:", {n: len(v) for n, v in rows.items()})
    first = rows["b01"] + rows["b02"] + rows["b03"]
    second = rows["b00"] + rows["b04"] + rows["b05"] + rows["b06"]
    fa, fb = arrays(first, cols), arrays(second, cols)
    allc = arrays(first + second, cols)
    vz = float(np.var(allc[0]))
    print(f"Var(z) over {len(allc[0])} coin games: {vz:.4f}; mean z {allc[0].mean():+.4f}")
    print(f"cells observed: {len(set(allc[2]))}")
    print("\nOut-of-sample residual share (Var(z - prediction) / Var(z)), fitted on one half, read on the other:")
    res = defaultdict(list)
    for tr, te, lab in ((fa, fb, "fit b01-b03, read b00+b04-b06"), (fb, fa, "fit b00+b04-b06, read b01-b03")):
        preds, kk = predictors(tr, beta_cfg)
        vt = float(np.var(te[0]))
        print(f"  {lab} (shrinkage constant k = {kk:.1f} games):")
        for name, fn in preds.items():
            s, se = share(te[0], fn(te), vt)
            res[name].append((s, se, len(te[0])))
            print(f"    {name:18s} {s:.4f} +- {se:.4f}")
    print("  pooled over the two folds (games-weighted):")
    for name, v in res.items():
        w = np.array([n for _, _, n in v], float)
        s = float(np.average([a for a, _, _ in v], weights=w))
        se = float(np.sqrt(np.sum((w / w.sum()) ** 2 * np.array([b for _, b, _ in v]) ** 2)))
        print(f"    {name:18s} {s:.4f} +- {se:.4f}")

    print("\nCoefficients fitted on all 60,000 coin games, frozen, read on the walk's uncoined argmax games:")
    preds, kk = predictors(allc, beta_cfg)
    walk = load([WALK])
    wz, wS, wcell, wam, waf = arrays(walk, cols)
    match = np.array([r["match"] for r in walk])
    zc = wz.copy()
    for m in set(match):
        zc[match == m] -= wz[match == m].mean()
    wt = (zc, wS, wcell, wam, waf)
    sets = {
        "own1 self-play (gen_own1_selfplay, own64_selfplay)": np.isin(match, ["gen_own1_selfplay", "own64_selfplay"]),
        "own2_64 vs own2_0 (r3)": match == "r3_own2_64_vs_own2_0",
        "own2_64 vs own1_64 (r1)": match == "r1_own2_64_vs_own1_64",
        "all 23 matches": np.ones(len(zc), bool),
    }
    for lab, mask in sets.items():
        t = subset(wt, mask)
        vt = float(np.var(t[0]))
        line = ", ".join(f"{n} {share(t[0], fn(t) - np.mean(fn(t)), vt)[0]:.3f}" for n, fn in preds.items())
        print(f"  {lab}, {mask.sum()} games: {line}")
    print("  (SE of a share on 800 games is about 0.04, on 1,600 about 0.03)")


if __name__ == "__main__":
    main()
