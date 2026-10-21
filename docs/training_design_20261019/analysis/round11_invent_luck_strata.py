"""Round 11, invent stance: do the luck walk's coefficients vary with the setup?

The design's chance head makes the luck coefficients a function of the position, beta(h), instead of
one constant vector. The coarsest version that existing records can test is beta varying with the
setup (map, faction of A, faction of B), which is fixed before every attack, so the control variate
keeps its zero mean. This script reads own1's 60,000 coin games of batch 1
(training/metrics/near_tie_coins_20261020/rows_b0*.jsonl: z, setup, the seven luck sums S) and asks
whether setup-interacted coefficients lower the out-of-sample residual share below the pooled beta's.

Folds as round9_simplify_residual.py (fit b01-b03, read b00+b04-b06, and the reverse).
Base: pooled beta refit plus the shrunk setup cell (round9_simplify_residual.predictors, 0.699 pooled).
Extensions, each a ridge on the base's residual, its penalty chosen by an inner two-way split of the
fitting half (grid below), then read on the other half:
  fa:   S x one-hot(faction of A)            (6 x 7 columns)
  fb:   S x one-hot(faction of B)            (6 x 7)
  map:  S x one-hot(scenario)                (21 x 7)
  all:  the three together                   (231 columns)
Reports each extension's residual share, its paired improvement over the base on the read games
(bootstrap SE over games), and the per-faction spread of the terminal and value_50 coefficients.
numpy and the standard library only.
"""
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # the laptop's threaded BLAS is 27x slower on these sizes

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COIN = ROOT / "training/metrics/near_tie_coins_20261020"
BETA = ROOT / "configs/luck_beta_20261019.json"
RNG = np.random.default_rng(20261021)
LAMBDAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)

spec = importlib.util.spec_from_file_location("res", HERE / "round9_simplify_residual.py")
res = importlib.util.module_from_spec(spec)
spec.loader.exec_module(res)


def setup_keys(rows):
    return ([r["faction_a"] for r in rows], [r["faction_b"] for r in rows], [r["scenario_id"] for r in rows])


def onehot(keys, levels):
    idx = {k: i for i, k in enumerate(levels)}
    out = np.zeros((len(keys), len(levels)))
    for n, k in enumerate(keys):
        if k in idx:
            out[n, idx[k]] = 1.0
    return out


def interactions(S_std, keys, levels):
    H = onehot(keys, levels)
    return (H[:, :, None] * S_std[:, None, :]).reshape(len(S_std), -1)


def ridge(X, y, lam):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)


def choose_lambda(X, y):
    n = len(y)
    perm = RNG.permutation(n)
    a, b = perm[: n // 2], perm[n // 2:]
    best = None
    for lam in LAMBDAS:
        loss = 0.0
        for tr, te in ((a, b), (b, a)):
            w = ridge(X[tr], y[tr], lam)
            loss += float(np.sum((y[te] - X[te] @ w) ** 2))
        if best is None or loss < best[1]:
            best = (lam, loss)
    return best[0]


def blocks(rows, S_std, levels):
    fa, fb, mp = setup_keys(rows)
    out = {
        "fa": interactions(S_std, fa, levels["fa"]),
        "fb": interactions(S_std, fb, levels["fb"]),
        "map": interactions(S_std, mp, levels["map"]),
    }
    out["all"] = np.hstack([out["fa"], out["fb"], out["map"]])
    return out


def paired(base_r, ext_r, var_ref):
    d = (base_r ** 2 - ext_r ** 2) / var_ref
    n = len(d)
    boots = [float(np.mean(d[RNG.integers(0, n, n)])) for _ in range(300)]
    return float(np.mean(d)), float(np.std(boots))


def main():
    cfg = json.loads(BETA.read_text())
    cols = cfg["columns"]
    beta_cfg = (cfg["intercept"], np.array(cfg["beta"]))
    rows = {f"b0{k}": res.load([COIN / f"rows_b0{k}.jsonl"]) for k in range(7)}
    first = rows["b01"] + rows["b02"] + rows["b03"]
    second = rows["b00"] + rows["b04"] + rows["b05"] + rows["b06"]
    every = first + second
    levels = {
        "fa": sorted(set(r["faction_a"] for r in every)),
        "fb": sorted(set(r["faction_b"] for r in every)),
        "map": sorted(set(r["scenario_id"] for r in every)),
    }
    S_all = np.array([[float(r["S"][c]) for c in cols] for r in every])
    mu, sd = S_all.mean(0), S_all.std(0)
    print(f"games: {len(every)}; factions {len(levels['fa'])}, scenarios {len(levels['map'])}; columns {cols}")
    acc = {k: [] for k in ("base", "fa", "fb", "map", "all")}
    gains = {k: [] for k in ("fa", "fb", "map", "all")}
    for tr_rows, te_rows, lab in ((first, second, "fit b01-b03, read b00+b04-b06"),
                                  (second, first, "fit b00+b04-b06, read b01-b03")):
        tr, te = res.arrays(tr_rows, cols), res.arrays(te_rows, cols)
        preds, _ = res.predictors(tr, beta_cfg)
        base = preds["beta+shrunk_cell"]
        y_tr = tr[0] - base(tr)
        r_te = te[0] - base(te)
        vt = float(np.var(te[0]))
        acc["base"].append((float(np.var(r_te)) / vt, len(te[0])))
        Btr = blocks(tr_rows, (tr[1] - mu) / sd, levels)
        Bte = blocks(te_rows, (te[1] - mu) / sd, levels)
        line = [f"base {acc['base'][-1][0]:.4f}"]
        for name in ("fa", "fb", "map", "all"):
            lam = choose_lambda(Btr[name], y_tr)
            w = ridge(Btr[name], y_tr, lam)
            ext = r_te - Bte[name] @ w
            s = float(np.var(ext)) / vt
            g, gse = paired(r_te, ext, vt)
            acc[name].append((s, len(te[0])))
            gains[name].append((g, gse, len(te[0])))
            line.append(f"{name} {s:.4f} (lambda {lam:g}, gain {g:+.4f} +- {gse:.4f})")
        print(f"  {lab}: " + "; ".join(line))
    print("pooled over the two folds (games-weighted residual share; gain = base share - extension share):")
    for name, v in acc.items():
        w = np.array([n for _, n in v], float)
        s = float(np.average([a for a, _ in v], weights=w))
        extra = ""
        if name in gains:
            g = np.array([a for a, _, _ in gains[name]])
            ge = np.array([b for _, b, _ in gains[name]])
            ww = w / w.sum()
            extra = f"   gain {float(ww @ g):+.4f} +- {float(np.sqrt(np.sum(ww ** 2 * ge ** 2))):.4f}"
        print(f"  {name:5s} {s:.4f}{extra}")

    # Heterogeneity magnitude: per-faction-of-A coefficients on the full data, unpenalised, raw units.
    t = res.arrays(every, cols)
    fa, _, _ = setup_keys(every)
    fa = np.array(fa)
    print("\nper faction of A, OLS of z on [1, S] (raw units), the terminal and value_50 coefficients:")
    for f in levels["fa"]:
        m = fa == f
        X = np.column_stack([np.ones(m.sum()), t[1][m]])
        coef, *_ = np.linalg.lstsq(X, t[0][m], rcond=None)
        resid = t[0][m] - X @ coef
        cov = np.linalg.inv(X.T @ X) * float(np.var(resid))
        ti, vi = 1 + cols.index("terminal"), 1 + cols.index("value_50")
        print(f"  {f:18s} n {m.sum():6d}  terminal {coef[ti]:+.4f} +- {np.sqrt(cov[ti, ti]):.4f}"
              f"  value_50 {coef[vi]:+.4f} +- {np.sqrt(cov[vi, vi]):.4f}")


if __name__ == "__main__":
    main()
