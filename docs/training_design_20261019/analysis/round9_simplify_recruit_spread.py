"""Round 9, simplify stance: the near-tie spread among recruit choices, and how much of it a head
of the design's form can express.

Input: the per-(cell, fold) label sums that round9_provenance_cells.py (provenance stance) extracted
from batch 1's 60,000 own1 coin games (round9_provenance_cells_b0*.json, partition "rr_units": cell =
mover's faction | top recruit's unit > runner-up's unit, for band decisions whose top two are
recruits of different units; every other decision in one cell "rest"). Labels T = s omega (z - beta.S),
b = 0, beta of batch 1 (readout_b.json); folds are 20 groups of games by SHA-256 of the game name, so
labels in different folds are independent.

Within the recruit cells only ("rest" excluded), this script reports:
  1. V_rr, the variance of the cell means of the true advantage among these decisions (weights: the
     cells' shares of them), by the U-statistic over distinct folds, delete-one-fold jackknife SE.
     A lower bound on the variance of the per-decision advantage there (law of total variance).
  2. The same for the coarser partition "mover's faction" within these decisions.
  3. The additive antisymmetric part: the head's form is f = h(a1) - h(a0) with h = w.phi(a), so with
     unit identity in phi it can express mu(fac, u0 > u1) = c + h(fac, u1) - h(fac, u0). Fitted by
     weighted least squares on one half of the folds, read against the other half's cell means
     (cross-half inner product, both directions): the spread the additive form captures, against V_rr.
  4. Cross-fitted first-order welfare per side-game (Elo, x 800/ln 10) of three switch rules on these
     decisions, each fold's rule chosen on the other 19 folds and scored on its own labels:
     switch all; switch a cell where its out-of-fold mean is above 0; switch where the out-of-fold
     additive fit is above 0. SE from the labels' second moment (labels within a game are
     uncorrelated: round9_simplify_labels.out, ratio 1.010).
numpy and the standard library only; reads about 13 MB of JSON.
"""
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MATCHES = ("b00", "b01", "b02", "b03", "b04", "b05", "b06")
FOLDS = 20
ELO = 800 / math.log(10)


def load():
    data = [json.loads((HERE / f"round9_provenance_cells_{m}.json").read_text()) for m in MATCHES]
    cells = sorted({c for d in data for c in d["partitions"]["rr_units"]["cells"]} - {"rest"})
    at = {c: i for i, c in enumerate(cells)}
    acc = {k: np.zeros((len(cells), FOLDS)) for k in ("n", "s_T", "ss_T")}
    for d in data:
        p = d["partitions"]["rr_units"]
        for j, c in enumerate(p["cells"]):
            if c == "rest":
                continue
            for k in acc:
                acc[k][at[c]] += np.asarray(p[k][j])
    side_games = sum(d["side_games"] for d in data)
    return cells, acc, side_games


def u_square(s, n):
    num = s.sum(1) ** 2 - (s * s).sum(1)
    den = n.sum(1) ** 2 - (n * n).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, 0.0)


def v_between(s, n):
    pi = n.sum(1) / n.sum()
    tot = s.sum(0, keepdims=True)
    return float(np.sum(pi * u_square(s, n))) - float(u_square(tot, n.sum(0, keepdims=True))[0])


def jackknife(fn, *arrays):
    full = fn(*arrays)
    reps = np.array([fn(*[a[:, [k for k in range(FOLDS) if k != f]] for a in arrays]) for f in range(FOLDS)])
    return full, math.sqrt((FOLDS - 1) / FOLDS * np.sum((reps - reps.mean()) ** 2))


def coarsen(cells, s, n, key):
    groups = sorted({key(c) for c in cells})
    at = {g: i for i, g in enumerate(groups)}
    S, N = np.zeros((len(groups), FOLDS)), np.zeros((len(groups), FOLDS))
    for i, c in enumerate(cells):
        S[at[key(c)]] += s[i]
        N[at[key(c)]] += n[i]
    return S, N


def parse(cell):
    fac, pair = cell.split("|", 1)
    u0, u1 = pair.split(">", 1)
    return fac, u0, u1


def additive_fit(cells, s, n):
    """Weighted LS of the cell means on c + h(fac, u1) - h(fac, u0); returns the fitted cell function."""
    units = sorted({(parse(c)[0], u) for c in cells for u in parse(c)[1:]})
    at = {u: i for i, u in enumerate(units)}
    X = np.zeros((len(cells), len(units) + 1))
    X[:, -1] = 1.0
    for i, c in enumerate(cells):
        fac, u0, u1 = parse(c)
        X[i, at[(fac, u1)]] += 1.0
        X[i, at[(fac, u0)]] -= 1.0
    w = n.sum(1)
    y = np.where(w > 0, s.sum(1) / np.maximum(w, 1), 0.0)
    sw = np.sqrt(w)
    coef = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    return X @ coef, coef[-1]


def cross_half_additive(cells, s, n):
    """Cross-half inner product of the additive fits' centred cell functions, against the cell means'
    cross-half inner product (an estimate of V_rr on the same split)."""
    halves = (list(range(0, FOLDS, 2)), list(range(1, FOLDS, 2)))
    fits, means, weights = [], [], []
    for h in halves:
        sh, nh = s[:, h], n[:, h]
        fits.append(additive_fit(cells, sh, nh)[0])
        means.append(np.where(nh.sum(1) > 0, sh.sum(1) / np.maximum(nh.sum(1), 1), 0.0))
        weights.append(nh.sum(1))
    pi = (weights[0] + weights[1]) / (weights[0] + weights[1]).sum()

    def centred(v):
        return v - np.sum(pi * v)
    add_ip = float(np.sum(pi * centred(fits[0]) * centred(fits[1])))
    mean_ip = float(np.sum(pi * centred(means[0]) * centred(means[1])))
    return add_ip, mean_ip


def welfare(cells, s, ss, n, side_games, rule):
    total = var = switched = 0.0
    for f in range(FOLDS):
        keep = [k for k in range(FOLDS) if k != f]
        choose = rule(cells, s[:, keep], n[:, keep])
        total += s[choose, f].sum()
        var += ss[choose, f].sum()
        switched += n[choose, f].sum()
    return total / side_games * ELO, math.sqrt(var) / side_games * ELO, switched / n.sum()


def rule_all(cells, s, n):
    return np.ones(len(cells), bool)


def rule_cell(cells, s, n):
    return s.sum(1) > 0


def rule_additive(cells, s, n):
    return additive_fit(cells, s, n)[0] > 0


def load_by_match(label):
    """Per match: (cells, s, ss, n, side_games) for the recruit cells, cells aligned across matches."""
    data = [json.loads((HERE / f"round9_provenance_cells_{m}.json").read_text()) for m in MATCHES]
    cells = sorted({c for d in data for c in d["partitions"]["rr_units"]["cells"]} - {"rest"})
    at = {c: i for i, c in enumerate(cells)}
    out = []
    for d in data:
        p = d["partitions"]["rr_units"]
        s, ss, n = (np.zeros(len(cells)) for _ in range(3))
        for j, c in enumerate(p["cells"]):
            if c == "rest":
                continue
            s[at[c]] += np.sum(p[f"s_{label}"][j])
            ss[at[c]] += np.sum(p[f"ss_{label}"][j])
            n[at[c]] += np.sum(p["n"][j])
        out.append((d["match"], s, ss, n, d["side_games"]))
    return cells, out


def robustness():
    """Leave-one-match-out cell rule, for T and for T0 (no luck columns); then the welfare by faction."""
    for label in ("T", "T0"):
        cells, per = load_by_match(label)
        tot = totv = 0.0
        sg_all = sum(x[4] for x in per)
        parts = []
        for k, (m, s, ss, n, sg) in enumerate(per):
            so = sum(x[1] for j, x in enumerate(per) if j != k)
            choose = so > 0
            tot += s[choose].sum()
            totv += ss[choose].sum()
            parts.append(f"{m} {s[choose].sum() / sg * ELO:+.1f}")
        print(f"5. [{label:2s}] leave-one-match-out cell rule: {tot / sg_all * ELO:+.1f} +- "
              f"{math.sqrt(totv) / sg_all * ELO:.1f} Elo per side-game; by held-out match: {', '.join(parts)}")
    cells, per = load_by_match("T")
    for fit_on, read_on in ((("b00", "b01", "b02", "b03"), ("b04", "b05", "b06")),
                            (("b04", "b05", "b06"), ("b00", "b01", "b02", "b03"))):
        sf = sum(x[1] for x in per if x[0] in fit_on)
        choose = sf > 0
        s_r = sum(x[1] for x in per if x[0] in read_on)
        ss_r = sum(x[2] for x in per if x[0] in read_on)
        n_r = sum(x[3] for x in per if x[0] in read_on)
        sg_r = sum(x[4] for x in per if x[0] in read_on)
        print(f"7. a frozen rule (cells with mean > 0 on {'+'.join(fit_on)}), read on {'+'.join(read_on)} "
              f"({sg_r // 2} games): {s_r[choose].sum() / sg_r * ELO:+.1f} +- "
              f"{math.sqrt(ss_r[choose].sum()) / sg_r * ELO:.1f} Elo per side-game; "
              f"{choose.sum()} of {len(cells)} cells, {n_r[choose].sum() / sg_r:.2f} switches per side-game")
        nf = sum(x[3] for x in per if x[0] in fit_on)
        fitted, const = additive_fit(cells, sf[:, None], nf[:, None])
        add = fitted > 0
        n_units = len({(parse(c)[0], u) for c in cells for u in parse(c)[1:]})
        print(f"   the same, additive form c + h(fac, u1) - h(fac, u0) ({n_units} unit values, c {const:+.4f}): "
              f"{s_r[add].sum() / sg_r * ELO:+.1f} +- {math.sqrt(ss_r[add].sum()) / sg_r * ELO:.1f} Elo per side-game; "
              f"{n_r[add].sum() / sg_r:.2f} switches per side-game")
    cells, a, sg = load()
    s, ss, n = a["s_T"], a["ss_T"], a["n"]
    facs = sorted({parse(c)[0] for c in cells})
    for fac in facs:
        idx = np.array([parse(c)[0] == fac for c in cells])
        tot = var = 0.0
        for f in range(FOLDS):
            keep = [k for k in range(FOLDS) if k != f]
            choose = idx & (s[:, keep].sum(1) > 0)
            tot += s[choose, f].sum()
            var += ss[choose, f].sum()
        print(f"6. {fac:18s}: {n[idx].sum() / sg:.2f} such decisions per side-game (all factions' side-games); "
              f"cell rule {tot / sg * ELO:+.1f} +- {math.sqrt(var) / sg * ELO:.1f} Elo per side-game")


def main():
    cells, a, sg = load()
    s, ss, n = a["s_T"], a["ss_T"], a["n"]
    nd = n.sum()
    print(f"recruit>recruit, different units: {int(nd)} band decisions in {sg} side-games "
          f"({nd / sg:.2f} per side-game), {len(cells)} cells (faction | top unit > runner-up unit)")
    mean = s.sum() / nd
    print(f"mean label {mean:+.4f} +- {math.sqrt(ss.sum()) / nd:.4f}")
    vb, se = jackknife(v_between, s, n)
    print(f"1. V_rr (variance of cell means within these decisions) {vb:.3e} +- {se:.2e} (z {vb / se:.2f}); "
          f"sd of cell means {math.sqrt(max(vb, 0)):.4f}, at least {math.sqrt(max(vb - 2 * se, 0)):.4f} at 2 SE")
    Sf, Nf = coarsen(cells, s, n, lambda c: parse(c)[0])
    vf, sef = jackknife(v_between, Sf, Nf)
    print(f"2. mover's faction alone (6 cells): {vf:.3e} +- {sef:.2e} (z {vf / sef:.2f})")
    add_ip, mean_ip = cross_half_additive(cells, s, n)
    print(f"3. cross-half inner products: additive fits {add_ip:.3e}; cell means {mean_ip:.3e}; "
          f"additive share {add_ip / mean_ip:.2f} (no SE: one split; a ratio of noisy inner products)")
    for name, rule in (("switch all", rule_all), ("switch cells with out-of-fold mean > 0", rule_cell),
                       ("switch where the out-of-fold additive fit > 0", rule_additive)):
        w, wse, share = welfare(cells, s, ss, n, sg, rule)
        print(f"4. {name:46s}: {w:+.1f} +- {wse:.1f} Elo per side-game (z {w / wse:+.2f}), "
              f"share of these decisions switched {share:.3f}")
    robustness()


if __name__ == "__main__":
    main()
