"""Round 11 (challenge stance): what a fork's exact baseline buys over the free value-head baseline.

Reads training/metrics/turn_value_20260925/bench_truth.npz (199 human positions, about 5 candidate
turns each, 28 playouts per candidate to the game's end at temperature 0.5 by obs8; the playouts'
two luck columns; obs8's value head read at the position, `value_reference`, and after the
candidate, `value_post`).

For a decision taken at such a position, its policy-gradient label is z - b. Its noise (everything
but the decision's own effect) is the continuation's variance plus the part of the position's value
the baseline b misses:
  none        b = 0                                   within + var(V)
  value head  b = a + c * v_head (cross-fitted)       within + var(V - b)
  fork K      b = mean of K - 1 sibling playouts      within * K / (K - 1)
  pre-fork    the decision just before a K-way fork, label = mean of the K results minus the
              value-head baseline: var(V - b) + within / K
`within` is the pooled within-candidate variance of one playout (raw, or luck-corrected with a
luck slope fitted on the other half of the positions); var(V) is the variance of candidate means
less within / 28. The value-head map (a, c) is fitted on one half of the positions and scored on
the other, then the halves swap (cross-fitting), so a useless head scores near zero.

Transferred conditions (human positions, obs8, temperature 0.5 playouts): a report for the design,
not a decision; the T = 1 decomposition on own2 is measured by the design's stage 1.
numpy and the standard library only.
"""
from pathlib import Path

import numpy as np

NPZ = Path(__file__).resolve().parents[3] / "training/metrics/turn_value_20260925/bench_truth.npz"
RNG = np.random.default_rng(20261021)
KS = (2, 4, 8, 16)


def halves(index, rng):
    positions = np.unique(index)
    rng.shuffle(positions)
    a = set(positions[: len(positions) // 2].tolist())
    return np.array([p in a for p in index])


def luck_corrected(outcomes, luck, mask):
    """Outcomes less a luck slope fitted within candidates on the other half (cross-fitted)."""
    out = outcomes.copy()
    for fit_on in (mask, ~mask):
        y = (outcomes[fit_on] - outcomes[fit_on].mean(axis=1, keepdims=True)).reshape(-1)
        L = luck[fit_on].reshape(-1, 2)
        Lc = L - L.mean(0)
        b = np.linalg.lstsq(Lc, y, rcond=None)[0]
        out[~fit_on] = outcomes[~fit_on] - luck[~fit_on] @ b
    return out


def decompose(out, v_head, mask):
    k = out.shape[1]
    means = out.mean(axis=1)
    within = out.var(axis=1, ddof=1).mean()
    var_v = means.var() - within / k
    resid = np.empty_like(means)
    for fit_on in (mask, ~mask):
        X = np.c_[np.ones(fit_on.sum()), v_head[fit_on]]
        coef = np.linalg.lstsq(X, means[fit_on], rcond=None)[0]
        resid[~fit_on] = means[~fit_on] - np.c_[np.ones((~fit_on).sum()), v_head[~fit_on]] @ coef
    var_resid = resid.var() - within / k  # the baseline's miss on V, playout noise of the mean removed
    return within, var_v, var_resid


def report(name, within, var_v, var_resid):
    total = within + var_v
    none = within + var_v
    head = within + var_resid
    lines = [f"{name}: one playout {total:.3f} = within {within:.3f} + var(V) {var_v:.3f}; "
             f"value head misses {var_resid:.3f} of var(V) (explains {1 - var_resid / var_v:.2f} of it)"]
    lines.append(f"  label noise: no baseline {none:.3f}; value-head baseline {head:.3f}; exact V {within:.3f}")
    for K in KS:
        fork = within * K / (K - 1)
        pre = var_resid + within / K
        lines.append(f"  K {K:2d}: post-fork {fork:.3f} (gain over value head {head / fork:.2f}x); "
                     f"decision just before the fork {pre:.3f} (gain {head / pre:.2f}x)")
    return lines


def main():
    d = np.load(NPZ, allow_pickle=False)
    outcomes = d["outcomes"].astype(float)
    luck = d["luck"].astype(float)
    index = d["index"]
    print(f"candidates {len(index)}, positions {len(np.unique(index))}, playouts {outcomes.shape[1]}")
    mask = halves(index, np.random.default_rng(1))
    lc = luck_corrected(outcomes, luck, mask)
    for head_name in ("value_reference", "value_post"):
        v = d[head_name].astype(float)
        print(f"\n== baseline from obs8's {head_name}")
        for name, out in (("raw", outcomes), ("luck-corrected", lc)):
            for line in report(name, *decompose(out, v, mask)):
                print(line)
    # bootstrap over positions for the headline (value_reference, luck-corrected, K 8)
    v = d["value_reference"].astype(float)
    positions = np.unique(index)
    rows_of = {p: np.flatnonzero(index == p) for p in positions}
    gains_post, gains_pre, explained = [], [], []
    for _ in range(300):
        pick = RNG.choice(positions, len(positions))
        rows = np.concatenate([rows_of[p] for p in pick])
        idx = np.concatenate([np.full(len(rows_of[p]), i) for i, p in enumerate(pick)])
        m = halves(idx, np.random.default_rng(RNG.integers(1 << 31)))
        o = luck_corrected(outcomes[rows], luck[rows], m)
        w, vv, vr = decompose(o, v[rows], m)
        head = w + vr
        gains_post.append(head / (w * 8 / 7))
        gains_pre.append(head / (vr + w / 8))
        explained.append(1 - vr / vv)
    for name, arr in (("K 8 post-fork gain", gains_post), ("K 8 pre-fork gain", gains_pre),
                      ("value head share of var(V)", explained)):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        print(f"bootstrap (luck-corrected, value_reference): {name} 95% {lo:.2f} to {hi:.2f}")


if __name__ == "__main__":
    main()
