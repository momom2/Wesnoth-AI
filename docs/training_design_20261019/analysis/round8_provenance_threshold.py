"""Round 8, provenance stance: what a switch threshold chosen on held-out labels costs and buys.

round8_provenance_rules.py shows that the sign rule (switch where f > 0, lambda by squared loss)
is the best threshold only when the near-tie advantage is Gaussian; for other shapes in the
design's own model the best threshold on f is elsewhere, on either side of 0. A threshold chosen
to maximise the held-out label sum over the switches (empirical welfare maximisation over the
one-parameter class {f > tau}; Kitagawa and Tetenov 2018) contains the oracle, but pays a
selection noise where the welfare curve is flat.

Model: the design's (round8_provenance_interim_shapes.py, imported): a = c + r, f = c + kappa (r + e),
e ~ N(0, eta_8), kappa = v / (v + eta_8). The held-out pool has N_sel labels at the student's own
f, each label = a + noise of variance s2 = 3.67 (current_design.md). f is cut into 300 bins; per
bin the label sum is Normal(n_b m_b, n_b s2), n_b = N_sel P_b. tau-hat maximises the cumulative
sum from the top; its true gain is compared with tau = 0 and with the oracle tau.
N_sel: 4 batches of band labels (6.09e7), the forecast's own count, and 7 batches (1.07e8).
Variant with a pre-test: tau-hat replaces tau = 0 only if its held-out sum exceeds tau = 0's by
2 SE of their difference (the labels between the two thresholds).
numpy and the standard library only; 400 replications per case.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round8_provenance_interim_shapes as I  # noqa: E402
import round8_provenance_rules as RU  # noqa: E402

S2 = 3.67
LABELS_PER_BATCH = I.N_BATCH
REPS = 400
RNG = np.random.default_rng(20261020)
CASES = [  # (shape, c, sd): the floors of the Gaussian model, then each two-point shape's worst case
    ("gaussian", 0.0, 0.0016), ("gaussian", -0.0018, 0.0034), ("gaussian", -0.0032, 0.0043),
    ("laplace", 0.0, 0.0016), ("laplace", -0.0032, 0.0047),
    ("two_0.02", -0.0016, 0.0055), ("two_0.05", -0.0008, 0.0047), ("two_0.1", -0.0080, 0.0047),
    ("two_0.5", -0.0076, 0.0090), ("two_0.8", -0.0056, 0.0124), ("two_0.95", -0.0008, 0.0055),
]


def bins(name, c, sd):
    r, w = RU.shape(name, sd)
    v, e = sd * sd, I.eta(8)
    kappa = v / (v + e)
    se = math.sqrt(e)
    lo = c + kappa * (r.min() - 7 * se)
    hi = c + kappa * (r.max() + 7 * se)
    edges = np.linspace(lo, hi, 301)
    cdf = RU.phi_cdf((edges[:, None] - c) / kappa / se - r[None, :] / se)   # P(f < edge | r)
    mass = np.diff(cdf, axis=0)                                             # bins x support
    p_b = mass @ w
    g_b = mass @ (w * (c + r))                                              # E[a 1(f in b)]
    return edges, p_b, g_b


def main():
    print(f"labels per batch {LABELS_PER_BATCH:.3e}; s2 {S2}; Elo per unit gain per band decision {RU.ELO_BAND:.0f}")
    print("case                      J8 at tau 0 | oracle | welfare tau-hat, mean (5th pct): 4 batches, 7 batches | "
          "with the 2-SE pre-test against tau 0: 4 batches, 7 batches")
    for name, c, sd in CASES:
        edges, p_b, g_b = bins(name, c, sd)
        tail_g = np.cumsum(g_b[::-1])[::-1]          # true gain of switching f above each lower edge
        k0 = int(np.searchsorted(edges, 0.0))
        true0 = tail_g[k0] if k0 < len(tail_g) else 0.0
        oracle = max(tail_g.max(), 0.0)
        out = []
        for batches in (4, 7):
            n_sel = batches * LABELS_PER_BATCH
            n_b = n_sel * p_b
            got, kept, moved = np.empty(REPS), np.empty(REPS), np.empty(REPS)
            tail_n = np.append(np.cumsum(n_b[::-1])[::-1], 0.0)
            for i in range(REPS):
                sums = n_sel * g_b + np.sqrt(n_b * S2) * RNG.standard_normal(len(n_b))
                est = np.append(np.cumsum(sums[::-1])[::-1], 0.0)   # the empty switch set is a candidate
                j = int(np.argmax(est))
                got[i] = tail_g[j] if j < len(tail_g) else 0.0
                # pre-test: keep tau 0 unless tau-hat's held-out sum beats it by 2 SE of the difference
                sd_diff = math.sqrt(S2 * abs(tail_n[j] - tail_n[k0]))
                jj = j if est[j] - est[k0] > 2 * sd_diff else k0
                kept[i] = tail_g[jj] if jj < len(tail_g) else 0.0
                moved[i] = jj != k0
            out.append((got.mean(), np.percentile(got, 5), kept.mean(), np.percentile(kept, 5), moved.mean()))
        e = RU.ELO_BAND
        print(f"  {name:9s} c {c:+.4f} sd {sd:.4f}: {e * true0:5.1f} | {e * oracle:5.1f} | "
              f"{e * out[0][0]:5.1f} ({e * out[0][1]:5.1f}), {e * out[1][0]:5.1f} ({e * out[1][1]:5.1f}) | "
              f"{e * out[0][2]:5.1f} ({e * out[0][3]:5.1f}), {e * out[1][2]:5.1f} ({e * out[1][3]:5.1f}); "
              f"moved off 0: {out[0][4]:.2f}, {out[1][4]:.2f}")


if __name__ == "__main__":
    main()
