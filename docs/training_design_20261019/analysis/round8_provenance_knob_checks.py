"""Round 8, provenance stance: is the S0(b) known answer's window honest, and do the knob records
bound any second-order term?

Data: the decode-knob search's stage-1 games (training/metrics/decode_knobs_20261012), 24 vectors
x 100 games, no selection (stage read from the seed as in round5_provenance_critique_knobs.py,
whose reader is imported). Each vector is own1 at 64 slots under a decode with four knobs
(e end_turn offset, a attack offset, r recruit offset, t attack time-of-day slope) against own1
at its own decode; kv00 is own1 against itself.

The S0(b) gate predicts "attack offset +1 on one seat" from a fit of these games. Round 6's fit
(round6_provenance_critique_attack_stage1.py, B) regresses z on the kinked attack offset plus
decision counts as the end_turn channel; its counts carry the response's own noise (the winning
share is subtracted with the observed z). Here:
  A. the reduced form: z on knobs only, kinked in a and in e around today's decode
     (a+, a-, min(e + 1.5, 0), max(e + 1.5, 0), r, t), no intercept, weights n, binomial SEs
     inflated by the dispersion when above 1; its prediction at a = +1, e = -1.5, r = t = 0;
  B. leave-one-vector-out: each vector predicted from the other 23, standardized by
     sqrt(1/n_v + x_v' Cov x_v); an honest SE gives an RMS near 1;
  C. curvature in a+: (a+)^2 added; the move at a = 0.35 and a = 1.0 under both fits;
  D. cross-knob interaction terms added one at a time: a+ x (e + 1.5), a+ x t, a+ x r.
numpy and the standard library only.
"""
import importlib.util
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ck", HERE / "round5_provenance_critique_knobs.py")
ck = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ck)
ELO = 800 / math.log(10)


def stage1_rows():
    knobs = ck.ks.knob_values()
    rows = []
    for v in range(24):
        label = f"kv{v:02d}"
        games = [g for g in ck.read(ck.KNOBS / f"games_{label}.tar.gz") if g[0] == 1]
        z = np.array([g[1] for g in games])
        rows.append((len(z), z.mean()) + knobs[label])
    n = np.array([r[0] for r in rows], float)
    z = np.array([r[1] for r in rows])
    e, a, rr, t = (np.array([r[k] for r in rows]) for k in (2, 3, 4, 5))
    return n, z, e, a, rr, t


def wls(X, y, w):
    sw = np.sqrt(w)
    coef = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    cov = np.linalg.inv((X * w[:, None]).T @ X)
    disp = float(np.sum(w * (y - X @ coef) ** 2) / (len(y) - X.shape[1]))
    return coef, cov * max(disp, 1.0), disp


def base_design(e, a, rr, t):
    return np.column_stack([np.maximum(a, 0), np.minimum(a, 0), np.minimum(e + 1.5, 0),
                            np.maximum(e + 1.5, 0), rr, t])


def main():
    n, z, e, a, rr, t = stage1_rows()
    print(f"stage-1 games: {int(n.sum())} over {len(n)} vectors; vectors with a > 0: {int(np.sum(a > 0))}, "
          f"a range {a.min():+.2f} to {a.max():+.2f}; t range {t.min():.2f} to {t.max():.2f}")
    X = base_design(e, a, rr, t)
    c, cov, disp = wls(X, z, n)
    se = np.sqrt(np.diag(cov))
    names = ("a > 0 per nat", "a < 0 per nat", "e below -1.5 per nat", "e above -1.5 per nat", "r per nat", "t per unit")
    print(f"\nA. reduced form (dispersion {disp:.2f}):")
    for nm, ci, si in zip(names, c, se):
        print(f"   {nm:22s} {ci:+.4f} +- {si:.4f}")
    print(f"   S0(b) prediction (a +1, e -1.5, r = t = 0): {c[0]:+.4f} +- {se[0]:.4f} z = "
          f"{ELO * c[0]:+.0f} +- {ELO * se[0]:.0f} Elo; label-code 2 Delta {2 * c[0]:+.3f} +- {2 * se[0]:.3f}")
    print("   round 6's decision-count fit on the same games: -0.2247 +- 0.0459 z (-78 +- 16 Elo)")
    match_se = 1 / math.sqrt(6400)
    for name, mid, s in (("reduced form", c[0], se[0]), ("decision-count fit", -0.2247, 0.0459)):
        half = 2 * math.sqrt(s ** 2 + match_se ** 2)
        print(f"   window at 6,400 games from the {name}: {ELO * (mid - half):+.0f} to {ELO * (mid + half):+.0f} Elo")

    print("\nB. leave-one-vector-out, reduced form")
    std = []
    for v in range(len(n)):
        keep = np.arange(len(n)) != v
        cv, covv, _ = wls(X[keep], z[keep], n[keep])
        pred = X[v] @ cv
        s = math.sqrt(1 / n[v] + X[v] @ covv @ X[v])
        std.append((z[v] - pred) / s)
        if a[v] >= 0.8:
            print(f"   kv{v:02d} a {a[v]:+.2f} e {e[v]:+.2f} t {t[v]:.2f}: observed {z[v]:+.3f}, predicted {pred:+.3f} "
                  f"+- {s:.3f}, standardized {std[-1]:+.2f}")
    std = np.array(std)
    print(f"   all 24: RMS standardized residual {math.sqrt(np.mean(std ** 2)):.2f}, max |.| {np.abs(std).max():.2f}, "
          f"share beyond 2: {np.mean(np.abs(std) > 2):.3f}")

    print("\nC. curvature in a > 0")
    Xq = np.column_stack([X, np.maximum(a, 0) ** 2])
    cq, covq, dq = wls(Xq, z, n)
    print(f"   (a+)^2 coefficient {cq[-1]:+.4f} +- {math.sqrt(covq[-1, -1]):.4f} (dispersion {dq:.2f})")
    for o in (0.35, 1.0):
        g_lin = np.zeros(X.shape[1])
        g_lin[0] = o
        g_q = np.zeros(Xq.shape[1])
        g_q[0], g_q[-1] = o, o * o
        print(f"   move at a = {o}: linear {g_lin @ c:+.4f} +- {math.sqrt(g_lin @ cov @ g_lin):.4f}; "
              f"quadratic {g_q @ cq:+.4f} +- {math.sqrt(g_q @ covq @ g_q):.4f}")

    print("\nD. interaction terms added one at a time")
    ap = np.maximum(a, 0)
    for nm, col in (("a+ x (e + 1.5)", ap * (e + 1.5)), ("a+ x t", ap * t), ("a+ x r", ap * rr)):
        Xi = np.column_stack([X, col])
        ci, covi, di = wls(Xi, z, n)
        s = math.sqrt(covi[-1, -1])
        print(f"   {nm:16s} {ci[-1]:+.4f} +- {s:.4f} z per unit product (dispersion {di:.2f}); "
              f"at one unit of each knob: {ELO * ci[-1]:+.0f} +- {ELO * s:.0f} Elo")


if __name__ == "__main__":
    main()
