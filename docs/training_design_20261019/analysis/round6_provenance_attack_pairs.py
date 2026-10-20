"""Round 6, provenance stance: what the decode-knob records say about pairs whose runner-up is
an attack.

A positive attack offset a (every attack logit raised by a nats) makes the decode play an
attack wherever a non-attack led an attack by less than a: each such change is a decision of
pair type (non-attack top, attack runner-up) at margin G in (0, a). A negative offset changes
the opposite type, (attack top, non-attack runner-up). The knob search
(training/metrics/decode_knobs_20261012, 24 vectors against own1, 6,400 games) reads the result's
response per nat (round5_provenance_knob_switches.out: -0.126 +- 0.027 z per unit, linear in a).

Here, from the same records and the same counting (round5_provenance_knob_switches.py, imported):
1. the result's response split by the sign of a (kinked fit; end_turn, recruit and time-of-day
   knobs and the end_turn offset's decision-count channel as covariates as in that script);
2. the first stage: extra attacks per side-game (side A minus own1 on side B, the winning share
   removed with the self-play slope) per nat of positive and of negative offset.
The ratio result-per-nat / attacks-per-nat is the mean advantage of one added (or removed)
attack, if each changed decision adds (removes) one attack; a changed decision can also move an
attack to a later decision without changing the count, so the count bounds the number of
changes from below and the ratio bounds the per-change magnitude from above.
numpy and the standard library only.
"""
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ks", HERE / "round5_provenance_knob_switches.py")
ks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ks)


def main():
    slope, _, _ = ks.reverse_slope()
    knobs = ks.knob_values()
    rows = []
    for v in range(24):
        label = f"kv{v:02d}"
        games = ks.read_games(ks.KNOBS / f"games_{label}.tar.gz")
        z = np.array([g[0] for g in games])
        d = np.array([g[1] for g in games], float)       # attack, recruit, move A-minus-B
        rows.append((len(z), z.mean(), d[:, 0].mean(), d.sum(1).mean()) + knobs[label])
    n = np.array([r[0] for r in rows], float)
    z = np.array([r[1] for r in rows])
    att = np.array([r[2] for r in rows]) - slope[0] * z
    allx = np.array([r[3] for r in rows]) - slope.sum() * z
    e, a, rr, t = (np.array([r[k] for r in rows]) for k in (4, 5, 6, 7))
    a_pos, a_neg = np.maximum(a, 0.0), np.minimum(a, 0.0)
    fewer, more = np.minimum(allx, 0.0), np.maximum(allx, 0.0)
    X = np.column_stack([a_pos, a_neg, fewer, more, rr, t])
    c, s = ks.wls(X, z, n)
    resid = z - X @ c
    disp = float(np.sum(n * resid ** 2) / (24 - X.shape[1]))
    s = s * np.sqrt(max(disp, 1.0))
    names = ("attack offset > 0, per nat", "attack offset < 0, per nat", "per decision removed",
             "per decision added", "recruit offset", "tod slope")
    print(f"1. result z on the kinked attack offset and covariates (SEs inflated by dispersion {disp:.2f}):")
    for nm, ci, si in zip(names, c, s):
        print(f"   {nm:28s} {ci:+.4f} +- {si:.4f}")
    X1 = np.column_stack([np.ones(24), a_pos, a_neg, e + 1.5, rr, t])
    c1, _ = ks.wls(X1, att, n)
    r1 = att - X1 @ c1
    sigma2 = float(np.sum(n * r1 ** 2) / (24 - X1.shape[1]))
    s1 = np.sqrt(np.diag(np.linalg.inv((X1 * n[:, None]).T @ X1)) * sigma2)
    print("2. extra attacks per side-game on (1, a>0, a<0, e+1.5, r, t): "
          + ", ".join(f"{ci:+.2f} +- {si:.2f}" for ci, si in zip(c1, s1)))
    k_pos, k_neg = c1[1], c1[2]
    print(f"   per nat of positive offset: {k_pos:+.2f} attacks; of negative offset: {k_neg:+.2f}")
    if k_pos > 0:
        print(f"   result per added attack (upper bound on |per change|): {c[0] / k_pos:+.4f} "
              f"(+- {s[0] / k_pos:.4f} from the result alone)")
    if k_neg > 0:
        print(f"   result per removed attack: {-c[1] / k_neg:+.4f} (+- {s[1] / k_neg:.4f} from the result alone)")


if __name__ == "__main__":
    main()
