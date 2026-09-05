"""Second reading: pooled-alternatives screen, sequential two-candidate
confirmation (with and without odd/even separation), pipeline costs."""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(r"C:/Users/amaur/Desktop/Perso/projects/Wesnoth_AI")
run1 = json.load(open(ROOT / "training/metrics/turn_gap/run1.json"))
conf = json.load(open(ROOT / "training/metrics/turn_gap/confirm1.json"))
R1 = {p["index"]: p for p in run1["positions"]}
C1 = {p["index"]: p for p in conf["positions"]}
C_PLAY = 0.90 / 12000  # dollars per playout, run1

# ------------------------------------------------ pooled screen
print("Pooled screen on run1: mean over the 4 alternatives minus base (160 vs 40 playouts).")
rows = []
for i in sorted(R1):
    r = R1[i]
    b = np.array(r["base"]["outcomes"], float)
    alts = np.concatenate([np.array(a["outcomes"], float) for a in r["alternatives"]])
    d = alts.mean() - b.mean()
    se = math.sqrt(b.var(ddof=1) / len(b) + alts.var(ddof=1) / len(alts)) if b.var() + alts.var() > 0 else 0.0
    rows.append((i, round(d, 3), round(se, 3), round(d / se, 2) if se else 0.0, round(r["gap"], 2)))
rows.sort(key=lambda t: -t[3])
print("  top by z (idx, pooled diff, se, z, best-of-4 gap):")
for t in rows[:12]:
    print("   ", t, "confirmed" if t[0] in (4, 18, 59) else ("borderline" if t[0] in (20, 29) else ""))

# ------------------------------------------------ sequential 2-candidate confirmation on the 12
print("\nSequential confirmation on (base, run1-selected alt) using confirm1's 160 fresh playouts in order.")
print("Rules: rounds of n0; confirm when gap_hat >= 0.25 and gap_hat - z*SE >= lo; reject when gap_hat + z*SE < 0.25.")


def seq_confirm(i, n0=20, z=2.0, lo=0.10, use="all"):
    r1, rc = R1[i], C1[i]
    sel = int(np.argmax([a["mean"] for a in r1["alternatives"]]))
    b_all = np.array(rc["base"]["outcomes"], float)
    a_all = np.array(rc["alternatives"][sel]["outcomes"], float)
    if use == "odd":
        b_dec, a_dec = b_all[1::2], a_all[1::2]
        b_est, a_est = b_all[0::2], a_all[0::2]
    else:
        b_dec, a_dec = b_all, a_all
        b_est, a_est = b_all, a_all
    n = 0
    verdict = "undecided"
    while n < len(b_dec):
        n = min(n + n0, len(b_dec))
        b, a = b_dec[:n], a_dec[:n]
        g = a.mean() - b.mean()
        se = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / n)
        if g >= 0.25 and g - z * se >= lo:
            verdict = "confirm"
            break
        if g + z * se < 0.25:
            verdict = "reject"
            break
    est = a_est[:n].mean() - b_est[:n].mean()
    playouts = 2 * n * (2 if use == "odd" else 1)
    return verdict, n, playouts, round(g, 3), round(est, 3)


for use in ("all", "odd"):
    tot = 0
    print(f"  mode={use} (decide on {'all' if use == 'all' else 'odd-indexed'} playouts, estimate on {'the same' if use == 'all' else 'even-indexed'}):")
    for i in sorted(C1):
        v, n, pl, g, est = seq_confirm(i, use=use)
        tot += pl
        print(f"    {i:2d}: {v:9s} after {n:3d} per candidate ({pl} playouts), gap_hat {g:+.2f}, estimate {est:+.2f}")
    print(f"    total playouts {tot} = ${tot * C_PLAY:.3f} (the run used 9600 = $0.84)")

# ------------------------------------------------ costs of pipeline A
seq_screen = 7790  # from analysis 1: n0=10, z=2, thr 0.25, budget 40 -> same 12 nominal hits
print(f"\nPipeline A: sequential screen {seq_screen} playouts = ${seq_screen * C_PLAY:.2f}; ")
for name, pl in (("flat 2-cand x160", 12 * 2 * 160), ("sequential all", None), ("sequential odd/even", None)):
    pass

# ------------------------------------------------ per-candidate SD near 0 vs at extremes
sds = []
for r in run1["positions"]:
    for c in [r["base"]] + r["alternatives"]:
        o = np.array(c["outcomes"], float)
        sds.append((abs(o.mean()), o.std(ddof=1)))
sds = np.array(sds)
for lo, hi in ((0, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.01)):
    m = (sds[:, 0] >= lo) & (sds[:, 0] < hi)
    print(f"  |mean| in [{lo},{hi}): n={m.sum()}, mean SD {sds[m,1].mean():.2f}")

# ------------------------------------------------ how many alternatives above base by class (blunder vs find) in run1
print("\nrun1: positions by number of alternatives >= base + 0.25 (40 playouts):")
from collections import Counter
cnt = Counter()
for i, r in R1.items():
    k = sum(a["mean"] - r["base"]["mean"] >= 0.25 for a in r["alternatives"])
    cnt[k] += 1
print("  ", dict(sorted(cnt.items())))
