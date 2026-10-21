"""Round 9, invent stance: merge round9_invent_cells_*.json (batch 1, all seven
coin matches) and print game-clustered label means per pre-coin cell.

A cell's mean label is unbiased for the cell's mean advantage (runner-up minus
top, oriented to the mover) because every cell key is fixed before the coin.
SE is game-clustered: Var(mean) = sum_g (S_g - n_g * mean)^2 / n^2.
Also prints the descriptive "other option played next" rates (post-coin).

Usage: python round9_invent_cells_summary.py   (standard library and numpy only)
"""
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ["round9_invent_cells_b00.json"] + [
    f"round9_invent_cells_b0{m}_p{p}.json" for m in range(1, 7) for p in (0, 1)]


def merge():
    acc = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0, 0])
    counts = defaultdict(int)
    for f in FILES:
        d = json.loads((HERE / f).read_text(encoding="utf-8"))
        for k, v in d["counts"].items():
            counts[k] += v
        for k, v in d["cells"].items():
            a = acc[k]
            for i in range(6):
                a[i] += v[i]
    return acc, counts


def stat(a):
    n, s, ss, ns, nn, g = a
    if n == 0:
        return float("nan"), float("nan"), 0, 0
    m = s / n
    var = (ss - 2 * m * ns + m * m * nn) / (n * n)
    return m, math.sqrt(max(var, 0.0)), n, g


def show(acc, key):
    m, se, n, g = stat(acc[key])
    z = m / se if se > 0 else float("nan")
    print(f"  {key:44s} {m:+.4f} +- {se:.4f}  z {z:+5.1f}  n {n:>9d}  games {g}")


def main():
    acc, counts = merge()
    print("games", counts["games"], "decisions", counts["decisions"], "capped dropped", counts["capped"])
    print("check against readout_b_offline.txt (pooled +0.0003 +- 0.0007; r>r +0.0070 +- 0.0024):")
    for k in ("pooled", "type:m>m", "type:r>r", "type:a>a", "type:e>m", "type:m>e"):
        show(acc, k)
    print("by actor relation (same unit or recruit type in both options, different, or end_turn involved):")
    for k in ("actor:same", "actor:diff", "actor:na"):
        show(acc, k)
    for t in ("m>m", "a>a", "a>m", "m>a", "r>r"):
        for a in ("same", "diff"):
            show(acc, f"type:{t}|actor:{a}")
    print("recruit>recruit cells:")
    for k in sorted(k for k in acc if k.startswith("rr|")):
        show(acc, k)
    print("by own side-turn:")
    for st in ("1", "2", "3-4", "5-8", "9+"):
        show(acc, f"st:{st}")
    print("r>r by side-turn:")
    for st in ("1", "2", "3-4", "5-8", "9+"):
        show(acc, f"type:r>r|st:{st}")
    print("attack>attack, weapon-only pairs (same attacker and target):")
    for k in ("aa|weapon_only:1", "aa|weapon_only:0"):
        show(acc, k)
    print("composition (band decisions by pair and actor relation):")
    tot = counts["decisions"]
    comp = sorted(((v, k[2:]) for k, v in counts.items() if k.startswith("n:")), reverse=True)
    for v, k in comp:
        print(f"  {k:14s} {v:>9d}  {v / tot:.3f}")
    print("post-coin, descriptive: the option not played is the next command (different actors):")
    for t in ("m>m", "a>m", "m>a", "a>a"):
        for d in (0, 1):
            a, b = counts.get(f"next_is_other:{t}|d{d}", 0), counts.get(f"next_total:{t}|d{d}", 0)
            if b:
                print(f"  {t} coin d={d}: {a}/{b} = {a / b:.3f}")
    print(f"family-wise: {len(acc)} cells defined, two-sided 5% Bonferroni line z {bonf(len(acc)):.2f}")
    big = [k for k in acc if stat(acc[k])[2] >= 20000]
    line = bonf(len(big))
    print(f"  {len(big)} cells with >= 20000 labels, line z {line:.2f}; cells past it:")
    for k in sorted(big, key=lambda k: -abs(stat(acc[k])[0] / stat(acc[k])[1])):
        m, se, n, g = stat(acc[k])
        if abs(m / se) > line:
            print(f"  {k:44s} {m:+.4f} +- {se:.4f}  z {m / se:+5.1f}  n {n}")
    print("largest |z| among recruit type pairs with >= 20000 labels:")
    rows = []
    for k in acc:
        if k.startswith("rrpair:"):
            m, se, n, g = stat(acc[k])
            if n >= 20000:
                rows.append((abs(m / se), k, m, se, n))
    rows.sort(reverse=True)
    print(f"  {len(rows)} pairs with >= 20000 labels; Bonferroni 5% line over them: z {bonf(len(rows)):.2f}")
    for r in rows[:8]:
        print(f"  {r[1]:44s} {r[2]:+.4f} +- {r[3]:.4f}  z {r[2] / r[3]:+5.1f}  n {r[4]}")


def bonf(k):
    # two-sided 5% family-wise line over k tests, by bisection on the normal tail
    target = 0.05 / max(k, 1) / 2
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if 0.5 * math.erfc(mid / math.sqrt(2)) > target:
            lo = mid
        else:
            hi = mid
    return lo


if __name__ == "__main__":
    main()
