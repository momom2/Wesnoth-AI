"""Per-seat, per-faction and per-map breakdown of a match's decisive games.

Read-only on the game result JSONs (not the .game.jsonl.gz records). Elo of
label A against label B, with the delta-method SE, by cell.
Usage: python breakdown.py <games_dir> [<games_dir> ...]
"""
import collections
import json
import math
import sys
from pathlib import Path

try:
    import psutil
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
except Exception:
    pass


def elo(w, n):
    if n == 0 or w in (0, n):
        return float("nan"), float("nan")
    p = w / n
    e = 400 * math.log10(p / (1 - p))
    se = 400 / math.log(10) * math.sqrt(p * (1 - p) / n) / (p * (1 - p))
    return e, se


def table(title, cells, min_n=0):
    print(f"\n  {title}")
    rows = []
    for key, (w, n) in sorted(cells.items(), key=lambda kv: -kv[1][1]):
        if n < min_n:
            continue
        e, se = elo(w, n)
        rows.append((key, n, e, se))
        print(f"    {str(key):45s} n={n:6d}  {e:+7.1f} +- {se:5.1f}")
    return rows


def spread(rows):
    """sd of the cells' Elo against their mean SE (noise-sized if close)."""
    es = [r[2] for r in rows if not math.isnan(r[2])]
    ses = [r[3] for r in rows if not math.isnan(r[3])]
    if len(es) < 2:
        return
    m = sum(es) / len(es)
    sd = math.sqrt(sum((e - m) ** 2 for e in es) / (len(es) - 1))
    # chi-square of the cells against their weighted mean
    wts = [1 / r[3] ** 2 for r in rows if not math.isnan(r[3])]
    wm = sum(w * r[2] for w, r in zip(wts, [r for r in rows if not math.isnan(r[3])])) / sum(wts)
    chi2 = sum(w * (r[2] - wm) ** 2 for w, r in zip(wts, [r for r in rows if not math.isnan(r[3])]))
    print(f"    spread: {len(es)} cells, sd {sd:.1f} against mean SE {sum(ses) / len(ses):.1f}; "
          f"chi2 {chi2:.1f} on {len(es) - 1} df")


def main(dirs):
    for d in dirs:
        seat, fac, opp, match, scen = (collections.defaultdict(lambda: [0, 0]) for _ in range(5))
        total, labels, procs, other = [0, 0], set(), set(), collections.Counter()
        for f in Path(d).glob("*.json"):
            r = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(r, dict) or "label_a" not in r:
                other[f"skipped {f.name}"] += 1
                continue
            o = r.get("outcome_a")
            labels.add((r["label_a"], r["label_b"]))
            procs.add((r["procedure_a"], r.get("raw_temperature_a"), r["procedure_b"], r.get("raw_temperature_b")))
            if o not in ("win", "loss"):
                other[o] += 1
                continue
            w = int(o == "win")
            fa, fb = r["faction_a"], r["faction_b"]
            for cells, key in ((seat, f"side {r['side_a']}"), (fac, fa), (opp, fb),
                               (match, "mirror" if fa == fb else f"{fa} v {fb}"),
                               (scen, r["scenario_id"].replace("multiplayer_", ""))):
                cells[key][0] += w
                cells[key][1] += 1
            total[0] += w
            total[1] += 1
        e, se = elo(*total)
        print(f"\n=== {d}\n  labels {sorted(labels)}\n  procedures {sorted(procs, key=str)}\n"
              f"  decisive {total[1]} (A won {total[0]}), not decisive {dict(other)}: {e:+.1f} +- {se:.1f}")
        srows = {r[0]: r for r in table("by A's seat", seat)}
        if {"side 1", "side 2"} <= set(srows):
            d = srows["side 1"][2] - srows["side 2"][2]
            dse = math.hypot(srows["side 1"][3], srows["side 2"][3])
            print(f"    side 1 - side 2: {d:+.1f} +- {dse:.1f} ({d / dse:+.1f} SE)")
        arows = {r[0]: r for r in table("by A's faction", fac)}
        spread(list(arows.values()))
        brows = {r[0]: r for r in table("by B's faction", opp)}
        spread(list(brows.values()))
        # A's edge with the factions' balance cancelled: the mean of A playing F and B playing F;
        # half their difference is F's strength over the average faction (when A and B are near-equal)
        print("\n  by faction, balanced: A's edge (mean of A as F and B as F); F's strength (half the difference)")
        for f in sorted(set(arows) & set(brows)):
            a, b = arows[f], brows[f]
            print(f"    {f:45s} edge {(a[2] + b[2]) / 2:+7.1f} +- {math.hypot(a[3], b[3]) / 2:4.1f}; "
                  f"strength {(a[2] - b[2]) / 2:+7.1f} +- {math.hypot(a[3], b[3]) / 2:4.1f}")
        mrows = table("mirror against not", {"mirror": match["mirror"],
                                              "not mirror": [sum(v[0] for k, v in match.items() if k != "mirror"),
                                                             sum(v[1] for k, v in match.items() if k != "mirror")]})
        del mrows
        spread(table("by map", scen))


if __name__ == "__main__":
    main(sys.argv[1:])
