"""Plain-python reading of training/metrics/turn_gap/{run1,confirm1}.json.
No torch. Answers: cost per playout, playout length distribution,
truncated-cap graders, blunder-vs-find, decisions-vs-gap, sequential
testing simulation, per-candidate SD."""
import json
import math
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(r"C:/Users/amaur/Desktop/Perso/projects/Wesnoth_AI")
run1 = json.load(open(ROOT / "training/metrics/turn_gap/run1.json"))
conf = json.load(open(ROOT / "training/metrics/turn_gap/confirm1.json"))
R1 = {p["index"]: p for p in run1["positions"]}
C1 = {p["index"]: p for p in conf["positions"]}


def cands(rec):
    return [rec["base"]] + list(rec["alternatives"])


def lengths(rec, cand):
    t0 = rec["turn_number"]
    return [t - t0 for t in cand["turns"]]


# ---------------------------------------------------------------- 1. cost
for name, d in (("run1", run1), ("confirm1", conf)):
    s = d["summary"]
    jobs = d["provenance"]["jobs"]
    n_play = s["playouts_total"]
    wall = s["wall_secs"]
    all_len = [L for rec in d["positions"] for c in cands(rec) for L in lengths(rec, c)]
    all_cap = [x for rec in d["positions"] for c in cands(rec) for x in c["capped"]]
    proc_s = wall * jobs / n_play
    print(f"{name}: playouts {n_play}, wall {wall/3600:.2f} h, jobs {jobs}, "
          f"process-s per playout {proc_s:.1f}, mean length {np.mean(all_len):.1f} turns "
          f"(median {np.median(all_len):.0f}), s per game-turn per process {proc_s/np.mean(all_len):.2f}, "
          f"$ per playout {s['dollars']/n_play*1000:.3f} per 1000, capped {np.mean(all_cap):.3f}")
    qs = [5, 10, 15, 20, 25, 30]
    arr = np.array(all_len)
    print("   fraction of playouts decided within t turns after the boundary:",
          {t: round(float(np.mean(arr <= t)), 3) for t in qs})
    # decisive length distribution (non-capped only)
    dec = np.array([L for L, cp in zip(all_len, all_cap) if not cp])
    print("   decisive playouts: mean length", round(float(dec.mean()), 1),
          "p25/p50/p75", np.percentile(dec, [25, 50, 75]))

# ---------------------------------------------------------------- 2. per-candidate SD
sds = []
for rec in run1["positions"]:
    for c in cands(rec):
        sds.append(np.std(c["outcomes"], ddof=1))
print(f"\nper-candidate outcome SD (run1, P=40): mean {np.mean(sds):.3f}, "
      f"p10 {np.percentile(sds,10):.2f}, p90 {np.percentile(sds,90):.2f}")
# outcome composition
allo = [o for rec in run1["positions"] for c in cands(rec) for o in c["outcomes"]]
print("   outcome mix run1: win", allo.count(1) / len(allo), "loss", allo.count(-1) / len(allo),
      "zero", allo.count(0) / len(allo))

# ---------------------------------------------------------------- 3. per-position table (run1)
print("\nrun1 per position: idx scenario T0 side | base n_dec mean | alts n_dec | alts mean | gap | n_alts>base+0.25 | mean len")
rows = []
for i in sorted(R1):
    rec = R1[i]
    b = rec["base"]
    alts = rec["alternatives"]
    n_beat = sum(a["mean"] - b["mean"] >= 0.25 for a in alts)
    ml = np.mean([L for c in cands(rec) for L in lengths(rec, c)])
    rows.append((i, rec["scenario_id"].replace("multiplayer_", ""), rec["turn_number"], rec["side"],
                 b["n_decisions"], b["mean"], [a["n_decisions"] for a in alts],
                 [round(a["mean"], 2) for a in alts], round(rec["gap"], 2), n_beat, round(ml, 1)))
for r in rows:
    print(" ", r)

# ---------------------------------------------------------------- 4. decisions vs gap
xs, ys = [], []
for rec in run1["positions"]:
    b = rec["base"]
    for a in rec["alternatives"]:
        xs.append(a["n_decisions"] - b["n_decisions"])
        ys.append(a["mean"] - b["mean"])
xs, ys = np.array(xs), np.array(ys)
print(f"\nalternatives (n={len(xs)}): n_dec(alt)-n_dec(base) mean {xs.mean():+.2f}; "
      f"alt mean - base mean: {ys.mean():+.3f}")
print("   Pearson r(decision diff, value diff) =", round(float(np.corrcoef(xs, ys)[0, 1]), 3))
for lo, hi in ((-99, -1), (0, 0), (1, 3), (4, 6), (7, 99)):
    m = (xs >= lo) & (xs <= hi)
    if m.any():
        print(f"   decision diff in [{lo},{hi}]: n={m.sum()}, mean value diff {ys[m].mean():+.3f}, "
              f"frac >= +0.25: {np.mean(ys[m] >= 0.25):.2f}")

# ---------------------------------------------------------------- 5. confirm positions: blunder vs find
print("\nconfirm1 (160 new playouts): idx | base n_dec / mean(new) | alts n_dec | alts mean(new) | "
      "run1-selected alt (k, run1 mean) | oos gap | #alts >= base+0.25 (new) | best-other")
for i in sorted(C1):
    r1, rc = R1[i], C1[i]
    b1, bc = r1["base"], rc["base"]
    sel = int(np.argmax([a["mean"] for a in r1["alternatives"]]))
    alts_c = rc["alternatives"]
    oos = alts_c[sel]["mean"] - bc["mean"]
    n_beat = sum(a["mean"] - bc["mean"] >= 0.25 for a in alts_c)
    others = [a["mean"] - bc["mean"] for k, a in enumerate(alts_c) if k != sel]
    print(f"  {i:2d} | {bc['n_decisions']:2d} / {bc['mean']:+.2f} | "
          f"{[a['n_decisions'] for a in alts_c]} | {[round(a['mean'],2) for a in alts_c]} | "
          f"(alt{sel}, {r1['alternatives'][sel]['mean']:+.2f}) | {oos:+.2f} | {n_beat} | {max(others):+.2f}")

# combined 200-playout means for the 12
print("\nconfirm positions, 200-playout means (run1 40 + confirm 160), SE ~ SD/sqrt(200):")
for i in sorted(C1):
    r1, rc = R1[i], C1[i]
    means = []
    for c1, cc in zip(cands(r1), cands(rc)):
        o = c1["outcomes"] + cc["outcomes"]
        means.append((round(float(np.mean(o)), 3), round(float(np.std(o, ddof=1) / math.sqrt(len(o))), 3)))
    print(f"  {i:2d}: base {means[0]}  alts {means[1:]}")

# ---------------------------------------------------------------- 6. truncated-cap graders
print("\nTruncated grader: outcome counted only if decided within c turns after the boundary, else 0.")
print("Per cap: mean |gap_c - gap_full| over positions, Spearman of candidate ranking within position (mean),"
      " the 3 confirmed positions' gap_c on the 160 new playouts, and mean playout length under the cap.")


def trunc_means(rec, cap):
    out = []
    for c in cands(rec):
        L = lengths(rec, c)
        out.append(float(np.mean([o if (l <= cap and not cp) else 0
                                  for o, l, cp in zip(c["outcomes"], L, c["capped"])])))
    return out


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


for cap in (5, 8, 10, 12, 15, 20, 30):
    diffs, rhos, lens = [], [], []
    for rec in run1["positions"]:
        full = [c["mean"] for c in cands(rec)]
        tm = trunc_means(rec, cap)
        gap_c = max(tm[1:]) - tm[0]
        diffs.append(abs(gap_c - rec["gap"]))
        rhos.append(spearman(full, tm))
        lens += [min(L, cap) for c in cands(rec) for L in lengths(rec, c)]
    conf_gaps = {}
    for i in (4, 18, 59, 20, 29):
        r1, rc = R1[i], C1[i]
        sel = int(np.argmax([a["mean"] for a in r1["alternatives"]]))
        tm = trunc_means(rc, cap)
        conf_gaps[i] = round(tm[sel + 1] - tm[0], 2)
    print(f"  cap {cap:2d}: mean|dgap| {np.mean(diffs):.3f}, mean Spearman {np.nanmean(rhos):.2f}, "
          f"oos gap_c at 4/18/59/20/29 {conf_gaps}, mean length {np.mean(lens):.1f} "
          f"(cost x{np.mean(lens)/np.mean([L for rec in run1['positions'] for c in cands(rec) for L in lengths(rec, c)]):.2f})")

# ---------------------------------------------------------------- 7. sequential testing simulation
print("\nSequential screening simulation on the recorded outcomes (in recorded order).")
print("Rule: rounds of n0 playouts per surviving candidate (base always). After each round, an alternative is")
print("dropped when gap_hat + z*SE_diff < thr (cannot be a large gap); the position stops as 'no hit' when")
print("no alternative survives; it stops as 'nominal hit' when an alternative has gap_hat - z*SE_diff > 0")
print("and gap_hat >= thr, or when the budget per candidate is exhausted (then hit if gap_hat >= thr).")


def simulate(rec_run, rec_conf, n0, z, thr, budget, min_rounds=1):
    """Returns (playouts_used, verdict, chosen_alt, n_rounds). Outcomes:
    run1's 40 followed by confirm's 160 when available."""
    base_o = list(rec_run["base"]["outcomes"]) + (list(rec_conf["base"]["outcomes"]) if rec_conf else [])
    alts_o = [list(a["outcomes"]) + (list(rec_conf["alternatives"][k]["outcomes"]) if rec_conf else [])
              for k, a in enumerate(rec_run["alternatives"])]
    avail = len(base_o)
    budget = min(budget, avail)
    alive = list(range(len(alts_o)))
    n = 0
    used = 0
    rounds = 0
    while True:
        n_new = min(n0, budget - n)
        if n_new <= 0:
            break
        used += n_new * (1 + len(alive))
        n += n_new
        rounds += 1
        b = np.array(base_o[:n], dtype=float)
        bm, bv = b.mean(), (b.var(ddof=1) if n > 1 else 1.0)
        stats = {}
        for k in alive:
            a = np.array(alts_o[k][:n], dtype=float)
            am, av = a.mean(), (a.var(ddof=1) if n > 1 else 1.0)
            se = math.sqrt((av + bv) / n)
            stats[k] = (am - bm, se)
        if rounds >= min_rounds:
            alive = [k for k in alive if stats[k][0] + z * stats[k][1] >= thr]
            if not alive:
                return used, "no", None, n
            best = max(alive, key=lambda k: stats[k][0])
            g, se = stats[best]
            if g >= thr and g - z * se > 0:
                return used, "hit", best, n
    if not alive:
        return used, "no", None, n
    best = max(alive, key=lambda k: stats[k][0])
    return used, ("hit" if stats[best][0] >= thr else "no"), best, n


full_used = 60 * 5 * 40
for (n0, z, thr, budget) in ((10, 1.5, 0.25, 40), (10, 2.0, 0.25, 40), (8, 1.5, 0.25, 40),
                             (10, 1.5, 0.25, 200), (10, 2.0, 0.25, 200), (20, 2.0, 0.25, 200),
                             (10, 1.5, 0.15, 200)):
    tot = 0
    hits = []
    for i in sorted(R1):
        used, verdict, k, n = simulate(R1[i], C1.get(i), n0, z, thr, budget)
        tot += used
        if verdict == "hit":
            hits.append((i, k, n))
    conf_found = [h for h in hits if h[0] in (4, 18, 59)]
    print(f"  n0={n0} z={z} thr={thr} budget={budget}: playouts {tot} (vs {full_used} for the flat 40), "
          f"hits {len(hits)} at positions {[h[0] for h in hits]}, confirmed-3 found: {[h[0] for h in conf_found]}, "
          f"playouts per candidate at stop for hits {[h[2] for h in hits]}")

# ---------------------------------------------------------------- 8. successive halving on the 12 (which alt is selected?)
print("\nSuccessive halving on the 12 confirm positions (200 outcomes each): after n playouts per candidate keep top half; "
      "does the finalist match the 200-playout best?")
for i in sorted(C1):
    r1, rc = R1[i], C1[i]
    outs = [c1["outcomes"] + cc["outcomes"] for c1, cc in zip(cands(r1), cands(rc))]
    best200 = int(np.argmax([np.mean(o) for o in outs[1:]]))
    alive = [0, 1, 2, 3]
    picks = {}
    for n in (10, 20, 40):
        alive = sorted(alive, key=lambda k: -np.mean(outs[k + 1][:n]))[: max(1, len(alive) // 2)]
        picks[n] = list(alive)
    print(f"  {i:2d}: best@200 alt{best200} (mean {np.mean(outs[best200+1]):+.2f} vs base {np.mean(outs[0]):+.2f}); "
          f"halving picks {picks}; run1-40 pick alt{int(np.argmax([a['mean'] for a in r1['alternatives']]))}")

# ---------------------------------------------------------------- 9. playouts needed for confirmation at 3 SE
sd = 0.9
for g in (0.25, 0.3, 0.5, 0.75, 1.0):
    for zz in (2.0, 3.0):
        P = (zz * math.sqrt(2) * sd / g) ** 2
        print(f"  gap {g}: P per candidate for {zz} SE = {P:.0f}", end=";")
    print()

# ---------------------------------------------------------------- 10. correlation base vs alt outcome at same r (CRN check)
cs = []
for rec in run1["positions"]:
    b = np.array(rec["base"]["outcomes"], float)
    for a in rec["alternatives"]:
        a = np.array(a["outcomes"], float)
        if b.std() > 0 and a.std() > 0:
            cs.append(np.corrcoef(b, a)[0, 1])
print(f"\nmean corr(base outcome, alt outcome) at the same playout index r: {np.mean(cs):+.3f} (n={len(cs)}); "
      "expected ~0: salts carry the candidate index, so playouts are independent")
