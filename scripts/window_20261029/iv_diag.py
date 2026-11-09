"""Where round N's I-V signal (v times the category residual) comes from.

usage: iv_diag.py <round_dir> [<out.json>]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RUN = Path(r"C:\Users\momom2\Desktop\perso\projects\wesnoth_ai-run")
sys.path.insert(0, str(RUN))

from tools.selfplay_readout import (  # noqa: E402
    CATEGORIES, decision_arrays, ratio_se, read_jsonl, residuals, run_records)

rd = Path(sys.argv[1])
play = rd / "play"
labels = {lab["game"]: lab for lab in read_jsonl(play / "labels.jsonl") if lab["side"] == 1}
rows = []
keys_shown = False
for shard in ("games_s00", "games_s01"):
    for rec in run_records(play / shard):
        if not keys_shown:
            print("record keys:", sorted(k for k in rec if k != "decisions"))
            keys_shown = True
        if rec.get("decisions") is None:
            continue
        d = decision_arrays(rec["decisions"])
        v_raw = d["v"]
        nan_v = int(np.isnan(v_raw).sum())
        v = np.nan_to_num(v_raw)
        res = residuals(d)
        dv = (v[:, None] * res).sum(axis=0)
        lab = labels.get(rec["game_label"], {})
        rows.append({"game": rec["game_label"], "shard": shard, "n": len(v), "dv": dv.tolist(),
                     "nan_v": nan_v, "absv_gt1": int((np.abs(v) > 1.0 + 1e-6).sum()),
                     "map": rec.get("scenario_id"),
                     "heldout": lab.get("heldout"), "turns": rec.get("turns"),
                     })

n = np.array([r["n"] for r in rows], float)
DV = np.array([r["dv"] for r in rows], float)
print(f"games {len(rows)}, NaN v total {sum(r['nan_v'] for r in rows)}, |v|>1 total {sum(r['absv_gt1'] for r in rows)}")


def z_of(mask, col):
    m, se = ratio_se(DV[mask, col], n[mask])
    return m, se, m / se if se else 0.0


cats = [c for c, _, _ in CATEGORIES]
allm = np.ones(len(rows), bool)
for i, c in enumerate(cats):
    m, se, z = z_of(allm, i)
    print(f"all {c:9s} mean {m:+.3e} se {se:.3e} z {z:+.2f}")
for key in ("shard", "heldout"):
    for val in sorted({r[key] for r in rows}, key=str):
        mask = np.array([r[key] == val for r in rows])
        print(key, val, mask.sum(), " ".join(f"{c}:{z_of(mask, i)[2]:+.2f}" for i, c in enumerate(cats)))
# Contribution concentration: drop the games with the largest |dv_recruit - r n|.
ri = cats.index("recruit")
r0 = DV[:, ri].sum() / n.sum()
contrib = DV[:, ri] - r0 * n
order = np.argsort(-np.abs(contrib))
for k in (1, 5, 20, 75, 375):
    mask = np.ones(len(rows), bool)
    mask[order[:k]] = False
    print(f"drop top {k:3d} |contrib|: recruit z {z_of(mask, ri)[2]:+.2f}, attack z {z_of(mask, cats.index('attack'))[2]:+.2f}")
# Kurtosis of the per-game contribution, against a Gaussian's 3.
c = contrib / contrib.std()
print("per-game contribution kurtosis %.2f, max |c| %.2f" % (float((c ** 4).mean()), float(np.abs(c).max())))
# Bootstrap of the recruit z's SE.
rng = np.random.default_rng(0)
G = len(rows)
boots = []
for _ in range(2000):
    idx = rng.integers(0, G, G)
    boots.append(DV[idx, ri].sum() / n[idx].sum())
m, se, z = z_of(allm, ri)
print(f"recruit: analytic se {se:.3e}, bootstrap sd {np.std(boots):.3e}")
# By map.
bymap = defaultdict(list)
for j, r in enumerate(rows):
    bymap[str(r["map"])].append(j)
print("by map (recruit z, attack z, games):")
for mp, idx in sorted(bymap.items(), key=lambda kv: -len(kv[1])):
    mask = np.zeros(len(rows), bool)
    mask[idx] = True
    print(f"  {mp[:40]:40s} {z_of(mask, ri)[2]:+.2f} {z_of(mask, cats.index('attack'))[2]:+.2f} {mask.sum()}")
if len(sys.argv) > 2:
    Path(sys.argv[2]).write_text(json.dumps(rows) + "\n")
