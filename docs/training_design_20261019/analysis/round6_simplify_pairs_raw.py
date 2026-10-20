"""Round 6, simplify stance: what side-swapped pairs keep of a match's variance with no luck
columns at all, so that the raw (PURE) reading of a paired adoption match can serve as the
strength claim, and so that ruling (b)'s "no" has a match-side number.

Reads training/metrics/match_luck_20261019/rows.jsonl (decisive games of 23 argmax matches).
Each match's mean is removed. The swap covariance Cov(f(c), f(swap c)) of the setup cells'
true effects is estimated as in round5_provenance_critique_pairs.py (whose function is reused),
here on the raw centred result instead of the column-corrected residual. Pairs keep
1 + Cov / Var per game.

Also printed: the SE of a 12,800-game match read raw, (a) as tools/elo_collect.py computes it
(games independent: 347.4 / sqrt(n) Elo at an even match) and (b) with the pairs' kept share.
numpy and the standard library only.
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cp", HERE / "round5_provenance_critique_pairs.py")
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)
ELO_PER_Z = 800 / math.log(10)
N_MATCH = 12_800


def main():
    rows = [json.loads(line) for line in open(cp.ps.ROWS)]
    rows = [r for r in rows if r.get("decisive")]
    z = np.array([float(r["z"]) for r in rows])
    match = np.array([r["match"] for r in rows])
    zc = z.copy()
    for m in set(match):
        zc[match == m] -= z[match == m].mean()
    total = float(np.var(zc))
    keys = [(r["scenario_id"], r["side_a"], r["faction_a"], r["faction_b"]) for r in rows]
    tau2 = cp.ps.cell_share(keys, zc, total)
    cov, se, npairs = cp.cross_cov(keys, zc)
    kept = 1 + cov / total
    print(f"decisive games {len(z)}; cell pairs with both cells observed {npairs}")
    print(f"raw result: full-cell tau2 share {tau2:.3f}; swap covariance share {cov / total:+.3f} +- {se / total:.3f}")
    print(f"pairs alone keep {kept:.3f} +- {se / total:.3f} of a game's variance (no columns)")
    se_indep = ELO_PER_Z * math.sqrt(1.0 / N_MATCH)
    se_pairs = ELO_PER_Z * math.sqrt(kept / N_MATCH)
    se_pc = ELO_PER_Z * math.sqrt(0.700 / N_MATCH)
    print(f"12,800-game match read raw: SE {se_indep:.2f} Elo as elo_collect computes it (independent games), "
          f"{se_pairs:.2f} Elo with the pairs' share; pairs + columns {se_pc:.2f} Elo")
    print(f"800-game PURE match: SE {ELO_PER_Z * math.sqrt(1.0 / 800):.1f} Elo")


if __name__ == "__main__":
    sys.exit(main())
