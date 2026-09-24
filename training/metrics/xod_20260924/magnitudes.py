"""Size of the trades in the largest blocking families (rejected by all 36)."""
import collections
import statistics
import sys

sys.path.insert(0, ".")
import analyze as A  # noqa: E402
import families as F  # noqa: E402  (prints its table on import)
from tools.combat_dominance import COMBOS, GT, dim_passes  # noqa: E402

R0 = COMBOS[0]
by = collections.defaultdict(list)
for r in F.rows:
    if r["r0"] or r["any36"]:
        continue
    dims = r["cdims"]
    gains = [d for d in dims if d.sym == GT and d.justifies]
    if r["tier"] == "D" and not gains:
        continue
    worse = collections.defaultdict(set)
    for d in dims:
        if dim_passes(d, R0) is False:
            worse[A.vectors.group(d.name)].add(d.sym)
    better = {A.vectors.group(d.name) for d in gains}
    by[F.family(r, worse, better, None)].append(A.gains_of(r) | {"trade": abs(r["gains"].get("d_hp_a1", 0)) + abs(r["gains"].get("d_hp_a2", 0))})
print("\nfamily (rejected by all 36): median / 90th percentile of |change|")
for f, gs in sorted(by.items(), key=lambda kv: -len(kv[1]))[:12]:
    parts = []
    for k in ("d_p_kill", "d_target_hp", "d_own_hp", "trade", "d_p_levelup"):
        vals = sorted(abs(g[k]) for g in gs)
        if vals[-1] == 0:
            continue
        parts.append(f"{k} {statistics.median(vals):.3f}/{vals[int(0.9 * (len(vals) - 1))]:.3f}")
    print(f"  {f} (n={len(gs)}): " + "; ".join(parts))
