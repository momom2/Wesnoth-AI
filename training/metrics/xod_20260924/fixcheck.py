"""Effect of the dead-unit status reading on admission."""
import collections
import sys
sys.path.insert(0, ".")
import load
import vectors
from tools.combat_dominance import COMBOS, admits, dim_passes
G = [g for g in load.games() if "error" not in g]
dec = sum(g["tally"]["decisions"] for g in G)
print("games", len(G), "decisions", dec, "attacks", sum(g["tally"]["attacks"] for g in G))
res = collections.Counter()
opp = collections.defaultdict(set)
examples = []
for g in G:
    for r in g["rows"]:
        if r["flags"].get("class_block"):
            continue
        old, new = vectors.dims_of(r), vectors.corrected_dims(r)
        changed = [(a.name, a.sym, b.sym) for a, b in zip(old, new) if a.sym != b.sym]
        if changed:
            res[("status_sym_changed", r["cls"])] += 1
        for label, cs in (("R0", [COMBOS[0]]), ("any36", COMBOS)):
            o = any(admits(old, r["tier"], c) for c in cs)
            n = any(admits(new, r["tier"], c) for c in cs)
            if o != n:
                res[(label, r["cls"], "gained" if n else "lost")] += 1
                if n:
                    opp[(label, r["cls"])].add((g["game"], r["anchor"]))
                    # blocked only by enemy dead-status dims under the old reading?
                    c = next(c for c in cs if admits(new, r["tier"], c))
                    fails = [d.name for d in old if dim_passes(d, c) is False]
                    only = fails and all(d.startswith("enemy_") and vectors.is_dead_status(vectors.Dim(d, "binary", "=")) or d.startswith(("slowed:t", "poisoned:t")) for d in fails)
                    res[(label, r["cls"], "gained_only_dead_enemy_status" if only else "gained_other")] += 1
                    if len(examples) < 6:
                        examples.append((g["game"][:40], r["cls"], r["anchor"], changed, r["gains"]))
                else:
                    if len(examples) < 12:
                        examples.append(("LOST", g["game"][:40], r["cls"], r["anchor"], changed, r["gains"]))
for k, v in sorted(res.items()):
    print(k, v)
for k, v in opp.items():
    print("opportunities", k, len(v))
for e in examples:
    print(e)
