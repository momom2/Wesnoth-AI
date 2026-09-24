import sys
import statistics
sys.path.insert(0, ".")
import load
import vectors
import analyze as A
from tools.combat_dominance import COMBOS, admits
G = [g for g in load.games() if "error" not in g]
dec = sum(g["tally"]["decisions"] for g in G)
for block in ("h_not_isolated", "a_kill_in_reach", "a_not_isolated"):
    sel = []
    for g in G:
        for r in g["rows"]:
            if r["flags"].get("class_block") == block:
                d = vectors.corrected_dims(r)
                if any(admits(d, r["tier"], c) for c in COMBOS):
                    r["game"] = g["game"]
                    sel.append(r)
    a = {(r["game"], r["anchor"]) for r in sel}
    gs = [A.gains_of(r) for r in sel]
    print(block, "admitted by some combination if the condition were dropped:", len(sel), "rewrites", len(a), "attacks",
          f"({1000*len(a)/dec:.2f}/1k)", {k: round(statistics.fmean(x[k] for x in gs), 3) for k in ("d_p_kill", "d_target_hp", "d_own_hp")} if gs else "")
