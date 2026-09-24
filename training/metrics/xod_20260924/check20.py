import collections
import sys
sys.path.insert(0, r".")
from pathlib import Path
import load
from tools.analysis.dominance_count import corpus_games
first20 = {g for _s, g, _w in corpus_games(Path(r"C:/Users/amaur/Desktop/Perso/projects/Wesnoth_AI/replays_dataset_imitation"), 20)}
G = [g for g in load.games() if g["game"] in first20]
print(len(G), "games; errors", sum(1 for g in G if "error" in g))
dec = sum(g["tally"]["decisions"] for g in G)
print("decisions", dec)
for name, combo in (("R0", load.COMBOS[0]), ("loosest", load.LOOSEST)):
    opp = {(g["game"], r["anchor"]) for g in G for r in g["rows"] if load.admitted(r, combo)}
    rw = sum(1 for g in G for r in g["rows"] if load.admitted(r, combo))
    per = collections.Counter(r["cls"] for g in G for r in g["rows"] if load.admitted(r, combo))
    print(name, "opportunities", len(opp), "rewrites", rw, dict(per))
