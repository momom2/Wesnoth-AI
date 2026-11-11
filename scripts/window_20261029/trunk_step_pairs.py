"""Paired readings between the match test's arms, by seed (CPU, seconds;
docs/trunk_step_match_prereg_20261111.md on main, departure 1). The arms play
the same seeds, so the same map, factions and seat. Score is 1 for a win of
side A and 0 for a loss; a seed is read only where both games are decisive.

    python scripts/window_20261029/trunk_step_pairs.py C:/Users/momom2/wesnoth_runs/trunk_step_match_20261111 full8 heads
"""
import json
import math
import sys
from pathlib import Path


def games(root: Path, arm: str) -> dict:
    out = {}
    for path in (root / arm / "games").glob("game_*.json"):
        j = json.loads(path.read_text(encoding="utf-8"))
        if j["outcome_a"] in ("win", "loss"):
            out[int(j["seed"])] = j
    return out


def elo(p: float) -> float:
    return 400.0 * math.log10(p / (1.0 - p))


def main() -> None:
    root, arms = Path(sys.argv[1]), sys.argv[2:]
    data = {arm: games(root, arm) for arm in arms}
    for arm, g in data.items():
        n = len(g)
        p = sum(j["outcome_a"] == "win" for j in g.values()) / n
        turns = sorted(j["turns"] for j in g.values())
        print(f"{arm}: {n} decisive, score {p:.4f} ({elo(p):+.1f} Elo), turns mean {sum(turns) / n:.2f} "
              f"median {turns[n // 2]}, forwards a side-A game {sum(j['forwards_a'] for j in g.values()) / n:.0f}")
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            seeds = sorted(set(data[a]) & set(data[b]))
            same = sum((data[a][s]["scenario_id"], data[a][s]["faction_a"], data[a][s]["faction_b"], data[a][s]["side_a"])
                       == (data[b][s]["scenario_id"], data[b][s]["faction_a"], data[b][s]["faction_b"], data[b][s]["side_a"])
                       for s in seeds)
            d = [(data[a][s]["outcome_a"] == "win") - (data[b][s]["outcome_a"] == "win") for s in seeds]
            n = len(d)
            mean = sum(d) / n
            se = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1) / n)
            pa = sum(data[a][s]["outcome_a"] == "win" for s in seeds) / n
            pb = sum(data[b][s]["outcome_a"] == "win" for s in seeds) / n
            unpaired = math.sqrt(pa * (1 - pa) / n + pb * (1 - pb) / n)
            print(f"{a} - {b}: {n} shared seeds ({same} with the same map, factions and seat), score {mean:+.4f} "
                  f"+- {se:.4f} paired ({mean / se:+.2f} SE; unpaired SE {unpaired:.4f}); "
                  f"Elo {elo(pa) - elo(pb):+.1f}, agree on {sum(x == 0 for x in d) / n:.3f}")


if __name__ == "__main__":
    main()
