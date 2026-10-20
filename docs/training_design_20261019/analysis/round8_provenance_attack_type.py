"""Round 8, provenance stance: the mean advantage of an attack-promoting switch, bounded from records,
and the sampling error of the band census the plan uses until S0(a).

1. Attack-promoting switches. A positive attack offset o on own1's decode switches exactly the
   decisions whose top action is not an attack and whose best attack is within o nats of it. By the
   performance-difference lemma the knobbed side's mean result moves by the sum of those switches'
   own1-advantages (at the states the knobbed player visits). The unselected stage-1 knob games give
   that move as -0.2247 +- 0.0459 z per nat of positive offset (round6_provenance_critique_attack_stage1.out B,
   a kinked fit linear in o over offsets 0.09 to 1.40, rerun identical in round 8).
   The number of switches at offset o is at most F_up(o): decisions (legal >= 2) whose top is not an
   attack, whose top-two gap is below o, and that either hold an attack among the top eight or have
   eight actions within o. Read on training/metrics/critic_step1_20261009/prior_gaps.games.jsonl
   (parity3 at 64 slots, offset -1.5, 100 side-games; the record does not name the runner-up's kind),
   per multi-legal reading, times own1's 224.6 decisions per side-game (the unit of the design's 76.1).
   So the mean advantage per attack-promoting switch over gaps in [0, o] is at most -0.2247 o / F_up(o),
   and at most -0.1329 o / F_up(o) at the fit's upper 2-SE slope.
2. Band census. The design's 76.1 band decisions per side-game is parity3's band share 0.339 times
   own1's decision count; its sampling error over the 100 side-games, by bootstrap over side-games.
numpy and the standard library only.
"""
import json
import math
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[3] / "training/metrics/critic_step1_20261009/prior_gaps.games.jsonl"
OWN1_UNIT = 224.6
SLOPE, SLOPE_SE = -0.2247, 0.0459
ELO = 800 / math.log(10)
RNG = np.random.default_rng(20261020)


def load():
    games = []
    for line in open(SRC):
        row = json.loads(line)
        rec = []
        for rd in row["readings"]:
            if rd["legal"] < 2 or not rd["gaps"]:
                continue
            gaps = rd["gaps"]
            eighth = gaps[6] if len(gaps) >= 7 else math.inf
            rec.append((gaps[0], rd["top_type"] == "attack", rd["attacks_top"], rd["legal"], eighth))
        games.append(rec)
    return games


def main():
    games = load()
    readings = [r for g in games for r in g]
    n = len(readings)
    g0 = np.array([r[0] for r in readings])
    top_att = np.array([r[1] for r in readings])
    att8 = np.array([r[2] for r in readings])
    legal = np.array([r[3] for r in readings])
    eighth = np.array([r[4] for r in readings])
    print(f"1. attack-promoting switches ({len(games)} side-games, {n} multi-legal readings)")
    for o in (0.35, 0.7, 1.0, 1.4):
        can = (~top_att) & (g0 < o) & ((att8 >= 1) | ((legal > 8) & (eighth < o)))
        f_up = can.mean() * OWN1_UNIT
        point = SLOPE * o / f_up
        hi = (SLOPE + 2 * SLOPE_SE) * o / f_up
        print(f"  o {o:.2f}: switches per own1 side-game <= {f_up:5.1f}; mean advantage per switch <= {point:+.4f} "
              f"(upper 2-SE slope: <= {hi:+.4f}); result move {SLOPE * o:+.3f} z = {ELO * SLOPE * o:+.0f} Elo")
    in_band_top_non_attack = ((~top_att) & (g0 < 0.35)).mean() * OWN1_UNIT
    print(f"  band decisions with a non-attack top: {in_band_top_non_attack:.1f} per own1 side-game")

    print("\n2. band census at w 0.35")
    per_game = np.array([sum(1 for r in g if r[0] < 0.35) for g in games], float)
    per_game_all = np.array([len(g) for g in games], float)
    share = per_game.sum() / per_game_all.sum()
    boot = []
    for _ in range(4000):
        idx = RNG.integers(0, len(games), len(games))
        boot.append(per_game[idx].sum() / per_game_all[idx].sum())
    boot = np.array(boot)
    print(f"  band per parity3 side-game {per_game.mean():.2f} (sd over side-games {per_game.std():.1f}); "
          f"share {share:.4f}, bootstrap SE {boot.std():.4f}")
    print(f"  own1 units: {share * OWN1_UNIT:.1f} +- {boot.std() * OWN1_UNIT:.1f} band decisions per side-game "
          f"(95% {np.percentile(boot, 2.5) * OWN1_UNIT:.1f} to {np.percentile(boot, 97.5) * OWN1_UNIT:.1f}); "
          f"sampling only, the parity3-to-own1 transfer not included")


if __name__ == "__main__":
    main()
