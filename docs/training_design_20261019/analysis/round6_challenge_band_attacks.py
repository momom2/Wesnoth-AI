"""Round 6 (challenge): how many near-tie decisions involve an attack, from the record.

Record: training/metrics/critic_step1_20261009/prior_gaps.games.jsonl (100 recorded side-games of
`parity3` at 64 slots, offset -1.5, argmax; per decision: legal count, the gaps top1 - top_k for
k = 2..8 in joint log-prior after the end_turn shift, the type of the top action, and the number
of attacks among the top eight).

The record does not name the runner-up's kind. Two bounds on "the top-two pair involves an
attack" at a band decision (G = top1 - top2 < w):
  lower: the top action is an attack;
  upper: at least one attack sits among the top eight.
Also: decisions with at least two attacks among the top eight (attack-against-attack choices are
possible there), and the same counts at margins up to 1.5 nats (the end_turn offset's reach).

Shares are per multi-legal reading; "per own1 side-game" multiplies a share by own1's 224.6 raw
decisions per side-game (round5_simplify_critique_decision_units.out), the unit the current
design uses for its band of 76.1.
numpy and the standard library only.
"""
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[3] / "training/metrics/critic_step1_20261009/prior_gaps.games.jsonl"
OWN1_DECISIONS_PER_SIDE_GAME = 224.6
WIDTHS = (0.35, 0.7, 1.5)


def load():
    gaps, top_attack, attacks = [], [], []
    n_side_games = 0
    for line in open(SRC):
        row = json.loads(line)
        n_side_games += 1
        for rd in row["readings"]:
            if rd["legal"] < 2 or not rd["gaps"]:
                continue
            gaps.append(rd["gaps"][0])
            top_attack.append(rd["top_type"] == "attack")
            attacks.append(rd["attacks_top"])
    return n_side_games, np.array(gaps), np.array(top_attack), np.array(attacks)


def main():
    n_sg, gaps, top_attack, attacks = load()
    n = len(gaps)
    print(f"side-games {n_sg}; multi-legal readings {n} ({n / n_sg:.1f} per side-game)")
    print(f"all readings: top is an attack {top_attack.mean():.3f}; >= 1 attack in top 8 "
          f"{np.mean(attacks >= 1):.3f}; >= 2 attacks {np.mean(attacks >= 2):.3f}")
    for w in WIDTHS:
        band = gaps < w
        share = band.mean()
        lo = np.mean(band & top_attack)
        hi = np.mean(band & (attacks >= 1))
        two = np.mean(band & (attacks >= 2))
        u = OWN1_DECISIONS_PER_SIDE_GAME
        print(f"\nw = {w:.2f}: band share {share:.3f} ({share * u:.1f} per own1 side-game)")
        print(f"  pair involves an attack, lower bound (top is attack): share {lo:.3f}, "
              f"{lo * u:.1f} per own1 side-game, {lo / share:.3f} of the band")
        print(f"  upper bound (>= 1 attack in top 8): share {hi:.3f}, {hi * u:.1f} per own1 side-game, "
              f"{hi / share:.3f} of the band")
        print(f"  >= 2 attacks in top 8: share {two:.3f}, {two * u:.1f} per own1 side-game, "
              f"{two / share:.3f} of the band")


if __name__ == "__main__":
    sys.exit(main())
