"""Round 5 (invent): where the turn-over rule would part from the -1.5 offset, as far as the
record shows it, and how many attack candidates sit near the top of the prior.

Record: training/metrics/critic_step1_20261009/prior_gaps.games.jsonl (100 recorded side-games
of `parity3`; per decision the gaps in nats from the top action to the next seven, under the
decode's end_turn offset c = -1.5, tools/prior_gaps.py at ea8536f; top_type; attacks among
the top eight).

The turn-over rule ends the side-turn iff the raw joint prior gives P(end) >= 1/2, i.e. iff
end_turn's mass is at least the total mass of all acting actions; otherwise it plays the joint
argmax among acting actions. The offset rule ends iff end_turn is the top action after its
logit is shifted by c. At a decision where the shifted top action is end_turn, the raw mass
ratio of an acting action a to end_turn is e^c * e^-gap(a), so the turn-over rule continues
there iff sum over acting actions of e^-gap(a) > e^-c = 4.48. The top seven gaps give a lower
bound on that sum, so the share counted below is a lower bound on where the turn-over rule
acts and the offset rule ends. (Where an acting action is the shifted top, the turn-over rule
can end only if raw P(end) >= 1/2; the record does not say which gap is end_turn's there, so
that direction is not measured here.)

Also reported: the distribution of the top-seven ratio sum at end-top decisions, and, at all
multi-legal decisions, the number of attacks among the top eight.
numpy and the standard library only.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[3] / "training/metrics/critic_step1_20261009/prior_gaps.games.jsonl"
C = -1.5


def main():
    n_sg = 0
    end_top_sums, end_top_n, multi, attacks_top, attack_top1 = [], 0, 0, [], 0
    for line in open(SRC):
        row = json.loads(line)
        n_sg += 1
        for rd in row["readings"]:
            if rd["legal"] < 2 or not rd["gaps"]:
                continue
            multi += 1
            attacks_top.append(rd["attacks_top"])
            attack_top1 += rd["top_type"] == "attack"
            if rd["top_type"] == "end_turn":
                end_top_n += 1
                end_top_sums.append(sum(math.exp(-g) for g in rd["gaps"]))
    s = np.array(end_top_sums)
    thr = math.exp(-C)
    print(f"side-games {n_sg}; multi-legal decisions {multi}; end_turn top under the offset: "
          f"{end_top_n} ({end_top_n / n_sg:.1f} per side-game)")
    print(f"turn-over rule continues where the offset ends (lower bound): {np.mean(s > thr):.4f} "
          f"of those decisions ({np.sum(s > thr) / n_sg:.2f} per side-game)")
    print("top-seven ratio sum at end-top decisions, quantiles 10/50/90/99: "
          + ", ".join(f"{q:.2f}" for q in np.percentile(s, [10, 50, 90, 99])))
    for k in (1.0, 2.0, 3.0):
        print(f"  share with the sum above {k}: {np.mean(s > k):.3f}")
    a = np.array(attacks_top)
    print(f"\nattacks among the top eight: mean {a.mean():.2f} per multi-legal decision; "
          f"share with >= 1: {np.mean(a >= 1):.3f}; >= 2: {np.mean(a >= 2):.3f}; "
          f"per side-game with >= 2: {np.sum(a >= 2) / n_sg:.1f}")
    print(f"attack as the top action: {attack_top1 / n_sg:.1f} per side-game")


if __name__ == "__main__":
    sys.exit(main())
