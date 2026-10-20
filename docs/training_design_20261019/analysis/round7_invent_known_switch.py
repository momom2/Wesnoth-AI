"""Round 7, invent stance: sizing the known-switch calibration.

The calibration places coins on one seat only, at p = 1/2, on the switch set S_a of an attack
offset +a: the decisions where a non-attack leads and an attack lies within a nats, so that the
offset would make the attack the argmax. The label sum over S_a then forecasts the match
"offset +a on one seat against own1" exactly up to third-order interactions (midpoint rule).

Inputs (read-only):
- training/metrics/critic_step1_20261009/prior_gaps.games.jsonl: 100 recorded side-games of
  parity3 at 64 slots, offset -1.5, argmax. Per decision: legal count, gaps top1 - top_k for
  k = 2..8 (joint log-prior after the end_turn shift), the top action's type, and the number of
  attacks among the top eight. The record does not say WHICH of positions 2..8 hold the attacks,
  so |S_a| is bounded:
    lower: the top is not an attack and the worst-placed possible best attack is within a
           (k attacks among positions 2..8: the best one is at position <= 9 - k, so its gap is
           at most gaps[7 - k]);
    upper: the top is not an attack, k >= 1, and the runner-up is within a.
  Shares are scaled to own1's 224.6 raw decisions per side-game
  (round5_simplify_critique_decision_units.out), as round 6 did for the band.
- The knob records' first stage (round6_provenance_attack_pairs.out): +2.35 extra attacks per
  side-game per nat of positive attack offset (all games); a lower bound on changed decisions.
- Label variance: E[R^2] = 0.688 x Var(z) = 0.688 x 1.0004 (current_design.md, full-cell
  baseline plus columns); at p = 1/2 one label's variance is E[R^2] / (p(1-p)) = 2.753.
- Prices: $2.51e-5 a one-network game in the pool (round6_provenance_prices.out).
- Elo per unit of z: 347.4 (800 / ln 10).
- Known answers: attack offset > 0 costs -0.2247 +- 0.0459 z per nat on unselected stage-1 games
  (round6_provenance_critique_attack_stage1.out); -0.1791 +- 0.0431 on all games.

Output: |S_a| bounds per side-game for a = 0.35 and 1.0; coin games and dollars for a given SE of
the forecast; the predicted match.
numpy and the standard library only.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "training/metrics/critic_step1_20261009/prior_gaps.games.jsonl"
OWN1_DECISIONS = 224.6
VAR_Z = 1.0004
KEPT = 0.688
P = 0.5
GAME_PRICE = 2.51e-5
ELO_PER_Z = 800 / np.log(10)
PER_NAT = {"stage-1 (unselected)": (-0.2247, 0.0459), "all games": (-0.1791, 0.0431)}


def load():
    rows = []
    n_sg = 0
    for line in open(SRC):
        rec = json.loads(line)
        n_sg += 1
        for rd in rec["readings"]:
            if rd["legal"] < 2 or not rd["gaps"]:
                continue
            g = rd["gaps"] + [np.inf] * (7 - len(rd["gaps"]))
            rows.append((rd["top_type"] == "attack", rd["attacks_top"], g))
    return n_sg, rows


def bounds(rows, a):
    lo = hi = 0
    for top_attack, k, g in rows:
        if top_attack or k < 1:
            continue
        if g[0] < a:
            hi += 1
        if g[min(7 - k, 6)] < a:
            lo += 1
    n = len(rows)
    return lo / n, hi / n


def main():
    n_sg, rows = load()
    n = len(rows)
    print(f"side-games {n_sg}; multi-legal readings {n} ({n / n_sg:.1f} per side-game)")
    label_var = KEPT * VAR_Z / (P * (1 - P))
    print(f"one label's variance at p = {P}: {label_var:.3f}")
    for a in (0.35, 1.0):
        lo, hi = bounds(rows, a)
        print(f"\nattack offset +{a}: |S_a| per own1 side-game between {lo * OWN1_DECISIONS:.1f} and "
              f"{hi * OWN1_DECISIONS:.1f} (shares {lo:.4f}, {hi:.4f})")
        print(f"  knob first stage, lower bound on changed decisions: {2.35 * a:.2f}")
        for name, (b, s) in PER_NAT.items():
            print(f"  predicted match ({name}): {ELO_PER_Z * b * a:+.1f} +- {ELO_PER_Z * s * a:.1f} Elo")
        for n_s in (lo * OWN1_DECISIONS, hi * OWN1_DECISIONS):
            for se_elo in (8.0, 5.0):
                se_z = se_elo / ELO_PER_Z
                games = n_s * label_var / se_z ** 2
                print(f"  |S| {n_s:5.1f}: forecast SE {se_elo:.0f} Elo needs {games:9.0f} one-sided coin games, "
                      f"${games * GAME_PRICE:.2f}")


if __name__ == "__main__":
    main()
