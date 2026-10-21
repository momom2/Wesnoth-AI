"""Round 11, simplify stance: the derived numbers of the proposal
(round11_simplify.md), each from measured inputs named beside it.
Standard library only.

[M] inputs (record, how it was read):
  GEN      $2.598e-5 a self-play game at $0.577/h: 50,000 coin games in 8,105 s
           (round9_simplify_labels.out [prices]); coins change no play cost.
  TWO_NET  $8.848e-5 a decisive two-network game (same file).
  WALL_G   155.0 +- 3.3 ms a game, FIXED 69 +- 30 s a match
           (round10_provenance_r_link.out section 4).
  POS_SIDE 230.4 decisions a side-game: 1,107,818 candidates over 4,808 own
           game-sides of own1 at argmax (memory_own_round2_20261011/candidates.log,
           own_trajectories CANDIDATES_DONE and the selection's "own": 4808).
  ENC      7,003 candidate positions a second on that box's CPU (same log).
  TRAIN    1,476.9 positions a second, configuration B with the compiled trunk
           (training/metrics/trainer_calibration_20261015/memory.json).
  LUCK     0.291 CPU seconds a game, the walk's own `secs` over the 1,600
           argmax self-play rows (round11_simplify_rows.out), on 31 usable cores
           (match_luck_20261019/box.txt).
  BUDGET   $82.34 left if the $5 window counted (current_design.md section 4),
           less match L, about $0.9 (BACKLOG.md, 0.30.5).
  BRIER    the seed's value head by turn bucket, states and Brier
           (training/metrics/value_head/seed_by_phase.json).
  POST     the LessWrong post's runs: 393K tokens x 2,372 steps (val 3.248);
           50M tokens x 350 steps fresh data (val 3.230).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

PRICE = 0.577
PRICE_LOW = 0.42
GEN = 2.598e-5
TWO_NET = 8.848e-5
WALL_G = 0.155
FIXED = 69.0
POS_SIDE = 1_107_818 / 4_808
ENC = 7_003.0
TRAIN = 1_476.9
BUDGET = 82.34 - 0.9
ELO_PER_Z = 800 / math.log(10)          # d Elo / d z at z = 0, z = 2p - 1
ROOT = Path(__file__).resolve().parents[3]


def per_game():
    pos = 2 * POS_SIDE
    train_s = pos / TRAIN
    enc_s = pos / ENC
    train_cost = train_s / 3600 * PRICE
    enc_cost = enc_s / 3600 * PRICE
    print("[per game]")
    print(f"  positions a game {pos:.1f}; train {train_s:.3f} s ({train_cost:.3e} $); encode {enc_s:.4f} s "
          f"({enc_cost:.3e} $, serial); generation {GEN:.3e} $")
    total = GEN + train_cost + enc_cost
    share = GEN / total
    print(f"  all-in a trained game {total:.3e} $ (generation share {share:.2f}); "
          f"without the serial encode {GEN + train_cost:.3e} $")
    low = total * PRICE_LOW / PRICE
    print(f"  at $0.42/h {low:.3e} $")
    wall = WALL_G + train_s + enc_s
    print(f"  wall a game in the serial loop {wall:.3f} s")
    return total, wall


def batch_rule(total, wall):
    print("\n[batch] B* = sqrt(B_noise * c_step / c_game) minimises the cost of reaching a loss under")
    print("  McCandlish et al.'s (S/S_min - 1)(E/E_min - 1) = 1 with B_crit = B_noise; c in seconds of the serial loop")
    for c_step in (FIXED, FIXED + 90.0, FIXED + 180.0):
        row = []
        for bn in (1_000, 3_000, 11_593, 30_000, 100_000):
            b = math.sqrt(bn * c_step / wall)
            over = c_step / (c_step + b * wall)
            row.append(f"Bn {bn:>7,}: B* {b:>6,.0f} (overhead {over:.2f})")
        print(f"  c_step {c_step:.0f} s: " + "; ".join(row))
    # cost relative to the optimum at B = r B*, in units of S_min * Bn * c_game
    print("  relative cost at B = r B* (Bn 11,593, c_step 159 s): ", end="")
    bn, c_step = 11_593, FIXED + 90.0
    bstar = math.sqrt(bn * c_step / wall)

    def cost(b):
        return bn * wall + b * wall + c_step + c_step * bn / b

    print("; ".join(f"r {r}: {cost(r * bstar) / cost(bstar):.2f}" for r in (0.25, 0.5, 1, 2, 4, 10)))


def budget(total):
    print("\n[budget]")
    games = BUDGET / total
    print(f"  ${BUDGET:.2f} buys {games:,.0f} trained games at $0.577/h, {BUDGET / (total * PRICE_LOW / PRICE):,.0f} at $0.42/h")
    s0_gen = 40_000 * GEN
    s0_grad = 40_000 * 2 * POS_SIDE / TRAIN / 3600 * PRICE
    s0_gap = 6_400 * TWO_NET
    s0_off = 6_400 * GEN
    s0 = s0_gen + s0_grad + s0_gap + s0_off
    print(f"  S0: 40,000 games {s0_gen:.2f}; gradient pass {s0_grad:.2f}; gap match 6,400 two-network {s0_gap:.2f}; "
          f"offset match 6,400 one-network {s0_off:.2f}; total {s0:.2f}")
    s1 = 3 * 12_800 * TWO_NET
    print(f"  S1: three 12,800-game two-network matches {s1:.2f}")
    s2 = 10 * 2_000 * total + 10 * (FIXED + 90) / 3600 * PRICE
    print(f"  S2: 10 steps of 2,000 games {s2:.2f}")
    claim = 12_800 * TWO_NET
    s3 = 100_000 * total + 2 * claim + 50 * (FIXED + 90) / 3600 * PRICE
    print(f"  S3: 100,000 games, 50 step overheads, two claims {s3:.2f}")
    left = BUDGET - s0 - s1 - s2 - s3
    print(f"  left after S3 {left:.2f}: {left / total:,.0f} trained games before claims")
    se = 400 / math.log(10) * 2 / math.sqrt(12_800)   # SE in Elo of a 50% match, 12,800 decisive games
    print(f"  claim match 12,800 decisive games: {claim:.2f} $, SE {se:.2f} Elo")


def trust_region():
    print("\n[trust-region bounds] TRPO's J(pi') >= L(pi') - C max KL, C = 4 eps gamma / (1 - gamma)^2")
    for h in (230.4, 460.8):
        c = 4 * 2.0 * h * h          # eps = max |A| <= 2 in result units, 1/(1 - gamma) = H
        print(f"  horizon {h:.0f}: C {c:.3g}; the bound certifies anything only if max KL < {2.0 / c:.2g} nats a decision")
    kl = 0.51 * math.log(0.51 / 0.5) + 0.49 * math.log(0.49 / 0.5)
    print(f"  moving one binary choice from 0.50 to 0.51: KL {kl:.2e} nats")


def value_baseline():
    d = json.loads((ROOT / "training/metrics/value_head/seed_by_phase.json").read_text(encoding="utf-8"))
    num = den = 0.0
    print("\n[value baseline] residual of z after the seed's value head, 4 x Brier against Var(z) = 1, by bucket")
    for k, b in d["buckets"].items():
        f = 4 * b["brier_head"]
        print(f"  turns {k}: states {b['n_states']}, factor {f:.3f}")
        num += f * b["n_states"]
        den += b["n_states"]
    print(f"  state-weighted factor {num / den:.3f}: {1 / (num / den):.2f}x effective games")


def end_turn_check():
    """The real-signal check of S0: the outcome-credit slope dJ/de along the
    end_turn logit, from the logged p(end) of 40,000 self-play games, against
    the offset match. Per side-game, Var(sum_i (1[end] - p_i) R) <= E[R^2] *
    sum_i p_i (1 - p_i) <= E[R^2] * (expected end_turns = side-turns), with
    20.08 turns a game at argmax (round11_simplify_rows.out) and E[R^2] <= 1."""
    turns, r2, sides = 20.08, 1.0, 80_000
    se_slope = math.sqrt(turns * r2 / sides)
    se_forecast = 1.5 * se_slope * ELO_PER_Z
    se_match = ELO_PER_Z / math.sqrt(6_400)
    print("\n[end_turn check]")
    print(f"  slope SE <= {se_slope:.4f} z per logit; forecast of the -1.5 offset SE <= {se_forecast:.1f} Elo; "
          f"6,400-game match SE {se_match:.2f} Elo; difference SE <= {math.hypot(se_forecast, se_match):.1f} Elo")


def misc():
    print("\n[misc]")
    print(f"  Elo per unit of mean result near 0: {ELO_PER_Z:.1f}; 1 Elo = {1 / ELO_PER_Z:.5f} z")
    print("  winners-only imitation (reward 1[win], baseline 0) against +-1: E[(y-b)^2] at p 0.5: "
          f"{0.5:.2f} against {0.25:.2f}, twice the variance")
    small = 393_216 * 2_372
    big = 50e6 * 350
    print(f"  post: small batch {small / 1e9:.3f}B tokens in 2,372 steps; fresh 50M x 350 = {big / 1e9:.1f}B tokens: "
          f"{big / small:.1f}x the data for {2_372 / 350:.1f}x fewer steps")
    luck_s = 0.291 / 31
    print(f"  luck walk {luck_s:.4f} s of wall a game on 31 cores ({1 / luck_s:.0f} games a second)")


def main():
    total, wall = per_game()
    batch_rule(total, wall)
    budget(total)
    trust_region()
    value_baseline()
    end_turn_check()
    misc()


if __name__ == "__main__":
    main()
