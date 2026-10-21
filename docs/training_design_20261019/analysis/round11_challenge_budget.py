"""Round 11 (challenge stance): prices, standard errors and the step-size rule of the certified
trust region. Arithmetic on measured inputs; no record is read except the constants below.

Measured inputs (record, how read):
  POOL_GAME      $2.598e-5  pool game, 120 in flight, $0.577/h host (analysis/round9_simplify_labels.out)
  TWO_NET_GAME   $8.848e-5  two-network match game, 20 connections, mean batch 5.5
                            (round9_simplify_labels.out; training/metrics/own2_base_20261020/
                            own2_vs_own1.server_0.json: max_batch 20, mean_batch 5.54)
  HOST           $0.577 an hour (seed section 7)
  TRAIN_RATE     1,477 positions a second, configuration B on a 4090 (BACKLOG, trainer_calibration_20261015)
  DECISIONS      475.7 a game, 237.9 a side-game, own2 v own1 at the reference decode
                 (analysis/round11_provenance_census.out); at temperature 1 unmeasured
  ENCODE         7,003 positions a second, the own-trajectory candidate walk on a box CPU
                 (training/metrics/memory_own_round2_20261011/candidates.log, last line), priced
                 serial: an upper side, since it can overlap the GPU's training
  FORWARD        a held-out position's forward priced as a trained position: an upper side
                 (forward-only speed is unmeasured)
Derived: ELO_PER_Z = 1 / (2 ln10 / 1600) = 347.4 Elo per unit of mean result near 50%.
numpy and the standard library only.
"""
import math

import numpy as np

POOL_GAME = 2.598e-5
TWO_NET_GAME = 8.848e-5
HOST = 0.577
TRAIN_RATE = 1477.0
DECISIONS = 475.7
SIDE_DECISIONS = 237.9
ENCODE_RATE = 7003.0
ELO_PER_Z = 1.0 / (2.0 * math.log(10.0) / 1600.0)
RNG = np.random.default_rng(20261021)


def position_price():
    return HOST / 3600.0 / TRAIN_RATE


def prices():
    pp = position_price()
    print("[prices]")
    print(f"  a trained position ${pp:.3e}; a game's decisions trained ${pp * DECISIONS:.3e} "
          f"({pp * DECISIONS / POOL_GAME:.2f}x its generation ${POOL_GAME:.3e})")
    print(f"  encoding a game's positions ${encode_price() * DECISIONS:.3e} (serial, upper side); a held-out "
          f"game's forward <= ${pp * DECISIONS:.3e}")
    print(f"  Elo per unit z near 50%: {ELO_PER_Z:.1f}; 800 decisive games at sd(z) 1: SE {ELO_PER_Z / math.sqrt(800):.1f} Elo")
    # two-network generation in the pool: if a forward's cost is flat in its batch (generation round 2,
    # BACKLOG), two networks each serving half the games double the cost a decision
    print(f"  cross-play game: measured ${TWO_NET_GAME:.3e} in the shallow match configuration; "
          f"${2 * POOL_GAME:.3e} if a pool forward's cost is flat in its batch (two half batches)")


def encode_price():
    return HOST / 3600.0 / ENCODE_RATE


def round_cost(n_train, n_held, n_cert, cert_price):
    """Generation, then training (trained positions, plus the held-out forwards and every game's
    encoding at their upper sides)."""
    pp = position_price()
    gen = (n_train + n_held) * POOL_GAME + n_cert * cert_price
    trained = n_train * DECISIONS + n_cert * SIDE_DECISIONS
    extra = n_held * DECISIONS * pp + (n_train + n_held + n_cert) * DECISIONS * encode_price()
    hours_train = trained / TRAIN_RATE / 3600.0
    return gen, trained * pp + extra, hours_train


def rounds():
    print("\n[a round: n_train self-play games trained, n_held held out (generated only), "
          "n_cert cross-play games (newest seat trained)]")
    for n_train, n_held, n_cert in ((20000, 10000, 10000), (40000, 10000, 10000), (80000, 20000, 10000)):
        for label, cp in (("measured shallow", TWO_NET_GAME), ("flat-batch pool", 2 * POOL_GAME)):
            gen, train, h = round_cost(n_train, n_held, n_cert, cp)
            print(f"  {n_train:6d}/{n_held:6d}/{n_cert:6d} cross-play {label:16s}: generation ${gen:5.2f}, "
                  f"training+encode+forecast ${train:5.2f} ({h:4.1f} h), total ${gen + train:5.2f}; "
                  f"certification share {(n_cert * (cp - POOL_GAME)) / (gen + train):.3f}")


def se_cert(n_cert, sd_y=1.0):
    return ELO_PER_Z * sd_y / math.sqrt(n_cert)


def se_forecast(eps, n_held, sd_y=1.0):
    """First-order forecast G = mean over held-out game-sides of sum_t w_t (r_t - 1).
    Per game-side variance ~ sd_y^2 * n_side * E[(r - 1)^2], E[(r - 1)^2] ~ 2 eps for a small step
    with per-decision KL eps; both sides of a held-out self-play game count (independent ratios)."""
    per_side = sd_y * math.sqrt(SIDE_DECISIONS * 2.0 * eps)
    return ELO_PER_Z * per_side / math.sqrt(2 * n_held)


def standard_errors():
    print("\n[standard errors, sd(y) 1 as the bound; luck correction lowers it]")
    for n in (5000, 10000, 20000, 50000):
        print(f"  certification over {n:6d} cross-play games: {se_cert(n):.2f} Elo")
    for eps in (0.002, 0.005, 0.01, 0.02, 0.05):
        row = ", ".join(f"{n} held out {se_forecast(eps, n):5.2f}" for n in (5000, 10000, 20000, 50000))
        print(f"  forecast at per-decision KL {eps:.3f}: {row} Elo")
    for p in (0.5, 0.75, 0.9):
        for n in (3200, 6400):
            se = (400 / math.log(10)) / math.sqrt(n * p * (1 - p))
            print(f"  a {n}-game match read at win rate {p}: SE {se:.1f} Elo")


def step_rule():
    print("\n[step size: first-order gain a*sqrt(eps), second-order loss C*eps (a quadratic in the step);"
          " eps* = (a / 2C)^2; at eps = k eps* the gain is (2 sqrt(k) - k) of the best, and the ratio"
          " actual / first-order is rho = 1 - sqrt(k) / 2]")
    for k in (1 / 16, 1 / 4, 1 / 2, 1, 2, 4, 9, 16):
        print(f"  k {k:7.4f}: gain {2 * math.sqrt(k) - k:+.3f} of the best; rho {1 - math.sqrt(k) / 2:+.3f}; "
              f"rule's next radius x{1 / (4 * (1 - (1 - math.sqrt(k) / 2)) ** 2):.3f}")
    print("  rule: next eps = eps / (2 (1 - rho))^2, capped at x4 and /4 a round; rho = 1/2 at the best step")


def noisy_rule():
    print("\n[the rule under noise: true k 1/4 (rho 0.75) or 4 (rho 0); forecast G with SE sG, reading R = rho G"
          " with SE sR; share of rounds the capped rule moves the radius the wrong way]")
    for G in (5.0, 10.0, 20.0, 40.0):
        for sG, sR in ((5.2, 3.5), (3.7, 2.5)):
            wrong = []
            for k in (0.25, 4.0):
                rho = 1 - math.sqrt(k) / 2
                g = G + sG * RNG.standard_normal(100000)
                r = rho * G + sR * RNG.standard_normal(100000)
                rho_hat = np.where(g > 2 * sG, r / np.maximum(g, 1e-9), 0.5)  # no move unless G is resolved
                step = 1.0 / (2 * (1 - np.clip(rho_hat, -10, 0.99))) ** 2
                step = np.clip(step, 0.25, 4.0)
                wrong.append(float(np.mean(step < 1.0)) if k < 1 else float(np.mean(step > 1.0)))
            print(f"  G {G:4.0f} Elo, sG {sG}, sR {sR}: wrong way at k 1/4 {wrong[0]:.3f}, at k 4 {wrong[1]:.3f}")


def bench():
    pp = position_price()
    print("\n[matched-KL optimizer bench on recorded temperature-1 games: each arm trains the same games to"
          " the same KL; held-out first-order gain read paired across arms]")
    for games, arms in ((5000, 6), (10000, 6)):
        pos = games * DECISIONS
        print(f"  {games} games, {arms} arms: {pos * arms / TRAIN_RATE / 3600:.1f} h, ${pos * arms * pp:.2f} "
              f"(+ generation ${games * POOL_GAME:.2f}, + a 10,000-game held-out set ${10000 * POOL_GAME:.2f})")


def stages():
    pp, ep = position_price(), encode_price()
    game_train = DECISIONS * pp
    game_fwd = DECISIONS * pp          # upper side
    game_enc = DECISIONS * ep
    print("\n[stages; generation priced as argmax pool games, every cost scales with the temperature-1 game length]")
    s1 = 20000 * POOL_GAME + 6400 * TWO_NET_GAME
    print(f"  stage 1: 20,000 own2 T=1 self-play games ${20000 * POOL_GAME:.2f} + 6,400-game claim own2@T1 v own1 "
          f"<= ${6400 * TWO_NET_GAME:.2f}: ${s1:.2f}")
    plant = 10000 * (game_train + game_enc) + 5000 * (game_fwd + game_enc) + 5000 * TWO_NET_GAME + 5000 * SIDE_DECISIONS * pp
    s2 = plant
    print(f"  stage 2: a planted round (10,000 games trained, 5,000 held out, 5,000 cross-play) ${plant:.2f}; "
          f"the revert check re-scores the same cross-play games with the plant's sign flipped, at no cost")
    arms, games, held = 6, 5000, 5000
    s3 = games * game_enc + held * game_enc + arms * (games * game_train + held * game_fwd)
    print(f"  stage 3: {arms} arms on {games} games, each read on {held} held-out games: ${s3:.2f}")
    print(f"  stages 1-3: ${s1 + s2 + s3:.2f}")
    from math import erf, sqrt
    print(f"  accept a step unless R < -2 SE: false revert at a true zero step {0.5 * (1 - erf(2 / sqrt(2))):.4f}")


def budget():
    print("\n[budget from $82]")
    for n_train, n_held, n_cert in ((40000, 10000, 10000),):
        for label, cp in (("measured shallow", TWO_NET_GAME), ("flat-batch pool", 2 * POOL_GAME)):
            gen, train, _ = round_cost(n_train, n_held, n_cert, cp)
            claim = 6400 * TWO_NET_GAME
            per3 = 3 * (gen + train) + claim
            print(f"  round {n_train}/{n_held}/{n_cert} ({label}): ${gen + train:.2f}; three rounds and a "
                  f"6,400-game claim ${per3:.2f}; rounds within $70 after stages 1-3: {int(70 // (per3 / 3))}")


if __name__ == "__main__":
    prices()
    rounds()
    standard_errors()
    step_rule()
    noisy_rule()
    bench()
    stages()
    budget()
