"""Round 8, critique of the invent stance (chance-effect supervision, CES). numpy and the standard library only.

1. Scale of S4's CES readings. The invent sizes the head's plant at 8 batches (800,000 games) and the
   kept-ratio readings at 100,000 held-out games. S4's distillation, costed at $2.1 by the
   own-trajectory recipe (docs/memory_own_trajectories_prereg_20261011.md, "Measured": a pass of
   2,000,384 positions, half own, half human; the teacher pass at about 390 positions a second),
   sees about 1,000,192 own positions. The head trains on the attack positions among them.
   Reported: the game-equivalents of fights the head sees, the plant's one-coefficient SE at that
   scale (the invent's own formula, round8_invent_chance_ledger.py part D), the paired SE of a kept
   ratio (the invent's scaling of 0.0011 at 18,800 games), the ridge overfit term r*d/N for
   r_frozen, and the cost of forwarding 100,000 held-out games through N.

2. A mis-join shared by the plant and the label passes the plant. Toy game: K strikes with hit
   chances c_k and true effects Delta_k on a +-1 result; the label path reads H through a wrong
   join (a permutation of strikes within the game). z' is built from the same mis-joined H, as the
   invent specifies nothing else. The effect of a hit depends on a pre-strike context (slope 0.04).
   A linear head [1, F*d, context] is fitted by least squares on the IPW labels: the plant's
   coefficient and the context slope are read with and without the mis-join.

3. A post-strike input breaks the control variate. Toy: a coin D (P 0.25) changes the next strike's
   chance; a control variate (H - c) * psi with psi depending on H (the record's damage field is 0
   on a miss) is not centred, and the coin label's mean moves.
"""
import math

import numpy as np

RNG = np.random.default_rng(20261020)

R = 0.688                  # cell + columns (round6_provenance_cells.out)
SS_HP = 5486.0             # mean summed law variance of hp luck per game (round8_invent_chance_ledger.out, part D)
KAPPA = 0.0005             # the invent's plant, per hp
SHARE = 0.5                # F ~ Bernoulli(1/2)
OWN_POSITIONS = 2_000_384 / 2   # the recipe's own half (prereg, "Measured")
DECISIONS = 449.1          # decisions a game (round5_synthesis_outcomes.out)
TEACHER_RATE = 390.0       # positions a second, own_targets teacher pass (prereg, "Measured")
BOX = 0.463                # $/h of that box (prereg); the design uses 0.42
SE_PAIRED_18800 = 0.0011   # round8_invent_chance_ledger.out part A


def part1():
    print("1. S4's CES readings at the distillation's scale")
    games = OWN_POSITIONS / DECISIONS
    for label, n in (("invent's sizing, 8 batches", 800_000), ("S4 distillation's own positions", games)):
        se = math.sqrt(R / (n * SHARE * SS_HP))
        print(f"  {label}: {n:,.0f} game-equivalents; plant SE {se:.2e} per hp; kappa/SE {KAPPA / se:.1f}; "
              f"a broken path (0) lies {KAPPA / se:.1f} SE from kappa")
        print(f"    paired SE of a kept ratio {SE_PAIRED_18800 * math.sqrt(18_800 / n):.4f}")
        for d in (1_152 + 8, 1_536):
            print(f"    r_frozen ridge, d {d}: overfit r*d/N {R * d / n:.3f}")
    held = 100_000 * DECISIONS
    hours = held / TEACHER_RATE / 3600
    print(f"  forwarding 100,000 held-out games through N (memory carried): {held / 1e6:.1f} M positions; "
          f"at the teacher pass's measured rate {hours:.1f} h, ${hours * BOX:.2f} (${hours * 0.42:.2f} at $0.42)")


def toy_game(n_games, k, kappa, misjoin, rng):
    c = rng.uniform(0.3, 0.8, size=(n_games, k))
    feature = rng.uniform(-1.0, 1.0, size=(n_games, k))     # a pre-strike context the head reads
    delta = 0.05 + 0.04 * feature                           # true effect of a hit: depends on the context
    d = rng.integers(4, 14, size=(n_games, k)).astype(float)
    h = (rng.random((n_games, k)) < c).astype(float)
    p_win = 0.5 + 0.5 * np.clip(((h - c) * delta).sum(1), -0.99, 0.99)
    z = np.where(rng.random(n_games) < p_win, 1.0, -1.0)
    flag = (rng.random((n_games, k)) < 0.5).astype(float)
    if misjoin:
        perm = np.argsort(rng.random((n_games, k)), axis=1)
        h_read = np.take_along_axis(h, perm, 1)
        c_read = np.take_along_axis(c, perm, 1)
    else:
        h_read, c_read = h, c
    plant = kappa * (flag * (h_read - c_read) * d).sum(1)
    z_prime = z + plant
    omega = (h_read - c_read) / (c_read * (1 - c_read))
    t_prime = omega * z_prime[:, None]
    # the head, linear: T' on [1, F*d, context]; the context's true slope is 0.04, the plant's kappa
    x = np.column_stack([np.ones(n_games * k), (flag * d).ravel(), feature.ravel()])
    coef = np.linalg.lstsq(x, t_prime.ravel(), rcond=None)[0]
    return coef[1:]


def part2():
    print("2. Mis-join shared by the plant and the label (toy, 40,000 games x 30 strikes, kappa 0.005)")
    for misjoin in (False, True):
        coefs = np.array([toy_game(40_000, 30, 0.005, misjoin, RNG) for _ in range(4)])
        print(f"  mis-joined {misjoin!s:5s}: plant coefficient {coefs[:, 0].mean():.4f} +- {coefs[:, 0].std():.4f} "
              f"(planted 0.005); context slope {coefs[:, 1].mean():+.4f} +- {coefs[:, 1].std():.4f} (true 0.04)")


def part3():
    print("3. A post-strike input in the control variate (toy, 2,000,000 games)")
    n = 2_000_000
    p = 0.25
    coin = (RNG.random(n) < p).astype(float)
    c = np.where(coin == 1, 0.7, 0.4)            # the coin's action changes the next strike's chance
    h = (RNG.random(n) < c).astype(float)
    adv = 0.0                                      # the coin's true effect on z is zero
    z = np.where(RNG.random(n) < 0.5 + 0.3 * (h - c) + adv * coin, 1.0, -1.0)
    omega = (coin - p) / (p * (1 - p))
    for name, psi in (("pre-strike psi (damage a hit would deal, 8)", np.full(n, 8.0)),
                      ("post-strike psi (record's damage field: 8 on a hit, 0 on a miss)", 8.0 * h)):
        cv = (h - c) * psi * 0.03
        label = omega * (z - cv)
        print(f"  {name}: E[CV] {cv.mean():+.5f}; coin label mean {label.mean():+.4f} +- "
              f"{label.std() / math.sqrt(n):.4f} (truth 0)")


def main():
    part1()
    part2()
    part3()


if __name__ == "__main__":
    main()
