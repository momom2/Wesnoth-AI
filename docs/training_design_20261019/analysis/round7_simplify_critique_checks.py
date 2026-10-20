"""Round 7, critique of the simplify stance: four checks on what its removals leave.

1. The intercept batch 1 loses with b = 0: side 1's mean result in own1 self-play
   (training/metrics/match_luck_20261019/rows.jsonl; z is label A's result, side_a maps it to a
   Wesnoth side).
2. I-V, the check that replaces the exact duel I2 for variate centring: its SE per pair type,
   from the stance's own inputs (band 76.1 per side-game, 100,000 games a batch, p 0.25,
   columns explain 1 - 0.839 of Var z in batch 1 and b + columns 1 - 0.688 later), and how often
   a 2-SE rule fires on correct code when applied to every type of every batch.
3. The plant read on T' - T against an error in the game-to-result mapping (the result field
   oriented to the wrong side): a simulation in which the record's z is flipped relative to the
   real outcome, the plant built from the record as the stance specifies. Also the repair that
   reads S0's attack known answer through the label code (one coin a game: which side carries
   the hook, P = 1/2).
4. The residual decode offset that undoes a measured closure c < 1 of a baked -1.5 shift.
numpy and the standard library only.
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ROWS = ROOT / "training/metrics/match_luck_20261019/rows.jsonl"
RNG = np.random.default_rng(20261020)
N_BAND, BATCH, P = 76.1, 100_000, 0.25
W2 = 1.0 / (P * (1 - P))


def phi_upper(x):
    return 0.5 * math.erfc(x / math.sqrt(2))


def section1():
    z1 = []
    for line in open(ROWS):
        r = json.loads(line)
        if r.get("decisive") and r["match"] in ("gen_own1_selfplay", "own64_selfplay"):
            z1.append(r["z"] if r["side_a"] == 1 else -r["z"])
    z1 = np.array(z1, float)
    m, se = z1.mean(), z1.std() / math.sqrt(len(z1))
    print(f"1. own1 self-play, {len(z1)} decisive games: side 1's mean z {m:+.4f} +- {se:.4f}; "
          f"E[z]^2 adds {m * m:.4f} to batch 1's E[R^2] (against 0.839)")


def section2():
    print("\n2. I-V per pair type: SE of mean s*w*(b + beta'S) = sqrt(w2 * E[(b+beta'S)^2] / n_type)")
    labels = 2 * N_BAND * BATCH
    for name, explained in (("batch 1 (b = 0)", 1 - 0.839), ("batch >= 2", 1 - 0.688)):
        cells = []
        for f in (0.02, 0.1, 0.3):
            se = math.sqrt(W2 * explained / (labels * f))
            cells.append(f"share {f}: SE {se:.5f}, 2 SE {2 * se:.4f}")
        print(f"  {name}: " + "; ".join(cells))
    p2 = 2 * phi_upper(2.0)
    for types in (10, 16):
        for batches in (8, 24):
            k = types * batches
            fw = 1 - (1 - p2) ** k
            z_bonf = 0.0
            lo, hi = 2.0, 8.0
            for _ in range(60):
                mid = (lo + hi) / 2
                if 2 * phi_upper(mid) * k > 0.05:
                    lo = mid
                else:
                    hi = mid
            z_bonf = hi
            print(f"  {types} types x {batches} batches = {k} tests: P(some 2-SE alarm | correct code) "
                  f"{fw:.3f}; 5% family-wise needs {z_bonf:.2f} SE")
    print("  spread floors it must resolve: sd 0.0011-0.0036 z per band decision (round7_simplify_power.out section 2)")


def simulate(n_games, n_side, flip_record, rng):
    """Labels and plant under a result field that is (flip) or is not the real side-1 result."""
    s = np.concatenate([np.ones(n_side), -np.ones(n_side)])
    d = (rng.random((n_games, 2 * n_side)) < P).astype(float)
    attack = rng.random((n_games, 2 * n_side)) < 0.1          # a pair type with a known negative effect
    effect = np.where(attack, -0.08, 0.0)
    latent = (s * d * effect).sum(1) + rng.standard_normal(n_games)
    z_real = np.sign(latent)
    z_field = -z_real if flip_record else z_real               # the bug sits upstream of the plant
    g = rng.standard_normal((n_games, 2 * n_side))
    inc = (s * (d - P) * g).sum(1)                              # plant from the record's s and D
    w = (d - P) / (P * (1 - P))
    t = s * w * z_field[:, None]
    t_planted = s * w * (z_field + inc)[:, None]
    diff = t_planted - t
    slope = float((g * diff).sum() / (g * g).sum())
    real_mean = float(t[attack].mean())
    real_se = float(t[attack].std() / math.sqrt(attack.sum()))
    return slope, real_mean, real_se


def section3():
    print("\n3. plant on T' - T against a result field oriented to the wrong side")
    for flip in (False, True):
        slope, m, se = simulate(20_000, 76, flip, RNG)
        print(f"  record flipped {flip}: plant slope {slope:.3f}; real labels of the planted-effect type "
              f"{m:+.4f} +- {se:.4f} (true -0.08 on the latent scale)")
    se_t = 2.0 / math.sqrt(6400)
    pred, pred_se = -0.2247, 0.0459
    half = 2 * math.sqrt(pred_se ** 2 + se_t ** 2)
    print(f"  repair, S0's 6,400 games read through the label code (P 1/2, w = +-2): SE {se_t:.4f} z; "
          f"window {pred:+.3f} +- {half:.3f} z = [{pred - half:+.3f}, {pred + half:+.3f}]; "
          f"a flipped result reads {-pred:+.3f}")


def section4():
    print("\n4. residual offset that restores an effective -1.5 at a measured closure c: -1.5 (1 - c)")
    print("  " + "; ".join(f"c {c:.2f}: {-1.5 * (1 - c):+.2f}" for c in (0.60, 0.82, 1.00)))


if __name__ == "__main__":
    section1()
    section2()
    section3()
    section4()
