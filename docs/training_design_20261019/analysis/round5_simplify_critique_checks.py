"""Round 5, critique of the simplify stance: four derivations from its own inputs.

1. P-coin's noise if its plants are booked into a separate result stream
   (round5_simplify.md, S1 batch 1): the stream holds only the planted terms,
   delta1 * (D - p) * (even contrast) and delta2 * (D - p) * (u . x), at
   2 x 62 band decisions a game, contrast variance 0.5 and Var(u . x) = 1 as
   the proposal assumes, 12.4 million labels in batch 1.
2. The plug forecast's missing intercept: SE of a mean label over the
   forecast half's 49.6 million labels (sigma_eps^2 = 3.91), times 31
   switches per side-game, in Elo; combined with the stated 2.4 (covariance
   ignored).
3. The exploitation check's resolution: PURE 800-game SE 12.3 for pi_1
   against parity3, 12.7 for own1's measured +88.6 over parity3, 2.65 for
   pi_1 against own1.
4. Critic re-entry's refit without stored features: replaying 400,000
   training games through own1 at every decision (memory), 465 decisions a
   game and 2,440 decisions a second (BACKLOG.md, generation round 5), at
   $0.42 an hour.
Standard library only.
"""
import math

P, N_BAND, LABELS_BATCH1 = 0.25, 62, 12.4e6


def main():
    var_stream = (0.01 ** 2 * 2 * N_BAND * P * (1 - P) * 0.5
                  + 0.002 ** 2 * 2 * N_BAND * P * (1 - P) * 1.0)
    var_t = var_stream / (P * (1 - P))
    se_a = math.sqrt(var_t / (LABELS_BATCH1 * 0.5))
    se_b = math.sqrt(var_t / LABELS_BATCH1)
    print(f"stream var per game {var_stream:.3e}; SE a {se_a:.2e} (stated 7.9e-4, ratio {7.9e-4 / se_a:.0f}); "
          f"SE b {se_b:.2e} (stated 5.6e-4, ratio {5.6e-4 / se_b:.0f})")
    se_alpha = math.sqrt(3.91 / 4.96e7)
    elo = se_alpha * 31 * 800 / math.log(10)
    print(f"SE alpha {se_alpha:.2e}; x31 switches in Elo {elo:.2f}; combined with 2.4: {math.hypot(2.4, elo):.2f}")
    print(f"exploit check combined SE {math.sqrt(12.3 ** 2 + 12.7 ** 2 + 2.65 ** 2):.1f}")
    hours = 4e5 * 465 / 2440 / 3600
    print(f"replay box-hours {hours:.1f}, dollars {hours * 0.42:.1f}")


if __name__ == "__main__":
    main()
