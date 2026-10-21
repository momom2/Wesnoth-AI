"""Round 10, provenance stance: the derived numbers the proposal's restated stages rest on.

Inputs are measured numbers, each named with the record or script it comes from. Standard library only.
1. The overhead o: the smoke's naive reading and the equal-size reading, from batch 1's walls.
2. A candidate host's cost per coin game: what a 2,000-game timing overstates.
3. Branch S's first step repriced at the measured host.
4. The dump's storage per 10,000-game match (processed and deleted per match).
5. What an L null would require of each named cause.
"""
import math

ELO = 800 / math.log(10)
# measured: walls.txt / matches.txt of near_tie_coins_20261020, fitted in round10_provenance_r_link.out section 4
A_FIX, B_GAME, RESID = 69.0, 0.1550, 24.0          # s per match, s per game, residual sd (s)
WALLS_10K = [1666, 1612, 1618, 1608, 1601]          # b01-b05, 10,000 games each (walls.txt)


def section1():
    m = sum(WALLS_10K) / len(WALLS_10K)
    sd = math.sqrt(sum((w - m) ** 2 for w in WALLS_10K) / (len(WALLS_10K) - 1))
    print("1. the dump overhead o")
    print(f"   consecutive 10,000-game matches on one box: walls {min(WALLS_10K)}-{max(WALLS_10K)} s, sd {sd:.1f} s "
          f"({sd / m:.3f} of the wall)")
    for n in (2_000, 10_000):
        noise = RESID if n == 2_000 else sd
        se = math.sqrt(2) * noise / (n * B_GAME)
        print(f"   equal-size dumped and undumped runs of {n:,} games: o = (W_d - W_u) / (W_u - a), "
              f"SE {se:.3f} (the fixed cost a cancels in W_d - W_u)")
    for o in (0.0, 0.56):
        naive = (A_FIX + 2000 * B_GAME * (1 + o)) / 2000 / ((A_FIX + 1000 * B_GAME) / 1000) - 1
        print(f"   current smoke (2,000 dumped against 1,000 undumped, per-game walls): reads {naive:+.3f} at o {o:.2f}")


def section2():
    per_2k = (A_FIX + 2000 * B_GAME) / 2000
    per_10k = (A_FIX + 10000 * B_GAME) / 10000
    print("\n2. a candidate host's cost per coin game")
    print(f"   per-game wall at 2,000 games {per_2k:.4f} s, at 10,000 games {per_10k:.4f} s: a 2,000-game timing "
          f"overstates the reference's per-game cost by {per_2k / per_10k - 1:.3f}")
    line = 0.934                                   # round9_synthesis_checks.out section 6
    print(f"   round 3's line is {line:.3f} x the measured host; read from 2,000 games, a host at true "
          f"{line:.3f} reads {line * per_2k / per_10k:.3f} and is rejected; a host passes only at a true cost <= "
          f"{line / (per_2k / per_10k):.3f} x the measured, so hosts between that and {line:.3f} are rejected wrongly")


def section3():
    b1_old, rate_old, rate_new = 2.59, 0.42, 0.577   # round8_synthesis_power.out (priced at $0.42/h)
    b1 = b1_old * rate_new / rate_old
    print("\n3. Branch S at the measured host's $0.577 an hour")
    print(f"   B1 (plant run and skill run, 6.17 box-hours) ${b1:.2f} (carried as ${b1_old:.2f}, priced at $0.42/h); "
          f"with B2 $0.98 and B3 $1.13: ${b1 + 0.98 + 1.13:.2f}; exits ${b1:.2f} and ${b1 + 0.98:.2f}")


def section4():
    census = 77.6                                   # band decisions per side-game, readout_b_offline.txt
    for d in (772, 815):
        gb = census * 2 * 10_000 * d * 2 / 1e9
        print(("\n4. dump storage\n" if d == 772 else "") +
              f"   d {d}: {gb:.2f} GB per 10,000-game match in bf16; {10 * gb:.1f} GB per batch if kept whole")


def section5():
    F, F_se, null_line = 21.3, 3.1, 2 * ELO / math.sqrt(37_800)
    need = F - null_line
    own, own_se = -3.5, 4.2                         # round10_provenance_r_link.out section 1
    print("\n5. what an L null (J <= 2 SE = %.2f Elo) would require" % null_line)
    print(f"   a shortfall of {need:.1f} Elo from F; R's own interactions read {own:+.1f} +- {own_se:.1f}: the null "
          f"needs {(need + own) / own_se:.1f} SE more ({(need + own) / math.hypot(own_se, F_se):.1f} SE counting F's own error)")
    t1, t1_se = 12.7, 1.5                           # turn-1 part, round10_provenance_r_link.out section 2
    ex, ex_se = -0.3, 1.7                           # first two switches, exact, round10_provenance_r_pdis.out section 1
    lo = t1 + ex
    se = math.hypot(t1_se, ex_se)
    print(f"   the turn-1 part ({t1:+.1f} +- {t1_se:.1f}, 0.6 prior deviations; 0.959 of its switches are a side's first "
          f"or second, where the exact reading is {ex:+.1f} +- {ex_se:.1f}) sits {(lo - null_line) / se:.1f} SE above the "
          f"null line: losing every later turn's value cannot produce a null; the turn-1 values themselves must fall")


def section6():
    """An adjacent-switch interaction correction for S1's forecast: the label at a student switch split by the
    coin at the side's previous student switch (P 0.25 played), times 0.75 x switches with a predecessor."""
    print("\n6. an adjacent-switch correction for S1's forecast (rejected): its SE")
    sd = math.sqrt(4.01)                            # label sd, E[T^2] round9_simplify_labels.out
    for n in (19, 38):                              # student switches per side-game at q 0.25, 0.5 (77.6 x q)
        pairs = 200_000 * (n - 1)                   # side-games in a 100,000-game batch x switches with a predecessor
        c_se = sd * math.sqrt(1 / (0.25 * pairs) + 1 / (0.75 * pairs))
        per_batch = (n - 1) * 0.75 * c_se * ELO
        print(f"   {n} switches per side-game: correction SE {per_batch:.1f} Elo on one batch, {per_batch / 2:.1f} on four "
              f"(forecast SE 3.41-4.88 on four)")
    miss = 2 * math.sqrt(3.1 ** 2 + 4.2 ** 2 + (ELO / math.sqrt(37_800)) ** 2)
    print(f"\n7. L sealed at F plus the first-coin correction: miss half-width 2 x sqrt(3.1^2 + 4.2^2 + 1.79^2) = {miss:.1f} "
          f"Elo, against 7.2 at F alone")


if __name__ == "__main__":
    section1()
    section2()
    section3()
    section4()
    section5()
    section6()
