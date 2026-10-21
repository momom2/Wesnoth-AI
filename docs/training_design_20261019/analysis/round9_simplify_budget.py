"""Round 9, simplify stance: the plan's costs, lines and derived readings from section 7's measurements.

Inputs, all measured (the scripts that read them are named):
  coin games all-in (match, walk, rows, plant, readout) 20,254 an hour; pool games 22,209 an hour;
    two-network games 6,521 an hour; box $0.577 an hour       round9_simplify_labels.out [prices]
  label second moment E[T^2] 4.01 with the frozen baseline; per-side-game welfare variance 158.1 at q 0.5
                                                               round9_simplify_labels.out
  band decisions 77.6 per side-game; pooled mean label +0.0003 +- 0.0007   readout_b_offline.txt
  own2 against own1 +22.7 +- 4.4 Elo                          own2_base_20261020/reading.json
  box setup and tests before the first match: 21:41:10 to 21:48:12 on the coin box (7 min)
                                                               near_tie_coins_20261020/walls.txt
  spent in section 7's window: the coin box 21:42 to 01:04 UTC, the own2 box 21:41 to 22:40 plus setup
Constants of the design [C]: 8 batches of 100,000 games a round, a 2,000-game smoke, 37,800-game adoption
matches, a 12,800-game claim, 772 network features in bf16, a 1,000-game stored subsample.
numpy is not needed; the standard library only.
"""
import math

ELO = 800 / math.log(10)
PRICE_MEASURED, PRICE_CHEAP = 0.577, 0.42
ALLIN_RATE, POOL_RATE, TWO_RATE = 20_254, 22_209, 6_521
SETUP_H = 7 / 60 + 0.25          # bring-up and tests (7 min measured) plus a quarter hour of staging, a choice
BUDGET = 85.0


def spent():
    """Section 7's two boxes as the seed states them (00_seed.md section 7): the coin box about $2.0,
    the own2 box about $0.66 (the coin box ran 21:42 to 01:04 UTC, 3.37 h at $0.577: $1.94)."""
    return 2.0, 0.66


def round_cost(price_h, overhead):
    batches = 8 * 100_000 / ALLIN_RATE * price_h * (1 + overhead)
    adoption = 37_800 / POOL_RATE * price_h
    return batches + adoption + SETUP_H * price_h, batches, adoption


def main():
    c1, c2 = spent()
    left = BUDGET - c1 - c2
    print(f"section 7's window: coin box ${c1:.2f}, own2 box ${c2:.2f}; left of $85: ${left:.2f}")
    claim = 12_800 / TWO_RATE * PRICE_MEASURED
    smoke = (2_000 / ALLIN_RATE + 1_000 / POOL_RATE + SETUP_H) * PRICE_MEASURED
    print(f"smoke (2,000 dumped games, 1,000 undumped on the same box for the overhead, setup): ${smoke:.2f}")
    print(f"final claim, 12,800 two-network games: ${claim:.2f} ({12_800 / TWO_RATE:.2f} h)")
    for price in (PRICE_MEASURED, PRICE_CHEAP):
        r0, b, a = round_cost(price, 0.0)
        # three rounds and the claim fit iff 3 * (b (1 + o) + a + setup) + claim + smoke <= left
        o_line = (left - claim - smoke - 3 * (a + SETUP_H * price)) / (3 * b) - 1
        n_rounds = math.floor((left - claim - smoke) / r0)
        print(f"at ${price} an hour: a round ${r0:.2f} at overhead 0 (batches ${b:.2f}, adoption ${a:.2f}); "
              f"{n_rounds} rounds fit at overhead 0; three rounds fit iff the dump's overhead <= {o_line:.3f}")
    calibration = 37_800 / POOL_RATE * PRICE_MEASURED
    first_box = smoke + calibration
    print(f"C1, the transfer calibration on own1 (37,800 pool games): ${calibration:.2f}; on the smoke's box, "
          f"the first box ${first_box:.2f} ({(2_000 / ALLIN_RATE + 1_000 / POOL_RATE + 37_800 / POOL_RATE + SETUP_H):.2f} h)")
    r0, b, a = round_cost(PRICE_MEASURED, 0.0)
    fixed = first_box + claim
    for k in (2, 3):
        o_line = (left - fixed - k * (a + SETUP_H * PRICE_MEASURED)) / (k * b) - 1
        h_line = left / (fixed + k * r0)
        print(f"with C1: {k} rounds and the claim cost ${fixed + k * r0:.2f} at overhead 0 on the measured host "
              f"(margin ${left - fixed - k * r0:.2f}); they fit iff the dump's overhead <= {o_line:.3f}, or, at overhead 0, "
              f"iff a host's all-in cost per game is <= {h_line:.3f} x the measured host's (${2.849e-5 * h_line:.3e} a coin game)")
    hours = 8 * 100_000 / ALLIN_RATE
    print(f"a round's batches take {hours:.1f} h of wall at overhead 0")
    dump = 100_000 * 2 * 77.6 * 772 * 2 / 1e9
    print(f"dump: {dump:.1f} GB per 100,000-game batch, {dump / 10:.2f} GB per 10,000-game match; "
          f"stored subsample {1_000 * 2 * 77.6 * 772 * 2 / 1e9:.2f} GB per batch")
    dev = 77.6 * 0.25
    cost, se = dev * 0.0003 * ELO, dev * 0.0007 * ELO
    print(f"coin play's first-order effect: {dev:.1f} deviations per side-game x mean label: {cost:+.1f} +- {se:.1f} Elo")
    coin_info = 2 * 77.6 / 4.01
    fork_info = 1 / (2 * 0.608)
    print(f"information on single-decision advantages per game-equivalent: coins {coin_info:.1f}, fork pair "
          f"{fork_info:.2f} (round 5's 0.608), ratio {coin_info / fork_info:.0f}")
    se_pf = math.sqrt(158.1 / (4 * 100_000 * 2)) * ELO
    miss = 2 * math.sqrt(se_pf ** 2 + (ELO / math.sqrt(37_800)) ** 2)
    print(f"forecast SE (4 batches, q 0.5): {se_pf:.2f} Elo; adoption miss line 2 x sqrt(SE_pf^2 + 1.79^2) = {miss:.1f} Elo")
    v_se = 1.27e-6 * 4.01 / 3.67
    print(f"spread estimator after 8 batches: SE {v_se:.2e}, null bound {2 * v_se:.2e} (sd {math.sqrt(2 * v_se):.5f}); "
          f"a pilot at a fraction f of the round has SE / f")
    for k in (1, 2, 3):
        pred_se = math.sqrt(4.35 ** 2 + k * 1.79 ** 2)
        line = 2 * math.sqrt(pred_se ** 2 + 3.07 ** 2)
        print(f"final claim after {k} adopted round(s): prediction 22.7 + sum of adoption readings, SE {pred_se:.2f}; "
              f"with the claim's 3.07, miss line {line:.1f} Elo")


if __name__ == "__main__":
    main()
