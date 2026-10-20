"""Round 8 synthesis: the plan's budget and the readings this synthesis adds.

Measured inputs (sources):
- pool price, steady state $1.912e-5 a one-network game (round7_provenance_price_tail.out); host spread 1.41x;
  eval driver $1.17e-4 a two-network decisive game (upper bound for the pool's two-network mode); box setup
  $0.018 (round6_provenance_prices.out); distillation recipe $2.1 (docs/memory_own_trajectories_prereg_20261011.md,
  an upper bound); sequence trainer 217 positions a second at $0.42 an hour (same prereg).
- dice control: the slope of match-centred z on side A's hp luck, match by match (round8_invent_chance_ledger.out
  part E, 23 player pairs, about 800 decisive games each) and its SE at 100,000 games (part B: 0.00004).
- skill settings: s_top +200 (round8_synthesis_skill_coverage.out, the coverage rule); s_low +40 (the rated sides'
  25th percentile in the same output, inside the coverage rule at +-52 Elo: 56 players, largest share 0.116).
Choices [C], their consequences derived here: 8 batches of 100,000 coin games a round; no interim match;
37,800-game one-network matches (SE 1.79); 12,800-game two-network matches (SE 3.07); S5's adoption against the
pre-reset incumbent pi_k (two networks); one E reserve per adoption reading; Branch S's plant 0.5 M and skill run
4 M positions (the challenge's sizes); S0(b), a 12,800-game two-network confirmation of own2 against own1
(R1 read +16.5 +- 12.3 Elo, training/metrics/memory_own_round2_20261011, docs/memory_own_round2_prereg_20261011.md).
Standard library only.
"""
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
POOL, SPREAD, EVAL, SETUP, DISTILL, BUDGET = 1.912e-5, 1.41, 1.17e-4, 0.018, 2.1, 85.0
HOUR, TRAIN_RATE = 0.42, 217.0
BATCH, BATCHES = 100_000, 8
MATCH, TWO_NET, ATTACK, SMOKE = 37_800, 12_800, 6_400, 2_000
ELO = 800 / math.log(10)
SE_MATCH, SE_TWO = ELO / math.sqrt(MATCH), ELO / math.sqrt(TWO_NET)
FORECAST_SE = 4.6                      # round6_synthesis_power.out section 4, at mu 0
S_TOP, S_LOW = 200.0, 40.0


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def coin_stages(pool):
    m, coin, two = MATCH * pool, BATCHES * BATCH * pool, TWO_NET * EVAL
    return {
        "S0 setup + attack known answer (smoke counted in batch 1)": SETUP + ATTACK * pool,
        "S0(b) base confirmation, own2 against own1, two networks": two,
        "S1 round 1 (8 batches)": coin,
        "S2 adoption = claim": m,
        "E reserve (S2, S3 one-network; S5 two-network)": 2 * m + two,
        "S3 round 2 (8 batches, adoption vs pi_1, claim vs own1)": coin + 2 * m,
        "S4 reset (distillation, N against pi_k two-network)": DISTILL + two,
        "S5 round on N (8 batches, adoption vs pi_k two-network, claim vs own1 two-network)": coin + 2 * two,
    }, m, coin


def branch_s():
    plant = 0.5e6 / TRAIN_RATE / 3600 * HOUR
    run = 4.0e6 / TRAIN_RATE / 3600 * HOUR + 25 / 60 * HOUR
    two = TWO_NET * EVAL
    return {
        "B1 plant run + skill run": plant + run,
        "B2 M1, N(s_top) against N(s_low), one network": MATCH * POOL,
        "B3 M2, N(s_top) against own1, two networks (the claim)": two,
        "B4 (conditional) 0-slot diagnostic": two,
        "B4 (conditional) own-trajectory remedy and its match": DISTILL + two,
    }


def section1():
    print("1. budget")
    for pool, label in ((POOL, "measured host"), (POOL * SPREAD, "host 1.41x slower")):
        st, m, coin = coin_stages(pool)
        total = sum(st.values())
        print(f"  {label}: ${pool:.3e} a game; batch ${BATCH * pool:.2f}; match ${m:.2f}; two-network match ${TWO_NET * EVAL:.2f}")
        for k, v in st.items():
            print(f"    {k}: ${v:.2f}")
        fixed3 = total - 3 * coin
        line3 = (BUDGET - fixed3) / (3 * BATCHES * BATCH)
        s5 = [v for k, v in st.items() if k.startswith("S5")][0]
        line2 = (BUDGET - (total - s5 - 2 * coin)) / (2 * BATCHES * BATCH)
        s0 = st["S0 setup + attack known answer (smoke counted in batch 1)"] + st["S0(b) base confirmation, own2 against own1, two networks"]
        print(f"    coin path total ${total:.2f}, left ${BUDGET - total:.2f}; three rounds fit iff price x (1 + overhead) <= "
              f"${line3:.3e} (room {line3 / pool - 1:+.3f}); two rounds iff <= ${line2:.3e}")
        print(f"    a null round 1 (S0, 8 batches, adoption match): ${s0 + coin + m:.2f}")
        b = branch_s()
        core = sum(v for k, v in b.items() if not k.startswith("B4"))
        print(f"    Branch S to its claim ${core:.2f}; with every conditional step ${sum(b.values()):.2f}; "
              f"coin path plus Branch S core ${total + core:.2f}, left ${BUDGET - total - core:.2f}; "
              f"plus the base-choosing match ${TWO_NET * EVAL:.2f}: left ${BUDGET - total - core - TWO_NET * EVAL:.2f}")
    st, m, coin = coin_stages(POOL)
    total = sum(st.values())
    extra = 2 * TWO_NET * EVAL
    print(f"  if S0(b) makes own2 the base: the claims of S2 and S3 against own1 become separate two-network matches "
          f"(+${extra:.2f}): coin path ${total + extra:.2f}, left ${BUDGET - total - extra:.2f}; "
          f"with Branch S's core ${total + extra + sum(v for k, v in branch_s().items() if not k.startswith('B4')):.2f}")
    for k, v in branch_s().items():
        print(f"  {k}: ${v:.2f}")
    b = branch_s()
    print(f"  Branch S exits: route closed at an empty permutation gap ${b['B1 plant run + skill run']:.2f}; "
          f"at a null M1 ${b['B1 plant run + skill run'] + b['B2 M1, N(s_top) against N(s_low), one network']:.2f}")


def read_part_e():
    text = (HERE / "round8_invent_chance_ledger.out").read_text().split("Part E")[1]
    rows = re.findall(r"^\s+(\S+)\s+([+-]\d\.\d+) \+- (\d\.\d+)$", text, re.M)
    return [(n, float(b), float(s)) for n, b, s in rows]


def q_stat(rows, tau2):
    w = [1 / (s * s + tau2) for _, _, s in rows]
    mean = sum(wi * b for wi, (_, b, _) in zip(w, rows)) / sum(w)
    return sum(wi * (b - mean) ** 2 for wi, (_, b, _) in zip(w, rows)), mean, 1 / math.sqrt(sum(w))


def section2():
    rows = read_part_e()
    q0, mean, se_mean = q_stat(rows, 0.0)
    df = len(rows) - 1
    # 2.5% quantile of chi-square on 22 df (Wilson-Hilferty), for the Q-profile upper bound on tau^2
    z = -1.959964
    chi_lo = df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3
    lo, hi = 0.0, 1e-5
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if q_stat(rows, mid)[0] > chi_lo:
            lo = mid
        else:
            hi = mid
    tau_up = math.sqrt(hi)
    se_batch, se_smoke = 0.00004, 0.00032 * math.sqrt(1600 / SMOKE)
    print("\n2. dice control (slope of the label path's z on side A's hp luck)")
    print(f"  {len(rows)} player pairs: pooled {mean:+.5f} +- {se_mean:.5f}; Q {q0:.1f} on {df} df; "
          f"95% upper bound on the between-pair sd tau {tau_up:.5f} (Q-profile, chi2 2.5% point {chi_lo:.2f})")
    for label, se in (("smoke (2,000 games)", se_smoke), ("a batch (100,000 games)", se_batch)):
        half = 3 * math.sqrt(se_mean ** 2 + tau_up ** 2 + se ** 2)
        lo_w, hi_w = mean - half, mean + half
        f_caught = 1 - lo_w / mean
        print(f"  {label}: window at 3 sd [{lo_w:+.5f}, {hi_w:+.5f}]; a flipped result reads {-mean:+.5f}; "
              f"a share f of games joined to the wrong result reads (1 - f) x {mean:.5f}, outside for f >= {f_caught:.2f}")
    for k in (2, 8):
        se_d = se_batch * math.sqrt(1 + 1 / (k - 1))
        print(f"  batch {k} against the mean of batches 1-{k - 1}: 3-SE line {3 * se_d:.5f}; "
              f"catches a change in the mis-joined share of {3 * se_d / mean:.3f} or more")


def section3():
    print("\n3. reset and round 3 (S4, S5)")
    pred_se = math.sqrt(SE_TWO ** 2 + FORECAST_SE ** 2)
    miss = 2 * math.sqrt(SE_TWO ** 2 + pred_se ** 2)
    print(f"  S5's adoption match predicts J_S4 + G_pf(S5): prediction SE {pred_se:.2f} Elo; the match's SE {SE_TWO:.2f}; "
          f"miss line {miss:.1f} Elo at mu 0")
    print(f"  false adoption of a round-3 player no better than pi_k: {phi(-2):.3f} (lower 2-SE bound rule); "
          f"a lossy reset cannot pass into the adopted player")
    for deficit in (2.8, 6.1):
        print(f"  under the round-7 rule (N kept iff J_S4 >= 0) a {deficit}-Elo reset deficit was kept with P "
              f"{phi(-deficit / SE_TWO):.2f}")


def section4():
    print("\n4. Branch S readings")
    gap = S_TOP - S_LOW
    print(f"  M1: settings +{S_TOP:.0f} and +{S_LOW:.0f}, gap {gap:.0f} Elo; human expected score at that gap "
          f"{1 / (1 + 10 ** (-gap / 400)):.3f}; SE {SE_MATCH:.2f} Elo; 2-SE detection of a transfer fraction "
          f"f = M1 / {gap:.0f} >= {2 * SE_MATCH / gap:.3f}")
    print(f"  M2: one pre-registered arm, SE {SE_TWO:.2f}, false adoption {phi(-2):.3f} (no selection over arms)")


def section5():
    print("\n5. per-decision storage for E, I-log and I-match")
    side_games_band = 76.1 * 2
    gb = 1_000 * side_games_band * 772 * 2 / 1e9
    print(f"  x on a fixed 1,000-game subsample: {side_games_band:.0f} band decisions a game x 772 features x 2 bytes "
          f"(bf16) = {gb:.2f} GB a batch, {gb * BATCHES:.1f} GB a round")
    print(f"  f as one float32 per band decision: {100_000 * side_games_band * 4 / 1e6:.0f} MB a batch")


def main():
    section1()
    section2()
    section3()
    section4()
    section5()


if __name__ == "__main__":
    main()
