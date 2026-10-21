"""Round 9, provenance stance: prices measured on the 2026-10-20 boxes, and the
plan's budget at those prices. Standard library only.

Measured inputs (files read below):
- training/metrics/near_tie_coins_20261020/matches.txt: each coin match's games and wall
  (one network, both sides coined, run_elo_batch at 120 jobs); stage A (attack coin only).
  Host $0.577 an hour (docs/near_tie_coins_prereg_20261020.md, amendment).
- training/metrics/own2_base_20261020/match.walls: own2 against own1, two networks, 20 jobs,
  6,402 games in 3,533 s; host $0.571 an hour (docs/own2_base_prereg_20261020.md, Measured).
- round7_provenance_price_tail.out: the pool's steady 21,966 games an hour (no coins) at $0.42.
- the distillation recipe: about $2.1 at about $0.42-0.46 an hour (docs/memory_own_trajectories_prereg_20261011.md),
  i.e. about 4.8 box-hours.
- spent in the 2026-10-20 window: about $2.0 (coins) + $0.66 (own2): seed section 7.
Choices [C]: match sizes, batch sizes and the stage list of round9_provenance.md.
"""
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COINS = ROOT / "training" / "metrics" / "near_tie_coins_20261020" / "matches.txt"
OWN2 = ROOT / "training" / "metrics" / "own2_base_20261020" / "match.walls"
ELO = 800 / math.log(10)
BUDGET, SPENT = 85.0, 2.0 + 0.66
DISTILL_HOURS = 2.1 / 0.44
SETUP_HOURS = 0.1          # bring-up, wheel and tests: 5.6 min on the coin box (walls.txt 21:42:39 -> 21:48:12)
BATCH, BATCHES, SMOKE = 100_000, 8, 2_000
LINK, MATCH, TWO_NET = 12_800, 37_800, 12_800
FORECAST_SE_R8 = 4.6       # round6_synthesis_power.out section 4, at sigma^2 3.67 and census 76.1
SIGMA2_R8, SIGMA2_CELL = 3.67, 4.05   # 4.05: round9_provenance_label_noise.out (0.760 x 5.333, decision-weighted)
CENSUS_R8, CENSUS = 76.1, 77.6


def coin_rates():
    rows = []
    for line in COINS.read_text().splitlines():
        m = re.search(r" (\w+): games (\d+) .* (\d+) s, results and decisive (\d+) (\d+)", line)
        if m:
            rows.append((m.group(1), int(m.group(4)), int(m.group(3))))
    return rows


def main():
    rows = coin_rates()
    print("1. one-network games an hour, host 56955777 ($0.577/h)")
    for name, games, wall in rows:
        print(f"  {name}: {games} games in {wall} s = {games / wall * 3600:,.0f} games/h")
    big = [(g, w) for n, g, w in rows if n.startswith("b") and g >= 8000]
    # wall = start + per_game x games, fitted on the full batch matches and the smoke
    smoke = [(g, w) for n, g, w in rows if n == "b00"][0]
    pts = big + [smoke]
    mx = sum(g for g, _ in pts) / len(pts)
    my = sum(w for _, w in pts) / len(pts)
    slope = sum((g - mx) * (w - my) for g, w in pts) / sum((g - mx) ** 2 for g, _ in pts)
    start = my - slope * mx
    steady = 3600 / slope
    a = [(g, w) for n, g, w in rows if n == "a"][0]
    a_steady = 3600 / ((a[1] - start) / a[0])
    print(f"  band-coin matches: start-up {start:.0f} s, steady {steady:,.0f} games/h; "
          f"stage A (attack coin only) at the same start-up {a_steady:,.0f} games/h: "
          f"band-coin logging costs {a_steady / steady - 1:+.3f}")
    full = sorted(g / w * 3600 for n, g, w in rows if n.startswith("b") and g >= 10_000)
    effective = full[len(full) // 2] if len(full) % 2 else (full[len(full) // 2 - 1] + full[len(full) // 2]) / 2
    print(f"  effective rate of the 10,000-game matches (median, start-up included): {effective:,.0f} games/h; "
          f"costs below use it")
    print(f"  against the pool's 21,966 games/h without coins (another host): {21966 / effective - 1:+.3f}")
    steady = effective
    own2 = OWN2.read_text()
    m = re.search(r"(\d+) s, (\d+) games", own2)
    two_rate = int(m.group(2)) / int(m.group(1)) * 3600
    print(f"2. two networks, 20 jobs (host 56955126): {two_rate:,.0f} games/h")
    left = BUDGET - SPENT
    print(f"3. budget: ${BUDGET:.0f} - ${SPENT:.2f} spent = ${left:.2f}")
    for rate in (0.42, 0.577):
        p1, p2 = rate / steady, rate / two_rate
        setup = SETUP_HOURS * rate
        one = lambda n: n * p1  # noqa: E731
        two = lambda n: n * p2  # noqa: E731
        round_cost = one(SMOKE + BATCHES * BATCH)
        stages = {
            "L+T: own1+R vs own1 and own2+R vs own2, 12,800 games each, one box": setup + one(2 * LINK),
            "claim own2+R vs own1 (two networks), if T adopts": two(TWO_NET),
            "S1: smoke + 8 batches on the base": setup + round_cost,
            "S2: adoption (one network) + claim vs own1 (two networks)": one(MATCH) + two(TWO_NET),
            "E reserve (S2, S3 one network; S5 two networks)": 2 * one(MATCH) + two(TWO_NET),
            "S3: 8 batches + adoption + claim": setup + round_cost + one(MATCH) + two(TWO_NET),
            "S4: distillation + N vs pi_k (two networks)": DISTILL_HOURS * rate + two(TWO_NET),
            "S5: 8 batches + adoption vs pi_k + claim (two networks each)": setup + round_cost + 2 * two(TWO_NET),
        }
        total = sum(stages.values())
        print(f"  at ${rate:.3f}/h: one network ${p1:.3e} a game, two networks ${p2:.3e}; a batch ${one(BATCH):.2f}, "
              f"{BATCH / steady:.2f} h")
        for k, v in stages.items():
            print(f"    {k}: ${v:.2f}")
        print(f"    total ${total:.2f}, left ${left - total:+.2f}")
        no_s5 = total - stages["S5: 8 batches + adoption vs pi_k + claim (two networks each)"]
        print(f"    without S5 ${no_s5:.2f}, left ${left - no_s5:+.2f}")
        null1 = stages["L+T: own1+R vs own1 and own2+R vs own2, 12,800 games each, one box"] + \
            stages["S1: smoke + 8 batches on the base"] + one(MATCH)
        print(f"    a null round 1 after L and T: ${null1:.2f}")
    # break-even hourly rate for the whole path
    lo, hi = 0.1, 2.0
    for _ in range(60):
        mid = (lo + hi) / 2
        p1, p2 = mid / steady, mid / two_rate
        rc = (SMOKE + BATCHES * BATCH) * p1
        tot = (5 * SETUP_HOURS * mid + 2 * LINK * p1 + TWO_NET * p2 + rc + MATCH * p1 + TWO_NET * p2
               + 2 * MATCH * p1 + TWO_NET * p2 + rc + MATCH * p1 + TWO_NET * p2 + DISTILL_HOURS * mid
               + TWO_NET * p2 + rc + 2 * TWO_NET * p2)
        lo, hi = (mid, hi) if tot < left else (lo, mid)
    print(f"  the whole path fits iff the hourly rate is <= ${lo:.3f} (games-per-hour as measured)")
    print("4. derived SEs")
    print(f"  match SE: {ELO / math.sqrt(LINK):.2f} Elo at {LINK:,} games, {ELO / math.sqrt(MATCH):.2f} at {MATCH:,}")
    f_se = FORECAST_SE_R8 * math.sqrt(SIGMA2_CELL / SIGMA2_R8) * math.sqrt(CENSUS / CENSUS_R8)
    print(f"  S1 forecast SE at the measured sigma^2 {SIGMA2_CELL} and census {CENSUS}: {f_se:.2f} Elo "
          f"(round 8: {FORECAST_SE_R8}); miss line {2 * math.sqrt(f_se ** 2 + (ELO / math.sqrt(MATCH)) ** 2):.1f}")
    for f_se_l in (3.2,):
        for n in (6_400, 12_800, 37_800):
            j = ELO / math.sqrt(n)
            print(f"  L: forecast SE {f_se_l}, match {n:,} games (SE {j:.2f}): miss half-width "
                  f"{2 * math.sqrt(f_se_l ** 2 + j ** 2):.1f} Elo; SE of J/F at F 22.9 "
                  f"{math.sqrt(j ** 2 + f_se_l ** 2) / 22.9:.3f}")


if __name__ == "__main__":
    main()
