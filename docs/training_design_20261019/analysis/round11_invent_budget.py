"""Round 11, invent stance: prices and precisions of the walk-corrected policy gradient.

Every input is read from a record; every output is arithmetic on them. Inputs:
  decisions per game   training/metrics/near_tie_coins_20261020/b0*.inference_server_0.json ("requests":
                       one network forward per decision, both sides) over matches.txt's results count;
  training rate        training/metrics/trainer_calibration_20261015/status.txt (positions_per_second,
                       configuration B, measured on the imitation loss);
  generation price     near_tie_coins_20261020/matches.txt: b01-b05, 50,000 games, their walls;
  two-network match    training/metrics/own2_base_20261020/match.walls (own2 against own1);
  host price           $0.577 an hour (the 2026-10-20 box, seed section 7);
  residual share r0    0.6988: the luck walk plus the shrunk setup cell, out of sample on 60,000 coin games
                       (round9_simplify_residual.out, reproduced by round11_invent_luck_strata.out).

Outputs:
  1. decisions per game, per match and pooled;
  2. dollars per game generated, per game trained on every decision, per two-network decisive game;
  3. the SE in Elo of a round's held-out first-order forecast,
       SE = (800 / ln 10) * sqrt(r * chi2_game / G_h),  chi2_game ~ 2 * D * kappa
     (the forecast sums (pi_new/pi_old - 1) * label over a held-out game's D decisions; each term has
     mean-zero weight with variance chi2(pi_new || pi_old) ~ 2 KL per decision for small steps, and the
     game's labels share one residual of variance r per game: round9_simplify_labels.out measured the
     within-game cross terms of coin labels at ratio 0.999, i.e. absent);
     for r = r0 only; any other r is read from M1 before anything uses it;
  4. a round's price as a function of its training games G_r (G_r is unknown until M2 measures the
     noise scale) and the number of rounds $81 buys.
numpy and the standard library only.
"""
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COIN = ROOT / "training/metrics/near_tie_coins_20261020"
CAL = ROOT / "training/metrics/trainer_calibration_20261015/status.txt"
OWN2 = ROOT / "training/metrics/own2_base_20261020/match.walls"
PRICE_H = 0.577
ELO = 800 / math.log(10)
R0 = 0.6988
BUDGET = 81.4          # current_design.md section 4: $82.34 left, less match L (about $0.9, BACKLOG)


def match_lines():
    out = {}
    for line in (COIN / "matches.txt").read_text().splitlines():
        m = re.search(r"(b0\d|a): games (\d+) .* (\d+) s, results and decisive (\d+) (\d+)", line)
        if m:
            out[m.group(1)] = dict(games=int(m.group(2)), secs=int(m.group(3)), results=int(m.group(4)),
                                   decisive=int(m.group(5)))
    return out


def main():
    ml = match_lines()
    print("1. decisions per game (server requests / results):")
    req_tot = res_tot = 0
    for name in [f"b0{k}" for k in range(7)]:
        srv = json.loads((COIN / f"{name}.inference_server_0.json").read_text())
        d = srv["requests"] / ml[name]["results"]
        req_tot += srv["requests"]
        res_tot += ml[name]["results"]
        print(f"   {name}: {srv['requests']:>9,} / {ml[name]['results']:>6,} = {d:.1f}")
    D = req_tot / res_tot
    print(f"   pooled b00-b06: {D:.1f} decisions a game (own1, argmax with coins, 64 slots)")

    cal = re.search(r'"positions_per_second": ([0-9.]+)', CAL.read_text())
    rate = float(cal.group(1))
    gen_games = sum(ml[f"b0{k}"]["games"] for k in range(1, 6))
    gen_secs = sum(ml[f"b0{k}"]["secs"] for k in range(1, 6))
    gen_price = PRICE_H * gen_secs / 3600 / gen_games
    train_price = PRICE_H / 3600 * D / rate
    w = re.search(r"(\d+) s, (\d+) games, (\d+) decisive", OWN2.read_text())
    two_price = PRICE_H * int(w.group(1)) / 3600 / int(w.group(3))
    print("\n2. prices at $%.3f an hour:" % PRICE_H)
    print(f"   generation: {gen_games:,} games in {gen_secs:,} s = {3600 * gen_games / gen_secs:,.0f} an hour, ${gen_price:.3e} a game")
    print(f"   training every decision: {D:.1f} / {rate} positions a second = {D / rate:.3f} s, ${train_price:.3e} a game"
          f" ({train_price / gen_price:.2f} x generation)")
    print(f"   one game generated and trained on once: ${gen_price + train_price:.3e}; a million: ${1e6 * (gen_price + train_price):,.0f}")
    print(f"   two-network decisive game: ${two_price:.3e}")
    print(f"   match SE in Elo with N decisive games: {ELO:.1f} / sqrt(N): N 1,600 -> {ELO / 40:.1f}, 6,400 -> {ELO / 80:.1f}")

    print(f"\n3. SE (Elo) of a round's held-out first-order forecast at r = r0 = {R0} (D = {D:.0f}):")
    kappas = (0.0005, 0.001, 0.002, 0.005)
    print("   G_h \\ kappa per decision " + "".join(f"{k:>9g}" for k in kappas))
    for gh in (5000, 10000, 20000, 50000):
        row = [ELO * math.sqrt(R0 * 2 * D * k / gh) for k in kappas]
        print(f"   {gh:>8,}                  " + "".join(f"{v:9.2f}" for v in row))
    print("   (the forecast's own size scales as sqrt(kappa) too, so its relative precision depends only on G_h,"
          " r and the unknown gain per unit KL)")

    print(f"\n4. a round: G_r training games (generated, trained once) + 10,000 held-out games (generated)"
          f" + a 6,400-game two-network match; rounds within ${BUDGET}:")
    for gr in (10_000, 30_000, 100_000, 300_000, 1_000_000):
        cost = gr * (gen_price + train_price) + 10_000 * gen_price + 6400 * two_price
        print(f"   G_r {gr:>9,}: ${cost:7.2f} a round, {int(BUDGET // cost):>3d} rounds")


if __name__ == "__main__":
    main()
