"""Round 11, critique of the invent stance: arithmetic checks on its derived numbers.

Standard library only. Inputs are the invent stance's own measured inputs (round11_invent_budget.out):
decisions a game D 453.3, residual share r0 0.6988, training 1,476.9 positions a second, $0.577 an hour,
the match SE 4.3 at 6,400 decisive games, 347.4 Elo per unit mean z.

1. chi2(pi_new || pi_old) against 2 KL(pi_old || pi_new) for one binary decision moved from p to q:
   the forecast's SE assumes chi2 ~ 2 KL; a step that raises a rare action breaks it.
2. The forecast summed over both seats of a self-play game estimates gain_A + gain_B; a one-seat match with
   seats alternated reads their mean. Halved forecast: its SE and the R1 miss line.
3. The price of one pass over the held-out positions (G_h 10,000 games) at the trainer's measured rate,
   an upper bound on one line-search forward (forward-only speed is unmeasured).
4. The ridge's degrees-of-freedom share at 70 columns x 4 turn bins.
5. The bench arm: one critical batch in positions against the arm's 1,000,448 positions, for G* in games.
"""
import math

D = 453.3
R0 = 0.6988
RATE = 1476.9
PRICE_H = 0.577
ELO = 800 / math.log(10)
MATCH_SE = ELO / math.sqrt(6400)


def kl(p, q):
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def chi2(p, q):
    return (q - p) ** 2 / p + (q - p) ** 2 / (1 - p)


def main():
    print("1. chi2(new||old) / 2 KL(old||new), one binary decision p -> q:")
    for p, q in ((0.5, 0.45), (0.1, 0.12), (0.01, 0.03), (0.001, 0.01)):
        print(f"   {p:>6} -> {q:<5}: KL {kl(p, q):.5f}, chi2 {chi2(p, q):.5f}, ratio {chi2(p, q) / (2 * kl(p, q)):.2f}")

    print("\n2. forecast SE and miss line, kappa 0.001, G_h 10,000:")
    se_both = ELO * math.sqrt(R0 * 2 * D * 0.001 / 10000)
    se_half = se_both / 2
    for name, se in (("summed over both seats (as written)", se_both), ("halved (one seat's expected gain)", se_half)):
        print(f"   {name}: SE {se:.2f}, miss line 2 sqrt(SE^2 + {MATCH_SE:.2f}^2) = {2 * math.hypot(se, MATCH_SE):.2f}")
    for k in (2.0, 6.0):
        se = se_both * math.sqrt(k)
        print(f"   chi2 = {k:g} x 2 KL: SE {se:.2f}, miss line {2 * math.hypot(se, MATCH_SE):.2f}")

    print("\n3. one pass over 10,000 held-out games at the trainer's rate:")
    pos = 10000 * D
    secs = pos / RATE
    print(f"   {pos:,.0f} positions / {RATE} = {secs:,.0f} s = ${PRICE_H * secs / 3600:.2f} (forward-backward; an upper bound)")

    print("\n4. ridge degrees of freedom: 70 x 4 = 280 columns:")
    for n in (10000, 20000):
        print(f"   n {n:,}: 70/n {70 / n:.4f}, 280/n {280 / n:.4f}")

    print("\n5. bench arm, 1,000,448 positions, against one critical batch G* (games) x D positions:")
    for g in (2207, 10000, 100000, 1000000):
        print(f"   G* {g:>9,}: {g * D:>13,.0f} positions; arm / G* = {1000448 / (g * D):.4f}; chord SNR ~ sqrt = {math.sqrt(1000448 / (g * D)):.3f}")


if __name__ == "__main__":
    main()
