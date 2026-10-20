"""Round 8, challenge stance: prices, standard errors, derived predictions and the budget of the skill-axis regimen.

Measured inputs (named where used):
  pool price $1.912e-5 a one-network game          round7_provenance_price_tail.out
  two-network price $1.17e-4 a decisive game        round6_provenance_prices.out (eval driver median, upper bound)
  worker cycle 39.3 ms, of which 33.2 ms on server  round7_provenance_price_tail.out
  own-trajectory pass 217 positions/s; run $2.1     docs/memory_own_trajectories_prereg_20261011.md "Measured"
  box $0.42 an hour                                 seed section 1
  human-scale gap s_top - s_mid = 253 - 84 Elo      round8_challenge_skill.out sections 1 and 2
No numbers are guessed: every choice is marked [C] and its consequence is derived here.
"""
import math

ELO_PER_Z = 800.0 / math.log(10.0)          # 347.4: Elo per unit of mean z near even
POOL = 1.912e-5
TWO_NET = 1.17e-4
CYCLE, SERVER = 39.3, 33.2
RATE = 217.0                                  # positions a second, sequence trainer with memory
BOX = 0.42
S_TOP, S_MID = 253.0, 84.0                    # 95th percentile of human regulars; action-weighted mean winner rating


def se_elo(n):
    return ELO_PER_Z / math.sqrt(n)


def main():
    two_fwd = (CYCLE + SERVER) / CYCLE
    print("1. prices")
    print(f"  two forwards a decision (guided reading), server-bound: x{two_fwd:.3f} -> ${POOL * two_fwd:.3e} a pool game")
    guided_vs_net = TWO_NET * (1 + (two_fwd - 1) / 2)
    print(f"  guided side against own1 (two networks, one side doubled): ${guided_vs_net:.3e} a game")

    print("2. training runs")
    plant_pos, main_pos = 0.5e6, 4.0e6       # [C]
    rebuild_h = 25 / 60                       # corpus rebuild and pre-encoding (the own-trajectory prereg's estimate)
    plant = plant_pos / RATE / 3600 * BOX
    main = main_pos / RATE / 3600 * BOX + rebuild_h * BOX
    print(f"  plant run {plant_pos / 1e6:.1f} M positions: {plant_pos / RATE / 3600:.2f} h, ${plant:.2f}")
    print(f"  skill run {main_pos / 1e6:.1f} M positions: {main_pos / RATE / 3600:.2f} h + rebuild, ${main:.2f}")

    print("3. matches")
    n1, n2 = 37800, 12800
    m1 = n1 * POOL
    m2 = n2 * TWO_NET
    m3 = n2 * guided_vs_net
    print(f"  M1, N(s_top) against N(s_mid), one network: {n1} games, SE {se_elo(n1):.2f} Elo, ${m1:.2f}")
    print(f"  M2, N(s_top) against own1: {n2} games, SE {se_elo(n2):.2f} Elo, ${m2:.2f}")
    print(f"  M3, N guided against own1: {n2} games, SE {se_elo(n2):.2f} Elo, ${m3:.2f}")
    gap = S_TOP - S_MID
    print(f"  human-scale gap {gap:.0f} Elo: the higher setting's expected score among humans {1 / (1 + 10 ** (-gap / 400)):.3f}")
    print(f"  skill transfer f = M1 / {gap:.0f}: SE {se_elo(n1) / gap:.4f}; M1 at 2 SE detects f >= {2 * se_elo(n1) / gap:.3f}")

    print("4. the first-order guidance prediction, M3 = M2 + gamma * M1 (BT transitivity, logits linear in s)")
    for g in (1.0, 2.0):
        se_pred = math.sqrt(se_elo(n2) ** 2 + (g * se_elo(n1)) ** 2)
        miss = 2 * math.sqrt(se_pred ** 2 + se_elo(n2) ** 2)
        print(f"  gamma {g:.0f}: prediction SE {se_pred:.2f} Elo; miss line (2 SE of M3 minus prediction) {miss:.2f} Elo")
    fa = 1 - (1 - 0.0228) ** 3
    print(f"  false adoption at zero gain, three candidates (M2, M3 at gamma 1 and 2) on lower 2-SE bounds: {fa:.3f}")

    print("5. distillation of the adopted setting")
    distil = 2.1
    claim = n2 * TWO_NET
    noninf = n2 * guided_vs_net
    print(f"  own-trajectory distillation ${distil:.2f} (measured upper bound); claim against own1 ${claim:.2f}; "
          f"against its guided parent ${noninf:.2f}")
    print(f"  a distilled network {2.8:.1f} Elo below its parent is accepted with P {0.5 * math.erfc(2.8 / se_elo(n2) / math.sqrt(2)):.2f}")

    print("6. self-play stage: coins on guided disagreements (both sides read two settings)")
    batch_games = 100_000
    batch = batch_games * POOL * two_fwd
    print(f"  a batch of {batch_games} games: ${batch:.2f} ({batch / (batch_games * POOL):.2f} near-tie batches)")
    print("  labels needed to resolve the guided band's mean mu_g at 2 SE, at sigma^2 3.67 a label: 4 * 3.67 / mu_g^2;")
    print("  mu_g = (M3 in Elo / 347.4) / n_dis, n_dis the guided disagreements per side-game, both read before S4")

    print("7. budget")
    s1 = plant + main
    s2 = m1 + m2 + 2 * m3
    s3 = distil + claim + noninf
    s4 = 8 * batch + 2 * m2
    total = s1 + s2 + s3 + s4
    print(f"  S1 ${s1:.2f}; S2 ${s2:.2f}; S3 ${s3:.2f}; S4 (8 batches, adoption and claim) ${s4:.2f}; total ${total:.2f}, "
          f"left of $85 ${85 - total:.2f}")
    print(f"  exits: an empty axis after M1 ${s1 + m1:.2f}; no setting above own1 after S2 ${s1 + s2:.2f}; "
          f"S4 skipped ${s1 + s2 + s3:.2f}")
    zero_slot = n2 * TWO_NET
    print(f"  the 0-slot diagnostic (best arm at 0 slots against own1 at 0 slots), run only when no arm beats own1 "
          f"while M1 > 0: ${zero_slot:.2f}; the route then closes at ${s1 + s2 + zero_slot:.2f} if it reads null")
    print(f"  on a 1.41x slower host the game-priced items scale: total ${s1 + 1.41 * (s2 + s3 - distil + s4) + distil:.2f}")


if __name__ == "__main__":
    main()
