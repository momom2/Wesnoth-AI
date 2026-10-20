"""Round 7 (challenge stance): sizes and prices of turn-level intent coins, from measured inputs.

Inputs (tag, source):
  R_SHARE 0.688        [M] label residual share of Var(z), full setup cell + seven columns
                           (round6_provenance_cells.py, current_design.md section 2)
  VAR_Z 1.0004         [M] same source
  TURNS 19.4           [M] side-turns per own1 side-game (round5_simplify_critique_decision_units.out)
  BAND 76.1            [D] near-tie band decisions per side-game (current_design.md section 3.1)
  GAME_PRICE 2.51e-5   [M] a pool game at 120 jobs (round6_provenance_prices.out)
  PAIR_KEEP 0.847      [M] PURE side-swapped pairs keep this share (current_design.md section 2)
  ELO_PER_Z 347.4      [D] 800 / ln 10
  TRAIN_RATE 1242      [M] sequence-trainer positions a second, columnar records, window batched
                           (BACKLOG.md, trainer_speed_2_20261014)
  TEACHER_RATE 385     [M] trainer-pipeline forwards a second: 1.1 M teacher targets in 2,854 s
                           (BACKLOG.md, round 2 of own-trajectory training)
  BOX 0.42             [M] dollars an hour (seed section 1)
Choices [C]: K intents 8, default probability P0 0.5, batch 100,000 games, selector features d 768 or 64 (PCA),
EM corpus 200,000 human positions, EM iterations 3.

The label for intent c at a side-turn whose arm D was drawn with probabilities P:
    T_c = s * R * (1[D = c] / P_c - 1[D = 0] / P_0),  E[T_c | H] = Q(c) - Q(0) exactly,
    Var(T_c) ~= Var(R) * (1/P_c + 1/P_0).
numpy-free: standard library only.
"""
import math

R_SHARE, VAR_Z = 0.688, 1.0004
TURNS, BAND = 19.4, 76.1
GAME_PRICE, PAIR_KEEP, ELO_PER_Z = 2.51e-5, 0.847, 347.4
TRAIN_RATE, TEACHER_RATE, BOX = 1242.0, 385.0, 0.42
K, P0, BATCH, D_SEL = 8, 0.5, 100_000, 768
EM_POSITIONS, EM_ITERS = 200_000, 3


def e_max_pos(k, steps=20000, top=8.0):
    """E[max(0, max of k iid standard normals)] = integral over x > 0 of 1 - Phi(x)^k."""
    h = top / steps
    phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    return sum((1 - phi((i + 0.5) * h) ** k) * h for i in range(steps))


def label_var(pc, p0):
    return R_SHARE * VAR_Z * (1 / pc + 1 / p0)


def main():
    pc = (1 - P0) / K
    s2 = label_var(pc, P0)
    turns_per_batch = BATCH * 2 * TURNS
    print(f"1. per-code label variance {s2:.2f} (P_c {pc:.4f}, P0 {P0}); a binary near-tie coin at p 0.25: "
          f"{R_SHARE * VAR_Z / 0.1875:.2f}")
    for b in (1, 2, 4):
        n = b * turns_per_batch
        se_mean = math.sqrt(s2 / n)
        fits = ", ".join(f"d {d}: {d * s2 / n:.5f} (sd {math.sqrt(d * s2 / n):.3f})" for d in (D_SEL, 64))
        print(f"   batches {b} ({b * BATCH:,} games, {n / 1e6:.2f} M turn labels a code): per-code mean advantage SE "
              f"{se_mean:.4f} a turn; selector fit noise variance d*s2/n {fits}")
    print("2. forecast SE (model-free, held-out labels summed over the turns the selector changes), batches 2-4:")
    n_sg = 3 * BATCH * 2
    for q in (0.1, 0.3, 0.5):
        se_z = math.sqrt(q * TURNS * s2 / n_sg)
        print(f"   share of turns changed q {q}: SE {se_z:.4f} z a side-game = {ELO_PER_Z * se_z:.1f} Elo")
    print("3. information break-even against near-tie coins at p 0.25 (same game price):")
    info_turn = 2 * TURNS / s2
    info_band = 2 * BAND / (R_SHARE * VAR_Z / 0.1875)
    ratio = math.sqrt(info_band / info_turn)
    print(f"   per unit stakes^2 a game: intent contrasts {info_turn:.2f} per code, near-tie coins {info_band:.2f}; "
          f"one intent contrast matches the band iff its stakes exceed {ratio:.2f} x the near-tie spread")
    print("   (K codes are read from the same turns, so the K contrasts together carry K times this)")
    print("4. oracle ceilings per side-game in z (formulas; neither stakes value is measured at own1):")
    e_max = {k: e_max_pos(k) for k in (1, K)}
    print(f"   E[max(0, max of k iid N(0,1))]: k 1 {e_max[1]:.3f}, k {K} {e_max[K]:.3f}")
    for s_t in (0.01, 0.03, 0.07):
        print(f"   intent stakes sd {s_t}: {TURNS * e_max[K] * s_t:.3f} z ({ELO_PER_Z * TURNS * e_max[K] * s_t:.0f} Elo)", end="; ")
    print()
    for v in (0.002, 0.004, 0.01):
        print(f"   near-tie spread sd {v}: {BAND * 0.399 * v:.3f} z ({ELO_PER_Z * BAND * 0.399 * v:.0f} Elo)", end="; ")
    print()
    print("5. prices:")
    e_step = EM_POSITIONS * K / TEACHER_RATE
    m_step = EM_POSITIONS / TRAIN_RATE
    em = EM_ITERS * (e_step + m_step) / 3600 * BOX
    plant_e = EM_POSITIONS * 2 / TEACHER_RATE
    plant = EM_ITERS * (plant_e + m_step) / 3600 * BOX
    print(f"   EM on {EM_POSITIONS:,} human positions, K {K}, {EM_ITERS} iterations: E-step {e_step / 3600:.2f} h, "
          f"M-step {m_step / 3600:.3f} h each; ${em:.2f}")
    print(f"   P-EM plant, K 2, same corpus size: ${plant:.2f}")
    div = 800 * 449.1 * (K + 1) / TEACHER_RATE / 3600 * BOX
    print(f"   diversity read: 800 own1 games x 449.1 decisions x {K + 1} readings: ${div:.2f}")
    batch = BATCH * GAME_PRICE
    match_games = 32_000
    print(f"   coin batch ${batch:.2f} before logging overhead; PURE paired match {match_games:,} games "
          f"${match_games * GAME_PRICE:.2f}, SE {ELO_PER_Z * math.sqrt(PAIR_KEEP / match_games):.2f} Elo")
    for nb in (1, 2, 4):
        total = em + plant + div + nb * batch + 2 * match_games * GAME_PRICE
        print(f"   round 1 with {nb} batch(es), interim and adoption matches: ${total:.2f}")
    print("6. P-intent plant (booked into z along a random unit u for code c*): slope SE after one batch")
    for var_ux in (1.0, 0.1):
        se = math.sqrt(s2 / (turns_per_batch * var_ux))
        print(f"   Var(u.x) {var_ux}: SE of the per-side slope on u.x {se:.4f} (delta for a 4-SE band: {4 * se:.4f})")


if __name__ == "__main__":
    main()
