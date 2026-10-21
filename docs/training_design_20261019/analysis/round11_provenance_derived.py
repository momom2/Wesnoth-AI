"""Round 11 (provenance stance): the derived numbers of round11_provenance.md, from measured inputs.

Every input is a measured record (named beside it); every output is arithmetic on them. Numbers
printed as "illustration" carry no decision: they show what a formula gives at values not yet
measured.
numpy and the standard library only.
"""
import math

# ---- measured inputs
HOST_RATE = 0.577                 # $/h, the coin box (seed section 7; round9_simplify_labels.out)
GEN_GAME = 2.598e-5               # $ per one-network pool game at HOST_RATE (round9_simplify_labels.out)
CLAIM_GAME = 8.848e-5             # $ per decisive two-network game (round9_simplify_labels.out)
TRAIN_RATE = 1477.0               # positions/s, configuration B (training/metrics/trainer_calibration_20261015)
DECISIONS = 475.7                 # per game at raw:t0+eo-1.5, own2 v own1 (round11_provenance_census.out)
SIDE_DECISIONS = 237.9
SIDE_TURNS = 20.16
R2_START_LUCK = (0.346, 0.298, 0.395)   # turn-start share, luck-corrected (round11_provenance_turn_signal.out)
I_TURN = (0.0204, 0.0064, 0.0394)       # per side-turn, luck-corrected (round11_provenance_turn_signal.out)
T1_TURN_COST = (-0.0679, -0.1264, -0.0159)
RESID_ARGMAX = 0.6988             # Var(z - b(cell) - beta S)/Var(z), own1 coin games (round9_simplify_residual.out)
SEPT_W, SEPT_L = 37, 3            # seed argmax v seed sampling (docs/raw_argmax_control_20260904.md)
ELO_PER_Z = 400 / math.log(10) / 0.25 * 0.5   # d Elo / d E[z] at E[z] = 0: (400/ln10)/(p(1-p)) x dp/dz, p = 1/2


def elo_of_z(z):
    p = (1 + z) / 2
    return 400 * math.log10(p / (1 - p))


def section_prices():
    train_game = DECISIONS / TRAIN_RATE / 3600 * HOST_RATE
    print("1. prices at $%.3f/h" % HOST_RATE)
    print(f"   training every decision of a game: {DECISIONS / TRAIN_RATE:.3f} s, ${train_game:.3e}; generation ${GEN_GAME:.3e};"
          f" ratio {train_game / GEN_GAME:.2f}; a game generated and trained ${GEN_GAME + train_game:.3e}")
    for n in (2500, 10000, 40000):
        print(f"   gradient batch of {n:,} games: ${n * (GEN_GAME + train_game):.2f}")
    fwd_bound = DECISIONS / TRAIN_RATE / 3600 * HOST_RATE   # a forward costs at most a trained position
    print(f"   certification forward of one candidate over 2,500 held-out games: at most ${2500 * fwd_bound:.3f}"
          f" (a forward priced as a trained position; forward-only speed is unmeasured)")
    print(f"   held-out set of 2,500 games generated: ${2500 * GEN_GAME:.3f}; claim match of 12,800 two-network games ${12800 * CLAIM_GAME:.2f}")
    return train_game, fwd_bound


def section_stage_costs(train_game, fwd_bound):
    print("2. stage costs (generation priced as pool games; the T=1 game length is unmeasured, so each scales with it)")
    b1 = 3200 * GEN_GAME
    b2 = 3200 * CLAIM_GAME
    b3 = 20000 * GEN_GAME
    print(f"   B1 own2 T1 v own2 reference decode, 3,200 decisive: ${b1:.2f}; B2 own2 T1 64 v 0 slots, 3,200: ${b2:.2f};"
          f" B3 20,000 own2 T1 self-play: ${b3:.2f}; B total ${b1 + b2 + b3:.2f}")
    c_grad = 20000 * train_game
    print(f"   C gradient pass over B3's 20,000 games: {20000 * DECISIONS / TRAIN_RATE / 3600:.2f} h, ${c_grad:.2f};"
          f" eleven certified candidates on 2,500 held-out games at most ${11 * 2500 * fwd_bound:.2f}")
    p = SEPT_W / (SEPT_W + SEPT_L)
    for gap_p in (0.9, 0.75):
        se_p = math.sqrt(gap_p * (1 - gap_p) / 3200)
        print(f"   B1's SE at p {gap_p}: {400 / math.log(10) / (gap_p * (1 - gap_p)) * se_p:.1f} Elo")
    print(f"   Sept seed argmax v sampling: p {p:.3f}, E[z] {2 * p - 1:.3f}, {elo_of_z(2 * p - 1):.0f} Elo (PURE fit +412 +- 97)")


def section_forks():
    print("3. forks (K continuations of one turn-start state, all trained, each weighted 1/K):")
    print("   noise factor per trained post-fork decision (1 - R2) K / (K - 1), against 1 for a single path")
    for r2 in R2_START_LUCK:
        row = ", ".join(f"K {k}: {(1 - r2) * k / (k - 1):.3f}" for k in (2, 4, 8, 16))
        print(f"   R2 {r2:.3f}: {row}; limit {1 - r2:.3f}")


def section_signal():
    print("4. signal indicators (obs8, human positions at game turn 7 or later, temperature-0.5 continuations)")
    lo, mid, hi = I_TURN[1], I_TURN[0], I_TURN[2]
    print(f"   I_turn {mid:.4f} ({lo:.4f} to {hi:.4f}) per side-turn; times {SIDE_TURNS} side-turns:"
          f" {mid * SIDE_TURNS:.2f} ({lo * SIDE_TURNS:.2f} to {hi * SIDE_TURNS:.2f}) per side-game, against the identity's"
          f" ceiling of 1/2 per side-game (both sides' choice variance plus chance plus the setup's share sum to Var(z) = 1)")
    c = T1_TURN_COST
    print(f"   T1 turn minus argmax turn {c[0]:+.4f} ({c[1]:+.4f} to {c[2]:+.4f}) per side-turn; times {SIDE_TURNS}:"
          f" {c[0] * SIDE_TURNS:+.2f}, beyond the bound |E[z]| <= 1, so per-turn costs do not add over a game")
    print(f"   per-decision share of a side-game's signal at the indicator: {mid * SIDE_TURNS / SIDE_DECISIONS:.4f}")


def section_certification():
    print("5. held-out importance-sampling certification of a step of per-side-game KL K (sigma^2 = residual share)")
    print("   SE(dJ) = sigma sqrt((exp(2K) - 1) / (2 |H|)) per game pair of sides; a match of |H| games: sigma / sqrt(|H|)")
    for sigma2 in (RESID_ARGMAX, 1.0):
        for k in (1 / 16, 1 / 4, 1 / 2, 1.0):
            for h in (2500, 10000):
                se = math.sqrt(sigma2 * (math.exp(2 * k) - 1) / (2 * h))
                match = math.sqrt(sigma2 / h)
                print(f"   sigma2 {sigma2:.3f} K {k:.4f} |H| {h:>6,}: SE {se:.4f} z = {se * ELO_PER_Z:.1f} Elo;"
                      f" match {match * ELO_PER_Z:.1f} Elo; ratio {se / match:.2f}; ESS share exp(-2K) {math.exp(-2 * k):.2f}")
    print(f"   Elo per unit of E[z] at E[z] = 0: {ELO_PER_Z:.1f}")


def section_illustrations(train_game, fwd_bound):
    print("6. illustrations (no decision rests on them): first-order gain sqrt(2 K I cos2) at K = 1/4")
    for i_side in (0.05, 0.2, 0.41):
        row = ", ".join(f"cos2 {c}: {math.sqrt(2 * 0.25 * i_side * c):.3f} z" for c in (1.0, 0.1, 0.01))
        print(f"   I {i_side}: {row}")
    c_g = GEN_GAME + train_game
    c_f = 2500 * GEN_GAME + 4 * 2500 * fwd_bound
    print(f"   cost-optimal batch N* = sqrt(c_f B / c_g), c_f ${c_f:.3f} a step (2,500 held-out games, four candidates), c_g ${c_g:.2e}:")
    for b in (1e3, 1e4, 1e5, 1e6):
        print(f"   B {b:,.0f} games: N* {math.sqrt(c_f * b / c_g):,.0f}")


def main():
    train_game, fwd_bound = section_prices()
    section_stage_costs(train_game, fwd_bound)
    section_forks()
    section_signal()
    section_certification()
    section_illustrations(train_game, fwd_bound)


if __name__ == "__main__":
    main()
