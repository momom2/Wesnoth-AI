"""Round 7 synthesis: the plan's budget and the readings it adds, at the round-7 inputs.

Inputs (measured, with their sources):
- pool price, steady state: $1.912e-5 a one-network game (round7_provenance_price_tail.out,
  completions 120-480 of arm_ref_p120x6; sub-windows 21,600 / 21,600 / 22,737 games/h in
  round7_provenance_critique_checks.out section 1); host spread 1.41x (BACKLOG.md via
  round6_provenance_prices.py); eval driver $1.17e-4 a decisive game (round6_provenance_prices.out),
  used for every two-network match until the pool's two-network price is timed; box setup $0.018.
- census replay: 2,000 own1 self-play games x 449.1 decisions at 2,440 decisions a second
  (round7_provenance_budget.py inputs).
- distillation recipe $2.1, the prereg's whole box run (docs/memory_own_trajectories_prereg_20261011.md;
  an upper bound, its two matches included).
- attack known answer: slope -0.2247 +- 0.0459 z per unit of attack offset on unselected stage-1 games
  (round6_provenance_critique_attack_stage1.out B).
- Elo per unit of mean z: 800 / ln 10 = 347.4.
Choices [C]: independent matches of 37,800 decisive games (SE 1.79, the paired 32,000's SE), 8 batches
of 100,000 coin games a round, interim after batch 3, a 12,800-game two-network match in S4 and for the
post-reset claim, an E reserve of two matches.
Standard library only (numpy for the small simulation in section 3).
"""
import math

import numpy as np

HOUR = 0.42
POOL = 1.912e-5
SPREAD = 1.41
EVAL = 1.17e-4
SETUP = 0.018
BUDGET = 85.0
BATCH, BATCHES = 100_000, 8
ELO = 800 / math.log(10)
PAIR_KEPT = 0.847                      # round6_simplify_pairs_raw.out
MATCH = round(32_000 / PAIR_KEPT, -2)  # independent games at the paired match's SE
S4_GAMES = 12_800
DISTILL = 2.1


def match_se(n, kept=1.0):
    return ELO * math.sqrt(kept / n)


def stages(pool):
    census = 2_000 * 449.1 / 2440 / 3600 * HOUR
    m = MATCH * pool
    coin = BATCHES * BATCH * pool
    two_net = S4_GAMES * EVAL
    st = {
        "S0 (setup, census replay, attack known answer 6,400 games)": SETUP + census + 6_400 * pool,
        "S1 (8 batches + interim)": coin + m,
        "S2 adoption": m,
        "E reserve (two matches)": 2 * m,
        "S3 round 2 on pi_1 (8 batches, interim, adoption vs pi_1, claim vs own1)": coin + 3 * m,
        "S4 reset (distillation, one two-network match)": DISTILL + two_net,
        "S5 round on the reset base (8 batches, interim and adoption vs the base, claim vs own1 two-network)":
            coin + 2 * m + two_net,
    }
    return st, census, m, coin


def report(pool, label):
    st, census, m, coin = stages(pool)
    print(f"\n{label}: ${pool:.3e} a pool game; batch ${BATCH * pool:.2f}; {MATCH:.0f}-game match ${m:.3f}")
    for k, v in st.items():
        print(f"  {k}: ${v:.2f}")
    total = sum(st.values())
    print(f"  all stages ${total:.2f}; left of ${BUDGET:.0f}: ${BUDGET - total:.2f}")
    cycle = st["S4 reset (distillation, one two-network match)"] + \
        st["S5 round on the reset base (8 batches, interim and adoption vs the base, claim vs own1 two-network)"]
    print(f"  one more reset-and-round cycle costs ${cycle:.2f}; the remainder buys {(BUDGET - total) / cycle:.2f} of one")
    fixed3 = total - 3 * coin
    line = (BUDGET - fixed3) / (3 * BATCHES * BATCH)
    print(f"  three coin rounds fit iff price x (1 + overhead) <= ${line:.3e} a coin game; "
          f"overhead room at this price {line / pool - 1:.3f}")
    s5 = st["S5 round on the reset base (8 batches, interim and adoption vs the base, claim vs own1 two-network)"]
    fixed2 = total - s5 - 2 * coin
    print(f"  two coin rounds (S5 dropped, S4 kept) fit iff price x (1 + overhead) <= "
          f"${(BUDGET - fixed2) / (2 * BATCHES * BATCH):.3e} a coin game")
    s0 = st["S0 (setup, census replay, attack known answer 6,400 games)"]
    stop = s0 + 3 * BATCH * pool + m
    full = s0 + coin + 2 * m
    print(f"  a null round 1: stopped at the interim ${stop:.2f}; run through ${full:.2f}; "
          f"expected at P(stop) 0.50 ${0.5 * (stop + full):.2f}")
    return st


def section1():
    print("1. matches")
    print(f"  independent {MATCH:.0f} games: SE {match_se(MATCH):.2f} Elo; paired 32,000: SE {match_se(32_000, PAIR_KEPT):.2f}")
    print(f"  extra cost of independent games at the steady price: ${(MATCH - 32_000) * POOL:.3f} a match")
    print(f"  S4 / post-reset claim, {S4_GAMES} independent two-network games: SE {match_se(S4_GAMES):.2f} Elo, "
          f"${S4_GAMES * EVAL:.2f} at the eval median")
    se_ka = match_se(6_400)
    print(f"  attack known answer, 6,400 independent games: SE {se_ka:.2f} Elo; prediction "
          f"{ELO * -0.2247:.1f} +- {ELO * 0.0459:.1f}; window +-{2 * math.hypot(ELO * 0.0459, se_ka):.1f} Elo")


def section2():
    print("\n2. the attack known answer read through the label code (one logged coin a game: which seat")
    print("   carries the hook, P = 1/2, omega = +-2, mover sign of side A)")
    d, d_se = -0.2247, 0.0459
    # E[T] = E[z_A | A hooked] - E[z_A | B hooked] = (m + D) - (m - D) = 2 D: the coin moves the hook
    # from one seat to the other, so the label reads the hooked side's effect twice.
    se_label = 2 / math.sqrt(6_400)
    half = 2 * math.hypot(2 * d_se, se_label)
    print(f"  E[T] = 2 Delta = {2 * d:+.3f} z, prediction SE {2 * d_se:.3f}; label SE at 6,400 games {se_label:.3f}")
    print(f"  window {2 * d:+.3f} +- {half:.3f} = [{2 * d - half:+.3f}, {2 * d + half:+.3f}]; "
          f"result oriented to the wrong side reads {-2 * d:+.3f}; half the games flipped read 0")
    print(f"  (round7_simplify_critique_checks.out section 3 predicted {d:+.3f}: the hooked side's effect once)")


def section3():
    print("\n3. simulation check of section 2 (symmetric game, side advantage m, hook effect Delta on the hooked side's z)")
    n, m_side, delta = 4_000_000, 0.04, -0.2247
    for seed in (7, 1, 2, 3):
        rng = np.random.default_rng(seed)
        hook_a = rng.random(n) < 0.5
        mean_za = np.where(hook_a, m_side + delta, m_side - delta)
        z_a = np.where(rng.random(n) < (1 + mean_za) / 2, 1.0, -1.0)
        omega = np.where(hook_a, 2.0, -2.0)
        t = omega * z_a
        print(f"  seed {seed}: mean label {t.mean():+.4f} (2 Delta {2 * delta:+.4f}); SE {t.std() / math.sqrt(n):.4f}")


def section4():
    print("\n4. the baked end_turn shift and its residual offset across resets: o_(k+1) = o_k (1 - c)")
    for c in (0.6, 0.82, 1.0, 1.3):
        o, seq = -1.5, []
        for _ in range(4):
            o = o * (1 - c)
            seq.append(f"{o:+.3f}")
        print(f"  closure c {c:.2f}: residual offsets after resets 1-4: {', '.join(seq)}")
    print("  converges to 0 for c in (0, 2)")


def main():
    section1()
    section2()
    section3()
    section4()
    print("\n5. budget")
    report(POOL, "steady-state price")
    report(POOL * SPREAD, "steady-state price on a host 1.41x slower")
    st, census, _, _ = stages(POOL)
    print(f"\n  census replay ${census:.3f}; wall per batch {BATCH * POOL / HOUR:.2f} h, per round {BATCHES * BATCH * POOL / HOUR:.1f} h")


if __name__ == "__main__":
    main()
