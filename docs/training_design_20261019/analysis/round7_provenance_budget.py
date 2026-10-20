"""Round 7, provenance stance: the plan's budget at the steady-state pool price, with the
round-7 regimen (census-only S0(a), DPI deferred, same-band round 2 conditional).

Measured inputs:
- pool price, steady state (games 120 to 480 of the ref arm): $1.912e-5 a game
  (round7_provenance_price_tail.out); the whole-wall reading $2.51e-5 is kept as a check;
- host spread 1.41x (BACKLOG.md generation rounds 4-5, via round6_provenance_prices.py);
- eval driver, two-network matches: $1.17e-4 a decisive game (round6_provenance_prices.out);
- replay forwards: 2,440 decisions a second (BACKLOG.md, via round5_simplify_critique_extra.py);
  own1 self-play records: 2,000 decisive games (gen_own1_selfplay 800, own64_selfplay 800,
  kv00 400; training/metrics/match_luck_20261019/rows.jsonl), 449.1 decisions a game
  (round5_synthesis_outcomes.out);
- distillation recipe $2.1 (docs/memory_own_trajectories_prereg_20261011.md); P-distill's training
  part: 1.4 M targets at 390 a second and 2.8 M positions at 1,477 a second (round6_provenance.md
  provenance table), plus 3,100 pool games and a 3,200-game eval reading.
Choices (round 6 design, unchanged): 100,000-game batches, 8 a round, 32,000-game matches,
interim after batch 3, S4's two 12,800-game eval matches. $0.42 an hour, $85 budget (seed).
Standard library only.
"""

HOUR = 0.42
POOL = 1.912e-5
POOL_WALL = 2.51e-5
SPREAD = 1.41
EVAL = 1.17e-4
BATCH, BATCHES, MATCH = 100_000, 8, 32_000
BUDGET = 85.0


def stages(pool):
    census = 2_000 * 449.1 / 2440 / 3600 * HOUR
    s0 = census + (3_200 + 6_400 + 2_000) * pool + 0.018
    coin = BATCHES * BATCH * pool
    m = MATCH * pool
    p_distill = (1.4e6 / 390 + 2.8e6 / 1477) / 3600 * HOUR + 3_100 * pool + 3_200 * EVAL
    s4 = p_distill + 2.1 + 2 * 12_800 * EVAL
    plan = {"S0": s0, "S1 (8 batches + interim)": coin + m, "S2 adoption": m,
            "S3 same-band round 2, conditional (+ interim, direct, claim)": coin + 3 * m,
            "S4 reset": s4,
            # after a reset the base is the distilled network: interim and direct match are one-network
            # (pool); the claim against own1 is two networks, priced at the eval median until timed
            "S5 round on the reset network (+ interim, direct; claim at eval price)": coin + 2 * m + MATCH * EVAL}
    return plan, census, m, coin


def report(pool, label):
    st, census, m, coin = stages(pool)
    print(f"\n{label}: ${pool:.3e} a pool game; coin batch ${BATCH * pool:.2f}; 32,000-game match ${m:.2f}")
    for k, v in st.items():
        print(f"  {k}: ${v:.2f}")
    reserve = 2 * m
    total = sum(st.values())
    without_s3 = total - st["S3 same-band round 2, conditional (+ interim, direct, claim)"]
    print(f"  all stages ${total:.2f} + miss reserve ${reserve:.2f} = ${total + reserve:.2f}; left ${BUDGET - total - reserve:.2f}")
    cycle = st["S4 reset"] + st["S5 round on the reset network (+ interim, direct; claim at eval price)"]
    print(f"  without S3 ${without_s3 + reserve:.2f}; left ${BUDGET - without_s3 - reserve:.2f}, "
          f"which buys {(BUDGET - without_s3 - reserve) / cycle:.2f} more reset-and-round cycles (${cycle:.2f} each)")
    print(f"  the round-6 plan priced S5's claim at the pool price; at the eval median it costs "
          f"${MATCH * EVAL - m:.2f} more")
    fixed3 = total + reserve - 3 * coin
    print(f"  three coin rounds fit iff price x (1 + overhead) <= ${(BUDGET - fixed3) / (3 * BATCHES * BATCH):.2e} "
          f"a coin game (fixed ${fixed3:.2f})")
    print(f"  a null round 1: stopped after batch 3 ${st['S0'] + 3 * BATCH * pool + m:.2f}; run through "
          f"${st['S0'] + coin + 2 * m:.2f}")
    return st


def main():
    st, census, _, _ = stages(POOL)
    print(f"S0(a) census replay of own1's 2,000 self-play games: ${census:.3f} "
          f"(the 18,800-game DPI replay: ${8.86e6 / 2440 / 3600 * HOUR:.2f})")
    report(POOL, "steady-state price")
    report(POOL * SPREAD, "steady-state price on a host 1.41x slower")
    report(POOL_WALL, "round 6's whole-wall price (check against round6_synthesis_power.out)")
    hours = BATCH / (3600 / (POOL / HOUR * 3600))
    print(f"\nwall per coin batch at the steady rate: {BATCH * POOL / HOUR:.2f} h; per round of 8: "
          f"{BATCHES * BATCH * POOL / HOUR:.1f} h ({hours:.2f} h check)")
    print("DPI's saving if admitted (round6_synthesis_power.out section 5): one batch a round at r <= 0.602 = "
          f"${BATCH * POOL:.2f}, two at r <= 0.516 = ${2 * BATCH * POOL:.2f}")


if __name__ == "__main__":
    main()
