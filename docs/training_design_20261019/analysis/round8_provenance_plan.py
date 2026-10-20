"""Round 8, provenance stance: the plan's budget with the interim match removed.

Inputs are round7_synthesis_power.py's, imported unchanged (pool price $1.912e-5 a game, steady
state; eval driver $1.17e-4 a two-network decisive game; 37,800-game one-network matches; 12,800-game
two-network matches; distillation $2.1; box setup $0.018; 8 batches of 100,000 games).
Changes from the round-7 plan: no interim match in S1, S3 or S5 (round8_provenance_interim_shapes.out:
its false-stop probability at an adoptable round is not bounded without a shape assumption); no
census replay (the smoke, the first 2,000 games of batch 1, reads the census); one fallback-E match
reserved per adoption reading (S2, S3, S5: three). The tested threshold adds no compute.
Standard library only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round7_synthesis_power as R  # noqa: E402


def stages(pool):
    m = R.MATCH * pool
    coin = R.BATCHES * R.BATCH * pool
    two_net = R.S4_GAMES * R.EVAL
    return {
        "S0 setup, attack known answer (6,400 games)": R.SETUP + 6_400 * pool,
        "S1 round 1 (8 batches)": coin,
        "S2 adoption match": m,
        "E reserve (three matches)": 3 * m,
        "S3 round 2 (8 batches, adoption vs pi_1, claim vs own1)": coin + 2 * m,
        "S4 reset (distillation, one two-network match)": R.DISTILL + two_net,
        "S5 round on the reset base (8 batches, adoption vs base, two-network claim)": coin + m + two_net,
    }, m, coin


def report(pool, label):
    st, m, coin = stages(pool)
    total = sum(st.values())
    print(f"\n{label}: ${pool:.3e} a game; batch ${R.BATCH * pool:.2f}; match ${m:.2f}")
    for k, v in st.items():
        print(f"  {k}: ${v:.2f}")
    print(f"  all stages ${total:.2f}; left of ${R.BUDGET:.0f}: ${R.BUDGET - total:.2f}")
    cycle = st["S4 reset (distillation, one two-network match)"] + \
        st["S5 round on the reset base (8 batches, adoption vs base, two-network claim)"]
    print(f"  one more reset-and-round cycle ${cycle:.2f}; the remainder buys {(R.BUDGET - total) / cycle:.2f} of one")
    fixed3 = total - 3 * coin
    line = (R.BUDGET - fixed3) / (3 * R.BATCHES * R.BATCH)
    print(f"  three rounds fit iff price x (1 + overhead) <= ${line:.3e} a coin game (overhead room {line / pool - 1:.3f})")
    s0 = st["S0 setup, attack known answer (6,400 games)"]
    print(f"  a null round 1 (S0, 8 batches, adoption match): ${s0 + coin + m:.2f}")


def ceiling_availability():
    """Round 2's sqrt(N) ceiling is stated only if every margin bin's mean label has a lower 2-SE bound
    >= 0. The first 0.1-nat bin of a same-kind pair type has mean about b x 0.05 (continuity at a tie
    gives 0 at G = 0), with SE 0.00085 at best (round7_provenance_scaling_bound.out section 4)."""
    import math
    print("\nchance that the first same-kind bin certifies a nonnegative mean (all bins must):")
    for b in (-0.004, -0.001, 0.0, 0.001, 0.004):
        z = b * 0.05 / 0.00085 - 2.0
        print(f"  margin slope b {b:+.3f} per nat: P {0.5 * (1 + math.erf(z / math.sqrt(2))):.3f}")


def main():
    report(R.POOL, "steady-state price")
    report(R.POOL * R.SPREAD, "host 1.41x slower")
    ceiling_availability()


if __name__ == "__main__":
    main()
