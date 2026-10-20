"""Round 8, critique of the simplify stance: checks on round8_simplify_power.py's claims.

1. The interim's futility line as a choice, not a keep-or-remove switch. Under the stance's own
   model and inputs (round8_simplify_power.powers, unchanged), stop iff student 3's match estimate
   is below c match SEs (the current design has c = 0). For each c: P(stop) at a null, expected
   coin spend of a null round 1, wall hours saved at a null, and adoption power at the stance's
   no-interim floors (sd 0.0016 at mu 0, 0.0034 at mu -0.0018).
2. The coin plant with a share f of coins read one decision off: the expected slope is exactly
   1 - f (a misread coin's score is independent of its own increment), so the distance from 1 on
   batch 1 is f / sqrt(3 / 100,000). The stance used 100-replicate simulated means.
3. False alarms of the plant gate on correct code: two sides, read on the smoke and on batch 1
   (four readings, independent to first order), at 2 and 3 SE.
numpy and the standard library only; imports the stance's script for its model.
"""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import round5_simplify_power as P  # noqa: E402
import round6_synthesis_power as R6  # noqa: E402
import round8_simplify_power as S  # noqa: E402

BATCH_HOURS = 4.55  # current_design.md section 4: a batch takes 4.55 h


def section1(cells):
    inp = R6.inputs_at(S.R_REF)
    b3, b8 = S.eff(cells, 3), S.eff(cells, 8)
    print("1. futility line c (stop iff student 3's estimate < c match SEs); stance's model and inputs")
    floors = {mu: R6.bisect(lambda x, mu=mu: S.powers(inp, x, mu, b3, b8)[0] >= 0.8)
              for mu in (0.0, -0.0018)}
    print("   no-interim floors (bisected as in the stance): " +
          ", ".join(f"mu {mu:+.4f} sd {sd:.5f}" for mu, sd in floors.items()))
    s0 = S.SETUP + S.ATTACK * S.POOL
    m = S.MATCH * S.POOL
    coin = S.BATCHES * S.BATCH * S.POOL
    stopped = s0 + 3 * S.BATCH * S.POOL + m
    through = s0 + coin + 2 * m
    print(f"   no interim: null round 1 ${s0 + coin + m:.2f}; power 0.800 at both floors")
    for c in (0.0, -0.5, -1.0, -1.5, -2.0):
        p_null = P.Phi(c)
        spend = p_null * stopped + (1 - p_null) * through
        hours = p_null * (S.BATCHES - 3) * BATCH_HOURS
        cells_txt = []
        for mu, sd in floors.items():
            g3, _ = R6.gain(inp, sd, mu, S.D_EFF, b3)
            g8, _ = R6.gain(inp, sd, mu, S.D_EFF, b8)
            p_stop = P.Phi(c - g3 / S.SE_M)
            power = (1 - p_stop) * P.Phi(g8 / S.SE_M - 2)
            cells_txt.append(f"mu {mu:+.4f} power {power:.3f}")
        print(f"   c {c:+.1f}: P(stop | null) {p_null:.3f}; expected null spend ${spend:.2f} "
              f"(saves ${s0 + coin + m - spend:.2f}); wall saved at a null {hours:.1f} h; " + ", ".join(cells_txt))


def section2():
    se_b1 = math.sqrt(3 / S.BATCH)
    se_smoke = math.sqrt(3 / S.SMOKE)
    print(f"\n2. plant with a share f misread: slope 1 - f exactly; SE {se_smoke:.4f} at the smoke, {se_b1:.4f} on batch 1")
    for f in (0.02, 0.05, 0.10, 0.20):
        print(f"   f {f:.2f}: {f / se_smoke:.1f} SE at the smoke, {f / se_b1:.1f} SE on batch 1; "
              f"detection probability on batch 1 at a 2-SE line {P.Phi(f / se_b1 - 2):.2f}, "
              f"at 3 SE {P.Phi(f / se_b1 - 3):.2f}")


def section3():
    print("\n3. plant gate false alarms on correct code, four readings")
    for k in (2.0, 3.0):
        one = 2 * (1 - P.Phi(k))
        print(f"   {k:.0f}-SE line: one reading {one:.4f}, any of four {1 - (1 - one) ** 4:.3f}")


def main():
    cells = S.R7S.read_cells()
    section1(cells)
    section2()
    section3()


if __name__ == "__main__":
    main()
