"""Round 9, critique of the provenance stance: arithmetic checks of its decision
numbers. Standard library only; inputs are the stance's own measured outputs
(round9_provenance_allin.out, recruit_cells.out, unit_cells.out, cells.out).

1. S5's host ceiling when S1-S4 run at $0.577/h (the stance's stated policy),
   against the $0.569/h uniform-rate ceiling its summary quotes.
2. L's detection probability with F's own uncertainty included.
3. What L's ratio rho_L = J/F says about S1's step under the design's own
   quadratic second-order model (round7_synthesis.md: half the switches,
   a quarter of the quadratic loss): J(n) = a n - c n^2, rho(n) = 1 - k n.
4. L and T read jointly, if T's prediction is J_T = J_L.
5. The attack-pair reading (b) with T, -8.3 +- 2.7 Elo: the per-switch mean it
   implies against the attack pairs' typed means (cells.out section 4).
"""
import math

ELO = 800 / math.log(10)
LEFT = 85.0 - 2.66


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def s5_ceiling():
    """Uses round9_provenance_allin.plan: S1-S4, L, T, C and E priced at $0.577/h,
    S5 alone at the rate r that keeps the path within $82.34."""
    import importlib.util
    from pathlib import Path
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("allin", here / "round9_provenance_allin.py")
    allin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(allin)
    match_s, post_s, _games = allin.post_share()
    post = post_s / match_s
    print("1. S5's ceiling with S1-S4 (and L, T, C, E) at $0.577/h")
    for h in (0.0, 0.05, 0.10):
        s = allin.plan(0.577, post, h)
        rest = sum(v for k, v in s.items() if k != "S5")
        lo, hi = 0.05, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if allin.plan(mid, post, h)["S5"] <= LEFT - rest else (lo, mid)
        print(f"  h {h:.2f}: S1-S4 path ${rest:.2f}, room for S5 ${LEFT - rest:.2f}; S5's host <= ${lo:.3f}/h "
              f"(uniform-rate ceiling {allin.ceiling(post, h):.3f})")


def detection():
    print("\n2. L: P(lower 2-SE bound > 0), J ~ N(rho F, 3.07^2 + rho^2 3.2^2)")
    for rho in (0.3, 0.5, 0.7, 1.0):
        known = phi((rho * 22.9 - 2 * 3.07) / 3.07)
        sd = math.sqrt(3.07 ** 2 + (rho * 3.2) ** 2)
        full = phi((rho * 22.9 - 2 * 3.07) / sd)
        print(f"  rho {rho:.1f}: F known {known:.3f}; F uncertain {full:.3f}")
    print("  a null (J <= 6.14) bounds rho at 2 SE by "
          f"{(6.14 + 2 * math.sqrt(3.07 ** 2 + 3.2 ** 2 * 0.25)) / 22.9:.2f} (at a J of 6.14)")


def step_transfer():
    print("\n3. rho at S1's step from rho_L, quadratic model rho(n) = 1 - k n")
    n_l, se_rho = 2.21, 0.194
    se_k = se_rho / n_l
    for n_s1, what in ((19.0, "19 switches (design's low end)"), (38.8, "q 0.5 x 77.6")):
        print(f"  {what}: SE of rho(n) {n_s1 * se_k:.2f}; rho_L 0.9 maps to {1 - n_s1 * 0.1 / n_l:+.2f}, "
              f"rho_L 0.97 to {1 - n_s1 * 0.03 / n_l:+.2f}")


def joint():
    print("\n4. L and T jointly, if T's sealed prediction is J_T = J_L")
    print(f"  SE of the mean of two 12,800-game matches {3.07 / math.sqrt(2):.2f} Elo; "
          f"SE of J_T - J_L {3.07 * math.sqrt(2):.2f}")


def attack():
    print("\n5. attack pairs, rule (b) with T: -8.3 +- 2.7 Elo, switching 0.122 of 13.23 a side-game")
    per_sg = 0.122 * 13.23
    w = -0.02402
    typed = {"attack>move": (-0.0003, 487948), "move>attack": (-0.0031, 452164),
             "attack>attack:weapon": (-0.0026, 221848), "attack>attack:other_target": (-0.0010, 213705),
             "attack>attack:same_target": (-0.0066, 128493), "recruit>attack": (0.0025, 23697),
             "attack>recruit": (-0.0072, 23130), "end_turn>attack": (-0.0135, 18994),
             "attack>end_turn": (0.0111, 18124)}
    n = sum(v[1] for v in typed.values())
    m = sum(v[0] * v[1] for v in typed.values()) / n
    print(f"  {per_sg:.2f} switches a side-game; implied mean per switched label {w / per_sg:+.4f}; "
          f"pooled typed mean of attack pairs {m:+.4f} over {n} labels "
          f"(explains {m * per_sg / w:.2f} of the reading)")


if __name__ == "__main__":
    s5_ceiling()
    detection()
    step_transfer()
    joint()
    attack()
