"""Round 7, critique of the invent stance: checks of its switch-set bound, budget line, batch
savings, the curvature a logit-linear effect would leave between J'(1/2) and J(1) - J(0), and the
conditioning of the M_T / F_D ratio.

1. |S_1| "upper bound" (round7_invent_known_switch.py counts a reading only if an attack is among
   the top eight). Readings whose top is not an attack, with no attack among the top eight, whose
   eighth action is still within a nats, may hold an attack within a further down: count them.
   Input: training/metrics/critic_step1_20261009/prior_gaps.games.jsonl, scaled to own1's 224.6
   raw decisions per side-game as the stance did.
2. The three-round price line after the stance's additions. round6_synthesis_power.out section 6:
   fixed costs $14.95, 3 rounds x 8 batches x 100,000 coin games, $85.
3. What r * m <= 0.602 saves per round with the design's fixed 100,000-game batches
   (ceil(8 r / 0.688) batches at m times the batch price) against batches resized to 100,000 / m.
4. Midpoint error if the coined seat's logit is linear in t (J(t) = 2 sigma(-b t) - 1, J(1) at
   -78 Elo): the size of the curvature term in a plausible smooth case.
5. F_H's per-seat SE at the $2 cap and the window's power against a 30% scale error.
numpy and the standard library only.
"""
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "training/metrics/critic_step1_20261009/prior_gaps.games.jsonl"
OWN1_DECISIONS = 224.6
ELO_PER_Z = 800 / math.log(10)


def part1():
    rows = []
    for line in open(SRC):
        rec = json.loads(line)
        for rd in rec["readings"]:
            if rd["legal"] < 2 or not rd["gaps"]:
                continue
            rows.append((rd["top_type"] == "attack", rd["attacks_top"], rd["gaps"], rd["legal"]))
    n = len(rows)
    print(f"1. multi-legal readings {n}")
    for a in (0.35, 1.0):
        hi = sum(1 for t, k, g, _ in rows if not t and k >= 1 and g[0] < a)
        hidden = sum(1 for t, k, g, legal in rows
                     if not t and k == 0 and len(g) == 7 and g[6] < a and legal > 8)
        print(f"   a = {a}: stance's upper bound {hi / n * OWN1_DECISIONS:.1f} per side-game; "
              f"uncounted candidates (no attack in top 8, eighth within a, more than 8 legal) "
              f"{hidden / n * OWN1_DECISIONS:.1f} per side-game")


def part2():
    fixed, games, budget, price = 14.95, 3 * 8 * 100_000, 85.0, 2.51e-5
    for add in (0.0, 3.59):
        line = (budget - fixed - add) / games
        print(f"2. additions ${add:.2f}: three rounds fit iff price x (1 + overhead) <= ${line:.3e}; "
              f"overhead room at the measured price {line / price - 1:.3f}")


def part3():
    batch = 2.51
    for m in (1.073, 1.122, 1.244):
        r = 0.602 / m
        fixed_batches = math.ceil(8 * r / 0.688 - 1e-12)
        saved_fixed = 8 * batch - fixed_batches * m * batch
        saved_resized = 8 * batch - 8 * r * m / 0.688 * batch
        print(f"3. m {m}: at r m = 0.602, r = {r:.4f}; fixed batches {fixed_batches} save ${saved_fixed:.2f}; "
              f"resized batches save ${saved_resized:.2f}")


def part4():
    target_z = 2 / (1 + 10 ** (78 / 400)) - 1          # z at -78 Elo
    lo, hi = 0.0, 5.0
    for _ in range(200):
        b = (lo + hi) / 2
        if 2 / (1 + math.exp(b)) - 1 > target_z:
            lo = b
        else:
            hi = b
    J = lambda t: 2 / (1 + math.exp(b * t)) - 1  # noqa: E731
    h = 1e-5
    jp = (J(0.5 + h) - J(0.5 - h)) / (2 * h)
    M = J(1) - J(0)
    print(f"4. logit-linear J: J(1) - J(0) = {M:+.5f} z; J'(1/2) = {jp:+.5f} z; "
          f"gap {(jp - M) * ELO_PER_Z:+.2f} Elo")


def part5():
    var_label = 0.688 * 1.0004 / 0.25
    for s in (4.8, 31.8):
        games = 2.0 / 2.51e-5
        se_all = math.sqrt(s * var_label / games) * ELO_PER_Z
        se_seat = se_all * math.sqrt(2)
        print(f"5. |S| {s}: at $2, F_H SE {se_all:.1f} Elo pooled, {se_seat:.1f} per seat; "
              f"a 30% scale error (-55 vs -78) against the M_1 line 2 sqrt(SE^2 + 1.79^2) = "
              f"{2 * math.sqrt(se_all ** 2 + 1.79 ** 2):.1f} Elo")


if __name__ == "__main__":
    part1()
    part2()
    part3()
    part4()
    part5()
