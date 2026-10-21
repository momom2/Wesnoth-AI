"""Round 9, critique of the challenge stance: checks of the factorial regimen's numbers.

A. The Sylvester Hadamard design's column 1 against run_elo_batch's side
   alternation (side_a = 1 for even slot index, 2 for odd; tools/run_elo_batch.py
   slot_for). Reads batch 1's rows: the first-mover effect on raw z, and the
   share of it a frozen cell baseline leaves (cell = scenario, side of A, factions,
   as round9_simplify_residual.py; fitted on b01-b03, read on b00, b04-b06).
B. The hook's opponent-side counts: is "moves lost" a count of changed
   decisions or a downstream effect? Reads round9_challenge_hook_flips.out's
   figures (no recomputation needed beyond the arithmetic shown).
C. Winner's curse in the claim's prediction: the sum of adopted confirmation
   readings, adoption iff read - 2 SE > 0, SE 1.79 Elo. Expected bias per
   adopted step for true steps of 2-10 Elo, and over three rounds.
D. Confirmation power at the challenge's 2.8-Elo proposal floor.

numpy and the standard library only; a few seconds.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REC = ROOT / "training" / "metrics" / "near_tie_coins_20261020"
ELO_PER_Z = 347.4


def phi(x):
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def read_rows():
    rows = []
    for b in ("b00", "b01", "b02", "b03", "b04", "b05", "b06"):
        with open(REC / f"rows_{b}.jsonl", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("decisive"):
                    rows.append((b, int(r["side_a"]), int(r["z"]), r["scenario_id"],
                                 r["faction_a"], r["faction_b"]))
    return rows


def section_a():
    rows = read_rows()
    side = np.array([r[1] for r in rows])
    z = np.array([r[2] for r in rows], dtype=float)
    n = len(z)
    # z is side A's result; side_a = 1 means A moves first.
    s = np.where(side == 1, 1.0, -1.0)
    m = float((z * s).mean())
    se = float((z * s).std(ddof=1) / math.sqrt(n))
    print("A. first-mover effect in batch 1 (z of side A times +1 if A is side 1, -1 if side 2)")
    print(f"   decisive games {n}; side_a = 1 in {int((side == 1).sum())}")
    print(f"   mean {m:+.4f} +- {se:.4f} z = {m * ELO_PER_Z:+.1f} +- {se * ELO_PER_Z:.1f} Elo")
    print("   Sylvester Hadamard column 1 is (-1)^g; run_elo_batch's side sign is (-1)^i (slot_for):")
    print("   with row g = i mod 2^m, block 1's raw-z main effect carries this whole term.")
    # residual after a frozen cell mean (scenario, side, factions), fitted on b01-b03
    fit = [k for k, r in enumerate(rows) if r[0] in ("b01", "b02", "b03")]
    read = [k for k, r in enumerate(rows) if r[0] not in ("b01", "b02", "b03")]
    sums = defaultdict(lambda: [0.0, 0])
    for k in fit:
        key = (rows[k][3], rows[k][1], rows[k][4], rows[k][5])
        sums[key][0] += z[k]
        sums[key][1] += 1
    gm = float(z[fit].mean())
    kshr = 15.0  # shrinkage constant of round9_simplify_residual.out (14.2 / 16.2 games)

    def b_of(k):
        key = (rows[k][3], rows[k][1], rows[k][4], rows[k][5])
        t, c = sums.get(key, (0.0, 0))
        return gm + (t - c * gm) / (c + kshr) if c else gm

    rr = np.array([z[k] - b_of(k) for k in read])
    sr = s[read]
    m2 = float((rr * sr).mean())
    se2 = float((rr * sr).std(ddof=1) / math.sqrt(len(rr)))
    m_raw = float((z[read] * sr).mean())
    print(f"   held-out games {len(read)}: raw {m_raw * ELO_PER_Z:+.1f} Elo; after a cell mean shrunk to the grand"
          f" mean, fitted on b01-b03: {m2 * ELO_PER_Z:+.1f} +- {se2 * ELO_PER_Z:.1f} Elo")
    # the stance's baseline puts side in an additive part (map x side), fitted on many games: the side
    # term is then removed up to its own estimation error and its drift between networks.
    ms = defaultdict(lambda: [0.0, 0])
    for k in fit:
        ms[(rows[k][3], rows[k][1])][0] += z[k]
        ms[(rows[k][3], rows[k][1])][1] += 1
    ra = np.array([z[k] - ms[(rows[k][3], rows[k][1])][0] / max(ms[(rows[k][3], rows[k][1])][1], 1) for k in read])
    m3 = float((ra * sr).mean())
    se3 = float((ra * sr).std(ddof=1) / math.sqrt(len(ra)))
    print(f"   after a map x side mean fitted on b01-b03 (the additive part of the stance's b): {m3 * ELO_PER_Z:+.1f}"
          f" +- {se3 * ELO_PER_Z:.1f} Elo")
    # drift of the first-mover effect across the luck walk's 23 matches (other networks and pairs)
    walk = REC.parent / "match_luck_20261019" / "rows.jsonl"
    per = defaultdict(list)
    with open(walk, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("decisive"):
                per[r["match"]].append(int(r["z"]) * (1 if int(r["side_a"]) == 1 else -1))
    est = []
    for name, v in sorted(per.items()):
        v = np.array(v, dtype=float)
        est.append((name, v.mean(), v.std(ddof=1) / math.sqrt(len(v)), len(v)))
    w = np.array([1 / e[2] ** 2 for e in est])
    mu = float((w * np.array([e[1] for e in est])).sum() / w.sum())
    q = float((w * (np.array([e[1] for e in est]) - mu) ** 2).sum())
    print(f"   luck walk, {len(est)} matches: pooled first-mover effect {mu * ELO_PER_Z:+.1f} Elo;"
          f" Q {q:.1f} on {len(est) - 1} df")
    for name, m_, se_, n_ in est:
        if name.startswith(("r1_", "r3_", "own64_selfplay", "gen_own1")):
            print(f"     {name:28s} {n_:5d} games {m_ * ELO_PER_Z:+6.1f} +- {se_ * ELO_PER_Z:.1f} Elo")


def section_b():
    print("\nB. hook counts (round9_challenge_hook_flips.out): hooked seat moves -9.379 +- 1.798, opponent +9.874 +- 1.867;")
    print("   recruits -0.429 / +0.429; attacks +1.046 / -0.193; side-turns -0.102 / +0.016")
    tot, tot_se = -9.379 + 9.874, math.hypot(1.798, 1.867)
    print(f"   moves, both seats summed: {tot:+.2f} +- {tot_se:.2f} per game: the hooked seat's lost moves reappear"
          " on the opponent, the signature of a material shift, not of a reordering inside the hooked seat's turns")


def section_c():
    se = 1.79
    line = 2 * se
    print("\nC. winner's curse in the claim's prediction (adopt iff read > 2 SE, SE 1.79 Elo)")
    for d in (2.0, 2.8, 4.0, 6.0, 10.0, 23.0):
        a = (line - d) / se
        p = 1 - Phi(a)
        bias = se * phi(a) / p
        print(f"   true step {d:5.1f} Elo: adoption {p:.3f}; E[read | adopted] - true = {bias:+.2f} Elo")
    a = (line - 2.8) / se
    b = se * phi(a) / (1 - Phi(a))
    ms = 2 * math.sqrt(4.4 ** 2 + 3 * se ** 2 + 3.07 ** 2)
    print(f"   three adopted steps near the 2.8-Elo floor: prediction biased by {3 * b:+.1f} Elo against the claim's"
          f" 2-SE miss line {ms:.1f}")
    # simulated miss rate of the claim at three adopted 2.8-Elo steps
    rng = np.random.default_rng(1)
    n = 200_000
    reads = rng.normal(2.8, se, (n, 3))
    keep = (reads > line).all(1)
    pred = 22.7 + reads[keep].sum(1)
    truth = 22.7 + 3 * 2.8
    claim = rng.normal(truth, 3.07, keep.sum())
    base_noise = rng.normal(0, 4.4, keep.sum())  # the 22.7's own error enters the prediction
    pred = pred + base_noise
    miss = np.abs(claim - pred) > ms
    print(f"   simulated: claim outside its miss line in {miss.mean():.3f} of trials (nominal 0.046)")


def section_d():
    se = 1.79
    print("\nD. confirmation power (adopt iff read - 2 SE > 0, SE 1.79 Elo)")
    for d in (2.8, 3.6, 5.0, 7.2):
        print(f"   true {d:.1f} Elo: power {1 - Phi((2 * se - d) / se):.3f}")


def section_e():
    """A switch block acts only at band decisions (G < w = 0.35), where coins run on both seats in R1.
    With the block on, the seat plays the block's choice unless its coin fires (P 0.25); with it off, it
    plays that choice only when its coin fires (P 0.25). Switch probability 0.75 against 0.25."""
    p = 0.25
    dil = (1 - p) - p
    k = 4.01 / (2 * 0.699)
    print(f"\nE. coins on both seats dilute a switch block's play effect by (1 - P) - P = {dil:.2f}")
    print(f"   SE on the uncoined effect at 100,000 coined games: {0.918 / dil:.2f} Elo (stated 0.92)")
    print(f"   coherence ratio, coined factorial: n V_T / (2 sigma_R^2) x {dil:.2f}^2 = {k * dil ** 2:.3f} n")
    for name, n in (("recruit rule R", 2.21), ("Loyalists", 0.602 * 0.42), ("Drakes", 0.602 * 0.90)):
        print(f"     {name:15s} n {n:.2f}: ratio {k * dil ** 2 * n:.2f} (uncoined {k * n:.2f})")
    print(f"   split-half prediction (coined) against the uncoined confirmation: expected ratio {1 / dil:.1f}")


def main():
    section_a()
    section_b()
    section_c()
    section_d()
    section_e()


if __name__ == "__main__":
    main()
