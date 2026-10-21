"""Round 9 synthesis: the numbers the new current design adds, from existing records.

Sections (cited as round9_synthesis_checks.out section n):
1. The recruit rule R's ridge chosen by the design's progressive loss over batch 1's seven matches
   (in game order), not by its welfare; the cross-fitted first-order welfare F at that ridge, its
   per-faction split, and the rule's switch rate. Reads the per-cell fold sums that
   round9_provenance_cells.py wrote (partition rr_units; labels T: batch 1's luck columns, b = 0).
2. L's lines (one rule on one seat, uncoined, raw z) at 12,800 and 37,800 decisive games; the bound a
   null puts on the transfer ratio; the per-faction SE of a stratified reading against a factorial block's.
3. H1, the head after S1's first batch, read against T: SE of the difference, false-fire rate,
   the first student's forecast SE.
4. The batch-1 feature replay the critiques proposed: its cost range and what it could resolve.
5. The winner's curse in a claim predicted from adoption readings: the truncated-normal conditional
   MLE (Zhong & Prentice 2008) and per-round checks, simulated.
6. The path's budget at the measured prices.

numpy and the standard library only; no network, no simulator; well under a minute.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MATCHES = ("b00", "b01", "b02", "b03", "b04", "b05", "b06")
FOLDS = 20
ELO = 800 / math.log(10)

# Measured inputs (scripts that read them named beside each).
POOL_RATE = 22_209      # pool games an hour, round9_simplify_labels.out [prices]
ALLIN_RATE = 20_254     # coin games an hour all-in, same
TWO_RATE = 6_521        # two-network decisive games an hour, same
PRICE = 0.577           # $ an hour, the measured host
COIN_GAME = 2.849e-5    # $ a coin game all-in, same
SETUP_H = 7 / 60 + 0.25 # 7 min measured bring-up (walls.txt) + 15 min staging [C], round9_simplify_budget.py
LEFT = 82.34            # $85 - $2.66 (seed section 7), assumes the window counted against $85
PERSIDE_VAR_Q50 = 158.1 # per-side-game variance of a rule's summed labels at switched share 0.5,
PERSIDE_VAR_Q25 = 77.0  # and at 0.25, frozen baselines (round9_simplify_labels.out)
DECISIONS_PER_GAME = 2 * (53.39 + 126.02 + 22.89 + 19.67)  # round9_challenge_hook_flips.out, unhooked level
OWN_TARGETS_RATE = 1_402_979 / (59 * 60)  # positions a second, two forwards each,
                                          # docs/memory_own_trajectories_prereg_20261011.md "Measured"


def units_of(cell):
    fac, pair = cell.split("|")
    u0, u1 = pair.split(">")
    return f"{fac}|{u0}", f"{fac}|{u1}"


def load():
    data = [json.loads((HERE / f"round9_provenance_cells_{m}.json").read_text()) for m in MATCHES]
    cells = sorted({c for d in data for c in d["partitions"]["rr_units"]["cells"]} - {"rest"})
    at = {c: i for i, c in enumerate(cells)}
    per_match = []
    for d in data:
        p = d["partitions"]["rr_units"]
        acc = {k: np.zeros((len(cells), FOLDS)) for k in ("n", "s_T", "ss_T")}
        for j, c in enumerate(p["cells"]):
            if c == "rest":
                continue
            for k in acc:
                acc[k][at[c]] += np.asarray(p[k][j])
        per_match.append(acc)
    side_games = [d["side_games"] for d in data]
    return cells, per_match, side_games


def design(cells):
    names = sorted({u for c in cells for u in units_of(c)})
    at = {u: i for i, u in enumerate(names)}
    X = np.zeros((len(cells), len(names)))
    for i, c in enumerate(cells):
        u0, u1 = units_of(c)
        X[i, at[u1]] += 1.0
        X[i, at[u0]] -= 1.0
    return X, names


def fit_pred(X, s, n, lam):
    """Ridge of the labels on e(a1) - e(a0), from per-cell sums: (X'NX + lam I)^-1 X's."""
    A = X.T @ (X * n[:, None]) + lam * np.eye(X.shape[1])
    return X @ np.linalg.solve(A, X.T @ s)


def section1(cells, per_match, side_games):
    print("1. R's ridge by the progressive loss (matches in game order b00..b06; loss of student j-1 on match j,")
    print("   sum over j of sum_cells [-2 S_cj p_c + n_cj p_c^2], labels T; lower is better)")
    X, names = design(cells)
    S = [m["s_T"].sum(1) for m in per_match]
    N = [m["n"].sum(1) for m in per_match]
    grid = [50.0, 200.0, 500.0, 1_000.0, 2_000.0, 5_000.0, 10_000.0, 20_000.0, 50_000.0, 200_000.0]
    losses = {}
    for lam in grid:
        loss = 0.0
        for j in range(1, len(MATCHES)):
            p = fit_pred(X, sum(S[:j]), sum(N[:j]), lam)
            loss += float(np.sum(-2 * S[j] * p + N[j] * p * p))
        losses[lam] = loss
    base = losses[grid[-1]]
    for lam in grid:
        print(f"   ridge {lam:>9.0f} labels: progressive loss {losses[lam]:+.3f} ({losses[lam] - base:+.3f} against the largest ridge)")
    lam_star = min(losses, key=losses.get)
    print(f"   chosen ridge: {lam_star:.0f} labels")

    s_all = sum(m["s_T"] for m in per_match)
    ss_all = sum(m["ss_T"] for m in per_match)
    n_all = sum(m["n"] for m in per_match)
    sg = sum(side_games)
    tot = var = switched = 0.0
    fac_tot, fac_var, fac_sw = {}, {}, {}
    facs = [c.split("|")[0] for c in cells]
    for f in range(FOLDS):
        so, no = s_all.sum(1) - s_all[:, f], n_all.sum(1) - n_all[:, f]
        choose = (no > 0) & (fit_pred(X, so, no, lam_star) > 0)
        tot += s_all[choose, f].sum()
        var += ss_all[choose, f].sum()
        switched += n_all[choose, f].sum()
        for i in np.nonzero(choose)[0]:
            fac_tot[facs[i]] = fac_tot.get(facs[i], 0.0) + s_all[i, f]
            fac_var[facs[i]] = fac_var.get(facs[i], 0.0) + ss_all[i, f]
            fac_sw[facs[i]] = fac_sw.get(facs[i], 0.0) + n_all[i, f]
    F, F_se = tot / sg * ELO, math.sqrt(var) / sg * ELO
    print(f"   F at the chosen ridge, cross-fitted over 20 folds: {F:+.1f} +- {F_se:.1f} Elo per side-game; "
          f"switches {switched / sg:.2f} per side-game ({switched / n_all.sum():.3f} of these decisions)")
    print("   per mover faction (contribution, Elo per side-game of all factions; switches per side-game):")
    for fac in sorted(fac_tot):
        print(f"     {fac:18s} {fac_tot[fac] / sg * ELO:+.1f} +- {math.sqrt(fac_var[fac]) / sg * ELO:.1f}; "
              f"{fac_sw[fac] / sg:.2f}")
    p_full = fit_pred(X, s_all.sum(1), n_all.sum(1), lam_star)
    sw_full = (n_all.sum(1) * (p_full > 0)).sum() / sg
    print(f"   the rule fitted on all 60,000 games switches {sw_full:.2f} decisions per side-game in batch 1's states "
          f"({int((p_full > 0).sum())} of {len(cells)} cells)")
    return F, F_se


def section2(F, F_se):
    print("\n2. L: R on one seat (a per-game coin keyed by seed, independent of side), uncoined, raw z")
    for n in (12_800, 37_800):
        se = ELO / math.sqrt(n)
        miss = 2 * math.sqrt(F_se ** 2 + se ** 2)
        null_line = 2 * se
        rho_bound = (null_line + 2 * se) / F
        p03 = 0.5 * math.erfc(-((0.3 * F - null_line) / math.sqrt(se ** 2 + (0.3 * F_se) ** 2)) / math.sqrt(2))
        p05 = 0.5 * math.erfc(-((0.5 * F - null_line) / math.sqrt(se ** 2 + (0.5 * F_se) ** 2)) / math.sqrt(2))
        cost = n / POOL_RATE * PRICE
        print(f"   {n:,} games (${cost:.2f}): SE {se:.2f}; miss half-width {miss:.1f}; null line {null_line:.2f}; "
              f"a null bounds rho by {rho_bound:.2f}; P(lower bound > 0) {p03:.2f} at rho 0.3, {p05:.2f} at rho 0.5")
    se = ELO / math.sqrt(37_800)
    strat = se * math.sqrt(6) / 6
    print(f"   per-faction contribution, stratified on the hooked seat's faction (1/6 of games each): SE {strat:.2f} Elo "
          f"(raw z) against a Hadamard block's {se:.2f} raw / 1.49 with variates at the same 37,800 games "
          f"(round9_challenge_factorial.out section 2)")


def section3():
    print("\n3. H1: pi_0 + student 1 against pi_0, 37,800 games, read against T (37,800 games)")
    se = ELO / math.sqrt(37_800)
    diff = math.sqrt(2) * se
    print(f"   SE of J_H1 - J_T: {diff:.2f} Elo; a fire at -2 SE when the head adds nothing beyond R and costs nothing: "
          f"P {0.5 * math.erfc(2 / math.sqrt(2)):.3f}")
    for q, v in ((0.25, PERSIDE_VAR_Q25), (0.5, PERSIDE_VAR_Q50)):
        print(f"   student 1's cross-fitted forecast on one 100,000-game batch, switched share {q}: SE "
              f"{math.sqrt(v / 200_000) * ELO:.1f} Elo")


def section4():
    print("\n4. Replaying batch 1's 60,000 games through own1 to dump features")
    pos = 60_000 * DECISIONS_PER_GAME
    gen_rate = POOL_RATE * DECISIONS_PER_GAME / 3600
    for name, rate in (("at generation's rate (the same forwards and steps)", gen_rate),
                       ("at own_targets' measured rate, one forward (2x its two-forward rate)", 2 * OWN_TARGETS_RATE),
                       ("at own_targets' measured rate as built (two forwards)", OWN_TARGETS_RATE)):
        h = pos / rate / 3600
        print(f"   {name}: {rate:,.0f} positions a second, {h:.1f} h, ${h * PRICE:.2f}")
    print(f"   ({DECISIONS_PER_GAME:.1f} decisions a game, {pos / 1e6:.1f} M positions)")
    sg = 120_000
    print(f"   cross-fitted welfare of a head on 60,000 games, switched share 0.5: SE "
          f"{math.sqrt(PERSIDE_VAR_Q50 / sg) * ELO:.1f} Elo (beyond-recruit cell rules read SE 11.2-12.4, "
          f"round9_provenance_unit_cells.out)")
    v_se_800k = 1.27e-6 * 4.01 / 3.67
    v_se_60k = v_se_800k * 800_000 / 60_000
    v_recruit = 0.0552 * 1.592e-3
    print(f"   spread estimator: SE {v_se_60k:.2e} at 60,000 games against {v_se_800k:.2e} at 800,000; "
          f"recruit cells' band-wide variance {v_recruit:.2e} (0.0552 x V_rr), z {v_recruit / v_se_60k:.1f} at 60,000")


def section5(rng):
    print("\n5. Winner's curse in the claim's prediction; the truncated-normal conditional MLE; per-round checks")
    se = ELO / math.sqrt(37_800)
    c = 2 * se
    grid = np.linspace(-20, 40, 6001)
    tails = 0.5 * np.vectorize(math.erfc)(((c - grid) / se) / math.sqrt(2))

    def cond_mle(x):
        ll = -0.5 * ((x - grid) / se) ** 2 - np.log(np.maximum(tails, 1e-300))
        return grid[int(np.argmax(ll))]

    for true in (2.8, 4.0, 6.0, 10.0):
        reads = true + se * rng.standard_normal(200_000)
        kept = reads[reads > c]
        corr = np.array([cond_mle(x) for x in kept[:4_000]])
        print(f"   true step {true:4.1f} Elo: adopted {len(kept) / len(reads):.3f}; mean read - true {kept.mean() - true:+.2f}; "
              f"conditional MLE - true {corr.mean() - true:+.2f} (sd {corr.std():.2f}; the read's sd {kept.std():.2f})")
    print("   per-round check: D = claim_k - claim_(k-1) (fresh two-network matches) against the adoption")
    print("   reading A (SE 1.79, kept iff A > 2 SE); miss iff |D - A| > 2 sqrt(SE_D^2 + 1.79^2)")
    for label, se_prev in (("claim_(k-1) a 12,800-game claim", 3.07), ("claim_0 = S0(b)'s 22.7 +- 4.35", 4.35)):
        se_d = math.sqrt(se_prev ** 2 + 3.07 ** 2)
        line = 2 * math.sqrt(se_d ** 2 + se ** 2)
        rates = []
        for true in (2.8, 4.0, 6.0, 10.0, 21.3):
            a = true + se * rng.standard_normal(400_000)
            d = true + se_d * rng.standard_normal(400_000)
            keep = a > c
            rates.append(f"{true}: {np.mean(np.abs(d[keep] - a[keep]) > line):.3f}")
        print(f"   {label}: SE of D {se_d:.2f}, line {line:.1f} Elo; false miss by true step " + ", ".join(rates))
    trials = 20_000
    reads = 2.8 + se * rng.standard_normal((trials * 40, 3))
    reads = reads[np.all(reads > c, axis=1)][:trials]
    base = 22.7 + 4.35 * rng.standard_normal(len(reads))
    claim = 22.7 + 3 * 2.8 + 3.07 * rng.standard_normal(len(reads))
    line = 2 * math.sqrt(4.35 ** 2 + 3 * se ** 2 + 3.07 ** 2)
    print(f"   cumulative prediction 22.7 + sum of three adoption readings at true 2.8 Elo each: bias "
          f"{np.mean(base + reads.sum(1) - claim):+.2f} Elo, outside its line ({line:.1f}) "
          f"{np.mean(np.abs(claim - base - reads.sum(1)) > line):.3f}")


def path_total(o, k=1.0, with_c=True):
    """The whole path: box A (L, T), C, round 1 (smoke, 8 batches, H1, adoption, claim), E reserve,
    round 2 (8 batches, adoption, claim), E reserve, S4 (distillation, N against pi_k), round 3 (8 batches,
    adoption against pi_k and claim, both two networks), E reserve (two networks). k scales the coin game's price."""
    pool = lambda n: n / POOL_RATE * PRICE
    two = lambda n: n / TWO_RATE * PRICE
    coin = lambda n: n / ALLIN_RATE * PRICE * (1 + o) * k
    setup = SETUP_H * PRICE
    box_a = 2 * pool(37_800) + setup
    c_stage = two(12_800) if with_c else 0.0
    smoke = coin(2_000) + pool(1_000)
    r1 = smoke + coin(800_000) + 2 * pool(37_800) + two(12_800) + setup
    r2 = coin(800_000) + pool(37_800) + two(12_800) + setup
    s4 = 4.8 * PRICE + two(12_800)
    r3 = coin(800_000) + 2 * two(12_800) + setup
    parts = dict(box_a=box_a, c=c_stage, r1=r1, e1=pool(37_800), r2=r2, e2=pool(37_800), s4=s4, r3=r3,
                 e3=two(12_800))
    return sum(parts.values()), parts


def solve(fn, lo, hi):
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fn(mid) else (lo, mid)
    return lo


def section6():
    print("\n6. The path at $0.577 an hour (rates of round9_simplify_labels.out), by the dump's overhead o")
    for o in (0.0, 0.05, 0.10, 0.20):
        total, p = path_total(o)
        upto2 = p["box_a"] + p["c"] + p["r1"] + p["e1"] + p["r2"] + p["e2"]
        null1 = p["box_a"] + p["r1"] - 12_800 / TWO_RATE * PRICE   # no claim after a null adoption
        print(f"   o {o:.2f}: box A ${p['box_a']:.2f}; C ${p['c']:.2f}; round 1 ${p['r1']:.2f}; round 2 ${p['r2']:.2f}; "
              f"S4 ${p['s4']:.2f}; round 3 ${p['r3']:.2f}; through round 2 with E reserves ${upto2:.2f} "
              f"(left ${LEFT - upto2:.2f}); whole path ${total:.2f}; a null round 1 after box A ${null1:.2f}")
    for left in (LEFT, 85.0):
        for with_c in (True, False):
            k = solve(lambda k: path_total(0.0, k, with_c)[0] <= left, 0.3, 1.5)
            print(f"   with ${left:.2f} left{', with C' if with_c else ', no C'}: the whole path fits at o 0 iff a host's "
                  f"all-in cost per coin game is <= {k:.3f} x the measured (${COIN_GAME * k:.3e})")
    o2 = solve(lambda o: (lambda p: p["box_a"] + p["c"] + p["r1"] + p["e1"] + p["r2"] + p["e2"])(path_total(o)[1]) <= LEFT,
               0.0, 3.0)
    print(f"   two rounds (with box A, C and E reserves) fit iff o <= {o2:.2f} on the measured host")
    short = path_total(0.0)[0] - LEFT
    print(f"   the whole path's shortfall at o 0: ${short:.2f}; twins' net saving over three rounds "
          f"(round9_invent_critique_checks.out section 3): $1.69 at r 0.95, $4.79 at r 0.90, $10.99 at r 0.80")


def main():
    rng = np.random.default_rng(20261021)
    cells, per_match, side_games = load()
    F, F_se = section1(cells, per_match, side_games)
    section2(F, F_se)
    section3()
    section4()
    section5(rng)
    section6()


if __name__ == "__main__":
    main()
