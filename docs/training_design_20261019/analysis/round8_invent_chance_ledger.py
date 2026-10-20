"""Round 8, invent stance: the dice's experiment against the coins', read on the luck walk.

Every coin game runs two randomized experiments with exact propensities: the coins (the design's
near-tie decisions) and the dice (every strike of every fight, hit with its chance-to-hit). This
script measures and derives what the second one holds for the first.

Part A (measured). Does the price of luck depend on the setup? On the walk's decisive games
(training/metrics/match_luck_20261019/rows.jsonl), z is match-centred and the seven luck columns
are fitted leave-one-match-out (round6_provenance_cells.py's procedure). The same columns are then
interacted with pre-game indicators (faction of A, faction of B, map, seat), still fitted
leave-one-match-out. A pre-game weight keeps every luck term centred, so these are legal control
variates. The kept ratio Var(residual)/Var(z) and its paired change against the plain columns are
reported, pooled and on own1 self-play. A drop is a measured lower bound on what conditioning the
price buys; per-game sums cannot condition on the fight's own context, which a token-priced
strike column does.

Part B (measured). The positive control. Each column's slope of z on S (side-A oriented luck) must
be positive: luck that favours A cannot lower A's result. Slope, SE and t, pooled and on own1
self-play, and the SE a 100,000-game batch would give. A result field flipped against the record
reads the negative slope at that t.

Part C (derived). The coins' own share of Var(z) at the design's floors against the dice's share,
the round-1 floor as a function of the kept ratio r, and the overfitting cost of a rich price
fitted on one batch.

Part D (measured, derived). On own1 self-play: Var(S_hp) against the summed law variances SS_hp
(equal when every fight's luck is centred), attacks a game, and the precision of a planted per-hp
price on a strike class, for sizing the chance-effect head's plant.

Part E (measured). The hp slope per match: whether the positive control's value depends on which
players are matched.

numpy and the standard library only.
"""
import collections
import json
import math
from pathlib import Path

import numpy as np

ROWS = Path(__file__).resolve().parents[3] / "training/metrics/match_luck_20261019/rows.jsonl"
COLUMNS = ("hp", "material", "material_nonterminal", "terminal", "value_50", "value_100", "value_200")
SELF_PLAY = ("gen_own1_selfplay", "own64_selfplay")
RNG = np.random.default_rng(20261020)

BAND = 76.1            # band decisions per side-game (current_design.md section 2, round 5)
P = 0.25               # coin probability
FLOOR_SD = 0.0016      # round-1 null bound at mu 0, r 0.688 (round8_provenance_spread_estimator.out)
FLOOR_SD_NEG = 0.0034  # at mu -0.0018 (round8_simplify_power.out section 1)
R_DESIGN = 0.688       # cell + columns, own1 self-play (round6_provenance_cells.out)
CELL_ALONE = 0.843     # cell alone (round8_simplify.md provenance, round6_simplify_pairs_raw.out)
GAMES_BATCH = 100_000


def load():
    rows = [json.loads(line) for line in open(ROWS)]
    rows = [r for r in rows if r.get("decisive")]
    z = np.array([float(r["z"]) for r in rows])
    match = np.array([r["match"] for r in rows])
    S = np.array([[r["S"][c] for c in COLUMNS] for r in rows])
    zc = z.copy()
    for m in set(match):
        zc[match == m] -= z[match == m].mean()
    return rows, zc, match, S


def one_hot(values):
    levels = sorted(set(values))
    index = {v: i for i, v in enumerate(levels)}
    out = np.zeros((len(values), len(levels)))
    for j, v in enumerate(values):
        out[j, index[v]] = 1.0
    return out[:, 1:]          # drop one level; the plain column carries it


def design(S, contexts):
    blocks = [S]
    for ctx in contexts:
        for c in range(S.shape[1]):
            blocks.append(S[:, [c]] * ctx)
    return np.hstack(blocks)


def cross_fit(X, zc, match):
    resid = np.zeros_like(zc)
    for m in set(match):
        test, train = match == m, match != m
        mu = X[train].mean(0)
        b = np.linalg.lstsq(X[train] - mu, zc[train], rcond=None)[0]
        resid[test] = zc[test] - (X[test] - mu) @ b
    return resid


def part_a(rows, zc, match, S):
    print("Part A. Kept ratio of match-centred z, columns fitted leave-one-match-out")
    fa = one_hot([r["faction_a"] for r in rows])
    fb = one_hot([r["faction_b"] for r in rows])
    mp = one_hot([r["scenario_id"] for r in rows])
    seat = one_hot([r["side_a"] for r in rows])
    models = collections.OrderedDict([
        ("plain columns (7)", []),
        ("x factions of A and B", [fa, fb]),
        ("x map", [mp]),
        ("x seat", [seat]),
        ("x factions, map, seat", [fa, fb, mp, seat]),
    ])
    var_all = float(np.var(zc))
    sp = np.isin(match, SELF_PLAY)
    var_sp = float(np.var(zc[sp]))
    base = None
    for name, ctx in models.items():
        X = design(S, ctx)
        resid = cross_fit(X, zc, match)
        kept = np.var(resid) / var_all
        kept_sp = np.var(resid[sp]) / var_sp
        line = f"  {name:28s} d {X.shape[1]:4d}: pooled {kept:.4f}, own1 self-play {kept_sp:.4f}"
        if base is None:
            base = resid
        else:
            diff = (resid ** 2 - base ** 2)
            se = diff.std(ddof=1) / math.sqrt(len(diff)) / var_all
            diff_sp = diff[sp]
            se_sp = diff_sp.std(ddof=1) / math.sqrt(len(diff_sp)) / var_sp
            line += (f"; change {diff.mean() / var_all:+.4f} +- {se:.4f}"
                     f" (self-play {diff_sp.mean() / var_sp:+.4f} +- {se_sp:.4f})")
        print(line)
    print(f"  games {len(zc)}, own1 self-play {int(sp.sum())}")


def part_b(zc, match, S):
    print("Part B. Positive control: slope of match-centred z on each side-A luck column")
    sp = np.isin(match, SELF_PLAY)
    for subset, name in ((np.ones_like(sp), "pooled"), (sp, "own1 self-play")):
        y = zc[subset]
        n = len(y)
        print(f"  {name}, {n} games")
        for c, col in enumerate(COLUMNS):
            x = S[subset, c] - S[subset, c].mean()
            b = float(x @ y / (x @ x))
            res = y - b * x
            se = math.sqrt(float(res @ res) / (n - 2) / float(x @ x))
            se_batch = se * math.sqrt(n / GAMES_BATCH)
            print(f"    {col:22s} slope {b:+.5f} +- {se:.5f} (t {b / se:+6.1f}); "
                  f"a 100,000-game batch: SE {se_batch:.5f}, a flipped result reads t {-b / se_batch:+.0f}")


def part_c():
    print("Part C. Derived")
    var_z = 1.0004
    for label, sd, mu in (("mu 0, floor", FLOOR_SD, 0.0), ("mu -0.0018, floor", FLOOR_SD_NEG, -0.0018)):
        share = 2 * BAND * P * (1 - P) * (sd ** 2 + mu ** 2) / var_z
        print(f"  coins' own share of Var(z) at {label} (sd {sd}): {share:.2e}; "
              f"dice's share (cell alone keeps {CELL_ALONE}, all of it chance at argmax): "
              f"{CELL_ALONE:.3f}, ratio {CELL_ALONE / share:,.0f}")
    print("  round-1 null bound (mu 0) against the label's kept ratio r: sd scales as sqrt(r)")
    for r in (0.688, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1):
        print(f"    r {r:.3f}: floor sd {FLOOR_SD * math.sqrt(r / R_DESIGN):.5f}, "
              f"worth {R_DESIGN / r:.2f} x the coin games")
    for d in (64, 772, 1536, 3072):
        for batches in (1, 3, 7):
            n = GAMES_BATCH * batches
            print(f"  rich price, d {d:5d}, fitted on {batches} batch(es): overfit adds about "
                  f"{R_DESIGN * d / n:.4f} to r")


def part_d(rows, zc, match, S):
    print("Part D. own1 self-play: centring check and the size of a planted per-hp price")
    sp = np.isin(match, SELF_PLAY)
    ss = np.array([r["SS"]["hp"] for r in rows])[sp]
    s_hp = S[sp, 0]
    attacks = np.array([r["counts"]["attacks"] for r in rows])[sp]
    print(f"  Var(S_hp) {s_hp.var():.0f} against mean SS_hp {ss.mean():.0f} (ratio {s_hp.var() / ss.mean():.3f}; "
          f"1 when every fight's luck is centred under its law); attacks per game {attacks.mean():.1f}")
    for share in (0.05, 0.1, 0.2):
        for kappa in (0.0005, 0.001):
            planted = kappa ** 2 * share * ss.mean()
            n = 8 * GAMES_BATCH
            se = math.sqrt(R_DESIGN / (n * share * ss.mean()))
            print(f"  plant kappa {kappa} per hp on a strike class holding {share:.2f} of the hp luck: "
                  f"adds {planted:.1e} of Var(z); per-hp SE over 8 batches {se:.1e} (kappa / SE {kappa / se:.0f})")


def part_e(zc, match, S):
    print("Part E. The hp slope match by match (does the positive control's value depend on the players?)")
    out = []
    for m in sorted(set(match)):
        sel = match == m
        y = zc[sel] - zc[sel].mean()
        x = S[sel, 0] - S[sel, 0].mean()
        b = float(x @ y / (x @ x))
        res = y - b * x
        se = math.sqrt(float(res @ res) / (len(y) - 2) / float(x @ x))
        out.append((b, se, m))
    out.sort()
    for b, se, m in out:
        print(f"  {m:28s} {b:+.5f} +- {se:.5f}")
    bs = np.array([o[0] for o in out])
    ses = np.array([o[1] for o in out])
    w = 1 / ses ** 2
    mu = float((w * bs).sum() / w.sum())
    chi2 = float(((bs - mu) ** 2 * w).sum())
    print(f"  weighted mean {mu:.5f}; chi2 {chi2:.1f} on {len(bs) - 1} df (no excess spread across the 23 player pairs "
          f"when chi2 is near its df); range {bs.min():+.5f} to {bs.max():+.5f}")


def main():
    rows, zc, match, S = load()
    part_a(rows, zc, match, S)
    part_b(zc, match, S)
    part_c()
    part_d(rows, zc, match, S)
    part_e(zc, match, S)


if __name__ == "__main__":
    main()
