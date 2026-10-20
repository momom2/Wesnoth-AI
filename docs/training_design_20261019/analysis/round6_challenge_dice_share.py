"""Round 6 (challenge): how much of a game's result the dice explain, against how much all the
coins of the current design can cause, and whether the dice's value depends on context.

Inputs (read-only):
- training/metrics/match_luck_20261019/rows.jsonl: 23 argmax matches, one row per game, side A's
  view; S holds each luck column's sum over the game's fights (tools/match_luck.py: L = Q(realised
  outcome) - sum_o p(o) Q(o), Q reading the two fighters only, with fixed hand-written weights).
- training/metrics/memory_own_round2_20261011/games_gen_own1_selfplay.tar.gz: own1 self-play, 800
  games; the strike tables recorded where the defender's counter weapon was a choice (a subsample
  of fights, not a random one: stated as such).

Parts:
1. Fixed columns: leave-one-match-out share of Var(z) the seven columns explain (reproduces the
   provenance stance's 0.847 kept), pooled and in own1 self-play.
2. Contextual columns: each column's weight varies additively with side A's faction, side B's
   faction and the map (pre-game quantities, so every correction stays mean-zero), ridge,
   leave-one-match-out. Does context add explained variance out of sample?
3. The current design's coins: the variance of z all coins of a game can cause, to first order,
   V_coin = sum over coins of p(1-p) Delta^2 = n_band * p(1-p) * (v^2 + mu^2), against the dice's
   explained share from part 1. Inputs: 2 x 76.1 band decisions a game, p = 0.25
   (current_design.md section 3.1); v and mu are unknown, so the ratio is printed over the range of
   the current design's own floors (sd 0.002 to 0.0038) and above.
4. Treatment variance of the dice: on the strike-table subsample, per fight, P(defender dies) and
   P(attacker dies) and their Bernoulli variances, against a coin's p(1-p) = 0.1875.
numpy and the standard library only.
"""
import gzip
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ROWS = ROOT / "training/metrics/match_luck_20261019/rows.jsonl"
TARBALL = ROOT / "training/metrics/memory_own_round2_20261011/games_gen_own1_selfplay.tar.gz"
COLUMNS = ("hp", "material", "material_nonterminal", "terminal", "value_50", "value_100", "value_200")
OWN1_SELFPLAY = ("gen_own1_selfplay", "own64_selfplay")
BAND_PER_GAME = 2 * 76.1
P_COIN = 0.25


def load_rows():
    return [r for r in (json.loads(line) for line in open(ROWS)) if r.get("decisive")]


def ridge(X, y, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def context_design(rows, S, maps, factions):
    """Columns, and each column times one-hots of faction A, faction B and map."""
    n, k = S.shape
    blocks = [S]
    for key, index in (("faction_a", factions), ("faction_b", factions), ("scenario_id", maps)):
        oh = np.zeros((n, len(index)))
        for i, r in enumerate(rows):
            oh[i, index[r[key]]] = 1.0
        blocks.append((S[:, :, None] * oh[:, None, :]).reshape(n, k * len(index)))
    return np.hstack(blocks)


def cross_fitted(rows, z, S, X_ctx, match, names):
    zc = z.copy()
    for m in names:
        zc[match == m] -= z[match == m].mean()
    sd = S.std(0)
    Ss = S / sd
    Xs = X_ctx / np.where(X_ctx.std(0) > 0, X_ctx.std(0), 1.0)
    out_fixed, out_ctx = np.zeros_like(z), {lam: np.zeros_like(z) for lam in (1.0, 10.0, 100.0)}
    for m in names:
        test, train = match == m, match != m
        b = np.linalg.lstsq(Ss[train] - Ss[train].mean(0), zc[train], rcond=None)[0]
        out_fixed[test] = z[test] - Ss[test] @ b
        for lam in out_ctx:
            mu = Xs[train].mean(0)
            c = ridge(Xs[train] - mu, zc[train], lam)
            out_ctx[lam][test] = z[test] - Xs[test] @ c
    return out_fixed, out_ctx


def pooled(g, z, match, names, subset=None):
    num = den = 0.0
    for m in names:
        if subset and m not in subset:
            continue
        sel = match == m
        num += np.var(g[sel]) * sel.sum()
        den += np.var(z[sel]) * sel.sum()
    return num / den


def strike_tables():
    p_def, p_att = [], []
    with tarfile.open(TARBALL) as t:
        for m in t.getmembers():
            if not m.name.endswith(".game.jsonl.gz"):
                continue
            rec = json.loads(gzip.decompress(t.extractfile(m).read()).decode().splitlines()[0])
            for entry in rec.get("outcomes", {}).values():
                cw = entry.get("counter_weapon")
                if not cw:
                    continue
                table = cw["tables"].get(str(cw["chosen"]))
                if not table:
                    continue
                tot = sum(row[-1] for row in table)
                p_def.append(sum(row[-1] for row in table if row[1] <= 0) / tot)
                p_att.append(sum(row[-1] for row in table if row[0] <= 0) / tot)
    return np.array(p_def), np.array(p_att)


def main():
    rows = load_rows()
    maps = {m: i for i, m in enumerate(sorted({r["scenario_id"] for r in rows}))}
    factions = {f: i for i, f in enumerate(sorted({r["faction_a"] for r in rows} | {r["faction_b"] for r in rows}))}
    match = np.array([r["match"] for r in rows])
    names = sorted(set(match))
    z = np.array([float(r["z"]) for r in rows])
    S = np.array([[r["S"][c] for c in COLUMNS] for r in rows])
    X_ctx = context_design(rows, S, maps, factions)
    out_fixed, out_ctx = cross_fitted(rows, z, S, X_ctx, match, names)

    print(f"1-2. decisive games {len(z)}, matches {len(names)}; context features {X_ctx.shape[1]}")
    for label, subset in (("pooled, 23 matches", None), ("own1 self-play (2 matches)", OWN1_SELFPLAY)):
        k_fixed = pooled(out_fixed, z, match, names, subset)
        print(f"  {label}: fixed columns keep {k_fixed:.3f} (explain {1 - k_fixed:.3f})")
        for lam, g in out_ctx.items():
            k = pooled(g, z, match, names, subset)
            print(f"    contextual columns, ridge {lam:>5}: keep {k:.3f} (explain {1 - k:.3f})")
    rng = np.random.default_rng(20261020)
    diffs = []
    best_lam = min(out_ctx, key=lambda lam: pooled(out_ctx[lam], z, match, names))
    for _ in range(200):
        idx = rng.integers(0, len(z), len(z))
        a = np.var(out_fixed[idx]) - np.var(out_ctx[best_lam][idx])
        diffs.append(a / np.var(z[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  context's added share (ridge {best_lam}), game bootstrap 95%: {np.mean(diffs):+.4f} ({lo:+.4f} to {hi:+.4f})")

    explained = 1 - pooled(out_fixed, z, match, names, OWN1_SELFPLAY)
    print(f"\n3. variance of z the coins can cause, n = {BAND_PER_GAME:.1f} band coins a game, p = {P_COIN}")
    for v in (0.002, 0.0038, 0.005, 0.01, 0.02):
        for mu in (0.0, -0.0018):
            v_coin = BAND_PER_GAME * P_COIN * (1 - P_COIN) * (v * v + mu * mu)
            print(f"  sd {v:.4f}, mu {mu:+.4f}: V_coin {v_coin:.2e} = {v_coin / 1.0004:.5f} of Var(z); "
                  f"fixed dice columns explain {explained:.3f}, ratio {explained / v_coin:,.0f}")

    p_def, p_att = strike_tables()
    print(f"\n4. strike-table subsample (counter weapon a choice): fights {len(p_def)}")
    for name, p in (("defender dies", p_def), ("attacker dies", p_att)):
        var = p * (1 - p)
        print(f"  P({name}): mean {p.mean():.3f}; share of fights with 0.05 < P < 0.95: {np.mean((p > 0.05) & (p < 0.95)):.3f}; "
              f"mean Bernoulli variance {var.mean():.4f} (a band coin: {P_COIN * (1 - P_COIN):.4f})")


if __name__ == "__main__":
    sys.exit(main())
