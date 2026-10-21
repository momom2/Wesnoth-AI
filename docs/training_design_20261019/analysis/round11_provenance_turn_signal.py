"""Round 11 (provenance stance): what a temperature-1 turn is worth against the argmax turn, and how
much the turn-start position explains, read from the 2026-09-25 validation playouts.

Source: training/metrics/turn_value_20260925/bench_truth.npz. 199 human positions (side-2 turn starts
of human games, game turn 7 or later); at each, candidate turns by tools/turn_gap.py: slot 0 the base
turn (obs8 at raw:t0+eo-1.5), slots 1-3 turns sampled from obs8 at temperature 1, slot 4 a continue
edit; 28 playouts per candidate to the game's end, both sides obs8 at temperature 0.5 (cap scored 0);
the playouts' two luck columns; turn_luck, the candidate turn's own two luck columns (both 0: the turn
made no attack, so its own dice are absent; recruit traits remain).

Reads, with a position bootstrap (400 resamples, seed 20261021):
  A. within-position variance of the true value of a temperature-1 turn (slots 1-3 only, attack-free),
     method of moments: var(candidate means) - pooled within-candidate variance / 28. Along a turn the
     advantages of its decisions are martingale differences, so this variance is the sum over the
     turn's decisions of E[Var_pi(A_t)] under the playout continuation: the tabular signal of one
     side-turn, I_turn;
  B. mean of (temperature-1 turn - base turn) where both are attack-free: the per-turn cost of
     sampling the turn instead of the argmax turn (the base turn is obs8's own decode);
  C. the share of a playout result's variance the turn-start position explains: level / (level +
     between-candidate spread + within-candidate noise), raw and luck-corrected (luck slope fitted
     in-sample on the playouts' two columns, as round5_provenance_playout_noise.py does).
numpy and the standard library only.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
NPZ = ROOT / "training/metrics/turn_value_20260925/bench_truth.npz"
RNG = np.random.default_rng(20261021)
REPS = 400


def luck_correct(outcomes, luck):
    y = outcomes.reshape(-1) - np.repeat(outcomes.mean(axis=1), outcomes.shape[1])
    lc = luck.reshape(-1, 2)
    lc = lc - lc.mean(0)
    b = np.linalg.lstsq(lc, y, rcond=None)[0]
    return outcomes - luck @ b


def stats(means, within, k, index, slot, free, positions):
    """(I_turn, mean T1 - base, level, spread_all, noise) over the given positions (with repeats)."""
    spreads, gaps, levels, spread_all, noise, level_noise = [], [], [], [], [], []
    for p in positions:
        rows = np.flatnonzero(index == p)
        t1 = rows[(slot[rows] >= 1) & (slot[rows] <= 3) & free[rows]]
        if len(t1) >= 2:
            spreads.append(means[t1].var(ddof=1) - within[t1].mean() / k)
        base = rows[(slot[rows] == 0) & free[rows]]
        if len(base) == 1 and len(t1) >= 1:
            gaps.append(means[t1].mean() - means[base[0]])
        levels.append(means[rows].mean())
        level_noise.append(means[rows].var(ddof=1) / len(rows) if len(rows) >= 2 else 0.0)
        if len(rows) >= 2:
            spread_all.append(means[rows].var(ddof=1) - within[rows].mean() / k)
        noise.append(within[rows].mean())
    lev = np.var(levels, ddof=1) - np.mean(level_noise)   # less the sampling variance of a position's mean
    return (np.mean(spreads), np.mean(gaps), lev, np.mean(spread_all), np.mean(noise),
            len(spreads), len(gaps))


def report(label, outcomes, index, slot, free):
    k = outcomes.shape[1]
    means, within = outcomes.mean(axis=1), outcomes.var(axis=1, ddof=1)
    positions = np.unique(index)
    point = stats(means, within, k, index, slot, free, positions)
    boots = np.array([stats(means, within, k, index, slot, free, RNG.choice(positions, len(positions)))[:5]
                      for _ in range(REPS)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    i_turn, gap, level, spread, noise, n_sp, n_gap = point
    r2 = level / (level + spread + noise)
    r2_boot = boots[:, 2] / (boots[:, 2] + boots[:, 3] + boots[:, 4])
    r2_lo, r2_hi = np.percentile(r2_boot, [2.5, 97.5])
    print(f"[{label}]")
    print(f"  A. I_turn (attack-free temperature-1 turns, {n_sp} positions): {i_turn:.4f} (95% {lo[0]:.4f} to {hi[0]:.4f})")
    print(f"  B. temperature-1 turn minus base turn (both attack-free, {n_gap} positions): {gap:+.4f} (95% {lo[1]:+.4f} to {hi[1]:+.4f})")
    print(f"  C. level {level:.4f}, between-candidate spread {spread:.4f}, within-candidate noise {noise:.4f};"
          f" share explained by the turn-start position {r2:.3f} (95% {r2_lo:.3f} to {r2_hi:.3f})")


def main():
    d = np.load(NPZ, allow_pickle=False)
    outcomes = d["outcomes"].astype(float)
    luck = d["luck"].astype(float)
    index, slot = d["index"], d["slot"]
    free = np.all(d["turn_luck"] == 0.0, axis=1)
    print(f"candidates {len(slot)}, positions {len(np.unique(index))}, playouts {outcomes.shape[1]};"
          f" attack-free candidates {int(free.sum())} (base turns {int((free & (slot == 0)).sum())},"
          f" temperature-1 turns {int((free & (slot >= 1) & (slot <= 3)).sum())})")
    report("raw", outcomes, index, slot, free)
    report("luck-corrected", luck_correct(outcomes, luck), index, slot, free)


if __name__ == "__main__":
    main()
