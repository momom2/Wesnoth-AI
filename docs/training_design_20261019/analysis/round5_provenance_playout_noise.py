"""Round 5 (provenance stance): playout noise and candidate-turn spread at human mid-game positions.

Reads training/metrics/turn_value_20260925/bench_truth.npz: 199 human positions (the turn is in
the replay file name), about 5 candidate turns each (the replay's own turn and alternatives
sampled at temperature 1, each played with its own dice), 28 playouts per candidate to the game's
end at temperature 0.5 (cap scored 0), the playouts' two luck columns and the fitted beta.

Reports, by the position's game turn:
  noise   the within-candidate variance of one playout's result (pooled sample variance), raw
          and luck-corrected with the recorded beta;
  spread  the between-candidate variance of the true candidate mean within a position,
          var(candidate means) - noise / 28, a method-of-moments estimate. Candidates are whole
          turns with their own dice, so this is an upper side for what any one decision's
          choice moves (the ruling forbids reading it as a decision spread);
  level   the between-position variance of position means.
numpy and the standard library only.
"""
import re
from pathlib import Path

import numpy as np

NPZ = Path(__file__).resolve().parents[3] / "training/metrics/turn_value_20260925/bench_truth.npz"
RNG = np.random.default_rng(20261020)


def decompose(outcomes, index):
    """Pooled within-candidate variance, mean within-position spread of candidate means, levels."""
    k = outcomes.shape[1]
    within = outcomes.var(axis=1, ddof=1)
    means = outcomes.mean(axis=1)
    noise = within.mean()
    spreads, levels = [], []
    for p in np.unique(index):
        m = means[index == p]
        if len(m) >= 2:
            spreads.append(m.var(ddof=1) - within[index == p].mean() / k)
        levels.append(m.mean())
    return noise, float(np.mean(spreads)), float(np.var(levels, ddof=1)), len(spreads)


def boot(outcomes, index, reps=300):
    positions = np.unique(index)
    rows_of = {p: np.flatnonzero(index == p) for p in positions}
    res = []
    for _ in range(reps):
        pick = RNG.choice(positions, len(positions))
        rows = np.concatenate([rows_of[p] for p in pick])
        new_index = np.concatenate([np.full(len(rows_of[p]), i) for i, p in enumerate(pick)])
        res.append(decompose(outcomes[rows], new_index)[:3])
    return np.percentile(np.array(res), [2.5, 97.5], axis=0)


def main():
    d = np.load(NPZ, allow_pickle=False)
    outcomes = d["outcomes"].astype(float)
    luck = d["luck"].astype(float)
    beta = d["beta"].astype(float)
    index = d["index"]
    turn = np.array([int(re.search(r"Turn_(\d+)", c).group(1)) for c in d["cluster"]])
    print(f"candidates {len(index)}, positions {len(np.unique(index))}, playouts per candidate {outcomes.shape[1]}")
    print(f"result values present: {sorted(set(np.unique(outcomes).tolist()))}")
    print(f"beta (recorded) {beta}")
    # The recorded beta has four entries for two luck columns: read it as two (column, fold) pairs is
    # not documented, so the luck-corrected rows use a fresh within-sample fit of the two columns.
    L = luck.reshape(-1, 2)
    y = outcomes.reshape(-1) - np.repeat(outcomes.mean(axis=1), outcomes.shape[1])
    Lc = L - L.mean(0)
    b = np.linalg.lstsq(Lc, y, rcond=None)[0]
    corrected = outcomes - (luck @ b)
    print(f"luck slope fitted in-sample on the two columns: {b}, within-candidate R2 "
          f"{1 - np.var(y - Lc @ b) / np.var(y):.3f}")
    for name, out in (("raw", outcomes), ("luck-corrected", corrected)):
        noise, spread, level, n = decompose(out, index)
        lo, hi = boot(out, index)
        print(f"\n{name}: one-playout noise {noise:.3f} ({lo[0]:.3f}-{hi[0]:.3f}); "
              f"candidate spread (variance) {spread:.4f} ({lo[1]:.4f}-{hi[1]:.4f}), "
              f"sd {np.sqrt(max(spread, 0)):.3f}; position level variance {level:.3f} ({lo[2]:.3f}-{hi[2]:.3f})")
        for lo_t, hi_t in ((1, 6), (7, 10), (11, 15), (16, 99)):
            sel = (turn >= lo_t) & (turn <= hi_t)
            if sel.sum() < 10:
                continue
            nz, sp, lv, npos = decompose(out[sel], index[sel])
            print(f"  turns {lo_t:2d}-{hi_t:2d}: positions {npos:3d}, noise {nz:.3f}, spread {sp:+.4f}, level {lv:.3f}")




def turn_dice_share():
    """How much of the candidate spread the candidate turn's own dice explain (within position)."""
    d = np.load(NPZ, allow_pickle=False)
    outcomes = d["outcomes"].astype(float)
    tl = d["turn_luck"].astype(float)
    index = d["index"]
    means = outcomes.mean(axis=1)
    m_c, t_c = means.copy(), tl.copy()
    for p in np.unique(index):
        sel = index == p
        m_c[sel] -= m_c[sel].mean()
        t_c[sel] -= t_c[sel].mean(0)
    b = np.linalg.lstsq(t_c, m_c, rcond=None)[0]
    r2 = 1 - np.var(m_c - t_c @ b) / np.var(m_c)
    print(f"\nwithin-position regression of candidate means on the candidate turn's two luck columns: "
          f"slope {b}, R2 {r2:.3f} of the within-position variance of candidate means "
          f"(which also holds the playout noise / 28)")


if __name__ == "__main__":
    main()
    turn_dice_share()
