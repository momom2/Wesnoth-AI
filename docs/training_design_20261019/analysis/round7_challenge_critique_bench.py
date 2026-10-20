"""Round 7, critique of the challenge stance: what the turn-value bench says about selectable turn stakes.

Source: training/metrics/turn_value_20260925/bench_truth.npz (199 human positions; per
docs/turn_value_prereg_20260925.md "Validation": slot 0 is obs8's own turn at raw:t0+eo-1.5, then
three turns sampled at temperature 1, then one "continue" edit, the base turn without its end_turn
plus one more argmax action; tools/turn_gap.py appends them in that order and drops duplicates,
so at the 173 positions with 5 candidates slot 4 is the continue edit). 28 playouts each.

Luck correction as in round7_challenge_turn_stakes.py (in-sample slope on the playouts' two luck
columns). Attack-free = turn_luck exactly 0 in both columns (the challenge's filter).

1. Reproduces the challenge's attack-free spread.
2. At positions with 5 candidates and an attack-free base: alternative minus base, by kind
   (T=1 samples, slots 1-3; continue edit, slot 4), attack-free alternatives only: mean and
   noise-corrected variance (var of the difference minus the mean noise of the difference).
3. Split-half selection (choose on playouts 0,2,4..., read on 1,3,5...): the gain over the base of
   (a) the best alternative (turn_gap's statistic), (b) the best candidate with the base allowed
   (a selector that may keep the default), over all candidates and over attack-free ones.
4. Gaussian check: if alternative-minus-base were N(mu, var) with the moments of step 2/3 and the
   bench's own noise, what split-half gain (b) would we expect? Monte Carlo, seeded.
Bootstrap over positions, 400 resamples. numpy and the standard library only.
"""
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
NPZ = ROOT / "training/metrics/turn_value_20260925/bench_truth.npz"
RNG = np.random.default_rng(20261020)
REPS = 400


def luck_correct(outcomes, luck):
    y = outcomes.reshape(-1) - np.repeat(outcomes.mean(axis=1), outcomes.shape[1])
    L = luck.reshape(-1, 2)
    b = np.linalg.lstsq(L - L.mean(0), y, rcond=None)[0]
    return outcomes - luck @ b


def load():
    d = np.load(NPZ, allow_pickle=False)
    out = luck_correct(d["outcomes"].astype(float), d["luck"].astype(float))
    free = np.all(d["turn_luck"].astype(float) == 0, axis=1)
    return out, free, d["index"], d["slot"]


def positions(index, slot):
    pos = {}
    for p in np.unique(index):
        rows = np.flatnonzero(index == p)
        pos[int(p)] = rows[np.argsort(slot[rows])]
    return pos


def diffs_by_kind(out, free, pos, keys):
    """Alternative minus base at 5-candidate positions with attack-free base; attack-free alts."""
    res = {"T1": [], "cont": []}
    k = out.shape[1]
    for p in keys:
        rows = pos[p]
        if len(rows) != 5 or not free[rows[0]]:
            continue
        b = rows[0]
        for j, r in enumerate(rows[1:], start=1):
            if not free[r]:
                continue
            dmean = out[r].mean() - out[b].mean()
            noise = (out[r].var(ddof=1) + out[b].var(ddof=1)) / k
            res["cont" if j == 4 else "T1"].append((dmean, noise))
    return {kk: np.array(v).reshape(-1, 2) for kk, v in res.items()}


def moments(arr):
    if len(arr) < 3:
        return float("nan"), float("nan")
    return float(arr[:, 0].mean()), float(arr[:, 0].var(ddof=1) - arr[:, 1].mean())


def split_half(out, free, pos, keys, attack_free_only):
    """Per position: (a) best alternative's odd-half gap; (b) best candidate incl. base."""
    ev, od = out[:, 0::2].mean(1), out[:, 1::2].mean(1)
    ga, gb = [], []
    for p in keys:
        rows = pos[p]
        b = rows[0]
        if attack_free_only and not free[b]:
            continue
        alts = [r for r in rows[1:] if (free[r] or not attack_free_only)]
        if not alts:
            continue
        best_alt = max(alts, key=lambda r: ev[r])
        ga.append(od[best_alt] - od[b])
        best_any = max([b] + alts, key=lambda r: ev[r])
        gb.append(od[best_any] - od[b])
    return np.array(ga), np.array(gb)


def boot(fn, keys):
    keys = np.array(keys)
    vals = [fn(list(RNG.choice(keys, len(keys)))) for _ in range(REPS)]
    return np.nanstd(np.array(vals, dtype=float), axis=0)


def gaussian_expected_gain(mu, var, noise_half, n_alts, reps=200_000):
    """Split-half gain (b) if each alternative minus base is N(mu, var), true values fixed,
    estimates on each half with noise variance noise_half (per candidate mean)."""
    sd = math.sqrt(max(var, 0.0))
    true = np.concatenate([np.zeros((reps, 1)), mu + sd * RNG.standard_normal((reps, n_alts))], axis=1)
    ev = true + math.sqrt(noise_half) * RNG.standard_normal(true.shape)
    pick = ev.argmax(1)
    return float(true[np.arange(reps), pick].mean())


def main():
    out, free, index, slot = load()
    pos = positions(index, slot)
    keys = sorted(pos)
    k = out.shape[1]
    n5 = sum(1 for p in keys if len(pos[p]) == 5)
    n5_free = sum(1 for p in keys if len(pos[p]) == 5 and free[pos[p][0]])
    print(f"positions {len(keys)}, with 5 candidates {n5}, of which attack-free base {n5_free}")

    d = diffs_by_kind(out, free, pos, keys)
    for kind in ("T1", "cont"):
        m, v = moments(d[kind])

        def fn(ks, kind=kind):
            return moments(diffs_by_kind(out, free, pos, ks)[kind])

        se = boot(fn, keys)
        print(f"2. {kind:4s} attack-free alt minus attack-free base: n {len(d[kind])}, mean {m:+.4f} (SE {se[0]:.4f}), "
              f"noise-corrected var {v:.4f} (SE {se[1]:.4f}), sd {math.sqrt(max(v, 0)):.3f}")

    for af in (False, True):
        ga, gb = split_half(out, free, pos, keys, af)

        def fn(ks, af=af):
            a, b = split_half(out, free, pos, ks, af)
            return [a.mean(), b.mean()]

        se = boot(fn, keys)
        tag = "attack-free base and alts" if af else "all candidates"
        print(f"3. split-half, {tag}: positions {len(ga)}; best alternative {ga.mean():+.4f} (SE {se[0]:.4f}); "
              f"best incl. base (may keep default) {gb.mean():+.4f} (SE {se[1]:.4f})")

    # Gaussian check on the attack-free subset: pooled moments of all attack-free alternatives vs base.
    pooled = np.concatenate([d["T1"], d["cont"]])
    mu, var = moments(pooled)
    noise_half = 2 * float(np.mean([out[r].var(ddof=1) for p in keys for r in pos[p] if free[r]])) / k
    ga, gb = split_half(out, free, pos, keys, True)
    n_alts = 3
    exp_gain = gaussian_expected_gain(mu, var, noise_half, n_alts)
    print(f"4. Gaussian check (attack-free): alt-minus-base N({mu:+.4f}, {var:.4f}), half-sample noise var {noise_half:.4f}, "
          f"{n_alts} alternatives: expected split-half gain (b) {exp_gain:+.4f}; measured {gb.mean():+.4f}")
    exp_noiseless = gaussian_expected_gain(mu, var, 1e-12, n_alts)
    print(f"   same with exact estimates (oracle pick of best of {n_alts} plus base): {exp_noiseless:+.4f} a side-turn")


if __name__ == "__main__":
    main()


def split_half_kind(out, free, pos, keys, kind, attack_free_only):
    """At 5-candidate positions: best of base and the chosen kind's candidates (T1: slots 1-3,
    cont: slot 4), chosen on even playouts, read on odd, minus base."""
    ev, od = out[:, 0::2].mean(1), out[:, 1::2].mean(1)
    g = []
    for p in keys:
        rows = pos[p]
        if len(rows) != 5:
            continue
        b = rows[0]
        if attack_free_only and not free[b]:
            continue
        cand = list(rows[1:4]) if kind == "T1" else [rows[4]]
        cand = [r for r in cand if free[r] or not attack_free_only]
        if not cand:
            continue
        best = max([b] + cand, key=lambda r: ev[r])
        g.append(od[best] - od[b])
    return np.array(g)


def by_kind():
    out, free, index, slot = load()
    pos = positions(index, slot)
    keys = sorted(pos)
    for af in (False, True):
        for kind in ("T1", "cont"):
            g = split_half_kind(out, free, pos, keys, kind, af)

            def fn(ks, kind=kind, af=af):
                return [split_half_kind(out, free, pos, ks, kind, af).mean()]

            se = boot(fn, keys)[0]
            tag = "attack-free" if af else "all"
            print(f"5. split-half, 5-candidate positions, {tag}, base or {kind}: positions {len(g)}, "
                  f"gain {g.mean():+.4f} (SE {se:.4f})")


if __name__ == "__main__":
    by_kind()
