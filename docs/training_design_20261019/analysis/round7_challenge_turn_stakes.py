"""Round 7 (challenge stance): how much the choice of a whole turn moves the result, read where the
turn itself rolls no dice.

Source 1, training/metrics/turn_value_20260925/bench_truth.npz: 199 human mid-game positions, about
5 candidate turns each (slot 0 the replay's own turn, the others sampled from the network at
temperature 1), 28 playouts per candidate to the game's end at temperature 0.5 (cap scored 0), the
playouts' two luck columns, and turn_luck, the candidate turn's own two luck columns. A candidate
whose turn_luck is exactly 0 in both columns made no attack (its turn rolled no combat dice), so the
spread among such candidates at one position is not the in-turn dice the ruling of 2026-10-19
objects to. It still holds recruits' trait draws (not in the luck columns; source 2 counts them).

Reported, by subset, with a position bootstrap (400 resamples):
  noise   pooled within-candidate variance of one playout's result;
  spread  mean over positions of var(candidate means) - mean within variance / 28, the
          method-of-moments between-candidate variance of the true candidate mean;
  base    for attack-free alternatives at positions whose own (slot 0) turn is attack-free:
          mean and noise-corrected variance of (alternative mean - own-turn mean).
Results are luck-corrected inside each candidate (a fresh in-sample slope on the playouts' two
luck columns, as round5_provenance_playout_noise.py does); raw results are reported too.

Source 2, training/metrics/turn_value_20260925/data.jsonl.gz: the same generator's training positions
(one playout per candidate, so no spread from it): the number of decisions in attack-free candidate
turns and the share of them that recruit. Used for the additive per-decision bound
    s_dev >= sqrt(spread / m)
(if the turn's value difference were a sum of m independent per-decision effects of sd s_dev, the
between-turn variance would be at most m s_dev^2).

Last, the derived break-even of a per-side-turn coin against a near-tie decision coin at the same
p: information per game is n * p(1-p) * stakes^2 / Var(R) for n coins a game, so a turn coin wins
iff its stakes exceed sqrt(n_band / n_turn) times the near-tie spread.
numpy and the standard library only.
"""
import gzip
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
NPZ = ROOT / "training/metrics/turn_value_20260925/bench_truth.npz"
DATA = ROOT / "training/metrics/turn_value_20260925/data.jsonl.gz"
RNG = np.random.default_rng(20261020)
REPS = 400

# Measured in round5_simplify_critique_decision_units.out (own1 self-play, 1,600 side-games).
SIDE_TURNS_PER_SIDE_GAME = 19.4
# Derived in current_design.md section 3.1: 0.339 x 224.6.
BAND_DECISIONS_PER_SIDE_GAME = 76.1


def luck_correct(outcomes, luck):
    y = outcomes.reshape(-1) - np.repeat(outcomes.mean(axis=1), outcomes.shape[1])
    L = luck.reshape(-1, 2)
    Lc = L - L.mean(0)
    b = np.linalg.lstsq(Lc, y, rcond=None)[0]
    return outcomes - luck @ b


def spread_stats(means, within, k, index, mask):
    """Per position with >= 2 candidates in mask: var(means) - mean(within)/k. Returns
    (mean spread, positions, candidates, pooled noise)."""
    sp, n_pos, n_c, noise = [], 0, 0, []
    for p in np.unique(index[mask]):
        m = mask & (index == p)
        if m.sum() >= 2:
            sp.append(means[m].var(ddof=1) - within[m].mean() / k)
            n_pos += 1
            n_c += int(m.sum())
            noise.append(within[m].mean())
    return (float(np.mean(sp)) if sp else float("nan")), n_pos, n_c, (float(np.mean(noise)) if noise else float("nan"))


def base_contrast(means, within, k, index, slot, free):
    """At positions whose slot-0 turn is attack-free: attack-free alternatives minus slot 0.
    Returns per-position lists of differences and their noise variances."""
    diffs, noises = [], []
    for p in np.unique(index):
        rows = np.flatnonzero(index == p)
        base = rows[slot[rows] == 0]
        if len(base) != 1 or not free[base[0]]:
            continue
        b = base[0]
        for r in rows:
            if r != b and free[r]:
                diffs.append(means[r] - means[b])
                noises.append((within[r] + within[b]) / k)
    return np.array(diffs), np.array(noises)


def bootstrap(fn, index, reps=REPS):
    positions = np.unique(index)
    rows_of = {p: np.flatnonzero(index == p) for p in positions}
    out = []
    for _ in range(reps):
        pick = RNG.choice(positions, len(positions))
        rows = np.concatenate([rows_of[p] for p in pick])
        new_index = np.concatenate([np.full(len(rows_of[p]), i) for i, p in enumerate(pick)])
        out.append(fn(rows, new_index))
    return np.array(out, dtype=float)


def part1():
    d = np.load(NPZ, allow_pickle=False)
    outcomes_raw = d["outcomes"].astype(float)
    luck = d["luck"].astype(float)
    tl = d["turn_luck"].astype(float)
    index = d["index"]
    slot = d["slot"]
    k = outcomes_raw.shape[1]
    free = np.all(tl == 0, axis=1)
    print(f"1. bench_truth: candidates {len(index)}, positions {len(np.unique(index))}, playouts {k}; "
          f"attack-free candidates {int(free.sum())} (slot 0: {int(free[slot == 0].sum())} of {int((slot == 0).sum())})")
    subsets = {
        "all candidates": np.ones(len(index), bool),
        "attack-free": free,
        "with attacks": ~free,
        "attack-free, alternatives only": free & (slot > 0),
    }
    for name, outcomes in (("luck-corrected", luck_correct(outcomes_raw, luck)), ("raw", outcomes_raw)):
        means = outcomes.mean(axis=1)
        within = outcomes.var(axis=1, ddof=1)
        print(f"  [{name}]")
        for label, mask in subsets.items():
            s, n_pos, n_c, noise = spread_stats(means, within, k, index, mask)

            def fn(rows, new_index, mask=mask):
                return spread_stats(means[rows], within[rows], k, new_index, mask[rows])[0]

            lo, hi = np.nanpercentile(bootstrap(fn, index), [2.5, 97.5])
            print(f"    {label:32s} positions {n_pos:3d} candidates {n_c:3d} noise {noise:.3f} "
                  f"spread {s:.4f} (95% {lo:.4f} to {hi:.4f}), sd {math.sqrt(max(s, 0)):.3f} "
                  f"({math.sqrt(max(lo, 0)):.3f} to {math.sqrt(max(hi, 0)):.3f})")
        diffs, noises = base_contrast(means, within, k, index, slot, free)

        def fn_b(rows, new_index):
            dd, nn = base_contrast(means[rows], within[rows], k, new_index, slot[rows], free[rows])
            return [dd.mean(), dd.var(ddof=1) - nn.mean()] if len(dd) > 2 else [np.nan, np.nan]

        b = bootstrap(fn_b, index)
        lo_m, hi_m = np.nanpercentile(b[:, 0], [2.5, 97.5])
        lo_v, hi_v = np.nanpercentile(b[:, 1], [2.5, 97.5])
        var_true = diffs.var(ddof=1) - noises.mean()
        print(f"    attack-free alternative minus attack-free own turn: pairs {len(diffs)}, mean {diffs.mean():+.4f} "
              f"(95% {lo_m:+.4f} to {hi_m:+.4f}), noise-corrected variance {var_true:.4f} (95% {lo_v:.4f} to {hi_v:.4f})")
    return free


def part2():
    lengths, recruits, total = [], 0, 0
    with gzip.open(DATA, "rt") as f:
        next(f)
        for line in f:
            r = json.loads(line)
            if "base" not in r:
                continue
            for cand in [r["base"]] + r.get("alternatives", []):
                acts = cand.get("actions") or []
                kinds = [a.get("type") for a in acts]
                if "attack" in kinds:
                    continue
                total += 1
                lengths.append(sum(1 for t in kinds if t != "end_turn"))
                recruits += "recruit" in kinds
    lengths = np.array(lengths)
    print(f"2. data.jsonl.gz attack-free candidate turns {total}: decisions before end_turn mean {lengths.mean():.2f}, "
          f"quantiles 10/50/90 {np.percentile(lengths, [10, 50, 90]).tolist()}, share with a recruit {recruits / total:.3f}")
    return lengths


def part3(spread_free, lengths):
    m_mean = lengths.mean()
    m_p90 = float(np.percentile(lengths, 90))
    print("3. additive per-decision bound s_dev >= sqrt(spread / m):")
    for label, m in (("m = mean decisions", m_mean), ("m = 90th percentile", m_p90)):
        print(f"   {label} {m:.2f}: s_dev >= {math.sqrt(spread_free / m):.4f}")
    ratio = math.sqrt(2 * BAND_DECISIONS_PER_SIDE_GAME / (2 * SIDE_TURNS_PER_SIDE_GAME))
    print(f"4. break-even: a side-turn coin carries more information a game than a near-tie decision coin at the "
          f"same p iff its stakes exceed {ratio:.2f} x the near-tie spread "
          f"(sqrt({2 * BAND_DECISIONS_PER_SIDE_GAME:.1f} band coins / {2 * SIDE_TURNS_PER_SIDE_GAME:.1f} turn coins a game))")
    for v in (0.0016, 0.0036, 0.01):
        print(f"   near-tie spread sd {v}: turn stakes needed {ratio * v:.4f}")


def main():
    part1()
    d = np.load(NPZ, allow_pickle=False)
    outcomes = luck_correct(d["outcomes"].astype(float), d["luck"].astype(float))
    free = np.all(d["turn_luck"].astype(float) == 0, axis=1)
    means, within = outcomes.mean(axis=1), outcomes.var(axis=1, ddof=1)
    spread_free = spread_stats(means, within, outcomes.shape[1], d["index"], free)[0]
    lengths = part2()
    part3(spread_free, lengths)


if __name__ == "__main__":
    main()
