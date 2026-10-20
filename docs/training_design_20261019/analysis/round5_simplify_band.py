"""Round 5, simplify stance: the near-tie band of the argmax decode, from the record.

Input: training/metrics/critic_step1_20261009/prior_gaps.games.jsonl
(`parity3` at 64 slots, offset -1.5, argmax, 100 recorded side-games of its
match against `parity2`; one reading per decision: number of legal actions,
the gaps top1 - top_k for k = 2..8 in joint log-prior after the end_turn
shift, and the type of the top action).

Output (stdout and round5_simplify_band.json): per side-game counts of
decisions in the band G = top1 - top2 < w, by top type; the coin design's
flips, information sum p(1-p) and shift sum p*G for a flat coin probability,
and the same for the KKT taper p = 1/2 (1 - G/w')_+ matched to the same
number of flips (the removal cost of the taper).

Standard library and numpy only; reads one 5.5 MB file.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
SRC = ROOT / "training/metrics/critic_step1_20261009/prior_gaps.games.jsonl"
OUT = pathlib.Path(__file__).with_suffix(".json")

WIDTHS = (0.175, 0.35, 0.7)
FLAT_P = (0.125, 0.25, 0.5)


def load():
    games = []
    for line in SRC.open():
        row = json.loads(line)
        gaps, types = [], []
        for r in row["readings"]:
            if r["legal"] < 2 or not r["gaps"]:
                continue
            gaps.append(r["gaps"][0])
            types.append(r["top_type"])
        games.append((np.array(gaps), np.array(types)))
    return games


def taper_flips(gaps, w2):
    p = 0.5 * np.clip(1.0 - gaps / w2, 0.0, None)
    return p


def per_game_mean(values):
    return float(np.mean(values))


def main():
    games = load()
    n_side = len(games)
    all_gaps = np.concatenate([g for g, _ in games])
    all_types = np.concatenate([t for _, t in games])
    out = {"side_games": n_side,
           "multi_legal_decisions_per_side_game": len(all_gaps) / n_side,
           "type_shares_all": {t: float(np.mean(all_types == t)) for t in sorted(set(all_types))}}
    # density per nat per side-game in the first bins
    dens = {}
    for lo in (0.0, 0.1, 0.2, 0.3, 0.6):
        hi = lo + 0.1
        dens[f"[{lo:.1f},{hi:.1f})"] = float(np.sum((all_gaps >= lo) & (all_gaps < hi)) / n_side / 0.1)
    out["density_per_nat_per_side_game"] = dens

    bands = {}
    for w in WIDTHS:
        inb = [(g < w) for g, _ in games]
        n_w = np.array([m.sum() for m in inb], dtype=float)
        sum_g = np.array([g[m].sum() for (g, _), m in zip(games, inb)])
        by_type = {}
        for t in sorted(set(all_types)):
            by_type[t] = float(np.mean([np.sum(m & (ty == t)) for (g, ty), m in zip(games, inb)]))
        rec = {"decisions_in_band": per_game_mean(n_w),
               "decisions_in_band_sd": float(np.std(n_w)),
               "mean_G_in_band": float(sum_g.sum() / n_w.sum()),
               "by_top_type": by_type,
               "flat": {}}
        for p in FLAT_P:
            flips = p * n_w
            info = p * (1 - p) * n_w
            shift = p * sum_g
            # taper matched to the same mean number of flips: find w2
            target = flips.mean()
            lo, hi = 1e-4, 50.0
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                f = np.mean([taper_flips(g, mid).sum() for g, _ in games])
                lo, hi = (mid, hi) if f < target else (lo, mid)
            w2 = 0.5 * (lo + hi)
            tp = [taper_flips(g, w2) for g, _ in games]
            t_info = np.mean([(q * (1 - q)).sum() for q in tp])
            t_shift = np.mean([(q * g).sum() for q, (g, _) in zip(tp, games)])
            t_minp = float(np.min(np.concatenate([q[q > 0] for q in tp])))
            rec["flat"][str(p)] = {
                "flips": float(flips.mean()),
                "info_sum_p1mp": float(info.mean()),
                "shift_sum_pG_nats": float(shift.mean()),
                "info_per_nat_shift": float(info.mean() / shift.mean()),
                "taper_same_flips": {"w_prime": w2, "info_sum_p1mp": float(t_info),
                                     "shift_sum_pG_nats": float(t_shift),
                                     "info_per_nat_shift": float(t_info / t_shift),
                                     "smallest_positive_p": t_minp},
            }
        bands[str(w)] = rec
    out["bands"] = bands
    json.dump(out, OUT.open("w"), indent=1)
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
