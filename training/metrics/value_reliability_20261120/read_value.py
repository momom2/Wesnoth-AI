#!/usr/bin/env python3
"""How reliable the simple loop's logged value estimate is
(docs/value_reliability_prereg_20261120.md), read from the decision logs
of the loop's own games. No game is played and no network is run.

Per game-side of the current network: the value head's expectation v_t at
each of its decisions, in play order, and the side's result z (+1, -1, or
0 without a winner).

    python training/metrics/value_reliability_20261120/read_value.py \
        --run C:/Users/momom2/wesnoth_runs/simple_loop --rounds 177 208 --out late.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.game_record import read_records  # noqa: E402

RECORD_SUFFIX = ".game.jsonl.gz"
LAMBDAS = (1.0, 0.98, 0.95, 0.9, 0.8)
BOOTSTRAPS = 400


def game_sides(run: Path, first: int, last: int) -> List[dict]:
    """The current network's game-sides of rounds first..last: both sides of
    a self game, side A of a past-self game (labels `sNNNN` against
    `sNNNNb` or `pNNNN`). Each: game, v (array, play order), z."""
    out, skipped = [], {"no_result": 0, "non_finite": 0, "empty": 0}
    for k in range(first, last + 1):
        for kind in ("games_self", "games_past"):
            for path in sorted((run / f"round_{k:04d}" / kind).glob("*" + RECORD_SUFFIX)):
                result = path.with_name(path.name[:-len(RECORD_SUFFIX)] + ".json")
                if not result.is_file():
                    skipped["no_result"] += 1
                    continue
                res = json.loads(result.read_text(encoding="utf-8"))
                records = list(read_records(path))
                if len(records) != 1:
                    skipped["no_result"] += 1
                    continue
                side_a = int(res["side_a"])
                z_a = {"win": 1.0, "loss": -1.0}.get(res.get("outcome_a") or "", 0.0)
                current = (side_a, 3 - side_a) if str(res["label_b"]).endswith("b") else (side_a,)
                for side in current:
                    v = np.array([float(d["v"]) for d in records[0]["decisions"]
                                  if int(d["side"]) == side and not d["refused"]], dtype=np.float64)
                    if v.size == 0:
                        skipped["empty"] += 1
                    elif not np.isfinite(v).all():
                        skipped["non_finite"] += 1
                    else:
                        out.append({"game": path.name, "round": k, "v": v,
                                    "z": z_a if side == side_a else -z_a})
    print(f"rounds {first}-{last}: {len(out)} game-sides, skipped {skipped}", flush=True)
    return out


def slope(v: np.ndarray, z: np.ndarray) -> Dict[str, float]:
    """The least-squares line of z on v."""
    vm, zm = v.mean(), z.mean()
    b = float(((v - vm) * (z - zm)).sum() / ((v - vm) ** 2).sum())
    return {"slope": b, "intercept": float(zm - b * vm), "n": int(v.size)}


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))


def flat(sides: List[dict]) -> Dict[str, np.ndarray]:
    """Every decision of `sides` in one set of arrays: v, z, progress, the
    change d_t (the last one z - v_T), the change before it (NaN at a
    side's first decision), whether it is the side's last, and the weight
    at each lambda."""
    cols: Dict[str, list] = {k: [] for k in ("v", "z", "progress", "d", "d_prev", "last", "side")}
    weights = {lam: [] for lam in LAMBDAS}
    for i, s in enumerate(sides):
        v, z, n = s["v"], s["z"], s["v"].size
        d = np.append(v[1:], z) - v
        cols["v"].append(v)
        cols["z"].append(np.full(n, z))
        cols["progress"].append(np.arange(n) / n)
        cols["d"].append(d)
        cols["d_prev"].append(np.append(np.nan, d[:-1]))
        cols["last"].append(np.arange(n) == n - 1)
        cols["side"].append(np.full(n, i))
        for lam in LAMBDAS:
            a = np.empty(n)
            acc = 0.0
            for t in range(n - 1, -1, -1):
                acc = d[t] + lam * acc
                a[t] = acc
            weights[lam].append(a)
    out = {k: np.concatenate(x) for k, x in cols.items()}
    for lam in LAMBDAS:
        out[f"w{lam}"] = np.concatenate(weights[lam])
    return out


def deciders(f: Dict[str, np.ndarray]) -> Dict[str, float]:
    """The three registered numbers: the slope in each third, the
    correlation of consecutive changes, the weight's mean square at
    lambda 0.95 over today's."""
    out = {}
    for third in range(3):
        m = (f["progress"] >= third / 3) & (f["progress"] < (third + 1) / 3)
        out[f"slope_third_{third + 1}"] = slope(f["v"][m], f["z"][m])["slope"]
    m = ~np.isnan(f["d_prev"])
    out["corr_consecutive_changes"] = corr(f["d"][m], f["d_prev"][m])
    out["mean_square_ratio_0.95"] = float((f["w0.95"] ** 2).mean() / (f["w1.0"] ** 2).mean())
    return out


def read(sides: List[dict], rng: np.random.Generator) -> dict:
    f = flat(sides)
    v, z, d = f["v"], f["z"], f["d"]
    res: dict = {"game_sides": len(sides), "decisions": int(v.size),
                 "results": {"wins": int(sum(s["z"] > 0 for s in sides)), "losses": int(sum(s["z"] < 0 for s in sides)),
                             "no_winner": int(sum(s["z"] == 0 for s in sides))}}
    # 1. calibration
    res["calibration"] = {"all": slope(v, z)}
    for third in range(3):
        m = (f["progress"] >= third / 3) & (f["progress"] < (third + 1) / 3)
        res["calibration"][f"third_{third + 1}"] = slope(v[m], z[m])
    order = np.argsort(v, kind="stable")
    res["calibration"]["bins"] = [{"mean_v": float(v[b].mean()), "mean_z": float(z[b].mean()), "n": int(b.size)}
                                  for b in np.array_split(order, 10)]
    # 2. how much it knows, and when
    known = lambda m: float(1.0 - ((z[m] - v[m]) ** 2).mean() / (z[m] ** 2).mean())  # noqa: E731
    res["explained"] = {"all": known(np.ones(v.size, bool)),
                        "mean_square_error": float(((z - v) ** 2).mean()), "mean_z_squared": float((z ** 2).mean()),
                        "by_tenth": [known((f["progress"] >= k / 10) & (f["progress"] < (k + 1) / 10))
                                     for k in range(10)]}
    # 3. whether it moves like a value
    m = ~np.isnan(f["d_prev"])
    inner = m & ~f["last"]
    res["changes"] = {"corr_with_previous": corr(d[m], f["d_prev"][m]),
                      "corr_with_previous_without_the_last": corr(d[inner], f["d_prev"][inner]),
                      "corr_with_v": corr(d, v),
                      "corr_with_v_without_the_last": corr(d[~f["last"]], v[~f["last"]]),
                      "mean_square_of_a_change": float((d[~f["last"]] ** 2).mean()),
                      "mean_square_of_the_last_change": float((d[f["last"]] ** 2).mean())}
    # 4. what lambda would do to the weight
    base = f["w1.0"]
    res["lambda"] = {str(lam): {"mean_square_over_todays": float((f[f"w{lam}"] ** 2).mean() / (base ** 2).mean()),
                                "corr_with_todays": corr(f[f"w{lam}"], base)} for lam in LAMBDAS}
    # 5. where the weight sits
    power = np.array([float(((s["z"] - s["v"]) ** 2).sum()) for s in sides])
    length = np.array([s["v"].size for s in sides], dtype=np.float64)
    top = np.sort(power)[::-1]
    root = np.sqrt(power)
    res["weight_by_game_side"] = {
        "share_top_tenth": float(top[:max(1, len(top) // 10)].sum() / top.sum()),
        "share_top_quarter": float(top[:max(1, len(top) // 4)].sum() / top.sum()),
        "share_top_half": float(top[:max(1, len(top) // 2)].sum() / top.sum()),
        "one_plus_cv2_of_root": float(1.0 + root.var() / root.mean() ** 2),
        "corr_power_with_length": corr(power, length),
        "median_decisions": float(np.median(length)),
        "share_of_sides_under_a_tenth_of_the_mean": float((power < 0.1 * power.mean()).mean())}
    # the deciding numbers, with standard errors from resampling whole games
    res["deciders"] = deciders(f)
    res["deciders_se"] = decider_errors(f, [s["game"] for s in sides], rng)
    return res


def decider_errors(f: Dict[str, np.ndarray], game_of_side: List[str], rng: np.random.Generator) -> Dict[str, float]:
    """The standard error of each deciding number over BOOTSTRAPS
    resamplings of whole games (a self game's two sides share a result).
    Each number is a function of sums over decisions, so a resampling is a
    reweighting of the games' sums."""
    names = {g: i for i, g in enumerate(sorted(set(game_of_side)))}
    game = np.array([names[g] for g in game_of_side])[f["side"].astype(int)]
    n_games = len(names)

    def sums(mask: np.ndarray, *columns: np.ndarray) -> np.ndarray:
        return np.stack([np.bincount(game[mask], weights=c[mask], minlength=n_games) for c in columns], axis=1)

    one, v, z, d = np.ones(f["v"].size), f["v"], f["z"], f["d"]
    counts = rng.multinomial(n_games, np.full(n_games, 1.0 / n_games), size=BOOTSTRAPS).astype(np.float64)
    counts = np.vstack([np.ones(n_games), counts])      # row 0: every game once, checked against `deciders`
    out = {}
    for third in range(3):
        m = (f["progress"] >= third / 3) & (f["progress"] < (third + 1) / 3)
        n, sv, sz, svv, svz = (counts @ sums(m, one, v, z, v * v, v * z)).T
        out[f"slope_third_{third + 1}"] = (svz - sv * sz / n) / (svv - sv * sv / n)
    m = ~np.isnan(f["d_prev"])
    p = np.where(m, f["d_prev"], 0.0)
    n, sd, sp, sdd, spp, sdp = (counts @ sums(m, one, d, p, d * d, p * p, d * p)).T
    out["corr_consecutive_changes"] = (sdp - sd * sp / n) / np.sqrt((sdd - sd * sd / n) * (spp - sp * sp / n))
    every = np.ones(f["v"].size, bool)
    s95, s1 = (counts @ sums(every, f["w0.95"] ** 2, f["w1.0"] ** 2)).T
    out["mean_square_ratio_0.95"] = s95 / s1
    direct = deciders(f)
    for k, x in out.items():
        assert abs(x[0] - direct[k]) < 1e-9, (k, x[0], direct[k])
    return {k: float(np.std(x[1:], ddof=1)) for k, x in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--rounds", type=int, nargs=2, required=True, metavar=("FIRST", "LAST"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261120)
    args = ap.parse_args()
    sides = game_sides(args.run, *args.rounds)
    res = read(sides, np.random.default_rng(args.seed))
    res["rounds"] = list(args.rounds)
    args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("game_sides", "decisions", "results", "deciders", "deciders_se")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
