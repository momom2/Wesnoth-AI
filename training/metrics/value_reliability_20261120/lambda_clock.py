#!/usr/bin/env python3
"""Two readings for the discussion of a weight with lambda under 1
(docs/value_reliability_prereg_20261120.md): what the estimate does right
after each kind of decision, and what the weight becomes when lambda
fades by the side's turn instead of by the decision.

    python training/metrics/value_reliability_20261120/lambda_clock.py \
        --run C:/Users/momom2/wesnoth_runs/simple_loop --rounds 177 208 --out clock_177_208.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from where_optimism import END_TURN, KINDS, load  # noqa: E402

PER_DECISION = (1.0, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8)
PER_TURN = (0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.0)


def progress_of(d: dict) -> np.ndarray:
    """Each decision's place in its game-side, from 0 to 1."""
    return d["index"] / d["n"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--rounds", type=int, nargs=2, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    d = load(args.run, *args.rounds)
    n = d["v"].size
    same_side = np.r_[d["gside"][1:] == d["gside"][:-1], False]
    change = np.where(same_side, np.r_[d["v"][1:], 0.0] - d["v"], d["z"] - d["v"])     # the last one: z - v_T
    games = int(d["game"].max()) + 1

    def mean_se(x: np.ndarray, mask: np.ndarray) -> dict:
        m = float(x[mask].mean())
        g = d["game"][mask]
        s = np.bincount(g, weights=x[mask], minlength=games)
        c = np.bincount(g, minlength=games).astype(float)
        return {"n": int(mask.sum()), "mean": m, "se": float(math.sqrt(((s - m * c) ** 2).sum()) / c.sum()),
                "mean_square": float((x[mask] ** 2).mean())}

    res: dict = {"rounds": list(args.rounds), "decisions": int(n)}
    res["change_after"] = {name: mean_se(change, same_side & (d["kind"] == k)) for k, name in KINDS.items()}
    res["change_after"]["a side's last decision (to the result)"] = mean_se(change, ~same_side)
    # the weights: fading by the decision, and by the side's turn
    turn_ends = d["kind"] == END_TURN
    today = d["z"] - d["v"]

    def weight(lam: float, by_turn: bool) -> np.ndarray:
        a = np.empty(n)
        acc = 0.0
        for t in range(n - 1, -1, -1):
            if not same_side[t]:
                acc = 0.0
                a[t] = change[t]
            else:
                a[t] = change[t] + (lam if (turn_ends[t] or not by_turn) else 1.0) * acc
            acc = a[t]
        return a

    def summary(a: np.ndarray) -> dict:
        return {"mean_square_over_todays": float((a ** 2).mean() / (today ** 2).mean()),
                "corr_with_todays": float(np.corrcoef(a, today)[0, 1])}

    assert np.allclose(weight(1.0, False), today)
    res["fading_by_the_decision"] = {str(lam): summary(weight(lam, False)) for lam in PER_DECISION}
    res["fading_by_the_turn"] = {str(lam): summary(weight(lam, True)) for lam in PER_TURN}
    res["mean_decisions_in_a_turn"] = float(d["turn_len"][d["place"] == 0].mean())
    res["mean_decisions_of_a_game_side"] = float(n / (int(d["gside"].max()) + 1))
    # What a lambda under 1 changes, decision by decision. The weight is
    #     a_t = -v_t + (1 - lam) * sum_{k>=1} lam^(k-1) v_{t+k} + lam^(later decisions) * z,
    # so `result_share_of_target` is the mean of lam^(later decisions), and
    # a_t - (z - v_t) = -sum_{k>=1} (1 - lam^k) (v_{t+k+1} - v_{t+k}): the later
    # changes of the estimate that lambda leaves out. Were the estimate
    # exact, those changes would average zero after any decision; their mean
    # by kind of decision, taken within the side's turn, is the part of the
    # lambda weight that differs from today's by what was decided.
    remaining = d["n"] - 1 - d["index"]
    turn_key = np.unique(d["gside"].astype(np.int64) * 4096 + d["turn"], return_inverse=True)[1]
    turn_size = np.bincount(turn_key).astype(float)

    def within_turn(x: np.ndarray) -> np.ndarray:
        return x - (np.bincount(turn_key, weights=x) / turn_size)[turn_key]

    tenth = np.minimum((10 * d["index"] / d["n"]).astype(int), 9)
    by_kind = {name: d["kind"] == k for k, name in KINDS.items()}
    res["today_by_kind"] = {name: mean_se(today, m) for name, m in by_kind.items()}
    res["today_by_kind_within_turn"] = {name: mean_se(within_turn(today), m) for name, m in by_kind.items()}
    res["by_lambda"] = {}
    for lam in PER_DECISION[1:]:
        a = weight(lam, False)
        left_out = a - today
        res["by_lambda"][str(lam)] = {
            **summary(a),
            "result_share_of_target": float((lam ** remaining).mean()),
            "mean_square_over_todays_by_tenth": [
                float((a[tenth == i] ** 2).mean() / (today[tenth == i] ** 2).mean()) for i in range(10)],
            "share_of_power_in_first_three_tenths": float((a[progress_of(d) < 0.3] ** 2).sum() / (a ** 2).sum()),
            "difference_from_todays_by_kind": {name: mean_se(left_out, m) for name, m in by_kind.items()},
            "difference_from_todays_by_kind_within_turn": {
                name: mean_se(within_turn(left_out), m) for name, m in by_kind.items()}}
    # how the weight's power is spread along the game under each
    progress = d["index"] / d["n"]
    res["share_of_power_in_first_three_tenths"] = {
        "today": float((today[progress < 0.3] ** 2).sum() / (today ** 2).sum()),
        **{f"per_turn_{lam}": float((weight(lam, True)[progress < 0.3] ** 2).sum() / (weight(lam, True) ** 2).sum())
           for lam in (0.8, 0.5)}}
    args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
