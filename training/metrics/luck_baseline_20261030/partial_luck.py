"""Does the luck dealt before a decision improve the value head's
per-decision baseline? (stage 1's labels, docs/selfplay_t1_local_prereg_20261028.md
"Label noise"; design §1 step 2.)

The label R_{g,s} = sigma_s (z - b - beta^T S) subtracts the game's whole
dice luck S. The best per-decision baseline is E[R | h_t] = sigma_s (E[z | h_t]
- b - beta^T S_<t): the value head reads E[z | h_t] from a state that already
carries the luck dealt so far, S_<t (every attack whose command precedes
decision t's), while R has it subtracted. So beside -alpha v_t, a correction
-gamma^T x_t with x_t = sigma_s S_<t should take out the part of v_t that
is luck. Both are functions of the history before a_t, so the corrected
label stays unbiased (design §1.1).

Per game-side, the sufficient statistics of sum_t (R - alpha v_t - gamma^T x_t)^2
are summed (n, sum v, sum v^2, sum x, sum v x, sum x x^T); the coefficients
are fitted on one fold of games and read on the other (two folds by game
index parity), as tools/selfplay_readout.py's label_noise fits alpha and
kappa; E[R^2] a decision and its SE are the ratio over game totals.

    <cpu python> training/metrics/luck_baseline_20261030/partial_luck.py \
        --run C:/Users/momom2/wesnoth_runs/selfplay_t1_local --jobs 6 --out partial_luck.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

K = 7                                  # match_luck.COLUMNS


def side_stats(item: Tuple[str, str, int]) -> Dict:
    """One game's per-side sufficient statistics: {side: dict}."""
    from tools import game_record
    from tools.match_luck import COLUMNS, attack_luck
    from tools.pg_positions import read_record
    path, game, side_a = item
    rec = read_record(Path(path))
    luck_at: List[Tuple[int, np.ndarray]] = []           # (command index, luck in side A's frame)
    for k, cs, cmd in game_record.walk_core(rec, verify=False):
        if cmd[0] != "attack":
            continue
        a = attack_luck(cs, cmd, side_a)
        if a.terms is not None:
            luck_at.append((k, np.array([a.terms[c] for c in COLUMNS], dtype=float)))
    ks = np.array([k for k, _ in luck_at], dtype=np.int64)
    cum = np.cumsum(np.stack([v for _, v in luck_at]), axis=0) if luck_at else np.zeros((0, K))
    from wesnoth_ai.server_priors import KIND_ATTACK, KIND_END_TURN, KIND_RECRUIT
    out = {}
    for side in (1, 2):
        sign = 1.0 if side == side_a else -1.0
        n, sv, svv = 0, 0.0, 0.0
        sx, svx, sxx = np.zeros(K), np.zeros(K), np.zeros((K, K))
        sxr, svr = np.zeros((K, 3)), np.zeros(3)      # I-V: each correction against 1[a in X] - p(X)
        for d in rec["decisions"]:
            if d["side"] != side:
                continue
            v = float(d["v"]) if d["v"] is not None else 0.0
            j = int(np.searchsorted(ks, int(d["cmd"]), side="left"))     # attacks strictly before the command
            x = sign * cum[j - 1] if j > 0 else np.zeros(K)
            kind = int(d["action"][1])
            r = np.array([kind == KIND_END_TURN, kind == KIND_ATTACK, kind == KIND_RECRUIT], dtype=float) \
                - np.array([d["p_end"], d["p_att"], d["p_rec"]], dtype=float)
            n += 1
            sv += v
            svv += v * v
            sx += x
            svx += v * x
            sxx += np.outer(x, x)
            sxr += np.outer(x, r)
            svr += v * r
        out[side] = {"n": n, "sv": sv, "svv": svv, "sx": sx.tolist(), "svx": svx.tolist(), "sxx": sxx.tolist(),
                     "sxr": sxr.tolist(), "svr": svr.tolist()}
    return {"game": game, "sides": out}


def fit(rows: List[Dict], use_v: bool, use_x: bool) -> np.ndarray:
    """(alpha, gamma) minimizing the summed squared corrected label."""
    p = int(use_v) + (K if use_x else 0)
    if p == 0:
        return np.zeros(1 + K)
    A, y = np.zeros((1 + K, 1 + K)), np.zeros(1 + K)
    for r in rows:
        R = r["R"]
        A[0, 0] += r["svv"]
        A[0, 1:] += r["svx"]
        A[1:, 0] += r["svx"]
        A[1:, 1:] += r["sxx"]
        y[0] += R * r["sv"]
        y[1:] += R * r["sx"]
    keep = ([0] if use_v else []) + (list(range(1, 1 + K)) if use_x else [])
    coef = np.zeros(1 + K)
    coef[keep] = np.linalg.solve(A[np.ix_(keep, keep)] + 1e-9 * np.eye(len(keep)), y[keep])
    return coef


def squared(r: Dict, coef: np.ndarray) -> float:
    a, g = coef[0], coef[1:]
    R = r["R"]
    return float(r["n"] * R * R - 2 * R * (a * r["sv"] + g @ r["sx"]) + a * a * r["svv"]
                 + 2 * a * (g @ r["svx"]) + g @ r["sxx"] @ g)


def ratio_se(num: np.ndarray, den: np.ndarray) -> Tuple[float, float]:
    """sum num / sum den over games, and its delta-method SE."""
    m = num.sum() / den.sum()
    resid = num - m * den
    G = len(num)
    return float(m), float(np.sqrt(G / (G - 1) * (resid ** 2).sum()) / den.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--labels", type=Path, default=None)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="the first N games only (0: all)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    labels = args.labels or args.run / "labels.jsonl"
    R = {}
    for line in labels.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            R[(row["game"], int(row["side"]))] = float(row["R"])
    items = []
    for walk in sorted(args.run.glob("walk_s*.jsonl")):
        chunk = walk.stem.split("_")[1]
        for line in walk.read_text(encoding="utf-8").splitlines():
            if line.strip():
                w = json.loads(line)
                rec = args.run / f"games_{chunk}" / f"{w['game']}.game.jsonl.gz"
                if (w["game"], 1) in R and rec.exists():
                    items.append((str(rec), w["game"], int(w["side_a"])))
    if args.limit:
        items = items[:args.limit]
    t0 = time.time()
    with get_context("spawn").Pool(args.jobs) as pool:
        games = pool.map(side_stats, items, chunksize=16)
    seconds = time.time() - t0
    rows = []
    for i, g in enumerate(games):
        for side, s in g["sides"].items():
            rows.append({"game": g["game"], "fold": i % 2, "R": R[(g["game"], side)], "n": s["n"], "sv": s["sv"],
                         "svv": s["svv"], "sx": np.array(s["sx"]), "svx": np.array(s["svx"]),
                         "sxx": np.array(s["sxx"]), "sxr": np.array(s["sxr"]), "svr": np.array(s["svr"])})
    np.savez(args.out.with_suffix(".npz"), R=np.array([r["R"] for r in rows]), n=np.array([r["n"] for r in rows]),
             sv=np.array([r["sv"] for r in rows]), svv=np.array([r["svv"] for r in rows]),
             sx=np.stack([r["sx"] for r in rows]), svx=np.stack([r["svx"] for r in rows]),
             sxx=np.stack([r["sxx"] for r in rows]), sxr=np.stack([r["sxr"] for r in rows]),
             svr=np.stack([r["svr"] for r in rows]), fold=np.array([r["fold"] for r in rows]))
    beta = None
    fits = args.run / "fits.json"
    if fits.exists():
        # The one-coefficient form along the labels' own beta (the training games' fit): x_t -> beta^T x_t.
        beta = np.array(json.loads(fits.read_text(encoding="utf-8"))["fits"]["train"]["beta"], dtype=float)
    n_game = np.array([r["n"] for r in rows]).reshape(-1, 2).sum(axis=1)
    result = {"games": len(games), "decisions": int(n_game.sum()), "seconds": seconds}
    luck = np.array([squared(r, np.zeros(1 + K)) for r in rows]).reshape(-1, 2).sum(axis=1)
    result["luck"] = dict(zip(("E", "se"), ratio_se(luck, n_game)))
    for name, use_v, use_x in (("v", True, False), ("x", False, True), ("v_and_x", True, True)):
        q = np.zeros(len(rows))
        coefs = []
        for f in (0, 1):
            coef = fit([r for r in rows if r["fold"] != f], use_v, use_x)
            coefs.append(coef.tolist())
            for i, r in enumerate(rows):
                if r["fold"] == f:
                    q[i] = squared(r, coef)
        qg = q.reshape(-1, 2).sum(axis=1)
        E, se = ratio_se(qg, n_game)
        cut, cut_se = ratio_se(luck - qg, n_game)
        result[name] = {"E": E, "se": se, "cut": cut, "cut_se": cut_se, "share": cut / result["luck"]["E"],
                        "coefficients_by_fold": coefs}
    if beta is not None:
        # beta^T x_t as one column: its sums are beta-projections of x's.
        proj = [dict(r, sx=np.r_[beta @ r["sx"], np.zeros(K - 1)], svx=np.r_[beta @ r["svx"], np.zeros(K - 1)],
                     sxx=np.pad(np.array([[beta @ r["sxx"] @ beta]]), ((0, K - 1), (0, K - 1)))) for r in rows]
        q = np.zeros(len(rows))
        coefs = []
        for f in (0, 1):
            coef = fit([r for r in proj if r["fold"] != f], True, True)
            coefs.append(coef[:2].tolist())
            for i, r in enumerate(proj):
                if r["fold"] == f:
                    q[i] = squared(r, coef)
        qg = q.reshape(-1, 2).sum(axis=1)
        E, se = ratio_se(qg, n_game)
        cut, cut_se = ratio_se(luck - qg, n_game)
        result["v_and_beta_x"] = {"E": E, "se": se, "cut": cut, "cut_se": cut_se, "share": cut / result["luck"]["E"],
                                  "coefficients_by_fold": coefs}
        # I-V (design §5): each correction's per-side-game sum against 1[a in X] - p(X), mean 0 within the
        # gates' z (3.45) when it is read before its event.
        iv = {}
        for name, per_side in (("v", np.stack([r["svr"] for r in rows])),
                               ("beta_x", np.stack([beta @ r["sxr"] for r in rows]))):
            g = per_side.reshape(-1, 2, 3).sum(axis=1)               # by game, as the readout clusters
            mean, se = g.mean(axis=0), g.std(axis=0, ddof=1) / np.sqrt(len(g))
            iv[name] = {c: {"mean": float(m), "se": float(s), "z": float(m / s)}
                        for c, m, s in zip(("end_turn", "attack", "recruit"), mean, se)}
        result["i_v"] = iv
    v, vx = result["v"], result["v_and_x"]
    paired, paired_se = ratio_se(
        np.array([squared(r, np.array(v["coefficients_by_fold"][r["fold"]])) for r in rows]).reshape(-1, 2).sum(1)
        - np.array([squared(r, np.array(vx["coefficients_by_fold"][r["fold"]])) for r in rows]).reshape(-1, 2).sum(1),
        n_game)
    result["x_over_v"] = {"cut": paired, "cut_se": paired_se, "share_of_v": paired / v["E"]}
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()
                                                                if kk != "coefficients_by_fold"})
                      for k, v in result.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
