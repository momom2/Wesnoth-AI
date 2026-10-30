"""The labels' per-decision corrections (docs/training_design_20261019/
current_design.md §1 step 2, §4, §5), fitted on held-out games and frozen
for the pass (tools/pg_pass.py --corrections):

    R_t = R_{g,s} - alpha v_t - gamma beta^T x_t

v_t is the value head's logged reading at decision t, the deciding side's
frame. x_t is the dice luck dealt before the decision, its side's frame:
tools/match_luck.attack_luck's terms summed over the attacks among the
commands before the decision's slot (tools/pg_positions.decision_slots). Its
own attack is never in it. beta is the labels' luck coefficients (fits.json,
the training labels' fit), so one coefficient gamma prices the luck that R
subtracts and that v_t already sees. On stage 1 it read -0.87/-0.90 by
fold, where the theory says -1 (training/metrics/luck_baseline_20261030).

The design's rule (§4): a correction enters iff it cuts E[R^2] a decision by
2 SE and by at least 5 % of the luck-only value. v first; then the luck term
over v, its cut read as the increment over v. Both readings are
cross-fitted: the coefficients are fitted on one fold of games and read on
the other (two folds by game). A correction that enters must also pass I-V
(§5): its per-game sum against 1[a in X] - p(X), for X end_turn, attack and
recruit, has mean 0 within the gates' z. One that does not enter, or fails
I-V, gets coefficient 0. The coefficients the pass uses are refitted on
every held-out game.

    python tools/pg_corrections.py --games DIR [DIR ...] --labels labels.jsonl --fits fits.json \\
        --out corrections.json --workers 10
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

CORRECTION_SE = 2.0           # design §4: a correction enters iff it cuts E[R^2] by this many SE ...
CORRECTION_SHARE = 0.05       # ... and by at least this share of the luck-only value
GATE_Z = 3.452432937387534    # tools/selfplay_readout.py GATE_Z: the gates' family-wise z
CLASSES = ("end_turn", "attack", "recruit")


def side_sums(path: str, beta: Sequence[float]) -> Dict[int, Dict[str, object]]:
    """One record's per-side sufficient statistics: n, sums of v, v^2, y, v y,
    y^2 (y = beta^T x), and the I-V sums of v and y against 1[a in X] - p(X).
    A missing or non-finite v counts as 0, as the pass reads it
    (tools/pg_positions.finite_v), and is counted in `v_nonfinite`."""
    from tools import game_record
    from tools.match_luck import COLUMNS, attack_luck
    from tools.pg_positions import decision_slots, finite_v, read_record
    from wesnoth_ai.pg_records import decisions_of
    from wesnoth_ai.server_priors import KIND_ATTACK, KIND_END_TURN, KIND_RECRUIT
    rec = read_record(Path(path))
    b = np.asarray(beta, dtype=float)
    attacks: List[Tuple[int, float]] = []              # (command index, beta^T luck in side 1's frame)
    for k, cs, cmd in game_record.walk_core(rec, verify=False):
        if cmd[0] == "attack":
            terms = attack_luck(cs, list(cmd), 1).terms
            if terms is not None:
                attacks.append((k, float(b @ np.array([terms[c] for c in COLUMNS]))))
    ks = np.array([k for k, _ in attacks], dtype=np.int64)
    cum = np.cumsum([y for _, y in attacks]) if attacks else np.zeros(0)
    decisions = decisions_of(rec)
    rejections = [(int(k), int(x), int(y)) for k, x, y in rec.get("rejections", ())]
    slots = decision_slots(decisions, rejections, len(rec["commands"]))
    out: Dict[int, Dict[str, object]] = {s: {"n": 0, "v": 0.0, "vv": 0.0, "y": 0.0, "vy": 0.0, "yy": 0.0,
                                             "iv_v": np.zeros(3), "iv_y": np.zeros(3), "v_nonfinite": 0}
                                         for s in (1, 2)}
    bad = Counter()
    for d, (k, _r) in zip(decisions, slots):
        j = int(np.searchsorted(ks, k, side="left"))          # the attacks among the commands before the slot
        y = (cum[j - 1] if j > 0 else 0.0) * (1.0 if d.side == 1 else -1.0)
        v = finite_v(d.v, bad)
        kind = d.kind
        r = np.array([kind == KIND_END_TURN, kind == KIND_ATTACK, kind == KIND_RECRUIT], dtype=float) \
            - np.array([d.p_end, d.p_att, d.p_rec], dtype=float)
        s = out[d.side]
        s["n"] += 1
        s["v"] += v
        s["vv"] += v * v
        s["y"] += y
        s["vy"] += v * y
        s["yy"] += y * y
        s["iv_v"] = s["iv_v"] + v * r
        s["iv_y"] = s["iv_y"] + y * r
        s["v_nonfinite"] += bad.pop("v_nonfinite", 0)
    return out


def _task(args: Tuple[str, str, Tuple[float, ...]]) -> Tuple[str, Dict]:
    game, path, beta = args
    return game, side_sums(path, beta)


def squared(row: Dict, alpha: float, gamma: float) -> float:
    """sum_t (R - alpha v_t - gamma y_t)^2 over a game-side's decisions."""
    R = row["R"]
    return (row["n"] * R * R - 2 * R * (alpha * row["v"] + gamma * row["y"]) + alpha * alpha * row["vv"]
            + 2 * alpha * gamma * row["vy"] + gamma * gamma * row["yy"])


def fit(rows: Sequence[Dict], use_v: bool, use_y: bool) -> Tuple[float, float]:
    """(alpha, gamma) minimizing the summed squared corrected label."""
    A = np.array([[sum(r["vv"] for r in rows), sum(r["vy"] for r in rows)],
                  [sum(r["vy"] for r in rows), sum(r["yy"] for r in rows)]])
    c = np.array([sum(r["R"] * r["v"] for r in rows), sum(r["R"] * r["y"] for r in rows)])
    keep = [i for i, used in enumerate((use_v, use_y)) if used]
    coef = np.zeros(2)
    if keep:
        coef[keep] = np.linalg.lstsq(A[np.ix_(keep, keep)], c[keep], rcond=None)[0]
    return float(coef[0]), float(coef[1])


def ratio_se(num: np.ndarray, den: np.ndarray) -> Tuple[float, float]:
    """sum num / sum den over games, and its delta-method SE."""
    m = num.sum() / den.sum()
    resid = num - m * den
    G = len(num)
    return float(m), float(math.sqrt(G / max(G - 1, 1) * float((resid ** 2).sum())) / den.sum())


def readings(rows: Sequence[Dict]) -> Dict:
    """Cross-fitted E[R^2] a decision under luck alone, v, and v with the
    luck term; rows of one game adjacent (side 1, side 2)."""
    n = np.array([r["n"] for r in rows], dtype=float).reshape(-1, 2).sum(axis=1)
    luck = np.array([squared(r, 0.0, 0.0) for r in rows]).reshape(-1, 2).sum(axis=1)
    out = {"luck": dict(zip(("E", "se"), ratio_se(luck, n)))}
    per_game = {}
    for name, use_v, use_y in (("v", True, False), ("v_and_luck", True, True), ("luck_term", False, True)):
        q = np.zeros(len(rows))
        coefs = []
        for f in (0, 1):
            coef = fit([r for r in rows if r["fold"] != f], use_v, use_y)
            coefs.append(coef)
            for i, r in enumerate(rows):
                if r["fold"] == f:
                    q[i] = squared(r, *coef)
        per_game[name] = q.reshape(-1, 2).sum(axis=1)
        E, se = ratio_se(per_game[name], n)
        cut, cut_se = ratio_se(luck - per_game[name], n)
        out[name] = {"E": E, "se": se, "cut": cut, "cut_se": cut_se, "share": cut / out["luck"]["E"],
                     "coefficients_by_fold": coefs}
    inc, inc_se = ratio_se(per_game["v"] - per_game["v_and_luck"], n)
    out["luck_over_v"] = {"cut": inc, "cut_se": inc_se, "share": inc / out["luck"]["E"]}
    return out


def enters(cut: float, cut_se: float, share: float) -> bool:
    return bool(cut >= CORRECTION_SE * cut_se and share >= CORRECTION_SHARE)


def i_v(rows: Sequence[Dict], key: str) -> Dict[str, Dict[str, float]]:
    """A correction's I-V: its per-game sum against 1[a in X] - p(X), mean
    and SE over games, by class."""
    g = np.stack([r[key] for r in rows]).reshape(-1, 2, 3).sum(axis=1)
    mean, se = g.mean(axis=0), g.std(axis=0, ddof=1) / math.sqrt(len(g)) if len(g) > 1 else np.full(3, np.inf)
    return {c: {"mean": float(m), "se": float(s), "z": float(m / s) if s > 0 else 0.0}
            for c, m, s in zip(CLASSES, mean, se)}


def decide(rows: Sequence[Dict]) -> Dict:
    """The corrections the pass uses, with the readings that decided them."""
    read = readings(rows)
    iv = {"v": i_v(rows, "iv_v"), "luck": i_v(rows, "iv_y")}
    iv_ok = {k: all(abs(x["z"]) <= GATE_Z for x in by.values()) for k, by in iv.items()}
    use_v = enters(read["v"]["cut"], read["v"]["cut_se"], read["v"]["share"]) and iv_ok["v"]
    luck = read["luck_over_v"] if use_v else read["luck_term"]
    use_y = enters(luck["cut"], luck["cut_se"], luck["share"]) and iv_ok["luck"]
    alpha, gamma = fit(rows, use_v, use_y)
    return {"alpha": alpha, "gamma": gamma, "enters": {"v": use_v, "luck": use_y}, "i_v_pass": iv_ok,
            "readings": read, "i_v": iv,
            "rule": {"correction_se": CORRECTION_SE, "correction_share": CORRECTION_SHARE, "gate_z": GATE_Z}}


def main(argv: Optional[Sequence[str]] = None) -> int:
    from tools.pg_positions import RecordIndex
    from wesnoth_ai.pg_records import read_labels
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--games", type=Path, nargs="+", required=True, help="the match directories")
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--fits", type=Path, required=True, help="tools/pg_labels.py's fits.json (beta: the train fit)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    beta = tuple(float(b) for b in json.loads(args.fits.read_text(encoding="utf-8"))["fits"]["train"]["beta"])
    rows_by_side = {(r.game, r.side): r for r in read_labels(args.labels) if r.heldout}
    games = sorted({g for g, _ in rows_by_side})
    index = RecordIndex(args.games)
    tasks = [(g, str(index.path(g)), beta) for g in games]
    if args.workers > 0:
        with get_context("spawn").Pool(args.workers) as pool:
            sums = dict(pool.map(_task, tasks, chunksize=16))
    else:
        sums = dict(_task(t) for t in tasks)
    rows = []
    for i, g in enumerate(games):
        for side in (1, 2):
            s = sums[g][side]
            rows.append({"game": g, "fold": i % 2, "R": rows_by_side[(g, side)].R, **s})
    result = decide(rows)
    result.update(beta=list(beta), games=len(games), decisions=int(sum(r["n"] for r in rows)),
                  v_nonfinite=int(sum(r.get("v_nonfinite", 0) for r in rows)),
                  fitted_on="the held-out games", labels=str(args.labels), fits=str(args.fits))
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=1, default=lambda x: x.tolist()) + "\n", encoding="utf-8")
    os.replace(tmp, args.out)
    r = result["readings"]
    print(f"PG_CORRECTIONS alpha {result['alpha']:.4g} gamma {result['gamma']:.4g} (enters {result['enters']}); "
          f"E[R^2] luck {r['luck']['E']:.4f}, v {r['v']['E']:.4f}, v and luck {r['v_and_luck']['E']:.4f} "
          f"on {result['games']} held-out games")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
