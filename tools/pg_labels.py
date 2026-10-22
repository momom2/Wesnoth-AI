"""The labels of a self-play run (docs/training_design_20261019/
current_design.md §1, step 2; build_contracts.md §2):

    R = z - b(cell) - beta . S

per game-side, in the side's frame (tools/selfplay_rows.py rows). b is the
shrunk map x factions x seat mean and beta the luck coefficients; both are
fitted on games other than the one labelled:

    training games   (index < heldout_from) by one fit on every held-out game;
    held-out games   half A by a fit on half B, half B by a fit on half A.

The fit (`fit_baseline`), on side rows with their luck sums:
  1. beta and an intercept: the least-squares fit of z on S
     (`match_luck_readout.fit_beta`, the fit of tools/fit_luck_beta.py).
     Capped games enter at z = 0, as they enter the labels: the fit is the
     best linear correction on the population it corrects, and dropping
     them would condition on an outcome. fit_luck_beta.py, written for
     argmax matches, keeps decisive games only.
  2. On the remainder y = z - intercept - beta . S, an additive baseline,
     map x seat plus the faction pair (own, opponent), by backfitting.
  3. Each cell's mean deviation from the additive baseline, shrunk by
     n / (n + k), with k the within-cell variance over the between-cell
     variance (one empirical-Bayes factor; round9_simplify_residual.py,
     whose out-of-sample residual share chose this form).
b(cell) = intercept + the additive terms + the shrunk deviation; a cell,
map x seat or faction pair the fit never saw contributes 0.

Self-play rows come in mirrored pairs (z, S and the cell's seat and
factions all swap with the side), so each fit is antisymmetric and a
game's two labels are opposite: R(side 1) = -R(side 2).

Splits: `split` is the game's index modulo 8 (the gradient pass's split
accumulators); a held-out game's `half` is "A" when (index // 2) is even,
so each half holds both seat assignments of run_elo_batch's alternation.

    python tools/pg_labels.py rows.jsonl --heldout-from 20000 --out labels.jsonl \\
        [--fits fits.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.match_luck import COLUMNS  # noqa: E402
from tools.match_luck_readout import fit_beta  # noqa: E402

SPLITS = 8
BACKFIT_SWEEPS = 20
# The between-cell variance's floor: a fit whose cells do not differ
# shrinks every cell to the additive baseline.
MIN_BETWEEN = 1e-9


def split_of(index: int) -> int:
    return int(index) % SPLITS


def half_of(index: int) -> str:
    """A held-out game's half: A or B by (index // 2) parity."""
    return "A" if (int(index) // 2) % 2 == 0 else "B"


def cell_parts(cell: str) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    """(map x seat, faction pair) of a "scenario|seat|own|opponent" cell."""
    scenario, seat, own, opp = cell.split("|")
    return (scenario, seat), (own, opp)


@dataclass
class BaselineFit:
    """b and beta as fitted on a set of side rows."""
    intercept: float
    beta: np.ndarray
    map_seat: Dict[Tuple[str, str], float] = field(default_factory=dict)
    pair: Dict[Tuple[str, str], float] = field(default_factory=dict)
    cell: Dict[str, float] = field(default_factory=dict)
    k: float = 0.0
    games: int = 0

    def b(self, cell: str) -> float:
        map_seat, pair = cell_parts(cell)
        return (self.intercept + self.map_seat.get(map_seat, 0.0) + self.pair.get(pair, 0.0)
                + self.cell.get(cell, 0.0))

    def beta_s(self, S: Sequence[float]) -> float:
        return float(np.asarray(S, dtype=float) @ self.beta)

    def summary(self) -> Dict:
        return {"columns": list(COLUMNS), "beta": [float(x) for x in self.beta],
                "intercept": self.intercept, "k": self.k, "games": self.games,
                "cells": len(self.cell)}


def group_means(keys: Sequence[Hashable], y: np.ndarray) -> Dict[Hashable, Tuple[float, int]]:
    """{key: (mean of y over its rows, its row count)}."""
    total: Dict[Hashable, float] = defaultdict(float)
    count: Dict[Hashable, int] = defaultdict(int)
    for k, v in zip(keys, y):
        total[k] += float(v)
        count[k] += 1
    return {k: (total[k] / count[k], count[k]) for k in total}


def additive_fit(y: np.ndarray, map_keys: Sequence, pair_keys: Sequence,
                 sweeps: int = BACKFIT_SWEEPS) -> Tuple[Dict, Dict]:
    """The two-way additive baseline y ~ a(map x seat) + f(faction pair),
    by backfitting from zero."""
    a: Dict = {}
    f: Dict = {}
    for _ in range(sweeps):
        a = {k: m for k, (m, _) in group_means(map_keys, y - np.array([f.get(k, 0.0) for k in pair_keys])).items()}
        f = {k: m for k, (m, _) in group_means(pair_keys, y - np.array([a[k] for k in map_keys])).items()}
    return a, f


def shrinkage_constant(dev: np.ndarray, cells: Sequence[str], means: Dict) -> float:
    """k = the within-cell variance over the between-cell variance. The
    within-cell variance is the residual sum of squares about the cell
    means over N - C degrees of freedom (N rows, C cells): over N it reads
    low by C / N, 0.15 at a held-out half's 6.6 rows a side-cell. The
    between-cell variance is the count-weighted mean square of the cell
    means less the within-cell variance's share of it, C / N."""
    N, C = len(dev), len(means)
    resid = dev - np.array([means[c][0] for c in cells])
    within = float(resid @ resid) / max(N - C, 1)
    ns = np.array([n for _, n in means.values()], dtype=float)
    ms = np.array([m for m, _ in means.values()])
    between = max(float(np.average(ms ** 2, weights=ns) - within * C / N), MIN_BETWEEN)
    return within / between


def shrunk_cells(dev: np.ndarray, cells: Sequence[str],
                 k: Optional[float] = None) -> Tuple[Dict[str, float], float]:
    """Each cell's mean of `dev` shrunk by n / (n + k), and k
    (`shrinkage_constant` when not given)."""
    means = group_means(cells, dev)
    if k is None:
        k = shrinkage_constant(dev, cells, means)
    return {c: m * n / (n + k) for c, (m, n) in means.items()}, float(k)


def fit_baseline(rows: Sequence[Dict], k: Optional[float] = None) -> BaselineFit:
    """b and beta fitted on `rows` (side rows with S); `k`, the cells'
    shrinkage constant, is estimated from them when not given."""
    if not rows:
        raise ValueError("a baseline needs rows to fit on")
    z = np.array([float(r["z"]) for r in rows])
    S = np.array([r["S"] for r in rows], dtype=float).reshape(len(rows), len(COLUMNS))
    beta = fit_beta(z, S)
    intercept = float(z.mean() - S.mean(axis=0) @ beta)
    y = z - intercept - S @ beta
    cells = [r["cell"] for r in rows]
    parts = [cell_parts(c) for c in cells]
    map_keys, pair_keys = [p[0] for p in parts], [p[1] for p in parts]
    a, f = additive_fit(y, map_keys, pair_keys)
    dev = y - np.array([a[m] for m in map_keys]) - np.array([f[p] for p in pair_keys])
    shrunk, k = shrunk_cells(dev, cells, k)
    return BaselineFit(intercept=intercept, beta=beta, map_seat=a, pair=f, cell=shrunk, k=k,
                       games=len({r["game"] for r in rows}))


def label(row: Dict, fit: BaselineFit, heldout_from: int) -> Dict:
    """The label row of contract §2 for one side row under `fit`."""
    b = fit.b(row["cell"])
    beta_s = fit.beta_s(row["S"])
    heldout = int(row["index"]) >= heldout_from
    return {"game": row["game"], "side": int(row["side"]), "split": split_of(row["index"]),
            "heldout": heldout, "half": half_of(row["index"]) if heldout else None,
            "z": float(row["z"]), "capped": bool(row["capped"]), "cell": row["cell"],
            "S": [float(x) for x in row["S"]], "b": b, "beta_S": beta_s,
            "R": float(row["z"]) - b - beta_s}


def zero_baseline() -> BaselineFit:
    """b = 0 and beta = 0: R = z. For tests on runs too small to fit them
    (a held-out half of a few games fits seven luck slopes exactly)."""
    return BaselineFit(intercept=0.0, beta=np.zeros(len(COLUMNS)))


def side_labels(rows: Sequence[Dict], heldout_from: int, zero: bool = False) -> Tuple[List[Dict], Dict]:
    """(the label rows of every walked side row, in the rows' order, the
    three fits' summaries and the rows left out). Training games are
    labelled by the fit on every held-out game, each held-out half by the
    fit on the other half; with `zero`, every fit is `zero_baseline`."""
    usable = [r for r in rows if "S" in r and "error" not in r]
    heldout = [r for r in usable if int(r["index"]) >= heldout_from]
    halves = {h: [r for r in heldout if half_of(r["index"]) == h] for h in ("A", "B")}
    if not halves["A"] or not halves["B"]:
        raise ValueError(f"both held-out halves need games (index >= {heldout_from}): "
                         f"A {len(halves['A'])} rows, B {len(halves['B'])}")
    fit = (lambda _rows: zero_baseline()) if zero else fit_baseline
    fits = {"train": fit(heldout), "A": fit(halves["B"]), "B": fit(halves["A"])}
    out = []
    for r in usable:
        which = half_of(r["index"]) if int(r["index"]) >= heldout_from else "train"
        out.append(label(r, fits[which], heldout_from))
    info = {"fits": {name: f.summary() for name, f in fits.items()},
            "fitted_on": ({"train": "zero", "A": "zero", "B": "zero"} if zero else
                          {"train": "every held-out game", "A": "held-out half B", "B": "held-out half A"}),
            "heldout_from": heldout_from, "rows": len(rows), "labelled": len(out),
            "left_out": sorted({r["game"] for r in rows if "S" not in r or "error" in r})}
    return out, info


def write_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    tmp.replace(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from tools.selfplay_rows import load_rows
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", type=Path, help="tools/selfplay_rows.py's side rows")
    ap.add_argument("--heldout-from", type=int, required=True, help="the first held-out game index")
    ap.add_argument("--out", type=Path, required=True, help="labels.jsonl")
    ap.add_argument("--fits", type=Path, default=None, help="the fits' summaries as JSON")
    ap.add_argument("--zero-baseline", action="store_true",
                    help="b = 0 and beta = 0 for every row (R = z): tests on runs too small to fit them")
    args = ap.parse_args(argv)
    labels, info = side_labels(load_rows(args.rows), args.heldout_from, zero=args.zero_baseline)
    write_jsonl(args.out, labels)
    if args.fits is not None:
        args.fits.write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{info['labelled']} game-sides labelled of {info['rows']}; {len(info['left_out'])} games left out "
          f"(no luck walk); beta (train) " + ", ".join(f"{c} {b:+.4g}" for c, b in
                                                    zip(COLUMNS, info["fits"]["train"]["beta"])))
    return 0 if not info["left_out"] else 1


if __name__ == "__main__":
    sys.exit(main())
