"""The coin plant and the dice control: the checks of the coin labels, by
code separate from the label code (tools/coin_labels.py;
docs/training_design_20261019/current_design.md 3.7). numpy and the
standard library only.

The coin plant. From the game records, parsed here and not by the label
code, a planted result

    z' = z + sum over the game's band decisions j of s_j (D_j - P_j) x_j,
    x_j = delta * r_j * 1[pair type of j == the chosen type],

with s_j the mover's sign against player A, D_j the coin, P_j its
propensity, r_j a sign (+1 or -1) drawn from SHA-256 of (salt, game, side,
decision index) and delta the plant's size. The label code then labels the
same games with z' in place of z, and T'_i - T_i = s_i omega_i (z' - z).
Its slope on x_i reads 1 for each mover side: the j = i term has
expectation s_i^2 E[omega_i (D_i - P_i)] x_i = x_i, the other terms have
expectation 0 (D_i is independent of D_j and of x_j). A label code that
reads the mover's sign wrong reads -1; one that reads the coin of the next
decision reads 0, since r_i and r_(i+1) are independent signs whatever the
pair types' sequence. z cancels in T' - T, so the plant cannot see a result
read with the wrong orientation or joined to the wrong game: the dice
control can.

The dice control. The slope of z, as the label code reads it, on side A's
hit-point luck sum S_hp (tools/match_luck.py's walk), fitted here: measured
+0.00416 +- 0.00009 over 23 player pairs and 18,800 games, between-pair sd
at most 0.00049 (analysis/round8_invent_chance_ledger.py Parts B and E, read
by round8_synthesis_power.py section 2). Its window at n games is that mean
+- 3 sqrt(0.00009^2 + 0.00049^2 + se_n^2), se_n = 0.00032 sqrt(1600 / n)
(own1 self-play's SE at 1,600 games): [+0.00245, +0.00587] at 2,000 games,
[+0.00268, +0.00565] at 100,000. A flipped result reads about -0.00416; a
share f of results joined to the wrong game reads (1 - f) x 0.00416.

    python tools/coin_plant.py plant --match NAME=PATH [--match ...] --out plant.jsonl
        [--delta 1.0] [--pair-type auto|KIND>KIND] [--salt coin_plant]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.match_luck import done_keys, match_items, parse_record, trim_partial_row  # noqa: E402

log = logging.getLogger("coin_plant")

# The dice control's measured inputs (see the module docstring).
DICE_MEAN = 0.0041627
DICE_MEAN_SE = 0.0000906
DICE_TAU = 0.0004852
DICE_SE_1600 = 0.00032
DICE_LINE_SD = 3.0


# ---------------------------------------------------------------------
# The plant, from the records
# ---------------------------------------------------------------------
def plant_sign(salt: str, game: str, side: int, index: int) -> int:
    """r_j: +1 or -1, a pure function of (salt, game, side, index)."""
    digest = hashlib.sha256(f"{salt}:{game}:{int(side)}:{int(index)}".encode("utf-8")).digest()
    return 1 if digest[0] & 1 else -1


def record_pair(entry: dict) -> str:
    """A logged band decision's ordered pair type, read off the record."""
    return f"{entry['a0']['kind']}>{entry['a1']['kind']}"


def planted_row(match: str, game: str, rec: dict, delta: float, pair: str, salt: str) -> dict:
    """One game's planted result and planted x's: z' and, for each band
    decision of the chosen type, [side, decision index, x]."""
    side_a = int(rec["players"]["a"]["side"])
    winner = int(rec["winner"])
    z = 0 if winner == 0 else (1 if winner == side_a else -1)
    shift, xs = 0.0, []
    for e in (rec.get("coins") or {}).get("decisions", []):
        if record_pair(e) != pair:
            continue
        x = delta * plant_sign(salt, game, e["side"], e["i"])
        sign = 1 if int(e["side"]) == side_a else -1
        shift += sign * (int(e["d"]) - float(e["p"])) * x
        xs.append([int(e["side"]), int(e["i"]), x])
    return {"match": match, "game": game, "z": z, "z_plant": z + shift, "pair": pair, "delta": delta, "x": xs}


def _records(sources: Sequence[Tuple[str, Path]]):
    for match, path in sources:
        for item in match_items(match, path):
            if item.record is not None:
                yield match, item.game, item.record


def most_frequent_pair(sources: Sequence[Tuple[str, Path]]) -> str:
    """The ordered pair type of most band decisions in these records."""
    counts: Counter = Counter()
    for _match, game, data in _records(sources):
        for e in (parse_record(data, game).get("coins") or {}).get("decisions", []):
            counts[record_pair(e)] += 1
    if not counts:
        raise ValueError("no band decision in these records")
    return counts.most_common(1)[0][0]


def plant_rows(sources: Sequence[Tuple[str, Path]], *, delta: float, pair: str, salt: str,
               skip: frozenset = frozenset()) -> List[dict]:
    """A planted row per recorded game, those in `skip` left out; a record
    that cannot be read is logged and left out (its game keeps z, x = 0)."""
    out = []
    for match, game, data in _records(sources):
        if (match, game) in skip:
            continue
        try:
            out.append(planted_row(match, game, parse_record(data, game), delta, pair, salt))
        except Exception as e:                            # noqa: BLE001 -- the game is left out, named
            log.warning("%s %s: %s: %s", match, game, type(e).__name__, e)
    return out


def plant_matches(sources: Sequence[Tuple[str, Path]], out: Path, *, delta: float, pair: str,
                  salt: str) -> int:
    """Append a planted row per game to `out`, the games already there
    skipped; returns the number written."""
    trim_partial_row(out)
    rows = plant_rows(sources, delta=delta, pair=pair, salt=salt, skip=frozenset(done_keys(out)))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    return len(rows)


# ---------------------------------------------------------------------
# Reading the plant through the label code's table
# ---------------------------------------------------------------------
def planted_inputs(table, plant_rows: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """(z' per game of the label code's table, x per band decision of the
    table): a game without a planted row keeps z and x = 0; x is joined on
    the decision's (game, side, decision index)."""
    by_key = {(r["match"], r["game"]): r for r in plant_rows}
    z_plant = table.z.astype(float).copy()
    games, sides, indices, values = [], [], [], []
    missing = 0
    for gi, key in enumerate(table.keys):
        row = by_key.get(key)
        if row is None:
            missing += 1
            continue
        if int(row["z"]) != int(table.z[gi]):
            raise ValueError(f"{key}: the plant reads z {row['z']}, the label code {table.z[gi]}")
        z_plant[gi] = float(row["z_plant"])
        for side, index, x in row["x"]:
            games.append(gi)
            sides.append(int(side))
            indices.append(int(index))
            values.append(float(x))
    if missing:
        log.warning("%d of %d games have no planted row: they keep z and x = 0", missing, len(table.keys))
    x = np.zeros(len(table.g))
    if values:
        span = int(max(max(indices), int(table.index.max()) if len(table.index) else 0)) + 1
        planted = (np.asarray(games) * 4 + np.asarray(sides)) * span + np.asarray(indices)
        order = np.argsort(planted)
        planted, vals = planted[order], np.asarray(values)[order]
        wanted = (table.g.astype(np.int64) * 4 + table.side) * span + table.index
        at = np.clip(np.searchsorted(planted, wanted), 0, len(planted) - 1)
        hit = planted[at] == wanted
        x[hit] = vals[at[hit]]
    return z_plant, x


def plant_reading(T: np.ndarray, T_plant: np.ndarray, x: np.ndarray, s: np.ndarray, g: np.ndarray,
                  keep: np.ndarray) -> dict:
    """The slope of T' - T on x per mover side ('a': the mover is player A),
    over the decisions `keep` selects, its standard error clustered by game;
    predicted 1."""
    from tools.coin_labels import clustered_slope
    y = T_plant - T
    out = {}
    for side, mask in (("a", keep & (s > 0)), ("b", keep & (s < 0))):
        out[side] = clustered_slope(y[mask], x[mask], g[mask])
        out[side]["planted"] = int(np.count_nonzero(x[mask]))
    return out


# ---------------------------------------------------------------------
# The dice control
# ---------------------------------------------------------------------
def dice_window(n: int) -> Tuple[float, float]:
    """The dice control's window at n decisive games."""
    se_n = DICE_SE_1600 * math.sqrt(1600.0 / max(1, n))
    half = DICE_LINE_SD * math.sqrt(DICE_MEAN_SE ** 2 + DICE_TAU ** 2 + se_n ** 2)
    return DICE_MEAN - half, DICE_MEAN + half


def dice_control(z: np.ndarray, s_hp: np.ndarray) -> dict:
    """The least-squares slope of z on S_hp (an intercept beside it) over
    decisive games, its standard error, and whether it lies in its window."""
    z = np.asarray(z, dtype=float)
    x = np.asarray(s_hp, dtype=float) - (np.mean(s_hp) if len(s_hp) else 0.0)
    n = len(z)
    lo, hi = dice_window(n)
    if n < 3 or float(x @ x) <= 0.0:
        return {"slope": None, "se": None, "n": n, "window": [lo, hi], "pass": False}
    y = z - z.mean()
    b = float(x @ y / (x @ x))
    res = y - b * x
    se = math.sqrt(float(res @ res) / (n - 2) / float(x @ x))
    return {"slope": b, "se": se, "n": n, "window": [lo, hi], "pass": bool(lo <= b <= hi)}


def _source(text: str) -> Tuple[str, Path]:
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"--match takes NAME=PATH, not {text!r}")
    return name, Path(path)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plant", help="the planted result of every recorded game")
    p.add_argument("--match", type=_source, action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--delta", type=float, default=1.0)
    p.add_argument("--pair-type", default="auto",
                   help="the ordered pair type planted (e.g. move>move); auto: the commonest in these records")
    p.add_argument("--salt", default="coin_plant")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    pair = most_frequent_pair(args.match) if args.pair_type == "auto" else args.pair_type
    n = plant_matches(args.match, args.out, delta=args.delta, pair=pair, salt=args.salt)
    log.info("%d planted rows written to %s (pair type %s, delta %g)", n, args.out, pair, args.delta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
