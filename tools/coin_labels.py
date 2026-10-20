"""The labels of the near-tie coins and of the attack coin, read off the
game records (docs/near_tie_coins_prereg_20261020.md; the design is
docs/training_design_20261019/current_design.md 3.2). numpy and the
standard library only.

The label. At band decision i of a game (tools/coin_decode.py), with z
side A's result (+1 a win, -1 a loss; a capped game is dropped and
counted), s_i the mover's sign (+1 when the mover is player A's side, -1
otherwise), D_i the coin, P_i its propensity and S the game's seven luck
sums (tools/match_luck.py's walk, side A's view, `match_luck.COLUMNS`),

    T_i = s_i * omega_i * (z - beta . S),   omega_i = (D_i - P_i) / (P_i (1 - P_i)),

with beta a fixed coefficient vector (`load_beta`; batch 1's comes from
tools/fit_luck_beta.py). The attack coin's label is the same formula with
the game's one coin: D = 1 puts the attack offset on player A, P = 1/2,
s = +1.

Claim: E[T_i | H_i] = Q(H_i, a1) - Q(H_i, a0), the mover's expected
result after playing the runner-up a1 less after playing the argmax a0,
both followed by the coin policy (the same player, coins included), for
any beta fixed before the game is read.

Proof. H_i is everything decided before coin i: the setup, the dice drawn,
the coins drawn and the memory. The band membership, P_i and s_i are
functions of H_i, and D_i is 1 with probability P_i independently of H_i
and of every later dice draw (coin_decode: distinct SHA-256 keys). (1) Given
H_i, D_i takes 1 with probability P_i, so
    E[omega_i z | H_i] = P_i (1 - P_i) / (P_i (1 - P_i)) E[z | H_i, D_i = 1]
                         - (1 - P_i) P_i / (P_i (1 - P_i)) E[z | H_i, D_i = 0]
                       = E[z | H_i, D_i = 1] - E[z | H_i, D_i = 0].
(2) S is the sum of the luck terms L_k of the game's fights. A fight before
decision i is in H_i, so its L_k is fixed given H_i while E[omega_i | H_i] =
0, and E[omega_i L_k | H_i] = 0. A fight after decision i has its luck term
of mean zero given F_k, the position and choices before its dice
(match_luck: L = Q(outcome) - the expectation of Q over the fight's exact
outcome distribution, or 0 decided on the position), and omega_i is a
function of F_k (D_i is drawn before the fight), so E[omega_i L_k | H_i] =
E[omega_i E[L_k | F_k] | H_i] = 0; whether fight k happens at all is a
function of F_k too, so the sum over a random number of fights keeps it.
Hence E[omega_i beta . S | H_i] = 0 for any fixed beta. (3) z is side A's
result and s_i turns it into the mover's: E[T_i | H_i] = s_i (E[z | H_i,
D_i = 1] - E[z | H_i, D_i = 0]) = Q(H_i, a1) - Q(H_i, a0). Dropping capped
games keeps it exactly when capping does not depend on the coins (the
I-coin check reads the coins within capped games). tests/test_coin_labels.py
checks the claim on a synthetic game whose decision effects are known,
for several beta, and that a variate read after the coin breaks it.

    python tools/coin_labels.py extract --match NAME=DIR [--match ...] --out coins.jsonl
    python tools/coin_labels.py readout --coins coins.jsonl [...] --rows rows.jsonl [...]
        --beta configs/luck_beta_20261019.json --band 0.35 [--plant plant.jsonl ...]
        [--save-json readout.json]

extract writes one row per game of a games directory or tarball (the
result files and their records, as run_elo_batch leaves them): its result,
its attack coin, its band decisions as columns, and the I-log count (every
band decision's logged choice against the command the record holds at its
index). A rerun skips the games already written. readout joins the rows
with the walk's rows (tools/match_luck.py walk) and prints the labels'
means with standard errors clustered by game (pooled, by mover side, by
0.05-nat margin bin, by ordered pair type), the attack coin's label and the
hooked player's Elo, and the checks: I-coin, I-log, the band decisions per
side-game, the dice control and the coin plant (tools/coin_plant.py).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.coin_decode import KIND_CODES  # noqa: E402
from tools.match_luck import COLUMNS, done_keys, match_items, parse_record, result_z, trim_partial_row  # noqa: E402

log = logging.getLogger("coin_labels")

KIND_NAMES = {code: name for name, code in KIND_CODES.items()}
BIN_WIDTH = 0.05
# A check's line in standard errors (the design's plant lines, 3.7).
CHECK_SE = 3.0
# The attack known answer at 6,400 games (current_design.md 2, round 8:
# analysis/round8_provenance_knob_checks.py): the hooked player's Elo and the
# attack coin's mean label.
ATTACK_ELO_WINDOW = (-111.0, -41.0)
ATTACK_LABEL_WINDOW = (-0.640, -0.236)


def pair_name(k0: int, k1: int) -> str:
    return f"{KIND_NAMES[int(k0)]}>{KIND_NAMES[int(k1)]}"


# ---------------------------------------------------------------------
# Extraction: one row per game
# ---------------------------------------------------------------------
def _hex(xs, ys, at: int) -> list:
    return [int(xs[at]), int(ys[at])]


def _attack_matches(cmd: list, want: dict) -> bool:
    return [int(cmd[3]), int(cmd[4])] == want["to"] and int(cmd[5]) == int(want["weapon"])


def played_status(entry: dict, commands: Sequence[list]) -> str:
    """Whether the record's command at the entry's index is the choice its
    coin made (a1 when D is 1, a0 otherwise): "matched", "mismatched", or
    "unresolved" when nothing was played (a refused recruit, an attack whose
    approach was cut short) or the index is unknown."""
    k = entry.get("cmd")
    if entry.get("refused") or k is None or not 0 <= int(k) < len(commands):
        return "unresolved"
    want = entry["a1"] if int(entry["d"]) else entry["a0"]
    cmd = commands[int(k)]
    kind = want["kind"]
    if kind == "end_turn":
        ok = cmd[0] == "end_turn"
    elif kind == "recruit":
        # The engine may place it on another castle hex (a hidden unit holds
        # the ordered one): the unit type names the choice.
        ok = cmd[0] == "recruit" and cmd[1] == want["unit"]
    elif kind == "move":
        ok = (cmd[0] == "move" and _hex(cmd[1], cmd[2], 0) == want["from"]
              and _hex(cmd[1], cmd[2], -1) == want["to"])
    elif cmd[0] == "attack":
        ok = [int(cmd[1]), int(cmd[2])] == want["from"] and _attack_matches(cmd, want)
    elif cmd[0] == "move" and _hex(cmd[1], cmd[2], 0) == want["from"]:
        # The simulator's approach move, then the attack.
        nxt = commands[int(k) + 1] if int(k) + 1 < len(commands) else None
        if nxt is None or nxt[0] != "attack":
            return "unresolved"
        ok = _attack_matches(nxt, want)
    else:
        ok = False
    return "matched" if ok else "mismatched"


def extract_game(match: str, game: str, result: dict, rec: dict) -> dict:
    """One game's row: its result, its attack coin, its band decisions as
    columns and the I-log count."""
    side_a = int(rec.get("players", {}).get("a", {}).get("side", result.get("side_a")))
    z = result_z(result.get("outcome_a"))
    if z and (int(rec["winner"]) == side_a) != (z == 1):
        raise ValueError(f"{game}: the record's winner {rec['winner']} contradicts outcome_a {result.get('outcome_a')}")
    coins = rec.get("coins") or {}
    entries = coins.get("decisions", [])
    status = {"matched": 0, "mismatched": 0, "unresolved": 0}
    for e in entries:
        status[played_status(e, rec["commands"])] += 1
    band = {"side": [int(e["side"]) for e in entries], "i": [int(e["i"]) for e in entries],
            "gap": [float(e["gap"]) for e in entries], "p": [float(e["p"]) for e in entries],
            "d": [int(e["d"]) for e in entries],
            "k0": [KIND_CODES[e["a0"]["kind"]] for e in entries],
            "k1": [KIND_CODES[e["a1"]["kind"]] for e in entries],
            "refused": [int(bool(e.get("refused"))) for e in entries]}
    attack = coins.get("attack")
    return {"match": match, "game": game, "seed": result.get("seed"), "side_a": side_a, "z": z,
            "decisive": z != 0, "ended_by": result.get("ended_by"),
            "attack": None if attack is None else {"d": int(attack["d"]), "p": float(attack["p"])},
            "bands": coins.get("bands", {}), "band": band, "i_log": status}


def extract_matches(sources: Sequence[Tuple[str, Path]], out: Path) -> int:
    """Append a row per game of each match to `out`, the games already
    there skipped; returns the number of rows written. A game without a
    record, or whose record cannot be read, is logged and left out."""
    trim_partial_row(out)
    done = done_keys(out)
    written = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as fh:
        for match, path in sources:
            for item in match_items(match, path):
                if (match, item.game) in done:
                    continue
                if item.record is None:
                    log.warning("%s %s: no record", match, item.game)
                    continue
                try:
                    row = extract_game(match, item.game, item.result, parse_record(item.record, item.game))
                except Exception as e:                    # noqa: BLE001 -- the game is left out, named
                    log.warning("%s %s: %s: %s", match, item.game, type(e).__name__, e)
                    continue
                fh.write(json.dumps(row, separators=(",", ":")) + "\n")
                written += 1
    return written


# ---------------------------------------------------------------------
# The label table
# ---------------------------------------------------------------------
def mover_sign(side: np.ndarray, side_a: np.ndarray) -> np.ndarray:
    """+1 where the mover's side is player A's, -1 otherwise."""
    return np.where(np.asarray(side) == np.asarray(side_a), 1, -1)


def coin_outcomes(band: dict) -> np.ndarray:
    """A game's coins D, in the order its decisions were played."""
    return np.asarray(band["d"], dtype=float)


@dataclass
class Table:
    """The games read (decisive or not) and their band decisions, flat."""
    keys: List[Tuple[str, str]]
    z: np.ndarray              # (G,) side A's result, 0 for a capped game
    S: np.ndarray              # (G, 7) the luck sums, side A's view
    side_a: np.ndarray         # (G,)
    attack_d: np.ndarray       # (G,) the attack coin, nan without one
    attack_p: np.ndarray
    i_log: Dict[str, int]
    g: np.ndarray              # (N,) the decision's game
    side: np.ndarray
    s: np.ndarray
    index: np.ndarray
    gap: np.ndarray
    p: np.ndarray
    d: np.ndarray
    k0: np.ndarray
    k1: np.ndarray
    refused: np.ndarray
    counts: Dict[str, int]

    @property
    def decisive(self) -> np.ndarray:
        return self.z != 0


def load_jsonl(paths: Sequence[Path]) -> List[dict]:
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    return rows


def build_table(coin_rows: Sequence[dict], luck_rows: Sequence[dict]) -> Table:
    """The coin rows joined with the walk's rows on (match, game). A game
    the walk has not read, or read with an error, is left out and counted;
    a result that differs between the two rows refuses the join."""
    luck = {(r["match"], r["game"]): r for r in luck_rows}
    counts = {"coin_rows": len(coin_rows), "unwalked": 0, "walk_errors": 0}
    keys, z, S, side_a, att_d, att_p = [], [], [], [], [], []
    cols = {k: [] for k in ("g", "side", "s", "index", "gap", "p", "d", "k0", "k1", "refused")}
    i_log = {"matched": 0, "mismatched": 0, "unresolved": 0}
    for row in coin_rows:
        key = (row["match"], row["game"])
        lr = luck.get(key)
        if lr is None:
            counts["unwalked"] += 1
            continue
        if "error" in lr:
            counts["walk_errors"] += 1
            continue
        if int(lr["z"]) != int(row["z"]):
            raise ValueError(f"{key}: the walk reads z {lr['z']}, the coin row {row['z']}: a join or result error")
        gi = len(keys)
        keys.append(key)
        z.append(int(row["z"]))
        S.append([float(lr["S"][c]) for c in COLUMNS])
        side_a.append(int(row["side_a"]))
        attack = row.get("attack")
        att_d.append(np.nan if attack is None else float(attack["d"]))
        att_p.append(np.nan if attack is None else float(attack["p"]))
        for k, v in row.get("i_log", {}).items():
            i_log[k] = i_log.get(k, 0) + int(v)
        band = row["band"]
        n = len(band["d"])
        cols["g"].append(np.full(n, gi))
        cols["side"].append(np.asarray(band["side"], dtype=int))
        cols["s"].append(mover_sign(np.asarray(band["side"], dtype=int), np.full(n, int(row["side_a"]))))
        cols["index"].append(np.asarray(band["i"], dtype=int))
        cols["gap"].append(np.asarray(band["gap"], dtype=float))
        cols["p"].append(np.asarray(band["p"], dtype=float))
        cols["d"].append(coin_outcomes(band))
        cols["k0"].append(np.asarray(band["k0"], dtype=int))
        cols["k1"].append(np.asarray(band["k1"], dtype=int))
        cols["refused"].append(np.asarray(band.get("refused", [0] * n), dtype=int))
    flat = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}
    return Table(keys=keys, z=np.asarray(z, dtype=float), S=np.asarray(S, dtype=float).reshape(len(keys), len(COLUMNS)),
                 side_a=np.asarray(side_a, dtype=int), attack_d=np.asarray(att_d, dtype=float),
                 attack_p=np.asarray(att_p, dtype=float), i_log=i_log,
                 g=flat["g"].astype(int), side=flat["side"].astype(int), s=flat["s"].astype(float),
                 index=flat["index"].astype(int), gap=flat["gap"], p=flat["p"], d=flat["d"],
                 k0=flat["k0"].astype(int), k1=flat["k1"].astype(int), refused=flat["refused"].astype(int),
                 counts=counts)


def load_beta(path: Path) -> np.ndarray:
    """The luck coefficients of a beta file (tools/fit_luck_beta.py), in
    `match_luck.COLUMNS` order; a file over other columns is refused."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if list(data["columns"]) != list(COLUMNS):
        raise ValueError(f"{path}: columns {data['columns']}, the walk's are {list(COLUMNS)}")
    return np.asarray(data["beta"], dtype=float)


def residuals(table: Table, beta: np.ndarray, z: Optional[np.ndarray] = None) -> np.ndarray:
    """R_g = z - beta . S per game (`z` in place of the table's when given)."""
    return (table.z if z is None else z) - table.S @ beta


def decision_labels(table: Table, beta: np.ndarray, z: Optional[np.ndarray] = None) -> np.ndarray:
    """T_i = s_i omega_i R_g for every band decision of the table."""
    omega = (table.d - table.p) / (table.p * (1.0 - table.p))
    return table.s * omega * residuals(table, beta, z)[table.g]


def attack_labels(table: Table, beta: np.ndarray) -> np.ndarray:
    """The attack coin's label per game, nan without one: omega (z - beta . S)
    with s = +1 (D = 1 is the offset on player A)."""
    omega = (table.attack_d - table.attack_p) / (table.attack_p * (1.0 - table.attack_p))
    return omega * residuals(table, beta)


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------
def clustered_mean(values: np.ndarray, groups: np.ndarray) -> dict:
    """The mean of `values` with its standard error clustered by `groups`
    (the CR1 sandwich: sum over groups of the squared within-group sum of
    deviations, times G / (G - 1), over N squared)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return {"mean": None, "se": None, "n": 0, "games": 0}
    m = float(values.mean())
    _, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=values - m)
    k = len(sums)
    se = math.sqrt(k / (k - 1) * float(np.sum(sums ** 2))) / n if k > 1 else None
    return {"mean": m, "se": se, "n": int(n), "games": int(k)}


def clustered_slope(y: np.ndarray, x: np.ndarray, groups: np.ndarray) -> dict:
    """The least-squares slope of y on x (an intercept beside it) with its
    standard error clustered by `groups` (CR1)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    xc = x - x.mean() if len(x) else x
    sxx = float(np.sum(xc ** 2))
    if len(x) < 3 or sxx <= 0.0:
        return {"slope": None, "se": None, "n": int(len(x))}
    b = float(np.sum(xc * (y - y.mean())) / sxx)
    e = (y - y.mean()) - b * xc
    _, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=xc * e)
    k = len(sums)
    se = math.sqrt(k / (k - 1) * float(np.sum(sums ** 2))) / sxx if k > 1 else None
    return {"slope": b, "se": se, "n": int(len(x)), "games": int(k)}


def normal_two_sided_p(zstat: float) -> float:
    return math.erfc(abs(zstat) / math.sqrt(2.0))


def chi2_upper_p(x: float, df: int) -> float:
    """P(chi-square on df degrees of freedom > x), by the Wilson-Hilferty
    normal approximation (exact enough for a check's line: within 0.01 of
    the exact tail for df >= 3 at the tails a line reads)."""
    if df <= 0:
        return 1.0
    zstat = ((x / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
    return 0.5 * math.erfc(zstat / math.sqrt(2.0))


def coin_frequency(d: np.ndarray, p: np.ndarray) -> dict:
    """The coins landed 1 against their propensities: count, expectation,
    the binomial chi-square on 1 degree of freedom and its p-value."""
    n = len(d)
    ones, expect, var = float(np.sum(d)), float(np.sum(p)), float(np.sum(p * (1.0 - p)))
    chi2 = (ones - expect) ** 2 / var if var > 0 else 0.0
    return {"n": int(n), "ones": int(ones), "expected": expect, "chi2": chi2,
            "p_value": normal_two_sided_p(math.sqrt(chi2)) if var > 0 else 1.0}


def elo_of_mean(m: float, se: float) -> Tuple[float, float]:
    """The Elo difference of a mean result m in [-1, 1] (score (1 + m) / 2),
    its standard error by the delta method (match_luck_readout.elo_from_mean)."""
    from tools.match_luck_readout import elo_from_mean
    return elo_from_mean(m, se)


# ---------------------------------------------------------------------
# The readout
# ---------------------------------------------------------------------
def _by(values: np.ndarray, groups: np.ndarray, cells: np.ndarray, names) -> Dict[str, dict]:
    return {names(c): clustered_mean(values[cells == c], groups[cells == c]) for c in np.unique(cells)}


def label_readouts(table: Table, T: np.ndarray, keep: np.ndarray, band: float) -> dict:
    """The mean label pooled, by mover side, by margin bin and by ordered
    pair type, over the decisions `keep` selects."""
    g, T = table.g[keep], T[keep]
    bins = np.minimum((table.gap[keep] / BIN_WIDTH).astype(int), max(0, int(math.ceil(band / BIN_WIDTH)) - 1))
    pair = table.k0[keep] * 4 + table.k1[keep]
    mover = np.where(table.s[keep] > 0, "a", "b")
    return {"pooled": clustered_mean(T, g),
            "by_mover": {m: clustered_mean(T[mover == m], g[mover == m]) for m in ("a", "b")},
            "by_bin": _by(T, g, bins, lambda b: f"[{b * BIN_WIDTH:.2f},{(b + 1) * BIN_WIDTH:.2f})"),
            "by_pair": _by(T, g, pair, lambda c: pair_name(c // 4, c % 4))}


def coin_checks(table: Table, keep: np.ndarray, band: float) -> dict:
    """I-coin: the coins against P over the decisive games' band decisions,
    per margin bin (their sum on as many degrees of freedom), in the capped
    games, and the slopes of D - P on G and on the decision index."""
    d, p, g = table.d, table.p, table.g
    capped = ~table.decisive[g]
    bins = np.minimum((table.gap / BIN_WIDTH).astype(int), max(0, int(math.ceil(band / BIN_WIDTH)) - 1))
    per_bin = {f"[{b * BIN_WIDTH:.2f},{(b + 1) * BIN_WIDTH:.2f})": coin_frequency(d[keep & (bins == b)], p[keep & (bins == b)])
               for b in np.unique(bins[keep])}
    chi2 = sum(v["chi2"] for v in per_bin.values())
    out = {"overall": coin_frequency(d[keep], p[keep]),
           "bins": per_bin, "bins_chi2": chi2, "bins_df": len(per_bin),
           "bins_p_value": chi2_upper_p(chi2, len(per_bin)),
           "capped": coin_frequency(d[capped], p[capped])}
    for name, x in (("gap_slope", table.gap), ("index_slope", table.index.astype(float))):
        out[name] = clustered_slope(d[keep] - p[keep], x[keep], g[keep])
    return out


def side_game_counts(table: Table) -> dict:
    """Band decisions per side-game over the decisive games: mean, standard
    error and quartiles."""
    dec = np.flatnonzero(table.decisive)
    per = np.zeros((len(table.keys), 2))
    side_b = 3 - table.side_a
    for col, sides in ((0, table.side_a), (1, side_b)):
        mask = table.side == sides[table.g]
        per[:, col] = np.bincount(table.g[mask], minlength=len(table.keys))
    x = per[dec].ravel()
    if not len(x):
        return {"side_games": 0}
    return {"side_games": int(len(x)), "mean": float(x.mean()), "se": float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else None,
            "quartiles": [float(q) for q in np.quantile(x, (0.25, 0.5, 0.75))]}


def attack_readout(table: Table, beta: np.ndarray) -> Optional[dict]:
    """The attack coin over the decisive games: its label's mean (games are
    independent), the hooked player's PURE Elo, the coin against 1/2."""
    has = ~np.isnan(table.attack_d)
    keep = has & table.decisive
    if not keep.any():
        return None
    T = attack_labels(table, beta)[keep]
    m, se = float(T.mean()), float(T.std(ddof=1) / math.sqrt(len(T))) if len(T) > 1 else float("nan")
    hooked = np.where(table.attack_d[keep] == 1.0, table.z[keep], -table.z[keep])
    hm, hse = float(hooked.mean()), float(hooked.std(ddof=1) / math.sqrt(len(hooked))) if len(hooked) > 1 else float("nan")
    elo, elo_se = elo_of_mean(hm, hse)
    return {"games": int(keep.sum()), "capped": int((has & ~table.decisive).sum()),
            "label": {"mean": m, "se": se, "window": list(ATTACK_LABEL_WINDOW),
                      "pass": bool(ATTACK_LABEL_WINDOW[0] <= m <= ATTACK_LABEL_WINDOW[1])},
            "hooked_elo": {"elo": elo, "se": elo_se, "mean_z": hm, "window": list(ATTACK_ELO_WINDOW),
                           "pass": bool(ATTACK_ELO_WINDOW[0] <= elo <= ATTACK_ELO_WINDOW[1])},
            "coin": coin_frequency(table.attack_d[has], table.attack_p[has])}


def _line(stat: Optional[dict], key: str = "slope", target: float = 0.0) -> dict:
    """A check whose value must lie within CHECK_SE standard errors of `target`."""
    if not stat or stat.get(key) is None or not stat.get("se"):
        return {"pass": False, "why": "not read"}
    zstat = (stat[key] - target) / stat["se"]
    return {"value": stat[key], "se": stat["se"], "target": target, "z": zstat, "pass": bool(abs(zstat) <= CHECK_SE)}


def readout(table: Table, beta: np.ndarray, band: float, plant_rows: Sequence[dict] = ()) -> dict:
    """Every readout and check of the coin games in `table`."""
    from tools import coin_plant
    keep = table.decisive[table.g]
    T = decision_labels(table, beta)
    out: dict = {"games": {"read": len(table.keys), "decisive": int(table.decisive.sum()),
                           "capped": int((~table.decisive).sum()), **table.counts},
                 "beta": {"columns": list(COLUMNS), "values": [float(b) for b in beta]},
                 "band": {"w": band, "decisions": int(keep.sum()), "refused": int(table.refused[keep].sum()),
                          "per_side_game": side_game_counts(table)},
                 "checks": {}}
    checks = out["checks"]
    dice = coin_plant.dice_control(table.z[table.decisive], table.S[table.decisive][:, list(COLUMNS).index("hp")])
    checks["dice_control"] = dice
    attack = attack_readout(table, beta)
    if attack is not None:
        out["attack"] = attack
        checks["attack_label"] = {"pass": attack["label"]["pass"]}
        checks["attack_elo"] = {"pass": attack["hooked_elo"]["pass"]}
        checks["attack_coin"] = {"p_value": attack["coin"]["p_value"],
                                 "pass": bool(attack["coin"]["chi2"] <= CHECK_SE ** 2)}
    if keep.any():
        out["labels"] = label_readouts(table, T, keep, band)
        coin = coin_checks(table, keep, band)
        out["i_coin"] = coin
        checks["i_coin_overall"] = {"chi2": coin["overall"]["chi2"], "pass": bool(coin["overall"]["chi2"] <= CHECK_SE ** 2)}
        checks["i_coin_bins"] = {"p_value": coin["bins_p_value"],
                                 "pass": bool(coin["bins_p_value"] >= normal_two_sided_p(CHECK_SE))}
        checks["i_coin_gap"] = _line(coin["gap_slope"])
        checks["i_coin_index"] = _line(coin["index_slope"])
        if coin["capped"]["n"]:
            checks["i_coin_capped"] = {"chi2": coin["capped"]["chi2"], "pass": bool(coin["capped"]["chi2"] <= CHECK_SE ** 2)}
        checks["i_log"] = {**table.i_log, "pass": table.i_log.get("mismatched", 0) == 0}
    if plant_rows and keep.any():
        z_plant, x = coin_plant.planted_inputs(table, plant_rows)
        T_plant = decision_labels(table, beta, z_plant)
        reading = coin_plant.plant_reading(T, T_plant, x, table.s, table.g, keep)
        out["plant"] = reading
        for side in ("a", "b"):
            checks[f"plant_{side}"] = _line(reading[side], target=1.0)
    failed = sorted(name for name, c in checks.items() if not c.get("pass"))
    out["verdict"] = "PASS" if not failed else "FAIL: " + " ".join(failed)
    return out


def _fmt(stat: Optional[dict], key: str = "mean") -> str:
    if not stat or stat.get(key) is None:
        return "n/a"
    se = stat.get("se")
    return f"{stat[key]:+.4f} +- {se:.4f}" if se is not None else f"{stat[key]:+.4f}"


def report_text(r: dict) -> str:
    g = r["games"]
    lines = [f"games {g['read']} read, {g['decisive']} decisive, {g['capped']} capped (dropped), "
             f"{g.get('unwalked', 0)} not walked, {g.get('walk_errors', 0)} walk errors"]
    dc = r["checks"]["dice_control"]
    lines.append(f"dice control: z on side A's hp luck {_fmt(dc, 'slope')}, window "
                 f"[{dc['window'][0]:+.5f}, {dc['window'][1]:+.5f}]: {'pass' if dc['pass'] else 'FAIL'}")
    if "attack" in r:
        a = r["attack"]
        lines.append(f"attack coin: {a['games']} decisive games ({a['capped']} capped); label {_fmt(a['label'])} "
                     f"z, window {a['label']['window']}; hooked player {a['hooked_elo']['elo']:+.1f} +- "
                     f"{a['hooked_elo']['se']:.1f} Elo, window {a['hooked_elo']['window']}; coin ones "
                     f"{a['coin']['ones']} of {a['coin']['n']}")
    if "labels" in r:
        b = r["band"]
        psg = b["per_side_game"]
        lines.append(f"band w {b['w']}: {b['decisions']} decisions in decisive games ({b['refused']} refused), "
                     f"{psg.get('mean', float('nan')):.1f} per side-game, quartiles {psg.get('quartiles')}")
        lab = r["labels"]
        lines.append(f"mean label pooled {_fmt(lab['pooled'])}; mover A {_fmt(lab['by_mover']['a'])}, "
                     f"mover B {_fmt(lab['by_mover']['b'])}")
        lines.append("by margin bin:")
        lines += [f"  {k}: {_fmt(v)} ({v['n']} decisions)" for k, v in lab["by_bin"].items()]
        lines.append("by pair type (argmax kind > runner-up kind):")
        lines += [f"  {k}: {_fmt(v)} ({v['n']} decisions)" for k, v in sorted(lab["by_pair"].items(), key=lambda kv: -kv[1]["n"])]
        c = r["i_coin"]
        lines.append(f"I-coin: ones {c['overall']['ones']} of {c['overall']['n']} (expected {c['overall']['expected']:.1f}, "
                     f"chi2 {c['overall']['chi2']:.2f}); bins chi2 {c['bins_chi2']:.2f} on {c['bins_df']} df "
                     f"(p {c['bins_p_value']:.3f}); slope on G {_fmt(c['gap_slope'], 'slope')}, on the index "
                     f"{_fmt(c['index_slope'], 'slope')}; capped ones {c['capped']['ones']} of {c['capped']['n']}")
        il = r["checks"]["i_log"]
        lines.append(f"I-log: {il['matched']} matched, {il['mismatched']} mismatched, {il['unresolved']} unresolved")
    if "plant" in r:
        p = r["plant"]
        lines.append(f"coin plant (slope of T' - T on the planted x, predicted 1): mover A {_fmt(p['a'], 'slope')}, "
                     f"mover B {_fmt(p['b'], 'slope')}")
    lines.append(f"verdict: {r['verdict']}")
    return "\n".join(lines)


def _source(text: str) -> Tuple[str, Path]:
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"--match takes NAME=PATH, not {text!r}")
    return name, Path(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract", help="one row per game: its result, coins and I-log count")
    e.add_argument("--match", type=_source, action="append", required=True, metavar="NAME=PATH")
    e.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("readout", help="the labels, their means and the checks")
    r.add_argument("--coins", type=Path, nargs="+", required=True)
    r.add_argument("--rows", type=Path, nargs="+", required=True, help="tools/match_luck.py walk rows")
    r.add_argument("--beta", type=Path, required=True)
    r.add_argument("--band", type=float, required=True, help="the band width w the games were coined at")
    r.add_argument("--plant", type=Path, nargs="*", default=(), help="tools/coin_plant.py rows")
    r.add_argument("--save-json", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.command == "extract":
        n = extract_matches(args.match, args.out)
        log.info("%d rows written to %s", n, args.out)
        return 0
    table = build_table(load_jsonl(args.coins), load_jsonl(args.rows))
    result = readout(table, load_beta(args.beta), args.band, load_jsonl(args.plant) if args.plant else ())
    print(report_text(result))
    if args.save_json is not None:
        tmp = args.save_json.with_name(args.save_json.name + ".tmp")
        tmp.write_text(json.dumps(result, indent=1), encoding="utf-8")
        tmp.replace(args.save_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
