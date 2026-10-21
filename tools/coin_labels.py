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
I-coin check reads the coins within capped games). Step (2) needs every
L_k of mean zero given F_k: a fight whose realised outcome lies outside
its enumerated support breaks it, so the readout requires the walk's
`outside_support` count to be 0. tests/test_coin_labels.py checks the claim
on synthetic games whose decision effects are known, for several beta,
with earlier luck setting band membership and effect size, and that a
variate read after the coin breaks it.

    python tools/coin_labels.py extract --match NAME=DIR [--match ...] --out coins.jsonl
    python tools/coin_labels.py readout --coins coins.jsonl [...] --rows rows.jsonl [...]
        --beta configs/luck_beta_20261019.json --band 0.35 [--plant plant.jsonl ...]
        [--plan matches.txt] [--save-json readout.json]
    python tools/coin_labels.py readout --from-records DIR [--matches NAME ...]
        --beta configs/luck_beta_20261019.json --band 0.35 [--save-json readout.json]

extract writes the coin rows of a games directory or tarball
(tools/coin_rows.py). readout joins them with the walk's rows
(tools/match_luck.py walk) and prints the labels' means with standard
errors clustered by game (pooled, by mover side, by 0.05-nat margin bin, by
ordered pair type), the attack coin's label and the hooked player's Elo,
and the checks: every match read whole (a match whose result files, coin
rows and walk rows do not all join is left out of the labels and fails the
verdict), the walk's fight counts, I-coin, I-log, the band decisions per
side-game, the dice control and the coin plant (tools/coin_plant.py). With
a plan (a box script's matches.txt) it counts each match's slots without a
result. --from-records reads a pulled records directory as the box left
it: games_NAME.tar.gz (or games_NAME/), rows_NAME.jsonl and matches.txt;
it extracts the coin rows and plants the result afresh from the records.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.coin_decode import KIND_CODES  # noqa: E402
from tools.coin_rows import STATUSES, extract_matches, extract_rows, parse_plan, record_sources, slot_report  # noqa: E402
from tools.match_luck import COLUMNS  # noqa: E402

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
# The walk's per-game fight counts (match_luck.game_luck) the readout sums.
FIGHT_COUNTS = ("attacks", "enumerated", "outside_support", "not_enumerable", "counter_weapon_differs", "no_units")


def pair_name(k0: int, k1: int) -> str:
    return f"{KIND_NAMES[int(k0)]}>{KIND_NAMES[int(k1)]}"


def bin_edges(band: float) -> np.ndarray:
    """The 0.05-nat margin bins' edges over [0, band), each the float of its
    decimal: k x 0.05 rounded, so the edge 0.15 is the float the literal
    0.15 reads as (3 x 0.05 is 0.15000000000000002, and 0.15 / 0.05 truncates
    to 2)."""
    n = max(1, math.ceil(round(band / BIN_WIDTH, 9)))
    return np.array([round(k * BIN_WIDTH, 10) for k in range(n + 1)])


def margin_bins(gap: np.ndarray, band: float) -> np.ndarray:
    """Each gap's bin: i where edges[i] <= gap < edges[i + 1], the last bin
    holding anything above."""
    edges = bin_edges(band)
    return np.clip(np.searchsorted(edges, np.asarray(gap, dtype=float), side="right") - 1, 0, len(edges) - 2)


def bin_name(edges: np.ndarray, b: int) -> str:
    return f"[{edges[b]:.2f},{edges[b + 1]:.2f})"


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
    """The games read (decisive or not) and their band decisions, flat, of
    the matches read whole; `matches` accounts for every match seen."""
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
    matches: Dict[str, dict] = field(default_factory=dict)
    excluded: List[str] = field(default_factory=list)
    fights: Dict[str, int] = field(default_factory=dict)

    @property
    def decisive(self) -> np.ndarray:
        return self.z != 0


def iter_jsonl(paths: Sequence[Path]) -> Iterator[dict]:
    """The rows of JSON-lines files, one at a time."""
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def load_jsonl(paths: Sequence[Path]) -> List[dict]:
    return list(iter_jsonl(paths))


def _account() -> dict:
    return {"results": None, "game_rows": 0, "extract_errors": 0, "unwalked": 0, "walk_errors": 0,
            "walk_rows_unjoined": 0, "fights_unread": 0, "fights": dict.fromkeys(FIGHT_COUNTS, 0),
            "i_log": dict.fromkeys(STATUSES, 0), "seeds": []}


def _game_columns(row: dict, lr: dict) -> dict:
    band = row["band"]
    n = len(band["d"])
    side = np.asarray(band["side"], dtype=np.int8)
    attack = row.get("attack")
    return {"key": (row["match"], row["game"]), "z": int(row["z"]), "S": [float(lr["S"][c]) for c in COLUMNS],
            "side_a": int(row["side_a"]),
            "attack_d": np.nan if attack is None else float(attack["d"]),
            "attack_p": np.nan if attack is None else float(attack["p"]),
            "side": side, "s": mover_sign(side, np.full(n, int(row["side_a"]))).astype(np.int8),
            "index": np.asarray(band["i"], dtype=np.int32), "gap": np.asarray(band["gap"], dtype=float),
            "p": np.asarray(band["p"], dtype=float), "d": coin_outcomes(band),
            "k0": np.asarray(band["k0"], dtype=np.int8), "k1": np.asarray(band["k1"], dtype=np.int8),
            "refused": np.asarray(band.get("refused", [0] * n), dtype=np.int8)}


def _drops(a: dict) -> List[str]:
    """Why a match is not read whole: what its result files, coin rows and
    walk rows fail to join."""
    why = []
    if a["results"] is None:
        why.append("no manifest (the extraction did not finish)")
    elif a["results"] != a["game_rows"]:
        why.append(f"{a['results']} result files, {a['game_rows']} coin rows")
    for k in ("extract_errors", "unwalked", "walk_errors", "walk_rows_unjoined"):
        if a[k]:
            why.append(f"{a[k]} {k.replace('_', ' ')}")
    return why


def build_table(coin_rows: Iterable[dict], luck_rows: Sequence[dict],
                plan: Optional[Dict[str, Tuple[int, int]]] = None) -> Table:
    """The coin rows joined with the walk's rows on (match, game), one
    match at a time: a match whose result files, coin rows and walk rows do
    not all join (`_drops`) is accounted for and left out whole. A result
    that differs between the two rows refuses the join."""
    luck = {(r["match"], r["game"]): r for r in luck_rows}
    joined, accounts, games = set(), {}, {}
    for row in coin_rows:
        m = row["match"]
        a = accounts.setdefault(m, _account())
        if "manifest" in row:
            a["results"] = int(row["manifest"]["results"])
            continue
        a["game_rows"] += 1
        a["seeds"].append((row.get("seed"), bool(row.get("decisive"))))
        if "error" in row:
            a["extract_errors"] += 1
            continue
        key = (m, row["game"])
        lr = luck.get(key)
        if lr is None:
            a["unwalked"] += 1
            continue
        joined.add(key)
        if "error" in lr:
            a["walk_errors"] += 1
            continue
        if int(lr["z"]) != int(row["z"]):
            raise ValueError(f"{key}: the walk reads z {lr['z']}, the coin row {row['z']}: a join or result error")
        fights = lr.get("counts")
        if fights is None:
            a["fights_unread"] += 1
        else:
            for k in FIGHT_COUNTS:
                a["fights"][k] += int(fights.get(k, 0))
        for k, v in row.get("i_log", {}).items():
            a["i_log"][k] = a["i_log"].get(k, 0) + int(v)
        games.setdefault(m, []).append(_game_columns(row, lr))
    for key in luck:
        if key[0] in accounts and key not in joined:
            accounts[key[0]]["walk_rows_unjoined"] += 1
    reports, read, excluded = {}, [], []
    for m in sorted(accounts):
        a = accounts.pop(m)
        seeds = a.pop("seeds")
        a["drops"] = _drops(a)
        if plan and m in plan:
            a["slots"] = slot_report(seeds, plan[m])
        reports[m] = a
        (excluded if a["drops"] else read).append(m)
    return _assemble([g for m in read for g in games.get(m, [])], reports, excluded)


def _assemble(games: List[dict], reports: Dict[str, dict], excluded: List[str]) -> Table:
    read = [m for m in reports if m not in excluded]
    i_log = {k: sum(reports[m]["i_log"].get(k, 0) for m in read) for k in STATUSES}
    fights = {k: sum(reports[m]["fights"][k] for m in read) for k in FIGHT_COUNTS}
    fights["rows_without_counts"] = sum(reports[m]["fights_unread"] for m in read)

    def flat(name, dtype):
        parts = [g[name] for g in games]
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)
    g_index = np.concatenate([np.full(len(g["d"]), gi, dtype=np.int32) for gi, g in enumerate(games)]) \
        if games else np.zeros(0, dtype=np.int32)
    return Table(keys=[g["key"] for g in games], z=np.array([g["z"] for g in games], dtype=float),
                 S=np.array([g["S"] for g in games], dtype=float).reshape(len(games), len(COLUMNS)),
                 side_a=np.array([g["side_a"] for g in games], dtype=int),
                 attack_d=np.array([g["attack_d"] for g in games], dtype=float),
                 attack_p=np.array([g["attack_p"] for g in games], dtype=float), i_log=i_log,
                 g=g_index, side=flat("side", np.int8), s=flat("s", float), index=flat("index", np.int32),
                 gap=flat("gap", float), p=flat("p", float), d=flat("d", float), k0=flat("k0", np.int8),
                 k1=flat("k1", np.int8), refused=flat("refused", np.int8),
                 matches=reports, excluded=excluded, fights=fights)


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
    edges = bin_edges(band)
    pair = table.k0[keep].astype(int) * 4 + table.k1[keep].astype(int)
    mover = np.where(table.s[keep] > 0, "a", "b")
    return {"pooled": clustered_mean(T, g),
            "by_mover": {m: clustered_mean(T[mover == m], g[mover == m]) for m in ("a", "b")},
            "by_bin": _by(T, g, margin_bins(table.gap[keep], band), lambda b: bin_name(edges, b)),
            "by_pair": _by(T, g, pair, lambda c: pair_name(c // 4, c % 4))}


def coin_checks(table: Table, keep: np.ndarray, band: float) -> dict:
    """I-coin: the coins against P over the decisive games' band decisions,
    per margin bin (their sum on as many degrees of freedom), in the capped
    games, and the slopes of D - P on G and on the decision index."""
    d, p, g = table.d, table.p, table.g
    capped = ~table.decisive[g]
    edges = bin_edges(band)
    bins = margin_bins(table.gap, band)
    per_bin = {bin_name(edges, b): coin_frequency(d[keep & (bins == b)], p[keep & (bins == b)])
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


def _match_checks(table: Table) -> dict:
    """Every match read whole; the walk's fights all inside their supports
    (the label's proof needs every luck term of mean zero) and counted."""
    f = table.fights
    return {"matches_whole": {"excluded": {m: table.matches[m]["drops"] for m in table.excluded},
                              "pass": not table.excluded},
            "luck_support": {"outside_support": f["outside_support"], "rows_without_counts": f["rows_without_counts"],
                             "pass": f["outside_support"] == 0 and f["rows_without_counts"] == 0}}


def readout(table: Table, beta: np.ndarray, band: float, plant_rows: Sequence[dict] = ()) -> dict:
    """Every readout and check of the coin games in `table`."""
    from tools import coin_plant
    keep = table.decisive[table.g]
    T = decision_labels(table, beta)
    out: dict = {"games": {"read": len(table.keys), "decisive": int(table.decisive.sum()),
                           "capped": int((~table.decisive).sum())},
                 "matches": table.matches, "fights": table.fights,
                 "beta": {"columns": list(COLUMNS), "values": [float(b) for b in beta]},
                 "band": {"w": band, "decisions": int(keep.sum()), "refused": int(table.refused[keep].sum()),
                          "per_side_game": side_game_counts(table)},
                 "checks": _match_checks(table)}
    checks = out["checks"]
    checks["dice_control"] = coin_plant.dice_control(table.z[table.decisive],
                                                     table.S[table.decisive][:, list(COLUMNS).index("hp")])
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


def _match_line(name: str, a: dict) -> str:
    line = f"  {name}: {a['game_rows']} coin rows of {a['results']} result files"
    s = a.get("slots")
    if s:
        line += f"; {s['with_result']} of {s['planned']} slots have a result ({s['decisive']} decisive)"
        if s["cut"]:
            line += (f", a prefix of its slots: 0-{s['complete_prefix'] - 1} complete, {s['missing']} missing "
                     f"(first {s['first_missing']})")
    il = a["i_log"]
    line += f"; I-log {il['matched']} matched, {il['relocated']} relocated, {il['mismatched']} mismatched"
    if a["drops"]:
        line += "; LEFT OUT: " + "; ".join(a["drops"])
    return line


def report_text(r: dict) -> str:
    g, f = r["games"], r["fights"]
    lines = [f"games {g['read']} read, {g['decisive']} decisive, {g['capped']} capped (dropped); matches:"]
    lines += [_match_line(name, a) for name, a in r["matches"].items()]
    lines.append(f"fights: {f['attacks']} attacks, {f['outside_support']} outside their support, "
                 f"{f['not_enumerable']} not enumerable, {f['counter_weapon_differs']} counter weapon differs, "
                 f"{f['no_units']} without units, {f['rows_without_counts']} walk rows without counts")
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
        lines.append(f"I-log: {il['matched']} matched, {il['relocated']} relocated recruits, {il['mismatched']} "
                     f"mismatched, {il['unresolved']} unresolved")
    if "plant" in r:
        p = r["plant"]
        lines.append(f"coin plant (slope of T' - T on the planted x, predicted 1): mover A {_fmt(p['a'], 'slope')}, "
                     f"mover B {_fmt(p['b'], 'slope')}")
    lines.append(f"verdict: {r['verdict']}")
    return "\n".join(lines)


def records_inputs(directory: Path, names: Optional[Sequence[str]], *, delta: float = 1.0, salt: str = "coin_plant"):
    """(coin rows, walk rows, plan, plant rows) of a pulled records
    directory: the coin rows extracted and the plant computed afresh from
    each match's records (a match without band decisions is not planted)."""
    from tools import coin_plant
    sources = record_sources(directory, names)
    plant_rows: List[dict] = []
    for m, path in sources:
        try:
            pair = coin_plant.most_frequent_pair([(m, path)])
        except ValueError:
            continue
        plant_rows += coin_plant.plant_rows([(m, path)], delta=delta, pair=pair, salt=salt)
    rows = [directory / f"rows_{m}.jsonl" for m, _ in sources]
    plan_file = directory / "matches.txt"
    coin_rows = itertools.chain.from_iterable(extract_rows(m, path) for m, path in sources)
    return (coin_rows, load_jsonl([p for p in rows if p.exists()]),
            parse_plan(plan_file) if plan_file.exists() else None, plant_rows)


def _source(text: str) -> Tuple[str, Path]:
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"--match takes NAME=PATH, not {text!r}")
    return name, Path(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract", help="the coin rows of each match (tools/coin_rows.py)")
    e.add_argument("--match", type=_source, action="append", required=True, metavar="NAME=PATH")
    e.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("readout", help="the labels, their means and the checks")
    r.add_argument("--coins", type=Path, nargs="+", default=None)
    r.add_argument("--rows", type=Path, nargs="+", default=None, help="tools/match_luck.py walk rows")
    r.add_argument("--plant", type=Path, nargs="*", default=(), help="tools/coin_plant.py rows")
    r.add_argument("--plan", type=Path, default=None, help="a box script's matches.txt")
    r.add_argument("--from-records", type=Path, default=None, metavar="DIR",
                   help="a pulled records directory, in place of --coins, --rows, --plant and --plan")
    r.add_argument("--matches", nargs="+", default=None, metavar="NAME", help="with --from-records: these matches")
    r.add_argument("--beta", type=Path, required=True)
    r.add_argument("--band", type=float, required=True, help="the band width w the games were coined at")
    r.add_argument("--save-json", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.command == "extract":
        n = extract_matches(args.match, args.out)
        log.info("%d rows written to %s", n, args.out)
        return 0
    if args.from_records is not None:
        if args.coins or args.rows or args.plant or args.plan:
            ap.error("--from-records reads the coins, rows, plant and plan itself")
        coin_rows, luck, plan, plant_rows = records_inputs(args.from_records, args.matches)
    else:
        if not args.coins or not args.rows or args.matches:
            ap.error("readout needs --coins and --rows (or --from-records DIR); --matches goes with --from-records")
        coin_rows, luck = iter_jsonl(args.coins), load_jsonl(args.rows)
        plan = parse_plan(args.plan) if args.plan else None
        plant_rows = load_jsonl(args.plant) if args.plant else ()
    table = build_table(coin_rows, luck, plan)
    result = readout(table, load_beta(args.beta), args.band, plant_rows)
    print(report_text(result))
    if args.save_json is not None:
        tmp = args.save_json.with_name(args.save_json.name + ".tmp")
        tmp.write_text(json.dumps(result, indent=1), encoding="utf-8")
        tmp.replace(args.save_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
