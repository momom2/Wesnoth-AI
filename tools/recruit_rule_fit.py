"""R0: the recruit rule Bd refitted on batch 1's near-tie coin labels and
frozen, and the capped rule's coin-world value sealed for match L
(docs/recruit_rule_l_prereg_20261021.md; the design is
docs/training_design_20261019/current_design.md, section 4 rows "R0" and
"Box A: L"). The definitions are those of the round-10 analysis scripts in
docs/training_design_20261019/analysis/ (named below); numpy and the
standard library only.

The rows (round10_invent_recruit_extract.py). Every band decision of the
coin games whose two candidates are recruits of different unit types: the
mover, the command index it played at, its coin d and propensity p, the
mover's faction and the enemy faction (the game record's setup), the two
types u0 (the argmax) and u1 (the runner-up), n0 and n1 = how many of each
the mover had recruited earlier in the game, the mover's side-turn, and the
label T = s (d - p) / (p (1 - p)) (z - beta . S) of tools/coin_labels.py
(capped games dropped). Every column is fixed before the decision's coin.
Each row also carries K and dev, the band decisions of both sides before it
and how many of them played the runner-up (round10_provenance_r_link.py,
refused decisions left out of both).

The rule (round10_synthesis_checks.py, Bd): a ridge fit of T on the columns
e(k(u1)) - e(k(u0)) over the units k(u) = "faction|unit", the same over the
matchups "faction|unit@enemy", and d = n0 - n1; ridge lambda on the unit and
matchup columns, 1 on d (tools/recruit_rule.py gives f). lambda minimises
the progressive loss over the synthesis's grid: matches in game order, the
student fitted on the matches before j read on match j by -2 w.b_j + w'G_j w.

The seal. The capped rule plays a1 at each side's first two rule decisions a
game (f > 0) and a0 at the later ones. Its exact coin-world value
(round10_provenance_r_pdis.py's identity): the labels of each side-game's
first two rule rows, the second weighted 3 if the first one's coin played a1
and 1/3 if not, in Elo (800 / ln 10 per unit of mean result per side-game),
each match's rule fitted on the other six (leave one match out, the
condition of a fresh match); standard errors clustered by game. Also read:
the uncapped first-order value F, the second-order term over all ranks
(TayPO-2 form), the capped value by mover faction, the turn-1 state shift
(round10_provenance_critique_checks.py section 2: the rank-0 turn-1 rule rows
reweighted by 1[no prior deviation] / 0.75^K to uncoined prior play), and L's
lines at the seal (round10_synthesis_budget.py section 1).

The differential check. With the enemy faction kept only where the synthesis
had it (the other side of the game has a row), the same code reads the
synthesis's numbers: progressive loss -586.7, -699.3, -658.4, -555.3 at the
four ridges and capped Bd +24.1 +- 3.0 Elo at 5,000.

    python tools/recruit_rule_fit.py --records training/metrics/near_tie_coins_20261020 \\
        --beta configs/luck_beta_20261019.json --out configs

writes configs/recruit_rule_bd_<hash>.json (the rule's units, matchups,
count coefficient, ridge, band and cap, the SHA-256 of that content, whose
first 12 characters name the file, and the code commit) and
configs/recruit_rule_bd_<hash>_readout.{json,txt}.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import subprocess
import sys
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.match_luck import COLUMNS  # noqa: E402
from tools.recruit_rule import SHORT_HASH, rule_hash  # noqa: E402

ELO = 800 / math.log(10)
ELO_SE_1 = 347.4               # Elo per sqrt(decisive game), round10_synthesis_budget.py
RIDGES = (2_000.0, 5_000.0, 10_000.0, 20_000.0)
COUNT_RIDGE = 1.0
BAND = 0.35                    # the coins' band, batch 1's
CAP = 2
L_GAMES = 37_800
# The synthesis's readings the differential check must reproduce (round10_synthesis_checks.out).
SYNTHESIS = {"progressive": {2_000.0: -586.7, 5_000.0: -699.3, 10_000.0: -658.4, 20_000.0: -555.3},
             "capped": (24.1, 3.0), "rows": 514_281, "enemy_known": 0.940}


# ---------------------------------------------------------------------
# Rows from game records
# ---------------------------------------------------------------------
class CommandContext:
    """Per command index of a game: the side to move, its side-turn, and
    each side's recruits by type before the command."""

    def __init__(self, commands: Sequence):
        self.side: List[Optional[int]] = []
        self.turn: List[int] = []
        self._recruits: List[Tuple[int, int, str]] = []
        side, turns = None, defaultdict(int)
        for k, c in enumerate(commands):
            if c[0] == "init_side":
                side = int(c[1])
                turns[side] += 1
            self.side.append(side)
            self.turn.append(turns[side] if side is not None else 0)
            if c[0] == "recruit" and side is not None:
                self._recruits.append((k, side, str(c[1])))

    def recruited(self, cmd: int, side: int, unit: str) -> int:
        return sum(1 for k, s, u in self._recruits if k < cmd and s == side and u == unit)


def row_features(rec: dict, ctx: CommandContext, side: int, cmd: int, u0: str, u1: str) -> dict:
    """The rule's inputs at a decision of side `side` played at command
    `cmd`, from the record: the factions of its setup, the recruits of its
    commands before `cmd`."""
    if ctx.side[cmd] != side:
        raise ValueError(f"{rec.get('game_label')}: a decision of side {side} at a command of side {ctx.side[cmd]}")
    setup = rec["setup"]
    return {"side": side, "fac": setup[f"faction{side}"], "enemy": setup[f"faction{3 - side}"], "u0": u0, "u1": u1,
            "n0": ctx.recruited(cmd, side, u0), "n1": ctx.recruited(cmd, side, u1), "st": ctx.turn[cmd]}


def is_recruit_pair(a0: dict, a1: dict) -> bool:
    return a0["kind"] == "recruit" and a1["kind"] == "recruit" and a0["unit"] != a1["unit"]


def recruit_rows(rec: dict) -> List[dict]:
    """Every band decision of a coin record whose candidates recruit two
    different types: its features, coin and gap, and K and dev before it."""
    ctx = CommandContext(rec["commands"])
    band, dev, out = 0, 0, []
    for e in (rec.get("coins") or {}).get("decisions", []):
        if is_recruit_pair(e["a0"], e["a1"]):
            cmd = int(e["cmd"])
            out.append({**row_features(rec, ctx, int(e["side"]), cmd, e["a0"]["unit"], e["a1"]["unit"]),
                        "cmd": cmd, "i": int(e["i"]), "gap": float(e["gap"]), "d": int(e["d"]), "p": float(e["p"]),
                        "K": band, "dev": dev})
        if not e.get("refused"):
            band += 1
            dev += int(e["d"])
    return out


def match_records(path: Path) -> Iterator[Tuple[str, dict]]:
    """(game label, record) of every game record in a match's tarball or directory."""
    def read(raw: bytes) -> dict:
        lines = [x for x in gzip.decompress(raw).decode("utf-8").splitlines() if x.strip()]
        return json.loads(lines[0])
    suffix = ".game.jsonl.gz"
    if path.is_dir():
        for p in sorted(path.glob(f"*{suffix}")):
            yield p.name[: -len(suffix)], read(p.read_bytes())
        return
    with tarfile.open(path, "r:*") as tf:
        for m in tf:
            name = m.name.rsplit("/", 1)[-1]
            if m.isfile() and name.endswith(suffix):
                yield name[: -len(suffix)], read(tf.extractfile(m).read())


@dataclass
class Rows:
    """The recruit rows of the decisive games read, as columns."""
    match: np.ndarray
    gid: np.ndarray
    side: np.ndarray
    cmd: np.ndarray
    d: np.ndarray
    fac: np.ndarray
    enemy: np.ndarray
    u0: np.ndarray             # "faction|unit"
    u1: np.ndarray
    dd: np.ndarray             # n0 - n1
    st: np.ndarray
    T: np.ndarray
    K: np.ndarray
    dev: np.ndarray
    side_games: int
    games_by_match: Dict[str, int]

    def subset_enemy(self, enemy: np.ndarray) -> "Rows":
        return Rows(**{**self.__dict__, "enemy": enemy})


def load_rows(records: Path, matches: Sequence[str], beta: np.ndarray) -> Rows:
    """The recruit rows of `matches` (games_<m>.tar.gz or games_<m>/ beside
    rows_<m>.jsonl, the luck walk) with their labels."""
    cols: Dict[str, list] = defaultdict(list)
    games, per_match = [], {}
    for mi, m in enumerate(matches):
        walk = {}
        with open(records / f"rows_{m}.jsonl", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if "game" in r and "S" in r:
                    walk[r["game"]] = r
        source = records / f"games_{m}.tar.gz"
        n = 0
        for game, rec in match_records(source if source.exists() else records / f"games_{m}"):
            w = walk.get(game)
            if w is None or not w.get("decisive"):
                continue
            side_a, z = int(rec["players"]["a"]["side"]), int(w["z"])
            if (int(rec["winner"]) == side_a) != (z == 1):
                raise ValueError(f"{m}/{game}: the record's winner and the walk's z disagree")
            R = z - float(np.dot(beta, [float(w["S"][c]) for c in COLUMNS]))
            for r in recruit_rows(rec):
                s = 1 if r["side"] == side_a else -1
                cols["T"].append(s * (r["d"] - r["p"]) / (r["p"] * (1 - r["p"])) * R)
                cols["game"].append(len(games))
                cols["match"].append(mi)
                for k in ("side", "cmd", "d", "fac", "enemy", "st", "K", "dev"):
                    cols[k].append(r[k])
                cols["u0"].append(f"{r['fac']}|{r['u0']}")
                cols["u1"].append(f"{r['fac']}|{r['u1']}")
                cols["dd"].append(r["n0"] - r["n1"])
            games.append(f"{m}/{game}")
            n += 1
        per_match[m] = n
    def arr(k, dtype=None):
        return np.asarray(cols[k], dtype=dtype)
    return Rows(match=arr("match", int), gid=arr("game", int), side=arr("side", int), cmd=arr("cmd", int),
                d=arr("d", int), fac=arr("fac"), enemy=arr("enemy"), u0=arr("u0"), u1=arr("u1"),
                dd=arr("dd", float), st=arr("st", int), T=arr("T", float), K=arr("K", int), dev=arr("dev", int),
                side_games=2 * len(games), games_by_match=per_match)


def synthesis_enemy(rows: Rows) -> np.ndarray:
    """The enemy faction as round10_synthesis_checks.py had it: the other
    side's faction where that side has a row in the same game, else ''."""
    have = set(zip(rows.gid.tolist(), rows.side.tolist()))
    return np.array([e if (g, 3 - s) in have else "" for g, s, e in zip(rows.gid.tolist(), rows.side.tolist(), rows.enemy)])


# ---------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------
@dataclass
class Design:
    """Per row the columns of its at most five nonzeros and their values;
    the column names; which columns the ridge lambda penalises."""
    idx: np.ndarray
    val: np.ndarray
    names: List[str]
    ridged: np.ndarray

    def penalty(self, lam: float) -> np.ndarray:
        return np.where(self.ridged, lam, COUNT_RIDGE)


def design(rows: Rows) -> Design:
    units = sorted(set(rows.u0) | set(rows.u1))
    known = rows.enemy != ""
    keys = sorted({f"{u}@{e}" for u, e in zip(np.r_[rows.u0[known], rows.u1[known]],
                                              np.r_[rows.enemy[known], rows.enemy[known]])})
    at = {name: i for i, name in enumerate(units + keys)}
    names = units + keys
    ridged = [True] * len(names)
    n = len(rows.T)
    idx = [np.array([at[u] for u in rows.u1]), np.array([at[u] for u in rows.u0])]
    val = [np.ones(n), -np.ones(n)]
    if not known.all():
        names.append("<no enemy>")         # a zero column for rows without the enemy faction
        ridged.append(False)
    blank = len(names) - 1
    idx.append(np.array([at[f"{u}@{e}"] if e else blank for u, e in zip(rows.u1, rows.enemy)]))
    val.append(known.astype(float))
    idx.append(np.array([at[f"{u}@{e}"] if e else blank for u, e in zip(rows.u0, rows.enemy)]))
    val.append(-known.astype(float))
    names.append("<count n0 - n1>")
    ridged.append(False)
    idx.append(np.full(n, len(names) - 1))
    val.append(rows.dd)
    return Design(np.column_stack(idx), np.column_stack(val), names, np.array(ridged))


def gram(des: Design, T: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """X'X and X'T over the masked rows."""
    k = len(des.names)
    I, V, t = des.idx[mask], des.val[mask], T[mask]  # noqa: E741
    flat = (I[:, :, None] * k + I[:, None, :]).reshape(-1)
    XtX = np.bincount(flat, weights=(V[:, :, None] * V[:, None, :]).reshape(-1), minlength=k * k).reshape(k, k)
    return XtX, np.bincount(I.reshape(-1), weights=(V * t[:, None]).reshape(-1), minlength=k)


def per_match_grams(des: Design, rows: Rows, n_matches: int):
    return [gram(des, rows.T, rows.match == j) for j in range(n_matches)]


def progressive_loss(grams, pen: np.ndarray) -> float:
    """Matches in game order: the student fitted on the matches before j,
    read on match j by -2 w.b_j + w'G_j w (the empty student reads 0)."""
    k = len(pen)
    loss, A, b = 0.0, np.zeros((k, k)), np.zeros(k)
    for j, (G, bj) in enumerate(grams):
        if j:
            w = np.linalg.solve(A + np.diag(pen), b)
            loss += -2 * w @ bj + w @ G @ w
        A, b = A + G, b + bj
    return float(loss)


def weights(grams, pen: np.ndarray, leave: Optional[int] = None) -> np.ndarray:
    A = sum(G for j, (G, _) in enumerate(grams) if j != leave)
    b = sum(bj for j, (_, bj) in enumerate(grams) if j != leave)
    return np.linalg.solve(A + np.diag(pen), b)


def values(des: Design, w: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """f per row, in tools/recruit_rule.RecruitRule.value's order of
    operations: (unit difference + matchup difference) + count term."""
    I, V = (des.idx, des.val) if mask is None else (des.idx[mask], des.val[mask])  # noqa: E741
    terms = V * w[I]
    return ((terms[:, 0] + terms[:, 1]) + (terms[:, 2] + terms[:, 3])) + terms[:, 4]


def lomo_switch(des: Design, rows: Rows, grams, pen: np.ndarray) -> np.ndarray:
    """Per row, whether its match's rule, fitted on the other matches, switches it."""
    sw = np.zeros(len(rows.T), dtype=bool)
    for j in range(len(grams)):
        held = rows.match == j
        sw[held] = values(des, weights(grams, pen, leave=j), held) > 0
    return sw


# ---------------------------------------------------------------------
# The readings
# ---------------------------------------------------------------------
def elo(v: np.ndarray, gid: np.ndarray, side_games: int) -> Tuple[float, float]:
    """Sum per side-game in Elo, its SE clustered by game (the synthesis's
    form: the root of the summed squared per-game sums)."""
    s = np.bincount(gid, weights=v, minlength=int(gid.max()) + 1 if len(gid) else 1)
    return float(v.sum()) / side_games * ELO, math.sqrt(float(np.sum(s * s))) / side_games * ELO


def ranked(rows: Rows, sw: np.ndarray):
    """The rule rows sorted by side-game and command: each one's rank among
    its side-game's rule rows, the coins landed 1 before it there, and the
    rank-0 row's coin."""
    r = np.nonzero(sw)[0]
    key = rows.gid[r] * 4 + rows.side[r]
    order = np.lexsort((rows.cmd[r], key))
    r, key = r[order], key[order]
    start = np.r_[True, key[1:] != key[:-1]]
    pos = np.arange(len(r))
    first = np.maximum.accumulate(np.where(start, pos, 0))
    rank = pos - first
    d = rows.d[r]
    cum = np.cumsum(d)
    played_before = cum - d - (cum - d)[first]
    return r, rank, played_before, d[first]


def capped_weights(rank: np.ndarray, d0: np.ndarray) -> np.ndarray:
    """The exact identity's weights at ranks 0-1: 1 at rank 0; at rank 1, 3
    when the rank-0 coin played a1 and 1/3 when not."""
    return np.where(rank == 1, np.where(d0 == 1, 3.0, 1.0 / 3.0), 1.0)


def pm(v: Tuple[float, float]) -> dict:
    return {"elo": v[0], "se": v[1]}


def rule_readings(rows: Rows, sw: np.ndarray) -> dict:
    """round10_synthesis_checks.py's `report` for one rule's switch set, and
    the capped value by mover faction and its turn-1 state shift."""
    T, gid, sg = rows.T, rows.gid, rows.side_games
    r, rank, pb, d0 = ranked(rows, sw)
    t, g = T[r], gid[r]
    cap = rank <= 1
    w1 = capped_weights(rank, d0)
    c = 2.0 * pb - (2.0 / 3.0) * (rank - pb)
    out = {"switches_per_side_game": float(sw.sum()) / sg, "F": pm(elo(t, g, sg)),
           "capped": {"switches_per_side_game": float(cap.sum()) / sg, **pm(elo((w1 * t)[cap], g[cap], sg)),
                      "first_order": elo(t[cap], g[cap], sg)[0],
                      "interaction": pm(elo(((w1 - 1.0) * t)[cap], g[cap], sg))},
           "second_order": pm(elo(t * c, g, sg)), "F_plus_second_order": pm(elo(t * (1.0 + c), g, sg))}
    out["by_rank"] = {lab: {"switches_per_side_game": float(m.sum()) / sg, "first_order": elo(t[m], g[m], sg)[0],
                            "second_order": pm(elo((t * c)[m], g[m], sg))}
                      for lab, m in (("1", rank == 1), ("2", rank == 2), ("3+", rank >= 3))}
    t1 = rows.st[r] == 1
    out["side_turn_1"] = {"switches_per_side_game": float(t1.sum()) / sg, "F": elo(t[t1], g[t1], sg)[0],
                          "capped": elo((w1 * t)[cap & t1], g[cap & t1], sg)[0]}
    out["capped_by_faction"] = {}
    for fac in sorted(set(rows.fac)):
        m = cap & (rows.fac[r] == fac)
        out["capped_by_faction"][fac] = {**pm(elo((w1 * t)[m], g[m], sg)),
                                         "switches_per_side_game": float(m.sum()) / sg}
    out["turn1_state_shift"] = state_shift(rows, r[(rank == 0) & t1], sg)
    return out


def state_shift(rows: Rows, r: np.ndarray, sg: int) -> dict:
    """round10_provenance_critique_checks.py section 2 on rows `r`: their
    labels reweighted by W = 1[no prior deviation by either side] / 0.75^K
    to uncoined prior play, the coin-world part, that part at uncoined prior
    states, and the shift between them (paired)."""
    T, g, K = rows.T[r], rows.gid[r], rows.K[r]
    W = np.where(rows.dev[r] == 0, 0.75 ** (-K.astype(float)), 0.0)
    est = W * T
    ess = float(W.sum() ** 2 / np.sum(W ** 2)) if np.any(W > 0) else 0.0
    return {"switches_per_side_game": len(r) / sg, "mean_K": float(K.mean()) if len(r) else None,
            "coin_world": pm(elo(T, g, sg)), "uncoined_prior": pm(elo(est, g, sg)), "shift": pm(elo(est - T, g, sg)),
            "mean_weight": float(W.mean()) if len(r) else None, "ess_share": ess / len(r) if len(r) else None}


def l_lines(seal: float, seal_se: float, games: int = L_GAMES) -> dict:
    """round10_synthesis_budget.py section 1 at `games` decisive games."""
    s = ELO_SE_1 / math.sqrt(games)
    null = 2 * s
    return {"games": games, "match_se": s, "half_width": 2 * math.hypot(seal_se, s), "null_line": null,
            "seal_above_null_se": (seal - null) / math.hypot(seal_se, s), "null_ratio_bound": (null + 2 * s) / seal}


# ---------------------------------------------------------------------
# R0
# ---------------------------------------------------------------------
def fit_rule(rows: Rows) -> dict:
    """lambda by progressive loss; the leave-one-match-out readings at it;
    the rule fitted on every match at it, and its switch rates in batch 1's
    states."""
    n_matches = int(rows.match.max()) + 1
    des = design(rows)
    grams = per_match_grams(des, rows, n_matches)
    loss = {lam: progressive_loss(grams, des.penalty(lam)) for lam in RIDGES}
    lam = min(loss, key=loss.get)
    pen = des.penalty(lam)
    readings = rule_readings(rows, lomo_switch(des, rows, grams, pen))
    w = weights(grams, pen)
    f = values(des, w)
    r, rank, _pb, _d0 = ranked(rows, f > 0)
    cap = r[rank <= 1]
    rates = {fac: float(np.sum(rows.fac[cap] == fac)) / rows.side_games for fac in sorted(set(rows.fac))}
    return {"design": des, "weights": w, "ridge": lam, "progressive_loss": loss, "readings": readings,
            "all_data_rule": {"switches_per_side_game": float(np.sum(f > 0)) / rows.side_games,
                              "capped_switches_per_side_game": len(cap) / rows.side_games,
                              "capped_switches_by_faction": rates}}


def rule_content(fit: dict) -> dict:
    des, w = fit["design"], fit["weights"]
    units, matchups, count = {}, {}, None
    for name, v in zip(des.names, w.tolist()):
        if name == "<count n0 - n1>":
            count = v
        elif name == "<no enemy>":
            raise ValueError("a frozen rule needs every row's enemy faction")
        elif "@" in name:
            matchups[name] = v
        else:
            units[name] = v
    return {"name": "Bd", "format": 1, "band": BAND, "cap": CAP, "ridge": fit["ridge"], "count_ridge": COUNT_RIDGE,
            "units": units, "matchups": matchups, "count": count,
            "value": "f = (units[fac|u1] - units[fac|u0]) + (matchups[fac|u1@enemy] - matchups[fac|u0@enemy]) "
                     "+ count * (n0 - n1), a missing key 0; play a1 iff f > 0, at most cap times a side-game"}


def reproduction(rows: Rows) -> dict:
    """The differential check: the synthesis's enemy coverage, its numbers."""
    sub = rows.subset_enemy(synthesis_enemy(rows))
    des = design(sub)
    grams = per_match_grams(des, sub, int(rows.match.max()) + 1)
    loss = {lam: progressive_loss(grams, des.penalty(lam)) for lam in RIDGES}
    capped = rule_readings(sub, lomo_switch(des, sub, grams, des.penalty(5_000.0)))["capped"]
    got = {"rows": int(len(rows.T)), "enemy_known": float(np.mean(sub.enemy != "")),
           "progressive": loss, "capped": (capped["elo"], capped["se"])}
    ok = (got["rows"] == SYNTHESIS["rows"] and round(got["enemy_known"], 3) == SYNTHESIS["enemy_known"]
          and all(round(loss[lam], 1) == v for lam, v in SYNTHESIS["progressive"].items())
          and tuple(round(x, 1) for x in got["capped"]) == SYNTHESIS["capped"])
    return {**got, "progressive": {f"{k:g}": v for k, v in loss.items()}, "pass": bool(ok)}


def code_commit() -> dict:
    root = Path(__file__).resolve().parent.parent
    def git(*a):
        return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"),
                "tool_changed_since": bool(git("status", "--porcelain", "--", "tools/recruit_rule_fit.py",
                                               "tools/recruit_rule.py"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "tool_changed_since": None}


def report_text(r: dict) -> str:
    rd = r["readings"]
    c = rd["capped"]
    L = r["l_lines"]
    lines = [f"rows {r['rows']} of {r['side_games'] // 2} decisive games ({r['rows'] / r['side_games']:.2f} per side-game); "
             f"enemy faction from the record on every row",
             "progressive loss by ridge: " + "; ".join(f"{k}: {v:+.1f}" for k, v in r["progressive_loss"].items())
             + f"; chosen {r['ridge']:,.0f}",
             f"Bd, leave one match out: {rd['switches_per_side_game']:.2f} switches per side-game; "
             f"F {rd['F']['elo']:+.1f} +- {rd['F']['se']:.1f} Elo",
             f"  capped at {CAP} a side-game: {c['switches_per_side_game']:.2f} switches; exact coin-world value "
             f"{c['elo']:+.1f} +- {c['se']:.1f} (first order {c['first_order']:+.1f}; interaction "
             f"{c['interaction']['elo']:+.1f} +- {c['interaction']['se']:.1f})",
             f"  second-order term, all ranks: {rd['second_order']['elo']:+.1f} +- {rd['second_order']['se']:.1f}",
             f"  side-turn 1: {rd['side_turn_1']['switches_per_side_game']:.2f} switches, F "
             f"{rd['side_turn_1']['F']:+.1f}; capped part {rd['side_turn_1']['capped']:+.1f}",
             "  capped by mover faction: " + "; ".join(
                 f"{k} {v['elo']:+.1f} +- {v['se']:.1f} ({v['switches_per_side_game']:.2f})"
                 for k, v in rd["capped_by_faction"].items())]
    s = rd["turn1_state_shift"]
    lines.append(f"  turn-1 state shift (rank-0 turn-1 switches, {s['switches_per_side_game']:.3f} per side-game, mean K "
                 f"{s['mean_K']:.1f}): coin world {s['coin_world']['elo']:+.2f} +- {s['coin_world']['se']:.2f}, at uncoined "
                 f"prior states {s['uncoined_prior']['elo']:+.2f} +- {s['uncoined_prior']['se']:.2f}; shift "
                 f"{s['shift']['elo']:+.2f} +- {s['shift']['se']:.2f} (mean weight {s['mean_weight']:.3f}, ESS share "
                 f"{s['ess_share']:.3f})")
    a = r["all_data_rule"]
    lines.append(f"the frozen rule (all matches) in batch 1's states: {a['switches_per_side_game']:.2f} rule decisions "
                 f"per side-game, {a['capped_switches_per_side_game']:.2f} capped switches; by mover faction "
                 + "; ".join(f"{k} {v:.3f}" for k, v in a["capped_switches_by_faction"].items()))
    lines.append(f"L's lines at {L['games']:,} decisive games (SE {L['match_se']:.2f}): seal {c['elo']:+.1f} +- "
                 f"{c['se']:.1f}, miss half-width {L['half_width']:.1f}, null line {L['null_line']:.2f}, the seal "
                 f"{L['seal_above_null_se']:.1f} SE above it; a null bounds the transfer ratio below "
                 f"{L['null_ratio_bound']:.2f}")
    rep = r["reproduction"]
    lines.append(f"differential check at the synthesis's enemy coverage ({rep['enemy_known']:.3f} of {rep['rows']} rows): "
                 "progressive " + "; ".join(f"{k}: {v:+.1f}" for k, v in rep["progressive"].items())
                 + f"; capped at 5,000 {rep['capped'][0]:+.1f} +- {rep['capped'][1]:.1f}: "
                 f"{'reproduces the synthesis' if rep['pass'] else 'DIFFERS FROM THE SYNTHESIS'}")
    lines.append(f"rule {r['hash']} ({r['rule_columns']} columns: {r['units']} units, {r['matchups']} matchups, count)")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=Path, required=True, help="batch 1's records: games_<m>.tar.gz and rows_<m>.jsonl")
    ap.add_argument("--matches", nargs="+", default=None, help="default: every games_b*.tar.gz, in name order")
    ap.add_argument("--beta", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="the directory the rule and its readout go to")
    args = ap.parse_args(argv)
    from tools.coin_labels import load_beta
    matches = args.matches or sorted(p.name[len("games_"): -len(".tar.gz")] for p in args.records.glob("games_b*.tar.gz"))
    rows = load_rows(args.records, matches, load_beta(args.beta))
    if np.any(rows.enemy == ""):
        raise SystemExit("a record without its enemy faction: R0 reads every row's from its record")
    fit = fit_rule(rows)
    rule = rule_content(fit)
    h = rule_hash(rule)
    stem = f"recruit_rule_bd_{h[:SHORT_HASH]}"
    c = fit["readings"]["capped"]
    readout = {"hash": h, "rows": int(len(rows.T)), "side_games": rows.side_games, "matches": matches,
               "games_by_match": rows.games_by_match, "beta": args.beta.as_posix(),
               "progressive_loss": {f"{k:g}": v for k, v in fit["progressive_loss"].items()}, "ridge": fit["ridge"],
               "readings": fit["readings"], "all_data_rule": fit["all_data_rule"],
               "seal": {"elo": c["elo"], "se": c["se"]}, "l_lines": l_lines(c["elo"], c["se"]),
               "reproduction": reproduction(rows), "rule_columns": len(fit["design"].names),
               "units": len(rule["units"]), "matchups": len(rule["matchups"])}
    text = report_text(readout)
    print(text)
    args.out.mkdir(parents=True, exist_ok=True)
    data = {"rule": rule, "hash": h, "provenance": {
        **code_commit(), "tool": "tools/recruit_rule_fit.py", "records": args.records.as_posix(), "matches": matches,
        "beta": args.beta.as_posix(), "rows": readout["rows"], "decisive_games": rows.side_games // 2,
        "readout": f"{stem}_readout.json"}}
    (args.out / f"{stem}.json").write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    (args.out / f"{stem}_readout.json").write_text(json.dumps(readout, indent=1) + "\n", encoding="utf-8")
    (args.out / f"{stem}_readout.txt").write_text(text + "\n", encoding="utf-8")
    return 0 if readout["reproduction"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
