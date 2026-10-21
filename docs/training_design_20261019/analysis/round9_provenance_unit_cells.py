"""Round 9, provenance stance: is there spread of near-tie advantage by unit
identity OUTSIDE the recruit pairs? round9_provenance_recruit_cells.py found it
in recruit>recruit pairs of different units (5.5% of band decisions); the other
nine partitions of round9_provenance_cells.py keyed nothing on units. This
script keys every band decision on the identity of the acting units.

Two modes, one process each, numpy and the standard library only:

    python round9_provenance_unit_cells.py extract MATCH   # games tarball -> per-cell fold sums
    python round9_provenance_unit_cells.py combine         # all matches -> readings

Cell of a band decision: (sub-kind, mover's faction, d(a0) > d(a1)), with
d(recruit) = the recruited unit, d(move) = the type of the unit standing on
the move's start hex, d(attack) = attacker type @ defender type, d(end_turn)
= "end". Unit types come from a tracker over the command stream BEFORE the
decision's command (recruits place a type on a hex, moves carry it along the
path, leaders are the setup's leader types on hexes never written). The
tracker ignores deaths and level-ups (a hex keeps its last written type until
overwritten; a unit keeps its recruited type), so some keys are stale. That
adds noise, never bias: every key is a function of the history before the
coin, so a rule chosen on other folds and read on this fold's labels is
unbiased for its first-order gain whatever the keys mean.

Readings per group of sub-kinds (the recruit pairs excluded or alone):
1. excess dispersion of cell means over cells with >= 200 labels;
2. V, the variance of the true cell means within the group (fold U-statistic,
   delete-one-fold jackknife SE);
3. cross-fitted welfare (rule chosen on 19 folds, read on the 20th) of
   (a) switch where the cell mean > 0, (b) the same for cells with >= 1,000
   out-of-fold labels, (c) for move>move between different units, an additive
   per-(faction, type) value fitted by ridge (2,000 labels), switch iff
   v(a1's unit) - v(a0's unit) > 0.
Per side-game, z and Elo (x 800 / ln 10); SE from the labels' second moment
(labels within a game are uncorrelated: design effect 1.00,
round9_provenance_cells.out).
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("pc", HERE / "round9_provenance_cells.py")
pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pc)

MATCHES = pc.MATCHES
FOLDS = pc.FOLDS
ELO = pc.ELO
GROUPS = {
    "move>move:other_unit": ("move>move:other_unit",),
    "move>move:same_unit": ("move>move:same_unit",),
    "attack pairs (any attack candidate)": ("attack>attack:weapon", "attack>attack:other_target",
                                            "attack>attack:same_target", "attack>move", "move>attack",
                                            "attack>end_turn", "end_turn>attack", "attack>recruit",
                                            "recruit>attack"),
    "end_turn pairs without attack": ("end_turn>move", "move>end_turn", "end_turn>recruit", "recruit>end_turn"),
    "recruit>move, move>recruit": ("recruit>move", "move>recruit"),
    "all but recruit>recruit": None,
    "recruit>recruit (both)": ("recruit>recruit:other_unit", "recruit>recruit:same_unit"),
}


class Tracker:
    """hex -> unit type, from recruits and moves only."""

    def __init__(self, leaders: dict):
        self.at = {}
        self.leaders = leaders

    def type_at(self, hx, side_hint: int) -> str:
        return self.at.get(tuple(hx), f"leader:{self.leaders.get(side_hint, '?')}")

    def apply(self, c: list, side: int) -> None:
        if c[0] == "recruit":
            self.at[(int(c[2]), int(c[3]))] = c[1]
        elif c[0] == "move":
            xs, ys = c[1], c[2]
            start, end = (int(xs[0]), int(ys[0])), (int(xs[-1]), int(ys[-1]))
            if start != end:
                self.at[end] = self.at.pop(start, f"leader:{self.leaders.get(side, '?')}")


def describe(a: dict, tr: Tracker, side: int) -> str:
    k = a["kind"]
    if k == "recruit":
        return a["unit"]
    if k == "move":
        return tr.type_at(a["from"], side)
    if k == "attack":
        return f"{tr.type_at(a['from'], side)}@{tr.type_at(a['to'], 3 - side)}"
    return "end"


def extract(match: str) -> None:
    beta = pc.load_beta()
    rows = pc.luck_rows(match)
    code, col, T, T0, F = {}, [], [], [], []
    games = side_games = 0
    for game, rec in pc.records(match):
        row = rows.get(game)
        if row is None or not row.get("decisive"):
            continue
        z = int(row["z"])
        side_a = int(row["side_a"])
        R = z - float(np.dot(beta, [row["S"][c] for c in pc.COLUMNS]))
        fac = {side_a: row["faction_a"], 3 - side_a: row["faction_b"]}
        setup = rec.get("setup", {})
        tr = Tracker({1: setup.get("leader1", "?"), 2: setup.get("leader2", "?")})
        f = pc.fold_of(game)
        games += 1
        side_games += 2
        decisions = sorted(((int(e.get("cmd", 0)), e) for e in (rec.get("coins") or {}).get("decisions", [])
                            if not e.get("refused")), key=lambda x: x[0])
        cmds = rec["commands"]
        k_done, side = 0, 1
        for k, e in decisions:
            while k_done < min(k, len(cmds)):
                c = cmds[k_done]
                if c[0] == "init_side":
                    side = int(c[1])
                tr.apply(c, side)
                k_done += 1
            mover = int(e["side"])
            p, d = float(e["p"]), int(e["d"])
            s = 1 if mover == side_a else -1
            om = (d - p) / (p * (1 - p))
            sub = pc.sub_kind(e["a0"], e["a1"])
            key = f"{sub}|{fac[mover]}|{describe(e['a0'], tr, mover)}>{describe(e['a1'], tr, mover)}"
            if key not in code:
                code[key] = len(code)
            col.append(code[key])
            T.append(s * om * R)
            T0.append(s * om * z)
            F.append(f)
    T, T0, F, c = np.asarray(T), np.asarray(T0), np.asarray(F), np.asarray(col)
    names = sorted(code, key=code.get)
    idx = c * FOLDS + F
    size = len(names) * FOLDS
    out = {"match": match, "games": games, "side_games": side_games, "decisions": int(len(T)), "cells": names}
    for label, v in (("T", T), ("T0", T0)):
        out[f"s_{label}"] = np.bincount(idx, weights=v, minlength=size).reshape(-1, FOLDS).round(6).tolist()
        out[f"ss_{label}"] = np.bincount(idx, weights=v * v, minlength=size).reshape(-1, FOLDS).round(6).tolist()
    out["n"] = np.bincount(idx, minlength=size).reshape(-1, FOLDS).tolist()
    with gzip.open(HERE / f"round9_provenance_unit_cells_{match}.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(out, fh)
    print(f"{match}: {games} games, {len(T)} band decisions, {len(names)} cells")


def merged():
    data = []
    for m in MATCHES:
        path = HERE / f"round9_provenance_unit_cells_{m}.json.gz"
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                data.append(json.load(fh))
    cells = sorted({c for d in data for c in d["cells"]})
    at = {c: i for i, c in enumerate(cells)}
    acc = {k: np.zeros((len(cells), FOLDS)) for k in ("n", "s_T", "ss_T", "s_T0", "ss_T0")}
    for d in data:
        rows = [at[c] for c in d["cells"]]
        for k in acc:
            acc[k][rows] += np.asarray(d[k])
    return data, cells, acc


def additive_pred(cells, s, n, lam):
    """Ridge fit of cell means on e(a1's unit) - e(a0's unit), per (faction, type)."""
    def units(c):
        _sub, fac, pair = c.split("|", 2)
        u0, u1 = pair.split(">")
        return f"{fac}|{u0}", f"{fac}|{u1}"
    names = sorted({u for c in cells for u in units(c)})
    at = {u: i for i, u in enumerate(names)}
    X = np.zeros((len(cells), len(names)))
    for i, c in enumerate(cells):
        u0, u1 = units(c)
        X[i, at[u1]] += 1.0
        X[i, at[u0]] -= 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(n > 0, s / np.maximum(n, 1), 0.0)
    A = X.T @ (X * n[:, None]) + lam * np.eye(len(names))
    return X @ np.linalg.solve(A, X.T @ (n * m))


def welfare(s, ss, n, sg, rule):
    total = var = switched = 0.0
    for f in range(FOLDS):
        so, no = s.sum(1) - s[:, f], n.sum(1) - n[:, f]
        choose = rule(so, no)
        total += s[choose, f].sum()
        var += ss[choose, f].sum()
        switched += n[choose, f].sum()
    return total / sg, math.sqrt(var) / sg, switched / max(n.sum(), 1)


def v_within(s, n):
    pi = n.sum(1) / n.sum()
    tot = s.sum(0, keepdims=True)
    return float(np.sum(pi * pc.u_square(s, n)) - pc.u_square(tot, n.sum(0, keepdims=True))[0])


def jack(s, n):
    full = v_within(s, n)
    reps = np.asarray([v_within(np.delete(s, g, 1), np.delete(n, g, 1)) for g in range(FOLDS)])
    return full, math.sqrt((FOLDS - 1) / FOLDS * np.sum((reps - reps.mean()) ** 2))


def combine() -> None:
    data, cells, a = merged()
    sg = sum(d["side_games"] for d in data)
    nd = sum(d["decisions"] for d in data)
    subs = np.array([c.split("|", 1)[0] for c in cells])
    unknown = sum(a["n"][i].sum() for i, c in enumerate(cells) if "leader:?" in c)
    print(f"matches {[d['match'] for d in data]}: {sum(d['games'] for d in data)} games, {nd} band decisions, "
          f"{len(cells)} cells; {nd / sg:.2f} per side-game; keys with an unknown leader {unknown / nd:.4f}")
    for gname, members in GROUPS.items():
        if members is None:
            mask = ~np.isin(subs, ["recruit>recruit:other_unit", "recruit>recruit:same_unit"])
        else:
            mask = np.isin(subs, list(members))
        gc = [c for c, m in zip(cells, mask) if m]
        n = a["n"][mask]
        print(f"\n== {gname}: {len(gc)} cells, {int(n.sum())} labels, {n.sum() / sg:.2f} per side-game")
        for label in ("T", "T0"):
            s, ss = a[f"s_{label}"][mask], a[f"ss_{label}"][mask]
            ns, sums, sss = n.sum(1), s.sum(1), ss.sum(1)
            big = ns >= 200
            K = int(big.sum())
            zc = sums[big] / np.sqrt(sss[big])
            exc = (np.sum(zc ** 2) - K) / math.sqrt(2 * K) if K else float("nan")
            v, se = jack(s, n)
            line = (f"  [{label:2s}] excess dispersion {exc:+.2f} SD over {K} cells with >= 200 labels; "
                    f"V {v:.3e} ({se:.2e}) z {v / se:+.2f}, cell-mean sd >= {math.sqrt(max(v - 2 * se, 0)):.4f} at 2 SE")
            print(line)
            rules = {"(a) cell mean > 0": lambda so, no: (no > 0) & (so > 0),
                     "(b) cell mean > 0, >= 1,000 labels": lambda so, no: (no >= 1000) & (so > 0)}
            if gname == "move>move:other_unit":
                rules["(c) additive unit values, ridge 2000"] = (
                    lambda so, no, gc=gc: (no > 0) & (additive_pred(gc, so, no, 2000.0) > 0))
            for rname, rule in rules.items():
                wz, wse, share = welfare(s, ss, n, sg, rule)
                print(f"      {rname:38s} W {wz:+.5f} ({wse:.5f}) z = {wz * ELO:+.1f} ({wse * ELO:.1f}) Elo per "
                      f"side-game; switches {share:.3f}")
    print("\nlargest |z| cells outside recruit>recruit, >= 2,000 labels (T):")
    mask = ~np.isin(subs, ["recruit>recruit:other_unit", "recruit>recruit:same_unit"])
    s, ss, n = a["s_T"].sum(1), a["ss_T"].sum(1), a["n"].sum(1)
    zs = s / np.sqrt(np.maximum(ss, 1e-12))
    order = [i for i in np.argsort(-np.abs(zs)) if mask[i] and n[i] >= 2000][:10]
    big = int(np.sum(mask & (n >= 2000)))
    print(f"  ({big} such cells; the largest of {big} null z values has expected size about "
          f"{math.sqrt(2 * math.log(max(big, 2))):.1f})")
    for i in order:
        print(f"  {cells[i]:70s} {int(n[i]):7d} {s[i] / n[i]:+.4f} ({math.sqrt(ss[i]) / n[i]:.4f}) z {zs[i]:+.2f}")


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(sys.argv[2])
    else:
        combine()
