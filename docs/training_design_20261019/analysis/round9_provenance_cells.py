"""Round 9, provenance stance: what batch 1's coin labels say about the spread of
near-tie advantages across observable cells, before any head is fitted.

Two modes, each one process, numpy and the standard library only:

    python round9_provenance_cells.py extract MATCH   # one games tarball -> per-cell sums (JSON)
    python round9_provenance_cells.py combine         # all extracted matches -> readings (stdout)

extract reads training/metrics/near_tie_coins_20261020/games_MATCH.tar.gz (the
game records, which hold every band decision's side, index, command index, gap,
P, D, a0 and a1) and rows_MATCH.jsonl (z and the seven luck sums per game), and
forms the label exactly as tools/coin_labels.py does:

    T_i = s_i * omega_i * (z - beta . S),  omega_i = (D_i - P_i) / (P_i (1 - P_i)),

beta the batch's frozen coefficients (readout_b.json), s_i = +1 when the mover
is side A; T0_i the same without the luck columns. Capped games are dropped.
Each game goes to one of 20 folds by SHA-256 of its name. Per partition of the
band decisions into cells, it writes n, sum T, sum T^2 per (cell, fold).

combine reads every extracted match and, per partition:
- V_between: the variance of the cell means of the true advantage, by a
  U-statistic over distinct folds (unbiased: folds hold independent games),
  with a delete-one-fold jackknife SE. It is a lower bound on the variance of
  the per-decision advantage (law of total variance).
- W_cf: the cross-fitted welfare of "switch where the cell mean is above 0":
  each fold's rule is chosen on the other 19 folds and evaluated on its own
  labels; per side-game, in z and in Elo (x 800 / ln 10). Unbiased for the
  rules' average first-order gain under the coin policy; SE from the labels'
  second moment (labels within a game are uncorrelated, round 8).
It also prints the label's measured second moment, and the same-kind pairs'
means by gap bin (whether a same-kind mean vanishes at a tie, the premise of
the head's antisymmetric constants, round 6).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import sys
import tarfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REC = HERE.parents[2] / "training" / "metrics" / "near_tie_coins_20261020"
MATCHES = ("b00", "b01", "b02", "b03", "b04", "b05", "b06")
FOLDS = 20
BAND, BIN = 0.35, 0.05
ELO = 800 / math.log(10)
KIND = {"move": "move", "attack": "attack", "recruit": "recruit", "end_turn": "end_turn"}
TURN_EDGES = (1, 2, 3, 5, 8, 12, 10_000)          # turn bins [1], [2], [3,4], [5,7], [8,11], [12,+)
PARTITIONS = ("all", "type", "type_gap", "type_sub", "type_turn", "type_faction", "type_map",
              "type_mover", "type_sub_gap", "type_sub_turn", "rr_units")


def fold_of(game: str) -> int:
    return hashlib.sha256(game.encode("utf-8")).digest()[0] % FOLDS


def gap_bin(g: float) -> int:
    return min(int(g / BIN + 1e-9), int(round(BAND / BIN)) - 1)


def turn_bin(t: int) -> int:
    for k in range(len(TURN_EDGES) - 1):
        if TURN_EDGES[k] <= t < TURN_EDGES[k + 1]:
            return k
    return len(TURN_EDGES) - 2


def sub_kind(a0: dict, a1: dict) -> str:
    """The ordered pair type, refined where both candidates share a kind."""
    k0, k1 = a0["kind"], a1["kind"]
    base = f"{k0}>{k1}"
    if k0 != k1:
        return base
    if k0 == "recruit":
        return base + (":same_unit" if a0["unit"] == a1["unit"] else ":other_unit")
    if k0 == "move":
        return base + (":same_unit" if a0["from"] == a1["from"] else ":other_unit")
    if k0 == "attack":
        if a0["from"] == a1["from"] and a0["to"] == a1["to"]:
            return base + ":weapon"
        return base + (":same_target" if a0["to"] == a1["to"] else ":other_target")
    return base


def load_beta() -> np.ndarray:
    return np.asarray(json.loads((REC / "readout_b.json").read_text())["beta"]["values"], dtype=float)


COLUMNS = ("hp", "material", "material_nonterminal", "terminal", "value_50", "value_100", "value_200")


def luck_rows(match: str) -> dict:
    out = {}
    with open(REC / f"rows_{match}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if "game" in r and "S" in r:
                out[r["game"]] = r
    return out


def records(match: str):
    with tarfile.open(REC / f"games_{match}.tar.gz", "r:*") as tf:
        for m in tf:
            name = m.name.rsplit("/", 1)[-1]
            if m.isfile() and name.endswith(".game.jsonl.gz"):
                game = name[: -len(".game.jsonl.gz")]
                lines = gzip.decompress(tf.extractfile(m).read()).decode("utf-8").splitlines()
                recs = [json.loads(x) for x in lines if x.strip()]
                rec = next((r for r in recs if r.get("game_label") == game), recs[0])
                yield game, rec


def extract(match: str) -> None:
    beta = load_beta()
    rows = luck_rows(match)
    codes = {p: {} for p in PARTITIONS}
    cols = {p: [] for p in PARTITIONS}
    T, T0, F = [], [], []
    games = side_games = 0
    omega2_sum = 0.0
    r2_by_game = []
    for game, rec in records(match):
        row = rows.get(game)
        if row is None or not row.get("decisive"):
            continue
        z = int(row["z"])
        side_a = int(row["side_a"])
        R = z - float(np.dot(beta, [row["S"][c] for c in COLUMNS]))
        fac = {side_a: row["faction_a"], 3 - side_a: row["faction_b"]}
        scen = row["scenario_id"]
        f = fold_of(game)
        games += 1
        side_games += 2
        cum, n = [], 0
        for c in rec["commands"]:
            if c[0] == "init_side":
                n += 1
            cum.append(n)
        decisions = (rec.get("coins") or {}).get("decisions", [])
        r2_by_game.append((R * R, len(decisions)))
        for e in decisions:
            if e.get("refused"):
                continue
            side, p, d = int(e["side"]), float(e["p"]), int(e["d"])
            s = 1 if side == side_a else -1
            om = (d - p) / (p * (1 - p))
            omega2_sum += om * om
            T.append(s * om * R)
            T0.append(s * om * z)
            F.append(f)
            a0, a1 = e["a0"], e["a1"]
            typ = f"{a0['kind']}>{a1['kind']}"
            sub = sub_kind(a0, a1)
            gb = gap_bin(float(e["gap"]))
            k = int(e.get("cmd", 0))
            turn = (cum[min(k, len(cum) - 1)] + 1) // 2
            tb = turn_bin(turn)
            mover = "first" if side == 1 else "second"
            keys = {"all": "all", "type": typ, "type_gap": f"{typ}|g{gb}", "type_sub": sub,
                    "type_turn": f"{typ}|t{tb}", "type_faction": f"{typ}|{fac[side]}",
                    "type_map": f"{typ}|{scen}", "type_mover": f"{typ}|{mover}",
                    "type_sub_gap": f"{sub}|g{gb}", "type_sub_turn": f"{sub}|t{tb}",
                    "rr_units": (f"{fac[side]}|{a0['unit']}>{a1['unit']}"
                                 if sub == "recruit>recruit:other_unit" else "rest")}
            for pname, key in keys.items():
                cmap = codes[pname]
                if key not in cmap:
                    cmap[key] = len(cmap)
                cols[pname].append(cmap[key])
    T, T0, F = np.asarray(T), np.asarray(T0), np.asarray(F)
    out = {"match": match, "games": games, "side_games": side_games, "decisions": int(len(T)),
           "sum_T2": float(np.sum(T * T)), "sum_T02": float(np.sum(T0 * T0)), "sum_omega2": omega2_sum,
           "mean_R2_games": float(np.mean([a for a, _ in r2_by_game])),
           "mean_R2_weighted": float(sum(a * b for a, b in r2_by_game) / max(1, sum(b for _, b in r2_by_game))),
           "partitions": {}}
    for pname in PARTITIONS:
        names = sorted(codes[pname], key=codes[pname].get)
        c = np.asarray(cols[pname])
        idx = c * FOLDS + F
        size = len(names) * FOLDS
        part = {"cells": names}
        for label, v in (("T", T), ("T0", T0)):
            part[f"s_{label}"] = np.bincount(idx, weights=v, minlength=size).reshape(-1, FOLDS).tolist()
            part[f"ss_{label}"] = np.bincount(idx, weights=v * v, minlength=size).reshape(-1, FOLDS).tolist()
        part["n"] = np.bincount(idx, minlength=size).reshape(-1, FOLDS).tolist()
        out["partitions"][pname] = part
    (HERE / f"round9_provenance_cells_{match}.json").write_text(json.dumps(out))
    print(f"{match}: {games} games, {len(T)} band decisions")


# ---------------------------------------------------------------------
# combine
# ---------------------------------------------------------------------
def merged():
    data = [json.loads((HERE / f"round9_provenance_cells_{m}.json").read_text()) for m in MATCHES
            if (HERE / f"round9_provenance_cells_{m}.json").exists()]
    parts = {}
    for pname in PARTITIONS:
        cells = sorted({c for d in data for c in d["partitions"][pname]["cells"]})
        at = {c: i for i, c in enumerate(cells)}
        acc = {k: np.zeros((len(cells), FOLDS)) for k in ("n", "s_T", "ss_T", "s_T0", "ss_T0")}
        for d in data:
            p = d["partitions"][pname]
            rows = [at[c] for c in p["cells"]]
            for k in acc:
                acc[k][rows] += np.asarray(p[k])
        parts[pname] = (cells, acc)
    return data, parts


def u_square(s: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Per cell, an unbiased-in-the-limit estimate of mu_c^2 from fold sums:
    sum over distinct folds of s_f s_f' over sum of n_f n_f'."""
    num = s.sum(1) ** 2 - (s * s).sum(1)
    den = n.sum(1) ** 2 - (n * n).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, 0.0)


def v_between(s: np.ndarray, n: np.ndarray) -> float:
    N = n.sum()
    pi = n.sum(1) / N
    second = float(np.sum(pi * u_square(s, n)))
    tot = s.sum(0, keepdims=True)
    mean_sq = float(u_square(tot, n.sum(0, keepdims=True))[0])
    return second - mean_sq


def jackknife(fn, s: np.ndarray, n: np.ndarray):
    full = fn(s, n)
    reps = []
    for f in range(FOLDS):
        keep = [k for k in range(FOLDS) if k != f]
        reps.append(fn(s[:, keep], n[:, keep]))
    reps = np.asarray(reps)
    se = math.sqrt((FOLDS - 1) / FOLDS * np.sum((reps - reps.mean()) ** 2))
    return full, se


def cross_fitted_welfare(s: np.ndarray, ss: np.ndarray, n: np.ndarray, side_games: int, min_n: int = 0):
    """Each fold's rule switches the cells whose mean over the other folds is
    above 0 (cells with fewer than min_n out-of-fold labels never switch);
    returns (welfare per side-game, its SE, share of labels switched)."""
    total, var, switched = 0.0, 0.0, 0.0
    for f in range(FOLDS):
        so, no = s.sum(1) - s[:, f], n.sum(1) - n[:, f]
        choose = (no > max(min_n, 0)) & (so > 0)
        total += s[choose, f].sum()
        var += ss[choose, f].sum()
        switched += n[choose, f].sum()
    return total / side_games, math.sqrt(var) / side_games, switched / n.sum()


def combine() -> None:
    data, parts = merged()
    games = sum(d["games"] for d in data)
    sg = sum(d["side_games"] for d in data)
    nd = sum(d["decisions"] for d in data)
    print(f"matches {[d['match'] for d in data]}: {games} decisive games, {nd} band decisions, "
          f"{nd / sg:.2f} per side-game")
    print("\n1. the label's second moment (measured)")
    print(f"  E[omega^2] {sum(d['sum_omega2'] for d in data) / nd:.4f} (exact 5.3333 at P 0.25)")
    print(f"  E[T^2] with the luck columns {sum(d['sum_T2'] for d in data) / nd:.4f}; "
          f"without (z only) {sum(d['sum_T02'] for d in data) / nd:.4f}")
    w = [d["decisions"] for d in data]
    r2 = sum(d["mean_R2_weighted"] * x for d, x in zip(data, w)) / sum(w)
    r2g = sum(d["mean_R2_games"] * d["games"] for d in data) / games
    print(f"  E[R^2] = E[(z - beta.S)^2]: per game {r2g:.4f}, weighted by band decisions {r2:.4f}")

    print("\n2. spread of the true advantage across cells (U-statistic over folds; jackknife SE)")
    print("  partition: cells | V_between (SE) | sd lower bound | cross-fitted switch welfare per side-game"
          " z (SE) = Elo (SE) | share switched")
    for pname in PARTITIONS:
        if pname == "all":
            continue
        cells, a = parts[pname]
        for label in ("T", "T0"):
            s, ss, n = a[f"s_{label}"], a[f"ss_{label}"], a["n"]
            vb, se = jackknife(v_between, s, n)
            wz, wse, share = cross_fitted_welfare(s, ss, n, sg)
            sdlb = math.sqrt(max(vb - 2 * se, 0.0))
            print(f"  {pname:14s} [{label:2s}] {len(cells):4d} | {vb:+.3e} ({se:.2e}) z {vb / se:+.2f} | "
                  f"sd >= {sdlb:.4f} at 2 SE | {wz:+.5f} ({wse:.5f}) = {wz * ELO:+.1f} ({wse * ELO:.1f}) Elo | {share:.3f}")

    print("\n3. typed means by gap bin, with the luck columns (mean, SE from the second moment)")
    cells, a = parts["type_sub_gap"]
    s, ss, n = a["s_T"].sum(1), a["ss_T"].sum(1), a["n"].sum(1)
    by = {}
    for c, si, ssi, ni in zip(cells, s, ss, n):
        typ, gb = c.split("|")
        by.setdefault(typ, {})[gb] = (si / ni, math.sqrt(ssi) / ni, ni)
    for typ in sorted(by, key=lambda t: -sum(v[2] for v in by[t].values())):
        tot_n = sum(v[2] for v in by[typ].values())
        if tot_n < 150_000:
            continue
        cells_txt = "  ".join(f"g{k[1:]} {v[0]:+.4f}({v[1]:.4f})" for k, v in sorted(by[typ].items()))
        ms = sum(v[0] * v[2] for v in by[typ].values()) / tot_n
        mse = math.sqrt(sum((v[1] * v[2]) ** 2 for v in by[typ].values())) / tot_n
        print(f"  {typ:32s} n {int(tot_n):8d} mean {ms:+.4f} ({mse:.4f}), z {ms / mse:+.2f}, "
              f"{tot_n / sg:.2f} per side-game")
        print(f"      {cells_txt}")

    print("\n4. sub-kinds pooled over gaps, with Elo of switching all of a sub-kind (first order)")
    cells, a = parts["type_sub"]
    for c, si, ssi, ni in sorted(zip(cells, a["s_T"].sum(1), a["ss_T"].sum(1), a["n"].sum(1)),
                                 key=lambda x: -x[3]):
        m, se = si / ni, math.sqrt(ssi) / ni
        print(f"  {c:36s} n {int(ni):8d} ({ni / sg:5.2f} per side-game) mean {m:+.4f} ({se:.4f}) z {m / se:+.2f}"
              f"  all switched {si / sg * ELO:+.1f} ({math.sqrt(ssi) / sg * ELO:.1f}) Elo")

    print("\n5. the selection check: is the largest typed mean reproduced across halves of the folds?")
    cells, a = parts["type"]
    halves = (list(range(0, FOLDS, 2)), list(range(1, FOLDS, 2)))
    for h, (pick, check) in enumerate((halves, halves[::-1])):
        sp, np_ = a["s_T"][:, pick].sum(1), a["n"][:, pick].sum(1)
        ssp = a["ss_T"][:, pick].sum(1)
        zp = sp / np.sqrt(ssp)
        top = int(np.argmax(np.abs(zp)))
        sc, nc, ssc = a["s_T"][top, check].sum(), a["n"][top, check].sum(), a["ss_T"][top, check].sum()
        print(f"  half {h}: largest |z| {cells[top]} z {zp[top]:+.2f} (mean {sp[top] / np_[top]:+.4f}); "
              f"other half {sc / nc:+.4f} ({math.sqrt(ssc) / nc:.4f}) z {sc / math.sqrt(ssc):+.2f}")


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(sys.argv[2])
    else:
        combine()
