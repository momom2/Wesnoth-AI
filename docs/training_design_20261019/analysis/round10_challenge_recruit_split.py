"""Round 10, challenge stance: is the recruit near-tie structure (rule R) a
property of the units' value under own1's continuation, or a by-product of
the factored joint argmax splitting a unit's mass over castle hexes?

Two modes, one process each, numpy and the standard library only:

    python round10_challenge_recruit_split.py extract MATCH   # one games tarball -> sums (JSON)
    python round10_challenge_recruit_split.py combine         # all matches -> readings (stdout)

extract reads training/metrics/near_tie_coins_20261020/games_MATCH.tar.gz and
rows_MATCH.jsonl, forms each band decision's label as
round9_provenance_cells.py does (T = s * omega * (z - beta.S), beta from
readout_b.json, capped games dropped, 20 folds by SHA-256 of the game name),
and writes, per (faction, unit):
  - recruits: the unit's recruit commands (all recruits, coined or not);
  - same:     band decisions whose two candidates recruit this unit on two
              different hexes (the unit's own mass split over hexes, gap < 0.35);
  - top:      band decisions whose top candidate recruits this unit;
and, per (faction, u0 > u1, same_hex, fold) for different-unit recruit pairs,
n, sum T, sum T^2. It also counts recruits per side-game and the share of
recruits that are band decisions.

combine:
  1. census: recruits per side-game; band share; same-hex share of
     different-unit pairs.
  2. the additive unit values v (ridge 5,000 labels, the progressive loss's
     choice in round9_synthesis_checks.out section 1), fitted on all folds,
     and the hex-split rate h(u) = same(u) / recruits(u);
  3. the hypothesis test, cross-fitted: in each fold, the different-unit
     pairs whose runner-up unit has the higher hex-split rate (rates computed
     on the other 19 folds) against those whose runner-up has the lower; the
     mean label of each group. A joint-argmax artifact predicts the runner-up
     is better (label > 0) when it is the more split unit and worse when it
     is the less split one;
  4. the weighted correlation of v(u) with log h(u) within faction;
  5. the welfare of a rule that uses only h: switch iff h(a1) > h(a0)
     (rates out of fold), Elo per side-game, against R's +21.3.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REC = HERE.parents[2] / "training" / "metrics" / "near_tie_coins_20261020"
MATCHES = ("b00", "b01", "b02", "b03", "b04", "b05", "b06")
FOLDS = 20
ELO = 800 / math.log(10)
COLUMNS = ("hp", "material", "material_nonterminal", "terminal", "value_50", "value_100", "value_200")
RIDGE = 5000.0


def fold_of(game: str) -> int:
    return hashlib.sha256(game.encode("utf-8")).digest()[0] % FOLDS


def load_beta() -> np.ndarray:
    return np.asarray(json.loads((REC / "readout_b.json").read_text())["beta"]["values"], dtype=float)


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


def recruit_counts(rec: dict, fac: dict) -> dict:
    """Recruit commands per (faction, unit), sides read from init_side markers."""
    out = defaultdict(int)
    side = None
    for c in rec["commands"]:
        if c[0] == "init_side":
            side = int(c[1])
        elif c[0] == "recruit" and side is not None:
            out[f"{fac[side]}|{c[1]}"] += 1
    return out


def extract(match: str) -> None:
    beta = load_beta()
    rows = luck_rows(match)
    recruits, same, top = defaultdict(int), defaultdict(int), defaultdict(int)
    pairs = defaultdict(lambda: [0, 0.0, 0.0])
    games = side_games = band_recruit = 0
    for game, rec in records(match):
        row = rows.get(game)
        if row is None or not row.get("decisive"):
            continue
        z = int(row["z"])
        side_a = int(row["side_a"])
        R = z - float(np.dot(beta, [row["S"][c] for c in COLUMNS]))
        fac = {side_a: row["faction_a"], 3 - side_a: row["faction_b"]}
        f = fold_of(game)
        games += 1
        side_games += 2
        for k, v in recruit_counts(rec, fac).items():
            recruits[k] += v
        for e in (rec.get("coins") or {}).get("decisions", []):
            if e.get("refused"):
                continue
            a0, a1 = e["a0"], e["a1"]
            if a0["kind"] != "recruit":
                continue
            side = int(e["side"])
            u0 = f"{fac[side]}|{a0['unit']}"
            top[u0] += 1
            band_recruit += 1
            if a1["kind"] != "recruit":
                continue
            if a0["unit"] == a1["unit"]:
                if a0.get("to") != a1.get("to"):
                    same[u0] += 1
                continue
            p, d = float(e["p"]), int(e["d"])
            s = 1 if side == side_a else -1
            t = s * (d - p) / (p * (1 - p)) * R
            sh = int(a0.get("to") == a1.get("to"))
            key = f"{u0}>{a1['unit']}|{sh}|{f}"
            acc = pairs[key]
            acc[0] += 1
            acc[1] += t
            acc[2] += t * t
    out = {"match": match, "games": games, "side_games": side_games, "band_recruit": band_recruit,
           "recruits": dict(recruits), "same": dict(same), "top": dict(top), "pairs": dict(pairs)}
    (HERE / f"round10_challenge_recruit_split_{match}.json").write_text(json.dumps(out))
    print(f"{match}: {games} games, {sum(recruits.values())} recruits, {band_recruit} band recruit decisions")


def load_all():
    data = [json.loads((HERE / f"round10_challenge_recruit_split_{m}.json").read_text()) for m in MATCHES]
    tot = {"games": 0, "side_games": 0, "band_recruit": 0}
    recruits, same, top = defaultdict(int), defaultdict(int), defaultdict(int)
    rows = []
    for d in data:
        for k in tot:
            tot[k] += d[k]
        for name, acc in (("recruits", recruits), ("same", same), ("top", top)):
            for k, v in d[name].items():
                acc[k] += v
        for key, (n, s, ss) in d["pairs"].items():
            pair, sh, f = key.rsplit("|", 2)
            u0, u1name = pair.split(">")
            fac = u0.split("|")[0]
            rows.append((u0, f"{fac}|{u1name}", int(sh), int(f), n, s, ss))
    return tot, recruits, same, top, rows


def ridge_values(rows, units, lam, keep):
    """Additive unit values: minimise sum over pairs of (S - n (v1 - v0))^2 / n + lam |v|^2,
    i.e. the label-weighted least squares of cell means on e(u1) - e(u0)."""
    at = {u: i for i, u in enumerate(units)}
    A = np.zeros((len(units), len(units)))
    b = np.zeros(len(units))
    for u0, u1, sh, f, n, s, ss in rows:
        if not keep(f):
            continue
        x = np.zeros(len(units))
        x[at[u1]] += 1
        x[at[u0]] -= 1
        A += n * np.outer(x, x)
        b += s * x
    return np.linalg.solve(A + lam * np.eye(len(units)), b), at


def split_rate(recruits, same, unit):
    r = recruits.get(unit, 0)
    return (same.get(unit, 0) + 0.5) / (r + 1.0)


def combine() -> None:
    tot, recruits, same, top, rows = load_all()
    sg = tot["side_games"]
    n_rec = sum(recruits.values())
    n_pairs = sum(r[4] for r in rows)
    n_same_hex = sum(r[4] for r in rows if r[2] == 1)
    print(f"matches {list(MATCHES)}: {tot['games']} decisive games, {sg} side-games")
    print("1. census")
    print(f"  recruits per side-game {n_rec / sg:.2f}; band decisions with a recruit on top {tot['band_recruit'] / sg:.2f}"
          f" ({tot['band_recruit'] / n_rec:.3f} of recruits)")
    print(f"  different-unit recruit pairs {n_pairs / sg:.2f} per side-game ({n_pairs / n_rec:.3f} of recruits);"
          f" on the same hex {n_same_hex / max(1, n_pairs):.3f}")
    print(f"  same-unit pairs on two hexes {sum(same.values()) / sg:.2f} per side-game")

    units = sorted({r[0] for r in rows} | {r[1] for r in rows})
    v, at = ridge_values(rows, units, RIDGE, lambda f: True)
    h = {u: split_rate(recruits, same, u) for u in units}
    print("\n2. per unit: recruits per side-game, hex-split rate h, additive value v (ridge 5,000, all folds)")
    for fac in sorted({u.split('|')[0] for u in units}):
        us = sorted([u for u in units if u.startswith(fac + "|")], key=lambda u: -v[at[u]])
        print(f"  {fac}")
        for u in us:
            print(f"    {u.split('|')[1]:22s} {recruits.get(u, 0) / sg:6.3f}  h {h[u]:.4f}  v {v[at[u]]:+.4f}")

    print("\n3. label mean by whether the runner-up unit is the more hex-split (h uses no labels)")
    groups = {+1: [0, 0.0, 0.0], -1: [0, 0.0, 0.0]}
    for u0, u1, sh, f, n, s, ss in rows:
        g = 1 if h[u1] > h[u0] else -1
        acc = groups[g]
        acc[0] += n
        acc[1] += s
        acc[2] += ss
    for g, name in ((+1, "runner-up more split"), (-1, "runner-up less split")):
        n, s, ss = groups[g]
        m = s / n
        se = math.sqrt(ss / n - m * m) / math.sqrt(n)
        print(f"  {name:22s} n {n:8d} ({n / sg:.2f} per side-game)  mean label {m:+.4f} ({se:.4f})  z {m / se:+.2f}")
    print("  (h is computed from recruit and band-pair counts only, never from labels, so the split cannot fit the labels)")

    print("\n4. weighted correlation of v(u) with log h(u), within faction (weights: recruits)")
    xs, ys, ws = [], [], []
    for fac in sorted({u.split('|')[0] for u in units}):
        us = [u for u in units if u.startswith(fac + "|")]
        w = np.array([recruits.get(u, 0) for u in us], dtype=float)
        x = np.log([h[u] for u in us])
        y = np.array([v[at[u]] for u in us])
        if w.sum() == 0 or len(us) < 2:
            continue
        x = x - np.average(x, weights=w)
        y = y - np.average(y, weights=w)
        xs.extend(x)
        ys.extend(y)
        ws.extend(w)
    xs, ys, ws = map(np.asarray, (xs, ys, ws))
    r = np.sum(ws * xs * ys) / math.sqrt(np.sum(ws * xs * xs) * np.sum(ws * ys * ys))
    print(f"  pooled within-faction weighted r = {r:+.3f} over {len(xs)} units")
    rng = np.random.default_rng(20261021)
    null = []
    for _ in range(2000):
        perm = np.empty_like(xs)
        start = 0
        for fac in sorted({u.split('|')[0] for u in units}):
            k = sum(1 for u in units if u.startswith(fac + "|"))
            idx = np.arange(start, start + k)
            perm[idx] = xs[rng.permutation(idx)]
            start += k
        null.append(np.sum(ws * perm * ys) / math.sqrt(np.sum(ws * perm * perm) * np.sum(ws * ys * ys)))
    null = np.asarray(null)
    print(f"  within-faction permutation p (two-sided) {np.mean(np.abs(null) >= abs(r)):.3f}")

    print("\n5. welfare of 'switch iff h(a1) > h(a0)' (no labels used), first order, per side-game")
    n, s, ss = groups[+1]
    w_z = s / sg
    se_z = math.sqrt(ss) / sg
    print(f"  {w_z:+.5f} ({se_z:.5f}) z = {w_z * ELO:+.1f} ({se_z * ELO:.1f}) Elo per side-game,"
          f" {n / sg:.2f} switches per side-game")
    print("  (Elo as in round9_provenance_recruit_cells: z per side-game x 800/ln10)")


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(sys.argv[2])
    else:
        combine()
