"""Round 9, critique of the simplify stance: how C1's frozen recruit rule clusters its switches.

C1's prediction (+22.0 +- 4.7 Elo) adds the coin labels of every decision the rule switches, as if each
switch were made alone. The labels were measured where the coin switched a decision with P 1/4, so a
second switch in the same side-turn was mostly measured after an unswitched first one; under the rule,
it follows a switched one. This script counts how much of the rule's switching sits in such sequences,
and reads, on the labels themselves, whether a switch's label differs when an earlier switch of the same
side-turn was played (D = 1) against not played (D = 0). Both are exact under the coin's randomisation:
H_j contains D_i, so the label at j is unbiased for the advantage given the earlier coin.

Rule: the additive form c + h(fac, u1) - h(fac, u0) fitted on b00-b03's cell sums (the simplify stance's
round9_simplify_recruit_spread.py, section 7), switched where the fit is > 0. Read on the game records of
the matches named on the command line (default b04), labels T = s omega (z - beta.S) with batch 1's beta,
as round9_provenance_cells.py forms them. numpy and the standard library only.
"""
import gzip
import importlib.util
import json
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REC = HERE.parents[2] / "training" / "metrics" / "near_tie_coins_20261020"
COLUMNS = ("hp", "material", "material_nonterminal", "terminal", "value_50", "value_100", "value_200")

spec = importlib.util.spec_from_file_location("rs", HERE / "round9_simplify_recruit_spread.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)


def frozen_rule():
    cells, per = rs.load_by_match("T")
    fit_on = ("b00", "b01", "b02", "b03")
    sf = sum(x[1] for x in per if x[0] in fit_on)
    nf = sum(x[3] for x in per if x[0] in fit_on)
    fitted, _ = rs.additive_fit(cells, sf[:, None], nf[:, None])
    return {c for c, v in zip(cells, fitted) if v > 0}


def games(match):
    rows = {}
    with open(REC / f"rows_{match}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if "game" in r and "S" in r and r.get("decisive"):
                rows[r["game"]] = r
    with tarfile.open(REC / f"games_{match}.tar.gz", "r:*") as tf:
        for m in tf:
            name = m.name.rsplit("/", 1)[-1]
            if not (m.isfile() and name.endswith(".game.jsonl.gz")):
                continue
            game = name[: -len(".game.jsonl.gz")]
            if game not in rows:
                continue
            lines = gzip.decompress(tf.extractfile(m).read()).decode("utf-8").splitlines()
            recs = [json.loads(x) for x in lines if x.strip()]
            rec = next((r for r in recs if r.get("game_label") == game), recs[0])
            yield rows[game], rec


def main():
    matches = sys.argv[1:] or ["b04"]
    rule = frozen_rule()
    beta = np.asarray(json.loads((REC / "readout_b.json").read_text())["beta"]["values"], dtype=float)
    side_games = 0
    per_turn = defaultdict(int)            # switches in a side-turn -> number of side-turns
    switches = after_switched = after_any = 0
    side_after_any = side_after_switched = 0   # preceded by a rule switch of the same side, any earlier side-turn
    split = {0: [], 1: []}                 # label at a switch, by D of the previous switch in its side-turn
    side_split = {0: [], 1: []}            # label at a switch in a later side-turn, by whether any earlier one played a1
    for row, rec in (g for m in matches for g in games(m)):
        side_games += 2
        side_a = int(row["side_a"])
        fac = {side_a: row["faction_a"], 3 - side_a: row["faction_b"]}
        R = int(row["z"]) - float(np.dot(beta, [row["S"][c] for c in COLUMNS]))
        cum, n = [], 0
        for c in rec["commands"]:
            if c[0] == "init_side":
                n += 1
            cum.append(n)
        last = {}                          # side-turn -> D of its latest rule-switched decision
        side_seen = {}                     # side -> 1 if an earlier side-turn of it had a rule switch with D = 1, else 0
        side_turn_of = {}                  # side -> side-turn of its latest rule switch
        count = defaultdict(int)
        for e in (rec.get("coins") or {}).get("decisions", []):
            if e.get("refused"):
                continue
            a0, a1 = e["a0"], e["a1"]
            if not (a0["kind"] == "recruit" and a1["kind"] == "recruit" and a0["unit"] != a1["unit"]):
                continue
            side = int(e["side"])
            if f"{fac[side]}|{a0['unit']}>{a1['unit']}" not in rule:
                continue
            k = int(e.get("cmd", 0))
            st = cum[min(k, len(cum) - 1)]          # side-turn index (init_side count)
            p, d = float(e["p"]), int(e["d"])
            T = (1 if side == side_a else -1) * (d - p) / (p * (1 - p)) * R
            switches += 1
            count[st] += 1
            if st in last:
                after_any += 1
                after_switched += last[st]
                split[last[st]].append(T)
            last[st] = d
            if side in side_turn_of and side_turn_of[side] != st:
                side_after_any += 1
                side_after_switched += side_seen[side]
                side_split[side_seen[side]].append(T)
            side_seen[side] = max(side_seen.get(side, 0), d)   # any earlier rule switch of this side played a1
            side_turn_of[side] = st
        for v in count.values():
            per_turn[v] += 1
    print(f"matches {matches}: {side_games // 2} decisive games; rule of {len(rule)} cells (additive, fitted on b00-b03)")
    print(f"rule switches {switches} = {switches / side_games:.2f} per side-game (the stance: 2.28 on b04-b06)")
    print(f"switches preceded by another rule switch in the same side-turn: {after_any} = {after_any / switches:.3f} "
          f"of switches; of those, the earlier one coined to a1 (D = 1): {after_switched} ({after_switched / max(after_any, 1):.3f})")
    print(f"switches preceded by a rule switch of the same side in an earlier side-turn: {side_after_any} = "
          f"{side_after_any / switches:.3f} of switches; of those, with an earlier one coined to a1: "
          f"{side_after_switched} ({side_after_switched / max(side_after_any, 1):.3f})")
    tot = sum(per_turn.values())
    print("side-turns with k rule switches (k: share of side-turns with any): "
          + ", ".join(f"{k}: {v / tot:.3f}" for k, v in sorted(per_turn.items())))
    for name, sp in (("same side-turn", split), ("earlier side-turn", side_split)):
        for d in (0, 1):
            a = np.asarray(sp[d])
            if len(a):
                print(f"[{name}] label at a switch after an earlier switch with D = {d}: n {len(a)}, "
                      f"mean {a.mean():+.4f} (SE {np.sqrt(np.mean(a * a) / len(a)):.4f})")
        a0, a1 = np.asarray(sp[0]), np.asarray(sp[1])
        if len(a0) and len(a1):
            diff = a1.mean() - a0.mean()
            se = np.sqrt(np.mean(a1 * a1) / len(a1) + np.mean(a0 * a0) / len(a0))
            print(f"[{name}] difference (D = 1 minus D = 0): {diff:+.4f} (SE {se:.4f}), z {diff / se:+.2f}")


if __name__ == "__main__":
    main()
