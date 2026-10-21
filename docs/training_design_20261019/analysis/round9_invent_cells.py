"""Round 9, invent stance: near-tie coin labels of batch 1 by pre-coin cells.

Reads one coin match of training/metrics/near_tie_coins_20261020 (its game
records in games_<M>.tar.gz, its walk rows in rows_<M>.jsonl), rebuilds every
band decision's label exactly as tools/coin_labels.py does on the branch
exp/near-tie-coins (T = s * (D - P)/(P(1 - P)) * (z - beta . S), beta from
configs/luck_beta_20261019.json, capped games dropped), and accumulates
game-clustered sums per cell. Every cell is defined by quantities fixed before
the coin (the pair's kinds and actors, the recruit types and their counts so
far, the side-turn, the gap), so its mean label is unbiased for the cell's mean
advantage. Post-coin facts (whether the other option was played next) are
counted, never used to cut labels.

Usage: python round9_invent_cells.py b01 [PART]   (PART 0 or 1: every other game record, so each
run stays under a minute; writes round9_invent_cells_b01[_pPART].json)
Standard library and numpy only; no simulator, no network.
"""
import gzip
import json
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REC = ROOT / "training" / "metrics" / "near_tie_coins_20261020"
OUT = Path(__file__).resolve().parent

COLUMNS = ("hp", "material", "material_nonterminal", "terminal", "value_50", "value_100", "value_200")
# configs/luck_beta_20261019.json on exp/near-tie-coins (the batch-1 labels' beta)
BETA = (0.0011885942625018315, 0.00028509220077075834, -0.00048074045056809554,
        -0.11838146825305246, 0.603040987021951, 0.0541244809103413, 0.022519783857895107)
BAND = 0.35
SHORT = {"attack": "a", "move": "m", "recruit": "r", "end_turn": "e"}


def unit_costs():
    units = json.loads((ROOT / "unit_stats.json").read_text(encoding="utf-8"))["units"]
    return {k: int(v.get("cost", 0)) for k, v in units.items()}


def side_turn_bucket(t):
    return "1" if t == 1 else "2" if t == 2 else "3-4" if t <= 4 else "5-8" if t <= 8 else "9+"


def same_actor(a0, a1):
    k0, k1 = a0["kind"], a1["kind"]
    if k0 == "end_turn" or k1 == "end_turn":
        return "na"
    if k0 == "recruit" or k1 == "recruit":
        if k0 == k1:
            return "same" if a0["unit"] == a1["unit"] else "diff"
        return "diff"
    return "same" if a0["from"] == a1["from"] else "diff"


def command_context(commands):
    """Per command index: (mover side, its side-turn, its action index within
    the turn, recruits by type of that side before the command)."""
    side, turn = None, defaultdict(int)
    within = 0
    army = defaultdict(lambda: defaultdict(int))
    ctx = []
    for c in commands:
        if c[0] == "init_side":
            side = int(c[1])
            turn[side] += 1
            within = 0
        ctx.append((side, turn[side], within, dict(army[side])))
        if c[0] == "recruit" and side is not None:
            army[side][c[1]] += 1
        if c[0] != "init_side":
            within += 1
    return ctx


def played_next(commands, k, other):
    """Post-coin, descriptive: is the other option the next command?"""
    j = k + 1
    if j < len(commands) and commands[k][0] == "move" and j < len(commands) and commands[j][0] == "attack":
        j += 1
    if j >= len(commands):
        return False
    c = commands[j]
    kind = other["kind"]
    if kind == "move":
        return c[0] == "move" and [c[1][0], c[2][0]] == other["from"] and [c[1][-1], c[2][-1]] == other["to"]
    if kind == "recruit":
        return c[0] == "recruit" and c[1] == other["unit"]
    if kind == "attack":
        if c[0] == "attack":
            return [int(c[3]), int(c[4])] == other["to"] and int(c[5]) == int(other["weapon"])
        return c[0] == "move" and [c[1][0], c[2][0]] == other["from"]
    return c[0] == "end_turn"


class Cells:
    def __init__(self):
        self.acc = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0, 0])  # n, sumT, sum_g S^2, sum_g n S, sum_g n^2, games

    def add_game(self, per_game):
        for key, (n, s) in per_game.items():
            a = self.acc[key]
            a[0] += n
            a[1] += s
            a[2] += s * s
            a[3] += n * s
            a[4] += n * n
            a[5] += 1


def main(match, part=None):
    costs = unit_costs()
    walk = {}
    with open(REC / f"rows_{match}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if "game" in r and "S" in r:
                walk[r["game"]] = r
    cells = Cells()
    counts = defaultdict(int)
    with tarfile.open(REC / f"games_{match}.tar.gz", "r:gz") as tf:
        seen = -1
        for m in tf:
            if not m.name.endswith(".game.jsonl.gz"):
                continue
            seen += 1
            if part is not None and seen % 2 != part:
                continue
            game = Path(m.name).name[: -len(".game.jsonl.gz")]
            w = walk.get(game)
            if w is None:
                counts["unwalked"] += 1
                continue
            if not w.get("decisive"):
                counts["capped"] += 1
                continue
            rec = json.loads(gzip.decompress(tf.extractfile(m).read()).decode())
            side_a = int(rec["players"]["a"]["side"])
            z = int(w["z"])
            if (int(rec["winner"]) == side_a) != (z == 1):
                raise ValueError(f"{game}: winner and z disagree")
            R = z - sum(b * float(w["S"][c]) for b, c in zip(BETA, COLUMNS))
            commands = rec["commands"]
            ctx = command_context(commands)
            per_game = defaultdict(lambda: [0, 0.0])
            counts["games"] += 1
            for e in rec["coins"]["decisions"]:
                a0, a1 = e["a0"], e["a1"]
                p, d, gap, k = float(e["p"]), int(e["d"]), float(e["gap"]), int(e["cmd"])
                s = 1 if int(e["side"]) == side_a else -1
                T = s * (d - p) / (p * (1 - p)) * R
                pair = SHORT[a0["kind"]] + ">" + SHORT[a1["kind"]]
                actor = same_actor(a0, a1)
                side, st, within, army = ctx[k]
                stb = side_turn_bucket(st)
                b = min(int(gap / 0.05), 6)
                keys = ["pooled", f"type:{pair}", f"type:{pair}|actor:{actor}", f"type:{pair}|st:{stb}",
                        f"type:{pair}|bin:{b}", f"fine:{pair}|{actor}|{b}|{stb}", f"st:{stb}",
                        f"actor:{actor}"]
                if pair == "a>a":
                    keys.append("aa|weapon_only:%d" % int(a0["from"] == a1["from"] and a0["to"] == a1["to"]))
                if pair == "r>r":
                    if a0["unit"] == a1["unit"]:
                        keys += ["rr|same_unit", f"rr|same_unit|st:{stb}"]
                    else:
                        c0, c1 = costs.get(a0["unit"], 0), costs.get(a1["unit"], 0)
                        n0, n1 = army.get(a0["unit"], 0), army.get(a1["unit"], 0)
                        keys += ["rr|diff_unit", f"rr|diff_unit|st:{stb}",
                                 "rr|diff_unit|cost:" + ("a1_dearer" if c1 > c0 else "a1_cheaper" if c1 < c0 else "equal"),
                                 "rr|diff_unit|army:" + ("a1_fewer" if n1 < n0 else "a1_more" if n1 > n0 else "equal"),
                                 "rr|diff_unit|a1_new:%d" % int(n1 == 0),
                                 f"rrpair:{a0['unit']}>{a1['unit']}"]
                for key in keys:
                    g = per_game[key]
                    g[0] += 1
                    g[1] += T
                counts["decisions"] += 1
                counts[f"n:{pair}|{actor}"] += 1
                if actor == "diff" and pair in ("m>m", "a>m", "m>a", "a>a"):
                    other = a0 if d else a1
                    counts[f"next_is_other:{pair}|d{d}"] += int(played_next(commands, k, other))
                    counts[f"next_total:{pair}|d{d}"] += 1
            cells.add_game(per_game)
    out = OUT / (f"round9_invent_cells_{match}.json" if part is None else f"round9_invent_cells_{match}_p{part}.json")
    out.write_text(json.dumps({"match": match, "counts": counts, "cells": cells.acc}), encoding="utf-8")
    print(match, dict(list(counts.items())[:4]), "cells", len(cells.acc), "->", out.name)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
