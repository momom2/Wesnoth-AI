"""Round 10, invent stance: per-decision rows of batch 1's recruit near-ties.

For every band decision of one coin match whose two candidates are recruits of
different unit types, writes (game key, mover side, command index, coin D, mover faction, a0's type, a1's type,
n0 and n1 = how many of each type the mover had recruited earlier in the game,
the mover's side-turn, the margin, the label T). T is rebuilt exactly as
tools/coin_labels.py on exp/near-tie-coins does and as
round9_invent_cells.py does: T = s (D - P)/(P(1 - P)) (z - beta . S), beta of
configs/luck_beta_20261019.json, capped games dropped. Every column is fixed
before the decision's coin, so any rule built on them is read without bias.

Usage: python round10_invent_recruit_extract.py b01 0 2   (match, part, parts:
every parts-th game record from offset part, so a run stays near a minute)
writes round10_invent_recruit_rows_<match>_p<part>.json.gz. Standard library and
numpy only; no simulator, no network.
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
BETA = (0.0011885942625018315, 0.00028509220077075834, -0.00048074045056809554,
        -0.11838146825305246, 0.603040987021951, 0.0541244809103413, 0.022519783857895107)


def command_context(commands):
    """Per command index: (mover side, its side-turn, recruits by type of that
    side before the command)."""
    side, turn = None, defaultdict(int)
    army = defaultdict(lambda: defaultdict(int))
    ctx = []
    for c in commands:
        if c[0] == "init_side":
            side = int(c[1])
            turn[side] += 1
        ctx.append((side, turn[side], dict(army[side])))
        if c[0] == "recruit" and side is not None:
            army[side][c[1]] += 1
    return ctx


def main(match, part, parts):
    walk = {}
    with open(REC / f"rows_{match}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if "game" in r and "S" in r:
                walk[r["game"]] = r
    rows = {k: [] for k in ("game", "side", "cmd", "d", "fac", "u0", "u1", "n0", "n1", "tot", "st", "gap", "T")}
    games = 0
    with tarfile.open(REC / f"games_{match}.tar.gz", "r:gz") as tf:
        seen = -1
        for m in tf:
            if not m.name.endswith(".game.jsonl.gz"):
                continue
            seen += 1
            if seen % parts != part:
                continue
            game = Path(m.name).name[: -len(".game.jsonl.gz")]
            w = walk.get(game)
            if w is None or not w.get("decisive"):
                continue
            rec = json.loads(gzip.decompress(tf.extractfile(m).read()).decode())
            side_a = int(rec["players"]["a"]["side"])
            z = int(w["z"])
            if (int(rec["winner"]) == side_a) != (z == 1):
                raise ValueError(f"{game}: winner and z disagree")
            R = z - sum(b * float(w["S"][c]) for b, c in zip(BETA, COLUMNS))
            ctx = command_context(rec["commands"])
            games += 1
            for e in rec["coins"]["decisions"]:
                a0, a1 = e["a0"], e["a1"]
                if a0["kind"] != "recruit" or a1["kind"] != "recruit" or a0["unit"] == a1["unit"]:
                    continue
                p, d = float(e["p"]), int(e["d"])
                mover = int(e["side"])
                s = 1 if mover == side_a else -1
                side, st, army = ctx[int(e["cmd"])]
                if side != mover:
                    raise ValueError(f"{game}: coin side {mover} at a command of side {side}")
                rows["game"].append(f"{match}/{game}")
                rows["side"].append(mover)
                rows["cmd"].append(int(e["cmd"]))
                rows["d"].append(d)
                rows["fac"].append(rec["setup"][f"faction{mover}"])
                rows["u0"].append(a0["unit"])
                rows["u1"].append(a1["unit"])
                rows["n0"].append(army.get(a0["unit"], 0))
                rows["n1"].append(army.get(a1["unit"], 0))
                rows["tot"].append(sum(army.values()))
                rows["st"].append(st)
                rows["gap"].append(float(e["gap"]))
                rows["T"].append(s * (d - p) / (p * (1 - p)) * R)
    out = OUT / f"round10_invent_recruit_rows_{match}_p{part}.json.gz"
    out.write_bytes(gzip.compress(json.dumps({"match": match, "part": part, "parts": parts, "games": games,
                                               "rows": rows}).encode("utf-8")))
    print(match, part, "games", games, "rows", len(rows["T"]), "->", out.name)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
