"""Round 9, invent stance: where two twin games would first part.

Twins: two games on one seed (same setup, same keyed dice), each with its own
independent coins. At argmax both are the same game until the first band
decision where their coins differ; there they are an exact fork of one
history. With independent coins of propensity P at each band decision, the
twins agree there with probability P^2 + (1 - P)^2.

Reads the coin logs of batch 1 (rec["coins"]["decisions"], P and command index
per band decision, in game order) from games_b00.tar.gz and every fourth game
record of games_b01.tar.gz, and computes, per game, the exact distribution of
the first disagreement: its band index, its command index as a share of the
game's commands (the prefix both twins replay), its own side-turn and pair
type. Standard library only; no simulator, no network.

Usage: python round9_invent_twin_prefix.py
"""
import gzip
import json
import tarfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REC = ROOT / "training" / "metrics" / "near_tie_coins_20261020"
SHORT = {"attack": "a", "move": "m", "recruit": "r", "end_turn": "e"}


def side_turns(commands):
    side, turn, out = None, defaultdict(int), []
    for c in commands:
        if c[0] == "init_side":
            side = int(c[1])
            turn[side] += 1
        out.append(turn[side] if side is not None else 0)
    return out


def first_split(decisions, n_commands, st_of_cmd, acc):
    alive = 1.0  # probability the twins still agree before this band decision
    for e in sorted(decisions, key=lambda e: (int(e["cmd"]), int(e["i"]))):
        p = float(e["p"])
        split = alive * (1.0 - p * p - (1.0 - p) * (1.0 - p))
        k = int(e["cmd"])
        acc["band_index"] += split * (acc["_idx"] + 1)
        acc["prefix_share"] += split * (k / n_commands)
        st = st_of_cmd[k]
        acc["st:" + ("1" if st == 1 else "2" if st == 2 else "3+")] += split
        acc["pair:" + SHORT[e["a0"]["kind"]] + ">" + SHORT[e["a1"]["kind"]]] += split
        acc["side:%d" % int(e["side"])] += split
        acc["_idx"] += 1
        alive -= split
    acc["never"] += alive
    acc["games"] += 1
    acc["_idx"] = 0


def main():
    acc = defaultdict(float)
    for name, every in (("games_b00.tar.gz", 1), ("games_b01.tar.gz", 4)):
        with tarfile.open(REC / name, "r:gz") as tf:
            seen = -1
            for m in tf:
                if not m.name.endswith(".game.jsonl.gz"):
                    continue
                seen += 1
                if seen % every:
                    continue
                rec = json.loads(gzip.decompress(tf.extractfile(m).read()).decode())
                commands = rec["commands"]
                first_split(rec["coins"]["decisions"], len(commands), side_turns(commands), acc)
    g = acc["games"]
    print(f"games {g:.0f} (b00 all, b01 every fourth)")
    print(f"expected band index of the first disagreement: {acc['band_index'] / g:.3f} (geometric at P 0.25: {1 / 0.375:.3f})")
    print(f"expected share of the game's commands before it (replayed by both twins): {acc['prefix_share'] / g:.4f}")
    print(f"twins that never part: {acc['never'] / g:.2e}")
    for key in sorted(k for k in acc if k.startswith(("st:", "pair:", "side:"))):
        print(f"  first disagreement at {key:10s} {acc[key] / g:.3f}")


if __name__ == "__main__":
    main()
