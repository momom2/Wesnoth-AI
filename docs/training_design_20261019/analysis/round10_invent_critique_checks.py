"""Round 10, critique of the invent stance: two light checks.

1. Hex splitting at recruit decisions (batch-1 match b00, every game record):
   recruit band decisions whose two candidates are the SAME unit type on two
   hexes (the type's mass split within 0.35 nats) against those of two
   different types, per side-game, and recruits per side-game. HA's q_t(o) is
   the type's actor mass summed over hexes; the reference decode takes the
   joint argmax over (type, hex). Where a type's mass is split, the two
   orderings are different objects.
2. Information per game on one fixed rule's value, labels against a match,
   from measured SEs: a rule's label welfare SE at 60,000 coin games
   (round10_invent_recruit_rules.out section 2) against a one-seat match's
   347.4/sqrt(N) (current_design.md section 3.4).
Standard library only.
"""
import gzip
import json
import math
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REC = ROOT / "training" / "metrics" / "near_tie_coins_20261020" / "games_b00.tar.gz"


def hex_split():
    games = same = diff = recruits = 0
    with tarfile.open(REC, "r:gz") as tf:
        for m in tf:
            if not m.name.endswith(".game.jsonl.gz"):
                continue
            rec = json.loads(gzip.decompress(tf.extractfile(m).read()).decode())
            games += 1
            recruits += sum(1 for c in rec["commands"] if c[0] == "recruit")
            for e in rec["coins"]["decisions"]:
                a0, a1 = e["a0"], e["a1"]
                if a0["kind"] == "recruit" and a1["kind"] == "recruit":
                    if a0["unit"] == a1["unit"]:
                        same += 1
                    else:
                        diff += 1
    sg = 2 * games
    print(f"1. b00: {games} games; recruits {recruits / sg:.2f} per side-game; recruit band pairs: "
          f"same type, two hexes {same / sg:.2f} per side-game; different types {diff / sg:.2f}")


def info_ratio():
    elo = 800 / math.log(10)
    for name, se in (("C", 3.2), ("R", 3.1)):
        n_eq = (elo / se) ** 2
        print(f"2. {name}: label welfare SE {se} Elo from 60,000 coin games; a one-seat match reaches it at "
              f"{n_eq:,.0f} games; information per game, match / labels = {60_000 / n_eq:.2f}; "
              f"per dollar at $2.598e-5 (pool) against $2.849e-5 (coin) = {60_000 / n_eq * 2.849 / 2.598:.2f}")


if __name__ == "__main__":
    hex_split()
    info_ratio()
