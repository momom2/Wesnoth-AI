"""Round 10, invent stance: does `own2` collapse its recruit mixture as `own1` does?

Reads the 6,402 games of S0(b) (training/metrics/own2_base_20261020/
games_own2_vs_own1.tar.gz): `own2` on one seat, `own1` on the other, both at the
reference decode with 64 slots, no coins. Reports, per seat's network, the
readings of round10_invent_composition.py (turn-1 recruits, distinct types,
commonest type's share, P(all one type | >= 3 recruits), P(repeat); whole
game: recruits, distinct types, Herfindahl index, P(repeat)), pooled and by
faction. Standard library and numpy only.
"""
from __future__ import annotations

import gzip
import json
import tarfile
from pathlib import Path

from round10_invent_composition import side_turn_recruits, stats, summarize

ROOT = Path(__file__).resolve().parents[3]
GAMES = ROOT / "training" / "metrics" / "own2_base_20261020" / "games_own2_vs_own1.tar.gz"


def main():
    rows = []
    with tarfile.open(GAMES, "r:gz") as tf:
        for m in tf:
            if not m.name.endswith(".game.jsonl.gz"):
                continue
            rec = json.loads(gzip.decompress(tf.extractfile(m).read()).decode())
            for seat in ("a", "b"):
                p = rec["players"][seat]
                side = int(p["side"])
                net = "own2" if "own2" in (p.get("label", "") + p.get("spec", "")) else "own1"
                rows.append((net, rec["setup"][f"faction{side}"], stats(side_turn_recruits(rec["commands"], side))))
    print(f"S0(b) sides read: {len(rows)}")
    for net in ("own2", "own1"):
        summarize(net, [s for n, _, s in rows if n == net])
    for fac in sorted({f for _, f, _ in rows}):
        print(f" {fac}")
        for net in ("own2", "own1"):
            summarize(net, [s for n, f, s in rows if n == net and f == fac])


if __name__ == "__main__":
    main()
