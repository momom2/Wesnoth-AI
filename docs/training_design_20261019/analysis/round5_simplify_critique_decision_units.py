"""Round 5, critique of the simplify stance: does the band count's derivation
(band share over prior_gaps readings x own1's decisions from
round5_provenance_stop_acts.py) multiply two different decision units?

prior_gaps (tools/prior_gaps.py at a72071a) reads one decision per command in
{move, attack, recruit, recall, end_turn}, merging a move with the attack that
follows it from its last hex only when the replayed argmax is that attack
(10 merges in 27,171 readings, prior_gaps.json). round5_provenance_stop_acts.py
merges every such move-then-attack and adds no end_turn (the simplify stance
adds one per side-turn).

Reads whole game records (commands only) of own1's self-play and of parity3 at
64 slots, and per side-game counts:
  raw    = commands in DECISION_KINDS (the prior_gaps unit, up to its rare merges)
  merged = recruits + attacks + moves not followed by an attack from their last
           hex, plus one end_turn per side-turn (the simplify stance's unit)
Standard library and numpy only.
"""
import gzip
import json
import tarfile
from pathlib import Path

import numpy as np

METRICS = Path(__file__).resolve().parents[3] / "training/metrics"
DECISION_KINDS = {"move", "attack", "recruit", "recall", "end_turn"}
MATCHES = {
    "gen_own1_selfplay (own1 both sides)": ("memory_own_round2_20261011/games_gen_own1_selfplay.tar.gz", None),
    "slots64_vs_slots0 (parity3 64 slots, side A only)": ("parity3_baselines_20261008/games_slots64_vs_slots0.tar.gz", "parity3"),
}


def counts(commands):
    out, side = {}, None
    for i, c in enumerate(commands):
        if c[0] == "init_side":
            side = c[1]
            out.setdefault(side, {"raw": 0, "merged": 0, "turns": 0})
            continue
        if c[0] not in DECISION_KINDS:
            continue
        rec = out[side]
        rec["raw"] += 1
        if c[0] == "end_turn":
            rec["turns"] += 1
            rec["merged"] += 1
        elif c[0] in ("recruit", "attack", "recall"):
            rec["merged"] += 1
        elif c[0] == "move":
            nxt = commands[i + 1] if i + 1 < len(commands) else None
            if not (nxt and nxt[0] == "attack" and (nxt[1], nxt[2]) == (c[1][-1], c[2][-1])):
                rec["merged"] += 1
    return out


def main():
    for name, (tarball, label) in MATCHES.items():
        raw, merged, turns = [], [], []
        with tarfile.open(METRICS / tarball) as t:
            for m in t.getmembers():
                if not m.name.endswith(".game.jsonl.gz"):
                    continue
                rec = json.loads(gzip.decompress(t.extractfile(m).read()).decode().splitlines()[0])
                per_side = counts(rec["commands"])
                sides = per_side.keys()
                if label is not None:
                    p = rec["players"]
                    sides = [p[k]["side"] for k in ("a", "b") if p[k]["label"] == label]
                for s in sides:
                    if s in per_side:
                        raw.append(per_side[s]["raw"])
                        merged.append(per_side[s]["merged"])
                        turns.append(per_side[s]["turns"])
        raw, merged = np.array(raw, float), np.array(merged, float)
        print(f"{name}: side-games {len(raw)}, raw decisions per side-game {raw.mean():.1f}, "
              f"merged (+ end_turn) {merged.mean():.1f}, ratio raw/merged {raw.sum() / merged.sum():.3f}, "
              f"side-turns {np.mean(turns):.1f}")
    print("prior_gaps (parity3 vs parity2, 100 side-games): 27171 readings = 271.7 per side-game, 10 merges")


if __name__ == "__main__":
    main()
