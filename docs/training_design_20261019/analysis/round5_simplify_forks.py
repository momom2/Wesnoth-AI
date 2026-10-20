"""Round 5, simplify stance: why rollouts_true.jsonl cannot give playout noise to the game's end.

Input: training/metrics/belief_worlds_20261012/rollouts_true.jsonl. Its rollouts run from a
candidate turn's end to the fourth player-turn start (docs/belief_worlds_prereg_20261012.md;
round5_invent_coupling_record.py), so a rollout carries a game result only when the game ended
inside that window. This script counts those rollouts and the cells (candidate turns) with at
least two of them. It replaces an earlier version of this file whose docstring wrongly said the
rollouts continued to the game's end; that version's within-cell variance (0.188 over 127 cells)
came from the few games that ended within four player-turns and is not a playout noise. The
playout noise to the game's end is round5_provenance_playout_noise.py's 0.608 (temperature 0.5).

Standard library only.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
SRC = ROOT / "training/metrics/belief_worlds_20261012/rollouts_true.jsonl"
OUT = pathlib.Path(__file__).with_suffix(".json")


def main():
    rollouts = with_result = cells_two = 0
    for line in SRC.open():
        row = json.loads(line)
        results = [r["result"] for r in row.get("rollouts", []) if r.get("result") is not None]
        rollouts += len(row.get("rollouts", []))
        with_result += len(results)
        cells_two += len(results) >= 2
    out = {"rollouts": rollouts, "rollouts_with_a_result": with_result,
           "share_with_a_result": with_result / rollouts, "cells_with_two_results": cells_two,
           "usable_as_playout_noise_to_the_end": False}
    OUT.write_text(json.dumps(out, indent=1))
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
