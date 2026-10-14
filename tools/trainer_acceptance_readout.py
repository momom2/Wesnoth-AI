"""The training acceptance's readout (docs/trainer_speed_20261013.md, round 3;
scripts/trainer_acceptance_box.sh): three passes from the same checkpoint
under the same seed, A (the padded trunk, today's arithmetic), A2 (A again:
the GPU's run-to-run nondeterminism) and B (a configuration that changes the
arithmetic), compared on their final holdout probes over the same holdout
game-sides.

    python tools/trainer_acceptance_readout.py --a A.probe_sides.jsonl --a2 A2.probe_sides.jsonl \\
        --b B.probe_sides.jsonl [--out acceptance.json]

Each file is a pass's <out>.probe_sides.jsonl (tools/sequence_train.py); its
final probe gives, per holdout game-side and per probe run, the decisions,
the summed policy cross-entropy over them, the positions and the summed
belief loss. The three must hold the same game-sides with the same counts
(the same holdout, the same labels), else the readout refuses.

Per reading (ce_all and belief at each probe run: k0, k64, k64_reset) the
pooled value of each pass (as the probe's ce_all and belief: summed over
every decision or position, divided by their count) and the differences
A2 - A, B - A and B - A2, each with its standard error paired over holdout
games: the two sides of a game are not independent draws, so a game is the
unit (cluster) and the variance is the ratio estimator's,
G / (G - 1) * sum_g (e_g - D * n_g)^2 / N^2, where e_g is the game's summed
difference, n_g its count, N the total and D the difference.

The criterion (round 1's, pre-registered by the lead): for ce_all at k0 and
k64 and belief at k0 and k64, |B - A| is no larger than |A2 - A| plus two
standard errors of B - A (`pass`); `not_worse` is the one-sided reading,
B - A no larger than |A2 - A| plus two standard errors.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

READINGS = (("ce_all", 0, 1), ("belief", 2, 3))     # (name, count index, sum index) in a side's run entry
JUDGED = ("ce_all_k0", "ce_all_k64", "belief_k0", "belief_k64")
SidesKey = Tuple[str, int]


def final_sides(path: Path) -> Dict[SidesKey, Dict]:
    """The final probe's game-sides of a .probe_sides.jsonl (its last row
    marked final, else its last row)."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{path}: no probe")
    finals = [r for r in rows if r.get("final")]
    row = finals[-1] if finals else rows[-1]
    return {(s["file"], int(s["side"])): s for s in row["sides"]}


def check_same_holdout(passes: Dict[str, Dict[SidesKey, Dict]]) -> List[str]:
    """The probe runs every pass read; refuses passes over other game-sides
    or other counts."""
    names = list(passes)
    first = passes[names[0]]
    for name in names[1:]:
        if set(passes[name]) != set(first):
            raise SystemExit(f"{name} probed other game-sides than {names[0]}")
    runs = sorted(k for k in next(iter(first.values())) if k.startswith("k"))
    for key, side in first.items():
        for name in names[1:]:
            other = passes[name][key]
            for run in runs:
                if (side[run][0], side[run][2]) != (other[run][0], other[run][2]):
                    raise SystemExit(f"{name} counted other decisions or positions at {key} {run}")
    return runs


def per_game(sides: Dict[SidesKey, Dict], run: str, count: int, total: int) -> Dict[str, Tuple[int, float]]:
    """(count, summed loss) per game, both sides pooled."""
    games: Dict[str, Tuple[int, float]] = {}
    for (file, _), side in sides.items():
        n, s = games.get(file, (0, 0.0))
        games[file] = (n + side[run][count], s + side[run][total])
    return games


def paired_difference(x: Dict[str, Tuple[int, float]], y: Dict[str, Tuple[int, float]]) -> Dict:
    """Pooled x minus pooled y and its standard error with games as clusters
    (module docstring); no error from fewer than two games."""
    games = [g for g in x if x[g][0] > 0]
    N = sum(x[g][0] for g in games)
    if N == 0:
        return {"diff": None, "se": None, "n_games": 0}
    e = {g: x[g][1] - y[g][1] for g in games}
    D = sum(e.values()) / N
    if len(games) < 2:
        return {"diff": D, "se": None, "n_games": len(games)}
    var = len(games) / (len(games) - 1) * sum((e[g] - D * x[g][0]) ** 2 for g in games) / N ** 2
    return {"diff": D, "se": math.sqrt(var), "n_games": len(games)}


def pooled(games: Dict[str, Tuple[int, float]]) -> float:
    n = sum(c for c, _ in games.values())
    return sum(s for _, s in games.values()) / n if n else float("nan")


def readout(a: Path, a2: Path, b: Path) -> Dict:
    passes = {"A": final_sides(a), "A2": final_sides(a2), "B": final_sides(b)}
    runs = check_same_holdout(passes)
    readings = {}
    for run in runs:
        for name, count, total in READINGS:
            g = {p: per_game(sides, run, count, total) for p, sides in passes.items()}
            rep = paired_difference(g["A2"], g["A"])
            new = paired_difference(g["B"], g["A"])
            entry = {"A": pooled(g["A"]), "A2": pooled(g["A2"]), "B": pooled(g["B"]),
                     "A2_minus_A": rep, "B_minus_A": new, "B_minus_A2": paired_difference(g["B"], g["A2"])}
            if new["se"] is not None and rep["diff"] is not None:
                bound = abs(rep["diff"]) + 2.0 * new["se"]
                entry.update(bound=bound, within=abs(new["diff"]) <= bound, not_worse=new["diff"] <= bound)
            readings[f"{name}_{run}"] = entry
    judged = [readings[k] for k in JUDGED if k in readings]
    return {"files": {"A": str(a), "A2": str(a2), "B": str(b)},
            "game_sides": len(passes["A"]), "games": len({f for f, _ in passes["A"]}),
            "pass": bool(judged) and all(r.get("within") for r in judged),
            "not_worse": bool(judged) and all(r.get("not_worse") for r in judged),
            "judged": [k for k in JUDGED if k in readings], "readings": readings}


def table(result: Dict) -> str:
    lines = [f"{'reading':<18} {'A':>9} {'A2-A':>18} {'B-A':>18} {'bound':>9}  within"]
    for key, r in result["readings"].items():
        def fmt(d):
            if d["diff"] is None:
                return "-"
            return f"{d['diff']:+.5f}" + ("" if d["se"] is None else f"+-{d['se']:.5f}")
        lines.append(f"{key:<18} {r['A']:9.5f} {fmt(r['A2_minus_A']):>18} {fmt(r['B_minus_A']):>18} "
                     f"{r.get('bound', float('nan')):9.5f}  {r.get('within')}")
    lines.append(f"pass {result['pass']}  not_worse {result['not_worse']}  "
                 f"({result['games']} games, {result['game_sides']} game-sides)")
    return "\n".join(lines)


def main(argv: Sequence[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", type=Path, required=True, help="pass A's .probe_sides.jsonl")
    ap.add_argument("--a2", type=Path, required=True, help="pass A2's (A repeated)")
    ap.add_argument("--b", type=Path, required=True, help="pass B's")
    ap.add_argument("--out", type=Path, default=None)
    opts = ap.parse_args(argv)
    result = readout(opts.a, opts.a2, opts.b)
    print(table(result))
    if opts.out:
        opts.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
