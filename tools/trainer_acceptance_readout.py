"""The training acceptance's readout (docs/trainer_speed_20261013.md, rounds 3
and 4; scripts/trainer_acceptance_box.sh): passes from the same checkpoint
under the same seed, A (the padded trunk, today's arithmetic), A2 (A again:
the GPU's run-to-run nondeterminism) and arms that each change the
arithmetic or the method (B, the fastest configuration, and B's variations),
compared on their final holdout probes over the same holdout game-sides.

    python tools/trainer_acceptance_readout.py --a A.probe_sides.jsonl [--a2 A2.probe_sides.jsonl] \\
        --arm B=B.probe_sides.jsonl [--arm NAME=FILE ...] [--out acceptance.json]

Each file is a pass's <out>.probe_sides.jsonl (tools/sequence_train.py); its
final probe gives, per holdout game-side and per probe run, the decisions,
the summed policy cross-entropy over them, the positions and the summed
belief loss. A pass is read only when it holds A's game-sides with A's
counts (the same holdout, the same labels): an arm or A2 that does not, or
whose file cannot be read, is set aside with its reason (`refused`), and the
others are read without it (without A2, with no bound).

Per reading (ce_all and belief at each probe run: k0, k64, k64_reset) the
pooled value of each pass (as the probe's ce_all and belief: summed over
every decision or position, divided by their count), the noise A2 - A, and
per arm its difference from A, from A2 and, for an arm other than B, from B
(the arm's own factor on top of B's), each with its standard error paired
over holdout games: the two sides of a game are not independent draws, so a
game is the unit (cluster) and the variance is the ratio estimator's,
G / (G - 1) * sum_g (e_g - D * n_g)^2 / N^2, where e_g is the game's summed
difference, n_g its count, N the total and D the difference.

The criterion (round 1's, pre-registered by the lead), per arm: for ce_all
at k0 and k64 and belief at k0 and k64, |arm - A| is no larger than |A2 - A|
plus two standard errors of arm - A (`pass`); `not_worse` is the one-sided
reading, arm - A no larger than |A2 - A| plus two standard errors. Without
A2 there is no bound: both are None and the differences stand alone.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

READINGS = (("ce_all", 0, 1), ("belief", 2, 3))     # (name, count index, sum index) in a side's run entry
JUDGED = ("ce_all_k0", "ce_all_k64", "belief_k0", "belief_k64")
REFERENCES = ("A", "A2")
ARM_NAME = re.compile(r"^[A-Za-z0-9_]+$")
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


def _arm_reading(name: str, g: Dict[str, Dict[str, Tuple[int, float]]], noise: Optional[Dict]) -> Dict:
    """One arm's reading: its value, its differences and the criterion."""
    entry = {"value": pooled(g[name]), "minus_A": paired_difference(g[name], g["A"])}
    if "A2" in g:
        entry["minus_A2"] = paired_difference(g[name], g["A2"])
    if name != "B" and "B" in g:
        entry["minus_B"] = paired_difference(g[name], g["B"])
    new = entry["minus_A"]
    if noise is not None and new["se"] is not None and noise["diff"] is not None:
        bound = abs(noise["diff"]) + 2.0 * new["se"]
        entry.update(bound=bound, within=abs(new["diff"]) <= bound, not_worse=new["diff"] <= bound)
    return entry


def _verdict(readings: Dict[str, Dict], key: str) -> Optional[bool]:
    """The criterion over the judged readings; None when one has no bound."""
    judged = [readings[k] for k in JUDGED if k in readings]
    if not judged or any(key not in r for r in judged):
        return None
    return all(r[key] for r in judged)


def _readable(files: Dict[str, Path]) -> Tuple[Dict[str, Dict[SidesKey, Dict]], Dict[str, str], List[str]]:
    """(the passes read, the reason each other pass was set aside, A's probe
    runs): A must be read; every other pass is checked against A alone."""
    a = final_sides(files["A"])
    runs = check_same_holdout({"A": a})
    passes, refused = {"A": a}, {}
    for name, path in files.items():
        if name == "A":
            continue
        try:
            sides = final_sides(path)
            check_same_holdout({"A": a, name: sides})
            passes[name] = sides
        except (OSError, ValueError, KeyError, SystemExit) as e:     # one bad pass leaves the others read
            refused[name] = str(e) or type(e).__name__
    return passes, refused, runs


def readout(a: Path, a2: Optional[Path], arms: Dict[str, Path]) -> Dict:
    """Every arm against A, with A2 - A as the noise (module docstring)."""
    if not arms or set(arms) & set(REFERENCES):
        raise SystemExit(f"arms {sorted(arms)}: at least one, none named {' or '.join(REFERENCES)}")
    files = {"A": a, **({"A2": a2} if a2 is not None else {}), **arms}
    passes, refused, runs = _readable(files)
    readings: Dict[str, Dict] = {}
    by_arm: Dict[str, Dict[str, Dict]] = {name: {} for name in arms if name in passes}
    for run in runs:
        for reading, count, total in READINGS:
            key = f"{reading}_{run}"
            g = {p: per_game(sides, run, count, total) for p, sides in passes.items()}
            noise = paired_difference(g["A2"], g["A"]) if "A2" in g else None
            readings[key] = {"A": pooled(g["A"])}
            if noise is not None:
                readings[key].update(A2=pooled(g["A2"]), A2_minus_A=noise)
            for name in by_arm:
                by_arm[name][key] = _arm_reading(name, g, noise)
    result_arms = {}
    for name in arms:
        if name in by_arm:
            r = by_arm[name]
            result_arms[name] = {"pass": _verdict(r, "within"), "not_worse": _verdict(r, "not_worse"), "readings": r}
        else:
            result_arms[name] = {"pass": None, "not_worse": None, "refused": refused[name], "readings": {}}
    return {"files": {k: str(v) for k, v in files.items()},
            "game_sides": len(passes["A"]), "games": len({f for f, _ in passes["A"]}),
            "judged": [k for k in JUDGED if k in readings], "readings": readings,
            "refused": refused, "arms": result_arms}


def _fmt(d: Optional[Dict]) -> str:
    if d is None or d["diff"] is None:
        return "-"
    return f"{d['diff']:+.5f}" + ("" if d["se"] is None else f"+-{d['se']:.5f}")


def table(result: Dict) -> str:
    lines = [f"{'reading':<18} {'A':>9} {'A2-A':>18}"]
    for key, r in result["readings"].items():
        lines.append(f"{key:<18} {r['A']:9.5f} {_fmt(r.get('A2_minus_A')):>18}")
    if "A2" in result.get("refused", {}):
        lines.append(f"A2 set aside: {result['refused']['A2']}")
    for name, arm in result["arms"].items():
        if "refused" in arm:
            lines += ["", f"{name}: set aside: {arm['refused']}"]
            continue
        lines += ["", f"{name:<18} {'value':>9} {name + '-A':>18} {name + '-B':>18} {'bound':>9}  within"]
        for key, r in arm["readings"].items():
            lines.append(f"{key:<18} {r['value']:9.5f} {_fmt(r['minus_A']):>18} {_fmt(r.get('minus_B')):>18} "
                         f"{r.get('bound', float('nan')):9.5f}  {r.get('within')}")
        lines.append(f"{name}: pass {arm['pass']}  not_worse {arm['not_worse']}")
    lines.append(f"({result['games']} games, {result['game_sides']} game-sides)")
    return "\n".join(lines)


def parse_arm(spec: str) -> Tuple[str, Path]:
    name, sep, path = spec.partition("=")
    if not sep or not ARM_NAME.match(name) or not path:
        raise SystemExit(f"--arm {spec!r}: NAME=FILE, the name of letters, digits and _")
    return name, Path(path)


def main(argv: Sequence[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", type=Path, required=True, help="pass A's .probe_sides.jsonl")
    ap.add_argument("--a2", type=Path, default=None, help="pass A2's (A repeated); without it, no bound")
    ap.add_argument("--arm", action="append", required=True, help="NAME=FILE: an arm's .probe_sides.jsonl")
    ap.add_argument("--out", type=Path, default=None)
    opts = ap.parse_args(argv)
    arms = dict(parse_arm(spec) for spec in opts.arm)
    if len(arms) != len(opts.arm):
        raise SystemExit("an arm named twice")
    result = readout(opts.a, opts.a2, arms)
    print(table(result))
    if opts.out:
        opts.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
