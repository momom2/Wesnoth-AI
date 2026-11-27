"""The matches from the switch as a mean and as a line through the rounds:
the mean of a set of matches, its difference from the switch match, and the
slope of a weighted least-squares line, each with its standard error. The
matches are taken as independent (they are not quite: see match_pairs.py).
Standard library only.

    python -I -X utf8 match_line.py --out line.json <run dir>/matches.json 624 640 656 672 688

The switch match (round 608's network on the new code) is not in the run's
matches.json; its figures are the registration's and are passed by hand.
"""
import argparse
import json
from pathlib import Path


def line(points: list) -> dict:
    """Weighted least squares of elo on the round, weights 1/se^2."""
    w = [1.0 / se ** 2 for _, _, se in points]
    sw = sum(w)
    mx = sum(wi * x for wi, (x, _, _) in zip(w, points)) / sw
    my = sum(wi * y for wi, (_, y, _) in zip(w, points)) / sw
    sxx = sum(wi * (x - mx) ** 2 for wi, (x, _, _) in zip(w, points))
    slope = sum(wi * (x - mx) * (y - my) for wi, (x, y, _) in zip(w, points)) / sxx
    return {"rounds": [p[0] for p in points], "elo_per_round": slope,
            "standard_error": sxx ** -0.5, "mean_round": mx, "elo_at_mean_round": my}


def mean(points: list) -> dict:
    n = len(points)
    return {"rounds": [p[0] for p in points], "elo": sum(p[1] for p in points) / n,
            "standard_error": sum(p[2] ** 2 for p in points) ** 0.5 / n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("matches", type=Path)
    ap.add_argument("steps", nargs="+", type=int)
    ap.add_argument("--switch", nargs=3, type=float, default=(608, 183.7, 14.0),
                    metavar=("ROUND", "ELO", "SE"), help="the switch match, from the registration")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    by_step = {r["step"]: r for r in json.loads(args.matches.read_text(encoding="utf-8"))}
    points = [(s, by_step[s]["elo"], by_step[s]["se"]) for s in args.steps]
    switch = (int(args.switch[0]), args.switch[1], args.switch[2])
    out = {"matches": [{"round": s, "elo": e, "se": se, "wins": by_step[s]["wins"]} for s, e, se in points],
           "switch_match": {"round": switch[0], "elo": switch[1], "se": switch[2]},
           "mean_all": mean(points), "line_all": line(points) if len(points) > 2 else None,
           "line_with_switch": line([switch] + points)}
    if len(points) > 1:
        out["mean_without_first"] = mean(points[1:])
    for key in ("mean_all", "mean_without_first"):
        if key in out:
            m = out[key]
            out[key + "_less_switch"] = {"elo": m["elo"] - switch[1],
                                         "standard_error": (m["standard_error"] ** 2 + switch[2] ** 2) ** 0.5}
    print(json.dumps(out, indent=1))
    if args.out:
        args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
