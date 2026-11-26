"""The crash barriers of a span of the simple loop's rounds (measurement 2 of
docs/audit_fixes_prereg_20261125.md): the exact check, the gradient's and the
update's norms whole and by parameter group, the value head's cross-entropy
and the pass's time, from each round's grad.json and pi.step.json. One row
per span, so that two spans can be set side by side. Standard library only.

    python -I -X utf8 round_checks.py --out checks.json <run dir> 593 608 610 624
"""
import argparse
import json
import statistics
from pathlib import Path


def spread(values: list) -> dict:
    out = {"mean": statistics.mean(values), "min": min(values), "max": max(values)}
    if len(values) > 1:
        out["standard_error"] = statistics.stdev(values) / len(values) ** 0.5
    return out


def read(run: Path, first: int, last: int) -> dict:
    series: dict = {}
    finite = True
    for k in range(first, last + 1):
        directory = run / f"round_{k:04d}"
        grad = json.loads((directory / "grad.json").read_text(encoding="utf-8"))
        step = json.loads((directory / "pi.step.json").read_text(encoding="utf-8"))
        finite = finite and bool(grad["finite"])
        row = {
            "exact_check": grad["replay"]["half_mean_square"],
            "gradient_norm": grad["norm"],
            "update_norm": step["update_norm"],
            "value_cross_entropy": grad["value"]["cross_entropy"],
            "decisions": grad["decisions"],
            "pass_seconds": sum(g["seconds"] for g in grad["groups"]),
        }
        for name, value in grad["norm_by_group"].items():
            row[f"gradient_norm.{name}"] = value
        for name, value in step["update_norms"].items():
            row[f"update_norm.{name}"] = value
        for name, value in row.items():
            series.setdefault(name, []).append(value)
    return {"rounds": [first, last], "count": last - first + 1, "all_finite": finite,
            **{name: spread(values) for name, values in series.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("bounds", nargs="+", type=int, help="first and last round of each span")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if len(args.bounds) % 2:
        ap.error("the rounds come in pairs: first last [first last ...]")
    spans = [read(args.run, args.bounds[i], args.bounds[i + 1]) for i in range(0, len(args.bounds), 2)]
    names = [n for n in spans[0] if n not in ("rounds", "count", "all_finite")]
    print("".ljust(28), *(f"{s['rounds'][0]}-{s['rounds'][1]}".rjust(12) for s in spans))
    for name in names:
        print(name.ljust(28), *(f"{s[name]['mean']:12.4g}" for s in spans))
    print("all finite".ljust(28), *(str(s["all_finite"]).rjust(12) for s in spans))
    if args.out:
        args.out.write_text(json.dumps(spans, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
