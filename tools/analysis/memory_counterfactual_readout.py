#!/usr/bin/env python3
"""The readings of docs/memory_in_play_prereg_20261002.md from the rows of
tools/analysis/memory_counterfactual.py: per source and turn bucket, how
often the memory player (m) and the 0-slot player (z) choose alike, how
their disagreements split (only m ends the turn, only z does, both act
differently), their mean end_turn and best-other priors, the reproduction
rate of a match source, and on a human source the label's log-prior
difference by label kind. Means of a game-level quantity carry the
standard error between games.

    python tools/analysis/memory_counterfactual_readout.py own=cf_own.jsonl \\
        other=cf_other.jsonl human=cf_human.jsonl [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

BUCKETS = ((1, 5), (6, 10), (11, 15), (16, 10 ** 6))
LATE = (6, 10 ** 6)                          # "turns 6 and later" of the reading rules


def bucket_name(lo: int, hi: int) -> str:
    return f"{lo}+" if hi >= 10 ** 6 else f"{lo}-{hi}"


def mean_se(xs: Iterable[float]) -> Tuple[Optional[float], Optional[float]]:
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, None
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var / len(xs))


def per_game(rows: List[Dict], value) -> List[float]:
    """The per-game means of value(row) over the rows that define it."""
    acc: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        v = value(r)
        if v is not None:
            acc[r["game"]].append(float(v))
    return [sum(v) / len(v) for v in acc.values()]


def ends(choice: Optional[str]) -> bool:
    return choice == "end_turn"


def split(rows: List[Dict]) -> Dict:
    """The decisions' agreement and the split of their disagreements."""
    n = len(rows)
    dis = [r for r in rows if not r["agree"]]
    m_only = sum(ends(r["choice_m"]) and not ends(r["choice_z"]) for r in dis)
    z_only = sum(ends(r["choice_z"]) and not ends(r["choice_m"]) for r in dis)
    both_act = len(dis) - m_only - z_only
    share = (lambda k: round(k / len(dis), 4) if dis else None)
    return {"decisions": n, "agreement": round(1 - len(dis) / n, 4) if n else None,
            "disagreements": len(dis), "only_m_ends": m_only, "only_z_ends": z_only,
            "both_act": both_act, "share_only_m_ends": share(m_only), "share_only_z_ends": share(z_only),
            "share_both_act": share(both_act)}


def priors(rows: List[Dict]) -> Dict:
    out = {}
    for name, value in (("p_end_m", lambda r: r["p_end_m"]), ("p_end_z", lambda r: r["p_end_z"]),
                        ("p_end_diff", lambda r: r["p_end_m"] - r["p_end_z"]),
                        ("max_act_m", lambda r: r["max_act_m"]), ("max_act_z", lambda r: r["max_act_z"]),
                        ("only_m_ends_rate", lambda r: float(ends(r["choice_m"]) and not ends(r["choice_z"]))),
                        ("only_z_ends_rate", lambda r: float(ends(r["choice_z"]) and not ends(r["choice_m"])))):
        m, se = mean_se(per_game(rows, value))
        out[name] = [None if m is None else round(m, 5), None if se is None else round(se, 5)]
    return out


def label_gain(rows: List[Dict]) -> Dict:
    """Per label kind: the mean of log prior(m) - log prior(z) of the
    recorded action (positive: the memory gives the label more mass), per
    game, and the decisions whose label is not among the legal actions."""
    out = {}
    for kind in sorted({r["rec"] for r in rows}):
        sel = [r for r in rows if r["rec"] == kind]
        scored = [r for r in sel if r.get("rec_prior_m") and r.get("rec_prior_z")]
        m, se = mean_se(per_game(scored, lambda r: math.log(r["rec_prior_m"]) - math.log(r["rec_prior_z"])))
        out[kind] = {"decisions": len(sel), "unscored": len(sel) - len(scored),
                     "log_prior_gain": [None if m is None else round(m, 5), None if se is None else round(se, 5)]}
    return out


def read_source(rows: List[Dict]) -> Dict:
    out: Dict = {"games": len({r["game"] for r in rows}), "decisions": len(rows)}
    if rows and "repro" in rows[0]:
        out["reproduction"] = round(sum(bool(r["repro"]) for r in rows) / len(rows), 4)
    for lo, hi in BUCKETS + (LATE,):
        sel = [r for r in rows if lo <= r["turn"] <= hi]
        name = "late (6+)" if (lo, hi) == LATE else bucket_name(lo, hi)
        out[name] = {**split(sel), **priors(sel)}
        if sel and sel[0]["src"] == "human":
            out[name]["label_gain"] = label_gain(sel)
    return out


def verdicts(readings: Dict) -> Dict:
    """The pre-registered tests that the rows alone decide."""
    out: Dict = {}
    own = readings.get("own")
    if own is not None:
        out["tool_check"] = (own.get("reproduction") or 0) >= 0.95
        late = own["late (6+)"]
        m, z, dis = late["only_m_ends"], late["only_z_ends"], late["disagreements"]
        out["T_rows"] = bool(dis) and m >= 2 * z and m >= dis / 3
        out["Q_rows"] = bool(dis) and late["both_act"] >= 2 * dis / 3
    if own is not None and "human" in readings:
        for name in ("p_end_diff", "only_m_ends_rate"):
            a, b = own["late (6+)"][name], readings["human"]["late (6+)"][name]
            if None not in (a[0], b[0], a[1], b[1]):
                d, se = a[0] - b[0], math.hypot(a[1], b[1])
                out[f"own_minus_human_{name}"] = [round(d, 5), round(se, 5),
                                                  "differs" if abs(d) >= 2 * se else "within 2 SE"]
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sources", nargs="+", help="NAME=ROWS.jsonl")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    readings = {}
    for spec in args.sources:
        name, _, path = spec.partition("=")
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
        readings[name] = read_source(rows)
    result = {"readings": readings, "verdicts": verdicts(readings)}
    text = json.dumps(result, indent=1)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
