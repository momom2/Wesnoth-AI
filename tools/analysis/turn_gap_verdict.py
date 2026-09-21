#!/usr/bin/env python3
"""The verdict of a turn-gap screen and its confirmation under
docs/turn_gap_ref_prereg_20260921.md: positions CONFIRMED out of the
screen's, against the KILL / SPARSE / RICH bars.

    python tools/analysis/turn_gap_verdict.py screen.json confirm.json

A position is confirmed when its confirmation gap (the replayed best
alternative against the replayed base, fresh playouts) is at least
the threshold and its lower 2-SE bound at least the margin, the
confirmation's own stop rule. Either file may be a `.partial.json`
(a cut run), which the verdict names.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.turn_gap import gap_stats  # noqa: E402

KILL_BELOW = 3          # confirmed positions: fewer than 5% of 60 kills
RICH_FROM = 6           # 10% of 60


def confirmation_gap(record: Dict) -> Tuple[Optional[float], float]:
    """(gap, SE) of a confirmation record's best alternative against
    its base on their fresh playouts; (None, inf) without an
    alternative (the screen's turn duplicated the base's on replay)."""
    alternatives = record.get("alternatives") or []
    if not alternatives:
        return None, math.inf
    best = max(alternatives, key=lambda a: a.get("mean", -math.inf))
    return gap_stats(best.get("outcomes", []), record["base"].get("outcomes", []))


def confirmed_positions(records: Sequence[Dict], *, threshold: float = 0.25,
                        margin: float = 0.10, z: float = 2.0) -> List[int]:
    out = []
    for r in records:
        gap, se = confirmation_gap(r)
        if gap is not None and gap >= threshold and gap - z * se >= margin:
            out.append(int(r["index"]))
    return sorted(out)


def verdict(screen: Dict, confirm: Optional[Dict], *, n_positions: int = 60,
            threshold: float = 0.25, margin: float = 0.10) -> Dict:
    screened = [int(r["index"]) for r in screen.get("positions", [])]
    nominal = sorted(int(r["index"]) for r in screen.get("positions", [])
                     if float(r.get("gap", 0.0)) >= threshold)
    confirm_records = (confirm or {}).get("positions", [])
    confirmed = confirmed_positions(confirm_records, threshold=threshold, margin=margin)
    n = len(confirmed)
    if len(screened) < n_positions:
        status = "PARTIAL"
    elif nominal and confirm is None:
        status = "SCREEN ONLY"
    elif n < KILL_BELOW:
        status = "KILL"
    elif n < RICH_FROM:
        status = "SPARSE"
    else:
        status = "RICH"
    denom = max(len(screened), 1)
    frac = n / denom
    se = math.sqrt(frac * (1 - frac) / denom) if denom else 0.0
    return {
        "status": status, "screened": len(screened), "n_positions": n_positions,
        "nominal": nominal, "confirmed": confirmed,
        "confirmed_fraction": frac, "confirmed_se": se,
        "confirmation_records": len(confirm_records),
        "screen_summary": {k: screen.get("summary", {}).get(k)
                           for k in ("frac_gap_ge_threshold", "null_frac_gap_ge_threshold",
                                     "mean_gap_split", "mean_gap_split_se",
                                     "playouts_capped_frac", "decisions_per_turn_base",
                                     "wall_secs", "dollars")},
    }


def render(v: Dict) -> str:
    lines = [
        f"turn-gap verdict: {v['status']}",
        f"  screened {v['screened']} of {v['n_positions']} positions; nominal hits "
        f"{len(v['nominal'])}: {v['nominal']}",
        f"  confirmed {len(v['confirmed'])} of {v['screened']} = "
        f"{v['confirmed_fraction']:.3f} +- {v['confirmed_se']:.3f}: {v['confirmed']}",
        f"  confirmation records {v['confirmation_records']}",
        f"  bars: KILL below {KILL_BELOW} confirmed, RICH from {RICH_FROM}",
    ]
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("screen", type=Path)
    ap.add_argument("confirm", type=Path, nargs="?", default=None)
    ap.add_argument("--n-positions", type=int, default=60)
    ap.add_argument("--gap-threshold", type=float, default=0.25)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--json", type=Path, default=None, help="also write the verdict here")
    args = ap.parse_args(argv)
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    confirm = (None if args.confirm is None or not args.confirm.exists()
               else json.loads(args.confirm.read_text(encoding="utf-8")))
    v = verdict(screen, confirm, n_positions=args.n_positions,
                threshold=args.gap_threshold, margin=args.margin)
    v["files"] = {"screen": str(args.screen), "confirm": None if confirm is None else str(args.confirm)}
    print(render(v))
    if args.json:
        args.json.write_text(json.dumps(v, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
