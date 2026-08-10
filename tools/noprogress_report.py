"""Offline would-fire analysis for the --no-progress-turns stalemate rule.

Prices candidate K values against per-game `noprogress` readouts
(wesnoth_sim.noprogress_summary, collected on every game since
2026-07-21) WITHOUT enforcing anything. For each K it reports:

  fired    d/n games with max_quiet >= K (the rule would have ended
           the game early)
  false    d/n games where a RESUMED streak >= K -- the rule would
           have called a draw while the game still had fighting left.
           This is the error rate that matters: a false fire mislabels
           a decisive-in-progress game as a stalemate draw.
  true     d/n games where the TERMINAL quiet streak >= K (the game
           was over in every sense but the clock)
  saved    total turns the true fires would have skipped
           (sum of tail_quiet - K), as a fraction of all turns played

Decision guidance (F2, user ruling 2026-08-10): enforcement is a
SEPARATE decision -- it changes game endings, so the Elo ladder needs
re-anchoring when it flips on. This report only prices the clock.

Usage:
    python tools/noprogress_report.py training/logs [MORE_DIRS...] \
        [--k 4 6 8 12 16] [--min-turns 10]

Directories are searched recursively for games.jsonl. `--min-turns`
drops smoke/short games whose noprogress readout carries no signal
(default 10).

NOTE 2026-08-10: the 2026-07-28..31 campaign leg's games.jsonl -- the
only real-game corpus post-dating the collector -- was never synced
off the box and is gone. This tool ships ahead of the data: the
handoff leg escrows games.jsonl (scripts/vast_onstart.sh), so the
analysis can run a few hours into that leg.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def iter_games(roots: Iterable[Path], min_turns: int):
    """Yield per-game dicts that carry a noprogress readout and at
    least `min_turns` turns."""
    for root in roots:
        for f in sorted(Path(root).rglob("games.jsonl")):
            with f.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        g = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    np_ = g.get("noprogress")
                    if np_ is None or g.get("turns", 0) < min_turns:
                        continue
                    yield g


def analyze(games: List[Dict], ks: List[int]) -> Dict:
    """Pure aggregation -- unit-testable without files."""
    n = len(games)
    total_turns = sum(g.get("turns", 0) for g in games)
    out = {"n_games": n, "total_turns": total_turns, "per_k": {}}
    for k in ks:
        fired = false_f = true_f = saved = 0
        for g in games:
            np_ = g["noprogress"]
            max_q = np_.get("max_quiet", 0)
            tail = np_.get("tail_quiet", 0)
            resumed = np_.get("resumed_streaks", []) or []
            if max_q >= k:
                fired += 1
            if any(s >= k for s in resumed):
                false_f += 1
            if tail >= k:
                true_f += 1
                saved += tail - k
        out["per_k"][k] = {
            "fired": fired, "false": false_f, "true": true_f,
            "turns_saved": saved,
        }
    return out


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path,
                    help="Directories searched recursively for games.jsonl.")
    ap.add_argument("--k", type=int, nargs="+", default=[4, 6, 8, 12, 16],
                    help="Candidate --no-progress-turns values to price.")
    ap.add_argument("--min-turns", type=int, default=10,
                    help="Ignore games shorter than this (smoke runs).")
    args = ap.parse_args(argv[1:])

    games = list(iter_games(args.roots, args.min_turns))
    rep = analyze(games, sorted(set(args.k)))
    n, tt = rep["n_games"], rep["total_turns"]
    print(f"games with noprogress readout (>= {args.min_turns} turns): {n}")
    print(f"total turns played: {tt}")
    if not n:
        print("no analyzable games -- nothing to price. (The campaign "
              "legs' games.jsonl must be escrowed off the box; see the "
              "module docstring.)")
        return 1
    print(f"{'K':>4} {'fired':>10} {'false-fire':>11} {'true-fire':>10} "
          f"{'turns saved':>12}")
    for k, r in rep["per_k"].items():
        frac = r["turns_saved"] / tt if tt else 0.0
        print(f"{k:>4} {r['fired']:>7}/{n} {r['false']:>8}/{n} "
              f"{r['true']:>7}/{n} {r['turns_saved']:>6} ({frac:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
