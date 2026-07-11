"""Fit Elo from a directory of elo_eval_game.py result files, under
TWO draw conventions from the same games:

  PURE (primary): draws are draws, weight 0.5 for each side. This is
    THE metric: the contract on policy performance is the game's own
    win/draw/loss, and material advantage must not factor into
    evaluation (user decision 2026-07-11, reversing the 2026-07-04
    material-primary lock -- material valuation is a training crutch,
    not part of what performance means).
  MATERIAL-SIGN (diagnostic only): a drawn/timed-out game whose final
    material margin from A exceeds +/-EPS counts as a win for the
    side ahead. More separating while ladder games are draw-heavy,
    useful for watching progress -- but never the headline number.

Usage:
    python tools/elo_collect.py GAMES_DIR [--anchor dummy]
        [--save-json PATH] [--eps 0.02]
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.elo_ladder import PairRecord, fit_elo


def load_games(games_dir: Path) -> List[dict]:
    games = []
    for p in sorted(games_dir.glob("game_*.json")):
        try:
            games.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            print(f"skipping unreadable {p.name}", file=sys.stderr)
    return games


def build_pairs(
    games: List[dict], eps: float,
) -> Tuple[List[str], Dict[Tuple[int, int], PairRecord],
           Dict[Tuple[int, int], PairRecord]]:
    labels = sorted({g["label_a"] for g in games}
                    | {g["label_b"] for g in games})
    idx = {l: i for i, l in enumerate(labels)}
    pure: Dict[Tuple[int, int], PairRecord] = {}
    mat:  Dict[Tuple[int, int], PairRecord] = {}
    for g in games:
        a, b = idx[g["label_a"]], idx[g["label_b"]]
        i, j = min(a, b), max(a, b)
        a_is_i = (a == i)
        for d in (pure, mat):
            d.setdefault((i, j), PairRecord())
        out = g["outcome_a"]
        if out == "win":
            win_i = a_is_i
        elif out == "loss":
            win_i = not a_is_i
        else:                                   # draw / timeout
            pure[(i, j)].draws += 1
            m = float(g.get("margin_a", 0.0))
            if abs(m) <= eps:
                mat[(i, j)].draws += 1
            else:
                ahead_is_a = m > 0
                win_i = ahead_is_a == a_is_i
                if win_i:
                    mat[(i, j)].wins_i += 1
                else:
                    mat[(i, j)].wins_j += 1
            continue
        for d in (pure, mat):
            if win_i:
                d[(i, j)].wins_i += 1
            else:
                d[(i, j)].wins_j += 1
    return labels, pure, mat


def _print_table(title: str, labels, elo, se, pairs) -> None:
    print(f"\n=== {title} ===")
    order = sorted(range(len(labels)), key=lambda k: -elo[k])
    for k in order:
        print(f"  {labels[k]:<10} {elo[k]:>8.1f} ± {se[k]:.0f}")
    for (i, j), rec in sorted(pairs.items()):
        print(f"    {labels[i]} vs {labels[j]}: "
              f"{rec.wins_i}-{rec.draws}-{rec.wins_j} (W-D-L)")


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("games_dir", type=Path)
    ap.add_argument("--anchor", default="dummy")
    ap.add_argument("--eps", type=float, default=0.02,
                    help="material dead zone: |margin| <= eps stays "
                         "a draw under the MATERIAL convention.")
    ap.add_argument("--save-json", type=Path, default=None)
    args = ap.parse_args(argv[1:])

    games = load_games(args.games_dir)
    if not games:
        print("no game files found"); return 2
    labels, pure, mat = build_pairs(games, args.eps)
    anchor_idx = (labels.index(args.anchor)
                  if args.anchor in labels else 0)
    n = len(labels)
    results = {}
    for title, pairs in (("PURE (draws=0.5, primary)", pure),
                         ("MATERIAL-SIGN (diagnostic)", mat)):
        elo, se = fit_elo(n, pairs, anchor_idx, anchor_elo=0.0,
                          prior_games=1.0, draw_weight=0.5)
        _print_table(title, labels, elo, se, pairs)
        results[title] = {l: {"elo": float(e), "se": float(s)}
                          for l, e, s in zip(labels, elo, se)}
    print(f"\ngames: {len(games)} | anchor: {labels[anchor_idx]} = 0")
    if args.save_json:
        args.save_json.write_text(
            json.dumps({"n_games": len(games), "tables": results},
                       indent=2), encoding="utf-8")
        print(f"written: {args.save_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
