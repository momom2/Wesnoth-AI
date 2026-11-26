"""Two matches of the simple loop set seed by seed, from their games'
summaries: do the same seeds give the same setup, and how often the same
outcome? Written when the matches at rounds 608 and 624 read the same count
(574 of 800), to tell a coincidence from a copy. Standard library only.

    python -I -X utf8 match_pairs.py --out pairs.json <games dir A> <games dir B>
"""
import argparse
import json
import re
from pathlib import Path

SETUP = ("scenario_id", "side_a", "faction_a", "faction_b")
PLAYER = ("checkpoint_sha256_a", "checkpoint_sha256_b", "code_version", "observation_epoch")


def load(directory: Path) -> dict:
    games = {}
    for path in sorted(directory.glob("game_*.json")):
        if path.name.endswith(".game.json"):
            continue
        games[int(re.search(r"_(\d+)\.json$", path.name).group(1))] = json.loads(
            path.read_text(encoding="utf-8"))
    return games


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    a, b = load(args.a), load(args.b)
    seeds = sorted(set(a) & set(b))
    n = len(seeds)
    win_a = sum(a[s]["outcome_a"] == "win" for s in seeds) / n
    win_b = sum(b[s]["outcome_a"] == "win" for s in seeds) / n
    both = sum(a[s]["outcome_a"] == "win" and b[s]["outcome_a"] == "win" for s in seeds) / n
    same = sum(a[s]["outcome_a"] == b[s]["outcome_a"] for s in seeds)
    # The correlation of the two outcomes on one seed; the standard error of the
    # two matches' difference is sqrt(1 - correlation) times the independent one
    # when the two matches have the same variance.
    spread = (win_a * (1 - win_a) * win_b * (1 - win_b)) ** 0.5
    out = {
        "a": str(args.a), "b": str(args.b), "games": [len(a), len(b)], "shared_seeds": n,
        "wins": [round(win_a * n), round(win_b * n)],
        "same_setup": {key: sum(a[s].get(key) == b[s].get(key) for s in seeds) for key in SETUP},
        "same_player": {key: sum(a[s].get(key) == b[s].get(key) for s in seeds) for key in PLAYER},
        "same_turn_count": sum(a[s]["turns"] == b[s]["turns"] for s in seeds),
        "same_outcome": same,
        "same_outcome_if_independent": round(n * (win_a * win_b + (1 - win_a) * (1 - win_b)), 1),
        "outcome_correlation": round((both - win_a * win_b) / spread, 3) if spread else None,
    }
    print(json.dumps(out, indent=1))
    if args.out:
        args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
