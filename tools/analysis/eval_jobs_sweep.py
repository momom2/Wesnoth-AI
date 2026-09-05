"""Eval throughput against the number of workers through the shared
inference server (box chain31, 2026-09-05): per row the wall time,
games per hour, the server's mean batch and time split, the argmax
W-D-L of seed against itself, and the slot-wise agreement of outcomes
between rows (the same slots replayed under different batch
compositions: the bf16 batched numerics are not bit-identical).

Usage:
    python tools/analysis/eval_jobs_sweep.py DIR   # DIR holds rows.txt and j<N>/
"""
import glob
import json
import math
import sys
from collections import Counter
from pathlib import Path


def load_row(root: Path, jobs: int):
    d = root / f"j{jobs}"
    games = {}
    for f in glob.glob(str(d / "game_*.json")):
        r = json.load(open(f, encoding="utf-8"))
        games[(r["seed"], r["side_a"])] = r
    stats = None
    for f in d.glob(".inference_server_*.json"):
        stats = json.load(open(f, encoding="utf-8"))
    return games, stats


def wdl(games):
    c = Counter(r["outcome_a"] for r in games.values())
    return c.get("win", 0), c.get("draw", 0), c.get("loss", 0), c.get("timeout", 0)


def main(argv):
    root = Path(argv[1])
    rows = [tuple(int(x) for x in line.split()[:3])
            for line in (root / "rows.txt").read_text().splitlines() if line.strip()]
    loaded = {j: load_row(root, j) for j, _, _ in rows}
    print("| jobs | games | wall s | games/h | s/game | mean batch | server infer % | "
          "gpu ms/batch | seed W-D-L-cap | turns median |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for j, g, secs in rows:
        games, st = loaded[j]
        w, d, lo, cap = wdl(games)
        turns = sorted(r["turns"] for r in games.values())
        mb = st["mean_batch"] if st else float("nan")
        infer = 100.0 * st["infer_s"] / st["wall_s"] if st else float("nan")
        gpu = st["gpu_ms"] / st["batches"] if st else float("nan")
        print(f"| {j} | {len(games)} | {secs} | {3600 * len(games) / secs:.0f} | "
              f"{secs / max(1, len(games)) * j:.1f} | {mb:.2f} | {infer:.0f} | {gpu:.1f} | "
              f"{w}-{d}-{lo}-{cap} | {turns[len(turns) // 2] if turns else 'n/a'} |")
    # Slot agreement across rows: the same (seed, side) under other batch mixes.
    keys = set.intersection(*(set(g.keys()) for g, _ in loaded.values()))
    base_j = rows[0][0]
    for j, _, _ in rows[1:]:
        same = sum(loaded[j][0][k]["outcome_a"] == loaded[base_j][0][k]["outcome_a"]
                   and loaded[j][0][k]["turns"] == loaded[base_j][0][k]["turns"]
                   for k in keys)
        print(f"slots shared with jobs={base_j}: {len(keys)}, identical outcome and "
              f"turn count at jobs={j}: {same}/{len(keys)}")
    allg = [r for g, _ in loaded.values() for r in g.values()]
    w = sum(r["outcome_a"] == "win" for r in allg)
    lo = sum(r["outcome_a"] == "loss" for r in allg)
    n = len(allg)
    p = (w + 0.5 * (n - w - lo)) / n
    se = math.sqrt(p * (1 - p) / n)
    print(f"pooled over rows (slots repeat across rows): seed score {p:.3f} +- {se:.3f} "
          f"over {n} games, decisive {w + lo}/{n}")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
