"""Round 6, provenance stance: the prices every stage's cost rests on, read from the records.

1. Generation (one network, 120 games in flight, `own1` against itself at the reference decode):
   training/metrics/generation_speed_r5_20261015/speed.json, every arm's games per hour,
   decisions per second and per game. The arms that matter:
     ref_p120x6      the reference generation path (full server outputs);
     priors_p120x6   priors-only, bit-identical to ref (BACKLOG.md, generation round 5);
     compile_bf16res_p120x6 / best_repeat  compiled loop + bf16 residual (rounding changes).
   base_1..3 are the eval configuration (20 jobs, one game per worker), the match driver's.
2. Matches (`run_elo_batch`, 20 jobs, persistent workers, shared inference): every
   "match <name> rc=0 ok <s> s" line of training/metrics/*/walls.txt, 800 decisive games each
   (the driver's target; the match.walls files beside some of them confirm 800 decisive).
   The look-ahead matches (material, exposure) run an extra search and are listed apart.
3. Box setup: seconds from each box's "entry:" line (stages.txt) to the end of its first
   stage that is not tests, reference, inputs, gpu check or fetch.

Prices at $0.42 an hour (seed section 1). numpy and the standard library only.
"""
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

METRICS = Path(__file__).resolve().parents[3] / "training/metrics"
PRICE_H = 0.42
SETUP_STAGES = {"tests", "reference", "inputs", "gpu", "fetch", "cuda-tests", "base-stage", "init", "tuned",
                "summary", "fp8-cuda-tests"}


def generation():
    arms = json.loads((METRICS / "generation_speed_r5_20261015/speed.json").read_text())["arms"]
    print("1. generation, generation_speed_r5_20261015/speed.json")
    for name in ("base_1", "base_2", "base_3", "ref_p120x6", "priors_p120x6", "compile_p120x6",
                 "compile_bf16res_p120x6", "best_repeat"):
        a = arms[name]
        gph = a["games_per_hour"]
        print(f"  {name:24s} games/h {gph:8.0f}  decisions/s {a['decisions_per_s']:7.0f}  "
              f"decisions/game {a['decisions_per_game']:5.1f}  $ per game {PRICE_H / gph:.2e}")
    return {k: PRICE_H / arms[k]["games_per_hour"] for k in ("ref_p120x6", "priors_p120x6", "base_1")}


def matches():
    rows = []
    for walls in sorted(METRICS.glob("*/walls.txt")):
        for line in walls.read_text().splitlines():
            m = re.search(r" match (\S+) rc=0 ok (\d+) s", line)
            if m:
                rows.append((walls.parent.name, m.group(1), int(m.group(2))))
    print("\n2. matches, 800 decisive games each, walls.txt")
    plain, lookahead = [], []
    for box, name, secs in rows:
        kind = "look-ahead" if name in ("material", "exposure") else "plain"
        (lookahead if kind == "look-ahead" else plain).append(secs)
        print(f"  {box:34s} {name:24s} {secs:5d} s  $ per decisive game {PRICE_H * secs / 3600 / 800:.2e}  ({kind})")
    s = np.array(plain, float)
    q = np.percentile(s, [0, 50, 100])
    print(f"  plain matches: {len(s)}, seconds min/median/max {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}; "
          f"$ per decisive game {PRICE_H * q[0] / 3600 / 800:.2e}/{PRICE_H * q[1] / 3600 / 800:.2e}/"
          f"{PRICE_H * q[2] / 3600 / 800:.2e}; the design's 17 min = {PRICE_H * 1020 / 3600 / 800:.2e}")
    return PRICE_H * q[1] / 3600 / 800, PRICE_H * q[2] / 3600 / 800


def setup():
    print("\n3. box setup, entry to the end of the first workload stage")
    out = []
    for stages in sorted(METRICS.glob("*/stages.txt")):
        walls = stages.parent / "walls.txt"
        if not walls.exists():
            continue
        entry = re.search(r"^(\S+Z) entry:", stages.read_text(), re.M)
        if not entry:
            continue
        t0 = datetime.fromisoformat(entry.group(1).replace("Z", "+00:00"))
        for line in walls.read_text().splitlines():
            m = re.match(r"(\S+Z) (\S+)(?: \S+)? rc=\d+ \w+ (\d+) s", line)
            if not m or m.group(2) in SETUP_STAGES:
                continue
            start = datetime.fromisoformat(m.group(1).replace("Z", "+00:00")).timestamp() - int(m.group(3))
            out.append(start - t0.timestamp())
            print(f"  {stages.parent.name:36s} first workload '{m.group(2)}' starts {start - t0.timestamp():6.0f} s after entry")
            break
    s = np.array(out)
    print(f"  boxes {len(s)}: median {np.median(s):.0f} s, max {s.max():.0f} s; "
          f"$ at the median {PRICE_H * np.median(s) / 3600:.3f} (the boot before 'entry' is not in the records)")


def main():
    g = generation()
    match_median, match_max = matches()
    setup()
    print("\n4. what a corrected match costs")
    for n in (12_800, 25_600, 51_200):
        print(f"  {n:6d} games: eval driver median ${n * match_median:.2f} (max ${n * match_max:.2f}); "
              f"generation pool, ref path ${n * g['ref_p120x6']:.2f}")
    print(f"  generation pool against eval driver, per game: {match_median / g['ref_p120x6']:.1f}x cheaper")
    print(f"  coin batch of 100,000 games at the ref path: ${1e5 * g['ref_p120x6']:.2f}")


if __name__ == "__main__":
    main()
