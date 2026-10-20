"""Round 8, challenge stance: how much of the stop decision the current design's near-tie band reaches.

Input: training/metrics/critic_step1_20261009/prior_gaps.games.jsonl (parity3 at 64 slots, raw:t0+eo-1.5, argmax,
100 side-games of its match against parity2; per decision the gaps top1 - top_k of the joint log-prior after the
end_turn shift, and the type of the top action; read as in round5_simplify_band.py).

At a decision whose top action is end_turn, the runner-up is an act (end_turn is one entry), so the top-two gap is the
end-against-best-act margin. The current design coins the pair only when that gap is below w = 0.35.
Reported: per side-game counts, and the share of end-top multi-legal decisions below w for several w.
The census is parity3's; own1's is read in the current design's S0(a) or the S0 smoke.
numpy and the standard library only.
"""
import json
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[3] / "training/metrics/critic_step1_20261009/prior_gaps.games.jsonl"
WIDTHS = (0.35, 0.7, 1.0, 1.5, 2.0)


def main():
    games = [json.loads(line) for line in open(SRC)]
    n = len(games)
    end_gaps, all_gaps = [], []
    for g in games:
        for r in g["readings"]:
            if r["legal"] < 2 or not r["gaps"]:
                continue
            all_gaps.append(r["gaps"][0])
            if r["top_type"] == "end_turn":
                end_gaps.append(r["gaps"][0])
    e, a = np.array(end_gaps), np.array(all_gaps)
    print(f"side-games {n}; multi-legal decisions {len(a) / n:.1f} per side-game; end_turn on top at {len(e) / n:.1f}")
    print(f"end-top gap (end minus best act): median {np.median(e):.3f} nats")
    for w in WIDTHS:
        print(f"  w {w:<4}: end-top decisions in band {np.mean(e < w):.3f} ({np.sum(e < w) / n:.1f} per side-game); "
              f"all decisions in band {np.mean(a < w):.3f}")
    # bootstrap over side-games for the w = 0.35 share
    rng = np.random.default_rng(20261020)
    per = [np.array([r["gaps"][0] for r in g["readings"] if r["legal"] >= 2 and r["gaps"] and r["top_type"] == "end_turn"])
           for g in games]
    boot = []
    for _ in range(1000):
        pick = rng.integers(0, n, n)
        x = np.concatenate([per[i] for i in pick])
        boot.append(np.mean(x < 0.35))
    print(f"  share at w 0.35, bootstrap over side-games: 95% {np.percentile(boot, 2.5):.3f} to {np.percentile(boot, 97.5):.3f}")


if __name__ == "__main__":
    main()
