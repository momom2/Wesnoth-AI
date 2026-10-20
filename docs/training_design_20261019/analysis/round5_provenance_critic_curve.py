"""Round 5 (provenance stance): the record's critic learning curve in games, and what more games of
plain result regression would buy.

Reads training/metrics/critic_step1_20261009/{T25,T50,T100}.summary.json and .holdout.jsonl:
true-state critics initialised from parity3, trained on 25%, 50% and 100% of 7,200 engine-match
games (at most 24 positions a game), each read every ~41,000 positions on the same 368 holdout
games; the value loss is the C51 cross-entropy against the game's result (nats; ln 2 = 0.693 for
a coin). The best read is selected on that same holdout, so each level is optimistic by the
selection; the three share the holdout, so their differences are paired.

Model, stated: each doubling of the games improves the best holdout loss by a constant ratio r of
the previous doubling's improvement (a power law in the games). With two measured doublings, r is
their ratio; the script extrapolates to 50,000 games and to infinity. The step-1 readout's ranking
test (training/metrics/critic_step1_20261009/readout.md) is quoted beside it: T100 against T25 in
corrected correlation +0.020 +- 0.047, selection gain +0.001 +- 0.016.
numpy and the standard library only.
"""
import json
import math
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parents[3] / "training/metrics/critic_step1_20261009"


def main():
    pts = []
    for name in ("T25", "T50", "T100"):
        s = json.load(open(DIR / f"{name}.summary.json"))
        reads = [json.loads(line) for line in open(DIR / f"{name}.holdout.jsonl")]
        games = s["provenance"]["train_games"]
        best = min(r["holdout_value"] for r in reads)
        epochs_at_best = s["state"]["best_positions"] / s["train_positions"]
        pts.append((name, games, best, epochs_at_best))
        print(f"{name}: {games} training games, best holdout CE {best:.4f} nats "
              f"(ln2 - CE = {math.log(2) - best:.4f}), best at {epochs_at_best:.2f} epochs, ended {s['ended_by']}")
    (_, g1, l1, _), (_, g2, l2, _), (_, g3, l3, _) = pts
    d1 = (l1 - l2) / math.log2(g2 / g1)
    d2 = (l2 - l3) / math.log2(g3 / g2)
    r = d2 / d1
    print(f"improvement per doubling: {d1:.4f} then {d2:.4f}; ratio r = {r:.2f}")
    for target in (50_000, 200_000):
        k = math.log2(target / g3)
        whole, frac = int(k), k - int(k)
        gain = sum(d2 * r ** (i + 1) for i in range(whole)) + d2 * r ** (whole + 1) * frac
        print(f"extrapolated improvement from {g3} to {target} games: {gain:.4f} nats (best CE about {l3 - gain:.4f})")
    print(f"extrapolated improvement to infinitely many games: {d2 * r / (1 - r):.4f} nats")


if __name__ == "__main__":
    sys.exit(main())
