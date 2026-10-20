"""Round 6, provenance stance: how much of a tempo signal the own-trajectory recipe transfers.

The recipe (docs/memory_own_trajectories_prereg_20261011.md) fine-tuned parity3 at 64 slots
toward parity3's own memoryless reading (0 slots, soft raw targets) on its own trajectories.
The memoryless reading acts more per side-turn than the 64-slot player, so the recipe carried a
tempo signal of known size. Its transfer is read here from whole game records, in the decision
unit of round5_provenance_stop_acts.py (recruits + attacks + moves not followed by an attack
from their last hex; end_turn not counted), per side-turn, decisive games:

  before  parity3 at 64 slots against parity3 at 0 slots   parity3_baselines/games_slots64_vs_slots0
  after   the tuned network at 64 slots against the same  memory_own_trajectories/games_m1_own64
  target  parity3 at 0 slots, read as side B of those two matches, and the tuned network at
          0 slots (games_m2_own0) as a second reading of the memoryless tempo.

closure = (after - before) / (target - before), SE by the game bootstrap.

Also printed, for P-distill's planted tempo: own1 at -2.5 against own1 at -1.5
(rollout_regrade_20261012/games_own1_eo25_vs_own1).
numpy and the standard library only.
"""
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sa", HERE / "round5_provenance_stop_acts.py")
sa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sa)
RNG = np.random.default_rng(20261020)


def per_turn(games, side):
    acts = np.array([g[f"acts_{side}"] for g in games], float)
    turns = np.array([g[f"turns_{side}"] for g in games], float)
    return acts, turns


def rate(acts, turns):
    return acts.sum() / turns.sum()


def boot(fn, n_games, reps=2000):
    vals = []
    for _ in range(reps):
        vals.append(fn(RNG.integers(0, n_games, n_games)))
    return float(np.std(vals))


def main():
    before_g = sa.read_match("parity3_baselines_20261008/games_slots64_vs_slots0.tar.gz", "parity3")
    after_g = sa.read_match("memory_own_trajectories_20261011/games_m1_own64.tar.gz", "own64")
    own0_g = sa.read_match("memory_own_trajectories_20261011/games_m2_own0.tar.gz", "own0")
    teach_g = sa.read_match("rollout_regrade_20261012/games_own1_eo25_vs_own1.tar.gz", "own1_eo25")
    Ab, Tb = per_turn(before_g, "a")
    Bb, TBb = per_turn(before_g, "b")
    Aa, Ta = per_turn(after_g, "a")
    Ba, TBa = per_turn(after_g, "b")
    A0, T0 = per_turn(own0_g, "a")
    print("decisions per side-turn (decisive games):")
    print(f"  before: parity3@64 {rate(Ab, Tb):.3f} +- {boot(lambda i: rate(Ab[i], Tb[i]), len(Ab)):.3f}; "
          f"its opponent parity3@0 {rate(Bb, TBb):.3f}  ({len(Ab)} games)")
    print(f"  after:  tuned@64  {rate(Aa, Ta):.3f} +- {boot(lambda i: rate(Aa[i], Ta[i]), len(Aa)):.3f}; "
          f"its opponent parity3@0 {rate(Ba, TBa):.3f}  ({len(Aa)} games)")
    print(f"  tuned@0 against parity3@0: {rate(A0, T0):.3f}")
    for name, target_fn in (("target = parity3@0 as side B of the after match", lambda i, j: rate(Ba[j], TBa[j])),
                            ("target = parity3@0 as side B of the before match", lambda i, j: rate(Bb[i], TBb[i]))):
        def closure(i, j):
            b = rate(Ab[i], Tb[i])
            return (rate(Aa[j], Ta[j]) - b) / (target_fn(i, j) - b)
        full_i, full_j = np.arange(len(Ab)), np.arange(len(Aa))
        est = closure(full_i, full_j)
        vals = [closure(RNG.integers(0, len(Ab), len(Ab)), RNG.integers(0, len(Aa), len(Aa))) for _ in range(2000)]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        print(f"  closure, {name}: {est:.2f} (95% bootstrap {lo:.2f} to {hi:.2f})")
    At, Tt = per_turn(teach_g, "a")
    Bt, TBt = per_turn(teach_g, "b")
    print(f"P-distill's tempo: own1 at -2.5 {rate(At, Tt):.3f} +- {boot(lambda i: rate(At[i], Tt[i]), len(At)):.3f}, "
          f"own1 at -1.5 {rate(Bt, TBt):.3f} +- {boot(lambda i: rate(Bt[i], TBt[i]), len(Bt)):.3f} ({len(At)} games)")


if __name__ == "__main__":
    main()
