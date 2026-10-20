"""Round 5 (invent): how often an argmax player's decision flips when its logits move by a
small amount, and whether keyed tie-breaking noise could make two branches stay coupled.

Record: training/metrics/critic_step1_20261009/prior_gaps.games.jsonl (100 recorded side-games
of `parity3` at its decode, offset -1.5 applied; per decision the log-prior gaps from the top
action to the next seven, tools/prior_gaps.py at ea8536f). Only decisions with two or more
legal actions are used. G is the top-two gap in nats.

Model (two candidates, stated): two runs that share every random draw take the same action at
a decision unless their logit difference moves the effective margin across 0. With keyed
Gumbel noise of scale T added to the logits (the same draw in both runs, keyed by the action's
identity), the effective margin is M = G + T*L, L standard logistic. For a perturbation of
size d in either direction the flip probability is f_M(0)*d, with
    f_M(0) = E_G[(1/T) f_L(G/T)],   f_L(u) = e^-u / (1 + e^-u)^2,
and for T -> 0 it tends to f_G(0+)/2 (exact argmax). The noise also makes the player deviate
from its argmax with probability P(L < -G/T) = 1 / (1 + e^(G/T)) per decision.

Output: f_G(0+) per nat per decision and per side-game; f_M(0) and deviations per side-game
for several T; the perturbation that reproduces the record's 1.76% replay disagreement.
numpy and the standard library only.
"""
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[3] / "training/metrics/critic_step1_20261009/prior_gaps.games.jsonl"
TEMPS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)


def load_gaps():
    gaps, side_games = [], 0
    for line in open(SRC):
        row = json.loads(line)
        side_games += 1
        for rd in row["readings"]:
            if rd["legal"] >= 2 and rd["gaps"]:
                gaps.append(rd["gaps"][0])
    return np.array(gaps), side_games


def logistic_pdf(u):
    e = np.exp(-np.abs(u))
    return e / (1.0 + e) ** 2


def main():
    G, n_sg = load_gaps()
    per_sg = len(G) / n_sg
    print(f"side-games {n_sg}, multi-legal decisions {len(G)} ({per_sg:.1f} per side-game)")
    for w in (0.02, 0.05, 0.1):
        f0 = np.mean(G < w) / w
        print(f"f_G(0+) from G < {w}: {f0:.3f} per nat per decision, {f0 * per_sg:.0f} per nat per side-game")
    f0 = np.mean(G < 0.05) / 0.05
    # The replay's 1.76% disagreement (prior_gaps.json: 26684 of 27171 agree; round5_provenance_numerics_ties.out)
    # read as d * f_G(0)/2 per decision under symmetric perturbations of size d.
    flip = 1 - 26684 / 27171
    print(f"replay disagreement {flip:.4f}; symmetric perturbation reproducing it: d = {2 * flip / f0:.4f} nats")
    print("\nT (nats)  f_M(0) per nat per decision  ratio to T->0  deviations per side-game")
    base = f0 / 2
    for T in TEMPS:
        if T == 0:
            fm, dev = base, 0.0
        else:
            fm = float(np.mean(logistic_pdf(G / T) / T))
            dev = float(np.sum(1.0 / (1.0 + np.exp(np.minimum(G / T, 700))))) / n_sg
        print(f"{T:<9} {fm:<29.3f} {fm / base:<14.2f} {dev:.1f}")
    # Expected flips per side-game for a symmetric perturbation of size d at every decision.
    print("\nexpected flips per side-game at T -> 0 for a perturbation of size d at every decision:")
    for d in (0.003, 0.01, 0.03, 0.1):
        print(f"  d = {d}: {base * d * per_sg:.1f}")


if __name__ == "__main__":
    sys.exit(main())
