"""Round 5, critique of the challenge stance: how many decisions the look-ahead's clipped tilt
flips on pure evaluator noise, when sigma is set "blind" to the noise of the candidate difference.

Input (read-only): training/metrics/critic_step1_20261009/prior_gaps.games.jsonl
(`parity3` at 64 slots, offset -1.5, argmax, 100 recorded side-games; per decision the gaps
top1 - top_k, k = 2..8, in joint log-prior after the end_turn shift).

Model: the challenge's operator (archived tools/lookahead_player.py `tilted_choice`):
    b* = argmax_i  log pi_i + clip((Q_i - V) / sigma, -c, c),  V = sum_i w_i Q_i,
w_i the prior renormalised over the top 8. A zero-information evaluator: Q_i = e_i, iid
N(0, s^2) per candidate (an equicorrelated common part cancels against V). The noise of a
candidate difference D(b) = Q(b) - Q(a0) then has SD sqrt(2) s. The challenge's blind rule sets
sigma^2 = 1/2 Var[D_1 - D_2] = Var(noise of D) = 2 s^2, i.e. s / sigma = 1/sqrt(2).

Reports, per side-game: flips caused by noise alone, by the ratio s / sigma, at c = 1, and the
mean top-two gap of the flipped decisions. numpy and the standard library only.
Run: python round5_challenge_critique_noise_flips.py  (writes the .out beside it)
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "training/metrics/critic_step1_20261009/prior_gaps.games.jsonl"
OUT = Path(__file__).with_suffix(".out")
RNG = np.random.default_rng(20261020)
DRAWS = 20


def load_decisions():
    rows, n_side = [], 0
    for line in SRC.open():
        game = json.loads(line)
        n_side += 1
        for r in game["readings"]:
            if r["legal"] >= 2 and len(r["gaps"]) == 7:
                rows.append([0.0] + list(r["gaps"]))
    return np.array(rows), n_side


def noise_flips(neg_logit_gap, ratio, c=1.0):
    """Mean flips over DRAWS noise draws; sigma = 1, noise SD per candidate = ratio."""
    logp = -neg_logit_gap
    w = np.exp(logp)
    w /= w.sum(1, keepdims=True)
    flips, gaps = 0.0, []
    for _ in range(DRAWS):
        q = RNG.normal(0.0, ratio, logp.shape)
        v = (w * q).sum(1, keepdims=True)
        score = logp + np.clip(q - v, -c, c)
        flipped = score.argmax(1) != 0
        flips += flipped.sum()
        gaps.append(neg_logit_gap[flipped, 1])
    return flips / DRAWS, float(np.concatenate(gaps).mean()) if flips else float("nan")


def main():
    lines = []
    gaps, n_side = load_decisions()
    lines.append(f"side-games {n_side}; decisions with 8 candidates {len(gaps)} "
                 f"({len(gaps) / n_side:.1f} per side-game)")
    lines.append("pure-noise flips per side-game at c = 1 (sigma = 1, per-candidate noise SD = s):")
    for ratio, label in ((0.25, ""), (0.5, ""), (1 / np.sqrt(2), "  <- blind sigma (SD of D's noise = sigma)"),
                         (1.0, ""), (2.0, "")):
        f, g = noise_flips(gaps, ratio)
        lines.append(f"  s/sigma {ratio:.3f}: {f / n_side:6.1f} flips per side-game, "
                     f"mean top-two gap of flipped decisions {g:.3f} nats{label}")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
