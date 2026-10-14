# Pre-registration: the trainer's faster options, accepted by their holdout losses (2026-10-14)

Written by the project lead before any box for it. The user asked for data
before deciding on the trainer options that change the arithmetic or the
method. These passes are measurements: no checkpoint they produce is used.

## Question

Does a network trained with each faster option come out as good as one trained
with today's default, over the same positions from the same start, within the
run-to-run noise of the default itself?

## Design

Code: `feature/trainer-speed-4` (tip 095f61a), `scripts/trainer_acceptance_box.sh`,
readout `tools/trainer_acceptance_readout.py`. Every arm starts from `own1`
under round 2's training preset, seed 20261012, and walks the same 500,000
positions: the corpus and round 2's own sequences (HF
`tier-b/trainer_speed_3_20261015b/own_sequences_r2.tar.gz`). It ends with the
holdout probe at 0 and 64 memory slots, read per game-side.

| arm | configuration |
|---|---|
| A | today's default (padded trunk; bit-exact with main) |
| A2 | A again: the run-to-run noise of GPU nondeterminism |
| B | the packed trunk, kept activations, columnar records, the batched window, the compiled post-attention block (no bf16 residual: it failed its CUDA test) |
| B8 | B with fp8 linear layers in place of the fused ReLU |
| B64 | B at 64 streams (half of them own), the learning rate times the square root of 2, so half the optimizer steps over the same positions |
| BHALF | B training every second window; the others run forward only, carrying the memory |

B's and B8's gradients are checked against A's on production windows before any
pass (cosine at least 0.99 per parameter group, loss within 0.5%). A failed
check removes the arm.

## Readings

Per arm X and per reading m: d = m(X) - m(A), its standard error paired
by holdout game-side, and the noise n = |m(A2) - m(A)|.
- **Primary:** m is the policy cross-entropy over all decisions (`ce_all`) at 64 slots.
- **Reported:** `ce_all` at 0 slots and the belief loss at 64 slots.

- **Pass:** d at most n + 2 SE (no worse than the default within its own noise).
- **Fail:** otherwise. The option costs quality at this scale; it stays off.

500,000 positions is a quarter of a production pass, and the holdout loss is a
proxy, not a strength claim. A pass means the option may be used in a run
whose checkpoint is then judged by a match, as every candidate is. A fail on a
short pass is still a reason to leave the option off.

## Predictions (the lead's)

| arm | prediction | probability of a pass |
|---|---|---|
| B | Pass | 0.85 |
| B8 | Pass | 0.6 |
| B64 | Pass | 0.7 |
| BHALF | Fail at equal positions walked | 0.3 |

BHALF trains on half the gradient signal for about 60% of B's cost per
position walked.

## Cost

About 2.5 hours on one 4090 box at $0.42-0.70 an hour: $1.1-1.8. BOX_MAX_H 4.
The speed bench of the same branch (fp8 timing, the streams sweep) runs
beside it on a second box, about 1 hour.
