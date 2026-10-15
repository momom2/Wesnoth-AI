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
| B8 | B with fp8 linear layers (`--fp8-trunk`) |
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

## Measured (2026-10-14)

Box 55962170 (RTX 4090, $0.724/h), stage
`tier-b/staging/stage_20261014_trainer4b_acceptance.tar.gz` (commit b2cef96 of
`feature/trainer-speed-4`), arms A, B, A2, B64 and BHALF. Every B arm ran with
an activation budget of 8 GiB and PyTorch's expandable segments, after the
first attempt's B ran out of memory at the 19.18 GiB budget with the compiled
block (kept activations are bit-exact, so the budget changes speed, not the
numbers). Records in `training/metrics/trainer_acceptance_20261014/`
(`acceptance.txt`, `acceptance.json`), 312 holdout games, 624 game-sides.

**B8 was removed by the gradient gate:** fp8's gradients read cosines of
0.96, 0.83 and 0.67 against A's on the three production windows (the encoder
lowest; the gate asks 0.99). fp8 stays off.

The primary reading (`ce_all` at 64 slots), each arm against A, with the
noise |A2 - A| = 0.00055:

| arm | d | bound (noise + 2 SE) | primary | positions a second |
|---|---|---|---|---|
| B | +0.00068 +- 0.00024 | 0.00102 | Pass | 794 (A: 279) |
| B64 | +0.00132 +- 0.00083 | 0.00221 | Pass | 765 |
| BHALF | +0.00018 +- 0.00101 | 0.00258 | Pass | 1,023 |

Reported, not judged:
- B is within its bound on every reading.
- B64 is outside on the belief loss at 0 and 64 slots, and on `ce_all` with
  the memory reset at every decision (+0.0036 +- 0.0007).
- BHALF is outside on the belief loss at 0 slots and with the memory reset.

B64 and BHALF pass the primary through wide standard errors of their own
(0.0008-0.0010, against B's 0.00024).

Predictions against the outcome:
- B Pass (0.85): Pass.
- B8 Pass (0.6): removed at the gate.
- B64 Pass (0.7): Pass on the primary.
- BHALF predicted Fail: Pass on the primary.

**The lead's decision:**
- **B becomes the trainer's configuration for real runs:** packed trunk, kept
  activations, columnar records, batched window, compiled block.
- **B64 and BHALF are not adopted.** B64 runs no faster than B at this budget
  (765 against 794). BHALF saves 1.29x for readings that are worse beyond
  noise on the memory and the belief. Both stay available behind their
  flags.
