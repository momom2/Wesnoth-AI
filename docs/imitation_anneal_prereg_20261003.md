# Pre-registration: imitation training under the anneal rule (2026-10-03)

Written before any box for it is rented. User decision 2026-10-03: keep
training the reference's recipe on imitation while it pays, with the
learning rate decided from the loss curve by the rule below, designed with
the user the same day.

## Question

Does more imitation training of `parity2`'s recipe, held at the peak
learning rate for as long as one more epoch is predicted to lower the
holdout loss by more than 0.03 and then lowered to zero, give a stronger
player than `parity2`?

## The rule

`tools/lr_law.py`, run by `tools/sequence_train.py --anneal-rule 0.03`:

- **The law** (Tissue et al., "Scaling Law with Learning Rate Annealing",
  2024): L = L0 + A·S1^(−α) − C·S2. S1, the sum of the learning rate over
  the optimizer steps, measures the progress made; S2 measures how much of
  the noise a high rate keeps in the weights has faded since the rate came
  down, which is what lowering the rate gains.
- **Hold** the peak rate, 2.8e-4. After every holdout probe (every 500,000
  positions), fit the law to every probe of the lineage: the two parity
  passes, replayed from their settings
  (`training/metrics/imitation_anneal_20261003/lineage.json`), and the hold
  passes so far. The loss read is the holdout policy cross-entropy at 64
  slots, the reference's setting.
- **Stop holding** at the first probe where one more epoch at the peak
  would lower the fitted loss (S1's term) by 0.03 or less.
- **Stop holding when the law fails:** when the two latest probes of a
  pass both sit above a fit made without them by more than twice that
  fit's error (exit 7), progress is slower than the law predicts. The
  holding ends there too; the lowering and the match follow, and the run's
  report flags it.
- **Spending cap:** at most two hold passes, each a full epoch on a new
  order, each starting where the previous one ended. The cap, not the
  threshold, bounds this run: the law, fitted mostly on the earlier passes,
  puts the threshold 6.46 epochs past the start (from the law in
  `first_decision.json`). Within the cap, the holding ends by the cap, or
  earlier by the failed-law stop. The first starts from
  pass 2's checkpoint where its lowering began
  (`tier-b/parity_memory_pass2_20261002/arm.stable.pt`).
- **The lowering:** a straight line from 2.8e-4 to 0 over 2,017,864
  positions (half an epoch, as in pass 2), from where the holding stopped.
  Its final probe gives the candidate's holdout loss.

Under the law, any early lowering slows the progress term for the rest of
the run, while the noise term only matters for the final weights. So the
rate is held, then lowered once at the end, over a stretch long enough for
the noise to fade.

Rejected: halving the rate whenever the loss stalls ("reduce on plateau").
Comparing two probes at the peak rate mostly measures noise, since
consecutive probes there differ by up to 0.1 at 64 slots (pass 1's last
two). And under the law, a rate
lowered early costs progress that a single final lowering keeps.

## What the law says before the run

Records in `training/metrics/imitation_anneal_20261003/`. The earlier
passes are replayed at 512 positions a step, as their probe rows read.

- **The fit** to the 18 probes of the parity passes, at 64 slots: L0 1.074,
  A 2.485, α 0.165, C 1.050, root-mean-square error 0.028, within the
  probes' noise (`first_decision.json`).
- **Only a lowering determines C.**
  - Fitted on the first 13 probes, taken before any lowering, the law
    leaves C at 0 and reports it undetermined (`backtest_first13.json`).
  - Fitted with the first two probes of pass 2's lowering, it puts that
    lowering's end 0.026 below the measured loss at 64 slots
    (`backtest_first15.json`) and 0.09 above it at 0 slots
    (`backtest_first15_k0.json`).
  - The seed of this run holds the whole lowering.
- **First decision**, at the start checkpoint: hold. One more epoch at the
  peak would lower the fitted loss by 0.166.
- **The law's predictions**, holding then lowering over half an epoch
  (`predictions.json`):

| epochs held from the start checkpoint | gain of the next epoch | loss at 64 slots after the lowering |
|---|---|---|
| 0 (pass 2 as run; measured 2.852) | 0.166 | 2.845 |
| 1 | 0.102 | 2.701 |
| 2 | 0.072 | 2.608 |

## Predictions, before the run

- **Holding:** the cap ends it (0.8); the failed-law stop ends it earlier
  (0.2); the threshold does not, within the cap. The law's gains are well
  above 0.03; I expect the true ones to be smaller, but still above it.
- **Holdout loss at 64 slots after the lowering:** 2.70 (2.62 to 2.80). The
  law says 2.61. I expect it to be optimistic: it extrapolates S1 to 2.3
  times the value it has seen, on data seen for the third and fourth time.
- **The match:** p 0.58 (0.50 to 0.66).

## The verdict

- **Match:** the candidate at 64 slots against `parity2` at 64 slots, both
  at the reference decode (`raw:t0+eo-1.5`). PURE, sides alternated, the
  Ladder maps with factions drawn uniformly and assigned openly, 800
  decisive games, seed base 90000 (disjoint from every earlier match).
- **Pass:** p ≥ 0.535, about +24 Elo, the bar of the earlier retrains. A
  pass makes the candidate the proposed reference, pending the user's
  ruling.
- **Holdout loss** is a reading, not a verdict.

## Amended before any box (2026-10-03, after the commit's review)

- A failed-law stop (exit 7) first ended the run with no lowering and no
  match. A re-entry would have stopped again, stranding the held weights.
  It now ends the holding and the run goes on.
- The replay of the earlier passes took their positions as spread evenly
  over their steps. It now uses their 512 a step: the fit moved by under
  0.001, and no decision changed.
- A rule's decision is saved with its probe, so a crash between the two
  cannot lose it. A short match is played once more; a failed one is
  reported as failed. The lowering retries once after a crash, like a hold
  pass.

## Cost

- **Setup:** about 25 minutes (code, wheel, tests, corpus, pre-encoding).
- **Each hold pass:** about 5.6 hours (pass 2's 336 minutes).
- **The lowering:** about 2.8 hours.
- **The match:** about 17 minutes.
- **Total:** two hold passes come to about 14.6 box-hours, about $6.10 at
  pass 2's rate ($0.42 an hour). `BOX_MAX_H` 20 caps it at about $8.40.
  The balance is $10.57 (2026-10-03).
- **Box:** `scripts/imitation_anneal_box.sh`, on the class of the
  parity-memory passes (RTX 4090, 32 cores, 64 GB).
