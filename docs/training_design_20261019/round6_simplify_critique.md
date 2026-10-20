# Critique of round 6, simplify stance

**Verdict: repairable.** The label, its proof and most of the cuts hold up. Three numbers that carry a stop or a branch are wrong or rest on an unknown (F1–F3). Each can be fixed without adding a mechanism. Script: `analysis/round6_simplify_critique_checks.py` (`.out` beside it).

## 1. Provenance audit

| Number | Class | Check |
|---|---|---|
| Pairs alone 0.847 ± 0.009; walk 0.321 s a game; floors, forecast SE, budget $77.3 | M; M; D | Reran `round6_simplify_pairs_raw.py`, `_walk_cost.py` and `_power.py`: identical outputs. The swap key (map, 3 − seat, f_B, f_A) is the right pairing. `secs` is the time per game, run as 31 jobs on 31 usable cores (box.txt). |
| 0.700; 0.882 = mean(0.885, 0.879); 0.732 = mean(0.730, 0.734) | M | Read in the round-5 outputs: correct. |
| SE 2.83 / 3.07; σ² 3.91; 2.4–6.6%; 17%; $3.36; $2.6 in matches; band 76.1 (91.36/269.17 × 224.6) | D | Redone: correct. |
| Capped 0 of 1,600 | M | Correct. |
| Early-stop P | D | Reproduced where printed. The range it leaves out is where the decision sits (F2). |
| Sign rule ≥ calibrated; E keeps 0.80–0.87 | D | Code read. At μ = 0, E's ratio is φ(0.674)/φ(0) = 0.797. In-model it is right. But the model is correctly specified, so ridge is its Bayes rule, and it cannot show what β̂ guarded against (misspecification). The removal stands only as a redundancy argument: λ chosen by held-out loss already calibrates the scale. |
| **P-coin ridge reading 1 ± 0.10** | claimed D | **Wrong** (F1). |
| **δ = 0.01** | C | Whether it can be detected depends on Var(u·x), which is unmeasured (F1). |
| **P-distill known answer 9.63** | M used as D | It assumes full closure, which the stance's own table lists as U (F3). |
| "Shared β: cost none" | claim | Unverified (§3). |
| Value head 0.22 against HP margin 0.40 | M, misapplied | This is a turn-ranking correlation, not a corrector's kept variance. Cost alone justifies the deferral it is used to support. |

## 2. Decision-carrying faults

**F1. The plant fails on correct code, and passes on some wrong code.**
- **Ridge reading.** "1 ± 0.10" is the OLS SE under an assumed Var(u·x) = 1 (`round5_simplify_critique_checks.py`). A ridge at the progressive λ shrinks a signal that lies along a single direction. In the stance's own model, ŵ·u/0.01 on batch 1 reads 0.33 (0.21–0.43) at Var(u·x) = 1 and 0.03 at trace 1; OLS reads 1.00 ± 0.05. "Outside its band: stop" therefore fires on a working path.
- **Slope reading.** At Var(u·x) ≤ 0.01 its 2-SE band contains 0, so the plant cannot fail. Var(u·x) is unknown. Same-actor pairs have no actor-block difference at all.
- **Sign.** The plant is booked with the logged s_i, the sign the label uses, and s_i² = 1. A wrong mover sign, or z read from the wrong side, therefore passes both readings while it inverts the real labels: side B's, or all of them. Pooled, the ridge then reads near 0, and the bug looks like a null round, the project's known failure mode. The dropped parity plant was built from the record without s and caught this. I-φ checks features, not s. I2 could catch it only if its duel exercises both movers, which the spec does not say.
- **Repair.**
  - Predict the ridge reading as δ·uᵀ(G + λ̂I)⁻¹Gu, its SE from batch 1's statistics, or read the plant at λ = 0.
  - Set δ after the smoke measures Var(u·x), so the band excludes 0 by at least 4 SE.
  - Book the plant with the mover's side taken from the record by separate code, and read it per side.

**F2. The early stop removes most of the stated power when μ < 0.**
- At d_eff 776 and μ = −0.0018, at the stated floor sd 0.0039 (final gain 8.3 Elo), the early stop fires with P = 0.54; at sd 0.0035, with P = 0.95. At d_eff 200, sd 0.0030 (7.5 Elo), P = 0.35.
- So power at the floor is about 0.46 × 0.8 ≈ 0.37, and a null bounds the spread near sd 0.0045 (P(stop) 0.11), not at 0.0039. S3-null is told more than the data shows.
- The cause: students 1–3 are judged against the final student's line.
- Repair: stop on the model's batch-8 prediction from the batch 1–4 readings, or state floors that include the stop.

**F3. P-distill's known answer requires full closure.**
- The one measured run of the recipe closed 58% of its agreement gap (`docs/memory_own_trajectories_prereg_20261011.md`). At that rate the count lands near 8.58 + 0.58 × 1.05 ≈ 9.2, outside 9.63 ± 2 SE. This is indicative, since that run had another target.
- "Keep π_k, widen" then fires on a working distillation.
- Repair: report the closed fraction, require only its direction, and let the distilled network's match decide the reset.

## 3. Other problems

- **The γV merge is not free.**
  - V_i already reflects the luck before i. A single shared β on S = S_{<i} + S_{>i} cannot weight past luck 0 and future luck 1 at once. Past luck is subtracted twice unless γ shrinks, which leaves V close to useless by construction.
  - Unbiased either way, and no decision rests on it, but "None" in §1 is wrong.
  - Fix: emit S_{<i} at band decisions and use V_i − κᵀS_{<i}. About 110 fights a game at 0.06–0.11 ms each costs under 1% of generation online.
- **A forecast miss has more than one cause.**
  - The labels price advantages under π̃, where both sides take about 19 coin deviations per side-game. The match plays π_1 against coin-free `own1`.
  - That excursion gap is first order in p and present for a single switch. Halving the switches (E) does not touch it.
  - Round 5's research target (excursion effects) is not taken up, so a miss followed by E failing does not name the stage that failed.
- **E's forecast** applies the final student's threshold t to students 4–7, whose f scales differ, so it forecasts another switch set. Use each student's own half-quantile.
- **S2 has no action** for "not adopted, match inside the forecast band"; S3 presumes adoption.
- **S4's non-inferiority test** "estimate ≥ 0 and lower bound > −5.7" reduces to "estimate ≥ 0". A true deficit of 2.8 Elo passes with P 0.16. This is inherited.
- **Pairs need new driver code.**
  - `tools/run_elo_batch.py` draws each game's seed, and so its map and factions, from the game index (`slot_for`).
  - A capped slot's replacement takes a new seed and breaks its pair: across the records, 666 of 19,466 games capped.
  - No rule says how a broken pair enters the SE.

## 4. Novelty

- **"One corrected result per game serves all labels"** is not new:
  - It is REINFORCE's structure: one return multiplies every score term, and D − P is the Bernoulli score ([Williams 1992](https://link.springer.com/article/10.1007/BF00992696)).
  - It is also the distal causal excursion effect of micro-randomized trials: hundreds of randomizations per participant, one distal outcome, cross-fitted, robust to outcome-model misspecification ([Qian 2025](https://arxiv.org/abs/2502.13500), abstract read).
- **λ from progressive loss on sufficient statistics** is prequential validation ([Dawid 1984](https://doi.org/10.2307/2981683), not re-read) applied to the ridge normal equations.

## 5. Strongest part

The strongest part is one paired PURE match serving as both the adoption test and the claim:
- 0.847 ± 0.009 is reproduced from the records.
- The SE rises only from 2.57 to 2.83 Elo.
- The cost is $2.6 over eight matches, derived and checked.
- The correction-term code and its fault mode leave the adoption path.

Also right: capped coin games enter the labels at z = 0 rather than being selected out, and S0 goes, since none of its outputs fed a decision before batch 1.
