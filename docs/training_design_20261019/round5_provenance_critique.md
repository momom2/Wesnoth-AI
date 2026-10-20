# Critique of round5_provenance.md

**Verdict: repairable.** The records it cites support its measured numbers. Three faults break the rule that every outcome teaches something specific, and all three can be fixed without changing the frame. New scripts: `analysis/round5_provenance_critique_knobs.py` and `analysis/round5_provenance_critique_pairs.py`, each with its `.out`.

## 1. Provenance audit

| Number | Tag | Check | Finding |
|---|---|---|---|
| 0.747 setup + columns; Match C −0.035 ± 0.035; 110/46/2.4 events; 19.6 decisions per unit | measured | scripts read, records opened | hold |
| pairs keep ≥ 0.696 | measured, lower side | swap covariance estimated directly (`critique_pairs`) | **0.700 ± 0.009**; antisymmetry holds (−0.147 ± 0.009 against −0.151) |
| Σt² = 17.7 on 23 df | measured | output | holds; detects only biases of about 7 Elo or more |
| +0.0033 removed / +0.0002 added | measured | variants (`critique_knobs`) | **added is fragile**: −0.0003 with kv15's fresh confirmation, −0.0023 ± 0.0015 on stage-1 games only; dispersion 1.32 is not applied; subtracting 40 × observed z puts the response's noise into the regressor |
| Match A −0.097 ± 0.048 | derived | reduced form, no decision counts | −0.116 ± 0.061 (all games), −0.159 ± 0.076 (stage 1): the sign holds, the SE is understated |
| winner's curse +35.7 → +1.7 | measured | `decode_knobs_prereg_20261012.md` | numbers right, mechanism wrong: the 800 stage-3 games were played after the selection that mattered |
| critic curve r = 0.57 | measured + extrapolated | summaries | best reads came at 0.25–0.5 epochs, stopped by patience. This is a curve of the stopping rule, not of data |
| ranking flat, +0.020 ± 0.047 | measured | `critic_step1/readout.md` | selective: the same file reads +0.091 ± 0.052 and +0.126 ± 0.050 for the "pre" readings. It ranks turns played with their own dice, which the rulings forbid |
| prices | derived | redone from the seed's throughputs | hold. The forward price comes from `relset`'s memoryless serving rate; S0(f) re-measures it |
| 347.4; SEs 12.3 / 5.3 / 2.65; 2.84; 7.5 Elo; P1 SE 0.022; $7.4 | derived | redone | hold |
| break-even 0.61 | derived | redone | uses the corrected σ_T² against the raw σ_f². Consistently corrected it is 0.70; the decision is unchanged |
| "advantage 0.01" → 49,000 pairs → "coins only calibrate" | derived | input | **guessed**: 0.01 is round 4's s, which this memo lists as removed |
| S0(e) line σ_f² < 0.012 | derived | input | **guessed**: rests on Σp(1−p) = 31, "an illustration" |
| S2 $6, S5 $3, S4 refits, four critics $1.9, remedies | — | no derivation shown | about $15 of the $63 is untraced (6 + 3 + 2 × 2.25 + 1.9), and the $22 reserve rests on it |
| SE(κ̂) "known blind, cross terms exactly zero" | derived | redone | exact only for the uncorrected z (z² = 1). With the corrected G, G² depends on the coins and the cross terms do not vanish |

## 2. Fatal issues (each repairable)

**F1. No planted decision signal before the gate on which every gain path depends.** S1b's plant column is empty:
- P1 checks how a booked lottery is corrected.
- P-coin checks the coin regression with a known X.

Neither runs the critic's sibling contrast X = C(x∘a1) − C(x∘a0). A bug there (the wrong side's perspective, a misaligned memory, the wrong child) produces κ̂ ≈ 0. That reads as "decision transfer failed" and routes S1b to windows or closes the route. The project's history is this failure mode, and the mandate requires a plant first.
- *Fix:* move a P2-type plant into S1a. Book 0.05·even(played hex) at every decision, so the critic must learn it. At held-out coins, the slope of X on the planted contrast is predicted to be 1.

**F2. S1b's table leaves its likely outcome unassigned.** Take the case where κ̂'s interval contains both 0 and κ_min while ρ̂ beats the columns. Neither (i), (ii) nor (iii) applies, and S1c's $20 has no rule.
- SE(κ̂) and κ_min are both read blind, so the case can be decided before truth.
- *Fix:* if the games needed for SE ≤ κ_min/2.84 exceed the cap, declare the type unresolvable within budget before unblinding.

**F3. S2's known answers are not answers for the estimator they check.**
- The stop machinery scores an offset rule one side-turn at a time on π's paths, with π valued afterwards. That is the first-order surrogate of the performance-difference lemma.
- Matches A and C change about 20 to 29 decisions per side-game over whole games. Their gap from the surrogate is the second-order term, and the "2 combined SE" does not contain it.
- Consequences: a correct machinery can fail, and "the stop leaves round 1" then blames the wrong part. A machinery that values every offset at 0 passes Match C, which reads −0.035 ± 0.035.
- *Fix:* take the known answer from the stop coins' own regression, which estimates the same first-order quantity. Match A then serves no decision, so drop it (−$0.48).

## 3. Other defects

- **Calibration at the switches.** κ is fitted on random a1 drawn from the top four, but the switches sit at the tail of the score. Round 4 rejected out-of-top-four coins for this very reason ("calibration on random alternatives does not cover the argmax switches"), and the reason applies here too.
  - Γ_C maximises noisy X over three alternatives with no cap, while the switch rule caps at 10 switches and requires posterior ≥ 0.8. So κ_min is too lenient.
  - *Fix:* in S2, randomise π_1's own switches with logged p, and compute Γ_C under the switch rule. The regression then measures the forecast's first-order truth directly, with no linear extrapolation.
- **Adoption check mis-sized.** If the 800-game PURE match is compared with the 12,800-game corrected one, the combined SE is about 12.6 Elo, so only a 25-Elo disagreement fails, larger than any gain expected. If both are read on the same games, the right test is the mean correction term (SE about 1.7 Elo), not a root-sum-square of 4.1.
- **Yardstick.** S4 adopts π_{k+1} against π_k, while the seed makes strength claims against `own1`, never against the current self: state an `own1` reading per round. Adoption at 7.5-Elo resolution by corrected matches also goes below what the 800-game PURE yardstick resolves, and needs the user's assent.
- **Exploitation check.** Its 2-SE threshold is about 36 Elo, against gains forecast at 7.5 to 20 Elo. Every outcome is uninformative.
- **"Skip S1c if 5,000 does not beat 2,500 by 2 paired SE"** treats absence of significance as a kill. The record's critics stopped before one epoch, so the test's power is unknown.
- **The knob slope.** "−0.017 says going below −1.5 gains" is one polynomial fitted through a kink. The kinked reduced form reads flat to slightly against acting more: +0.001 ± 0.025 (all games), +0.057 ± 0.032 (stage 1) z per unit below −1.5.
- **I15's pair drop** is harmless at today's 0–0.5% cap rate, but a stop candidate can raise it, making the censoring informative.

## 4. Novelty

- **Nearest precedent the memo misses.** [Bottou et al. 2013](https://jmlr.org/papers/v14/bottou13a.html) puts randomised perturbations with logged propensities inside a deployed learning system to predict counterfactual outcomes.
- **Same machinery as κ.** [DR-WCLS](https://arxiv.org/abs/2306.16297) puts ML outcome models inside the WCLS estimator. Using it would also cut σ_T in κ̂.
- **No game use found.** A search for micro-randomised designs in game self-play found nothing, so that claim stands as far as I searched.

## 5. Strongest part

Section 1's critic-free yardstick:
- Columns plus setup, fitted leave-one-match-out, keep 0.747.
- Side-swapped pairs keep 0.700 ± 0.009, which I confirmed independently from the swap covariance. That is about 30% off every match, unbiased by construction, with no fitted model.

Also sound: performance-difference scaling replaces the λ/c0 taper, and guessed ranges are removed rather than re-dressed.
