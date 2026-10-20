# Round 7 critique of the simplify stance

**Verdict: repairable.** The numbers reproduce, and no guessed number carries a decision. But two removals undo protections that round 6 had installed against specific failures:
- Dropping the exact duel I2 leaves nothing that checks the real result's orientation to the mover.
- The replacement check, I-V, fires on correct code and cannot resolve biases at the scale of the signal.

Both repairs cost nothing. The new script is `analysis/round7_simplify_critique_checks.py`, with its `.out` beside it (cited as §n).

## 1. Provenance audit

I reran `round7_simplify_power.py`; its output is byte-identical.

**Measured numbers, checked in their records:**
- prices $2.51e-5, $1.17e-4 and $0.018 (`round6_provenance_prices.out`);
- pairs keep 0.847 (`round6_simplify_pairs_raw.out`; a cell-level swap covariance, which fits its use for sizing);
- 0.688, 0.839 and 0.696 (`round6_provenance_cells.out`). The 0.839 is match-centred, with β fitted leave-one-match-out. Batch 1's b = 0 has no intercept: side 1's mean in `own1` self-play is +0.040 ± 0.025, which adds 0.0016 to E[R²], a negligible amount (§1);
- caps 0 of 1,600, walk 0.321 s;
- attack slope −0.2247 ± 0.0459 (`round6_provenance_critique_attack_stage1.out` B);
- knob slopes, tempo closures, numerics 1.76%;
- $2.1 for distillation (the prereg's whole box run, two matches included, so an upper bound).

**Derived numbers, redone:**
- matches: 37,800 games give SE 1.79 Elo for $0.95 each, $1.16 more in all; the S0 window is ±33.2;
- label variance: σ² 3.67, and 4.48 in batch 1;
- frozen coefficients: 7.78 effective batches against 7.93;
- budget: $71.61; three rounds fit at up to $3.07e-5 a game, two at up to $4.78e-5; a null round costs $8.66, $22.16 or $15.41;
- miss line 9.9; false adoption 0.023;
- plant SE √(3/n). (T′ − T)·g summed over a side is p(1 − p)·W_A(W_A − W_B), with variance 3σ⁴ for Gaussian W.

All are correct.

**Guessed numbers that carry a decision: none.** The ĉ band 0.82–1.30 is labelled indicative and gates nothing beyond ĉ > 0.

**Faults:**
- **S4's closure price uses the wrong sample.** It takes the all-games slope (−0.0774). S0 takes unselected stage-1 games, because successive halving selected the later stages. On stage 1, the price at c 0.82 is −9.9 ± 4.7 Elo, not −7.3 ± 3.8 (the stance's own §6). One sample should serve both.
- **Item 6 misdescribes the current design.** Current §3.7 already reads (T′ − T)/δ, so item 6 removes only δ's sizing.

## 2. What would make it fail

**F1. The label path no longer checks the result's orientation.**
- **Why the plant cannot see it.** The plant adds z′ − z = Σ s_j^rec (D_j − P_j) g_j to the result field. So T′ − T = s_i ω_i (z′ − z) contains no real result, and the plant passes whatever the field's orientation.
- **Simulated (§3).** With the field flipped against the real outcome, the plant reads 1.006. Real labels of a negative-effect type read +0.066 ± 0.004 instead of −0.063.
- **Where the bug would come from.** `rows.jsonl` orients z and S to label A, and `side_a` is 1 in 400 and 2 in 400 games of each `own1` self-play match. The coin log's mover is a Wesnoth side.
  - A label code that reads z as side 1's flips half the games: every real label averages to zero while the plant reads 1, and the round reads null.
  - A full flip makes the student learn the anti-advantage. Its forecast, from the same labels, predicts a gain, and the match then reads a "transfer miss".
  - Either way, a bug is reported as a finding.
- **Round 6 had closed this hole.** Its simplify critique named this failure, and round 6 repaired it with I2, the coin on each seat. The stance credits "sign (plant)" with I2's coverage, but none of the checks it keeps sees the error:
  - I1 and I9 never touch z;
  - S0 is read by `elo_collect`, not by the label code.
- **Repair at no cost:** read S0's 6,400 games through the label code, one coin a game (the hooked side, P = 1/2; sides alternate independently of the setup draw).
  - Predicted −0.225 z, window [−0.329, −0.120]. A flipped field reads +0.225; a half-flipped one reads 0 (§3).
  - Or restore I2.

**F2. I-V cannot stand in for the exact duel.**
- **Its SE (§2).** Per type, SE = √(ω²·E[(b + βᵀS)²]/n_type): 0.00075 in batch 1 and 0.00105 later at a 10% type share, up to 0.0023 at 2%.
- **It fires on correct code.** Applied "every batch" at 2 SE to 10–16 types over 8 batches, the rule fires with probability 0.976–0.997. No action on failure is specified.
- **At a 5% family-wise threshold (3.4–3.8 SE) it is too coarse:**
  - one batch resolves 0.0037 at a 10% share, above the whole spread range the round seeks (sd 0.0011–0.0036);
  - pooled over 8 batches it resolves 0.0013, still at the floors.
- **Such a bias matters.** A per-type centring bias enters the unpenalised c_uv, which drives switching directly.
- **The stance misdescribes what I-V detects.** "I-V's statistic weighted by g is this slope" describes the plant. The unweighted mean has no power against a coin-correlated term whose sign varies within a type.
- **Repair:** read I-V once per round at a family-wise threshold, as a report, and restore I2's exact centring test.

**F3. S4 keeps the unvalidated arm and drops the validated one.**
- **The arms.** The raw target played at −1.5 is the recipe behind `own1`'s +88.6. Round 7 replaces it with the baked target played at offset 0.
- **A likely halt.** At ĉ 0.82, the baked arm costs −7.3 to −9.9 Elo against a non-inferiority reading at SE 3.07. It then likely fails, and the plan halts at S3-null on the user's choice. The stance prices the closure but not this halt.
- **The plant is only a sign test.** It passes at ĉ > 0 by 2 SE, so ĉ 0.3 passes. This gap is inherited from round 6.
- **Repair at no cost:** ĉ is measured on held-out positions before the match, so play the baked network at the residual offset −1.5(1 − ĉ), −0.27 at c 0.82 (§4).
  - The offset is derived, and it vanishes as ĉ → 1.
  - Ruling (d) should say whether a residual offset counts as "gone".

**Lesser points:**
- **γV.** It is dropped at unknown cost while V_i stays logged.
  - Keeping it costs one frozen coefficient from batch 2, and it is exact by the same proof.
  - Σ_i x_i T_i = Σ_g R_g A_g − γ Σ_i ω_i V_i x_i is still a per-game sum, so dropping it saves little.
  - Give it DPI's derived line: it enters iff batch 1's kept ratio with V is ≤ 0.602.
- **The baked target.** It lowers end_turn by a fixed 1.5, so a later reset of an offset-0 base would apply it twice. Lower it by π_k's own offset.

## 3. Novelty

**The forward-chained coefficients are standard.** In adaptive-experiment AIPW, the outcome model m̂_t is built on the history H^{t−1} ("the sample mean of arm rewards up to period t−1 as the plug-in estimator", [Hadad et al. 2021](https://arxiv.org/abs/1911.02768), §2.1 and experiments, read in the ar5iv text). That is closer than cross-fitting, and the stance should cite it.

**The other claims are details, not contributions.** The T′ − T reading is algebra, and the baked offset as its own plant is a design detail.

## 4. Strongest part

Two pieces each remove an unknown and a code path at a small derived price:
- **Frozen coefficients.** Labels are exact by construction and computed once, with no folds; I4 is automatic. The price is 0.15 of a batch.
- **Independent matches.** Pair scheduling, I8 and the pair-correlation unknown go, for $1.16.
