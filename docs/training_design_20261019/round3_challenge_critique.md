# Critique of round 3 challenge: play each candidate out, then read

**Verdict: repairable.** The label is unbiased for what it defines, Q_k. But the central claim, that a weak critic costs forks and not correctness, is false by the design's own section 2.1, and the calibration of that bias does not transfer to the decisions the student picks. Both can be fixed in the spec.

## Strongest part

Each fork label compares one decision with another, both sides continuing at argmax with fresh dice under the core's exact laws and re-deciding after every roll. This meets the per-decision ruling directly and removes the three faults the seed finds in the +0.049 result (own dice, linear luck, temperature 0.5). E[Y_k | x] = Q_k holds for any f. The labels come from interventions, so a memory-reading student cannot relearn the causal confusion that cost 105 Elo. P0f and the −2.5 check test the fork and performance-difference code against known match results.

## Issues, most severe first

**1. The critic's error is a bias, but the design calls it noise.** The stance says the critic's errors "become label noise, which more forks average away". S1 says "C sets noise, not direction", and section 3 says a weak critic "costs forks, not correctness". Section 2.1 derives the opposite: D_k is biased by E[e(x_τ^{a0})] − E[e(x_τ^b)], and more forks at a fixed k leave that bias unchanged. The current design's look-ahead is the k = 0 case of the same estimator; this one reads the critic later, it does not remove it. The record shows the reader matters as much as the horizon: ranking turns, the static HP margin scored 0.40, the rollout read three side-turns later 0.42, `obs8`'s static head 0.22 (`docs/turn_value_prereg_20260925.md`). A critic near material, the record's level, pulls k = 3 labels toward material, as the material look-ahead did (−8.7, mostly rewarding recruits). Withdraw the claim and give S2 a kill line on direction (item 3).

**2. m̂ is measured where C did not select the decisions, then applied where the student did.** The student switches where D̂_k is largest, so its switches over-sample decisions where C overrates x_τ^b relative to x_τ^{a0}. Ĝ_plug(k) reads the same C at the same horizon and inherits that error; S2's sets were not chosen by C, so m̂ does not contain it. The forecast m̂·Ĝ_plug is biased upward by an amount that grows with λ, and section 2.4 would blame interacting switches or myopia. Repairs:
- Fit the labels' critic and Ĝ_plug's critic on disjoint games. This removes selection on sampling error, though not on blind spots both share.
- Across the three λ matches, read the gap between m̂·Ĝ_plug(λ) and the match. A gap that grows with λ means the student is exploiting C.

**3. m_act is too noisy to choose k\*, and it does not measure actions alone.**
- **Noise.** The stated combined SE of 25 Elo, against 0.75 × 88.6 ≈ 66 Elo, puts m_act's standard error near 0.25 to 0.3, against about 0.07 for m_stop by the same arithmetic. On the action side, noise decides both "smallest k with m ≥ 0.75 on both" and the kill "m(6) < 0.5 on both".
- **Composition.** The `own1`–`parity3` disagreements are largely about stopping. The memory changed about a quarter of decisions, 19% of its disagreements on the player's own trajectories were turn endings, and a stronger offset recovered 65 of its 105 Elo (`docs/memory_in_play_parity3_prereg_20261009.md`).
- **Repair.** Split the fork sums by decision type (the lemma is additive) and take the non-stop part's D_∞ from end reads on a subsample. Otherwise, name m_act for what it measures.

**4. Some checks pass with wrong code, and some code paths have no planted signal.**
- **P-luck** tests the game-level mean of Σℓ. Since z is signed to a side and sides alternate, a law error common to both sides (one terrain's chance to hit) shifts both sides' terms alike and cancels. Test the mean per event class (attacker side × terrain) and a χ² of realised outcomes against p_j.
- **The invariant** Σ_o p(o)(f(o) − E_p f) = 0 is an algebraic identity and cannot fail. The needed test checks p_j against the core's enumeration, as the current design's I1 does.
- **P1** never runs the fight-law path, and its miss means a poor f (more variance), yet it stops everything.
- **The horizon has no plant.** A payoff of known size, paid L side-turns after the decision and hidden from the critic, makes m(k) a known step at k = L. The stop head has no plant either.
- **P2** compares π_1 with its own fork forecast, so a bug shared by labels and forecast passes. Compare with the optimum the planted rate implies.
- **S0(b)** needs batch-invariant kernels if (a) measures agreement per decision: at 98%, a 450-decision record survives with probability 0.98^450 ≈ 10⁻⁴. Replay in the record's batch slots, or report the first divergence.

**5. The round-1 forecast can hardly miss, and its priors have no derivation.** The match's 5 Elo, Ĝ_plug's 9 Elo scaled by m̂, and m̂'s error times about 67 Elo give a combined SE of 15 to 20 Elo, so the 2-SE window is ±30 to ±40 around +50. The forecast comes from π_{k+1}'s own games after the student exists, so only the prior of +25 to +95 is fixed before the stage. That prior and the 0.4–0.8 decay have no derivation, so "+110 against +60" compares two guesses. Derive the gain from the χ² through the student's reliability, or mark the priors unsourced.

**6. The stop variants carry no improvement guarantee.** Parallel rollout's guarantee, no worse than the best base policy at each state ([Chang, Givan & Chong 2004](https://scholarworks.sogang.ac.kr/item/9f189e84-3bda-4a83-af8c-e43b21120ce1)), rests on re-evaluating the same base set at the next state. Re-evaluated one act later, "act 5 more, then π" becomes a six-act commitment from the first state, so {end, 1, 2, 5} is not closed under that shift. The family "continue while L < c, for this turn", over a grid of c, is closed under it, contains π (c = 1.5) and every offset, and makes the rule a parallel rollout over offsets whose variants are per-decision rules.

**7. The "5 times the χ²" comparison sets one design's variance against the other's bias.** The current design's student regresses a dice-free look-ahead; here the χ² reach the student with C's error inside their mean. The figures also assume ρ_f = 0.7 (0.85 on record for linear columns) and s_3 ≈ 0.8 s_∞.

**8. Numbers.** M_A = 25 cannot occur: offset 0 ends the turn at its first disagreement with −1.5, so fewer than 20 per side-game (the SE is conservative). Points per round: 300,000 in section 2.5, 400,000 in S3. The $8 round omits the C/f refit every second round; the other costs follow from the seed's throughputs. Tesauro & Galperin's 10-hidden-unit net used 11-step rollouts, not 7 ([Table 3](https://arxiv.org/html/2501.05407v1)).

## Novelty

- **Claim 1.** GNU Backgammon ships truncated rollouts with luck cancelled by its own net ([manual](https://www.gnu.org/software/gnubg/manual/html_node/Rollouts-in-GNU-Backgammon.html); [Montgomery](https://bkgm.com/articles/GOL/Feb00/var.htm)), and its mailing list describes a pipeline that rolls out positions where 0-ply and 2-ply disagree and trains the net on them ([2020](https://lists.nongnu.org/archive/html/bug-gnubg/2020-10/msg00040.html)). Only the amortised exact-law afterstate head is new, close to [Stochastic MuZero](https://openreview.net/forum?id=X6D9bAHhBQ1).
- **Claim 3** is n-step TD on luck-corrected targets ([De Asis & Sutton](https://arxiv.org/abs/1807.01830); [Mesnard et al.](https://proceedings.mlr.press/v139/mesnard21a.html)). Reading it as m(1) rising is new.
- **Claim 4** is the performance-difference lemma estimated by rollouts from resets ([Kakade & Langford](https://mlanthology.org/icml/2002/kakade2002icml-approximately/)).
- **Claim 5** is Baisero & Amato, and [Suphx](https://arxiv.org/abs/2003.13590) also trains a fogged player from a full-information oracle.
- **Claim 2**, a truncated label's bias calibrated against matches at disagreement sets with known answers, I found nowhere. Item 2 limits what it delivers.
