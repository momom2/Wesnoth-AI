# Critique of round 10, provenance stance

**Verdict: repairable.** The exact identity is right and reproduces. The headline ("own interactions cannot produce an L null, so a null names the coined continuation") does not hold: on the stance's own rows, R's second-order interaction term reads −12.3 ± 4.5 Elo. The first-coin correction has the wrong coefficient. The repairs are offline and free.

New script: `analysis/round10_provenance_critique_checks.py` (*checks*, §n of its `.out`).

## 1. Provenance audit

The *link*, *pdis* and *derived* summaries rerun byte-identical; b00's rows, re-extracted from its games, match exactly.

| Number | Class | Check |
|---|---|---|
| F 20-fold +21.3 ± 3.1; F_LOMO +18.0 ± 3.1; turn 1 +12.7 ± 1.5 | M | Reproduced |
| Fold gap +3.3 ± 1.2 | M | Reproduced. The choice of +21.3 as the seal is post hoc (§2.4) |
| Ranks 0–1 exact, J − F −0.3 ± 1.7 | M | Reproduced. Rank 0 adds 0 by construction, so this is the rank-1 term alone, and it has no power (§2.2) |
| First-coin correction −3.5 ± 4.2 | M as labelled; in fact a wrong derivation | Should be −2.3 ± 2.8 (§2.1) |
| Within-rank contrast −0.032 ± 0.012, called "randomized" | M | Conditions on a post-treatment event (§2.3) |
| Turn-1 state shift "small, 0.6 prior deviations" | Argued from a count, not measured | Now measured: −1.2 ± 1.3 (*checks* §2) |
| Null needs 17.7 Elo of shortfall, 2.7 SE beyond own interactions | D | Fails: it rests on the first coin only (§2.2) |
| Turn-1 part "3.9 SE above the line" | D | Omits L's own noise: 3.06 SE (*checks* §4) |
| Learning curve flat past 17,000 games | M | Confirmed paired: 17k → 51k reads +0.1 ± 1.9; 43k → 51k reads −0.7 ± 1.1 (*checks* §3) |
| Wall model a 69 ± 30 s; o SE 0.024 | M; D | Reproduced. a rests almost entirely on b00, the only 2,000-game match; the equal-size protocol does not need it |
| Branch S B1 $3.56 | D | B1 is GPU training (4.5 M positions at 217 a second, `round8_synthesis_power.py`); pricing it at the generation host's rate assumes it runs there. Not "stale" |
| Dump 2.40 GB; L's lines 1.79, 7.2, 3.57; 2.08× at p 0.1 | D | Recomputed, correct |

No pure guess carries a decision; the seal at +21.3 and B1's host are choices presented as findings.

## 2. What would make it fail

**2.1 Coefficient.** The stance's exact identity, with the coin policy as base, is

J(π'_R) − J(π'_0) = 0.75 E_R[ΣΔ] + 0.25 E_0[ΣΔ].

Weight only the side's first R coin: a later label gets 3 if that coin was played and 1/3 if not. Its excess over F is then 0.5 (E1 − E0), not 0.75. *link* §1's 0.75 is the first step of (J(π'_R) − J(π_c))/0.75, which is R against the coin policy and not L's comparison. Corrected: −2.34 ± 2.80 (*checks* §1). The decision does not move.

**2.2 Own interactions are not bounded; they read large.** The stance reads own interactions only by the exact estimator (variance 4^k) and the first coin. The second-order term of the same expansion sums, over pairs of a side's R decisions, each later label times +2 if the earlier coin was played and −2/3 if not (the TayPO-2 form, §4); its variance grows with the pair count.

It reads **−12.3 ± 4.5 Elo** (z −2.7), and **F_LOMO plus it is +5.8 ± 5.3, against L's null line of 3.57** (*checks* §5). The stance's own within-rank contrasts (z −2.8) point the same way.

The expansion is not clearly convergent: at rank 2 the second-order term is −3.3 ± 2.0 and the exact one +4.3 (*pdis* §1), so third order there is about +7.6. The records neither bound R's own interactions nor show them small; by themselves they could produce a null.

The rank-1 "no interaction" reading has no power: if rank 1's value vanished after a played rank 0, it would read −(2/3) × 3.7 = −2.5 Elo, 1.3 SE from the observed −0.3 ± 1.7.

So the L table's null row ("own interactions would have to sit 2.7 SE beyond their reading … coined continuations do not price uncoined ones") is unsupported, and its listed responses (one coined side, p 0.1) target the wrong cause. Round 9's "recruit-specific second-order effect" reading was removed on a bound that does not exist.

**2.3 "Randomized" within-rank contrasts.** The first coin changes how many later R decisions occur (−0.138 ± 0.015 per side-game, *link* §1), so a within-rank comparison conditions on a post-treatment event. Only sums are causal (the per-side-game contrast, the IS and second-order estimators); the per-switch −0.032 ± 0.012 is descriptive.

**2.4 The seal.** The paired learning curve is flat (§1), so the +3.3 ± 1.2 fold gap is not a training-size effect. It is either within-match information that a new match lacks, or chance among many readings.

The stance keeps +21.3 because fold rules agree with the all-data rule on more cells. That measures rule similarity, not which held-out unit resembles L, a new match on fresh seeds: the leave-one-match-out condition. Choosing the higher estimate after seeing both is a forking path, and the rows depend on it: an L reading of 12 is "partial transfer" under +21.3 and "inside" under +18.0.

**2.5 Smaller.** The undumped match 1 removes a tenth of student 1's features, unstated beside H1's SE. "0.71 of F" divides by F_LOMO, not the sealed F. The continuation derivative (±19.1) carries nothing.

## 3. Repairs (offline, $0)

1. **Report the second-order term,** and drop the "2.7 SE" claim. The null row then names two live causes: R's own interactions and the continuation.
2. **Let L play R capped at each side's first two switches a game,** for the same $0.98. *pdis* §1's ranks 0–1 terms are exactly that rule's coin-world value: +14.9 ± 3.0, interaction −0.3 ± 1.7, (14.9 − 3.57)/√(3.0² + 1.79²) = 3.2 SE above the null line. A miss then isolates the coin-to-uncoined link, which is what L exists to test.
   Ranks 3+ read −8.7 ± 3.7 at second order: full R may not beat the capped rule in play.
3. **Seal at F_LOMO,** or seal the interval spanning both fold schemes, fixed before L.
4. **Replace the dose-response with exact reweighting** by 1[no prior deviation]/0.75^K at turn 1: the shift reads −1.2 ± 1.3 (*checks* §2), a measurement in place of a model.
5. **Correct the first-coin coefficient to 0.5,** and price B1 on the host it would run on.

## 4. Novelty

Splitting a first-order forecast into higher-order terms estimated from behaviour data is Taylor Expansion Policy Optimization ([Tang, Valko & Munos, ICML 2020](https://arxiv.org/abs/2003.06259); html version read): J(π) expanded around μ in powers of π/μ − 1, first order the TRPO surrogate, second order two ratio terms times Q^μ at the later step, "estimated only via samples under μ". The stance's own-interaction split is TayPO with μ the coin policy, uncited.

The narrow use as a pre-match attribution may be new; the tool is not. The Farias et al. abstract says "a large penalty in variance"; "possibly exponentially larger" is not in it (body not read).

## 5. Strongest part

The exact performance-difference identity with the coin policy as base, read by per-decision importance weights over a side's earlier R coins. It is correct, reproduces, and reports its 4^k reach honestly. With it the stance localized 0.71 of R's value to turn-1 recruits, 0.96 of them a side's first or second switch: a verified fact about `own1` that makes the capped L possible. Second: the equal-size overhead protocol, immune to the fixed cost whatever its size.
