# Round 1 synthesis

## Verdict

The design changed in substance this round:
- **Core mechanism.** The seed's fork labels and grader are replaced. A self-play game becomes one factorial experiment: nature's ~160 chance events, plus designed per-decision coins, fitted jointly by a privileged critic trained to zero the luck-corrected residual.
- **A fatal flaw fixed.** Distilling per-world tilts fused strategies under fog. The fogged student now regresses on the privileged advantage by squared loss.
- **A new instrument.** The midpoint audit measures each candidate's gain per decision type, unbiased and exact to second order.
- **Stage order.** Instruments, then the critic, then calibration, then the player, then rounds, then distillation. Every stage opens with its own plant.

The seed's frame survives: policy iteration on single decisions, with the performance-difference lemma predicting each round.

## What changed, and why

**From harden (factorial coins; critique: repairable).**

Kept:
- Coins in argmax self-play as the label source: 10 to 15 times the information per dollar of fork pairs, and the critique checked the orthogonality.

Repaired:
- The stop window is now centred on the decode's real boundary (L = +1.5). On raw L, the old window could only sample ending earlier.
- The correlation kill lines were replaced by a nested model (mean cost plus log-prior gaps), and the stop rule must beat the best constant offset.
- The planted check now goes through the game record and the real decode: P0 compares the coin estimate of the offset-0 switch with Match A, an answer measured by an independent code path.

Demoted:
- The frozen ridge head becomes fallback F, because a linear head on sparse deviation outcomes is starved of data (critique items 2 and 8).
- The coins now calibrate and audit. Whether outcome regression falls under the user's killed "outcome policy gradient" is flagged for the user's ruling; the critique showed it is episodic natural actor-critic.

**From challenge (zero-floor critic; critique: repairable).**

Kept: the identity, and the ratio ρ_C as a readout with an absolute zero.

Repaired:
- **Accounting.** The critique's "one label per game" holds for the semi-gradient refresh rule. The design trains instead on the joint objective, Σ_t R_t² differentiated through every evaluation of C, unrealised outcomes included. In the linear case this objective is a noiseless regression per game, so information is bounded by misspecification. The semi-gradient fit only initialises.
- **The prediction of ρ_C** is now the power-law extrapolation of the first two curve points, not an unexplained 0.55.
- **Off-policy candidates.** The coins are chance events of known law, so the critic is trained on action outcomes at the decisions where candidates are compared. Its look-ahead is calibrated per decision against coin labels before any player is built.
- **Fusion.** Squared regression replaces the tilt distillation. The step-size argument and the 0.6 transfer fraction are dropped.
- **Planted checks.** The result-coin plant tests the residual variance, which a result-regression critic cannot pass. The windfall plant regresses truth on critic, unattenuated by the truth's noise.
- **Missing pieces.** Advancement draws are now included as chance events; memories are recomputed rather than stored (the critique's 1.8–5.9 TB); the critic reads the matchup tag, so later rounds price the forecast's opponent.

**From invent (chance contrasts and turn closure; critique: repairable).**

Kept: the forecast protocol. With `own1` deterministic, the performance-difference lemma reduces to side A's changed decisions.

Repaired:
- The forecast is now audited on the selected switches by coins at those switches, which removes the optimizer's curse the critique found.
- Turn-closure labels remain as a variant of the look-ahead, chosen by calibration.

Rejected:
- Forced-outcome contrasts, because the joint fit of the realised game's events carries about 70 times the information per playout.
- Coupled dice, because strike-order misalignment breaks them (critique item 8).

Adopted from the critique:
- the decision-moved plant: the village tax, with a hidden rate, which doubles as the leak test of the fog boundary;
- per-outcome forecasts in the match's decisive-game estimand.

**New this round.**
- The midpoint audit.
- Candidate coins as a per-decision form of conservative policy iteration's mixture.
- Truth-on-prediction calibration as the shrinkage that corrects selection.
- The raw-base player π_1^raw, which tests directly whether the learned stop rule replaces the −1.5 offset.

## Open questions

1. **ρ_C.** Will the critic fall well below the linear columns' 0.83? No first-principles number exists. Backgammon nets reach about 0.04 ([Montgomery 2000](https://bkgm.com/articles/GOL/Feb00/var.htm)), but Wesnoth's state is far richer. The record's critics failed, though under result regression.
2. **Stability of the joint objective.** Its gradient on unrealised outcomes is the residual times a centred indicator: noisy while the residual is large, so it may learn slowly at first, as residual-gradient methods do. Does the semi-gradient initialisation cure that, or is a schedule between the two rules needed?
3. **The advantage spread s** among the top-3 candidates. Below 0.005, calibration needs millions of deviations, and gains near the argmax base are small anyway. The harden critique's integral already shows that s cannot be as large as 0.02 everywhere.
4. **The fog ceiling** Var(E[Â | h]) / Var(Â): how much of the privileged advantage the fogged player can know at all.
5. **Compounding.** Do single-decision switches compound into the multi-decision changes strong play needs, round after round, or does the method stall near the imitation policy's neighbourhood?
6. **Numerics.** Does generation numerics agree with eval numerics on at least 98% of decisions? If not, every game costs about 5 times more.
7. **The user's ruling** on coin regression as outcome-driven learning.
8. **Exploitability** of a deterministic fogged player, which no Elo against one reference shows.

## Research targets for round 2

- **Learned martingale control variates and deep BSDE.** Convergence and failure modes when the residual is large at the start (Han, Jentzen & E 2018; Vidales, Šiška & Szpruch 2021; RELAX's joint training of the control variate). Do these papers report the slow early phase, and what cures it?
- **Experimental design for policy improvement.** Optimal coin propensities to minimise the variance of the switch decision or of the audit:
  - adaptive micro-randomized trials (Thompson sampling with known propensities);
  - Differences-in-Q and its bias–variance results (Farias et al. 2022);
  - local experimentation in equilibrium (Wager & Xu 2021).
  Whether a non-uniform t, or stratified t, beats the midpoint for the audit.
- **Information-state advantages under memory policies.** Whether the 64-slot memory is a sufficient history summary for the critic's identity to hold exactly; the asymmetric-critic literature after Baisero & Amato.
- **Compounding.** Evidence from approximate policy iteration with rollout-calibrated heads in long stochastic games: how many rounds before gains decay, and what restarts them (larger candidate sets, distillation, critic growth). Tesauro & Galperin's backgammon rollouts and AlphaGo's policy-network RL as data points.
- **The stop rule as optimal stopping.** Whether regression-based stopping with cross-fitting (Longstaff–Schwartz; Woo, Liu & Choi) suggests a better candidate set than "end versus the best act".
- **A cheap pre-box estimate of s and of the ratio's scale** from records already on disk (`training/metrics/critic_step1_20261009/`, the luck walk's rows), so S1's size is set before renting.
- **Divergent alternatives.** Whether a generative world model (Stochastic MuZero's afterstates, chance codes) could replace the exact simulator's role in the look-ahead at a lower cost per candidate. It should be kept as a challenger, not adopted.
