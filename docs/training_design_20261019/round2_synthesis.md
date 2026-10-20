# Round 2 synthesis

## Verdict

The design changed in substance this round. Its frame, the stage order and the critic's core objective stand; the changes are these:

- **A new mechanism for end_turn.** The stop decision is now an optimal-stopping problem, solved by Longstaff–Schwartz on forced continuation paths with the critic's end-of-turn value as payoff (invent's layer 0, fitted as its critique asked). This replaces the one-step stop switch and π_1^raw. The −1.5 offset leaves the decode in round 1, not at distillation.
- **Two flaws in the identity fixed (harden).** Bernoulli coins made every eligible decision a chance node whose term the residual omitted, which biased the identity. Single draws of sampled tails biased the gradient toward critics flat across rare outcomes. The fixes are a schedule drawn at game start and read by the critic as part of the closed-loop state, and a double-sampled loss.
- **A flaw in the audit fixed.** The midpoint rule was exact only to second order in switch interactions. J(t) has the degree of the disagreements per trajectory, 20 to 40, so P0 could not tell truncation from a bug (harden critique 3; challenge critique 7). Drawing t per game from the arcsine law and weighting by its density makes the estimate unbiased at every order. This costs 2.47 times the midpoint's variance, and the same games give an exact per-type split and the interaction curve J′(t).
- **Coins changed form.** Weak coins (deviation probability 0.2) at about 30 decisions per side-game replace a few strong ones. Per deviation they carry about 1.5 times the information, and they shrink the no-coin extrapolation that the harden critique flagged.

## What changed, and why

### From harden (critique: repairable)

**Kept:**
- the closed-loop critic state (true state, memories, rejection sets, remaining schedule), which the critique verified makes the identity exact without assuming the memory is a sufficient statistic;
- double sampling, also verified;
- the information accounting, which names the design's weakest point (chance differences carry several hundred times more χ² than decision differences);
- the per-type gate;
- the incumbent rule;
- head-only distillation by default;
- the two numerics checks;
- the reward-stream form of P2;
- the corrected P1 claim.

**Repaired as the critique asked:**
- **P1's rate test.** It now uses four disjoint fits tested by χ² on 3 degrees of freedom, replacing a ratio of two errors that rejected correct code five times in six.
- **The windfall plant.** Its turn and size are randomised, so the slope's standard error (predicted 0.10 to 0.15) fits its band.
- **The κ gate.** A type now opens on a lower bound above zero instead of closing on an upper bound below 0.3, which almost never fires. κ is named as the best-linear-predictor slope of Chernozhukov et al.
- **The no-coin extrapolation.** It is checked by κ's interaction with the remaining coin count.
- **κ and β.** κ is no longer called a ceiling on β.
- **Folds.** Each student now trains on half of the other fold's states and is calibrated on the other half, so no calibration game's state or result reaches it. The deployed head averages calibrated predictions.
- **The linear baseline.** It is computed on S1's own games.
- **The budget.** It is priced for the case where generation and eval numerics disagree.
- **Compounding.** The Newton and Bertsekas convergence claims are struck: policy iteration in zero-sum Markov games converges only under restrictive conditions (Winnicki & Srikant), and fog adds posterior shift. The audit and the cross-round matches are named as the only guards.
- **Stability.** The design responds to the contradicting evidence (Hu & Laurière on global-in-time losses) in three ways: Baird's interpolation from the semi-gradient fit, a predicted gap ρ_C(U_0) − ρ_C(joint) ≥ 0.05, and the backward scheme as fallback.
- **Novelty.** The claims now cite Off-policy DAE, MIVAT, Shani et al. and the BLP slope.

**Changed:** Dunnett coin laws gave way to weak coins. The critique noted that no per-decision contrast is ever estimated, so the law should maximise E[Var_p Q] per unit of cost. That leads to small deviation probabilities.

### From invent (critique: repairable)

**Kept:**
- **Layer 0**, which the critique called the strongest part. It targets the largest lever on record and includes π as a special case.
- **Turn coins**, which become the stop factor of the audit. The critique verified that dropping turns where the two rules agree is unbiased.

**Repaired:**
- **Fitting.** Layer 0 is fitted by Longstaff–Schwartz on realised path payoffs, which is low-biased and scores the rule as played, rather than by bootstrapped fitted iteration that four passes cannot converge.
- **The forecast.** It evaluates the rule actually played on held-out paths, so no maximum over noisy heads enters it.
- **Known answers.** They now cover both sides of π's stop: offset 0 (stops earlier, Match A) and offset −2.5 (stops later, a fresh match).
- **Cost.** The passes run on path states only, which keeps them near $1.

**Deferred:**
- **Swing layers 1 to 3.** A 10-Elo layer marginal needs about 94,000 audit games, the maximisation bias grows with depth, and deviation budgets are prior art (limited-discrepancy choice functions, Saute RL, LICRA).
- **P3**, which belonged to the layers.

### From challenge (critique: repairable)

**Deferred:** intent iteration, for four reasons:
- its evidence did not discriminate: the critique showed the −1.5 offset is a per-decision stop rule, and the +412 is confounded by the sampled tail;
- its cited support (Hu et al.) shows one-hot latents losing to direct imitation;
- codes fitted to realised turns name dice branches;
- retraining the trunk with a code token reopens the memory's causal-confusion channel.

**Kept:** its diagnosis that an action-level step sees a d-decision change only at its last decision. The design answers it in two places:
- Stopping, the one multi-decision change on record, is fitted as a multi-step problem.
- The audit's J′(t) curve measures whether switches interact. A flat curve says single-decision steps suffice. A bending one is the measured trigger for intents or swing layers in the $1,000 case.

### New this round (synthesis)

- **The arcsine-randomised audit.** It is unbiased for J(1) − J(0) at every interaction order. The optimal prior on t is proportional to the root of the expected squared score (Gelman & Meng 1998), which here is Beta(½, ½). Shared t gives an additive Aumann–Shapley split by type, and a synthetic chain test (I8) separates it from the midpoint.
- **Coin-onward residuals in the audit.** C(x_i) is subtracted as a baseline, along with every later event's term, so the audit's noise is the critic's error after each coin rather than the whole result's variance.
- **The round-1 player has no offset.** Within the stop window the stop head decides. Outside it every offset from 0 to −4.5 makes the same choice, so the sign of L decides.
- **P0 in two forms.** It runs with turn coins and with decision coins. With an exact estimator, a miss now indicts the code.
- **Prices.** A critic forward is priced at $5e-8, not `relset`'s $3.6e-8, since the critic's memory tokens are unmeasured. The planned path costs about $54. One of round 3 or an S1 rerun fits in the $85, not both.

## Open questions

1. **ρ_C.** No first-principles value exists, and the record's critics failed under result regression. Global-in-time residual losses are reported to converge poorly over long horizons, and a game has about 160 chance events.
2. **κ_move.** Even with weak coins, S1 carries only 140 to 320 χ² on decision differences. Will any move signal survive calibration, or is round 1 limited to stops, attacks and recruits?
3. **Over-continued states.** The stop rule's gain lies past π's stop, where coins visit at rate 0.2 per stop-window decision. The −2.5 known answer has a true effect near zero, so it bounds the critic's error there only weakly.
4. **Audit tails.** Per-game estimates are heavy-tailed near t ∈ {0, 1}. Does the bootstrap standard error match π²Mσ_a²/n, and is σ_a² (coin-onward residual) near the predicted 0.3 to 0.5?
5. **The fog ceiling** Var(E[Â | h]) / Var(Â). The student's R² is only its lower bound.
6. **Compounding under zero-sum play and fog.** Changing π shifts the posterior of the true state given the memory, which Ĝ_plug cannot see. Whether rounds compound is unmeasured until S4.
7. **Numerics.** Agreement within generation (a1) and between generation and eval (a2) are both unmeasured; batch-invariant kernels cost about 1.6 times throughput by secondhand reports.
8. **User ruling.** Are β calibration and fallback F, which regress outcomes on coins (episodic natural actor-critic), covered by the killed "outcome policy gradient"?
9. **Exploitability** of a deterministic fogged player, which one reference does not show.

## Research targets for round 3

- **Variance reduction for the arcsine audit.** Candidates:
  - control variates built from the forecast (an AIPW form per coin);
  - stratified t over the arcsine's quantiles;
  - separate laws of t per factor, which keep exactness but change the attribution path.

  Target: get the 2.47 factor back toward 1 without losing exactness. Check also whether the path-sampling literature gives a lower-variance estimator for a known endpoint J(0).
- **Value-of-information coin design.** Set ε per state with known propensities, higher where the student's calibrated advantage is near the switch threshold. Sources: adaptive micro-randomized trials; [Xu & Wager, weak-signal asymptotics for sequentially randomized experiments](https://ideas.repec.org/a/inm/ormnsc/v70y2024i10p7024-7041.html). How much does this cut the coins the switch decision needs?
- **Optimal stopping under partial information with a learned memory.** Whether rules measurable in the memory come close to the full-information Snell envelope; whether deep optimal stopping (Becker, Cheridito & Jentzen), which fits the decision directly, beats regression for a window rule; duality upper bounds (Rogers; Haugh & Kogan) as a check on the stop rule's suboptimality from the same paths.
- **Global against local residual losses for deep critics** over about 160 events: the deep-BSDE survey's evidence (Hu & Laurière), Kapllani & Teng's local losses, and Huré, Pham & Warin's backward scheme. Also what Baird's interpolation does in practice.
- **Policy iteration in zero-sum Markov games.** Which lookahead or mixing conditions give convergence (Winnicki & Srikant 2023 and their sources), and whether training each round against a mixture of earlier rounds is the cheap remedy.
- **A pre-box estimate of s.** Records hold commands and seeds, not priors. Can a one-hour box job replaying Match A's records give s per type before S1 is sized?
- **The multi-decision challengers, priced for the $1,000 case.** Intents with chance-free latents (Dichotomy of Control), with the critique's repaired spread readout (one estimator for codes and actions, within-code variance subtracted). Swing layer 1 with a powered audit. Both conditioned on J′(t) bending.
