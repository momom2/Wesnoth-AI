# Critique of the round 1 challenge (learn the evaluator from the dice)

**Verdict: repairable.** The identity at its core is correct. What the design builds on it is not yet supported: the accounting counts labels a game does not contain, the critic is never trained or read on the states the operator prices, two planted checks fail for correct code and one passes without the mechanism, and the cited guarantees do not hold at the chosen step.

## Strongest part

The telescoping holds. At argmax V^π is constant along deterministic segments, so Σ_{k≥t} L_k(V^π) = z − V^π(x_t), and E‖R_t‖² = e(x_t)² + Σ_k E Var_o e(x_k^o) follows from orthogonal martingale increments. This gives the project its first critic readout with an absolute zero and no playouts, and it keeps the critic off the actor's trunk. It also composes with the seed: any critic frozen before the games it corrects is an unbiased control variate on fork labels, so a good critic makes the forks cheaper.

## 1. The accounting counts one label per game as 150

With U frozen, Y_t(U) − Y_t'(U) = Σ_{t≤k<t'} L_k(U). The targets inside a game differ only by U's own corrections, and z is the game's only fresh number. A chance event carries a label of noise 0.25 only once U already prices the later events, which is the thing being learned, so section 1.3 assumes its conclusion.

The training rule is also narrower than section 1.2 says. Regressing C(x_t) on Y_t(U) never evaluates C at an unrealized outcome, so the Σ Var_o e term sits in the readout, not in the gradient. The target is the λ = 1 per-decision control-variate return (Sutton and Barto 2018, §7.4; De Asis and Sutton 2018), with nature as the behaviour policy. That is bootstrapping, and it beats result regression only through generalization across games.

So the record's "effective sample about the game count" still applies until U is good. The stage 0 prediction of 0.55 (0.40-0.70) is derived from nothing, and a miss would not say which refresh stalled. Repair: read the ratio at each refresh, predict each refresh from the one before, and price the case where the refreshes stall near the linear columns' 0.85.

## 2. The operator prices states the critic never sees

Generation is argmax self-play with no exploration. For 8 of the 9 candidates T(x, a, o) lies off the data, and the chance training covers only outcomes of the action π chose; E‖R_t‖² bounds error on those states only.

R_0 at game start sees even less. An error that chance does not change, such as a function of surviving units' hexes, leaves R_0 untouched whenever it is zero at x_0. The readout is blind to the failure section 4 names as the main risk, and nothing reads the operator's ranking before stage 2's $16.

Repairs:
- Average R_t² over all decisions.
- Make exploration a chance event of known law: at one uniformly drawn decision per game, a uniformly drawn candidate among the top 8, with the flag carried in the state. The identity then covers it, and its L_k trains action differences as a fight's does.
- Before stage 2, regress luck-corrected fork labels on the operator's Â at 10,000 single decisions. At 8 playouts a side that is 160,000 playouts, about $2: the per-decision calibration the ruling asks for.

## 3. The guarantees do not hold at the chosen step

- **Conservative policy iteration.** f = 10% flips about 22 of a side's 225 decisions a game. Kakade and Langford's bound is informative only when the expected deviations per trajectory are well below one; at 22 its penalty exceeds the advantage, and every state after the first flip is off π's distribution. G_pred also comes from the critic that chose the flips by their top Â, so the optimizer's curse biases it upward (Smith and Winkler 2006). The range s in 0.5-1.0 absorbs both by assumption. Repair: choose flips with one critic, score them with another trained on disjoint games, and score a sample against fork labels.
- **Magnetic mirror descent** converges for the stochastic iterate with the exact Q of that iterate in a zero-sum game. Here Q is that of argmax play on the true state, play is argmax, and there are three or four iterations. It is a regularized greedy step, with no equilibrium property to claim.
- **Fog.** The first-order agreement needs η·sd(Â | history) ≪ 1, but η is set by the flip rate. At a flipped decision η|Â| is at least the top-two logit gap (median 0.59 nats on `parity3`'s record). At that scale E[q_x | history] moves toward the share of worlds in which a is best, which is perfect-information Monte Carlo's vote, the textbook case of strategy fusion.
  - The record puts the hidden-unit share high: selection on the true state gained +0.049, from the mover's view −0.002, and O100 kept little of T100's edge.
  - The actor's target has the floor Var(Â | history), so "no noise floor" stops at the critic. Stage 3's fraction of 0.6 contradicts the record without a derivation.
  - Stage 2's operator reads hidden units, so its gain is no evidence for the fair actor.
  - Repair: drop the 0.6 and predict stage 3 from G_pred on the actor's own flips, scored by cross-fitted critics.
- **Opponent.** Stage 4's critic prices play against π_k, while the match is against `own1`. From round 2 on, G_pred and the Elo it predicts refer to different opponents.

## 4. Planted checks

- **Windfall.**
  - The truth is 16 luck-corrected paired playouts a position, a noise variance of about 1.5/16 ≈ 0.096. Regressing the critic on that truth attenuates the slope to Var_true / (Var_true + 0.096). A perfect critic with a spread of 0.1 across positions reads about 0.1 and trips the kill at 0.6.
  - The rate check is dominated by the same noise and reads about 1, not 1.7. Repair: regress the truth on the critic, and subtract the known truth noise.
  - Plain result regression on 30,000 games learns the coin too (standard error about 0.011), so the check passes without the mechanism.
- **Chest.** With the prior at 0.001 and argmax play, no training game takes the chest, so Â(chest) extrapolates a feature the critic has never seen. The check fails for correct code, or passes by luck. Under exploration-as-chance it becomes the right test of the off-policy path.
- **Null.** `parity3`'s replayed choices agreed with the recorded ones at 98.2%, so the 99% agreement bar can fail on arithmetic alone. Set the bar from stage 0's determinism measurement.
- **A missing chance source.** Self-play draws each advancement uniformly over its targets (`draw_uniform_advance`, `rust/wesnoth_core/src/core_units.rs`), 1,777 in the luck match's 804 games. Enumerate them with the fights.

## 5. Costs

The listed items add up from the seed's prices: stage 0 is about $2.2 of generation, $1.6 of correction forwards at 750 a game, and under $1 a training epoch. The missing item is the memories. The critic reads both, float32 64 × 384 per side, which is 196 KB a position. At 150 positions a game that comes to 1.8 TB for stage 0's 60,000 games and 5.9 TB for stage 2's 200,000. The alternative is rerunning the actor over each game's recorded actions, 450 forwards a game, about the cost of generating the game. The CPU cost of encoding the outcome states (about 750 a game) is not measured either.

## 6. Novelty

- **Claim 1** is covered. Training a deep network so that the martingale-corrected estimator's residual vanishes is the deep BSDE loss (Han, Jentzen and E 2018), and Algorithm 4 of Vidales, Šiška and Szpruch minimizes the variance of a learned martingale control variate. Zero-variance estimates along sampled trajectories from a predictive baseline are in Davis, Schmid and Bowling (2020), in tabular form. What may be new is the transfer to a game critic with exact chance enumeration, and the ratio as a readout.
- **Claims 2 to 5.** Claim 2 is correct but small. Claim 3 rests on section 1, and claim 4 fails at the chosen η. Claim 5 is piKL's human anchor (Jacob et al. 2022; Bakhtin et al. 2023) at a finer grain.

## References

- Davis, Schmid and Bowling 2020, ICML. https://proceedings.mlr.press/v119/davis20a.html
- De Asis and Sutton 2018, UAI. https://arxiv.org/abs/1807.01830
- Han, Jentzen and E 2018, PNAS 115(34). https://arxiv.org/abs/1707.02568
- Jacob et al. 2022, piKL, ICML. https://arxiv.org/abs/2112.07544
- Smith and Winkler 2006, "The optimizer's curse", Management Science 52(3).
- Sutton and Barto 2018, *Reinforcement Learning*, 2nd ed., §7.4.
- Vidales, Šiška and Szpruch 2021, Applied Mathematical Finance 28(4), https://arxiv.org/abs/1810.05094 (Algorithm 4 checked on the arXiv text).
