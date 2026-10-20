# Round 1, hardening: factorial self-play and closed-form advantage heads

Stance: keep the seed's policy iteration on per-decision labels; replace its label source. A coupled fork costs about 2.25 playouts per label. A self-play game with a few randomized single-decision deviations gives one label per deviation, each with the game's outcome noise: about 10 times more information per dollar at 8 deviations a game (1.4).

## 1. The algorithm

### 1.1 What is learned

Two advantage heads, read off `own1`'s frozen trunk (memory included), each a linear function fitted in closed form:
- **Stop head**: `D_stop(h) = Q(h, end) − Q(h, best act)`. Here `best act` is the non-end action with the largest joint prior, and `h` is the acting side's observation plus memory. Features: the end_turn token's context, the best act's actor and target contexts, the policy's own end and best-act log-priors, and the value. About 1,160 dimensions plus a constant.
- **Action head**: `D_act(h, b) = w·(φ(h,b) − φ(h,a0))`. Here `a0` is the argmax non-end action and `b` is one of the next 3 by joint prior. `φ` is the actor context, the target-hex context, the weapon and kind one-hots and the four factored logits. About 780 dimensions.

The deployed player picks actions with the network's logits corrected by these heads. It never picks turns. The −1.5 offset is not used anywhere in the new player.

### 1.2 The labels: randomized single deviations in argmax self-play

Both sides play the argmax decode. At an eligible decision a coin with known probability `p` decides whether to play an alternative instead:
- **Stop-eligible**: the raw log-odds of end against best act lies in [−4.5, +1.5], that is, the −1.5 boundary ± 3. The alternative flips the choice: end becomes best act, or best act becomes end. `p` is set for about 1.5 flips per side per game.
- **Action-eligible**: any non-end argmax with at least 2 candidates. The alternative `b` is drawn uniformly from the top 3. `p` is set for about 2.5 deviations per side per game.

That makes D ≈ 8 deviations a game. Each deviation is one decision, and the play after it re-decides after every roll, as the ruling requires. The coins come from a generator keyed by (game seed, decision index), separate from the dice. The label is the game result `z` ∈ {−1, 0, +1} from side A's view, a capped game counting 0. No fork and no extra playout are needed, and generation runs at the seed's 17,000 games an hour.

### 1.3 The estimator

**Regressor.** At chosen decision i, with side sign `s_i` = ±1:
- `x_i = 1[dev_i]·(φ(h_i,b_i) − φ(h_i,a0_i))`;
- its conditional mean is `m_i = p·Σ_b q(b|h_i)(φ(h_i,b) − φ(h_i,a0_i))`, computed at play time from the logged candidates;
- the centered regressor is `x̃_i = x_i − m_i`, and per game `X_g = Σ_i s_i x̃_i`.

The stop head is the same with `φ_stop(h)` and a flip direction of ±1. The box keeps only `(X_g, z_g, setup_g, S_g)` per game: a few KB per head.

**Orthogonality, the core invariant.** `E[x̃_i | h_i] = 0` exactly, and the coin is independent of everything before it. So `x̃_i` is uncorrelated with any quantity fixed before decision i: the setup, the earlier dice, the earlier deviations. Likewise `E[x̃_i x̃_j] = 0` for i ≠ j.

It follows that the least-squares target `w*` of `z_g ≈ c + X_g·w` solves `E[X X^T] w* = E[X z]`. Its fitted values are the best linear approximation, in the feature space, of `Q^{π_D}(h,b) − Q^{π_D}(h,a0)`, the advantage under the deviating policy π_D. Two consequences:
- The luck before decision i adds noise but no bias. This is the orthogonalization of the R-learner and double machine learning, here with known propensities ([Nie & Wager](https://arxiv.org/abs/1712.04912); [Chernozhukov et al.](https://arxiv.org/abs/1608.00060)).
- In the centered directions the information per game is `Σ_i E[x̃_i x̃_i^T]/σ²`, so it grows with D. It is the factorial principle (Fisher), the decision-space analogue of simultaneous-perturbation gradient estimation (Spall 1992).

**Variance reduction, unbiased by construction.** The label is `z'_g = z_g − ĉ(setup cell) − β̂·S_g`:
- `ĉ` is the cell mean over (map, factions, side);
- `S_g` holds the seven chance columns of `tools/match_luck.py`, each mean zero given the position before its fight;
- both are cross-fitted.

Since `x̃` is mean zero given `h` and each chance term is mean zero given its fight's prior state, `E[x̃_i S_g] = 0`. The record says these columns leave 0.848 of the variance, so σ² ≈ 0.8. At most 7 of about 807 games were capped in the `own1` matches, so Var(z) ≈ 1.

**Fit.** `ŵ = (Σ_g X_g X_g^T + λI)^{-1} Σ_g X_g z'_g`, with λ = σ̂²/τ̂² set by maximizing the evidence (MacKay 1992). It is closed form: a d×d solve with d ≈ 1,200. The posterior covariance comes with it.

**Error, predicted before the fit.** For a held-out decision, `E[(D̂ − D*)²] ≈ σ²·d_eff(λ)/N` plus a shrinkage bias. Here N is the number of deviations and `d_eff = tr(Σ(Σ + λ/N)^{-1})` (random-design ridge, [Hsu, Kakade & Zhang](https://arxiv.org/abs/1106.2363)). Let v be the variance of the true advantage across eligible decisions. The head's squared correlation with the truth is then `r² ≈ v/(v + σ²d_eff/N)`. Example: σ² = 0.8, d_eff = 500 and v = 5×10⁻⁴ give r² = 0.5 at N = 800,000 deviations, that is 100,000 games, about 6 box-hours or $2.5.

**Fog.** `h` is the acting side's information and `z` the true result. On-policy games draw the true state from its posterior given `h`, so the regression estimates `E[Q(s,·) | h]`, the information-state advantage of a fogged player whose future play is also fogged. No full-state teacher plays any continuation, so there is no strategy fusion. Hidden units enter only through the outcome.

**Invariants (each becomes a test):**
- I1: `m_i` equals the exact mean of `x_i` over the coin and `q`, enumerated on logged decisions, to float precision.
- I2: on data, `X_g` regressed on setup and on `S_g` gives zero coefficients within their SE.
- I3: the head's features at a decision equal the player's own forward, bitwise, on the same weights.
- I4: no game's result enters the head that forecasts that game (cross-fitting by game).
- I5: permuting `z` across games puts the held-out correlation and the forecast at null.

### 1.4 Why this extracts more signal per dollar

**Against the seed's fork pairs.**
- Fork: one label of noise variance `2(1−ρ)σ_s²` costs 2 playouts plus 2/K for the base game; that is 2.25 with K = 8. Here ρ is the outcome correlation between the coupled branches.
- Factorial: D labels of noise σ² cost one game, 2 playout units.
- The ratio is about `2.25·D·(1−ρ)·σ_s²/σ²`. It is about 10 at D = 8, ρ = 0.3 and σ_s²/σ² = 0.85.
- Forks win only above ρ ≈ 0.93. The only coupling measurement on record found a median of 0 shared fights, though on a collapsed checkpoint and within one turn, so ρ that high is not expected.
- The seed's 8 replicates per option add cost without adding coverage. Independent decisions beat replicates for regression.

**Against policy gradient.** Same kind of games, but the update is a closed-form estimate with a predicted error and a sign-based switch: no value head, no tilt of a sampled action's log-probability, the withdrawn design's two faults. The kill arithmetic on record (about 19,000 games per decision context) certifies each context separately; pooling through a head is what makes per-decision labels affordable, for forks and deviations alike.

**Against critics trained on results.** A value critic spends its capacity on between-position variance, mostly material. The coin removes that variance by design: the target is the within-position difference only.

### 1.5 Improvement step, forecast, iteration

**Improvement.** `π_{k+1}(h)` is the argmax over {a0} ∪ top-3 of the posterior-mean `D̂`, with a0 scored 0. For the stop decision, the player ends iff `D̂_stop > 0` inside the window, and keeps the raw decode outside it. Bayesian shrinkage is the remedy Smith & Winkler (2006) propose for the optimizer's curse.

**Forecast, written down before each match.**
- `Ĝ` = the per-side mean, over held-out `π_k` games, of `Σ D̂^{(other fold)}(h_i)` over the decisions where `π_{k+1} ≠ π_k`, in result units.
- In Elo, `≈ 350·Ĝ` (capped fraction near 0); one match SE of 12.3 Elo is about 0.035 result units.
- Cross-fitting removes look-ahead bias. In regression-based stopping that bias grows with the ratio of regressors to paths ([Woo, Liu & Choi](https://arxiv.org/abs/1810.02071)). This is double Q-learning's idea applied to a forecast (van Hasselt 2010).

**During the match.** The same sum is evaluated on `π_{k+1}`'s own states, without using results. This is the plug-in performance-difference lemma (Kakade & Langford 2002). A gap between it and `Ĝ` is covariate shift; a gap between it and the match is head error.

**Iteration.** Round k+1 generates its games under `π_{k+1}` on both sides. Each head's prior is the previous posterior widened by a drift variance fitted by evidence, a Kalman update across rounds (Geist & Pietquin 2010). Stale labels are discounted by measurement, not by a guessed step size. `parity3` stays out of all training as the transfer yardstick.

## 2. The regimen

Matches are 800 decisive games, 12 to 17 minutes, about $0.1. Games run at 17,000 an hour, $0.42 an hour.

| Stage | Compute, cost | Predicted readout | Kill line |
|---|---|---|---|
| **0. Instruments, two anchors** (no training) | 2 matches, $0.2 | A: `own1` at offset 0 vs `own1` at −1.5, E[z] −0.25 to −0.60 for offset 0. B: `own1` with deviations vs `own1`, −0.02 to −0.08 result units (dilution). | B below −0.15: halve the rates. |
| **1. Planted game** (code path P1, logger and fit) | 10k games, $0.25 | Training-only scoring adds +0.04 per deviation *to an attack* and +0.03 per deviation *to end*. The head recovers both within 2 SE of the formula SE, about 0.008 and 0.005. A permuted-z run reads null. | Recovery outside 2 SE, or a placebo ≠ 0: a bug in that path; nothing downstream runs. |
| **2. Round-1 data** under `own1` | 150k games, $3.7 (1.2M deviations) | σ̂² 0.75–0.85. Held-out game-level correlation (30k games, SE 0.006): stop 0.02–0.10, action 0.02–0.07. These give the first measurement of v. Known answer: on Match A's games the stop head's summed prediction at offset 0's early ends equals Match A's E[z] within 2 SE. Dose-response: games at half rate (a third of the set) give a forecast within 2 SE of the rest. | Correlation under 0.012: that head carries no signal at this N; stop it. Known answer missed: stop head not adopted. Dose-response missed: dilution biases the labels; lower D. |
| **2b. Extension rule** | at most $15 | If v̂ puts r² below 0.6, add games up to r² = 0.75 (formula). | — |
| **3. Player `π1`** = `own1` + stop head (no offset) + action head | 4 matches, $0.4 | Forecast `Ĝ`, computed before: stop alone +10 to +55 Elo; both +20 to +90, central +50. The matches: `π1` vs `own1`; stop alone vs `own1`; `π1` vs `parity3`; `π1` at 0 slots vs 64. | Adopt if at least +2 SE and within 2 SE of `Ĝ`. Below `Ĝ` by more than 2 SE: the head misgeneralizes (plug-in vs `Ĝ` says whether by covariate shift). Gain over `parity3` under half the gain over `own1`: exploitation of `own1`. Stop head within ±1 SE: replaces the offset if the user accepts parity for removing the kludge. |
| **4. Rounds 2–3** under `π_k` | 150k games + matches per round, $4.2 each | Gain per round about half the previous; forecast matched within 2 SE. | Forecast under 0.035 result units twice in a row: the frozen features are exhausted, go to 5. |
| **5. Trunk stage** (code path P3) | about $5 | The network is fine-tuned on (a) the same orthogonalized objective by SGD over windows ending at chosen decisions, the memory carried, (b) distillation of `π_k`'s head-corrected choices into the logits, (c) the human-imitation anchor. Gradient share is capped by the existing probe. Planted first: +δ per deviation onto a village hex, a planted signal outside the frozen span. SGD recovers it at least as well as ridge on frozen features recovers its projection. Then the held-out correlation of the next round rises above the frozen head's. | No rise: the trunk does not learn the advantage; report before spending more. |

The planted check of the decode path (P2) is a unit test: a head with planted weights must switch exactly the decisions its sign predicts.

**Total for the $85.** About $20 on the planned path. The extension rule and one rerun per stage bring it to $50 at most. The $1,000 case is argued only from stages 3–4. If forecasts and matches agree for two rounds, a large run is 25–40 rounds with trunk stages between them, and its predicted gain is the measured per-round decay extrapolated, stated with its error.

Whether this reaches the strongest humans cannot be predicted from the record. Nothing links `own1` to human ratings, and that link is a decision for the user.

## 3. What is new, and the nearest prior work

**Nearest prior work.**
- Per-decision rollouts with the base policy in a dice game: [Tesauro & Galperin 1996](https://proceedings.neurips.cc/paper/1996/hash/996009f2374006606f4c0b0fda878af1-Abstract.html), backgammon.
- Rollout-labelled policy iteration: Lagoudakis & Parr (2003); [Dimitrakakis & Lagoudakis 2008](https://arxiv.org/abs/0805.2015).
- Branch rollouts with common random numbers: [TRPO's vine](https://arxiv.org/abs/1502.05477); event-keyed versions in [Buffalo et al. 2026](https://arxiv.org/abs/2603.11084) and [Sakib 2026](https://arxiv.org/abs/2609.24144).
- Monte Carlo advantages in place of a critic: [VinePPO](https://arxiv.org/abs/2410.01679).
- Least squares on deep features: LSPI; [LS-DQN](https://arxiv.org/abs/1705.07461).
- Regression-based stopping: Longstaff & Schwartz (2001).
- Known-policy correction terms: [AIVAT](https://arxiv.org/abs/1612.06915).

**New here** (searches found no precedent; unverified beyond that):
1. Factorial randomized single-decision deviations inside argmax self-play, analysed jointly by propensity-orthogonalized regression of one terminal result. The pieces are known (factorial designs, R-learner orthogonalization, AIVAT-style centering); their use as the label source of policy iteration in a long stochastic game was not found.
2. An error curve predicted in closed form before training (measured σ², exact `d_eff`), checked by a planted game at a pre-stated size.
3. A learned end-of-turn rule as regression-based optimal stopping over the policy's own turn, cross-fitted against look-ahead bias.
4. A readout chain that localizes failure: planted recovery (code); held-out correlation (signal); a known answer from a measured decode match (extrapolation); forecast, plug-in, match (generalization, covariate shift).
5. Bayesian (Kalman) heads across policy-iteration rounds, the drift variance fitted by evidence.

## 4. The weakest point

**The frozen trunk may not carry the advantage linearly.** Every number above assumes the advantage across eligible decisions has a variance v near 10⁻³, and that a linear head on imitation features can see a useful part of it. Neither is measured. Two records warn against it:
- Decode offsets by action kind and time of day came out flat (+1.7 ± 12.3).
- Learned critics did no better than material on the player's observation.

The per-game signal-to-noise is `D·v/σ²`, about 0.003 to 0.03. If v is 10 times smaller, round 1 needs 10 times the games, about $37: affordable, but most of the budget. If the signal is not linear in these features, stage 2's correlation reads near zero at any N and everything rests on the trunk stage, the least predictable step.

Lesser risks: the labels price π_D, not π (the dose-response check bounds this only to its SE); round 1's training opponent is the yardstick (the `parity3` match measures exploitation); and the method learns from outcomes, which the killed list touches. It is a regression with a predicted error, not outcome policy gradient, but the user should rule on it.

## Decision record

- **Rejected: coupled forks as the main label source**, because they lose by about 10 times in information per dollar below ρ ≈ 0.93 (1.4). They are kept as a reserve, in case a later stage needs labels at chosen states.
- **Rejected: truncated value-bootstrapped labels, including multilevel corrections**, because the noise after the horizon cannot be reduced. When the value cannot predict the outcome, the correction level costs as much as full labels.
- **Rejected: the 199-position turn benchmark as a test for graders**, because it compares turns already played with their own dice.
- **Rejected: a full-state teacher for continuations**, because the label must price the fogged player's own future; otherwise strategies fuse.
