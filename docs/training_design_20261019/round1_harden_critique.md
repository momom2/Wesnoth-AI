# Critique of round1_harden.md (round 1)

**Verdict: repairable.** The label source is sound and is the best on the table. As written, though, three things are wrong: its kill lines pass a head that carries no usable signal, its stop window cannot sample the side the record favours, and its sizing contradicts its own forecast. It is also outcome policy gradient under another name.

## Strongest part

The estimator. Three properties make each per-decision contrast unbiased from ordinary self-play games, with no fork and no critic: a coin independent of the dice, known propensities, and luck terms that are mean-zero given the position before each fight. The continuation re-decides after every roll, so the ruling holds.

I checked the orthogonality: the coin and q depend only on h_i, so E[x̃_i | full history before i] = 0, the cross terms vanish, and a linear advantage is recovered exactly. The comparison with forks is conservative: fresh-dice branches are independent given the state, so ρ = 0 and the ratio is about 15. The fog argument is right (the fogged player's information-state advantage, no strategy fusion). The costs add up ($18 on the planned path).

## 1. It is episodic natural policy gradient, and novelty claim 1 is covered

Tilt the deviation mixture: π_θ(b) ∝ p·q(b)·exp(θ·Δφ_b), with Δφ_a0 = 0. At θ = 0 the score ∇log π_θ(a_i|h_i) equals x̃_i exactly. So X_g is the game's summed score, and fitting z on [X_g, 1] by least squares is the critic of the episodic Natural Actor-Critic, which regresses the episode's return on the summed ∇log π and a constant ([Peters & Schaal 2007](https://www.esann.org/sites/default/files/proceedings/legacy/es2007-125.pdf), eNAC table).

The rest is in the literature too:
- That w·x̃ approximates the advantage is the compatible-function-approximation theorem ([Sutton et al. 2000](https://papers.nips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)).
- The greedy switch is where the natural gradient points, "toward choosing a greedy optimal action" ([Kakade 2002](https://papers.nips.cc/paper/2001/hash/4b86abe48d358ecf194c56c69108433e-Abstract.html)).
- Micro-randomized trials randomize hundreds of decisions per person with known probabilities, centre the treatment by them ([Boruvka et al. 2018](https://arxiv.org/abs/1601.00237)), and estimate per-decision effects on one distal outcome with cross-fitting ([Qian 2025](https://arxiv.org/abs/2502.13500)).
- In games, outcome-sampling MCCFR with baselines estimates per-decision values from one terminal result under known sampling probabilities ([Schmid et al. 2019](https://arxiv.org/abs/1809.03057)).

So §1.4's "no tilt of a sampled action's log-probability" is false: this is the killed list's "outcome-driven policy gradient". Its differences from the withdrawn design are real (argmax base, rare deviations, luck columns, closed-form fit, no value head), but the user should rule on it as policy gradient. What is new is the assembly for this game.

## 2. The sizing contradicts the forecast

The worked example takes v = 5×10⁻⁴, an advantage spread (sd) of 0.022. Stage 0B predicts −0.02 to −0.08 per side for about 4 deviations, so a deviation's mean advantage μ is about −0.005 to −0.02.

I integrated the first-order gain, Σ E[max(0, max_b D̂_b)], assuming three alternatives per decision, calibrated posterior means distributed N(μ, r²v) at r² = 0.5, and about 200 action-eligible decisions per side. It comes to 0.44 result units at μ = −0.02 and 1.3 at μ = −0.01, the player switching at 28% and 60% of decisions; 1.3 is outside a result's ±1 range. That is 5 to 15 times stage 3's +20 to +90 Elo, a range the document derives from nothing.

So either v is 10⁻⁴ or less, or the forecast does not follow from the model. At v = 10⁻⁴, stage 2's 750k action deviations give r² ≈ 0.16, and the gain is about +2 Elo at μ = −0.01 and +40 at μ = −0.005. The gain comes from the tail: it depends on the head's resolution relative to |μ|, not on r². Size N on the predicted switch gain at the μ and v that stage 2 measures; expect 300k to 1M games ($8 to $25), not 150k.

## 3. Kill lines that a head with no usable signal passes

- **Stage 2's correlation kill line.** The log-prior gaps among the features carry the mean cost of deviating. A head that knows only μ predicts results with correlation √(5μ²/0.8) ≈ 0.025 at μ = −0.01, with v = 0: twice the kill line of 0.012. The stop head's mean does the same, and the "first measurement of v" reads v + μ².
- **Match A's known answer.** An intercept-only head meets it, and that head is the constant offset.
- **The seed's kill was dropped.** The seed required the head to predict held-out labels better than the best constant offset. Restore it: test each head against a nested model with only the mean and the log-prior gaps, and read out the held-out gain in the directions that change a decision.

## 4. The stop window does not straddle the boundary

`shift_end_turn_prior` (tools/raw_player.py) adds the offset to end_turn's logit, so own1 ends iff L = log P(end) − log P(best act) ≥ +1.5. On raw L, the window "−1.5 ± 3" = [−4.5, +1.5] lies entirely on the acting side. Own1 never ends inside it, every flip is act→end, and the head can only learn to end earlier than own1. Acting longer, the direction the record favours (6 of the 7 better turns took more decisions), is never sampled. Centre the window on L = +1.5, or define L after the offset, and say which.

## 5. The planted check cannot fail on the game side

The plant adds +0.04 to z for each *logged* deviation to an attack, so it is recovered even when the deviation is logged but not played, played at the wrong decision, or not written into the memory, and when the plant and the regressor share a wrong side sign, which cancels.

Plant through the simulator instead. Key the bonus on the game record (tools/game_record.py), not the deviation log. Add a planted effect that only the continuation realizes, such as a village held at game end, and measure its true size with forks at sampled positions (the reserve the proposal keeps), so two independent code paths must agree. I5 detects leakage only.

## 6. The step is greedy, so the forecast holds only to first order

The seed kept each step small; the proposal discounts stale labels "by measurement" but switches wherever D̂ > 0. The eNAC authors rule out greedy improvement: with approximated value functions, "a policy improvement ... cannot be guaranteed". At §2's switch rates, Ĝ computed on π_k's states does not forecast π_{k+1}. Repairs:
- Switch only where the posterior P(D > 0) exceeds a threshold, set so that the plug-in and Ĝ agree.
- Select with one fold's head and score with the other's. Scoring other-fold D̂ on decisions picked by the full-data head keeps the optimism cross-fitting should remove.
- Watch for cycling: greedy self-play steps can cycle in a fogged zero-sum game, and `parity3` detects exploitation of own1, not cycling.

## 7. The throughput figure is for a different configuration

The 17,400 games an hour are the priors-only server with compiled bf16 numerics, "for generation only, never eval" (BACKLOG.md). The heads need about 4,300 floats per decision shipped (about 37 MB/s, unmeasured), fitted under generation numerics and deployed under eval numerics, so I3 (bitwise) cannot hold. Ĝ needs per-decision features of held-out games, which the (X_g, z, setup, S) record drops: budget a forward pass over the records.

## 8. The stage that has to scale is the fire-and-hope one

Frozen linear heads over the top four actions are a decode correction; under the assumed halving per round they total about twice round 1. Everything beyond rests on stage 5, whose readout ("rises above") has no number.

Minor: the luck columns are hand-written material evaluators. They are unbiased for any Q, but using the learned value as Q keeps "learned only" literal.
