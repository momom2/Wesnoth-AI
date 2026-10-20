# Round 2 critique of "invent" (swing layers, turn-scale policy iteration)

**Verdict: repairable.** No ruling is broken: the player picks single actions and re-decides after every roll, the backup takes its maxima over fogged Q values, and the turn coins compare policies, not fixed sequences. However, the proposal's main claim, measuring the depth at which critic errors start to be exploited, cannot be measured at the stated sizes. The forecast reads a biased quantity, the fitted iteration stops short of its fixed point, and the stop layer's known answer misses the states where its gain lies.

## Strongest part

Layer 0 is optimal stopping along π's own forced-continuation path, with C as the payoff. It targets the largest lever on record (the offset, +229) without adding new candidate actions, and it includes π as a special case. The turn coins are also sound. Dropping the turns where the two layers agree is unbiased: the agreement event depends on the turn-start state and the dice, not on the coin.

## Where it fails

**1. The audit is underpowered for the depths it is meant to measure.** The standard error is 347·(4·n_d·σ_r²/n)^½ Elo. At n = 20,000 and σ_r² = 0.6, it is 10.7 Elo at n_d = 8, 7.6 at n_d = 4, and 5.4 at n_d = 2.
- The prior for layer 2's marginal is 3–24 Elo, and adoption needs at least 2 SE.
- A true 10-Elo marginal at n_d = 4 passes with probability about 0.25. Since the first failure ends the search, n largely decides the adopted depth before any data arrive.
- 80% power at 10 Elo needs about 94,000 games per audit: about $5 under generation numerics, about $15 under eval numerics.
- n_d is not predicted per layer, and Ĝ_plug's own error is not stated.

**2. A miss would not point to the critic.** Ĝ_plug sums max Q_θ − C.
- With independent head errors of standard deviation σ_e, the maximum over five candidates adds about 1.16·σ_e. That bias compounds along each in-turn chain.
- At σ_e = 0.01 this is comparable to the proposal's own per-decision advantage of 0.015–0.03.
- The number of maxima grows with j. So even with a perfect critic, the gap Ĝ_plug − audit is expected to widen with depth.

The current design's safeguards (β shrinkage, the 0.8 threshold, M = 10) are gone. β is read only on the π-stopping output, which no layer player uses. Layer 3 may make about 30 uncalibrated deviations a game.

From round 2 on, the forecast takes V_j, a maximum under the `own1` tag, while the player chooses under the self-play tag. V_j is then not the value of the rule that is played.

**3. Four passes do not converge.** Each target refresh moves value back by one decision. In-turn chains reach about 14 decisions, and longer when the player continues past π's stop. After four passes, V_0 means "stop within four actions, then the initial head", not the Snell envelope.

The head's inputs cannot be cached: about 0.3 MB per state (384 tokens × 384 × 2 bytes) times 19.5M states is about 5.7 TB. So each pass reruns the trunk, about $1 a pass, and 14 passes take S2 from $6 to about $16.

For layer 0, the bootstrapped max is Tsitsiklis and Van Roy's form, which carries the fitted value forward. Longstaff–Schwartz uses the fitted value only to decide and scores each path by its realised payoff. That is low-biased and evaluates the rule actually played ([Oxford notes](https://courses.maths.ox.ac.uk/mod/resource/view.php?id=8720); Glasserman not re-read). The a* path does not depend on when the player stops, so one path per state serves every rule in a single backward pass.

**4. The stop layer's known answer misses the states where its gain lies.** Offset 0 stops wherever −1.5 stops, and sometimes earlier. Evaluating it therefore reads u only at states on π's path, never past π's stop, which is where layer 0's gain is expected (section 5's own worry). The check "V_1 ≥ V_0 ≥ C" passes for any head biased upward, so it certifies nothing.

**5. P3 can fail a correct implementation and pass a broken one.**
- **The layer-1 count.** After one pincer move, π's own re-decided a* may be the second move, and a* is free. Layer 1 can then collect pincers that the script, reading π's original path, counts as needing two deviations.
- **The audit line.** r × 0.7 ≈ 0.035 per feasible turn is about 1.1 audit SE. At one feasible turn a game, a layer 2 that collects nothing still passes.
- **The 0.7** is not derived.
- **The critic.** At $2.5, no critic is retrained on the planted game. P3 certifies the recursion and the counter, not the pricing of flanks, and should say so.

**6. Round 1's deep states have no defined source.** The ε-greedy layer-3 rule needs Q_θ,3, which is the head being fitted. A collect–fit–collect schedule is required, and it is neither specified nor costed.

Because the 64-slot memory is not a sufficient statistic, E[y | h] also depends on the policy that generated the data. Q-learning on observations misbehaves ([Singh, Jaakkola & Jordan 1994](https://mlanthology.org/icml/1994/singh1994icml-learning); this reading is mine). So the layers shift when π^(j) plays its own states.

**7. Guarantees that do not hold here.**
- **Property 2's improvement theorem** needs x | h to be the same under π and π^(j). With a lossy memory it depends on the policy. The performance-difference identity stays exact in x, but the inequality does not follow.
- **Section 1's "last i stages optimal after round i"** assumes unrestricted greedy improvement with exact values on a fixed MDP. Here the candidates are limited to the top 4 with at most 3 deviations, the values are fitted, the opponent changes, and the turn count varies. The claim cannot support "tens of rounds".

**8. The turn-scale evidence comes from the kind of comparison the ruling invalidates.** The 7 of 60 (docs/turn_gap_ref_prereg_20260921.md) graded fixed sequences replayed under the screen's turn salt, so the in-turn dice of the selection survived. The playouts ran at temperature 0.5 from the true state, and the screen sat inside its permutation null. In its favour, the fight-by-fight luck check explains only 2 of the 7. Treat the headroom as a hypothesis for the audits, not a premise.

## Costs

Twelve children per decision assumes at most one attack among the four candidates; three attacks at about 8 outcomes each give about 26. The play-time head is not priced in the audit games. With 14 passes and powered audits for layers 2–3, the plan costs about $68 under generation numerics and about $90 under eval numerics, which exceeds $85.

## Novelty

- **Deviation budgets.** A cap on discrepancies from a base policy, over restricted candidate sets that always include the base action, guaranteed to improve when leaves equal V^π, is the limited-discrepancy choice function ([Issakkimuthu, Fern & Tadepalli 2019](https://arxiv.org/html/1910.00614)). It descends from [limited discrepancy search](https://mlanthology.org/ijcai/1995/harvey1995ijcai-limited) (Harvey & Ginsberg 1995).
- **Values over a remaining budget.** [Saute RL](https://proceedings.mlr.press/v162/sootla22a.html) learns values over a state augmented with the remaining budget, and [LICRA](https://arxiv.org/abs/2205.15953) learns values for agents limited to k actions.
- **"Only in bandits"** is therefore wrong. What remains new is amortising such a look-ahead into a fogged learned Q, with free stopping and a privileged turn-end payoff.
- **Turn coins.** Randomising the policy over time blocks is a switchback experiment ([Bojinov, Simchi-Levi & Zhao 2023](https://arxiv.org/abs/2009.00148)). Dropping units where both arms coincide is triggered analysis ([Deng & Hu 2015](https://exp-platform.com/dilution/)).

## Repairs, in order

1. Predict n_d and the marginal for each layer, size each audit for 80% power, and state Ĝ_plug's error. Otherwise, drop the depth claim.
2. Forecast by evaluating the rule actually played, with a held-out head under the `own1` tag. Calibrate a deviation margin per layer, and cap deviations per game.
3. Fit layer 0 by Longstaff–Schwartz. For deeper layers, refresh the targets once per decision of the longest chain, and price it.
4. Add a known answer past π's stop: offset −2.5 against −1.5.
5. Rebuild P3 on forked trees, put its audit line at 3 SE or more, and derive the 0.7.
6. Specify and cost the round-1 collect–fit schedule.
7. Strike the section 1 bound and the novelty sentence.
