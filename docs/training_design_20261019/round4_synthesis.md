# Round 4 synthesis

## Verdict

The design changed in substance this round, though less than in rounds 1–3. The frame, the stage order and the instruments stand: the exact luck-and-coin identity, the privileged critic, Longstaff–Schwartz stopping, the fusion-free fogged student, corrected matches and the forecast ladder. Five changes go beyond wording:

1. **The critic's objective changed.** It is the core of S1, the stage where the design can die. The λ family is replaced by two terms:
   - **Game paths:** harden's multistep loss, with a frozen cross-fold corrector.
   - **Windows:** round 3's exact within-window residual, kept against harden's frozen window target.

   Together they are one fixed objective. The change fixes three faults: λ < 1 was an oblique projection; λ = 1 weighted late events by the number of states before them; and harden's frozen window target lost the convergence argument.
2. **A new data channel.** Luck-corrected consequence heads on the critic, gradient-gated: challenge's surviving part, in the role its critique proposed.
3. **A new student arm.** Law-factored attacks (invent), restricted to the prior's top four and repaired. It runs only if the user rules it within "learned only".
4. **A planted check fixed.** P1 as a level plant passed with the corrections dropped (harden critique 1). It is now a variance plant, and P-cons rides on the same games.
5. **A record error of mine corrected.** `prior_gaps.json` names `training/checkpoints/parity3.pt` and player `cand64`: the 100 side-games and the 98.2% replay agreement are `parity3`'s decisions in its match against `parity2`. The round-3 design and synthesis labelled them `parity2`'s, and the round-3 synthesis listed that wrong label as a correction. I checked the file this round.

## What changed, and why

### From harden (critique: repairable)

**Kept:**
- **The multistep loss.** The critique re-derived (1) and (2), and checked that MDBDP's bound (Theorem 3.1) lacks the factor N that the one-step DBDP bound carries (Theorem 3.3).
- **The band sized from the record.** The critique reproduced its counts.
- **The cross-critic score for the stop rule,** a double estimator.
- **The action prior with the posterior threshold:** 0 to +33 Elo, about 0 at the low corner. The stop is round 1's likeliest gain.
- **Window placement by p*(G).**
- **The record label,** confirmed above.

**Repaired as the critique asked:**
- **P1** is a variance plant. A mean-zero lottery with an asymmetric logged law sits at every fight and coin and is booked into the result. The held-out residual is regressed on the lottery sum: correct code reads 0, dropped corrections 1, sign-flipped corrections 2, with SE about 0.02 at 5,000 games.
- **The event-class split** reports each class's removed variance. A class's residual share is not identified.
- **The pilot no longer tests λ.** ρ_C weights each event once, as the multistep loss does, so a λ = 1 arm could not refute the multistep loss. λ = 1 is dropped, not piloted. The pilot now carries the consequence heads.
- **Propensities and pairs are logged at generation.** I2 checks them exactly, and nothing recomputes a margin.
- **Sampling.** Events and states are sampled uniformly. Sampling "by gradient weight" broke the cross terms.
- **Window placement is floored** at a quarter of the mean, so decisions with G ≥ w get windows too.
- **The band cost model** has an intercept. The taper becomes ½(1 − (G + c0/λ)/w′)₊, which follows from the same KKT condition. w's range is [0.35, 0.7]: at λ = 0.01 the cap gives about 0.38. The λ anchor's bias direction is stated as unknown, since truncation and saturation pull opposite ways.
- **Costs** include the multistep loss's children forwards, about $3.3 over S1.

**Not kept:**
- **Frozen cross-fold window targets.** Across passes they are fitted value iteration at off-policy siblings, with no convergence argument ([Tsitsiklis & Van Roy 1997](https://doi.org/10.1109/9.580874)).
- **The horizon rung.** It reads C′'s own Bellman residuals, not the distance from truth, and its SE (about 0.15) lets it fire only below about −0.6. The ladder's Ĝ_plug(2) against the match reads the same bias at the policy level.
- **I12's tolerance for numerics flips.** Logging the propensities replaces it.

### From challenge (critique: repairable on a ruling)

**Kept:** the consequence labels, which the critique called the strongest part: one decision per fork, fresh dice, both sides re-deciding, no critic in the label. They become auxiliary heads on the privileged critic, with three additions:
- **The critique's "free reduction":** an exact component-wise luck correction of the labels over the fights in between, which is CPU arithmetic on outcome keys.
- **The repaired P-cons.** It is planted through the reply's combat (strikes at units on even hexes hit 0.2 more often) and read on fork truth. The hidden-unit components are checked against the true state (I11).
- **No information multiple.** By Zellner, a vector regression with identical regressors adds nothing to the scalar the decode uses. The heads are an inductive-bias bet, supported by KataGo's ablation (−190 Elo, a 1.65× training-time factor, verified in [Table 2](https://arxiv.org/html/1902.10565v5)).

**Not kept:**
- **The valuation g = w·φ fitted on results.** It is a Texel-tuned material evaluation over a hand-chosen basis, against the learned-only ruling; with embedding cumulants it is fallback F.
- **The factorial of decode-trust groups.** At its own prior it cannot tell working from null: P(any adoption) is 0.29 against 0.19. It is also outcome tuning of decode scalars at scale.
- **The claims of "2.3 times per dollar, unbiased" and "15–60 times per entity".** The first holds for the expected board, not for the decision's value. The second fails by Zellner.
- **The board-sufficiency test S−1(b).** It resolves 0.03 of Var(y), against a within-position signal near 10⁻⁴.
- **The "reads after play beat static reads" evidence.** It ranks turns played with their own dice. At H1 the record shows a tie (0.266 against 0.27).

### From invent (critique: repairable)

**Kept:**
- **The exact factorisation,** as an S2 A/B on identical targets: with w set to the critic's values, the factored score is Â_C plus a per-state constant (I9).
- **The law census in S−1.**

**Repaired:**
- **Every outcome key is trained, weighted by p,** and terminal keys take the result exactly. A leader's 1.5% death decides the risky attacks.
- **The stream reward booked at the transition** enters the target.
- **The readouts** are within-top-four R² and argmax regret against Â_C, and the outcome head's within-fight R² over a key-only baseline. R² over all legal attacks measured coverage and easy variance.
- **P-law depends on h.** It is a bonus for ranged kills of units on villages. Key-only and token-shuffled heads must read ≤ 0.3. It uses 64 playouts per candidate, and its band is set at 2 SE of the measured stream variance.
- **Round 1 is restricted to top-four attacks.** The arm then needs no off-path enumeration, because its targets are the children that the scalar student's targets already evaluate; P-law rides on P2's games. It costs about $1.
- **The claimed gain is approximation only.** The observation carries no hit chance or kill probability; I checked the unit features in `rust/wesnoth_core/src/encode.rs`. Factoring adds decision information only for weapon choice, which S−1 counts.
- **The user ruling is flagged.** The factorisation computes the law in play, and Wesnoth's default AI uses the same decomposition with hand-written values.

**Not kept:**
- **Option tokens:** their values are out-of-distribution hypotheticals.
- **Same-unit coins and outside-top-four candidates in round 1:** calibration on random alternatives does not cover the argmax switches. They return in round 2 if the arm wins, with coins restricted to outside-top-four alternatives.
- **The attacks-only match.**
- **The +14 to +29 Elo prior.** It assumed q = 0.6 and a spread s 1.5–2 times the design's, and ignored the posterior gate. The honest band is inside the actions-only 0 to +33.

### New this round (synthesis)

- **One fixed objective.** The multistep game loss's expected gradient does not depend on C′, and the window residual is a fixed function of C. So the critic descends one objective whatever the pass, and fitted value iteration, harden's window form, never enters. Over two side-turns the window multiplier's variance is about 0.03, so the global loss's stall does not arise there.
- **Supporting and contradicting evidence on residual forms:**
  - [Zhang, Boehmer & Whiteson 2020](https://arxiv.org/abs/1905.01072) find that residual algorithms handle model-based planning's distribution mismatch better than TD(k) (supporting).
  - Baird's slow convergence stands (against).
- **The cross-fit leak removed with two folds.** Pass 0's corrector is the seven luck columns refitted on the luck walk's 18,800 recorded games, outside both folds. Pass 1's corrector is the other fold's pass-0 critic, which has therefore never seen this fold. The leak the round-3 critique found needed three folds otherwise.
- **Gradient-similarity gating** of the consequence heads ([Du et al. 2018](https://arxiv.org/abs/1812.02224)) replaces an arbitrary gradient cap. The auxiliary gradient enters the trunk only while its cosine with the value gradient is positive, which keeps convergence to the value objective's critical points. That is why adoption on non-inferiority is defensible: the heads cannot move the objective, only which minimiser is reached.
- **The decision χ², recounted.**
  - Weak coins run at a third of round 2's density, about two deviations per side-game, so they carry 50–110 χ², not 140–320.
  - Band coins carry 360–1,360, from harden's information ratio times the band's share of the shift budget.
  - Windows carry 460–1,040.
- **Luck steering.** Tilting dice toward contested positions (invent's deferred idea) is valid for the window residuals and local residuals, whose zero set ignores the path law. It is not valid for the multistep target, which a tilted law q biases by Σ_j (q_j − p_j)(V − C′) at each later event. It stays deferred, now with the reason.
- **Budget.**

  | Path | Cost |
  |---|---|
  | S−1 to S3, round 2 and S5 | about $60 |
  | Adding round 3 | $72 |

  - The S1 total of $25 includes the multistep loss's children ($3.3), backward positions ($0.8), P1 with P-cons ($2.5), the pilot arm ($1.5) and the gating's extra backward pass ($1.2).
  - The remaining $13 covers the kernels with a partial rerun, or an S1 rerun in place of round 3, at the $85 edge.

## Open questions

1. **ρ_C.** It still has no first-principles value. The multistep loss repairs weighting and gradient noise, not capacity. If the record's critics failed for features, S1 dies at the same line and fallback F follows.
2. **Consequence heads.** Do they raise κ_move? The pilot resolves only a pooled κ gain above about 0.15, so adoption rests on non-inferiority plus KataGo's evidence, which comes from a different scale and from targets on the final board.
3. **The factored head.** Does exactness beat the scalar head's approximation of fight laws within the top four? The A/B answers it well (held-out R² on identical targets), but only after the user's ruling.
4. **Windows.** How fast does the residual objective converge at 100,000 windows? Their weight is equal per state, set, not derived.
5. **The band.**
   - Are the prior's errors at ties, σ_η²(G), as large as elsewhere?
   - How are runner-ups composed: what share are same-unit adjacent hexes?
   - What are λ and c0?
   - S−1 and S0 answer the second and third; only S1's κ answers the first.
6. **Numerics.** Replay against the record (S−1) and generation against eval (S0), at 98%. Failure costs about $7.
7. **Unchanged:** drift under 43 band deviations per side-game; the fog ceiling; compounding under zero-sum play and fog; exploitability; Match C's true value.
8. **User rulings.**
   - (a) κ, β, fallback F and P-coin regress outcomes on coins: are they covered by the killed "outcome policy gradient"?
   - (b) Is computing an attack's outcome law in play, and summing learned values over it, within "learned only"?

## Research targets for round 5

- **The mixed objective.** Monte Carlo regression with control variates on paths, plus residual gradient on short exact-model windows. Look for theory on how their minimisers combine under misspecification, and on residual minimisation's bias toward smooth solutions at short horizons:
  - Scherrer 2010;
  - Zhang, Boehmer & Whiteson 2020 and its bidirectional target network;
  - gradient-TD methods as the convergent alternative (Sutton et al. 2009).
- **Power for critic comparisons.** Find a decision-information readout with more power than κ for paired critic arms, for example regression adjustment of κ's coin truth by window contrasts (a CUPED-style control variate), or common-coin paired designs. The pilot's resolution of about 0.15 is the bottleneck for every future critic factor.
- **Auxiliary weighting at scale.** Evidence on gradient-cosine gating beyond Du et al.'s small benchmarks (alternatives: GradNorm, PCGrad), KataGo's auxiliary loss weights, and whether luck-corrected auxiliary targets appear anywhere in the literature.
- **Precedents for ruling (b).** TD-Gammon's one-ply expectimax, DeepStack's chance enumeration, Stochastic MuZero's chance codes, and the default AI. Also price the alternative the user may prefer: law tokens in the observation (what the attack dialog shows), at the first trunk training, as a new observation epoch.
- **Compounding.** Per-round decay of approximate policy iteration with rollout-calibrated or value-expanded critics in long stochastic games: Tesauro & Galperin, CBMPI on Tetris, gnubg's iterated rollout training. Harden found no per-round figure.
- **Coin design with costs.** Li & Owen's tie-breaker optimum under a per-deviation cost with an intercept, and adaptive propensities once S0 has measured λ and c0.
