# Critique of round 4 challenge (forks for consequences, results for valuation)

**Verdict: repairable, on a ruling.** If the user rules out the valuation (issue 1), the stance is broken as a core and survives only as an auxiliary target. As written, its two deciding readouts cannot be read at the planned samples (issues 2 and 3), which fails the no-fire-and-hope rule. The repairs cost little.

## Fatal issues

**1. The valuation is a Texel-tuned material evaluation.** g = w·φ weighs hand-chosen rule quantities (units per type, HP and XP sums, leaders, villages, gold), fitted to game results. That is how classical engines tune evaluations: Texel's method fits about 400 parameters to the results of 8.8 million positions by logistic regression ([chessprogramming](https://www.chessprogramming.org/Texel%27s_Tuning_Method)). Every A_f is this evaluation applied to a predicted board. On 2026-10-19 the user rejected GOFAI, and with it reaching for hand-built structure because learned evaluators failed. §2.3 calls g "the level those critics failed to beat", which is that move. With embedding cumulants, g becomes a linear critic on frozen features fitted on results, the current design's fallback F. The proposal asks for a ruling on the result fit, not on the hand-chosen basis. The closest record, the material look-ahead (−8.7 ± 12.3, "mostly rewarded recruiting"), gets no prediction for the recruit groups.

**2. S1's deciding readout cannot be measured.** The kill line is "value-unit r under 0.3". Per pair, the true contrasts have variance s² ≈ 1–2.25×10⁻⁴ against label noise 0.075, a reliability of 0.0013–0.003. A corrected r needs the true-contrast variance in its denominator. Estimating it from duplicate forks by covariance (SE ≈ 0.075/√n) to within 25% takes 2 to 9 million duplicated pairs, and S1 has 300,000 points. The comparison values (0.40, 0.52–0.53) come from whole turns at reliability 0.51–0.58, which is a different estimand. Two things can be read: the calibration slope of realised contrasts on predicted ones, with SE √(0.075/(n·Var(pred))) ≈ 0.15–0.22 at 60,000 held-out pairs if Var(pred) ≈ (0.5s)², and the per-entity r, whose reliability per pair is 0.02–0.15. Kill on the slope, with a derived range.

**3. The factorial cannot read its own prior.** The prior for the 9 action groups is 0 to +35 Elo in total, central 8, so about 1 Elo per group. A held-out contrast has SE 4.4 Elo, and adoption needs its lower 2-SE bound above 0. A 1-Elo group is therefore adopted with probability 1 − Φ(2 − 1/4.4) ≈ 0.04, against 0.023 under the null. Over 9 groups, P(at least one adopted) ≈ 0.29 if the mechanism works at the central prior and 0.19 if it does nothing, so S3's kill cannot tell the two apart. Two repairs:
- Pool into one action factor and one stop factor at two levels. The variance becomes 2σ²/n, the held-out SE 3.1 Elo at 40,000 games, and the central 8 sits at 2.6 SE (power about 0.7).
- Buy the games. A held-out SE of 1 Elo per contrast takes about 19 × 40,000 games, about $38.

Keep the 66 interactions out of the forecast: at 2 SE, 66 null readings give about 3 false positives. Caps "count as losses for both sides", so the response is not antisymmetric: a side-A-minus-side-B regression misreads levels that change the cap rate, which stop levels do, and the yardstick drops caps. Model the decisive result and the cap separately.

**4. The planted checks bypass the paths they certify.**
- **P-fact.** The null groups pass even if the decode never reads λ, and the +12 gold plant runs through its own code, not through λ. Plant through λ instead: a group whose level switches a known share of decisions to an action planted to be better.
- **P-cons.** The plant ignores the dice and the reply, so it passes with recorded dice, an idle opponent, labels at H2 instead of H1, or wrong hidden-unit cells. Plant through the reply's combat (a changed hit chance against units on even hexes during the opponent's turn) and check hidden cells against the true state. The 2.0 HP ignores the unit's later moves and deaths; state the enumerated expectation.
- **P-val** certifies w_gold only. The confounded coefficients are the unit cells: on-policy, material at H tracks the position quality that φ omits, so g overprices the material a candidate gains. A randomised windfall cannot show this.
- **S−1(b)** reads between-position R² at a resolution of 0.03 of Var(y). The within-position value that decisions act on is about s²/Var(y) ≈ 10⁻⁴ of it. The test also fails on between-position omissions that do not matter for decisions, so it certifies the factorisation in neither direction.

## Wrong or overstated math

- **"Its minimiser is E[y | φ(x_H)]".** For the linear class it is the best linear predictor on on-policy boards. z saturates in material, so g is most wrong in lopsided positions, and wrong differently on sibling boards: a second, unnamed approximation beside board sufficiency.
- **"15 to 60 times" the information per entity does not reach decisions.** With identical regressors, least squares on the vector then w equals least squares on w·Y ([Zellner 1962](https://doi.org/10.1080/01621459.1962.10480664)): the extra components add nothing to the scalar the decode uses. In a network they help only through the shared representation, KataGo's evidence (190 Elo at 2.5G queries, realised final-board targets, [Wu 2019](https://arxiv.org/abs/1902.10565)): an inductive-bias bet, not a χ² multiple.
- **"2.3 times per dollar, unbiased".** The labels are unbiased for E[g(φ(x_H)) | h, b], not for the decision's value. The bias moves from C at the window's end to material at H, and the record has privileged critics about 2 SE above material.
- **The evidence that reads after play beat static reads** ranks turns played with their own dice, a comparison the user ruled invalid. At H1 the record is against it: `obs8`'s value at read 1, over 8 playouts, ranked at 0.266, level with the static graders (0.27). Material after a turn ranks well partly because it reads that turn's dice, which per-decision forks average away.
- **A free reduction is missed.** For linear φ, the exact luck correction of the labels is cheap and has mean zero component by component: subtract Σ_k [φ(x_k) − Σ_o p_k(o) φ(x_k^o)] over the window's fights.
- **S−1(a) predicts g_0 at 0.78–0.84.** That is better than every material column on record (0.865–0.94) and than the seven-column fit (0.829 on `gen_own1_selfplay`). The range is not derived, and it feeds σ² ≈ 0.80.

## Costs

The prices follow from the seed's throughputs: 300,000 points × 154 forwards ≈ $2.6, and S1's training (about 17 million examples an epoch at 1,500 a second) costs about $1.3 an epoch. The saving does not hold: "the central bet decided at $8" rests on issue 2's readout.

## Novelty

- **Claim 1** is Direct Future Prediction, with forked siblings in place of ε-greedy experience. DFP predicts measurement changes per action at offsets 1 to 32 and acts by argmax gᵀF ([ar5iv](https://ar5iv.labs.arxiv.org/html/1611.01779)).
- **Claim 2** is successor features' reward regression ([Barreto et al.](https://arxiv.org/abs/1606.05312)) plus Texel tuning. Only the AIVAT-corrected target is new.
- **Claim 3.** Tuning parameters on self-play win rate has precedent: AlphaGo's hyperparameters, by Bayesian optimisation, from 50% to 66.5% ([Chen et al. 2018](https://arxiv.org/abs/1812.06855)). The current design rejected outcome tuning of one decode scalar (piKL λ), and 36 contrasts do the same at scale. Decode offsets per action kind measured flat (+1.7 ± 12.3).

## Strongest part

The consequence labels: one decision per fork, fresh dice, both sides re-deciding after every roll; a fogged regression onto the information-state expectation, with no critic and no strategy fusion; $8.6e-6 a point, per-entity reliability high enough to measure, and labels that survive any change of valuation. As an auxiliary target, with KataGo's ablation behind it, they are worth keeping whichever core wins.
