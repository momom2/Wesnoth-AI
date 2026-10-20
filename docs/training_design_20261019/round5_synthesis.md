# Round 5 synthesis

## Verdict

The design changed in substance this round.

1. **New core mechanism.** In rounds 1–4, decisions were informed by a privileged critic, gated by κ, distilled into a student and calibrated by coins. Now they are informed by the coins directly. Each near-tie coin's result gives an exactly unbiased label of one decision. A ridge second actor head pools the labels, and a match certifies the result.
2. **Fatal flaw fixed.**
   - Every gain path of the round-4 design passed through κ. κ had no measurement, and the record's critics did not improve in ranking with data.
   - The new core's correctness depends on no unknown: only its gain does, through the near-tie spread v, and a derived floor states what a null means.
3. **Guessed or untested numbers replaced by measurements and derivations that changed decisions.**
   - **The stop.** Its mean is measured at zero, so the stopping machinery is removed and end-against-act becomes a pair.
   - **Matches.** Critic-free side-swapped pairs keep 0.700, so the critic leaves the matches.
   - **Learned luck corrector.** Its derived cost line keeps the identity critic out of round-1 labels.
   - **Forecasting.** A match is 17 times more informative per dollar than a forecast batch, so no frozen forecast half.
   - **Band count.** Corrected from 62 to 76 per side-game.
4. **Stage order changed.**
   - No critic is trained before the first coin round.
   - The critic's sibling information (κ_T) is read only after a round shows the frozen features are the binding limit.
   - The look-ahead player is the branch for a "no" on ruling (a).

## What changed, and why

### From simplify (critique: repairable). Its core is now the design's core

**Kept.** These are the mechanisms whose guarantees do not depend on an unknown:
- the label T_i and its proof, re-derived by its critique;
- the flat coin;
- end_turn as an ordinary pair;
- the closed-form ridge from per-game sufficient statistics;
- critic-free corrected matches.

**Repaired as its critique asked:**
- **Intercepts and the calibrated rule.** Each pair type gets an unpenalised intercept, and the rule switches where μ̂ + β̂(f − μ̂) > 0. The sign rule loses 7 to 14 Elo at μ = −0.0018 (`analysis/round5_simplify_critique_rule.out`).
- **Model-free forecast.** Held-out labels are summed over the switched decisions; the plug forecast is no longer used alone.
- **Band count in the decision unit of its share:** 76.1 per side-game, not 62.
- **Student features.** q(actor) ⊙ k(hex) replaces the hex token, so that moves of the same unit differ. S0 counts the runner-ups' composition.
- **Plants booked into z itself.** They now pass through the real label code, with SEs from the real noise.
- **Control-variate invariance test.** Every control variate is checked with and without, and a deliberately post-coin variate serves as a negative control showing the test can fail.
- **P-distill.** Its known answer is the decision count (9.63 per side-turn). Agreement is predicted from the measured 58% closure, stated as indicative.
- **S0's flip line.** It is now a diagnostic for a forecast miss, not a $7 trigger.
- **Critic re-entry.** It no longer refits labels from replays, which cost $17.8. The critic enters as one ridge column, computed at generation, after a κ_T reading sized blind.
- **Weak readouts removed.** "β's lower bound ≤ 0 certifies the spread below the floor" and the exploitation check are gone; the match-only floor states what a null bounds.

### From provenance (critique: repairable)

**Kept:**
- the critic-free yardstick: pairs keep 0.700 ± 0.009 by the critique's direct estimate, and the setup baseline plus columns keep 0.747;
- the record audit;
- Match C as a record, so no match is run for it;
- the performance-difference framing;
- blind SE computations, now applied to the critic column's resolution;
- fresh-seed adoption.

**Its three fatal points are now moot or met:**
- **F1** (no planted decision signal before the gate): P-coin runs through the real label path in S1's first batch, and nothing downstream runs if it fails.
- **F2** (S1b's unassigned outcome): the κ gate is gone.
- **F3** (known answers for the wrong estimator): the stop machinery and Matches A and C are gone.

**Also met:**
- The critique's point that calibration on random alternatives does not cover the switches is answered by construction: the forecast sums labels at exactly the decisions the rule switches.
- The adoption check uses the paired correction term (SE 1.56 Elo), not a root-sum-square.

**Not kept:**
- κ as the hub;
- Longstaff–Schwartz stopping;
- Match A;
- the coin schedule scaled from a trial cost, replaced by the flat coin read per margin bin.

### From challenge (critique: repairable)

**Kept:**
- certification by play before anything is distilled;
- the per-switch advantage as the common unit with the record's look-ahead matches (−0.0021 and −0.0008 per flip);
- the placebo principle.

The look-ahead player becomes **branch L**, for a "no" on ruling (a), with the critique's repairs:
- a placebo arm per critic (values permuted across each decision's candidates);
- end_turn left to the prior in the fair arm;
- a god-view material arm that separates the world from the critic;
- port checks with known answers: the decode at c = 0, and the recorded material match's per-flip reading.

**Not kept as the main route:**
- Its critics were trained on argmax games that contain no sibling contrasts, and they value an older mixture of policies.
- Its blind σ admits 93 noise flips per side-game.
- Its data ladder tests extrapolation only.

### From invent (critique: repairable)

**Kept:**
- The measured rejection of keyed coupling between runs. Its critique called this the strongest part: the margin density is nearly flat near 0, and keyed tie-breaking at T = 0.25 cuts flips by only 31% while deviating 40.7 times per side-game.
- The turn-over rule, as counts logged in S0's replay. It is deferred as a candidate for removing the offset.
- Law supervision, deferred to the first trunk training.

**Not kept:**
- S0(j)'s early adoption of a hand-set rule, whose readout is mostly predictable;
- the sibling-transfer probe, which tests a deterministic local function that κ does not depend on;
- the exact-truth plant, exact only for the immediate term;
- law heads on three networks. With a frozen trunk and a linear student, a linear law readout of the same tokens adds no span.

### New this round (synthesis)

1. **The learned luck corrector's cost line.** Script: `analysis/round5_synthesis_outcomes.py`.
   - Measured on `own1`'s 800 recorded games: 449 decisions and 110 fights per game, and 11.9 outcomes per fight on the counter-weapon subsample (an upward-biased sample).
   - A learned Q at fight children multiplies a generation game's forwards by 2.0–3.9.
   - It therefore pays in labels only if it keeps under 0.19–0.37 of the variance, against the columns' 0.732.
   - This closes, for round 1, the identity critic's role as a label corrector. Its role in matches (multiple about 1.3) stays open, measured in S0 with own1's value head for $0.04.
2. **The repaired power analysis.** Script: `analysis/round5_synthesis_power.py`. It uses the band count of 76.1, the calibrated rule and the progressive protocol, reusing the simplify model.
   - At $19.8 and d_eff 776, a match detects a near-tie spread of sd 0.0020 (μ 0) to 0.0038 (μ −0.0018), at 7.5 Elo with 80% power.
   - The progressive forecast's SE is 3.3–4.7 Elo.
   - The early stop after batch 4 fires with probability 0.15 at a μ = 0 null, and with probability ≈ 1 when μ < 0 and there is no spread.
   - p = 0.5 lowers the floors by only 5–10%.
   - A critic column needs κ·sd(X) ≥ 0.0005 per band decision to be worth 2 match SEs. 200,000 coin games read a column worth 10.7 Elo.
3. **No frozen forecast half.** One 100,000-game batch read model-free has SE 9.5 Elo for $2.47; the 12,800-game match has SE 2.65 for $1.90, which is 17 times the information per dollar. Every batch therefore trains, and the progressive readings provide the sealed forecast.
4. **The conservative mixture match.** A forecast miss is answered by a match of π_½ (each switch taken with probability ½). Together with the main match, it identifies the second-order term, G2 = 2(J(1) − 2J(½)), and the step size α* = −G1/(2G2). It replaces simplify's threshold rule, which rested on the Gaussian model, with conservative policy iteration's step ([Kakade & Langford 2002](https://mlanthology.org/icml/2002/kakade2002icml-approximately/)).
5. **A correction of the group's record.** Round 4 called the stop "the likeliest gain" (`round4_synthesis.md`). The decode-knob records measure its mean at the margin at about zero (+0.0002 ± 0.0010 per added decision; Match C −0.035 ± 0.035), so that statement was wrong.

## Open questions

1. **Rulings for the user.**
   - (a) Are coin-label regressions allowed? If not, branch L.
   - (b) Are the hand-written luck columns allowed as control variates?
   - (c) Adoption by corrected matches below the PURE yardstick's resolution.
   - (d) The offset stays inside π until a distilled network at offset 0 is non-inferior.
2. **The spread v.** How large is the near-tie spread of advantages that the frozen features express linearly, and what is the mean μ per pair type? S1 measures both, and nothing before it relies on them.
3. **d_eff(λ)** of the q ⊙ k feature set. S0 measures the spectrum for free.
4. **Second-order loss.** How much is lost at 19–38 switches per side-game, and is the memory pushed off its distribution by switching?
5. **Coin cost by margin.** If it is small, p = 0.5 and w = 0.7 become cheap.
6. **Logging overhead.** If it exceeds about 15%, round 3 is cut.
7. **Distillation.** Does own-trajectory distillation preserve the switches' gain, and does the factored head's end_turn bias return in the distilled network?
8. **The critic.** Do privileged critic contrasts carry near-tie information (κ_T)? This has been open since round 1, and is now read only when it is the next lever.
9. **Compounding, exploitability, human level.** All unmeasured; nothing rests on them.

## Research targets for round 6

- **Beyond first order.** Bounds on the performance difference past first order for many simultaneous greedy switches:
  - safe policy iteration ([Pirotta et al. 2013](https://proceedings.mlr.press/v28/pirotta13.html));
  - the surrogate bound of constrained policy optimisation ([Achiam et al. 2017](https://arxiv.org/abs/1705.10528));
  - CPI's step-size rule.

  Aim: derive α* or a switch cap from the measured coin-label variances, before the mixture match.
- **Excursion effects.** The excursion effect against the policy-level effect in micro-randomized trials (Qian 2025 in the original, not secondary sources; Boruvka et al. 2018; DR-WCLS), for the gap between Δ under π̃ and under π at p = 0.25.
- **What the frozen features can express.** Is the advantage between a policy network's near-tie candidates linearly decodable from its token outputs? Look for evidence from probes of game networks, and on compatible features ([Kakade 2001](https://papers.nips.cc/paper/2073-a-natural-policy-gradient)).
- **Distilling an improved policy.** Agreement after one pass on a sharp argmax target ([Rusu et al. 2016](https://arxiv.org/abs/1511.06295), policy distillation; [Czarnecki et al. 2019](https://arxiv.org/abs/1902.02186), "Distilling policy distillation"). These links were not re-read this round.
- **The end_turn bias after distillation.** Label bias and the brevity problem ([Murray & Chiang 2018](https://arxiv.org/abs/1808.10006)), and whether a separate turn-over head trained on own trajectories removes the offset without tuning.
- **SPSA in practice.** Fishtest's SPSA tuning: games used per parameter at given effect sizes, as an empirical comparator for the floor table.
- **Kernels.** A price for batch-invariant inference kernels, the only remedy for generation-numerics flips at large gaps.

## Unknowns that remain

| Unknown | Measured where | What relies on it |
|---|---|---|
| Near-tie spread v; mean μ per pair type | S1's progressive fits | The round's gain, and nothing before S1. Floors are stated for each case. |
| d_eff(λ) | S0(a) feature spectrum | Floor between the d_eff 200 and 776 cases |
| E[R²] with the value-head baseline | S0(a) | σ² (0.732 until then) |
| `own1`'s band census and runner-up composition | S0(a) | Band count (76.1, derived from `parity3`) |
| Value head's kept ratio as a corrector, and the support census | S0(b) | The labels' and matches' corrector |
| Logging overhead per coin game | S0(c) | S1's price; whether round 3 fits |
| Coin cost per margin bin | S1, batch 1 | Round 2's p and w |
| Second-order ratio (match against forecast) | S2; mixture match | Round 2's step size and forecast |
| Distillation closure on a policy target | S4's P-distill | Band reset |
| κ_T | Coin games, only after S3-null or a saturated band | The critic column |
| Generation-numerics flips at G ≥ w | S0(a) | Diagnosis only |
| Compounding, exploitability, human level | Each round's match; not measured; not measured | The $1,000 case's argument |
