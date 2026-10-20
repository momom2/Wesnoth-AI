# Round 8, provenance stance: critique

**Verdict: repairable.** No guessed number carries a costly decision. One gate rests on an unmeasured transfer. Two "shape-free" statements hold only under assumptions the text omits. S4's acceptance rule contradicts its own prediction. Checks: `analysis/round8_provenance_critique_checks.py`, with its `.out` beside it.

## 1. Provenance audit

| Number | Class | Check |
|---|---|---|
| Prices $1.912e-5, $1.17e-4, 1.41×; Var(z) 1.0004; residual 0.688, 0.839; 224.6 decisions | M | `round7_provenance_price_tail.out`, `round6_provenance_prices.out`, `round5_simplify_variance.json`, `round6_provenance_cells.out`, `round5_simplify_critique_decision_units.out`: match |
| F(0.35) 16.5, F(1.0) 45.1 | M | `attack_type.py` reread: match. It is parity3's record in own1's units. The transfer moves the magnitude, not the sign. |
| Attack move −0.2247 ± 0.0459, −0.2178 ± 0.0487; LOO RMS 0.83; demotion −0.12 ± 0.08; ending earlier 0.0033 ± 0.0017 | M | `round6_provenance_critique_attack_stage1.out` B, `knob_checks.out` A–B, `round5_provenance_knob_switches.out`: match |
| Per-switch mean ≤ −0.0024 | D | Arithmetic right; the inputs are smooth fits (§2.1) |
| S0 windows [−111, −41] Elo, [−0.640, −0.236] z | D | Redone: match |
| False stops 0.028–0.497; sign-rule ratio 0.283–1.000; E ≥ 0.500 | D | Gaussian path reproduces `round5_synthesis_power` (8.08, 5.71 Elo). Valid in the single-cell model. |
| Threshold: ≤ 0.1 Elo lost, up to 13.7 gained | D | `threshold.out`: match |
| E[T²] = 5.333; within-game cross moment 0 | D | Algebra right, for the label without variates |
| Spread SE 1.27e-6, bound 2.53e-6 | D | Redone: σ²√d/(n√28) |
| Scarf caps 21.0 to 2.7 Elo | D | Formula right. "Checked on 20,000 random laws" has no script or output on disk. |
| Budget $56.18 and stage costs; $3.113e-5; $77.12 | D | `plan.out` reproduces them. The two-round line $4.808e-5 is printed in `round8_simplify_power.out`, not `plan.out`. |
| Census 76.2 ± 1.8 | M (parity3) | Match, but used as own1's smoke prediction (§2.2) |
| Negative control, P 0.04–0.10 | guessed for its use | Copied from `threshold.out`'s Gaussian floor cases, not computed at the linear plant's size |
| w, p, 8 batches, four-pair window, 2-SE pre-test, plant at 10 SE | C | Consequences derived |

## 2. What would make it fail

**2.1 The near-tie bound borrows the slope of the far gaps.** Only three vectors have 0 < a⁺ < 0.35 (0.09, 0.29 and 0.31). All three have end_turn offsets from −2.0 to −3.3. Split a⁺ at 0.35 and the near slope reads −0.45 ± 0.22 per nat (`critique_checks.out` A). The move at 0.35 is then −0.159 ± 0.077. So the per-switch mean is at most −0.0003 at 2 SE (point −0.0096), not −0.0024. The sign holds at 2.06 SE. The "three fits" are three smooth extrapolations of the same far-gap data. Nothing costly rests on this: the √N ceiling dies anyway, since its certification passes with P 0.013–0.039. Still, batch 1's report should predict ≤ −0.0003, and the summary should not call the bound settled.

**2.2 A smoke gate on an unmeasured transfer.** The smoke predicts own1's census at 76.2 ± 1.8, an SE that covers only parity3's sampling. The proposal's own table marks own1's census [U]. No own1 gap record exists; S0(a) was that measurement, and it was dropped. A difference of about 5% between the two networks fails the smoke and sends correct code to debugging. The census should be a reading that restates costs, not a pass/fail item.

**2.3 Scarf's cap is not shape-free.** It bounds E[(E[a|x])⁺] when E[a|x] = c + wᵀx. With a = c + wᵀx + u and u uncorrelated with x, a threshold rule on the fitted index also collects E[u·1(wᵀx > τ)], which v does not bound unless E[u | wᵀx] = 0. So the cap holds for any law of wᵀx only under a linear conditional mean, and S3-null (ii)'s "span exhausted" reading is model-bound.

S2 also pairs each type's upper v̂ with its measured mean, and the cap rises with m. At a measured −0.005 the cap reads 3.3 Elo: exhausted. At that mean's 2-SE upper bound, about −0.004 (SE ≈ 0.0005 after 8 batches at the proposal's assumed 10 attack pairs per side-game), it reads 4.0: not exhausted. Use the upper bound on m, or the welfare curve's upper confidence bound, which is model-free for the rules the head can play.

**2.4 E's half-share guarantee holds only within one cell.** The monotone-likelihood argument needs one noise level. E pools types, and where one type's f is miscalibrated, the pooled top half can keep less than half. Miscalibration is exactly the non-Gaussian regime this round studies: E[a | f] < 0 at f > 0. Example: two types with equal counts, one with mean +0.001 at f 0.001, the other with mean −0.0005 at f 0.002. The switch set sums to +0.0005 a decision, while E keeps −0.0005. Applying E within each type restores the guarantee, since it then sums over types. E's sealed forecast still protects the claim.

**2.5 S4 predicts rejection and accepts on noise.** S4 predicts J ≤ −(1 − ρ̂)Ĝ_k + Σ n μ̂, which is negative whenever ρ̂ < 1 and the disagreements cost. Yet N is accepted iff J ≥ 0. At ρ̂ 0.8 and Ĝ 8 Elo, the bound is −1.6, and noise alone gives P ≈ 0.30 of acceptance (SE 3.07), on an outcome the stage predicted against. By the stage's own logic that reading is a miss of the off-band sign argument, which rests on argmax beating temperature-1 sampling, not on measured flips of a distilled network. State the price the reset may cost and accept against it.

**2.6 Smaller points.**
- "Label variance capped at 5.33" holds for the label without variates. The designed label carries variates, whose second moment has no cap. It is a usable fallback only if both labels are computed.
- The negative control is one binary outcome with up to 10% false alarms and no stated action.

**What survives checking.**
- The proposal's weakest point on transfer across students is small in its model. Choosing τ on students 4–7 and applying it to student 8 loses at most 0.4 Elo against student 8's oracle in all eleven cases (`critique_checks.out` B).
- The toy label is unbiased against exact advantages over eight seeds (z +0.58, C); the recorded run's six same-sign deviations were Monte Carlo error in its forced-arm truth plus chance.

## 3. Novelty

Claim 1 is covered beyond the cited EWM papers (abstracts read):
- Chernozhukov, Demirer, Duflo and Fernández-Val fit a proxy score on an auxiliary sample. With known propensities, they test whether the proxy carries heterogeneity (BLP) and report effects by proxy quantile (GATES). E is the top GATES group. [NBER w24678](https://www.nber.org/system/files/working_papers/w24678/revisions/w24678.rev0.pdf).
- Imai and Li evaluate a scoring rule on experimental data under a budget, the share treated. [arXiv 1905.05389](https://arxiv.org/abs/1905.05389), JASA 2023.
- RATE and TOC curves score prioritisation rules by their top-q effects. [Yadlowsky et al.](https://arxiv.org/abs/2111.07966), JASA 2025.

Keeping the plug-in unless rejected is a classical pre-test estimator; only the setting is new. Claim 2's estimator is the cross-split inner product already cited (Kong & Valiant).

## 4. Complexity

The welfare curve, a cumulative sum over logged labels, earns its place: ≤ 0.1 Elo lost where the plug-in is right, 2–14 Elo gained in the two-point cases (`threshold.out`, 4 batches with the pre-test). The Scarf reading earns little: at m = 0 the cap is 21 Elo, so it closes nothing unless typed means sit well below 0, where it is model-bound (§2.3).

## 5. Strongest part

Rules whose error rates came from the Gaussian model give way to readings valid by construction: the welfare curve is unbiased for every threshold because the propensities are logged. Two rules that fail on inspection go: the interim stop, whose false stops reach 0.50 when only the residual's shape changes, and round 7's ceiling certification, which passes with P 0.013–0.039 even at nonnegative slopes, so it would almost never have fired.

