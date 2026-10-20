# Round 8 synthesis

## Verdict

The design changed in substance this round, but not at its core. The coin label, the closed-form head on the frozen network and adoption only by match are round 5's. No proposal replaced the core, and no record moved the main unknown, the near-tie spread v. What changed, by the criteria the round is judged on:

1. **Decision rules that rested on an unmeasured shape were replaced by rules valid for any shape.** Four rules got their error rates from round 5's Gaussian model of the near-tie advantage. Varying only the shape breaks two of them, the third had no shape-free form, and the fourth survives in a narrower form:
   - **The interim stop.** False stops reach 0.497 when a minority of decisions carries the gain.
   - **The sign rule.** It keeps 0.283–0.589 of the best threshold's gain under two-point shapes. It is replaced by a threshold tested on held-out labels.
   - **The reading of a null.** It becomes an unbiased cross-batch spread estimator.
   - **E** survives within each pair type.
2. **Stage order changed:**
   - no interim match;
   - S0 gains a confirmation of `own2` as the base and ends with the coin smoke;
   - round 3 is adopted against the pre-reset incumbent;
   - Branch S runs before S1 if its rulings and the optimizer discussion come first.
3. **Three defects fixed:**
   - **E could not be computed from the stated logs:** f is now logged per band decision, and x is stored on a 1,000-game subsample.
   - **The head's swap inside the match had no check:** I-match is added.
   - **S4 accepted N on an outcome its own prediction ruled out:** a 2.8-Elo deficit passed with P 0.18.
4. **A choice replaced by a derivation:** Branch S's top setting is +200 by a coverage rule, not +253. At +253, one player supplies 0.605 of the sides.
5. **Ruling (a) restated under its real name.** The head's fit is one natural policy-gradient step for a coin policy, and the seed lists outcome-driven policy gradient as killed.

Measured against the user's standard, the round removed the last decision rules whose error rates came from an assumed shape. It added a check on every batch that sees what the coin plant cannot (the result's orientation and join). It also added a cheap, independent route to a better base. The spread v is still unmeasured, and a null round 1 costs $17.66 (`analysis/round8_synthesis_power.py` §1).

## What changed, and why

### From provenance (critique: repairable)

**Kept:**
- **Removing the interim stop.** In the design's own model, varying only the residual's shape gives false stops up to 0.497 at negative typed means (`round8_provenance_interim_shapes.out`). The simplify critique supplies a model-free reason as well (below).
- **The tested threshold.** The welfare curve W_j(τ) is unbiased for every τ because the propensities are logged. The 2-SE pre-test keeps the plug-in cut unless the labels reject it.
  - Where the plug-in is right (Gaussian, Laplace), the cost is ≤ 0.1 Elo.
  - In two-point shapes it gains 1.9–13.7 Elo at 4 batches (`round8_provenance_threshold.out`).
  - Choosing τ on students 4–7 for student 8 costs ≤ 0.4 Elo (`round8_provenance_critique_checks.out` B).
  - The critique judged it worth its place.
- **The cross-batch spread estimator.** It reads v per type without a shape assumption. After 8 batches a null bounds v ≤ 2.53e-6 (sd 0.0016) (`round8_provenance_spread_estimator.out`).
- **The exact second moment of the variate-free label,** 5.333. It gives every SE a worst case.
- **The S0 known answer's window,** now the union of two fits: [−111, −41] Elo and [−0.640, −0.236] z. Its leave-one-out RMS is 0.83.
- **Removing the √N ceiling.** Its certification passes with P 0.013–0.039.

**Repaired:**
- **The attack-type bound is reported, not settled.** A hinge at 0.35 gives ≤ −0.0003 at 2 SE (point −0.0096), against the stance's −0.0024. The stance's three fits were smooth extrapolations of far-gap data. It is a report, not a gate.
- **The census is a reading, not a pass/fail item.** The 76.2 ± 1.8 covers only `parity3`'s sampling.
- **E is applied within each pair type.** The half-share guarantee holds within one noise level. Pooled types can keep less (the critique's two-type example).
- **S4's acceptance.** The stage predicted J ≤ −(1 − ρ̂)Ĝ + Σ n μ̂, yet accepted N on J ≥ 0. Now N becomes the next band's generator on the distillation check alone, and round 3 is adopted against π_k (two networks). A lossy reset therefore cannot reach the adopted player. The S5 match predicts J_S4 + Ĝ_pf, SE 5.53, miss line 12.7 Elo (`round8_synthesis_power.out` §3).

**Not kept:**
- **Scarf's cap as an exhaustion rule.** It bounds the gain only under a linear conditional mean. At m = 0 it caps at 21 Elo, so it closes nothing. v̂ with its upper bound carries the reading of a null instead.
- **The threshold plant's negative control.** It is one binary outcome with up to 10% false alarms and no stated action, and its probability was copied from other cases.
- **Ruling (f) on the threshold.** The threshold is a statistical choice inside ruling (a), and (a) now names it.

### From simplify (critique: repairable)

**Kept:**
- **No interim, for a model-free reason.** The interim and the adoption match are independent games, so a stop can only lower power, and at a null it fires with P ½.
  - It saves $4.42 at a null for 0.17 of power at the μ −0.0018 floor.
  - The budget does not bind.
  - If the user values dollars over power, the critique's line at −0.5 SE ($2.45 for 0.08) is the fallback, not 0 (`round8_simplify_critique_checks.out` §1).
- **γV and the census replay removed.** They change cost, not guarantees, and no record holds V on `own1`'s games.
- **The coin plant read on the smoke as well** (SE 0.0387), so gross label-path errors stop the run before 798,000 games are paid for.
- **The merged invariants** I-coin and I-log.

**Repaired:**
- **E's logs.** f_i is logged in place of the switch bit, and x is stored for a fixed 1,000-game subsample of each batch (0.23 GB, `round8_synthesis_power.out` §5). Each forecast student's half-quantile is fixed on the previous batch's subsample, so E's sealed forecast stays out of sample.
- **I-match.** On the match's head seat, f and the switch are logged at every band decision. Every switch must have played a1, and f must recompute from stored x. The switch rate is reported beside the forecast's, not gated: the opponent is uncoined in the match, so the state distribution differs.
- **Plant lines at 3 SE.** Four readings at 2 SE raise a false alarm with P 0.17; at 3 SE, 0.011. A 2% misread sits 3.7 SE away on batch 1, not 4.6.
- **The core's name.** With one residual per game, the fit is the episodic natural actor-critic with the expected Fisher matrix. Ruling (a) is put under that name.
- **"The regime the record leans toward" removed.** The stop-act readings span 0: −0.0028, +0.0043 on `parity3`, −0.0001 and −0.0018.

### From invent (critique: repairable)

**Kept: the dice positive control.** The slope of the label path's z on side A's hp luck is free, and it is measured on 23 player pairs with no excess spread.
- **Window.** This synthesis replaced the stance's range window with a Q-profile bound: the between-pair sd is ≤ 0.00049 at 95%, giving a 3-sd window of [+0.00268, +0.00565] on a batch. Each later batch must also sit within 3 SE of the earlier batches' mean (`round8_synthesis_power.out` §2).
- **What it catches.** A flipped result reads −0.00416. A mis-joined share f reads (1 − f) × 0.00416: the window catches f ≥ 0.36, and the batch-to-batch line catches changes of 0.03–0.04.
- **Why it is needed.** The coin plant sees neither error, because z cancels in T′ − T.

**Deferred to the $1,000 case: chance-effect supervision (CES).**
- **Its plant cannot work at the distillation's scale.** At the 2,227 game-equivalents of fights the distillation processes, κ/SE is 1.5, so a broken path cannot be told from a working one.
- **Its readings are unfunded or overfit.** The held-out kept-ratio replay costs $13–15 and is unfunded. The frozen-token ridge overfits by 0.36 (`round8_invent_critique_checks.out` §1).
- **Two failure modes have no guard.** The plant shares the label's join, so it reads κ on a mis-join. A post-strike input breaks exactness and biases coin labels (+0.0075 ± 0.0016 against 0 in the toy, §2–3).
- **The repairs are recorded:** κ ≥ 0.0014, a separate parse, pre-strike inputs with I2 and I-V extended, a funded replay, and the per-strike head's overhead measured.
- **The record leans against its premise:** hand prices saturate at 0.152, and setup conditioning reads null.

**Kept as a reading:** setup-conditioned luck prices are null after overfit (−0.0006 ± 0.0011).

### From challenge (critique: repairable)

**Kept, as Branch S, behind rulings (f) and (g):** a rating-conditioned base, certified by one match. The stance found a signal the project owns and never used, and it holds out of sample: ratings cut held-out log loss from 0.693 to 0.605. The stance also established that the earlier top-rated test never ran. The critique's repairs are all applied:
- **Out-of-fold ratings,** and an "unrated" input below 30 games. In-sample ratings leak the result: E[s | won] − E[s | lost] is +116 Elo, and one-game winners sit at s_mid.
- **An indicator for own-trajectory targets,** not a reserved s on the axis.
- **Jitter by the posterior sd,** and a top setting chosen for player coverage. This synthesis ran the coverage analysis (`analysis/round8_synthesis_skill_coverage.py`):
  - The highest centre where ≥ 10 players supply the sides within ±52 Elo, with none above 0.2, is +200: 42 players, largest share 0.146.
  - Above about +210, one player supplies half or more.
  - The low setting is +40, the rated 25th percentile (56 players). The 160-Elo gap lets M1 detect a transfer f ≥ 0.022.
- **A permutation gap before any match.** The holdout loss with true s against permuted s must be above 0 by 2 SE, so a null M1 can be told apart from "s was ignored".
- **One pre-registered arm,** with no selection over three arms. False adoption is then the standard 0.023.

**Not kept:**
- **Guidance past the top.** It rests on extrapolation that [UniMaia](https://arxiv.org/abs/2605.27767) (as cited by the challenge, not re-read), [Transcendence](https://arxiv.org/abs/2406.11741) and [Decision Transformer](https://arxiv.org/abs/2106.01345) argue against. The raw contrast also lets a −3-nat candidate win at a 1.5-nat contrast when γ is 2.
- **The three-arm claim.**
- **Directed coin bands.** The head gains from spread, not from the band's mean.
- **M1's sign as a known answer.** It is the hypothesis.

**Kept as record corrections:**
- At w 0.35, the (end, best act) pair covers only 0.140 of end-on-top decisions (`round8_challenge_stop_census.out`). Round 7 had called the "continue" edits covered.
- `own1` acts 7.56 times per side-turn early on, against 8.32 for lower-rated regulars and 10.07 for the top quartile. Yet the stop margin is flat at `own1`: offset −2.5 reads −12.2 ± 12.3. The tempo gap is therefore not reached through the end_turn logit. It is logged in Branch S's matches, with no gate.

**Taken from the stance's decision record:** "adopt `own2`" was called a $1.50 confirmation worth running on its own. It is now S0(b). `own2` read +16.5 ± 12.3 against `own1` at 64 slots (predicted +15; `docs/memory_own_round2_prereg_20261011.md`). A 12,800-game two-network match at SE 3.07 settles it as the base. Claims stay against `own1`, which adds $3.00 of two-network claims if `own2` wins.

### New in this synthesis

`analysis/round8_synthesis_power.py` and `analysis/round8_synthesis_skill_coverage.py`, outputs beside them.
1. **Budget:**
   - coin path $59.23 (E reserve $2.94, S5's adoption and claim two-network); $62.23 with `own2` as the base;
   - Branch S to its claim $4.81, exits at $2.59 and $3.32;
   - three rounds fit at up to 56% logging overhead on the measured host, 8% on a 1.41× slower one.
2. **The dice control's window** from a Q-profile bound on the between-pair spread, and the batch-to-batch line.
3. **S5 adopted against π_k,** with its prediction and miss line.
4. **The coverage rule for Branch S's settings.**
5. **The Differences-in-Q bias bound, read** ([Farias et al. 2022](https://arxiv.org/abs/2206.02371), Theorem 1, ar5iv). It is a constant times (1/(1 − λ))²·r_max·δ², where δ bounds the total-variation distance between the two actions' transition laws and λ is a mixing rate of the experimentation chain. Neither δ nor λ is observable for this game, so the bound cannot size a miss in advance. This closes round 7's research target: a miss stays detected, not sized.
6. **S5 runs unless J_S4's upper 2-SE bound is below minus the largest per-round gain measured so far.** This protects money without a guessed loss tolerance.

## Open questions

1. **Rulings:**
   - (a) restated: natural policy gradient on a near-tie coin policy, adopted only by match.
   - (b) narrowed to the luck columns.
   - (c) adds two-network claims when `own2` is the base and for S5.
   - (d) and (e) unchanged.
   - (f) new: rating-conditioned training on both sides; is it "imitation, parked"?
   - (g) new: two zero-initialised inputs.
2. **The spread v per type, the typed constants c and slopes b.** Nothing on disk bounds them; S1's cross-batch estimator reads them without a shape assumption.
3. **The shape of near-tie advantages.** Does τ̃ leave 0? It moves off 0 with P 0.04–0.10 when the plug-in is right.
4. **E's guarantee when the noise level varies within a pair type.** The half-share proof assumes one level per type; the sealed forecast protects the claim either way.
5. **Second-order and excursion loss.** No measurable bound exists in advance; S2 against Ĝ_pf, then E.
6. **Distillation closure** on an exchanged-pair target with a baked logit shift.
7. **Branch S:**
   - Does the network use s (the permutation gap)?
   - Does the axis transfer to play (f̂)?
   - Does N(+200) beat `own1`?
   - Does training on both sides bring back the memory's confusion?
8. **`own2` against `own1`** at SE 3.07.
9. **The server's feature-gather overhead.**

## Research targets for round 9

- **What the frozen tokens express.** Probing results on game networks that report how well small action-value differences, or move rankings at near-ties, are linearly decodable from token outputs ([Jenner et al. 2024](https://proceedings.nips.cc/paper_files/paper/2024/file/37d9f19150fce07bced2a81fc87d47a6-Paper-Conference.pdf) found look-ahead in bilinear form, not value gaps). Still the main unknown, and the only one a record might bound before S1.
- **Rating-conditioned networks at their top setting.** Do Maia-2, ALLIE or KataGo's human networks report that the top covered setting plays stronger than an unconditioned imitation model of the same data? Does training on losers' moves under a rating input change the top setting's strength? This feeds Branch S's case before ruling (f).
- **A finite-horizon second-order bound that uses measurable quantities,** for example the switch count and the range of per-decision advantages, as conservative policy iteration's bound does ([Kakade & Langford 2002](https://dl.acm.org/doi/10.5555/645531.656005)). Check whether it is tight enough at 19–38 switches per side-game to size a miss before S2.
- **Distilling a rule-modified argmax target:** closure when 5–15% of decisions change ([Rusu et al. 2016](https://arxiv.org/abs/1511.06295); [Czarnecki et al. 2019](https://arxiv.org/abs/1902.02186)), still not read.
- **The pre-test's selection.** Would a split pre-test (τ chosen on two pairs, tested on the other two) give a cleaner 2-SE line than the in-sample one, at what cost in power?
- **For the $1,000 case only:** CES's repaired plant at a funded scale, and the capture of antithetic pairs under learned prices.

## Unknowns that remain

| Unknown | Measured where | What relies on it |
|---|---|---|
| Spread v per type; typed c and b; switched share q; τ̃ | S1's batches (cross-batch estimator, welfare curve) | The round's gain; nothing before S1 |
| The base's census, runner-up mix, spectrum; logging overhead | S0(c), the 2,000-game smoke | The restated floors and costs before S1 is proposed; whether three rounds fit (≤ $2.986e-5) |
| `own2`'s strength against `own1` | S0(b), $1.50 | π_0 |
| Second-order and excursion loss | S2 against Ĝ_pf; E's match | Fallback; round 2's reading |
| Re-pricing under π_1 | S3 | Round 2's value |
| ĉ, ρ̂, J_S4, the price of an off-band flip; pool two-network price | S4, held out before its match; timed at the first two-network match | Whether N generates the next band; S5's prediction |
| Whether the network uses s; transfer f; N(+200) against `own1` | Branch S: B1's permutation gap, M1, M2 | The base for coin rounds |
| κ_T of a privileged critic | Only if chosen after a null | The critic column |
| r_DPI, antithetic capture, CES at a funded scale | The $1,000 case's replay | Nothing in the plan |
| Compounding, exploitability, human level | Each round's claim; none; none | Nothing |
