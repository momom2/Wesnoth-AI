# Round 8, critique of the simplify stance

**Verdict: repairable.** No guessed number carries a decision. The removals hold, and the interim's removal is right for a reason that does not depend on the model. Two things must be fixed before S1. As written, fallback E cannot be computed from the logs the stance specifies. And the stance's account of a forecast miss leaves out a code path that no plant exercises. Script: `analysis/round8_simplify_critique_checks.py` (`.out` beside it), cited as §n.

## 1. Provenance audit

I reran `round8_simplify_power.py`, and its output is byte-identical to the `.out` file. I also reopened the input records it reads.

| Number | Class | Check |
|---|---|---|
| Pool price $1.912e-5; host spread 1.41×; two-network price $1.17e-4; box setup $0.018 | M | `round7_provenance_price_tail.out` (ref_p120x6 row), `round6_provenance_prices.out` ✓ |
| Var(z) 1.0004; caps 0 of 1,600; walk 0.321 s; numerics flips 1.76% | M | Their cited records ✓ |
| Residual 0.688 | Tagged M, actually D | `round6_provenance_cells.out` ✓. It is own1's column share minus the cell share pooled over 23 matches. Own1's games alone give 0.651 (0.581 to 0.721) |
| Band 76.1; 125 at w 0.7 | D | parity3's shares × own1's 224.6 decisions per side-game ✓. A transfer between networks; the smoke re-measures it before S1 |
| Floors; power 0.80 → 0.77 / 0.63; P(stop) 0.50 | D, Gaussian model | Rerun ✓. These hold at hypothetical spreads. Only the direction of the trade carries a decision, and the direction does not depend on the model (§3) |
| Budget $56.18, stage costs; price lines; null round 1 $16.16 against $11.74; slack 5.0 batches, V's break-even share 0.614 | D | Rerun, and checked by hand ✓ |
| Plant SE 0.0387 / 0.0055 | D | Rederived: per game, the numerator's variance is about 3n² against a denominator of n ✓. One misread distance is wrong (§2.4) |
| Attack known answer: −78 ± 16 Elo; 2Δ = −0.449 | M via D | `round6_provenance_critique_attack_stage1.out` B ✓. A kinked fit over 2,399 stage-1 games, extrapolated to an offset of +1 |
| Forecast SE 4.6 and miss line 9.9 (3.6 to 7.8 at μ −0.0018); residual offset; tempo closure | D / M | ✓ |
| v, census, overhead, ĉ, ρ̂ | U | Nothing uses any of them before it is measured ✓ |

## 2. What would make it fail

**2.1 E cannot be run from the stated logs.** Per band decision, §2 logs only the log-only student's switch bit (f > 0); A_g and the Gram are sums. E needs two things: the final student's half-quantile of f, and a sealed forecast in which students 4–7 each switch above their own half-quantile (`round6_synthesis.md`). Neither can be recovered from a bit or from sums. I6 and I-log's recomputation also need per-decision x, which nothing stores. Repair: log f_i in place of the bit, and keep x for a fixed 1,000-game subsample of each batch (about 0.24 GB in bf16). Student 8's half-quantile is read on that subsample.

**2.2 A forecast miss has a third possible cause.** §2 says that "with the plants passed, a miss is curvature or the opponent's coins". The plants test the label and fit code, and a constant per-seat offset through the hook. No plant exercises the head's swap in the match: reading the features at match time, picking the pair, and exchanging a0 and a1 on one seat. I-log covers generation games only. A defect in the swap would read as curvature, and E would inherit it. Repair at no cost: on the match's head seat, log f and the switch at every band decision. Then check that every switch played a1, that f recomputes from stored x, and that the switch rate sits near the forecast's.

**2.3 Removal 1 leaves two costs out of its table.**
- At a null, false adoption doubles, from 0.011 to 0.023 (`round8_simplify_power.out` §1, sd 0). That is still the standard 2-SE rate, but it belongs in the table.
- The interim saves 11.4 hours of wall time in expectation at a null (§1).

The trade holds, since every futility line trades dollars for power at a roughly fixed rate (§1, at the μ −0.0018 floor):
- line 0: $4.42 saved for 0.17 of power;
- line −0.5 SE: $2.45 for 0.08;
- line −1 SE: $0.91 for 0.03;
- below about −1.4 SE: a loss of both money and power.

At the μ 0 floor, the lower lines give up less power: 0.010 at −0.5 SE against 0.032 at 0. So if the user values the dollars, the fallback should be a line near −0.5 SE, not the interim as it was.

**2.4 The plant gate has no stated line, and one distance is overstated.**
- With four readings (two sides, on the smoke and on batch 1), a 2-SE line raises an alarm on correct code with probability 0.17. A 3-SE line lowers that to 0.011, and every gross error the stance lists stays 27 to 56 SE away (§3).
- In the stance's own model, the slope with a share f of misread coins is exactly 1 − f. A 2% misread therefore sits 3.7 SE away on batch 1, not 4.6, and is detected with probability 0.74 at 3 SE (§2). The 4.6 came from a 100-replicate simulated mean whose own noise is two thirds of batch 1's SE.

**2.5 The nearest work is misnamed.** The appendix of [Bottou et al.](https://ar5iv.labs.arxiv.org/html/1209.2355) that the stance cites (D.1) is titled "Infinitesimal Interventions and Policy Gradient" (text read). With one residual per game, the identity with natural policy gradient is exact:
- At every band decision, P_i(1 − P_i) = 0.1875. So A_g = ψ_g/0.1875, where ψ_g = Σ s_i(D_i − P_i)x_i is the game's summed score of a logistic coin policy at θ = 0.
- Hence ŵ = (F̂ + 0.1875λI)⁻¹ Σ_g R_g ψ_g, with F̂ = 0.1875 G, the expected Fisher matrix.
- That is the episodic natural actor-critic, which regresses each episode's return on its summed score, with the regression's constant replaced here by the cell and the luck columns ([Peters & Schaal 2007](https://www.esann.org/sites/default/files/proceedings/legacy/es2007-125.pdf); the PDF was unreadable to me, and the form is taken from [Su et al. 2017](https://arxiv.org/abs/1707.00130), Eq. 7).
- The sign rule is the greedy step that the natural gradient points to ([Kakade 2001](https://papers.nips.cc/paper/2073-a-natural-policy-gradient), abstract read).

Rounds 1 and 5 already said this, yet §4 names RESLOPE and LOLS as the nearest work. What is new is the assembly: near-tie exploration with logged propensities, exact chance-law control variates, frozen features, and certification by match. Ruling (a) should be put to the user under this name, because the seed lists outcome-driven policy gradient as killed.

**2.6 "The regime the record leans toward" overstates the record.** The −0.0018 reading's interval spans 0. The same file also holds −0.0028, −0.0001 and, on parity3, +0.0043 (+0.0021 to +0.0062) (`round5_provenance_stop_acts.out`), and all four readings price only (end, act) pairs. No decision rests on the phrase; drop it.

## 3. Strongest part

Removing the interim, for a reason that holds in any model. The interim match and the adoption match are independent games, so a stop can only lower adoption power, and at a null P(stop) is exactly ½. What remains is money against power, and the budget does not bind ($28.82 left). The two other removals are sound too:
- The census is now read by the generator's own band computation, under the coin policy the labels sample.
- The record supports deferring γV: the value head ranks turns at 0.22 against the HP margin's 0.40, and critics that see only the observation do no better than material (seed §3).
