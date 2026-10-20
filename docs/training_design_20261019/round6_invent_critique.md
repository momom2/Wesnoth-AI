# Round 6 critique: invent stance (dice-priced innovations)

Verdict: **repairable**. The estimator is correct. The planted check cannot measure what it claims, the admission line is set by significance rather than by what DPI is worth, and the lottery column breaks two rulings. New scripts: `analysis/round6_invent_critique_checks.py` and `analysis/round6_invent_critique_breakeven.py`, each with its `.out`.

## 1. Provenance audit

| Number | Class | Check |
|---|---|---|
| r 0.732 (setup + columns, `own1` self-play); pairs 0.700 ± 0.009 | M | Both hold. The pairs figure is a cell-covariance estimate, not a reading of played pairs. |
| Columns 0.848; value_50 − material −0.0428 ± 0.0043 | M | Reproduced. Control I added: material + terminal (both fitted) keeps 0.8951, and value_50 beats it by −0.0420 ± 0.0043. The gain is conditioning on the board, not the terminal term (`checks.out` §1). |
| Paired SEs "0.0013–0.0043" | M, misapplied | These SEs are pooled over 18,800 games. S0(d)'s gate is read on `own1`'s 800 games, where the same SEs are 0.0044–0.0147 (`checks.out` §2). |
| 109.7 fights, 45.8 recruits, 0/800 capped, 449.1 decisions | M | Hold (`readout.txt`, `round5_synthesis_outcomes.out`). |
| S0(d) 11.5 M forwards, $0.55 | D, inconsistent | Multiplies the decisive-game count by the mean turns of all walked games (27.27). Decisive games average 21.12 turns, giving 8.86 M forwards and $0.42 (`checks.out` §3). The error is conservative. |
| Power table; fit penalty +0.02; CPU, storage and round costs | D | Redone; all hold. The "r_match scaled alike" column is an assumption, but it is only illustrative, because S0(d) reads r in pairs. |
| P-DPI bands ± 0.15 / ± 0.054 | D | The arithmetic is right, but the derivation does not apply (F1). |
| u 25, ψ 13, Θ 420, PCA-8, δ | C | These fix what S0(d) measures (F3). |

No guessed number carries a decision outright (the gate's line is computed in the run), but two decisions rest on numbers that do not mean what the stance says (F1, F2).

## 2. What would make it fail

**F1. P-DPI is either a tautology or biased.**
- **Tautology.** The plant δ Σ_k s_k(1[defender dies] − P_k)(e·u_k) lies exactly in the regressors' span: it is the defender-dead row of X̃_g contracted with δe. OLS is linear in the response, so Θ̂(z′) = Θ̂(z) + δe exactly. Read as the planted fit minus the unplanted fit on the same games, the slope is 1 by algebra.
- **Bias.** Read from the planted fit alone, which is the reading the band √(r/(0.01N)) presumes, the slope is 1 + e·Θ_dead/δ. That second term is the true price's projection on e. It is not zero, since kills dominate fight luck and e has a component on u's constant entry, and nothing bounds it.
- **Band too narrow.** The band also omits collinearity: the defender's death fixes the attacker's XP gain, "advanced" and "game ends". Correct code can fail the check.
- **Blind to real bugs.** The plant is built from the same logged s_k, u_k and P_k that the fit reads, so a misaligned token, a flipped sign or a wrong law all pass.
- **Weak negative control.** Bias comes from centring, which only I-D1 and I-D2 test; a post-fight price biases coin labels only weakly at near-ties, so the negative control may not show the test can fail.
- **Repair.** Fit the planted and unplanted versions on disjoint folds, with the SE from σ̂²(X̃ᵀX̃)⁻¹. Compute P_k independently, for example by Monte Carlo of the core's attack on forks. Test L_k for zero mean per event over the ~2 M fights; this is exact under correct code.

**F2. The admission line is significance, not worth.**
- **Where the gate admits DPI.** On 800 games the gate admits DPI at about r 0.70–0.72.
- **What DPI saves there.** Under the stance's own model, DPI first saves a batch at r ≤ 0.640 and a second at r ≤ 0.548 (`breakeven.out`). At 8 batches, r 0.6 moves the floor only from 0.0020 to 0.0019 (`round6_invent_dpi_power.out`).
- **The cost of admitting it.** The gate can admit token logging through the inference server, the trait laws, cross-fitting, I4′ and relabelling for $0 saved. "DPI pays whenever r_DPI < 0.732" counts compute only, while the project's failure mode is bugs and DPI enters every label's R_g.
- **Repair.** Admit DPI at the derived r ≤ 0.640, or at a stated floor change, and read r pooled as well as on `own1`'s games alone.

**F3. A null at S0(d) is ambiguous.**
- **What a null would show.** Only that bilinear prices on PCA-8 projections of three tokens fail.
- **Why PCA-8 is a weak basis.** Principal axes rank variance, not value. Conditioning on the board margin, the only lever the record shows, need not lie on them.
- **No floor under the fit.** The fit does not nest the seven columns, so it can read worse than value_50. "DPI is not built" would then rest on a [C].
- **Repair.** Nest the columns. Add the value head's scalar to u. Fit a closed-form reduced-rank regression on logged full tokens (2.1 M fights × 3 × 384 in bf16, about 5 GB [D]), so that a null bounds the linear-in-tokens class.

**F4. The lottery column breaks two rulings.**
- **Play ruling.** It feeds a ψ-based evaluator into the switch rule. Ruling (b) covers control variates "never in play", and the stance's note to the user does not cover this use.
- **Fog ruling.** Its law is "the core's, at a fork with the unit on the hex", which is the true state. At decision time the hexes around a visible defender can be fogged. A hidden adjacent leader (leadership), an illuminator, or an ambusher on the path then changes the law, so ℓ leaks hidden information into play.
- **No known answer.** Its coefficient of 1 holds for true prices only, and nothing predicts that chance-sibling prices transfer to decision siblings.
- **Repair.** Defer the column until DPI is certified, compute it in the mover's world, and get an explicit ruling.

**F5. Novelty: mostly covered.**
- **Learned value with AIVAT.** DeepStack corrected its matches against professionals with AIVAT, using its own learned value function, exactly unbiased, at an 85% cut in SD ([Moravčík et al. 2017](https://arxiv.org/abs/1701.01724); read via a search summary).
- **Unbiased learned baselines.** Learned baselines that read future exogenous randomness are proven bias-free in policy-gradient training ([Mao et al., ICLR 2019](https://arxiv.org/abs/1807.02264)).
- **The fit.** DPI's fit is MIVAT's closed-form variance minimiser with a tensor-product basis.
- **Retroactive relabelling.** This is the linearity of least squares.
- **Cross-fitting.** [arXiv 2605.14261](https://arxiv.org/abs/2605.14261) shows a value fitted on the evaluation data makes the variance pathologically low: this supports the cross-fitting, and the deferred nonlinear head must be fixed before the batch it is read on.
- **What is new.** The one-forward bilinear price, a cost device inside a known estimator, and its use on coin labels. §7 items 1, 3 and 4 should not be listed as new.

**Smaller point.** Undead have `num_traits=1`, filled by their musthave trait (`units.cfg`), so "46 trait draws a game" overcounts the free experiments.

## 3. Strongest part

The exactness argument, and the order of work it permits.
- **Bias.** Correctness rests only on three conditions: L_k is centred under the enumerated law, u_k is fixed before the event, and Θ̂ is fitted outside the batch it labels. A poor price therefore costs variance, never bias.
- **Confounding.** Because the dice are randomized with known laws, the price regression cannot be confounded by the players' choices.
- **Order of work.** Its value is measurable for about $0.4–0.6 before any generation code exists.
- **Evidence.** The record supports trying it: conditioning on the board removes 0.042 ± 0.004 of the variance beyond material + terminal.
