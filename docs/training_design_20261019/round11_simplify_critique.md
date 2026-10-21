# Round 11 critique of the simplify stance (plain REINFORCE)

**Verdict: repairable.** The estimator is right. Several checks and readings around it measure something other than what they claim, and one optimizer claim is wrong. Each has a cheap repair. No guessed number carries a decision. New script: `analysis/round11_simplify_critique_rank.py` (*rank*).

## 1. Provenance audit

| Number | Class | Check |
|---|---|---|
| $2.598e-5 and $8.848e-5 a game; 230.4 decisions a side-game; 69 ± 30 s and 155 ms; 0.291 CPU-s, 20.08 turns, 0 caps; window 16 × 32 | M | Records opened: all confirmed. All are argmax readings, and the proposal tags the temperature-1 values as unknown |
| Trainer 1,476.9 positions/s | M | Confirmed. It comes from 40 timed windows (`trainer_calibration_20261015/memory.json`); own2's real pass ran at about 197/s on the older trainer (`memory_own_round2_20261011/train.log`). The 0.58 training share stands until S0's 18.4 M-position pass reports its speed, and S3's size must be re-derived from that reading |
| B_simple 11,593; z 4.13; split-half z 1.29 | M | Rerun: reproduced to the digit |
| Residual 0.699 / 0.692 | M | Confirmed. The 0.692 comes from own2_64 against own2_0, 800 games, SE about 0.04 |
| Luck share 0.072 at T 0.5 | M | Confirmed, but from a different estimator (two columns, mid-game playouts). A report only |
| $8.654e-5 a trained game; 0.533 s; B* and relative costs; stage costs; 347.4 Elo per unit; TRPO 4.7e-6; KL 2.0e-4; end_turn SE 8.3; 0.242; 1.19× | D | Redone: all correct. The S2 and S3 costs include a guessed 90 s trainer restart (about $0.9, deciding nothing). The claim SE of 3.07 holds only at a 50% win rate (§2.8). The 1.19× is measured against raw z, not against the luck-corrected R |
| Post: 3.50 against 3.248; 18.8×; 6.8× | D | Checked: 3.521 final, 3.498 best. The post's matrices ran a Muon-family optimizer (ANVIL), which the proposal does not say |

## 2. What would make it fail

**2.1 Fresh-moment AdamW is not the step S1 calibrates, and the forecast is biased.**
- The claim that "v̂'s own-batch weight is 0.001" ignores bias correction. Under fresh moments the weight is (1−β₂)/(1−β₂ᵏ): 1 at k = 1, 0.10 at k = 10, 0.02 at k = 50.
- So each segment's first step is lr·sign(g) on a noise-dominated gradient, across 15.6 M weights, and g_AᵀPg_B is no longer unbiased for the gain.
- S1 measures one step θ₀ + ηd under S0's preconditioner, while S3 runs fresh moments with a 2,000-game v. "Travel η" has no defined mapping between them.

*Repair:* warm-start v from S0 on sub-batches of S3's size, with no bias-correction reset. P is then nearly fixed, the 0.001 holds, and K steps of B games travel η·d at lr = η·B/20,000.

**2.2 S0's gate does not test whether steps carry signal.**
- Against a sign-flip null, the z of |G|² is (n/B_simple)·√(r_eff/2), where r_eff = (tr Σ)²/tr Σ².
- On the one measured gradient, r_eff is 18.9 of 43 columns, and z is 15.9 against the null but 4.0 on its full SE (*rank*).
- At 15.6 M parameters, B_simple = 10⁶ with r_eff = 10⁴ still reads z 2.8 at 40,000 games (*rank*, derived). The gate therefore passes on gradients that are mostly noise at B* ≈ 2,000.
- "Below the coin analog's strength" compares noise scales across parametrisations, which are not comparable.

*Repair:* gate on the z of the preconditioned split-half forecast at the planned batch. That is the quantity S1 sets η₁ from.

**2.3 P3 can fail when the code is right.**
- It extrapolates a slope taken at offset 0 out to −1.5, an offset worth about 229 Elo at argmax.
- If J(e) is not monotone on [−1.5, 0], the signs disagree and the design stops on correct code. A sign match proves little.

*Repair:* one one-network match, own2@T1 at +0.75 against own2@T1 at −0.75. The game is antisymmetric: f_aa = −f_bb, and the symmetric part of f_ab vanishes at (0,0). So f(e, −e) = 2e·∂ₐf + O(e³), and the forecast is exact to second order.

**2.4 Nothing checks that actions are drawn from the logged p, which unbiasedness rests on.**
- I-logp compares server with trainer probabilities, and P1 is algebra on logged p.
- A decode that kept the −1.5 offset while logging unshifted priors would pass both.

*Repair:* add I-sample, the analogue of I-coin. Run a binned χ² of 1[a ∈ c] against p(c) for end_turn, attack and recruit, and test that the sampled action's rank under its logged distribution is uniform.

**2.5 S3's seal leaves out a term known to be nonzero.**
- "θ_K vs θ₀ = Σ forecasts" is first-order only.
- At B ≈ 0.1–0.2·B_noise, each step's noise adds a second-order loss that accumulates, so "below" is the expected reading, not a diagnosis. The record already shows second-order terms of −12.3 ± 4.5 Elo.

*Repair:* at each η, S1 plays two matches:
- θ₀+ηd against θ₀−ηd, which isolates the first-order term to O(η³);
- θ₀+ηd against θ₀; its difference from the first gives the curvature.

The seal then adds a K·c₂·lr² term.

**2.6 The Muon arm runs in the wrong regime.**
- The cited failure is whitening of low-SNR policy gradients ([Fan et al.](https://arxiv.org/abs/2605.19282): "collapses to zero"; checked).
- The arm runs on S2's planted signal at z ≥ 5 a step, where that failure cannot show.

*Repair:* score each optimizer's first step on S0's real sub-batch gradients by held-out forecast gain at matched step size. This needs no training and no extra games.

**2.7 Stage order and the plant.**
- S1 plays weights from the update and hand-off code before S2 has checked that code. Run S2 first.
- S2 replaces R, so the plant is never recovered "at the predicted rate" through real outcome noise. Train on R + plant, with δ set from S0's noise.
- "Σ p(X) moves as forecast" includes the state-distribution shift that the plant's gradient excludes. Read p_{θ_{k+1}}(X) on batch k's own histories by teacher forcing.

**2.8 Smaller.**
- **Code.** `own_policy_loss` keeps only rows with `prob > 0` (`live = batch.prob > 0`), so a signed weight routed through the target becomes winners-only imitation. The trainer also steps every 512-position window and adds imitation and value terms, so accumulating over about 1,800 windows is new code. P1 catches the sign trap only if it runs first.
- **Claims.** At a 400-Elo gap (the seed's +412), the claim SE is 5.4, not 3.07: 173.7/√(12,800·0.091·0.909). The reference is now own2 (BACKLOG 0.31.4).
- **Telemetry.** Under β₁ = 0.9 and noisy batches, the consecutive-update cosine reads the momentum's overlap, not edge of stability (the post's −0.945 came from full-batch descent). Use cross-step split products instead.
- **Caps.** "−1 for both" cannot be written as s_j(z − …), and it diverges from the yardstick. Memory players can stall: `selfpin` capped 220 of 1,020 games. S0 must read the cap rate first.
- **Entropy.** No telemetry tracks it, and the credit signal vanishes as the policy sharpens.
- **Novelty.** [Mesnard et al.](https://arxiv.org/abs/2011.09464) already separate skill from luck with hindsight baselines that are independent of the action. What is new is only the exact chance-law form. The centred plant is the score-function identity with p(X) as baseline.

## 3. Strongest part

The update is exactly unbiased for the self-play gradient except for three departures it names itself (memory truncation, server arithmetic, refused recruits).
- **The luck control variate is zero-mean by construction.** `tools/match_luck.py` defines L = Q(realized) − Σ p(o)Q(o), with L = 0 decided before the attack wherever it is not computed. So βᵀS cancels for any fixed β, under fog and memory.
- **P1 is an exact identity at the logits.** It catches sign, alignment and R errors, the `own_loss` trap included.
- **The cost case against data reuse and critics follows from the measured cost shares.**
- **The optimizer citations check out.** Ruan et al.'s RMS-matched control removes Muon's gain ([2607.16169](https://arxiv.org/abs/2607.16169)), and [Liu et al.](https://arxiv.org/abs/2502.16982) Table 6 shows no Muon advantage after AdamW pretraining.
