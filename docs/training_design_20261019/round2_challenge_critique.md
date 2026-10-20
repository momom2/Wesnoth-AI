# Critique of round 2 challenge: intent iteration

**Verdict: repairable.** The premise is gated early and cheaply, and every flaw below can be fixed in the spec. Two of them must be fixed before S1 is sized: the premise readout cannot fail (item 1), and the base player and the refit break the forecast (items 4 and 5).

## Strongest part

The premise is split into four measurable claims (information, obedience, competitiveness, spread), tested for about $18 before any player is built; a kill costs about $6 because the critic is shared. Within one round the performance-difference lemma over intent points is exact, since π and π′ share the decoder. It also names a real gap in the current design: an action-level step sees a d-decision change only at its last decision.

## Issues

**1. The premise readout cannot fire.** S2 compares ŝ_z (turn closure, 2 samples per code) with ŝ (the critic's exact one-step look-ahead). By the design's own numbers, a closure sample carries about 0.011 of luck variance after correction, and a code's mean over 2 samples carries 0.0055. The signal is s_z² ≈ 10⁻³. A naive spread therefore gives E[ŝ_z²] ≈ s_z² + 0.0055. At s_z = s = 0.01 the ratio reads about 7.5. At s_z = 0 it still reads 7.4, so the kill at < 1.5 can never trigger.

"Corrected" is not defined; if it means luck-corrected only, the readout measures noise. Repairs: subtract the within-code variance the two replicates estimate, and price the top-4 joint actions by the same closure (one action, the base's play to end_turn, C at x_e), so both spreads share an estimator; the critic's look-ahead is also shrunk by its own error.

Spread is not upside: a codebook whose non-modal codes are all worse than the modal one passes. S2 should also read the share of intent points where a non-modal code beats the modal one by more than 2 noise-corrected standard errors, against the same share for actions. The ρ̄ = 0.3–0.6 behind "8 to 24 times" has no source.

**2. The evidence for intents does not discriminate.**
- The −1.5 offset (+229) is a per-decision stop rule, applied the same way at every decision. It is exactly the kind of change single-decision policy iteration prices, so it supports the action-level design, not intents.
- The +412 of argmax over temperature-1 sampling is confounded. Sampling puts mass on the long tail of bad actions, and the current design's coins (top 3 only) avoid that tail. Incoherence is one explanation, not the measured one.
- "6 of 7 better turns" comes from turns played with their own dice, the comparison the seed marks invalid.
- Hu et al. 2019, which the design cites "for", contradicts it on the point that matters. Its latents were supervised language instructions, not learned codes. Its **one-hot** (non-compositional) instructor won 44.6 / 49.7 / 43.1% against direct imitation at 50 / 250 / 500 instructions (draws exist), while the compositional encoders won 55–58% ([Table 3, arXiv 1906.00744](https://arxiv.org/abs/1906.00744)).

A flat 16-way code is the one-hot case. Lower the prior on (iv), and state it as a number.

**3. Held codes and re-selection.**
- **Round 1 holds z through every roll.** The decoder re-decides, but the plan component does not, which is the adaptation the ruling exists to protect. Re-selection should run from round 1, not round 2.
- **From round 2, the closure prices a different continuation.** It holds z to end_turn, while π re-selects after each fight. A^π(h, z) is then not π's advantage: the selector's target and the lemma's forecast are biased by the value of re-selection the closure leaves out. Repair: price to the next intent point (the end of the next fight, or end_turn) and read C there, which is also cheaper.
- **Codes learned from realized turns name dice branches.** From one realization, a posterior cannot separate a contingent plan from the branch the dice chose, so a code partly means "the attack worked, push" ([Dichotomy of Control](https://arxiv.org/abs/2210.13435), [ESPER](https://arxiv.org/abs/2205.15967)). Fresh-dice pricing keeps selection unbiased, but obedience and competitiveness suffer, and readouts (i) and (ii) count branch information as plan. P-I cannot see it: its variants do not depend on the dice. Add an invariant: on human turns, the turn's logged luck columns regress on the posterior code with null effect.

**4. The base player has no mechanism.** The trunk is retrained on human turns with a code token the memory records. The record's worst failure was the memory continuing its own history (−105 Elo, four times the end_turns); p(z | h) reading past codes gives that confusion a direct channel. `own1`'s repair trained toward its memoryless reading; "own1's recipe with sampled codes" names no teacher, so (iii)'s ±20 Elo band has no derivation. Name the teacher, or freeze `own1`'s trunk and heads and add the code through an adapter, so (iii) is near zero by construction.

**5. Rounds 2 and later break the forecast.** Refitting the codebook retrains decoder and posterior, so "actions given a code coincide" fails between π_k and π_{k+1}, and the critic reads codes whose meaning changed. Append new codes with the decoder frozen on the old ones, and audit the refit itself, with a predicted number, before selecting on top of it.

**6. Stages without a predicted number.**
- 2.5 predicts only that the offset-0 penalty "shrinks".
- S5's "at most the last round's gain" and S6's "at most round 1's gain" are caps. A zero reading cannot be told apart from a broken stage.
- The design admits that self-intents have no predicted readout.

This is the fire-and-hope pattern the user ruled out. Each needs a central number and a band.

**7. The audit at turn level.** The midpoint rule is exact only to second order in interactions among switches. Eight whole-turn switches per side-game reshape the states that later switches see, so the third-order term is likely larger than for single actions, and nothing bounds it. The current design used two-point Gauss–Legendre for P0; P0_z and S4 use the midpoint alone. Use the two points, which are exact for cubic J, at the same game count.

P0_z runs before any critic exists (σ_r² ≈ 0.94, consistent with its 19 Elo); its ±38 Elo band catches a sign or factor-2 propensity error, not a 30% one.

## Costs

They check out: closure ≈ 80 policy forwards at $5.5×10⁻⁸ (17,000 games an hour × 450 decisions) plus 104 critic forwards at $3.6×10⁻⁸ ≈ $8×10⁻⁶ (stated $1.1×10⁻⁵ with simulation); calibration standard error 0.041 and audit 0.035 reproduce; stages sum to $56. Caveats: S1's $6 holds only at 2–3 epochs each for the corpus and P-I learners, unstated; the audit's $0.4 assumes generation numerics ($2.1 under eval numerics).

## Novelty

- **Claims 1 and 2** (latents of future behaviour, chosen by value, under stochasticity): Dichotomy of Control is the nearest work and is missing. It learns the value prior from data and suppresses chance with a mutual-information penalty. Latent action priors learned from demonstrations: [SPiRL](https://proceedings.mlr.press/v155/pertsch21a.html), [SAQ](https://arxiv.org/abs/2310.11731), OPAL. Options evaluated by rollouts with interruption are classical.
- **Claim 3**: a vocabulary regrown from the improving agent's own solutions is DreamCoder's wake-sleep library learning ([arXiv 2006.08381](https://arxiv.org/abs/2006.08381)), and recent agent skill banks co-evolve libraries with the policy (unverified beyond abstracts).
- **Claim 4**, synthetic recovery of a planted latent, is standard practice.

What remains new is narrow and real: pricing learned codes by fresh-dice simulation under the policy's own play, with known-propensity coins on them, which removes Dichotomy of Control's luck bias by construction rather than by a penalty. The design should claim exactly that.
