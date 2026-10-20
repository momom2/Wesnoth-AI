# Critique of round 3 (invent): margin experiments

**Verdict: repairable.** The KKT taper and the certificate estimator are algebraically right. Three things are wrong:
- the band path's planted check is a tautology;
- three predicted readouts contradict the proposal's own derivation;
- the luck heads change what the critic learns, and S1's predictions are not re-derived for it.

## 1. P-M cannot fail (fatal as written)

Let c = 1 mean a1 was played, so even(a_c) = c·even(a1) + (1−c)·even(a0). The booked stream is then exactly r(c−p)(even(a1) − even(a0)), which is the regressor itself. The slope is 1 by algebra, whatever c and p were logged:
- a coin drawn at the wrong probability passes;
- 1−p logged for p passes;
- a sign flip passes;
- the wrong runner-up passes.

Only a broken regression routine fails it. This is not the user's planted-signal check, which needs a modified game whose signal travels through play. I10 is circular in the same way: it recomputes p* from the logged margins instead of re-deriving the margins from the actor.

**Repair.**
- Put the parity bonus into the simulator's result, as P2 does with its tax. At r = 0.03, 15,000 band games give Σp(1−p)Δ² ≈ 100, so the slope's SE is about 0.07.
- Check I10 against margins recomputed by replaying the actor.

## 2. Readouts that contradict the derivation

Use the proposal's numbers: B = 0.06, λ = 0.005, Ḡ = 2. Then w = √(144/f), which is 1.55 to 3.1 nats over f = 60 to 15.

- **The "4 to 8 times per unit of shift" headline.** It needs Ḡ/w = 2 to 4. These widths give Ḡ/w = 0.65 to 1.3, so the ratio 2.13·Ḡ/w is 1.4 to 2.75.
- **S1(iii), "2.5 to 5 times the design's χ²".** With weak coins at a third, the band gets 0.04 of the budget.
  - The η-information comes to 1.4 to 2.6 times the design's.
  - The design's 140 to 320 includes the 0.16·m̄² ranking term, which band coins at m ≈ 0 lack. On that scale the ratio is 0.9 to 1.5.
  - κ's SE should be about 0.09 to 0.18, not 0.05 to 0.11. Correct code misses this readout, and it has no kill line.
- **S0(f).** The band's cost per coin decision, relative to a weak coin's, is w/(2.4Ḡ). That is 0.65, 0.50 and 0.32 at f = 15, 25 and 60.
  - "At most a third" holds only at the top of the prior range.
  - The kill line (at least half) fires at f ≤ 25, where the band is still 1.4 to 1.8 times better per unit of shift.
  - It gates on cost per decision, but the figure of merit is information per unit of shift.

## 3. Luck heads change the critic's training

With D in the event terms, C(x_t) regresses on z − ΣL_k(D), a luck-corrected Monte Carlo target. The joint coupling of C to its own values at unrealised outcomes, which carries the 125,000 χ² on chance, now trains D instead. The student's look-ahead and κ still read C(T(x,b,o)), and nothing ties C there to D.

The two tests do not reach this:
- I12 checks plumbing.
- The A/B reads ρ_C, which measures the control variate.

The A/B's 0.01 margin has no SE. A held-out mean of R̂¹R̂² over 1,000 games has an SE of about 0.03, and the variation between seeds of two 5,000-game critics is unmeasured. S1's predicted gap of at least 0.05 below ρ_C(U_0) is not re-derived under heads.

**Repair.**
- Read the attack look-aheads through D.
- Add a consistency loss C(x_k^{o_real}) ≈ D(x_k, o_real).
- Pre-register the A/B's SE.

## 4. An unmeasurable assumption carries the band

The taper is optimal only for the assumed σ_η²(G). The decode takes the argmax over every legal joint action (`tools/raw_player.py`). Near-ties are therefore often near-duplicates, such as the same unit sent to an adjacent hex, where Δ is tiny. So σ_η² may be smallest at G ≈ 0.

No step measures σ_η²(G) before S1:
- S−1 measures f, and S0 measures m(G).
- Forks cannot: an 8-playout contrast has noise variance of about 0.125 against a σ_η² of about 1e-4.

The first test is S1's κ SE, after $15 is spent. S−1 should at least report what near-ties consist of (same unit or not, hex distance).

## 5. Certificates are outcome policy gradient

For a ramp-smoothed rule π = ½ + (S + δφ)/b, the score at π = ½ is (φ/b)·4(c − ½). So Ĵ′ is the likelihood-ratio gradient ([Williams 1992](https://link.springer.com/article/10.1007/BF00992696)) in the rule's offset and slope, with a critic baseline. Two stages step along it: S2's fallback "c* whose certificate reads 0", and S3's threshold move. Only S3 is flagged for the user's ruling.

S3's prediction has the wrong sign. If the posterior is calibrated, a 0.8 threshold is conservative, so lowering it gains at first order. A plausible effect of 0.01 to 0.03 against an SE of 0.05 reads "zero" for lack of power.

The SEs "per nat" were derived for the log-probability offset. They are reused for D (in value units) and for a posterior probability, whose densities at their thresholds are unmeasured.

## 6. P-C's power is one-sided

Two combined SE come to 0.15.
- A halving bug passes: 2(c−½) for 4(c−½), or b for b/2, shifts the integral by 0.125 when Match A is 0.25.
- A correct code can fail: three-node Simpson overestimates J′ ∝ e^(−c/0.2) by 37%.

**Repair.** Use five nodes, and add an in-game plant (a bonus r per turn ended) that shifts J′ by a known f(c)·r at each node.

## 7. Reply-read: cost and coverage

"About 10 forwards per path state" assumes the opponent recruits nothing before its first attack.
- Most races draw 2 of 4 traits, 6 pairs; dwarves and elves draw 2 of 5, 10 pairs (`wesnoth_src/data/core/units.cfg`).
- The opponent's actor sees the traits, so its decisions branch: two recruits make 36 to 100 branches, each needing its own forwards.
- The $1 can therefore be off by one to two orders of magnitude.

I14 removes the coins from the reply, so the read is the no-coin C, not V_w. Over-continued coverage is unchanged: C still prices those states, one ply later.

## 8. Minor

- **S−1(c) at c = 2.5.** 800 side-games give an SE of at least √(8·0.8/16,000) ≈ 0.02, not 0.0075. The SE must also be clustered by game. [Ghosh & Wager (2025)](https://arxiv.org/abs/2512.15244) show that static regression discontinuity fails when past treatments move the running variable, so "textbook" understates the problem.
- **Eval numerics.** S0 alone rises from about $4 to $22 (173,000 games at $1.3e-4), against the claimed $15 overall.
- **Deviations.** At the cap there are 12 to 23 deviations per side-game against the design's 6. The program prices value shift, not state drift.
- **I13.** "Permuting the coins nulls" a regression discontinuity, which uses no coins.

## 9. Novelty

- **Band coins.** Softmax exploration with logged propensities already sets deviation odds from the network's own margin: fair at a tie, falling with G. Known noise used as local randomization is [Eckles, Ignatiadis, Wager & Wu (2025)](https://arxiv.org/abs/2004.09458). The linear taper is new.
- **Certificates.** The derivative of a sequential threshold rule's value with respect to its cutoff is Ghosh & Wager's "dynamic marginal policy effect", which they relate to the policy-gradient theorem. A randomized smoothed threshold for RL appears in [Roy et al. 2019](https://arxiv.org/abs/1912.10325) (search snippet only, unverified). New here: the Simpson known answer against a measured match, and use as gates.

## 10. Strongest part

**The stop-offset certificate.** It is unbiased for any baseline, so it tests from outcomes the critic's pricing of over-continued states (the design's weak point 3), whether or not the critic is right. It has a measured known answer, Match A.

**S−1** is the best practical step: for $0.5 it runs S0(a2)'s numerics check on games already played.
