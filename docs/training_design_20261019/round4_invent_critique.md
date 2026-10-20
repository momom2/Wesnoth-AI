# Critique of round 4 (invent): law-factored students

**Verdict: repairable.** The identity is correct and the parametrisation may help. But the information argument is overstated, the S3 prior is assumed rather than derived, and the planted check and two readouts can pass while the part that matters is broken.

## Strongest part

The factorisation E[Q − V | h] = Σ_o p_b(o | h) ω(h, b, o) is exact whenever the law is h-measurable. For visible fighters the law nearly always is: hidden ambushers in default-era factions carry no leadership or illumination, as far as I know. At decode the law costs about 0.1 ms per attack and no forward. I-law1 makes the A/B clean: both students fit the same numbers, so S2 compares two parametrisations on identical targets. The law census in S−1 is cheap and decides whether the extension is worth anything.

## Failures, most serious first

**1. The information argument does not hold; the gain is approximation only.** I-law1 itself shows that with w set to the critic's values, Q̂ equals Â_C plus a constant. The design's student already receives the chance information through Â_C = Σ_o p C(x∘b∘o). The 125,000 χ² pin differences among the children of fights that were *fought*. For an unplayed attack, the critic's values at its children are generalisation, exactly as in the design. And Q(x, b) − Q(x, a0) compares across fights: that is the ~1,000 χ² of decision information, which factoring does not raise.

One narrow class is a real exception: same attacker, same defender, another weapon. The core's outcome key is `(Side, Side)`: type, hit points, slowed, poisoned, petrified (`rust/wesnoth_core/src/outcomes.rs`). So children coincide on shared keys (plague aside), and weapon choice is a reweighting of on-path sibling values. S−1 should count this class.

**The S3 prior inherits the overstatement.** q is the calibrated prediction's R² against the *true* advantage. S2 measures R² against Â_C (0.35–0.7 predicted), which is not q. The critic's error is a function of the state: the student copies the part of it that h predicts and averages out the rest. So q can sit above or below S2's number, and only the coins (κ_outside, SE 0.11–0.15) tie it to truth. q = 0.6 is an assumption, and every critic on record failed its correlation test.

The spread s = 0.015–0.03 is also 1.5–2 times the design's 0.010–0.015, with no source. The memo's own formula gives 9.5 Elo at q = 0.6, s = 0.01 and 3 Elo at q = 0.3, s = 0.01. So the honest band before S2 is about +3 to +29, not +14 to +29, and it should be written as a function of S2's R² and κ_outside.

The formula also ignores the design's own gate. Under the same Gaussian model, posterior ≥ 0.8 means X ≥ 0.84·√(1−q)·s. That keeps 0.61 of E[max(X, 0)] in the memo's worked example, so 19 Elo becomes about 12. The design's prior has the same omission.

**2. P-law can pass while the outcome head ignores the position, and its SE is understated.**
- *It does not test position-reading.* The defender answers at the attacker's range (`counter_weapon` filters `w.ranged == a_ranged`). So the immediate plant is a function of the outcome key and the weapon's range bit only. A head that reads only those fits it. A wrong token gather (another unit's row, a stale memory) still reads near slope 1 if the future part is a small share of the planted variance. The plant does catch the bugs it names: a permuted law, a weapon shift, a sign error.
- *Its SE is too small.* My assumptions, not measured: about five ranged kills remain per side from mid-game, and the net count has variance about 5 per playout. That gives a playout SD of r√5 ≈ 0.067, and 0.024 for a mean of 8. The signal is r times the SD of P(ranged kill) across a decision's attacks: about 0.004–0.008 if half the attacks are melee. Over 1,200 attack points, the slope SE is about 0.024 / (√1200 × 0.006) ≈ 0.12, not 0.05. The band 1 ± 0.1 is then under 1 SE wide, and a correct path fails it roughly 40% of the time.
- *Spec gap.* For a stream, the target must be y = r(o) + C_s(x∘F∘o) − C_s(x). The §2.2 formula drops the reward booked at the transition.

Repair: add a plant whose immediate part needs h (a bonus only when the victim stands on a village, or for a hashed subset of unit types), which a key-only and a token-shuffled head must fail, predicted slope ≤ 0.3; use 64 playouts per candidate (about 154,000 playouts, about $2); set the band at 2 SE of the pilot's measured stream variance.

**3. Two readouts measure the easy variance.**
- *S1's gate.* The kill line on w's R² over outcome deltas (under 0.3) is dominated by between-key variance within one fight: τ ≈ 0.13 between outcome draws, against decision spreads of 0.01–0.03. A head given only the key, the two unit types and the law gets most of it. Report R² over a key-only baseline, within fight.
- *S2's A/B.* It scores the design's student over the full legal set, but that student was trained on the top four, so the A/B partly measures coverage. R² over all legal attacks is also dominated by attacks nobody would play. The "paired gain ≥ 0.1" can come entirely from candidates outside the top four, where the factored head also extrapolates (off-path targets at one decision in ten). The decision-relevant readouts are R² within the top four and the regret of each student's argmax against Â_C.

**4. Tails are summed but never trained.** Targets (a) keep outcomes with p ≥ 2%; the decode sums the whole law. Take an attacking leader that dies with probability 1.5%. The game ends, so ω ≈ −1 − E[V(x) | h], and the term contributes about −0.015 to −0.03: the size of s itself. So untrained keys decide exactly the risky attacks. Their correct value also needs the fogged value level, which the record never learned from observation ("no better than material").

Repair: train on every key with p-weighting, and give terminal keys the result exactly.

**5. Option tokens are not "exact-law values".** v_F evaluates w at the pre-move h, with the attacker's token on its old hex. w was trained only at attack decisions. The move can uncover fog, so the post-move h cannot be known before moving. The law is exact for a hypothetical; the values are out of distribution. The +0.03–0.10 R² gain is not derived.

These tokens and the in-play factorisation are rule-computed hypotheticals chosen by design, like the threat tokens, which alone are flagged to the user. Wesnoth's default AI uses the same decomposition with hand-written outcome values: `value = chance_to_kill*target_value - avg_losses*(1.0-aggression)` ([attack.cpp at 1.18.4](https://raw.githubusercontent.com/wesnoth/wesnoth/1.18.4/src/ai/default/attack.cpp)). Since the factorisation stays in the player permanently (§6), the user should rule on it under "learned only" before S1 pays for its targets.

**6. Calibration does not cover the switches.** Same-unit coins draw among the unit's other attacks, top-four ones included, and cover only the unit π chose; the decode also switches to other units' attacks and to attacks where a0 was a move. The joint law with a band coin at the same decision is unspecified (I-law3 tests the marginal). Calibration on random alternatives does not transfer to the argmax over 2–6 extrapolated candidates unless it holds conditionally (the optimiser's curse); the memo names this as a ladder outcome, not a guard. Repair: restrict the coins to outside-top-four alternatives.

## Costs

They add up: $58.8 through round 2 and S5, $70.8 with round 3. The final-critic pass is 25,000 × 900 × $5·10⁻⁸ ≈ $1.1; the off-path $0.8 is conservative. In play, the law calls take about 70 ms a game, roughly 2% of a worker's time. P-law's repair (about $2) fits the $14 margin.

## Novelty

- **Claim 1** (exact law × learned per-outcome values, under fog) is expectimax with a learned leaf: TD-Gammon; 2048's afterstate learners ([Szubert & Jaśkowski](https://www.cs.put.poznan.pl/mszubert/projects/2048.html); [Yeh et al.](https://arxiv.org/abs/1606.07374)); [Stochastic MuZero](https://openreview.net/forum?id=X6D9bAHhBQ1), which values chance codes from the parent's latent. Closest, in imperfect information, is [DeepStack](https://arxiv.org/abs/1701.01724): pre-flop it enumerates all 22,100 flops through the flop network, and its auxiliary network is trained on those averages, the unfactored amortisation built for speed. What remains is an outcome-conditioned head on the parent's tokens with privileged targets: weakly new.
- **Claim 2** is distillation of an expectimax node's child values. Weak.
- **Claim 3** is the default AI's inputs with learned values, as stated.

