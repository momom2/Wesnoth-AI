# Critique of round2_harden.md (round 2)

**Verdict: repairable.** The identity is exact and the double-sampled loss is unbiased. Four instruments fail as written: P1's rate test rejects correct code five times in six, the κ gate almost never closes, P0's kill line blames bugs for an approximation error it cannot exclude, and π_1^raw's known answer is not an estimand. The budget breaks if S0(a2) fails, compounding rests on single-agent theory, and two novelty claims are covered.

## Strongest part

Sections 1.1–1.2. Conditioning the critic on the closed-loop state (true state, both memories, rejection sets, remaining schedule) makes z = V_S(x_0) + Σ L_k(V_S) hold without assuming 64 slots summarise a history. I checked the double sampling: with R̂ʲ = R + εʲ, the tail errors independent and mean zero given the trajectory, E[R̂¹R̂²] = E[R²] and every gradient cross term vanishes. The information accounting (8M × 0.0086/0.55 ≈ 125,000 against 0.9M × s²/0.55 ≈ 160–370) is right and names the real weak point.

## 1. P1's tests are narrower than their instruments

- **Rate test.** If "error" is the realised |L̂ − 0.30|, the ratio of two half-normal errors with scales 2σ and σ lies in [1.5, 2.5] with probability (2/π)(atan 1.25 − atan 0.75) = 0.16. If it is an estimated standard error, any estimator scales as 1/√n and a broken one passes. Repair: four disjoint 5,000-game fits from the 20,000; test their mean against 0.30 and their spread against σ_r/√2,500 ≈ 0.015 by χ² on 3 degrees of freedom.
- **Windfall slope.** Truth per position (16 against 16 luck-corrected playouts) has sd ≈ √(0.55·2/16) ≈ 0.26. If the critic's windfall value varies by 0.03 across positions at a fixed turn 4 (assumed), the slope's standard error is 0.26/(√1000·0.03) ≈ 0.27, wider than the ±0.2 band. Randomise the windfall's turn and size, and predict the spread before setting the band.

## 2. The κ gate cannot close, and gates another estimand

- At SE 0.13–0.25, a 2-SE upper bound below 0.3 needs κ̂ below +0.04 to −0.20; a move critic with true κ = 0.2 passes 95% of the time. Against a record where every learned critic failed, require the lower bound above zero, and gate on the calibrated spread κ·s, which sets the switch gain.
- κ is not a ceiling on β: the slope of truth on E[Â_C | h] equals κ only if the critic's error is uncorrelated with what the player observes. The band β/κ ∈ [0.5, 1.0] is not a theorem.
- The truth is a contrast under V_S, priced with the remaining coins; student and switches read the critic at ∅. Only a side-game's last coin has ∅ after it (1 in 12). Fit κ on last coins and on the rest separately; a gap means the ∅ extrapolation fails. So "a coin's information costs nothing extra" is false: tripled density shifts states (S0(d) predicts up to −0.14 per side) and puts 11 of 12 coin contrasts off the deployed estimand.

## 3. P0 blames bugs for truncation

Two-point Gauss–Legendre is exact for J′ cubic, so J of degree 4 (one better than the file says). But J(t) has degree equal to the disagreements on a trajectory, 20–40 per side-game, and stops in one turn compound: the k-th act needs k consecutive "continue" picks, a t^k term. A P0 miss can be truncation, which the kill line ("coin logging, side sign or propensity is broken; stop") cannot tell from a bug. Repair: plant first with one candidate coin per side-game (J linear, any node exact); then the two-node test as a measurement of interaction order, with a three-node read as the truncation check.

## 4. π_1^raw's known answer is not an estimand

A_stop correlates with L, so −α/γ is not the best constant boundary; that is a separate fit on α + γL alone. The band [0.5, 2.5] is not from the record: −2.5 and −1.5 are level and 0 is about 229 Elo worse, which leaves 0.5–1.0 unmeasured and does not exclude boundaries beyond 2.5. Predict [1.5, 2.5] for the constant-only fit, or measure −3.5 first. Uncapped changes also restore the interactions the cap of 10 bounded; the audit then carries all of it.

## 5. The budget breaks if S0(a2) fails

Under eval numerics a game costs $1.3e-4, not $2.5e-5. The path generates about 270,000 games (S1 70,000; S2 ~20,000; S3 40,000; S4 2 × 70,000): +$28, +$57 if the memory recompute must also run at eval numerics, plus $7.5 for S0 at four times. Total $88–116 against $85, with no rerun. The critic forward's $3.6e-8 is `relset`'s serve rate without memory; the critic adds 128 memory tokens and the fogged units, unmeasured. Price the a2-fail branch and say which stage drops.

## 6. Compounding cites single-agent theory

Policy iteration converges as Newton's method in MDPs. In zero-sum Markov games naive policy iteration "is only provably convergent under very restrictive assumptions" ([Winnicki & Srikant 2023](https://arxiv.org/abs/2303.09716)); van der Wal (1978) on Pollatschek–Avi-Itzhak's non-convergence is from search summaries, unchecked. Under fog one step is not guaranteed either: changing π shifts the posterior of the true state given the memory, which Ĝ_plug cannot see (the student was fitted under π_k's posterior; Ĝ_plug − Ĝ_old measures shift in memories only). S4's "Ĝ_plug at most the previous round's" and the two-to-three-round decay are therefore not derived. Name the audit and cross-round matches as the only guards, and give S4 a number extrapolated from round 1's audit.

## 7. Stability: contradicting evidence exists

The deep-BSDE survey says global-in-time losses "may have difficulty in converging" on long horizons ([Hu & Laurière 2023](https://arxiv.org/abs/2303.10257), read in a search excerpt). Kapllani & Teng's per-step form was shown on tens of steps; a game has about 160 chance events. On the value level the joint objective's gain over the record's failed critics is 1/ρ_C ≈ 1.3–1.8 times the effective sample, and the record says more games did not help; the rest must come from chance contrasts. S1 should predict ρ_C(U_0) − ρ_C(joint) on the same games.

## 8. Novelty already covered

- **κ and β** are the BLP slope of [Chernozhukov, Demirer, Duflo & Fernández-Val](https://arxiv.org/abs/1712.04802) (Econometrica 2025): Y on (D − p)(S − E S), weights 1/(p(1−p)), β₂ = Cov(s₀, S)/Var(S), 1 for a perfect proxy, with data splitting; GATES are the dropped per-decile slopes, and their β₂ = 0 test is the gate.
- **The identity and loss** are Off-policy DAE's ([Pan & Schölkopf, ICLR 2024](https://arxiv.org/abs/2402.12874), Eq. 10–11): return = V(s₀) + ΣA + ΣB, B = V(s′) − E V(s″), squared residual minimised subject to Σ_a πA = 0 and E_{s′}B = 0, known transitions as their oracle (the CVAE exists only because p is unknown), uniqueness needing an exploratory behaviour policy. With coins as nature, claim 3 is that loss with exact transitions and B tied to the critic. New: simulator enumeration, double-sampled tails, a privileged state for a fogged player.
- **Schedule in the state:** [Shani, Efroni & Mannor 2019](https://arxiv.org/abs/1812.05551) fold a fixed exploration mechanism into a surrogate MDP, nearer than NGU.

## 9. Smaller

- The deployed head averages two students, each calibrated alone; averaging cuts noise and raises the slope. Calibrate the average on a third split.
- I11 admits the calibration fold's post-coin states into the student's training: post-treatment covariates of the games it is calibrated on. Exclude each coin's own game.
- Dunnett minimises per-decision contrast variance, but none is estimated (one draw per decision); coins enter pooled slopes of variance σ_r²/(N·E[Var_p Q]). Choose p to maximise E[Var_p Q].
- "Linear baseline about 0.82": the record's pooled seven-column ratio is 0.848, `gen_own1_selfplay` 0.833 (`training/metrics/match_luck_20261019/readout.txt`). Use S1's own games' figure before the 0.05 margin.
- No ruling is broken: every contrast is one decision followed by a re-deciding continuation, and no evaluator is hand-written.
