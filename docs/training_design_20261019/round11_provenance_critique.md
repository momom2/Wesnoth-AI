# Round 11 provenance stance: critique

**Verdict: repairable.** The update is a sound estimator: on-policy REINFORCE with the luck and cell control variates, a KL-normalized step, and certification by importance sampling on held-out games. Four faults need repair before the discussion:
- stage C's "a miss is a bug" line uses a critical-batch formula that is wrong for a KL-normalized step;
- the cited loss is not the score of the policy that plays;
- per-step adoption power is not derived;
- the plant bypasses the outcome path.

The optimizer section also inverts one of its sources.

## 1. Provenance audit

**Measured, checked.**
- Census (475.7 ± 3.2 decisions a game, 237.9 a side-game, 20.16 side-turns, kind shares, 2 of 6,402 capped, 36.0 departures): `round11_provenance_census.py` rerun, every digit reproduced.
- Turn signal (I_turn 0.0204, R² 0.346, T1 turn −0.068): `round11_provenance_turn_signal.py` rerun, reproduced. I_turn uses only 64 of the 199 positions.
- Prices: $2.598e-5 a pool game and $8.848e-5 a claim game, `round9_simplify_labels.out`.
- σ_R² at argmax: 0.6988 ± 0.0029, `round9_simplify_residual.out`.
- Trainer speed: 1,477 positions/s, BACKLOG, measured on the pass's six largest windows.
- Code facts: BPTT window 16 (`tools/sequence_setup.py`), smoothing 0.05 (`wesnoth_ai/imitation_loss.py`).
- Memory reading: +13.0 ± 12.3, CLAUDE.md.

**Derived, redone.**
- Correct:
  - training at $5.16e-5 a game, 1.99× generation;
  - 347.4 Elo per unit of z;
  - false adoption 0.023;
  - the fork factor (1 − R²)K/(K − 1), for a leave-one-out baseline;
  - N* = sqrt(c_f B/c_g), under McCandlish's trade-off;
  - B_nat ≥ rank(F), by Cauchy–Schwarz;
  - the IS SE, to second order, with the Rényi-2 divergence ≈ 2K.
- **Wrong:** B_P = σ_R² tr(PF)/(gᵀPg) (§2.1).

**Choices carrying decisions.**
- **|H| = 2,500** sets the adoption power, and through it E's cost and stop. It is not derived (§2.3).
- **"Two misses at N\*: double N once, then stop"** has no stated meaning.
- **"Temperature 1 lowers R²"** is asserted, not derived. Nothing is spent on it, so it is not fatal.
- **B, I_net and the length of a T1 game** are measured before money depends on them. This ordering is right.

## 2. What would make it fail

### 2.1 Stage C's prediction uses the wrong noise scale

The step is d = Pĝ scaled so that ½s²dᵀFd = K. Its first-order gain is

    sqrt(2K)·gᵀPg / sqrt(gᵀPFPg + tr(PFPΣ)/N),

so the curve sqrt(N/(N+B)) holds with B = tr(PFPΣ)/(gᵀPFPg). With Σ ≈ σ²F this is σ² tr((PF)²)/(gᵀPFPg). The proposal's formula is McCandlish's B_simple in the P metric, which assumes isotropic curvature there ([McCandlish et al. 2018](https://arxiv.org/abs/1812.06162)). The two agree only at P = F⁻¹, which the proposal excludes.

For SGD with an evenly spread g they differ by tr(F²)·dim/tr(F)². That factor is large for a dispersed Fisher spectrum. Correct code can therefore miss C2's curve and be called a bug. Repair: estimate d_iᵀFd_j across the accumulators (Fisher-vector products, or the KL along d_i + d_j).

### 2.2 The loss in §1.1 is not ∇ log π

The sequence trainer's `imitation_loss_parts` takes a softmax over every non-pad actor, hex and weapon, with no legality mask. Its target row depends on the actor alone. The player samples from `enumerate_legal_actions_with_priors`, which masks each factor, uses separate attack and move target masks, and renormalizes over the legal joint.

For a masked factor, E_π[∇log q(a)] = ∇log q(L), which is not zero. The signed-weight loss therefore carries a term E[(V(h) − b)·∇log q(L)] that moves parameters without moving π. C0 would catch the mismatch, but the claim that the code exists is false. The repair is a masked joint log-probability from the own-trajectory path's mask columns (`own_policy_loss`, `tools/sequence_inputs.py`). It is unpriced.

### 2.3 Adoption is underpowered, and E's price ignores it

The chosen candidate is read on half of H, 1,250 games.
- At K = 1/4 and σ² = 0.70 the SE is 0.0135 z (4.7 Elo). Half the time, a step must truly gain about 9.4 Elo to be adopted (11.2 at σ² = 1).
- With a gain of sqrt(2K·I_net·cos²), that requires I_net·cos² ≥ 0.0015 at every step. The signal-to-noise ratio does not depend on K, so the ladder buys no power; only |H| and N do.
- N* minimizes cost per unit of progress; it does not make a step certifiable. A working update gaining 3 Elo a step is adopted 9% of the time, and "two misses, then stop" can end it.
- The price (Δ_s/ΔĴ)(N*c_g + c_f) assumes every step is adopted and that gains add. Divide it by the adoption probability and label it an extrapolation.

Repair: after C, set N and |H| jointly from the measured ΔĴ(N) for a power of about 0.8.

### 2.4 The plant does not test the outcome path

The plant adds δΣ(f − E_π f) beside z, and its readout is free of outcome noise. It never passes through σ_s, the z join, β·S or b. An inverted seat sign on z leaves D1 and D2 green. The CI toy game covers the estimator, not the production records.

Repair: plant through the result, as S0(a) did. For example, flip z with a probability that rises with the side's recruit count.

### 2.5 Costs and estimand

- **KL search.** Setting each candidate's s to its KL target needs memory forwards over held-out games, each as costly as a certification forward. At two to five trials, C's candidate cost is $2.8–7.1, not $1.42.
- **Inconsistent counts.** "Eleven candidates" does not match the listed grid (up to 14). C is priced at $2.5 and at $2.75.
- **Caps.** A cap scores −1 for both sides, but `elo_collect` treats caps as absences. Certified J then rewards fewer caps, which the yardstick never sees, and adds a cooperative term. No line says at what T1 cap rate this scoring must change.

### 2.6 Literature

- **Ruan et al. is inverted.** [Ruan et al. 2026](https://huggingface.co/papers/2607.16169) report Muon raising success from 0.290 to 0.546 against AdamW in sparse-reward agentic RL. The paper runs no matched-RMS comparison: each optimizer is tuned over its own learning-rate range. This is contradicting evidence for "Muon predicted to lose", from the setting closest to this update.
- **Fan et al. supports the proposal.** [Fan et al. 2026](https://arxiv.org/abs/2605.19282) report Muon "collapses to zero" in RLVR. The record is split.
- **The bench claim is false.** "The nearest is Ruan et al.'s RMS-matched control" fails for the same reason.
- **Certification is partly covered.** [POIS](https://arxiv.org/abs/1809.06098) already optimizes a per-batch trajectory-IS objective bounded through the exponentiated Rényi-2 divergence, the e^{2K} term here. The new part is narrower: chance-law control variates inside IS, with a held-out split.
- **The LessWrong post** figures check (0.2 nats, cosine −0.945, about 7× fewer steps at about 19× the tokens), and its regime is correctly set aside.

**Minor.**
- **The default contradicts its own argument.** By the regime argument, SGD, not diag(F)^{−1/2}, should be the default. C decides either way.
- **The indicator does not cohere with the chance share.** I_side ≈ 0.41 per side implies the choice variance takes about 0.8 of E[z²], while at argmax chance takes all of it. It carries nothing, but "99.8% noise" leans on it.

## 3. Strongest part

The order of work, and the identity under it. The martingale ceiling shows one game holds at most σ_R² of signal, so the optimizer is subordinate to the noise regime. B, the one number everything rests on, is measured first, for about $3.6, with an honest stop.

The certification is unbiased under fog and memory for the reason stated: the dice's law does not depend on the policy, so the luck terms keep zero mean under any reweighting, and the opponent's ratios are 1. Nothing beside the network decides play, and nothing chooses whole turns.
