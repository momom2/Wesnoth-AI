# Round 5 critique of the invent stance (law supervision, exact-truth plants, the turn-over rule)

**Verdict: repairable.** Each added mechanism costs cents, and no guessed number funds a stage. But S0(j) adopts a hand-set rule that breaks the kept frame, and the record already prices most of that rule's effect. The student's stated mechanism and the gate's guarantee are wrong. Two of the new checks do not test what they claim to.

## 1. Provenance audit

I read and reran the three invent scripts and `analysis/round5_invent_critique_turnover_bounds.py` (mine); all four reproduce their `.out` files exactly.

| Number | Class | Check |
|---|---|---|
| Margin density 1.19 per nat per decision; keyed density 0.94–0.52; deviations 10.6–63.1 | measured (`parity3`) / derived | Reproduced. f_M(0) = E[f_L(G/T)/T] and P(deviate) = 1/(1 + e^{G/T}) are right. |
| Replay disagreement 1.79% | measured, misattributed | `prior_gaps.log` lacks the "inference bf16 autocast ON" line that `transformer_policy.py` logs at ea8536f, so the replay ran fp32 against the bf16 server's games: precision and batching confounded. "Batch arithmetic alone" is the provenance stance's S0(b). |
| d ≈ 0.030 nats | derived, model-bound | The provenance quantile model reads 0.0136 from the same record. Nothing depends on d: 477 flips in 100 side-games is 4.8 directly. |
| Coins ≈ 40× forks; ρ > 0.975 | cites a withdrawn figure | `round5_provenance.md` §6 replaced it by Σp(1−p) ≥ 0.61. Σp(1−p) comes from S0(c); 0.61 uses temperature-0.5 noise. |
| Turn-over continues where the offset ends: ≥ 1.0% | measured lower bound | Reproduced. The upper bound is **90.0%**, and the rule could end at up to **44.7%** of act-top decisions. The record brackets almost nothing; "mostly agrees" (§1.5) is unsupported. |
| 52.9, 67.8, 40.5 per side-game; 0.747; Match C; 5.3 Elo; $0.48; S1b $2; S2 ≤ $0.4 | measured / derived | Checked. S1b scales with outcomes per fight (9, unknown until S0(a)'s census, which runs first). |
| Provenance stage costs | misquoted | `round5_provenance.md` §3 prices S4 at $9.4 and S5 at $3, not $10 and $5. The budget is about $66, not $70, a conservative error. |
| Step-1 critic 0.518 against 0.380 | measured, selectively read | Paired: +0.137 ± 0.093 (read0), +0.046 ± 0.107 (pre). It is a turn benchmark and carries no decision. |

## 2. What would make it fail

**2.1 S0(j) adopts a hand-set rule early, and its readout is mostly predictable.**
- **The band is too loose.** "Within 2 SE" admits −10.6 Elo. Ruling (c) asks for an estimate ≥ 0, and it was posed for a learned stop. The ½ threshold is the 0–1 loss for predicting whether a human ends, not a strength optimum.
- **Swapping π₀'s stop before the coin games voids what is defined around −1.5:**
  - S2's known answers (Match C, Match A);
  - the band census;
  - the stop window L ∈ [−1.5, 4.5];
  - the forced paths.
- **The record prices the decision-count channel.** A dropped decision costs 0.0033 ± 0.0017 z and an added one gains 0.0002 ± 0.0010 (`round5_provenance_knob_switches.out`). So the rule can gain only through heterogeneity.
- **3,200 games cannot resolve that gain** unless √q·τ ≥ 0.0059 (`round5_provenance_stop_bound.out`). A null is likely and cannot tell "no heterogeneity" from "not captured".

Repair: generate at −1.5. Count the decisions the rule drops in S0(a)'s replay (P(end) is the actor head's end entry, which the record lacks), and forecast the count channel from the knob fit. Then enter the rule in S3's stop comparison, where 12,800 games resolve √q·τ ≥ 0.0038.

**2.2 The student's mechanism is misstated.** Â_C = Σ_o p_b(o|x)·C(x∘b∘o) − C(x∘a0) is already summed over the exact law by the critic, so it carries no result noise, and 0.747 is irrelevant. What remains is a bet on representation, with S0(h) as its prior and S2's A/B as its evidence.

**2.3 The gate's guarantee assumes exact gradients.** [Du et al.](https://arxiv.org/abs/1812.02224) prove Proposition 1 for steepest descent; their Atari runs smoothed the cosine with a 0.999 moving average. At a main-loss stationary point the minibatch gradient is noise ε. The gated update's mean, E[ε·1(⟨ε,a⟩>0)] + a·P(⟨ε,a⟩>0), then has a positive component along the auxiliary gradient a, so the fixed point moves. Where label noise dominates the critic's gradient, the gate acts like a fixed weight near ½. Either let the A/B decide alone, or smooth the cosine and test the claim with a planted auxiliary loss whose minimum conflicts with the main one.

**2.4 The sibling-transfer probe tests something κ does not depend on.**
- **What it reads.** φ is the same local function of the fighters, the terrain and the time of day whether an attack was played or not.
- **What κ needs.** κ fails if the trained critic misvalues off-policy children, which the probe never sees.
- **Its readings do not follow.** Label noise lowers both fits, so a gap is still shift.
- **R² is the wrong statistic.** Each set divides by its own variance, and unplayed siblings include attacks argmax avoids. MSE is the right comparison.

S1b fits both arms anyway. Remove the probe.

**2.5 The exact-truth plant is exact only for the immediate term.**
- **Why.** A working stream head at the children also predicts later ranged kills on villages. Through the builder, the slope is then 1 only if the head fails.
- **Restricted to the immediate term,** it re-checks law summation (I9, I-L1, I-L2).
- **A role-sign error passes** if truth and prediction share extraction code.

Make it an invariant against independently coded truth, or remove it.

**2.6 The law gates have no stated power.**
- **Two metrics each.** S1b gates on ρ̂ or κ_attack, S2 on attack R² or regret, so false adoption is up to 4.5%, not 2.3%.
- **The SE is computable in advance.** The paired SE of the κ difference is blind, computed from the logged p and the two critics' X, as the provenance stance does for κ.
- **A null would be uninformative.** Round 4 put the pilot's resolution near 0.15 in κ, so a null cannot tell "no effect" from "too few coins" (seed §6).

Repair: name one metric per gate and state the minimum detectable gain before running.

**2.7 §2.5 over-reaches.**
- **Most of it is on record.** Common random numbers were rejected "under any keying" on measurement (`docs/selfplay_redesign_20260904.md`, ✅5: a median of 0 matched downstream fights). Only keyed tie-breaking is new, and its rejection is sound.
- **"Paired matches" must mean shared dice.** The provenance stance's side-swapped setups share no draws and remove a measured 0.151 of the variance.
- **Drop** 40×, ρ > 0.975 and "batch arithmetic alone".

**2.8 Complexity.** Law heads on three networks, each with gating, normalisation and four invariants, plus two probes. All of it buys a gain the memo calls approximation only. The S5 law target has no readout.

## 3. Novelty

- **Law supervision.** Training-only, engine-computed auxiliary targets for a fogged agent are prior art:
  - [Gehring et al. 2018](https://arxiv.org/abs/1811.08568): the full StarCraft state;
  - [Lample & Chaplot 2017](https://arxiv.org/abs/1609.05521): enemy presence;
  - [Mirowski et al. 2017](https://arxiv.org/abs/1611.03673): depth.

  This is learning using privileged information (Vapnik & Vashist 2009, not re-read). Exact chance laws at every legal stochastic action are weakly new.
- **The turn-over diagnosis** is NMT's brevity problem. [Murray & Chiang 2018](https://arxiv.org/abs/1808.10006) trace it to label bias and fix it with a per-word reward tuned by perceptron, the offset's analogue. I found no parameter-free stopping rule of this kind (search not exhaustive).

## 4. Strongest part

The keyed-noise analysis (§1 item 1, §2.5):
- it is measured and it reproduces;
- the gap density is nearly flat near 0 (1.24, 1.19 and 1.15 per nat below 0.02, 0.05 and 0.1), so smoothing barely lowers it;
- at T = 0.25, keyed tie-breaking cuts flips by only 31% while deviating 40.7 times per side-game.

It closes a deferred mechanism without a run.
