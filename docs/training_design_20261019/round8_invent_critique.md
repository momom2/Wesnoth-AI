# Critique of round 8, invent stance (the dice as a second experiment)

Verdict: **repairable.** The core is the current design's and is untouched. The dice positive control is sound. Chance-effect supervision (CES) at S4 is not informative as sized: its plant and its kept-ratio readings were sized on 300 to 360 times the data the stage processes, and two failure modes have no guard. Scripts: `analysis/round8_invent_chance_ledger.py` (rerun, output identical) and `analysis/round8_invent_critique_checks.py` (`.out` beside it, cited as §1–§3).

## 1. Provenance audit

| Number | Class | Check |
|---|---|---|
| hp slope +0.00416, range +0.00327 to +0.00513, χ² 24.8 / 22 | M | Reproduced from `rows.jsonl`. The window is the range of 23 estimates (SE 0.00044 each). A batch whose games are 20% mis-joined reads 0.8 × 0.00416 = 0.0033, at the window's edge, so the gate catches only gross errors. |
| Control SE 0.00029 (2,000 games), 0.00004 (batch); flipped −0.0042 | D | Correct. |
| Setup conditioning +0.0039 ± 0.0011 | M | Reproduced. It equals the overfit term r·d/N ≈ 0.847 × 70 / 18,000 = 0.0033, so the true gain is about −0.0006 ± 0.0011: a null, as stated. |
| Hit probability error < 3e-8 | D | True; the actual error is under 1e-9 (2³² mod 100 = 96). |
| Centring ratio 1.020 | M | Reproduced. |
| Floor ∝ √r; 2.29× at r 0.3 | D | The √r scaling holds: the spread estimator's null bound is v ≤ 2·SE ∝ s²/N (`round8_provenance_spread_estimator.py`). r 0.3 has no source, but no decision rests on it. |
| Dice share 0.843 ("all within-cell variance"); ratio 11,544 | D, overstated | At argmax, within-cell chance also includes recruit traits (seed §2; about 48 recruits a game, `rows.jsonl` counts) and the numerics flips (1.76%). The strikes' share is at most 0.843 and has not been measured. The ratio compares the variance each source causes, not what each tells the decision learner; the proposal itself concedes the dice cannot rank decisions. No decision rests on it. |
| κ 0.0005 [C]; plant SE 1.8e-5 over 8 batches | D, misapplied | The arithmetic is right at 800,000 games. The head trains inside S4's $2.1 distillation, which holds about 1,000,000 own positions (`memory_own_trajectories_prereg_20261011.md`, "Measured"), the fights of **2,227 games**. There the SE is 3.4e-4, so κ/SE = 1.5 (§1). A broken path, which reads 0, lies within 2 SE of κ, so the plant cannot tell working code from broken code. **Fatal as sized.** |
| Paired SE of a kept ratio 0.0005 at 100,000 games | D, unfunded | S4 forwards no held-out coin games. Forwarding 100,000 of them through N, with its memory carried, is 44.9 M positions: 32 h at the teacher pass's measured 390 positions a second, about $13.4 (§1). That is not in the $0.51. On S4's own 2,227 games the SE is 0.0032. |
| r_frozen, a ridge on frozen tokens (d ≈ 1,160) | D, unfunded | On 2,227 games it overfits by r·d/N = 0.36 (§1). The decision "neither below 0.688 → learned pricing closed" would then follow from lack of power, not from the truth. |
| Comparator 0.688 | M, misused | It is a constant from 1,600 `own1` games with the pooled cell share. Against new coin games the comparison is unpaired: the cell baseline's finite-sample term alone is 0.0105 at 100,000 games (`round6_provenance_cells.out`), several times the stated SE. |
| Gradient cap 0.1 | C | Not derived, as the proposal admits. The $2.10 control covers a closure miss, but "plant outside its window: distil without it" needs a second distillation that is not budgeted. |
| Head cost ≤ $0.51 | D | 108.8 / 449.1 × $2.1 = $0.509. It covers training only. |
| Total $55.50 / $56.01; three-round line $3.141e-5 | D, mis-cited | `round8_provenance_plan.out` reads $56.18 (S0 $0.14, a three-match E reserve $2.17) and $3.113e-5. The proposal mixes round 7's S0 and E with provenance's stages. No decision flips (overhead room about 0.63). |
| Per-strike head evaluation during round-3 generation | U, unlisted | If CES enters round 3's labels, every strike of 800,000 games needs ψ_k. That takes a new server path (tokens out, or the head over each fight's states) with unmeasured overhead. S5's cost of $17.52 assumes none. |

## 2. What else would make it fail

**A plant that passes on a mis-join.** z′ is built from the same (H, c) join the labels read; unlike the coin plant's s^rec and D^rec, nothing says it comes from separate code. If H is joined to the wrong strike, E[ω̃_k(H̃_k − c̃_k)] is still 1, so the plant reads κ while the real effects vanish. In the toy, the plant reads 0.0053 ± 0.0004 against 0.005 planted, while the context slope falls from 0.039 to −0.0003 against a true 0.04 (§2). The proposal's claim that "a mis-joined (H, c) reads about 0" holds only for a join error that the plant does not share.

**Post-strike inputs break exactness, and nothing catches it.** The core's strike record is (cth, hits, dmg, dies), with dmg 0 on a miss and dies set after the strike (`rust/wesnoth_core/src/combat.rs:437–446`). If the head reads "damage" or "kills" from that record, it sees the outcome. Then Σ_k (H_k − c_k)ψ_k is not centred, r_CES looks strong out of sample, and round 3's coin labels are biased: in the toy, +0.0075 ± 0.0016 against a truth of 0 (§3). The F-flag plant still reads κ. The design's I2 negative control ("a post-coin variate must fail") and I-V do not cover the CES column.

**Stages whose outcomes are not informative.** At S4's scale, the plant, r_frozen and "neither below" cannot separate truth from lack of power. Every outcome is informative only after the readings are funded.

**Complexity.** CES adds a head, a flag input, a z′ path, per-fight luck columns (the design logs only per-game sums), strike logging, ruling (f), a cap, a conditional second distillation and a generation-time head in round 3. Its own §5 says the record leans against its premise. Repaired, it costs about $14 and several code paths for a variance cut, which is a cost saving where the budget does not bind (round 7 deferred DPI and ACS on that ground).

**Novelty.** CES regresses an IPW transformed outcome, (H − c)/(c(1 − c))·R, on covariates with a known propensity. That is standard pseudo-outcome CATE learning (Athey & Imbens 2016, not re-read; surveyed in [Fisher, ICML 2024](https://proceedings.mlr.press/v235/fisher24a.html), [arXiv 2307.09700](https://arxiv.org/abs/2307.09700), which also note that this pseudo-outcome is noisy even when the propensity is known). Used as a control variate, it is [AIVAT](https://arxiv.org/abs/1612.06915)'s chance-node correction with a learned contrast in place of a value function. The combination may be new; the parts are not. The dice control is a standard positive control.

## 3. Repairs

1. κ ≥ 0.0014 (κ/SE ≥ 4 for the ideal estimator at 2,227 games; adds 0.0049 of Var(z), CES labels only).
2. z′ from a separate parse of the strike records, plus a real-effect check: per game, the CES join's Σ_k s_k(H_k − c_k)d_k must track `match_luck`'s S_hp.
3. Pre-strike inputs only, with I2's negative control and I-V extended to the CES column.
4. Nested against hand-only on the same held-out games; fund the replay (about $13.4) or drop r_frozen.
5. The per-strike head's overhead measured before CES enters round 3.
6. Simpler: defer CES to the $1,000 case beside DPI and ACS; keep the dice control now.

## 4. Strongest part

The dice positive control. It is free, measured on 23 player pairs with no excess spread, and reads a flipped result field at about 100 SE in every batch. That closes the coin plant's blind spot continuously, not once at S0. The underlying algebra is also right: for a binary chance event, the martingale increment is exactly (H − c)Δ, so any head fixed before a batch and fed only pre-strike inputs gives an exactly centred control variate.
