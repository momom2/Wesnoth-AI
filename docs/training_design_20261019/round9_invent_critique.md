# Critique of round 9, invent stance: chance-coupled twins

**Verdict: repairable.** No guessed number carries a decision: ρ is read by a $0.35 pilot before any batch is sized, and every T0 outcome has an action. Two defects make the component fail as written: a sibling-mean bug that biases every label passes the stated plants and reads as a twin gain, and batches are sized on a residual ratio that is not the head's noise ratio. Checks: `analysis/round9_invent_critique_checks.py` (*checks*), output beside it.

## 1. Provenance audit

| Number | Class | Check |
|---|---|---|
| `own2` +22.7 ± 4.4; S0(a) −0.242 ± 0.023 z, −0.118 | M | `own2_base_20261020/progress.txt` (6,400 decisive), seed §7 ✓ |
| Pooled +0.0003 ± 0.0007; r>r +0.0070 ± 0.0024; four cells z 5.8–4.2 over 3.72 | M | `round9_invent_cells_summary.out` ✓; all four pass 4.07 too |
| V_rr 1.592e-3; A's forecast +20.2 ± 3.1 | D from M | `round9_provenance_recruit_cells.out` §2, §7 ✓ (19.1–22.9 across ridges) |
| Prices, $82.34; 16 of 28 units | M | `round9_provenance_budget.out`, `..._own2_recruits.out` ✓ |
| $77.46 plan, $4.88 left, totals at r | D | arithmetic ✓; stage list differs from the regimen (§2.4) |
| r = 1 − ρ²(K−1)/(1+(K−2)ρ) | D | ✓ |
| The head's noise scales by r | D, asserted | not derived; fails (§2.2) |
| Hit correlation 0.667 | D | ✓ (`combat.rs:436`); per draw position only (§2.5) |
| 2.667, 0.0226, 0.667, 0.785 | D from M | `round9_invent_twin_prefix.out` ✓ |
| 0.701, 0.680 ⇒ r ≥ 0.97 | M ⇒ D | inputs ✓; r is 0.9975 at K 4 (§2.3) |
| Leaders −0.0002; 0 shared seeds; Q8 | M | `round9_invent_setup_share.out` ✓ (12,000 games, no SE; "+0.0005 on 22,000" has no output); archive lines 201–212 ✓ |
| Pilot SE, P(certify), shared plant r′, plant SE 0.019 | D | *power* §2–3 ✓; binary-latent model on z, not R |
| A's window [11.3, 29.1]; SE 3.07, 1.79 | D | ✓ |
| K 4, the 0.95 line, a 0.5 | C | the 0.95 line's stated saving is wrong (§2.4) |
| ρ, κ, J/Ĝ, v on `own2` | U | read before use ✓ |

## 2. What would make it fail

**2.1 A sibling-mean bug passes every check and certifies twins.** If the sibling mean includes the game itself, κ fits to 1 and every label shrinks by 1 − 1/K: 0.074 against a planted 0.10 at ρ 0, 0.078 at ρ 0.2. r̂ reads 0.749 at ρ 0, so T0 certifies twins on the bug (*checks* §1). The checks miss it:
- the coin plant as worded ("z′ only in a game's own residual") reads 0.997, since the sibling vector stays unplanted;
- the shared-component plant passes if ρ̂ is inverted from r̂ (0.598 measured, 0.616 predicted, 1.6 pilot SE);
- I-V would need to resolve Δ/K, about 0.002 on r>r, roughly one batch SE.

Repair: plant into every twin's z (the bug reads 0.747); ρ̂ from pairwise covariances in separate code (0.598 against 0.914); I2, the enumerable duel, on the twin label, a self-included sibling required to fail.

**2.2 r̂ is not what sizes batches.** §2.3 calls cross-twin label covariances second order. Given the shared chance U (setup, keyed dice), the term is E_U[Δ_k(U)Δ_j(U)]: each twin's realized response to its own coin under the same dice, not the square of the small mean effect. It peaks in the shared prefix, two-thirds turn-1 recruits, where the spread sits. OSCC, which the stance cites, says that with branch-local policy randomness gradient noise moves with "policy-Jacobian-weighted off-diagonal return covariance", not return-contrast variance ([abstract](https://arxiv.org/abs/2609.33543)). Toy (*checks* §2, ρ 0.3, one prefix coin): r̂ stays 0.832 while the true variance ratio is 0.832, 0.849, 0.882, 0.944 as the realized effect's sd across shared chance goes 0, 0.2, 0.3, 0.5. That sd is unmeasured.

Repair: T0 reads the seed-clustered variance of the head's statistics per pair type against batch 1's, and S1 is sized on it; every SE in twin batches (forecast, welfare pre-test, plants, I-V) is clustered by seed. The stance's side check, the type means' SE ratio, is the right sizing quantity.

**2.3 The exact fork is 2.3% of the game.** Twins share 0.0226 of their commands; afterwards their coins are independent at every band decision, about 19 deviations per side-game each. "One label gives the paired contrast" is false: the pair's difference carries every later coin difference. The fork's value lies wholly inside ρ, and Q8 points low. The luck columns already remove the linear part of each twin's luck, so ρ must be read on R (*power* §2 simulates z). The pessimistic bound is misstated: ρ 0.030 gives r 0.9975 at K 4; 0.97 is the limit as K grows.

**2.4 The costs do not add up.**
- The totals ($78.88 … $54.08; $4.88 left) carry A as provenance's L+T plus a two-network claim, $1.85; the regimen's A is one 12,800-game arm, about $0.40. Headroom is understated by about $1.45.
- "Saves ≥ $3.10" at the 0.95 line compares twins at r 0.95 with twins at r 1. Against the plain plan, r 0.95 saves $0.68 a round: $0.33 on round 1 after the pilot, $1.69 over three. Break-even r is 0.966 for round 1, 0.977 for three (*checks* §3). Rounds 2–3 need an adoption first.
- S1's "$21.07 × r̂ + start-ups" counts the start-ups twice. Best case (r 0.8): $11 over three rounds.

**2.5 Complexity precedes the measurement.** T0 needs only twin coin keys, seeds shared across the K matches and an offline correlation. The label path (join, κ, two plants, I-twin, twin I-V, seed clustering) and the ordinal keys can wait for a certified r̂. The ordinal key changes every salted sim's RNG, so it must be gated off the match and search paths; the stance does not say so. Fights at the same (t, s, n) couple only on aligned strikes: when strike counts or firststrike differ, a draw passes to the other striker.

**2.6 Novelty.**
- (ii) is RLOO: the mean of the other K − 1 returns as an unbiased baseline for every score term of a sample ([Kool, van Hoof & Welling 2019](https://mlanthology.org/iclrw/2019/kool2019iclrw-buy)); uncited.
- (iii) is OSCC's branch-local randomness; OSCC also certifies a coupling by a lower confidence bound with fallback to independent sampling, T0's structure.
- (iv) is wrong: the counter stream is equally exogenous, so Mao et al.'s condition holds under both keyings, and §2.2's proof never uses the keying. Ordinal keys change alignment ([Buffalo et al. 2026](https://arxiv.org/abs/2603.11084)).
- New: the application to near-tie coins at argmax. [Sakib](https://arxiv.org/abs/2609.24144)'s abstract names no leave-one-out baseline; the stance's description is unverified.

**2.7 Smaller.** I-twin's digest equality must allow numerics flips across matches. The pilot's games are S1 data, so at r 1 only start-ups and setup (about $0.15) are lost, not the pilot.

## 3. Strongest part

§2.2's argument: a coin drawn under its own key is independent of every sibling, so any function of the siblings is an exactly centred control variate, with no value function, for a project whose critics never beat material. T0 is the right shape: $0.35, a zero-code arm, a fallback to the plain design, no r predicted or relied on. Section 7 is read honestly: knob-derived sizing is dropped after S0(a) read half its effect.

## 4. Repairs, in order

1. The coin plant into every twin's z; pairwise ρ̂; I2 on the twin label.
2. Size on the seed-clustered variance of the head's statistics; seed-cluster every SE.
3. T0 with twin coin keys only; the label path and gated ordinal keys after r̂ certifies.
4. Restate the budget (A as one arm), the 0.95 line's saving against the plain plan, r ≥ 0.9975, and the novelty (RLOO, OSCC).
