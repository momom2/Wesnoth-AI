# Round 7, critique of the provenance stance

**Verdict: repairable.** The price, the budget and the ceiling's algebra hold. The ceiling's premise does not: the design assumes it by default, batch 1 cannot establish it, and the record points the other way. D2, the interim's stated error and the exhaustion reading all rest on that premise. Two smaller readings misquote their sources.

New checks: `analysis/round7_provenance_critique_checks.py` (`.out` beside it). The stance's three scripts rerun with identical output.

## 1. Provenance audit

| Number | Carries | Class | Check |
|---|---|---|---|
| $1.912e-5 a game, steady | every cost | M | Holds. The three sub-windows read 21,600, 21,600 and 22,737 games/h, within the 1-s quantization (checks §1). "All 120 slots busy" is wrong: 113 in flight on average, minimum 100. The rate is still the one a long batch gets. |
| Stage costs, $61.59 / $44.46, null $6.63 / $16.80, cycle $26.43 | plan | D | Recomputed by hand; they match. |
| Worker cycle, walk ≤ 1.8%; σ² 3.67, SE 1.79, η_k, bin SEs | overhead, power | M, D | Match their sources; arithmetic correct. |
| √N ceiling | interim, D2, exhaustion | D, given a premise | The derivation is correct, and it holds for the posterior-mean rule at μ ≥ 0 (checks §3: g16/g8 ≤ 1.36). The premise is effectively guessed (§2.1). |
| Interim P ≤ 0.041 | S1's error statement | D | Conditional on that premise and on first order. |
| D2 line 8.6 Elo | stage order | D | The arithmetic holds, but "cannot certify" misreads it (§2.2). |
| DPI saving "at most $1.91–3.82" | deferral | presented as D | Not derived (§2.4). |
| Flip rates; "none resolved at ≥ 0.35" | the mirror null's gate | presented as M | Misread (§2.3). |
| ~57 identical decisions | null coverage | D | From parity3's cross-box rate, not the pool's same-box rate. Labelled as such. |
| −78 ± 16 / ±33; kv23 −0.240 ± 0.097 | S0(b) | M via D | Arithmetic correct. |
| E[R²] read by S0(a) on 2,000 games | replaces 0.688 | M planned | 2,000 games give about 1.3 games per cell over 1,512 cells (`round6_provenance_cells.out`). The full cell cannot be refitted there. Only V's paired reduction against the pooled 0.688 is readable. |
| "The host's line" in S0(c) | host choice | unstated | Give the value: $2.89e-5 × (1 + overhead), per `round7_provenance_budget.out`. |

## 2. What would make it fail

### 2.1 The ceiling's premise is the default, and batch 1 cannot test it

The bound needs E[a | noise scale] ≥ 0. The intercepts c_uv + b_uv·G are unpenalised, so the premise applies to each margin bin's mean m(G). Continuity gives m(0) = 0 for same-kind pairs. If the logit gap carries any information, the slope is negative, and the stance says so itself ("away from ties the argmax's runner-up is likely worse").

The record points the same way:
- The one near-tie reading on disk, extra acts at the end/act margin, is −0.0018 (−0.0061, +0.0017) per act (`round5_provenance_stop_acts.out`).
- Argmax beats sampling by +412 ± 97 (seed §3). This shows direction only; it is not specific to near-ties.

Take m(G) = b·G with the band's margin density, the per-bin SEs and an in-span sd of 0.0015–0.002 (checks §2):

| b per nat | Band mean | P(batch 1 flags any bin at −2 SE) | Posterior-mean rule | Sign rule |
|---|---|---|---|---|
| −0.001 | −0.00016 | 0.13 | g8/g3 1.81–2.06 (ceiling 1.633) | 1.94–3.04 |
| −0.002 | −0.00032 | 0.19 | g16/g8 1.50–1.68 (ceiling 1.414) | 1.65–3.53 |
| −0.004 | | 0.37 | g16/g8 1.65–1.85 | |
| 0 | | 0.09 (false flags) | | |

So the test misses the slopes that already break the ceiling about four times in five. When it misses, the design declares the "nonnegative regime" and applies a bound that is false. At b = −0.002 and sd 0.002, the same-band increment is 0.50 of g8, not ≤ 0.414. D2's line falls from 8.6 to about 7.2 Elo, and to 5.5 at b = −0.004.

This is a guessed premise carrying a stage-order decision. Batch 1 does not settle it, because it measures at a resolution that cannot. Nothing becomes falsely claimed, since adoption stays a PURE match. What goes wrong is how money and nulls are read.

### 2.2 D2 states a probability as an impossibility, and weighs a bound against an unknown

- **"Cannot certify" is wrong.** At the ceiling increment, π_2 is adopted against π_1 with P ≤ 0.21, 0.35 and 0.44 at round-1 gains of 5.1, 7.0 and 8.0 Elo (checks §4; the stance's own `scaling_bound.out` §3 gives 0.20 and 0.44). The line 8.6 is where that probability reaches 0.5.
- **The ceiling ignores re-pricing.** It also excludes re-pricing under π_1, which the stance calls unbounded.
- **The alternative's value is unknown.** S4 + S5 costs $26.43 for a gain that depends on ρ̂ and on the reset span's spread, both unknown. "S4 before S3" therefore compares a partial bound with an unknown.

That is a choice [C], not a derivation. It should be stated as one, with what it costs in each regime.

### 2.3 The mirror null's gate rests on a non-reading, and the method is standard

- **The source does not support the gate.** The cited source says of G ≥ 0.35: "the record neither confirms nor bounds zero flips at G >= 0.35". Its coefficient is −0.0089 (−0.0174, −0.0009), a misfit of an additive model on 100 side-games (`round5_simplify_critique_extra.out` §1). "None resolved at ≥ 0.35" turns that into evidence of rarity.
- **It is a different setting.** The rates are cross-box for parity3. The pool's numerics are same-box, with varying batch composition, which nothing has measured.
- **Fix.** Make the digest match the only gate, and report the gap census without a threshold.
- **What the null certifies.** With identical players, a seat or sign bug that is consistent across both games of a pair still gives z₂ = −z₁. The null certifies determinism and seed pairing only; the attack known answer carries the rest.
- **Not new.** State checksums with a first-divergence search are how deterministic lockstep games detect desyncs. Examples: Bettner & Terrano 2001, [1500 Archers](https://www.gamedeveloper.com/programming/1500-archers-on-a-28-8-network-programming-in-age-of-empires-and-beyond) (checksum detail not verified in the text); Spring's running replay checksum to find "the last agreeing point" ([Spring wiki](https://springrts.com/wiki/Syncing_System)); Wesnoth's own OOS checks. Novelty claim 3 is a use of this, not a method.

### 2.4 DPI's saving is not capped

The 0.602 and 0.516 lines are the first two steps of ⌈8r/0.688⌉ batches (`round6_synthesis_power.out` §5), not a ceiling. At r = 0.34, DPI would save four batches, $7.65 a round. r_DPI is unknown and S0 no longer reads it. Deferring DPI is a defensible complexity choice, but it should be tagged [C] and the "at most" removed.

### 2.5 Checked and immaterial

Labels in one game are uncorrelated only to first order (z² = 1 kills the leading term; S and V do not). Student 3's realised gain varies about 13% around its expectation at d = 776 (√(η₃/d)/sd(a)), small next to the match SE.

## 3. Strongest part

The price re-measurement: reproducible, stationary across sub-windows, consistent over three configurations, and it cuts the plan by 1.31× with the budget reproduced to the cent. The ceiling is also correct mathematics with its failure shown honestly; where its premise holds it is a valid shape-free bound, including for the ridge rule the design uses (checks §3).

## 4. Repairs

1. **Invert the regime default.** Apply the ceiling to a bin only when that bin's lower 2-SE bound is ≥ 0. Otherwise use no bound, rather than the Gaussian one. On the record this will rarely fire, so state that the shape-free reading will usually be unavailable.
2. **Recast D2 as a choice [C].** Say what skipping S3 forgoes at b ∈ {−0.002, −0.004}, and either run S3 by default when round 1 is adopted or ask the user.
3. **The mirror null.** Gate on digests only and report the gap census. Retitle it a determinism check and credit lockstep desync detection.
4. **DPI.** Tag the deferral [C] and drop "at most".
5. **S0(a).** State that it reads V's paired reduction, not a refitted full-cell residual.
