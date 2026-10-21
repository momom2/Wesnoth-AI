# Round 10, critique of the simplify stance

**Verdict: repairable.** The headline merger, "R is the head with a unit block and zero token weights", is false as the features are specified. As written, R0's known answer goes red on a correct processor, and L plays a rule with twice R's switches against R's sealed prediction. One feature definition repairs it. The other numbers hold or need minor fixes.

New scripts: `analysis/round10_simplify_critique_mixed.py` (cited as "mixed §n") and `analysis/round10_simplify_critique_claimstop.py`, with outputs beside them.

## 1. Provenance audit

| Number | Class | Check |
|---|---|---|
| `own2` +22.7 ± 4.35; coin plant, dice slope, I-log, census 77.6, pooled mean | M | `reading.json` and `readout_b_offline.txt`, read. Hold |
| R: F +21.3 ± 3.1, 2.19 switches; F by ridge; τ curve | M | Reran `round10_simplify_checks.py`: output byte-identical |
| One ridge "costs at most 3.6 Elo" | M, but only to λ 50,000 | Extended to λ → ∞: +19.1 to +19.3 ± 3.1 (mixed §4). **Holds at every ridge** |
| Typed cells z +0.39; shrinkage 0.857 / 0.988 | M; D | `round9_provenance_cells.out` line 10; 30,000/35,000. Hold |
| One `coin_p` per player | M (code) | `git show exp/near-tie-coins:tools/raw_player.py`: a single float |
| Prices: 20,254, 22,209, 6,521 an hour; $0.577 | M | `round9_simplify_labels.out`. Hold |
| Setup 22 min | 7 min M, **15 min guessed** | Enters four setups and the third slot's $2.77 margin; at 45 min it costs $1.15 of it. Measure at box A |
| Path costs $1.28 / $25.12 / $53.47 / $79.57; budget left $82.34 | D | Recomputed. Hold |
| Third slot iff o ≤ 0.040 | D | 68.37·o ≤ 2.77 holds, but the line omits the hook overhead o_h on S2, E and the claims, and the processor's time. Both are measured before the slot is decided, so only the formula is incomplete |
| Match SEs 1.79 / 3.07; false adoption 0.023; forecast SE 4.88 at q 0.5 | D | 347.4/√N; Φ(−2); 347.4·√(158.1/800,000). Hold |
| L's lines 7.2 / 3.57 / 0.34 / 0.92 | D | Hold for R's 2.19 switches, **not for the rule the specified hook plays** (§2.1) |
| Claim lines 9.4 / 11.2; false miss 0.042–0.053 | D | Reproduced by simulation (claimstop) |
| Clustering 0.583; 0.56× | M | 0.56 is the odds ratio (0.157/0.843)/(0.25/0.75). Holds |

No guessed number carries a decision except the staging minutes, and they are too small to move one.

## 2. What would make it fail

**2.1 R is not the head's unit block (measured).**
- φ(a) carries the one-hot when a is a recruit, so x = φ(a1) − φ(a0) is non-zero on every band pair with one recruit candidate: 4.85 per side-game (recruit>move 2.11, move>recruit 1.95, four small types; mixed §1). R was fitted only on recruit>recruit pairs of different units.
- R's values are identified only up to a constant per faction, since every row is a difference within one faction. The ridge sets each faction's sum to 0 (mixed §2), so on a mixed pair the sign of f = −w(u0) or +w(u1) is measured from an arbitrary origin.
- **In play:** through the specified hook, R adds 2.36 switches per side-game to its own 2.19, worth +0.2 ± 3.3 Elo first order. L would play about 4.55 switches. Its sealed value becomes +21.5 ± 4.5 with a miss half-width of 9.7, not 7.2. Its switch-rate report would read about 4.5 against 2.21, and its transfer ratio would be read on a rule that is half noise.
- **In R0:** fed the per-candidate block, the processor fits values that correlate 0.871 with R's (max difference 0.027; R's rms 0.022), with cross-fitted F +20.0 ± 4.5 (mixed §3). So R0's "R's 43 values to float tolerance" fails on correct code.
- **Repair:** define the block as the pair feature 1[both candidates recruit]·(e(u1) − e(u0)). The mixed part is measured worthless (−1.2 ± 3.3 when fitted jointly).

**2.2 R0's known answer is under-specified.**
- R was fitted on labels with b = 0, while the processor subtracts the frozen b(cell).
- The progressive loss is flat at its minimum: −273.186 at 5,000 against −272.985 at 10,000 (r9s §1). A change of label can therefore move λ̂.
- F is cross-fitted over 20 game-hash folds, while the processor emits per-match statistics.
- **Repair:** run R0 with b ≡ 0 and `readout_b`'s β, and add per-fold sums to the processor's output.

**2.3 The token block has no known answer.**
- L's token weights are zero, so L never exercises the 772 columns that carry everything beyond recruits.
- The merged plant is planted through the dumped x, so it recovers its signal whatever x contains.
- I-φ checks only the log-probability block.
- **Consequence:** a wrong token index, or an a0/a1 swap confined to the token block, passes every listed check. Its symptom is a null S1, read as "band exhausted": $22.79 spent and a wrong conclusion. This is the bug class the seed names.
- **A free exact check exists.** `actor_logits = actor_head(actor_ctx)` (`wesnoth_ai/model.py:383`) and `target_logits = q·kᵀ/√d` (lines 399–401). The actor head applied to the dumped actor-token difference must reproduce the actor-logit difference. The q⊙k block summed over d and divided by √d must reproduce the target logit. Run both on the generation dump and on S2's hooked seat.

**2.4 Removing the R² report is not free.**
- The smoke's 2,000 dumped `own2` games give the cross-validated R² of R's values on the token block at $0. The target is deterministic, so no labels are needed.
- Under (h) = no, that R² is the only reading, before S1's $22.79, of whether round 1 can bank the one measured spread. The proposal names this as its weakest point 3, then drops the measurement with "Cost: None".
- With T gone, L on `own1` is the only recruit evidence before S1. `own2`'s recruit rates moved toward R's preference for 16 units and away from it for 12 (`round9_provenance_own2_recruits.out`).
- **Repair:** keep the R² as a report, with no threshold.

**2.5 The claim stop is two-sided.** The miss is |D − A| > line, and the upper tail holds 0.005–0.023 of the 0.042–0.053 false misses (claimstop). There the stated reason, a gain that does not reach `own1`, does not apply. Stop on the lower tail only.

**2.6 Smaller points.**
- The variance formula assumes homoskedastic labels. T² scales with ω² ∈ {16, 16/9}, so use the sandwich form with Σ T²xxᵀ.
- "$79.57 against $86.87" sets a refit of the same band against a reset followed by a new band. The like-for-like saving is $3.10 through round 2.
- **What I checked and found sound:**
  - The exactness derivation: earlier luck terms are H_i-measurable, later ones have mean zero given their pre-fight state, and b is fixed per setup.
  - The expectation of the progressive loss.
  - The fog reading: the target is the true-state advantage averaged given x.

## 3. Novelty

- **Claim 1** (a finding used as its own known answer): S0(a) already played a known effect through the match path, and positive controls are standard practice. Only the source of the effect is new.
- **Claim 2:** the nearest work is LSPI ([Lagoudakis & Parr 2003](https://jmlr.org/papers/v4/lagoudakis03a.html), JMLR 4:1107–1149), which the seed cites, rather than LS-DQN. LSPI fits closed-form least squares on fixed features, off-policy with sample reuse, inside policy iteration. The difference is the target: an unbiased propensity-weighted Monte Carlo advantage on a two-action band, in place of LSTD's bootstrapped fixed point.
- **Claim 3** stands.

## 4. Strongest part

R0 is a $0 differential test of the production processor against independent code on 60,000 real games, with the merged plant on the same records. Once 2.1 and 2.2 are fixed, it is the cheapest known answer this design has had.

Two simplifications are sound:
- **Removing H1.** Students never play in generation and the futility stop stays rejected, so H1 could only change S2's step form, which S2 followed by E does from logs.
- **One ridge.** It holds at every λ.

## 5. Repairs, in order

1. Make the unit block a recruit>recruit pair feature (2.1).
2. Run R0 with b ≡ 0 and per-fold sums (2.2).
3. Add the head-identity check on the token block (2.3).
4. Keep the R² report (2.4).
5. Make the claim stop one-sided (2.5).
6. Measure staging at box A, and state the third-slot line with o_h and the processor's time.
