# Round 7 synthesis

## Verdict

The design changed in substance this round, but not at its core. The coin label, the closed-form head on the frozen network and adoption only by match are round 5's. No proposal brought a core mechanism that survived its critique, and no fatal flaw was found in the round-6 design. What changed, by the criteria the round is judged on:

1. **A misread measurement was replaced, and decisions moved.**
   - Round 6 priced a pool game at the whole wall of a 600-game arm, $2.51e-5. A long batch pays the steady-state rate, $1.912e-5. The steady rate is read between the 120th and 480th completions, with three sub-windows and two other arms agreeing (`round7_provenance_price_tail.py`, `round7_provenance_critique_checks.py` §1).
   - **The host is no longer a gate before S1.** At the old price, a 1.41× slower host did not fit three rounds. At the steady price it does, with up to 9% logging overhead. So the S0(c) smoke stage folds into batch 1.
   - **The plan costs $57.67 of $85,** leaving $27.33, which buys 1.25 more reset-and-round cycles (`round7_synthesis_power.py` §5).
2. **Stage order changed.**
   - S0 loses DPI's $0.42 admission reading and the mirror null, and gains the label-code reading of the attack known answer.
   - The S0(c) smoke moves into batch 1.
   - The P-distill run merges into S4's distillation: the baked end_turn shift is that run's own plant.
   - Round 2 still precedes the reset. Its reason is now the user's 2026-10-15 ruling: no network training before the optimizer discussion. Provenance's rule for skipping round 2 was rejected.
3. **Mechanisms removed:**
   - seed-shared pairs, pair scheduling, I8 and the mirror null;
   - cross-fitting;
   - DPI from round 1;
   - the separate P-distill run;
   - the kind indicators in φ;
   - the d_eff 200 floor case;
   - scaling round 2's forecast by round 1's ratio.
4. **A check added, and an error in it fixed.**
   - The coin plant cannot see a result field oriented to the wrong side; the simplify critique showed this. S0's attack games are now also read through the label code, with the hooked seat a logged coin.
   - The critique predicted −0.225 z for that reading. The correct value is 2Δ = −0.449 ± 0.092: the coin moves the hook from one seat to the other, so the label reads the effect twice (§"New in this synthesis").
   - With the critique's window [−0.329, −0.120], a correct label path would have failed its gate.

Measured against the user's standard, the round's main gain is fewer code paths for the same guarantees. Correctness gates now cover every code path, the label orientation included. The main unknown, the near-tie spread v, is still unmeasured, and no record on disk can bound it.

## What changed, and why

### From provenance (critique: repairable)

**Kept:**
- The steady price and the overhead room. A worker's 39.3-ms cycle spends 33.2 ms waiting on the server, so the walk costs at most 1.8% and the per-game statistics under 0.1%.
- The deferral of DPI, tagged [C] as the critique asked, with no "at most" on its saving. DPI saves $1.91 per batch dropped: one batch at r ≤ 0.602, more below. That saving does not bind inside $85, while DPI adds a label-path code path.
- The two-network price for the post-reset claim.
- Dropping the d_eff 200 case.
- The √N ceiling, as mathematics. For near-tie advantages of any shape with nonnegative mean, the sign rule's first-order gain grows at most as √N in the labels (`round7_provenance_scaling_bound.py`, 16 shapes). The critique confirmed it holds for the posterior-mean rule.

**Repaired:**
- **Regime default inverted.** The ceiling is stated only where every margin bin's mean label over round 1 has a lower 2-SE bound ≥ 0. It then serves as a sealed upper bound in round 2's preregistration (0.414 × round 1's gain), not as a stage gate.
- **S0(a)** reads V's paired reduction against the columns, not a refitted full cell: 2,000 games give about 1.3 per cell.
- **The interim's error** is stated in both regimes: ≤ 0.041 where the premise holds, up to 0.15 under the Gaussian model at μ −0.0018.

**Not kept:**
- **D2,** which skipped a same-band round 2 below 8.6 Elo of round-1 gain.
  - Its premise is the default, and batch 1 cannot test it. Margin slopes of −0.001 to −0.004 per nat break the ceiling, yet batch 1 flags them with P 0.13–0.37 (`round7_provenance_critique_checks.py` §2).
  - "Cannot certify" was in fact P ≤ 0.21–0.44 (§4).
  - Round 2 runs by default instead. The reset falls under the optimizer ruling, which settles the order without the premise.
- **The exact mirror null.** The critique asked to gate it on digests alone. The synthesis drops it with the pairs: with independent games, nothing relies on determinism. Its gate also cited a non-reading ("none resolved at G ≥ 0.35"), and digest checks are standard lockstep desync detection.
- **The S3-null option "twice the batches in the convex regime".** Only the Gaussian model forecasts it, and option (i), w = 0.7, covers widening.
- **The 32,000-game two-network claim** after a reset. 12,800 games give SE 3.07 for $1.50 instead of $4.42 [C].

### From simplify (critique: repairable)

**Kept:**
- **Independent matches.** 37,800 games keep SE 1.79 Elo for $0.11 more a match at the steady price (`round7_synthesis_power.py` §1). This removes pair-aware scheduling, I8, the mirror null and the pair-correlation unknown.
- **Forward-chained frozen coefficients.** Labels are computed once, and I4 holds by construction, for 0.15 of a batch (`round7_simplify_power.py` §2). The critique's citation is checked: [Hadad et al. 2021](https://arxiv.org/abs/1911.02768) §2.1 builds m̂_t "based on the history H_{t−1}" (ar5iv text read).
- **The plant read on T′ − T:** SE 0.0051 per side, with δ free and no prior Var(u·x).
- **The smoke folded into batch 1.**
- **The baked end_turn shift** as the distillation path's plant.
- **d = 772.**
- **Each round's own Ĝ_pf.**

**Repaired:**
- **I2 restored** (critique F1). Simulated, the plant reads 1.006 against a flipped result field while real labels change sign. S0(b) is also read through the label code, with the prediction corrected to 2Δ.
- **I-V** (F2) is read once a round at a 5% family-wise line (3.4–3.8 SE). A 2-SE rule on every type and batch fires on correct code with P 0.976–0.997.
- **The reset** (F3).
  - The target lowers end_turn by π_k's own offset, not a fixed 1.5.
  - The distilled network plays at the residual o_k(1 − ĉ), derived from ĉ measured on held-out positions before the match. It is −0.27 at ĉ 0.82 and shrinks geometrically over resets for ĉ in (0, 2) (`round7_synthesis_power.py` §4).
  - The critique's other point is applied to ruling (d): the closure price for offset 0 comes from unselected stage-1 games (−9.9 ± 4.7 Elo at ĉ 0.82), the sample S0's known answer uses.
- **γV kept** as one frozen coefficient. It is admitted iff S0(a)'s replay reads its reduction above 2 SE, and fitted there, since the replay computes V at every decision.

**Not kept:**
- **Dropping E.** E stays as one rule for both the interim and adoption, with a two-match reserve ($1.45). It responds to the one failure the plants cannot exclude, second-order loss, with a derived prediction: half the switches keep 0.80–0.87 of the first-order gain, and the quadratic loss falls to a quarter. It costs a $0.72 match.
- **Census from batch 1's logs only.** S0(a)'s $0.04 replay lets the sealed floors and costs rest on `own1`'s measured band before $16 is committed.

### From invent (critique: repairable)

**Kept in the record, not in the plan:** antithetic chance siblings (ACS). Each fight is replayed once with reflected draws, and the network prices the reflected child. The correction is exactly centred for any price, and its capture is (1 − ρ)/2 ≥ ½ for monotone prices.
- ACS saves a batch a round only if a child price explains 0.25–0.28 of Var(z) beyond the columns. All hand prices together explain 0.152, and board conditioning adds 0.042 (`round7_invent_antithetic.py`, `round6_invent_conditioning.py`).
- Like DPI, it changes cost, not guarantees, and the budget does not bind. It is deferred to the $1,000 case, read in one replay with DPI (about $0.57).
- The critique's repairs are recorded for revival:
  - terminal and single-action children priced by one function;
  - a Φ = identity plant;
  - admission on the measured cost multiple;
  - resized batches.

**Not kept:** known-switch calibration (H, M₁, T).
- Its readings can fire together, and no outcome changes a decision.
- The midpoint gap at −78 Elo is 0.31 Elo under a logit-linear response (`round7_invent_critique_checks.py` §4).
- The Differences-in-Q estimator ([Farias et al. 2022](https://arxiv.org/abs/2206.02371), abstract read) covers its mathematics.
- The $3.59 it spent was the coin-overhead allowance, not spare money.
- Its main purpose, a real-sized known answer for the label path, is now met at no cost by reading S0(b) through the label code.

### From challenge (critique: repairable)

**Kept:**
- The multi-arm label is correct (E[T_c | H] = Q̃(c) − Q̃(0); the plant slope is exactly 1).
- Intents are recorded as S3-null option (v), with the critique's repairs:
  - pre-roll E-step scoring;
  - natural-forward features tested across arms;
  - S1's null gated on P-EM's closure;
  - a budgeted placebo arm;
  - P_0 set after the diversity read.

**Not kept: intents as the route to run first.**
- **Its motivation fails on its own record** (`round7_challenge_critique_bench.py`).
  - Sampled temperature-1 turns score −0.055 ± 0.031 against the base turn.
  - Split-half selection among attack-free turns reads −0.035 ± 0.040, against +0.066 predicted by a Gaussian model with the stance's own moments.
  - The only positive kind is single-decision "continue" edits (+0.049 ± 0.028). The (end, best act) pair type already covers that.
- **Fitting codes is network-side training,** under the 2026-10-15 ruling.
- **A per-turn choice needs a ruling** against "the network picks actions, never whole turns".

### New in this synthesis

`analysis/round7_synthesis_power.py` (`.out` beside it):

1. **The label-code known answer.** Each game carries one logged coin, D = 1 iff side A carries the hook, with P ½ and ω = ±2.
   - E[T] = E[z_A | A hooked] − E[z_A | B hooked] = (m + Δ) − (m − Δ) = 2Δ = −0.449 z. The prediction SE is 0.092 and the label SE 0.025 at 6,400 games, so the window is [−0.640, −0.259].
   - Simulated with four seeds at 4 M games each: −0.4491 to −0.4497, SE 0.0010 (§3).
   - A result oriented to the wrong side reads +0.449; half the games flipped read 0.
2. **Budget at the steady price, with independent matches:**
   - S0 $0.18; S1 $16.02; S2 $0.72; E reserve $1.45; S3 $17.46; S4 $3.60; S5 $18.24.
   - Total $57.67.
   - Three rounds fit iff price × (1 + overhead) ≤ $3.05e-5 (overhead room 0.60), and two up to $4.76e-5.
   - On a 1.41× host the plan costs $79.20, and three rounds fit at up to 9% overhead.
   - A null round 1 costs $6.64 stopped at the interim, $16.92 run through, $11.78 expected.
3. **E as one rule:** the same trigger and action at the interim and at adoption.
4. **Ruling (e), new:** whether a closed-form head fit on frozen features counts as training under the 2026-10-15 ruling. If it does, nothing past S0 runs before the optimizer discussion.
5. **Branch L re-sized to independent games:** 3,800 a match, SE 8.0 per arm-minus-placebo, about $4.35.
6. **S4's base acceptance at SE 3.07** accepts a 2.8-Elo deficit with P 0.18 [D: Φ(−2.8/3.07)]; the paired SE gave 0.16.

## Open questions

1. **Rulings (a)–(e).**
   - (a) Coin labels pooled by one linear head: unchanged.
   - (b) Revised: the luck columns and V_i as label-only control variates; DPI is no longer asked for.
   - (c) Revised: independent 37,800-game matches; after a reset, a 12,800-game two-network claim.
   - (d) Revised: a residual offset o(1 − ĉ), and whether it counts as "gone".
   - (e) New: the closed-form head under the optimizer ruling.
2. **The spread v, the typed constants c and slopes b.** Nothing on disk bounds them; S1 measures them.
3. **The sign of the margin-bin means.** Does every bin certify a nonnegative mean, so that round 2's preregistration can state the √N ceiling? The record leans against it.
4. **Second-order and excursion loss** at 19–38 switches with both sides coined. Can the DQ estimator's bias bound give it a size before S2?
5. **Distillation.** Does a baked logit shift distil linearly enough for the first-order residual o(1 − ĉ)? What is the closure on a target that differs from the base at the switched decisions?
6. **Prices still unmeasured:** the server's feature gather under logging, and the pool's two-network mode.
7. **γV.** Does the value head reduce the label residual beyond the columns? S0(a) reads it.

## Research targets for round 8

- **What the frozen tokens express.** Probes of game-playing networks (chess and Go probing work, e.g. [McGrath et al. 2022](https://www.pnas.org/doi/10.1073/pnas.2206625119), not re-read) on whether small action-value differences are linearly decodable from token outputs. This is still the design's main unknown, and the only one a record might bound before S1.
- **A bound in the negative-mean regime.** Find a shape-free bound for the sign rule with a precisely estimated negative typed mean, the regime the record suggests, starting from the convex-concave value of Gaussian information ([Keppo, Moscarini & Smith 2008](https://cowles.yale.edu/node/145025)). Round 2's preregistration would then carry a bound there too.
- **The DQ estimator's Theorem 1.** Read its conditions and constants ([Farias et al. 2022](https://arxiv.org/abs/2206.02371)), to see whether the second-order bias bound gives a usable size for a miss at the design's scale.
- **Distilling a rule-modified argmax target.** Closure when 5–15% of decisions change ([Rusu et al. 2016](https://arxiv.org/abs/1511.06295); [Czarnecki et al. 2019](https://arxiv.org/abs/1902.02186)), still not read. And whether additive logit shifts are reproduced linearly by a distilled student.
- **For the $1,000 case only:** the capture of antithetic pairs under non-monotone learned prices.

## Unknowns that remain

| Unknown | Measured where | What relies on it |
|---|---|---|
| Spread v; typed constants c and slopes b; switched share q | S1's progressive fits | The round's gain; nothing before S1. Floors are stated for μ 0 and μ −0.0018. |
| `own1`'s band census, runner-up mix, feature spectrum; V's reduction | S0(a), $0.04 | The power rerun before S1; γV's admission |
| Server feature-gather overhead; host speed | S1 smoke | How many rounds fit (≤ $3.05e-5 for three) |
| Margin-bin means (the ceiling's premise) | All of round 1's labels | Whether round 2's preregistration states the ceiling |
| Second-order and excursion loss | S2 against Ĝ_pf; E's match | The fallback; round 2's reading |
| Re-pricing under π_1 | S3 | Round 2's value beyond the ceiling |
| Closure ĉ, share ρ̂, spillover | S4, before its match | Base reset and its residual offset |
| Two-network pool price | S4's match | S4 and S5 costs, bounded by the eval median until timed |
| κ_T of a privileged critic | Only if chosen after a null | The critic column |
| r_DPI; ACS capture on `own1`'s prices | The $1,000 case's replay | Nothing in the plan |
| Compounding, exploitability, human level | Each round's claim; none; none | Nothing |
