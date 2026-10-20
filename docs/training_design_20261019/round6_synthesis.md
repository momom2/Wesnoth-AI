# Round 6 synthesis

## Verdict

The design changed in substance this round, but not at its core. The coin label, the linear head on the frozen network and adoption by match are round 5's. What changed, by the criteria the round is judged on:

1. **An assumed number was replaced by a measurement, and the measurement changed decisions.**
   - Round 5 priced a match game at $1.49e-4: 17 minutes per 800 games, the seed's figure.
   - The records give $2.51e-5 for a one-network match in `run_elo_batch` at 120 jobs, the configuration the generation-speed rounds measured (`round6_provenance_prices.py`). The arm's log confirms that this is the match driver.
   - The cheaper price changed four decisions:
     - adoption and the strength claim become one PURE paired match, since columns would save only $0.14 a match;
     - matches grow to 32,000 games (SE 1.79 Elo, against 2.57);
     - a model-free interim match replaces the model-based early stop;
     - the budget leaves room for three rounds and a miss reserve.
2. **Stage order changed.**
   - S0 now carries two things before any coin batch: DPI's admission reading on recorded games, and a known answer for the match path (attack offset +1, predicted −78 ± 16 Elo, through the same per-seat hook as the candidate).
   - S1 has an interim match after batch 3.
3. **Three derivation errors were fixed.** Two were found by the provenance stance and its critique, one by this synthesis.
   - The constant intercept per ordered pair type broke the pair-swap invariant and was misspecified at ties.
   - α* = −G1/(2G2) took G1 from labels that estimate g + A at the midpoint, not g.
   - The band-exhaustion rule (forecast upper bound under 2 match SEs) needed a forecast below −5.6 Elo and would almost never have fired, so the plan would never have reached its reset by its own rule.
4. **No new core mechanism; no fatal flaw.** DPI is a conditional variance component that cannot bias. The exhaustion error would have stalled the plan after round 2, but it did not threaten correctness.

## What changed, and why

### From provenance (critique: repairable)

**Kept:**
- the measured prices: the pool at $2.51e-5 a game and the eval driver at $1.17e-4;
- the full-cell label baseline (0.688, so σ² is 3.67, not 3.91);
- antisymmetric intercepts with margin slopes;
- the interim match, in place of the progressive stop;
- the exchanged-logit distillation target;
- the seed-shared mirror null;
- the attack-offset known answer, at its unselected value −78 ± 16 (the critique's refit on stage-1 games).

**Repaired as the critique asked:**
- **Attack known answer.** It now runs through the per-seat hook the candidate uses, not through a decode offset.
- **Mirror null.** It is read as a divergence census by gap bin.
- **Sign checks.**
  - The duel invariant runs through the production generator with the coin on each seat.
  - A new invariant ties every logged D_i = 1 to the command in the game record.
  - The sign check becomes a report, not a gate.
- **S4's prediction** now prices the off-switch band disagreements with μ̂.
- **P-distill** reads the end_turn logit shift at held-out positions instead of a tempo band borrowed from another signal.

**Not kept:**
- **The miss branch** (the mixture match with g, A and C, then α*). g and A carry ±15–16 Elo at 2 SE, C conflates two causes, and α* is undefined at Â ≥ 0.
- **p = q̂/2 for round 2.** q depends on the unknown μ, and no readout makes the attribution the rule needs.
- **Corrected matches.** They lose to PURE pairs at pool prices.

### From simplify (critique: repairable)

**Kept:**
- one PURE paired match as both the adoption test and the claim. Its 0.847 is reproduced, and at pool prices it costs $0.14 more than a corrected match;
- the sign rule with λ chosen by held-out loss, with no calibration slope;
- Ĝ_pf as the only forecast;
- fallback E;
- the γV term in the label;
- one corrected result per game;
- the removal of the value head at fight children;
- the per-decision luck split;
- the critic and branch L kept as options rather than built.

**Repaired as the critique asked:**
- **P-coin.** It is booked with the mover's side and the coin read from the game record by separate code, and read per side, so a sign error reads −1. Its prediction is stated for (T′ − T)/δ and for the ridge reading δ·uᵀ(G + λ̂I)⁻¹Gu. δ is set after S0 measures Var(u·x).
- **Early stop.** Replaced by the interim match; with the stop counted, the bound a null leaves is unchanged within 0.0002 (`round6_synthesis_power.py` §3).
- **P-distill.** Its known answer no longer assumes full closure.
- **Fallback E.** Its threshold uses each student's own half-quantile.
- **S2's outcomes.** Every case now has an action: they are split by the match's lower bound and by Ĝ_pf's lower bound.
- **Pair-aware scheduling** in `run_elo_batch` is listed as new code.

**Not kept:**
- **Dropping S0 entirely.** S0(a) now carries DPI's admission, and S0(b) carries the match path's known answer.
- **Scoring caps at z = 0.** That conflicts with the user's 2026-08-17 ruling that a capped game is no result. Labels drop caps, as the yardstick does, and each batch tests that capping is independent of the coins.

### From invent (critique: repairable)

**Kept:** DPI as the labels' control variate, exactly unbiased for any price fixed before the batch it labels, and identified by the dice without confounding.

**Repaired as the critique asked:**
- **Admission line.** It is set by worth, not significance: r_DPI ≤ 0.602 saves a batch per round (`round6_synthesis_power.py` §5).
- **Basis.** It nests the seven columns and includes the value head's scalar.
- **The regression plant P-DPI** (tautological or biased) is replaced by exact centring tests: the probability-integral transform of outcomes against their laws, with a Monte Carlo spot check, and the luck sum's independence from the coins, with a negative control.
- **S0's replay cost** is corrected to $0.42.

**Correction found in this synthesis.** The fit regresses one result per game, so its sample size is games, not fights. The admission reading therefore needs the 18,800 walked games: at 1,600 games, r·d/N would be about 0.45 of the variance.

**Not kept:**
- **The lottery column.** It conflicts with the play ruling, its law reads the true state through fog, and it has no known answer.
- **A full-token reduced-rank price at S0.** It overfits at 18,800 games.
- **Retroactive relabelling, trait draws and status effects.** Deferred until DPI is admitted.

### From challenge (critique: repairable)

**Kept:** the dice as a randomized experiment, used as the corrector. This is the same object as invent's DPI; the critique calls it MIVAT, credited in the design.

**Not kept:**
- **X as a decision score, and the real-against-anti arms.** Randomization identifies the slopes of ψ across outcomes, not its level across candidates. The arms are not mean-zero: the expected exchange favours the attacker in 0.749 of fights (`round6_challenge_critique_exchange_sign.py`). And X is a learned form of the prior attack bias the user retired on 2026-07-16.
- **The 320–1,400 ratio as an argument.** It compares two different estimands.

**Smaller errors the critique found:**
- P-dice's band should be ±0.096, not ±0.136.
- The null's per-switch line was misread.

### New this round (synthesis)

`analysis/round6_synthesis_power.py` (`.out` beside it) reuses round 5's stated Gaussian model with the round-6 inputs. Its derivations:

1. **Matches.** PURE paired matches cost $0.32–1.29 for SEs of 2.83–1.41 Elo; 32,000 games give SE 1.79 for $0.80.
2. **Floors.** The round-1 floors at that adoption size are sd 0.0016 (μ 0) or 0.0034 (μ −0.0018) at d_eff 776.
3. **The interim match after batch 3.**
   - P(stop) is 0.50 at a null. With the stop counted, the null bound is sd 0.0016 or 0.0036, against 0.0016 and 0.0034 with no stop.
   - Placing it after batches 2–5 trades $1.25 of expected null spend per batch against at most 0.0004 of the bound.
4. **DPI's admission line.** Seven batches hold the floors at r ≤ 0.602 and six at r ≤ 0.516. If ruling (b) is "no", 1.225 times the coin games are needed.
5. **Budget.**
   - $73.58 in all, with $1.61 reserved for E3 and E.
   - Three rounds fit iff price × (1 + overhead) ≤ $2.92e-5 a coin game, which allows 16% logging overhead at the measured price but not the 1.41× host spread.
   - A null round 1 costs $9.07 or $22.42.
6. **The interaction term.** It cannot be read from the coins. Products of two coin scores estimate the switches' interaction with the opponent's coins without bias, but the SE is 38–93 Elo at 0.8–2.4 M coin games. Attributing a miss stays a question for matches.
7. **The exhaustion-rule correction** (verdict, item 3).

## Open questions

1. **Rulings for the user.**
   - (a) Are coin-label regressions allowed? Unchanged.
   - (b) Revised: control variates in labels only, the seven columns and DPI if admitted; matches are PURE.
   - (c) Revised: adoption and the claim by one PURE paired match in `run_elo_batch` at 120 jobs, departing from the standing 20-job, 800-game procedure.
   - (d) Unchanged.
2. **The spread v and the per-type means c and b.** Nothing on disk bounds them; S1 measures them.
3. **DPI's kept ratio.** Do token-conditioned fight prices reach r ≤ 0.602 where hand forms stop at about 0.69?
4. **Second-order and excursion loss.** How large is it at 19–38 switches with both sides coined at p = 0.25, and which of curvature or the opponent's coins dominates? The coins cannot say at this scale (synthesis §7).
5. **Distillation.** Closure on an exchanged-logit target, and whether the end_turn bias returns.
6. **Prices still unmeasured.** Logging overhead and host speed; the price of a two-network match in the pool.
7. **Seed-shared pairs.** Do common seeds within a pair keep less than 0.847? The first adoption match measures it, and the planned SE uses 0.847.

## Research targets for round 7

- **Excursion against policy effects.** Read the full text of [Qian 2025](https://arxiv.org/html/2502.13500v2) (§2.2 and its appendix A.2, τ(s) = β_s − (α_{s−1} + α_s)p) for a bound on the gap between an excursion effect at p and the effect of a deterministic switch. Also look for designs that randomize one side only, priced against half the labels per game.
- **Distilling a rule-modified argmax policy.** Closure when the target differs from the base at 5–15% of decisions ([Rusu et al. 2016](https://arxiv.org/abs/1511.06295); [Czarnecki et al. 2019](https://arxiv.org/abs/1902.02186)), still not re-read. And whether the factored head's end_turn bias returns ([Murray & Chiang 2018](https://arxiv.org/abs/1808.10006)).
- **Learned control variates in games.** Read DeepStack's evaluation section in the text: which value is evaluated at which chance nodes, and the 85% cut, so far known from search summaries only. Look for fight-local or parent-conditioned prices in backgammon rollout variance reduction.
- **What the frozen tokens express.** Evidence from probes of game networks on whether near-tie action-value differences are linearly decodable from token outputs. This is the design's main unknown, and the only one a record might bound before S1.
- **Interim placement under the measured S0 inputs.** Rerun §3b on the census, d_eff and E[R²] that S0(a) measures, before S1 starts.

## Unknowns that remain

| Unknown | Measured where | What relies on it |
|---|---|---|
| Spread v; intercepts c and slopes b per type; switched share q | S1's progressive fits | The round's gain, and nothing before S1. Floors are stated for each case. |
| DPI's kept ratio r_DPI (and in pairs) | S0(a), on the 18,800 walked games | Whether DPI is built; S1's batch count |
| `own1`'s band census, runner-up mix, d_eff, E[R²] with V | S0(a) | The power script's rerun before S1 (76.1, d_eff 200 or 776, 0.688 until then) |
| Logging overhead; host price | S0(c) | Whether three rounds fit (≤ $2.92e-5) |
| The pool's generation-numerics band | S0(b)'s mirror null | Its own tolerance; the diagnosis of a miss |
| Pair correlation with shared seeds | The first adoption match | Nothing: the SE is measured per match, and planning uses 0.847 |
| Second-order and excursion loss | S2 (match against Ĝ_pf); E's match | Round 2's forecast scale; the fallback |
| Two-network match price | S4's first match | S4's cost; bounded by the eval median |
| Distillation closure ρ̂; transfer of a planted shift | S4 | Band reset |
| κ_T | Only if chosen after a null round 1 | The critic column |
| Compounding, exploitability, human level | Each round's claim; none; none | Nothing |
