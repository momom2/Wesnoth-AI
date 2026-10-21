# Round 9, invent stance: chance-coupled twins

Tags: **[M]** measured (record and script named), **[D]** derived (inputs named), **[U]** unknown (measured before anything relies on it, or nothing relies on it), **[C]** a choice whose consequences are derived. Scripts in `analysis/`: `round9_invent_cells.py` with `round9_invent_cells_summary.py` (*cells*), `round9_invent_twin_prefix.py` (*prefix*), `round9_invent_setup_share.py` (*setup*), `round9_invent_twin_power.py` (*power*, §1–§4), each with its `.out`.

## Summary

1. The core stays: near-tie coins with logged propensities, the closed-form head on the frozen network, adoption only by a PURE match.
2. **New component.** Each seed is played as K = 4 twins: same setup, dice keyed by event ordinal, independent coins. Each twin's coin labels are centred on its siblings' mean residual: an exactly unbiased control variate that needs no value function. At argmax the twins are one history until their first coin disagreement, which falls at a turn-1 recruit choice in 0.667 of pairs [D from M, *prefix*]: an exact fork at no snapshot cost.
3. **Section 7 moves three decisions:** the base is `own2`; money binds (the plan leaves $4.88 [D, *power* §4]); spread is measured in `own1`'s recruit near-ties, so a cheap label-to-play test runs before round 1's money.
4. **The twins' correlation ρ is unknown.** A $0.35 pilot, which is also `own2`'s smoke and S1 data, measures it. At a measured variance ratio r, the coin rounds need r times the games for the same floors: $66.48 instead of $78.88 at r = 0.8 [D, *power* §4].
5. **Weakest point.** If play is chaotic once twins part, they recover only what the cell baseline misses of the setup, r ≥ 0.97 [D from M], and the component is dropped for $0.35.

## 1. What section 7 moves

- **Base.** `own2` beats `own1` by +22.7 ± 4.4 Elo [M]: π_0 = `own2`. Claims against `own1` become two-network matches ($1.13 each [D, `round9_provenance_budget.out`]).
- **Attack known answer.** The labels predicted the match they came from (match mean −0.118, half the label's 2Δ of −0.242 ± 0.023 [M]); the decode-knob records overstated the effect twofold. No stage below sizes anything from decode-knob matches; every forecast comes from labels.
- **Mean zero, spread present.** The pooled label, +0.0003 ± 0.0007, reproduces exactly from the game records [M, *cells*]. Of 251 pre-coin cells with ≥ 20,000 labels, four pass the family-wise 5% line, z 3.72 [M, *cells*]:
  - recruit pairs whose runner-up is a type the side has fewer of: +0.0235 ± 0.0041 (z 5.8);
  - runner-up a type not yet recruited: +0.0194 ± 0.0042 (z 4.7);
  - recruit>recruit in side-turn 1: +0.0195 ± 0.0046 (z 4.2);
  - all band decisions in side-turn 1: +0.0173 ± 0.0041 (z 4.2).

  The provenance stance reads V_rr = 1.592e-3 ± 3.70e-4 and a cross-fitted forecast of +20.2 ± 3.1 Elo per side-game for additive recruit values [M, `round9_provenance_recruit_cells.out` §2, §7]. So v is above zero for `own1`'s recruits. On `own2` it is unmeasured: its recruit rates moved toward those values in 16 of 28 units, away in 12 [M, `round9_provenance_own2_recruits.out`].
- **Price.** $2.583e-5 a one-network game, 35% above the design's $1.912e-5 [M, `round9_provenance_budget.out`]; the three-round path costs $77.46 of $82.34 [D, *power* §4]. Round 8 deferred variance cuts because the budget did not bind. It binds now.

## 2. The component: chance-coupled twins

### 2.1 Construction

- **Twins.** Seed g is played K times [C, K = 4]. Twin k draws coins under the key (domain, g, k, side, index); twin 0 keeps today's key (`tools/coin_decode.py`, branch `exp/near-tie-coins`). Setup and leaders follow from g.
- **Ordinal-keyed dice.** The seed of the n-th attack (or recruit) of side s in turn t is sha256(salt : kind : t : s : n)[:8], in place of sha256(salt : request counter) (`tools/wesnoth_sim.py:1093–1105`; salt `elo:<seed>`, `tools/elo_eval_game.py:464`).
  - Keys stay distinct within a game, so each game's law is unchanged. Wesnoth takes any per-command seed, records store each seed, and the unsalted export branch is untouched.
  - A strike hits iff draw % 100 < cth (`combat.rs:436`). Two strikes of chances c₁, c₂ fed one draw correlate at (min(c₁, c₂) − c₁c₂)/√(c₁(1−c₁)c₂(1−c₂)): 1 at equal chances, 0.667 at 0.4 against 0.6 [D]. Different fights at the same (t, s, n) are coupled; no identity match is needed.
  - A zero-code arm keeps the counter key.
- **One match per twin index,** so no inference batch holds a game with its sibling, and batch-dependent arithmetic cannot carry one twin's coins into another's result.

### 2.2 The label

    R_k = z_k − βᵀS_k − b̃(cell),   X_k = R_k − κ · mean_{j≠k} R_j,   T_i = s_i ω_i X_k(i)

ω_i = (D_i − P_i)/(P_i(1 − P_i)) as now; β, b̃, κ are fitted on earlier batches and frozen (κ for S1's first batch from the pilot).

**Claim:** E[T_i | H_i] = Q̃(H_i, a1) − Q̃(H_i, a0), exactly. **Proof.** The design's argument (`current_design.md` §3.2) covers R_k. Sibling j is a function of the setup, the keyed draws, twin j's coins and its own batch-mates; coin D_i of twin k comes from a key independent of all of these, its propensity fixed by H_i. So E[(D_i − P_i)R_j | H_i, sibling j] = 0. ∎ Caps are dropped as now; a sibling's cap depends only on its own coins.

### 2.3 Variance

- **Ratio.** Exchangeable twins have one common residual correlation ρ. At the optimal κ, r = E[X²]/E[R²] = 1 − ρ²(K − 1)/(1 + (K − 2)ρ): 0.914 at ρ 0.2, 0.625 at 0.5 for K = 4 [D, *power* §1].
- **Why labels improve by r.** ω_i is independent of the siblings and only weakly dependent on its own residual, so E[T_i²] = E[ω_i²]E[X²] to first order in the coin effects, and the head's noise covariance Σ_g E[X_g² A_g A_gᵀ] scales by r in every direction. Cross-twin label covariances are mixed second differences: second order.
- **Coupled-rollout caveat.** [Sakib 2026](https://arxiv.org/abs/2609.24144) and [OSCC](https://www.alphaxiv.org/abs/2609.33543) show coupling can lower return-contrast variance yet raise gradient variance; there the score sums a whole trajectory's sampling, here each score is one coin the sibling never saw. The pilot checks: the type means' SE ratio must read √r̂.

### 2.4 Exact forks without snapshots

Twins whose coins agreed so far are the same game, memory included, up to numerics flips (1.76% at gaps < 0.014 [M, carried]). Over batch 1's coin logs [D from M, *prefix*], the first disagreement falls at band index 2.667 (geometric at P 0.25); both twins replay 0.0226 of the game's commands before it; it is recruit>recruit with probability 0.667 and in side-turn 1 with 0.785. There the pair is an exact fork at the decision type where spread was measured, and afterwards both arms draw the same numbers at each (t, s, n). One label gives the paired contrast and the coupling of later luck. No state-and-memory snapshot is needed, which is the path round 8 held against prefix-branched coin games.

**What it cannot do.** If one recruit of difference scrambles the rest of the game, ρ is only what twins share of the setup beyond the cell baseline: that baseline's estimation noise (0.701 against a within-cell asymptote of 0.680 [M, `round9_provenance_label_noise.out`]) and the two leaders (−0.0002 of Var(z − βᵀS) [M, *setup*]). So r ≥ 0.97 [D]. Q8 found a median of 0 matched fight identities after single turn edits under a collapsed checkpoint [M, `docs/archive/credit_assignment_design_20260817.md`]; it never measured the correlation of results, and ordinal keys need no identity match. ρ is [U].

### 2.5 Checks of the twin code path

| Check | Prediction | Failure reading |
|---|---|---|
| I-twin: first band decision's coins across twins | Agree at 0.625 (P² + (1−P)²), χ² as I-coin; turn digests equal up to the first disagreement | Shared coin keys: agreement 1 |
| Coin plant with twin labels (z′ only in a game's own residual) | Slope 1 per mover side, SE ≈ 0.019 [D: 0.0387 × √(2,000/8,000)] | A shared coin key reads 1 − κ |
| Planted shared component: every twin's z shifted by 0.5·η_g in a copy, η = ±1 keyed by seed | Ratio moves to r′(ρ̂, 0.5): 0.914 at ρ 0, 0.774 at 0.2; simulated recovery 0.915 ± 0.011, 0.776 ± 0.018 [D, *power* §3] | Sibling joined to the wrong seed: r′ ≈ 1 |
| I-V on the twin column: mean of s ω κ R̄_{−k} per type | 0 within the 5% family-wise line | A post-coin leak |
| Dice control | The design's window, SE clustered by seed | The law changed |

## 3. The algorithm

- **Learned:** the current design's head, f = c_uv + b_uv·G + ŵᵀx on the frozen token outputs, fitted by ridge regression: one episodic natural policy-gradient step for the coin policy (`current_design.md` §1, ruling (a)). It switches where f exceeds the tested threshold τ̃, is adopted only by match, and is distilled at the reset.
- **Labels:** coin labels, twin-centred. Each compares one decision followed by argmax play that re-decides after every roll; the network picks every action.
- **Signal per dollar.**
  - A coin game carries about 155 orthogonal experiments (77.6 band decisions per side-game [M]); a per-decision fork pays a continuation for one contrast.
  - Learned critics never beat material in the record (seed §3); twins need no value function. CES needs a head, a distillation and a funded replay (round 8); twins need one seed function.
  - Twins cost 0.0226 of each game replayed [D, *prefix*] and K start-ups of 69 s a batch [M, `round9_provenance_budget.out`], and buy the same floors at r times the games.

## 4. Regimen

| Stage | Cost [D, *power* §4] | Readout and prediction | Decision per outcome |
|---|---|---|---|
| **T0** twin pilot, `own2` smoke | $0.35 | 1,000 seeds × 4 twins × two keyings (ordinal, counter), `own2`, coins w 0.35, p 0.25. r̂ cross-fitted per keying, SE 0.0023 at ρ 0, 0.0122 at 0.2 [D, *power* §2]; §2.5's checks; the design's smoke gates for the new base. No r predicted: no seed is shared between any two of 53 archived and 23 walked match records [M, *setup*] | A check fails: debug. An arm's upper 2-SE bound below 0.95 [C; saves ≥ $3.10 over three rounds]: S1 takes the lower-r arm (counter on a tie: no simulator change), batches of 100,000 × r̂_upper games. Neither: no twins, S1 as designed. Certifies ρ ≥ 0.2 with P 0.80 |
| **A** label-to-play test, same box | $0.36 | `own1` with the provenance stance's additive recruit values against `own1`, 12,800 one-network games, SE 3.07. Forecast +20.2 ± 3.1 Elo cross-fitted [M, `round9_provenance_recruit_cells.out` §7], miss half-width 8.9 [D, `round9_provenance_budget.out` §4]. I-match: every switch played the rule's choice | J in [11.3, 29.1]: first-order transfer holds on the largest measured signal; S1 proceeds; a PURE claim. J > 0 by 2 SE but below: S1 proceeds, J/Ĝ reported beside its forecast. J not > 0: coined-self-play labels do not reach play; S1 is not proposed until the cause is found ($21 saved) |
| **S1** round 1 on `own2` | $21.07 × r̂ + start-ups | The design's plants and readouts, v̂ per type for `own2`; per batch the twin ratio out of sample against T0's (2 SE) and the twin column's I-V | Plant fails: batch quarantined. Ratio drifts: batch size re-derived |
| **S2** | $2.11 | As the design: 37,800 fresh-seed games, no twins, PURE; claim against `own1`, two networks | As the design (E reserve $3.08) |
| **S3, S4, S5** | $2.11 + batches; $3.89; $2.26 + batches | As the design | As the design |
| **S6** round 4 | One round at r̂ | As S3 | Runs iff the headroom after S5 covers it: 0.15 rounds fit at r 1.0, 0.83 at 0.8, 1.30 at 0.7 [D, *power* §4] |

**Totals** [D, *power* §4]: $78.88 at r = 1 (pilot wasted, $3.46 left), $72.68 at 0.9, $66.48 at 0.8, $54.08 at 0.6. The $1,000 case is argued as the design says, after two adopted rounds within their miss lines; twins change its price by r̂. Nothing measured links `own1` to human strength, and no stage claims it.

**Option behind ruling (c):** seat-swapped twins in matches (duplicate format), estimator the plain mean. For π₁ ≈ π₀ it cancels the setup's side effect, 0.125 of Var(z) [D from M: 0.8239 − 0.6988, `round9_simplify_residual.out`]. Nothing relies on it.

## 5. What is new, and the nearest work

Searched 2026-10-21; finding nothing does not prove absence. The parts exist: common random numbers ([Glasserman & Yao 1992](https://doi.org/10.1287/mnsc.38.6.884), not re-read); fixed scenarios for policy search ([PEGASUS](https://arxiv.org/abs/1301.3878)); common random numbers across branch rollouts (TRPO's vine, as cited by Sakib); input-dependent baselines from rollouts sharing an exogenous input ([Mao et al. 2019](https://arxiv.org/abs/1807.02264), unbiased "provided that the input process is independent of the states and actions"); event-keyed hashing ([Buffalo et al. 2026](https://arxiv.org/abs/2603.11084)); certified coupling of counterfactual rollouts in Leduc poker ([OSCC](https://www.alphaxiv.org/abs/2609.33543)); duplicate formats; AIVAT. **Nearest:** [Sakib 2026](https://arxiv.org/abs/2609.24144), event-keyed noise shared within GRPO groups with a leave-one-out baseline (gradient-covariance traces 21–63% lower).

Not found:
- **(i)** twins under deterministic argmax whose only policy randomness is logged near-tie coins, so each pair is an exact fork of one history, memory included, until its first coin disagreement, with no snapshot, landing where measured spread sits;
- **(ii)** the sibling mean as an exactly centred control variate for every per-decision inverse-propensity label of a game at once, with the head's noise scaling by r at first order;
- **(iii)** exactness against cross-game inference numerics by one match per twin index;
- **(iv)** endogenous chance (which fights happen depends on play) made exogenous by ordinal keys on an engine that accepts any per-command seed, so Mao et al.'s condition holds by construction.

## 6. Weakest point

**ρ.** The record leans against coupling after divergence: Q8 found no matched fights, and one recruit of difference changes which fights happen. At ρ 0.15 the pilot certifies a ratio below 0.95 with P 0.04 [D, *power* §2]; then the component cost $0.35, a seed function and a K-way join for nothing. At best it cuts cost; the ceiling (the band, the frozen span, v on `own2`) is untouched. Second: exactness rests on separate matches; merging twins into one match for speed would break it silently, and I-V catches only a large leak.

## 7. Decision record

- **Rejected: antithetic first coins** (P 0.5, D₂ = 1 − D₁): one paired fork per pair, as independent coins already give at band index 2.667, for a second label formula.
- **Rejected: single-coin twins:** one contrast for a whole game, against about 155 labels per coin game.
- **Rejected: identity keys:** Q8 measured zero matches.
- **Rejected: twins in one match:** shared inference batches break exactness.
- **Rejected: siblings in place of the cell baseline:** a 3-game mean is noisier than a cell's 40, so the twin term sits on top of it.

## Provenance

| Number | Value | Tag | Source or derivation | Used for |
|---|---|---|---|---|
| `own2` against `own1` | +22.7 ± 4.4 Elo, lower bound +14.0 | M | seed §7; `training/metrics/own2_base_20261020/progress.txt` | π_0 |
| Attack known answer | −0.242 ± 0.023 z (2Δ); match mean −0.118; −41.3 ± 4.4 Elo | M | seed §7 | No knob-derived sizing |
| Pooled label; recruit>recruit | +0.0003 ± 0.0007; +0.0070 ± 0.0024 | M | `readout_b_offline.txt`, reproduced by *cells* | §1 |
| Cells past the family-wise line | +0.0235 ± 0.0041; +0.0194 ± 0.0042; +0.0195 ± 0.0046; +0.0173 ± 0.0041; line z 3.72 over 251 cells (4.07 over all 1,049) | M | *cells* (game-clustered, pre-coin cells, labels rebuilt as `tools/coin_labels.py`) | §1, A |
| V_rr; forecast of the additive recruit rule | 1.592e-3 ± 3.70e-4; +20.2 ± 3.1 Elo per side-game | M | `round9_provenance_recruit_cells.out` §2, §7 | A's prediction |
| `own2` recruit rates | 16 of 28 units toward the head, 12 away | M | `round9_provenance_own2_recruits.out` | v on `own2` [U] |
| Prices | $2.583e-5 one network, $8.846e-5 two networks, $0.577/h, start-up 69 s, $82.34 left | M | `round9_provenance_budget.out` §1–§3 | Every cost |
| Plan without twins | $77.46, $4.88 left | D | *power* §4, stage list of `round9_provenance_budget.py` | Money binds |
| Twin variance ratio | 1 − ρ²(K−1)/(1 + (K−2)ρ) | D | Exchangeability, optimal κ | §2.3 |
| First-order scaling of the head's noise by r | — | D | ω independent of siblings, mean zero (§2.3) | Pilot checks the SE ratio |
| Shared-draw hit correlation | (min − c₁c₂)/√(…): 1 equal, 0.667 at 0.4/0.6 | D | Hit iff draw % 100 < cth (`combat.rs:436`) | §2.1 |
| First disagreement | band index 2.667; prefix 0.0226; r>r 0.667; side-turn 1 0.785 | D from M | *prefix*, 4,502 games of batch 1 | §2.4 |
| Census | 77.6 band decisions per side-game | M | seed §7 | §3 |
| Numerics flips | 1.76% at gaps < 0.014 | M | `current_design.md` §2 (carried) | §2.4 |
| Cell residual; within-cell asymptote | 0.701; 0.680 | M | `round9_provenance_label_noise.out` | r ≥ 0.97 without dice coupling: (0.701 − 0.680)/0.701 = 0.030 |
| Leader share within (side, faction) | −0.0002 of Var(z − βᵀS) (12,000 games; +0.0005 on 22,000) | M | *setup* (`leaders`), method of moments | r ≥ 0.97 without dice coupling |
| Cell share for the duplicate option | 0.125 of Var(z) | D from M | 0.8239 − 0.6988, `round9_simplify_residual.out` | Option only |
| Q8 | median 0 matched fight identities; 28% of pairs share one | M | `docs/archive/credit_assignment_design_20260817.md` Q8 | §2.4, §6 |
| Seeds shared across match records | 0 of 95,895 seeds in 53 archived records; 0 of 19,466 in 23 walked matches | M | *setup* (`seeds`) | ρ cannot be read from disk |
| Pilot resolution | SE 0.0023–0.0207; P(certify) 0.04 at ρ 0.15, 0.80 at 0.2, 1.00 at 0.25 | D | *power* §2, simulated binary results | T0 |
| Planted shared component | r′ 0.914 / 0.774; recovered 0.915 ± 0.011 / 0.776 ± 0.018 | D | *power* §3 | T0 check |
| Coin plant SE in T0 | ≈ 0.019 | D | 0.0387 × √(2,000/8,000) (`current_design.md` §2) | §2.5 |
| Costs at r | $78.88 / 72.68 / 66.48 / 54.08 at r 1 / 0.9 / 0.8 / 0.6; T0 $0.35; A $0.36 | D | *power* §4 | Regimen |
| A's miss half-width | 8.9 Elo at 12,800 games | D | `round9_provenance_budget.out` §4 (2√(3.2² + 3.07²)) | A |
| Match SEs | 3.07 at 12,800; 1.79 at 37,800 | D | (800/ln 10)/√N | A, S2 |
| K = 4; the 0.95 line; a = 0.5 | — | C | K 8 gains at most about 0.06 of ratio for ρ ≤ 0.5, at twice the start-ups (*power* §1); the 0.95 line saves ≥ $3.10 (*power* §4); a recovered within 1–2 SE (*power* §3) | T0 |
| ρ, r̂ per keying, κ | — | U | T0, before S1 is sized | Batch size |
| J/Ĝ for recruit switches | — | U | A, before S1 | Whether S1 is proposed |
| v, c, τ̃ on `own2` | — | U | S1 (the design's estimators) | Round gain |
| Second-order loss, ĉ, ρ̂ of the reset, compounding, human level | — | U | As `current_design.md`; nothing here rests on them | — |
