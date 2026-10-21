# Round 9, provenance stance: a measured near-tie gain, and a test of its link to play before a round is bought

Tags: **[M]** measured (record and script named), **[D]** derived (inputs shown), **[U]** unknown (where it gets measured, or why nothing rests on it), **[C]** a choice whose consequences are derived. Scripts are `analysis/round9_provenance_{cells, recruit_cells, unit_cells, fold_se, label_noise, own2_recruits, budget, allin}.py`, outputs beside them.

## Summary

- **Section 7 moves four decisions.** The base is `own2`, so batch 1's `own1` labels cannot train S1's head; they serve the rule R below and the baselines. **At the all-in price the whole path does not fit on the measured host** ($83.49 against $82.34 at $0.577 an hour); S5's host is chosen under a ceiling of $0.569 an hour, lower once the feature dump's overhead is measured. The derived SEs held. S0(a) read half the decode-knob extrapolation, so knob windows from another network become reports.
- **New from batch 1's records.** A 43-value linear head on the recruit candidates' unit identities, fitted by ridge to the coin labels, gains **+22.9 ± 3.2 Elo** first order on held-out games (between-fold SE 2.9); +17.5 ± 3.2 of it comes from spread between unit pairs, which the choice of this partition did not touch. Outside the recruit pairs, cells keyed on unit identity show no certified spread.
- **Reorder.** Before S1's $22.73, one box ($0.72) plays R twice: **L** on `own1` (does a first-order coin label predict uncoined play?) and **T** on `own2` (does a learned rule survive a change of base, which pooling labels across rounds assumes?). S1 is proposed only after L.

## 1. What the records settle now

### 1.1 Prices (`allin.out`, `budget.out`)

- **One network, coin matches** (120 jobs): 22,341 games an hour of match wall on the $0.577/h host. The work after each match that the labels need (luck walk, coin rows, plant rows, readout) adds 0.095 of match wall. The whole coin box ran 19,770 games an hour all-in, $2.92e-5 a game, matching the record's "about $2.0".
- **Two networks, 20 jobs:** 6,523 games an hour ($8.85e-5 a game); 6,256 with the box's staged checks.
- **The path** (L, T, C, S1–S5, E's reserve), batches all-in: $83.49 at $0.577/h, $60.77 at $0.42/h, against $82.34 left. It fits iff the rate is at most $0.569/h with no feature-dump overhead h, $0.526 at h 0.10. Without S5: $58.50 at $0.577/h. **Decision:** S1–S4 run on any host up to $0.577/h; S5's host is chosen under the ceiling restated once the S1 smoke measures h. A null round 1 after L and T costs $24.42 ($17.78 at $0.42/h).

### 1.2 Label noise (`cells.out` §1, `label_noise.out`)

- E[ω²] 5.330 (exact 5.333); E[T²] 4.549 with the luck columns, 5.330 without.
- Kept shares of Var(z) per game: frozen β 0.824, refitted 0.822 (no gain), with the setup cell fitted on other matches 0.701 (round 6 derived 0.688). Weighted by band decisions, as the label weighs them: 0.858, and 0.760 with the cell, so σ² = 5.333 × 0.760 = 4.05 [D from M].
- Design effect within a game 1.00: the clustered pooled SE 0.00070 equals √(4.549/9,317,547).
- **Consequences [D]:** S1's forecast SE 4.88 Elo at a switched share q of 0.5 (6.9 at q = 1), miss line 10.4; S5's miss line 13.1 (table).
- **[C]** S1's batch 1 takes b(cell) from `own1`'s 60,000 games: any b fixed before a batch keeps the label exact.

### 1.3 Where the spread is: recruit pairs (`cells.out` §2, `recruit_cells.out`, `fold_se.out`)

Games go to 20 folds by hash. Per partition of the 9,317,547 band decisions into cells: **V**, the variance of the true cell means (U-statistic over distinct folds), and **W**, the cross-fitted first-order welfare of a switch rule chosen on the other folds.

**Nine coarse partitions show nothing:** pair type, × gap bin, turn, faction, map, mover, and three sub-kind refinements. No V exceeds 1.2 SE; W runs from −16 to +16 Elo with SEs of 13–16.

**Recruit pairs of different units**, cells (mover's faction, a0's unit, a1's unit): 262 cells, 514,281 labels, 4.29 per side-game.
- Excess dispersion +11.2 SD (+9.7 with T0, the label without variates). V_rr 1.59e-3 ± 3.7e-4: cell means with sd 0.040, at least 0.029 at 2 SE.
- Where both orders are seen, the strongest pairs flip sign with the order: Vampire Bat>Ghost +0.126 ± 0.023, Ghost>Bat −0.115 ± 0.034; Wolf Rider>Grunt +0.092 ± 0.018, Grunt>Wolf Rider −0.072 ± 0.022.
- **The additive head R:** one value v per (faction, unit), 43 values, switch iff v(a1) − v(a0) > 0, ridge 2,000 labels. Cross-fitted W **+22.9 ± 3.2 Elo** per side-game, switching 2.21 decisions per side-game; +20.6 ± 3.4 read with T0; honest halves +21.3 and +24.8 (± 4.5).
- **Two checks.** The SE from the folds' spread, which allows any correlation within a game, is 2.9. Centring each fold's labels on the group's out-of-fold mean leaves +17.5 ± 3.2, earned from spread between unit pairs. The partition was examined because recruit>recruit had section 7's largest typed mean; that choice touches only the mean's share, about 5 Elo.

**What it settles:** Var(E[Δ | cell]) over the band is at least 0.0552 × (1.59e-3 − 2 × 3.7e-4) = 4.7e-5 [D], 18 times the bound a null S1 would have given; the unit columns of §2.2 span the rule that collects it. **Not settled:** uncoined play (L), `own2` (T).

### 1.4 Outside the recruit pairs (`unit_cells.out`, `fold_se.out`)

Every band decision is keyed on (sub-kind, faction, acting unit types), the types tracked through the command stream before the decision. The tracker ignores deaths and level-ups; that adds noise, never bias, since a pre-coin key keeps a cross-fitted rule unbiased whatever it means.
- Over the other 71.3 band decisions per side-game (90,280 cells): excess dispersion +0.90 SD over 2,433 cells, V 1.4e-4 ± 3.3e-4.
- Cross-fitted cell rules: moves of different units +19.0 ± 8.0 Elo (T), +23.5 ± 8.6 (T0); an additive unit model on them +9.5 ± 5.5; same-unit moves +8.3 ± 5.8; end_turn pairs +0.8 ± 4.1; attack pairs −8.8 ± 5.3 (their typed means lean negative, so a rule switching half of them reads below 0 without spread). Between-fold SEs are 0.77–1.04 of the label SEs.
- **What it settles:** no certified spread outside recruit pairs at unit-identity resolution: the pre-named reading for moves (T, rule a) sits at 2.38 SE, under the 2.45 family-wise line for seven groups. Nothing relies on it.

### 1.5 Does `own2` recruit like `own1`? (`own2_recruits.out`)

In the 6,402 base games, Vampire Bats per side-game 4.12 and 4.10, Wolf Riders 3.52 and 3.52; of 28 units whose rate moved by more than 2 SE, 16 moved toward R's preference. A reading, not a forecast.

## 2. The algorithm

The core is the current design's: coin labels with logged propensities, a closed-form linear head on frozen features, a tested threshold, adoption by PURE match. The changes are unit columns in the features and §3's stage order.

**2.1 The label.**

    T_i = s_i ω_i (z − b_k(cell) − βᵀS),   ω_i = (D_i − P_i)/(P_i(1 − P_i))

E[T_i | H_i] = Q̃(H_i, a1) − Q̃(H_i, a0) under the coin policy: b and β are fixed before the batch, earlier luck is fixed by H_i, and each later luck term has mean zero given its pre-fight state. Plant slope 0.994 ± 0.008 per mover side [M]. Var T = E[ω²]E[R²] = 4.05 [D from M].

**2.2 The head.**

    f = c_uv + b_uv·G + ŵᵀx,   x = φ(a1) − φ(a0)
    φ = [actor token, q ⊙ k, four log-probabilities, e(faction, unit) if the candidate is a recruit]

- 43 unit columns are added to d = 772, unconditionally: the measured rule is linear in e(unit), so whether the tokens carry unit identity linearly no longer matters. x stays antisymmetric under a swap (I3).
- ŵ = (G + λI)⁻¹ Σ_g R_g A_g, a natural policy-gradient step for the coin policy.
- Same-kind pairs vanish at a tie where checked: recruit pairs of different units at G < 0.05 read −0.003 ± 0.007.

**2.3 The estimators used before any head is fitted.** V̂ = Σ_c π̂_c Û_c − Û_all, with

    Û_c = (S_c² − Σ_f S_cf²)/(n_c² − Σ_f n_cf²)

Given the counts, Û_c is unbiased for μ_c² because distinct folds hold independent games, so E V̂ = Var(E[Δ | c]) ≤ Var(Δ); SE by a delete-one-fold jackknife. Ŵ = SG⁻¹ Σ_f Σ_{i∈f} T_i·1[r₋f(c_i)], SG the side-games, r₋f the rule chosen on the other folds: each rule is independent of the labels it is read on, so Ŵ is unbiased for the fold rules' first-order gain (cross-fitted empirical welfare maximisation, [Kitagawa & Tetenov 2018](https://doi.org/10.3982/ECTA13288)), with variance SG⁻² Σ T_i²·1[·], confirmed by the folds' spread.

**2.4 From labels to play.** By the performance-difference lemma, a match's gain J sums the new policy's advantages under the old policy along the new policy's states. The labels give advantages under the coin policy, where both sides deviate 19 times a side-game. So J = ρF plus a second-order term; no a-priori bound on either is measurable ([Farias et al. 2022](https://arxiv.org/abs/2206.02371), round 8). L splits J = n_L × a_L: n_L, R's switches per side-game in uncoined play, is logged (against 2.21 in the coin games), so a shortfall is attributed to the switch frequency or to the value per switch.

**2.5 Signal per dollar [M].** About $1.75 of all-in games bought a 43-parameter rule worth +22.9 ± 3.2 Elo first order out of sample; each result priced 155 labels, with no playout or critic. A match prices one rule at SE 3.07 for $0.33; the labels price every rule in the span at once.

**2.6 Invariants and plants.** I-coin, I-log, I2, I3, I4, I6, I-φ, I-V and I-match carry over. New:
- **R plant**, offline on batch 1's records: z′ = z + Σ_j s_j (D_j − P_j) uᵀx_j, x_j = e(a1) − e(a0), parsed by `tools/coin_plant.py`'s separate code; u is ±δ on a third of the units, δ at 10 SE. **Prediction:** ŵ′ − ŵ = (XᵀNX + λI)⁻¹XᵀNX·u, slope 1 along u, line at 3 SE; a wrong mover sign or swapped candidates read −1.
- **I-R, the hook:** on the head seat, every band recruit decision between two units logs v(a1) − v(a0). Every positive one plays a1 and no other does; n_L is reported. A unit test with a planted table derives from this.

## 3. Regimen

Costs at $0.577/h, coin batches all-in [D, `allin.out`].

| Stage | Cost | Readout and prediction | Decision per outcome |
|---|---|---|---|
| **R0**, offline | $0 | R plant at slope 1; the table frozen and hashed; F sealed at +22.9 ± 3.2 Elo | Plant fails: fix the fit before any box |
| **L** (one box with T) | $0.36 | `own1`+R against `own1`, the seat hooked by a per-game coin, 12,800 decisive games, SE 3.07. Prediction F, miss half-width 8.9; n_L against 2.21; I-R. J clears 0 by 2 SE with P ≥ 0.96 if ρ ≥ 0.5 | **Lower bound > 0, \|J − F\| < 8.9:** labels predict play; a PURE claim over `own1`; S1 proposed with ρ = 1. **Lower bound > 0, J < F − 8.9:** ρ̂ = J/F, split by n_L/2.21, scales later forecasts. **Lower bound ≤ 0:** first-order labels do not carry to play where strongest; no S1; the S3-null menu. **J > F + 8.9:** restate the link |
| **T** | $0.36 | `own2`+R (`own1`'s table) against `own2`, same size. No magnitude predicted | **Lower bound > 0:** π_0 = `own2`+R, then C, and later rounds pool earlier rounds' labels. **Otherwise:** π_0 = `own2`; each round fits on its own labels only (pooling changes power, never a claim) |
| **C** | $1.13 | `own2`+R against `own1`, two networks, 12,800 games: the claim; read beside 22.7 + J_T | Reported |
| **S1 smoke** | in S1 | I-φ, the ridge plant, I-coin, I-log, coin plant, dice control; spectrum (d_eff), feature-dump overhead h, π_0's census | A gate fails: debug. Else floors and costs are restated from d_eff, the census and h, and S1 is proposed |
| **S1** | $22.73 | 8 batches on π_0; v̂ per type (SE 1.40e-6); forecast (SE 4.88 at q 0.5) | As the current design |
| **S2** | $2.11 (+ E) | Adoption against π_0 (37,800 games), claim against `own1` (two networks); forecast × ρ̂ | As the current design |
| **S3** | $24.84 | Round 2 | As S2 |
| **S4** | $3.89 | Reset, after the optimizer discussion | As the current design |
| **S5** | $24.99 | Round 3, on a host under the restated ceiling | As S2 |

**Why this order.** Every costly stage after L relies on L's link reading and ρ̂; T settles pooling before round 2 refits; the smoke's d_eff and h come before S1's floors and S5's host. L, T and C cost at most $1.85. Branch S stays behind rulings (f) and (g).

**Toward human level:** coin rounds and resets, each certified by a match, then the $1,000 case after two adopted rounds within their miss lines. Nothing measured links `own1` to human ratings, and nothing rests on it.

## 4. What is new, and the nearest work

Searched 2026-10-21; finding nothing does not prove absence.

1. **The band coin's nearest prior work is the tie-breaker design** of regression discontinuity: treatment randomised only near a cutoff of a running variable, deterministic elsewhere ([Owen & Varian 2020](https://arxiv.org/abs/1808.07563); [Li & Owen 2023](https://arxiv.org/abs/2202.12511), Annals of Statistics, abstract read). **New here:** the running variable is the network's own margin, the cutoff is the argmax tie, and the dice's exact laws remove part of the outcome noise without bias.
2. **Checking an off-policy estimate against an online test before scaling** is recommender practice: Criteo found capped and normalised importance sampling poorly matched to A/B outcomes ([Gilotte et al. 2018](https://arxiv.org/abs/1801.07030)). **New here:** the estimate is exact by construction, so a disagreement isolates the coin-to-uncoined shift and the second-order term, not estimator bias.
3. **Shape-free spread on pre-coin cells** ([Kong & Valiant 2018](https://arxiv.org/abs/1805.01626)'s split inner product, with cross-fitted welfare) as a go/no-go reading before a round: known parts.
4. **The finding:** an imitation-trained network's near-tie recruit choices are miscalibrated by unit identity. No prior report found; a fact about this network, not a method claim.

## 5. Weakest point

- **F is first order and under the coin policy.** On a shortfall, later forecasts rest on ρ̂ from one rule covering 5.5% of band decisions; moves (47 a side-game) may transfer differently.
- **Outside recruit pairs nothing is certified;** S1's value beyond R is unmeasured, and a null round 1 costs $24.42.
- **R is a table over unit identities.** If the user rules it a strategy concept, L and T are measurements only.
- **Money is tight:** the path overruns by $1.15 at $0.577/h before the feature dump's overhead.

## 6. Rulings and decision record

**Rulings.**
- **(h), new.** May a closed-form linear head on the two candidates' unit identities (43 values fitted to coin labels) play in L and T, and stay in the player? May the production head carry unit columns?
- **(e)** stands: does a closed-form head fall under the 2026-10-15 ruling?
- **(a)–(g)** are otherwise unchanged.

**Rejected or retired:** S1 on `own1` (π_0 is `own2`); a wider recruit band (unmeasured; after L); refitting β (0.822 against 0.824); the derived attack-type bound (measured means replace it); knob windows from other networks as gates (S0(a) −41.3 against −76 ± 17); 37,800-game L and T (miss half-width 8.9 → 7.3 for three times the cost); an R² threshold for dropping the unit columns (a number from nowhere); a move-cell arm in L (not certified, and its tracker key ignores level-ups, so a hook could not reproduce it); pricing batches at match wall.

## Provenance

| Number | Value | Tag | Source or derivation | Used for |
|---|---|---|---|---|
| One-network rate | 22,341 games/h of match wall (median 10,000-game match) | M | `budget.out` §1 from `near_tie_coins_20261020/matches.txt` | Every cost |
| Post-match work | +0.095 of match wall; 15.3 s per 1,000 games | M | `allin.out` §1 from `walls.txt` (b01–b06) | Coin batch prices |
| All-in coin box | 19,770 games/h, $2.92e-5 a game at $0.577/h | M | `allin.out` §1; record "about $2.0" | Check of the model |
| Band-coin logging | +5.7% against the attack-coin stage | M | `budget.out` §1 | Costs; feature dump h [U] |
| Two-network rate | 6,523 games/h ($8.85e-5); 6,256 with staged checks | M | `budget.out` §2, `allin.out` §1 from `own2_base_20261020` | C, S2–S5 |
| Spent; left | $2.66; $82.34 | M, D | Seed §7; 85 − 2.66 (assumes the window counts against $85) | Budget |
| Path total | $83.49 at $0.577/h, $60.77 at $0.42/h; without S5 $58.50 at $0.577 | D | `allin.out` §2 | S5's host |
| Rate ceiling | $0.569/h at h 0; 0.547, 0.526, 0.489 at h 0.05, 0.10, 0.20 | D | `allin.out` §3 | S5's host |
| Null round 1 after L, T | $24.42 at $0.577/h; $17.78 at $0.42/h | D | `allin.out` §2 | Risk |
| Distillation | 4.8 box-hours | M | $2.1 at about $0.44/h, `docs/memory_own_trajectories_prereg_20261011.md` | S4 |
| E[ω²]; E[T²]; E[T0²] | 5.330; 4.549; 5.330 | M | `cells.out` §1 | σ² |
| Kept shares per game | 0.824 frozen β, 0.822 refitted, 0.701 cell, 0.680 asymptote | M | `label_noise.out` | b kept, β frozen |
| Kept shares, decision-weighted | 0.858; 0.760 with the cell | M | `label_noise.out` (b00, b06) | σ² |
| σ² | 4.05 | D | 5.333 × 0.760 | Every SE |
| Design effect | 1.00 overall; between-fold SE ratios 0.77–1.05 in every group read | M | `readout_b.json` against √(4.549/9,317,547); `fold_se.out` | W's SE |
| Census, `own1` | 77.6 ± 0.13 per side-game | M | `readout_b.json` | Storage; `own2`'s [U], smoke |
| Coarse partitions | No V above 1.2 SE; W −16 to +16, SEs 13–16 Elo | M | `cells.out` §2 | "No coarse spread" |
| Recruit pairs of different units | 514,281 labels, 4.29 per side-game, 262 cells | M | `recruit_cells.out` | R |
| Excess dispersion | +11.2 SD (T), +9.7 (T0) | M | `recruit_cells.out` §1 | Spread exists |
| V_rr | 1.59e-3 ± 3.7e-4; sd ≥ 0.029 at 2 SE | M | `recruit_cells.out` §2 | Lower bound |
| Lower bound on Var(E[Δ \| cell]) | 4.7e-5, 18× round 8's 2.53e-6 | D | 0.0552 × (1.59e-3 − 2 × 3.7e-4) | Weakest point of round 8 closed from below |
| F | +22.9 ± 3.2 Elo (fold SE 2.9); T0 +20.6; halves +21.3, +24.8 (± 4.5); ridge sweep +19.1 to +22.9 | M | `recruit_cells.out` §3, §5, §7; `fold_se.out` | L's prediction |
| F from spread alone | +17.5 ± 3.2 Elo | M | `fold_se.out` (labels centred on the out-of-fold mean) | Selection caveat sized |
| R's switches | 2.21 per side-game (0.516 of 4.29) | M | `recruit_cells.out` §3 | n_L's reference |
| Rule (a) by faction | Undead +9.2 ± 1.7 … Rebels −1.7 ± 1.2 | M | `recruit_cells.out` §6 | Report |
| Tie check | −0.003 ± 0.007 at G < 0.05 | M | `cells.out` §3 | Antisymmetric terms kept |
| Unit-keyed cells outside recruits | +0.90 SD over 2,433 cells; V 1.4e-4 ± 3.3e-4; moves of different units +19.0 ± 8.0 (T), 2.38 SE | M | `unit_cells.out`; `fold_se.out` | Weakest point 2; nothing relies on it |
| Family-wise line, seven groups | 2.45 SE one-sided at 5% | D | Φ⁻¹(1 − 0.05/7) | Reading of 1.4 |
| `own2`'s recruit mix | Bat 4.12 → 4.10, Wolf Rider 3.52 → 3.52; 16 toward, 12 away | M | `own2_recruits.out` | T's reading |
| `own2` against `own1` | +22.7 ± 4.4 Elo | M | `own2_base_20261020/own2_vs_own1.fit.json` | π_0; C |
| Match SE | 3.07 (12,800 games), 1.79 (37,800) | D | 347.4/√N | L, T, C, S2 |
| L's miss half-width; SE of J/F | 8.9 Elo; 0.19 | D | 2√(3.2² + 3.07²); √(3.2² + 3.07²)/22.9 | L's decisions |
| L's detection | P(lower bound > 0) 0.96 at ρ 0.5, 0.59 at 0.3 | D | Φ((ρF − 6.14)/3.07) | L's size |
| S1 forecast SE; miss line | 4.88 at q 0.5 (6.9 at q 1); 10.4 | D | 4.6 × √(4.05/3.67) × √(77.6/76.1), `round6_synthesis_power.out` §4 | S2; recomputed from logs |
| S5 prediction SE; miss line | 5.77; 13.1 | D | √(3.07² + 4.88²); 2√(3.07² + 5.77²) | S5 |
| Spread estimator SE, 8 batches | 1.40e-6 | D | 1.27e-6 × 4.05/3.67 | S1's null reading |
| Coin plant SE, 100,000 games | 0.0064 | D | 0.0082 × √(60,000/100,000) | S1's plant line |
| S0(a) | −41.3 ± 4.4 against [−111, −41] | M | `readout_a_offline.txt` | Knob windows become reports |
| Attack-promoting means | move>attack −0.0031 ± 0.0032; end_turn>attack −0.0135 ± 0.0154 | M | `readout_b.json`, `cells.out` §4 | Replaces the derived bound |
| Caps | 23 of 60,023 | M | `readout_b.json` | Cap rule |
| Elo per unit z | 347.4 | D | 800/ln 10 | All |
| J_L, n_L, J_T, additivity at C | — | U | L, T, C | Nothing before them |
| `own2`'s census, d_eff, h | — | U | S1 smoke | S1's floors; S5's host |
| q, v per type, second order beyond R | — | U | S1, S2 | Nothing before S1 |
| ĉ, ρ̂ of the reset, J_S4 | — | U | S4 | S5 |
| Branch S numbers | As current design | M, D, U | `current_design.md` §2 | Branch S |
| Human level, exploitability | — | U | None | Nothing |
| 12,800-game L and T, ridge 2,000, 43 columns, w 0.35, p 0.25, 8 batches of 100,000, 20 folds | — | C | Consequences derived above | Design |
