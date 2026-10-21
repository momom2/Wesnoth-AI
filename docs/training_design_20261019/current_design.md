# Current design (after round 10)

Supersedes the round-9 design. The core is unchanged: near-tie coins with logged propensities, labels exactly unbiased through the simulator's chance laws, a closed-form linear head on the frozen network, adoption only by match, a fresh claim against `own1` after each adoption. What changed:
- **Fitted rules interact with themselves.** Every fitted recruit rule on record loses 8–13 Elo of second-order value over its own later switches in a side-game, so its first-order sum is not a prediction of play. Capped at each side's first two switches a game, the interaction is measured near zero and the rule's coin-world value is exact.
- **L plays a capped rule.** Its sealed prediction is complete in the coin world, so a miss isolates the shift from coined to uncoined play.
- **The recruit structure is matchup- and composition-dependent.** The design's progressive loss picks a rule Bd keyed on unit, unit × enemy faction and the side's earlier recruits of each type. Capped, it is worth +24.1 ± 3.0 Elo in the coin world, against +14.9 ± 3.0 for capped R.
- **S2 searches the step in play.** Three screens of different switch-set sizes are played, and the best is confirmed on fresh seeds. This replaces S2 then E, and it measures how the second-order loss grows with the step.
- **Simpler.**
  - H1 and the typed block are gone.
  - T and C run only if ruling (h) allows a rule in the player.
- **Fixed.**
  - A token-block identity check covers the 772 columns no other check saw.
  - The unit block is a pair feature on recruit pairs.
  - o is read from equal-size runs.
  - The host test runs 10,000 games.
  - The claim stop is one-sided.

Reasons: `round10_synthesis.md`. Tags: **[M]** measured (record or script named), **[D]** derived (inputs named), **[U]** unknown (measured before anything relies on it, or nothing relies on it), **[C]** a choice whose consequences are derived. Scripts are in `analysis/`; *checks* is `round10_synthesis_checks.out` and *budget* is `round10_synthesis_budget.out`, with § for their sections.

## 1. The idea

At argmax both players are deterministic, so with the setup fixed, all of a result's variance is chance. At a decision where the network's top two choices nearly tie, a logged coin plays the runner-up. Play then continues under the network, which re-decides after every roll. The coin is independent of the past, so the result identifies that decision's effect without a critic. A linear head on the frozen token outputs pools the labels across positions and proposes which near-ties to switch. With P(1 − P) constant, the head's ridge fit is one episodic natural-actor-critic step for a logistic coin policy ([Peters & Schaal 2008](https://doi.org/10.1016/j.neucom.2007.11.026)).

Labels choose the direction, and play chooses the step. A head's first-order gain is exact from labels at any switch set, but its gain in play includes the switches' interactions and the shift from coined to uncoined play, which labels cannot price. So the design plays a few step sizes and adopts only what a fresh match certifies. The record shows why. On the one structure found, recruit choices between different units, the rule's later switches in a side-game lose value, and only a rule capped at two switches has a complete forecast.

## 2. Measurements first

**Settled by existing records**

| Quantity | Value | Tag, source | Decision it feeds |
|---|---|---|---|
| Base | `own2` over `own1` +22.7 ± 4.4 Elo, 6,400 decisive games | [M] `own2_base_20261020/reading.json` | π_0 is `own2` (or `own2` + capped Bd, T) |
| Label path | Coin plant 0.994 and 0.995 ± 0.008; dice slope 0.00436; I-coin, I-log over 9.3 M decisions | [M] `near_tie_coins_20261020/readout_b_offline.txt` | Certified on `own1`; the gates rerun on `own2`'s smoke |
| Mean near-tie advantage | +0.0003 ± 0.0007; coin play costs +2.0 ± 4.7 Elo | [M] seed §7; [D] `round9_simplify_budget.out` | Exploration is free |
| Spread outside recruits | Nine partitions, none above 1.2 SE; unit-keyed cells +0.90 SD | [M] `round9_provenance_cells.out`, `round9_provenance_unit_cells.out` | Nothing is built on them; the typed block goes (type V z +0.39) |
| Recruit rules, leave one match out | R (43 unit values) F +18.0 ± 3.1; B − R (matchup keys) +8.6 ± 2.1; Rd − R (count d) +5.1 ± 1.7; Bd − R +10.1 ± 2.3. Progressive loss at ridge 5,000: R −273.2, Rd −403.1, B −557.0, Bd −699.3 (lowest) | [M] *checks*; B and d first found in `round10_challenge_recruit_context_ridge.out`, `round10_invent_recruit_rules.out` | Bd is L's rule and the recruit block under (h) |
| Fold-scheme spread | R's 20-fold F +21.3 (SHA-256 folds) or +19.9 (crc32 folds); leave-one-match-out +18.0 | [M] `round9_synthesis_checks.out` §1; `round10_invent_recruit_rules.out` §2; *checks* | Seals use leave-one-match-out, the condition of a fresh match |
| Own interactions | Second-order term over all ranks (TayPO-2 form): R −12.3 ± 4.5, Rd −7.8 ± 4.8, B −10.3 ± 4.4, Bd −13.1 ± 4.7; the parameter-free count rule −3.4 ± 4.8. The expansion is not convergent at rank 2 (second order −3.3, exact +4.3) | [M] *checks*; `round10_provenance_critique_checks.out` §5 | No full rule's F is sealed; S2 searches the step |
| Capped at two switches a side-game | Exact coin-world value: R +14.9 ± 3.0 (interaction −0.3 ± 1.7); Bd +24.1 ± 3.0 (interaction +0.6 ± 1.8; side-turn 1 +14.3); capped Bd − capped R +9.3 ± 2.6 | [M] *checks* | L's seal |
| State shift at turn 1 | R's rank-0 turn-1 switches, reweighted exactly to uncoined prior play: −1.2 ± 1.3 | [M] `round10_provenance_critique_checks.out` §2 | An L shortfall is read as the continuation |
| The collapse | Turn-1 distinct types: `own1` 1.345, `own2` 1.356, humans 3.164. Whole-game HHI 0.514, 0.514, 0.274. Count slope inside unit identity +0.00267 ± 0.00048 (z 5.6) | [M] `round10_invent_composition.out`, `round10_invent_composition_own2.out`, `round10_invent_recruit_rules.out` §7 | d enters Bd; `own2` collapses as `own1` does, which T's seal J_T = J_L rests on |
| Not a decode artifact | Split-only rule +1.3 ± 2.9; r(v, log split rate) −0.20, p 0.21 | [M] `round10_challenge_recruit_split.out` | No marginal recruit decode |
| Learning curves | R flat past 17,000 games (paired +0.12 ± 1.90 to 51,000); B still rising (30,000 → 54,000 +2.47 ± 1.25) | [M] `round10_provenance_critique_checks.out` §3; `round10_challenge_critique_checks.out` §3 | S1's size stays; per-block curves are read in-run |
| R as a unit block | A per-candidate one-hot adds 2.36 switches per side-game on mixed pairs worth +0.2 ± 3.3; one ridge holds R at every λ (+19.1 to +22.9) | [M] `round10_simplify_critique_mixed.out` §2, §4 | The recruit block is a pair feature on recruit pairs; one ridge |
| Threshold | On R the best τ beats τ = 0 by +0.8 ± 1.6 | [M] `round10_simplify_checks.out` §2 | τ̃ kept as a guard |
| Label noise | E[T²] 4.01; residual 0.699 ± 0.003 (0.692 on `own2`); within-game ratio 0.999 | [M] `round9_simplify_labels.out`, `round9_simplify_residual.out` | Frozen b and β; every SE |
| Forecast | SE 4.88 Elo at switched share 0.5, 3.41 at 0.25 | [D] `round9_simplify_budget.out` | The screens' lines |
| Walls | A fixed 69 ± 30 s a match; 155.0 ± 3.3 ms a game; GPU busy 0.77 of the wall | [M] `round10_provenance_r_link.out` §4 | o read from equal-size runs (SE 0.024 at 10,000 games [D, `round10_provenance_derived.out` §1]); the host test runs 10,000 games, since 2,000 overstate cost by 17% (§2) |
| Prices, $0.577/h host | Coin game all-in $2.849e-5; pool game $2.598e-5; two-network game $8.848e-5 | [M] `round9_simplify_labels.out` | Every cost |
| Claims | Per-round checks, false miss 0.042–0.053; the lower tail alone 0.023–0.048 | [D] `round9_synthesis_checks.out` §5; `round10_simplify_critique_claimstop.out` | One-sided stop |
| Transferred windows | S0(a) read −41.3 ± 4.4 against −76 ± 17 from decode-knob records | [M] seed §7 | Windows from other networks are reports |
| First-mover effect | +9.3 ± 1.4 Elo | [M] `round9_challenge_critique_checks.out` A | I-side |

**Still to measure**

| Quantity | Cheapest way | Cost [D, *budget* §3] | Decision it feeds |
|---|---|---|---|
| The coin-to-uncoined link at the capped step | L | $0.98 | Whether any round is bought |
| Capped Bd on `own2` | T, iff (h) | $0.98 | π_0 |
| π_0 against `own1` | C, iff T changes π_0 | $1.13 | The claim chain |
| `own2`'s gates, census, E[T²], processor time, the server's dump timer; R² of Bd's values and of d on the dumped recruit x | The smoke, 3,000 dumped coin games | $0.09 | Gates; whether the token span carries the recruit structure (report) |
| Staging minutes (15 of the 22 are [C]) | Box A's log | $0 | Every setup; the third slot's margin |
| The dump overhead o | Batch 1's match 1 undumped and match 2 dumped, 10,000 each | inside S1 | Rounds that fit |
| v per type, τ̃, each screen's forecast | S1 | inside S1 | The round's gain |
| The second-order loss against the step, and the link on the new base | S2's step search | $1.98 | Adoption; round 2's grid |
| A host's cost per coin game | One 10,000-game coin match on the candidate | ≤ $0.29 + setup | The third slot with the reset |
| Whether the $5 window counted | Ask the user | $0 | The third slot |
| ĉ, ρ̂ of the reset | S4, held out | $3.90 | The next base |
| Branch S: s used, transfer, M2 | B1–B3 | $4.70 at $0.42/h, $5.67 at $0.577/h | Base choice |
| Herded argmax in play | H, iff ruling (i) | $0.98 | The recruit decode |
| Compounding, exploitability, human level | Each claim; none; none | — | Nothing rests on them |

## 3. The algorithm

### 3.1 Base and generation

- **π_0.** `own2` at `raw:t0+eo-1.5` with 64 memory slots; or `own2` + capped Bd if T adopts it under ruling (h).
- **Coins.** At a decision whose top two joint log-priors after the offset differ by G < 0.35 [C], a coin keyed by seed, side and decision index deviates from π_k's choice with probability ¼ [C, built]. P(a1) is logged per decision: ¼ where π_k plays a0, ¾ where it plays a1. The first match on a π_k that switches gates this code through I-coin, I-log and the coin plant.
- **Logs.** Per band decision: P, D, pair type, mover, G, both candidates with their unit identities, the enemy faction, and the side's earlier recruits of each candidate type. Per game: z and the seven luck sums S_g.
- **Dump.** The inference server writes x = φ(a1) − φ(a0) per band decision in bf16. An offline processor makes one sequential pass per match, at 2.40 GB a match [D, `round10_provenance_derived.out` §4]. It computes the Gram matrix G_j, the vector b_j, sandwich sums Σ T²xxᵀ, the plant sums and the per-block progressive losses, then deletes the match's x. It keeps x for a fixed 1,000-game subsample.

### 3.2 The label

    T_i = s_i ω_i R_g,   ω_i = (D_i − P_i)/(P_i(1 − P_i)),   R_g = z − b(cell) − βᵀS_g

- **Baselines.** b and β are fitted once on `own1`'s 60,000 coin games and frozen.
- **Claim.** E[T_i | H_i] = Q̃(H_i, a1) − Q̃(H_i, a0) under the coin policy, for any coefficients fixed before the game. Under fog it is the information-state advantage.
- **Caps.** Capped games are dropped (23 of 60,023 [M]).

### 3.3 The head and the switch rule

- **Features.** φ = [actor token output, q(actor) ⊙ k(target hex), four factored log-probabilities], d 772. If ruling (h) allows, recruit pairs of different units add Bd's 301 pair columns: e(u1) − e(u0) over 43 units, the same over unit × enemy faction, and d = n1 − n0, the side's earlier recruits of a1's type less a0's. These columns are zero on every other pair.
- **Fit.** ŵ = (Σ_j G_j + λI)⁻¹ Σ_j b_j, one ridge for all columns, with λ̂ minimising the progressive loss. Earlier rounds' statistics are pooled iff the progressive loss says so.
- **Switch rule.** τ̃ maximises the unbiased welfare curve W(τ) = Σ_{f_i > τ} T_i of student j − 1 on batch j ([Kitagawa & Tetenov 2018](https://doi.org/10.3982/ECTA13288)) over the latest four pairs if it beats τ = 0 by 2 SE, else 0. The head reorders only the network's top two at a band decision, so every switch it makes is labelled.

### 3.4 The step search, adoption, claims

- **Screens.** Each screen plays π_k + the head on one seat, picked by a per-game seat coin (I-side), against π_k: 12,800 decisive games, SE 3.07 [D]. Three step sizes, each with a forecast sealed from batches 5–8 (students 4–7):
  - **full:** every change with f past τ̃; first-order forecast.
  - **quarter:** the top quarter of those changes by predicted gain within each pair type; first-order forecast.
  - **capped:** each side's first two changes a game; its exact coin-world value (rank 1 weighted 3 or ⅓ by the rank-0 coin), the one forecast with no second-order unknown.
- **Confirmation and adoption.** The highest screen is replayed on fresh seeds, 37,800 games (SE 1.79). Adoption needs its lower 2-SE bound above 0. False adoption is 0.023 under an all-null truth, whatever the screens selected. At illustrative truths of (2, 6, 5) Elo, adoption is 0.80 and mean regret 0.74 [D, *budget* §2].
- **Lines.**
  - A screen's miss line is 11.5 at forecast SE 4.88 and 9.2 at 3.41.
  - The confirmation's line is 10.4 [D].
- **Claims.** After an adoption, π_k plays `own1` in a two-network match of 12,800 games. D_k = claim_k − claim_{k−1} is checked against the confirmation's reading at 9.4 Elo, or 11.2 against S0(b) [D].
  - A lower-tail miss means the round's gain does not reach `own1`. It stops the chain until the user decides.
  - An upper-tail miss is reported.

### 3.5 Invariants and plants

| Invariant | Check |
|---|---|
| I-coin, I-log | Coins match P (χ², capped games too) and are independent of dice and pre-coin quantities; every logged decision played as the coin chose |
| I2, I3 | An enumerable duel through the production generator gives E[T] = Q(a1) − Q(a0) exactly; a post-coin variate fails it; a swap flips x and T |
| I-φ | The dump's log-probability block reproduces the logged gap, by separate code |
| I-φ2 (new) | The actor head, a linear layer (`wesnoth_ai/model.py` lines 210 and 1047), applied to the dumped actor-token difference, reproduces the actor log-probability difference, since the softmax normaliser cancels within a decision. On same-actor pairs, the q⊙k block summed and divided by √d reproduces the target log-probability difference (lines 1053–1055). Run at bf16 tolerance on the smoke, every batch and every screen's seat. A wrong token index or a swap confined to the token block fails it |
| I-proc | The statistics recompute from the stored subsample |
| I-V | Per pair type, the mean of sω(b + βᵀS) is 0 at the family-wise 5% line |
| I-cap, I-key | On a capped rule's seat, every rule decision logs f, its rank and the switch. At most two fires a side-game, each played a1, all recomputed by separate code. The fire count is above 0 on every faction. The enemy-faction key equals the setup's |
| I-match | On a head's seat, f, the screen's mask and the switch are logged at every band decision, with x on a 1,000-game subsample; every switch played as the mask says; f recomputes |
| I-side | The seat coin's (seat, side) χ² |

| Plant or known answer | Prediction | Failure reading |
|---|---|---|
| R0's differential test: the processor fed batch 1's records (b ≡ 0, readout_b's β, per-fold sums, x = the recruit pair block) | R's 43 values and F to float tolerance | Any difference |
| Recruit-block plant, offline: z′ = z + Σ_j s_j(D_j − P_j)uᵀx_j, u = ±δ on a third of Bd's keys, parsed by `tools/coin_plant.py` | Slope 1 along u (3 SE) | Wrong mover sign −1; a mis-joined key about 0 |
| Merged feature plant, batches 1–2, at 10 SE | τ̃ leaves 0; planted welfare within 2 SE of τ*'s; fit change as the processor predicts | τ̃ = 0 |
| Coin plant, every match | Slope 1 per mover side (SE 0.0387 smoke, 0.0064 batch [D]) | Side sign −1.005 |
| Dice control, separate code | Smoke [+0.00245, +0.00587], batch [+0.00267, +0.00565] | Flipped result −0.00416 |
| L, on real signal | Capped Bd within its seal ± 7.0 | §4 |
| Distillation (S4) | ĉ above 0 by 2 SE | ĉ near 0 |

## 4. Regimen

Costs are at $0.577 an hour with o = 0 [D, *budget* §3].

| Stage | Cost | Readout and prediction | Decision per outcome |
|---|---|---|---|
| **R0**, offline | $0 | Bd refitted on batch 1 with each game's enemy faction read from its record (the 6% of rows this synthesis lacked), λ by progressive loss, hashed. Its capped exact value sealed, now +24.1 ± 3.0 Elo with 94% coverage, with its per-faction split and its turn-1 state shift. The differential test and the plants. CI: I2, I3, I-match, I-cap, I-side, I-φ, I-φ2 on a CPU forward | Red: fix before any box |
| **Box A: L** | $0.98 | `own1` + capped Bd on one seat against `own1`, uncoined, 37,800 games. Sealed at the R0 value ± 7.0 (now 24.1 ± 7.0); null line 3.57, which sits 5.9 SE below the seal [D, *budget* §1]. Reported: fire rate by faction, I-cap | **Inside:** first-order labels price uncoined play at a step whose interaction is measured near zero; S1 is proposed. **Below, lower bound > 0:** a partial link. Own interactions are measured near zero and the turn-1 state shift at −1.2 ± 1.3, so the shortfall is the coined continuation. S1 is proposed; S2's capped screen reads the same link on `own2`. **Null** (ratio below 0.30): if I-cap and the label path are clean, coined continuations do not price uncoined ones. No round is bought. The user chooses: one coined side per game (2.0× games per label) or p 0.1 (2.08×) [D], which shrink the coin world's distance from play, or Branch S, the reset, Branch L. **Above:** reported |
| **Box A: T**, iff (h) | $0.98 | `own2` + capped Bd against `own2`. Sealed J_T = J_L, difference SE 2.53, line 5.1. The basis: `own2`'s collapse equals `own1`'s | **Lower bound > 0:** π_0 = `own2` + capped Bd. **\|J_T − J_L\| > 5.1:** the rule depends on the base; reported. **Lower bound ≤ 0, L positive:** π_0 = `own2` |
| **C**, iff π_0 changed | $1.13 | π_0 against `own1`; D against T's reading, line 11.2 | **Inside:** the chain starts at π_0. **Lower tail:** stop and report. **Upper tail:** reported. The rule was fitted on `own1`'s self-play, so a gain specific to `own1` shows here |
| **Smoke** (box A) | $0.09 | 3,000 dumped `own2` coin games: I-coin, I-log, coin plant, dice control, I-φ, I-φ2, I-proc. Reports: census, E[T²], dump timer, processor time, R² of Bd's values and of d on the recruit x (deterministic targets) | A gate fails: debug. The R² is a report, no threshold |
| **S1**, batches 1–8 | $22.79 (1 + o) | Match 1 undumped and match 2 dumped give o, with SE 0.024. Every match: plants, dice control, I-V. Batch 8: λ̂, τ̃, v̂ per type, per-block learning curves, three sealed forecasts. v is not predicted | **o above 0.59** (0.64 without T): one round fits, and the user is told before match 3. A check fails: the batch is quarantined |
| **S2**, step search + claim | $1.98 + $1.13 | Three screens against their forecasts; the best confirmed; the claim check | **Adopt** iff the confirmation's lower bound > 0, then claim. **The capped screen inside its complete forecast, the larger ones below theirs:** the loss is second order and grows with the step, so round 2's grid moves down. **The capped screen below its forecast:** the link fails on this base, read beside L. **All screens null, forecasts not above 0:** S3-null |
| **Round 2** | $26.11 (o = 0) | As S1–S2 on π_1; pooling by progressive loss | As S2 |
| **Third slot** | reset path $30.17; third refit $26.11 | The user's choice after round 2. The reset needs the optimizer discussion and a host test. The refit re-prices the same band | As S2 |
| **S3-null** | when chosen | v̂ per type, per-type welfare curves, the forecasts' upper bounds | The user chooses: band 0.7, Branch S's base, the reset, or stop |

**Budget** [D, *budget* §3], from $82.34 left if the $5 window counted:
- **With T and C:** through round 2 costs $55.62, leaving $26.72; two rounds fit iff o ≤ 0.59. The third slot fits as a refit iff o ≤ 0.009. With the reset it fits iff a host's all-in cost per coin game is ≤ $2.705e-5 (0.950× the measured).
- **Without (h):** through round 2 costs $53.51. A third refit fits iff o ≤ 0.040; the reset iff the cost per coin game is ≤ $2.793e-5.
- **If the window did not count:** a refit fits iff o ≤ 0.048 (with T) or 0.080 (without).
- **A null round 1** after box A costs $26.26–$27.24.
- These lines omit the hook's overhead and the processor's time, which the smoke and S1 measure before the third slot is chosen.
- A round's batches take 39.5 h.

**Branch S** (rulings (f), (g); it trains weights, so it waits for the optimizer discussion) is unchanged: B1 $2.59–$3.56 by host, B2 $0.98, B3 $1.13. **Branch L,** if ruling (a) is no, is unchanged. **The $1,000 case** is argued only after two adopted rounds pass their checks. It is sized from measured v̂, the step search's second-order curve and ρ̂. Nothing measured links `own1` to human ratings, and nothing rests on it.

## 5. What is new, and the nearest work

Searched 2026-10-21; finding nothing does not prove absence.
1. **Near-tie coins at an argmax decode,** made exact by the chance laws, pooled by a closed-form natural-gradient head and certified by match, as in round 9. The nearest designs randomise only near a cutoff ([Owen & Varian 2020](https://arxiv.org/abs/1808.07563); [Cattaneo, Frandsen & Titiunik 2015](https://www.doi.org/10.1515/JCI-2013-0010)). Here about 155 coins share one result.
2. **A probe built so its forecast is complete.** Capping a rule at two decisions a side-game makes its performance-difference terms exact from behaviour data, with importance weights over at most one earlier coin. The full expansion is TayPO's ([Tang, Valko & Munos 2020](https://arxiv.org/abs/2003.06259); its second-order form was read by the provenance critique). The same records show the uncapped second-order term large and the expansion unreliable, which is why the probe is capped. One search found no estimator for policies that deviate at most k times. The nearest is state-based importance sampling, which drops ratios where an action cannot affect the return (abstract-level: [deepai listing](https://deepai.org/publication/low-variance-off-policy-evaluation-with-state-based-importance-sampling)).
3. **A step search on the true objective.** The screens run along a label-fitted direction, each with an exact first-order forecast, one complete, and a fresh confirmation.
   - CPI+ line-searches the step on a 2^i α grid but does not state its criterion ([Scherrer 2014](https://arxiv.org/abs/1405.2878), read).
   - HCPI selects a candidate on one data split and tests it on another ([Thomas, Theocharous & Ghavamzadeh 2015](https://proceedings.mlr.press/v37/thomas15.html); the split is from its erratum, unverified in the body).
   - CLOP and SPSA tune engines by match outcomes along fitted or random directions.

   The assembly was not found.
4. **Findings about the network,** not method claims:
   - its argmax recruits one type where humans field three;
   - its near-tie recruit errors depend on the matchup and on the side's composition, and are not a decode artifact;
   - every fitted rule over them loses value to its own later switches.

## 6. Weakest points

1. **Outside recruits, nothing is certified.** A null round 1 costs $26.26–$27.24.
2. **The second-order loss at the head's scale is unknown.** On recruits a full fitted rule loses 8–13 Elo at about 2.2 switches a side-game. The head may switch 19–38 times, about 0.25–0.5 of 77.6 band decisions [D]. If every uncapped screen loses and the capped one carries little, round 1 adopts a small step.
3. **Money.** The third slot fits only at o near 0, or on a cheaper host.
4. **Rulings (a), (e), (h)** decide whether the coin route plays at all and whether recruit structure enters a player before the reset.
5. **The band moves only through the reset,** which waits for the optimizer discussion.

## 7. Rulings for the user

- **(a), unchanged:** one natural policy-gradient step for a near-tie coin policy, adopted only by match.
- **(b), unchanged:** luck columns and the frozen cell baseline in labels only.
- **(c):** per-round claims; a lower-tail miss stops the chain.
- **(d), (f), (g), unchanged.**
- **(e):** do a closed-form head on frozen features, and the capped rule played in L and T, fall under the 2026-10-15 ruling? L and T play them as measurements.
- **(h), revised:** may the player carry Bd's 301 columns on recruit pairs (unit identity, unit × enemy faction, the side's earlier recruits of each type)? Both forms are covered: the capped rule in π_0 and head columns. If not, T and C do not run, and the recruit structure reaches the player through the token span or the reset.
- **(i), new (invent):** may the player decode recruits by herding its own probabilities? If yes, H runs in box A with the critique's repairs. If no, herding is not used.
- **Budget:** did the $5 window come out of the $85? And after round 2, the reset or a third refit?

## 8. Decision record

**Rejected or deferred this round:**
- **Sealing a full rule's F** (round 9's L). Second-order terms of −12.3 ± 4.5 (R) and −13.1 ± 4.7 (Bd), with a non-convergent expansion, make it no prediction of play. A miss could not separate own interactions from the link.
- **Sealing at the 20-fold F.** It is a forking path between fold schemes (+21.3, +19.9, +18.0). A fresh match is the leave-one-match-out condition.
- **"Own interactions cannot produce a null"** (provenance). On its own rows the second-order term reads −12.3 ± 4.5, and the first-coin coefficient is 0.5, not 0.75 (−2.3 ± 2.8).
- **H1** (simplify). Students never play in generation, so its only action moves into S2's screens.
- **E as a separate stage.** It is absorbed as a screen, for $0.02 more than S2 plus its reserve.
- **The class-wide shift past the band, and η screens** (challenge). Past-band flips and third-candidate flips are unlabelled, so they have no forecast. The screens vary the switch set within the top two.
- **The attack hook as the shift path's known answer.** A within-class shift of a constant changes no decision (challenge critique §2.1).
- **30,000-game recruit iterations with a two-null stop.** The iteration size was measured on `own1`'s first step only. They are deferred until round 1's per-block learning curves show when the recruit block saturates.
- **Herded argmax in the regimen.** Deferred to ruling (i). Inside the band it is not the count rule, since it depends on the unmeasured ΔQ. As specified, its T → 0 limit is not the reference decode, and W under overrides is unspecified. The repairs if run: q from best-hex joint priors, W following the played type, per-faction readings, the bottom row conditional on (h). Kept from it: the collapse, the count column d, composition statistics reported on every head seat.
- **κ_comp as a gate.** Its target of 1 is not a derivation.
- **The 2,000/1,000 smoke for o.** It reads −0.154 at o = 0. Likewise the 2,000-game host test, which reads 17% high.
- **A futility stop.** It has no trigger without H1, and its power cost cannot be derived.
- **Removing T and C unconditionally** (simplify). They stay under (h): T is the cheapest bankable gain on record.
- **Moving the reset and Branch S out of the plan** (simplify). They stay, gated on the optimizer discussion; the third slot is the user's choice.
- **The per-candidate unit block.** It adds 2.36 switches worth +0.2 ± 3.3.
- **The adjacent-switch forecast correction.** Its SE is 5.7–8.2 at four batches.
- **The "~$7" critic column.** It has no derivation and is off the menu until priced.

**Kept from simplify:** frozen b and β, the dump and offline processor, the merged plant, R0's differential test, one ridge, no typed block, the one-sided claim stop, the R² report, I-φ2 (from its critique).

**Record corrections:**
- The "78×" information ratio holds for the whole band; on one rule's switch set it is 4.8–5.2×. The "42–87×" is 5.2× for B.
- Farias et al.'s abstract says "a large penalty in variance", not "exponentially larger".
- 0.56 is an odds ratio.
- B1's cost depends on its host.
- "Logits + ηf" is Gumbel MuZero's improved-policy form.
- HA's bound is proven without masks only.

**Standing:** everything rounds 8 and 9 rejected (factorial core, twins deferred, the batch-1 replay, the conditional MLE, scaling forecasts by L's ratio, an R² threshold, the offset block); single-decision comparisons only; logged propensities; fresh-seed adoption; PURE claims; nothing distilled before a match certifies it.
