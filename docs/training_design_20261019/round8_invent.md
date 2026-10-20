# Round 8, invent stance: the dice as the second experiment in every coin game

Tags: **[M]** measured (record and script named), **[D]** derived (inputs named), **[U]** unknown (measured before anything relies on it, or nothing relies on it), **[C]** a choice whose consequences are derived. This stance's script is `analysis/round8_invent_chance_ledger.py` (`.out` beside it), cited as §A–§E.

## Summary

1. The core is the current design's, with the round-8 simplify stance's removals of the interim match, γV and the √N ceiling; costs follow `round8_provenance_plan.out` ($55.50 for S0–S5).
2. Every coin game runs a second randomized experiment with exact propensities: each strike hits with a known chance. At the round-1 floor the coins' effects explain 7.3e-5 of Var(z); the dice explain all within-cell variance, 0.843, about 11,500 times more [D, §C]. Seven hand-weighted columns capture 0.155 of it [M].
3. **Now, free:** the result's response to the dice is a known answer for the label path. The slope of z on side A's hp luck is consistent with one value, +0.00416, across 23 player pairs (χ² 24.8 on 22 df) [M, §E]; a result oriented to the wrong side reads −0.0042, about 100 SE away in a batch [D, §B]. It catches, on every batch, the orientation error the coin plant is blind to.
4. **At the reset, new:** chance-effect supervision (CES). Each strike's inverse-propensity label estimates, exactly unbiased, how much that hit changed the result. A head on the trunk learns them during the distillation, giving an exactly centred learned control variate for later labels, a trunk representation of what consequences are worth, and a yardstick for evaluators. A random input flag carries a planted effect of known size.
5. Measured against CES's premise: conditioning the luck price on the setup removes nothing (+0.0039 ± 0.0011 to the kept ratio) [M, §A]. Whatever a learned price adds must come from the fight's own context.

## 1. What the records say

| Reading | Value | Tag, source |
|---|---|---|
| Columns × factions / map / seat / all | kept ratio change +0.0039 ± 0.0011, +0.0029 ± 0.0019, −0.0005 ± 0.0007, +0.0139 ± 0.0049, from 0.847 | [M] §A, leave-one-match-out as `round6_provenance_cells.py` |
| hp slope | own1 self-play +0.00438 ± 0.00032; 23 matches +0.00416, range +0.00327 to +0.00513 | [M] §B, §E |
| Centring | Var(S_hp) 5598 against summed law variances 5486 (ratio 1.020) | [M] §D |
| Floor against kept ratio r | sd 0.0016·√(r/0.688): 0.00106 at r 0.3 | [D] §C, the spread estimator's σ² scaling (`round8_provenance_spread_estimator.out`: 0.0016 at σ² 3.67, 0.0019 at 5.33) |

## 2. The algorithm

### 2.1 Unchanged

As `current_design.md` §3: near-tie coins (w 0.35, p 0.25, logged propensities), the label T_i = s_i ω_i R_g with R_g = z − b(cell) − βᵀS_g and forward-chained coefficients, the ridge student (d 772, antisymmetric typed terms, sign rule), the sealed Ĝ_pf, adoption by a 37,800-game PURE match (SE 1.79 Elo), fallback E, rounds, and the own-trajectory reset with the baked end_turn shift. The network picks single actions; every comparison is one decision followed by play that re-decides after every roll.

### 2.2 Strike coins

The core records each strike's chance to hit, whether it hit, its damage and whether the target died (`rust/wesnoth_core/src/combat.rs:425–447`). A strike hits iff `draw() % 100 < cth` on a u32 MT19937 draw. Its exact hit probability is c = #{r < 2³²: r mod 100 < cth}/2³², which differs from cth/100 by less than 3e-8.

For strike k with striker side s_k:

    T_k = s_k (H_k − c_k) / (c_k(1 − c_k)) · R_k,   R_k = z − b(cell) − βᵀ(S_g − L_f(k)),

where L_f(k) is the luck column vector of the fight containing k, removed because it contains H_k.

**Claim.** E[T_k | state before k] = Δ_k = E[z | H_k = 1] − E[z | H_k = 0], oriented to the striker.

**Proof.** The cell and the earlier fights' luck are fixed by the past; each later fight's luck has mean zero given the state before it, which contains H_k; and E[(H − c) z | past] = c(1 − c) Δ_k. ∎ It is the coin label's proof with the dice in place of the coin.

Strikes ride in the game record's per-attack optional data (`tools/game_record.py`, "outcomes"). Logging costs no forward; its overhead is read in the S0 smoke with the design's other logging.

### 2.3 Use 1, now: the dice positive control

The coin plant reads 1.006 even when the result field is flipped, while real labels change sign (`round7_simplify_critique_checks.py` §3).
- **Reading.** In every batch, separate code fits the slope of the label path's z (side A, as the labels read it) on the walk's S_hp (side A, as `tools/match_luck.py` computes it from the record). z and S_g are already logged, so it costs nothing.
- **Prediction.** Inside the 23-match range, +0.0033 to +0.0051 [M, §E]; SE 0.00029 at the smoke's 2,000 games [D: 0.00032 × √(1600/2000)], 0.00004 per batch [D, §B].
- **Failure readings.** A flipped result reads −0.0042; results flipped on one seat, or residuals joined to the wrong game, read about 0.
- **Coverage.** With the coin plant (s, ω, D, P per mover side), I9 and I2, the coin path's arithmetic and orientation are tested on real games every batch. S0(b) keeps its 6,400 games for the match path (window −78 ± 33 Elo).

### 2.4 Use 2, at the reset: chance-effect supervision

**Head.** f_θ(t_s, t_d, g, e_k, F_k) → Δ̂_k, on the trunk's outputs for striker, defender and global token at the attack decision (all visible to the attacker), the strike's descriptors e_k (whose strike, damage, whether a hit kills, both fighters' hp), and a random flag F_k.

**Loss.** Squared error to T′_k, the strike label computed with z′ (below), on the coin games of rounds 1–2. Over the head's inputs, its population minimiser is E[Δ_k | inputs] + κ F_k d_k (damage and the flag are inputs). The head's share of the trunk's update norm is capped at 0.1 [C]; in the record, the value gradient took 94–99% of every update (seed §3).

**Plant in the same run.** F_k ~ Bernoulli(½) is keyed by seed and logged. The CES labels use

    z′ = z + κ Σ_j F_j s_j (H_j − c_j) d_j.

Expanding E[T′_k | past] gives κ F_k d_k in addition to Δ_k, since E[ω_k (H_k − c_k)] = 1 and the cross terms vanish.
- **Reading.** On held-out strikes, the head is evaluated with F toggled. The mean output difference divided by d is predicted at κ = 0.0005 per hp. The one-coefficient SE floor is 1.8e-5 over 8 batches [D, §D's formula at share 0.5]; the window is ±2 held-out SE computed at run time.
- **Failure readings.** A sign error reads −κ. A mis-joined (H, c) or a head that does not learn reads about 0.
- **Isolation.** Only the CES labels see z′. Coin labels and policy targets never do.

**Readouts before any match.**
- **(a) Kept ratios.** r_frozen is a closed-form ridge of the same head form on the frozen teacher's tokens, gathered in `tools/own_targets.py`'s teacher pass, which already forwards every position. r_CES is the trained head's ratio. Both are read on held-out coin games and nested with the hand columns: R = z − b − βᵀS − γᵀΣ_k s_k (H_k − c_k) ψ_k, with ψ_k the trained head's output or the frozen tokens' features, coefficients forward-chained. Each is compared with 0.688; the paired SE is about 0.0005 at 100,000 games [D: §A's 0.0011 at 18,800 × √(18,800/100,000)].
- **(b) Closure.** ĉ and ρ̂ are read as in the design.
- **(c) Span.** The next round's v̂ (provenance's estimator) is read against this round's.

**Exactness.** Any f_θ fixed before a batch gives Σ_k s_k (H_k − c_k) f_θ(·) a mean of zero under the coin law and the dice law. It can therefore enter coin labels as a control variate without bias, whatever the head learned. Head quality costs variance, never bias.

**Why this is not the critic that was killed.** The record's critics regressed one result per game on about 24 positions: levels, an effective sample near the game count, judged by ranking turns. CES estimates a contrast at each randomized event, identified by an exact propensity; its innovations are orthogonal within a game (coin labels' design effect is 1.002 in provenance's simulation, `round8_provenance_label_variance.out`; strike labels' is read at S4); it is judged by variance removed under exact centring, which fitted noise cannot pass out of sample. It never ranks decisions: chance contrasts do not identify the level across candidates (round 6's rejection of the exchange score stands), so coins remain the only decision labels.

### 2.5 Why more signal per dollar

The decision labels are capped by deviations. Information per game is Σ p(1 − p)A²/σ², about 7.3e-5 of Var(z) at the floor [D, §C], and no reweighting changes that. The dice's experiment is about 10⁴ times larger and costs nothing: every game already has about 109 fights [M].

Its value to the decision learner runs through two channels. For labels, the floor scales as √r: a learned price at r 0.3 is worth 2.29 times the coin games [D, §C]. For the span, a trunk trained on what hits are worth in context is a candidate source of the spread v that round 1 may lack. Against the rest of the corrector class: hand columns are free but capture 0.155 of 0.843 and saturate [M]; setup conditioning captures nothing [M, §A]; a child-forward value costs 2.0–3.9× the forwards [D, round 5]; ACS costs a forward per fight [D, round 7]. CES costs no generation forward, only a head inside a distillation that runs anyway.

## 3. Regimen

| Stage | Cost [D] | Readout and prediction | Decision per outcome |
|---|---|---|---|
| **S0** | $0.18 | As the current plan. The smoke also reads the overhead of strike logging. | Overhead above the three-round line ($3.141e-5 a game × (1 + overhead), `round8_provenance_plan.out`): strikes are not logged and CES is dropped, and nothing else changes. |
| **S1** | $15.30 | The dice control on the smoke and on each batch: inside +0.0033 to +0.0051. The design's plants and readings. | **Outside:** the result's orientation or the residual join is broken; stop and debug. **Otherwise:** as the design. |
| **S2** | $0.72 (+ E $1.45 reserve) | As the design | As the design |
| **S3** | $16.74 | As S1 | As the design |
| **S4** (after the optimizer discussion and ruling (f)) | $3.60 + head ≤ $0.51 [D: 108.8 of 449.1 positions × $2.1]; + $2.10 for a control only on a closure miss | **(1)** The plant reads κ̂ within its window. **(2)** r_frozen and r_CES against 0.688, no prediction (§5). **(3)** ĉ and ρ̂ as the design. | **Plant outside its window:** the CES path is broken; distil without it. **r_CES below 0.688 by 2 SE:** the CES columns, nested with the hand columns, enter round 3's labels; the floor becomes 0.0016·√(r_CES/0.688). **r_frozen also below:** the frozen tokens already price hits, so a generation-time gather is worth building for later rounds. **Neither below:** learned pricing of hits on this lineage is closed. **ĉ or ρ̂ miss:** the no-CES distillation separates CES's cost from the target's. |
| **S5** | $17.52 | As the design, plus v̂ against rounds 1–2 | Adopt by the rule |
| **Total** | $56.01, + $2.10 conditional | | |

**Beyond the yardstick.** The $1,000 case is argued only on measured trends across cycles of a coin round and a reset with CES: J/Ĝ_pf within miss lines, r falling (cheaper labels), v̂ rising (a richer span). If neither r nor v̂ moves over two cycles, the coin route has saturated on this lineage and the case is not made. Nothing measured links `own1` to human ratings, and nothing rests on it.

**Branch L** (ruling (a) no) is unchanged. The dice control applies to its matches once walked.

## 4. What is new, and the nearest work

Searched 2026-10-20 (WebSearch, three queries on chance-outcome supervision); finding nothing does not prove absence.

1. **CES.** Nearest: MIVAT learns a value function minimising an evaluation estimator's variance ([White & Bowling 2009](https://webdocs.cs.ualberta.ca/%7Ebowling/papers/09ijcai-mivat.pdf)); DeepStack corrected its matches with AIVAT and its own value net ([Moravčík et al. 2017](https://arxiv.org/abs/1701.01724), via round 6's critique); GNU Backgammon's rollouts subtract the luck its own equities estimate ([manual](https://www.gnu.org/software/gnubg/manual/html_node/Rollouts-in-GNU-Backgammon.html)). Baselines from exogenous randomness: [Mesnard et al. 2021](https://arxiv.org/abs/2011.09464), [Mao et al. 2019](https://arxiv.org/abs/1807.02264). Stochastic MuZero models chance outcomes for planning, trained through value and policy losses (Antonoglou et al., ICLR 2022; not re-read). Auxiliary tasks: [Jaderberg et al. 2017](https://arxiv.org/abs/1611.05397), [Zheng et al. 2021](https://arxiv.org/abs/2102.04897). Not found: a policy trunk trained on inverse-propensity labels of each chance event's effect on the result under the exact law, used both as the exactly centred control variate of randomized-decision labels and as an evaluator yardstick.
2. **The dice positive control.** Positive and negative controls (epidemiology) and A/A tests (online experiments) are standard. Not found: a natural randomized experiment inside the same games, its slope measured on independent games, as the orientation check of a treatment-effect pipeline.
3. **Derived, not methods:** the 10⁴ ratio, and that the span cannot be piloted (§6).

## 5. Weakest point

**CES's value is a hypothesis, and the record leans against its premise.** Hand prices saturate at 0.152 of Var(z) (`round6_invent_conditioning.out`), setup context adds nothing [M, §A], and the only context that helped was the board margin, at 0.042 (round 6). If r_CES lands near 0.688 and round 3's v̂ does not exceed round 2's, the reset carried a head, a ruling and a code path for a negative result, which still closes learned consequence pricing on this lineage for the $1,000 case.

**The cap may not protect closure.** The 0.1 cap [C] is not derived. A closure miss is detectable (ĉ, ρ̂) and attributable only by the $2.10 control.

**The dice control is small.** It adds a check, not gain.

**v remains unmeasured.** Neither component touches round 1's v, and a null round 1 still costs $16.20 [D, provenance plan].

## 6. Decision record

**Rejected or deferred this round:**
- **A rich frozen price as round 1's control variate** (round 6's DPI with full tokens), deferred to S4's free reading (r_frozen). It would fit cheaply on a coin batch (overfit 0.011 at d 1,536 [D, §C]), but §A's null and round 6's saturation do not justify a generation-time token gather before S1.
- **Setup-conditioned luck prices.** Measured null (§A).
- **Afterstate features, or any span change chosen on a pilot.** A batch-1 half bounds v only at sd 0.0052, three times the round's floor (`round8_provenance_spread_estimator.out`), so a feature set can be judged only by a full round [D].
- **Sentinel coins at confident decisions,** deferred to S3-null as option (vii): 148.5 non-band decisions per side-game [D: 224.6 × 0.661]; at p 0.02, 3.0 deviations per side-game [D], each costing about 0.005 (attack-promoting switches at offsets 0.35–1.4 [M bound, `round8_provenance_attack_type.out`]); label weight variance 51 against 5.33 [D]. Nothing measures the mass of positive advantages there.
- **Prefix-branched coin games with leave-one-out baselines.** Their gain is the variance resolved before the branch, unmeasured for `own1`, and they need a state-and-memory snapshot path in the pool.

**Rulings:**
- (a)–(e) are unchanged (`current_design.md` §7).
- **(b) may simplify:** where CES alone matches the nested fit, the labels need no hand price.
- **(f), new:** at the reset, may an auxiliary head be trained into the trunk on the results' measured response to randomized hits, its gradient share capped? If no, CES is not built; nothing else changes.

## Provenance

| Number | Value | Tag | Source or derivation | Used for |
|---|---|---|---|---|
| Hand columns kept, pooled / own1 self-play | 0.847 / 0.839 | M | §A (leave-one-match-out, as `round6_provenance_cells.py`) | Baseline of §A |
| Setup-conditioned columns | +0.0039 ± 0.0011 (factions), +0.0029 ± 0.0019 (map), −0.0005 ± 0.0007 (seat), +0.0139 ± 0.0049 (all) | M | §A | §6 rejection; §5 |
| Cell + columns; cell alone | 0.688; 0.843 | M | `round6_provenance_cells.out`; `round8_simplify.md` provenance | Ratio, floors |
| Chance share at argmax | 0.843 of Var(z) | D | Seed §3: with setup fixed at argmax all variance is chance; cell alone keeps 0.843 | Ratio |
| Coins' own share at the floors | 7.3e-5 (μ 0, sd 0.0016); 4.2e-4 (μ −0.0018, sd 0.0034) | D | 2 × 76.1 × 0.1875 × (sd² + μ²)/1.0004, §C | Ratio |
| Dice/coin ratio | 11,544; 1,997 | D | 0.843 / the above, §C | §2.5 |
| Hand columns' capture of chance | 0.155 | M/D | 0.843 − 0.688 | §2.5 |
| Hand price saturation; board-margin gain | 0.152 explained; 0.042 | M | `round6_invent_conditioning.out` | §5 |
| hp slope, own1 self-play | +0.00438 ± 0.00032 | M | §B | Control |
| hp slope, 23 matches | +0.00416 weighted; range +0.00327 to +0.00513; χ² 24.8 / 22 df | M | §E | Control window |
| Control SE | 0.00029 at 2,000 games; 0.00004 per batch | D | 0.00032 × √(1600/2000); §B | Smoke and batch gates |
| Flipped-result reading | −0.0042, t ≈ −100 a batch | D | §B | Control failure reading |
| Strike record and hit rule | (chance, hits, damage, dies); hit iff draw % 100 < cth | M | `rust/wesnoth_core/src/combat.rs:425–447` | §2.2 |
| Exact hit probability error | < 3e-8 | D | 2³² mod 100 = 96 partial values over 2³² draws | §2.2 |
| Centring ratio | 1.020 | M | §D | Exactness check |
| Attacks a game; decisions a game | 108.8; 449.1 | M | §D; `round5_synthesis_outcomes.out` | Head cost |
| Floor against r | 0.0016·√(r/0.688); 2.29× games at r 0.3 | D | §C; spread-estimator σ² scaling | §2.5, S4 |
| Rich-price overfit | 0.0106 at d 1,536, one batch | D | r·d/N, §C | §6 |
| Paired SE of a kept ratio | ≈ 0.0005 at 100,000 games | D | 0.0011 × √(18,800/100,000), §A | S4 gate |
| Plant κ; one-coefficient SE | 0.0005 per hp; 1.8e-5 over 8 batches at share 0.5 | C; D | §D formula √(0.688/(N·share·5486)) | S4 plant |
| Label design effect | 1.002 | D (simulated) | `round8_provenance_label_variance.out`, a toy of coin labels | §2.4 |
| Value-gradient share in the record | 94–99% | M | Seed §3 | Cap |
| Gradient cap | 0.1 of the trunk's update norm | C | — | S4 |
| Distillation cost | $2.1 (upper bound) | M | `docs/memory_own_trajectories_prereg_20261011.md` via `round8_simplify.md` | S4 head bound, control |
| Head cost bound | ≤ $0.51 | D | 108.8/449.1 × $2.1 (a trunk pass per attack position, worst case) | Budget |
| Plan | $55.50; null round 1 $16.20; three-round line $3.141e-5 | D | `round8_provenance_plan.out` | Regimen |
| This plan | $56.01, + $2.10 conditional | D | $55.50 + $0.51 (+ $2.10) | Budget |
| Spread bound from a batch-1 half | sd 0.0052 | D | `round8_provenance_spread_estimator.out` | §6 |
| Non-band decisions; sentinel deviations | 148.5; 3.0 per side-game at p 0.02 | D | 224.6 × 0.661; × 0.02 | §6 |
| Cost per deviation off the band | mean advantage ≤ −0.0045 to −0.0052 per attack-promoting switch, offsets 0.35–1.4 | M (bound) | `round8_provenance_attack_type.out` | §6 |
| Match SE; S0(b) window | 1.79 at 37,800; −78 ± 33 Elo | D; M via D | `current_design.md` §2 | Unchanged stages |
| Strike-logging overhead | — | U | S0 smoke; decides whether strikes are logged | S0 |
| r_frozen, r_CES | — | U | S4, held-out coin games; nothing before S4 uses them | S4 decisions |
| v̂ by round; v | — | U | Provenance's estimator after each round | The $1,000 argument |
| CES's effect on ĉ, ρ̂ | — | U | S4; the control on a miss | S4 |
| Compounding, exploitability, human level | — | U | Not measured; nothing rests on them | — |
| w 0.35, p 0.25, d 772, the match sizes | — | C | As `current_design.md` | Core |
