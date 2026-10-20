# Round 6, provenance stance: the coin-label design with every number traced

The frame of `current_design.md` stands: near-tie coins with logged propensities; labels exactly unbiased through the simulator's chance laws; a closed-form ridge head on the frozen network; adoption by a corrected match against `own1`.

I audited every number it rests on (table at the end) and measured what the records settle. I then changed what rested on a guess or on a wrong derivation:

1. **Cheaper adoption matches.** They run in the generation pool, 4.7 times cheaper per game (from two measured prices).
2. **Lower label noise.** Labels use the full setup cell as baseline, so σ² is 3.67, not 3.91.
3. **Antisymmetric intercepts.** I3 now holds exactly.
4. **An interim match** replaces the early stop.
5. **The step size comes from play.** Coin labels at p > 0 do not estimate the gradient at `own1`, so the current α* formula is biased (section 2.3).

Scripts: `analysis/round6_provenance_*`, each under 10 s; `round5_synthesis_power.py` and `round5_simplify_critique_extra.py` §3 reran identically.

## 1. What the records settle now

**Prices** (`round6_provenance_prices.py`):
- **Generation.** `own1` against itself, 120 games in flight, ref path: 16,744 games an hour, **$2.51e-5 a game**. The bit-identical priors-only path costs $2.45e-5.
- **Host spread.** The same code ran 1.41 times slower on round 4's host (BACKLOG.md).
- **Eval matches.** The 18 recorded plain 800-decisive matches took 654–1,093 s: **$1.17e-4 a decisive game** at the median (range $9.5e-5 to $1.59e-4). The design's $1.49e-4 is near the top of that range.
- **Box setup.** Median 156 s, $0.02.

**So adoption matches move into the generation pool.** The candidate is `own1` plus a linear head on `own1`'s token outputs: one network. Its match against `own1` is the coin generator with p = 0 and the head live on one seat, so it adds no code path, costs **$0.64** for 25,600 games (SE 1.82 Elo [D: 347.4·√(0.700/25,600)]), and plays under the labels' numerics. Strength claims stay PURE on the eval driver (800 games, $0.09).

**Label baseline** (`round6_provenance_cells.py`, which reproduces round 5's pooled shares 0.847 and 0.151). Columns plus the full cell (map, seat, both factions) keep 0.688 of Var(z) in `own1` self-play (pooled cell share; self-play cells alone 0.651, 0.58–0.72), against 0.732 for the additive baseline. At about 66 games a cell per batch, fitting the cell on earlier batches adds 0.0015–0.0105. **σ² = 1.0004 × 0.688 / 0.1875 = 3.67.**

**Distillation transfer** (`round6_provenance_closure.py`).
- **Tempo closure.** The own-trajectory recipe moved `parity3` at 64 slots from 7.34 to 8.46 decisions per side-turn, against a memoryless target of 8.39–8.45. That is a **tempo closure of 1.00–1.06 (95%: 0.82–1.30)**.
- **Agreement closure** on the same run: 0.58.
- **P-distill's teacher.** `own1` at −2.5 plays 9.625 ± 0.076 decisions per side-turn; `own1` at −1.5 plays 8.580 ± 0.065.

**Runner-up attacks** (`round6_provenance_attack_pairs.py`: 24 decode-knob vectors, 6,400 games, kinked fit). Raising every attack logit costs **−0.179 ± 0.043 z per nat** and changes at least 2.35 decisions per side-game per nat, so one change is worth at most −0.076 ± 0.018; lowering it is flat (+0.005 ± 0.019 per removed attack). For this pair type the runner-up's mean is far from 0: intercepts are required, and batch 1 gets a sign check on real decisions.

**Nothing on disk bounds the spread v:** game records hold commands and fight laws but no logits, and the knob records give type means at margins up to 1.4 nats, not near-tie spreads.

## 2. The algorithm

### 2.1 Data and label

As the current design (§3.1–3.2): `own1` at `raw:t0+eo-1.5`, 64 slots; a logged coin plays the runner-up a1 with p = 0.25 where the post-offset top-two gap G < 0.35; T_i = s_i (D_i − p)/(p(1 − p))·(z − c_i − β·S_{>i}), unbiased for Q̃(a1) − Q̃(a0) for any pre-coin c_i. **Change:** c_i's setup term is the full cell fitted on earlier batches (batch 1 cross-fits its halves).

### 2.2 Student with antisymmetric intercepts

The student is a ridge regression of T_i on x_i = φ(a1) − φ(a0), with d = 776, fitted from per-game sufficient statistics. λ is chosen by progressive validation. This choice is exact, because any w's held-out error follows from the next batch's statistics.

**Derivation.** For action kinds u and v, let D_uv(g) = E[Q(v) − Q(u) | l_u − l_v = g].
- Relabelling the pair gives D_vu(g) = −D_uv(−g).
- Continuity at a tie gives D_uu(0) = 0.

**The unpenalised terms** are therefore c_uv + b_uv·G, with c_vu = −c_uv, c_uu = 0 and b_vu = b_uv. That is one constant per unordered pair of distinct kinds, and one margin slope per unordered type.

The prediction then flips sign exactly when numerics flip a pair's order (I3 holds at the 1.76% of flipped decisions), and a same-kind pair's mean vanishes at a tie (moves lead 66 of `parity3`'s 91 band decisions; S0(a) counts the runner-ups' kinds). The design's constant per ordered type broke both.

### 2.3 Why p = 0.25: the coin point is the midpoint

**Setup.** Let x_h be the probability that the decode plays its runner-up at information state h. J(x) is multilinear: a path's probability is a product of terms x_h or 1 − x_h.

**What labels and match each measure.**
- The labels, summed over a switch set S, estimate **exactly** ∇J(x̃)·1_S at the coin point x̃, where x̃ is p on both sides' band states.
- The match measures J(1_S) − J(0) = ∫₀¹ ∇J(t·1_S)·1_S dt.

**The gap between them.** With pairwise interactions H, the label sum minus the realised gain is

  p·1_band′H1_S − ½·1_S′H1_S + p·1_opp′H1_S.

**When the own-side terms cancel.** Suppose switches interact with other own band states as they do with each other: 1_S′H1_S = q·1_band′H1_S, with q the switched share. Then the own-side terms cancel at **p = q/2**: the coin point is the midpoint of the path from `own1` to the candidate.
- Round 1's p = 0.25 matches q = 0.5, the share a calibrated rule switches near μ = 0 [D: 38.1 of 76.1 in the power model; S1 measures q].
- Round 2 keeps p = 0.25 if round 1's match landed within its forecast. After a miss attributed to curvature, it sets p = q̂/2 and pays label noise 0.1875/(p(1 − p)) times round 1's (1.47 at q̂ = 0.3).

**A forecast miss then means** the opponent's coins (p·1_opp′H1_S, absent against an uncoined `own1`), curvature among switches that is not exchangeable, or higher-order terms.

**Correction to the current design.** It sets α* = −G1/(2G2) with G1 taken from the labels. The labels estimate g + A, not g, so the formula computes −(g + A)/(2A). At g = 20 and A = −15 Elo that gives 0.17 where the truth is 0.67 (`round6_provenance_power.out` §3).

**What play identifies.** With J(α) = gα + Aα²:
- g = 4J(½) − J(1), SE 7.5 Elo;
- A = 2J(1) − 4J(½), SE 8.1;
- the labels' offset C = L − g, SE 8.8. C separates the opponent-coin bias from curvature.

### 2.4 Switch, forecast, adoption

**Switch rule.** Play a1 where m̂ + β̂(f − m̂) > 0, with m̂ the intercept part and β̂ the progressive calibration slope.

**Forecast.** Students 4–7 log f on batches 5–8; their labels summed at their own switches give Ĝ_pf (SE 4.6 Elo [D: 4.7·√(0.688/0.732)]), sealed before the match. The candidate, student 8, has an expected label sum at least theirs (in the stated model; a failure would itself be a learning failure), so the comparator is one-sided: **miss** = J(π_1) below Ĝ_pf by more than 2·√(4.6² + 1.82²) = 9.9 Elo.

**Adoption.** 25,600 fresh-seed pool games, in seed-shared mirror pairs. Adopt iff the lower 2-SE bound is above 0 and the correction term's mean is within 2 × 1.10 Elo of 0. False adoption at zero gain is 0.023.

### 2.5 Distillation target

The recipe trains on soft raw targets, so the offset applies once, at play (docs/memory_own_trajectories_prereg_20261011.md). π_k is a rule, not a distribution. Its target is therefore `own1`'s raw distribution with the switched pair's post-offset logits exchanged: the margin is kept and its sign reversed.

Before S4's match, held-out own positions give ρ̂ (the share of switches reproduced) and the agreement elsewhere; the match against π_k is then predicted at −(1 − ρ̂)·Ĝ_k to first order, and a shortfall beyond 2 SE is spillover.

### 2.6 Invariants and plants (changes)

- **I3** is exact (section 2.2).
- **I9, an exact null.** With identical seats and a shared seed, a mirror pair produces the same command stream with seats swapped, so z_A(2) = −z_A(1). This holds up to the first decision whose logged gap lies inside the generation-numerics band, measured in S0(a) by replaying against replay. A divergence at a larger gap is a seat or seed bug.
- **P-coin bands at σ² 3.67:** 1 ± 0.069 and 1 ± 0.097.
- **Sign check.** The label mean of (non-attack top, attack runner-up) pairs must not be positive at 2 SE (SE 0.0061 at 0.5 such pairs per side-game, known once S0(a) counts them). The plants cannot catch a mover-sign error, since one convention books and reads them; this check comes from real matches.
- **P-distill.** Plant: a soft raw target with end_turn lowered by 1.0, at the measured run's scale (1.4 M own positions plus as many human). The student at −1.5 must play 8.580 + c × 1.045 decisions per side-turn, c in [0.82, 1.30], plus 2 SE of a 3,200-game reading: **[9.37, 10.01]**. Below, the path loses a signal the recipe transfers; above, it overshoots.

## 3. Regimen

| Stage | Cost [D] | Readout | Decision per outcome |
|---|---|---|---|
| **S0** | $0.80 | **(a)** `own1`'s 800 recorded games replayed twice at generation numerics ($0.04): band census by type and runner-up; flips by gap, against the record and replay against replay; E[R²] with the value head as c_i; d_eff(λ). **(b)** Value-head corrector r_L ($0.05). **(c)** 2,000 coin games timed on the rented host ($0.05): I1, I6, I7. **(d)** Two 12,800-game pool matches ($0.64): the I9 null; `own1` at attack offset +1 against `own1`, predicted **−62 ± 15 Elo**. | (a)–(c) feed the power script, rerun before S1. It states what a null would bound; whether that is worth S1 is your call. (c): **three rounds fit iff price × (1 + logging overhead) ≤ $2.97e-5 a game; two fit iff ≤ $4.58e-5.** On a slow host, re-rank hosts. (d): a divergence outside the numerics band, or the attack match outside ±30 Elo (2 combined SE), means the match path is broken; fix it before S1. |
| **S1** | $20.08 + $0.64 | **Batch 1:** plants, control-variate invariance, coin cost by margin bin, sign check. **After batch 4:** student 4 against `own1` over 25,600 pool games (J4), and students 1–3's forecast. **After batch 8:** v̂, c, b, q̂, Ĝ_pf. | **Plant or sign check failed:** the label path is broken; debug. **J4 < 0 while students 1–3's forecast is above 0 by 2 SE:** the labels overstate play; run student 4's mixture ($0.64) before buying more. **J4 < 0 otherwise: stop** (P 0.50 at a null; 0.03 at sd 0.0015, 0.00 at sd ≥ 0.002, μ 0 [D]; the design's rule: 0.15). **Students 1–3 switch nothing and their upper bound is under 2 match SEs:** stop. **Otherwise** continue to batch 8, then S2. |
| **S2** | $0.73 | π_1 against `own1` over 25,600 pool games (SE 1.82); PURE 800. J8/J4, which the model puts in [1, √2] at μ 0 [D: gain ∝ v/√(v + c/N)]. | **Adopt** per section 2.4. **Miss beyond 9.9 Elo:** the mixture π_½ ($0.64) gives g, A, C and α* = −g/(2A); π_α* gets a fresh match ($0.64). **Correction mean outside ±2.2 Elo:** a fault in the correction code; the PURE reading stands. **Null:** no in-span spread above sd 0.0016–0.0034 [D]; go to S3-null. |
| **S3** | $22.09 | Coins on π_1 (p per section 2.3); π_2 against π_1 and against `own1` in the pool; PURE. | As S2. A forecast upper bound under 2 match SEs means the band is exhausted: go to S4. |
| **S3-null** | — | — | **Your choice** between two options. **Critic column:** κ_T on 200,000 coin games, costing $5.0 of games at pool price plus the design's ≈ $7 of critic forwards. **Wider band:** w = 0.7, giving 125 band decisions per side-game. |
| **S4** | $6.19 | P-distill ($1.09); π_k distilled ($2.1 [M]); ρ̂ and non-switch agreement; the distilled network against π_k at −1.5 and at offset 0, on the eval driver since two networks play ($3.00). | **P-distill outside [9.37, 10.01]:** distillation is broken. **Shortfall from −(1 − ρ̂)Ĝ_k by 2 SE:** spillover; keep π_k and widen to w = 0.7. **Offset 0 non-inferior:** the offset leaves the decode. |
| **S5** | $22.09 | As S1–S2, on the new band | As S2 |

## 4. Budget

S0 to S5 cost $72.6, plus $1.28 for the miss branch, leaving $11.1. The fixed costs (S0, three interim matches, the S2/S3/S5 matches, S4, the miss branch) are $13.67, so three rounds of 800,000 games fit iff the price is at most (85 − 13.67)/2.4 M = **$2.97e-5 a game**: 18% above the measured $2.51e-5, which the 1.41× host spread alone exceeds. S0(c)'s timing is a decision, not a formality. A null round 1 costs $11.5 (stopped at batch 4) or $21.5. The $1,000 case is argued only after two adopted rounds that landed within their forecasts, sized from measured J8/J4 ratios and gains. Nothing measured links `own1` to human ratings; nothing rests on it.

## 5. What is new, and the nearest work

Searched 2026-10-20; finding nothing does not prove absence.

1. **The midpoint reading of coin labels**: labels as partial derivatives of the game's multilinear value at the coin point, the gain as a line integral, p = q/2 cancelling exchangeable own-side curvature, and g, A, C identified from two matches. Nearest: Owen's multilinear extension, whose diagonal integral of partials gives the Shapley value ([Owen 1972](https://doi.org/10.1287/mnsc.18.5.64), read through secondary sources such as [Casajus & Hüttner](https://unpaywall.org/10.1007%2FS00186-010-0313-6)); the p-dependence of distal excursion effects ([Qian 2025](https://arxiv.org/html/2502.13500v2): "different MRTs with different p will yield different DCEE"); conservative policy iteration's mixtures ([Kakade & Langford 2002](https://mlanthology.org/icml/2002/kakade2002icml-approximately/); [CPI beyond the tabular case, 2025](https://arxiv.org/pdf/2506.07134)). I found no use of the midpoint condition to choose a randomisation rate. Safe policy iteration's bound carries a γ/(1−γ)² penalty ([Pirotta et al. 2013](https://proceedings.mlr.press/v28/pirotta13.html); the form is recalled, as the PDF was unreadable to me). At about 225 decisions per side that bound is vacuous, so no bound sets a switch cap.
2. **One-network adoption matches inside the generation pool,** in seed-shared mirror pairs with an exact null. Nearest: Fishtest's paired openings, and AIVAT ([Burch et al.](https://arxiv.org/abs/1612.06915)). The exact null is particular to deterministic self-play.
3. **Antisymmetric typed intercepts with margin slopes,** derived from continuity at ties. Nearest: tie-breaker designs ([Owen & Varian](https://arxiv.org/abs/1808.07563)).

## 6. Weakest point

**The spread v is still unmeasured, and the records cannot bound it**: round 1 can spend $11.5–21.5 and return only an upper bound. Behind it: the midpoint argument assumes exchangeability, while switches chosen by one head may all push play the same way (more aggression, say) and compound; and the opponent's coins bias the labels by p·1_opp′H1_S, which only one-sided coin games would remove, at half the labels per game. A miss would point at these, and the mixture match separates them only for effects of about 15 Elo or more (SE 7.5–8.8).

## 7. Rulings and decision record

**Rulings for you:**
- **(a)** Are coin-label regressions allowed?
- **(b)** May the seven luck columns serve as control variates?
- **(c)** Adoption by corrected matches at 1.82 Elo resolution, **run in the generation pool at generation numerics**, with strength claims staying PURE on the eval driver.
- **(d)** The offset stays in π until S4 shows non-inferiority.

**Rejected:** the constant intercept per ordered type (breaks I3, misspecified at ties); G1 from the labels in α* (biased by A); the model-bound Ĝ_8 as comparator; the progressive-forecast stop as sole rule (P 0.15 at a null); eval-driver adoption matches (4.7 times dearer, other numerics); Match C as the pool's known answer (its 12 Elo SE cannot see a sign error).

## Provenance table

| Number | Value | Used for | Tag | Source or derivation |
|---|---|---|---|---|
| Elo per unit z | 347.4 | all SEs | D | 800/ln 10 at an even match |
| Var(z), `own1` self-play | 1.0004 | σ² | M | `round5_simplify_variance.json` |
| Additive setup + columns kept | 0.732 (own1 self-play); 0.747 pooled | design's σ²; comparison | M | `round5_provenance_match_variance.out` |
| Full-cell baseline kept | 0.688 (pooled cell share); 0.651 (0.58–0.72) from self-play cells | σ² | M | `round6_provenance_cells.out` |
| Cell baseline's finite-sample term | 0.0015 (700,000 games) to 0.0105 (100,000) | batch 1's cross-fit | D | within × 1,512 cells / games |
| Label noise σ² | 3.67 at p 0.25 | every size | D | 1.0004 × 0.688 / 0.1875 |
| Pairs + columns kept | 0.700 ± 0.009 | match SE | M | `round5_provenance_critique_pairs.out` |
| Mean preservation | Σt² 17.7 on 23 df | I8 | M | `round5_provenance_match_variance.out` |
| Match SE | 2.57 / 1.82 / 1.28 Elo at 12,800 / 25,600 / 51,200 | adoption, interim | D | 347.4 √(0.700/n) |
| Correction term SE | 1.56 at 12,800 games; 1.10 at 25,600 | adoption check | M / D | `round5_simplify_critique_extra.out` §3 (rerun); × √½ |
| Generation price | $2.51e-5 (ref path), $2.45e-5 (priors-only) | every cost | M | `round6_provenance_prices.out` (generation_speed_r5 speed.json) |
| Host spread | 1.41× (1,612–1,625 vs 2,285 decisions/s, same code) | three-round line | M | BACKLOG.md, generation rounds 4–5 |
| Eval match price | $1.17e-4 per decisive game median (9.5e-5–1.59e-4, 18 matches) | PURE, S4 matches | M | `round6_provenance_prices.out` (walls.txt) |
| Pool against eval, per game | 4.7× | match placement | D | 1.17e-4 / 2.51e-5 |
| Box setup | median 156 s after entry, $0.02 | S0 | M | `round6_provenance_prices.out` |
| Forward price | $5.1e-8 (`own1`); $3.6e-8 (relset, no memory) | S0(b) | D | $0.42/h ÷ 2,285 decisions/s (ref path, generation_speed_r5); ÷ 3,200 leaves/s (docs/box_specs.md) |
| Band share | 0.339 at w 0.35, 0.558 at 0.7; mean G in band 0.161 | band count | M (`parity3`) | `round5_simplify_band.json` |
| Band composition | moves 65.8, recruits 10.3, attacks 9.6, end 5.7 of 91.4 | intercept design | M (`parity3`) | same |
| `own1` decisions per side-game | 224.6 (raw); 19.4 side-turns | band count | M | `round5_simplify_critique_decision_units.out` |
| Band per side-game | 76.1; 125 at w 0.7 | label volume | D | 0.339 × 224.6; 0.558 × 224.6; S0(a) re-measures |
| Decisions, fights, support per game | 449.1; 109.7; 11.86 | S0 costs | M | `round5_synthesis_outcomes.out` |
| S0(a) cost | $0.04 | S0 | D | 718,400 decisions at 2,285/s × $0.42/h |
| S0(b) cost | $0.05 | S0 | D | 109.7 × 11.86 × 800 forwards × $5.1e-8 |
| Numerics flips | 1.76% at G < 0.0136; rates 0.53 / 0.12 / 0.01 per reading in [0, 0.0136) / [0.0136, 0.1) / [0.1, 0.35) | logged propensities, I3, I9 | M (`parity3`) | `round5_provenance_numerics_ties.out`; `round5_simplify_critique_extra.out` §1 |
| Attack promotion | −0.179 ± 0.043 z per nat; ≥ 2.35 changes per side-game per nat; ≤ 0.076 ± 0.018 per change | sign check; intercepts; S0(d) | M | `round6_provenance_attack_pairs.out` |
| Attack demotion | +0.005 ± 0.019 per removed attack | context | M | same |
| S0(d) attack-match prediction | −62 ± 15 Elo at offset +1 | pool known answer | D | 347.4 × (−0.179 ± 0.043) |
| Value of ending earlier / later | +0.0033 ± 0.0017 / +0.0002 ± 0.0010 (fragile: −0.0023 ± 0.0015 at stage 1) | end pairs are ordinary pairs | M | `round5_provenance_knob_switches.out`, `round5_provenance_critique_knobs.out` |
| Match C | −0.035 ± 0.035 z | rejected as known answer | M | `round5_provenance_stop_acts.out` |
| Tempo closure of the recipe | 1.06 (0.88–1.30) / 1.00 (0.82–1.23) | P-distill band | M | `round6_provenance_closure.out` |
| Agreement closure | 0.672 → 0.862 (0.58) | ρ̂ context | M | docs/memory_own_trajectories_prereg_20261011.md |
| P-distill teacher / base | 9.625 ± 0.076 / 8.580 ± 0.065 | P-distill | M | `round6_provenance_closure.out` |
| P-distill band | [9.37, 10.01] | S4 kill line | D | 8.580 + [0.82, 1.30] × 1.045 ± 2 × 0.035 (0.065–0.076 × √(800/3,200)) |
| Recipe cost; targets rate; trainer rate | $2.1; 390 positions/s; 1,477 positions/s | S4 | M | prereg above; BACKLOG.md (configuration B calibrated) |
| P-distill cost | $1.09 | S4 | D | 3,100 games × c_g + 1.4 M/390 s + 2.8 M/1,477 s at $0.42/h + 3,200 × c_e |
| Floors at $20.08 | match-only sd 0.0016 (μ 0, d 776), 0.0034 (μ −0.0018); 0.0012 / 0.0027 at d 200 | what a null means | D | `round6_provenance_power.out` §1 (stated Gaussian model) |
| Adoptable gain | 5.2 Elo (80% power, 25,600 games) | floors | D | 2.84 × 1.82 |
| Interim stop probabilities | 0.50 (null), 0.20 / 0.03 / 0.00 at sd 0.001 / 0.0015 / 0.002 (μ 0); 0.30 at μ −0.0018, sd 0.003 | S1 rule's meaning | D | `round6_provenance_power.out` §2 |
| Forecast SE | 4.6 Elo | comparator | D | 4.7 (`round5_synthesis_power.out`) × √(0.688/0.732) |
| Miss line | 9.9 Elo | S2 | D | 2√(4.6² + 1.82²) |
| Identification SEs | g 7.5, A 8.1, C 8.8 Elo | miss branch | D | `round6_provenance_power.out` §3 |
| Switch share q | 0.5 at μ 0 (38.1 of 76.1) | p = q/2 | D (model); M in S1 | `round5_synthesis_power.out` §2 |
| J8/J4 bound | [1, √2] at μ 0 | round-2 sizing check | D | g ∝ v/√(v + c/N) |
| Plant bands | 1 ± 0.069; 1 ± 0.097 | P-coin | D | √(3.67/(15.2 M × 0.5))/0.02 × 2; same for δ2 |
| Sign-check SE | 0.0061 at 0.5 pairs per side-game | batch 1 | D | √(3.67/(2 × 100,000 × 0.5)); census from S0(a) |
| False adoption | 0.023 a match | adoption | D | one-sided 2 SE |
| Stage costs | S0 0.80; S1 20.72; S2 0.73; S3 22.09; S4 6.19; S5 22.09; miss 1.28 | budget | D | prices above × game counts in section 3 |
| Three-round / two-round price line | $2.97e-5 / $4.58e-5 per coin game | S0(c) decision | D | (85 − 13.67)/2.4 M; (85 − 11.65)/1.6 M |
| Winner's curse | +35.7 → +1.7 | fresh seeds for adoption | M | docs/decode_knobs_prereg_20261012.md |
| Batch-invariant kernels | 1.6–2.1× wall (vLLM, one GPU) | nothing (propensities logged; adoption at generation numerics) | literature | [Thinking Machines 2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) |
| w 0.35, batch 100,000, 8 batches, interim at 4, δ1 0.02, δ2 0.01, 25,600-game matches | — | design | C | consequences derived above |
| p | 0.25 (round 1); q̂/2 after a curvature miss | label point | D (conditional on exchangeability) | section 2.3; noise ratio 0.1875/(p(1 − p)) |
| v; intercepts c and slopes b by type | — | the round's gain | U | S1's progressive fits; nothing before S1 relies on them |
| `own1` band census, runner-up mix, d_eff, E[R²] with the value head, r_L | — | power inputs | U | S0(a), S0(b); the power script reruns before S1 |
| Logging overhead; host price | — | three-round line | U | S0(c), on the rented host |
| Generation-numerics band (replay against replay) | — | I9's tolerance | U | S0(a) |
| Opponent-coin term, non-exchangeable curvature | — | what a miss means | U | interim match, S2, mixture match |
| ρ̂, non-switch agreement, spillover | — | S4 | U | held-out reading before S4's match |
| Two-network pool rate | — | nothing | U | S4 uses the eval price |
| κ_T | — | critic column | U | only after S3-null; $5.0 of coin games at pool price plus about $7 of critic forwards (`current_design.md` §2) |
| Compounding, exploitability, human level | — | the $1,000 case only | U | each round's match; nothing else rests on them |
| Budget; hourly price | $85; $0.42 | everything | given | seed §1 |
