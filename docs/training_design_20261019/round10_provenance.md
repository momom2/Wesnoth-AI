# Round 10, provenance stance: L's prediction decomposed on batch 1's records, and two confounded measurements repaired

Tags: **[M]** measured (record and script named), **[D]** derived (inputs shown or script named), **[U]** unknown (where it gets measured, or why nothing rests on it), **[C]** a choice whose consequences are derived. New scripts: `analysis/round10_provenance_r_link.py` (*link*), `round10_provenance_r_pdis.py` (*pdis*), `round10_provenance_derived.py` (*derived*), each with its `.out`; "§n" is a section of that output.

## Summary

1. The core is the current design's. Six things it rests on were unmeasured, confounded, stale, untraced, or decided on a premise the measured prices overturned. Each is replaced by a measurement from batch 1's records, a derivation, or a restructuring.
2. **L's sealed prediction F, decomposed before L runs.**
   - Each side's first two R switches show no interaction: their exact all-order value differs from first order by −0.3 ± 1.7 Elo (*pdis* §1).
   - Over all switches, R's own interactions read −3.5 ± 4.2 Elo (*link* §1).
   - 0.71 of F sits in turn-1 recruits, whose states carry 0.6 prior coin deviations (*link* §2).

   An L null therefore needs the turn-1 values themselves to fall. A null that passes I-R would name the coined continuation, which the records cannot read (−10.8 ± 19.1 Elo, *link* §2b) and L does.
3. **The smoke's overhead reading is confounded.** Batch 1's walls carry a fixed 69 ± 30 s a match. Comparing 2,000 dumped with 1,000 undumped games reads o = −0.154 at a true 0 and +0.233 at 0.56, so the gate "o > 0.56" cannot fire (*derived* §1). Equal-size runs cancel the fixed cost. The same confound makes round 3's 2,000-game host test reject hosts whose true cost lies between 0.798 and 0.934 of the measured one (*derived* §2).
4. **Stale and untraced numbers.**
   - Branch S's B1 was priced at $0.42 an hour; it costs $3.56 at the measured host, not $2.59 (*derived* §3).
   - The "~$7 critic column" has no derivation.
   - Round 8 rejected a futility stop because "the budget does not bind". It now binds, so the trade goes to the user.
5. **R's learning curve is flat past about 17,000 training games** (+17.9 Elo, against +19.7 at 51,000; *link* §3). The recruit readout's refit of R on π_0's labels is therefore saturated within batch 1, and needs no features.

## 1. Readings and the decisions they move

| Reading | Value | Decision moved |
|---|---|---|
| S0(b), seed §7 | `own2` +22.7 ± 4.4 over `own1` | π_0 = `own2` |
| S0(a), seed §7 | −41.3 ± 4.4 against −76 ± 17 predicted from decode knobs | Windows from other networks are reports, not gates |
| Batch 1's typed means | none past the family-wise line | No stage keyed on a type's mean |
| Batch 1's prices | $2.849e-5 a coin game all-in | Round 3 does not fit at o = 0 on this host |
| Batch 1's walls (new) | fixed 69 ± 30 s a match; consecutive 10,000-game matches sd 25.9 s | o read on equal-size runs; the host test runs 10,000 games |
| R's own interactions (new) | first two switches exact −0.3 ± 1.7 Elo (1.31 of 2.16 switches, +15.1 of +18.0 Elo); first-coin correction −3.5 ± 4.2; later switches after a played one, within rank, −0.032 ± 0.012 each | L's miss attribution |
| R by turn (new) | turn 1 +12.7 ± 1.5 of +18.0, at 0.6 prior deviations; −0.00046 ± 0.00064 per prior deviation | An L null names the continuation |
| Continuation (new) | Taylor step to an uncoined continuation −10.8 ± 19.1 | None: L is the measurement |
| Fold schemes (new) | 20-fold +21.3, leave-one-match-out +18.0, difference +3.3 ± 1.2 | F stays +21.3: fold rules match the all-data rule L plays on 250–259 of 262 cells, match rules on 238–260 |
| Learning curve (new) | +16.3 at 8,600 games, +17.9 at 17,000, +19.7 at 51,000 | The recruit readout's R, refitted on π_0's labels, is saturated within batch 1 |

A correction to my own first reading. Pooled over ranks, the "earlier switch played" contrast read −0.033 ± 0.011. That pooling mixes ranks, since P(an earlier one was played) grows with rank. The within-rank figures above are the randomized ones.

## 2. The algorithm

**Labels.** At a decision whose top two joint log-priors differ by G < 0.35 [C], a logged coin plays the runner-up a1 with P = 0.25 [C]. Play continues at argmax and re-decides after every roll.

    T_i = s_i ω_i R_g,   ω_i = (D_i − P_i)/(P_i(1 − P_i)),   R_g = z − b(cell) − βᵀS_g   (b, β frozen)

- **Exact.** E[T_i | H_i] = Q̃(H_i, a1) − Q̃(H_i, a0) under the coin policy. The coin is independent of H_i with P logged, and each luck term has mean zero given its pre-fight state.
- **Noise.** E[T²] is 4.01 and the design effect within a game is 1.00 [M]. About 155 labels share each result.
- **What it avoids.** No critic, no tree, and single decisions only.

**Learned:** a linear head on the frozen token outputs, f = c_uv + b_uv·G + ŵᵀx with x = φ(a1) − φ(a0).
- **Features.** d is 772, plus 43 unit columns iff ruling (h) allows them.
- **Update.** ŵ = (Σ_j G_j + λΛ)⁻¹ Σ_j b_j, with λ by the progressive loss. This is one episodic natural-gradient step for a logistic coin policy ([Peters & Schaal 2008](https://doi.org/10.1016/j.neucom.2007.11.026)).
- **Switch rule.** The player switches where f > τ̃, which is tested on held-out welfare ([Kitagawa & Tetenov 2018](https://doi.org/10.3982/ECTA13288)).
- **The network still picks every action;** the head only reorders its top two at a near-tie.

**Signal per dollar** [M].
- **Information.** Coins carry 38.7 units of single-decision information per game-equivalent, against 0.82 for a fork pair (`round9_simplify_budget.out`).
- **R itself.** R came from $1.75 of games, and 8,600 games of labels already give +16.3 Elo (*link* §3).
- **Exploration** costs +2.0 ± 4.7 Elo first order. Nothing rests on its all-order cost.
- **Critics**, the alternative evaluator, failed on record.

**Reading a sparse rule's own interactions exactly (new, used at R0).**
- **The identity.** Take the coin policy π_c as the old policy of the performance-difference lemma ([Kakade & Langford 2002](https://dl.acm.org/doi/10.5555/645531.656005)). Let π'_R and π'_0 play one side's rule decisions a1, or a0, for sure. Since A^π_c(s, a1) = 0.75Δ and A^π_c(s, a0) = −0.25Δ,

      J(π'_R) − J(π'_0) = 0.75 E_π'_R[Σ_j Δ_j] + 0.25 E_π'_0[Σ_j Δ_j].

- **The estimator.** Per-decision importance weights over the side's earlier rule decisions make both terms exact from coin games ([Precup, Sutton & Singh 2000](https://mlanthology.org/icml/2000/precup2000icml-eligibility/)). The weight is 4^k if all k earlier ones were switched and (4/3)^k if none was. With no interaction the estimate equals F.
- **Its reach.** The variance grows as 4^k: the estimate reads R's first two switches at SE 1.7 Elo, and all of them only at ±40.5.
- **Past that reach,** two randomized contrasts bound the rest: the side's first coin against its later labels, and later labels by whether an earlier switch was played, within rank.
- **The continuation's sensitivity** is an exact derivative, dΔ_j/dp = E[T_j Σ_{k>j}(D_k − p)/(p(1 − p))]: each later coin's score function, carried through the label identity.
- **For the head,** at 19–38 switches per side-game, the weights explode, so its link is read in play (H1, S2).

**Plants, one per training code path** (unchanged):
- **Label path:** the coin plant on every match, and the dice control for the join.
- **Head fit:** the merged feature plant at 10 SE, and the R plant offline.
- **Switch path:** I-R, I-match and I-side.
- **Dump:** I-φ and I-proc.
- **Distillation:** S4's closure ĉ, on held-out positions.

## 3. Regimen

Costs at $0.577 an hour (`round9_synthesis_checks.out` §6 unless marked).

| Stage | Cost | Readout and prediction | Decision |
|---|---|---|---|
| **Rulings** | $0 | (a), (e), (h), the $5 window, (i) | L needs (e); T's reading needs (h) |
| **R0** | $0 | R frozen at ridge 5,000 and hashed; F +21.3 ± 3.1 sealed with §1's decomposition; R plant; unit tests | Red: fix first |
| **Box A: L** | $0.98 | `own1`+R on one seat against `own1`, 37,800 games, SE 1.79 | Table below |
| **Box A: T** | $0.98 | `own2`+R against `own2`; sealed J_T = J_L, SE 2.53 | As now |
| **C** | $1.13 | iff π_0 = `own2`+R, line 11.2 | As now |
| **Smoke** | $0.06 + setup | 2,000 dumped coin games on `own2`: every gate; census, E[T²], R² of R's values on x | A gate fails: debug |
| **Batch 1** | $2.85(1 + 0.9o) | Match 1 undumped and match 2 dumped, 10,000 games each on one box: o = (W₂ − W₁)/(W₁ − a), SE 0.024 (*derived* §1). A dump timer in the server's counters cross-checks it; the server binds, with the GPU busy 0.77 of the wall. Match 1's labels feed the typed columns, plants and R's refit | **o > 0.56:** one round fits; the user is told before match 3 |
| **H1** | $0.98 | Student 1 against π_0, read against T (SE 2.53) | As now, plus (i) |
| **Batches 2–8** | $19.94(1 + o) | Plants, dice control, v̂ per type, τ̃, sealed Ĝ_pf | A plant fails: quarantine |
| **S2–S5** | as now | Adoption at 37,800 games; per-round claims, lines 9.4 and 11.2 | As now |
| **Host test** | ≤ $0.27 + setup [D: 10,000 × $2.66e-5] | One 10,000-game coin match on the candidate, the reference price's own size | ≤ $2.66e-5 a coin game ($2.77e-5 if the window did not count): round 3 runs |

**L's table:** F = +21.3 ± 3.1, miss half-width 7.2, null line J ≤ 3.57.

| L reads | Meaning, from §1 | Decision |
|---|---|---|
| Within F ± 7.2 | First-order labels price uncoined play at R's step | S1 proposed; own1+R's PURE number |
| Below, lower bound > 0 | Partial transfer. Own interactions can supply up to 11.9 Elo at 2 SE, and turns 2 onward hold +5.3 of F | S1 proposed with E; the shortfall reported with this split |
| Null | It needs 17.7 Elo of shortfall. Own interactions would have to sit 2.7 SE beyond their reading. Losing every later turn's value still leaves the turn-1 part 3.9 SE above the line (*derived* §5). So the turn-1 values must fall, though their states carry 0.6 prior deviations and 0.959 of their switches are a side's first or second, where no interaction is read (−0.3 ± 1.7). With I-R clean: coined continuations do not price uncoined ones | No round at this coin density. Options: Branch S, the reset, Branch L, or one coined side per game (2.0× games per label) or p 0.1 (2.08×) [D] |
| Above F + 7.2 | Reported | As the first row |

**Budget** (unchanged by this stance, `round9_synthesis_checks.out` §6):
- Box A, C, two rounds, their claims and E reserves cost $56.57 at o = 0 of the $82.34 left.
- Two rounds fit iff o ≤ 0.56.
- The whole path is $4.53 short.
- A null round 1 costs $27.22.
- Branch S costs $5.67 (*derived* §3).
- The dump is processed and deleted per match, at 2.40 GB at a time rather than 24 GB (*derived* §4): every processor statistic is per match and sequential.

**Toward human level:** coin rounds, each certified by a PURE match and a per-round claim; the reset, which folds the switches into the network so the band moves; round 3; and the $1,000 case, sized from measured v̂, the link readings and ρ̂. Nothing measured links `own1` to human ratings, and nothing rests on it.

**New ruling (i).**
- **Proposal.** S1 stops after batch 1 when H1 reads the head below T by 2 SE while its recruit readout is in line.
- **Saving.** $19.94(1 + o) at a null.
- **Cost.** Its power cost cannot be priced, since the value beyond R is unknown.
- **Default.** No stop.

The other rulings are as in the current design.

## 4. Provenance

| Number | Value | Tag | Source or derivation | Used for |
|---|---|---|---|---|
| Base | `own2` +22.7 ± 4.4 Elo, 6,400 games | M | `own2_base_20261020/reading.json` | π_0 |
| Label path | coin plant 0.994, 0.995 ± 0.008; dice slope 0.00436; I-coin, I-log pass | M | `near_tie_coins_20261020/readout_b_offline.txt` | Gates |
| E[T²]; residual share; design effect | 4.01; 0.699; 1.00 | M | `round9_simplify_labels.out`, `round9_simplify_residual.out` | Every SE |
| Census | 77.6 band decisions per side-game | M | `readout_b_offline.txt` | Storage; `own2`'s is [U], read at the smoke |
| Recruit spread | 4.29 different-unit pairs per side-game; V_rr 1.59e-3 ± 3.7e-4; +11.2 SD dispersion | M | `round9_provenance_recruit_cells.out` | R exists |
| F, 20-fold | +21.3 ± 3.1 Elo; 2.19 switches per side-game; ridge 5,000 | M | `round9_synthesis_checks.out` §1 | L's prediction |
| F, leave-one-match-out | +18.0 ± 3.1 (clustered by game); 2.16 switches | M | *link* §1 | Fold-scheme check |
| Fold-scheme difference | +3.3 ± 1.2 Elo; 0.335 disagreeing decisions per side-game | M | *pdis* §2 | Reported; F kept |
| All-data rule's agreement | 250–259 of 262 cells (fold rules), 238–260 (match rules); 2.21 switches | M | *pdis* §2 | Which F seals |
| First two switches, exact | J − F −0.3 ± 1.7 Elo (1.31 switches, +15.1 Elo first order) | M | *pdis* §1 | L's table |
| First three, exact | J − F +4.0 ± 3.8; all ranks ±40.5 | M | *pdis* §1 | Reach of the exact reading |
| Rank-1 contrast | −0.0015 ± 0.0197 per switch | M | *pdis* §1, *link* §1 | Own interactions |
| Within-rank later-switch contrast | −0.032 ± 0.012 (ranks 2, 3, 4+: −0.052, −0.051, −0.039) | M | *link* §1 | Own interactions |
| First-coin correction | −3.5 ± 4.2 Elo (D1 → later labels −0.0174 ± 0.0207 per side-game; later R decisions −0.138 ± 0.015) | M | *link* §1 | L's table |
| F by turn (leave-one-match-out rules) | turn 1 +12.7 ± 1.5 (0.49 switches, 0.6 prior deviations; ranks 0, 1, 2+: 0.767, 0.192, 0.042); turns 2+ +5.3 | M | *link* §2 | L's table |
| Adjacent-switch correction's SE | 5.7–8.2 Elo over four batches | D | *derived* §6 | Rejected |
| L sealed with the correction | miss half-width 11.0 | D | *derived* §7 | Rejected |
| Dose-response | −0.00046 ± 0.00064 per prior deviation; linear extrapolation +5.4 ± 7.4 Elo | M; extrapolation D under a linear model | *link* §2 | Report only |
| Continuation derivative | Taylor step −10.8 ± 19.1 (all later coins), −23.2 ± 13.5 (own), +12.4 ± 13.7 (opponent), −8.4 ± 7.3 (4 side-turns) | M; step D (Taylor) | *link* §2b | Nothing: L reads it |
| Learning curve | +16.3, 17.9, 18.8, 19.2, 18.8, 19.7 Elo at 8.6k–51k training games (subset means) | M | *link* §3 | The recruit readout's refit of R |
| L null's requirement | shortfall 17.7; own interactions 3.4 SE beyond their reading (2.7 with F's error); turn-1 part 3.9 SE above the line | D | *derived* §5 | L's table |
| L's lines | SE 1.79; miss half-width 7.2; null line 3.57; ratio bound 0.34 | D | 347.4/√37,800; `round9_synthesis_checks.out` §2 | Box A |
| T, H1 | difference SE 2.53; false fire 0.023 | D | `round9_synthesis_checks.out` §3 | T, H1 |
| Wall model | a 69 ± 30 s a match; b 155.0 ± 3.3 ms a game; residual 24 s | M | *link* §4 from `walls.txt`, `matches.txt` | o, host test |
| Consecutive-match spread | sd 25.9 s on 1,621 s (b01–b05) | M | *derived* §1 | o's SE |
| Smoke's naive o | −0.154 at o 0; +0.233 at o 0.56 | D | *derived* §1 | Rejected protocol |
| o by equal-size runs | SE 0.109 at 2,000 games; 0.024 at 10,000 | D | *derived* §1 | Batch 1's o decision |
| Server load | GPU busy 1,282 s of 1,658 s (0.77) | M | `b01.inference_server_0.json` | o read from the server's counters |
| Host-test bias | a 2,000-game timing reads 1.170× the reference's per-game cost; hosts between 0.798 and 0.934 rejected wrongly | D | *derived* §2 | Round-3 host test |
| Prices | coin game all-in $2.849e-5; pool $2.598e-5; two-network $8.848e-5 | M | `round9_simplify_labels.out` | Every cost |
| Path | $56.57 through round 2; two rounds iff o ≤ 0.56; path $4.53 short; line $2.66e-5 | D | `round9_synthesis_checks.out` §6 | Budget |
| Branch S | B1 $3.56; total $5.67; exits $3.56, $4.54 | D | *derived* §3 from `round8_synthesis_power.out` | Budget |
| S4 | $3.90 (4.8 measured box-hours, plus a two-network match) | D | prereg 20261011 "Measured"; `round9_synthesis_checks.out` §6 | S4 |
| Dump | 2.40 GB per match (2.53 with unit columns) | D | *derived* §4 | Storage |
| Signal ratio | 38.7 against 0.82 per game-equivalent | D from M | `round9_simplify_budget.out` | §2 |
| Exploration cost | +2.0 ± 4.7 Elo, first order | D | `round9_simplify_budget.out` | Nothing beyond first order |
| Forecast | SE 4.88 at q 0.5 (3.41 at 0.25); miss line 10.4 | D | `round9_simplify_labels.out`; recomputed from logs | S2 |
| Claims | line 9.4 (11.2 against S0(b)); false miss 0.042–0.053 | D | `round9_synthesis_checks.out` §5 | Per-round checks |
| Lower-density options | one coined side 2.0×; p 0.1 2.08× games per label | D | labels per game halved; 0.1875/0.09 | Options after an L null |
| Critic column "~$7" | — | U | no derivation found | Removed from the menu until priced |
| Continuation shift at R's step | — | U | L | Box A's decision |
| `own2`'s census, E[T²], o, R² | — | U | Smoke and batch 1 | Rounds that fit |
| Value beyond R; q; τ̃; Ĝ_pf | — | U | H1, S1 | The round's gain; nothing before S1 |
| Link at the head's step | — | U | H1, S2 | Adoption; E |
| ĉ, ρ̂, J_S4; Branch S's readings | — | U | S4; B1–B3 | Round 3; base |
| Human level, exploitability | — | U | none | Nothing |
| w 0.35, p 0.25, 8 × 100,000 games, four-pair τ̃, 2-SE pre-test, 3-SE plant lines, 37,800 and 12,800 games, 1,000-game subsample | — | C | consequences derived in rounds 5–9 | Design |

## 5. What is new, and the nearest work

Searched 2026-10-21; finding nothing does not prove absence.
1. **The core assembly** is as in the current design §5: near-tie coins at an argmax decode, exact through the simulator's chance laws, pooled by a closed-form natural-gradient head, certified by match. The nearest designs randomise near a cutoff: tie-breaker designs ([Owen & Varian 2020](https://arxiv.org/abs/1808.07563)) and local-randomisation regression discontinuity ([Cattaneo, Frandsen & Titiunik 2015](https://www.doi.org/10.1515/JCI-2013-0010)).
2. **New this round: a sealed first-order forecast is split before its match**, from the randomised data alone, into own interactions (exact up to k switches), state shift, and continuation sensitivity (an exact derivative), so that the match isolates one cause.
   - The parts are known: the performance-difference lemma, per-decision importance sampling, and the score-function (likelihood-ratio) derivative, the last recalled, not re-read.
   - So is the variance trade: [Farias et al. 2022](https://arxiv.org/abs/2206.02371) report a second-order bias for Differences-in-Q against possibly exponentially larger variance for off-policy estimators. It is measured here: SE ±1.7 Elo at two switches, ±40.5 at all.
   - The use as a pre-match attribution of a forecast was not found.
3. **Selection after adoption.** [Andrews, Kitagawa & McCloskey 2024](https://nber.org/papers/w25456) (QJE 139(1); abstract only) give median-unbiased estimators and hybrid intervals for a selected winner. They are the candidate for sizing the $1,000 case from adopted rounds. Round 9 measured the truncated-normal correction as too noisy near the adoption floor; this paper is not yet tested here.
4. **The finding.** `own1`'s near-tie recruit miscalibration sits mostly in turn-1 recruits, and its first two switches per side show no interaction (−0.3 ± 1.7 Elo). This is a fact about this network, not a method claim.

## 6. Weakest point

- **The continuation.** It is the one cause of an L miss the records cannot bound: its Taylor step reads −10.8 ± 19.1 Elo. It is also the cause that would carry to the head, since every S1 label is priced under a coined continuation, while L tests it only at R's step.
- **Beyond R, nothing is certified.** S1's $22.79 of batches buys a measurement whose payoff is unknown, and a null round 1 costs $27.22. Ruling (i) is the only lever on that, and its power cost cannot be priced.
- **The fold-scheme gap is unexplained.** Two out-of-sample readings of R differ by 3.3 ± 1.2 Elo. Its size is inside L's half-width, but it shows that R's value moves with its training sample at the 3-Elo scale.

## Decision record

- **Rejected:** the smoke's o from 2,000 dumped against 1,000 undumped games, because it reads −0.154 at o = 0. Also a 2,000-game host test, because it overstates cost by 17%.
- **Rejected:** an adjacent-switch interaction correction for S1's forecast. Its SE over four batches is 5.7–8.2 Elo at 19–38 switches (*derived* §6), larger than the forecast's own 3.41–4.88, and E is the response to a miss whatever it reads.
- **Rejected:** sealing L at F plus the first-coin correction. That widens the miss half-width from 7.2 to 11.0 (*derived* §7), and L's question is the first-order link. The correction is kept as attribution.
- **Kept:** everything else in the current design.
