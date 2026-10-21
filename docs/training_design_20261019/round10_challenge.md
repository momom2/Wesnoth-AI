# Round 10 challenge: labels choose the direction, play chooses the step

Tags as in `current_design.md`: **[M]** measured, **[D]** derived, **[U]** unknown (with when it is measured), **[C]** a choice whose consequences are derived. Scripts in `analysis/`, each with its `.out`: `round10_challenge_recruit_split.py` (*split*), `round10_challenge_recruit_context.py` (*context*), `round10_challenge_recruit_context_ridge.py` (*ridge*), `round10_challenge_shift_curve.py` (*curve*), `round10_challenge_recruit_halves.py` (*halves*), `round10_challenge_budget.py` (*budget* §n).

## Summary

1. **Core.** The current design asks its coin labels for a direction (the head) and also for the step's size and certificate (welfare threshold, sealed forecast, E). Here labels give only the direction. The step is the length η of a logit shift along it, measured in play by a line search of cheap one-network matches and a fresh confirmation. For a fixed rule, a 12,800-game screen gives 42–87 times the information per dollar of a 100,000-game batch's forecast [D, *budget* §2].
2. **The step reaches past the band.** A shift acts on every decision of its class: for recruits, 22.89 per side-game, where band switches reach 4.29 [M, *split*].
3. **Small, fresh iterations.** A recruit direction fitted on 30,000 games keeps 0.91 of the welfare of one fitted on 54,000 [M, *halves*]. An iteration costs $3.63–$7.62 against about $26 for a round [D, *budget* §6], and its labels always price the current policy.
4. **New measurements that move decisions.** The recruit structure is matchup-dependent: keys on (faction, unit, enemy faction) collect +28.5 ± 3.1 Elo first order against R's +21.3, a paired +7.2 ± 1.7, ridge chosen by the progressive loss [M, *ridge*]. Turn 1 holds 0.59 of R [M, *context*]. R is not an artifact of the factored decode [M, *split*].
5. **Weakest point.** No direction outside recruits is on record; if the rest is diffuse, iterations grow toward the current design's batch size and only the cheaper evaluator remains.

## 0. Why change the core

- **Section 7 moved:** the base (`own2`, +22.7 ± 4.4); the labels' validity (plants and invariants over 9.3 M decisions); the prices; and one play reading of real size, S0(a)'s hook at −41.3 ± 4.4 Elo, which becomes the shift path's known answer (§2).
- **Only recruits carry structure, and a small batch finds it.** Round 9 found nothing else at 60,000 games (nine partitions; unit cells +0.90 SD) [M, `round9_provenance_cells.out`, `round9_provenance_unit_cells.out`]. Rule B, fitted on 6,000 / 12,000 / 30,000 / 54,000 games, reads +17.7 / +20.6 / +24.7 / +27.2 Elo out of sample: 0.65 / 0.76 / 0.91 / 1.00 of the largest [M, *halves*]. The current design buys 800,000 games per adoption ($22.79 of a $26.18 round [D, `round9_synthesis_checks.out` §6]); that pays only for a diffuse structure the record has never shown.
- **The structure is counter-recruiting, and strategic.** B − A +7.2 ± 1.7 Elo paired over 20 folds; progressive loss −579.9 for B against −273.2 for A, ridge 5,000 for both; A reproduces round 9's R exactly (+21.3 ± 3.1, loss −273.19) [M, *ridge*]. Turn-1 pairs (0.87 per side-game) carry +12.5 ± 1.5 of R; phase keys add −1.3 ± 1.3 [M, *context*]. Under the network's own continuation the values penalise scouts and fodder against line units (Vampire Bat −0.061, Goblin Spearman −0.038, Wolf Rider −0.031 per decision; Troll Whelp +0.037, Orcish Grunt +0.036) [M, *split* §2].
- **Not a decode artifact.** The joint argmax splits a unit's mass over castle hexes, the mechanism behind end_turn's +193, so R could have been its by-product. It is not: the most often split units are the scouts (split rate 0.17–0.19), which R disfavours; the within-faction correlation of v with log split rate is −0.20 (permutation p 0.21); a split-only rule reads +1.3 ± 2.9 [M, *split* §2–5]. A marginal recruit decode would move the wrong way.
- **The step is what is poorly measured.** The current design spends L, T, H1, S2 and E on one question: do first-order labels price play at a given step. A shift's in-band first-order gain is exact from labels at any η [M, *curve*]; past the band nothing is labelled (gaps over 0.35 are not logged). Play reads both at once.

## 1. The algorithm

### 1.1 What is learned

- **A direction per decision class c**, f_c(h, a) = ŵ_cᵀx_c(a), in closed form on the label T_i = s_i ω_i R_g (as built): ŵ_c = argmin Σ_i (T_i − ŵᵀΔx_i)² + λ‖ŵ − ŵ_parent‖², keys hierarchical ((faction, unit, enemy faction) shrunk to (faction, unit), shrunk to 0), λ by the progressive loss. With P(1 − P) constant this is one episodic natural-actor-critic step for the logistic coin policy ([Peters & Schaal 2008](https://doi.org/10.1016/j.neucom.2007.11.026)): an unbiased natural-gradient direction, under ruling (a) as now. The enemy faction is in the observation, and the harness assigns factions as a lobby shows them ([observation parity](../observation_parity_20260926.md) item 1); against a Random pick the key falls back to (faction, unit).
- **A step length η_c per class**, measured in play.
- **The policy.** The network decides the action kind; when its argmax is in class c, the choice among class-c candidates is re-made by argmax of log prior + η_c f_c. For recruits the network decides whether to recruit and the shift picks the unit (0.983 of different-unit pairs share the hex [M, *split*]). The network picks every action and re-decides after every roll, traits included.

### 1.2 The step and its measurement

- J(η): the result of π_{k,η} on one seat against π_k, one-network, raw z (ruling (b)), seat drawn by a per-game coin (I-side). By the performance-difference lemma, to first order J(η) sums the advantages under π_k of the flipped decisions:

      J(η) = ρ·W_in(η) + B(η) + second order

  W_in(η) is the exact cross-fitted gain of the flips inside the band [M, *curve*]; ρ is the label-to-uncoined-play ratio [U, stage 1 on `own1`]; B(η) is the effect of flips past the band [U, the screens].
- **Line search.** Screens at a few η, 12,800 games each (SE 3.07 Elo, $0.333); the best is confirmed on fresh seeds at 37,800 (SE 1.79, $0.982) and adopted iff its lower 2-SE bound is above 0. Monte Carlo [D, *budget* §5]: false adoption 0.023 under an all-null truth (the screen winner's +2.59 bias never reaches the decision); at true arms (20, 25, 15) the best is picked 0.871 of the time, mean regret 0.68 Elo.
- **Why measured, not bounded.** An argmax policy's J is piecewise constant in η, so no smooth trust region exists. Step-size bounds of the conservative-policy-iteration family ([Pirotta et al. 2013](https://papers.nips.cc/paper_files/paper/2013/file/f64eac11f2cd8f0efa196f8ad173178e-Paper.pdf)) are reported conservative by their own authors ([2015 journal version](https://re.public.polimi.it/retrieve/e0c31c09-6408-4599-e053-1705fe0aef77/MachLearn2015-reprint.pdf)). Here a point of the true objective costs $0.33.

### 1.3 Estimators and invariants

- **Direction:** unbiased up to shrinkage for the class's linear advantage. Its noise only lowers the cosine with the true gradient, so a noisy direction reads as a flat J(η), never as a false gain.
- **Screens:** SE 347.4/√N on decisive games. J(40) − J(band-only) has SE 4.34; B on `own2` against B on `own1` (T's question) SE 3.55, line 7.1 [D, *budget* §4].
- **Claims:** as now, π_k against `own1` after each adoption ($1.13), D_k against A_k at 11.2 (first) or 9.4; false miss 0.042–0.053 [D, `round9_synthesis_checks.out` §5].
- **Invariants:** I-coin, I-log, coin plant and dice control on every batch (as built); I-side; **I-shift**: on the shifted seat every class-c decision logs G, f, the network's choice and the shift's, recomputed by separate code, every logged change played, and the flips past the band counted per η (a free census of the step's reach); **I-key**: the matchup key equals the setup's enemy faction, or "unknown" after a Random pick.
- **Why more strength per dollar:** evaluation 42–87× cheaper per unit information (consistent with round 9's 78× per game); search saturated at 30,000 games ($0.855) for the one structure on record; reach 22.89 decisions per side-game instead of 4.29; no pooling, threshold, forecast or E. The dump and head are needed only once recruits are exhausted.

## 2. Regimen

Measured host, $0.577 an hour: pool game $2.598e-5, coin game all-in $2.849e-5, two-network game $8.848e-5 [M, `round9_simplify_labels.out`]; $82.34 left [D].

| Stage | Cost [D] | Readout and sealed prediction | Decision per outcome |
|---|---|---|---|
| **R0**, offline | $0 | B frozen and hashed (257 keys, ridge 5,000); W_in(η); label plant on B's fit; CI: I-shift with a planted table, I-key | Red: fix before any box |
| **Box A″, `own1`:** B band-only, 37,800 | $0.98 | ρ·28.5: sealed +28.5, miss half-width 7.2, null line 3.57 [D, *budget* §4] | **Within:** labels price uncoined play at B's step. **Below, above 0:** partial link, read per faction; screens still decide. **Null** (ρ < 0.13–0.16): no coin batch is bought (Branch S, the reset, or stop) |
| **Box A″, `own1`:** S0(a)'s hook through the shift path (f = 1[attack], η = 1, one seat), 12,800 | $0.33 | Window [−52.0, −30.6] around −41.3 ± 4.4, measured by a separate path | **Outside:** the shift path is broken; no screen is read (a seat swap reads about +41, an unapplied shift 0) |
| **Box A″, `own2`:** screens B band-only, shift η 8, shift η 40; 12,800 each | $1.00 | Band-only against the `own1` reading, line 7.1; in-band parts +28.5, +24.3, +28.4; B(40) not predicted [U] | **J(40) − J(band-only) > +8.7:** flips past the band pay, later grids extend η. **< −8.7:** confident recruits are right; band-only form. **Within:** the winner alone is confirmed |
| **Confirmation**, fresh seeds, 37,800 | $0.98 | The winner's reading | Adopt iff lower 2-SE bound > 0 |
| **Claim** | $1.13 | D_1 against A_1, line 11.2 | Outside: intransitivity or a two-network defect, reported |
| **Phase R, iteration k** | $3.63 | 30,000 coin games on π_k; B refitted; screens at η̂ and 2η̂; confirmation; claim | Screens run only if B's progressive loss beats "no structure"; two iterations without adoption end recruits |
| **Phase H, iteration k** | $7.62 (+$0.29 smoke once) | The current design's dump and head on 100,000 games, one direction per class, two screens per class, confirmation, claim | The smoke reads o and I-φ first; two null iterations end phase H for this network |
| **Reset**, after the optimizer discussion | $3.90 | Soft target = raw logits + η f (the shift exactly, unlike exchanged pairs); held-out ĉ; N against π_k | ĉ not above 0 by 2 SE: distillation broken. Else N is the next base; phases R and H restart |

**What fits** [D, *budget* §6]: box A″ $4.43 (the current box A plus C: $3.31), leaving $77.91: 21 recruit iterations, 10 all-class ones, or 15 recruit ones with a reset every third adoption. A null box A″ costs $4.43 and a null phase-R pair $7.26, against $27.22 for the current design's null round 1. No stage relies on an unmeasured number: ρ, B(η), the transfer to `own2` and o are read before anything uses them.

**Planted checks, one per code path:** the direction fit, by the offline label plant (`tools/coin_plant.py`'s parse, z′ = z + Σ_j s_j(D_j − P_j)u(key_j), u = ±δ on a third of the matchup keys; slope 1 at 3 SE; a wrong mover sign reads −1, a mis-joined key about 0); the shift in play, by the hook window; the re-choice logic, by I-shift and its CI table; batches, by the coin plant and the dice control as passed on batch 1; screens and claims, by the match path certified in S0(a) and S0(b); the reset, by the own-trajectory planted relabel.

**Toward the strongest humans.** Each adoption is a certified natural-gradient step on the true objective; resets let the network absorb the shifts and move the band. In the $1,000 case the loop runs at $4–8 a step, with the network grown at a reset (`docs/network_size_20261014.md`). Nothing measured links `own1` to human ratings and nothing rests on it; no gain after the first step is predicted [U].

## 3. Provenance

| Number | Value | Tag | Source or derivation | Used for |
|---|---|---|---|---|
| Base | `own2` +22.7 ± 4.4 | M | seed §7, `own2_base_20261020` | π_0 |
| Coin labels valid | plants 0.994 / 0.995 ± 0.008; I-coin, I-log over 9.3 M | M | seed §7 | Directions |
| R (A) | +21.3 ± 3.1 Elo, ridge 5,000, progressive loss −273.19, 2.21 switches | M | `round9_synthesis_checks.out` §1; reproduced by *ridge* | Comparison |
| B | +28.5 ± 3.1; B − A +7.2 ± 1.7 (20 folds); loss −579.9; ridge 5,000; 2.14 switches | M | *ridge* | Stage 1's direction and sealed F |
| B at other ridges | +33.1 / +28.5 / +24.1 at 2,000 / 5,000 / 20,000 | M | *context* | Not used: the ridge is chosen by loss |
| Phase | turn-1 pairs 0.87 per side-game carry +12.5 ± 1.5 of R; phase keys −1.3 ± 1.3 | M | *context* | No phase keys |
| Census | recruits 22.89 per side-game; band decisions with a recruit on top 8.89; different-unit pairs 4.29 (0.187 of recruits); same hex 0.983 | M | *split* §1 | Reach; the re-choice form |
| Decode hypothesis | scouts' split rate 0.17–0.19; r(v, log split) −0.20, p 0.21; split-only rule +1.3 ± 2.9 | M | *split* §2–5 | Marginal recruit decode rejected |
| In-band curve | B: η 8 → +24.3 (2.4), 1.27 flips; η 40 → +28.4 (3.0); ∞ → +28.5; sd of p 0.0455 | M | *curve* | Screens' in-band parts; η grid |
| η grid | 8 and 40: shifts of 0.36 and 1.8 nats at one sd of p | D, C | 8 × 0.0455, 40 × 0.0455 | Screens |
| Learning curve | 6,000 / 12,000 / 30,000 / 54,000 games → 0.65 / 0.76 / 0.91 / 1.00 | M | *halves* | 30,000-game batches |
| Prices | pool $2.598e-5; coin all-in $2.849e-5; two-network $8.848e-5 | M | `round9_simplify_labels.out` | Every cost |
| Match SE | 3.07 (12,800), 1.79 (37,800) | D | 347.4/√N, Var z ≤ 1 | Lines |
| Screen against forecast | 0.319 against 0.0037–0.0076 per (SE²·$): 42–87× | D | *budget* §2; forecast SE 6.8–9.8 from `round9_synthesis_checks.out` §3 | §1.4 |
| Lines | L′ miss 7.2, null 3.57 (ρ < 0.13–0.16); T′ line 7.1; beyond-band SE 4.34, line 8.7 | D | *budget* §4 | Stage 1 decisions |
| Hook known answer | −41.3 ± 4.4; window [−52.0, −30.6] | M, D | seed §7 S0(a); 2√(4.4² + 3.07²) | Shift path gate |
| Line-search selection | null adoption 0.023; best picked 0.871, regret 0.68 at (20, 25, 15) | D | Monte Carlo, *budget* §5 | Adoption rule |
| Claim check | lines 11.2 / 9.4; false miss 0.042–0.053 | D | `round9_synthesis_checks.out` §5 | Claims |
| Stage costs | box A″ $4.43; phase R $3.63; phase H $7.62; reset $3.90; smoke $0.29 | D | *budget* §3, §6; `round9_simplify_budget.out` | Regimen |
| Budget | $82.34 left; 21 / 15 / 10 iterations fit | D | seed §7; *budget* §6 | What fits |
| Current design | round $26.18 / $25.12; null round 1 $27.22; box A + C $3.31 | D | `round9_synthesis_checks.out` §6 | Comparison |
| ρ on `own1` | — | U | Box A″, own1 arm | Whether coins are bought |
| B on `own2`; B(η) past the band | — | U | Box A″ screens | η̂, form of the step |
| Structure on π_1 and beyond recruits | — | U | Each iteration's batch, then its screens | Nothing before them |
| o, I-φ | — | U | The smoke, only in phase H | Phase H's cost |
| ĉ | — | U | Reset, held out | Next base |
| Compounding, human level, exploitability | — | U | Claims; none; none | Nothing |
| 12,800 screens, 37,800 confirmations, 30,000 batches, two-null stop | — | C | Consequences above | Regimen |

## 4. What is new, and the nearest work

Searched 2026-10-21; not finding something does not prove it absent.

1. **A line search on the true objective, read in paired self-play matches, along an exact natural-gradient direction from near-tie micro-randomisation.** TRPO's backtracking search checks a surrogate and a KL bound, not the return ([OmniSafe's TRPO notes](https://omnisafe.readthedocs.io/en/latest/baserl/trpo.html)); Pirotta et al. size the step by a lower bound; [HCPI](https://proceedings.mlr.press/v37/thomas15.html) (Thomas, Theocharous & Ghavamzadeh 2015), the nearest in spirit, deploys a candidate only when an off-policy importance-sampling bound clears a baseline; Stockfish's SPSA measures in play but along random directions ([chessprogramming.org](https://www.chessprogramming.org/Stockfish's_Tuning_Method), as the round 9 critique read it). A search for a return-measured line search found none.
2. **The class-wide logit shift as the step** (the softmax natural-gradient form), with a band-only arm that separates in play the gain inside the band from the gain past it.
3. **Facts about the network:** its near-tie recruit errors are matchup-dependent, concentrated on turn 1, and not a by-product of the factored decode.
4. **The reset's soft target is the shifted logits**, the step itself.

**Nearest applied work:** the [Wesnoth ML Recruiter](https://wiki.wesnoth.org/Machine_Learning_Recruiter), a learned recruiter that beat the RCA recruiter inside a scripted AI; matchup-dependent composition in MOBA drafting ([Chen et al. 2018](https://arxiv.org/abs/1806.10130), MCTS over hero sets; [JueWuDraft](https://arxiv.org/pdf/2012.10171)); AlphaStar's build-order statistics (`docs/literature_sparse_signal_20260921.md`). None uses exact per-decision labels in self-play or a play-measured step.

**Contradicting evidence:** learned build-order selectors overfit their training opponents (CherryPi 0.88 against 0.64–0.71 held out, repo survey); claims here are against `own1`, which the shifts were not fitted on, but `own1` is a close relative of the base. Steps computed from gradient and curvature overshot in round 9's ES toy, which is why the step is measured.

## 5. Weakest point

**The engine past recruits.** The gain per dollar rests on directions a small batch resolves and steps that a screen reads at SE 3 Elo. The record holds one such direction. Phase H is the current design's head with a cheaper evaluator: if the remaining band value is diffuse (thin, spread over many decisions, linear only in the token span), its directions need hundreds of thousands of games and its steps sit under a screen's 6-Elo resolution. This regimen then converges to the current design's cost per adoption and keeps only the cheaper evaluation and the shorter null path. Nothing on disk says which case holds.

Lesser: one η per class averages contexts where the shift helps and hurts (splitting turn 1 from later turns doubles the screens); flips past the band extrapolate a near-tie model into confident contexts, and a screen sees harm without locating it; the matchup keys extend ruling (h), and the shift stays a learned table in the player until a reset absorbs it. If (h) is no, the direction comes from the head on the recruit token outputs, and the smoke's R² decides whether that carries the structure.

**Rulings asked:** (a) and (e) as now; (h) extended to (faction, unit, enemy faction) keys and to a shift reaching confident recruit decisions. **Rejected here:** the marginal recruit decode (refuted by *split*); a reversed-direction known answer for the shift path, which catches a sign error only at ρ ≥ 0.49 [D, *budget* §4], where the hook needs no ρ.
