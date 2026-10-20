# Round 5 challenge: search first, certified by play

**Stance.** The current design builds a privileged critic, calibrates it per decision type with coins, and amortizes it into a switch head on a frozen trunk before any of it plays. This proposal inverts the order:
- **The improvement operator** is a one-step expectimax player over the network's own candidates. It exists and has played 1,791 games.
- **Its value** is a learned critic.
- **The certificate** is the player's own matches, before anything is distilled, as the standing rule requires.
- **Distillation** into the whole network comes last, through the own-trajectory trainer behind +88.6.

There are no coins, no κ or β, no student calibration and no forced paths. The new script is `analysis/round5_challenge_lookahead.py`; I also reused the provenance stance's `match_variance` and `playout_noise` scripts after reading them.

## 0. Why change the core

1. **The question every critic design hangs on has never been asked in play.** Does a learned critic's look-ahead choose better than the prior?
   - The 2026-10-08 program built the player (tag `archive/exp-value-policy-iteration`: evaluators `material`, `critic`, `rollout`; worlds `observed`, `godview`) and four critics (HF `tier-b/critic_step1_20261009`).
   - It then dropped the critic's gate (BACKLOG Q3) on the turn benchmark. That benchmark is a proxy, which the standing rules make a crash barrier, never a verdict. Its candidates were whole turns played with their own dice, a design since ruled suspect.
   - Even there, the true-state critic T100 selected better than the HP margin by +0.041 ± 0.017.
2. **With uninformed values, the operator is measured and neutral.** A flip is a decision the player changes from the raw argmax. Per flip, corrected for luck and setup:

   | Evaluator | Advantage per flip | Flips per side-game | Capped games |
   |---|---|---|---|
   | Material | −0.0021 ± 0.0033 | 9.9 | 19% |
   | Material plus exposure | −0.0008 ± 0.0016 | 18.8 | 0% |

   What learned values add is the open number, and it costs about $3 to read.
3. **A match on the flips is the cheapest truth about them.**
   - By the performance-difference lemma (PDL), a rule's gain is the sum of its flips' advantages under the old policy, since A^π(h, π(h)) = 0.
   - Compare a corrected match on m flips per side-game with coins spread over n_c decisions, at equal advantage. The match carries m²/(n_c·p(1−p)) times the information per game: 2.9, 11.7 and 47 at m = 10, 20 and 40.
   - The match measures the flips themselves. Coins calibrate at random deviations, and the flips' gain must then be extrapolated.
   - A 3,200-game corrected match resolves 0.0015 per flip at m = 20.
4. **Per-decision labels cannot carry the signal.**
   - Against `own1`, `own1` scores J = 0 and no policy scores above 1. So even an exact greedy step averages at most 1/N = 0.0042 per decision, at N = 237.6 decisions per side-game.
   - Certifying one decision of that size takes about 276,000 fork pairs.
5. **The critics on record were limited by games.**
   - Best held-out loss was 0.4915, 0.4712 and 0.4595 at 1.8k, 3.5k and 6.8k games.
   - Each was reached at 0.25 to 0.5 epochs, and the loss then rose to 0.57–0.66.
   - A plain self-play game costs $2.5e-5, against $3.1e-4 for a critic-data game of the current design: 12 times less.

## 1. The algorithm

### 1.1 What is learned, from which labels, by which update

**π_k**, the network: π_0 = `own1`, with its 64-slot memory, at `raw:t0+eo-1.5`.

**C_k**, the privileged critic (training only). It estimates V^{π_k}(x) for the side to move.
- **Structure:** two folds by game, the step-1 architecture, warm-started from the previous critic.
- **Label:** y_t = z − Σ_{e≥t} λ·(φ(x_e) − Σ_o p_e(o) φ(x_e^o)), where φ is the luck walk's linear columns and λ is fitted on the other fold.
- **Unbiased:** each term has mean zero given the past, so E[y_t | x_t] = V^{π_k}(x_t) for any fixed λ. From the start of the game, 0.848 of the variance is kept.
- **Sampling:** at most 8 positions per game, inside the 6–12 per game at which the old critics peaked.
- **Loss:** the existing C51 head, with labels projected onto its atoms.
- **No memories:** within `own2`, 64 slots against 0 read +13.0 ± 12.3.

**U_k**, the information-state value. It reads the mover's observation and memory.
- **Label:** C_k(x_t) at the same positions. Squared loss gives E[C_k(x) | h].
- **Noise:** the label noise is Var(C(x) | h). A result label carries E[Var(z | x) | h] + Var(V | h).
- **Gain:** Var(V | h) ≤ Var(V) = 0.305, and Var(z | x) = 0.608 at human mid-game positions. So the label variance falls at least 3.0-fold.
- U needs no belief sampler.

**S_k**, the search player. This is the archived player with U as its evaluator. At each decision:
- **Candidates:** B(h), the prior's top 8 plus end_turn.
- **Expansion:** each candidate is expanded exactly in the mover's observed world W(h), with unseen units removed. Attacks expand over their enumerated outcome laws; terminal outcomes take the result.
- **Value:** Q̂(b) = Σ_o p_W(o | b)·U_k(h(W∘b∘o)).
- **Play:** b* = argmax_b ℓ_k(b) + clip((Q̂(b) − V̄)/σ, −c, c), where ℓ_k is the joint log-prior and V̄ is the prior-weighted mean of Q̂.
- **The oracle S_k\*** uses C_k on the true state. It is a diagnostic only.

**π_{k+1}** is π_k trained on its own trajectories (DAgger) toward S_k's tilted distribution over B, mixed with the corpus as in the own-trajectory recipe.

**Rulings respected:**
- Each comparison is a single decision followed by π_k's own play, which re-decides after every roll. That is how V^{π_k} is defined.
- The search picks one action per decision.
- S_k reads only h.

### 1.2 Estimators and their properties

**With exact values, every flip gains.** The tilt is monotone in Q, so b* ≠ a0 only if Q(b*) > Q(a0). Every flip then has a positive advantage, and the PDL gain is positive.

**With noisy values, a flip can lose.** A flip happens wherever the noise beats the prior's margin, and its expected advantage can then be negative. That is the failure mode of the 32-simulation search that lost 124 Elo to argmax.
- **σ bounds it, from S2 on.** σ_b² = ½·Var[D_1(b) − D_2(b)], where D_j = Q̂_j(b) − Q̂_j(a0) under fold j's critic.
- **Blind:** σ is read on held-out decisions with no result. A full tilt therefore needs one standard error of the critic's own estimation noise.
- **S1 keeps σ = 0.1** from the archive, so the evaluator is the only factor changed against the two recorded matches.

**Readouts per match:**
- **Ĝ:** the corrected gain, with the provenance stance's leave-one-match-out setup baseline and luck columns. r = 0.747, so SE = 347.4·√(r/n) Elo.
- **m̂:** flips per side-game, by kind.
- **â = Ĝ/m̂:** bootstrapped over games.
- **Also recorded:** caps and decisions per side-turn.

**Attribution:** the oracle's gain is the critic's decision value, oracle minus fair is the fog cost, and fair minus distilled is the amortization loss.

### 1.3 Why more signal per dollar

- **Direct reading:** the critic's decision value is read in play, as in section 0, item 3.
- **Exact laws:** outcome laws enter exactly at decision time. The observation holds no hit chance (`encode.rs`), and U never needs one, since it values states after the outcome.
- **Cheap data:** critic data costs 12 times less.
- **Few mechanisms:** four in all (C, U, the player, the trainer), and the player and the trainer have already run.

### 1.4 Invariants (tests from this spec)

- **I1:** with c = 0, or a constant evaluator, the player reproduces the raw argmax decision for decision. This test exists.
- **I2:** expanded laws match the core's enumeration by χ²; terminal outcomes take the result.
- **I3:** corrected labels keep the raw mean (P1).
- **I4:** with nothing hidden and U = C, S and S\* choose alike.
- **I5:** U trained on a constant C is constant.
- **I6:** flips are counted against the raw argmax at the same state and memory.
- **I7:** after distillation, the network agrees with S on its training states at least at the generation self-agreement rate (P5).

## 2. The regimen

**Prices:**
- a game: $2.5e-5;
- a corrected 3,200-game match: $0.48;
- a look-ahead 3,200-game match: $1.35 (1.6 times the wall time, plus 15.5 critic forwards per decision);
- 2 SE at 3,200 corrected games: 10.6 Elo.

**S1: two look-ahead matches against `own1` raw** ($2.7). Each is 3,200 corrected games with the operator untouched: k 8, c 1, σ 0.1.
- **Arms:** the oracle, with T100 on the true state; the fair player, with O100 in the observed world.
- **Readout:** no prediction, since no learned critic has played. The uninformed evaluators read −0.0021 ± 0.0033 and −0.0008 ± 0.0016 per flip.
- **A null means** |a| < 2·SE/m, which is 0.0015 at m = 20.
- **Decisions:**
  - Oracle above 2 SE: S3 on T100 now, S2 in parallel.
  - Fair above 2 SE: confirm on fresh seeds (PURE 800), then S4.
  - Oracle within 2 SE, or below −2 SE (exploited errors): S2a, with σ set blind.
- **Plant:** none, since nothing trains. I1 and I2 run on the box, and the two recorded matches are the references.

**S2a: one data step** ($4.5). 50,000 `own1` self-play games give critics C_a (two folds) and σ; then an oracle match.
- **Readout:** the epoch of the best held-out loss, Ĝ_a and â_a.
- **No prediction:** on the proxy, 4 times the data moved selection by +0.001 ± 0.016.
- **Decisions:**
  - Ĝ_a above 2 SE: buy S2b.
  - Otherwise: stop the ladder at about $7 in all, and hand the critic objective to the identity loss, judged by the same match.
- **Plant P1:** a mean-zero lottery with a logged law, booked into z. The corrected residual's slope on it reads 0 when correct, 1 with corrections dropped, 2 when sign-flipped. SE 0.007 on 50,000 relabelled games.
- **Plant P-score:** a scoreboard feature carrying δ per use of one natural action. The critic's coefficient must read 1 ± SE, and in a planted-game match the oracle's share of that action must rise.

**S2b: 400k games** ($11.4). Critics C_b, then an oracle match.
- **Readout:** the oracle's gain against critic data: 6.8k engine games, then 50k, then 400k.
- **Decision:** the $1,000 case rests on this curve.
- **Plant:** P1 again if the code changed.

**S3: U from the best C** ($3.1), then a fair match.
- **Readout:** the fog cost, oracle gain minus fair gain.
- **Decisions:**
  - Fair above 2 SE: confirm, then S4.
  - Fair null while the oracle gains: belief worlds next. The RB and RU code exists, and the 2026-10-12 record says what exists matters more than where.
- **Plant P-U:** a hidden per-game bit B, seen by C, with δ·B booked into the label. U must read δ/2 whatever B is: slope 0, where a leak would read 1.

**S4: distillation** ($7.4). P5 first, then 5,000 own game-sides labelled by S, giving π_1.
- **Matches:** π_1 against `own1` (corrected 3,200 plus PURE 800); π_1 against S; π_1 at offset 0 against −1.5.
- **Decisions:** adopt if the corrected lower bound is above 0 and PURE agrees within 2 combined SE. The offset leaves on non-inferiority (ruling b).
- **Plant P5:** `own1` at −2.5 distilled by the same trainer. Agreement must reach the generation self-agreement minus 2 SE, and the level must sit within 2 SE.

**Rk: round k ≥ 2** ($9.5 each). 100k games of π_k, warm refits of C and U, S_k certified against π_k, distillation and matches.
- **Decision:** continue only if the previous round's corrected lower bound is above 0.
- **Plants:** P1, and P5 whenever the trainer changes.

**Budget:**

| Path | Cost |
|---|---|
| Round 1 with S2b | $29.1 |
| Round 1 without S2b | $17.7 |
| Rounds 2 to 4 | $28.5 |
| Plan with S2b and rounds 2 to 4 | $57.6 |

The reserve of about $27 covers the identity-loss arm or the kernels, plus one remedy.

**The $1,000 case** is argued only from the measured oracle-gain curve and two certified rounds. Its content:
- critic data at 10⁶ to 10⁷ games ($25 to $250);
- 20 or more rounds;
- more test-time search for the final player, read by a k 8 against k 16 match;
- an exploitability probe.

Nothing measured links `own1` to human ratings.

## 3. Provenance

| Number | Value | Used for | Tag | Source or derivation |
|---|---|---|---|---|
| Advantage per flip, material look-ahead | −0.0021 ± 0.0033 corrected (raw −0.0025 ± 0.0035); 9.89 flips/side-game; caps 0.193 | S1 reference | measured | `round5_challenge_lookahead.out` §1–3 (`games_material.tar.gz`, `rows.jsonl`) |
| Advantage per flip, exposure look-ahead | −0.0008 ± 0.0016 corrected (raw −0.0013 ± 0.0019); 18.83 flips | S1 reference | measured | same (`games_exposure.tar.gz`) |
| Decisions per side-game | 237.6, 235.5 | PDL bound | measured | same |
| Greedy mean advantage per decision | ≤ 0.0042 | per-decision labels rejected | derived | PDL; J = 0, J ≤ 1; 1/237.6 |
| Fork pairs per decision at 0.0042 | ≈ 276,000 | same | derived | (2·√(2·0.608)/0.0042)²; noise read at temperature 0.5 (argmax's unknown; nothing else rests on it) |
| r, setup + luck columns | 0.747 (0.733–0.758) | every SE | measured | `round5_provenance_match_variance.out` (script read and checked) |
| Elo per z; SE at 800 / 3,200 / 12,800 | 347.4; 10.6 / 5.3 / 2.7 | thresholds | derived | 800/ln 10; 347.4·√(r/n) |
| a_min at 3,200 games, m = 10/20/40 | 0.0031 / 0.0015 / 0.0008 | meaning of a null | derived | 2·√(r/n)/m (script §4) |
| Match against coins, information per game | 2.9 / 11.7 / 46.8 | instrument choice | derived | m²/(n_c·p(1−p)); n_c 180, p(1−p) 0.19 from the provenance stance (parity3's band) |
| Old critics' best loss | 0.4915 / 0.4712 / 0.4595 at 1.8k / 3.5k / 6.8k games; best at 0.25 / 0.5 / 0.5 epochs; last 0.61 / 0.66 / 0.57 | data-limited claim; 8 positions per game | measured | script §6 (`critic_step1_20261009/*.summary.json`, `*.holdout.jsonl`) |
| Proxy readings | T100 − HP margin selection +0.041 ± 0.017; T100 − T25 +0.001 ± 0.016 | context; S2a predicts nothing | measured | `critic_step1_20261009/readout.md` |
| U's label variance reduction | ≥ 3.0 | U over result labels | derived | (0.608 + 0.305)/0.305 (`round5_provenance_playout_noise.out`: human positions, turns ≥ 7, temperature 0.5) |
| Luck columns alone keep | 0.848 | label correction | measured | `match_luck_20261019/readout.txt` |
| Game price | $2.5e-5 | S2, rounds | derived | $0.42/h ÷ 17,000 games/h (seed) |
| Decisive match game | $1.49e-4 | matches | derived | 17 min per 800 at $0.42/h (provenance stance) |
| Critic-data game, current design | $3.1e-4 | the 12× comparison | derived | round-4 design, checked by its critique |
| Critic forward | $5e-8 | look-ahead and U costs | derived | the current design's price, from 3,200 leaf evaluations/s with a margin |
| Look-ahead wall factor | 1.6 | match cost | measured | material: 1,569 s for 991 games (`match.walls`); raw: 929 s for 944 games (its pre-registration) |
| States expanded per decision | 14.7, 16.4 | match cost | measured | `material.lookahead.json`, `exposure.lookahead.json` |
| Look-ahead match, 3,200 games | $1.35 | S1–S4 | derived | 3,200·1.49e-4·1.6 + 3,200·237.6·15.5·5e-8 |
| Critic training rate | 283 positions/s | S2, S3 costs | measured | T100: 245,952 positions in 869 s (`walls.txt`, summary) |
| Position building | 21,485 games in 51 s on 62 cores | S2 cost (negligible) | measured | `critic_step1_20261009/positions.log` |
| Luck walk | 800 games in 9–18 s on 31 cores | label cost (negligible) | measured | `match_luck_20261019/walls.txt` |
| Own-trajectory training | $2.1 for round 1; `own1` +88.6 ± 12.7 over `parity3` | S4 cost; trainer path | measured | BACKLOG; `memory_own_trajectories`, `own_confirm` records |
| 64 against 0 slots within `own2` | +13.0 ± 12.3 | C reads no memories | measured | `memory_own_round2` record |
| P1 SE | 0.022 at 5,000 games; 0.007 at 50,000 | P1 band | derived | provenance stance: 1/(√n·√(160·0.05²)) |
| k 8, c 1, σ 0.1 in S1 | — | S1 | choice | archived defaults, so the evaluator is the one factor |
| 8 positions per game; ladder 50k, 400k | — | S2 | choice | inside the measured 6–12 at peak; each step gated by a match |
| Critic decision value in play | — | stage order after S1 | unknown | S1 ($2.7) and S2a ($4.5) measure it; nothing before them rests on it |
| Its scaling with data | — | S2b, $1,000 case | unknown | S2a and S2b; S2b runs only on S2a's reading |
| Fog cost; amortization loss | — | S3, S4 | unknown | the three-match attribution |
| Per-round decay | — | rounds, $1,000 case | unknown | continuation gated each round; the case waits for two rounds |
| Jones: 10× training ≈ 15× test-time compute | — | nothing | measured elsewhere | [Jones 2021](https://arxiv.org/abs/2104.03113) §IV-C (AlphaZero on Hex) |
| `own1` against strong humans | — | nothing | unknown | needs rated human games |
| Budget | $85 at $0.42/h | everything | given | seed §1 |

## 4. What is new, and the nearest work

Searched 2026-10-20; finding nothing does not prove absence.

1. **A fair one-step expectimax under fog, with its value distilled from a privileged critic onto the mover's observation.** The labels are C's readings, not results, which lowers their variance at least 3-fold. No belief sampler is needed, and outcome laws enter exactly at decision time.
   - **Nearest:**
     - asymmetric critics ([Pinto et al.](https://arxiv.org/abs/1710.06542); [Baisero & Amato](https://arxiv.org/abs/2105.11674));
     - expert distillation under partial observability ([Cai et al. 2024](https://arxiv.org/abs/2412.00985));
     - SePoT, look-ahead on top of a policy network with a critic of its value ([Kubíček, Burch & Lisý](https://arxiv.org/abs/2312.15220));
     - ReBeL's belief-state values ([Brown et al.](https://arxiv.org/abs/2007.13544));
     - IIMC's imperfect-information continuations (Furtak & Buro 2013, which I know only through [Rebstock et al.](https://arxiv.org/pdf/1903.09604)).
   - I found no perfect-information critic regressed onto information states as a fair search's leaf.
2. **Certification by play at three levels,** whose differences attribute a gap to the critic's decision value, to fog and to amortization, together with a per-flip PDL readout. Nearest: the evaluation of the search player in ExIt ([Anthony et al.](https://arxiv.org/abs/1705.08439)) and AlphaZero.
3. **The critic's decision value as a data-scaling curve read by matches** rather than by loss. Nearest: [Jones 2021](https://arxiv.org/abs/2104.03113).
4. **A tilt scale set blind from fold disagreement.** Weakly new. Nearest: piKL's anchor ([Jacob et al.](https://arxiv.org/abs/2112.07544)) and double estimators.

**Not claimed:**
- one-ply expectimax (TD-Gammon);
- Muesli's clipped target ([Hessel et al.](https://arxiv.org/abs/2104.06159));
- DAgger and ExIt;
- corrected matches;
- luck-corrected labels.

## 5. The weakest point

**The critic.** The measured priors run against it:
- On the proxy, 4 times the data left selection flat.
- O100 ranked candidate turns no better than material.
- Uninformed evaluators read about 0 per flip.

If S1 and S2a both read null, the plan has spent about $7 and closed the critic route without gain. It will at least have named the failure: no decision value at 50,000 games. The identity objective and rollouts, the next routes, are priced elsewhere.

**Lesser points:** a one-step look-ahead on V^π sees a multi-decision plan only at its last step, and compounding has no theory, as in the current design. The observed world also shows hexes a move would reveal as empty, which favours scouting; the fog cost includes this bias.

**Rulings for the user:**
- **(a)** The search player uses the simulator and exact outcome laws at play time. Is that within "learned only" and "the network picks actions"? If not, S_k is only a training-time teacher, and π_{k+1} plays. The stages stay the same; only the player whose strength is claimed changes.
- **(b)** The offset is removed on non-inferiority, as in the provenance stance's ruling (c).

## 6. Decision record

**Rejected:**
- **Coins of every kind.** A match on the flips carries 3 to 47 times more information per game and measures the flips themselves.
- **Longstaff–Schwartz stopping.** end_turn is a candidate in the same look-ahead. The offset's mean effect is measured at zero, so the remaining gain is heterogeneity, which the match reads.
- **The frozen-trunk student with κ and β.**
- **Critic-data games with children forwards.** They cost 12 times a plain game. They return only if S2a is flat.

**Deferred:**
- **A luck-corrected turn completion as the leaf.** It needs about 7 times the forwards and gets its own match once a one-ply gain is certified.
- **Belief worlds,** used only if S3's fog cost is large.
