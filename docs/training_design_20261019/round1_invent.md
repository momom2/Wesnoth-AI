# Round 1 (invent): chance-made contrasts and turn-closure labels

## Summary

The binding constraint is information per dollar: a decision moves the result by about 0.03, a full-playout label carries noise variance 2, and $85 of such labels pins down about 170 free directions. The dice move results far more, so full playouts are spent only on contrasts the dice make large (forced fight outcomes, turn twins), which train a training-only evaluator V of end-of-turn states. V grades single decisions cheaply through labels that end at the mover's end_turn; a pairwise mirror-descent update anchored on `own1` moves the policy; each round's Elo is predicted before its match by the performance-difference lemma, from labels at the candidate's own changed decisions calibrated by full playouts.

## 1. The binding constraint

**Information.** A label y with true effect D and noise variance σ² carries D²/σ². Summed over labels, this gives χ². Fit a regression with d free directions to labels of total χ², and its error variance stands to the signal variance in the ratio d/χ². It therefore recovers a share χ²/(χ²+d) of the signal, and recovering 90% needs χ² ≈ 9d.

**Effects.**
- *Decisions:* δ ≈ 0.03 result units (win +1, loss -1). The end_turn offset's +229 Elo raises the win rate to 0.789, which is g = 0.58 result units a game. That gain is spread over the roughly 20 to 40 decisions a game that the offset changes.
- *Dice:* at argmax both players are deterministic, so with the setup fixed, R - V(s0) = Σ_c L_c. The sum runs over chance events c, and L_c = V(after c) - E[V(after c) | before c] is the Doob increment. The increments are uncorrelated, so Var(R | setup) = Σ_c E[L_c²]. In the 18,800 decisive games of the 2026-10-19 luck walk, a game has 111 fights on average (`training/metrics/match_luck_20261019/rows.jsonl`; 47 recruits roll traits too). So the average fight carries E[L²] ≤ 0.95/111 ≈ 0.0086. Two outcomes drawn independently from one fight's distribution differ in value by E[(V_k - V_k')²] = 2·E[L²] ≈ 0.017 (τ ≈ 0.13). The two extreme outcomes of a wide fight differ by about three times that.
- *Noise:* σ² = Var(R1|x1) + Var(R2|x2) - 2·Cov ≤ 2(1 - ρ), where ρ is the correlation the coupling of dice buys (section 2.5).

**Costs.** At 17,000 games an hour of 450 decisions on a $0.42 box, a forward costs $5.5e-8. A full-playout pair from a random decision costs about 450 forwards, $2.5e-5.

| label | effect² | noise σ² | cost | χ² per $ |
|---|---|---|---|---|
| decision fork, full playouts (seed §4) | 0.0009 | 2 | $2.5e-5 | 18 |
| forced fight outcomes, natural pair, full playouts | 0.017 | 2 | $2.5e-5 | 340 |
| forced fight outcomes, extreme pair | ≈0.05 | 2 | $2.5e-5 | ≈1,000 |
| turn twins (two turns from one turn start, each with its own dice) | ≈0.045 | 2 | $2.5e-5 | ≈900 |
| turn-closure label (section 2.3) | 0.0009 | ≈0.022 | ≈$5.5e-7 | ≈74,000, about V's opinion |

The seed's design buys about 1,500 χ² for $85, enough for about 170 directions. My reading of the record's grader B (docs/turn_value_prereg_20260925.md), unmeasured: a head on the trunk (hundreds of free directions) fitted to 36,209 playouts, at most about 420 χ² even as pairs, and in fact fitted to levels (rank weight 0), which spends the data on who is winning rather than on differences within a position. It read 0.27.

**Design rule:** full playouts go only where chance makes the effect large. Decisions are graded by the evaluator those playouts train.

## 2. The algorithm

### 2.1 The evaluator V (used only in training, reads the true state)

V(x) = E[R_mover | x, both sides play π] at the mover's end_turn, just before the opponent's first decision.

- **Input.** The true state, encoded with fog switched off on a forked core (the observation carries a fog flag), then read by `own1`'s frozen trunk; whether a fog-trained trunk reads hidden units well is part of what S2 measures.
- **Head.** A linear head over the pooled trunk features at first.
- **Growth.** The head gets wider only when the accumulated χ² of its labels exceeds 9 times its width.
- **Loss.** On pairs from one parent: (V(x1) - V(x2) - y)². The loss is antisymmetric by construction, and the parent's value cancels.

### 2.2 Forced-outcome contrasts (new)

1. Take a fight from `own1`'s self-play games and enumerate its K outcomes exactly (`GameCore.attack_outcomes`).
2. Pick two outcomes: for half the pairs, two independent draws from p; for the other half, the pair the current V spreads most.
3. Force each by rejection over seeds until the core's outcome key matches (about 1/p_k tries at 0.05 ms; the key fixes the successor state, per the `tools/combat_outcomes.py` docstring).
4. From each forced state the mover re-decides to its end_turn (x_T1, x_T2), then the game runs to its end under π with coupled dice.
5. The label y = R1 - R2 has E[y | x_T1, x_T2] = V(x_T1) - V(x_T2): each twin is a legal continuation of a reachable state.

The effect is the dice's whole share, rare outcomes get labels (a leader hurt, an attacker killed on its own attack), and what it teaches is what the prior lacks: the price, in context, of hit points, kills, experience, slow and poison (the deployed player has no combat arithmetic in its priors, docs/selfplay_redesign_20260904.md, S1).

**Turn twins** complete V's data: from one turn start, the argmax turn and the argmax turn with one decision replaced by the prior's second choice, each re-decided and rolling its own dice. Turns with their own dice are valid here because V scores states; the ruling forbids them only for comparing choices.

### 2.3 Turn-closure labels (new as a construct)

At a decision with observation history h, the candidates are the top 4 joint actions under π, plus end_turn:

  Q̂(h, a) = Σ_k p_k(a) · V(x_T^{a,k})

- k runs over the exact outcomes of a's fight. All outcomes with p ≥ 0.02 are enumerated; the rest form one stratum, sampled in proportion to p. A move or a recruit has K = 1.
- x_T^{a,k} is reached by applying a with outcome k, then letting π re-decide after every later roll until end_turn. The dice are coupled across candidates.
- For end_turn, Q̂ = V(x_now): one forward, no dice.

Properties:
- **Bias.** E[Q̂ | s] = Q_π(s, a) + b, where b is V's bias at the end-of-turn states.
- **Noise.** The horizon stops before the first opponent decision, so no opponent appears in the label. The decision's own dice are integrated, never selected (the record's flaw in comparing turns played with their own dice). What remains is the later in-turn dice: with 2.6 fights per side-turn (111/42), about 1.3 left after a mid-turn decision, 2 × 1.3 × 0.0086 ≈ 0.022 at ρ = 0.
- **Cost.** About 2 × (4 decisions + 1 read of V) for a pair, times K for a fight. That is 60 to 115 forwards per decision for the 5 candidates, so $3 to $6 per million decisions.

### 2.4 The policy update

The target is magnetic mirror descent ([Sokota et al.](https://arxiv.org/abs/2206.05825)) with `own1`, frozen, as the magnet m:

  ℓ*(a) = [ℓ_t(a) + αη·ℓ_m(a) + η·Q̂(a)] / (1 + αη)

ℓ is a joint log-probability, the sum of the factored heads. The fit uses the [REBEL](https://arxiv.org/abs/2404.16767) pairwise squared loss over candidate pairs:

  Σ_{i<j} [(ℓ_θ(a_i) - ℓ_θ(a_j)) - (ℓ*(a_i) - ℓ*(a_j))]²

A term KL(π_t ‖ π_θ) holds the non-candidates. Zero-mean label noise leaves the squared-loss minimizer unchanged; no value gradient enters the trunk (the record's 94-99% value share is gone); η is the largest value that changes the argmax on at most 5% of held-out decisions (about 11 per side per game). Play stays argmax.

**The prediction.** With π′ the candidate on side A against `own1` on side B, the performance-difference lemma gives, exactly, J(π′) - J(`own1`) = E_{π′}[Σ_flips A_own1(h_t, a_t)]: side B's decisions and every unchanged decision contribute 0, since `own1` is deterministic. Before the match: play 2,000 candidate games against `own1`; at every changed decision take the turn-closure label under round 1's V, with `own1`'s memory carried in shadow; calibrate with full playouts at 2,000 changed decisions, 16 pairs each; predict Elo ≈ 347·ĝ (the slope of Elo near a 0.5 win rate). Labels taken fresh on the candidate's own changed decisions carry no selection bias from the training labels.

### 2.5 Coupled dice (adapted, used only if measured useful)

**Per-unit luck streams.** Each fight's seed in a label playout is hash(salt, turn, the attacker's birth key, that unit's fight count this turn). A trait roll's seed is hash(salt, side, turn, recruit ordinal). A unit's birth key is (side, turn recruited, recruit ordinal).

Within one branch each key is used once and is a function of the past, independent of the draw, so each branch's law is exact. A unit that attacks in the same turn in both twins rolls the same draws whatever its target, and its strikes agree with probability 1 - |cth1 - cth2|, the maximal coupling of two Bernoullis. Only label playouts use the keys, through the fight's existing seed argument; engine games and replays keep the engine's RNG.

The record killed common random numbers on 2026-08-17 (median 0 shared fights under fight-identity keys, on a collapsed checkpoint rolling a median of 0 dice per turn). This is a different key on a healthy policy, read as a variance ratio. Every cost above assumes ρ = 0.

### 2.6 Fog, memory, invariants

**Fog.** Every decision inside a label is made by the fogged π on its own observation and memory; only V reads the true state, which is a draw from the posterior the game itself induces. So E[label | h] = Q_π(h, a) + E[b | h], with no strategy fusion: no decision inside a label sees hidden units.

**Memory.** Forks copy both memories. Round 1's update reads the stored memory at each labelled decision as a fixed input. The size stays 64.

**Invariants, each a test:** twins forced to the same outcome under the same keys give y = 0 exactly; the forced key equals the realized key; Σ p_k = 1; per-strike hit frequency matches cth (χ² test over 10⁶ strikes); a candidate identical to `own1` predicts ĝ = 0.

## 3. The regimen

Every stage runs on a 4090 at $0.42 an hour unless marked. The ranges are my predictions, written before any run.

| stage | compute, cost | predicted readout | kill line |
|---|---|---|---|
| S0 plumbing, planted check P-C | CPU box, $0.3 | every invariant above holds; strike law p > 0.01 | any failure stops the line |
| S1 coupling | 8,000 twin pairs, 16,000 playouts, $0.2 | ρ: decision twins 0.15 (0.05-0.30), forced twins 0.30 (0.10-0.50); independent dice \|ρ\| < 0.05 | ρ of forced twins < 0.10: keys dropped |
| S2 evaluator V₀ | 400,000 pairs (60% forced, 30% turn twins, 10% stop decisions), $10; training $0.3 | held-out pair-MSE reduction 0.012 (0.006-0.020) against E[y²] ≈ 1.9, 2 standard errors ≈ 0.002 on 100,000 held-out pairs; 199-position benchmark 0.45 (0.38-0.55); P-A recovered | MSE reduction < 0.004, or benchmark < 0.40 (the HP margin's reading) |
| S2b audit | 2,000 decision forks × 32 pairs, 128,000 playouts, $1.6 | slope of full-playout labels on turn-closure labels 0.8 (0.6-1.0; standard error 0.12) | slope < 0.4 |
| S3 learned stop | 300,000 stop decisions, $0.3; match $0.12 | a head on the frozen trunk shifts end_turn's logit; its mean implied offset -1.5 (-2.2 to -0.8); +20 (+5 to +40) Elo over the offset player | under +5, or offset outside its range |
| S4 round 1 | 1M decisions × 5 candidates, $4.4; audit at changed decisions $0.8; match $0.12 | ĝ, computed before the match; my prior +35 (+15 to +60) Elo | match < +12, or off ĝ by more than 2 combined standard errors |
| S5 rounds 2-4 | $10 a round: V refitted on 200,000 new pairs from π_k's play, warm-started; 1M labels; audit; match | each round's ĝ, predicted shrinking (about +25, then +15) | stop when ĝ < 25 Elo (2 standard errors) |

The total is about $50 of the $85, which leaves room to repeat S2 or S4 once. One factor changes per stage. Every claim of strength is an 800-decisive-game match against `own1`.

**Planted-signal checks.** Each check modifies the terminal reward and adds the planted term to recorded results, so it costs no playouts. The continuation policy is fixed, so under the modified reward A' = A + δ_p·f exactly, and the right answer is known.

- **P-A (evaluator path).** Add δ_p = 0.2 to the mover's label when the forced fight's defender is dead at the mover's end_turn, a fact visible in the state. Train V on the same pairs with and without the term. On pairs that differ in that fact, the two Vs must differ by 0.20 ± 0.02 once 5,000 such pairs are in (standard error √(2/N_d)). The error must fall as 1/√N. The measured curve sizes every later stage.
- **P-B (label and update path).** Add δ_p = 0.05 to Q̂(end_turn) in the S3 and S4 labels. Prediction: end_turn's logit rises against every candidate by ηδ_p/(1+αη). The flip rate rises by the held-out share of decisions whose gap between end_turn and the best action falls inside that shift, a share read from `own1`'s logits. Both must hold within 10%.
- **P-C (simulator path).** The invariants of section 2.6.

**The case for about $1,000** stands only if S4 and S5 each land within 2 standard errors of ĝ: 10M contrast pairs ($250, χ² ≈ 2·10⁵, about 2·10⁴ directions, enough for adapters on the top trunk layer), 30 rounds ($300), and a league check each round against `own1`, `parity3` and every earlier round (a round that gains on `own1` but loses to an earlier round flags non-transitivity); about $650. No human anchor is measured, so the record does not price the distance to the strongest humans.

## 4. What is new, and the nearest prior work

Searches (2026-10-19) for value training on paired or forced chance outcomes found nothing that manufactures contrasts at chance events.

- **Forced-outcome contrasts (new).** Nearest: GNU Backgammon's rotation of first rolls ([manual](https://www.gnu.org/software/gnubg/manual/html_node/Quasi_002dRandom-Dice.html)), which stratifies one rollout estimate and creates no labels; [MIVAT](https://webdocs.cs.ualberta.ca/~bowling/papers/09ijcai-mivat.pdf) (White & Bowling 2009) and [AIVAT](https://arxiv.org/abs/1612.06915), which learn or use a luck-correcting value passively, from games already played; afterstate values in Stochastic MuZero ([Antonoglou et al. 2022](https://openreview.net/forum?id=X6D9bAHhBQ1)).
- **Turn-closure labels (new as a construct).** Nearest: model-based value expansion with a fixed horizon ([Feinberg et al. 2018](https://arxiv.org/abs/1803.00101)) and truncated rollouts ([Tesauro & Galperin 1996](https://proceedings.neurips.cc/paper/1996/hash/996009f2374006606f4c0b0fda878af1-Abstract.html)). New: a horizon at the mover's end_turn, so no opponent decision falls inside a label; the decision's own chance integrated exactly; a privileged evaluator paired with deciders under fog.
- **Adapted, not new:** per-unit dice (common-random-number synchronization, Law & Kelton; [Veness, Lanctot & Bowling 2011](https://proceedings.neurips.cc/paper_files/paper/2011/file/d736bb10d83a904aefc1d6ce93dc54b8-Paper.pdf), whose shared outcomes suit chance that does not depend on the action; event-keyed noise in [Sakib 2026](https://arxiv.org/abs/2609.24144), which also finds that lower contrast variance need not speed learning); the pair loss (Relative Value Learning, [Höftmann et al., ICLR 2026](https://arxiv.org/abs/2607.21120); Tesauro's comparison training, 1989); the update (REBEL, MMD, piKL [Jacob et al. 2022](https://arxiv.org/abs/2112.07544), conservative policy iteration); the privileged critic ([Pinto et al. 2017](https://arxiv.org/abs/1710.06542)). The χ² routing rule is classical statistics, not claimed as a method. The 2026 entries are checked only against their arXiv pages.

## 5. The weakest point

V's transfer from what the dice change to what decisions change. Chance contrasts teach the value of what dice move: hit points, kills, experience, statuses. Positional value (a hex's defence, zones of control, villages) varies only with decisions, at effects of size δ, so V reads positions through the frozen imitation trunk and through the decision part of turn twins, about 2% of their effect². A policy steered toward positions V misreads gains in its labels and loses in the match; S2b measures V's calibration on average, not on the states the policy will steer to, and the audits at changed decisions in S4 and S5 are the only defense. Expected consequence: tactics improve first (which fights, their order, when to stop, recruits), and positional play stays the prior's until decision contrasts accumulate.

Further risks: the record's learned evaluators all failed, and my explanation (fitted to levels, short of χ²) is a reading, not a measured cause; deterministic argmax under fog is exploitable, which Elo against one reference cannot show; if luck is spread thinly over many small fights (τ < 0.08), forced contrasts lose most of their advantage, which S2's MSE readout catches.
