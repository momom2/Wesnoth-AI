# Round 4 (invent): law-factored students

## Summary

The current design learns most of what it knows from chance: S1 carries about 125,000 χ² on chance differences, about 1,000 on decisions. The fogged student sees none of it directly. It regresses the critic's aggregated look-ahead Â_C on four candidates, from frozen features that hold weapons, resistances and traits but no hit chance or kill probability, so it must rebuild each fight's law inside a two-layer head.

This proposal changes the student and keeps the frame (critic, stopping, coins, matches, forecast ladder, rounds):

1. **Exact-law factorisation.** For an attack b, E[Q(x, b) − V(x) | h] = Σ_o p_b(o | h) · ω(h, b, o) exactly when the law is a function of the mover's observation. The student learns only the outcome values ω; the core supplies the law, in play as in training, from the mover's own view.
2. **Free targets.** The residual objective already evaluates C at every enumerated outcome of every fight, about 900 values a game. Centred on the decision state, they are the outcome head's targets, for one inference pass.
3. **Wider candidates.** Every legal attack is a candidate. Approach moves carry option tokens: the exact-law value of each attack their destination opens.
4. **Same-unit attack coins** calibrate attacks outside the prior's top four.

## 1. The gap

- **Attacks are frequent.** An `own1` game has about 110 fights (`training/metrics/match_luck_20261019/readout.txt`). In the 100 recorded `parity2` side-games of `training/metrics/critic_step1_20261009/prior_gaps.games.jsonl` (my count), an attack is the top action at 53 of 272 decisions per side-game, and 38% of decisions have an attack in the top eight. Attacks start from the unit's current hex (`wesnoth_ai/action_sampler.py`, `start_hex: unit_pos`), so most follow an approach move in the same side-turn.
- **The network does not see the law.** The unit row holds damage per strike, strikes, damage type, specials, resistances, traits, abilities and the time-of-day modifier (`wesnoth_ai/encoder.py`, the parity unit row); no hit chance, strike-order distribution or kill probability, as far as I read. The hand-placed combat prior is off (`COMBAT_TARGET_ALPHA = 0.0`, `wesnoth_ai/constants.py:150`). A human sees each weapon's outcome distribution in the attack dialog before confirming.
- **The design's student must learn it.** Its target Â_C(x, b) = Σ_o p_b(o | x) C(x∘b∘o) − C(x∘a0) is one scalar per candidate. A kill probability is a binomial tail with a damage threshold that jumps when the defender has one hit point less; a smooth head on frozen tokens reaches it only through capacity, and that error enters every attack advantage.

## 2. The algorithm

### 2.1 Decomposition

Let x be the true state at a decision, h the mover's information state (observation and memory), b a legal attack (actor, target, weapon), and p_b(o | x) the core's enumerated law over outcome keys o (both units' hit points, death, experience, slow, poison, petrify, advancement). The fight is the next event, so Q^π(x, b) = Σ_o p_b(o | x) V^π(x∘b∘o) exactly. When the law is a function of h, written p_b(o | h) (both fighters visible, and every unit whose ability enters the fight, such as leadership or illumination),

    E[Q^π(x, b) − V^π(x) | h] = Σ_o p_b(o | h) ω(h, b, o),    ω(h, b, o) = E[V^π(x∘b∘o) − V^π(x) | h]

Given h, the outcome is independent of what h hides, so the law factors out. Centring on V^π(x) = Q^π(x, a0) makes π's own choice score about zero, so scores compare as advantages. Recruits factor the same way over the trait law, known before recruiting.

### 2.2 The outcome head

w_θ(h, F, o) reads `own1`'s frozen token outputs (attacker, defender, global token, memory), an embedding of the outcome key and the fight's hex. Squared loss on y = C(x∘F∘o) − C(x), x the decision state where F was chosen; the minimiser is ω, the outcome's information-state value, fusion-free as in the design ([Baisero & Amato](https://arxiv.org/abs/2105.11674)). π's choice of F is a function of h, so P(x | h, F) = P(x | h): training on chosen attacks biases coverage, not the target.

Targets:
- **(a)** every on-path fight of every critic-data game, both sides: about 8 enumerated outcomes (p ≥ 2%) a fight, about 900 a game, children the residual evaluates anyway. They need one inference pass of the final critic, since training saw a moving critic;
- **(b)** off-path attacks: at one attack-legal decision in ten, the outcomes of every legal attack (about 30 forwards), covering targets and weapons π did not pick;
- **(c)** recruits: the two trait draws the residual already evaluates.

**Moves** keep the design's student m_θ(h, b) over the top four and its target E[C(x∘b) − C(x) | h]. A move whose destination H borders visible enemies also carries option tokens, one per attack F it opens from H, holding F's law and v_F = Σ_o p_F(o | h) w_θ(h, F, o) (stop-gradient). The head learns how options become a move's value: π may not attack, and the destination exposes the unit.

### 2.3 Decode and calibration

Every legal attack scores Q̂(h, b) = Σ_o p_b(o | h) w_θ(h, b, o); recruits the same sum over the trait law; the top four moves m_θ; end_turn stays with the Longstaff–Schwartz head. The switch rule is the design's (κ gate, β_avg, posterior ≥ 0.8, at most 10 switches per side-game), over the types attack in the prior's top four, attack outside it, move, recruit and stop.

**Same-unit attack coins.** When a0 is an attack and its unit has another legal attack, the side plays a uniformly drawn other attack of that unit with probability 0.05. The propensity is a fixed function of the state and the critic reads the rate in the schedule, so the identity stays exact. S1 holds about 2.5 million such decisions; at an assumed spread of 0.02 among a unit's attacks and residual variance 0.55, κ_outside gets about 90 of information, an SE of 0.11 (0.15 per fold). The shift is about 1.4 deviations × 0.015 ≈ 0.02 per side-game; weak-coin density falls to keep the total within 0.06.

### 2.4 Why it should extract more per dollar

In decreasing order of confidence:
1. **The law's error leaves the student.** The factored error on an attack is Σ_o p_b(o)(w − ω), outcome-value errors averaged by the exact law; the unfactored one also carries the head's approximation of p_b. With w set to the critic's own values, Q̂ equals the design's target exactly (I-law1), so the two students target the same quantity and differ only in what they must learn.
2. **One head serves all fights.** An outcome (a dead archer, a slowed troll at 12 hit points) recurs across candidates and positions, so an attack never played gets values other fights pinned. The aggregated target pins one law-weighted sum per candidate; the factored target pins about 8 numbers per fight, for forwards the residual already spends. These are the chance differences the critic learns best (τ ≈ 0.13 between two outcome draws, `round1_invent.md` §1), against decision spreads near 0.01.
3. **Candidates grow where the prior is weakest**, at 0.06–0.11 ms of core time per law and no forward.

**Derived prior** for the attack part of S3, with the design's N·E[max(X, 0)], X ~ N(−m, q·s²): N = 40 attack decisions per side-game with at least two legal attacks, m/s = 0.75, s = 0.015–0.03 (assumed; S−1 and S2 measure it). At q = 0.3, the design's student, 4–9 Elo; at q = 0.6, 14–29 Elo. Example: s = 0.02, q = 0.6 gives σ = 0.0155, E[max(X, 0)] = 0.00137, 0.055 in z over 40 decisions, 19 Elo at 347 Elo per unit of z. Moves gain only as far as S2's option R² shows.

### 2.5 Invariants (tests from this spec)

- **I-law1.** On recorded positions, with w replaced by the critic's centred outcome values, Q̂(h, b) = Â_C(x, b) + C(x∘a0) − C(x) to float tolerance for every legal attack. Permuted probabilities, a shifted weapon index or swapped roles break it.
- **I-law2.** law(h) = law(x) on recorded fights where every unit within two hexes of both fighters is visible; mismatches elsewhere are counted.
- **I-law3.** The logged same-unit propensity equals 0.05/n_alt, recomputed by replaying the actor; no coin when n_alt = 0.
- **I-law4.** On the design's enumerable duel, w trained on (a) alone converges to V's outcome values.

## 3. Regimen

Prices are the design's. **P-law**, the planted check of the new path, takes P2's reward-stream form:
- **Plant:** r = 0.03 booked to a side for each kill it makes with a ranged weapon, retaliation included. The critic learns it as an additive stream head; both students get stream heads.
- **Known answer:** fork truth at 300 positions over every legal attack and the top-four approach moves. The immediate part r · P(ranged kill | b) is exact; the rest comes from 8 playouts per candidate, the stream counted from the record.
- **Prediction:** the factored planted advantage regresses on fork truth with slope 1 ± 0.1 (SE about 0.05); unfactored reported, predicted 0.4–0.8; approach moves 0.7–1.1. A permuted law, a weapon shift or a perspective sign error drives the factored slope toward zero or below, because the bonus sits on one outcome key per fight.

P-coin's stream also covers the same-unit coins (target-hex parity).

| Stage | Cost | Predicted readout | Kill line |
|---|---|---|---|
| **S−1 replay** | $0.6 | The design's (a)–(c). **Law census**: legal attacks per attack-legal decision 2–6; decisions with two or more, 30–70 per side-game; SD across a decision's legal attacks of P(defender dies) 0.1–0.3; approach moves 30–50% of moves; law(h) ≠ law(x) on under 1% of fights. | The design's. Under 15 eligible decisions per side-game, or SD under 0.05: no outside-top-four type, factorisation kept. Mismatch over 5%: factorisation declared approximate; S2's A/B decides. |
| **S0 instruments** | $4 | The design's (a)–(e); same-unit coins inside (d)'s arms: shift 0.01–0.03 per side-game. | The design's. Shift over 0.04: rate halved. |
| **S1 critic** | $23 | The design's (P1, pilot, ρ_C, κ). New: the final-critic pass over 25,000 games' enumerated children ($1.4) and off-path enumeration ($0.8). Fold-crossed held-out R² of w on outcome deltas 0.5–0.85; κ_outside 0.5–1.0, SE 0.11–0.15. | The design's. w's R² under 0.3: the fogged player cannot read outcome values; the design's student stands. |
| **S2 operators** | $8.5 | The design's (P2, P3, stop). **P-law.** **A/B** on the same folds, held out by position, attack advantages over the full legal set against Â_C: factored R² 0.35–0.7, design's 0.15–0.45, paired gain ≥ 0.1. Option tokens raise approach-move R² by 0.03–0.10. β_avg (attacks) 0.6–1.0. | P-law outside: factored path broken, the design's student for attacks. Paired gain under 2 SE: the design's student. Option gain under 2 SE: tokens dropped. κ_outside lower bound ≤ 0: top four only. |
| **S3 round 1** | $5.7 | The design's matches (SE 5 Elo each) plus **attacks-only** (+$0.7); Ĝ_plug(0) per factor written first. Priors: attacks-only +14 to +29; actions-only +14 to +45; stop-only +5 to +35. | The design's. Attacks-only under +4: read the ladder for attacks; Ĝ_plug(0) > Ĝ_plug(2) ≈ match means static outcome values, a match below Ĝ_plug(2) selection among many candidates. |
| **S4 rounds 2–3** | $12 each | The design's, w refit warm on each round's free targets; the band on the factored margin for attacks. Prior 0.2–0.8 of the previous gain (not derived). | The design's. |
| **S5 distillation** | $5 | The design's P5; the factorisation stays in the player, like the legality mask, and heads are distilled around it. | The design's. |

**Budget.** S−1 to S3, round 2 and S5: about $59; round 3 brings it to $71. The remaining $14 covers the batch-invariant kernels ($7) with a partial S2 rerun, not an S1 rerun.

**Beyond `own1`.** The $1,000 case keeps the design's gate (two rounds with lower 2-SE bounds above zero, landing near Ĝ_plug(0)). The outcome head grows with the critic, and its targets scale with critic data at no generation cost. Flagged for the user: at the first trunk training, law tokens could enter the observation for legal and option attacks, giving the network the arithmetic the factorisation now supplies. That opens a new observation epoch; observation changes gave this lineage +56 and +73. Its test is a match against `own1`. Nothing measured links `own1` to human ratings.

## 4. What is new, and the nearest work

Searched 2026-10-19 (four searches); absence is not proven.

1. **A fogged player's advantage factored through the exact, observation-computable law** of the fight a candidate executes, with learned per-outcome information-state values from one trunk pass. Nearest: [Stochastic MuZero](https://openreview.net/forum?id=X6D9bAHhBQ1) (an afterstate head predicts a learned distribution over VQ chance codes; a dynamics step runs per code); afterstate values ([Sutton & Barto §6.8](http://incompleteideas.net/book/the-book-2nd.html)); TD-Gammon's 1-ply search, one network evaluation per successor ([Tesauro 1995](https://dl.acm.org/doi/10.1145/203330.203343)); [Expected Policy Gradients](https://arxiv.org/abs/1706.05374), which integrate over the policy's own actions, not nature's. Here the law is exact and given, no successor is simulated in play, and the values are privileged regression targets.
2. **Per-outcome targets from the critic's chance-sibling evaluations**, a by-product of the exact residual objective that carries its chance information to the fogged player. Nearest: asymmetric actor-critic and oracle guiding ([Baisero & Amato](https://arxiv.org/abs/2105.11674); [Suphx](https://arxiv.org/abs/2003.13590)), which pass a privileged value or policy; [AIVAT](https://arxiv.org/abs/1612.06915), which uses per-outcome values for variance only.
3. **Option tokens**: exact-law values of the attacks a move opens, as inputs to a learned move head. Nearest: Wesnoth's default AI rates attacks with hand-written formulas over the same laws (`src/ai/default/attack.cpp` at 1.18.4; not re-read). Here every value is learned.

Not claimed: same-unit coins (an instance of the design's coins), the calibration.

## 5. Weakest point

**The exact part covers only the fight a candidate executes.** Much of an attack's value may be decided one decision earlier, by the approach move. There the option tokens supply the arithmetic, but whether π then attacks, and what the destination exposes the unit to, are learned from the aggregated target. If `own1`'s errors lie mainly in approach and exposure rather than in target and weapon choice, the gain shrinks toward the design's; S2's R² split and S3's attacks-only match show which.

Further risks: w trains mostly on π's chosen attacks, and off-path enumeration covers one decision in ten, so unusual targets rest on generalisation. Law mismatch from hidden units' abilities is uncounted until S−1, and I have not checked whether the human dialog accounts for it either. The spread s is assumed; at s = 0.015 and q = 0.4 the attack part falls to about 7 Elo. The chain inherits the critic's accuracy at outcome children, which S1 must establish.

## 6. Decision record

- **Deferred: luck steering in critic-only games.** Each local residual uses the enumerated law, so the loss's zero set does not depend on the law that drew a path's outcomes: dice could be tilted toward contested positions with no importance weight in the critic ([OFFER](https://ojs.aaai.org/index.php/AAAI/article/view/10810) tilts environment variables and needs weights). A rule reading only what both sides see leaves the information-state posterior unchanged; coin, stop and match games keep the true law. Its gain depends on the share of decisions in decided positions, which no record holds; S−1 can count it.
- **Deferred: threat tokens** (laws of visible enemies' next-turn attacks on a destination): rule-computed, learned-valued, one factor per stage; flagged, since the user may read them as structure.
- **Rejected: distilling the factored decode into a trunk without law inputs**, which would have to relearn the law the factorisation removed.
- **Luck heads stay rejected**: this proposal leaves the critic's residuals alone; w is a student, not a control variate.
