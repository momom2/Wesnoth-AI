# Round 2, hardening: the identity made exact, decision differences gated, stages resized

Stance: keep the current design's frame (privileged zero-floor critic, one-step look-ahead, fogged student by squared loss, coin calibration, midpoint audit) and repair what is wrong in it. Whatever this file does not change stands as written in `current_design.md`.

## 0. Defects found

1. **The coins break either the identity or the budget.** A coin firing with probability p makes every eligible decision a chance node, with L = p·(Q(a0) − mean_b Q(b)) ≠ 0 where it does not fire. Counting L only where coins fired (the 800-forward budget) biases the residual; counting it everywhere costs about 4,000 forwards a game (1.1).
2. **Sampled expectations bias the gradient.** A single trait redraw or tail draw gives E[R̂²] = E[R²] + m²·Var_tail(C), rewarding a critic flat across rare outcomes (a leader nearly killed): the double-sampling bias the design says it avoids (1.2).
3. **"Each game is a noiseless equation"** holds only if the critic class contains V; otherwise a coin's contrast is measured against the critic's residual σ_r² (1.2, 1.3).
4. **Cross-fitting leaks.** A student trained on both folds' labels depends on fold-1 results through critic 1's labels of fold-2 states, so its slope on fold-1 coins is optimistic (1.3).
5. **P0 assumed M = 10.** Between offsets 0 and −1.5 a turn can hold several disagreements (each act continues it): M ≈ 20–40, and the standard error at 20,000 games is about 0.09, not 0.06.
6. **The plants leave the critic's output space** (P(win), P(loss), P(cap)): a ±0.30 bonus or a village tax cannot be represented.
7. **π_1^raw cannot work as defined.** Ten switches cannot restore offset 0's 10–20 early ends per side-game, and the critic prices the −1.5 continuation.
8. **Continuing and adopting are one rule,** so a real gain under 2 standard errors blocks the next round.
9. **Distilling into the trunk invalidates the critic,** since memories are written from the trunk's outputs.
10. **Determinism: right floor, wrong reason.** Logit near-ties are not value near-ties; a flip costs about s²/4, still under 0.002 of Var(z) at 2% of 450 decisions. Replay agreement within generation (the identity's floor) and generation-versus-eval agreement (the critic's estimand) are separate measurements.
11. **"A result-regression critic cannot pass P1" is false:** the coin is in the state, so result regression learns it. The plant certifies the path and its rate.
12. **Per-decile slopes** have a standard error near 0.4 at the planned sample; the audit tests the selected tail instead.

## 1. The algorithm

### 1.1 Generation: a pre-drawn coin schedule

At game start, each side-game draws a schedule S from a generator separate from the dice. S says which stop-window decisions carry a stop coin, and which decisions with at least two candidates carry an action coin. A decision is in the stop window when L ∈ [−1.5, +4.5]; the decode ends the turn iff L ≥ 1.5.

Given S, both players are deterministic except at scheduled coins and nature's events. With V_S(x) = E[z | x, S],

    z = V_S(x_0) + Σ_{k ∈ nature ∪ scheduled coins} L_k(V_S)

holds exactly. Unscheduled decisions carry no term.

The critic reads:
- the true state;
- both memories;
- the per-turn recruit-rejection sets;
- the matchup tag;
- the remaining schedule: coins left and eligible decisions until the next one, per side and type.

That tuple is the closed-loop Markov state. Each policy reads only its observation and its memory, and its next memory is a function of the present one and the next observation. So the identity holds by construction, without assuming that 64 slots summarise a history (research target 3). Summarising the schedule by counts leaves a floor that I estimate, without measurement, below 10⁻³.

**Coin laws.**
- A stop coin plays end or the best act, ½ each.
- An action coin draws from {a0, b1, b2, b3} with p(a0) = √3/(3+√3) = 0.37 and 0.21 for each alternative. This is Dunnett's square-root allocation, which minimises the summed variance of the contrasts against a0 ([Penn State STAT 503 §3.4](https://online.stat.psu.edu/stat503/book/export/html/644); near-optimal per [Hayter & Tamhane](https://users.iems.northwestern.edu/~ajit/papers/12%29%20Hayter-Tamhane%20JSPI.pdf)). A switch needs exactly those contrasts.

**Density.** Each coin side-game carries 4 stop and 8 action coins, three times the current design's rate. A coin's information costs nothing extra, and the look-ahead reads the critic at an empty remaining schedule, which prices the coin-free continuation. A quarter of games are coin-free, which keeps an empty schedule on-distribution at every turn. S0 checks that the cost of coins is linear in their density.

### 1.2 The critic objective, exact

R_t is as in the current design, with V_S. An event may have a sampled part: a fight's tail below 2%, a trait roll, an advancement. For such an event, two independent draws give two copies R̂¹_t and R̂²_t, and the loss is R̂¹_t·R̂²_t. Its expectation is E[R_t²] and its gradient is unbiased, by [Baird's double sampling](http://www.leemon.com/papers/1995b.pdf). The readout ρ_C is the held-out mean of R̂¹_0·R̂²_0 divided by Var(z). This costs about 1,200 forwards a game: 110 fights × 7, 46 recruits × 3 and 24 coins × 12.

**Stability (open question 2).** The games do not depend on C, so the objective is a fixed least-squares problem, convex in the linear case.
- No target is bootstrapped, so TD's divergence cannot occur.
- Every R_t is anchored on z, so residual-gradient's slow backward propagation does not arise either.
- What remains is gradient variance of about the current misfit. The semi-gradient initialisation lowers it to result regression's level, and it shrinks as C improves.

BSDE solvers use the same structure: one loss per time step, each containing the terminal condition, minimised jointly ([Kapllani & Teng 2020](https://arxiv.org/abs/2010.01319)). If S1's curve stalls, the fallback is a backward scheme ([Huré, Pham & Warin 2020](https://arxiv.org/abs/1902.01599)): fit the last turns, freeze them, step back. My search found no report of the joint form being unstable.

**Information.**
- **Coins.** A coin informs about decision differences at about E[Var_{c∼p} Q(x,c)]/σ_r² ≈ s²/σ_r². S1's 0.9 million coins give 160 to 370 χ², at s = 0.010 to 0.015 and σ_r² = 0.55.
- **Nature's events.** S1's 8 million events give about 125,000 χ² about chance differences, at E[L²] ≈ 0.0086 per fight.

The critic will therefore know what a hit is worth far better than what a hex is worth. Its move advantages exist only by generalisation.

### 1.3 From critic to player

**The gate (new).** On held-out coins, regress the corrected residual (excluding the coin's own term) on the critic's centred coin value, Q_C(x,c) − Σ_b p_b Q_C(x,b). The slope κ is fitted per type (stop, attack, move, recruit), with a standard error of σ_r/(√N·s), about 0.13 to 0.25. κ is the ceiling of what the fogged student can reach. A type whose κ has a 2-SE upper bound below 0.3 gets no switches.

**Students.** Student j trains only on critic j's look-ahead targets Â (read at an empty schedule) at the other fold's states, and is calibrated on that fold's coins, luck-corrected by critic j; no result it is calibrated on reaches it. The deployed head averages the two. Its target is E[Â_C | h], the information-state advantage, and its held-out R² against Â_C bounds the fog ceiling from below (research target 4).

**Stopping.** As in optimal stopping (Longstaff–Schwartz), end is compared with the best continuation: the best of the four acts by calibrated value, not the prior's best act. Action coins supply the contrasts of the other acts.

**Switch rule.** As in the current design (at most 10 switches per side-game, posterior at least 0.8), applied only to gated types.

**π_1^raw, redefined.** Inside the window, end iff α + γL + β·A_stop > 0; outside it, offsets 0 and −1.5 already agree. It differs from `own1`(−1.5) only where the learned rule disagrees with L ≥ 1.5, which is what the critic prices; those changes are not capped, and the audit is sized on their measured number. Known answer: the implied constant boundary L* = −α/γ lies in [0.5, 2.5]. `own1` at −2.5 and at −1.5 are level (−12.2 ± 12.3), and offset 0 is predicted to be worse.

### 1.4 Forecast, audit, incumbent

Ĝ_old, Ĝ_plug and the midpoint audit are as in the current design.
- **Audit variance.** It is 4Mσ_r² per game, about 20 times a match's. The audit is bought for its split by type and its exactness on the selected switches, not for precision.
- **Choice of t.** The midpoint minimises the score variance 1/(t(1−t)) and is exact to second order. Two-point Gauss–Legendre is exact to third order at 1.5 times the variance.
- **Why fresh games.** The record shows why the selected switches need fresh games: the best of eight decode knobs read +35.7 ± 12.3 on its selection games and +1.7 ± 12.3 on confirmation (kv15).

**Incumbent rule (new).** Round k+1 starts from π_{k+1} when audit and match agree within 2 combined standard errors and their inverse-variance mean gives P(gain > 0) ≥ 0.9. Otherwise it starts from π_k, with the round's games added to the critic. A strength claim is still a PURE match at +2 standard errors.

### 1.5 Rounds, distillation, compounding

**Round data mix:**
- half self-play with coins;
- a quarter coin-free;
- a quarter against `own1`, with coins on π_k's side;
- plus the audit games.

**Distillation.** By default it trains a deeper head on the frozen trunk, so the memories, and the critic that reads them, stay valid. The trunk is trained only if S5 shows the frozen features limit the head, and that step budgets a full refit of the critic.

**Compounding.** Policy iteration is Newton's method on Bellman's equation (Puterman & Brumelle 1979; [Bertsekas 2021](https://arxiv.org/abs/2108.10315)). Exact, it converges in a few iterations. Approximate, it settles within a band set by the evaluation error; that is Bertsekas & Tsitsiklis 1996, §6.2, recalled and not re-checked. Prediction: round gains fall to the critic's error on decision differences within two or three rounds. What restarts them is a better critic (more coins, more capacity, the new policy's data), not more rounds.

### 1.6 New invariants

- **I8:** on an enumerable duel with one scheduled coin, R_t(V_S) = 0 exactly.
- **I9:** with a forced sampled tail, the mean of R̂¹R̂² equals the enumerated E[R²]. A single-draw loss fails I9 by m²·Var_tail.
- **I10:** the look-ahead passes an empty schedule, and the critic's inputs include the rejection sets.
- **I11:** student j's training set holds no result from the fold it is calibrated on.

## 2. The regimen

**Prices:**
- a game: $2.5e-5 under generation numerics, or $1.3e-4 if S0(a2) fails;
- a critic forward: $3.6e-8;
- a critic training position: $1.2e-7 (estimated);
- a critic-data game: $1.6e-4 (generation, the memory recompute, and 1,200 forwards plus 100 backward positions per epoch, over two epochs);
- a match: $0.12.

| Stage | Cost | Predicted readout | Kill line |
|---|---|---|---|
| **S0 instruments** | $2.5 (×4 if a2 fails) | (a1) replay agreement within generation ≥ 98%; (a2) agreement between generation and eval numerics ≥ 98%. (b) Match A: E[z] for offset 0 in [−0.60, −0.25]; the L log gives the early ends per side-game M_A (prior 10–20), and \|E[z]\|/M_A ≈ 0.02 is the record's first per-decision advantage. (c) **P0** at Gauss–Legendre t ∈ {0.211, 0.789} over 40,000 games reproduces Match A within 2 combined standard errors (one is about 0.07 at M ≈ 30). (d) Coins on one side against `own1`, at 4 and at 12 per side-game, 20,000 games each: −0.012 to −0.047 and −0.035 to −0.14; the second within 2 standard errors of three times the first. | (a) below 95%: batch-invariant inference first. (c) outside: coin logging, side sign or propensity is broken; stop. (d) non-linear or below −0.2: 4 coins. |
| **S1 critic** | $13 | **P1**, on 20,000 modified games: (i) at a random turn a coin, visible to the critic, names a side that wins outright at the end with probability 0.3 (both draws logged). Mean s(c)·L_coin = 0.30 ± 0.02, and the error at 5,000 games is 2 ± 0.5 times that at 20,000. (ii) A +60-gold windfall at turn 4: L within 0.02 of luck-corrected forks at 1,000 positions; slope of truth on critic 1 ± 0.2. Then 50,000 `own1` games, with critics fitted at 10,000, 20,000 and 50,000. ρ_C(50k) in 0.55–0.75, at least 0.05 below the same games' cross-fitted linear baseline (about 0.82); the three points on a power law within 2 standard errors. κ prior: attack 0.6–1.0, recruit and stop 0.4–1.0, move 0.0–0.6. Also: s per type and σ_r². C(x_0;S) − C(x_0;∅) equals S0(d)'s cost within 2 standard errors. | P1 outside its bands: that path is broken. ρ_C not 0.05 below the baseline: fallback F. A type with κ's upper bound under 0.3 gets no switches; if all types do, F. Curve stalls: the backward scheme before more data. |
| **S2 operator** | $6 | **P2**: a village tax as a reward stream, r per own village at own end_turns, with r = 0.01 under a hidden bit (probability 0.3), else 0.002. An additive head learns it (zero in the real game), with R_t = z + Σ_{τ≥t} r_τ − C(x_t) − Σ_k L_k(C). The taxed-minus-untaxed critic advantage regresses on fork truth with slope 1 ± 0.1. The student's planted part reads 0.0044 per village-turn whatever the bit, while the critic's shows the gap (leak test). π_1 on the taxed game holds villages as forecast. Real game, per gated type: student R² in 0.2–0.6; β/κ in 0.5–1.0; a held-out gain over the nested model above 2 standard errors; one-step against turn closure decided by β. | P2 outside its bands: stop. β < 0.3 for every gated type: F on coin labels. |
| **S3 round 1** | $3 (+$2 under eval numerics) | Ĝ_old, Ĝ_plug and the per-type split are written down before any match. Prior, for sizing only: +10 to +40 Elo (at most 10 switches per side-game, 0.003–0.012 each). Known answers: Ĝ_plug on Match A's games reproduces Match A within 2 combined standard errors, and π_1^raw's L* lies in [0.5, 2.5]. The audit (20,000 games) lands within 2 combined standard errors of Ĝ_plug, and the match within 2 of the audit. | As in the current design, plus the incumbent rule. |
| **S4 rounds 2–3** | $11 each | The critic is warm-started on 50,000 π_k games; then students, audit, and matches, including against every earlier round. Each round's Ĝ_plug is at most the previous round's. | Two rounds under one match standard error: go to S5. |
| **S5 distillation** | $5 (head) or $13 (trunk with critic refit) | **P5**: `own1` at −2.5 distilled into the head: agreement ≥ 97%, level within 2 standard errors. Then π_k, level within 2 standard errors with no offset. The grown student's R² rises by ≥ 0.05, and the next forecast by ≥ one match standard error. | P5 outside: that path is broken. No rise: report. |

The planned path costs about $52. One S1 rerun brings it to about $65 of the $85. The $1,000 case is unchanged.

**Pre-box estimate of s (research target).** None is possible from disk: game records hold commands and seeds, not priors (`tools/game_record.py`). The first per-decision advantage will be S0(b)'s |E[z]|/M_A.

## 3. What is new, and the nearest work

Searched on 2026-10-19; finding nothing does not prove absence.

1. **The deviation schedule as part of the privileged critic's state.** Exploration becomes one chance event at game start, and the deployed continuation is priced at an empty schedule. Nearest: [NGU and Agent57](https://arxiv.org/abs/2002.06038), which condition one network's values on the exploration parameter and act with its exploitative member.
2. **A per-type privileged calibration gate (κ)**, read on randomised coins before any fogged student is trained. Nearest: micro-randomised trials (Boruvka et al.); [Baisero & Amato](https://arxiv.org/abs/2105.11674)'s history-state critics.
3. **An exact-expectation residual objective for a deep game critic**, double-sampled only where it must sample. Novelty claim 1 of the current design must now cite:
   - [MIVAT](https://www.ijcai.org/Abstract/09/326), which learns the luck-correcting value from past games (whether it minimises the estimator's variance is unverified);
   - Kapllani & Teng 2020;
   - Han, Jentzen & E 2018.

   What stays new is the transfer to a deep critic with exact chance enumeration and designed treatments.
4. **Dunnett allocation for coin laws, and the audit-plus-match incumbent rule.** Both tools are classical; their use here is new.

## 4. The weakest point

**Decision differences.** S1 gives the critic several hundred times more information on chance differences than on decision differences (125,000 against 160–370 χ²). Move advantages therefore rest on generalisation. If κ_move fails, round 1 can change only attacks, recruits and stops, and its prior straddles the +25 adoption bar. Two levers remain:
- more coins, which buy information at the cost of dilution;
- more games, whose cost is linear: 200,000 games, about $32, quadruple χ².

**The fog ceiling.** The record's selection gain fell from +0.049 on the true state to −0.002 from the mover's view. That record is suspect, but it warns that β/κ may be small.

**No first-principles value for ρ_C.**

**Still for the user:** whether the coin regression, used to calibrate and to audit, falls under the killed outcome policy gradient.

## Decision record (added)

- Rejected: per-decision Bernoulli coins, single-draw expectations in the loss, one student on both folds' labels, trunk distillation by default, per-decile kill lines; the reasons are defects 1, 2, 4, 9 and 12.
- Rejected: [Stochastic MuZero](https://openreview.net/forum?id=X6D9bAHhBQ1)'s learned chance model in the look-ahead, because the cost is the critic's forwards (about 20 per decision), while the exact core forks in 0.02 ms and enumerates a fight in 0.06–0.11 ms on CPU; a learned model adds error and saves nothing.
