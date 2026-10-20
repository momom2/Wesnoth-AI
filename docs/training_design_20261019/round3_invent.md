# Round 3 (invent): margin experiments

## Summary

The current design's weakest point is decision information: S1 carries 140 to 320 χ² on decision differences against about 125,000 on chance. This proposal keeps the design's frame:
- the privileged zero-floor critic and the exact luck identity;
- the fogged student;
- Longstaff–Schwartz stopping;
- the arcsine audit.

It adds four parts:
1. **Band coins.** Coin propensities become a known function of the decode's own margin: fair at a prior tie, tapering linearly to zero. The taper is derived as the most information about the prior's errors per unit of state shift.
2. **Margin certificates.** Fair coins in a narrow band around any learned threshold measure, from outcomes, the first-order gain of moving that threshold. They test the stop rule's pricing of over-continued states without the critic, and a quadrature of three certificates must reproduce Match A.
3. **Luck heads.** Each fight's outcome values come from one critic forward and a small head, which halves the cost of critic data.
4. **Reply-read payoffs.** The stop payoff is read after the opponent's deterministic reply.

A pre-box replay of `own1`'s recorded games sizes the bands before anything is rented.

## 1. The constraint, in numbers

**Setup.** At a decision, let a0 and a1 be the top two joint actions in decode order, with the end_turn offset applied. Define:
- G = ℓ(a0) − ℓ(a1) ≥ 0, the log-prior margin;
- Δ = Q(a1) − Q(a0) = −m(G) + η, the true contrast;
- m(G) = E[−Δ | G], the runner-up's mean cost;
- η, the prior's error, with variance σ_η²(G).

With exchangeable labels at a tie, m(0) = 0.

**What a coin buys and costs.** A coin that deviates with probability p gives p(1−p)σ_η²/σ_a² of information about the prior's errors, where σ_a² ≈ 0.55 is S1's coin-onward residual variance. It costs p·m(G) of state shift. The design caps the shift at 0.06 per side-game (S0(d)), about 6 deviations.

**The weak coin.** The design's weak coin (ε = 0.2 over the next three candidates) has E[Var_p Q] = 0.16 m̄² + 0.19 s², which is the design's 0.35 s² at m̄ = s.
- The first term is the prior's known ranking; only the second measures its errors.
- It pays 0.2·m̄ per coin decision.

**Where the gain is.** The gain available at a decision, E[max(η − m(G), 0)], peaks where m(G) ≈ 0, near prior ties. So the cheapest coins are also where switches will happen.

## 2. The algorithm

The design stands as written (`current_design.md` §2):
- argmax self-play with coins;
- a critic on the closed-loop state, trained on the double-sampled joint residual;
- a Longstaff–Schwartz stop head;
- a fogged student of Â_C, calibrated by κ and β on the other fold's coins;
- switches, forecasts and the arcsine audit;
- head-only distillation.

The changes follow.

### 2.1 Band coins

**The program.** Maximise Σ_d p_d(1 − p_d)σ_η²(G_d) subject to Σ_d p_d m(G_d) ≤ B and 0 ≤ p_d ≤ ½. The KKT conditions give

    p*(G) = ½ · [1 − μ m(G) / σ_η²(G)]⁺

With m(G) ≈ λG and σ_η² constant, p*(G) = ½(1 − G/w)⁺: fair at a tie, linear to zero at w = σ_η²/(μλ), with the cap B setting w.

**Per coin.** Averaged over the taper (G locally uniform on [0, w]), a band coin carries σ_η²/6 at cost λw/12, that is 2σ_η²/(λw) per unit of shift. A weak coin carries 0.94σ_η²/(λḠ), with Ḡ the mean margin of candidates 2 to 4. The ratio is 2.1·Ḡ/w, so a band coin is 4 to 8 times as informative per unit of shift at Ḡ/w = 2 to 4.

**In total.** The total gain is limited by how many decisions lie near a tie. With f near-tie decisions per nat of margin per side-game, the cap gives w = √(12B/(fλ)) and information f·w/6.
- At the design's cap (λ ≈ 0.01/Ḡ, B = 0.06) and f = 15 to 60, Ḡ = 2, this is 1.4 to 2.7 times the design's η-information.
- Weak coins stay at a third of their density for candidates 3 and 4 and for large margins. This costs a little of that gain.

**The stop band.** Its margin is L − c (the rule ends iff L ≥ c), with the same taper. The weak continuation coins stay at ε = 0.1 over [c, c + 3], because the Longstaff–Schwartz payoffs need states several actions past π's stop.

**Exactness.** Every decision with p > 0 is a chance node with term

    L_d = C(x^c) − [p C(x^{a1}) + (1 − p) C(x^{a0})]

- Each side draws its band width at game start from {0, w/2, w}, a quarter of games at 0; the critic reads it with the weak-coin schedule.
- The propensity is a fixed function of a margin the actor computes from the state and memory the critic reads. So V_w is the closed-loop value, and the identity stays exact.
- The no-coin reading C(·; 0) is checked by κ's interaction with the width.

κ and β are fitted on band and weak coins separately. If they differ by more than 2 combined standard errors (SE), the switch rule uses the band slope only inside the band. Known propensities keep the estimates unbiased, but calibration at ties need not transfer to large margins; margin-based sampling is the classic source of sampling bias in active learning ([Dasgupta & Hsu 2008](https://www.cs.columbia.edu/~djhsu/papers/hier.pdf)).

### 2.2 Margin certificates

Every rule in the decode thresholds a score S(h):
- the stop head ends iff D(h) ≤ 0;
- the switch rule switches iff its posterior score exceeds a threshold.

Shift the rule by δ·φ(h). Against the deterministic `own1` the performance-difference lemma is exact, and only decisions with S ∈ (−δφ, 0] change, so

    dJ/dδ |₀ = E[ Σ_d δ_Dirac(S_d) · φ(h_d) · T_d ],    T_d = Q(x_d, act) − Q(x_d, not act) at S_d = 0

This is the boundary density times the jump, [Dong & Lewbel](https://ideas.repec.org/a/tpr/restat/v97y2015i5p1081-1092.html)'s marginal threshold treatment effect, extended to a sequential argmax rule and a direction φ. (I read their abstract only.) Fair coins in |S| < b/2 estimate it:

    Ĵ′_φ = (1/b) Σ_{|S_d| < b/2} φ_d · 4(c_d − ½) · R_d

Here R_d is the coin-onward residual (critic-corrected, or the seven linear luck columns before S1).
- The bias is O(b²).
- The variance per game is about (4/b²)·n_b·φ̄²·σ², with n_b the decisions in the band.
- The directions are φ = 1 (offset) and φ = S (slope).

A locally optimal rule reads zero in both.

**Its use.** The design's weak point 3 is that the critic prices states past π's stop. If it underprices them, the stop head stops early and Ĵ′₁ > 0 at its own boundary, measured from outcomes alone.

**Known answer (P-C).** `own1` plays at offsets 0, −0.75 and −1.5 (c = 0, 0.75, 1.5), 20,000 games each, with stop-band coins at b = 0.5.
- **Prediction.** The Simpson integral (1.5/6)(J′(0) + 4J′(0.75) + J′(1.5)) reproduces Match A within 2 combined SE.
- **Size.** Each certificate's SE is about 0.07 per nat with luck columns (n_b ≈ 7, σ² ≈ 0.8). Simpson's SE is 0.25·√18·0.07 ≈ 0.07, so 2 combined SE ≈ 0.15.
- **What a miss means.** A factor-2 error in density weighting or band width, or a side sign error, fails at Match A's predicted size, 0.25 to 0.60.

### 2.3 The natural boundary in recorded games

Coin-free argmax games have a sharp boundary at every margin. The record's `own1` games therefore already hold a regression-discontinuity design at L = c. With a uniform kernel, the boundary local-linear jump has variance 8σ²/N (N decisions per side in the bandwidth): four times fair coins on the same decisions (my derivation).

Its use is sizing. Replaying the records through the actor, with memories recomputed, logs every margin, which gives f, Ḡ and the stop band's density. It also compares the replayed argmax with the recorded actions, the design's S0(a2), on games already played.

### 2.4 Luck heads

The identity needs each event term centred given the past, not built from C: L_k(D) = D(x_k, o_real) − Σ_o p_k(o) D(x_k, o) has mean zero for any D.
- **The head.** A fight's outcome values come from one critic forward on the pre-fight state and a two-layer head over the two units' tokens and the core's outcome key (HP, slow, poison, petrify, type after advancement; XP and death follow from the key, `tools/combat_outcomes.py`).
- **Exactness.** Every outcome is enumerated, so fights need no sampled tail and no double sampling. Traits are handled the same way.
- **What keeps full forwards.** Decision siblings do: a move changes positions, an outcome changes two units.
- **Cost.** Critic forwards fall from about 1,600 to about 500 a game, and a critic-data game from $2.4e-4 to about $1.2e-4.
- **Risk.** The head imagines the post-fight state, which may raise σ_a². S1's A/B decides.

### 2.5 Reply-read payoffs

At argmax, the opponent's play from x∘end to its first attack is deterministic except for recruit traits. So u(x) = V(x∘end) = E[V(N(x∘end)^o)] exactly: N runs the opponent's actor on its own fogged observation and memory to its first attack, with traits enumerated, and o is that attack's outcome.

Reading C there prices an over-continued position by the reply it draws, with the chance law computed rather than learned. It costs about 10 forwards per path state, about $1 in S2. It is a payoff variant, chosen in S2.

### 2.6 New invariants

- **I10:** the logged propensity equals p*(G; w) recomputed from the logged margins.
- **I11:** R_t(V_w) = 0 on an enumerable duel with one band decision and one fight.
- **I12:** with D replaced by full forwards C(T(x, o)), the head path's residuals equal the forward path's exactly.
- **I13:** on a synthetic score with known jump T(s) and density f, Ĵ′ recovers f(0)T(0) within its SE and the uniform-kernel RDD recovers T(0). Permuting the coins nulls both.
- **I14:** a reply segment contains no own decision and no coin. Where the opponent's first action is an attack, the reply-read payoff equals that attack's enumerated value.

## 3. Regimen

Prices are the design's, plus:
- a critic-data game, about $1.2e-4 with heads ($2.4e-4 without);
- a critic-corrected certificate game, about $7e-5;
- a plain game, $2.5e-5 under generation numerics ($1.3e-4 under eval).

| Stage | Cost | Predicted readout | Kill line |
|---|---|---|---|
| **S−1 replay** | $0.5 | `own1`'s recorded games (about 6,000 side-games at −1.5, 800 at −2.5) replayed. (a) Replayed argmax equals the recorded action on ≥ 98% of decisions. (b) f, prior 15–60 per nat per side-game, unmeasured; Ḡ 1–3 nats. (c) RDD stop jumps at c = 1.5 and 2.5 within 2 SE (about 0.015) of zero; reported, not gated. | (a) below 95%: the design's numerics branch applies. f below 10: band coins are dropped and the design's coins stand. |
| **S0** | $5 | The design's (a)–(d). (e) **P-C**: J′(0) > J′(0.75) > J′(1.5), with J′(1.5) within 2 SE of 0, and the Simpson integral on Match A. (f) Band cost, 20,000 games: m(G) rising from about 0, cost per coin decision at most a third of a weak coin's. This sets w at the cap. | (e) outside: the certificate path is broken and certificates leave the plan. (f) band cost at least half a weak coin's: band coins are dropped. |
| **S1** | $15 | The design's S1 on about 100,000 games: band coins, weak coins at a third, heads; ρ_C read at 10k, 20k, 50k and 100k. (i) Heads A/B on 5,000 games: ρ_C(heads) ≤ ρ_C(forwards) + 0.01. (ii) **P-M** slope 1, SE about 0.1. (iii) Decision χ² 2.5–5 times the design's (about 350–1,600); κ's SE per type 0.05–0.11. (iv) κ_band and κ_weak within 2 combined SE. | (i) failed: heads off, 50,000 games as in the design. P-M outside: the band path is broken. Otherwise the design's kill lines. |
| **S2** | $9 | The design's S2 (P2, the two stop known answers, actions). Longstaff–Schwartz is fitted with both payoffs; reply-read is kept if its known-answer errors, in SE units, sum lower (predicted: it wins at −2.5). The fitted rule's certificate on 20,000 games (SE about 0.05 per nat) reads within 2 SE of 0 in offset and slope. | Offset certificate beyond 2 SE: the critic misprices over-continued states, and the stop factor enters round 1 at the constant c*, whose own certificate must read 0. |
| **S3** | $9 | The design's S3: forecasts written first, then audit and matches. The switch rule's certificate (20,000 games against `own1`, SE about 0.05) reads within 2 SE of 0. | Beyond 2 SE: round 2's posterior threshold moves from 0.8 to 0.7 if positive, 0.9 if negative. Pre-registered; flagged for the user's outcome-regression ruling. |
| **S4** | $13 per round | The design's refit at heads' prices, then S2–S3, with certificates every round. | The design's. |
| **S5** | $5 | The design's: P5, then head distillation. | The design's. |

**P-M (band path).** At every band decision a centred parity stream is booked: r·(even(a_c) − [p·even(a1) + (1 − p)·even(a0)]), with r = 0.01 and even(·) meaning the action ends on a hex with x + y even.
- The stream enters only the band estimator's outcome, never the play or the critic.
- Each term has mean zero given the past, so the slope on (c − p)·r·(even(a1) − even(a0)) is exactly 1.
- With the stream alone the slope is exactly 1. With z added, its SE is √(0.55 / Σ p(1−p)Δ²_plant) ≈ 0.1 at S1's size.

The design's P0, P1, P2 and P5 stand.

**Budget.** The planned path (S−1 to S3, one S4 round, S5) costs about $57. A second S4 round or an S1 rerun adds $13–15. Both fit in $85 only under generation numerics; eval numerics add about $15 and drop the second round.

**The $1,000 case** is argued only after two rounds whose matches land within 2 SE of their audits and whose certificates read zero.
- **What it buys:** 40 to 70 rounds at about $13, distillation every three rounds, a larger critic, and the band width re-measured from each round's margins.
- **Multi-decision operators:** the deferred ones enter only if J′(t) bends.
- **Prediction:** the measured per-round decay, extrapolated with its error. No measured link from `own1` to human ratings exists.

## 4. What is new, and the nearest work

Searched 2026-10-19; absence is not proven.

1. **Band coins.** The policy network's own margins set known-propensity coins, tapered by the information-per-shift program, and those coins are chance nodes of an exact luck identity.
   - Nearest work: tie-breaker designs ([Owen & Varian 2020](https://arxiv.org/abs/1808.07563); [Li & Owen 2023](https://arxiv.org/abs/2202.12511), whose optimum is piecewise constant under a two-line regression objective; [Morrison & Owen](https://arxiv.org/abs/2202.10030)); local-randomization regression discontinuity ([Cattaneo, Frandsen & Titiunik 2015](https://www.doi.org/10.1515/JCI-2013-0010)); uncertainty sampling ([Lewis & Gale 1994](https://arxiv.org/abs/cmp-lg/9407020)); [Xu & Wager](https://ideas.repec.org/a/inm/ormnsc/v70y2024i10p7024-7041.html); and the design's weak coins.
   - I found tie-breaker designs used for scholarships and loyalty perks, never on an agent's action margins.
   - The linear taper comes from a different objective. It is derived here and not checked against Li & Owen's theorem.
2. **Margin certificates.** A first-order optimality test of a learned sequential argmax rule's thresholds, in chosen directions, from designed band randomization, with a quadrature known answer against a measured match.
   - Nearest work: Dong & Lewbel; [Zhang, Ben-Michael & Imai](https://arxiv.org/abs/2208.13323) (safe cutoff learning under regression discontinuity); the jump terms of smoothed perturbation analysis.
   - I found no use in reinforcement learning.
3. **Reply-read payoffs.** A post-decision state, extended through the opponent's deterministic network play to its next chance event. Nearest: afterstates and truncated rollouts. The claim is modest.
4. **Not claimed.** Luck heads: [AIVAT](https://arxiv.org/abs/1612.06915) accepts any value function, and [Stochastic MuZero](https://openreview.net/forum?id=X6D9bAHhBQ1) predicts per-outcome quantities from afterstates. Regression discontinuity on recorded games is textbook.

## 5. Weakest point

The band's gain assumes two things:
- the prior's errors are as large at ties as elsewhere;
- near-tie decisions are dense.

The record contradicts the first once: the largest lever on record, the −1.5 offset, corrected a confident error spread over margins up to 1.5 nats, not a tie. That is why the width is set by the cap rather than kept small, and why the weak coins stay. If S−1 finds few near-ties, the band adds little, and learning that costs $0.5.

Further risks:
- **Certificates are coarse.** At about 0.05 per nat they catch a threshold that is off by about 0.1 per nat, not fine miscalibration.
- **Luck heads may raise σ_a²** and undo their saving.
- **Band coins label only the runner-up.** Candidates 3 and 4 rest on the thinned weak coins.
- **The critic** remains the single point of failure, and nothing here predicts ρ_C.
- **For the user:** S3's threshold move is outcome-driven tuning of one scalar and needs the same ruling as β.

## 6. Decision record

- **Rejected: natural regression discontinuity as a main-loop source.** It has four times the variance of coins on the same decisions and rests on continuity. It is kept for pre-box sizing.
- **Rejected: certificates as a training gradient for the stop head.** That would be outcome policy gradient on a threshold. They gate only.
- **Rejected: value-rank coupled twins** (dice made comonotone by critic value). To beat coins they need a twin correlation above about 0.94, which one decision's butterfly effect at argmax makes unlikely.
- **Rejected: posterior twins by resimulating hidden traits.** Their acceptance rate is unknown and the student's targets are cheap. They are kept as a possible diagnostic of the fog ceiling.
