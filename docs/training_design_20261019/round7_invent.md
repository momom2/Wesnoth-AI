# Round 7, invent stance: antithetic chance siblings and a known-switch calibration

Tags as in `current_design.md`: [M] measured (record and script named), [D] derived, [U] unknown, [C] a choice whose consequences are derived. Scripts are in `analysis/`.

## Summary

1. The current design's core stays: near-tie coins, exact labels, a ridge head on frozen tokens, adoption by a PURE paired match. This memo adds two components. ACS is one more label-only control variate, of the class ruling (b) asks the user about, so the ruling's list would name it beside the columns and DPI. KSC uses only matches and coin labels the design already has.
2. **Antithetic chance siblings (ACS).** Each fight is replayed on a fork of its pre-fight state with every strike's draw reflected, and the network prices the reflected child with one forward. The realized child's price is free: it is the next decision's forward. Half their difference has mean zero exactly, so it is a valid control variate for every coin label. It keeps (1 − ρ)/2 of what enumerating all outcomes keeps, at least ½ for prices monotone in the dice (Harris's inequality), for one forward per fight instead of 11.86.
3. This revives the corrector class round 5 priced out. Enumerating every child costs 3.90× the forwards and pays only if the price explains more than 0.511 of Var(z) beyond the columns. ACS on the riskier half of fights costs 1.122× and pays above 0.121 (`round7_invent_antithetic.py`). Whether `own1`'s reading of a board explains that much is unknown; S0 measures it for $0.15, before any generation code.
4. **Known-switch calibration (KSC).** Coins on one seat at p = ½, restricted to a fixed switch set, forecast that switch's match effect exactly up to third-order interactions. This is the midpoint rule on Owen's multilinear extension. On the attack-offset switch set, whose effect the records put at −78 ± 16 Elo, it gives the label path a known answer of real size. It also measures, before S1, whether summed first-order labels predict a match.
5. Added cost is at most $3.59, inside the design's $9.81 reserve. An admitted ACS saves at least one batch ($2.51) per round.

## 1. What the records say (this round's scripts)

- **Reflection, stated fight model** (`round7_invent_antithetic.py` Part A: 1,500 random fights, alternating strikes, 30–70% hit chances). ρ ≤ 0 in every fight, as Harris predicts. Captured share κ: 0.791 for the hp margin, 0.670 for kills, 0.735 for hp fraction plus a half-unit kill bonus. For one strike with an additive price, κ is 0.714 at c = 0.3 or 0.7, 0.833 at 0.4 or 0.6, and 1 at 0.5 [D].
- **Concentration** (Part B: 6,130 enumerated laws of `own1` self-play fights, the counter-weapon subsample, not random). The 30% and 50% of fights with the largest proxy-price variance under the law carry 0.644 and 0.839 of its summed variance [M].
- **Lines** (Parts C–D). Evaluating a share f of fights costs m = 1 + f·109.7/449.1. With κ 0.735 and the measured shares, ACS beats the cell plus columns (r·m < 0.688) iff the enumerated price explains more than 0.099, 0.121 or 0.184 of Var(z) beyond the columns, at f = 0.3, 0.5 or 1. It saves a batch per round (r·m ≤ 0.602, the design's DPI line) iff that share exceeds 0.269, 0.246 or 0.278 [D].
- **Switch-set sizes** (`round7_invent_known_switch.py`, `parity3`'s 100 recorded side-games, scaled to `own1`'s 224.6 decisions). An attack offset of +1 changes the argmax at 4.8 to 31.8 decisions per side-game, an offset of +0.35 at 0.2 to 16.3 [M, bounds: the record does not say which top-eight slot holds the attack].
- **Against ACS.** `obs8`'s value head ranks candidate turns at 0.22, against the HP margin's 0.40 (seed §3). Hand prices explain 0.152 of Var(z); conditioning on the board adds 0.042 [M, `round6_invent_conditioning.py`]. No record reads a network price at fight children.

## 2. The algorithm

### 2.1 Unchanged

Coin generation, the label T_i = s_i·ω_i·R_i, the ridge student with antisymmetric type terms, the sign rule, progressive validation, the interim and adoption matches, fallback E and S4's distillation follow `current_design.md` §3. What is learned is also unchanged: the switch head, the labels' control-variate coefficients, and per round the distilled network. ACS adds a control variate to R_i. KSC adds readings, not training.

### 2.2 Antithetic chance siblings

**Construction.**
- Fight k starts from state x_k. Its strikes draw u32 numbers from the command's seeded MT19937, and a strike hits iff r mod 100 < cth (`rust/wesnoth_core/src/combat.rs`, `perform_hit`).
- The reflection Φ maps r ↦ r − 2(r mod 100) + 99 within each full block of 100 values, and x ↦ 95 − x on the top partial block of 96. Φ is an involution, so Φ(r) has the law of r.
- On a fork (0.02 ms), the attack is replayed with every draw reflected, through a `StrikeRng` wrapper beside the existing `ScriptedRng`. This gives o′_k.
- The price is h_θ(child) = θ₀·V(child) + θ₁ᵀP·g(child): `own1`'s value and a 16-dimensional projection P, fixed in S0 [C], of its global-token output g.

    L_k = s_k · (h_θ(o_k) − h_θ(o′_k)) / 2
    R_i = z − b(cell) − βᵀS − γ·s_i·V_i − Σ_k L_k(θ̂)   (plus DPI's terms if admitted)

**Exactness.** Given x_k, o_k and o′_k have the same law: the outcome is a fixed function of the draws, and Φ preserves their law. h_θ and s_k are fixed once θ̂ is fitted outside the batch. So E[L_k | before k] = 0. L_k is orthogonal to every coin innovation D_i − p_i: by the tower property for coins before the fight, and because it is pre-coin for coins after it. The design's proof (§3.2) carries over, and the price's quality enters only the variance.

**Capture.** Let e_k = h(o_k) − E h. If z depends on fight k through e_k, the optimal coefficient on L_k removes Var(e_k)(1 − ρ_k)/2, against Var(e_k) for enumeration, with ρ_k = Corr(h(o_k), h(o′_k)).
- Let y_j = −(r_j mod 100) on the attacker's strikes and +(r_j mod 100) on the defender's, so that a larger y_j favours the attacker.
- A price monotone in the attacker's favour then increases in every y_j, and the reflected fight's price decreases in every y_j.
- The draws are independent, so Harris's inequality gives ρ_k ≤ 0, hence κ ≥ ½.
- Kills, drain, slow, poison and advancement heals keep this monotonicity for a price that rewards hit points and kills. A learned price need not be monotone, so S0 measures κ directly.

**Cost.**
- **Realized child.** Its price comes from the next decision's forward: the same state and the same memory, since the memory is written from the decision's trunk outputs before the action (`wesnoth_ai/memory.py`). Where the next decision is not forwarded (a single legal action), that child is requested too.
- **Reflected child.** It needs one forward, and its memory write is discarded.
- **Selection.** Only fights whose proxy-price variance under the enumerated law passes a cutoff fixed in S0 are evaluated, at 0.06–0.11 ms of CPU each to compute. The selection is pre-fight, so exactness holds.

**Fit.** θ follows MIVAT's criterion ([White & Bowling 2009](https://www.ijcai.org/Abstract/09/326)): a ridge regression of z − b on per-game sums Σ_k s_k(f(o_k) − f(o′_k))/2, with f = (V, Pg).
- It is closed-form in 17 coefficients and cross-fitted per batch, as DPI is. It is refitted each round on that round's coin games.
- The dice have known laws, so the fit cannot be confounded.
- Overfitting costs r·d/N ≈ 0.688 × 24/18,800 ≈ 0.001 of r [D].

**Admission.** ACS is built iff r_ACS·m ≤ 0.602, with the paired upper 2-SE bound below 0.688/m, read on `own1` self-play and pooled. This is the design's DPI rule: a batch saved per round. If both qualify, the stack is admitted only if it saves one more batch than the better alone. Between the "pays" line and the batch line, ACS would buy less than a batch per round for a new code path, so it is not built.

### 2.3 Known-switch calibration

**Statement.**
- Fix a switch set S, a rule on histories ("an attack offset of +1 changes the argmax here").
- On one seat, a coin at p = t plays the switched action at each decision of S, and the other seat plays `own1` exactly. Let J(t) be the coined seat's expected result.
- J is Owen's multilinear extension (Owen 1972, *Management Science* 18(5); not re-read) restricted to the diagonal of S, a polynomial in t.
- The labels summed over S in a side-game estimate J′(t) without bias. Each estimates its decision's advantage at reach probability t, so their sum is the directional derivative.
- The match measures M = J(1) − J(0) = ∫₀¹ J′. The midpoint rule gives

    M − J′(½) = J‴(ξ)/24  for some ξ in (0, 1)

which is zero when interactions among three or more switches vanish. One-sided coins at p = ½ on S therefore forecast its match exactly to second order. The design's generator coins both seats at 0.25 over the whole band, so its forecast differs from M by second-order terms, among them the opponent's coins (`current_design.md` §3.5). Round 6 found no affordable way to read those terms.

**Arms.**
- **H (S0).** One-sided p = ½ coins on S₁ (attack offset +1), through the per-seat hook S0(b) already builds, with the coined seat alternating by game. SE 8 Elo costs $0.62–4.14 for |S₁| of 4.8–31.8 [D]. It is capped at $2 [C], for an SE of at most 11.5 Elo.
- **M₁ (S0).** S0(b)'s attack known answer, raised from 6,400 to 32,000 PURE paired games: SE 1.79 Elo for $0.80 [D]. Predicted −78 ± 16 Elo [M via D, unselected records].
- **T (S1, batch 1).** A hand-set student plays the runner-up wherever the band's top is a non-attack and the runner-up an attack (set S_T). It runs through the student's own per-seat hook in a $0.80 match M_T. Batch 1's labels summed over S_T give the design's own forecast F_D at no cost. The sign is predicted negative (attack promotion costs −0.179 ± 0.043 z per nat, `round6_provenance_attack_pairs.out`); the size is not.

**What the readings mean.**
- **F_H of the wrong sign on a seat, or outside −78 ± 2√(16² + SE_H²) (36–39 Elo).** The label path mislabels real effects; P-coin tests only its arithmetic. Fix it before S1.
- **|F_H − M₁| within 2√(SE_H² + 1.79²).** First-order label sums predict a match on a real set of tens of switches.
- **Beyond that line, with P-coin, I2 and I9 passing.** Interactions among three or more switches matter at this scale, so S2's forecast is reported as indicative.
- **F_D − M_T within 2 SE.** The design's coin point forecasts an in-band switch, so Ĝ_pf predicts S2.
- **F_D − M_T beyond 2 SE.** S2's prediction is Ĝ_pf scaled by M_T/F_D, marked indicative because S_T is not the student's set.

Adoption stays by match in every case: KSC decides what a forecast is worth, never what is adopted.

### 2.4 Why more signal per dollar

- **Where label information comes from.** A coin's information is p(1 − p)/(E[R²]·m). The design lowers E[R²] only through hand columns and DPI, a bilinear price in 13 rule numbers. Of the correctors considered in rounds 1–7, ACS is the only one that reads the network's own valuation of the post-fight board at a cost independent of the outcome support. The share it must explain (0.10–0.28) is a fifth to a half of what enumeration needs (0.511).
- **Alternatives.**
  - Forks carry 1/40 of a coin game's information [M, round 5].
  - Branching continuations give at most 1.125× [D, round 6].
  - Randomized telescoping truncation (Rhee–Glynn) with keep probability q gives (1 − a + a/q)(1 − f + fq), with a the tail's share of label noise and f its share of cost. This product, convex in q, falls below 1 only if a < f [D]. The same value spent as an ACS price removes the tail's luck instead of dropping the tail.
- **KSC.** It turns a forecast miss at S2 from a surprise into a reading whose meaning was measured first.

## 3. Regimen

Prices are the pool's: $2.51e-5 a one-network game, replay at 2,440 forwards a second, $0.42 an hour [M].

| Stage | Cost [D] | Readout and prediction | Decision per outcome |
|---|---|---|---|
| **S0(a)+** | design $0.42; + $0.099 (a reflected child per fight, 2.08 M forwards); + $0.050 (enumeration on 800 `own1` games) | E_full (the enumerated price's share beyond the columns), κ_real, r_ACS at f = 0.3, 0.5 and 1, r_DPI, the stack, \|S₁\| and \|S_T\|. E is not predicted: no record reads it. κ_real ≥ ½ is predicted for monotone prices. | **r_ACS·m ≤ 0.602 at the best f:** ACS enters S1, and the power script reruns. **Otherwise:** not built. **E_full < 0.099:** child-evaluated prices from `own1`'s readings are closed. **κ_real < ½:** the price is not monotone, and r_ACS is used as measured. |
| **S0(b)+** | design $0.24; + $0.64 (M₁ to 32,000 games); + H ≤ $2.00 | M₁ −78 ± 16; F_H = M₁ to second order, negative per seat | §2.3. Label path broken: fix before S1. |
| **S0(c)+** | design $0.05 | ACS overhead per game (fork, reflected fight, side request) | **Three rounds fit iff price × (1 + overhead) × m × B/8 ≤ $2.92e-5,** with B the batches per round that ACS's measured r·m buys [D: the design's line with the batch cost scaled]. **Otherwise:** a smaller f. |
| **S1** | design $20.88, minus $2.51 a batch ACS saves; + M_T $0.80 | Design's readouts; I-A1 to I-A4 on generated fights in batch 1; F_D against M_T | Design's rules; F_D − M_T as in §2.3 |
| **S2–S5** | as design | θ refitted per round; r_ACS read in every batch from the logs | Design's rules. ACS is dropped for any round whose cross-fitted r·m fails the batch line. |

**Budget [D].**
- The additions cost at most $3.59 ($0.099 + $0.050 + $0.64 + $2.00 + $0.80), leaving at least $6.22 of the $9.81 reserve.
- An admitted ACS returns at least $2.51 a round, $7.53 over three rounds.

**Beyond `own1`.** The $1,000 case is argued only after two adopted rounds, as in the design.
- ACS is also how a privileged critic would enter cheaply. As a price at the reflected child it is training-only, cannot bias a label, and costs one forward per selected fight instead of 11.86. Its admission line is S0's, re-read.
- Nothing measured links `own1` to human ratings, and nothing rests on it.

## 4. Invariants and checks for the new code

- **I-A1.** Φ is a bijection (exhaustive block check), and reflected outcomes pass a χ² test against the enumerated law on 10⁵ fights.
- **I-A2.** On scripted fights, the joint law of (o, o′) matches Part A's pair DP, so the reflection is per strike and in order.
- **I-A3.** The realized child's price equals the next decision's value output to numerics tolerance.
- **I-A4.** L_k has mean zero per event over about 2 M fights, uncorrelated with every D_i − p_i and with pre-coin quantities.
- **Negative control.** The non-bijective map r ↦ min(r, Φ(r)), which favours hits, must fail I-A1 and I-A4.
- **KSC.** H's coins pass I1 and I9. M₁ and M_T exercise the per-seat hook with a known sign. P-coin, I2 and I9 are read before H.

**No regression plant for θ.** OLS recovers a plant lying in its own span by algebra (round 6, for DPI). ACS's bias rests only on centring, which I-A1 and I-A4 test exactly.

## 5. What is new, and the nearest work

Searched 2026-10-20; finding nothing does not prove absence.

1. **ACS.**
   - **Enumerating chance outcomes.** AIVAT evaluates a value at every chance outcome ([Burch et al. 2018](https://arxiv.org/abs/1612.06915)), and DeepStack used it with its own network ([Moravčík et al. 2017](https://arxiv.org/abs/1701.01724)). Backgammon bots enumerate rolls with their evaluator ([GNU Backgammon](https://www.gnu.org/software/gnubg/manual/html_node/Rollouts-in-GNU-Backgammon.html); [Montgomery](https://bkgm.com/articles/GOL/Feb00/var.htm)).
   - **Variance reduction in rollouts.** [Veness, Lanctot & Bowling 2011](https://proceedings.neurips.cc/paper_files/paper/2011/file/d736bb10d83a904aefc1d6ce93dc54b8-Paper.pdf) pair whole rollouts antithetically, and their control variates need known means (read).
   - **Antithetic variates.** They help for monotone integrands (Hammersley & Morton 1956; not re-read).
   - **New here:** one reflected-dice sibling priced by the network, with the realized sibling free, as an exactly centred chance-node price. Also new: its capture (1 − ρ)/2 ≥ ½, and pre-fight selection by law variance.
2. **KSC.**
   - **Owen's integral.** Quadrature over the inclusion probability is known in Shapley estimation ([Okhrati & Lipani 2021](https://arxiv.org/abs/2010.12082); [Mitchell et al.](https://arxiv.org/abs/2104.12199)).
   - **Excursion against policy effects.** The gap is known in micro-randomized trials ([Qian 2025](https://arxiv.org/html/2502.13500v2)).
   - **Off-policy estimators against A/B tests.** They have been compared ([Gilotte et al. 2018](https://arxiv.org/abs/1801.07030)).
   - **New here:** coin sums read as Owen's integrand for a deterministic policy switch, so that one-sided p = ½ coins forecast its match to second order. A matched real effect then serves as the label path's known answer.

## 6. Weakest point

**ACS rests on E, which is unknown, and the record leans against it.**
- `obs8`'s value head ranked turns below the HP margin.
- Observation-only critics did no better than material.
- Hand prices stop at 0.152.

ACS captures part of what a price explains and creates no explanation. To save a batch, the child price must explain about 0.25 of Var(z) beyond the columns, 1.6 times what all hand columns explain together. Below that, ACS is not built and $0.15 is spent. The reading still settles the corrector class round 5 deferred, for `own1`'s readings.

**KSC's set is not the student's.** Attack promotions are few and large (up to 0.076 z each [M, upper bound]); the student's switches are many and small. Its reading is evidence about the label path, not a calibration of S2.

**Neither component touches the near-tie spread v,** the design's main unknown. A null round 1 stays possible at the design's price.

## 7. Decision record

**Rejected:**
- **Telescoping truncation.** It needs a < f (§2.4), and a value function is better spent as a centred price.
- **Rotation instead of reflection.** For a linear price of one uniform it keeps 0.75 of the variance where reflection keeps 1 [D].
- **Antithetic trait draws.** Trait pools are unordered, so Harris bounds nothing, and recruits would add 45.8 forwards a game [M].
- **A trapezoid between round 1's and round 2's coin points.** It keeps the background-coin term p(1_B − 1_S)ᵀH1_S; a $0.80 match reads the target directly.

**Deferred:**
- A two-point Gauss arm (one-sided coins at t = ½ ± 1/(2√3), exact for J of degree ≤ 4), if H misses with P-coin passing.
- Privileged prices at the reflected child (the $1,000 case).

## Provenance

| Number | Value | Tag, source |
|---|---|---|
| Pool price, one-network game | $2.51e-5 | M, `round6_provenance_prices.py` |
| Replay speed; box price | 2,440 forwards/s; $0.42/h | M, BACKLOG.md via `round5_simplify_critique_extra.py`; seed §1 |
| Fights, decisions per `own1` self-play game | 109.7; 449.1 | M, `round5_synthesis_outcomes.out` |
| Walked decisive games; attacks; turns | 18,800; 110.6; 21.12 | M, `round7_invent_antithetic.py` Part E (`match_luck_20261019/rows.jsonl`) |
| Outcome support | 11.86 (6,130 laws, counter-weapon subsample) | M, Part B; `round5_synthesis_outcomes.out` |
| Enumeration cost multiple; its line | 3.90; E > 0.511 | D, Part C: 1 + 109.7 × 11.86/449.1; 0.688 − 0.688/m |
| κ, strike reflection | 0.791, 0.670, 0.735; ρ ≤ 0 in all 1,500 fights | D under a stated fight model, Part A |
| κ, one strike, additive price | 0.714–1.0 | D, closed form, Part A |
| Harris bound | κ ≥ ½ for prices monotone in each strike | D, Harris's inequality |
| Variance in the top 30% / 50% of fights | 0.644 / 0.839 (mixed proxy) | M, Part B (non-random subsample) |
| ACS cost multiples | 1.073, 1.122, 1.244 at f = 0.3, 0.5, 1 | D, Part C |
| ACS lines | pays at E > 0.099, 0.121, 0.184; saves a batch at E > 0.269, 0.246, 0.278 | D, Part D (κ 0.735 × share) |
| Label residual; Var(z) | 0.688; 1.0004 | M, `round6_provenance_cells.py`; `round5_simplify_variance.json` |
| Batch line | 7 batches iff r ≤ 0.602 ($2.51 a round) | D, `round6_synthesis_power.out` §5 |
| Hand prices' share; board conditioning | 0.152; 0.042 | M, `round6_invent_conditioning.py`, `round6_invent_critique_checks.py` |
| Value head against HP margin, turn ranking | 0.22 against 0.40 | M, seed §3 (`docs/turn_value_prereg_20260925.md`) |
| ACS fit penalty at S0 | about 0.001 of r | D, r·d/N with d = 24 (17 + 7 columns), N = 18,800 |
| S0 ACS replay; enumeration | $0.099; $0.050 | D, Part E |
| \|S₁\|; \|S_0.35\| per `own1` side-game | 4.8–31.8; 0.2–16.3 | M (bounds), `round7_invent_known_switch.py` |
| Knob first stage | +2.35 attacks per side-game per nat | M, `round6_provenance_attack_pairs.out` |
| Attack known answer | −78.1 ± 15.9 Elo (unselected); −62.2 ± 15.0 (all games) | M via D, `round6_provenance_critique_attack_stage1.out`; `round7_invent_known_switch.out` |
| Per added attack | at most 0.076 z in magnitude | M (upper bound), `round6_provenance_attack_pairs.out` |
| Label variance at p = ½; at p = 0.25 | 2.753; 3.67 | D, 0.688 × 1.0004/(p(1 − p)) |
| H's cost for SE 8 Elo; cap | $0.62–4.14; $2 → SE ≤ 11.5 Elo | D, `round7_invent_known_switch.out`; cap C |
| 32,000-game PURE paired match | SE 1.79 Elo, $0.80 | D, `current_design.md` §2 |
| Elo per unit of z | 347.4 | D, 800/ln 10 |
| Midpoint remainder | J‴(ξ)/24 | D, midpoint rule on J′ |
| Truncation condition | pays iff a < f | D, convexity of (1 − a + a/q)(1 − f + fq) in q |
| Rotation's capture, linear price | 0.75 | D, closed form |
| Recruits per game | 45.8 | M, `match_luck_20261019/readout.txt` (round 6) |
| Plan; reserve; additions | $73.58; $9.81; at most $3.59, leaving at least $6.22 | D, `round6_synthesis_power.out` §6, plus §3 here |
| Projection size 16, selected share f, cap $2, Φ's partial-block map | — | C; S0 sets f from the measured concentration |
| E_full, κ_real, r_ACS, r of the stack | — | U, S0(a); ACS's admission waits for them |
| \|S₁\| and \|S_T\| for `own1` | — | U, S0(a) census; H's size waits for it |
| M₁, F_H, M_T, F_D | — | U, S0(b) and S1 batch 1; they decide what a forecast is worth, nothing else |
| ACS overhead per game | — | U, S0(c) |
| v, μ, second-order loss, distillation closure, κ_T, human level | — | U, as in `current_design.md` |
