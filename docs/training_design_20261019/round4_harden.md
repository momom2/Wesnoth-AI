# Round 4, hardening: a multistep critic loss, a band sized from the record, horizon and cross-critic checks

Stance: keep the current design's frame and repair it. Kept: the exact luck-and-coin identity, the privileged critic, decision windows, Longstaff–Schwartz stopping, the fusion-free fogged student, corrected matches and the forecast ladder. Anything this file does not change stands as written in `current_design.md`. Every comparison is still one decision followed by play that re-decides after every roll, and the network picks every action.

## 0. Defects found

1. **The λ family does not share a target** except when V lies in the critic class; under misspecification λ < 1 is oblique, and λ = 1 weights each event by its position (§1.1).
2. **The global loss is noisy by construction:** each event's gradient is multiplied by R_t, which carries the level error and every other event. The pilot also selects λ by ρ_C, λ = 1's own t = 0 term.
3. **The band rests on assumptions the record answers:** the tie density falls with G, the runner-up's cost has an anchor, Ḡ is 1.47, not 2 (§1.3).
4. **Record label.** The side-games in `critic_step1_20261009/prior_gaps*` are `parity3`'s decisions (player `cand64`, checkpoint `parity3.pt`, +82.2 ± 12.6 over `parity2` in `imitation_anneal_20261003/box/cand64_vs_ref.fit.json`), not `parity2`'s.
5. **Windows** have no placement rule in S1 (no student exists yet) and no horizon check.
6. **The stop rule is scored by the critic that chose it.**
7. **The action prior ignores the posterior threshold;** its low corner is about 0 Elo, not 1 (§1.5).

## 1. The algorithm: what changes

### 1.1 What the residual loss is

Let e = C − V. On every path z = V(x_t) + Σ_{k≥t} L_k(V), so R_t = −[e(x_t) + Σ_{k≥t} L_k(e)]. Each L_k(e) has mean zero given the past under the core's exact law, and so does ∇L_k(C) = ∇C(x_k^{o*}) − Σ_o p_k(o)∇C(x_k^o). All cross terms vanish:

    E[R_t²]     = E[e(x_t)²] + Σ_{k≥t} E[Var_{o~p_k} e(x_k^o)]              (1)
    E[R_t ∇R_t] = E[e(x_t) ∇C(x_t)] + Σ_{k≥t} E[L_k(e) ∇L_k(C)]           (2)

with E[L_k(e)∇L_k(C)] = ½∇E[Var_{p_k} e] for fixed V. Equation (1) is the standard variance formula for martingale control variates. Three consequences follow.

- **λ = 1 is a projection with a tilt.** The loss is Σ_t E[R_t²] = Σ_t E[e(x_t)²] + Σ_k n_k E[Var e(x_k^o)], where n_k counts the sampled states at or before event k. Its minimiser is a weighted L2 projection, but late fights weigh up to the whole game's state count and early ones almost nothing. The look-ahead uses the critic uniformly across the game.
- **λ < 1 is an oblique projection.** Summation by parts gives R^λ_t = −[e(x_t) − (1−λ)Σ_j λ^{j−1} e(x_{b_j}) + Σ_k λ^{n(k)−n(t)} L_k(e)], with b_j the j-th side-turn boundary after t. The square couples errors at different times, so the minimiser is an oblique projection ([Scherrer 2010](https://arxiv.org/abs/1011.4362)), with no performance guarantee for the TD-like end. Its error bound carries the horizon factor that one-step backward schemes have (below).
- **The stall.** In the stochastic form of (2), event k's term is multiplied by the realised R_t. By (1), R_t's variance holds the level error and all events from t onward. With about 160 events, each event's signal is a small share of its multiplier's variance.

### 1.2 The multistep loss

Replace the λ family with the multistep form:

    event k (chance or coin):  ℓ_k = (y_{k+1} − C(x_k) − L_k(C))²,   y_{k+1} = z − Σ_{j>k} L_j(C′)
    decision state t:          ℓ_t = (y_t − C(x_t))²,                 y_t     = z − Σ_{j≥t} L_j(C′)

C′ is the other fold's critic from the previous pass, frozen. Pass 0 uses the seven linear columns fitted on the other fold. Write η = y_{k+1} − V(x_{k+1}), which has mean zero given the past through k+1 for any C′ that did not see the game. Then y_{k+1} − C(x_k) − L_k(C) = −e(x_k) − L_k(e) + η, so

    E[ℓ_k] = E[e(x_k)²] + E[Var_{p_k} e(x_k^o)] + E[η²],   E[η²] = Σ_{j>k} E[Var_{p_j}(V − C′)(x_j^o)].

The same terms appear as in (1), with one weight per event, chosen by the use. The multiplier's noise covers the future only, under the frozen corrector, and it falls to zero as C′ → V. That is the zero floor, now independent of the critic being trained. Each pass is a plain regression whose fixed point does not depend on C′, so no λ, no warm-start pass and no Baird interpolation are needed. Put another way, this is λ = 1 differentiated through its starting state and first event only, with the rest frozen and cross-fitted.

It has the structure of the multistep deep backward scheme MDBDP ([Germain, Pham & Warin 2022](https://arxiv.org/abs/2006.01496), eq. 2.14). There the target is the terminal payoff minus the later steps' fitted Z·ΔW terms. Their error bound has no factor N on the value-approximation errors, while the one-step DBDP bound has one (their Remark 3.1, as I read it). With a terminal-only payoff and exactly centred terms, no backward ordering is needed.

**Mechanics.**
- A sampled event costs three backward positions: the pre-event state, the realised child, and one child drawn from p_k (unbiased for Σ_o p_k∇C(x_k^o), since the multiplier does not depend on it). 200 per game per epoch instead of 100 adds $0.8 to S1.
- Sampled tails (fights below 2%, traits, advancement) keep two independent draws.
- Events weigh uniformly; inverse-variance weights would favour late events again.
- Windows take the same frozen form: sibling b regresses on C′(x_τ^b) − Σ_{k in window} L_k(C′), n-step TD with a frozen cross-fold target. Within a pass it is a regression and cannot diverge; its bias is C′'s error at τ (§1.4).

**Pilot** (first 10,000 games, training compute only, about $0.6): the multistep loss against the λ = 1 joint residual, compared by paired held-out ρ_C. ρ_C favours λ = 1, since it is λ = 1's t = 0 term, so a multistep win counts double.

**Readout split.** By (1), ρ_C·Var(z) splits by event class: setup, fights, traits, coins. S1 reports the split. Decision siblings enter only through coins, so ρ_C cannot see the move weakness; κ and the horizon rung can.

**P1's rate, derived.** At the planted result coin, the learned contrast's error at N games is at most √(σ_η²/(N p(1−p))). σ_η² is E[η²] at the coin's event, read from C′ before the fit (about 0.6 at pass 0). It is an upper bound because later states' level terms also see the plant. The test: the mean within 2 SE of the planted 0.30, and the four disjoint fits' χ² on 3 degrees of freedom below 11.3 against that SE.

### 1.3 The band, sized from the record

All figures come from `prior_gaps.games.jsonl`: `parity3`, 100 side-games, 269 multi-legal decisions each. They were counted for this file.

- **Density per nat per side-game.** 309 for G in [0, 0.1), 230 in [0.2, 0.3), 138 in [0.6, 0.7).
- **Band counts per side-game, by width w:**

| w | decisions in band | expected deviations | Σp(1−p) | Σp·G |
|---|---|---|---|---|
| 0.35 | 91 | 25 | 16 | 2.6 nats |
| 0.70 | 150 | 43 | 28 | 8.6 nats |

- **The runner-up's cost λ.** The seed at argmax beat itself sampling at temperature 1, 37-0-3 (+412 ± 97 Elo, z ≈ 0.85; `docs/raw_argmax_control_20260904.md`). Over `parity3`'s top-8 margins, temperature 1 spends Σ P·G ≈ 164 nats per side-game, so λ ≈ 0.85/164 ≈ 0.005 per nat. This is an order of magnitude only: 40 games, the seed's weights on `parity3`'s margins, saturation near z = 0.85, and the mass beyond 8 candidates ignored. Saturation makes it a lower bound.
- **Predicted band costs.** At λ = 0.005–0.01 per nat, the band costs 0.013–0.026 per side-game at w = 0.35 and 0.043–0.086 at w = 0.7. Weak coins take about 0.015 of the 0.06 cap: two deviations per side-game at Ḡ = 1.47 nats. So w ends in [0.45, 0.7], fixed by S0(d), whose 20,000-game arms have SE about 0.006.
- **Information per unit shift.** Band over weak is 2.1·Ḡ/w ≈ 4.4 at w = 0.7 and 6.9 at 0.45. That is before the falling density, which lowers the mean G in band and raises the ratio.
- **The taper and Li & Owen.** [Li & Owen](https://arxiv.org/abs/2202.12511)'s two-level optimum maximises a parametric two-line information matrix. The taper maximises separable per-decision information, which is concave in each p. Both are right for their objectives, so they do not conflict. κ and β are slopes on the critic's contrast X, with information proportional to Σp(1−p)X². The band does not target that, so κ's SE in S1 is computed from X's spread inside the band, which S1 measures.

### 1.4 Windows: placement and the horizon rung

**Placement.** In S1, windows are drawn at decisions with probability proportional to p*(G; w), the band's own propensity: where the decode is undecided, which is where switches occur. From round 2, they sit on the previous switch score's band.

**Horizon rung.** 10% of windows are extended to four side-turns. For each sibling pair, take the turn-3–4 increment

    I(b) − I(a0),   I(b) = C′(x_{τ4}^b) − Σ_{turns 3–4} L_k(C′) − C′(x_{τ2}^b)

and regress it on the root contrast Â_C(x, b). If C′ is Bellman-consistent, E[I | x_{τ2}] = 0 whatever the dice. Â_C is fixed at the root, so the slope is zero under that null, without errors-in-variables bias. A negative slope means the critic's two-side-turn reading overstates decisions, because their consequences fade by turn 4.

The SE is about √(0.065/(30,000 × 1.5·10⁻⁴)) ≈ 0.12, using 10,000 extended windows with 3 pairs each, pair noise 0.065 and Var(Â) ≈ s². Regressing D_4 on D_2 instead fails: D_2's dice noise (0.065) against s² ≈ 1.5·10⁻⁴ attenuates the slope by a factor of about 400.

### 1.5 Cross-critic scoring, and the action prior

**Stop.** The Longstaff–Schwartz rule fitted on critic j's payoffs is scored on held-out paths by critic 3−j's payoffs. This is the double estimator of [van Hasselt 2010](https://papers.nips.cc/paper/3964-double-q-learning). The gap between own-critic and cross-critic scores is the optimism from selecting on one critic's sampling error. The rule's gain over c* is read cross-critic. Shared blind spots remain, and the stop-only match catches those.

Actions need no counterpart. β_avg is the slope of E[T | X], which is exactly the shrinkage that corrects selection on top values when E[T | X] is linear.

**Action prior with the threshold.** Posterior ≥ 0.8 requires X > 0.84·s·√(1−q).
- Low corner (s = 0.01, q = 0.2, m/s = 1, N = 100): 0.005 switches per side-game pass, about 0 Elo.
- High corner (s = 0.015, q = 0.4, m/s = 0.5, N = 200): 6.9 switches at mean 0.0136, so 0.094 in z, about +33 Elo.
- The cap of 10 does not bind.
- Round 1's action factor is therefore 0 to +33. The stop is its likeliest gain.

### 1.6 Invariants added

- **I17.** On the enumerable duel, with C a constant and a random network, sampled means of R_t² and R_t∇R_t equal (1) and (2). The λ = 1 expected gradient equals the multistep one with weights n_k.
- **I18.** The multistep expected gradient is identical for C′ ∈ {constant, random network, V}, and moves for a C′ fitted on the same sampled games.
- **I19.** A reward paid in side-turn 3 after the root enters four-side-turn windows and not two-side-turn ones. With the duel's exact V as C′, the increment's slope is 0.
- **I12** tolerates the 1.8% numerics flips when it recomputes G.

## 2. Regimen

Prices as `current_design.md` §3. Stages not listed change only as noted.

| Stage | Cost | Predicted readout | Kill line |
|---|---|---|---|
| **S−1 replay** | $0.5 | `own1`'s recorded games, memories recomputed. (a) Agreement ≥ 98% (`parity3` 98.2%). (b) f(0) 200–400 per nat per side-game (`parity3` 309); density at 0.6–0.7 nats about half of f(0) (`parity3` 138); Ḡ of candidates 2–4 1.2–1.8 (`parity3` 1.47). The runner-up's unit, action kind and hex distance logged, with the same-unit-adjacent-hex share. (c) Stop-window and per-type counts. | (a) under 95%: kernels first. f(0) under 30, or duplicates over 70% after re-pairing: no band. |
| **S0 instruments** | $4 | (a), (b), (c) P-coin, (e) as current; P-coin's SE 0.046 at the record's Σp(1−p). (d) Band cost 0.013–0.026 at w = 0.35 and 0.043–0.086 at w = 0.7 per side-game; weak coins at 3 and 9 deviations 0.01–0.04 and 0.03–0.12. λ read to about ±15%. | As current. w is the largest width with total shift ≤ 0.06 (predicted 0.45–0.7). |
| **S1 critic** | $22.5 (+$1.75 conditional) | P1 with the derived rate (§1.2). 50,000 `own1` coin games; 100,000 windows placed by p*(G; w), 10,000 of them extended. Pilot: multistep minus λ = 1 in held-out ρ_C, −0.05 to 0. Fit: ρ_C 0.55–0.75 at 10k, 20k and 50k games, ≥ 0.05 below the same games' seven-column ratio, split by event class. Horizon slope −0.3 to 0. κ as current; band over weak per unit shift 4–7. C(x_0; S) − C(x_0; ∅) equals S0's cost within 2 SE. | P1 outside: path broken. λ = 1 better by 2 paired SE: it replaces the multistep loss, and §1.1's variance argument is reported wrong. ρ_C or κ as current: fallback F. Horizon slope below −0.3 at 2 SE: the second half's windows run four side-turns (+$1.75). Band over weak under 1: no band from round 2. |
| **S2 operators** | $7 | As current, plus: the stop rule over c* +5 to +25 Elo, read cross-critic; the own-minus-cross gap reported (assumed 10–40% of the own-critic gain). | As current. Cross-critic gain over c* under 2 SE: round 1 stops at c*. |
| **S3 round 1** | $5 | As current. Priors: stop-only +5 to +35; actions-only 0 to +33 (§1.5); interaction within ±10. | As current. |
| **S4 rounds 2–3** | $11 each | As current. Prior 0.2–0.8 of the previous gain, still not derived. Tesauro & Galperin's 5-fold error reduction ([1996](https://arxiv.org/abs/2501.05407)) is one step with full rollouts, and I found no per-round decay figure for iterated one-step improvement in a long stochastic game. | As current. |
| **S5 distillation** | $5 | As current. | As current. |

**Planted-signal checks, by path:**
- P-coin: coin logging and propensities.
- P1: the multistep critic's event terms, at the derived rate.
- P2: decision-moved advantages through game states only and window states only, plus the fog leak.
- P3: Longstaff–Schwartz.
- P5: distillation.
- I19: window horizon bookkeeping.

**Budget.**
- The planned path (S−1 to S3, round 2, S5): about $55.5.
- Round 3 brings it to $66.5.
- The remaining $18.5 covers the kernels ($7) with a partial S1 rerun, or a full S1 rerun in place of round 3.

## 3. What is new, and the nearest work

Searched 2026-10-19; finding nothing does not prove absence. Items 1–5 of `current_design.md` §4 stand. This round adds:

1. **The level-and-sibling decomposition of the identity loss**, (1)–(2). It shows that λ = 1 tilts its weight toward late events and that λ < 1 is oblique. (1) itself is textbook for martingale control variates; using it to read a critic objective is modest.
2. **The multistep identity loss for a privileged game critic**: one term per chance or coin event, with a cross-fold frozen corrector whose choice cannot move the fixed point. Nearest: MDBDP (Gobet & Turkedjiev's scheme with networks), [Off-policy DAE](https://arxiv.org/abs/2402.12874) (the global return decomposition), and frozen target networks.
3. **The horizon rung**: an increment regressed on the root contrast, zero under Bellman consistency, testing a decision window's horizon. Nearest: [STEVE](https://arxiv.org/abs/1807.01675), which weights horizons by ensemble variance and does not test them.
4. **The band sized from the record**, with the runner-up's cost anchored to the argmax-against-sampling match.
5. Not claimed: cross-critic scoring (van Hasselt's double estimator), and window placement by propensity.

## 4. The weakest point

**Moves.**
- Their contrasts reach the critic only through band and weak coins and through two-side-turn windows bootstrapped on C′.
- By (1), ρ_C carries no decision-sibling term outside coins, so S1's main readout cannot see this weakness.
- Only κ_move and the horizon rung can. κ_move's SE depends on the spread of the critic's own move contrasts, which is unmeasured.
- The band's value still rests on the prior's errors being as large at ties as elsewhere. 73% of ties are moves, and the runner-up's identity is unlogged until S−1.

**The multistep loss is a variance repair.** If the record's critics failed for capacity or features rather than gradient noise, it changes nothing, and S1 dies at the same line as before. ρ_C still has no first-principles value.

**For the user:** κ, β, fallback F and P-coin regress outcomes on coins to calibrate, gate and check. Whether that falls under the killed "outcome policy gradient" is the user's call. The critic's coin terms are value regression and do not move the policy by themselves.

## 5. Decision record

**Rejected this round:**
- **λ < 1 for game states:** oblique under misspecification, with a horizon factor in its error bound.
- **The λ = 1 joint residual as default:** a late-event tilt and a multiplier carrying the whole game's error; it stays as the pilot's comparison arm.
- **The separate warm-start pass:** the multistep loss's pass 0 without its sibling terms.
- **Inverse-variance event weights:** they reproduce the tilt.
- **D_4 on D_2 as the horizon test:** attenuated about 400-fold.
- **A pre-box tie census from records:** they hold commands and seeds, not margins or runner-ups (`tools/game_record.py`).

**Deferred** to the $1,000 case: an exploitability probe, one round trained against a frozen π_k (about $11), whose gain bounds π_k's one-step exploitability from below.

**Standing:** the rejections of rounds 1–3 (`current_design.md` §6).
