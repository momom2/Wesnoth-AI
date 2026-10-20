# Critique of round4_harden.md (round 4)

**Verdict: repairable.** The algebra of §1.1–1.2 holds and the record's counts reproduce. Around them: P1 passes with the corrections dropped, the event-class split is not identified, the horizon rung tests the critic against itself, the pilot's asymmetry argument runs backwards, and the window terms lose the convergence argument the design relied on.

## Strongest part

§1.1–1.2. I re-derived (1) and (2): each L_k is centred given its past, so every cross term vanishes (provided sampled positions do not depend on later dice), and the multistep loss's expectation splits into C's terms plus E[η²], so its game-path fixed point does not depend on C′. The MDBDP citation checks: Theorem 3.3's DBDP bound carries N in front of the error sum, Theorem 3.1's MDBDP bound does not ([Germain, Pham & Warin](https://arxiv.org/abs/2006.01496)). §1.3 reproduces from `prior_gaps.games.jsonl`: densities 309/230/138; 91, 24.7, 16.1, 2.64 at w = 0.35; 150, 43.2, 27.8, 8.64 at 0.7; Ḡ 1.473; Σ P·G 163.8; `prior_gaps.json` names `parity3.pt`. §1.5's tail arithmetic checks (0.0044 switches; 6.9 at mean 0.0135, +33 Elo).

## 1. P1 passes with the event terms broken

The plant is a level effect visible in every later state, so plain result regression learns it too. At pass 0 the seven columns cut the future variance only from about 0.7 to 0.6. Dropping every correction raises the contrast's SE by √(0.7/0.6) ≈ 1.08, flipping their sign by about 1.3; against the derived SE, the mean and χ² tests together pass the first with probability about 0.9, the second about 0.8. Also, σ_η² cannot be "read from C′", since η involves V: the computable E[(y − C′(x_{k+1}))²] exceeds it by E[(V − C′)²], loosening the test. And "at most" is no bound for a network that shrinks small-sample contrasts; correct code can fail on that bias.

Repair: plant a variance, not a level. Book a ±a lottery with a logged law into z at some fights and coins; the trained critic's held-out ρ_C must fall by the planted share within its SE. Dropped corrections leave the lottery in the residual, sign-flipped ones quadruple it.

## 2. The event-class split is not identified

A class's share of ρ_C·Var(z) is E[L(e)²] = E[L(C)²] − 2E[L(C)·z] + E[L(V)²]. One result per game identifies E[L(V)²] only in total. Identified instead: each class's removed variance, Var(z − C(x_0) − Σ_{other classes} L(C)) − Var(R̂_0) = E[L(V)²] − E[L(e)²]. Report that.

## 3. The horizon rung tests C′ against itself

On every path I(b) = −Σ r_s(C′) over turns 3–4: the slope reads C′'s own Bellman residuals against Â, not the window's distance from the truth. A bias persisting from τ2 to τ4 reads 0. I19's second half cannot fail: at C′ = V, I ≡ 0 over any interval, so an interval bug passes. The SE is optimistic: the three pairs share I(a0) and contrasts against a0 share its value (about 1.5× in variance, SE near 0.15), and Var(Â) is the critic's spread, below s² if κ_move is 0–0.6. "Below −0.3 at 2 SE" then fires only below about −0.6; a positive slope has no action; −0.3 to 0 is not derived. Repair: on the duel, plant a C′ whose sibling contrast decays by a known share from τ2 to τ4 and require the slope to recover it. Novelty: a forecast-revision efficiency test (Nordhaus 1987, *Rev. Econ. Stat.* 69:667; [Patton & Timmermann 2012](https://ideas.repec.org/a/taf/jnlbes/v30y2012i1p1-17.html), some of whose tests need no target data).

## 4. The pilot's asymmetry is backwards; the opening gains nothing

By (1) at t = 0, ρ_C·Var(z) = E[e(x_0)²] + Σ_k E[Var e]: every event once. That is the multistep weighting, not λ = 1's n_k, so ρ_C favours the multistep loss; "counts double" is wrong, and a λ = 1 win would not refute §1.1.

With C′ ≈ C, the multistep multiplier keeps λ = 1's future term (η is C′'s future residuals); what goes is the past, weighted (n_j/n_k)², about a third of its variance. The noise ratio per unit of signal is about 1 + (k/3)/(K − k): 1 in the opening, 1.33 at mid-game, 2 at three quarters. The opening, where η² is largest, gains nothing, so §1.1's stall applies to the fix there. The floor "independent of the critic being trained" is λ = 1's own floor lagged one pass, and it is not zero: the 1.8% numerics flips are unlogged randomness. The pass count is unstated; with two epochs, pass 1's corrector is pass 0's critic.

## 5. Windows lose their convergence argument

The frozen cross-fold window target is fitted value iteration across passes at off-policy siblings. `current_design.md` §5 point 3 relied on residual gradients converging where semi-gradient TD may not; "within a pass a regression" does not cover the passes ([Tsitsiklis & Van Roy 1997](https://doi.org/10.1109/9.580874)). Independence from C′ holds for game terms only: I18 must exclude windows, and nothing tests pass-to-pass stability. Round 3's cross-fit leak stands: from pass 2, C′ was trained on targets corrected by this fold's critic (O(1/n); three folds remove it).

Placement by p*(G; w) leaves no window at G ≥ w: 119 of 269 multi-legal decisions per side-game at w = 0.7, 157 at 0.45. Switches are not restricted to the band, and "where switches occur" is unverified: floor the placement or restrict switches. Likewise β's linearity is fitted on tie-heavy coins and used in the tail X > 0.84·s·√(1−q).

## 6. The band's anchor and range

The control sampled "from its full distribution over ~350 legal actions" (`raw_argmax_control_20260904.md`); the 164 nats count the top 8. Truncation biases λ upward, against saturation, so "lower bound" fails. At λ = 0.01 the cap gives w ≈ 0.38, not 0.45 (0.01 × (4.13 + 2.94) = 0.071 at 0.45). The cost model has no intercept: 0.001 per deviation adds 0.043 at w = 0.7. S0(d)'s two widths can fit one; say what it does to the taper.

## 7. Propensities and numerics

I12's tolerance hides a bias: the identity needs the law used, and with batch-dependent arithmetic a recomputed G is not it. The 1.8% flips sit at ties, where band coins sit, and can swap a0 and a1. Log p and the pair at generation, use them everywhere, make I12 exact. Likewise `current_design.md` samples positions "by gradient weight", which breaks (1)'s cross terms without importance weights.

## 8. Costs

The pilot's $0.6 omits forwards: the λ = 1 arm alone evaluates every event's children, about 1,600 forwards a game, $0.8 per epoch over 10,000 games at $5·10⁻⁸; with the other arm, about $3. The multistep loss also needs C on all children of each sampled event (about 660 forwards a game per epoch) beside C′'s refresh: about $3.3 over S1 by my count, beyond the "+$0.8" for backward positions. The plan still fits, with a margin near $12.

## 9. Smaller

- [Scherrer 2010](https://arxiv.org/abs/1011.4362): Bellman-residual minimisation "enjoys a performance guarantee", TD(0) does not. The λ objective is a full-gradient residual, so the horizon factor, not "no guarantee", is the reason to reject λ < 1.
- Novelty item 2 is the doubly robust per-step control variate with a separately fitted value ([Jiang & Li 2016](https://arxiv.org/abs/1511.03722)) moved to chance nodes, MDBDP's linear case. A frozen mean-zero corrector not moving the fixed point is what makes it a control variate.
