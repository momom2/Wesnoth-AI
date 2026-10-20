# Round 3 synthesis

## Verdict

The design changed in substance this round. The frame stands: the exact luck-and-coin identity, the privileged critic, Longstaff–Schwartz stopping and the fusion-free fogged student. Four changes go beyond wording:

- **A flaw in the instrument fixed.** The arcsine audit was the round's measurement, at about a thirtieth of a corrected match's information per dollar: per-game variance π²·12·0.4 ≈ 47 against ρ_C·Var(z) ≈ 0.65 (harden, checked by its critique). Corrected matches replace it. From round 2 the round's gain is one direct match of π_{k+1} against π_k, 12,800 games at SE 2.5 Elo, which keeps the advantage the critique said two matches against `own1` would lose. S3 and S4 change accordingly.
- **A new core mechanism in the critic.** The identity decomposes exactly: R_t = −Σ_{s≥t} r_s, where r_s = C(x_s) − E[C(x_{s+1}) | x_s] uses the enumerated chance and coin laws. Two consequences follow:
  - the critic's objective becomes a λ family with one zero set, from round 2's global loss (λ = 1) to the local Bellman residual, answering the stall risk of global-in-time losses without moving the target;
  - **decision windows**, the same residuals on two-side-turn forks at the look-ahead's siblings, are a new data source aimed at the design's named weakest point, decision differences.
- **Coins changed form.** Band coins at the decode's own margin come from invent, sized by the record. Recorded `parity2` games show about 310 near-ties per nat per side-game, against invent's prior of 15–60.
- **Stage order and known answers.** A replay stage S−1 comes first. Match C (`own1` at −2.5 against −1.5) moves into S0, because the "−0.035 ± 0.035" it was to reproduce exists in no record. P-coin, a planted stream on the training coin path, replaces the arcsine P0.

## What changed, and why

### From harden (critique: repairable)

**Kept:**
- corrected matches as the instrument, with C frozen beforehand. [Kim & Sandholm (2026)](https://arxiv.org/abs/2605.14261), which I checked, show that a value function tuned on the evaluation games fakes precision;
- run-length stop coins;
- β_avg, the calibration of the averaged students;
- privileged twins for the stop head and the student, as fog diagnostics;
- the warm start's cross-fitted targets;
- Ĝ_plug read on the match's own games with results sealed;
- the forecast's calibration on the forecast's opponent, which is now the self-play coins;
- the fresh −2.5 match.

**Repaired as the critique asked:**
- **The default critic.** The residual objective is the default, not the warm start (critique 1). The warm start's minimiser is result regression's, and the chance-sibling information enters only through the residual loss. The warm start is a single pass whose corrector, the seven linear columns, is fitted on the other fold, so the cross-fit leak the critique found at the second refresh cannot occur.
- **P3, the stop plant.** The gold-parity plant could not be learned (gold enters as gold/500 and is cast to bf16) and was monotone. It is replaced by a non-monotone plant on the turn number, which the encoder's global features carry: −0.01, −0.01, then +0.05. A one-step greedy fit fails it. The stop head now reads its depth past the decode's stop, a function of its own history. The mean-zero fight term, which no regression can see, becomes the variance invariant I16.
- **P0** is replaced by P-coin on the training path.
- **The dual bound** is dropped (decision record).
- **The stop sizing.** The offset sweep measured `relset`, not `terrain`: −2.5 and −1.5 tied against offset 0, p 0.801 and 0.789 ± 0.014 (`docs/endturn_offset_sweep_prereg_20260919.md`). `parity3` at 64 slots gained 65 Elo from −1.5 to −2.5 (`docs/memory_in_play_parity3_prereg_20261009.md`). Match C's band, z in [−0.03, +0.11], spans both.
- **The opponent-mix citations** (Robinson; Winnicki & Srikant) are dropped; the mix stays as a hedge.
- **Rounds can now fail.** The direct match has SE 2.5, the forecast's lower bound is written first, and the gate to the $1,000 case needs lower bounds above zero.
- **Run-length visit rates.** The end-of-turn state at depth d is visited at 1/4 per deviating stop coin, not (5 − d)/4.
- **One record correction.** The 98.2% replay agreement is from `parity2`'s games (`training/metrics/critic_step1_20261009/prior_gaps.json`), not `parity3`'s.

### From challenge (critique: repairable)

**Kept:**
- forks with fresh dice under the core's exact laws, re-deciding after every roll, in two roles: as critic training data (decision windows) and as the middle rung of the forecast ladder (Ĝ_plug(2));
- the calibration of a truncated read against matches with known answers, the one claim the critique found nowhere. It becomes the ladder, which locates a forecast's bias at horizon 0, at horizon 2 or beyond.

**Not kept:**
- **D_k as the student's direct target.** The critique showed the critic's error at the window's end is a bias, not noise. Feeding windows into the critic keeps the student's target dice-free and lets the critic's function class denoise the labels, as the seed's grader intended.
- **The luck head f.** Windows use exact residuals.
- **The λ curve of a piKL decode read by matches**, which is outcome tuning.
- **The stop variants.** The critique showed the set {end, 1, 2, 5} is not closed under the shift. The forced path already contains every offset rule c ∈ [0, 4.5] as a stopping time, because an offset rule plays a* wherever it does not stop. So the current design's stop class is closed, and the critique's closure condition holds without change.

**Critique items answered:**
- item 1: the bias is stated, and κ checks the windows-trained critic against coins;
- item 2: Ĝ_plug(2) beside Ĝ_plug(0) and the match separates selection on the critic from myopia;
- item 3: no m_act stage remains;
- item 4: per-event-class χ² of outcomes against laws (I1), fork reproduction to the first argmax flip (I15), and P2 with a head trained on window states only, which isolates the window path.

### From invent (critique: repairable)

**Kept:**
- **Band coins.** The KKT taper is correct, and the record changes their value. `prior_gaps.games.jsonl` holds 100 recorded `parity2` side-games with 269 multi-legal decisions each. 3,093 decisions have a top-two margin under 0.1 nats: 310 per nat per side-game, 73% of them moves. With the critique's own scaling, which grows as √f, the band carries about 3 to 6 times the weak coins' information per unit of value shift; the critique computed 1.4 to 2.75 at f = 15 to 60. Their width falls to about 0.7 nats.
- **S−1**, which the critique called the best practical step: the numerics check on games already played, plus the tie census.

**Repaired:**
- **P-M.** The planted stream is booked into the simulator's result, uncentred, so a wrong propensity, a flipped side or a wrong runner-up moves the slope. That is P-coin; I13 tests it with injected bugs.
- **I12** recomputes margins by replaying the actor.
- **The band gate** is on total value shift instead of cost per decision.
- **Duplicate ties.** The critique's concern that ties are near-duplicates is sharper given the 73% share of moves. S−1 counts same-unit adjacent-hex ties, and the runner-up is re-paired to another unit or action kind if they dominate.
- **Drift.** About 50 band deviations per side-game is named as a weakest point.
- **The band's χ² readout** gets a kill line: band over weak below 1 drops the band.

**Rejected:**
- margin certificates and the Simpson known answer;
- luck heads;
- reply-read by enumeration;
- natural regression discontinuity.

Reasons are in the decision record.

### New this round (synthesis)

- **The exact local decomposition and the λ family.** This is Sutton's TD-error sum with [De Asis & Sutton](https://arxiv.org/abs/1807.01830)'s per-step control variates, made exact by enumeration. In this setting, deterministic players plus enumerated chance, the local residual needs no sampling except at sampled draws, so residual-gradient descent carries no double-sampling bias there. [Kapllani & Teng](https://arxiv.org/abs/2010.01319) report that global minimisation of local losses avoids the poor minima and divergence of the global deep-BSDE loss at long horizons. I found their method description, not a head-to-head proof.
- **Decision windows** are trained by the same residuals, so they need neither a lagged target nor a luck head. The nearest work is value expansion ([Feinberg et al. 2018](https://arxiv.org/abs/1803.00101), learned models, error bound growing with model error) and GNU Backgammon's training on luck-cancelled rollouts of disputed positions.
- **Windowed stop payoffs** take invent's reply-read intent by sampling instead of enumeration. S2 chooses between the static and the windowed payoff by the known answers.
- **The band on the previous switch score from round 2.** It replaces harden's targeted coins with the same aim (coins where the next rule decides), now with a taper and known propensities.
- **The setup draw as the first chance event** of the corrected match.
- **The S3 action prior is derived.** It is N·E[max(X, 0)] over the calibrated prediction, giving 0 to +30 Elo, and about 1 Elo at its low corner. Round 1 may therefore move only the stop.

**Budget:** S−1 to S3, round 2 and S5 cost about $54; round 3 adds $11. The remaining $20 covers an S1 rerun or the kernels with a partial rerun, not both.

## Open questions

1. **ρ_C, and λ.** Does the local end of the λ family beat the global end at 10,000 games? Is the residual minimiser worse than a TD fixed point under misspecification, as Bellman-residual minimisation can be? ρ_C still has no first-principles value.
2. **The prior's errors at ties.** σ_η²(G) near G = 0 decides the band's value, and 73% of ties are moves, where near-duplicates are likeliest. Nothing measures it before S1's κ_band.
3. **Windows.**
   - Do they open κ_move?
   - Do two side-turns carry a decision's consequences into what the critic prices?
   - Does bootstrapping at off-policy siblings stay stable?
   - What weight should windows have in the loss? Equal per state is set, not derived.
4. **Drift.** About 50 band deviations per side-game cap the value shift, not the state distribution. Is C(·; ∅) read well enough off coin games, with windows and the coin-free quarter as its anchors?
5. **Match C for `own1`.** The range [−0.03, +0.11] brackets two other players' records. Its true value sets how informative the −2.5 known answer is about over-continued states.
6. **The fog ceiling** Var(E[Â | h]) / Var(Â). The twins give a lower bound for it.
7. **Compounding** under zero-sum play and fog: no theory, and the measured per-round decay arrives only at S4.
8. **Numerics.** S−1 measures replay against the record, S0 measures generation against eval; failure costs about $7.
9. **User ruling.** Are κ, β, fallback F and P-coin, which regress outcomes on coins, covered by the killed "outcome policy gradient"?
10. **Exploitability** of a deterministic fogged player, which one reference does not show.

## Research targets for round 4

- **Residual minimisation against TD fixed points under function approximation**, and where the λ family sits between them: Scherrer's oblique-projection comparison of Bellman-residual minimisation with the TD fixed point (ICML 2010), Sutton & Barto §11.5–11.6, and the deep-BSDE evidence on local against global losses over about 160 events (Hu & Laurière; Kapllani & Teng; Huré, Pham & Warin). Target: a principled default λ, or a reason to fit λ by held-out ρ_C.
- **Window horizon and weight.** Horizon selection in MVE and STEVE and CBMPI's rollout-length theory: whether the window weight can be set by each residual's own variance (inverse-variance weighting) and the horizon by a measured materialisation of consequences.
- **The taper against tie-breaker theory.** Check the linear taper against [Li & Owen](https://arxiv.org/abs/2202.12511)'s piecewise-constant optimum. Find a cheap early estimate of σ_η²(G), and decide whether the band should exclude same-unit ties outright.
- **A pre-box tie census.** Find out whether `own1`'s recorded game files hold the legal joint actions with their acting units, so that tie composition can be counted from records (laptop-cheap) before S−1 is rented.
- **Exploitability at low cost.** One round of this method trained against a frozen π_k, as a best-response probe whose gain bounds π_k's exploitability from below.
- **Compounding data.** Per-round gains of approximate policy iteration with rollout-calibrated or value-expanded critics in long stochastic games: Tesauro & Galperin, CBMPI on Tetris, gnubg's iterated rollout training.
- **The $1,000 case.** Price the multi-decision operators conditional on the mixture curvature exceeding 2 SE.
