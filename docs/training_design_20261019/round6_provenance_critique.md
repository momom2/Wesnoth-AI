# Round 6 critique of the provenance stance

**Verdict: repairable.** The label, the ridge and adoption by match stand, and the audit's numbers reproduce. Four decisions rest on assumptions the proposal does not measure: the sign check, the miss branch, P-distill's band and S4's prediction. Each can be repaired without changing the core.

## 1. Provenance audit

I reran `round6_provenance_{prices,cells,closure,power}.py`, and each reproduced its `.out` byte for byte. `cells.py` and `power.py` had been edited after their outputs were written, so the rerun mattered. My new script `analysis/round6_provenance_critique_attack_stage1.py` reproduces `attack_pairs.out` (its part A) and refits it on unselected data (part B).

**Measured numbers, checked against the records:**
- **Generation price, $2.51e-5 a game.** Confirmed: one 600-game run of 129 s, conservative through its tail.
- **Eval price, $1.17e-4 a decisive game.** Confirmed, but the 4.7× ratio is confounded.
  - The "generation pool" is `run_elo_batch` itself at `--jobs 120` (`arm_ref_p120x6.log`). The recorded matches predate the generation-speed rounds (BACKLOG.md lines 100–134).
  - Same box, same code: the 20-job base arms cost $6.47–6.91e-5 a game (`speed.json`), a ratio of 2.6–2.75×.
  - Adoption at 120 jobs stands, but the rejection "eval-driver adoption matches, 4.7 times dearer" sets one program against itself, and the eval-priced items (S4's $3.00, PURE) are overstated (safe side).
- **Capped games.** At most 0.5% in `own1` matches (`rows.jsonl`), so per-game and per-decisive prices are interchangeable.
- **0.688 and σ² 3.67.** Confirmed (self-play cells alone 0.651, 0.58–0.72).
- **Closure 1.00 / 1.06, and 9.625 / 8.580.** Confirmed. A student read against `own1` at −1.5 compares with `own1` self-play, **8.42 ± 0.05** per side-turn (my reading, `stop_acts.read_match` on `gen_own1_selfplay`), not 8.580 (`own1` against `own1` at −2.5). The band edges move by at most 0.04.
- **Attack slope, −0.179 ± 0.043.** Confirmed.
  - The knob search was successive halving (`search.jsonl`), and all 13 vectors with a positive attack offset stopped at stage 1.
  - On stage-1 games only, which no selection touched (2,399 games), the slope is −0.225 ± 0.046 and the S0(d) prediction is **−78 ± 16 Elo**.
  - That lies inside the ±30-Elo window around −62, so S0(d) survives the selection check.
- **Host spread 1.41×.** Confirmed (BACKLOG.md lines 108 and 134).

**Derived numbers, redone:** the SEs 1.82, 1.10 and 4.6, the miss line 9.9, the plant bands, the sign-check SE, the SEs of g, A and C, the stage sums ($72.6, with $13.67 fixed), the price lines $2.97e-5 and $4.58e-5, J8/J4 in [1, √2], and the α* correction. All are correct.

**Guessed or assumed numbers that carry a decision:**
1. **The switch share q = 0.5.** It is the Gaussian model's value at μ = 0, and μ is unknown. At μ = −0.0018 and sd 0.003, the same model switches 9.3 of 76.1 band decisions (q = 0.12, `round5_synthesis_power.out` §2). The labels then estimate g + (2p/q)·A = g + 4.1A, not J(1). The comparator's reading rests on q; p = 0.25 itself is a design choice.
2. **The band mean of (non-attack top, attack runner-up) labels.** The sign check rests on it.
3. **The memory-tempo closure, carried over to an end_turn plant.** S4's first kill line rests on it.
4. **Exchangeable interactions and a quadratic J along the path.** α* and round 2's p rest on them.
5. **I9's tolerance.** It takes the widest flip seen in replay-against-replay as the band for live pairs.

## 2. Issues that would make it fail

**F1. The sign check can pass while the sign is wrong.**
- A mover-sign error turns the label mean m into +|m|; the check catches it with 80% power only if |m| ≥ 2.84 × 0.0061 = 0.017 per pair.
- The record only bounds |per change| from above (≤ 0.076, over margins up to 1.4 nats); inside G < 0.35 the proposal's own model gives c + b·G, which can sit near 0. The power is unknown.
- No invariant ties a logged D_i to the command the core applied. If "played hex" is read from the coin log (the text leaves it open), a coin that never reaches the game passes every plant.
- **Repair:**
  - Make I2 (the enumerable duel, exact E[T]) carry the mover sign: run it through the production generator, once with the coin on each seat.
  - Add an invariant: at every logged D_i = 1, the game record's command equals a1.
  - Keep the sign check as a report, not a gate.

**F2. The miss branch rarely tells anything, and one of its rules has nothing to act on.**
- **Resolution.** g carries ±15 Elo at 2 SE and A ±16, against the model's gains of 4.7–16.7 Elo at sd 0.0015–0.003 (`power.out` §2).
- **No rule for Â ≥ 0.** α* = −g/(2A) is then undefined, and the proposal says nothing for that case. At a true A of −5 Elo this happens with probability 0.27 [D: Φ(−5/8.1)].
- **C cannot separate the causes.** C − A equals the opponent-coin term plus the non-exchangeable part: one statistic for two causes. The claim "C separates the opponent-coin bias from curvature" is false. Round 2's rule, p = q̂/2 "after a miss attributed to curvature", rests on an attribution no readout makes.
- **Repair:**
  - On a miss, match π_½ and adopt whichever of π_1 and π_½ certifies.
  - To separate the opponent term, run part of round 2's batches with coins on one seat only (half the labels per game), or drop the attribution.

**F3. P-distill's band [9.37, 10.01] borrows a closure measured on another signal.**
- The 0.82–1.30 comes from a 64-slot network learning to act like its own 0-slot reading; a uniform end_turn shift may transfer differently, so a reading outside means "another signal", not "distillation is broken", yet S4 stops on it.
- **Repair:** read the student's end_turn logit shift at held-out positions, predicted −1.0 less the shortfall the fit's held-out KL implies, both before any match; tempo follows from it.

**F4. S4's prediction leaves out a term, so the spillover reading is confounded.**
The forecast −(1 − ρ̂)Ĝ_k omits disagreements with π_k away from the switches (the measured recipe ended at agreement 0.862). S1's μ̂ per pair type, the mean value of playing the runner-up at a band decision, prices them. Without that term, "spillover" and the widening to w = 0.7 rest on an unidentified residual.

**Lesser points:**
- **S0(d) tests a different code path.** It changes a seat's decode offset; adoption runs a seat's switch head (new code: the per-seat hook, seed-shared pairs, columns over 25,600 records). Drive S0(d)'s attack change through the hook with a planted ŵ, its count predicted from S0(a)'s census.
- **I9's null covers a prefix.** At fp32-against-bf16 rates the lowest gap bin gives about 3.9 flips a game [D: 449 × 1.19 × 0.0136 × 0.53]; 12,800 live pairs will exceed the widest flip of 800 replays. State the tolerance as a rate per gap bin.
- **I3** holds up to b·(G + G′) ≤ 0.027·|b|, not exactly.
- **"Run student 4's mixture before buying more"** maps no outcome to an action.

## 3. Novelty

The midpoint reading is the one-point midpoint rule for a line integral of gradients (Owen's multilinear extension; [integrated gradients](https://arxiv.org/abs/1703.01365), Sundararajan et al. 2017), exact iff J is quadratic along the path; the mixture gradient is CPI's identity ([Kakade & Langford 2002](https://mlanthology.org/icml/2002/kakade2002icml-approximately/)). Two searches found only first-order mixture bounds ([Iwaki & Asada 2017](https://arxiv.org/abs/1710.03442)) and nothing setting a randomisation rate this way. The application may be new; it is a textbook rule under an exchangeability nothing measures.

## 4. Strongest part

The audit: every cited script reproduces, and it correctly fixes two derivation errors in the current design, the ordered-type intercepts (continuity at a tie gives c_vu = −c_uv, c_uu = 0, b_vu = b_uv) and α* fed by labels that estimate g + A. With it, S0(d): a predicted −62 ± 15 Elo is a real known answer for the match path's seat sign, and it survives removing the knob search's selection.
