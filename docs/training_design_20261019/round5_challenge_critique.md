# Critique of round 5 challenge: search first, certified by play

**Verdict: repairable.** No costly stage rests on a guessed number, and the first question it asks is the right one. Three faults stand:
- **The blind σ is no bound on noise.** Under it, an evaluator with no information flips about 93 decisions per side-game. S2a's null is the one that closes the route, and it cannot separate "no decision value" from noise flips.
- **The fair arm's end_turn** is valued from a view no player has.
- **The oracle's gain** mixes the critic with god-view expansion.

## 1. Provenance audit

| Number | Class | Check |
|---|---|---|
| Per flip −0.0021 ± 0.0033 and −0.0008 ± 0.0016; 9.9 and 18.8 flips; 237.6 decisions | measured | Reran `round5_challenge_lookahead.py`: identical. The caps, 19% against 0%, follow the base player (`parity3`'s self-pin caps 21.6%, `own1`'s matches under 1%, `rows.jsonl`), not the evaluator. |
| r = 0.747 | measured | Script read: leave-one-match-out, mean-preserving. |
| SE 5.3 Elo; a_min | derived | Arithmetic right. A null bounds \|a\| by \|â\| + 2·SE/m, not by 2·SE/m. |
| 1/N = 0.0042; 276,000 pairs | derived | Arithmetic right; the inference fails (§2, item 6). |
| Ratio 2.9 / 11.7 / 46.8 | derived from mis-sourced inputs | n_c = 180 and p(1−p) = 0.19 are not in `round5_provenance.md`. The 0.19 is the band's share of decisions at w = 0.175 (`round5_simplify_power.out`), not a p(1−p). With the provenance stance's illustrative Σp(1−p) = 31, the ratios are 3.2, 12.9 and 52. The conclusion survives, but for certification only, and it carries "Rejected: coins of every kind". |
| Old critics' losses | measured | They match `*.summary.json`. Training games: 6,832 from matches of the `parity2`, `obs8` and `parity3` era (`positions_manifest.jsonl`), none `own1`'s. |
| T100 − HP margin +0.041 ± 0.017 | measured, selected | The best of four readings (+0.041, +0.038, +0.021, +0.017; `readout.md`), whose verdict is Kill. |
| U's variance, at least 3-fold lower | derived, other population | Human positions, temperature 0.5. Nothing rests on it. |
| Wall factor 1.6; $5e-8 a critic forward; $1.35 a match | measured for CPU evaluators; derived across regimes | The archived `CriticEvaluator` loads one model per worker (CPU by default), about 16 states a batch, so the shared server's price does not apply. S1 stays small even at several times the price. |
| σ 0.1, c 1, k 8 | placeholders | The archive's `configs/lookahead.json` says so. |
| 64 against 0 slots: +13.0 ± 12.3 | measured, other checkpoint | That is `own2`'s reading. `own1` at 64 slots reads −21.7 ± 12.3 against itself at 0. |
| S4 $7.4 | derived, low | Labelling, the trainer, P5 and four matches rebuild to about $9. The plan stays inside $85. |

## 2. What would make it fail

**1. Blind σ is not a bound.**
- Set σ² to the noise variance of D. A zero-information critic's score (Q − V̄)/σ then has an SD of about 0.7 nats before clipping.
- On `parity3`'s recorded margins, that flips 93 of 268 decisions per side-game, at a mean top-two gap of 0.37 nats. With noise at a quarter of σ, it still flips 43 (`analysis/round5_challenge_critique_noise_flips.py`).
- Material and exposure flipped only 9.9 and 18.8 because they are flat across most move siblings; a learned critic is not.
- The record prices deviations from the argmax only in aggregate (argmax over temperature-1 sampling, +412 ± 97 Elo). S2a's Ĝ nets the critic's signal against flips of unmeasured cost, and a null there ends the route.
- Fold disagreement also misses shared bias: a deterministic evaluator such as material has none and gets the full tilt.

**2. The fair arm's end_turn leaf.**
- Checked in the archive: `lookahead_world.py` removes every unit the mover cannot see and gives the opponent the mover's gold. `CriticEvaluator` encodes the observation of the side to move and negates it.
- So after end_turn, O100 reads the opponent's view of its own army, pruned to the units the mover saw. Every other candidate is read from the mover's real observation.
- The bias sits on the decision priced at +229 Elo, and U as specified is undefined there.
- The record's one reading of this world: rollouts with hidden units deleted lost the whole selection gain, −0.002 ± 0.018 against +0.049 ± 0.017 on the true state (whole rollouts there, not one ply; suspect turn protocol).

**3. The oracle is not the critic.** God view also informs the expansion (ambushes, fogged castles, hidden attackers), so "oracle minus fair" changes the world and the critic at once. A material-in-god-view match separates them.

**4. Data without contrasts.**
- S2a's 50,000 argmax games vary only in dice and setup. Siblings b ≠ π(x) are never played, so the critic's sibling differences are extrapolation.
- The record agrees: T25 and T100 select alike (+0.040 and +0.041 over the HP margin; T100 − T25 +0.001 ± 0.016), though four times the data lowered the loss by 0.032. "Limited by games" holds for loss, not decisions, and it is the claim behind S2b and the $1,000 curve.
- Coins filled this role; a ratio about certification does not replace it.

**5. Estimand.** T100 and O100 value an older mixture of policies; the performance-difference lemma needs V^own1. C drops the memory that changes `own1`'s play, and U = E[C|h] inherits that bias ([Baisero & Amato](https://arxiv.org/abs/2105.11674)). U's labels must come from the other fold's C, or they carry C's memorised results.

**6. "Per-decision labels cannot carry the signal" proves too much.** 1/N bounds the mean over all decisions, while the gains sit in a few. The critic itself pools per-position labels of the same noise.

**7. Plants and invariants.**
- A constant U reads δ/2 for every B, so it passes both P-U and I5. Each needs a positive control: a bit U sees, read at slope 1.
- P-score's "share must rise" has no predicted size and no price.
- P1 with a fixed linear φ tests one column, not the fight laws.
- P5 plants a uniform shift, while S's flips depend on the state.
- I7's threshold, the generation self-agreement rate, is a numerics figure. It is not a derived capacity for distilling a teacher that may flip a third of its decisions.

**8. Novelty.**
- [Bertram, Fürnkranz & Müller 2024](https://arxiv.org/abs/2407.05876) already regress perfect-information evaluations of sampled states onto information states (two or three samples per position, in poker and reconnaissance blind chess). Learned information-state values were used inside search earlier ([Buro et al. 2009](https://cs.du.edu/~sturtevant/papers/skat.pdf)).
- What remains is Baisero & Amato's identity, for a critic of a fogged policy, used as a one-ply leaf. Claim 4 fails by item 1.

**9. Engineering.** The player sits on archive branches at core phase 31; main is at 35, with memory states on the server since 34. The port is unpriced.

## 3. Strongest part

- It asks first, in play and for a few dollars, whether a learned critic's look-ahead beats the prior, with code that has played 1,791 games, and it certifies before distilling.
- The per-flip readout gives every evaluator a common unit and reproduces exactly.
- U trains on C's readings, not results, so its positions are not limited by games; C values a fogged policy, so averaging it over h involves no strategy fusion.

## 4. Repairs

- **A placebo arm at every look-ahead stage.** Use the fold difference (Q̂_1 − Q̂_2), scaled to the player's noise: with independent fold errors it has the critic's noise and no shared signal.
  - Real minus placebo is the critic's decision value at that σ.
  - The placebo alone prices the noise flips.
- **Shrink the tilt** by the blind reliability Cov(D_1, D_2)/Var(D) per kind, stating that shared bias passes through.
- **End_turn from the mover's observation.** Train U on both sides' views at each sampled position. Until then, the fair arm keeps the prior's stop.
- **Two more S1 arms:** material in god view, and attacks only.
- **S2a's games:** give them a logged exploration schedule, or state that the ladder tests extrapolation only.
- **Smaller fixes:** source the ratio's inputs; cross-fold labels for U; positive controls for P-U and I5; a predicted size for P-score; a state-dependent P5; derive I7's threshold or drop it; price the port.
