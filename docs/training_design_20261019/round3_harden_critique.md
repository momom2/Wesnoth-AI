# Critique of round3_harden.md (round 3)

**Verdict: repairable.** Switching the instrument to corrected matches is right, and its arithmetic checks. Five problems remain: the default critic is the record's failed recipe with less noise; P3 stops correct code and passes a myopic fit; P0 certifies an instrument no stage uses; the dual bound bounds less than claimed; the stop sizing comes from another player.

## Strongest part

§0.1 with §1.5, which I checked. The audit's π²·12·0.4 = 47 per game gives SE 0.034 in z over 40,000 games, or 12 Elo (347 Elo per unit of z near even). The corrected match's 0.65 over 3,200 games gives 0.014, or 5 Elo.

The corrected match is unbiased for any frozen C, because each L_k has mean zero under the core's exact law given its pre-event state; critic error and numerics flips cost variance only. Reading Ĝ_plug on the match's own states pairs forecast and match: at the exact critic, z − V(x_0) − Σ_t A(x_t, a_t) − Σ_k L_k = 0 on every path. β_avg is also right: two calibrated predictors with correlation r average to slope 2/(1+r).

One correction: the 30× holds in round 1 only, where `own1` against itself scores 0 by symmetry. From round 2 on, a gain needs two matches (SE 7), about 8× per dollar.

## 1. The default critic is the record's critic with less noise

Phase 1's population minimiser is E[z | x_t], the same as result regression's. The control variate only scales target noise by ρ(C′), and target errors still share each game's tail, so the effective sample rises by 1/ρ ≈ 1.2–1.7 and stays near the game count. The record says more games of the same targets did not help, and nothing derives the drop from pass 0 (0.70–0.85) to pass 1 (0.58–0.75).

The chance-sibling information §1.7 credits enters only through gradients at unrealised outcomes, which means only in Phase 2, and Phase 2 is optional. Its gate barely opens. At 10,000 games κ's SE is about 0.25–0.4 per type. "Lowers no type's κ", read on point estimates over four types with independent differences, adopts an equally good Phase 2 with probability about 1/16, and one that raises every κ by half an SE of the difference with probability 0.69⁴ ≈ 0.23. The design's main information source is off by construction.

Repair: make the joint residual the default after a Phase-1 warm start, gate it on a paired held-out difference in ρ_C with its SE, and derive pass 1's band from the record's critic curves.

## 2. P3 stops correct code and passes a myopic fit

- **(i) cannot be learned.** Gold enters as one scalar, gold/500 (global feature 2 in `wesnoth_ai/encoder.py`), and bf16 serving casts it. Its spacing is about 1 gold between 125 and 250 gold, and adjacent values coincide above 250. The frozen trunk never learned parity, and the head reads four token outputs. A correct Longstaff–Schwartz fit therefore fails P3(i), and its kill line stops the program.
- **(i) is monotone.** Every act past the stop pays, so a one-step greedy fit also runs to the path's end, and the check cannot tell LS from the rule `current_design.md` §6 rejected. Repair: plant a non-monotone payoff on a feature the head reads (turn ≥ 10, say): −0.01 for each of the first two acts past the stop, +0.05 at the third.
- **(iii) does not test the correction.** A mean-zero term independent of h moves no regression on h, whether the luck correction is applied, missing or sign-flipped.

## 3. P0 certifies the retired instrument, at overstated power

With the audit gone, P0's arcsine coins between two decodes exercise a code path no stage uses. S1's weak, run-length and targeted coins get only S0(d)'s cost check and the unit invariants.

P0(B)'s SE is not 0.07. B's coins sit wherever `own1`'s top action is an attack, about 50 per side-game: π²·50·0.5 ≈ 250 per game, an SE of about 0.1 at 25,000 games. A propensity weight off by 2× on a −0.25 effect moves the reading by 0.125, inside 2 combined SE (about 0.2), so the check passes.

Repair: replace P0 with a known answer on the training path, for example weak coins between `own1` and Match B's decode, read by the BLP slope.

## 4. The dual bound bounds less than claimed

- Its payoff is C's, so a critic that overvalues continuation raises the rule and the bound together.
- It covers stopping rules on the a* path only, not "any rule's gain over π".
- M is the payoff's own martingale part, so u_t − M_t = u_0 + A_t with A predictable (Doob decomposition). The bound is then E[max_t A_t], which includes the value of foreseeing how later fights shift the drift. Tight bounds use the continuation value's martingale (Andersen–Broadie; Brown, Smith & Sun's "good penalties").

Its band of 0.01–0.06 has no derivation, and no stage acts on it.

## 5. The stop sizing comes from another player

The cited sweep (`endturn_offset_sweep_prereg_20260919.md`) measured `relset`, not `terrain`: memoryless, an earlier observation epoch, a forced Knalgan side. So the per-act values, c* ∈ [1.5, 3.0], the 36 extra acts behind the "0.001 per act" power claim and Match A's −0.58 anchor are all `relset`'s.

The "record −0.035 ± 0.035" for −2.5 appears in no record outside this folder. The one memory-player measurement points the other way: `parity3` at 64 slots gained 65 Elo from −1.5 to −2.5 (`memory_in_play_parity3_prereg_20261009.md`). `own1`'s knob search came out flat (+1.7 ± 12.3), but it moved four knobs together.

Cheap fix: in S0, match `own1` at −2.5 against −1.5 and log acts per side-turn.

## 6. Rounds 2 and 3

- **The forecast's estimand.** The student regresses advantages averaged over the opponent mix and is calibrated on all coins, but Ĝ_plug forecasts the gain against `own1`. Where advantages depend on the opponent, the forecast is biased, and §1.5 would misread the miss as critic bias. Calibrate the forecast on the `own1` quarter.
- **The cited theory does not apply.**
  - The mix is not fictitious play. A one-step improvement is not a best response, the weights are fixed, and the last iterate is deployed; Robinson's theorem concerns time averages of best responses in matrix games.
  - [Winnicki & Srikant](https://arxiv.org/abs/2303.09716) assume α < 1, and as α → 1 no finite look-ahead meets their condition.

  Keep the hedge, drop the citations.
- **The predictions cannot fail.** Suppose round 1 reads +20 Elo. Round 2's correct prediction of +10 against SE 7 then clears P(gain > 0) ≥ 0.9 only about half the time, while "within 2 SE of the forecast" accepts a round that gained nothing. The gate to the $1,000 case thus rests on readings that cannot fail. Power S4's matches (12,800 games, SE 3.5) or require the forecast's lower bound above zero.

## 7. Novelty already covered

- **§3.3** is AIVAT ([Burch et al.](https://arxiv.org/abs/1612.06915)), as used for DeepStack; [Kim & Sandholm](https://arxiv.org/abs/2605.14261), cited by the file, study a learned value function inside it.
- **§3.1:** a mean-zero control variate leaving the conditional mean unchanged is textbook. Cross-fitted value nuisances are [Kallus & Uehara's double RL](https://arxiv.org/abs/1908.08526).
- **§3.2:** penalties from an approximate value function's martingale differences are information-relaxation duality ([Brown, Smith & Sun 2010](https://doi.org/10.1287/opre.1090.0796)). A POMDP version appears in [Ruiz Lacedelli's thesis](https://academiccommons.columbia.edu/doi/10.7916/d8-mwkk-mr35), which I saw in a search excerpt only. The game-turn application may be new.
- **§3.4** is linearity of covariance.

## 8. Smaller

- **Run-length visit rates.** The stop payoff reads x∘end at depth d, which training visits only when K = d (1/4 each), not (5 − d)/4.
- **Cross-fit leak.** Pass 2's C′ was trained on targets corrected by pass 0's critic of this game's own fold, so "independent of the game" fails at the second refresh. The leak is O(1/n), three folds remove it, and I10 does not test it.
