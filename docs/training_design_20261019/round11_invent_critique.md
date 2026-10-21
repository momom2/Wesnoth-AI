# Round 11 critique: invent stance (exact-law advantage walk, KL-set policy gradient)

**Verdict: repairable.** The label algebra is right and the plan measures its two unknowns first. But several sealed numbers rest on assumptions that fail, the plants miss the path that carries most of the signal, and three thresholds are guesses. Arithmetic: `analysis/round11_invent_critique_checks.py` (*checks*).

## 1. Provenance audit

| Number | Class | Check |
|---|---|---|
| 453.3 decisions a game | M | Budget script rerun, reproduced. Measured on `own1` at argmax + offset; unknown at temperature 1, as stated |
| 109.7 attacks; 0 outside support | M | `readout.txt` line 54 |
| r 0.699 | M | `round9_simplify_residual.out`. The setup cell takes it from 0.82 to 0.70, at argmax |
| β by setup +0.0007 | M | Reproduced; every penalty at the grid's top |
| Slope +0.0157; ratio 0.999 | M | `round9_simplify_labels.out` lines 13, 18 |
| Prices, 1,476.9/s, $8.848e-5 | M | `matches.txt`, `status.txt`, `match.walls` |
| 347.4 Elo/z; match SE 4.3 | D | Correct |
| Forecast SE 2.77; line 10.3 | D | Arithmetic reproduces (*checks* §2); assumptions fail (§2.1–2.2) |
| $8.34 a round, "nine rounds" | D | "Nine" leaves out M1, M2, P, the bench and the forwards (§2.9) |
| $81.4 left | D | Assumes the $5 window counted (unknown, `current_design.md`) |
| 0.7% df cost | D, wrong | 280 columns with turn bins: 2.8% (*checks* §4); nothing rests on it |
| κ₀ | guessed | "Set at P from M2" with no rule; it carries P's and R1's step |
| 1.2× wall | guessed | Decides an optimizer switch; not derived from positions saved |
| Bench arm $0.14 | misapplied | Priced at $0.42/h; sized for the wrong regime (§2.8) |

None of the three guesses is fatal if M2 supplies it before use. The proposal must say how.

## 2. What would make it fail

**2.1 χ² ≈ 2 KL.** The line search bounds KL(π_k‖π_α), which is lenient where the importance ratio grows: on actions raised from a rare base. Raising 0.01 to 0.03 gives χ²/2KL = 2.2; 0.001 to 0.01 gives 6.0 (*checks* §1). Then SE_f is 3.9–6.8 and the miss line 11.7–16.1. The 0.999 cited measured coin labels, not score products under a shared residual.
- Repair: the empirical per-game variance of the held-out forecast sums.

**2.2 Seats.** Summed over both seats, the held-out forecast estimates gain_A + gain_B. R1, one seat with sides alternated, reads half of it. As sealed (the SE's D = 453 counts both seats), a correct step reads "below" and κ is halved for nothing. Halved: SE 1.38, line 9.1.

**2.3 The plants miss ε_g.**
- The surrogate Σ(π_θ/π_k)Ã_t omits σ_t while Ã_t is signed to side A; as written, side B learns to lose.
- M1's plant tests the ridge. P trains on "the plant alone". If C is a walk column, M2's plant flows through I_t.
- None routes signal through ε_g, which carries everything outside the walk's span. A wrong game join or side sign on ε_g passes all three; I-duel covers it only in a toy duel.
- Repair: plant z′ with C excluded from the walk, read per seat at scale.

**2.4 Loss.** "The own-trajectory loss with a signed one-hot target" is −Ã log π_θ. It lacks the factor π_θ/π_k, so the claimed drift correction does not exist in it.
- Repair: the ratio form, with π_k(a_t) logged.

**2.5 Capped games are unaddressed.** Dropping them conditions on a post-action event (bias). Scoring them 0 makes stalling a lost game worth +1. `elo_collect` counts caps as absences, so the claim rewards the same stalling, and the memory player already stalls more (CLAUDE.md).
- Repair: report the cap rate each round and count caps in claims.

**2.6 Stale yardstick.** The reference has been `own2` since 2026-10-21 (`configs/reference_player.json`); D₀ and the claims are set against `own1`.

**2.7 M2.** If |g|² is unresolved at 20,000 games, M2 yields only a lower bound on G*, whose height depends on d_eff = tr(Σ)²/tr(Σ²), estimable from the same gradients. There is no branch for this outcome.

**2.8 The bench arm.** With the noise scaled to B_simple, 1,000,448 positions make one critical batch only if G* ≤ 2,207 games. At G* = 100,000 they make 0.022 of one, chord SNR about 0.15 (*checks* §5): every arm reads noise. Size it from M2 or drop it.

**2.9 Forwards.** Each α needs a pass over 4.5M held-out positions: up to $0.49 at the trainer's rate (*checks* §3), against a $0.57 match.

**2.10 Complexity against a bounded gain.**
- The walk saves at most a factor 1/r in games.
- Dice plus the setup cell already reach 0.699 with certified code.
- The new action block needs expectations over every legal action for 280 columns, and outcome tables for unchosen attacks, which the generator does not compute. There are few, since attacks start from the unit's hex (`server_priors.compact_action`), but "costs nothing at the margin" is unshown.
- M1's keep rule (+2 SE, about 0.01 of r) keeps a block that saves 1% of games.
- Keep it on dollars saved against slowdown and code risk.

## 3. The optimizer argument

- **"Adam whitens; Muon gives noise full-size steps" is no contrast.** On pure noise, Adam's per-coordinate step is fixed by β₁ too. The real difference is linearity:
  - with a stable v, Adam is a fixed diagonal preconditioner that keeps the summed gradient's SNR;
  - Muon orthogonalizes each noisy step, flattening a low-rank signal into full-rank noise.

  The argument is not made.
- **The fine-tuning evidence is neutral.** [2502.16982](https://arxiv.org/html/2502.16982v1) Table 7 favours Adam on all four benchmarks. Table 6, the authors' own AdamW-pretrained model, has Muon ahead on three of four (MMLU 55.3 against 52.0). The authors read both as "on par". The grounds that stand: AdamW trained `own2`, and the diagonal metric.
- **[2605.19282](https://arxiv.org/abs/2605.19282)'s abstract attributes the collapse** jointly to low SNR and to preserving per-head specialization.
- **The [full-batch post](https://www.lesswrong.com/posts/9gqjdRyJxSM6kS7zi/i-trained-gpt-2-with-full-batch-gradient-descent-so-you-don) is another regime.** Its numbers check out (−0.945; 48 → 113; 3.411 against 3.228). But it ran a Muon-family optimizer on the exact gradient over 800M tokens. Its edge-of-stability readings describe deterministic steps, not a minibatch AdamW pass scaled afterwards. The reuse result supports one pass per round only by analogy (one run, noise confounded with repetition, by the author's caveat).

## 4. Novelty

- **Claims 3 and 4 are standard control-variate facts.** AIVAT's unbiasedness proof is the same argument ([1612.06915](https://ar5iv.arxiv.org/html/1612.06915)). TrajCV subtracts centred future-action terms and notes that dynamics need their law ([1908.03263](https://ar5iv.labs.arxiv.org/html/1908.03263)).
- **Claim 1 is AIVAT moved to training.** A search found no such use, so the assembly may be new.
- **Missing neighbour:** input-dependent baselines ([Mao et al. 2019](https://arxiv.org/abs/1807.02264)); dice are an exogenous input.

## 5. Strongest part

**The label identity, checked.** E[Ã_t | h_t, a_t] = Q(h_t, a_t) − c − Σ_{s<t} I_s, that is, Q less a function of the history before a_t. It holds for any frozen w, with no value level, under fog, memory and self-play, since each event is centred by its logged law. It is a variance device, not a shaped reward.

**The order.** M1 and M2, about $2 with no training, measure r and G* first, and G* ∝ r prices the mechanism.

## 6. Repairs

1. σ_t in the surrogate.
2. The ratio loss.
3. Per-seat forecasts with an empirical SE.
4. An ε_g plant.
5. The cap rule.
6. `own2` as the yardstick.
7. κ₀ from M2's ĝ and the match SE, plus M2's lower-bound branch.
8. The bench arm sized from M2, with a priced switch criterion.
9. The action block kept on dollars.
10. The forwards priced.
