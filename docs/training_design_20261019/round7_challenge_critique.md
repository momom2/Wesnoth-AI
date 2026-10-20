# Critique of round 7 challenge ("plans, not ties")

**Verdict: repairable.** The label mechanics are correct, and the null exit is cheap. The reason given for running intents before the current design (§0, "the stakes are in plans") does not survive a correct reading of its own record, and S1's gate tells you nothing. New script: `analysis/round7_challenge_critique_bench.py` (`.out` beside it).

## 1. Provenance audit

| Number | Class | Check |
|---|---|---|
| Attack-free spread 0.0193, sd 0.139 | M | Rerun of `round7_challenge_turn_stakes.py`: output identical. Misread (§2). |
| Alternative minus "the human's own" turn, −0.029 | M, mislabelled | Slot 0 is `obs8`'s turn at `raw:t0+eo-1.5`, not the human's (`docs/turn_value_prereg_20260925.md` "Validation"; `tools/turn_gap.py`). The figure pools T=1 turns with continue edits, which are single-decision changes. S5's "within 0.03 of human ones" is therefore wrong. |
| 7.03 decisions; 0.707 recruit | M, other positions | These come from `data.jsonl.gz`, positions of `obs8` against `terrain`. On the bench's own positions, turns take 9.6 decisions (base) and 11.8 (alternatives) (`validation.md`). |
| ≥ 0.040–0.052 per deviation; "11 to 33×", "39 to 87×" | D | The arithmetic is right; the comparison is not. It compares deviations sampled at T=1, mostly at confident decisions, with the band's detection floors, not with any measured spread. |
| 0.688, 19.4, 76.1, $2.51e-5, 0.847 | M/D | Found in `round6_provenance_cells.out`, `round5_simplify_critique_decision_units.out`, `round6_provenance_prices.out`. |
| 12.4; 3.64; SEs 0.0018/0.0013/0.0009; fit noise 0.014/0.007; forecast SE 2.2–4.9; δ 0.0071; 24 Elo; P-EM SE 0.0035; $5.9 | D | Rerun of `round7_challenge_power.py`: output identical. Rederived by hand: correct. |
| Placebo as a ninth arm | inconsistent | P_0 + 8·P_c = 0.5 + 8/16 = 1 leaves no probability for a ninth arm. With nine arms at 1/18, the per-code variance becomes 13.8, not 12.4. |
| EM $1.51 | D, from a borrowed rate | The 385 forwards a second is the memoryless teacher-target pass (BACKLOG.md l. 96). The E-step instead needs sequential forwards that carry memory, under each code, plus the natural prefix. Little money rides on it. |
| Round 1 $7.03–14.56 | D, minor | The script adds an interim match that the stage table does not have, and leaves out S0 and the $0.20. The table itself sums to $6.71–14.24. |
| "16 slots play as 64"; "a quarter of decisions" | M (`parity3`) | The 0.0 ± 12.3 Elo tie bounds what slots 17–64 are worth; it does not show that they are empty. Nested dropout (`wesnoth_ai/memory.py`) is the better argument. For `own1`, [U], as the stance tags it. |
| P_0 = 0.5; one gradient pass per M-step; the S1 "differs by 2 SE" statistic | C, consequences not derived | §3a, §3e. |

## 2. The core evidence, re-read

The bench's alternatives are of two kinds. At the 173 positions with all five candidates, attack-free alternatives against an attack-free base (luck-corrected, position bootstrap):

- **T=1 turns** (the "plans"): mean **−0.055 (SE 0.031)**, noise-corrected variance 0.033, n 146.
- **Continue edit** (the base turn plus one argmax action instead of end_turn, a single decision): mean **+0.049 (SE 0.028)**, variance 0.0096, n 68.

**The spread is not selectable.** Choose on even playouts, read on odd ones, with the base allowed:
- attack-free, base or T=1: **−0.035 ± 0.040**;
- attack-free, any candidate: −0.019 ± 0.037;
- all candidates: +0.042 ± 0.021.

A Gaussian model with the stance's own moments and the bench's noise predicts **+0.066** for the attack-free procedure. The measured −0.019 sits 2.3 SE below it. The original record agrees: the share of positions with a gap ≥ 0.25 equals its permutation null (0.285 against 0.282, `validation.md`).

So the variance of T=1 turns is mostly downside: sampled plans are worse than argmax, with no resolved upside. The only kind of alternative with a positive mean is a one-decision stop change, which the current design's (end, best act) pair already covers. The bench also compares fixed sequences played once with their own trait draws, the comparison the play ruling forbids.

Nothing costly rests on this number, since S2 batch 1 measures intent stakes before anything relies on them. But it is the stance's only reason to put this route first, and the record points the other way.

## 3. What else would make it fail

**a. S1 cannot fail.**
- The diversity read is deterministic on recorded positions. At 359,000 decisions, a 1% argmax disagreement has SE 0.00017 [D], so any code that moved at all "differs by 2 SE".
- The null branch reads "the slot channel is too weak". That conflates channel weakness with under-fitting: one gradient pass per M-step is a guess.
- P-EM's pass line (agreement > 0.514) admits a fitter that recovers almost nothing, and its closure is reported, not gated. S1's null reading needs P-EM's closure as its condition.

**b. Hindsight in the E-step.**
- The E-step assigns each turn by the likelihood of all its realized actions, including actions taken after its rolls (618 of 966 bench candidates attack).
- So the codes absorb dice outcomes: a "retreat" code is fitted on turns whose attacks failed. Trajectory-inferred conditioning in stochastic environments fails in exactly this way ([Paster, McIlraith & Ba 2022](https://arxiv.org/abs/2205.15967)).
- "It cannot assume dice that did not fall" holds for the labels, not for the repertoire's quality. Fix: score only the actions before the turn's first chance event.

**c. Post-treatment features.**
- "The features come from the turn's first forward." On an intent turn, that forward runs under the pinned slots unless a natural forward runs first.
- A feature computed after D is drawn makes the regression of T_c on x biased.
- The fix costs one natural forward per coined turn start, about 5% of forwards [D: 0.5 / (184.3 / 19.4)]. Add an invariant: x_t is identical across arms on forks.
- "No extra forward" is false for π_1's switched turns.

**d. Memory spill.**
- Slots 0–47 are written from outputs that attended to the pinned slots, so an intent outlives its turn, and "intent 0 is `own1` exactly" fails after any intent turn.
- The labels price this without bias, but I-pin does not test it.
- The placebo's mean is exactly 0 only if "identity" means no pinning at all.

**e. Excursion, multiplicity, stop line.**
- P_0 = 0.5 steers half of both sides' turns, each across many decisions. That is further from both `own1` and π_1 than the band's 19 deviations in 225 decisions. The diversity read measures this before S2, so P_0 should be set then.
- Across 8 codes, P(some code passes 2 SE at a null) = 0.17 [D].
- "No cross-fit forecast above 0" has no margin, so a null continues up to about half the time, to $14.24 [D]. "A null stops at $5.9" is the best case, not the expectation.
- "Intent stakes are then bounded" overclaims: per-code means bound the mean advantage, not the context-dependent spread that §1.4 says the selector exploits.

**f. Novelty is narrower than claimed.** Each part already exists:
- soft prompts on a frozen transformer policy, chosen by a bandit ([Rietz et al. 2025](https://arxiv.org/abs/2502.04979); [Hu et al. 2023](https://arxiv.org/abs/2305.09648); abstracts only, details unverified);
- options fitted from demonstrations by EM-style iterations ([Fox et al. 2017, DDO](https://arxiv.org/abs/1703.08294));
- importance sampling at the option level ([Guo, Thomas & Brunskill 2017](https://arxiv.org/abs/1703.03453)).

What remains new is per-turn, within-game selection with exact randomized arms.

## 4. Strongest part

The multi-arm label. E[T_c | H] = Q̃(c) − Q̃(0) holds for any control variate fixed beforehand. P-intent's slope is exactly 1: E[(1[D=c] − P_c)(1[D=c]/P_c − 1[D=0]/P_0)] = (1 − P_c) + P_c. A sign error reads −1, and an arm logged but not played reads 0. Together with the cheap first exit, this makes the route a sound probe, once its motivation is withdrawn.

## 5. Repairs required

1. Withdraw §0's argument, its "39 to 87×", and S5's human comparison. Rank the route on no evidence advantage. The one measured positive is the stop decision.
2. Gate S1's null reading on P-EM's closure, and give S1 a go statistic that can fail.
3. Score the E-step on pre-roll actions only.
4. Use natural-forward features, tested across arms.
5. Budget the placebo arm, or drop it.
6. Set P_0 after the diversity read.
7. State the expected null spend.
