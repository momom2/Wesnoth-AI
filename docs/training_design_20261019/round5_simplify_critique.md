# Critique of round5_simplify.md (round 5)

**Verdict: repairable.** The label, its proof and the information argument hold, and most numbers trace to their records. Three things fail as written:
- The switch rule and the forecast assume the runner-up's mean advantage is zero.
- Two kill lines rest on guessed numbers.
- One branch is mis-costed.

Each has a repair that fits the design.

## Strongest part

The label T_i. I redid the proof:
- D_i is independent of H_i.
- Later luck terms have mean zero given a past that includes D_i.
- E[(D−p)z | H] = p(1−p)(E[z|1] − E[z|0]).

So E[T_i | H_i] = Δ_i for any pre-coin c_i and fixed β. Conditioning on h_i gives the information-state advantage, so there is no strategy fusion. Each label compares one decision followed by play that re-decides after every roll. A game's labels are nearly uncorrelated, so one result informs about 150 decisions, which gives the 39× over fork pairs (recomputed). Nothing before S1 depends on its one unknown, the in-span spread v.

## 1. Provenance audit

| Number | Class | Check |
|---|---|---|
| Variance kept 0.732 / 0.747 / 0.897 / 0.847; Var(z) 1.0004; Σt² 17.7; band share 0.339; `parity3` band 91.4; agreement 98.2%; flips 1.76%; −0.0018 per act; 9.63 / 8.58; 0.608; 285 per second; $2.1; +35.7 → +1.7 | measured | Records and scripts reread: as quoted. The −0.0018 interval spans 0. |
| 182.7 decisions per side-game | measured, wrong unit | It merges every move-then-attack; the share is per command reading. In the share's unit: 224.6 (`round5_simplify_critique_decision_units.out`, rerun), so the band holds about 76, not 62. Conservative. |
| σ_ε² 3.91; 31.7 / 0.82; 347.4; SE 2.65 and $1.90; $2.47e-5; 4.67 / 6.67; 61% / 32%; 0.023; 150 GB | derived | Redone: correct. |
| Floors, gain table, forecast SE 2.4 | derived | `round5_simplify_power.py` rerun reproduces them; the switch rule silently takes μ = 0 (2.1). |
| Plant SEs; critic re-entry $1–4 | derived | Wrong noise (2.3); incomplete (2.4). |
| P-distill agreement ≈ 98%; zero flips at G ≥ 0.35 | **guessed** | 2.2 |

## 2. What would make it fail

### 2.1 The rule and the forecast under a nonzero mean (fatal as written)

a0 is always the argmax, so the runner-up's mean advantage μ need not be 0; the proposal's own sensitivity case is μ = −0.0018. Its model rerun at 8 batches, d_eff 776 (`round5_simplify_critique_rule.out`):

| sd | Sign rule f > 0 | Calibrated rule μ̂ + β̂(f − μ̂) > 0 |
|---|---|---|
| 0.002 | −11.8 Elo | 0.0 |
| 0.003 | −7.1 Elo | +0.5 |
| 0.005 | +6.3 Elo | +8.7 |

Ĝ = β̂·Σ_switched f is positive whenever β̂ > 0. At sd 0.003 it forecasts a gain while the switches lose 7 Elo, and S2 blames second-order error. Adoption blocks the loss, but the miss points at the wrong stage. A ridge absorbs μ only partly: it shrinks toward 0, not μ.

Repair:
- an unpenalized intercept;
- the calibrated rule, calibrated on batch 4's progressive reading;
- the model-free forecast, held-out labels summed over the switches (SE 4.3 Elo), beside the plug forecast with its intercept (SE about 3.9; `round5_simplify_critique_checks.out`).

### 2.2 Kill lines on guessed numbers (fatal as written)

**P-distill.** It requires agreement at S0's replay level, about 98%. The one measured run of this recipe went from 0.672 to 0.862 agreement with its target (`docs/memory_own_trajectories_prereg_20261011.md`, "Measured"), closing 58% of the gap in one pass. A student starting near 90% (1.05 extra acts in 10.63 decisions per side-turn) ends near 96% at that rate, so a working distillation reads "broken". The 58% comes from a different target, so this is indicative. The stage also gives no rate prediction, which the plant rule requires. Repair: use the decision count (9.63) as the known answer, and predict agreement from the measured closure.

**S0's flip line.** It spends $7 on any flip at G ≥ 0.35. The record holds no per-decision flags. A per-game regression puts the flips below 0.1 nats and misfits the ≥ 0.35 bin with a negative rate (`analysis/round5_simplify_critique_extra.out` §1), so zero is neither confirmed nor bounded. One memory-drift flip in 360,000 decisions fires the branch. Repair: state the tolerance as its consequence on Ĝ.

### 2.3 Plants that miss the real path

Both P-coin plants go into a separate stream, but their bands use the result's noise. The stream's own noise makes the SEs 24 times smaller (`round5_simplify_critique_checks.out`), so the stated bands pass a 16% scale error in (a) and 55% in (b).

Neither plant passes through z, c_i or S_{>i}. A control variate read after the coin (the value head at i+1, off by one) absorbs part of every real effect and passes both plants, and I5 too. Repair: fit every batch with and without each control variate. For a valid c the difference has mean 0 and variance E[(c−c′)²]/(p(1−p)), far below the label's.

### 2.4 Costs

**Critic re-entry "with no new games".** The x_i are not stored, and a refitted student needs new predictions on the forecast half. The replay costs $8.9 per 400,000-game half, $17.8 for both (`extra.out` §4), before any critic forward. The $4 covers only a re-read of β̂ from logged predictions, which cannot improve the student. With re-entry as specified, the regimen passes $85.

**The early stop.** Under the proposal's model, batches 2–4 leave an upper bound of 7.5 Elo even at a true null. The $9.9 skip therefore fires with probability 0.27 at sd 0, 0.11 at sd 0.0015 and 0.02 at the floor (`extra.out` §2). A null round costs about $20.

### 2.5 Readouts that cannot mean what they say

- **"β̂'s lower bound ≤ 0 certifies the spread below the floor".** At the floor β̂ sits about 2.2 SE above 0, so the lower bound reaches 0 in about 42% of such rounds. Only an upper bound certifies.
- **The exploitation check.** Its PURE readings have a combined SE of 17.9 Elo, so its line sits at 36 Elo against forecasts of 5–25 Elo.
- **"Corrected and PURE agree within 2 combined SE".** On the same 12,800 games, the root-sum-square SE is 4.03 Elo but the paired difference's is 1.56 (`extra.out` §3). The check lets through a correction bias 2.6 times larger than intended.

### 2.6 The student's span

φ joins the actor token and the target-hex token. When a0 and a1 move the same unit:
- the actor difference is zero;
- the hex tokens do not know which unit moves;
- the network scores a target by the interaction q(actor)·k(hex) (`wesnoth_ai/model.py`).

The head then ranks hexes for any unit. Moves top 66 of the 91 band decisions (`round5_simplify_band.json`), but no record names the runner-up, so the same-unit share is unknown. This cuts v, the number the design lives on. Repair at the same width: q(actor) ⊙ k(hex) in place of the hex token, and a runner-up census (unit, kind, distance) in S0.

### 2.7 Premise, novelty, ruling

**The premise.** "No critic ranks alternatives better than material" cites a readout that ranked turns played with their own dice. There, T25 and T100 beat the HP margin in selection gain by 2.2–2.4 SE. Removing the critic stands on cost alone.

**The novelty.**
- T_i is the score-function estimate of a Bernoulli logit, divided by its Fisher information.
- Switching where its regression on x is positive is the greedy policy-iteration step under compatible approximation, which the natural gradient moves toward ([Kakade 2001](https://papers.nips.cc/paper/2073-a-natural-policy-gradient)).
- Random flips of adjacent pairs are FairPairs ([Radlinski & Joachims 2006](https://arxiv.org/abs/cs/0605037)).
- Many simultaneous perturbations read from one game result are SPSA, as Stockfish uses it ([chessprogramming](https://www.chessprogramming.org/Stockfish's_Tuning_Method)).

New: exact simulator-law control variates inside that estimator, in self-play under fog.

**The ruling.** This is outcome-driven policy gradient, which the seed lists as killed. Pooling across contexts answers the "19,000 games per context" arithmetic, but a "no" on ruling (a) leaves the design without a label.
