# Round 8 critique: the challenge stance (the skill axis)

**Verdict: repairable.** The rating axis is real out of sample, and S1 with M1 is a cheap, well-posed first gate. As specified, though:
- the conditioning input leaks each game's result, and at s_top it mostly names one player;
- a run length that was chosen, not derived, decides whether the route closes;
- one decision rule and the budget contain arithmetic errors;
- past the human range, the route rests on extrapolation that the evidence does not support, then falls back on the current design's estimator.

New checks: `analysis/round8_challenge_critique_checks.py` (C§n). The stance's three scripts rerun with byte-identical output.

## 1. Provenance audit

**Measured, checked:**
- **Rerun identical** (`round8_challenge_skill.py`): the regulars, their spread and percentiles, the winner mean +84.1, the out-of-sample log loss 0.6051 against 0.6931 (a sound 80/20 refit), the win shares by gap, and tempo +0.436 ± 0.051.
- **"Same factions and luck" is wrong in words.** The two sides have different factions, and the dice act on them in opposite directions. The number survives: with the faction difference regressed out, the gap is +0.383 ± 0.050 (C§3).
- **Confirmed:** test 3 never ran (escrow `fit.log`, `train.log`, `vs_seed.log`). The stop census is `parity3`'s. The figures 217 positions/s and $2.1 are in `docs/memory_own_trajectories_prereg_20261011.md` "Measured".

**Derived, redone:**
- **Correct:** the SEs, costs and miss lines (9.39 and 11.25), f ≥ 0.021, and ×1.845. The last is conservative, since the pool's server was busy 0.71 of its wall (`round7_provenance_price_tail.out`).
- **S3's "P 0.18" is wrong for the rule as written.** Φ(−2.8/3.07) is the acceptance under "estimate ≥ 0". The written rule, "not below its parent by 2 SE", accepts a 2.8-Elo deficit with P 0.862 and a 6.1-Elo deficit half the time (C§4).
- **The $46.02 budget omits** S4's always-follow match ($2.13) and its census.

**Choices that carry decisions:**
- **4 M positions [C] decides the route.** A null M1 closes the route "at this run size", but no readout says whether N learned to use s at all. So a null cannot separate "s was ignored" from "s does not transfer". This is fatal as written.
  - Zero-cost repair: compare the holdout loss with true s against s permuted among sides, and read M1 only once that gap is above 0 by 2 SE.
- **8 batches [C]** price S4, while the text sizes the batches from μ_g.
- **The plant's gate** ("above 0 by 4 SE") predicts no closure rate. Its s = +500 lies outside the human range, so the plant shows that the column reaches the logits, not that values from −178 to +253 carry signal.

## 2. What would make it fail

1. **s carries the game's own result.** `ratings.json` was fitted on every game, each side's own game included.
   - **The leak (C§1):** E[s | won] − E[s | lost] is +116 Elo in-sample and +70 out-of-fold; for one-game players, +148 against 0.
   - **Who sits at s_mid:** within ±30 Elo of s_mid, 0.201 of the training sides are one-game winners (rated +78 ± 21) and 0.561 are non-regulars. These sides won 0.699 of their games.
   - **Consequence:** N(s_mid) partly learns "a side that won", the outcome conditioning that §6 rejects.
   - **Repair:** out-of-fold ratings, and an "unrated" input below 30 games.
2. **s_top is mostly one player.** Within ±30 Elo of +253, 46 players supply the training sides; the largest supplies 0.605 of them and the largest three 0.779 (C§1).
   - **Consequence:** a scalar that differs for every regular can act as a player identifier. ℓ_top − ℓ_mid is then largely one player's style against a broad mixture, and guidance extrapolates that style. A positive M1 would certify a style, not an axis.
   - **Repair:** the permuted-rating placebo, which the stance defers to the $1,000 case, separates style from axis and belongs before any γ > 0 arm. Also jitter s by its posterior sd during training (median 52 Elo for regulars), and choose s_top for player coverage.
3. **The reserved s for own trajectories lies on the axis.** One scalar column has no "off the axis".
   - Placed below the human range, it tilts ℓ_top − ℓ_mid away from the memory fix's targets.
   - Placed above, guidance heads toward them.
   - **Repair:** an indicator column, which changes ruling (g).
4. **Guidance lacks a plausibility constraint.** Contrastive decoding ([Li et al. 2023](https://arxiv.org/abs/2210.15097), abstract read) restricts the contrast to plausible candidates, because the raw contrast favours implausible ones.
   - At γ = 2, a candidate at −3 nats under N(s_top) overtakes a near-certain top candidate once its contrast exceeds about 1.5 nats (C§4). Rare actions carry the noisiest contrasts, the more so given item 2.
   - This failure would be misread as "the axis bends".
5. **Two memories.** ℓ_mid needs N(s_mid) carrying its own memory along the guided trajectory. I-guide passes even with a shared memory.
   - **Repair:** an invariant that, on recorded guided games, the ℓ_mid stream equals N(s_mid) replayed with its own memory.
6. **The claim is a selected arm.** Taking the best of three on one match gives false adoption 0.067, against the standing 0.023, and a 2.6-Elo selection bias at equal arms (C§4).
   - **Repair:** fresh-seed adoption stands, so a confirmation match ($1.50–2.13) is needed.
7. **M1's sign is not a known answer.** It is the hypothesis: a negative reading closes the route rather than flagging a bug, and given items 1–2 even a positive sign is partly circular.
8. **The tempo link misses own1's lever.**
   - **For the stance:** at side-turns 2–4, own1 makes 7.56 actions per side-turn, against 8.32 for lower-rated regulars and 10.07 for the top quartile, faction-matched (C§2). So the axis has a tempo component own1 lacks.
   - **Against it:** acting more through the stop margin measures flat at own1. Offset −2.5 against −1.5 adds 19.4 decisions per game and reads −12.2 ± 12.3 Elo (`round5_provenance_stop_acts.out`).
9. **The stance misreads the current design.**
   - Its head gains from the band's spread, with floors stated at μ = 0 and μ = −0.0018, so a negative mean is no defect.
   - S4's rule "resolve μ_g at 2 SE" sizes the batches for the mean, which the always-follow match already reads; the head needs the spread.
10. **Beyond the human range,** the stance's own sources argue against extrapolation (UniMaia's reversion, Transcendence's 1500 cap).
    - An informal model card, [fastelo](https://huggingface.co/irsotarriva/fastelo) (unverified), reports that strength plateaus above an Elo input of about 2400.
    - Decision Transformer exceeded the dataset's best return only "sometimes" ([Chen et al. 2021](https://arxiv.org/abs/2106.01345), §5.2).
    - Past the top, the route is S4: the current design's estimator on another band. The mandate is beyond the strongest humans, and this stance's new part ends near them.

**Novelty.** The parts are covered:
- classifier-free guidance on a quality condition (Decision Diffuser);
- expert-minus-amateur contrast (Li et al.);
- Elo-conditioned game models (ALLIE, Maia-2, KataGo);
- prompting above the trained Elo ([KempnerInstitute/chess-research](https://github.com/KempnerInstitute/chess-research) sets this up and reports no result).

As far as I found, only the combination with match-certified γ steps is new.

**Timing.** S1 trains weights, so it waits for the optimizer discussion and for ruling (f). The stance says so.

## 3. Strongest part

The stance found a signal the project owns and never used, and showed it holds out of sample: on 3,405 held-out games the ratings cut log loss from 0.693 to 0.605. It also established that test 3 never ran.

Its first gate is cheap and coherent: a zero-initialised column, so that N starts as own1, then one single-network match between two settings (SE 1.79, $0.72), $3.32 in all. With repairs 1–3 and the s-permutation readout, S1 and S2a form an experiment in which every outcome says something specific. Guidance past the top and S4 need to be argued again.
