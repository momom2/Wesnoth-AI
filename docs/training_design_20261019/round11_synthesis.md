# Round 11 synthesis

## Verdict

**The design changed in substance, and the main cause is the user's rulings of 2026-10-21, not this round's analyses.**

The round-10 design was a near-tie coin experiment at argmax: a closed-form head on the frozen network, and a recruit rule played beside it. The rulings removed all three:
- the player samples;
- nothing beside the network decides play;
- the coins are not a training mechanism.

What survives is the label algebra (the result less a frozen cell baseline and the dice's exact luck), the census and prices, the claim machinery and the planted-signal discipline.

The four stances converged on the same update without coordination: on-policy policy gradient at temperature 1 from `own2`, every decision labelled by its side's luck-corrected result, no critic, no anchor, the current self as opponent. They differed on how the step is sized, how it is certified and what the optimizer is.

This round's own contributions are those choices:
1. **One step per round, on a single gradient.** Each round makes one step from one gradient: a direction from the round's summed gradient through a metric, scaled to a KL from π_k. It replaces a minibatch pass with a learning rate (challenge, invent, simplify). This makes the optimizer comparison the update itself, run on one recorded gradient.
2. **The step is read on the true objective, offline.**
   - Three radii are read by trajectory importance weights on held-out games, selected on one half and confirmed on the other.
   - A step is reverted only on confirmed harm.
   - Under the quadratic step model the ladder keeps 0.62–0.90 of the best round's gain over 12 rounds from a radius 1/16 of the best. The challenge stance's ratio rule keeps 0.39–0.83, and a fixed step 0.44 (`analysis/round11_synthesis_plan.out` §6; `round11_challenge_critique_rule.out`).
3. **The critical batch is corrected and read cheaply.** For a KL-normalized step it is B = tr(PFPΣ)/(gᵀPFPg) (`round11_provenance_critique.md` §2.1). Stage C reads it from the KLs of eight split-accumulator directions against their sum, on a 5% subsample.
4. **Temperature probes.** The sharpening slope comes from stage 1's logs alone: the first-order test of the user's hypothesis that self-play claws back what sampling loses. A floor on the attainable gain per unit of KL comes from the heads' temperature directions, which are weights of the network.
5. **The yardstick is `own2`.** It became the reference on 2026-10-21 (`configs/reference_player.json`, BACKLOG 0.31.4); every stance still named `own1`.

No box ran this round. Nothing measured changes what the record says about strength.

## What the round's analyses moved

| Reading | Source | Decision moved |
|---|---|---|
| Ladder 0.62–0.90 of the best gain; ratio rule 0.39–0.83; fixed 0.44 | `round11_synthesis_plan.out` §6; `round11_challenge_critique_rule.out` | The ladder of three radii with revert-on-harm |
| Ratio rule under a null: no "two reverts" stop in 0.994 of 12-round chains | `round11_challenge_critique_rule.out` | A pause rule on the last three confirmed readings: a null pauses at 0.977, a 3.8-Elo round passes at 0.8 (`round11_synthesis_plan.out` §4) |
| The sign-flip gate's z overstates resolution: 15.9 against a null, 4.0 on the full SE, r_eff 18.9 | `round11_simplify_critique_rank.out` | No gradient-norm gate; stage C reads the held-out gain directly |
| χ²/(2 KL) 1.0–6.0 when a step raises rare actions | `round11_invent_critique_checks.out` §1 | Every SE of Ĝ and ΔĴ is read empirically; the ESS share is checked |
| The bandit-converted arm's chord SNR is about 0.15 at B = 10⁵ | `round11_invent_critique_checks.out` §5 | That arm is rejected |
| Turn-start R² 0.346; forks ≤ 1.53× (provenance); 1.19× over a value-head baseline (challenge) | `round11_provenance_turn_signal.out`, `round11_challenge_forks.out` | Forks rejected |
| Setup-varying luck coefficients +0.0007 ± 0.0002 of variance | `round11_invent_luck_strata.out` | No chance head |
| Census at argmax: 475.7 decisions a game, 237.9 a side-game | `round11_provenance_census.out` | Prices; K per decision |
| Stage costs: 1 $1.65; C $2.73–5.71; P $0.65; a round $2.78–3.97 at N 20,000; 18–27 rounds | `round11_synthesis_plan.out` §2 | The regimen and budget |
| Claim SE grows at lopsided win rates: 5.4 Elo at 0.09, not 3.07 | `round11_synthesis_plan.out` §3 | The transfer check's line |

## What changed, by stance

### Provenance (certified policy gradient; critique: repairable)

**Kept.**
- One step per batch, from the reduced AdamW.
- KL-normalized candidates.
- Held-out trajectory importance sampling (ΔĴ) with the luck terms inside, and the split of H into selection and confirmation.
- No auxiliary loss.
- No anchor, with a magnet or a pool on a transfer miss.
- B measured before money depends on it.
- The B1 measurement, as D₀ against the new reference.
- B2 (the memory at temperature 1), as a report.

**Repaired.**
- **The critical-batch formula** (critique §2.1).
- **The loss.** The provenance stance said the loss existed in the sequence trainer; it does not. The masked joint log-probability exists as `legal_log_probs` in `wesnoth_ai/own_loss.py`, but its `prob > 0` filter drops negative weights. The signed weight is applied outside it. A legal-mass check catches any joint renormalization the masks miss (critique §2.2).
- **Adoption.** Certify-to-adopt was underpowered (a 3-Elo step adopted 9% of the time). It becomes revert-on-confirmed-harm, with claims checking the sum (§2.3).
- **The plant now passes through the result table and the production join** (§2.4).
- **KL-search costs are priced** (§2.5).

**Not kept.**
- "Muon predicted to lose." The noisy quadratic model's abstract says preconditioning raises the critical batch; it does not say preconditioning hurts at small batch.
- K = 1/16 as a fixed rung.

### Challenge (certified steps; critique: repairable)

**Kept.**
- Ĝ, the held-out first-order forecast free of every baseline. Its critique named this the strongest part.
- The free control variates (the value head's reading, the opponent's surprisal), each admitted on measurement.
- The trust-region idea that games, not the surrogate, size the step.
- One pass per round.
- Logging refused recruits.
- The replay identity and the orthogonality checks (critique §2.3).
- The score of inverse temperature as a free measurement. The critique found it; it is now the sharpening slope.

**Repaired.**
- **The radius rule** became the ladder. The ratio rule froze at low z and overshot under its ×4 cap.
- **The null case.** It now pauses instead of spending the budget silently.
- **The planted identity** is a z-test, not float tolerance.
- **The plant** enters the result in side A's frame.

**Not kept.**
- Training on the cross-play games' new seat.
- The parity memory plant (likely unlearnable).
- The belief loss at the recipe's weight.

### Invent (exact-law advantage walk; critique: repairable)

**Kept.**
- The label identity: any frozen linear walk of centred events keeps the label unbiased.
- Measuring before training.
- The chord-scaling idea, which became KL scaling of a single direction.
- No anchor.
- `own2`'s AdamW as the default.
- The fine-tuning evidence from Liu et al. 2025, read as the critique corrected it.

**Not kept.**
- **The action block of the walk.** Its expectations over every legal action are not computed by the generator, and its saving is bounded by 1/r. Only the free columns (v̂, surprisal) are tested.
- **The chance head.**
- **The bandit-converted bench arm.**
- **The two-seat forecast against a one-seat match.** The forecast now reads one seat's switch, averaged over seats.

### Simplify (plain REINFORCE; critique: repairable)

**Kept.**
- The minimal label.
- z = 0 for caps, replacing its −1 for both, which is not zero-sum.
- The P1 identity at the logits.
- I-count.
- The antisymmetric match from its critique (§2.3), now used for the sharpening slope.
- I-sample from its critique.
- The cost case against data reuse.
- The cost-optimal batch with a fixed per-round cost.

**Not kept.**
- **A fixed learning rate per segment with fresh moments.** Each segment's first steps are lr·sign(g) (critique §2.1).
- **The |G|² gate** (critique §2.2).
- **The end_turn extrapolation to −1.5** (§2.3).

## Record corrections

- **Ruan et al. 2026 has four versions.**
  - v1 (17 July 2026): single seed, Muon 0.290 → 0.546 success with GiGPO, no magnitude-matched control.
  - v4 (2 August 2026): a full-budget RMS-matched control removes the late-success gain, and tuned AdamW nearly matches at 3B.

  The provenance critique read v1, the simplify critique v4. Both were right about the version they read; v4 is current.
- **Liu et al. 2025, Moonlight.**
  - Table 6: the AdamW-pretrained model fine-tuned with Muon leads AdamW fine-tuning on 3 of 4 benchmarks.
  - Table 7: on Qwen2.5-7B, Adam leads on all 4.

  The authors read both as no significant advantage for a mismatched optimizer.
- **The built Muon arm is already RMS-matched to AdamW** (`scripts/optimizer_bench_box.sh` header).
- **The trainer's speeds.** 1,477 positions a second is configuration B on 40 timed windows. `own2`'s actual pass ran about 197 a second on the older trainer (`round11_simplify_critique.md` §1).
- **The invent stance's ridge cost.** With turn bins its columns are 280, not 70: 2.8% of r, not 0.7%.
- **The combat-oracle decode biases are off** (`COMBAT_TARGET_ALPHA = COMBAT_TYPE_ALPHA = 0.0`, `wesnoth_ai/constants.py`; the server's priors refuse them otherwise). This was checked because the rulings forbid a component beside the network. Stage 1's I-sample checks the rest of the decode against the logged law; no other decode path was audited here.

## Open questions

1. **For the user** (`optimizer_discussion.md`):
   - the update;
   - the metric;
   - the step rule;
   - the human prior;
   - the yardstick's decode;
   - whether stage C counts as a measurement;
   - the ceiling for rounds;
   - whether the $5 window counted.
2. **A memory-credit plant.** It needs a bit the current observation hides. The cheapest candidate is a planted reward on a choice conditioned on an earlier decision's weapon index, whose visibility in the current observation would first be measured on the frozen network. It is not designed. The user's rule asks a planted check of every training path. The memory's back-propagation is the recipe's, validated by the own-trajectory rounds, but the user may still require this plant.
3. **Distance.** How far must the policy move to repay D₀? If the debt is spread over many decisions, rounds of at most K = ½ per side-game, read offline, may be too short. Past that, each round needs a cross-play match ($0.88).
4. **Cap scoring at temperature 1.** If stage 1's cap rate exceeds 0.009, scoring caps as draws in labels and as absences in claims trains and measures different objectives.
5. **Muon as an optimizer.** Stage C reads Muon as one orthogonalized step on a summed gradient. Its usual form, orthogonalizing each minibatch step, is not tested. That form matters only if the update moves to minibatch passes.

## Unknowns that remain

| Unknown | Measured where | What relies on it |
|---|---|---|
| Game length, cap rate, price at temperature 1 | Stage 1 | Every price; cap scoring |
| E[R²] at temperature 1 and each correction's saving | Stage 1 | Which corrections enter; every SE |
| D₀ | Stage 1 | The claims' origin; the size of the task |
| The sharpening slope | Stage 1 | The user's hypothesis at first order; the link match |
| Memory at temperature 1 | Stage 1 | Report |
| Legal mass, replay identity, forward-only speed | Stage 1, C0 | Gates; C's arm list |
| B in the step's metric; the gain curve | C | Round size; whether rounds resolve |
| The best metric; ε₀; the head-temperature floor | C | The optimizer; round 1's radius |
| Second-order loss against the radius | Rounds' ladders | The radius |
| Transfer to the reference | Claims | Continue or stop |
| Exploitability; human level | None | Nothing |

## Research targets for round 12

- **Policy-gradient fine-tuning from imitation in stochastic games of long horizon**, with measured sample counts against the supervised start: AlphaGo's RL policy (Silver et al. 2016, read directly), AlphaStar's league ablations, and recent self-play work on imperfect-information games. Find cases where outcome policy gradient from a human prior failed.
- **Importance-sampled certification at larger KL.** Per-decision importance sampling, or weights truncated at the memory's window, could extend ΔĴ past K = ½ without a cross-play match. A truncated product is biased, so look for exact forms that exploit the opponent's ratio being 1.
- **The distance question.** For imitation policies made stronger by sharpening, how much per-decision KL did the improvement take? Look for chess and Go policy networks, and for language models after RL fine-tuning.
