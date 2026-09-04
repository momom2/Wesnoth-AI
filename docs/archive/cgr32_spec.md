# CGR-32 — Certified Gumbel Re-decision

**Status: DRAFT for user review. Every decision below is a proposal, not a ruling. Two items explicitly require a user ruling before implementation (marked ⚖).**

Proposed location: `docs/cgr32_spec.md`. Synthesized 2026-08-26 from three adversarially-reviewed designs; base = CGR-32 (survived review with no fatal flaw), grafts from HAG and TCS-R noted inline and in §10.

---

## 1. Motivation

The 2026-08-26 2x2 matrix (40 games/cell, sims 32, same seeds, all decisive) is the ground truth:

| match | result | Elo |
|---|---|---|
| pin+MCTS vs seed+MCTS | 9-0-31 | seed +208 ± 64 |
| pin+TCS vs seed+TCS | 22-0-18 | pin +34 ± 55 |
| pin+TCS vs seed+MCTS | 6-0-34 | seed +290 ± 74 |
| seed+TCS vs seed+MCTS | 9-0-31 | MCTS +208 ± 64 |

Row 4 is the verdict: **on identical weights, TCS as a play procedure is ~200 Elo weaker than per-decision Gumbel MCTS-32.** Leg 5 trained the policy toward TCS-refined play — the teacher was worse than the prior. CGR-32 therefore makes the measured-strongest procedure BOTH the data generator and the deployed player (deployment sampling = training sampling, user ruling 2026-08-26), and spends all new engineering on certifying the distillation target and instrumenting the leg.

Honesty about the causal mass: the matrix does not prove teacher-weakness carried the whole ~200. If training had distilled toward TCS-refined play, pin+TCS should clearly beat seed+TCS; it reads +34 ± 55 (CI includes 0). The numbers decompose at least as well as generic prior erosion (~200 through either deployment) plus a small TCS-alignment term. So the base case is: CGR removes the one identified bad teacher AND installs gates that catch a generic erosion channel earlier and cheaper than leg 5 did. It does not claim to have identified the erosion channel.

Also priced in: the matrix was not compute-matched (~113 TCS vs ~384 MCTS forwards/side-turn); part of the +200 may be budget. This does not change the design choice — the deployed player pays the full budget either way — but tempers the expected leg outcome.

## 2. Play-time procedure

Executed identically in generation, eval, and deployment. One search per micro-action, K≈12 per side-turn. This is the current default `tools/mcts.py` path; steps marked (=) are shipped behavior, cited by the review against source.

1. (=) Snapshot: deepcopy the observable `GameState` (live-gs identity check raises, `mcts_policy.py:356`); fork the live sim. `SIM_FORK_GUARD=1` in rung-0 smoke and rung-3's first iteration; OFF in steady-state generation (two full fingerprints per search is not free).
2. (=) Root expand: one forward → factored priors over all legal actions (action_sampler enumeration; annealed combat-oracle bias mirrored by the trainer reforward; note the end_turn prior bias is env-gated, mini-category-only, default OFF — it is NOT part of this procedure). Legality mask = pure function of observable state; `_NOOP_KEY`/`_STEP_ERROR_KEY` sentinels unchanged.
3. (=) Gumbel root: draw m=16 distinct candidates by g + log(prior). No force-inclusion of anything; end_turn competes for a slot like every action (force-inclusion is safe only under a linear link, which this design does not use — see §10/HAG).
4. (=) Sequential halving, B=32 sims over 4 phases; each sim pinned through its root candidate, PUCT descent below (c_puct 1.5, FPU as shipped), one forward per leaf, sign-flipped visit-averaged backup. Descents cross the turn boundary freely: an end_turn edge's child is the opponent's decision node.
5. (=) Chance nodes ON, fresh-salt re-forking (unsalted fork = hard error), exact outcome enumeration at the shipped mass threshold.
6. (=) Halving score g + log(prior) + sigma(completed_q); play = argmax over final survivors; unspent budget handling as shipped (exact spill behavior to be cited from `mcts.py` during implementation, not asserted here).
7. (=) Commit exactly ONE action. Fresh search next call; tree_reuse iff live state_key matches (combat almost never reuses). No plans, no incumbents, no warm re-plans.
8. Record one `_PendingMCTSState` per decision with plan-time decision_step; finalize_game, z-perspective, game_weight, holdout diversion, replay buffer unchanged.

Honest scope note (review-forced correction): boundary blindness is **reduced in amplitude, not structurally eliminated**. Phase-1 gives all 16 candidates a single visit; a 1-visit Q IS one boundary-blind value forward. Only multi-visit survivors mix opponent replies. The matrix says this amplitude is empirically sufficient at play time; §3 keeps 1-visit reads out of the *targets*.

## 3. Training-target construction

Base: shipped `extract_gumbel_policy_target`, lam pinned 1.0 — pi(a) ∝ softmax(log prior(a) + sigma(completed_q(a))) over all legal actions. Three modifications, all config flags wired through all three generation paths:

**(a) ⚖ Min-visit completion** (`--distill-min-visits 2`, graft from the HAG verdict's repair clause): edges with fewer than 2 visits are completed at v_mix exactly like never-sampled actions; only phase-2+ survivors carry measured Q into the target. Effect: (i) the tried-and-cut tax vanishes at its largest surface — ~14/16 candidates cut on 1-2 samples can no longer grade below the v_mix shelter; (ii) no single boundary-blind forward ever teaches; (iii) the winner's-curse selection noise of phase-1 allocation is excluded. **This revises the standing extraction-unchanged ruling** (which was justified by a per-state below-detection-floor measurement, not the cumulative effect at 14-cut-actions-per-decision scale). Needs the user's explicit go; default until then = 1 (current behavior).

**(b) Snap deadband** (graft from HAG): if the completed-Q spread over the evaluated set is < 0.04 (one C51 atom), emit **no target** (`recorded=False`) — not a proportionally faded sigma, which at exactly one atom of spread still granted the full ~5.5-logit span (the review's correction of the original CGR claim). Expected to bind rarely at m=16; the real noise guard is (c) plus the rung-2 shuffled-Q audit.

**(c) KL trust region armed ON** (`--distill-kl-max`, default ON in the leg config — reversal of original CGR's default-OFF, which left rungs 3-4 running the unbounded configuration): per-state binary-search alpha ∈ (0,1] scaling sigma so KL(target‖prior) ≤ cap. It can only shrink toward the prior, never flatten or invert it. Mutually exclusive with any lam<1 (Grill-stacking forbidden; the leg-4 lam ruling stands). The cap value must be **derived before rung 1** from the measured prior-gap scale on the committed state set and entered in `docs/design_constants.md` — a hand-picked cap is the magic-number class of failure. Rationale for arming: at lam=1.0 the equilibrium logit gap is unbounded (`mcts.py:328-341` derivation) and the 2026-08-05 one-hot end_turn collapse occurred at exactly this setting; the sigma floor fixed only the noise component, not consistent-bias integration.

### Non-degradation property

**Property (per-state, in expectation):** if completed-Q at a state carries no information about action quality — noise exchangeable across evaluated actions — the emitted target equals the prior, or no target is emitted.

**What is proved and what is a condition:**
- *Exact:* under the deadband, or with sigma ≡ 0, target = prior identically (softmax(log p) = p). lam=1.0 means no flattening term exists anywhere — the leg-4 killer is structurally absent.
- *First order:* softmax is locally linear at sigma=0; under exchangeable noise with **equal exposure** across evaluated actions, the first-order expected deviation cancels. Second-order terms are O(sigma²), bounded by the KL cap.
- *The exact condition needed:* evaluated-set membership and Q-exposure must be exchangeable with respect to the noise. Sequential halving violates this by construction (allocation is Q-adaptive). Modification (a) restores it at the largest surface (only ≥2-visit survivors teach); **residual selection bias among survivors remains and is not proved away.** It is measured, not assumed: the rung-2 shuffled-Q audit puts an empirical number on the operator's noise-sharpening floor, with a pre-registered separation bar.
- *Not covered by any target transform:* a systematically biased judge (tempo, unit-count — both measured) that mis-orders survivor Q coherently. Hedging reduces variance, not bias; both self-play sides share the bias, so internal telemetry can stay green while Elo burns (the leg-5-resume signature). Only the game-level gates catch this, and the ladder is honest about their resolution (§7).

## 4. Failure-mode coverage

| # | Failure mode | Mechanism | Honest residual |
|---|---|---|---|
| 1 | Boundary blindness | Survivor Q mixes adversarial replies through the tree (end_turn's child = opponent node); 1-visit boundary reads excluded from targets by min-visit rule | 1-visit reads still steer *cuts* at play time; frozen pass-bias meter (§6) watches drift |
| 2 | Goodhart at full gain | Visit-averaged Q over ≤16 prior-plausible candidates; per-action amplitude with re-observation between decisions; KL cap bounds per-state target movement | Shared systematic bias caught only by Elo gates; "per-action amplitude" ≈ turn amplitude minus re-observation, no more claimed |
| 3 | Commitment vs stochasticity | One micro-action per search, full re-decision after every realized roll; chance nodes + exact enumeration; no incumbents | Eliminated (review-confirmed, not relabeled) |
| 4 | Compute asymmetry | Teacher = player = ~396 fwd/side-turn at K≈12, same budget as the winning matrix arm; counted per-side-turn forward telemetry (TCS-R graft), never nominal sims | Matrix itself was not compute-matched; expected gain tempered accordingly |
| 5 | Validation trap | Match-first ladder; no self-graded statistic is a pass criterion anywhere; dark-instrument-invalidates-the-rung clause; do-not-preempt-the-gate clause | Gate resolution is ±60 Elo per 40 games (§7 prices this) |
| T | Truncation | end_turn priced through opponent subtrees in survivors; snap deadband kills the noise ratchet at emission; `--abort-k-median 10` explicit; per-iteration frozen p(end_turn) + et_target-vs-et_prior slope tripwires; pass-bias meter; depth-1 projection pre-planned as the rung-5 response if the meter and K trend adversely | MCTS-era K collapse never root-caused; attribution to the legacy 1e-8 floor is honest-uncertain |
| TC | Tried-and-cut tax | ⚖ Min-visit v_mix completion removes the tax structurally (cut edges keep exactly shelter mass); paired-state prior-mass tripwire stays armed; midgame-recruit prior-mass frozen series | If the user declines (a), fallback = tripwires only, and the tax runs at maximal surface (14/16 cut per decision) |

## 5. Cost model

Per side-turn at K≈12: K·(1+B) ≈ 396 forwards (TCS ~113, MCTS-32 ~384 — same family, so rung-1 equal compute holds by construction). Sim steps ~13% of a forward; chance forking CPU-only. Binding resource = the ~54 fwd/s pool inference server: **~3.5x fewer games/hour than TCS legs.** Consequences priced, not hidden: actor-pool iteration deadlines MUST be re-sized in the leg config or the leg-3 survivorship filter (17/21 iterations discarding 30-60% of games) recurs silently; discarded-game fraction goes on the abort dashboard with a <10% bar. Budget reductions (B=16, ~204 fwd) permitted only after a rung-1 re-run at that budget. No playout-cap randomization (deployment = training sampling ruling). Money: rungs 0-2 ~$3, rung 3 ~$4, rung 4 ~$20-30 at ~$0.37/h — the original $15-25 was tight against the 3.5x throughput hit plus gate evals; worst-case waste per unidentified-erosion recurrence ≈ one inter-gate interval, realistically detected ~160k steps (§7), ~$10-15.

## 6. Implementation plan

| Module | Change | Est. diff |
|---|---|---|
| `tools/mcts.py` | `extract_gumbel_policy_target`: min-visit completion flag, snap deadband, KL projection (binary-search alpha); wire `search_q_spread`/`overturn_frac` (dark since leg 4) | ~200 lines |
| `tools/mcts_policy.py` | Config passthrough, telemetry drain as rates via `drain_distill_stats` | ~50 |
| `tools/frozen_probe.py` (new) | Committed ~256-state audit (stratified by turn phase, stored in-repo): target stats, shuffled-Q twin, pass-bias meter (raw boundary V vs depth-1 projected V for end_turn on fixed states — HAG's bias meter as an offline probe, zero play-time cost) | ~300 |
| `sim_self_play.py`, `selfplay_worker.py`, actor_pool | Forward new flags on all three generation paths; extend the flag-symmetry test to MCTSConfig | ~60 |
| `configs/leg_l6.json` | Asserts every ruling at startup; launcher emits nothing unconditionally (leg-4 config-drift class); startup fails on mismatch — the assert must be written and tested, not assumed | ~40 |
| `tools/elo_ladder` / `run_elo_batch.py` | Per-side-turn counted-forward telemetry in match output | ~40 |
| Tests | Deadband, min-visit, KL-cap monotonicity (alpha shrinks KL, never inverts ordering), symmetry, frozen-probe determinism | ~200 |

Total ~900 lines. `tools/turn_policy.py`/`turn_search.py` untouched, retired from generation via config, kept in-tree for the rung-5 macro-edge arm.

## 7. Pre-registered validation ladder

**Rung 0** (local, ~$0): unit tests; every new telemetry counter drains nonzero through the actor-pool path; `SIM_FORK_GUARD=1` 10-game smoke. KILL: any dark counter or path asymmetry.

**Rung 1** (mandatory first, ~$1-2, no training): **40-game seed-vs-seed match, seed+CGR-32 vs seed+MCTS-32 catalog protocol, equal measured forwards/side-turn, same seed schedule, idle GPU** (`--device cuda`; CPU measured 27/40 timeouts), decisive-only BT fit. Since the play path with target-side flags is byte-identical to catalog MCTS-32, this is a **harness/config identity null test**: it must read ~even. KILL: CGR side loses with CI excluding 0 → implementation or config defect, root-cause, never tune around. Honest limit: certifies only against >~150-Elo gross defects; that is what it is for.

**Rung 2** (<1h idle GPU, no training): frozen-state target audit on the committed 256 states, **two-sided** (repair of original CGR's one-sided rung 2):
- KILL low: mean sharpen_top < 0 (net-flattening, the leg-4 signature) or et_target > et_prior with t>3.
- KILL high: sharpen_top or KL(target‖prior) above pre-registered upper bounds (values fixed from the seed-state baseline before the rung runs, recorded in the leg doc) — the over-sharpening direction lam=1.0 leaves open.
- **Shuffled-Q twin** (TCS-R's placebo graft): recompute targets with permuted completed-Q assignments. KILL if real-Q target statistics are indistinguishable from shuffled (pre-registered separation bar) — an operator that moves as much on noise as on signal is distilling noise, regardless of sharpen_top's sign.
- Record baselines: deadband-emit fraction, KL-clip fraction, pass-bias meter, midgame-recruit prior mass.

**Rung 3** (~$4): ~40k-step leg from the seed. Per-iteration frozen-probe deltas with slope tripwires (prior entropy, top80, p_et, sharpen_top, pass-bias meter); `--abort-k-median 10`; value tripwire = 3-independent-redraw design; decisive-rate + CPU watchdogs; discarded-game fraction <10%; any pre-registered instrument reading empty invalidates the rung. A 40-game match vs seed runs as a **gross-failure barrier only** (kill iff seed better with CI excluding 0); at ~32 Elo of expected drift it cannot certify health and is not claimed to.

**Rung 4** (~$20-30): 250k-step leg, launcher-wired qualify gates at **80k / 160k / 250k**, 40 games each vs seed, both sides CGR-32 — which IS the catalog MCTS-32 protocol, so pre-2026-08-26 catalog numbers (seed +211, 2516k +140) stay comparable; every entry carries its protocol tag. KILL at any gate: seed better with CI excluding 0. Pooled 120 games (~±37) is the secondary read at 250k. PASS: point estimate ≥ 0 at 250k. Honest limit, stated up front: a genuinely −50-Elo leg can pass one gate cycle; the guard is that every subsequent leg re-gates against the same seed baseline, so cumulative erosion becomes detectable across legs. The gate is the verdict — no proxy story preempts it in either direction (leg-5 lesson runs both ways).

**Rung 5** (only after rung-4 pass; one variable per arm, 40 games each, losers dropped not tuned): (a) end_turn depth-1 projection at the root (the HAG mechanism, pre-planned response to an adverse pass-bias trend); (b) KL cap off; (c) B=16 at matched wall-clock; (d) one whole-turn continuation as a 17th Gumbel candidate (`tcs_spec` §6 Mode 2 — the sanctioned test of turn-scale candidates on top of per-decision search).

## 8. Open questions

1. ⚖ Min-visit completion (`--distill-min-visits 2`) revises the standing extraction-unchanged ruling — user go/no-go required. If declined, the tried-and-cut tax runs at 14/16-cut scale under tripwires only.
2. ⚖ KL cap default ON and its derived value (design_constants entry due before rung 1) — user sign-off on both the arming and the derivation.
3. Was part of the matrix's +200 budget rather than shape? A 40-game seed+TCS-at-384-forwards arm would answer it (~$2); optional, does not block the ladder.
4. If rung 4 erodes anyway, the teacher-weakness hypothesis is spent; next suspects are self-play distribution narrowing and the trunk proxy rotation (leg-5 value diagnosis). Postmortem headline = Elo per 100k steps.
5. Actor-pool deadline values at 3.5x per-game cost — set from rung-3 measured wall-clock, not estimated.
6. Value-head supervision density is unchanged (a target-or-not decision does not gate `_PendingMCTSState` value supervision under this design's every-decision recording) — verify at rung 0 rather than assume; TCS-R's starvation failure must not be reproduced by the deadband path.

## 9. Rejected alternatives

- **TCS / TCS-R (refereed turn commitment):** whole-turn commitment measured −200 Elo as a play procedure (row 4); the refereed gate re-ships the projection-accept configuration already measured at 1.0:1 real:placebo separation at the same n=3 sample count, and CRN pairing is void because dice streams decohere at the first divergent action (Q8: median zero matched fights). Salvaged: counted-forward telemetry; shuffled-grade audit → rung-2 twin.
- **HAG (linear link + end_turn projection on the MCTS path):** the linear link's exposure invariance was derived for non-adaptive evaluation and breaks under sequential halving (winner's-curse selection; convex clip grants expected mass to the noisiest edge — a new end_turn ratchet). Play-time depth-1 projection at 1-3 rollouts carries the 1.0:1-separation precedent and mixes pre/post-reply estimands with a measured class-dependent skew. Salvaged: snap deadband, offline pass-bias meter, min-visit repair idea, projection as rung-5 arm.
- **lam < 1:** the leg-4 killer under the linear link; never re-derived under the exp link. KL cap is the sole permitted damper.
- **end_turn force-inclusion:** safe only coupled to a linear link; not carried.
- **Playout-cap randomization for throughput:** violates deployment = training sampling.
- **Policy/value anchors:** default OFF, user ruling 2026-08-26.
- **Gate projection as an accept gate:** parked; separation 1.0:1 as implemented.

## 10. Provenance of grafts

From CGR-32 (base): procedure, target family, ladder shape, estimand self-consistency, compute honesty. From HAG: snap deadband, pass-bias meter, min-visit exclusion (via its verdict's repair clause), depth-1 projection as the pre-planned rung-5 arm. From TCS-R: counted per-side-turn forward telemetry, shuffled-grade audit as the rung-2 twin, deadline/survivorship pricing. Review-forced corrections to the base: boundary-blindness claim downgraded to amplitude reduction; KL cap armed by default with a derived value; rung 2 made two-sided; ladder resolution and money priced honestly; env-gated flags (end_turn prior bias, SIM_FORK_GUARD) no longer described as standing.