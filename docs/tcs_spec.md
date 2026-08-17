# TCS — Turn-Commitment Search: approved spec (2026-08-13)

**Status: INTEGRATED, DEFAULT ON (user ruling 2026-08-14).** The
rung-0/1 probe ran 2026-08-14 (300 ladder states, imitation seed +
F1-arm final, cheap CPU box, ~$0.4): revalidated accept 0.640/0.460,
median accepted Δ 0.070/0.106 (~2 C51 atoms), placebo 0.130/0.180,
ρ(Δ,survival) 0.016/0.061, naive-vs-revalidated under the 2×
tripwire. The KL gate (perturbation-magnitude proxy) FAILED as
pre-registered on both arms (matched 0.184/0.383 vs Gumbel baseline
0.343/0.463); the user ruled PROCEED with the gate recorded as
failed-and-disputed. Rung-0 re-registration: **imitation seed plays
K≈11.5–15.4 (end_turn 6–8.5%) on ladder; the F1-arm self-play
policy plays K median 2–4.5** — turn truncation is a property the
self-play lineage ACQUIRED, contradicting the history-CSV "curing
itself" reading and restoring the design's original motivation.

## Integration (2026-08-14): what shipped, and deviations

- `tools/turn_search.py` — shared core (spine, materialization,
  two-stage acceptance, `tcs_target_distribution` reusing
  `_gumbel_sigma` verbatim). The probe imports from it, so
  measurement and production cannot diverge.
- `tools/turn_policy.py::TurnCommitPolicy` — SUBCLASSES MCTSPolicy;
  everything downstream (pending schema, z-sealing, per-side game
  weights, aux/ml targets, holdout diversion, harvest, buffer,
  train_step, checkpoint I/O) inherited unchanged. Chance-node
  re-planning is exact: each coordinate's expected pre-state key is
  stored; a live mismatch (diverged combat outcome) triggers a warm
  re-plan with the remaining commands as incumbent. Recruit bounces
  invalidate the plan via the `drop_last_pending` override.
- Wired through ALL THREE generation paths (`sim_self_play` inline,
  `selfplay_worker` spool + `_cmd_tail` forwarding, `actor_pool`
  pickled turn_cfg); flag surface `--turn-*`, symmetry pinned by
  `tests/test_turn_policy.py::test_turn_flag_symmetry_across_paths`.
  `--no-turn-search` restores the per-decision Gumbel generator.
  Eval tooling (elo_ladder `mcts:` players, sim_demo `--mcts`) is
  deliberately unchanged.
- **Deviation 1 — no boundary value-only experiences** (spec §4).
  With TCS on both sides, every turn-boundary state is already the
  opponent's coordinate-0 experience; extra value-only rows would
  duplicate states AND require the normalizer/harvest changes §4
  warns about. Zero trainer changes shipped. Follow-up knob if the
  value head needs it: upweight coordinate-0 experiences.
- **Deviation 2 — reply arm default OFF** (`--turn-reply none`;
  spec §3 said ON). It is implemented (`reval`/`all` modes) but
  unmeasured; turning unprobed machinery on by default contradicts
  measure-before-deciding. It is the next single-variable A/B, and
  §5.3's "sole guard" ruling makes it the first knob to flip if the
  leg drifts passive under healthy tripwires.

  **2026-08-17 addendum — multi-turn projection** (user directive
  "implement multi-turn planning, default off"; leg-3 postmortem).
  The reply arm is generalized to `--turn-project` in
  `tools/turn_search.py::project_value`: candidate turns are graded
  by the value **H half-turns past our boundary**
  (`--turn-project-halfturns`, default 1), each half-turn played
  closed-loop by the same policy — a single line, no branching, so
  cost is LINEAR in depth, not exponential. Per half-turn the mover
  plays ≤ `--turn-project-max-actions` (default 40) then `end_turn`
  is forced, keeping half-turns well-defined (deviation from the
  old reply arm, which evaluated mid-opponent-turn at its 4-action
  cap). Placement keeps the reply vocabulary: `reval` grades only
  the stage-2 acceptance pairings (guard placement: the climb
  proposes by the cheap boundary objective, the gate re-grades with
  projection — tempo-blind wins like premature `end_turn` die at
  the gate); `all` additionally drives stage-1 selection and the
  distill targets (the training signal itself learns the projected
  ordering). The stage-2 deterministic-pair shortcut is disabled
  whenever projection is on (projection rollouts sample the policy,
  so grades never replicate; and under `reval` the gate MUST
  re-grade). `--turn-reply X` remains as a deprecated alias for
  `--turn-project X` at depth 1 with the old action cap.
  **DEFAULT OFF** — unmeasured; motivated by the leg-3 turn-length
  collapse (K median 12→2 over ~12 iterations with draws rising to
  0.75: boundary-only grading let the search exploit the value
  head's tempo blindness through the force-included `end_turn`
  alternative). Telemetry: `TurnPlan.projections` →
  `tcs_projections` in `drain_tcs_stats`.
- CRN identity keying (gate 0e) remains DEFERRED — the probe found
  revalidated improvements without it; it is a variance upgrade,
  not a correctness fix, and touches the bit-exact sim.
- Playout-cap analog: `--turn-full-prob 0.25` (full budget +
  targets); fast turns run `--turn-fast-rounds 1` with no targets.

Authorization history: concept approved 2026-08-13 scoped to rungs
0–1; probe results 2026-08-14; production integration ruled by the
user 2026-08-14.

Provenance: Opus-workflow research report + adversarial review
(session 8b044cb2, workflow `wf_884c8c53-16d`, journal results #4
and #6), amended by the 2026-08-13 user discussion (rulings in §5).
Line numbers below were verified by the review against the code as
of 2026-08-12; treat them as approximate after any refactor.

---

## 1. Claim

> TCS is Gumbel AlphaZero in which the Q of a candidate action is
> estimated by **completing the turn and evaluating at the turn
> boundary** (optionally after one sampled opponent turn) instead of
> by a 1–3-ply subtree; and the improvement operator is
> **coordinate-wise refinement of a full turn commitment** instead
> of visit allocation over micro-actions.

Everything downstream of the target — σ q-transform, completed-Q,
prior damping λ, 5-tuple target schema, factored CE loss, C51 value
loss, replay buffer — is reused verbatim. The trainer does not
change (two semantic caveats, §4).

The reframe: with K ≈ 4–10 and branching ~100+, a side-turn is not
a deep sequential plan but a **joint assignment** whose value is
only revealed jointly. PUCT over micro-actions must commit to
coordinate 1 before seeing coordinates 2..K and grades it on a
half-finished turn — the value head's worst regime. TCS always
evaluates complete turns at the boundary, its best measured regime.

## 2. Measured foundations

From the research session (scratchpad probes; **mini maps,
`imit_tierb_start.pt`, top-8 sampler — rung 0 must replicate on
ladder maps with the live checkpoint before any number is quoted
again**):

- **M1 unit costs** (ladder map, 15M net): fork 0.11 ms, sim step
  1.95 ms, encode 6.3 ms, enumerate 5.9 ms, forward ~1.5 ms (4090)
  / 773 ms (laptop CPU). GPU-regime leaf ≈ 16 ms; a bare sim step
  inside a materialized plan is **13% of a leaf**. Every published
  option method pays a dynamics-network forward per option step; we
  pay 2 ms of exact Python. **The cost arithmetic is inverted, and
  no published design is built for the inverted arithmetic.**
- **M4 suffix survival**: replaying a recorded suffix after a
  1-coordinate substitution keeps 78.2% of commands (whole suffix
  intact 51.1%) — coordinates are weakly coupled, which is what
  makes coordinate-wise improvement work.
- **M5 boundary SNR**: sd of boundary V across 1-coordinate
  variants (signal, 0.116) vs paired variant−incumbent noise at
  fixed salt (0.048): **signal/paired-noise = 2.4** (median ratio
  5.4). The gate that could have killed the bet at zero cost, and
  it passed.
- **Stale-baseline correction (review finding A)**: the report's
  K = 3.71 / end_turn 27.1% motivation was a cross-lineage pooled
  artifact. Live leg (last iterations of `history_15m.csv`):
  **K ≈ 9.4, end_turn ≈ 10.5% and falling** — the turn-truncation
  degeneracy is curing itself under the 2026-08-12 rescale-floor
  fix. TCS's justification is the evaluation point and
  coordination, NOT turn truncation. Pre-registrations must use
  K_baseline ≈ 9.4 re-measured at rung 0.

## 3. Algorithm

**Spine.** At own side-turn start s₀, run the current policy
closed-loop to `end_turn`, recording per step j: a fork of
s_{j−1}, chosen action a_j, the full
`enumerate_legal_actions_with_priors` list L_j with priors, and
decision_step d_j (combat-oracle anneal symmetry — carried to the
target). Cost: K leaf-equivalents. One trunk forward prices the
ENTIRE legal set at a state (factored heads), so alternatives are
proposed for free.

**Coordinates** are positions in the recorded sequence, not
entities. L_j spans every unit's every action, so "open with a
different unit" is a single-coordinate perturbation at j = 1.

**Perturbation.** Pick coordinate j and a'_j ∈ L_j (Gumbel-top-k by
prior; `end_turn` always included). Fork at s_{j−1}, step a'_j,
replay the recorded suffix a_{j+1..K} open-loop through the sim
(the sim is the legality oracle; bounced commands are skipped, the
replay continues, `end_turn` always executes — every variant
terminates at a real boundary). Cost per variant: ~zero policy
forwards + ~K sim steps + ONE boundary value forward.
Δ_j(a'_j) = V(variant boundary) − V(incumbent boundary) at the
same CRN salt. This is the COMA counterfactual computed by exact
rewind instead of a learned centralized critic.

**Variant classes.** (a) open-loop substitution (default, ~free);
(b) extension — replace terminal `end_turn` with the top non-end
action and continue closed-loop (costs real forwards; own budget
slice ~20%); (c) closed-loop tail re-decision after an early edit
(same budget slice; the coordination-valley reacher).

**Acceptance — two-stage, mandatory (review finding B).** Naive
argmax-over-~50-variants accepts pure noise (E[max of noise over
50 draws] ≈ 2.0–2.8σ, at/above the 2σ threshold) — structurally
the `gumbel_rescale_floor = 1e-8` bug relocated one level up.
Rule: take argmax_j Δ_j, **re-evaluate winner and incumbent at 3
fresh salts**, accept only if the re-evaluated paired mean exceeds
2σ/√3. Log naive vs revalidation-surviving accept rates.

**Acceptance semantics — user ruling: grade-what-you-commit.** The
new incumbent is the **materialized turn** — the literal command
sequence that executed, drops included — never "the edit + the
original suffix". Estimator unbiased by construction; mismatched
completions are legitimate discoveries, not confounds.

**Boundary evaluation + reply arm.** v1: encode the boundary state
(opponent to move), take −V (existing perspective-flip convention).
Reply arm (default ON, B = 1): generate one capped closed-loop
opponent turn first and evaluate at the SECOND boundary — lookahead
becomes two complete turns ≈ 16–26 plies vs the measured 1–3
today. **User ruling: the reply arm is the SOLE
anti-value-exploitation guard** (a serene board that is a lost
exchange gets un-flattered by the opponent demonstrating it).

**Stochasticity.**
- *CRN keying (deferred gate 0e — do NOT touch `_next_seed` until
  the rung-1 probe shows revalidated improvements exist).*
  `WesnothSim._next_seed` (two call sites: per-attack, per-recruit
  trait roll) keys on a global request counter, so perturbations
  shift every later fight's dice — measured: paired sd ≈ unpaired
  sd, CRN currently does nothing. Fix: search-only keying by
  plan-invariant identity — attack (salt, turn, side, attacker_id,
  defender_id, attack_index); recruit (salt, turn, side, unit_type,
  target_hex). Live path (no salt) byte-identical; replay-export
  parity preserved by construction. Pre-registered: paired sd drops
  ≥30% while unpaired stays flat, else residual noise is plan
  divergence and the remedy is branching, not seeding.
- *Rao-Blackwellization at a perturbed attack.* Use
  `combat_outcomes.enumerate_attack_outcomes` (dynamics oracle —
  outcome masses only, never scores, never priors; same admissible
  role as `exact_outcome_enumeration=True`) to branch the suffix
  under enumerated outcome classes weighted by exact masses:
  V̂ = Σ_observed p(o)·V(o) + (1−Σp)·mean(V(observed)). Requires
  `advancement_choice="uniform"`; **returns None on complexity
  caps → explicit sampling fallback required.** Shared fights
  cancel under CRN within each branch; only changed fights pay for
  branches. Median perturbation touches no combat (attacks ~11% of
  actions) and is exact with zero variance.

**Execution: plan once, re-plan when surprised.** Execute τ* one
command at a time. Deterministic commands: no search. At each
stochastic action, execute, observe the realized outcome, warm
hill-climb (≈N/3 budget) from the realized state with the
remaining plan as incumbent — the option formally terminates at
the first unrealized chance outcome (semi-MDP-honest). ~1 re-plan
per turn at current attack rates. **Second invalidation trigger
(review): a live recruit bounce** — `play_one_game`'s retry loop
calls `policy.drop_last_pending`; `TurnCommitPolicy` must
implement it, and a mid-plan bounce invalidates the remaining
coordinates → re-plan there too.

## 4. Targets and trainer

Per coordinate j: (action, paired boundary value) pairs at
s_{j−1}; build the target with the EXISTING transform
π_j = softmax(λ·log prior + σ(completed_q)) reusing `_completed_q`
/ `_rescale_q` / `_gumbel_sigma` verbatim (factor into
`tools/target_transform.py` so search and TCS provably cannot
diverge). Emit the existing 5-tuple `MCTSExperience` per
coordinate. Additionally emit value-only boundary experiences
(empty visit_counts → zero policy loss, verified
`trainer.py:756-762`) — shifts the value head's training
distribution to exactly where TCS queries it.

Two semantic corrections (review) to "zero trainer changes":
1. Value-only experiences enter the policy-loss normalizer
   `total_gw` → **~10% silent policy/value coefficient shift at
   K≈9**; compensate or measure.
2. `harvest_boundary_pairs` pairs consecutive experiences with
   differing `current_side`; injected boundary experiences corrupt
   `boundary_sum` — the instrument this design leans on hardest.
   **Boundary experiences need a flag and a harvest exclusion.**

Prior art note: the existing tree does NOT silently drop rejected
actions — it routes them to `_NOOP_KEY`/`_STEP_ERROR_KEY` terminal
sentinel children (`mcts.py:873-890`). TCS's skip-and-continue is
a deliberate, different choice; the materialized-turn acceptance
semantics (§3) is what makes it sound.

## 5. User rulings (2026-08-13 discussion — binding)

1. **Acceptance is over materialized turns** (grade-what-you-
   commit).
2. **The survival filter and repair-on-bounce are DEAD.** No
   behavioral gate may consult suffix survival. `survival_j` is a
   logged error-bar covariate only (low-survival variants share
   fewer CRN-paired fights → noisier Δ → the accept threshold may
   widen, never redirect). This also moots the review's fog
   concern about survival-gated targets conditioning on god-view
   truth.
3. **Reply arm is the sole anti-value-exploitation guard.** If the
   search drifts passive under a sound estimator, that is either
   truth or a value-head defect fixed at the value head. No
   aggression priors, ever.
4. CRN keying is plan-invariant identity, not stream position.
5. Combat DP as dynamics oracle: admissible (masses in an
   expectation; never scores, never priors).

## 6. Integration seams

New code (not before rung 2+ authorization):
`tools/turn_search.py` (spine/perturbation/CRN/hill-climb, mirrors
`tools/mcts.py`'s role) and `tools/turn_policy.py`
(`TurnCommitPolicy` mirroring `MCTSPolicy`; `select_action(
pre_state, game_label=, sim=)` signature unchanged, returns the
next command of the cached plan, re-plans on exhaustion/
invalidation; must implement `drop_last_pending`).

Config plumbing has **three** generation paths, not two:
`sim_self_play.py`, `selfplay_worker.py` (`MCTSConfig`-from-args
block), **and `tools/actor_pool.py`** (builds its own policy per
actor process — the topology the live legs actually use). The
mis-damped-target incident is the precedent for missing one.

Reused unchanged: `action_sampler.enumerate_legal_actions_with_
priors`, `wesnoth_sim.fork/step`, the target-transform trio,
`_terminal_value`, `mcts_policy.finalize_game` z/aux/game_weight
logic (lift to a mixin), `harvest_boundary_pairs`. Guarded change
(deferred): `wesnoth_sim._next_seed` CRN mode. Playout-cap analog:
full perturbation budget on a random `turn_full_prob` ≈ 0.25 of
turns; cheap budget, no targets on the rest.

**Mode 2 (minimal-risk A/B arm):** keep the current MCTS and add
one macro-edge at the Gumbel root whose child is the turn-boundary
state (`MCTSEdge.children` maps `state_key → MCTSNode`, verified).
The literal OptionZero integration; isolates "does a turn-length
edge help" from "does coordinate refinement help."

## 7. Failure modes → observables

| Failure | Observable | Response |
|---|---|---|
| Noise-climbing (accepts within paired noise) | naive vs revalidated accept rate; placebo arm | two-stage acceptance is the fix; placebo ≥ half real accepts = STOP |
| Estimator noise on low-survival variants | ρ(Δ, survival) per iteration; survival logged per variant | widen accept threshold (never redirect); more outcome branching |
| Fog-differential leak (scouting variants gain god-view-informed suffix advantage) | Δ split by whether the perturbation changed the acting side's visible-unit set | pre-registered report split; ruling if material |
| Basin collapse (single-spine ascent over-learns one shape) | plan entropy over games; accepts→0 with Elo flat | multi-spine (rung-4 plan-codebook head); more Gumbel noise on perturbation sampling |
| Value-head exploitation | reply-arm on/off comparison | reply arm (sole guard, §5.3) |
| Hierarchy stops paying (value head strong → operator no-op) | accepted improvements/turn trending to 0 with Elo rising | expected success mode; retire gracefully |

## 8. Validation ladder (rungs 0–1 authorized)

**Rung 0 (laptop/cheap CPU box, ≤$1):** replicate M2–M5 on ladder
maps with the live campaign checkpoint; measure K_baseline
(current best estimate 9.43) and per-decision `kl_prior` baseline
in-script (absent from local history CSV).

**Rung 1 — `tools/turn_counterfactual_probe.py`** (offline, no
training, no production code; cheap CPU box; ladder maps, live
checkpoint). 200 side-turn states; per state: spine → Gumbel-top-k
perturbations per coordinate (always incl. `end_turn`) → open-loop
suffix replay logging survival and the fog-visibility split →
boundary eval (v1, **no reply arm — single-variable first
experiment**) → two-stage acceptance → placebo arm (boundary
values shuffled across variants before argmax).

Pre-registered decision rule:
- **PROCEED** iff revalidation-surviving accept_rate ≥ 0.50 AND
  median accepted Δ ≥ 0.05 AND per-coordinate KL(π_TCS ‖ π_prior)
  ≥ measured Gumbel `kl_prior` on the same states AND placebo
  accept rate < half the real arm's.
- **STOP — noise-climbing** if placebo ≥ half real, or naive
  accepts ≥ 2× revalidated.
- **STOP — no new information** if accepts are high but KL ≈ the
  Gumbel target's (TCS reproduces the existing target,
  expensively).
- K prediction (replaces the stale 1.5× form): TCS raises median K
  vs the rung-0 baseline, concentrated in turns where extension
  perturbations show positive revalidated Δ. K not moving is now
  an expected outcome, not a surprise.

**Later rungs (NOT authorized; re-propose on rung-1 numbers):**
rung 2 = self-play games probe (3 arms incl. placebo, cheap CPU
box, ~days); rung 3 = training leg vs control at equal wall-clock
with both micro-step and forward counters logged for both arms
(~$10–20); rung 4 = plan-codebook head (C≈8, hard-EM
self-distillation on improved turns; architecture change —
explicit approval required).

## 9. Cost (review-corrected)

At K = 8, GPU regime, parity with current search spend (~2.8 s per
side-turn): spine 128 ms + **~33 variants WITH reply** (~80 ms
each; the report's "~55" under-priced the reply's encode+enumerate)
or ~167 without. Alternatives per coordinate ≈ 4 with reply.
Evaluation horizon 2 full turns ≈ 16 plies vs measured 1–3.
Variants are batch-parallel across cores and batchable through
`forward_batch` — fits the measured CPU-bound box shape (4090 at
2–6% util).

## Addendum (2026-08-17): target link function — exposure invariance

**User ruling:** "random draw among the evaluated actions should not
push their probability up." The distill-target transform must be
EXPOSURE-INVARIANT: under an uninformative value head, E[target] =
prior for every action regardless of how often it is evaluated. The
sigma/exp transform violates this (convexity converts symmetric
grader error into expected mass gain proportional to evaluation
frequency — the leg-3 R2 `end_turn` ratchet), so `TurnSearchConfig`
gained `target_link` (`--turn-target-link`):

- `linear` (**DEFAULT**, leg-4 ruling): `prior^lam * max(0, 1 +
  beta*(q − LOO mean of the other evaluated q))`. Unbiased to first
  order under symmetric grader error (renormalization residual is
  second-order and non-positive for evaluated actions); beta = 5.0
  derived in docs/design_constants.md; `link_clip_frac` telemetry
  rides the distill stats.
- `exp`: the previous behavior, byte-shared with the Gumbel-MCTS
  sigma transform (pinned by
  tests/test_turn_target_link.py::test_exp_path_is_byte_identical...).
  Mirror descent concentrates faster under a KNOWN-GOOD grader;
  re-enable only with measured value-head trust (the A1 gate).

Consequence for §3's force-inclusion debate: with the linear link,
`end_turn` force-inclusion **stays ON** — representability is kept
and the exposure lottery is dead by construction, resolving the
"unsure" ruling on A5(ii) (2026-08-17). The rung-1 probe instrument
(`tcs_target_kl`) pins `link="exp"` so its 2026-08-14 baselines stay
comparable; pass `link="linear"` to measure the production target.
