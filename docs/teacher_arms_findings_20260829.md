# Teacher arms — findings (2026-08-29, autonomous run)

Design: docs/teacher_arms_20260829.md. Two boxes from the imitation
seed (leg-5 config + fixed GBC labels), arm T = TCS teacher,
arm M = plain Gumbel MCTS-32 teacher. ~40k steps/hour (5.9GHz
cores). Probes: 24-game pin-vs-seed matches, MCTS-32 frame both
sides, on-box GPU.

## Result 1: plain-MCTS teaching dies of K-collapse (arm M)

Tripwired at iteration 10 (~124k steps): median actions/side-turn
1 for 3 consecutive iterations — the leg-3 passivity shape,
reproduced from a healthy seed in ~4h. Mechanism visible in the
final iterations' target telemetry: end_turn prior mass INFLATING
through distillation (0.212 -> 0.255 at iter 9). Turn-truncation
is teacher-intrinsic to Gumbel-MCTS distillation, NOT a TCS
artifact. Curve before death: -162, -374, +52, -374; final
(collapsed) checkpoint -263 +/- 90. Artifacts:
eval_games/teacher_arms/armM/.

## Result 2: the invisible erosion channel is OFF-DISTRIBUTION
## VALUE CORRUPTION, amplified by search (arm T)

Arm T (TCS teacher) never K-collapses (K median 17-22 throughout)
but oscillates violently vs the seed: -28, -137, -61, -478, -104
at ~27k-step spacing — while every internal metric stays healthy
(losses flat, value_auc fine, 24/24 decisive, no tripwire).
The -478 pin (step 2,922,263) reproduces under fresh seeds
(3-0-21, ~-340): real, not instrument noise.

The attribution triad on that pin:

    pin raw   vs seed raw : 12-0-12  (priors EQUAL)
    seed mcts vs seed raw :  9-0-1   (search = +321 for the seed)
    pin mcts  vs seed raw :  3-0-21  (search = ~-340 for the pin)

Search flips from a +320 amplifier to a -340 saboteur between
checkpoints whose raw strength is identical and whose weights
differ by <=1.2% per component (value_head 0.5%, actor_head 0.3%;
largest movers: target_k/q projections and the encoder trunk —
leaf-evaluation pathways).

Mechanism: training drifts the value head's judgments on IMAGINED
states — the counterfactual positions only search visits. All
value telemetry (value_auc, fresh_value_ce, the redraw tripwire)
measures REAL-game states, where the head stays accurate; the
drift is invisible by construction. At play time search consults
the head exactly on the unmeasured states and its verdicts
overrule the (healthy, anchor-protected) prior. The drift wanders,
so strength oscillates instead of eroding monotonically. This
explains the leg-5 resume verdict's signature (all proxies green,
~200 Elo gone) and why the 2x2 found play-procedure effects the
weights couldn't explain.

## Weight-diff table (healthy pin 2,891,504 -> collapsed 2,922,263)

    target_k_proj   0.0124   encoder        0.0116
    target_q_proj   0.0110   gbc_heads      0.0100
    aux_score_head  0.0089   weapon_head    0.0087
    type_head       0.0083   value_head     0.0051
    actor_head      0.0027   token_kind_embed 0.0004

(relative L2 per component; nothing exceeds 1.2%)

## Result 3: GBC exonerated (arm G)

Arm G (TCS teacher, --no-gbc, otherwise identical) oscillates the
same: -201, -85, -263 at ~26k/57k/85k steps. The drift needs no
GBC gradient; the common denominator across all three arms is the
core self-play value training itself (unprotected C51 head on
terminal outcomes; policy head anchored, value behavior not).

Corrections from redraws: arm M's lone positive point (+52)
re-measured 10-14 (~-60) under fresh seeds — noise on a
seed-level checkpoint, not a real gain. Arm T's -478 re-measured
3-21 (~-340) — real. Final arm T curve (24-game probes, ~27k-step
spacing): -28, -137, -61, -478, -104, -255, -201, -104, -382.

## Addendum (2026-08-30..31): the value-memory arms — the fix
## collapsed the policy, twice, and localized the disease further

Arm V (arm T's recipe + --value-memory-iters 20, one extra value
gradient step per iteration over a widening per-game outcome
reservoir): K-COLLAPSED at iteration ~5 (median 9 actions/turn),
distill targets healthy throughout (et mass flat 0.03-0.06 — NOT
the arm-M ratchet). First patch (freeze everything but the value
head in the memory step, verified parameter-exact): arm V2
K-collapsed FASTER, iteration ~3. Quick entropy check does NOT
support value saturation (fresh_pred_entropy rose 0.38->0.44).
Probe points before death: V1 +56 then -182; V2 none completed.

What this pins down: ONE extra value-HEAD-ONLY fit per iteration
is sufficient to collapse TCS's turn length within ~3 iterations,
with target telemetry blind to it — the strongest causal handle
yet on the search-consumes-value channel. What it leaves open: the
mechanism (saturation disconfirmed at first look; candidates: the
head's fit drifting off the distribution search co-adapted to;
TCS's accept-threshold interacting with a faster-moving head).
Two consecutive failed patches -> step-back rule; HOLDING for user
review (2026-08-31). Artifacts: armV1/armV2 finals escrowed at
tier-b/teacher_arms_20260829/, probes/logs under
eval_games/teacher_arms/armV/.

## Open

- Whether arm T's oscillation and leg-5's smooth -200 are the
  same channel at different sampling density: plausible, not
  proven.
- Fixed-vs-broken GBC labels never isolated (moot for the drift
  after arm G, still open for historical attribution).
- The healthy/collapsed study pair is escrowed:
  tier-b/teacher_arms_20260829/armT_pin_{2891504,2922263}.pt
  (+ local copies in training/checkpoints/). These two, 30k steps
  and <=1.2% weight change apart, bracket a ~660-Elo swing in the
  value of search — the natural test articles for any
  imagined-state value telemetry.

## Implications (for the next design round, user rulings pending)

1. NEW TRIPWIRE / TELEMETRY: evaluate the value head on
   SEARCH-VISITED (imagined) states each iteration — the blind
   spot is now a measurable quantity (collect leaf states during
   generation, grade the head's verdicts against deep-search or
   rollout ground truth).
2. VALUE GROUNDING ON IMAGINED STATES: the sim replays imagined
   states perfectly; salted closed-loop rollouts from leaf states
   give ground-truth value targets exactly where the head is
   drifting. This is what "reanalyze" machinery is FOR — aimed at
   the value head, not the policy targets (consistent with the
   redesign panel's rejection of policy-side re-search).
3. SEARCH ROBUSTNESS: a drifting value head argues for search
   trusting it less (cliffness-aware weighting exists, default
   off) — mitigation, not cure.
4. The E-ladder's teacher question is ANSWERED for MCTS (K-collapse)
   and reframed for TCS: the teacher procedure was never the root
   cause; the value function's off-distribution behavior is.
