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

## Open

- Arm G (TCS teacher, --no-gbc) running: does removing the GBC
  gradient calm the value drift? (Caveat: G tests gbc-vs-none,
  not fixed-vs-broken labels — a broken-label arm would be needed
  to fully explain history.)
- Whether arm T's oscillation and leg-5's smooth -200 are the
  same channel at different sampling density: plausible, not
  proven.

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
