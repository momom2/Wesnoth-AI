# Teacher arms — findings (2026-08-29, autonomous run)

Design: docs/archive/teacher_arms_20260829.md. Two boxes from the imitation
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

## Signal-profiling rounds 1-2 (2026-08-31, gradient-amplitude tree)

Round 1 (v1, post-clip): on the SEED, the applied update is ~99.7%
value gradient; policy distill projects +0.017 -- with the
target-link amplitude finding, the quantified "why +320 does not
transfer": weak targets x ~2% gradient share.

Round 2 (v1.1, pairwise cosines): FALSIFIED the antiparallel
policy/value hypothesis (cos ~ 0 everywhere) and exposed three
instrument defects, each now fixed in v1.2: (a)
peek_checkpoint_arch dropped gbc heads from every gbc-trained
checkpoint ("5 unexpected keys" in every arm eval since
2026-08-14 -- also an EVAL bug, fixed at the source); (b) the
v1.1 coefficient overrides were no-op/wrong (defaults already
production); (c) separately-clipped gradients cannot be compared
against a clipped sum -- the "96% unaccounted" paradox was clip
artifact. v1.2 profiles unclipped with a built-in linearity
self-check and target-amplitude (KL/TV of targets vs prior).

Round 3 (v1.2): SEED rows are the keepers — linearity residual
0.000 (the instrument is exact unclipped), and the headline stands
quantified: the value gradient is 12x the policy gradient in norm
(3.07 vs 0.25) and owns 99.6% of the applied update's direction;
distill targets are homeopathic (KL median 0.014 vs the prior).
With target-link amplitude, this is the measured "why a +320
teacher signal does not transfer": weak targets x ~0% update share.

Round 3 arm rows INVALIDATED — and rounds 1-2 arm rows with them:
the armV3 checkpoints were never staged on the profiler box, and
_load_policy silently random-inits on a missing path, so every arm
profile in rounds 1-3 measured a fresh random network (the logs
say "no checkpoint -> random init" seven times per run). This
retroactively explains the round-3 anomalies attributed to the V3
checkpoints: the "140% linearity residual", the "policy norm
growth across the cliff arc", and the zero-GBC-gradient mystery
(a random-init build has no gbc heads because peeking a missing
file returns the base arch) were all properties of random nets.
Guard shipped: signal_profiler.make_policy now raises
FileNotFoundError on a missing checkpoint. Invalid JSONs moved to
eval_games/signal_profiles/invalid_random_init/. Round 4 = the
first real arm profiles (same v1.2 protocol, staging verified by
arch peek: gbc+aux True on all three).

## Round 4 (2026-09-01): the real arm profiles

8 games / ~900 experiences per checkpoint, seed 31337, load lines
verified, linearity residual 0.0000 on all runs (the round-3
"140% residual" was the random-net artifact; no repeatability
check needed). Table = gradient norm / signed share of the
applied update's direction (proj_frac):

                     seed      early     precliff  cliff
    total            3.07      2.99      2.84      2.18
    value_inbatch    3.07 .996 2.96 .989 2.83 .992 2.17 .991
    policy_distill   0.25 .004 0.24 .009 0.29 .008 0.19 .009
    gbc              --        0.03 .002 0.02 .000 0.02 .000
    value_memory     0.37 .035 1.17 .138 0.76 .080 0.56 .079

Verified findings:

1. VALUE DOMINANCE IS STRUCTURAL, NOT TRANSIENT: value_inbatch
   owns ~99% of the update direction at EVERY point in the arc
   (seed included). The policy-side signal is ~1% of the update.
2. TARGETS ARE HOMEOPATHIC EVERYWHERE: KL(target||prior) median
   0.0018 (seed, rerun) / 0.0025-0.0062 (arms), TV ~0.03-0.045
   over ~350 legal actions -- the teacher's accepted plans barely
   perturb the prior. Combined
   with (1): the measured reason a +320-Elo teacher signal does
   not transfer -- near-zero-amplitude targets carried by a ~1%
   gradient share.
3. GBC is gradient-inert (norm ~1% of value's) -- consistent with
   arm G's behavioral exoneration.
4. value_memory is the largest non-inbatch term and 2-3x bigger
   on the arm checkpoints than the seed -- the V3 memory step was
   a real second value channel on top of an already value-
   dominated update.
5. No gradient-amplitude signature of the cliff: the cliff
   checkpoint's gradients are modestly SMALLER overall. The
   -676 collapse is not visible in update magnitudes -- coherent
   with the off-distribution-value mechanism (the damage lives in
   where the value head is wrong, not in how hard it trains).
6. aux_margin unprofiled in rounds 1-4 (harvest lacked
   draw_tiebreak; fixed for v1.3+).

## Round 5 (2026-09-01, v2): update space, consultation movement,
## provenance — the transfer failure fully quantified

Protocol: 8 games/checkpoint on a 5.9GHz box, aux targets now
attaching (v1.3), 400 search-consulted boundary states captured
per checkpoint (reservoir over ~22-24k consultations), 200
real-state controls. Update tree = one production step per
isolated term, real Adam (checkpoint second moments, exp_avg
zeroed — loaded momentum otherwise dominates any single step and
erases attribution; a total_momentum variant keeps it for scale),
production clip 1.0. Caveat: the seed's optimizer_state key is
EMPTY (stripped at handoff), so seed update rows are fresh-Adam;
arm rows are true production moments.

1. THE EROSION CHANNEL, MEASURED DIRECTLY: the isolated value
   step moves value predictions on IMAGINED (search-consulted)
   states as much as on real ones — dv_consult/dv_real 0.91-0.98
   on all three arms (~0.062-0.072 vs ~0.063-0.079 per step, in
   [-1,1] units). Nothing in the loss anchors the states search
   reads; they move in lockstep with the trained states.
2. SCALE: one production step's value movement on consulted
   states is ~0.06-0.08 ~= 3-4 C51 atoms — LARGER than TCS's
   median accepted delta (~2 atoms). A single training step
   re-scrambles value differences of the size search uses to pick
   plans; over an iteration the search's ranking substrate is
   fully churned. This is the mechanism behind "search flips from
   +321 to -340 while raw policy stays put", now in units.
3. ADAM REBALANCES MAGNITUDE, NOT DIRECTION: post-Adam the policy
   step's weight movement is ~40-65% of the value step's (|du|
   0.012-0.022 vs 0.029-0.034; gradient space said 8%), so the
   policy channel is not starved of step SIZE — it is starved of
   target CONTENT (KL median 0.0024-0.0044 across the arc).
4. TARGETS PUSH TOWARD PASSIVITY: per-category mass, the
   accepted-plan targets consistently REMOVE mass from attacks
   (-0.003..-0.006) and ADD mass to end_turn, growing across the
   arc (seed +0.0018 -> precliff +0.0049 -> cliff +0.0077). The
   homeopathic policy signal that does exist points in the
   K-collapse direction even under mover frame + projection.
5. THE VALUE GRADIENT IS A CANCELLATION RESIDUAL:
   cos(winner-state grad, loser-state grad) = -0.83..-0.95 — the
   two label groups push the trunk in nearly opposite directions,
   and the net value gradient is their small difference. Which
   side dominates FLIPS along the arc (seed/early: loser-side,
   cos_full +0.99/+0.88; precliff/cliff: winner-side, +0.95/+0.87)
   — a fragile direction, consistent with leg-5's coin-flip trunk
   rotation and arm T's oscillation. (This REFUTES the round-4
   guess that winners and losers would push the same proxy
   direction.)
6. GBC: gradient-inert in update space too (|du| ~0.01, dv
   ~0.0001). Aux (first measurement): norm 0.32-0.74, ~4-6%
   projection, modest dv — a real trunk regularizer, not a
   dominant channel; NOTE the user intends aux as telemetry-only,
   but as shipped it trains the trunk at coef 0.15 (trainer.py
   ~1268) — pending ruling.

Synthesis: search training fails because (a) each value step
churns the imagined-state valuations search depends on by more
than search's own discrimination threshold, while (b) the policy
channel — adequately sized after Adam — carries near-empty targets
whose systematic component points at passivity. The fix space this
measures out: anchor/ground value on consulted states (reanalyze-
style targets exactly where the head is read), and strengthen the
target link (beta) — matching arm-W's W2 and sharpening W1 into
"control value movement per step", not merely "shrink value_coef".

## Round 6 (2026-09-01): aleatoric-label probe — near-random-head
## refuted; the value loss spends its mass fighting opening noise

User challenge: winner/loser gradient anti-parallelism "should
only happen with an almost random value head". Discriminating
test: per-turn-decade outcome AUC + winner/loser value-gradient
cosine, seed + cliff, 8 games each.

    seed:  AUC 0.40 (t1-10) -> 0.80 (t11-20) -> 0.96 (t21-30)
    cliff: AUC 0.77        -> 0.78          -> 1.00 (CE 0.11 vs
           floor 0.69; t31-40 AUC 1.00, CE 0.02)
    cos(win,lose): seed -0.91/-0.90/-0.84; cliff -0.73/-0.79/-0.48
    gradient mass: turns 1-20 carry 3-7x the late-game mass
           (seed d1_10: |g_lose| 12.4 vs |g_win| 4.8)

Verdict: NOT a random head — endgame discrimination is perfect and
the cancellation fades exactly where labels become informative.
The +-1 outcome labels on undecided early positions are
substantially aleatoric, and that is where the value gradient's
bulk sits: the loss spends most of its budget on label noise while
the informative endgame (already solved) contributes almost none.
Two sharpenings: (a) the SEED is actively miscalibrated on
openings (AUC 0.40 BELOW chance, CE above the state-blind floor)
and self-play training fixed it (0.40 -> 0.77) — early decades
contain real signal plus noise, arguing for TD/bootstrapped early
targets or phase-weighted value loss over simply zeroing them;
(b) which class's noise-mass dominates (seed: loser 12.4/4.8;
cliff: winner 7.6/4.9) is what the net trunk direction inherits —
the round-5 arc flip, explained.

Shipped alongside: aux + moves-left heads DETACHED (user ruling:
telemetry-only; validated — the aux term's gradient is now 100%
in its own head, zero trunk), and the per-decade fresh-probe
telemetry (fresh_{ce,floor,auc,n}_{d1_10..d61p} CSV columns) so
this structure is live in every future leg.

## Arm VG (2026-09-01..02): value grounding — K-collapse in 4
## iterations, mechanism captured live by the new signal telemetry

User order: implement (2) rollout grounding + (3) consistency
targets, launch, measure. Leg: docs/archive/arm_vg_leg_20260901.md; killed
by the K-median tripwire at iter 4 (median 8 < 10 x3) — the FOURTH
arm and THIRD distinct value-channel design to collapse turn
length within ~3-5 iterations (V1 memory ~5, V2 frozen-trunk ~3,
VG ~4). Probe at ~59k steps: 6-0-18 vs seed (~-190), mid-collapse.

The always-on signal telemetry (sig_* columns, shipped this leg)
recorded the mechanism as it happened — per-source gradient norms
and per-step value movement on consulted states:

    it K_med dv_consult policy game_val ground consist  atk%
     0  10     0.170     0.12    3.8     24.6    36.5   21.4
     1   9     0.066     0.11    6.4     17.7    41.7   21.2
     2   4     0.030     0.12    6.5      0.0    25.2   14.0
     3   8     0.149     0.07    4.8     13.9    32.8   21.3

1. THE CONSISTENCY TERM DOMINATED EVERYTHING: norms 25-42 vs the
   entire game-outcome value signal at 3.8-6.5 and policy at 0.1.
   The 0.25 value_weight guard was ineffective because LOSS WEIGHT
   does not bound GRADIENT MAGNITUDE: the labels are the head's
   own boundary optimism mirrored (-0.4 vs +0.4, the WYSIATI
   bias), and C51 CE gradients explode when a confident head is
   pushed toward a strongly disagreeing label. The rollout (truth)
   term was similarly inflated (14-25) by the same effect.
2. THE CURE AMPLIFIED THE DISEASE: dv_consult ran 0.03-0.17 per
   step — up to 8 atoms, vs the round-5 baseline 0.06-0.08 and
   the ~0.08 search decision threshold. Training the head on
   boundary states rewrote the accept gate's substrate within 2
   iterations; end_turn candidates started winning gates
   (attack% 21->14, K 10->4) — the leg-3 passivity shape.
3. fresh_value_ce on GAME states degraded 1.0 -> 1.7 alongside:
   the boundary-state gradients dragged the whole head off its
   own distribution.

Lesson (sharpens the value-channel law): TCS turn length is
exquisitely sensitive to value movement on boundary-adjacent
states, and ANY added value channel whose gradient is large
relative to the game signal collapses it in a handful of
iterations.

## Arm VG2 (2026-09-02): principled mixture + trust region --
## K-collapse in ONE iteration; the trust region anchored the
## wrong frame

Design: docs/archive/arm_vg2_leg_20260902.md (Gaussian consistency term
with b, sigma2 estimated from paired labels; trust region with
dual-ascent lambda vs the 2-atom resolution). Result: iteration 0
trained at lambda=1 (no reading yet), consulted-state movement
0.31, head optimism +0.28 overshot to -0.23; iteration 1 played
K median 1 (gate shorten-accepts 0.10 -> 0.58/plan). The
consistency term was tame (norm 3-19 vs VG's 25-42): the Gaussian
form worked. The pusher was the rollout-truth term (16-33),
coherent by nature.

Probe (seed vs iter-1, 80 states): the MOVER-FRAME valuation of
the incumbent turn swung +0.37 -> -0.35 (0.71), more than twice
the 0.31 measured on the trained/anchored states -- because the
grounding captures were the stage-2 projection pairs (post-flip,
opponent to move) while stage 1 grades the pre-flip mover-frame
state. Candidate contrast fell 3x (best_delta 0.086 -> 0.026),
projection re-grades turned positive, end_turn accepts 3x. The
value change generalized ACROSS THE FLIP with amplification: the
leg's own instance of the erosion channel it was built to fix --
training one frame away from where the search reads.

Two fixes follow directly: ground/anchor/measure on the states
the gate reads (mover-frame pre-flip boundaries; the round-5
consult hook), and warm-start lambda from the calibration
harvest's predicted movement (first-iteration protection; the
collapse horizon is one iteration). Fourth collapse in the series;
each one localized the mechanism further.

## Arm VG3 (2026-09-02..03): gate-frame grounding + measured
## trust region -- the churn is held; the mixture self-distills

Design: docs/archive/arm_vg3_leg_20260902.md. What worked: capture/anchor/
measure on the mover-frame pre-flip states the gate reads, with
lambda0 = 12.1 measured by an offline production iteration; per-
iteration movement on those states fell from 0.83 (unregulated) to
~0.1 (0.035-0.063 held-out), K stayed 12 -> 10 -> 8/9/10 over six
iterations with no tripwire, and the checkpoint continuation
metadata restored the controller on a real resume. What failed:
the consistency (bootstrap) term's precision, 1/(2 sigma2) with
sigma2 = var(search - rollout) - (1 - V^2), is a small difference
of two ~0.95 quantities; the proxy overshot, sigma2 hit its floor,
and the first valid provenance profile (pin 2862807, 53k steps,
linres 0.054) read: consistency 98.8% of the update direction,
game outcomes 2.6%, rollout truth 0.8%, policy 0.0%;
cos(winner-state grad, loser-state grad) = +0.82 (the value
gradient no longer depends on the outcome); late-game gradient
opposing the net (-0.65). Pin 6-0-18 vs seed (~-190); second pin
at 83k: 5-0-19 (~-230), profile consistency 100% / rollout truth
-0.9% (now pushing against the net) / game 1.2%. The controller
saturated (dv 0.23 at lambda 267). Trainer stopped by hand at
iteration 5; box destroyed.

The series now reads: rate control works (VG3), direction is set
by the labels' precision weighting, and a self-referential label
whose precision is over-estimated wins the mixture. Fixes
committed: precision = upper 90% CI on sigma2 (ad5c87f); rollout
noise measured from 2 playouts/state (9d04c2f). VG4 proposal in
the leg doc: (a) VG3 + fixes, or (b) rollout truth only, no
bootstrap -- the never-run variant the profile points at.

CORRECTION (2026-09-02, from the VG2 calibration harvest): the
"categorical CE explodes on confident misses" explanation of the
25-42 norms was wrong in mechanism -- CE's gradient w.r.t. the
logits is bounded (p - onehot) regardless of confidence; the LOSS
value explodes, the gradient does not. The real mechanism is
COHERENCE: every consistency label said the same thing ("you are
~0.3-0.4 too optimistic here"), so 192 per-state gradients added
up ALIGNED, while the ~1850 game-state gradients largely cancel
(round 6: winner/loser anti-parallel, cos -0.9) and leave a small
residual. Aligned systematic push vs cancelling noise -- a 280x
per-state ratio in the sum. Bias correction removes exactly the
coherent component; the trust region bounds whatever coherent
correction remains to <= 2 atoms per iteration. That is the VG2
design (docs/archive/arm_vg2_leg_20260902.md), with b and sigma2 measured
(-0.321, 0.100 on the seed) rather than chosen.

Cost: ~$5 (incl. ~8h idle after the 22:17 tripwire — no auto-
teardown by design). Artifacts: eval_games/arm_vg/, escrow
tier-b/arm_vg_20260901/ (final checkpoint, CSV, probes).
