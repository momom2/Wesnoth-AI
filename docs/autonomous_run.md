# Autonomous run — Claude + Fable (72h from 2026-07-28)

**Standing mandate (user, 2026-07-28).** Claude and Fable run autonomously
for 72 hours in a persistent discussion loop. Design decisions, commits and
pushes at will. Subagents/workflows allowed, **Opus 5 and smaller models
only**. Don't waste tokens frivolously. Box experiments allowed — credits
topped up but NOT unlimited; spend compute wisely. The backlog may be
followed, ignored, or extended with our own ideas.

> "Always prefer the short, principled solution over the long kludgy one.
> Always prefer the long, rigorous solution over the short ad hoc one."
> "Review your designs and your code implementations."
> "I recommend you don't hyperfixate on specific issues and consider how to
> enable training at scale on a larger network, though that's ultimately
> your call."

**Success is measured on BOTH:** (a) algorithmic improvements landed in the
codebase, (b) actual measured performance of the trained policy.

This file is the run's DURABLE STATE. Claude's context compacts and Fable's
is separate, so anything that must survive lives here, not in a context
window. Update it every cycle.

---

## Strategic thesis (revise if evidence says otherwise)

Measured on 2026-07-28 (`docs/eval_20260728.md`):
- The campaign lineage **peaked ~2.3-2.75M decision steps and REGRESSED**
  by 3.74M (0-1-3 vs each predecessor).
- It **loses to the built-in RCA AI**, ~9 decisive games, 0 wins.
- Under-recruiting / gold hoarding **worsened**, and is **not** mechanically
  blocked (62-69% of decisions offer a recruit, zero mask bugs) — it is a
  learned actor-preference shift. The prior `weight_gold=0` fix did nothing.

=> **Scaling compute or parameters on top of a broken learning signal buys
nothing.** But diagnosing forever is also failure. So run three tracks in
parallel, and let the scale track proceed while the signal track digs:

| track | goal | owner (default) |
|---|---|---|
| **T1 signal** | find + fix WHY the policy is taught to hoard/degrade | Fable leads, Claude reviews |
| **T2 scale** | make a larger net trainable at throughput on the box | Claude leads, Fable reviews |
| **T3 campaign** | run + measure a real campaign; Elo vs predecessors AND RCA | Claude (box ops) |

**Gate before spending serious box credit:** T1 must produce either a fix
with a measurable behavioural delta, or a positive verdict that the signal
is sound. Do not launch a long campaign into a known-regressing setup.

---

## Roles and protocol

- **Claude (Opus 5)** — architecture, integration, commits/pushes, box
  operations, final call on design disputes. Runs the loop.
- **Fable** — independent investigation and experiments, adversarial review
  of Claude's designs and code. Fable is a peer, not a tool: it is expected
  to disagree, and disagreement is recorded here.

**Cycle:** read this doc -> set agenda -> dispatch Fable a scoped work
package -> Claude does its own package -> exchange results and CRITIQUE
each other -> land code + tests -> update this doc -> schedule next wake.

**Review discipline (non-negotiable):** every design gets an adversarial
pass by the other party before it lands. Every implementation gets a code
review. Findings from a review go in the log below even when rejected.

## Guardrails (learned the hard way; violating these wastes hours)

1. **No background/detached runs on the dev box** — they get SUSPENDED
   (0 CPU, ~11 MB WS). Foreground with an explicit `timeout` < 9 min.
2. **Never two pytest invocations at once** (multi-GB each).
3. **Full slow tier (`pytest -m ""`) before any training-critical commit**;
   run it in GROUPS to fit the timeout.
4. **Sim fidelity is sacred** (architecture principle 4). Never "fix" the
   sim toward convenience without checking the engine source first.
5. **Checkpoints: date-stamped names only**, verify by `decision_step`, and
   `tier_a_campaign.pt` is a RESERVED pipeline name.
6. **Box:** stop it when the next step waits on a human; never leave it
   idling. Verify what a restart RESUMES from before paying for it.
7. Prefer measurement over inference; label inferred claims as inferred.

---

## Standing decisions

- (2026-07-28) The detector advice signal is built, tested and gradient-
  verified (13% fire rate, 5.7% grad share at graft). `MCTS_ADVICE=1` is
  the box default. It is a candidate lever for T1 but NOT assumed to be the
  fix.
- (2026-07-28) Box instance 45230879 (RTX 4090 / 64 vCPU, $0.50/hr) start
  request is QUEUED — host resources unavailable. If it never frees, a new
  instance is a CREATE, which the user launches.

## Open questions (kill or answer, don't let them rot)

1. Which term in the training signal pays the model to bank gold and
   decline recruits? (T1's first probe: C51 value around recruit vs
   hoard-and-end_turn, bucketed by turn.)
2. Does the real engine leave a leader off-keep on Tombs_of_Kesorak /
   Sablestone_Delta (7.5% of leader-sides can't recruit at start), or does
   it place differently? Check `wesnoth_src/src/` before changing placement.
3. What is the actual throughput ceiling on the box, and what net size is
   trainable within it? (T2.)

---

## Cycle log

Newest first. Each entry: what was attempted, what was MEASURED, what was
decided, what is next. Keep entries short and factual.

### Cycle 9 — 2026-07-29 — campaign steady; rebooted onto the telemetry build

Two iterations before the reboot, ~18 min each => **~230 iterations
projected** for the run:

```
iter 0: train_step 313.4s loss=2.9040 policy=2.2274 value=0.6652
iter 1: train_step 363.9s loss=2.9429 policy=2.2622 value=0.6652
iter 0: decisive -- ladder 6/6,  mini/drill 12/12
iter 1: decisive -- ladder 11/11, mini/drill 7/8
```

Near-total decisiveness on ladder maps, which is the distribution the eval
actually cares about. Loss is flat across two iterations — expected this
early and not yet informative.

**Rebooted onto `4069190`** to pick up the boundary telemetry. Cost ~1
iteration out of ~230 (0.4%); the payoff is a `boundary_sum` baseline
covering the whole remaining run, which is what decides whether the T1-F
penalty is needed at all. Verified: resumed from the campaign checkpoint
(decision_step 2,298,935 — carried, NOT reset), 101 workers back up.

NEXT CYCLE: confirm `boundary_sum` is appearing on the train_step line and
start the baseline series.

### Cycle 8 — 2026-07-29 — TRAINING IS HEALTHY; box decision made; boundary telemetry landed

**First iteration completed and the numbers are good:**

```
iter 0: rolled 24 games in 521.7s (3880 actions, 581 turns; 7 actions/s)
iter 0: train_step in 313.4s  loss=2.9040 policy=2.2274 value=0.6652
        z_comp=0.54/0.46/0.00      <-- 54% win / 46% loss / ZERO draws
```

~14 min/iteration => ~300 iterations over the run (vs the ~70 the old
cadence would have allowed). **Zero draws** is the headline: draw-inflation
was this lineage's chronic pathology, and the first post-fix iteration is
fully decisive. z_comp_w 0.51/0.49 — balanced, no side collapse.

**BOX DECISION: KEEP (no recreate).** The sm_89 gap is real but PTX JIT
compiles to native SASS at module load and caches; steady-state cost is
small and the warmup is already paid. A recreate costs ~20 min plus
fresh-image risk for a speculative gain, against a campaign that is now
producing healthy decisive data. Revisit only if throughput degrades.

**Boundary-sum telemetry landed (a9d6e4f)** — Fable's diff, reviewed and
tested here. Deliberately shipped BEFORE the proposed consistency penalty
so we get a baseline series from the live campaign: if `boundary_sum`
trends toward 0 on its own under the fixed q-transform, the penalty is
unnecessary and T1-F dies the way T1-D did. Tests pin the extraction
contract (switch-only pairing, single-side game, side-3 interleave,
playout-cap gaps, NaN below threshold). 579 fast + 5 slow green including
the concurrent rollout/train-step races.

### Cycle 7 — 2026-07-29 — two config bugs on the live box; the CUDA guard pays for itself immediately

**1. The trainer stalled, and the cause was config, not code.** After 45 min:
0 train_steps, workers healthy (100 procs, 43 min CPU each, 131% CPU, load
130), 812 GB free, no OOM, trainer alive but nearly idle (54 s CPU in
33 min). Root cause: `games-per-iter` was HARD-TIED to `SPOOL_WORKERS`, so
100 workers meant the trainer waited for **100 finished games** before a
single gradient step. Measured game time ~9-40 min => roughly ONE iteration
per hour, i.e. ~70 iterations and ~1100 gradient updates for a 70h run.
Fixed (f2d8765): `GAMES_PER_ITER` is its own knob, default
`min(SPOOL_WORKERS, 24)`. Extra cores now DEEPEN the replay buffer instead
of lengthening the iteration. (Also caught in my own edit before it shipped:
the arithmetic read `SPOOL_WORKERS` before its default was applied, which
would have produced `games_per_iter=0`.)

**2. `$WORKDIR/env.sh` exists — a live box can be retuned without
recreating it.** The onstart sources it and it beats create-time `-e`
values. This removes the "env is frozen at create" constraint I had been
planning around; recorded here because it changes what is cheap to change.

**3. Fable's CUDA smoke test found a real defect on the box I picked,
within seconds of landing.** The pinned `pytorch 2.4.0/cu124` image reports
compiled archs `['sm_50','sm_60','sm_61','sm_70','sm_75','sm_80','sm_86',
'sm_90']` — **`sm_89` is absent, and the RTX 4090 IS sm_89**. The GPU is
running via **PTX JIT fallback**. `torch.cuda.is_available()` returns True
throughout, which is exactly the blind spot the guard was written for. This
is a candidate contributor to the slow throughput (GPU util was 71%, so GPU
work is not negligible).

**Decision point, deliberately gated on measurement:** with the cadence
fixed to 24 games/iter, watch the first iterations. If throughput is
acceptable, keep the box and accept JIT (kernels cache after first use). If
throughput is poor, recreate on an image whose torch includes sm_89
(pytorch >= 2.5 / cu124). Recreating is cheap RIGHT NOW — zero completed
iterations means nothing is lost — and gets more expensive every hour.

### Cycle 6 — 2026-07-28 — T1-D gate says NO; a correction to my offset-invariance claim

**The gate worked: Fable disconfirmed its own hypothesis before I built it.**
T1-D measured whether the winner records more states (which would bias the
value target toward "to move => winning"): over 8 decisive games the
winner's share of recorded states is **0.519 mean, above 0.5 in only 4/8**,
and the marginal E[z] over states is **+0.03..+0.05**. To explain the
measured boundary bias (+0.4..+0.65) the mechanism would need an order of
magnitude more. **DISCONFIRMED — the per-(game,side) value reweighting is
NOT landing.** It would have been plausible-looking, harmless-looking, and
useless. This is exactly why proposals get a measurement gate before code.
(Caveat retained: on the skill-imbalanced HUMAN corpus the count mechanism
could be larger; untestable locally, unclaimed.)

**CORRECTION TO MY OWN CLAIM (important).** I wrote that the q-transform's
offset invariance immunises the target against the side-to-move bias. That
is WRONG as stated, and Fable is right: the bias is not a uniform offset
across a node's children — only children that CROSS A TURN BOUNDARY are
evaluated from the opponent's perspective and negated, so it acts as an
**end_turn-child-specific relative handicap** (~-2b in Q). Min-max rescale
is monotone, so the end_turn child still ranks ~2b too low in every target
and every interior PUCT comparison. Offset invariance protects against a
*global* baseline drift; it does not touch this. The 0722 probe
(end_turn 0.174 prior -> 0.054 target) is consistent with the handicap
operating right now, i.e. the search systematically UNDER-ends turns.

Fable's proposed instrument (not built, needs review): a boundary-consistency
penalty `lambda*(V(s_i) + V(s_{i+1}))^2` over consecutive recorded states
where the side switches, since the true sum is ~0 in expectation. Gated on:
boundary-sum metric drops materially on a short local leg WITHOUT
fresh_value_ce degradation.

**T1-E tripwire recorded** (when to take the denominator floor
`max(hi-lo, 0.1)` off the shelf): fixed 20-state probe set, 2 independent
full-budget searches per state, total-variation distance between the two
extracted targets; trigger at mean TV > 0.20 (or > 0.30 on the contested
subset) on 2 consecutive evaluations. Secondary corroboration: fresh
value CE improving while ladder decisiveness and strength stay flat.

**DISAGREEMENT ON RECORD — MCTS_ADVICE during the first post-fix leg.**
Fable argues the first leg should isolate the q-transform fix, since advice
ON adds a second simultaneous variable and we will not be able to attribute
a change. The objection is methodologically correct. **My ruling: keep
advice ON, because the confound is MEASURABLE rather than assumed.** The
advice path is a zero-init graft — it contributes exactly nothing until
`advice_out` grows off zero — and we log `advice_out_norm` and
`advice_grad_share` every train_step. So: if `advice_out_norm` stays ~0
through the leg, there is provably no confound and we got the fix cleanly;
if it grows, we know precisely when attribution became ambiguous and can
split the leg there. Flipping it off would also cost a reboot and restart
of a run that is already rolling. Revisit if the telemetry shows
`advice_out_norm` climbing early.

### Cycle 5 — 2026-07-28 — T3 LIVE: new box created, campaign running

User granted full autonomy over the Vast credits ("do whatever... don't
request my help again"). Acted on it.

**New box: instance 46142270** — 128 cores @ 3.7 GHz, RTX 4090, 1032 GB
RAM, 80 GB disk, reliability 1.00, **$0.536/hr** (~$37 for a 70h run).
Old box 45230879 DESTROYED (superseded; its lineage is on HF, its token was
plumbed to the new box shell-internally without ever surfacing).

Chose reliability 1.00 over the marginally cheaper 96-core option: an
unattended 70h run loses more to a dead box than the price delta.

**Caught before launch:** my own first pick (RTX 5060 Ti, best cores/$) is
**Blackwell (sm_120)** and the pinned `pytorch 2.4.0 / cu124` image does not
support it — that would have been a launch failure. Filtered to Ada-or-older.

**CORRECTION to cycle 3's projection.** I predicted the GPU would sit at
2-6% utilisation. The live box shows **71%**. My projection was for ONE
serial decision; with 100 concurrent spool workers (13 CUDA + 87 CPU) the
GPU is genuinely well-used. The measurement was right, the extrapolation to
a many-worker box was wrong — and it means choosing the 4090 over the
weak-GPU option was correct for the wrong stated reason. Keep the
cores-per-dollar figure of merit, but do NOT conclude the GPU is idle.

**Bug found and fixed by launching (51ff30d).** Setting `DRILL_RATIO=0` —
per the standing decision that the broken drills stay at 0 — made the five
scenario-mix ratios sum to 0.95, and `sim_self_play` exits rc=2. The
supervisor then relaunched into the identical error 20 times. The ratios
were five independent env vars whose DEFAULTS happened to sum to 1, so
changing any ONE broke the launch. Fixed properly rather than by patching
env: LADDER_RATIO is now DERIVED as the remainder, so any single-ratio
change is valid by construction, and an over-subscribed mix fails loudly at
onstart instead of 20 relaunches deep. (Likely why the old box was sitting
`exited`, too.)

**Live config:** seeded from `selfplay_seed_20260718.pt` (2.30M, the
measured peak), `drill=0`, ladder 0.45, `MCTS_ADVICE=1`, 100 spool workers,
and — the point of the whole exercise — the **fixed q-transform**. Verified
running: mix line correct, holdout set full, games training.

### Cycle 3 — 2026-07-28 — T3: profiled the box requirement; the current box is the WRONG SHAPE

User (2026-07-28): "don't feel beholden to the current Vast.ai box... use
profiling to determine the characteristics of the box you most want."

**MEASURED — the workload is CPU-bound, not GPU-bound.** One MCTS decision
(32 sims), real 5M net vs a tiny net that makes the forward ~free:

```
real 5M net (d256/6L)   9094 ms / decision
tiny net (forward ~0)    940 ms / decision
-> CPU side (sim + encode + enumerate + MCTS bookkeeping) = 940 ms
-> model forward on CPU                                   = 8154 ms (90%)
```

On a GPU box the forward collapses to ~0.5-2 ms/leaf, i.e. 16-64 ms per
decision, so:

| GPU leaf time | decision | GPU share | **CPU share** |
|---|---|---|---|
| 0.5 ms | 956 ms | 1.7% | **98.3%** |
| 1.0 ms | 972 ms | 3.3% | **96.7%** |
| 2.0 ms | 1004 ms | 6.4% | **93.6%** |

**A 4090 sits ~2-6% utilised during rollout.** We are paying for silicon we
do not use. The figure of merit is CPU throughput per dollar
(cores x GHz / $), subject to **>= 2 GB RAM per core** — spool workers are
separate processes, each with its own torch runtime.

**Market scan (50 offers, >=32 effective cores, 1 GPU, <$1.50/hr):**

| cores | GHz | $/hr | GPU | RAM | GB/core | rel | id |
|---|---|---|---|---|---|---|---|
| **96** | **3.4** | **0.268** | RTX 5060 Ti | 252 | 2.6 | 1.00 | **33925788** |
| 80 | 3.6 | 0.283 | RTX 4060 Ti | 252 | 3.1 | 1.00 | 46008923 |
| 128 | 3.7 | 0.486 | RTX 4090 | 336 | 2.6 | 0.97 | 42902076 |
| *64* | *?* | *0.500* | *RTX 4090* | *516* | *8.1* | — | *45230879 (current)* |

**DECISION: replace the current box with 33925788** (96 cores @ 3.4 GHz,
252 GB, RTX 5060 Ti, $0.268/hr, reliability 1.00): **1.5x the cores at 54%
of the price**, and a GPU still ~20x larger than rollout needs. Backup for
headroom if T2 lands a bigger net: 42902076 (128 cores @ 3.7 GHz + 4090,
same price as today's box for 2x the cores). Creating an instance is a
CREATE, which the user launches, and it needs `-e HF_TOKEN=...` supplied by
them (env is baked at create time).

**SECOND FINDING, and it matters: `DRILL_RATIO=0.15` on the current box.**
The standing decision (BACKLOG 2026-07-21) is that the capability drills
are broken and **DRILL_RATIO stays 0**. The box was created ~2026-07-18,
BEFORE that verdict, and Vast bakes env at create time — so **the entire
regressing leg (2.75M -> 3.74M) trained with 15% known-broken drill
scenarios.** That is a second concrete contributor to the regression,
independent of the q-transform bug, and it is corrected simply by creating
the new box with the right env.

Also confirmed: `HF_SEED_FILE=selfplay_seed_20260718.pt` already seeds a
FRESH instance from the 2.30M checkpoint — the peak region per the ladder
(2.30M vs 2.75M was 3-1-4, statistically tied). So no upload is needed and
the new campaign starts from the peak by construction.

Onstart delivery is a thin bootstrap (`scripts/vast_onstart_bootstrap.sh`)
that git-pulls the real script, so the q-transform fix, the advice signal
and the telemetry all reach the new box automatically.

### Cycle 2 — 2026-07-28 — T1 ROOT CAUSE FOUND AND FIXED: the Gumbel q-transform

**Fable found the regression's root cause; I verified and landed it (4fecbca).**

`sigma(q) = (c_visit + max_N) * c_scale * q` ran with `c_scale=1.0` on RAW
q in [-1,1] and NO rescale. That multiplies Q differences by ~50-82 and
saturates the softmax, so `extract_gumbel_policy_target` emitted a near
ONE-HOT label: **every training iteration distilled argmax instead of a
policy improvement.**

MEASURED (Fable, real box search config, 32 sims, 18 offered-recruit states):

| | 2.75M | 3.74M |
|---|---|---|
| recruit prior -> target mass | 0.159 -> 0.142 (**crushed to <=0.006 in 12/18**) | 0.188 -> 0.355 |
| end_turn prior -> target mass | 0.126 -> **0.278** | 0.174 -> 0.054 |

Repeat probe: recruit target mass **0.000-0.002 across 4 independent
searches** of the same state — systematic teaching, not noise.

Also measured and worth keeping: the value head is NOT pro-hoarding (it
moved strongly PRO-recruit, +0.72 by 3.74M, while behaviour hoarded more —
opposite directions), and the draw-z=0 framing is **disconfirmed** (draws
were only 4-23% of training games). Both earlier hypotheses are dead; the
amplifier was the mechanism all along.

**Fix = match the reference, verified against mctx source by me
independently** (`qtransform_completed_by_mix_value`: value_scale=0.1,
maxvisit_init=50, rescale_values=True). The subtle part, and how the bug
survived review: our CONSTANTS were the paper's — what was missing was the
paper's NORMALIZATION. `_rescale_q` (min-max to [0,1]) + `c_scale=0.1`,
applied through a vector-valued `_gumbel_sigma` used by BOTH
sequential-halving selection and target extraction so they cannot drift.
Buys bounded sharpening (~8.2 logits at 32 sims, independent of the value
head's scale) and OFFSET INVARIANCE. Full suite green (572 fast + 10 slow);
derivation in `docs/design_constants.md`; 6 tests in
`tests/test_gumbel_qtransform.py`.

**CORRECTION (cycle 4, Fable's T1-B):** the "side-to-move bias drifting
0.06 -> 0.37" line above is WRONG and is retracted. A clean boundary probe
at natural end_turns shows the bias is **inherited from the SL prior**
(+0.43/+0.65), dipped at 2.75M and rebounded at 3.74M — it is structural,
not campaign drift. The encoder's side-identity feature is EXONERATED
(mirror-state V is antisymmetric to within ~0). My original number
conflated the structural bias with the genuine badness of ending a turn
prematurely at offered-recruit states. Leading hypothesis now: per-side
recorded-state imbalance (the stronger side logs more decisions, so
winner-perspective states dominate the value target). Gated on a
covariance test before any fix lands.

**Review fix (bbe1132):** Fable's review of 4fecbca caught that `_score`
rescaled over RAW `edge.q_value`, which is 0.0 for unvisited edges — on an
all-negative node those zeros anchored the window's top, distorting
sharpness state-dependently and diverging from the very reference the fix
cites. Factored `_completed_q()`; both call sites now consume it.

**Open, flagged to Fable:** the min-max rescale always spans full [0,1], so
sharpening is uniform across nodes and Q-scale information is discarded.
That is reference behaviour and I implemented it rather than inventing a
variant — but it deserves a second opinion.

**Consequence for T3:** the regressing leg was trained under a broken
target. Seeding the next campaign from the PEAK (2.75M / 2.30M) with this
fix landed is now the plan, per `docs/eval_20260728.md` §0.

### Cycle 1 — 2026-07-28 — T2: the scaling constraint is SEQUENCE LENGTH, not parameters

MEASURED (CPU, 4 torch threads, 40 real mid-game states, current 5M net):

| | encode | forward | enumerate | total | fwd share |
|---|---|---|---|---|---|
| 5M (d256/6L) | 3.6 ms | **95.7 ms** | 5.5 ms | 104.8 ms | **91.3%** |
| 14.8M (d384/8L) | 3.8 | 225.7 | 5.7 | 235.1 | 96.0% |
| 38.9M (d512/12L) | 4.3 | 616.5 | 6.2 | 626.9 | 98.3% |

So on CPU the forward DOMINATES the decision (91%), and naive parameter
scaling costs ~linearly in wall-clock (x2.24 for 14.8M, x5.98 for 38.9M).
My prior assumption that "the rollout is CPU/sim-bound so a bigger net is
nearly free" is REFUTED for the dev box. (On the CUDA box the split will
differ — the 2026-07-22 perf campaign found the CPU side was the pie there
— but that is unverified while the box is down, so treat it as open.)

Then the diagnostic that matters. Sequence composition on a 29x22 map:

```
SEQ LEN = H 638 + U 1 + R 8 + 2 = 649   -> hex tokens are 98.3% of it
H=638 seq=649  forward=120.3 ms
H=319 seq=330  forward= 46.0 ms   (2.6x faster for 2x fewer tokens)
H=159 seq=170  forward= 20.6 ms   (5.8x faster for 4x fewer tokens)
```

**The model spends essentially all of its compute doing self-attention over
every hex of the map, and the cost is SUPER-linear in hex count** (quadratic
attention + linear FFN). Parameter count is not the binding constraint;
sequence length is.

**T2 thesis (revised):** the cheapest path to a larger network is to stop
paying for a 638-token hex stream, then reinvest the savings in depth/width.
A 3-4x sequence cut buys roughly the 14.8M net for free.

**Design direction to evaluate next (NOT yet decided — needs Fable's
adversarial review before any code):** every hex currently gets a token, but
the pointer-network target head masks all illegal destinations anyway, so
most hex tokens only ever act as context. Candidate: restrict hex tokens to
a RELEVANT set (union of unit reach, attack-adjacent hexes, villages, castle
network) plus a context margin. Risks to settle first: (a) the target head
indexes hex tokens, so every legal target MUST keep a token — the relevant
set must provably cover the legality mask; (b) it changes what the net sees,
so warm-starting an existing checkpoint shifts its input distribution;
(c) a per-decision "relevant set" computation must not cost more than it
saves, and must be a pure function of OBSERVABLE state (mask contract,
CLAUDE.md §6). Alternative if (a)/(b) prove ugly: keep all hexes as pointer
TARGETS but take them out of the self-attention stack (units cross-attend to
hexes), which preserves the action space exactly.

### Cycle 0 — 2026-07-28 — setup
- Wrote this doc; relayed the mandate to Fable; started the persistent loop.
- Inherited state: advice signal landed; lineage regression + RCA losses
  documented; hoarding traced to a learned preference, not mechanics.
- Next: T1 value-attribution probe (Fable); T2 throughput/scale assessment
  (Claude); confirm box availability.
