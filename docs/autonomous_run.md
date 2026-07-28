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
