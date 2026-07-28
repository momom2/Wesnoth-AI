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
