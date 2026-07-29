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
8. **The loop runs on a RECURRING CRON (job 23fee23c, `7,37 * * * *`), not
   a self-re-arming ScheduleWakeup chain.** The chain died twice — once
   because I forgot to re-arm (cycle 19), once because the link itself
   failed after a confirmed schedule (cycle 20) — and each death cost a
   user rescue that is no longer available. A cron fires independently of
   whether any single cycle remembers anything. Do NOT go back to the
   chain. If a future cycle finds no cron, recreate it before doing
   anything else.

---

## Standing decisions

- (2026-07-29) **T2 relevant-hex flag: PLUMBED AND VALIDATED-AS-MECHANISM,
  NOT validated-as-drop-in.** Warm-start value MAE is **0.217** vs the
  net2net acceptance precedent of ~0.017 — an order of magnitude over. The
  weights load; the function they compute does NOT carry over, because
  cutting ~62% of hex tokens shifts the attention/pooling statistics. So:
  **cross-flag evaluations are INCOMPARABLE at warm-start.** The flag's
  first real use must be a FINE-TUNED leg gated on `fresh_value_ce`
  recovering to its pre-switch level (not a fixed iteration count), or
  better, start the next campaign leg flag-ON from its seed and compare
  leg-vs-leg. Do NOT run a same-weights ON-vs-OFF eval and read it as the
  encoding's ceiling — it measures warm-start damage.
- (2026-07-29) **Speedup numbers, kept distinct:** 4.3-4.8x is the
  FORWARD-COMPONENT figure. End-to-end decision throughput is Amdahl-bounded
  by the 91% forward share at **<= ~3.4x**, and the trainer's padded batches
  get ~2x. Plan with ~3x.

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

### Cycle 24 — 2026-07-29 — T2-C fails its own gate: the encoding is NOT a drop-in

**Fable ran the warm-start validation it specified, and it failed the bar
it set** — the most useful kind of result.

| | flag-OFF vs flag-ON warm-start (220 shared states) |
|---|---|
| **value MAE** | **0.217** (p90 0.446, max 0.708) |
| aux-margin MAE | 0.253 (max 0.663) |
| hex fraction ON/OFF | mean 0.385 (min 0.068, max 0.829) |

Against the net2net precedent (~0.017) that is **an order of magnitude over
the bar**. Cutting ~62% of hex tokens moves the value head by ±0.2-0.45 on
its ±1 scale: the weights load, the FUNCTION does not carry over.

Fable then **cancelled its own part-2 experiment** rather than run it: a
same-weights ON-vs-OFF ladder eval would measure warm-start damage, not the
encoding's ceiling, so the 200 games buy nothing. Its earlier prediction
("degradation <= noise") was wrong and it said so plainly. Recorded as a
standing decision above so a future cycle cannot misread T2's status.

**Both judgement calls I sent it came back with corrections:**

1. **The speedup number I had been quoting is the component figure.**
   4.3-4.8x is the forward-cost win; end-to-end decision throughput is
   Amdahl-bounded by the 91% forward share at **<= ~3.4x**, trainer batches
   ~2x. My "~3x instinct" was right as the planning number and the 4.3-4.8x
   should not be repeated as a throughput claim.
2. **My ingest drop-and-continue was right for the transient but incomplete.**
   Adopted its escalation (3d8650a): >=50% of an iteration's games rejected
   for 2 CONSECUTIVE iterations exits code 6, which lands in the
   supervisor's tripwire range so it writes ABORTED_6 and blocks
   auto-relaunch. Drop handles noise; a systemic mismatch must halt rather
   than quietly train on starved iterations — the same silent-boundary class
   as the three bugs already found. A streak is required so one bad
   iteration cannot halt a healthy run.

598 fast + 3 slow green.

### Cycle 23 — 2026-07-29 — T2 fully plumbed (holdout basis stamp closes it)

**Landed edc8349:** the holdout probe stores its index basis and
`load_holdout` DISCARDS a probe whose basis differs. This one is subtle
enough to state: the probe is persisted *precisely so* holdout CE is one
continuous comparable curve across restarts (2026-07-18 — resampling per
relaunch made levels jump 0.44<->0.88 and made the capacity question
unanswerable). Restoring a foreign-basis probe would keep that curve
looking continuous while silently making it a DIFFERENT measurement — the
worst failure mode for the one metric relied on for cross-restart
comparison.

**T2 is now fully plumbed, all default OFF:** shared source-of-truth set
(c4a2504) -> encoder threading (a964fbe) -> hex_subset marker + firing
superset assert (f4a0bf8) -> config gate + worker/learner payload rejection
(685b1c2) -> holdout basis stamp (edc8349). 597 fast + slow green
throughout. Remaining before any A/B: Fable's warm-start validation
(value MAE + same-weights ladder eval), now dispatched as T2-C.

**Campaign (post-restart, coherent advice):** 1 iteration —
`fresh_value_ce` 0.7698, `advice_out_norm` 0.3066 (resumed from 0.3073, so
the weights carried), `advice_fire` 0.040, boundary +0.014. 101 workers.
Tripwire |mean| 0.014. Too early to read anything.

**Two judgement calls sent to Fable rather than assumed:** whether the
4.3-4.8x projection survives the actual wired encoder (my H=870->119 was
ONE state; pooled f=0.30 predicts nearer 3x, and the projection assumed
unpadded per-state forwards), and whether my ingest rejection should DROP
the game (my choice: a stale worker after a flag change is the realistic
cause, so dropping costs an iteration where aborting costs the run) or be
fatal.

### Cycle 22 — 2026-07-29 — relevant-set config gate complete; the seam is now guarded

**Loop scheduling: cron 23fee23c confirmed alive** (`CronList`), so this
cycle did NOT add a competing ScheduleWakeup — two schedulers would
double-fire and burn tokens. Rule for future cycles: check `CronList`
first; the cron IS the loop.

**T2 wiring finished (685b1c2), still default OFF.** `--relevant-set-hexes`
-> `TransformerPolicy` -> BOTH encoders -> spool workers, flag recorded in
the checkpoint.

The load-bearing part is the seam, not the flag. The hex stream defines
what `target_idx` MEANS, so ingesting a worker payload built under the
other basis poisons every replayed transition **silently** — no exception,
just wrong gradients. So workers STAMP each payload with its basis and the
learner REJECTS a mismatch loudly and drops the game. A stale worker after
a flag change is the realistic cause; dropping its games costs one
iteration, accepting them costs the run.

**This is the third bug of that class this run** (dead spool telemetry,
`_combine_stats` swallowing advice stats, dead acting-side advice), so the
guard is a test that READS the boundary: AST-checks that the worker parses
the flag, builds its policy with it, and stamps the payload; and that the
learner forwards the flag and rejects a mismatch. The pattern is now
explicit — *anything that must hold across worker/learner gets a boundary
test, not a comment*.

596 fast + 4 slow green, including the spool-workers e2e that exercises
this exact seam.

**Remaining for T2:** only the holdout-stamp guard (discard a holdout file
whose basis differs — its stored indices are meaningless in the other
space), then the warm-start validation Fable specced (value MAE + a
same-weights ladder eval) before any A/B.

Campaign: restarted onto coherent advice at 07:11; first post-restart
iteration still in flight at cycle time.

### Cycle 21 — 2026-07-29 — loop moved to a cron; box restarted onto coherent advice

**Scheduling fixed STRUCTURALLY.** The self-re-arming wakeup chain died
twice (cycle 19: I forgot; cycle 20: the link failed after a confirmed
schedule), and the user is away ~3 days with no further rescues available.
Replaced with recurring cron **23fee23c** at `7,37 * * * *` — it fires
whether or not any cycle remembers to re-arm, and auto-expires after 7 days
(covers the mandate). Recorded in the guardrails so a future cycle
recreates it rather than reverting to the chain.

**Box restarted onto the acting-side advice fix — and I changed my mind
about whether to.** My first instinct was to leave the leg clean, since
advice-at-act-time had never run and that made this a pure q-transform
measurement. Reasoning it through reversed that:

The state was NOT "advice inert, baseline clean". The trainer attaches
advice tokens to stored states and shapes the gate against MCTS targets
**generated without advice** — so the gate was being trained on a
distribution where advice had NO CAUSAL ROLE in the target. That is a
train/act mismatch: advice is uninformative noise with respect to the
objective it is fitted against, and `advice_out_norm` climbing 0.19 -> 0.31
under that objective is fitting noise, not learning a gate. A defect beats
a clean baseline, so: restart.

Cost ~1 iteration. Verified after reboot: box on 7034de0, 101 workers,
`pgrep` confirms **`--mcts-advice` is now actually in the worker command
line** (the check that would have caught the original bug), decision_step
2,395,399 carried not reset.

Leg is now split: iterations 0-16 = q-transform only, acting-side advice
dead; iterations 17+ = coherent advice loop. Recorded so no later analysis
treats the leg as homogeneous. T3-A already ruled attribution is not being
bought this run, so the split costs little.

Campaign at 16 iterations pre-restart: `fresh_value_ce` 0.3803,
`advice_out_norm` 0.3073.

### Cycle 20 — 2026-07-29 — the ACTING half of the advice signal never ran in production

**My process error first: the loop stopped because I ended cycle 19 without
calling ScheduleWakeup.** The user had to restart it. Re-armed; noting it
so the failure mode is on the record rather than silently repeated.

**Found while wiring the relevant-set config gate: `grep advice
tools/selfplay_worker.py` returned NOTHING.** The spool workers generate
every training game, and they built BOTH `TransformerPolicy` and
`MCTSConfig` with no advice. So with `MCTS_ADVICE=1` on the box and the
learner dutifully reporting `advice_out_norm` climbing, **all 100 workers
played every game with zero advice conditioning at the search root**, and
each worker's checkpoint load was dropping the grafted advice weights as
unexpected keys.

The design's acting half — root-conditioned priors, the entire point of the
prospective advisor — has never run in production. Only the trainer
reforward saw advice tokens.

**This REVISES cycle 12.** I recorded attribution for this leg as
"ambiguous because advice is active". It is much less ambiguous than that:
the games were generated identically to an advice-OFF run, and only the
gradient path differed. My mitigation-by-telemetry reasoning was sound, but
the thing I was watching (`advice_out_norm`) was measuring the LEARNING
half while the ACTING half was dead — so the confound I conceded to Fable
was largely imaginary.

Fixed (9a17d65): worker parses `--mcts-advice` and honours it in the model
(also enabling it when the CHECKPOINT carries the flag, so grafted weights
load) and in the search config; the learner forwards it in the spawn tail.
Tests AST-inspect the seam rather than trusting a comment — the third bug
of this exact class (T1-H telemetry, `_combine_stats` swallow, this), so
the pattern is now: **anything that must hold across the worker/learner
boundary gets a test that reads the boundary.**

**Flaky test observed, NOT caused by this change:**
`test_inference_seam::test_mcts_search_through_seam_matches_direct` failed
once in the full tier (one edge differing by a single visit), then passed
3/3 in isolation AND on the stashed prior tree, and the next full tier was
green (594). Recording the hypothesis rather than dismissing it: the
q-transform rescale normalises Q to [0,1], which AMPLIFIES tiny float
differences between the direct and seam paths, so near-ties can flip. If it
recurs, that is where to look.

### Cycle 19 — 2026-07-29 — the superset assert SHIPS, and it fires

**Landed f4a0bf8:** `hex_subset` marker on RawEncoded/EncodedState (stamped
in `encode_raw`, propagated through BOTH `encode_from_raw` paths — the
batched one is what the trainer uses, and T1-H's lesson was that the
production path is exactly where guards quietly die), plus the assert at
the two raw-position lookups that can drop an offered action (the landable
loop and `_recruit_hex_mask`).

Why mode-aware rather than unconditional: under the FULL board a position
miss legitimately means off-board. Under the subset it means **the mask
offered an action with no token to point at** — an unorderable action, with
no error anywhere. Silent action-space shrinkage is the whole risk of this
design, so the guard exists precisely for it.

**The test asserts that the assert FIRES**, not that it exists: shrink the
relevant set behind the encoder's back to a third of its hexes while the
mask still offers them, and `enumerate_legal_actions_with_priors` must
raise `"relevant-set gap"`. It does. Also pinned: the marker really reaches
EncodedState — without that the guard would be dead code that passes every
test.

592 fast + 4 slow green. Still default OFF; campaign untouched.

**Campaign at 12 iterations — the value metric is NOISIER than any trend
claim I have made.** `fresh_value_ce` jumped to **1.1860** after touching
0.3912 the iteration before. Full within-process series:

```
1.018  0.530  0.718  0.491  0.485  0.500  0.709  0.391  1.186
```

That is a 3x swing between consecutive iterations. Recording it plainly:
the "-51% then plateau" reading from cycle 16 stands as a description of
the early drop, but **no trend claim on this metric is supportable at this
noise level** — including any I might be tempted to make from a future low
reading. What would be needed is a fixed-probe-set evaluation rather than
one computed on each iteration's own incoming games.

### Cycle 18 — 2026-07-29 — encoder wiring landed (default OFF); H 870 -> 119 measured

Took the wiring myself (Fable's context spent). **Landed a964fbe:**
`encode_raw(relevant_set=)` + `GameStateEncoder(relevant_set_hexes=False)`.

Measured on a real pool state: **full board H=870 -> relevant H=119**
(0.137 there; T2-A's pooled mean is 0.30). That is the saving that buys
`d_model` 256->512 at equal wall-clock.

The flag changes the ACTION SPACE's index basis, so the properties that
matter are ordering and determinism, and all four were VERIFIED rather than
assumed:

```
relevant is a SUBSET of full:            True
ordering is the filter of full order:    True    <- filtered, never re-sorted
deterministic across re-encode:          True
default encoder unchanged (full board):  True
```

The third is the one that would have bitten silently: the trainer
re-encodes STORED states and replays `target_idx` against them, so a
re-sort or any set-iteration nondeterminism corrupts every replayed
transition without an error. 590 fast + 4 slow green (incl. spool-workers
e2e, ladder export validation).

**Still OFF by default — the running campaign is untouched.** Remaining:
the shipping debug assert, the config gate with worker/learner flag
agreement (loud rejection at the seam), and the holdout stamp guard. All
specced at insertion points in cycle 17.

Campaign at 11 iterations: `fresh_value_ce` **0.3912**, a new low
(series 1.018, 0.530, 0.718, 0.491, 0.485, 0.500, 0.709, 0.391 — noisy,
but the floor keeps dropping). `advice_out_norm` 0.2742, still monotone.

### Cycle 17 — 2026-07-29 — relevant-hex core landed (INERT); wiring specced to insertion points

**Landed c4a2504:** `relevant_hex_positions` / `relevant_hexes_in_slot_order`
in `visibility.py`, beside `hexes_in_slot_order` where the slot contract
lives, built from the SAME primitives the mask consumes (shared code, not a
mirror — the 2026-07-16 lesson). 4 contract tests; 586 fast green.
**Nothing calls it, so behaviour is unchanged** — verified by grep, not
assumed.

Determinism is BY CONSTRUCTION: the set derives only from deterministic
observable-state components, and ordering comes from FILTERING the
canonical `(y,x)` sort rather than re-sorting, so it cannot drift even if
set-iteration order does. This is the requirement most likely to bite
subtly, because the trainer re-encodes stored states and replays
`target_idx` — any nondeterminism corrupts every replayed transition.

Fable stopped here deliberately (context budget) rather than ship a
half-verified encoder diff. Correct call by this run's standard, and it
left the derivation rather than a vague TODO:

**REMAINING WIRING — each item bounded, insertion points exact:**
1. `encoder.py:997` — `hexes = relevant_hexes_in_slot_order(gs) if
   relevant_set else hexes_in_slot_order(gs)`; `relevant_set: bool = False`
   on `encode_raw` (:954), threaded from a
   `GameStateEncoder(relevant_set_hexes=False)` ctor flag at BOTH call
   paths (`encode()` :446/:455 and the raw-cache path the trainer uses);
   add `hex_subset: bool` to RawEncoded/EncodedState.
2. **Shipping debug assert** — the two raw-position lookups where an
   excluded hex would silently vanish: the landable loop's
   `pos_to_hex.get(_lpos)` (action_sampler ~:1189) and `_recruit_hex_mask`
   (~:1441). `if __debug__ and encoded.hex_subset: assert _j is not None`.
   Gated on the marker so full-hex mode is untouched. Test by
   monkeypatching the set to drop one landable hex — the assert MUST fire.
3. **Config gate** `--relevant-set-hexes` -> TransformerPolicy -> both
   encoders, AND the spool workers must get the same flag; the payload
   carries `"relevant_set": bool` and ingest REJECTS a mismatch loudly.
   (T1-H's lesson applied in advance: the worker/learner seam is exactly
   where a silent mismatch would corrupt every replayed index.)
4. **Persistent-crossing guards** — live buffers are safe by construction
   (in-memory, flag is process-lifetime), but stamp the CHECKPOINT with the
   flag and discard a HOLDOUT file whose stamp mismatches: its stored
   indices are meaningless in the other hex space.

**Warm-start validation plan (specified, NOT run):** value MAE between
flag-OFF and flag-ON-warm-started forwards on ~200 shared states (net2net
precedent: ~0.017 acceptable); short ladder eval same-weights ON vs OFF
(~100-200 games, expect degradation <= noise); a smoke leg watching
`fresh_value_ce` re-settle over ~10-20 iterations. Gate any A/B campaign on
the first two; treat cross-flag evals as INCOMPARABLE until then.

### Cycle 16 — 2026-07-29 — T2-A says GO (4.3-4.8x); TWO of my cycle-15 claims corrected

**T2-A verdict: the relevant-set design SURVIVES measurement.** Fable, over
**1,840 decisions across 10 ladder maps** (H = 696-2162):

- **|relevant set| / board: mean 0.30, median 0.29, p90 0.44-0.62, p99
  0.58-0.76, max 0.789.** f > 0.5 on ~10% of decisions, > 0.7 on ~1%.
- **Superset check: ZERO violations in 1,840 decisions.** Every
  mask-offerable target (move, attack, recruit) was in the set, with the
  set built from independent primitives rather than from the mask. The
  kill case (0.6-0.7 of board, ~1.4x) is not close.
- **Reach dominates** (0.22-0.32); the statics I worried would bloat it are
  tiny — castles 3-6%, villages ~2%, network ~1%, visible units ~2%. So
  the myopia mitigation is nearly FREE, which was the objection that made
  me shrink the expected win.
- **Throughput gain 1/E[f^alpha] = 4.3x (alpha=1.27) to 4.8x (alpha=1.38)**
  on the rollout forward (the measured 91% bottleneck). At equal
  wall-clock that buys roughly **d_model 256 -> 512, i.e. the 5M net ->
  ~20M class**. Honest caveats from Fable: the TRAINER's batched forward
  pads to the batch max (~p99 0.6) so its win is nearer 2x (fine — not the
  bottleneck), and f will GROW if a stronger policy fields bigger armies,
  so 4.3x is today's number, not a constant.

**CORRECTION 1 — there is no instrument gap.** Cycle 15 recorded
"boundary_sum NaN on 2 of 5 iterations". Fable deduced this was impossible
within one process (the FIFO only grows) and must mean a learner RESTART.
Checked: correct. The "missing" rows were pre-telemetry iterations from
EARLIER learner processes that my log-tail conflated with the current run.
Since the last restart, boundary_sum is present on EVERY iteration. My
report of a gap was a parsing error, not a defect.

**CORRECTION 2 — the headline number was inflated by the same mistake.**
Cycle 15 claimed `fresh_value_ce` 1.387 -> 0.485, "-65%, trending". The
1.387 came from a DIFFERENT learner process. Within the current process the
series is:

```
1.0182  0.5295  0.7178  0.4910  0.4854  0.5002
```

i.e. **-51%, and FLAT for the last three iterations** (0.491, 0.485,
0.500). The drop is real; the trend is not continuing. "Improved then
plateaued" is the accurate description, and I should not have quoted a
cross-process delta as a trend.

**Tripwire, now on 6 clean readings:** rolling mean **-0.0077**, |mean|
0.008 vs the 0.25 trigger — and it oscillates in BOTH signs (-0.143,
-0.015, +0.160, -0.019, +0.069, -0.098), which is what no systematic bias
looks like. T1-F closure is solid on live data, not just on probes.

`advice_out_norm` monotone across all six: 0.1947 -> 0.2598.

### Cycle 15 — 2026-07-29 — fresh_value_ce is now trending, not bouncing

Eight iterations. The series that matters:

| loss | fresh_value_ce | advice_out_norm | boundary_sum |
|---|---|---|---|
| 2.9346 | 1.3870 | — | — |
| 2.8640 | 1.0182 | 0.1947 | -0.143 |
| 2.6883 | 0.5295 | 0.2139 | -0.015 |
| 2.8711 | 0.7178 | 0.2276 | — |
| 2.7082 | 0.4910 | 0.2381 | -0.019 |
| 3.0504 | **0.4854** | **0.2552** | — |

**`fresh_value_ce` 1.387 -> 0.485 over six points (-65%).** In cycle 14 I
cautioned that the 0.53 was not an achievement because the series bounced;
with six points the decline survives that bounce, so this now reads as
signal rather than noise. Still NOT claimed as a win — the metric depends
on the incoming game distribution, and the campaign is young.

**Loss is flat-to-noisy** (2.93..3.05, no trend) — worth stating because it
is the metric one would naively quote. The generalization metric moving
while train loss does not is the expected shape when the value head stops
being fed a saturated target.

`advice_out_norm` monotone across all five readings (0.195 -> 0.255).
Tripwire: boundary rolling |mean| 0.059 over 3 readings, clear of 0.25.

**Minor instrument gap noted, not chased:** `boundary_sum` is NaN on 2 of 5
telemetry-carrying iterations, implying <4 pairs despite 24 games each.
Flagged to Fable as low priority; the tripwire still has readings.

T2-A dispatched: the relevant-set size measurement that gates the whole
sequence-length scaling design, including the empirical superset check
against the legality mask. Explicitly told Fable that "the set is 60-70% of
the board and the win is ~1.4x" is an acceptable answer that would kill the
direction — better learned from a measurement than after an encoder rewrite.

### Cycle 14 — 2026-07-29 — ablation harness landed; fresh_value_ce is NOISY, not a trend

**Harness landed (0a7ca78), unused until there is a checkpoint worth
testing.** The property that makes it evidence rather than decoration: it
reproduces TWO known-nulls before being trusted — OFF/OFF returns exactly
0.5 (0-2-0), and ON/OFF on a zero-init graft returns the theoretically
required null (gate=0 => identical play). Fable also priced it honestly
against my 30-minute ask: `sims=8`/400 games projects to **60-130 min, not
30**; fitting configs are `sims=4 --max-turns 24` at 400 games, or
`sims=8` at 200 games with the CI widening to ~+-48 Elo. A box-side null
settles the true rate before the live config is sized.

**Campaign series (6 iterations), with the honest reading:**

| iter | loss | fresh_value_ce | advice_out_norm |
|---|---|---|---|
| 0 | 2.8640 | 1.0182 | 0.1947 |
| 1 | 2.6883 | **0.5295** | 0.2139 |
| 2 | 2.8711 | **0.7178** | 0.2276 |

`fresh_value_ce` went 1.02 -> 0.53 -> 0.72: **not monotonic**. My cycle-12
note already said two points were not a trend; this is the confirmation,
and it is a caution against reading the 0.53 as an achievement. Loss
likewise bounced (2.86 -> 2.69 -> 2.87). Nothing here yet distinguishes
improvement from noise, and the pre-committed discipline is to keep
collecting rather than narrate a story around three points.

`advice_out_norm` is the one clean monotone series: 0.1947 -> 0.2139 ->
0.2276, i.e. the gate keeps learning up from its zero-init graft.

Tripwire: boundary_sum rolling |mean| still well under 0.25.

### Cycle 13 — 2026-07-29 — T1-F closed on evidence; attribution deliberately NOT bought

**T1-F closed, and the reconciliation is now measured rather than argued.**
Fable probed the one checkpoint it had never measured — the 2.30M SEED this
campaign started from — and got **-0.190 fogged (n=73)**. Lineage series:

| SL | **2.30M (seed)** | 2.75M | 3.74M |
|---|---|---|---|
| +0.43/+0.65 | **-0.19** | +0.24/+0.45 | +0.43/+0.62 |

So the live ~0 needs no "fogless dilution" or "fixed-transform" explanation:
**the seed simply does not carry the bias.** Probes and live instrument
agree once lineage is accounted for. Also settles my named residual — with
the offset ~0 on the actual training distribution, end_turn children carry
only the legitimate unspent-moves negativity, not a binding artifact.

**But the same series shows the bias RE-EMERGED once before** (-0.19 ->
+0.35 -> +0.52 across the old leg), so the tripwire is load-bearing, not
ceremonial. **TRIPWIRE ADOPTED: 8-iteration rolling |mean boundary_sum| >
0.25 reopens T1-F.** Two-sided (2.30M shows it can point either way), and
the threshold already accounts for the pair FIFO pooling ~20% fogless games
(a fogged-only +0.30 reads ~+0.24 pooled). Checked each cycle from the
train_step series; current rolling |mean| = 0.079 over 2 readings — clear.

**T3-A: attribution is NOT worth buying now — accepted with its reasoning.**
Fable's statistics, which corrected my read of our own logs: the `+-1.27` /
`+-0.52` I quoted is the PER-STATE spread of the ~256-state probe; the
MEAN's SE is only ~0.03-0.08. What actually binds an arm comparison is
between-iteration variance, and early-leg that is **trend-dominated**
(1.02 -> 0.53 in one iteration — the curve is falling faster than any
plausible arm gap), so a split started now measures the TREND, not the arm.
On the strength endpoint, each-vs-reference needs ~800 games/arm for +50
Elo, and option (a) buys that with doubled spend and n=1 run per arm whose
run-level variance is unmeasured and plausibly larger than the effect.

Staged plan adopted:
1. **Now:** dose telemetry only. If `out_norm` plateaus small with
   `grad_share ~1.5%`, that alone justifies "effect <= noise".
2. **At checkpoint K, IF the leg improves:** inference-time ablation —
   same weights, advice attached vs not at act time, ~400 ladder games
   (~$1-2, zero training cost, +-34 Elo resolution). Bounds the acting
   channel; does NOT measure trunk shaping (stated, not hidden).
3. **Only if non-null:** paired branch continuations FORKED FROM THE SAME
   checkpoint K, 30-50 iterations each (~$5-10), compared head-to-head
   (~400 games total for 50 Elo — half the cost of each-vs-reference).
   This keeps my option-(b) cheapness while killing its flaw: the flip
   point becomes a FORK point, so both arms share full history.
4. **Pre-committed floor:** if stage-2's CI covers zero and branches are
   within noise, record "below detection floor; confound accepted; advice
   stays ON (costs ~nothing, out_norm shows it learning)" and STOP. A
   legitimate terminal answer, agreed in advance so it cannot be
   rationalised away later.

### Cycle 12 — 2026-07-29 — the baseline lands and KILLS T1-F; my cycle-10 inference was wrong

First live reads of both channels, and they overturn two of my own claims.

| metric | iter 0 | iter 1 | I predicted |
|---|---|---|---|
| `boundary_sum` (n=16) | **-0.143** | **-0.015** | +0.4..+0.6 |
| `advice_fire` | 0.053 | 0.052 | ~0.13 |
| `advice_grad_share` | 0.0129 | 0.0150 | — |
| `advice_out_norm` | 0.195 | **0.214** | ~0 if inert |
| `fresh_value_ce` | 1.0182 | **0.5295** | — |
| loss | 2.8640 | 2.6883 | — |

**1. The boundary bias is NOT present in the live campaign — T1-F should
die.** Predicted +0.4..+0.6 from the fogged-lineage probes; measured
-0.143 then -0.015, i.e. essentially zero and if anything the wrong sign.
The consistency penalty was designed to remove a systematic offset that
this campaign does not exhibit. **This is exactly what landing telemetry
BEFORE the penalty was for** — the third proposal killed by its own gate
(after T1-D and the draw-framing). Candidate reasons the probes and the
campaign disagree: the probes measured the RAW 0722 policy under the OLD
q-transform, while this is the training net seeded from 2.30M under the
FIXED transform, on a mix that is 20% fogless. Not resolved, and it does
not need to be to decline building the penalty.

**2. My cycle-10 inference "the advice signal is likely INERT" was WRONG.**
`advice_out_norm` is 0.195 -> 0.214 and CLIMBING, with a 1.3-1.5% gradient
share. The advice reforward had been working the whole time; only the
REPORTING died in `_combine_stats`. I inferred a broken mechanism from a
broken instrument — the exact error the telemetry existed to prevent.
Retracted.

**3. Consequence for the cycle-6 disagreement, on my own stated terms.** I
kept `MCTS_ADVICE=1` arguing the confound was measurable via
`advice_out_norm`, and committed to naming the split point if it climbed.
It has climbed from the first iteration, so: **attribution for this leg is
ambiguous between the q-transform fix and the advice signal from iter 0
onward.** Fable's methodological objection was right; my mitigation
correctly detected it rather than assuming it away, but the mitigation does
not undo the ambiguity. A clean attribution needs an advice-OFF arm.

**4. `fresh_value_ce` 1.0182 -> 0.5295** on the project's default success
metric, with loss 2.864 -> 2.688. Two points is not a trend (the +-
spreads are 1.27 and 0.52), but the direction is right.

### Cycle 11 — 2026-07-29 — telemetry fixed on the spool path (+ a holdout leak caught)

Fable traced both channels exactly, and one cause is a general hazard:

1. **Boundary:** workers build their OWN `MCTSPolicy` and run
   `finalize_game` worker-side; the learner ingests
   `payload["experiences"]` straight into `policy._queue`
   (`sim_self_play.py:1292`), so the LEARNER's `finalize_game` never runs
   and its FIFO stays empty.
2. **Advice: `_combine_stats` is a default-swallowing chokepoint.** Under
   `--replay-buffer`, `train_step` aggregates through it, and it built a
   fresh `TrainStats` naming only ELEVEN fields — the four `advice_*` stats
   silently reverted to NaN defaults and the log guard then hid them. The
   legacy no-replay path returns the step's stats directly, which is
   *exactly* why every in-process validation showed advice working.

**Fix:** `harvest_boundary_pairs(exps)` reconstructs pairs at INGEST from
each experience's own `current_side` — chosen over shipping pairs because
it needs no wire-format change (mixed worker/trainer versions survive a
rolling restart) and adjacency is intact (spool payloads are atomic per
game and order-preserving). `_combine_stats` now carries the advice fields
(NaN-aware means; last-non-NaN for `advice_out_norm`, mirroring
`grad_norm`).

**Fable also caught a HOLDOUT LEAK in its own previously-landed harvest:**
pairs were collected BEFORE holdout diversion, so holdout games would have
fed the future consistency penalty. Now harvested only for games that
TRAIN.

Validated by Fable on the PRODUCTION topology (spool workers + replay
buffer + advice), not the in-process path that hid the bug. Tests pin both,
including the `_combine_stats` chokepoint so the next stat added inside
`step_mcts` cannot die silently. 582 fast + 6 slow green (incl. the
spool-workers e2e). Landed 100ea16; box rebooted onto it.

**Expected signatures on the next box iteration** (these ARE the reads that
matter): `advice_fire ~ 0.13` and `boundary_sum ~ +0.4..+0.6 / n=16`.
Anything else is new information — and that boundary number IS the T1-F
baseline we have been missing.

### Cycle 10 — 2026-07-29 — BOTH telemetry channels read ZERO on the box's actual path

Campaign healthy (post-reboot iter 0: train_step 247s, loss 2.9346,
z_comp 0.52/0.48/0.00, fresh_value_ce 1.3870). But the instrumentation I
landed is **blind on the configuration the box runs**:

```
trainer_history_local.csv, latest row:
  boundary_sum = 'nan'      boundary_pairs_n = '0'
train_step log line: no advice_fire / advice_grad_share / advice_out_norm
```

Verified on the box that the CODE is present (`advice_fire` x2 and
`boundary_sum` x4 in sim_self_play.py, `has_advice` guard in trainer.py,
"detector advice path ON" logged, 2 boundary columns in the CSV). So the
code ships and the values are simply never populated:

- `boundary_pairs_n = 0` => `finalize_game` harvesting never fires. The box
  runs **spool workers**: games are played in 100 SEPARATE PROCESSES and
  shipped to the trainer as experiences, so the trainer process's
  `MCTSPolicy.finalize_game` — where adjacency is knowable and pairs are
  harvested — does not run for those games.
- advice telemetry NaN => `n_advice_states == 0`, i.e. the advice-attach
  block in `_trainer_step_mcts` is not executing on this path either
  (`Trainer.step_mcts = _trainer_step_mcts` IS the entry point, so the
  cause is more specific than "wrong function" and needs tracing).

**Consequences, stated plainly:**
1. The advice signal is very likely INERT in this campaign. That
   incidentally VINDICATES Fable's methodological objection from cycle 6
   from the opposite direction — there is no confound because there is no
   effect — but it also means `MCTS_ADVICE=1` is currently buying nothing.
2. We have no boundary baseline, so the T1-F penalty decision is still
   unmeasured.
3. **General lesson: instrumentation validated only against the
   in-process path can be silently dead on the production path.** Both my
   advice telemetry and Fable's boundary telemetry were verified locally
   (in-process) and both read zero in the campaign. Any future telemetry
   must be validated against the SPOOL path before being trusted.

Not yet fixed — delegated with the diagnosis above. The campaign continues
meanwhile: it is producing decisive games and the q-transform fix (the
thing that actually mattered) is active in the workers' search.

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
