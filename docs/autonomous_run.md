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

1. ~~Which term in the training signal pays the model to bank gold?~~
   **ANSWERED, cycle 29: none does.** Every gold channel is inert under
   `--mcts` (`MCTSPolicy.observe` is a no-op), the value head is
   pro-recruit (+0.229), search does not starve recruits, and end_turn is
   taught DOWN. What is real is the "tried-and-cut tax" in
   `extract_gumbel_policy_target` — ≤2-visit halving losers grade below
   v_mix while ~98% untried mass shelters at v_mix, so concentrated-prior
   classes (midgame recruit) pay most. Whether that is pathology or good
   play is deferred to question #5.
2. Does the real engine leave a leader off-keep on Tombs_of_Kesorak /
   Sablestone_Delta (7.5% of leader-sides can't recruit at start), or does
   it place differently? Check `wesnoth_src/src/` before changing placement.
3. What is the actual throughput ceiling on the box, and what net size is
   trainable within it? (T2.)
4. **Why did iteration time rise (~16 -> >25 min) when acting-side advice
   went live, given the advisor measures 0.3 ms/decision?** Refuted: the
   advisor itself. **The "load doubled 130 -> 272" half is also refuted
   as evidence about us** — `/proc/loadavg` in a Vast container is
   HOST-wide (cycle 27), so it counted co-tenants. On the new box our
   processes use 91.6 of a 92.16-core quota, i.e. quota-saturated, not
   oversubscribed. Remaining live hypotheses for the *time* rise: host
   contention from co-tenants (now the leading one, and it would make
   this a non-bug), model-side cost of the advice module, or trainer
   reforward cost. Test: watch iteration time on the new box.
5. ~~Does the q-transform fix actually produce a stronger policy?~~
   **ANSWERED YES, cycle 46.** Triangulated against a fixed anchor over
   340 games in one joint fit: **new_2p52M +133 ±57** vs the 2.29M
   peak-region seed, direct arm +146 [+72,+219] at p=0.0001, and the
   transitive check (direct 146 vs chained 134) rules out
   style-exploitation. Cycle 30's "+35 ±68, undetermined" was not wrong,
   just **too early**: undetectable at 113k steps, clearly detectable at
   225k. The 2026-07-28 regression is recovered and surpassed. Residual:
   the co-peak candidate 2,747,117 was not directly tested, and vs-RCA
   remains unmeasured.

---

## Cycle log

Newest first. Each entry: what was attempted, what was MEASURED, what was
decided, what is next. Keep entries short and factual.

### Cycle 47 — 2026-07-30 — campaign accruing on fixed code; credit now the binding constraint

**Box.** 77 workers, learner alive, **zero aborts**, HEAD `42fa1ab`.
Decision step **2,546,945** — +20,150 since the cycle-45 resume, i.e. the
campaign is accruing normally on fully-fixed code (all four sim/encoder
fixes plus the dual-import normalization).

```
iter 2  CE 0.7985  floor 0.9134  -> -0.115   boundary -0.254/n=64  draws 0.08
iter 3  CE 0.6252  floor 0.8005  -> -0.175   boundary -0.116/n=64  draws 0.04
iter 4  CE 1.0076  floor 1.0592  -> -0.052   boundary -0.088/n=64  draws 0.17
```

CE-minus-floor over the leg: −0.157, −0.145, −0.091, −0.175, −0.052.
Two crossings of the −0.10 bar, **never 3 consecutive → tripwire NOT
tripped.** Boundary rolling |mean| 0.153, under 0.25 (one individual
reading, −0.254, marginally exceeds it; on the now-trustworthy k=64
estimator that is worth noting but not acting on at n=1).

**Credit is the binding constraint again: $5.60 (~17 h).** The 72h mandate
has ~27 h left, so **the box will expire before the run does.** No action
needed — the campaign simply runs until credit exhausts, which is the
correct use given cycle 46 measured this window as genuinely productive.
Deliberately NOT topping up or re-provisioning: the user authorized
spending the existing credit, not acquiring more.

**Dispatched: the EXTERNAL anchor.** Every strength number this run has
produced is **within-lineage** — Elo against our own ancestors. The
founding thesis recorded "loses to the built-in RCA AI, ~9 decisive games,
0 wins", and that reading is now stale in BOTH directions: taken at
pre-fix 3.74M weights, before the q-transform fix, the four sim fixes, and
the measured +133. It is the only non-self-referential number available and
maps most directly onto the user's actual goal.

Package framed feasibility-first, because the live-Wesnoth path has been
eval-only since 2026-05-11 and the package reorg has already broken one
entry point (`main.py --check-setup`, fixed earlier this run) — so "here
is precisely what bit-rotted, and the minimal fix" is an acceptable
deliverable rather than sinking the package into a blind repair. Fallback
if it is not viable at acceptable cost: the direct **new_2p52M vs
2,747,117** edge, which would close cycle 46's one stated gap (the
untested co-peak candidate) at ~2 h.

Constraints restated in the dispatch: headless/minimized only (no focus
stealing, per standing user veto), pure-outcome eval contract verified
rather than assumed, and a game budget fixed up front since real Wesnoth
subprocesses run 7-15 min/game.

### Cycle 46 — 2026-07-30 — ABSOLUTE PROGRESS: the regression is recovered and surpassed

**The run's best result, and it answers open question #5.** Triangulation
against a FIXED external anchor, 340 games in one joint Bradley-Terry fit,
seed anchored at 0, predictions and interpretation rules registered at
07:28:39Z before any game:

```
player                      joint Elo (ref=0)   CI95
ref_2p29M (seed, peak-region)      0             --
old_2p40M (2,403,615)            +30            50
new_2p52M (2,515,896)           +133            57
random_init                     -300           232

arm A direct: new +146 [+72,+219], 70-30-0, sweeps 23-3, p=0.0001
```

**The check that matters — no style-exploitation signature.** Direct
+146 vs chained (+35)+(+99) = +134; difference **12**, nowhere near the
pre-registered ±145 inconsistency band. So this is **not** a lineage
beating its own parent while going nowhere — the exact failure mode I
asked Fable to look for, and the one this lineage has historically shown.
**Verdict: ABSOLUTE_PROGRESS**, the registered cell.

**Open question #5 — ANSWERED.** The q-transform fix (cycle 2, the run's
central claim) produces a measurably stronger policy. Cycle 30's
pessimism is revised rather than contradicted: **undetectable at 113k
steps (+35 ±68), clearly detectable at 225k (+133 ±57).** The fixed
signal buys real strength; the earlier read was simply taken too early.

**The 2026-07-28 regression is recovered and surpassed** relative to the
2.30M peak-region anchor — with the honest limit Fable stated: the other
tied peak candidate (2,747,117, locally available) was not directly
tested, and the tie (3-1-4) is what licenses the word "region". A direct
edge costs ~2h if belt-and-braces is wanted.

**Method notes worth keeping.** Arm B (seed-vs-old, n=40 on shared
setups) replicated cycle 30 within noise (20-0-20, z=−0.54), which
independently confirms that `fa95da5`'s encoding delta does not move this
measurement — the reused edge was safe. A sensitivity fit dropping the
pre-fix edge entirely gives new +130 ±63: conclusion unchanged. The
random-init gauge came in at −300 and, notably, **random took one game
off the new checkpoint (5-1)**, missing the registered ≥90% bar — but at
n=6 that has P≈0.47 even at a true 90% win rate, so it is recorded and
NOT interpreted. That restraint is right.

**This vindicates the cycle-45 reversal**, and by extension makes the
cycle-44 stop the clear error it was recorded as: the campaign was
gaining absolute strength throughout the window I halted it in.

**Standing caveats, unchanged:** the anchor is within-lineage (vs-RCA
remains unmeasured, and the last such reading is stale in both directions
— taken at pre-fix 3.74M); scope is the ladder pool; and **the mini
passivity drift remains real and untouched** — correctly scoped now as a
mini-pool behavioural problem, not a strength regression.

**Landed `tools/mini_anatomy.py`** — the early-warning instrument for that
drift. It plays mini-pool games under the box's exact production search
config and records per-side per-turn trajectories (units, HP, gold,
villages, distance to enemy leader, attacks, recruits, unused-MP, idle
fraction, ended_by, the jittered cap). Its value is that the GRADED
precursors move well before the first draw appears — non-end_turn actions
per side-turn 2.55 -> 1.49 and median end turn 10 -> 30 while games are
still decisive — so it detects the pathology earlier than any draw-rate
column can. Chunk-friendly (`--append`/`--seed`) for the 9-min guardrail;
no network, no box access. 633 fast tests pass.

**Box:** 77 workers, zero aborts, iterating (iter 4 at 11:12Z).
CE-minus-floor across the leg: −0.157, −0.145, −0.091, −0.175, −0.052 —
two crossings of the −0.10 bar but never 3 consecutive, so the amended
tripwire is NOT tripped.

### Cycle 45 — 2026-07-30 — I WAS WRONG: the primary metric reverses cycle 44; campaign RESUMED

**Cycle 44's stop was not supported. I reversed it.** The campaign is
running again on HEAD `42fa1ab` (all fixes plus the dual-import
normalization), 77 workers, no ABORTED marker, resumed from **2,526,795**
— nothing lost by the stop/start.

**The measurement, with my prediction registered before it.** I asked for
a pre-registered prediction precisely so this could not be rationalized
afterwards. Mine: *old ahead, new at −45, P(new significantly ahead) =
0.05.* The result landed in that 5% cell.

```
LADDER, n=100, 4e445c2 protocol (mirrored pairs, max_turns 100, one BT fit)
  pooled W-D-L (old's view):   36-0-64
  new_2p52M:  Elo +99   CI95 [+28, +169]   <- excludes zero, favouring NEW
  mirror: 5 swept old / 26 split / 19 swept new
  sign test p=0.0066 ; binomial on decisives p=0.0066  (three instruments agree)
  0 draws in 100; new wins FASTER (median turn 14 vs 19)

MINI, n=100 (30 registered + 70 post-hoc, disclosed as such)
  old 58-1-41 ; new -60 +-69, CI [-129, +9]  -> old-lean, NOT significant
  only 1 non-decisive game in 100
```

**What was actually true.** The drift is real — the matched-seed
p=0.0008 stands — but it is **map-scoped and behavioural, not a general
strength regression**. The window traded a modest, mostly
self-play-visible mini passivity (15% of the mix) for a **significant
ladder improvement** (45% of the mix), and produced the first significant
window-on-window Elo gap this lineage has managed (the prior window was
+35 ±68, undetermined). Head-to-head, an aggressive opponent punishes the
stall, so the self-play draw pathology **does not convert into
head-to-head draws**.

**BEST-KNOWN CHECKPOINT FLIPS.** By the primary metric it is
**`campaign_live_20260730.pt` (2,515,896)**, not `..._20260729.pt`
(2,403,615) as cycle 44 claimed. The box is now 11k steps beyond that, at
2,526,795 (unmeasured).

**My error, stated precisely.** The controlled experiment was sound; the
INFERENCE I drew from it was not. I generalized a behavioural finding
measured on 15% of the curriculum to the whole policy, and then took an
irreversible-feeling action on it. **The sequencing was the mistake**: I
had a free, local, already-built Elo harness and a standing lesson that
this project's plausible-reasoning levers keep measuring wrong
(`gumbel_m` 16->8). The correct order was measure-then-decide, and I
decided first. "Training value ~0, now negative" is refuted for this
window — it was the most productive 112k steps recently measured.

Cycle 44's own hedge ("a flat Elo would be consistent with a mini-only
pathology") turned out to be the wrong worry: the metric was not blind,
it saw an *improvement*, in the opposite direction from my concern.

**Limitation carried forward, flagged by Fable and not used as an
excuse:** adjacent-checkpoint head-to-head measures relative exploitation
WITHIN a lineage, not absolute strength — a newer checkpoint can beat its
predecessor while both drift somewhere unhelpful. Triangulating both
against a common reference (`seed_20260718.pt`) would settle it, ~2h per
pairing.

**What stands from cycle 44:** the mini passivity is real, accelerating,
mutual (the materially-ahead side declines free leader-kills), and
mechanistically the founding thesis — banks gold, declines to commit. It
is now correctly scoped as a *mini-pool behavioural* problem worth an
early-warning instrument (`tools/mini_anatomy.py`), not a reason to halt
training.

### Cycle 44 — 2026-07-30 — CAMPAIGN STOPPED: the policy was measurably degrading

**The most consequential decision of the run. Box training killed, instance
STOPPED, credit preserved at ~$7.27.**

**The evidence that forced it** — Fable's controlled matched-seed
experiment, same code (post-fix local HEAD), same 24 seeds (same
scenarios, factions, caps), two checkpoints from this lineage:

```
                       decision_step  mini draws  endings          median end turn
campaign_live_20260729   2,403,615     0/24       24x leader_killed      10
campaign_live_20260730   2,515,896     9/24       9x max_turns, 15 kill  64
Fisher one-sided p = 0.0008
```

~112k decisions turned a policy that kills in **10** median turns into one
that draws 3/8 of mini games. Because the 0/24 arm ran OLD weights on
POST-fix code, this is also an independent, code-held-constant
confirmation of cycle 42: **the fixes don't cause draws; the weights do.**

**Mechanism — "mutual armistice at arm's length":** non-end_turn actions
per side-turn **0.20** in draws vs 2.55 pre-drift; unused-MP fraction
**0.95** vs 0.33; idle-unit fraction ~0.94. The armies are **ADJACENT**
(median min-separation 2-3 hexes, `dist_to_enemy_leader` min 1-3), so this
is no-commit, not never-meet. Combat ceases outright — 6/9 draws have ≤1
turn-with-HP-loss in the back half, back-half HP flat (85→85, 92→92,
122→122). Gold banks to a mean max of **303**/side (peaks 428-486) vs 86,
with 0-3 recruits, while the pre-drift arm recruits 3.6/side on the SAME
maps. **This is the founding thesis verbatim: banks gold, declines to
commit.**

**It is graded, not draw-only:** even still-DECISIVE games at current
weights show median 30 turns (vs 10), unused-MP 0.68 (vs 0.33), gold-max
138 (vs 86). Draws are the thresholded tip of a continuous slowdown.

**Symmetry matters for any future fix:** 7/9 draws are *mutually* quiet in
the back half — the materially-ahead side, standing adjacent to a near-dead
opponent (seed 100: 81 HP vs 23), **also stops attacking**, forfeiting a
free +1 leader-kill. So "help the stuck side close" would miss the point.

**Structural explanation DEAD:** minis are not inherently drawish (0/24
pre-drift, median kill turn 10, caps 60-100 give 6-10x headroom); every
draw ends by `max_turns`, `max_actions` never fires; draws concentrate on
the *plain smallest* maps while the two largest minis are 0/6. Also
fact-checked: `DrawTiebreakConfig.weight_gold = 0` since 2026-07-20, so
"hoarding is paid in-tree" is NOT the mechanism.

**Trend is ACCELERATING, not saturating:** 0/55 draws over 65k decisions
-> ~10% over the next 30k -> 26-37% in the last ~15k.

**Why I stopped rather than watched.** Training value was already ~0
(cycles 28-30: ~180k affordable steps against a 450k-1M detectable gap).
This makes it **negative** — continuing would spend the user's credit
making the checkpoint worse. Stopping preserves both the credit and the
better checkpoint. Restarting from 2,403,615 was considered and rejected:
without a fix the same signal reproduces the same drift.

Box stopped cleanly: no ABORTED marker, final step 2,526,795. Best-known
checkpoint is **`campaign_live_20260729.pt` (2,403,615)**, secured locally
and on HF.

**Box-ops trap, recorded:** `pkill -f sim_self_play.py` also matches the
SSH session's OWN bash (its command line contains that string), killing
the session mid-command — my first attempt truncated its output and left
the learner alive with its workers dead. `pgrep -fc` likewise counts the
supervisor `bash -c` wrapper. **Kill by PID** (supervisor first so it
cannot relaunch, then the learner).

**Honest correction to my own bar.** The overall draw mass FELL across the
post-restart iterations (0.25 -> 0.10 -> 0.05) as composition normalized,
so the box telemetry alone would **not** have triggered my ≥20% condition
— only the mini-specific column I happened to nominate (4/6) did, on n=6.
**The controlled local experiment is what carried this finding**, not the
telemetry. CE-minus-floor read -0.157, -0.145, -0.091, so iter 2 crossed
the amended -0.10 bar (1 of 3).

**Dispatched:** confirm the regression on the PRIMARY metric — head-to-head
Elo, 2,403,615 vs 2,515,896, n≈100 under the `4e445c2` protocol, with the
prediction registered BEFORE measuring. Explicitly asked whether stopping
was correct, and flagged the confound that matters: the ladder eval runs
ladder maps, which are 100% decisive in every iteration, so **a flat Elo
would be consistent with a mini-only pathology rather than a refutation
of it** — and if the honest answer is "our primary metric is blind to this
pathology", that is itself an important limitation to record.

### Cycle 43 — 2026-07-30 — the mandated reading is STILL PENDING (and why that is expected)

**Box.** 03:20Z, 77 workers, zero aborts, HEAD `8780c0c`. Credit ~$8.

**The cycle-42 test could not be run: iteration 1 has not landed.** 50 min
after iter 0 (02:30:13Z), still only one post-restart iteration exists.
Recorded rather than glossed, because cycle 42 named iters 1-3 as this
cycle's mandated reading.

**Why this is expected, labelled INFERRED:** it follows from Fable's own
composition finding. A cold-start batch is mini-heavy because tiny maps
finish first and the batch takes the first 24 completions — so iter 0
harvested the FAST games, and iter 1 must wait on the slower ladder games
only now completing. Pre-restart steady-state was 17-43 min/iteration, but
that was a steady mix; the first post-cold-start balanced batch should be
slower than either. Falsifiable: if iter 1 still has not landed by ~90 min
after iter 0, this explanation is wrong and something is wrong with
generation instead.

**Nothing was read into the pending gap.** No tripwire evaluation, no
draw-rate claim, no CE trend — one artifact iteration (excluded by the
cycle-42 amendment) is all the post-restart data that exists. The reading
carries to the next cycle unchanged:

- read **CE-minus-floor from iter 1 onward** (iter-0-after-restart
  excluded);
- Fable's prediction to test: iters 1-3 resemble the 01:23 row — CE
  ~0.4-0.7, floor ~0.68, draws ~0-10%;
- **if draws hold ≥20% at steady composition**, the mini-drift graduates
  from watch-item to real campaign-health problem.

**Dispatched:** the mini-map draw drift — is it the founding
passivity/hoarding thesis re-emerging where it is most visible (fewest
hexes, fewest options, stalling cheapest), a structural property of tiny
maps, or noise on small n? Package asks for the drawn games characterised
BEHAVIOURALLY (recruits/turn, gold trajectory, attacks, idle units,
distance to enemy leader), symmetric-vs-asymmetric stalling, which
termination actually fires (`max_turns` vs `max_actions_per_side`), and the
trend across the whole leg rather than two endpoints. Explicitly barred
from proposing a training-config change on this evidence alone — the run's
standing lesson is that levers chosen from plausible reasoning measure
wrong (`gumbel_m` 16->8).

### Cycle 42 — 2026-07-30 — draw jump is BENIGN: the box ran the control itself; my hypothesis dead

**My cycle-41 hypothesis is REFUTED, exactly by the falsifier I set.**
WML census over all pools: **only Aethermaw** carries latch-suppressible
events (its 8 morphs). Every other ladder/mini/drill scenario's turn
events are `first_time_only=no`, which `fire_event` never latch-suppresses
(`scenario_events.py:1503`). 1/21 maps cannot produce 25% draws.
Doubly dead: the draws concentrate in the MINI pool, which has no
suppressible events, and the spike batches contained 0-1 ladder games —
Aethermaw was not even present.

**The box ran the controlled experiment itself and nobody noticed.** The
trainer restarted **twice** that night. Verified independently on the box:
`training exited rc=1 at 2026-07-29T23:38:15Z` + `relaunch 1/20 in 60s` —
an ordinary crash down the supervisor's rc-1 path (no marker), on the
pre-fix launch clone. So there is a **pre-fix cold-start arm** for exactly
this question:

```
                  code      weights   draws  zD_w   CE     floor   bnd_n
00:19:38 iter 0   PRE-fix   2.51M     5/24   0.213  0.943  1.073   16
02:30:16 iter 0   POST-fix  2.52M     6/24   0.250  0.925  1.082   64
Fisher on mini draws 5/19 vs 6/18:  p = 0.46
floor-relative: post -0.157  vs  pre -0.130   (post mildly BETTER)
```

Fable's proof the 00:19 row is pre-fix is clean and I verified the logic:
`boundary_pairs_n = 16` (the k=64 estimator shipped only with the reboot
and flips to 64 exactly at the 02:30 row), and decision_step 2,511,831 <
2,515,896 — the step of the checkpoint I secured before rebooting.

**So: none of the four fixes caused the draw jump, and `fa95da5` is NOT
GUILTY.** My restart decision is not implicated. Ladder + fogless +
midgame decisiveness is **100% in every row of the leg**, including all
post-restart rows — there is no global-passivity signal.

**Two stacked effects explain the anomaly.**
1. **Iteration-0-after-any-restart is a COMPOSITION artifact.** Every
   cold start is mini-heavy — "other" (= mini pool) is 17, 19, 18 of 24
   games at the three restart iter-0s, with 0-1 ladder games — because
   tiny maps finish first and the batch takes the first 24 completions.
   CE, floor and z-mix at a restart iter-0 are **not comparable to steady
   state, full stop.** My cycle-41 comparison was invalid on both sides.
2. **A real, slow, PRE-EXISTING drift toward mini-map turn-cap draws.**
   Within pre-fix code at matched cold-start composition: 0/17 mini draws
   at 2.40M vs 5/19 at 2.51M (**p = 0.031**); corroborated mid-leg on old
   code, 0/55 minis at iters 0-8 vs 2/20 at iters 9-13 (p = 0.068). It
   began around iter 9, **before any fix existed.** All 11 post-restart
   draws are mini games (`draws` == `other_games − other_decisive`
   exactly, every row).

**TRIPWIRE PROTOCOL AMENDED (adopted):** exclude **iter-0-after-any-restart
entirely**, and read CE-minus-floor from iter 1 onward. The cycle-41 bar
was doubly wrong — raw instead of floor-relative, AND applied to a
composition artifact.

**Falsifiable prediction now on the record** (Fable's, adopted as the
next cycle's test): post-reboot iters 1-3 should resemble the 01:23 row —
CE ~0.4-0.7, floor ~0.68, draws ~0-10%. **If draws instead hold ≥20% at
steady composition, the mini-drift graduates to a real campaign-health
problem** — and that, not the fixes, is the thing to watch (column:
`other_decisive/other_games`; the abort tripwire at 0.05 decisive is
nowhere near firing). Only iter 0 exists as of 02:58Z, so this is next
cycle's reading.

**Corrections to the record.** The 0/116 cap-endings measurement is
**cycle 30**, not cycle 38 as cycle 41 said. The draw label is honest
(`winner==0` -> z=0 exactly; `--train-draw-tiebreak` not passed;
`--draw-value-weight 0.25` is a LOSS weight, not a game weight — zD_w
6/24 = 0.250 exactly confirms game-weight 1). Draws are turn-cap /
`max_actions` timeouts, consistent with mean_turns 20 -> 40+ under caps
jittered 60-100. Also: `campaign_live_20260730.pt` contains two extra
old-code iterations (~8k decisions) nobody knew about — harmless, but the
record should say so.

**Reconciliation with cycle 30's 0/116:** no tension once split by map
class. That eval was raw-policy on **ladder** maps, and ladder games are
100% decisive in training too. **The draws are mini-only.**

### Cycle 41 — 2026-07-30 — the tripwire READ: raw bar exceeded, but it is CONFOUNDED; draws 0% -> 25%

**The reading I committed to taking.** First post-restart iteration landed
02:30:13Z (55 min, as predicted). Against the original cold launch on
pre-fix code, **same config, same curriculum mix**:

```
post-restart iter 0: CE 0.9250 +-1.3421  floor 1.0817  z_comp_w 0.42/0.33/0.25
original     iter 0: CE 0.4654 +-0.6201  floor 0.6823  z_comp_w 0.56/0.44/0.00
                                                        draws 0% -> 25%
```

**The raw tripwire bar (>~0.749) IS exceeded at 0.9250.** Stated first
and without softening, because it is the unfavourable reading for my
cycle-39 restart decision.

**But the bar is confounded, and the confound is identifiable
independently of the outcome.** `fresh_ce_floor` — the marginal-label CE
baseline — jumped **0.6823 -> 1.0817**, because the label distribution
gained a 25% draw class and a three-way distribution has higher marginal
entropy. CE measured against a 1.08 floor is not the same quantity as CE
against 0.68. Floor-relative:

```
post-restart:  0.157 BELOW floor      original:  0.217 below floor
```

So the model still beats its marginal baseline, by somewhat less. **I am
not using this to rescue the result** — the corrected metric is also
mildly worse, and I am recording that. The tripwire should have been
specified floor-relative from the start; that is a flaw in how I wrote
it, not a licence to move it now. **Amendment, stated before further
data: read the tripwire as CE-minus-floor, bar = 3 sustained iterations
at worse than −0.10** (vs −0.217 pre-restart, −0.157 now). Held to 1/3.

**The real signal is the draw rate: 0% -> 25% under identical config.**
Not explainable by the curriculum mix, which is unchanged (mini 0.15,
ladder 0.45, sims 32, max-turns 100 — verified bit-for-bit after the
reboot). It also sits in direct tension with Fable's cycle-38 measurement
of **ZERO cap-endings in 116 games**, which must be reconciled.

**Hypothesis, labelled INFERRED and dispatched for refutation:** the
event-latch fix (`933888d`) means `first_time_only` turn events now fire
in LIVE games across **all** scenarios, not only Aethermaw — previously
ANY search fork crossing a turn boundary latched them and suppressed them
live, everywhere. If ladder scenarios broadly carry turn-triggered
events, games are now materially different and **more draws could be the
CORRECT behaviour we were previously missing** — a fidelity improvement
surfacing as a metric regression. The falsifier is sharp: if only
Aethermaw qualifies (~2% of games at ladder-ratio 0.45), 1/21 maps cannot
produce a 25% draw rate and the hypothesis is dead.

The alternative that would mean **my restart was wrong** is `fa95da5`'s
encoder change pushing play toward passivity globally. Fable was asked to
give that a fair test and to say so plainly if it holds — reverting is
preferable to defending the decision.

Note `8b68a25` (dual-import) is NOT on the box: pushed after the restart,
box remains `8780c0c`.

**Vindication of `d03d50c`:** `boundary_sum=-0.323/n=64/**pool=1721**`.
The pool is 1721 pairs, so sampling 16 of them was indeed the noise
source, and k=64 was justified. This also means the reading is
individually interpretable for the first time — and |−0.323| is outside
the 0.25 band. Second watch item, now on a trustworthy estimator.

### Cycle 40 — 2026-07-30 — dual-import audit: NINE modules, and one bug that can't exist single-flavour

**Box post-restart: healthy, tripwire not yet readable.** HEAD
`8780c0c`, resumed at 2,515,896, 77 workers, **zero errors, zero ABORTED
markers**, and our processes at **9181% of the 9216% CPU quota (99.6%)**.
But **0 iterations 55 min after restart** vs 37 min on the original cold
launch. Diagnosed, not assumed: generation is at full quota and games run
50-80 min each (cycle 28), and post-restart the spool starts cold AND the
holdout must refill — so 60-90 min to the first iteration is within
expectation. **Not a regression from the four fixes.** The `fresh_value_ce`
tripwire (>~0.749 sustained 3 iters vs a ~0.649 pre-restart level) is
therefore still UNREAD; next cycle must check it.

**Measurement trap found, worth remembering:** the worker-heartbeat
aggregate is **NON-MONOTONIC across a restart**. Each worker writes
`stats/w{id}.json` with its own counters, so new worker processes
overwrite old files with fresh (lower) counts — I watched games go
229 -> 227 and decisions 75,094 -> 74,247 over three minutes. Cross-restart
heartbeat deltas are meaningless; use summed process CPU against the
cgroup quota instead. (Cycle 28 already found this file is a heartbeat,
not a per-game record; this is the second trap in the same file.)

**The dual-import audit (landed `8b68a25`) was far bigger than either of
us framed it.** Not one module — **NINE**, and `wesnoth_sim` was the
widest with 20 bare importers, with `mcts.py` the bare-side hub for three
more. My cycle-39 note saying "`wesnoth_sim.py` imports bare" was true
but understated.

**Four pairs co-reside in every production worker process, and a FIFTH
was live in the learner:** spool payloads pickle `GameOutcome` under
`tools.sim_self_play` while the learner runs that same file as
`__main__`, so the first unpickle **re-executed the entire 3,700-line
module** as a second in-process copy.

**One REAL reachable bug, and note the shape:** `scenario_events.py`
checked `composite in _UNIT_DB` against the PREFIXED copy while games are
constructed through the BARE copy. In any process without the encoder
warm (probes, reconstruction tools) the prefixed DB is still the pre-load
`{}`, so variation-carrying spawns silently degrade to base type. **This
bug cannot exist single-flavour** — it is a pure dual-import artifact.
Production escaped it only because the encoder touches the prefixed copy
at decision 1: correct by luck, not by construction.

Everything else classified with evidence rather than waved through: the
unit/movetype/race DBs and `wesnoth_sim`'s cost caches are
deterministic-identical (HARMLESS, though the box wastes a second ~400KB
JSON parse per worker × 77); `VALIDATION_EXPORTER` is coherent only
because `selfplay_worker` installs it on the matching flavour by hand;
`openers._REGISTRY` would fail LOUDLY. Four of the nine have no
module-level mutable state at all. No `isinstance`/`except`/enum
comparison crosses flavours anywhere, and **no test monkeypatches a dual
module**, so nothing was silently weakened by this class.

**Guard:** `tests/test_no_dual_imports.py` pairs a static AST lint (which
catches function-level imports a runtime check cannot see) with a runtime
`sys.modules` detector that permits same-object aliases so the `__main__`
pin stays legal. Both verified RED against planted violations. Residual
risk recorded: bare imports still RESOLVE since `tools/` stays on
`sys.path` — only the lint forbids them; stripping `sys.path` was
considered and rejected as a much riskier diff.

633 fast + 11 slow green, the slow tier covering spool-workers e2e,
concurrent train-step and export validation — exactly the seams touched.
Needs no fresh-CE gate: no encoder inputs, no sim mechanics.

### Cycle 39 — 2026-07-30 — the gate ran, and the box RESTARTED onto all four fixes

**The gate result, reported as registered.** Thresholds were written into
the probe header BEFORE any measurement: SAFE if dCE ≤ +0.03 AND MAE ≤
0.02; NOT-SAFE if dCE ≥ +0.10 OR MAE ≥ 0.05.

```
raw arm (primary, 20 games / 4,457 states, 20/20 decisive):
    dCE = +0.0037  CI95[-0.0014,+0.0089]    MAE = 0.0493   -> UNDETERMINED
mcts arm (validation, 5 games / 796 states, no ladder):
    dCE = -0.0126  CI95[-0.0876,+0.0441]    MAE = 0.0666   -> nominal NOT-SAFE
decomposition (mean per state):
    no-flip  n=3425   |dv| 0.0000   dCE +0.0000  (exact zero: recon validated)
    A-only   n= 232   |dv| 0.0245   dCE -0.0198  (NEW IMPROVES)
```

**A third sub-case neither of us anticipated, and it dominates 15:1.**
Under pre-fix code, stored states' hexes ALIAS the live map and the
trainer re-encodes at train time — so **a state at turn 3 showed
ownership flags for every village captured by turn 30.** The OLD encoding
leaked the FUTURE into training inputs: 5,865 state-hex flips across
1,311/4,457 states, vs 385 states for pre-owned villages (A) and **zero**
for search-imagined captures (B1, in 796 MCTS states). Also corrected:
pre-owned villages are **17/21 ladder scenarios**, not "scenarios like
Arcanclave" — the fix's commit message undersold it.

**DECISION: restarted the box, overriding the UNDETERMINED verdict.**
Recorded plainly because overriding one's own pre-registered gate needs
justification, not just a preference:
1. The CE clause — the project's default success metric — passes with ~8×
   margin (+0.0037 vs a +0.03 bar), ~1/60th of one organic iteration step.
2. The MAE gray-zone sits almost entirely on **B2, the future-leak whose
   removal IS the fix**. The head's predictions MUST move where a leaky
   input feature ceased to exist; that is the fix working, not damage.
3. On the only sub-case persisting at ACT time (A), CE **improves**.
4. The alternative was never "keep a good leg": the running leg was
   **actively training on future-leaked inputs**. That reframes the
   choice from risk-vs-safety to which-defect-do-I-accept.
5. The leg's training value is already ~0 (cycles 28-30), so the downside
   lands on something established as unmeasurable while the
   fidelity/export upside is real.
6. T2-C's catastrophe mode (weights load, function does not carry) is
   excluded by measurement, not assumed.

**Restart mechanics, learned by reading before acting.** Two traps:
- `rc >= 128` (signal kill) makes the supervisor **stand down** without
  an `ABORTED` marker (`vast_onstart.sh`), so a `pkill` does NOT
  auto-relaunch — an operator must start the next run. Only rc 3-9 write
  the marker.
- **The baked env vars are NOT visible to an SSH session** (all empty).
  Re-running `onstart.sh` over SSH would have silently substituted
  DEFAULT ratios and changed the campaign config. That is why the reboot
  path was chosen: the container restarts with its baked env.

Checkpoint secured first: pulled locally as
`training/checkpoints/campaign_live_20260730.pt`, decision_step
**2,515,896**, flags verified.

**Verified after reboot:** HEAD **`8780c0c`** (all four fixes live), 77
workers, learner alive, **zero ABORTED markers**, resumed from exactly
2,515,896, and the baked config preserved bit-for-bit (sims 32, mini
0.15, ladder 0.45, max-turns 100, advice ON, draw-weight 0.25, workers
76). The leg now also gets `d03d50c`'s k=64 boundary estimator, so
`boundary_sum` readings become individually interpretable.

**TRIPWIRE ARMED** (operationalizing the pre-registered NOT-SAFE bar):
pre-restart `fresh_value_ce` rolling level was ~**0.649** (iters 9-12:
0.793, 0.614, 0.540, 0.647). If post-restart CE sits **> ~0.749
sustained ≥3 iterations**, treat as gate failure and halt. Expected: no
visible step outside the organic 0.54-0.79 band.

**Carried forward — production-relevant, from Fable:** `tools.replay_dataset`
and bare `replay_dataset` are **TWO distinct module objects for the same
file** (`wesnoth_sim.py` and `supervised_train.py` import bare, nearly
everything else imports `tools.`-prefixed). Any module-level mutable
state in that file is DUPLICATED across the two. It already cost a probe
a silent zero-capture run. Same hazard likely applies to other
dual-imported `tools/` modules. Deserves an audit.

### Cycle 38 — 2026-07-30 — both audit items closed; a FOURTH bug, engine-verified

**Box (22:20Z).** Iteration 13, 77 workers, zero aborts. Credit **$9.64**
(~29 h). `boundary_sum` −0.073, −0.119 — rolling |mean| back down; the
cycle-37 −0.317 reads as k=16 noise, as flagged.

**Item 1 — `combat_outcomes.py` DP: CLOSED-clean, with a legible
negative.** Line audit found no parent write (`_strike_dp` pure over
locals; `build_attack_context`/`_to_combat_unit` build fresh snapshots;
advancement runs on an isolated carrier with `_rebuild_unit` copies).
Empirical: `SIM_FORK_GUARD=1` **silent over ~1,025 searches**, 14 games,
8 maps, exercising **1,725 DP enumerations (100% exact, zero fallbacks),
2,628 advancement enumerations, 254 live + 1,750 fork attacks, 677
advancement applications, 253 turn-boundary event fires**. That is what
makes the negative worth something. Bonus: in a cut=2 Aethermaw playout
**all 8 live morphs fired on schedule (walls opened turns 4-6)** — first
end-to-end production confirmation that `933888d` works with search live.

Guard gap recorded: `deep_state_fingerprint` covers `u.defenses` but NOT
the `_defense_table` stash dict, so a future in-place mutation there
would be guard-invisible. No current writer mutates it in place.

**Item 2 — Aethermaw export OOS: CLOSED-found-a-bug, a DIFFERENT bug
than hypothesized.** Landed `a21030c`.

The hypothesized channel is **refuted**: engine-side morphs only ever
make terrain MORE permissive, so recorded actions stay legal. The real
bug: `_terrain_action` stored the overlay-**STRIPPED** base code into
`_terrain_codes` (`'Chw^Xo' -> 'Chw'`). Movement/defense resolvers walk
the alias graph from that code and an overlay can DOMINATE it — `^Xo` is
the Impassable Overlay, `mvt_alias=Xt`
(`wesnoth_src/data/core/terrain.cfg:1746-1748`, verified independently).
So Aethermaw's whirlpool walls — impassable at ALL times — were priced as
walkable water-castles and self-play walked onto them.

The old code was inconsistent **on its own terms**: `parse_terrain_codes`
documents that it returns the "full WML terrain code INCLUDING overlay"
because the overlay is what resolves defense keys. The stale comment
claiming the base is what the defense table keys by was simply wrong.

```
census of 49 Aethermaw exports in the HF pull:
  ladder/fogless : 0/32 touch ANY morph hex despite all reaching turn>=4
                   (the signature of latch-bugged morphless play)
  midgame        : 776 engine-LEGAL touches, PLUS 9 engine-ILLEGAL moves
                   across 6/17 files -- every violation onto (22,19) or
                   (28,22), the never-passable walls
engine-verified 4/4: the 2 violating exports OOS with exactly "found
  corrupt movement in replay"; a 236-touch midgame and an 83-turn
  morphless ladder export played CLEAN (1313/1313, 1129/1129)
```

Also corrects the record: the morphs cover **22 hexes across 8 events**,
not 13.

**Interaction worth remembering: bug 3 was MASKING bug 4.** The event
latch kept ladder games morphless, so they never reached the walls. Now
that `933888d` makes fresh games fire morphs, from-scratch exports would
have started hitting the overlay bug. Fixing one bug armed the next —
which is an argument for auditing a class exhaustively rather than
stopping at the first live instance.

**Restart calculus, updated but unchanged in conclusion.** These two
fixes change sim MECHANICS only, so on their own they would need no
fresh-CE gate. But a restart necessarily also picks up `fa95da5`, which
DOES change encoder inputs — so the T2-C warm-start hazard still gates
it. Decision: **still no mid-leg restart**; instead, run the fresh-CE
gate as real measured work, THEN decide. Consequence to record honestly:
this leg's Aethermaw ladder games remain morphless, and its Aethermaw
midgame exports may contain the illegal-move signature — that subset is
suspect fidelity evidence, and the signature is now known exactly.

**Known-bad artifacts on HF:** 6 midgame Aethermaw exports contain
engine-illegal moves (the 2026-07-15 `ladder_fogless` OOS on record is
the pre-07-18 terrain-leak era realizing the same class). Morphless-era
ladder exports remain valid REPLAYS but document games played under wrong
terrain from turn 4+. Cosmetic residue deliberately untouched: the
encoder still shows the two walls as castle terrain (`_parse_hex_code`
ignores `^Xo`) — changing that IS encoder-input territory.

### Cycle 37 — 2026-07-29 — the fork-shared-state audit: a THIRD live instance, and a general guard

**Box (21:10Z).** Iteration 12, 77 workers, zero aborts. Landed
`933888d` (fix) + slow tier verified 630 fast + 11 slow.

```
fresh_value_ce  0.7933, 0.6140, 0.5399
boundary_sum    -0.118, -0.317, -0.073   <- now NEGATIVE
  rolling |mean| last 3 = 0.169  (highest yet; still under 0.25, but
  the -0.317 reading individually exceeds the band)
```

WATCH ITEM, with the caveat stated: the box runs its launch-time clone,
so this is still the **noisy k=16** estimator — `d03d50c` raised the
sample cap to 64 but is not live there. A single −0.317 at k=16 is well
within what cycle 31 showed this estimator does on its own, so the sign
flip is not yet evidence of anything. Re-read after any restart, when
k=64 makes a single reading worth interpreting.

**A live sim-fidelity violation, found by auditing the class rather than
waiting for a symptom.** `_scenario_events` is a shallow-copied list, so
its `ScenarioEvent` ELEMENTS are shared across every fork — and
`fire_event` latches `ev.fired = True`. A search fork stepping `end_turn`
crosses a turn boundary (`_begin_side_turn` -> `_apply_command(init_side)`
-> `_fire_turn_events`), latches the shared event, and **the LIVE game
then never fires it.**

Measured in production init: **Aethermaw (ladder pool #1) carries exactly
8 unfired `first_time_only` morph events past init — the map's walls
never opened.** Since the 2026-07-18 terrain COW fix this was the leak's
residue (before that it was masked by the larger terrain leak). Impact:
Aethermaw self-play games since 07-18 effectively morphless (symmetric,
1/21 maps — not campaign-invalidating), a sim-fidelity violation, and
**Aethermaw validation exports carry OOS risk**, since Wesnoth replaying
them WOULD morph. Sibling forks within one search also saw inconsistent
futures.

**Second instance, dormant but code-real:** `_object_action` pulled
shared units out of `gs.map.units` and `_apply_effect_to_unit` rebound
their fields in place. This was the `u.attacks` suspicion from cycle 35,
now confirmed AT THE CODE LEVEL and measured DORMANT — every
`[object]`-bearing event in the current pool latches pre-fork
(prestart/turn-1 fire inside `__init__`). One scenario addition from
live, and the mini-map tentacles roadmap heads exactly there.

**The attack surface, now pinned three ways** (docstrings on both
`__deepcopy__`s, plus an executable spec
`tests/test_fork_isolation.py::test_fork_alias_contract`):

```
ALIASED across forks (mutation = leak):
  map.mask, map.fog, map.hexes (+ each Hex's terrain_types/modifiers)
  GlobalInfo._terrain_codes (deliberate, Dijkstra-cache keying)
  Unit OBJECTS (units is a fresh set; contents shared)
  _scenario_events ELEMENTS   <- this was the hole
  values inside shallow-copied stash dicts/lists
PER-FORK: sides (deep), stash containers, command_history, _actions_by_side
```

**Fixes.** `GlobalInfo.__deepcopy__` now shallow-copies only UNFIRED
events (fired ones stay shared — their sole later write is an idempotent
re-latch, so steady-state cost is zero copies; the branch was already an
O(n) list rebuild, so the added cost is a `getattr` per element).
Verified independently: `fired` is the only mutable field on
`ScenarioEvent` (`actions` is read-only after parse), so a shallow copy
is exactly sufficient. `_object_action` moves to the replace-unit
pattern, and `_apply_effect_to_unit` documents+upholds "rebind fresh
containers, never mutate in place".

**A general guard, default OFF (`SIM_FORK_GUARD=1`).** Three instances of
this class in 11 days, and **all three were invisible to `state_key`** —
which is why a general detector earns its cost where N specific fixes
would not. `deep_state_fingerprint(gs)` hashes exactly what `state_key`
deliberately excludes (hex modifiers, attack tables, event latches,
terrain codes) and the flag asserts it unchanged around every
`mcts_search`. Reviewed: module-level constant read once, `None` when
off, so the cost when disabled is one `if` per search. A sensitivity test
proves it flips on all three historical surfaces; an e2e test proves no
false positive on a real 8-sim search. Plan: enable for ONE box smoke
iteration at the next campaign start, then off.

**Note this fix needs no fresh-CE gate**, unlike `fa95da5`: it changes
sim MECHANICS, not encoder inputs, so it is safe to pick up whenever the
box next restarts for other reasons.

**Not ruled out (Fable's own list):** `combat_outcomes.py`'s DP
enumeration was not line-audited for parent writes (evidence is indirect;
a box smoke with the guard closes it properly); the Aethermaw export OOS
is a RISK STATEMENT, not measured — whether any exported replay actually
traverses the 13 morph hexes is unverified.

### Cycle 36 — 2026-07-29 — a carried-forward "fix" REFUTED before it was written

**Box.** Iteration 10, 77 workers, learner alive, zero aborts. Credit
**$10.74** (~32 h).

```
fresh_value_ce  0.7267, 0.3855, 0.7933 (+-1.52)   still no readable trend
boundary_sum    -0.002, +0.096, -0.118
  rolling |mean| last 3 = 0.008   -> very clear of the 0.25 band
advice_out_norm 0.3737 -> 0.3845  (RETIRED as evidence, cycle 32)
```

**Cycle 35 carried forward a spec to fix "cross-process encoding
nondeterminism". Measured first; it is REFUTED.** The concern was that
`encoder.py` tie-breaks multi-terrain hexes with
`next(iter(terrain_types))`, and that enum-set iteration order is
address- or `PYTHONHASHSEED`-dependent — which would let ~76 worker
processes and the learner encode the same hex differently, the same
act/train desync class as the village bug. That would have been serious.

It is not real: **`Terrain` and `TerrainModifiers` are `IntEnum`**, so
`hash(member) == hash(member.value)` — an int hash, identical in every
process. Verified in fresh interpreters at `PYTHONHASHSEED` 1/17/424242:
iteration order byte-identical every time.

Two things this bought beyond the answer: no production change, and no
encoder-input change that would have needed its own before/after
fresh-CE validation we cannot currently afford. **Measuring the premise
was cheaper than the fix would have been.**

Kept as a **guard**, not a note (`7a8dc9f`): the property is
load-bearing and invisible, and `IntEnum -> Enum` looks like a harmless
refactor while silently restoring identity-derived hashing. One test
pins the hash basis; the other spawns fresh interpreters at differing
hash seeds and asserts the orders agree — the only way to actually prove
a cross-process property. 624 fast tests pass.

**Pattern worth naming:** Fable's instinct about the fork-shared-state
CLASS produced the village bug (real, serious); this particular instance
was not real. That is a good trade, and it is exactly why claims get
labelled inferred — being cheaply wrong about an instance is the price
of being right about a class.

**Fable dispatched: audit the rest of the class.** The village bug's
shape was "`Map.__deepcopy__` aliases a structure for speed, something
mutates it during search, hypothetical futures rewrite the real
present". The fast-path deepcopy is an optimization all of MCTS depends
on, so the aliasing is not wrong — the mutations are, and there is no
reason to think we found the only one. Package: enumerate what
`Map.__deepcopy__` / `GlobalInfo.__deepcopy__` / `fork()` alias vs copy
(that list belongs in the codebase regardless of findings), find every
in-place mutation of those structures reachable during search, classify
LEAKS / SAFE / UNREACHABLE with evidence, and measure the top suspect
(`scenario_events.py:1263-1308` rebinding `u.attacks` on fork-shared
Units) rather than settling for the inference. Also asked whether a
general guard — a debug assertion that a parent's `state_key` is
unchanged across a search — beats N specific fixes.

### Cycle 35 — 2026-07-29 — the flaky test was a REAL production bug: search imagination rewrote the real game

**The biggest correctness find of the run**, and it came from refusing to
dismiss a 1-in-3 flaky test. Landed as `fa95da5`.

`_capture_village` stamped `TerrainModifiers.VILLAGE` onto a `Hex` that
`Map.__deepcopy__` **ALIASES across every MCTS fork**
(`classes.py:238`, `hexes = self.hexes  # alias`). So a **hypothetical**
village capture explored inside a search **permanently rewrote the live
game's encoder input**. The aliasing docstring claimed the add was
"idempotent on actual villages anyway" — measured FALSE: scenario-pool
builds never stamp the modifier, so the first capture is a real
mutation.

Divergence chain, measured end to end:

```
modifier flips -> encoder.py:1105 reads it into the hex token
  -> all logits shift -> priors renormalize ~1e-4..1e-3
  -> _expand's stable sort reorders near-ties
  -> rng.gumbel assigns noise BY POSITION
  -> different candidate tournament -> different visits
```

Both earlier symptoms reduce to this: "root priors stepped ≤8e-3 after a
search" IS the leak (magnitude scales with how many captures the search
imagined), and "enumeration order permutes run-to-run" is the leak
re-sorting the prior-ordered edge list.

**Why it mattered beyond the test.** Stored transitions hold
**references** to game states whose hexes alias the live map, and the
trainer re-encodes them at train time (`trainer.py:449`, `:1075`,
`:1440`). Any search-imagined capture occurring AFTER a state was stored
silently changed that state's training encoding — **distillation was fit
against inputs the search never saw.** It also violates the CLAUDE.md §6
observable-state contract: the bit encoded "some search line once
imagined capturing this", which no player can observe.

**Sim fidelity untouched** — verified by grep over every reader: no
mechanics path reads the modifier (healing uses terrain codes, income
uses `_village_owner`), so replay parity and the combat oracle are
unaffected. `encoder.py:1315`'s reader is dead code (no callers).

**My inferred hypothesis was REFUTED.** I recorded lazy vocab
registration as the likely mechanism (cycle 34, labelled inferred).
Vocab held at 44 entries through every failing iteration and the leak
reproduces with zero growth. No vocab change is warranted. Recorded
because the label "inferred" is what made it cheap to be wrong.

**Fix:** ownership has ONE source of truth, the per-fork
`_village_owner` map (already deep-copied per fork, already hashed by
`state_key`). Encoder/rewards/diff_replay read `owner-map OR modifier`,
keeping the modifier honored for the live-Wesnoth converter path which
legitimately stamps it. Repro loop went **12/24 leaking to 0/24**;
622 fast + 11 slow green; the flaky test itself untouched.

**Behaviour delta accepted (Claude's call):** scenario-pool PRE-OWNED
villages now encode as owned from turn 1 — which `scenario_pool.py:859-864`
always said should happen — and imagined captures now read unowned. The
alternative (copy-on-write hex replacement) would preserve today's bits
exactly, but preserving a measured inconsistency is worse.

**Decision: do NOT restart the box to pick this up.** The fix changes
encoder inputs, and warm-starting a checkpoint into changed encoder
semantics without a fresh-CE gate is exactly the T2-C hazard (warm-start
MAE 0.217). The running leg's marginal value is already near zero
(cycles 28-30: ~180k steps against a 450k-1M detectable gap), so the
trade is "unvalidated lineage disturbance" against "a small, measured,
non-collapse-scale corruption for ~33 more hours". The fix's value is for
FUTURE legs, and its first use should be gated on a fresh-CE check.

**Carried forward (specs only, not implemented):**
- Cross-process encoding nondeterminism: `encoder.py:1104` and
  `_first_terrain_id` (`:1300`) use `next(iter(terrain_types))`, and
  enum-set order is address-dependent — so multi-terrain hexes can
  encode differently across process restarts (resume/serve boundaries).
  Within-process stable, so unrelated to this flake. Fix = deterministic
  pick (`min(tt, key=...)`), needs its own before/after CE check.
- Same bug class, dormant (INFERRED, not measured):
  `scenario_events.py:1263-1308` `[effect]` handlers rebind `u.attacks`
  in place on `Unit` objects shared across forks. A turn-boundary event
  mutating a STATIONARY unit inside a search would leak exactly like the
  village bit. Worth a probe on Hornshark before trusting it.

### Cycle 34 — 2026-07-29 — 32 sims is NOT the constraint; a committed rule was measured wrong; a determinism test is flaky

**Box.** Iteration 9, 77 workers, zero aborts. Credit **$11.02** (~33 h).
`fresh_value_ce` 0.3855 then 0.7933 (±1.52) — still no readable trend;
`boundary_sum` +0.096, −0.118, rolling |mean| far under 0.25.

**The premise of my own package failed, and that IS the finding.** I
asked for a quality-vs-sims curve against a high-budget reference as
"ground truth". Measured: **two independent 512-sim searches on the same
state differ by TV 0.85-1.00** (7-state floor: mean 0.59). The m=16
Gumbel target does not CONVERGE with budget — it **concentrates**. σ
scales with `(c_visit + max_N)`, so from N=32→512 max weight goes
0.39→0.62, shelter mass 0.18→0.04, entropy 2.4→1.15, and the mass lands
on a winner picked by the Gumbel draw among 16 prior-selected
candidates. **Coverage never widens with N: visited = 16 at every
budget.** `m` is the coverage knob; N only sharpens the tournament.

So cycle 32's E0 generalizes: it is not "32 sims can't resolve the
advised action's merit" but **"no N in this family resolves
individual-action merit per decision; the signal exists only in
expectation across draws."**

**Answer to the question asked: 32 is at or near compute-optimal.**
Class-level bias to the 512-ref: 0.24 (N=8), 0.21 (32), 0.22 (64), 0.17
(128), 0.12 (256). **Quality-per-sim strictly declines above 32**; 64 is
not better than 32 (0.223 vs 0.210, within noise) at 2× cost;
compute-matched pairs (2×N vs 1×2N at equal total sims) split 9-12/20,
point estimates favouring MORE STATES over more sims. **Keep 32 sims,
m=16.**

**A rule I committed last cycle is measured WRONG — amended.**
`tools/recruit_prior_drift.py` told a future reader to prefer
`gumbel_m` 16→8 if the tripwire fires. Measured at 32 sims: m=8 moves
midgame recruit mass **−0.0135 vs m=16 — the wrong direction** (CI
[−0.052..+0.015]), leaves the cut band unchanged, and raises shelter
mass 0.20→0.32. The original reasoning ("halve the tried-and-cut
population") was plausible and still wrong, which is exactly why naming
the lever in code beat leaving it as folklore. Levers now in EVIDENCE
order: (1) playout-cap randomization (`--mcts-playout-cap`, already
implemented) at full-move N=128 with matched average cost — 128-targets
roughly halve the class bias, caveats being fewer targets/game and a
hotter label temperature, both unmeasured end-to-end; (2)
extraction-semantics changes, cheaply testable by re-running the
target-quality probe on a new checkpoint rather than by Elo. **Not
`gumbel_m`→8.**

**The tax vs N: constant magnitude, changing meaning.** The cut band
(q̂−v_mix) is ≈ **−0.07 at every N from 32 to 512** while visits-at-cut
go 1→8. It does not shrink — but its interpretation flips from
selection-on-noise (1-visit cuts) to mostly-real inferiority (8-visit
cuts), and its class-mass consequence fades anyway because sharpening
drains both cut and shelter mass into winners. Also measured: 32 sims
under-delivers recruit mass **mainly where recruiting is RIGHT** (the
low-N gap concentrates in 3/15 states where the 512-ref puts recruit at
+0.60/+0.62/+0.20), and mostly agrees, attenuated, where it is wrong.
n=15, so magnitude indicative, direction solid.

**A determinism test is FLAKY, and that is a real signal.**
`tests/test_inference_seam.py::test_mcts_search_through_seam_matches_direct`
failed once in three full fast-tier runs this cycle (620 passed / 619+1
failed / 620 passed) and passes in isolation. My only change was
docstrings, so it is not order-dependence from my edit — it is
intermittent. It corroborates Fable's independent observation that
**root edge enumeration order permutes run-to-run** (same key set) and
that root priors on a fixed state stepped by ≤8e-3 after a search.
**Not weakened, not quarantined** — handed to Fable to root-cause.

Mechanism hypothesis, **LABELLED INFERRED**: lazy unit-type/vocab
registration during search leaf encodes (uniform advancement reaching a
type absent from the checkpoint vocab → a fresh embedding row). Note
vocab growth here is INTENTIONAL and armed in production
(`watch_vocab_growth`, and `freeze_vocab()` is the existing lock), so
this is not obviously a bug — which is why nothing training-critical was
changed on the hypothesis.

### Cycle 33 — 2026-07-29 — the cycle-30 tripwire is now real code, and it reproduces the finding exactly

**Box.** Iteration 7, 77 workers, learner alive, zero aborts. Credit
**$11.95** (~35 h at $0.334/hr).

```
fresh_value_ce  0.4321, 0.5081, 0.7148, 0.4664   (noisy, no trend at n=7)
boundary_sum    +0.168, +0.208, -0.024, -0.002
  rolling |mean| last 3 = 0.061  -> well clear of the 0.25 band
advice_out_norm 0.3594 -> 0.3711  (RETIRED as evidence, cycle 32)
```

**A decision was one directory sweep from being unenforceable.** Cycle 30
decided to leave target extraction alone and "arm a cheap tripwire
instead". That tripwire existed only as probe scripts in the temp
scratchpad, which gets cleaned. Promoted to **`tools/recruit_prior_drift.py`**
(`e215314`) with `collect`/`compare` subcommands — the right shape,
because snapshots are game STATES and therefore checkpoint-independent:
collect once, re-compare as the lineage advances. Fixed what made the
scripts unshippable (a hardcoded absolute Windows ROOT; a dead
comprehension).

Design point worth keeping: the analysis core (`summarize` /
`paired_delta` / `escalates`) is deliberately **torch-free**, so the
DECISION is unit-testable even though collection needs a checkpoint and
a sim. `paired_delta` RAISES on mismatched state lists rather than
truncating, because a length mismatch means the checkpoints did not score
the same states and the pairing — the entire point — would be a lie. The
cycle-30 escalation rule is a function, not prose, so it cannot drift
from its own justification.

**Validated end-to-end against Fable's independent implementation** on
the same 70 states, reproducing cycle 29 exactly:

```
seed_20260718 (2,290,529) -> campaign_live_20260729 (2,403,615)
  turn>=3 : rec mean 0.264 -> 0.122   median 0.123 -> 0.037
  turn<=2 : rec mean 0.700 -> 0.731   (opening ROSE)
  paired  : midgame UP in only 4/53
```

Two independent implementations agreeing is a genuine check on both.
**Escalation did NOT fire**: 0.122 is still above the 0.05 floor, so the
tax stays "recorded", not "actionable" — exactly as cycle 30 specified.
620 fast tests pass. Snapshots are regenerable via `collect` and are
deliberately NOT committed (pickled `GameState`s are class-version
fragile and would bloat the repo); the T1 set is preserved locally at
`training/logs/recruit_snaps/` (gitignored).

**Fable dispatched: is 32 sims the root constraint?** Two independent
findings now trace to one root — the tried-and-cut tax exists because at
32 sims/`gumbel_m=16` only ~16 of 100-1300 actions are visited and ~8 are
cut after 1-2 sims, and cycle 32's E0 showed the target cannot resolve a
specific action's merit at that budget. The package is a
**quality-vs-sims curve measured against a high-budget (512-1024 sim)
reference, normalized by COMPUTE not by sims** — because sims and games
trade off, so a 64-sim target must be more than twice as informative per
unit compute to be worth it. Plus: does the cut-vs-shelter gap shrink
with N, and at what N does it stop mattering? Framed so that "32 is
fine" and "undetermined at affordable cost" are both acceptable answers.

### Cycle 32 — 2026-07-29 — the ADVICE channel carries no information (and `advice_out_norm` was never evidence)

**A claim I published repeatedly is REFUTED.** I read `advice_out_norm`
growing from zero-init (0.3417 -> 0.3711 over 7 iterations) as the model
"learning the scale up rather than ignoring it". Fable reproduced that
growth signature **identically with informationally void tokens**. A
zero-init parameter under Adam moves whether or not the signal carries
information. **`advice_out_norm` is retired as evidence of anything
except "the optimizer is running."**

**The control.** Placebo = cross-state permutation of complete advice
content (motif ids, feats, grounding vectors moved as a unit, a
per-seed derangement, train permuted only within train). This holds
parameter count, token count/shape, gradient pathway, fire pattern and
the batch-marginal token distribution EXACTLY fixed, varying only the
token<->state correspondence. Token-construction parity with
`build_advice_tokens` is asserted via `torch.allclose`, so the
reconstruction cannot drift from production. Explicitly NOT controlled:
presence information ("an opportunity exists here") cancels between arms,
so the test is silent on presence-only value.

**Three instruments agree.**

```
E0  does the 32-sim Gumbel target up-weight the advised action?
    prior-matched controls: mean D = +0.0010 +-0.0094
    same-class controls:    mean D = -0.0022 +-0.0092
    -> target treats the advised action like ANY same-prior edge

E1a live checkpoint, no training, 104 states:
    CE none 3.10 / real 3.11 / permuted 3.11   (real-perm ~ +0.001)

E1b fresh graft, matched training, 2 seeds:
    out_norm  real 5.13/5.55   placebo 5.06/5.58   (indistinguishable)
    heldout   paired real-placebo: +0.13 and -0.21 (OPPOSITE signs)
    both arms DEGRADE heldout CE -> what is learned is memorization
```

**Mechanism, and it is cycle 29's tax again.** The advised edge attracts
visits (1.03 vs 0.43 for controls), so it tends to be sampled, cut, and
graded below the unvisited v_mix shelter — the apparent same-actor
asymmetry is a prior-magnitude artifact that vanishes under
prior-matching. At 32 sims the target **cannot resolve the merit of a
specific advised action**, so there is no content signal in the objective
for the gate to couple to. This is an information-starved OBJECTIVE, not
a broken pathway — E3 shows the pathway learns fine when the target does
contain signal.

**Verdict: not carrying information, at this search budget with this
coupling.** Honest scope: E0 ran with advice OFF in search to avoid
circularity while the box runs advice ON (justified by the tiny act-time
footprint, prior-TV ~0.015 — but it IS a deviation); E1b's frozen-trunk,
62-state, hot-lr regime is a learnability probe, not a box replica. The
strong claim rests on E0 bounding what the target offers to learn.

**The user's requirement IS met, and is being exercised.** "A stronger
model may learn to ignore the signal (e.g. exp management)" — softplus is
asymptotically 0 with never-vanishing-sign gradient, exact zero is in
`advice_out`'s span, and empirically a suppress phase closed the CE gap
from +1.01 to **+0.0002** in 150 steps, while a selectivity phase reached
contribution **20.6 on follow-advice states vs 3.8 on ignore states**.
Production corroboration: the LIVE gate reads 0.28 mean vs 0.55-0.85 for
a fresh graft on the same states — the model is training the gate DOWN,
which is the opposite of the story `advice_out_norm` seemed to tell.

**Design-doc gap, verified and corrected.** `detector_training_signal.md`
promised `softplus(gate(state, advice_features))`; the code implements
`softplus(Linear(d_model,1)(actor_ctx))` (`model.py:331,406`) — state
only. The doc was wrong, not the code. State-conditional ignoring (the
user's case) still works; what is lost is reading the gate scalar as
"trust in THIS finding" — that would need attention weights. Doc fixed.

**Decision: keep `MCTS_ADVICE=1` for now.** It is a user-requested
feature, the measured cost is small (advisor ~0.3 ms/decision, ~1% of
gradient, act-time TV ~0.015), and flipping it mid-leg would add a THIRD
regime to an already-split leg — the same reasoning that kept target
extraction untouched in cycle 30. What changes is the JUSTIFICATION: it
is retained as a cheap standing experiment, NOT because telemetry shows
it working. If it is to become real, the candidate couplings are the
not-yet-implemented ΔV-weighted retrospective distillation push, or a
search budget large enough to actually judge advised actions.

**Note on box/code skew:** the box runs the tree as of its launch clone
(`2997de3`). `d03d50c` (boundary `pool=`), `5a62f80` and `b0d5468` are
NOT live there and will not appear in this leg's logs; picking them up
would need a restart, which is not worth it for telemetry.

### Cycle 31 — 2026-07-29 — box steady; boundary reading made watchable; two stale cycle-prompt facts

**Box health.** Instance 46182445, iteration 5 at 14:01Z, 77 workers,
learner alive, **no automated tripwire fired**, HF escrow uploading.
Credit **$12.54** (~37 h at $0.334/hr).

```
advice_out_norm   0.3417 -> 0.3533 -> 0.3594 -> 0.3633  (zero-init, growing)
advice_grad_share ~0.010-0.013                           (~1% of gradient)
boundary_sum      +0.053, +0.334, -0.066, +0.168, +0.208
  rolling |mean| last 3 = 0.103  -> clear of the 0.25 band
fresh_value_ce    0.47, 0.60, 0.58, 0.43, 0.51  (n=5, no trend readable)
```

**Two facts in the recurring cycle prompt are STALE — do not act on them
without checking:**
1. It names the box as `141.0.85.212:45924`. That host died in cycle 27.
   The live box is `ssh2.vast.ai:22444` (instance 46182445).
2. It lists T2 relevant-set wiring as remaining work. **Verified
   complete**: `transformer_policy.py:89,127` (ctor gate),
   `selfplay_worker.py:82,125` (flag + pass-through),
   `sim_self_play.py:1307-1309` (spool payload basis REJECTION),
   `mcts_policy.py:679,726` (holdout stamp). What actually remains on
   T2 is the T2-C problem — warm-start MAE 0.217 vs the ~0.017
   precedent, so the flag is not a drop-in and needs a fine-tuned leg
   gated on `fresh_value_ce` recovery. That is compute-blocked (cycle
   28/29 budget), NOT wiring-blocked.

**Clarification: the 0.25 boundary threshold is a MANUAL watch item,
not an automated tripwire**, despite the cycle prompt calling it one.
Only `--abort-decisive-rate` and the holdout-stall check actually abort
a run (`sim_self_play.py`). So a noisy boundary reading was never a
false-abort risk — it was a legibility problem.

**Landed `d03d50c` — boundary telemetry precision.** The reading is
watched against ±0.25 but was a mean over only k=16 sampled pairs, and
the five readings above swing most of that band on their own, making a
single reading close to uninformative. Sample cap 16 -> **64** (mean, so
sampling SE scales 1/sqrt(k); cost ≤128 no-grad forwards against a
~184 s train_step — negligible). Kept as a named constant, not a CLI
knob: telemetry precision, not a behavioural parameter.

Also now reports the **pool** size. `boundary_pairs_n` saturates at the
cap, so alone it could not distinguish "only 16 pairs exist" from "4000
exist, sampled 16" — i.e. it could not answer whether the noise was
fixable by sampling more, which is exactly what raising the cap assumes.
Iter line now reads `boundary_sum=<v>/n=<sampled>/pool=<population>`.
Tests call the production `_attach_boundary_sum` (stubbed only at the
model/encoder boundary) and pin all four properties. Full slow tier run
as required: **613 fast + 11 slow**.

**Fable dispatched:** is the ADVICE channel carrying information, or
drifting? `advice_out_norm` growing from zero-init is NOT evidence of
usefulness — a zero-init parameter under gradient moves regardless. The
package is a **placebo control** (structurally identical but
informationally void advice tokens, holding parameter count, token count
and gradient pathway fixed), plus the architectural question of whether
the gate can suppress to ~zero — which is the user's own stated
requirement that "a stronger model may learn to ignore the signal".

### Cycle 30 — 2026-07-29 — Q5 measured at n=100: UNDETERMINED (+35 ±68), and my tail theory was wrong

**Open question #5, the run's central claim, measured at 4x the power of
cycle 26.** Players verified by READING `decision_step`, not filenames —
and the filename trap fired again: `selfplay_local_20260718.pt` reads
2,299,999 and is NOT the seed.

```
LIVE campaign_live_20260729.pt  ds 2,403,615
SEED seed_20260718.pt           ds 2,290,529
protocol: 4e445c2 mirrored pairs, raw policy, max_turns=100,
          draw_weight=0.0, 19 chunks (seeds 101-119), 0 pairs abandoned
pooled:   55-0-45 (live)   Elo(live) = +35 +-68
mirror:   11 swept-by-live / 33 split / 6 swept-by-seed / 0 mixed
          sign test on sweeps p=0.33; binomial on decisives p=0.37
```

**Verdict: the q-transform fix is neither supported nor contradicted.**
The +35 lean points the right way and the CI is 4x tighter than cycle
26's −137 ±260 (**which this supersedes**), but it comfortably covers
zero. 113k steps is small against the 450k-1M gaps that ever separated
checkpoints. Honest headline: **no detectable strength change.**

**My tail explanation was WRONG — recorded because I published it.** In
`20f75e7` I attributed the long-game tail to "passive shuffling games"
running to the cap. Measured: **0 cap-endings in 116 games** (0/100
cross-play, 0/8 live self-play, 0/8 seed self-play; `timeout` maps
exactly to `ended_by in ("max_turns","max_actions")`, `eval_sim.py:200`).
The wall-clock monsters are long **decisive** games. There is no
live-vs-seed passivity difference to find — both are 100% decisive at
this horizon. Scope note: this is measured on RAW-POLICY eval games;
whether the BOX's MCTS training games ever cap out is still unmeasured,
so the iteration-time half of question #4 stays open with its leading
explanation now removed.

Bonus: this extends cycle 25's decisiveness observation to n=116 and
removes cap-truncation as a label-noise concern for current training.

**Timing tail, for whoever plans the next eval:** median ~45-65 s/game
(so 46.5 s/game was a fair MEDIAN), mean ~76 s, single games up to
~16 min. Budgeting off the median underestimates ~1.6x on the mean and
misses the tail entirely; per-pair JSON checkpointing is what made it
survivable.

**T1 verdict with the eval in hand: pull NEITHER lever.** The
tried-and-cut tax is a real structural bias whose measured consequences
sit below every detection floor we have — strength undetermined at ±68,
recruits/turn flat (2.12→2.19), zero passivity endgames. Changing target
extraction now would put a THIRD regime into an already-split leg.
Decision: **keep the mctx reference semantics; arm a cheap tripwire**
(watch midgame-recruit prior drift in the concentrated-prior buckets —
the bucket-for-bucket match was the original evidence). It graduates to
actionable only if drift accelerates or recruits/turn falls. If it ever
does, prefer `gumbel_m` 16→8 (a documented mctx *knob*) over
`max(q̂, v_mix)` (which rewrites completed-Q *semantics*).

**Disagreement, and the synthesis (Claude's call).** Fable argued the
`begin_turn` bool I landed in `5a62f80` is the wrong shape: the ctor
firing init_side is *correct by contract* — production midgame-splicing
(`midgame_starts.py:62-104`) deliberately cuts BEFORE the boundary
init_side and relies on the ctor firing it — so the invariant "ctor
states are boundary states" is worth preserving, and a
`from_midturn_state` classmethod mirroring `fork()` would touch zero
existing paths. **Fable is right about the contract and I have recorded
it in the code**, which my original comment got wrong by implying the
ctor was buggy. I am keeping the bool as the primitive, because a
classmethod built on `__new__` duplicates ctor field-init and drifts as
fields are added. The synthesis: a future `from_midturn_state` should
simply CALL the ctor with `begin_turn=False` and additionally restore
`_rng_requests` — sugar over the primitive, no duplication.

Fable's two subtleties applied to the shipped flag (both were real gaps
in what I wrote): `_rng_requests` starts at 0, so bit-exact stochastic
continuation needs the caller to restore it, and `command_history`
starts empty, so replay export is unsupported. Both now documented.

### Cycle 29 — 2026-07-29 — T1 ANSWERED: nothing pays for gold; the tax is in target extraction

**Open question #1 is answered** (Fable's probe; instruments in scratchpad,
70 affordability-gated recruit states from the live checkpoint, box search
config, advice=True, loaded via `peek_checkpoint_arch` so advice tensors
actually load). Verdict, in one line: **no active term in the training
signal pays for banked gold.**

What was RULED OUT, each with a cite:
- `rewards.py` gold terms are **structurally inert under `--mcts`** —
  `MCTSPolicy.observe` is a no-op (`mcts_policy.py:469-476`), gated by
  `uses_step_rewards` (`sim_self_play.py:346`). **This finally explains
  why the old `weight_gold=0` fix "did nothing": it could not have done
  anything.** A whole prior investigation was aimed at a dead channel.
- Combat-oracle attack bias: alphas are literally 0.0
  (`constants.py:164-165`) — dead machinery.
- Draw tiebreak / aux margin: `weight_gold=0.0` (`draw_tiebreak.py:79`);
  `--train-draw-tiebreak` not passed.
- Value head pro-hoarding: **refuted** — V(after recruit) − V(after
  end_turn) = **+0.229 paired, recruit better in 45/70**.
- Visit starvation: **refuted** — recruit visit share 0.307 ≥ prior 0.270.
- end_turn taught as a hoard move: **refuted** — taught DOWN in 63/64
  states once 6 already-decided positions are excluded.

**What IS real — the "tried-and-cut tax."** `extract_gumbel_policy_target`
(`tools/mcts.py:1855`) grades VISITED edges by backed-up q̂ but parks
UNVISITED edges at v_mix (`_completed_q`, `mcts.py:1502`). At the box
budget (32 sims, `gumbel_m=16`) only 16 of 100-1300 legal actions are
visited and ~8 are cut after 1-2 sims. Measured per edge:

```
cut candidates (1-2 visits): -0.018 BELOW v_mix (68/120)
survivors       (3-6 visits): +0.054 ABOVE v_mix (30/40)
~98% of legal actions never sampled -> shelter AT v_mix
```

So **being sampled and cut costs probability mass; never being sampled is
free.** Selection-on-noise makes a cut edge's q̂ biased low. This is NOT
recruit-specific in the search (paired q−v_mix: recruit −0.020, move
−0.044, attack −0.025 — recruit is treated *mildest*); recruit loses
because its prior is CONCENTRATED, so it reliably enters the candidate
set (56/70) while diffuse move mass shelters. The predicted drift matches
the leg: turn≥3 recruit prior **0.264 → 0.122** over 113k steps (down in
49/53) while turn≤2 ROSE 0.700 → 0.731. Reproducible: same sign 22/24 on
an independent-rng repeat.

Note the aggregate hides the sign: recruit mass goes DOWN at gold<25 and
turns 3-4, but **UP at turn 1-2 (+0.043) and UP at gold>50 (+0.021,
median 0.238 → 0.595)** — i.e. the target pushes recruiting up exactly
where gold is piled up. That is evidence AGAINST "it is taught to hoard".

**Env assumption closed (Claude, box-side).** No `env.sh` exists, so
nothing overrides the baked config, and the live command carries
`--mcts-sims 32 --max-turns 100 --max-turns-min 60 --draw-value-weight
0.25`. `--mcts-aux-score` IS set but that is the boolean enabling the aux
training HEAD — a different thing from `--mcts-aux-value-bonus`, the
search-time material shaper, which is absent and therefore **0.0**.

**Throughput, measured definitively** as a delta between two steady-state
samples (so uncounted in-flight decisions cancel):

```
T0 10:42:10 iters=1 games=42 dec=5767
T1 10:58:52 iters=2 games=52 dec=7797
--> +2030 decisions / 16.7 min = ~7,300 decision-steps/hour
--> ~17 min/iteration at steady state (earlier 27-38 min included warmup)
```

**CORRECTION (same day, from more data).** The ~7,300/hr above was
measured over a single 16.7-min window and is an OVERESTIMATE — it
happened to land on a fast stretch. Iteration cadence over a longer
span: iter 0 at 09:55, iter 3 by 12:05 = **~43 min/iteration**, i.e.
**~4,400 decision-steps/hour**. Iteration time is highly variable
(16.7 min for 1->2, ~67 min for 2->3).

The cause is almost certainly the same heavy tail Fable independently
hit in eval, where per-pair times spanned 8 s to 900 s+: an iteration
blocks until all 24 games finish, so a few pathologically long
"passive shuffling" games gate the whole iteration. One phenomenon,
two symptoms. Note this is ALSO a behavioural signal about the policy,
not just a cost problem, and it was invisible at the old eval horizon
of 30 turns where such games got truncated.

The budget conclusion is unchanged and in fact STRENGTHENED: at
4,400/hr, ~41 h buys ~180k steps, further below the 450k-1M
detectable gap. Use **4,400-7,300/hr with high variance** as the
honest figure, not the single-window number.

**Strategic consequence, and the cycle's decision.** ~41 h of remaining
credit x 7,300/h = **~300k decision steps**, against a 450k-1M gap that
has ever separated checkpoints detectably on this lineage. **More
training cannot buy a measurable strength gain on this budget.**
Measurement can, and is orders of magnitude cheaper (46.5 s/game
raw-policy eval vs ~80 min for one MCTS self-play game). So: training
continues (it is cheap and still accrues steps + validation exports), but
the RUN's effort reallocates to settling open question #5 at n≈200.

Fable's own recommendation agrees and is adopted: **do not touch target
extraction until Q5 resolves** — whether midgame recruit down-teaching is
pathology or good play is exactly what a strength eval arbitrates, and
both proposed levers (`max(q̂, v_mix)`, `gumbel_m` 16→8) diverge from the
mctx reference that the cycle-2 fix deliberately restored.

**Carried forward — CLOSED same cycle as `5a62f80`.**
`WesnothSim.__init__` unconditionally fired `_begin_side_turn`
(income, healing, MP refresh, turn bump), so any tool reconstructing a
MID-TURN state via the ctor silently got a free turn's worth of gold and
HP — invisible, because the state stays structurally valid. Fable hit it
and worked around it with a sacrificial-copy swap. Now there is a
`begin_turn: bool = True` opt-out: default unchanged (no game-playing
caller affected), and skipping also keeps a stray `init_side` out of
`command_history`, which would desync a replay export. Tests pin BOTH
directions so the guard cannot rot into a no-op. Slow tier run as
required for a sim change: 609 fast + 11 slow, all green.

One testing trap recorded from it: units live on `map.units`, **not**
`hex.unit`. The first draft of the HP/MP assertion read them off hexes,
found zero, and compared `{} == {}` — green while proving nothing. The
test now asserts non-emptiness before comparing. Worth remembering: a
passing assertion over an empty collection is indistinguishable from a
real one unless you check.

### Cycle 28 — 2026-07-29 — the budget is the binding constraint (credit $13.90)

**The number that should drive T3 planning, found by looking:**
`vastai show user` reports **credit = $13.90**, not the large balance the
run had been assuming. At the current box's $0.334/hr that is **~41 h of
runtime against ~58 h of mandate remaining.**

**Decision: do NOT upgrade the box.** I had costed a migration to a
192-core host (192x2.6GHz = 499 core-GHz vs this box's 96x3.4 = 326, so
~1.5x throughput for 1.8x price). At $13.90 that buys only ~23 h. The
current box stays. Recorded so a future cycle doesn't re-derive it.

**A stall scare that was not a stall.** Iteration 1 ran >45 min with zero
heartbeat movement (30 workers / 40 games / 5,226 decisions, unchanged
over 3 min). Checked before concluding: worker process states were
**76 `Rl` (running)**, the learner `Sl` blocked in `collect()`, no worker
tracebacks, no crashed workers. Nothing is stuck — self-play games are
simply very long:

```
worker 0: 1 game, 243 decisions, 56.6 min   -> ~20 s per decision
avg over completed games: 130.7 decisions/game (high variance)
```

At ~20 s/decision a 243-decision game costs ~80 min of one worker, which
is why most workers were still inside their FIRST game 80 min in.

**Two measurement traps hit and avoided in this cycle, both worth
remembering.** (a) `spool/stats/w*.json` is a **per-worker heartbeat
written at game completion**, NOT a per-game record — an early reading of
"18 -> 19 files per 60s" as a game rate was wrong. (b) Because heartbeats
land only at completion, in-flight decisions are uncounted, so the naive
"5,226 decisions / 82 min = ~4,100 decisions/hr" **understates** steady
state, possibly by ~2x. Estimates by different methods spanned
4k-13.7k decisions/hr, which is too wide to plan on.

**So no throughput figure is published here yet.** The definitive
measurement is a `decision_step` delta across a known wall-clock
interval; it is in progress. This matters because the historical gaps
that ever separated checkpoints detectably were 450k-1M steps, and
whether 41 h of credit clears that bar depends entirely on which end of
that range is real.

**Next:** land the decision_step/hr measurement; then decide between
(a) continue training, or (b) reallocate credit to a decisive n~200
strength eval — Fable measured raw-policy eval at 46.5 s/game vs ~80 min
for an MCTS self-play game, so measurement is orders of magnitude cheaper
than training and may be the better buy.

### Cycle 27 — 2026-07-29 — box replaced; the load signal was never ours; eval made steerable

**T3 — the box died and is replaced.** Instance 46142270 (host 18135) went
`actual_status: offline` with both SSH routes dead; a reboot did not
recover it after ~9 min. Nothing was lost: HF `tier_a_campaign.pt` escrows
decision_step **2,403,615**, identical to the local pull.

New instance **46182445** (machine 24774): 96-core quota, RTX 3090,
seeded from HF, **RESUME** (no `--reset-decision-step`, so the
combat-oracle anneal correctly continues), 76 workers, `games_per_iter=24`.

Two operational lessons, both cheap and both learned the hard way:
- **`vastai create` returns `success: False` while still handing back a
  contract id.** The first replacement (46180515, 192 cores) returned
  `False`, sat in `loading` for 26 min, and never created a container
  (`vastai logs` -> "No such container"). A second create on another
  machine returned `success: True` and was up in **under a minute**.
  Treat `success: False` as failed regardless of the contract id.
- **Hedge instead of waiting.** Racing a second box cost cents and saved
  ~half an hour. Only ONE may survive: two boxes both seed from and
  upload to `tier_a_campaign.pt` and would clobber each other's lineage.

**Open question #4 is half-answered, and the premise was wrong.**
`/proc/loadavg` inside a Vast container is **HOST-wide** (no lxcfs), so
"box load doubled 130 -> 272" was never a measurement of *our* workload —
it included co-tenants. Measured on the new box with the true quota known
(cgroup `cpu.cfs_quota_us/period` = **92.16 cores**):

```
host loadavg          173.16      <- includes other tenants, NOT us
our processes, sum %CPU  91.6      <- we use ~100% of our 92.16 quota
threads per worker         3       <- thread cap in force (2997de3)
```

So we are **quota-saturated, not oversubscribed**. What remains of #4 is
the *iteration-time* rise (~16 -> >25 min), which was an our-side
measurement; host contention from co-tenants now explains it at least as
well as any code cause, and is testable by watching iteration time here.
Recorded as: the load half is REFUTED as evidence about us.

**T-eval — Fable's package, reviewed and landed as `4e445c2`.** The run's
primary metric could not be steered by (cycle 26: ±260 Elo at n=8).
Fable confirmed by reading code, not assuming: horizon mismatch REAL but
not dominant (eval ran max_turns 30, below the training band's 60 floor,
yet 7/8 games were decisive by turn 30); low power REAL and dominant;
no pairing REAL (`_play_pair` drew a fresh setup per game). Now: mirrored
setup pairs (same setup, sides swapped) with sweep/split counts as paired
evidence, horizon default 100, and protocol (`max_turns`/`seed`/
`draw_weight`) stamped into every header and saved JSON.

Power, stated honestly: CI95 ≈ **681/√N** — n=50 -> ±96, n=200 -> ±48.
**8 games can never settle anything**; ~200 is the ±50 point, ~2.6 h
locally at a measured 46.5 s/game.

**A live bug, same class as the others.** `eval_sim._load_policy` built
policies without the checkpoint's structural flags, so an advice-trained
checkpoint's advice tensors were silently dropped as unexpected keys.
Confirmed live: the campaign checkpoint reports `advice: True`. Inert for
raw-policy eval, but it would have biased any `mcts:` eval **against the
live model specifically**. This is the fourth bug of the form "the
convenient path differs from the production path".

**Review changes on top of Fable's work.** The flag peek existed in three
copies each with a silent `except: pass`; collapsed into one
`eval_sim.peek_checkpoint_arch` that LOGS on failure, since a silent
fallback there means "evaluated a structurally different model".

**Disagreement recorded (Claude's call).** Fable asked to align
`elo_ladder`'s `draw_weight=0` to `elo_collect`'s PURE 0.5, citing the
user's 2026-07-11 decision. **Declined** — on checking, that decision
rules that *material* must not factor into evaluation, which both tools
already obey; the draw *weight* is a separate statistical question and
`elo_ladder` documents its own rationale (a Wesnoth draw is a turn-budget
timeout, not equality evidence). Documented the divergence instead so
neither can be quoted as the other. Fable was invited to push back with
an argument about what a timeout is evidence OF.

**Next:** confirm iteration time on the new box; re-measure strength at a
later checkpoint with n≈200 under the new protocol (open question #5);
Fable is on T1 (which term pays the model to hoard gold).

### Cycle 26 — 2026-07-29 — first STRENGTH read: no improvement detected (and a refuted hypothesis)

**The primary-objective measurement, stated without spin.** Ladder, 8
games, shared seeds, LIVE post-fix checkpoint (2,403,615) vs the SEED it
started from (2,290,529), i.e. 113k steps of the post-fix leg:

```
live_2p40M vs seed_2p29M:  2-1-5 (W-D-L)   ->  Elo -137 +-260
```

**Point estimate NEGATIVE, CI spans -397..+123.** So: *no evidence of
improvement*, and not evidence of harm either — the sample cannot
distinguish them. I am recording this as the headline rather than leading
with the caveats, because the q-transform fix is the run's central claim
and its first strength test came back unfavourable.

Caveats that are real but do NOT rescue the result: 113k steps is small
against the 450k-1M that separated earlier legs; 8 games is a wide CI; the
leg is SPLIT (advice acting-side dead for iterations 0-16, live after); and
the eval runs max_turns 30 while training runs 100, a genuine distribution
mismatch. What would settle it is more games at a later checkpoint, not
argument.

**Hypothesis refuted by measurement.** Iterations slowed (>25 min vs
~16 min) and box load doubled (130 -> 272 on 128 cores) right after
acting-side advice went live, so I predicted the prospective advisor was
the cost. **Measured on midgame states (7.8 own units): 0.3 ms/decision.**
Not the cause. The slowdown is UNEXPLAINED and is now an open question, not
a theory I keep repeating. (My cycle-1 "~0-3 ms" figure was measured on a
turn-1 state with ONE unit; the midgame number vindicates it, which is why
I re-measured rather than trusting it.)

### Cycle 25 — 2026-07-29 — first behavioural read of the post-fix leg (and a confound caught)

Pulled the LIVE campaign checkpoint off the box (decision_step 2,403,615)
and the seed it started from (2,290,529) — **113,086 steps into the
post-fix leg** — and ran the hoarding probe on shared seeds.

| checkpoint | bank | end gold | recruits/game | turns | **recruits/TURN** |
|---|---|---|---|---|---|
| SEED 2.29M | 38.3 | 30.7 | 41.7 | 19.7 | **2.12** |
| LIVE 2.40M | 44.2 | 61.3 | 26.3 | **12.0** | **2.19** |

**Read naively this says the pathology got worse** — recruits down 37%,
end gold doubled. It does not. **Game length nearly halved** (19.7 -> 12.0
turns), which mechanically depresses recruits-per-game and inflates end
gold. Normalised, the recruiting RATE is 2.12 -> 2.19 per turn: **flat**.

What actually changed is that games got much SHORTER, which is consistent
with the near-total ladder decisiveness logged since the fix (6/6, 11/11).
INFERRED, not measured: shorter decisive games are what a working policy
improvement looks like here; but 3 seeds and 113k steps cannot support a
strength claim, and this is explicitly NOT one.

**Probe bugs found by using it (3512584):**
1. It built `TransformerPolicy` without `advice=`, so the live checkpoint
   loaded with **12 unexpected keys** and its advice tensors were silently
   DROPPED. Inert for this probe (no advice tokens attach at act time), but
   a probe that quietly discards weights is one assumption away from lying.
   **Fourth instance of this class this run.**
2. `recruits/game` is confounded by game length. The probe now prints
   `recruits/TURN` so the un-confounded number is the one on screen — the
   confound nearly produced a false "hoarding got worse" headline in this
   very cycle.

Campaign: 2 post-restart iterations, `advice_out_norm` 0.3066 -> 0.3326,
boundary |mean| 0.0605, 101 workers.

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
