# BACKLOG

Forward-looking only. **History was removed 2026-07-31** — it lives in the
git log (`git log -- BACKLOG.md` recovers every prior entry, and commit
messages carry the reasoning). Cycle-by-cycle provenance for the
2026-07-28..31 autonomous run is `docs/autonomous_run.md`; established
Wesnoth rules are `docs/wesnoth_rules.md`.

---

## Where the project stands

- **The learning signal was broken in four measurable ways. Fixed.** The
  lineage provably improves against its own past: **+133 ±57 Elo**
  (2,515,896 vs the 2,290,529 anchor), triangulated over 340 games, with a
  transitive check ruling out "beats its own parent while going nowhere".
- **The external gap is unchanged: 0-0-30 vs the built-in RCA AI**, median
  leader death turn 10, re-confirming the founding number at 3.3x the
  sample. Read this as *"the signal was broken and is now fixed"*, NOT
  *"the policy got good"*.
- Caveat that must travel with those two numbers: **+133 was measured WITH
  search; the RCA eval is RAW policy.** Different objects — never merge.
- **2026-08-04 eval results (full detail: `docs/tier_b_brief.md` +
  `docs/eval_box.md`):**
  - **T-C: +61.2 ±18 Elo raw-vs-raw** in-lineage (395 games) — search
    gains DO reach the deployed raw policy (~half of +133).
  - **T-B: KILL** — the 32-sim search root value scores **−0.141 AUC
    WORSE** than the raw head at outcome prediction on human states
    (CI [−0.201, −0.079], worse in every phase). No value-distillation
    channel; "repair the value head from search values" is dead.
  - **Mini passivity mechanism found** (workflow, adversarially
    verified): confined to the 3 fixed-ToD mini maps; a self-referential
    Gumbel PRIOR ratchet (~3.9-logit prior gap vs ~0.5 logits of value
    restoring force), no reward asymmetry. Decisive de-confound pending:
    force `random_start_time` for the mini pool.
  - **Export sweep: 538/574 clean.** 2 fogless OOS = known-fixed
    (9133cca, bare-clone schedules.cfg); **30 midgame OOS = phantom
    advancement `[choose]`s — ROOT-CAUSED AND FIXED (6712c70)**:
    `sample_midgame_start`'s prefix walk left accumulated advancement
    events on the start state; the sim's first attack flushed them
    into the export. Residual open item: one real advancement exported
    with NO [choose] (Den of Onis attack 168) — rarer, separate; **2 mini OOS = spawned-tentacle
    divergence** (engine re-rolls the `random_traits=yes` turn-1 spawn;
    our reconstruction is self-consistent, engine's monster differs).
- **All boxes stopped 2026-08-04** (eval box work complete; T2 box start
  was queued by the host, then canceled — resume deliberately).

## The decision to make before spending again

The lineage improves at ~4,000-7,000 decision-steps/hour, and the step
gaps that have ever been *detectable* on it are 450k-1M — i.e. **100-200
box-hours per measurable increment**, and no evidence yet that this closes
an external gap. Choose deliberately between:

- **(a) more compute on the current signal** — expensive, unproven;
- **(b) a cheaper/faster net so the same money buys more steps** — this is
  item 1 below;
- **(c) a better learning signal** — highest ceiling, least specified.

Do not default to (a) by momentum.

---

## Do next (ordered)

### 1. T2 — fine-tune leg for the relevant-set encoder

The only concrete route to "the same money buys more steps". The encoder
is built, wired and tested behind `--relevant-set-hexes` (default OFF) and
cuts hex tokens **~870 → ~119**. The scaling constraint is **sequence
length, not parameters**: hex tokens are 98.3% of the sequence and the
forward pass is ~91% of decision cost.

- Expected gain: **≤ ~3.4x end-to-end** (Amdahl-bounded). Do NOT quote the
  4.3-4.8x figure — that is the forward-component win only.
- **Blocker (T2-C):** warm-start value MAE **0.217** against a ~0.017
  precedent. The weights load; the function does not carry.
  **NB 2026-08-04: this number is SUSPECT** — it was measured through
  `eval_value_metrics`, which until `cdf263a` dropped the
  `relevant_set` flag on its encode path (same bug family as the
  8e8 policy-loss explosion). Re-measure through the fixed path
  before treating 0.217 as real warm-start damage.
- **Needed:** a short fine-tune leg with the flag ON, gated on
  `fresh_value_ce` recovering to its flag-OFF level (read it
  **floor-relative**, and skip iteration-0-after-restart).
- **Do not** judge it with a same-weights ON-vs-OFF eval — that measures
  warm-start damage, not the encoding's ceiling.

### 2. Mini-map passivity drift — root cause
**ANSWERED 2026-08-04 (workflow verdict, adversarially verified — see
"Where the project stands" above and `docs/tier_b_brief.md`).** What
remains here is the DE-CONFOUND EXPERIMENT (force `random_start_time`
for the mini pool) and the config-level repairs (`--no-progress-turns`,
`_rescale_q` floor). Original framing kept below for context.

Real, weights-driven, accelerating, and it is the founding thesis in
miniature: banks gold, declines to commit, armies **adjacent** but not
fighting, and the materially-ahead side declining free leader-kills.

- Controlled evidence: **0/24 mini draws at 2.40M vs 9/24 at 2.52M**
  (Fisher p=0.0008), same code, same seeds.
- Instrument exists: **`tools/mini_anatomy.py`**. Its *graded* precursors
  (non-end_turn actions per side-turn, unused-MP, median end turn) move
  well before the first draw appears.
- Scope honestly: ladder/fogless/midgame stayed 100% decisive, and
  head-to-head an aggressive opponent punishes the stall. This is
  behavioural on ~15% of the mix, **not** a strength regression.

### 3. Systematic export-fidelity sweep
**DONE 2026-08-04: 538/574 clean, every failure root-caused** (2
known-fixed 9133cca, 30 fixed 6712c70, 2 tentacle-spawn open).
**2026-08-06 closures:** midgame RE-VERIFIED post-fix — 15/15 fresh
exports clean in real Wesnoth (one flaky playback stall passed on
retry), which also closes the attack-168 missing-[choose] residual as
an old-code artifact (current code records unconditionally and
validates clean). **Tentacle class root-caused:** the embedded
scenario's side 3 carries controller="ai", so playback RE-RUNS a live
RCA (tentacles wander) while the sim's neutrals are stationary by
design — export-framing defect — THEORY REFUTED BY EXPERIMENT
(2026-08-06): a purpose-built repro (DummyPolicy tentacle games, 10
exports across all 3 tentacle maps, 273+ side-3 commands) validated
10/10 CLEAN in real Wesnoth — the engine replays AI-side recorded
commands faithfully, controller=ai is NOT re-simulated. The class
does not reproduce at HEAD; dispositioned as pre-provenance
campaign-era artifacts (same closure as attack-168). SWEEP LEDGER
FULLY CLOSED: all 574 verdicts explained, fixed, or shown
unreproducible. Repro harness: scratchpad tentacle_repro*.py.
Original framing below.

**Started, never completed** (the agent stopped without a result). The
question is NOT "clean up 6 known-bad files" — it is *how much sim
unfaithfulness is still hiding?* The 6 known-bad Aethermaw exports were
found by a **targeted census, not a sweep**.

- Path: `tools/run_validation_batch.py --hf-pull`; real-Wesnoth playback
  is proven working (engine-verified OOS and clean 1313/1313 runs).
- Want: pass/fail as d/n **per category and per code era** (pre/post the
  terrain-overlay + event-latch fixes), every failure root-caused as
  KNOWN-CLASS (terrain overlay, already fixed) or **NEW CLASS** — a new
  class would be a live sim-fidelity defect and the most important find
  available.
- Quarantine by manifest; **do not delete anything on HF unilaterally.**

### 3b. Throughput program (user orders 2026-08-05, from the 15M profile)

**Stage-1 measurements (2026-08-05 evening, all on the 3060 box):**

- **Distill damping VALIDATED on the trained 15M net** (paired arms,
  same weights, seeds shared): sharpen_top +0.130 undamped vs +0.030
  at lambda=0.9 (4.3x damping; end_turn re-teaching +0.124 -> +0.020).
  Honest note: the pre-registered bar was <= +0.02; measured +0.030 on
  a 3-game sample -- direction emphatic, bar narrowly missed. Next
  campaign can consider lambda=0.8 (pre-register first).
- **Intra-search batch-16: ~1.3-1.4x only.** Forward CALLS drop 9.4x
  but padded variable-length batches blunt the 3060 gain (per-call
  123ms for ~8 leaves), and serial per-leaf encode (18.6%) +
  enumerate (17.1%) become co-dominant. Keep learner default 16 on
  CUDA; not the lever.
- **bf16 inference: NO-GO on the 3060** -- 10.52ms vs 9.52ms fp32
  (cast overhead > tensor-core gain at batch-1/d384). Flag stays
  opt-in; re-bench on datacenter GPUs or at real batch sizes.
- **T2 checkpoint re-measure (clean --dest instrument): MAE 0.3513,
  NOT-A-WARM-START** -- worse than the suspect 0.217, so the artifact
  hypothesis is dead. Caveat: t2_relevant_set.pt trained briefly under
  the broken 8e8-loss trainer, so 0.351 = switch + poisoned gradients.
  Stage-2 measures the PURE switch (same weights, flag off vs on) and
  runs the recovery leg FRESH from tier_a_campaign.pt on the fixed
  trainer.
- **Probe run itself (the 2x24-game 15M leg): clean end-to-end.**
  policy loss 2.75/3.21 (sane), value 0.67 -> 0.45, holdout CE 2.406
  baseline, 0 draws in 48 games, VRAM peak ~2.4GB of 12.


Profile (15M, 3060, single actor): forward 53% at 20.7ms/leaf with ONE
forward per sim (batch-1 -- sequential halving cannot batch within a
game), enumerate 17%, encode 16%, sim 7%. Orders:

- **T2 relevant-set encoder: FINISH IT** (user, emphatic). Re-measure
  the suspect 0.217 MAE through the fixed eval path (cdf263a), then
  the fine-tune leg. <= ~3.4x end-to-end, attacks forward AND encode.
- **Batched central inference: ALREADY BUILT -- validate + activate.**
  `tools/actor_pool.py` is a full SEED-RL-pattern server (weightless
  actor processes, central GPU dynamic batching, no weight sync; B2 on
  the B1 seam; un-broken in the 2026-07 pre-flight), and it takes the
  learner's MCTSConfig object DIRECTLY, so the distill knobs and
  playout-cap ride along with no extra plumbing (unlike spool, which
  re-parses CLI flags -- two forwarding bugs caught there already).
  Remaining work is a box A/B: `--actor-pool N` vs the 76-spool shape,
  decision-steps/hour at 15M, playout-cap on. Near-mandatory for
  Tier-b: 15M CPU forwards would drop the spool to ~2k steps/hr.
- **Playout-cap randomization: ON by default -- DONE** (CLI layer
  only; library MCTSConfig and eval paths stay uncapped; workers now
  receive the trio on their command line).
- **Forward-kernel: improve** -- bf16 inference autocast + compile,
  opt-in flags, A/B on the box before defaulting.
- **CPU-side quantization (int8 dynamic) for spool workers: DEFERRED
  by user ruling 2026-08-05** ("no CPU-side quantization for now").
  Revisit when the batched-inference server lands: if workers stop
  doing forwards entirely, the item dies; if CPU forwards remain in
  the loop, int8 is a ~2-4x candidate that needs a target-quality
  check (pre-registered A/B) before shipping.

### 3c. Passivity mechanics follow-ups (user session 2026-08-05 evening)

- **Combat oracle -> PRIOR HARDCODED BIAS (user order 2026-08-06):**
  the machinery is RETAINED as a general facility for hand-placed
  prior nudges -- every instance defaults OFF, activated only in
  specific situations on explicit user order (pinned by
  test_prior_hardcoded_bias_defaults_off). The plumbing-strip plan
  is OBSOLETE. First new instance: end_turn bias on mini games
  (WESNOTH_PRIOR_BIAS_END_TURN_MINI=<float>, env-inherited for
  worker/trainer symmetry; scoped by the _scenario_category stash).
  NOT yet activated -- awaits the user's explicit per-run order.
- **Hierarchical Gumbel search** (principled fix for the factored-
  prior edge asymmetry): halve over ACTORS first with un-split actor
  mass, then targets within survivors. Fixes single-edge actions
  (end_turn) structurally out-competing split-mass actors in
  sequential halving everywhere -- measured at random init: end_turn
  is the fattest edge on ladder (2.9x median) though rank 3-14 on
  mini. **IMPLEMENTED 2026-08-06 behind `MCTSConfig.gumbel_hierarchical`
  (default OFF; CLI `--mcts-hierarchical-gumbel`, spool-forwarded).**
  Two-level candidate pick in `_gumbel_root_search`: sample m ACTORS
  without replacement by Gumbel(log total-actor-prior-mass), one
  Gumbel-argmax representative edge per actor, halving score rebased
  to the actor-level base (full mass survives cuts; sigma(q) and
  target extraction untouched). Pinned by
  test_hierarchical_gumbel_actor_mass_competition (distinct-actor
  guarantee + mass-rate slot occupancy + flat-mode same-actor
  double-booking pathology). **Pre-registered A/B (do not flip
  without it):** arm = identical leg + `--mcts-hierarchical-gumbel`;
  endpoints: (1) mini decisive rate up, (2) end_turn candidate-slot
  share down at matched sims, (3) ladder in-lineage Elo not degraded
  (the flat pathology is worst on ladder where end_turn is 2.9x
  median mass). Prediction: helps mini passivity only via slot
  composition, NOT expected to fix it alone (zugzwang illusion is
  the deeper mechanism; see T1-F).
- **DISCRIMINATOR MEASURED (2026-08-05 late): the passivity gradient
  is an HONEST self-play equilibrium, not target-math.** Cut-edge
  probe (125 pairs, live mini roots, 15M probe ckpt): cut
  alternatives under-graded by only +0.07 (loser's curse real but
  modest), while END_TURN deep-searches HIGHER than its shallow
  value (+0.42 -> +0.61 median, 0/27 revised down; paired
  differential ~0). Against the current passive opponent, passing
  genuinely evaluates well -- MUTUAL passivity is self-reinforcing
  equilibrium: each side's passivity makes the other's correct.
  Eliminated tonight as primary pumps: structural prior (refuted at
  init), combat oracle (was already zero), tried-and-cut tax
  (secondary, +0.07), indifference (gradient is positive). The
  escape is changing the GAME, not the search: make draw-bound
  material matter (train-draw-tiebreak) and/or price the clock,
  and/or exploration/opponent-diversity so punishment lines get
  discovered.
- **BOUNDARY DISCRIMINATOR MEASURED (2026-08-06): the pump is a
  DEPTH-LEVEL ZUGZWANG ILLUSION.** 22 pass-boundary pairs, mini, full
  32-sim search both perspectives: raw head sums +0.34 (mover
  optimism, T1-F reproduced) but SEARCH INVERTS it to -0.22 (0/22
  pairs > +0.15): at depth both movers read as LOSING (deep_a -0.14,
  deep_b -0.31). A mover's-curse inconsistency directly rewards
  passing for both sides (hand the curse to the opponent). Root
  cause of mini passivity = value-consistency defect, inverted at
  depth; the T1-F batch-mean boundary-consistency repair is
  ACTIVATED (targets both signs). Supersedes both the "honest
  equilibrium" and "raw-optimism pump" readings.
- **Mini-draw incentive repair pending user ruling:** draws measured
  as 100% mini artifact (leg: ladder 51/51 + midgame 24/24 decisive,
  mini 30/45 with the draw rate CLIMBING under lambda=0.9). The
  "rational passivity" reading is refuted by free-damage-taking
  (user observation) -- it's INDIFFERENCE: draw-bound => all actions
  EV 0 => no opposing gradient. Candidate: --train-draw-tiebreak
  (existing, default-off) restores a material gradient in drawn
  games (training only; eval stays pure per the eval contract), plus
  --no-progress-turns pricing the clock. DISCRIMINATOR still owed:
  are cut alternatives' completed-Q taxed below deep-searched truth
  (tried-and-cut mechanism) or honestly ~0 (pure indifference)?

### 4. Next-campaign hygiene (cheap, do before provisioning)

- **Size the GPU for the learner.** Two CUDA OOMs in ~26h on a 12GB 3060
  (learner alone held 11.63 GiB); each cost ~1h of spool refill. Take
  ≥24GB or lower `--replay-minibatch`.
- Consider `SIM_FORK_GUARD=1` for ONE smoke iteration at campaign start —
  it catches the fork-aliasing bug class and is free when off.

---

## Standing user ideas and decisions (preserved)

- **DECISION (user, 2026-08-05b): the 15M net gets the relevant-set
  encoder by RE-GROW from the T2-5M checkpoint** (net2net, d_head 32
  head-aligned), then a 15M recovery leg — not switch+recover at 15M.
- **DECISION (user, 2026-08-05b): `--train-draw-tiebreak` stays OFF.**
  The "mutual-passivity equilibrium" reading is NOT accepted as root
  cause — evidence audit: target-math ruled out as primary (measured)
  and the trap is self-play-local (aggressive opponents punish,
  earlier measurement), BUT deep_q(end_turn)~+0.61 for BOTH sides
  violates zero-sum consistency, matching the T1-F boundary bias
  (V(pre)+V(post) ~ +0.4..0.6 under fog). LIVE COMPETITOR: the
  side-to-move optimism bias may be UPSTREAM (both sides chase an
  illusion; passivity is its behavioral shadow). DISCRIMINATOR OWED:
  paired full-depth evals of identical mini states from both
  perspectives across a pass — sums ~0 = equilibrium story, ~+0.5 =
  bias is the pump and the fix is the T1-F value-consistency repair
  (batch-mean boundary penalty), not incentive surgery.
- **DECISION (user, 2026-08-05) — FULL MOVE TO TIER-B.** All training
  from now on runs the MEDIUM net (15.55M, d384/L8/H12/ff1536) unless
  stated otherwise. The Tier-a 5M lineage is measurement history, not
  a training target. Consequences: every launcher/runbook default
  should assume the 15M arch; throughput planning uses the 15M cost
  model (box_bench with a 15M checkpoint); the T2 relevant-set
  adoption question moves to "how does the 15M net get the encoder"
  (re-grow from a recovered T2-5M — d_head stays 32, head-aligned —
  vs switch+recover directly at 15M), with recovery-leg feasibility
  proven at 5M (fresh CE beat floor by widening margins in 8 iters).

- **DECISION (user, 2026-08-03) — evaluation runs on a RENTED BOX, not
  the laptop.** Operate under this assumption: do not plan, schedule or
  cost any eval as local work. A *separate, cheap, short-lived* box from
  the Tier-b campaign box — many cores, `--device cpu`, no GPU (one CUDA
  context per concurrent game exhausts VRAM before the cores are busy),
  hours not weeks. Runbook + the T-B/T-C queue: **`docs/eval_box.md`**.
  Why: measured 2026-08-03, the laptop has 7.6 GB RAM and ran a single
  eval game for **9 minutes of wall clock on ~1 second of CPU**,
  producing nothing — it page-thrashes, and had already crashed once.
  The "free local measurement track" was never free.
- **DECISION (user, 2026-08-02) — Tier-b directly**, accepting that
  Tier-a's exit gate (≥90% vs RCA) was not met (0-0-30). Arch locked at
  `d384/L8/H12/ff1536` = 15.55M on measurement, seed escrowed on HF as
  `tier_b_15m.pt`. **The grow is NOT a drop-in (value MAE 0.226 vs the
  0.017 precedent), so a recovery leg is mandatory before any strength
  number from it is interpretable** — see `docs/tier_b_runbook.md` §3
  and `docs/superhuman_training_plan.md` §11. Box is parked, not booked.

- **IDEA — Wesnoth add-on: human vs a trained model.** Let a human play in
  the real client against a checkpoint. Builds on the existing
  live-Wesnoth bridge (`wesnoth_ai/wesnoth_interface.py` +
  `add-ons/wesnoth_ai/` Lua), which today is eval-only; the new piece is a
  human-vs-model setup.
- **IDEA — tactical oracle / dominated-move detector.** Permute a turn's
  moves and check whether some permutation strictly dominates the played
  line; penalize dominated lines, the way a stronger player points out
  concrete errors. Run OBSERVE-ONLY first. (Partly realized as the swap
  detector; the *reward* half was never enabled.)
- **IDEA — `combat_outcomes`: exact DP past the complexity caps.**
  `enumerate_attack_outcomes` still returns `None` (caller samples) beyond
  `MAX_SCHEDULE=512` / `MAX_DP_STATES=4096`, chiefly BERSERK fights.
- **PARKED — moves-left utility OFF, indefinitely.** User: "the training
  signal is already complicated enough". The head still trains as
  telemetry (~0.03% of gradient); drop that too if it ever costs anything.
- **REVISIT — drills, redesigned.** Current drills are unusable
  (`DRILL_RATIO=0`). A refined version needs per-drill victory conditions,
  scenario-faithful economies, and a measured transfer test.
- **LIMITATION — swap detector matches by POSITION, not unit id.** Two
  identical units close together can mis-attribute a move/flank/MP-bank.

---

## Known hazards and traps

- **`training/checkpoints/tier_a_campaign.pt` is STALE locally**
  (`decision_step` 155464). That filename is *reserved* for the live
  campaign, so it will mislead. The real campaign is on HF
  (`momom2/wesnoth-model-checkpoints`, tier-a/) at **2,670,682**; the best *measured*
  checkpoint is **`campaign_live_20260730.pt` (2,515,896)**.
  **Always verify a checkpoint by reading `decision_step`, never by
  filename** — this trap has fired three times.
- ~~`campaign_live_20260730.pt` may exist only locally.~~ **ESCROWED
  2026-07-31**: uploaded to HF as `campaign_live_20260730.pt` and verified
  by round-trip (`decision_step` 2,515,896 intact). Deliberately NOT
  uploaded over the reserved rolling name `tier_a_campaign.pt`, which
  holds the newer-but-unmeasured 2,670,682.
  The other tier_a-era checkpoints were **deleted 2026-07-31** (user
  decision), freeing 954 MB: `training/checkpoints` is now 146 MB and
  holds only git-tracked files plus `campaign_live_20260730.pt`.
  **What that cost, recorded so nobody re-derives it by surprise:** the
  intermediate triangulation anchor `campaign_live_20260729.pt`
  (2,403,615) and the co-peak candidate `tier_a_campaign_20260719.pt`
  (2,747,117) are gone. The +133 result's two ENDPOINTS survive on HF
  (seed 2,290,529 as `selfplay_seed_20260718.pt`, best 2,515,896 as
  `campaign_live_20260730.pt`), so the headline is reproducible; the
  three-point table and any co-peak comparison are not.
- **Overlays dominate base terrain codes.** `^Xo` is impassable over a
  walkable base, and `^Kov` is a keep with no `K` base. Any code that
  inspects only the base of a `Base^Overlay` code is wrong. (Both bit us.)
- **`deep_state_fingerprint` does not cover the `_defense_table` stash**,
  so an in-place mutation there would be guard-invisible. No current
  writer does it.
- **`tools/eval_vs_builtin.py` has `--limit` but no `--offset`**, so
  chunked runs need a driver. One-flag fix.
- Box operations: see the `vast-box-ops-traps` memory (pkill matches your
  own SSH session; baked env vars are invisible over SSH; `rc >= 128`
  makes the supervisor stand down; `/proc/loadavg` is host-wide).

---

## Explicitly NOT doing (user decision, 2026-07-31)

- **Finish search-vs-RCA.** 0/30 raw and 0/5 with search already answer
  it; the one residual question (does the bridge forward-model track the
  engine?) is covered more cheaply by `reuse_frac`.
- **Co-peak comparison** (2,515,896 vs 2,747,117). Within-lineage; changes
  no decision.
