# BACKLOG

Forward-looking only. **History was removed 2026-07-31** — it lives in the
git log (`git log -- BACKLOG.md` recovers every prior entry, and commit
messages carry the reasoning). Cycle-by-cycle provenance for the
2026-07-28..31 autonomous run is `docs/autonomous_run.md`; established
Wesnoth rules are `docs/wesnoth_rules.md`.

---

## LEG-5 LAUNCH CONFIG (2026-08-21, user: "proceed! Let leg-5 train
## tonight with the mover frame grading and the various fixes (no
## projection)")

Config file: **configs/leg_l5.json** (file-mode launch — schema
requiredness is structural; extra_env carries the flag rulings):
- campaign tier_b_l5.pt, seed tier-b/a3/seed_imit_tierb_start.pt
  (restart from the Elo +211 seed; legs 1-4 weights all abandoned)
- PROBE_T0 3.2070; policy anchor v2 (sole prior protection — the
  F1 one-protection rule; lambda GONE: launcher emit env-gated,
  code default 1.0, the leg-4 killer)
- TURN_BOUNDARY_FRAME=mover (bake-off: accept 0.90 vs 0.67, robust
  verdicts; E2: blind 8%->0) — NEW estimand, fresh gate baseline
  measured 2026-08-21
- ABORT_K_MEDIAN=10 (K-collapse tripwire, exit 7)
- projection OFF (bake-off STOP: separation 1.0:1 as implemented;
  mechanism validated, parked pending variance reduction)
- everything else = code defaults per the leg-4 ruling set below
  (caps 60-100 jitter, GBC 0.1, value censoring, linear link b=5,
  tripwires CE/AUC/decisive/stall armed)
- scenario mix stays 60/20/20 (unchanged from legs 2-4; change one
  variable class per leg)
- **Pre-registered external gate: 40-game Elo vs the seed at
  ~100k steps (~12h). Leg must hold >= even with its own start or
  it stops there.** New telemetry to watch: tcs_blind_coord_frac
  (expect ~0), tcs_gate_shorten_per_plan, distill_prior_entropy
  TREND (the leg-4 disease channel), boundary_pairs_n (> 0 at
  last).
- LAUNCHER GAP found at launch: the qualify gate is NOT wired into
  vast_onstart.sh (run by hand for legs 4 and 5; leg-5 PASS 0.700).
  Wire it before leg 6.

## LEG-4 LAUNCH CONFIG (consolidated rulings, 2026-08-17 — read
## BEFORE writing any launch env; every item is a USER RULING or a
## shipped default, and the leg-3 cap accident was precisely a
## ruling that lived only in a launch env)

- **Value head restarts**; grader chosen by the A4 bake-off between
  the A3 seed on the current trunk and the seed on the imitation
  trunk (matrix running). If Q1 says trunk-limited → RESTART the
  leg from the imitation checkpoint (nothing in legs 1-3's weights
  earned keeping: erosion, collapse, 0-18). NO fallback graders
  (GBC-as-grader, frozen judge, dedicated net, hand-built features
  ALL REFUSED — "the only real solution is to fix the value head").
- **learned-or-exact principle** (standing): every training-loop
  quantity is exact (sim/DP) or learned; nothing hand-designed, no
  hand-tuned values. Kills heuristic playouts + feature graders.
- **Launch gate**: `holdout_probe_loop.py --qualify CKPT` must PASS
  (value_auc ≥ 0.60) or the leg does not launch. Drift tripwire
  default-armed (AUC floor 0.52 ×3).
- **TCS on, linear target link** (default; exp needs measured
  trust), **force-inclusion ON** (safe under linear),
  **projection OFF** (Q7 offline sign test under the seeded head
  gates any future use; approved).
- **Winnerless games value-censored** (value_weight 0, shipped) +
  ended_by demix telemetry; caps are NO-RESULT in eval too
  (replacement + guard shipped).
- **Turn cap jittered 60-100 as CODE DEFAULT** (shipped; do NOT
  override in the env). Pool drain-not-abandon default 1800s grace.
  Batched TCS boundary evals (shipped).
- **GBC aux ON** (event supervision, coef 0.1) — as auxiliary ONLY.
- **Anchors**: policy anchor v2 game-normalized; value anchor
  updates now redundant with the A3 seed? — decide at launch with
  the A4 numbers (do not run A2 value anchor AND wonder why the
  seeded head drifts; one owner for value training).
- **DEAD, do not relitigate**: luck ledger (Q3, rho^2 0.03/0.05),
  CRN keying (Q8, median 0 shared events), Q5/Q6 (moot), β head,
  --turn-project all placement, min_delta 0 arm, hindsight-credit
  mechanism (parked as measurement idea only).
- Pending pilots gated on the seeded head: Q7 (projection sign +
  noise gates), Q9 (forward cost + batching benchmark), M1/M2 (in
  A3/A4's fold as specced).

## NEXT ACTIONS (2026-08-13 — the short list)

0. **TCS is INTEGRATED, DEFAULT ON (user ruling 2026-08-14).** The
   rung-0/1 probe ran on a rented CPU box (300 ladder states, both
   the imitation seed and the F1-arm final, ~$0.4): revalidated
   accept 0.640/0.460, median accepted Δ ~2 C51 atoms,
   placebo-separated 5:1, ρ(Δ,survival)≈0. KL gate failed as
   pre-registered; user ruled PROCEED (gate = magnitude proxy,
   disputed). **Rung-0 headline: the imitation seed plays K≈12
   (end_turn ~8%) on ladder; the F1 self-play policy plays K median
   2–4.5 — turn truncation was ACQUIRED by self-play, not cured.**
   Shipped: `tools/turn_search.py` (core, shared with the probe),
   `tools/turn_policy.py::TurnCommitPolicy` (subclasses MCTSPolicy;
   trainer untouched), wired default-on through all three
   generation paths (`--no-turn-search` opt-out), 14 new tests +
   full-suite green + 1-iter production smoke. Deviations recorded
   in `docs/tcs_spec.md` §Integration: no boundary value-only
   experiences (redundant both-sides); reply arm implemented but
   default OFF (next single-variable A/B); CRN keying deferred.
   LEG 1 (tier_b_tcs, 2026-08-14, ~$2): ran 7 iters on a 4090;
   turn structure held at K 14-17 and floor-relative fresh CE hit
   −0.43, but the human-holdout probe eroded MONOTONE (3.207 →
   3.717 → 3.785 → 3.874) and the pre-registered abort tripwire
   killed it at 3 consecutive points over t0+0.5. Verdict: TCS
   fixes turn structure but does not hold the human prior without
   a policy anchor. GBC pivoted the same day (0d attribution test:
   events predict outcomes AUC 0.79, the value head's turn
   movement is noise AUC 0.53 — user's miscalibration hypothesis
   confirmed) and is now INTEGRATED as the event-supervision
   auxiliary, default ON (`wesnoth_ai/gbc.py`, docs/gbc_spec.md).
   LEG 3 CONFIG DELTA (recorded 2026-08-16, user-ruled): the policy
   anchor's rehearsal draw is now GAME-NORMALIZED (cache v2: pairs
   grouped per game; draw = game-first, one pair each) — v1 was
   uniform over pairs, weighting a game by its length (~15x spread),
   inconsistent with the trainer's per-game principle. The F1 arm
   and leg 2 were MEASURED under v1; leg 3's anchor behavior differs
   by exactly this delta. Stale v1 caches are auto-rebuilt by
   onstart's new validation block.
   LEG 2 config (TCS + GBC + F1 policy anchor):
   CAMPAIGN_FILE=tier_b_tcs2.pt, HF_SEED_FILE=tier-b/
   imit_tierb_start.pt, HUMAN_ANCHOR_POLICY_FILE=training/
   checkpoints/policy_anchor.npz (the measured erosion antidote:
   the F1 arm held ≤+0.27 for 356k steps), probe abort at
   t0=3.207+0.5 unchanged. ITS parked
   (`docs/planning_abstractions_litreview_20260812.md`).
   **LEG 3 PAUSED 2026-08-17 (turn-length collapse).** K
   (actions/side-turn) slid mean 12.2→3.6 / median 10→2 over iters
   7→19 while draws rose 0.00–0.10 → 0.68–0.75; the human-CE probe
   held 3.46–3.52 throughout (the anchor holds the PRIOR while
   search-driven behavior degrades — the probe is structurally
   blind to this). Predates the co-located Elo match (not
   contention). Mechanism hypothesis: boundary-only grading +
   force-included end_turn alternative + tempo-blind value head =
   the value-exploitation channel the reply arm was designed to
   guard (it was default OFF). Unconfirmed directly — TCS accept
   stats are NOT aggregated from actor-pool workers (telemetry gap,
   fix before leg 4). **Multi-turn projection shipped 2026-08-17
   (user directive, default OFF): `--turn-project reval|all` +
   `--turn-project-halfturns H` grades candidates H closed-loop
   half-turns past the boundary (linear cost; tcs_spec.md §3
   addendum).**
   **ROOT-CAUSE REPORT (2026-08-17, 17-agent Opus workflow):
   `docs/leg3_passivity_rootcause_20260817.md` — READ IT before any
   leg-4 design. Headlines: the collapse is turn truncation (moves
   per turn), NOT aggression aversion (attacks/game flat, contact
   1.0 every iter); K = 1/p(end_turn) exactly (corr 0.988, the
   hill-climb was NOT truncating — gate-confined fixes incl.
   projection@reval cannot restore K); seed = a draw flood at iters
   7-8 (won fights stopped converting inside the cap) BEFORE K
   moved; value head was a below-chance turn-ranker (AUC mean
   0.434) from leg entry, unread; the end_turn force-inclusion
   target defect is real but sign-conditional and reversed in the
   late phase (amplifier at most). 8 mechanisms refuted incl.
   anchor-v2 and fog stories. NO leg-2 telemetry survives (the
   assumed baseline was May REINFORCE data). Next: the M0-M6
   measurement ladder in the report (M1 frozen-state p_et probe
   first — weights vs state drift), NOT a training leg.
   ELO (2026-08-17, interrupted 18/40 by user, 9 games
   wall-clock-censored): tcs3 (mid-collapse 3.37M ckpt) lost
   0-0-18 to new_2p52M -> catalog -321.5 +- 149.7, ~460 Elo below
   the July tier-a anchor; coheres with RCA 0/28. Leg-3 end
   checkpoint pulled local
   (training/checkpoints/tier_b_tcs2_leg3_end.pt) for M1; all
   boxes STOPPED.**

1. 🟡 **Launch-system redesign — architectural, not guarded**
   (user ruling 2026-08-19: no incident-specific guards; fix at the
   root so the error class is unrepresentable). Replace the
   env-string launch surface + hand-maintained preflight with: ONE
   typed leg-config file (schema-validated; requiredness structural
   -- parser fails on missing decisions; supersedes the preflight
   list which has its own silent-omission mode one level up);
   launcher as an idempotent RECONCILER (N invocations converge to
   declared state; retires flock + pkill-before-spawn); process-
   group ownership for teardown (retires pkill bracket-pattern
   folklore); structured status file written by daemons (retires
   log-grep watchdogs/monitors). The flock + preflight shipped
   2026-08-19 are explicit STOPGAPS pending this. Also fold in:
   baked project image evaluation (PTX JIT + 10GB cold pulls).

1. 🟡 **Revisit the luck compensator** (user directive 2026-08-17).
   The Q3 probe retired the luck LEDGER as a value-label variance
   tool (rho^2 0.03/0.05, docs/credit_assignment_design_20260817.md
   "Q3 ANSWERED") — but the user wants an in-depth discussion of
   luck compensation as its own topic later, noting the Wesnoth
   player community's long experience complaining about luck (their
   accumulated intuitions are a real input: which fights feel
   luck-decided, ladder norms around RNG variance, etc.). Scope for
   that discussion: compensation in TRAINING targets vs in SEARCH
   grading vs as a PLAYER-facing/eval concept are different things;
   the probe instrument (tools/luck_probe.py) measures any of them
   for free, incl. on self-play games. Not blocking leg 4.

2. 🟡 **Hindsight credit assignment — idea parked, mechanism
   rejected** (user ruling 2026-08-17). The specific Q11 proposal
   (P(a|s,outcome)/pi log-ratio as an additive target term, HCA
   recast) is NOT approved: correlation-not-causation unbaselined,
   likely reduces to a scalar end_turn knob (hand-tuned = refused
   under learned-or-exact), and its logit-space integration surface
   predates the linear target link. But the user finds MEASURING
   credit via hindsight interesting in itself — revisit as a
   measurement/diagnostic direction (e.g., what do outcome-
   conditioned action statistics reveal about where credit is
   misallocated?) after the leg-4 value-head repair. Literature
   anchors: HCA (Harutyunyan 2019), CCA (Mesnard 2021);
   docs/credit_assignment_design_20260817.md Q11.

3. 🟡 **Revisit the no-progress stalemate rule** (user ruling
   2026-08-17: unsatisfied with the current detection criteria;
   default-off stands). Current rule (tools/wesnoth_sim.py:783-821):
   progress = unit-count change OR net-HP decrease OR village flip,
   counted per full quiet turn. Known blind spots to address in the
   redesign discussion: healing doesn't refresh progress (intended),
   but maneuvering/fortress sieges with sporadic 1-HP chip damage
   reset the clock arbitrarily; no notion of *reversible* progress
   (village trading back and forth counts as progress forever).
   Revisit alongside the leg-4 draw-label work (ended_by demix).

--- (below: the 2026-08-08 list, kept for provenance) ---

1. **DONE 2026-08-10 (rescued at 94%)**: tier-b imitation checkpoint =
   HF `imit_tierb_rescued_2368k.pt`, CE 3.102. **Stall follow-ups
   CLOSED 2026-08-10:** (a) hang mechanism found — _ParallelStream's
   bare blocking `out_q.get()` waits forever when a worker dies
   without its worker_exit message (OOM-kill/segfault); exactly the
   observed signature (silent, ~0 CPU, near file-list end where
   workers retire). Hardened: bounded get + corpse reconciliation
   with a loud error and lost-file accounting; regression tests pin
   it. Cannot be CONFIRMED as the 2026-08-08 cause (box gone) — it is
   the only unbounded wait in the path. (b) BOX-SIDE stall watchdog
   shipped: `scripts/stall_watchdog.py` (per-process CPU flatline →
   marker + SIGKILL; the onstart supervisor treats a marker-kill as a
   crash and relaunches). A hung leg now loses ≤~35 min, not days.
2. **Evaluate it**: holdout curve vs both A/B arms (expect ≤3.107 CE
   with a ~0.95-AUC value head), then the external probes — this is the
   first checkpoint with a plausible claim to move the 0-30 RCA number.
3. **Decide the imitation → self-play handoff** — the periodic
   human-holdout CE probe is WIRED (2026-08-10):
   `scripts/holdout_probe_loop.py` (launched by vast_onstart, CPU
   subprocess, hourly, 1200 pairs, arch-peeked; CSV escrowed by
   hf_upload_loop as `holdout_probe.csv`). Needs
   `replays_dataset_imitation/` staged on the box or it announces
   itself OFF. t0 reference: CE 3.102.
4. Multi-epoch question: one epoch left CE still falling slowly — a
   second epoch is ~$4.50/14h; measure marginal gain before habit.
5. **Handoff-leg launch — ASSEMBLED 2026-08-10; template is
   launch-ready.** `scripts/vast_onstart.sh` now DEFAULTS to the
   handoff leg: 15M arch, CAMPAIGN_FILE=tier_b_handoff.pt, seed
   fetched from HF tier-b/imit_tierb_start.pt (flags fixed + re-
   escrowed: aux/moves head booleans were still True after D2's
   tensor strip), NO decision-step reset (anneal pinned off;
   RESET_DECISION_STEP=1 to override), actor-pool topology (F3;
   SPOOL_WORKERS>0 = debug fallback), --distill-prior-discount 0.9
   (A1), MINI_RATIO 0 (A1 caveat), value-anchor build default ON
   (A2), turn-cap jitter 60-100 (A3), tripwires 0.35/20 + floor-
   relative stall 60 (A5), SIM_FORK_GUARD smoke gate before launch
   (A6; PASSED locally vs the seed, ~30 min at 2x15 turns, trimmed
   to 1x8), imitation dataset staged from HF
   (tier-b/replays_dataset_imitation.tar.gz, 67 MB, uploaded),
   holdout-probe + games-log escrow armed. Launching = one vast
   create call; propose specs+cost to the user first (standing
   rule). Original ruling record: ACTIVATED: `--distill-prior-discount 0.9`
   (A1), `--human-anchor-file` value rehearsal (A2, cache build
   pending), turn-cap jitter `--max-turns-min 60 --max-turns 100` (A3),
   `--replay-buffer` now DEFAULT ON at the training CLI (A4, shipped),
   one `SIM_FORK_GUARD=1` smoke iteration at campaign start (A6;
   `_defense_table` fingerprint gap CLOSED per user order).
   **A5 abort tripwires — THRESHOLDS DECIDED 2026-08-10 (user):**
   decisive-rate floor 0.35 over a 20-iter window; holdout stall on
   the FLOOR-RELATIVE fresh CE (fresh_value_ce − fresh_ce_floor,
   min-delta 0.01) with a 60-iter window. Stall metric repointed in
   code (the raw-CE version mis-fired twice in the 72h run);
   vast_onstart defaults updated, env-overridable
   (ABORT_DECISIVE_RATE / ABORT_WINDOW / ABORT_HOLDOUT_STALL).
   **F-item rulings (user, 2026-08-10):**
   - F1: policy-head extension of the human anchor — WIRE NOW, but
     OFF for leg 1 (A1 prior-discount is that leg's ONE protection;
     this becomes leg 2's arm if A1 fails the CE observable).
   - F2: no-progress would-fire offline analysis — RUN NOW over
     available games.jsonl data; enforcement stays a separate
     decision.
   - F3: actor pool ACTIVATED for tier-b without a fresh A/B (the
     200 req/s ceiling is 3-10× the 15M requirement; the "losing
     design" verdict was tier-a-specific — techniques.md corrected).
     The handoff leg launches with `--actor-pool N`, not spool.
   - F4: relevant-set (T2) recovery leg CONFIRMED after the handoff
     leg (basis change severs checkpoint/buffer continuity).
   - F5: non-default --reward-config + --mcts now REFUSED at arg
     validation (shipped, with test).
   - F6: scripted openers DELETED (tools/openers.py, --opener-spec;
     unreachable on both production topologies; the config-flip
     opener goal stays in CLAUDE.md — re-add from git with evidence).
6. **Self-play handoff design (the next big decision)** — the
   2026-08-10 literature scan (docs/literature_scan_20260810.md,
   grounded in docs/techniques.md) converges on: nothing protects the
   imitation prior once self-play starts (self-referential Gumbel
   target + strictly current-vs-current play). Candidate package for
   the first tier-b self-play campaign, each config-gated and
   pre-registering an EXTERNAL observable (RCA probe / human-holdout
   CE), per the standing +133-moved-nothing caveat:
   - piKL-style CE anchor to the frozen BC policy (trainer-side,
     cheapest; lambda sweep pre-registered);
   - frozen BC checkpoint as permanent league opponent (PFSP-lite;
     elo_ladder already dispatches two policies per side);
   - extend --human-anchor-file rehearsal to the POLICY heads
     (currently value-only);
   - search-side cheap pair: mctx-compatible cut-arm debias (reserve
     non-adaptive sims per halving phase) + Go-Exploit archive starts;
   - cheap corpus wins to consider: opponent-reply auxiliary head,
     HL-Gauss value targets.
7. Corpus follow-ups (cheap, parked): extractor gap for menu-picks
   armed by MOVES (Necromancer-pick shape, zero measured corpus impact);
   `setaside_pickadvance_force` support if force-choice games ever
   matter; eras beyond default/dunefolk-clean re-ruling.
8. **2026-08-12 diagnosis ("the self-play loop is distilling its own
   noise" — user's read-only investigation artifact) — fixes shipped
   same day:** F1 root cause = the Gumbel sigma's min-max rescale
   amplified value noise into a fixed ~5-logit target perturbation
   (rescale floor raised 1e-8 → 0.04 = one C51 atom, CLI +
   worker-forwarded, `distill_kl_prior` instrument added); F4 blind
   spots = distill telemetry now ships from actor-pool actors, and
   the human-holdout probe ABORTS training at t0+0.5 ×3 consecutive
   probes (PROBE_T0). STILL OPEN from the diagnosis: F2 gate (no
   self-play spend unless frozen-holdout value loss falls over 20
   iters and human AUC ≥ 0.75 — a launch-discipline rule, not code);
   F5 index-basis divergence root cause (guarded, un-diagnosed); F6
   side-2 win bias (57% of 1,255 decisive games, ~5σ — audit on a
   side-swapped subset); F3 replay-ratio 0.21 + RawEncoded/mask
   caching (deferred BY DESIGN until the target is trustworthy —
   fixing it first would fit the noise faster).
9. **Deleted-technique design notes (2026-08-10 review, user rulings
   X2/X5)** — preserved so a future revival doesn't re-derive them:
   - *Tier-2 adaptive outcome bucketing* (X5, deleted at
     `tools/outcome_buckets.py` + the mcts.py integration; recover via
     `git log -S outcome_buckets`). Design worth keeping: event
     hard-split on the discrete part of OutcomeKey (dead/slow/poison
     flags — legality-invariant, so a bucket's shared edges stay
     valid); ground-stats aggregation per MEMBER so a split is a warm,
     unbiased re-grouping; split trigger = visit-weighted-median
     bisection on an HP axis when the two halves' mean values differ
     by > z_sig standard errors (no hand-tuned threshold). Lit: PARSS
     (Hostetler et al.) for coarse→fine split-in-half convergence;
     OGA-UCT (Anand et al.) for ground-stat aggregation + the
     value-heterogeneity significance trigger. Deleted because it was
     serial-path-only (conflicts with leaf batching, the actual GPU
     win) and never measured.
   - *Cliffness consumers* (X2, deleted: `cliffness_bootstrap_alpha`
     Bayesian backup shrinkage + `adaptive_sim_budget`; recover via
     `git log -S cliffness_bootstrap_alpha`). `output.cliffness`
     (= std of the C51 distribution) and the root-cliffness debug
     log STAY. Future-improvement marker (user order): cliffness is
     the network's own per-state uncertainty estimate — a natural
     input for epistemic-uncertainty work (exploration bonuses,
     targeted search budget, OOD detection) once there's an
     experiment that measures it; the deleted Bayesian-shrinkage
     derivation (uniform prior on [-1,1], var 1/3, scale =
     σ²p/(σ²p+α·cliff²)) is in docs/design_constants.md.

Context for all of it: **CLAUDE.md §Current status (2026-08-08)** —
corpus certified 100% (24,796), imitation pool 19,367 games / 2.57M
winner pairs, A/B verdict (warm policy + fresh value head,
`--reinit-value-head`).

---

## Where the project stands (2026-08-04, pre-imitation — kept for context)

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

The lineage improves at ~4,000-7,000 decision-steps/hour, and there is
no evidence yet that more steps close an external gap. Choose
deliberately between:

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

### Corpus fidelity status (2026-08-06 full sweep)

**24,776 / 24,893 accepted replays reconstruct bit-exact (99.53%).**
Full-corpus extract+diff sweep (35 min, 10 workers) after five
engine-parity fixes found by two 1% samples: quick_4mp leader
refresh, pick_advance narrowing (menu + forced mode), add-on
map-header start offset, random-ToD derivation from attack labels,
init_side healing-gate split. Residue: 78 divergent (payloads kept
in the session scratchpad ledger; classes: src_missing/defender_
missing cascades, late insufficient_gold, path_non_adjacent
teleport-shaped, 5 small-shortfall turn-2 gold files), 39
extract_none (mini_edited 15 + Aethermaw 7, un-diagnosed). Six
provably OOS-corrupt recordings deleted (turn-1/2 spend exceeds
recorded gold; engine itself OOS-errors, user-verified) --
`tools/check_replay_consistency.py` proves this class from a file's
own data; run it at ingestion. `training/logs/
replay_dispositions.jsonl.gz` tracks every file's class.

### 3b. Throughput program (user orders 2026-08-05, from the 15M profile)

**Always-on fleet profiling LANDED (2026-08-06, 88ea911, user order):**
`--prof` on the learner arms `tools/prof_hooks.py` in every spool
worker (env-inherited `WESNOTH_PROF=1`); per-component seconds
(sim.step, sim.fork, encode, forward, enumerate) accumulate into the
existing heartbeat JSONs and `tools/prof_report.py` renders the
fleet-wide breakdown on demand mid-campaign. Measured overhead
0.23us/wrapped call (~0.006% of a decision) — cheap enough to keep ON
for the whole campaign (armed in the §4c launch checklist). CUDA
caveat: no synchronize in production, so GPU-worker forward
attribution is async-skewed; the CPU fleet (the norm) is exact.
`profile_rollout.py` gained production-parity flags in the same
commit (checkpoint-driven relevant-set, playout-cap ON by default,
--torch-threads).

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
