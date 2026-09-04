# Technique activation/deletion review — 2026-08-10

Opus review over docs/techniques.md, cross-checked against the measured
record (docs/autonomous_run.md, BACKLOG.md, docs/literature_scan_20260810.md,
CLAUDE.md §Current status). Verdicts with observables and an order of
operations relative to the first tier-b self-play campaign.

Effort key: **XS** <1h · **S** half-day · **M** 1-3 days · **L** box-hours + days.
External observables (human-holdout CE, RCA probe) marked with a star.

## ACTIVATE (default-OFF, evidence says turn on)

- **A1. `--distill-prior-discount 0.9`** — measured in-repo: sharpen_top
  +0.130 undamped vs +0.030 at λ=0.9; end_turn re-teaching +0.124 → +0.020.
  The in-repo-measured version of the self-referential-target hole the
  literature scan flags as fatal to the BC prior. Caveat: pre-registered
  bar was ≤+0.02, landed +0.030 on a 3-game sample. Observable:
  distill_sharpen_top + star human-holdout policy CE probe. XS (flag
  exists, spool-forwarded — verified). Do NOT stack with piKL anchor or
  Grill targets in the same leg; keep --mini-ratio 0 (draw rate climbed
  under damping on minis).
- **A2. `--human-anchor-file` (value rehearsal)** — protects the fresh
  value head (AUC 0.951 A/B) against the documented self-play erosion
  (0.88 → 0.60 in ~80 iters). Observable: star value AUC on imitation
  holdout. XS + cache build. Don't enable the policy-head extension in
  the same arm.
- **A3. Turn-cap jitter `--max-turns-min 60 --max-turns 100`** — fixed
  caps teach last-turn gold banking (founding pathology). Observable:
  banked-gold telemetry, decisive rate unchanged. XS. Re-anchor Elo.
- **A4. `--replay-buffer` as training-CLI default** — every campaign
  passes it and fresh_value_ce (the success metric) only exists with it;
  a leg without it silently loses the metric. XS. Precedent: playout-cap
  default. Keep no-replay path as debugging fallback.
- **A5. Abort tripwires (`--abort-decisive-rate`, `--holdout-size`,
  `--abort-holdout-stall`)** — the last two runs lost real money to
  silence ($15 hung box; mini collapse needed a purpose-built probe).
  Specify the stall metric FLOOR-RELATIVE (raw-CE tripwire mis-fired
  twice in the 72h run). XS. Holdout = tripwire, not success gauge.
- **A6. `SIM_FORK_GUARD=1` for one smoke iteration at campaign start** —
  caught three aliasing bugs invisible to state_key; the handoff is a
  new weight/config combination. XS. Known gap: `_defense_table` stash.

## DEACTIVATE (default-ON, should be off)

- **D1. REINFORCE as entry-point default** — the documented default is
  not the production path; the two-layer disagreement produced two spool
  forwarding bugs and one investigation aimed at a structurally dead
  channel. Make `--mcts` default with a `--reinforce` escape. S. Audit
  non-mcts callers first.
- **D2. Optional heads riding checkpoint stickiness** — aux_score /
  moves_left stick to checkpoints (strict=False re-enables silently);
  the box legs passed --mcts-aux-score, so the imitation lineage may
  carry never-measured heads. Read the keys of
  imit_tierb_rescued_2368k.pt and DECIDE explicitly; strip via the
  --reinit-value-head pattern (keys + optimizer moments) if dropping.
  XS to check, S to strip.

## FINISH (partially implemented, worth completing)

- **F1. Policy-head extension of the human anchor** — the BACKLOG's
  leading handoff-protection candidate (RLPD-shaped); loop, cache, and
  four-head CE all exist. Observable: star human-holdout policy CE flat
  across the leg (3.102 = t0 reference) + star RCA probe at leg end. M.
  ONE protection per leg (this vs A1 vs piKL) for attribution. Anchor
  CE is the BC objective (hard human action), not a search target.
- **F2. `--no-progress-turns`: READ the accrued would-fire stats** —
  collected since 2026-07-21, never read; free offline analysis that
  prices the clock (passivity escape #2). S offline; enforcement is a
  separate decision (re-anchor Elo; no interaction with draw-tiebreak).
- **F3. Actor pool: resolve the docs/BACKLOG contradiction by
  arithmetic** — techniques.md says "measured losing design (200 req/s
  cap, GPU idle)"; BACKLOG says "validate + activate, near-mandatory for
  tier-b". Decide: required req/s = steps/hr × sims / 3600 vs the
  measured 200; >2× gap = server rewrite, not validation. XS to decide,
  M to fix, S to delete. Cross-actor batching breaks bit-determinism;
  OpenerPolicy incompatible.
- **F4. Relevant-set encoder (T2) recovery leg — AFTER the handoff** —
  user order stands ("finish it"), mechanism confirmed (≤~3.4× Amdahl),
  but warm-start gate failed twice (re-measured 0.3513 = not a warm
  start). L. STRONGEST SEQUENCING CONSTRAINT: changes the action index
  basis — checkpoints/replay buffers/targets not interchangeable across
  the flag; keep far from the handoff leg. Never judge with same-weights
  ON-vs-OFF.
- **F5. Make the shaping seam honest under `--mcts`** — WeightedReward
  (1,522 LOC) is structurally inert on the production path; add a loud
  refusal when a non-default --reward-config is passed with --mcts, and
  document that draw_tiebreak.py is the live seam. XS. Do NOT delete
  rewards.py (customization is a first-class goal; REINFORCE consumes it).
- **F6. Scripted openers: spool support or delete** — first-class goal
  in CLAUDE.md but unreachable on both production topologies; S to
  wire when wanted, or a clean 369-line delete if unused this year.

## DELETE (remove the machinery)

- **X1. Detector-advice channel (whole conditioning path, ~1.2k LOC)** —
  refuted against a placebo with three instruments; the published
  "gate is learning" growth signal was retracted (reproduced from void
  tokens). Keep tools/swap_detector.py (independent diagnostic). M.
  Warnings: telemetry-schema break rotates history CSVs — migrate the
  hand-maintained history_15m.csv header in the same commit; checkpoints
  carrying advice params must load clean (cycle-48 loader precedent);
  do it BETWEEN campaigns.
- **X2. Cliffness consumers (`cliffness_bootstrap_alpha`,
  `adaptive_sim_budget`)** — OFF since 2026-05-10, no schedule ever
  picked, absent from the 72h measurement record; lit scan supplies the
  principled reason (C51 spread mixes aleatoric dice noise with
  epistemic uncertainty). Keep output.cliffness + the root log. S.
  Preserve the _BOOTSTRAP_PRIOR_VAR derivation in design_constants.md.
- **X3. `FORBID_IDLE_END_TURN`** — binds on 92.4% of decisions, forbids
  69% of HUMAN end_turns; with a BC-warm-started lineage it is
  counter-doctrinal and violates the mask contract's purpose. S. Delete
  its gate-specific tests wholesale (never weaken).
- **X4. Drill scenarios (ids, `--drill-ratio`, templates)** — declared
  unusable, cost real money twice (baked-env leg trained 15% broken
  drills; ratio-sum rc=2 relaunch storm). Redesign shares nothing with
  present code. S. KEEP the ratio-sum validation; update launcher
  templates + spool arg forwarding.
- **X5. Adaptive outcome bucketing (297 LOC + hot-path hook)** — never
  measured, serial-path only; motivation eaten by exact enumeration
  (1,725/1,725, zero fallbacks). Weakest delete: preserve the
  PARSS/OGA-UCT citations + split-trigger design in a note. S.
- **X6. Documentation-truth housekeeping** — --replay-pool parsed but
  ignored; _enemy_unit_at dead; stale combat_oracle "default 0.1";
  NUM_HEX_DYNAMIC_FLAGS comment vs value; --holdout-size help;
  design_constants weight_village 10.0 vs authoritative 2.0. XS.
  (imitation.json dead key already fixed in b4f6ded.)

## Notable KEEP-AS-IS (with warnings)

- **Forced faction ON** (settled) — NEW CONFOUND: self-play is all-
  Knalgan while the human-holdout CE spans all factions; prior decay
  will look faction-skewed. Log holdout CE PER FACTION during the
  handoff leg.
- **Playout-cap randomization ON** (settled). Cycle-34 note: full-move
  N=128 with matched cost roughly halves target class bias — the
  best-supported untried lever; lives in playout-cap knobs, NOT
  --mcts-sims. Pre-register before touching.
- **--mcts-sims 32-50, gumbel_m=16** — permanent do-not-retry entries
  (raising sims refuted; m 16→8 measured the wrong direction).
- **--train-draw-tiebreak OFF, moves_left_utility OFF** — user rulings.
- **draw_value_weight 1.0** — decent case for 0 but never run as an
  arm; do not flip during the handoff leg; own arm later.
- **value_label_smoothing OFF** — if value-target shape is ever
  touched, HL-Gauss is the shape-correct replacement, not larger ε.
- **gumbel_rescale_floor OFF** — real argument, but no CLI flag and no
  spool forwarding: activating as-is would break the symmetry contract.
- **--mcts-hierarchical-gumbel OFF** — pre-registered A/B never run;
  external support exists (MA Gumbel MuZero). Trigger: minis return.
- **midgame/mini/fogless ratios OFF for the first leg** — trigger for
  midgame/Go-Exploit: closest-approach telemetry shows the BC-warm
  policy still fails to reach contact.
- **--infer-bf16 OFF** — measured NO-GO on 3060; re-bench on datacenter
  GPUs.
- **Chance nodes + exact enumeration, tree reuse, transposition, FPU,
  virtual loss** — clean positive audit; no action.

## Suggested ORDER relative to the first tier-b campaign

**Phase 0 — pre-launch, no box (hours to ~2 days):**
1. X6 housekeeping, then X1, X2, X3, X4, X5 — migrate the
   history_15m.csv header in the same commit as X1.
2. D2: read the imitation checkpoint's head keys; decide explicitly.
3. D1 + A4 + F5: --mcts and --replay-buffer as entry-point defaults;
   inert-shaping refusal.
4. F2: read the no-progress would-fire stats offline.
5. Build the human-anchor cache; wire the periodic
   `--imitation-config --eval-only` probe (this IS the external
   observable for everything downstream).
6. Full suite (`pytest -m ""`) before committing.

**Phase 1 — the handoff leg:** imitation checkpoint + A1 + A2 + A3 +
A5 + A6; 100% ladder; nothing else changed. Pre-registered questions:
does the human prior survive self-play (star holdout CE + value AUC
probe), and does it move the external number (star RCA probe, n≥30,
raw policy, leg end — never merged with in-lineage Elo). Optional
cheap control: a short bare leg (probe on, protections off) first.

**Phase 2 — one protection per leg:** F1 or piKL, not both, not
stacked onto A1's arm if attribution is wanted.

**Phase 3 — throughput only, signal frozen:** F3 decision, then F4
recovery leg. Never stack a throughput change with a signal change.

**Deferred with explicit triggers:** hierarchical Gumbel (minis),
rescale floor (flag + forwarding first), draw_value_weight=0 (own
arm), midgame ratio (no-contact telemetry), playout-cap N=128 target
shape (pre-register).

**Cross-cutting warning:** every knob the WORKERS build targets with
(damping, playout-cap trio, prior biases, relevant-set) must be
forwarded to spool workers or the CE fights targets the live priors
never produced. Two forwarding bugs already caught on that seam;
damping is forwarded today (verified), the rescale floor is not.
