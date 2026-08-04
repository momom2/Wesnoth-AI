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

### 3. Unfinished: systematic export-fidelity sweep

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

### 4. Next-campaign hygiene (cheap, do before provisioning)

- **Size the GPU for the learner.** Two CUDA OOMs in ~26h on a 12GB 3060
  (learner alone held 11.63 GiB); each cost ~1h of spool refill. Take
  ≥24GB or lower `--replay-minibatch`.
- Consider `SIM_FORK_GUARD=1` for ONE smoke iteration at campaign start —
  it catches the fork-aliasing bug class and is free when off.

---

## Standing user ideas and decisions (preserved)

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
  (`momom2/wesnoth-tier-a`) at **2,670,682**; the best *measured*
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
