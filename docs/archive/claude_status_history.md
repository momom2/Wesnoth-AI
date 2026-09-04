# CLAUDE.md status history (superseded sections, moved 2026-09-04)

Every "Current status" block CLAUDE.md carried from 2026-06-11 to 2026-09-04, verbatim, newest first. Kept for provenance; the current status lives in CLAUDE.md and the plan in docs/plan_20260904.md.

## Current status (2026-09-04): PROJECT ON HIATUS

**User decision 2026-09-04, after the minimal loop's first pin
(raw policy 11-0-29 vs the seed): the project is on hiatus.** All
Vast instances destroyed (2026-09-04 07:25 UTC); nothing is
running or billing. Everything identity-critical is on HF
`momom2/wesnoth-model-checkpoints`: the seed
(`tier-b/imit_tierb_start.pt`), leg checkpoints and histories under
`tier-b/arm_az*_2026090[34]/` (az5: `tier_b_az5.pt`,
`az_history.csv`, `train.log`, `abort_escrow.tar.gz`), and the
earlier legs. The last raw pin's game files were not escrowed
separately; the result and the reading are in
`docs/archive/az_leg_20260903.md`. If the project resumes, start from
`BACKLOG.md` NEXT ACTIONS (the three untested target levers) and
the leg record; the search half of the pin was never measured.

## Current status (2026-09-03, superseded by the hiatus note)

**Restart from a minimal self-play loop (user decision 2026-09-03).**
Everything above the loop is quarantined, not deleted:
`quarantine/INVENTORY.md` lists the 78 mechanisms with evidence
(none has a measured strength benefit; 27 were never isolated).
The loop is `tools/az_loop.py` per `docs/archive/az_minimal_spec.md`: plain
PUCT (32 sims), visit-count targets, result labels, squared-error
value loss on the C51 mean, one step per iteration, signal and time
telemetry on, pins vs the seed every 10 iterations. Launcher
`scripts/az_launch.sh` (tests -> Rust wheel -> smoke -> loop).
Leg record: `docs/archive/az_leg_20260903.md`.

**First finding (measured, `training/metrics/step_scale_20260903/`):**
the seed has no Adam moments, so the first update is lr * sign(g)
on every parameter; it shifts the value level down uniformly through
the trunk (-0.6), and search converts a mover-frame level error b
into an act-vs-end_turn preference (2b), so K collapsed to 1 in one
step. The seed's own +0.2..0.3 optimism is what holds its K 10-12
under search; correcting it (which outcomes do) gives K ~7. Every
earlier leg from the seed paid the same first step (VG2's
"unprotected first iteration" is this). Fix in place:
`tools/step_control.py` backtracking on held-out games; K tripwire
threshold 3.

**Legs az2/az3/az4 (2026-09-03..04, box 49739163, one game per
actor = 19/iteration, ~16 min each):** az2 showed held-out loss
alone accepts a level swing (+0.48 -> -0.38); az3 added a level cap
(2 atoms/step) and held K 7-11 for 8 iterations, but the cap
throttled EVERY step to 1/8-1/16 (the proposal keeps a level
component from momentum), so the policy heads moved by KL ~1e-4
per step -- no policy learning possible. az4 removed the level from
search instead (`MCTSConfig.value_center` = the head's mean value,
subtracted from every value search reads; cap off; full-size steps
accepted when the held-out games are not significantly worse):
steps went through but K fell to 4 -- with the level gone, PUCT
falls back on the priors where end_turn is the single largest
action, and the visit targets teach more end_turn. **The seed's
+0.44 level was pricing the tempo that end_turn hands over.**
**az5 (running, from the az3 iteration-7 checkpoint): centering +
`--tempo-bonus 0.44`** (search sees `V - mean_V + 0.44`;
design_constants.md): K 11-14, 17-20/19-20 decided, full steps,
KL ~0.0015/step, held-out loss improving. **First pin (raw policy
vs the seed's raw policy, 40 games): 11-0-29 -- clearly worse.**
Pre-registered reading: the 32-sim visit targets teach something
the seed's raw policy already did better; levers are the search
budget or the target temperature, not the step rule. The searched
half of the pin reruns after the relaunch (~08:30 UTC 2026-09-04;
columns `raw_vs_seed_wdl` / `search_vs_seed_wdl` in
`/workspace/az_history.csv`, escrowed to HF
`tier-b/arm_az5_20260904/az_history.csv`). Pre-registered
predictions and the record: `docs/archive/az_leg_20260903.md`. Ops: `scripts/az_stop.sh`
(launcher first, then loop, daemons, orphan sweep);
`LAUNCH_SKIP_TESTS=1` for resumes (a flaky old-loop test stopped
the box once). Open: generation throughput varies 2x between
iterations (GPU-bound; tokens/leaf and GC time now logged); the
value head's level drifts under momentum (harmless to search with
centering, watch the MSE).

## Current status (2026-08-25, superseded — kept for provenance)

**TRAINING IS DOWN by explicit user order (2026-08-25); do NOT
resume without their go.** The user intends to oversee the resume
directly. Objective restated by the user in plain terms: **"my
objective is to make the policy play better. Nothing else
matters."** Strength is measured by GAMES (Elo matches), never by
proxies; internal metrics are crash barriers, not verdicts.

**Elo board (catalog `tools/elo_catalog.py show`; lineage-path
naming per docs/checkpoint_naming.md, aliases preserved):** the
**imitation seed `2516k-b-294k-l4-0k` is the strongest checkpoint
ever measured: +211 ± 67**, ABOVE the old 5M champion `2516k`
(+140 ± 29). Legs 3 and 4 each destroyed ~500 Elo from it (leg-4
pin `2516k-b-294k-l4-495k`: −309 ± 105; 0-19 vs its own seed, 0-20
at sims 0).

**Leg-4 postmortem (docs/archive/leg4_erosion_rootcause_20260820.md — the
full arc: workflow synthesis + E1/E2/E3/Q7 + final board):**
killer = `--distill-prior-discount 0.9` alone: under the linear
link the lam-decay flattens ALL ~386 legal actions while the boost
touches <=5, so it is net-flattening PER STEP regardless of grader
quality (computed sharpen_top −0.033 vs measured −0.032). The A1
ruling that justified 0.9 was measured under the EXP link and never
re-derived. Launcher no longer emits it (env-gated; code default
1.0). Grader exonerated on every axis (ICC 0.96-0.995, blind 8%->0
on own games). Fog-frame flaw real but not the killer:
`--turn-boundary-frame mover` shipped (bake-off: accept 0.90 vs
0.67, robust verdicts; ADOPTED for leg 5+). Gate projection
REJECTED as implemented (separation 1.0:1; mechanism validated —
depth-1 removes the measured +3.7-atom pass bias — parked pending
variance reduction).

**Leg 5 (tier_b_l5, from the seed, lam=1.0 + mover frame +
K-tripwire + policy anchor):** ran 8 iterations / 122,231 steps
(checkpoint 2,931,890 ESCROWED at tier-b/tier_b_l5.pt) with the
healthiest policy-side telemetry ever (CE flat, attack% rising, K
12, 24/24 decisive, blind_coord 2.3%, boundary_pairs live) — then
the value tripwire killed it. Diagnosis
(docs/archive/leg5_value_inversion_20260825.md, workflow RAN the decisive
experiments): the value head's below-chance reading on human games
is a TRUNK proxy rotation onto raw unit count (head grafts move
<0.01; trunk swaps move everything; count anti-predicts human
winners while material predicts 0.89 — in self-play both
coincide, so the direction is unidentified and lands coin-flip:
leg 4 rotated the GOOD way, 0.912). AND the probe instrument was
unsound: 1,200 pairs = ~3 games, invalid CI. Whether the rotation
transient matters for STRENGTH is unmeasured — the leg was killed
before the pre-registered 100k-step Elo-vs-seed gate.

**Instrument + tripwire state (user rulings 2026-08-25, all
shipped):** probe stratified (8 pairs/game seeded-random across
~150 games, per-game stats, BETWEEN-game SE; seeded baseline ~0.85
± 0.03, opening-only sampling reads lower); CE abort REMOVED
("learning to play better does not necessarily mean learning how
humans play" — CE is telemetry only); value tripwire = user's
design: reading < 0.60 -> up to 3 INDEPENDENT sample redraws,
abort only if all fail; K-collapse tripwire `--abort-k-median 10`
(pass explicitly); decisive-rate, CPU watchdog, holdout-stall
unchanged. No automatic money cap exists.

**The agreed resume plan (awaiting user):** fresh non-VM box
(vms_enabled=false — VM-class hosts refuse ssh keys, 4 rentals
lost to it), resume from escrow via configs/leg_l5.json (already
points at the rolling file), train to ~250k+ steps, then a
40-game Elo match vs the seed is THE verdict: improved -> continue;
else stop. Launcher gap: the qualify gate is still run BY HAND
(BACKLOG). Boxes 48108334/48607224 exist stopped (storage only).

## Current status (2026-08-14, superseded — kept for provenance)

**Turn-Commitment Search (TCS) is the production data generator,
DEFAULT ON.** The 2026-08-11..12 handoff legs (A1 prior-discount,
F1 policy-anchor arms) were stopped and discarded by user ruling
after the human-holdout probe showed erosion and evals showed no
external movement; the project pivoted to novel-algorithm research
(three Opus-workflow tracks). Outcome: **TCS integrated**
(`docs/archive/tcs_spec.md` — plan complete side-turns by counterfactual
coordinate refinement, graded at turn boundaries; validated by a
300-state probe: revalidated accept 0.64, median accepted Δ ≈ 2 C51
atoms, placebo-separated; `tools/turn_search.py` +
`tools/turn_policy.py::TurnCommitPolicy`, trainer untouched,
`--no-turn-search` opt-out), **GBC integrated as the
event-supervision auxiliary, DEFAULT ON** (`wesnoth_ai/gbc.py` +
`docs/archive/gbc_spec.md`: after the 0d attribution test showed events
predict outcomes at AUC 0.79 while the value head's turn-scale
movement is noise at 0.53, GBC's role became value-head repair —
fog-censored dies/flips labels attached in finalize_game, BCE
through the trainer at `--gbc-coef 0.1`), **ITS parked**
(`docs/archive/planning_abstractions_litreview_20260812.md`). Probe
side-finding that re-motivates TCS: the imitation seed plays
K≈12 actions/turn on ladder while the self-play-trained F1 policy
plays K median 2–4.5 — turn truncation was ACQUIRED during
self-play. TCS leg 1 (tier_b_tcs) ran 7 iterations and was killed
by the pre-registered probe-abort tripwire (monotone human-CE
erosion; turn structure and value learning were healthy) — leg 2
adds the F1 policy anchor; config in BACKLOG item 0. Leg 3
(2026-08-16..17) was PAUSED after turn length collapsed back to K
median 2 with draws at 0.68–0.75 while the CE probe held (the
probe is blind to search-driven passivity): boundary-only grading
let the climb exploit the value head's tempo blindness through the
force-included end_turn alternative. Counter shipped 2026-08-17:
**multi-turn projection** (`--turn-project reval|all` +
`--turn-project-halfturns`, DEFAULT OFF; generalizes the reply
arm; tcs_spec.md §3 addendum) grades candidate turns H closed-loop
half-turns past the boundary at linear cost, and TCS planning
telemetry now rides the distill drain (it was dark on every path
during the collapse). Leg-4 candidate: pre-slide escrow restart +
`--turn-project reval`; details in BACKLOG item 0.

## Current status (2026-08-11, superseded — kept for provenance)

**The tier-b self-play handoff leg is RUNNING (Vast on-demand 4090,
instance 47452778, ~$0.37/h) from `imit_tierb_start.pt` (the rescued
imitation checkpoint, aux heads + flags stripped, step 2,809,659
carried).** Launch config = the executed 2026-08-10 technique review
(all A/D/F/X rulings shipped; `docs/archive/technique_review_20260810.md`):
A1 `--distill-prior-discount 0.9`, A2 value anchor, A3 turn-cap
jitter 60-100, A5 tripwires (decisive 0.35/20, FLOOR-RELATIVE stall
60), A6 fork-guard smoke gate, F3 actor-pool topology (sized from
cgroup quota — nproc is HOST-wide on Vast), mini-ratio 0. Everything
identity-critical escrows to HF every 30 min (checkpoint+sidecar,
telemetry, hourly human-holdout probe CSV, games log); a fresh box
reseeds itself from escrow unattended (proven 3x, incl. one
cross-machine migration).

**First measured results (protocol-matched hourly probe,
`scripts/holdout_probe_loop.py`, t0 = CE 3.207 / value-AUC 0.627 on
the imitation manifest holdout):**
- **A1 verdict: PLATEAU, not washout** — CE 3.648 (+26k steps) →
  3.782 (+58k) → 3.713 (+85k). The handoff costs ~+0.5 nats of
  human-play CE, then stabilizes; the pre-registered "flat" bar
  failed but the washout the literature scan feared does not occur.
  Leg-2 comparison arm (F1 policy-head anchor,
  `tools/policy_anchor.py`) is wired, default OFF.
- **Value head healthy**: AUC dipped to 0.417 in the handoff shock,
  recovered to 0.76-0.79 (> t0) under the A2 anchor.
- **Self-play is hyper-decisive**: 23-24/24 decisive per iteration
  (leader kills ~turn 15), ~8.5k decision-steps/hour. The F2
  would-fire analysis over the first 112 games: ZERO stalemate-rule
  fires at any K — the no-progress rule is priced irrelevant for
  this regime.
- Launch-day ops lessons are encoded: actor-pool smoke test (slow
  tier), OOM fixes (train chunk 32, pool fuse 16), stall watchdog +
  supervisor marker protocol, holdout-sidecar carry.

Next: let the leg run (~24-48h), then leg-end evals (Elo vs the
imitation seed; RCA probe on a cheap CPU box) and the multi-epoch /
leg-2 decisions. BACKLOG NEXT ACTIONS is current.

## Current status (2026-08-08, superseded — kept for provenance)

**The human-replay corpus is CERTIFIED 100% bit-exact (24,796/24,796)
and the imitation-learning phase is running.** The 2026-08-06..07
fidelity grind (viewer-ledger process with the user + an automated
engine-OOS harness validated on 10/10 clean controls) closed every
residual divergence: five sim/extractor root causes fixed
(tentacle rest-heal `a41b059`, chatter-window + pattern-A trailer
`d92f949`, pickadvance-recruit + [object]-through-advancement
`422675c`), 57 corrupt recordings deleted engine-verified,
39 empty saves deleted, 6 Dunefolk-fielded quarantined, 1 parked
(`setaside_pickadvance_force` — pick_advance's force-mode RANDOM
narrowing on ignored dialogs; docs/wesnoth_rules.md has the entry).
Ledger: `training/logs/replay_dispositions.jsonl.gz`.

**Imitation dataset + pipeline (commit `7dfe44b`, config-first):**
preprocessing quarantined short (<5 turns, 3,644), duplicates
(save-chain dedup by stream-prefix identity, 276) and uninformative
(<10 attacks or <3 kills, 1,509) games → **training pool 19,367;
17,124 with explicit winners = 2.57M winner-side pairs** in
`replays_dataset_imitation/` (build: `tools/build_imitation_dataset.py`;
outcome labels: `training/logs/replay_outcomes.jsonl.gz`). Trainer:
`supervised_train.py --imitation-config configs/imitation.json` =
winners-only policy CE + per-game equal weighting + both-sides ±1
value supervision + manifest holdout (369 games).

**Imitation A/B (HF `tier-b/imitation_ab_20260808/`): warm-start wins
policy, fresh head wins value.** One epoch each, 15M net, 4090:
seeded holdout CE **3.107** / actor@1 56.6% vs random 3.449 / 54.4%
(seeded better at every matched pair count); but value AUC: fresh
head **0.951** stable vs warm head 0.538 oscillating. Verdict wired
as `--reinit-value-head` (`49df952`). Caveat: the seeded arm ran
pre-instrumentation code and silently skipped ~21% of files; the
rerun below is the clean version. `bfa96e7` added per-epoch
accounting (`files_seen/file_errors/pairs`) so underruns are audible.

**Tier-b imitation checkpoint (2026-08-10): rescued at 94%, escrowed.**
The clean run (warm trunk+policy + `--reinit-value-head`) HUNG
silently at 2.368M/2.515M pairs (no traceback; suspect: parallel-
stream worker teardown near end of file list — open BACKLOG item)
and billed ~2 idle days before discovery (laptop Modern-Standby
sleep killed the supervising session; see memory
`session-bound-watchers-die`). Rescued periodic checkpoint = HF
`tier-b/imitation_ab_20260808/imit_tierb_rescued_2368k.pt`:
holdout **CE 3.102 — statistically indistinguishable from the
seeded arm's 3.107** (23/39 matched evals ahead, mean −0.002 vs
~0.04 adjacent-eval jitter; the supported claim is EQUALITY: the
fresh value head cost the policy nothing). actor@1 0.545, value
AUC 0.69 still climbing at cut. Curves:
`training/metrics/imitation_15m/imitation_curves.html`.
All boxes stopped; credit ~$3.

**Open training-design questions:** winners-only is the default;
outcome-conditioning rejected for now (architecture identity with
self-play preserved). Next after the checkpoint lands: evaluate it
(vs RCA + probe metrics), then decide the imitation→self-play
handoff for tier-b.

## Current status (2026-07-31, superseded — kept for provenance)

**A 72h autonomous run (Claude + Fable, 2026-07-28..31) fixed four
measurable defects in the learning signal. The lineage now provably
improves against its own past; the external gap is unchanged.**
82 commits; suite **633 fast + 11 slow, green**. Full provenance:
`docs/archive/autonomous_run.md` (cycle log, newest first).
**Read `BACKLOG.md`'s "NEXT ACTIONS" block first — it is the short list.**

**Measured performance (both numbers matter, do not quote one alone):**
- **In-lineage: +133 ±57 Elo** — 2,515,896 vs the 2,290,529 anchor, 340
  games, one joint Bradley-Terry fit, prediction pre-registered. A
  transitive check (direct +146 vs chained +134 against a ±145 band)
  rules out "beats its own parent while going nowhere". The 2026-07-28
  regression is recovered and surpassed *within-lineage*.
- **External: 0-0-30 vs the built-in RCA AI**, median leader death turn
  10, 15 games per side. Re-confirms the founding number at 3.3x the
  sample. Four fixes and +133 moved it **not at all**.
- Honest summary: *"the signal was broken and is now fixed"*, NOT *"the
  policy got good"*. Note the +133 was measured WITH search and the RCA
  eval is RAW policy — different objects; never merge them.

**What was actually fixed** (all with fail-before/pass-after tests):
- `fa95da5` — a search-imagined village capture permanently rewrote the
  REAL game's encoder input (`Map.__deepcopy__` aliases hexes). Stored
  transitions are re-encoded at train time, so a state at turn 3 showed
  ownership for villages captured by turn 30: **the old encoding leaked
  the future into training inputs.**
- `933888d` — search forks latched `first_time_only` scenario events, so
  Aethermaw's morphs never fired in live games (walls never opened).
- `a21030c` — terrain events stripped overlays from `_terrain_codes`, so
  impassable whirlpool walls were priced walkable; produced
  engine-verified OOS ("found corrupt movement in replay").
- `8b68a25` — nine `tools/` modules were imported both as `tools.X` and
  bare `X` (two module objects, duplicated module state), including a
  bug that **cannot exist single-flavour**.
- Guard: `SIM_FORK_GUARD=1` (`deep_state_fingerprint`) catches that whole
  class — all three instances were invisible to `state_key`. Free when off.

**Diagnoses that closed old questions:** nothing in the training signal
pays for banking gold — the shaping-reward path is **structurally inert
under `--mcts`** (`MCTSPolicy.observe` is a no-op), which finally explains
why the old `weight_gold=0` fix did nothing. The real mechanism is a
"tried-and-cut tax" in Gumbel target extraction (edges sampled and cut
grade below `v_mix`; never-sampled mass shelters at it). Also measured:
the **detector-advice channel carries no information** (placebo control),
and raising `--mcts-sims` is the WRONG lever — the Gumbel target
*concentrates* rather than converges, with coverage fixed at `m`=16
visited edges at every N.

**Compute state:** Vast credit **$0**, box stopped/exited, nothing
running. Last campaign checkpoint **decision_step 2,670,682** on HF
(`momom2/wesnoth-model-checkpoints`, `tier-a/tier_a_campaign.pt`); best *measured* is
**2,515,896** (`training/checkpoints/campaign_live_20260730.pt`).
Throughput was ~4,000-7,000 decision-steps/hour. Weigh the cost per
step before buying more of the same compute.

## Current status (2026-07-02, superseded — kept for provenance)

**Kaggle pre-flight DONE (2026-07-02): Phase 0 executed, repo made
self-contained, a warm-start-corrupting vocab bug found+fixed; next
action = USER creates a Kaggle account (phone-verify) and runs
`kaggle/tier_a_phase1.ipynb` (Phase 1), then Vast.ai Phase 2.**
Highlights (details in **BACKLOG.md §2026-07-02**, suite **370 passed**):
- 🔴 `load_checkpoint` rebind orphaned the shared trainer/inference
  vocab dicts → after ANY warm-start, MCTS rollouts ran with an empty
  vocab and scrambled unit-type embeddings (affected all resumes since
  2026-06-29; smoke runs only). Fixed in-place + regression test.
- `training/checkpoints/tier_a_5m.pt` (5.02M, vocab+step carried) is
  COMMITTED — Phase 0 done; don't re-grow.
- The sim's runtime WML subset (`wesnoth_src/data/multiplayer/*`,
  Mini Maps, ~840KB) is now TRACKED; a bare `git clone` trains
  (verified from a scratch index export). Do NOT pip-install
  requirements.txt on Kaggle (Windows-only torch-directml pin).

The 2026-07-01 deep review returned **GO for the default in-process
`--mcts` path** (reward core sound; itemised in BACKLOG.md §2026-07-01);
its pre-flight fixes are on `origin/main` (`8468c5b`, `c83a3dd`,
`f706400`/`c6b314c`).

What a new instance MUST know:
- **The go-forward path is the Tier-a runbook: `docs/archive/tier_a_runbook.md`**
  (decisions locked in `docs/archive/superhuman_training_plan.md` §10). Phase 0
  grow (done-able locally) → Phase 1 Kaggle free T4×2 (pipeline test +
  profile) → Phase 2 Vast.ai spot 4090 on a ≥32-vCPU host (~$30
  calibration run). Objective is a single Elo-vs-compute point, NOT a
  strong model, before any Tier-b spend.
- **Tier-a net = 5.0M params** (`--d-model 256 --num-layers 6
  --num-heads 8 --d-ff 1024`), Net2Net-grown from the 471K checkpoint via
  `tools/net2net.py` (measured value MAE ~0.017 — the grow preserves the
  trained value head well enough to warm-start). Grow output =
  `training/checkpoints/tier_a_5m.pt`.
- **Fresh campaign uses `--reset-decision-step`** (weights-only init;
  combat-oracle anneal restarts at full strength). **OMIT it on a
  spot-preemption resume** or the anneal restarts mid-run.
- **Pre-flight fixes that change how you run it:** checkpoint save is now
  atomic (`.tmp`+`os.replace`+rolling `.bak`) and resume falls back to
  `.bak`, so same-path `--checkpoint-in`==`--checkpoint-out` on spot is
  safe; `--mcts-batch-size` now defaults device-aware (B=16 on CUDA);
  `configs/reward_selfplay.json damage_dealt` corrected 0.005→0.0005;
  `--actor-pool` un-broken (was crashing → 0 experiences) if you reach
  for it to feed the GPU.
- **Deferred CUDA-only perf (profile-first on the GPU node):** the
  in-process rollout's per-leaf `.item()`/`.tolist()` on GPU tensors is
  the biggest expected stall (fix = adopt the actor-pool's
  forward-on-GPU/sampler-on-CPU split); plus B2 (per-leaf value/cliffness
  batch read) and B3 (pinned H2D, now impl'd on `main`). Spec in
  `docs/gpu_perf_patches.md`; profile with `tools/profile_rollout.py`.

**Free-compute question (RESOLVED, see plan §10 / the runbook's cost
section):** startup credit programs (MS Founders Hub / AWS Activate /
Google / NVIDIA Inception) all need a registered company + website — NOT
viable for a solo unincorporated individual. Google TPU Research Cloud is
the only large free grant open to individuals but a poor fit (needs a
`torch_xla` port and its low-vCPU driver would starve the CPU-bound
rollout). So the plan is **Kaggle free T4×2 (Phase 1) → Vast.ai spot 4090
~$30 (Phase 2)**, with small free smoke credits available (Modal ~$30/mo,
RunPod ~$5). A final background re-run of this research was lost when the
process exited, but its conclusions were already captured — no need to
redo it unless chasing a specific new grant.

## Current status (2026-06-11, superseded — kept for provenance)

**Training is local-only.** The ENSTA Mesogip cluster is permanently
inaccessible (2026-06); all SLURM/sync/GUI infrastructure was removed
(history: `cluster/` up to commit 8a31ea1, plus the
`recovery-snapshot` branch). Reward/weight configs moved to
`configs/`. The project was recovered onto a new machine 2026-06-11;
`replays_raw/` and `replays_dataset/` did not survive and must be
re-downloaded via `tools/download_replays.py` + re-extracted
(~21 tests skip until then).

**The simulator is the production training path.** `tools/wesnoth_sim.py`
is a pure-Python headless reimplementation of Wesnoth 1.18.4's game
logic, ~1000× faster than driving a Wesnoth
subprocess. Combat math is bit-exact verified (731/731 strikes
matched against `[mp_checkup]` oracle on strict-sync replays);
full-replay reconstruction at **99.93% clean (5,482/5,486)** on the
re-extracted competitive-2p corpus (2026-05-11 sweep; the oft-quoted
98.57% is the stale May-4 snapshot). 4 residual divergences remain
un-root-caused (3 classes: recruit gold drift, defender_missing,
src_missing) — parked 🟡 in the 2026-05-11 BACKLOG (commit 931b574),
still open.

**Self-play training is end-to-end ready.** `tools/sim_self_play.py`
drives self-play in the simulator with either REINFORCE or MCTS
(`--mcts` flag, AlphaZero-style PUCT + virtual loss + transposition
table). Auto-resumes from `sim_selfplay.pt` or warm-starts from the
highest `supervised_epoch*.pt`. Checkpoints through 2026-05-21
(latest at decision_step 5.68M) live in `training/checkpoints/`.

**Distributional value head + cliffness signal landed 2026-05-10.**
C51 head (K=51 atoms, `[V_MIN, V_MAX] = [-1, +1]`) replaces the
prior tanh-bounded scalar value. `output.cliffness = std(Z(s))`
exposes the network's per-state value uncertainty. Two MCTS
consumers are wired but default OFF pending calibration: Bayesian-
precision bootstrap weighting in `_backup`, and adaptive sim budget
based on root cliffness. Always-on root-cliffness debug log
collects distributions for tuning.

**The live-Wesnoth IPC bridge is now eval-only.** `game_manager.py`
and `main.py --display` were retired 2026-05-11; `tools/sim_demo_game.py`
+ `tools/sim_to_replay.export_replay_from_scratch` cover the demo
case by exporting a Wesnoth-loadable `.bz2`. The only remaining live-
Wesnoth consumer is `tools/eval_vs_builtin.py` (via `tools/eval_runner.py`
+ `wesnoth_ai/wesnoth_interface.py`), which needs real Wesnoth subprocesses to
pit the trained model against the built-in RCA AI.

