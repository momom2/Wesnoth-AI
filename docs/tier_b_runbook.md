# Tier-b campaign runbook (2026-08-02)

Go-forward path for the Tier-b net, decided in this order: **grow →
recover → campaign → measure.** The middle step is not optional and is
the thing most likely to be skipped under time pressure.

Companion to `docs/tier_a_runbook.md` (the Tier-a mechanics — bidding,
preemption, tripwires, eval protocol — are unchanged and not repeated
here). Arch evidence: `docs/superhuman_training_plan.md` §11.

---

## 0. What you are buying, stated up front

Tier-a's exit criteria were **not** met: the gate was ≥90% vs the
built-in RCA AI with non-zero leaderkills on ladder maps; the measured
result is **0-0-30 with median leader death on turn 10**. The lineage
does improve against its own past (+133 ±57 Elo in-lineage, with
search) but has never converted that into an external gain.

Tier-b is therefore a **capacity experiment run on an unproven signal**,
launched by explicit user decision (2026-08-02) with that understood.
The plan's own Phase-2 gate ("only after Phase 1 shows the pipeline
learns *and* the bigger net helps") is being deliberately skipped, not
overlooked. Budget accordingly and read §5's stopping rules.

---

## 1. Architecture — LOCKED: `d_model=384, layers=8, heads=12, d_ff=1536`

**15.55M params**, mid the plan's 10–30M Tier-b band.

- **heads = 12 is load-bearing, not cosmetic.** It keeps `d_head = 32`,
  matching the source. Net2Net's per-Q/K/V leading-block copy is
  head-aligned only when d_head is unchanged; at heads=8 (d_head 48)
  the same grow measures **0.339 vs 0.226** value MAE. Never set
  `--num-heads 8` at d384, or 8 at d512 — §11 has the table.
  **The §3.2 plan table is wrong on this point.**
- 27.62M (512/L8/H16) was the alternative. Rejected on **throughput,
  not warm-start**: its warm-start damage is within 10% (0.248 vs
  0.226) while its forward cost is ~1.7x higher, and forward is ~91%
  of decision cost. At a fixed budget the smaller net buys materially
  more steps, and steps are what this experiment is short of.

## 2. Phase 0 — grow, locally, once

```bash
python tools/net2net.py --in training/checkpoints/campaign_live_20260730.pt --out training/checkpoints/tier_b_15m.pt --d-model 384 --num-layers 8 --num-heads 12 --d-ff 1536
```

Grow from `campaign_live_20260730.pt` (**decision_step 2,515,896** —
the best *measured* checkpoint), NOT from `tier_a_campaign.pt`, whose
local copy is stale at 155,464 and whose HF copy (2,670,682) is newer
but unmeasured. **Verify by reading `decision_step`, never by
filename** — that trap has fired three times.

`net2net.py` carries `aux_score` / `moves_left` / `advice` /
`relevant_set_hexes` from the source and **refuses to write a
checkpoint that would discard trained tensors**. Before 2026-08-02 it
silently dropped the advice and moves-left heads (14 trained tensors)
because the transfer walks the *destination's* params.

Then confirm the grow is what you think it is:

```bash
python tools/measure_warm_start.py --source training/checkpoints/campaign_live_20260730.pt --arch 384,8,12,1536 --states 32 --stride 7 --max-decisions 25
```

Expect ~0.226 and `NOT-A-WARM-START`. That is the **known, accepted**
starting condition for §3 — not a reason to stop. Stopping conditions
are a number far from 0.226, or `head_aligned: false`.

## 3. Phase 1 — the RECOVERY leg (the step that gets skipped)

The grown net is **13x** past the acceptance precedent (0.017) and past
the bar at which T2-C's encoding change was rejected (0.217). So:

> **No strength number from this net means anything until holdout CE
> has returned to its pre-grow level.** Not an Elo run, not an RCA eval,
> not a "quick sanity game". A comparison against the Tier-a lineage
> before recovery measures grow damage, not capacity.

- Gate on **holdout value CE / `fresh_value_ce` recovering to its
  flag-OFF, pre-grow level**, read **floor-relative** (CE − floor; raw
  CE moves with the label mix), **skipping iteration-0-after-restart**.
- Gate on the curve, **not on an iteration count**.
### The pre-grow baseline — RECORDED 2026-08-03

Measured from the Tier-a campaign's `trainer_history_local.csv` (pulled
from HF; 41 logged iterations, `holdout_n` 526 throughout). **This is
the number the recovery leg must return to.**

| metric | last 10 iters | last 20 iters |
|---|---|---|
| **`fresh_value_ce` − floor** | **−0.228** (sd 0.138) | −0.277 (sd 0.156) |
| `holdout_value_loss` | 1.396 (sd 0.330) | 1.486 (sd 0.480) |

**Use floor-relative CE as the gate, and use a trailing mean.** Raw
`fresh_value_ce` moves with the label mix, so only CE − floor is
comparable across legs. And a single iteration is worthless here: the
floor-relative value swings between −0.02 and −0.56 from iteration to
iteration (sd 0.14–0.16), so **any recovery claim needs a trailing mean
over ≥10 iterations**, not a good-looking row. Concretely: recovered
means the trailing-10 mean of (CE − floor) is back at roughly **−0.23**,
i.e. comfortably negative, not merely improving.

Two caveats that must travel with these numbers:
- The step series is **non-monotonic** near 2,534,542 → 2,529,503 — a
  resume. Treat the CSV as concatenated legs, not one continuous curve.
- `holdout_value_loss` is the noisier of the two (range 0.71–2.56 across
  the run) and its basis can reset across boxes. It is a secondary
  read; the floor-relative CE is the gate.

- The raw CSV is **gitignored** (`.gitignore:105`, it is regenerable).
  Re-pull with `hf_hub_download("momom2/wesnoth-model-checkpoints",
  "trainer_history_local.csv")` to recompute rather than trust these.

### Drift, not degradation — the gate on Tier-b spend, 2026-08-03

A value-head AUC regression was measured across the +133 interval on
**human-corpus** states (2,290,529 → 2,515,896: open 0.713→0.636, mid
0.784→0.727, late 0.683→0.609). Extended to 160 games with a
game-cluster bootstrap it is **probably real but not decisive**: pooled
**−0.078**, CI95 [−0.153, +0.001], and a clean out-of-sample replication
on the 120 games the hypothesis was *not* formed on (−0.078, one-sided
p≈0.049). A within-game check (−0.068) excludes a cross-game calibration
artifact. Alongside it, an unambiguous **C51 confidence collapse**:
Z-entropy 0.591→0.505 (CI excludes 0), max-atom mass 0.669→0.835, late
game averaging **0.905 on a single atom**.

That raised a real gate: if the training signal degrades its own
evaluator while climbing in-lineage Elo, scaling capacity inherits the
pathology and signal work must precede Tier-b spend.

**The discriminator says drift.** In-distribution value telemetry over
the same campaign CSV moved the *other* way, and more strongly:

| metric (floor-relative; more negative = better) | first half | last half | delta |
|---|---|---|---|
| `fresh_value_ce` − floor | −0.087 | −0.285 | **−0.198** (t=−3.62) |
| `fresh_decisive_ce` − floor | −0.160 | −0.407 | **−0.246** (t=−3.56) |

In-distribution value learning **improved**, at roughly twice the
t-statistic of the human-corpus decline. Improving on self-play states
while declining on human states is the signature of **distribution
drift**, not a degrading evaluator — so **the capacity argument for
Tier-b survives** and this is not a reason to halt the scale-up.

Three caveats that must travel with that conclusion:
- **Coverage is imperfect.** The CSV spans 2,406,235→2,670,682, so it
  misses the first 115,706 steps of the +133 interval and extends
  154,786 past it. It is not a clean overlay of the same interval.
- Iterations are **autocorrelated**, so the t-values overstate
  significance; read them as "large relative to spread", not as p.
- The series is non-monotonic across a resume, so first/last half is
  approximate time ordering.

The collapse finding stands regardless of drift-vs-degradation, and is
an independent argument against **value-side** self-distillation: a head
already sharpening toward delta-spikes would take a bootstrapped target
faster than terminal z could correct it.
- If holdout CE plateaus above the pre-grow level for a long window,
  that is the real result: the capacity did not pay for its own
  warm-start damage. Report it; do not keep buying hours.

## 4. Phase 2 — campaign

Reuse the Tier-a Phase-2 machinery unchanged. `scripts/vast_onstart.sh`
is now arch- and identity-parameterized (2026-08-02); set these at
instance-create time:

```
-e D_MODEL=384 -e NUM_LAYERS=8 -e NUM_HEADS=12 -e D_FF=1536
-e CAMPAIGN_FILE=tier_b_campaign.pt
-e SEED_CKPT=training/checkpoints/tier_b_15m.pt
-e HF_TOKEN=hf_...
-e SPOOL_WORKERS=64
```

- **`CAMPAIGN_FILE` is the run's identity** — it names the local rolling
  checkpoint *and* the HF escrow object. `tier_a_campaign.pt` is
  RESERVED for the Tier-a lineage; leaving the default would roll
  Tier-b weights forward over that escrow. `hf_upload_loop.py` reads the
  same variable so the two cannot drift.
- The arch vars must agree with the seed checkpoint's `arch` —
  `sim_self_play` raises on mismatch, which is the guard that catches a
  half-applied override.
- **VRAM is a measured quantity, not a predicted one.** The 5M learner
  alone held 11.63 GiB and OOM'd a 12GB card twice in ~26h. Activation
  memory scales with d_model × layers, so 15.55M plausibly lands near
  ~23 GiB — close enough to 24GB that a 48GB card removes the question
  rather than answering it. Run one smoke iteration and read actual
  peak VRAM before committing to a long run; lower `--replay-minibatch`
  if it is tight.
- Consider `SIM_FORK_GUARD=1` for exactly one smoke iteration at
  campaign start — it catches the fork-aliasing bug class and is free
  when off.
- Sizing: `SPOOL_WORKERS` deepens the replay buffer; `GAMES_PER_ITER`
  sets trainer cadence and defaults to min(workers, 24). On a many-core
  host raise workers, leave cadence near the default, and re-tune on
  the node — the laptop sweep does not carry over.

## 4b. Host requirements — derived from actual failures, not preference

Every line cites the incident that motivates it. An offer failing any of
these is excluded outright; these are crash causes.

| Requirement | Why (incident) |
|---|---|
| **VRAM ≥ 40 GB** | The 5M learner held **11.63 GiB of an 11.64 GiB card** and OOM'd twice in ~26h (`autonomous_run.md` cycle 49); the earlier 24 GiB 4090 was fine. Tier-b scales activations ~2.0x → **~23 GiB**, so a 24 GB card reproduces that 100%-utilization failure exactly. |
| **`gpu_frac` == 1.0** | A shared GPU means a co-tenant's spike OOMs us — same crash, someone else's cause, undebuggable from our logs. |
| **RAM ≥ 128 GB** | Replay buffer holds `--replay-capacity 24000` deep-copied GameStates in host RAM (`vast_onstart.sh`); Tier-a wanted ≥32 GB at capacity 6000. |
| **duration ≥ 30 days** | Host 18135 went away mid-campaign. Detectable increments need hundreds of box-hours. |
| **reliability ≥ 0.98** | Same. |
| **disk ≥ 60 GB**, **inet_up ≥ 100 Mbps** | 63 MB checkpoint escrowed every 30 min; the 2026-07-19 balance exhaustion nearly stranded checkpoints on an unreachable disk. |
| **CUDA ≥ 12.0, Python ≥ 3.11** | Already hard-checked in `vast_onstart.sh`, including a real matmul (an arch-mismatched wheel passes `is_available()` and fails hours later at the first kernel). |

**Not a crash risk, but budget for it:** `SpoolDir.collect`'s deadline is a
hardcoded 3600 s (`sim_self_play.py:1266`) with no CLI flag. On slower
cores with a 2.6x heavier net, an iteration collects *fewer* games and
logs `spool collect timed out with N/M` — already seen on the 3060. It
degrades gracefully; it does not crash.

**The VRAM number above is an estimate, and must not be trusted.** Run a
bounded smoke and read actual peak VRAM before committing to the long
run. `PYTORCH_ALLOC_CONF=expandable_segments:True` and a lower
`--replay-minibatch` are the tuning levers.

## 4c. Campaign box selection — measured, 2026-08-06

The §4b host requirements were derived for FULL-BOARD Tier-b and are
superseded for the relevant-set lineage by direct measurement:

- **Shape: spool (many CPU cores) + any modest GPU.** The learner
  peaks at **2.4 GiB** VRAM at the measured-safe 64/32 batches; the
  actor-pool loses to spool on consumer cards (flat batching curve,
  measured on the 3060 at every batch size). Cores/$ is the metric.
- **The incumbent 192-core + RTX 3060 box (~$0.20/h) beat every
  2026-08-06 market offer on cores/$** (best alternates: 80-core
  3060 @ $0.234, 64-core 3090 @ $0.148, 128-core 4060Ti @ $0.348).
  Recommendation: campaign on the incumbent box class; re-run
  `tools/box_bench.py --checkpoint <t2b ckpt>` on any candidate
  before committing (2-minute answer, idle box only).
- **RAM**: 76 workers ran in <60 GiB; 63 GiB hosts are adequate,
  252 GiB is waste.

Launch checklist (supersedes ad-hoc launchers; all from standing
rulings): seed = the recovery leg's output (verify by
decision_step), `--save-every 1`, playout-cap default ON,
`--distill-prior-discount 0.9`, `--relevant-set-hexes`,
train-draw-tiebreak OFF, batches 64/32 (12 GiB) — scale up only
after a measured peak on the actual card, abort tripwires ON, arm
`watchdog.sh`, campaign identity = a DISTINCT rolling name
(`t2b_campaign.pt`) + HF escrow under that name only, never the
Tier-a rolling name. Iteration budget: at the leg's measured
~15-20 min/iteration (~5.5k decisions/iter), a 450k-step detectable
increment costs ~80-100 box-hours (~$16-20 on the incumbent) — set
campaign length in iterations from that, not wall-clock.

## 5. What to measure, and when to stop

Measure exactly as Tier-a did (`docs/tier_a_runbook.md` "What to
measure"): **ladder maps only** for Elo, holdout CE for value learning,
throughput on that GPU compared only to other GPU runs.

Two numbers must travel together in any report, and must never be
merged: **in-lineage Elo (measured WITH search)** and **the external
RCA anchor (RAW policy)**. They are different objects.

**Pre-register these before the first hour is billed:**

1. **Recovery**: holdout CE returns to the pre-grow level. If it does
   not, stop — nothing downstream is interpretable.
2. **Beats its own seed**: the recovered 15.55M net vs
   `campaign_live_20260730.pt`, ladder maps, one joint Bradley-Terry
   fit. This is the *only* question Tier-b was bought to answer.
3. **External**: RCA anchor re-run. Expect no movement — four fixes and
   +133 in-lineage moved it **not at all**. Treat any movement as a
   result requiring confirmation, not as vindication.

**Cost reality, so the stopping rule has teeth.** Detectable step gaps
on this lineage are 450k–1M decision-steps. The 5M net ran
4,000–7,000 steps/hour; a 15.55M net's forward pass costs ~2.6x more,
though a higher-core host partly offsets it. Expect **hundreds of
box-hours per measurable increment** and decide in advance how many
increments you are buying.

## 6. Hazards specific to this tier

### 6.0 Hazards measured 2026-08-05/06 (the first 15M legs)

- **`--save-every` defaults to 10 and saves happen ONLY at launch,
  every Nth iteration, and clean exit.** A crash before iteration N
  loses EVERYTHING since launch -- the 2026-08-05 overnight OOM at
  iteration ~6 lost the whole night while the in-memory counter (and
  the CSV) showed healthy progress. **Pass `--save-every 1` on every
  paid run**; the save costs ~1s against 10-20min iterations.
- **15M on a 12GB card: `--replay-minibatch 64 --train-batch-size 32`
  is the measured-safe pair** (2.4 GiB backward peak). The 5M-era
  128/64 OOMs the backward pass at 15M -- that exact carry-over
  caused the overnight crash.
- **Verify resumes by decision_step AND checkpoint mtime, never by
  loss levels.** Post-crash loss values are incomparable across
  holdout/replay-buffer regimes and false-confirmed a "continuation"
  that was actually a from-seed restart (2026-08-06).
- **Arm the on-box watchdog** (`watchdog.sh` pattern: relaunch on
  learner death, max-restart budget, RUN_COMPLETE/WATCHDOG_GAVE_UP
  beacons). Laptop-side monitors die with the operator's session --
  the OOM night went 8h unnoticed. The box cannot self-stop (no API
  key on rented hardware, by design); instance-stop decisions stay
  operator-side.
- **Seeding lineage (user rulings 2026-08-05):** the Tier-b working
  seed is the RE-GROW of the recovered T2-5M relevant-set checkpoint
  (t2_recovery -> t2b_15m_seed, d_head 32 head-aligned, grow-gate
  MAE 0.2399), NOT the full-board tier_b_15m.pt. All training runs
  the 15M net; playout-cap ON; lambda=0.9; train-draw-tiebreak OFF.

- **`tier_a_campaign.pt` is a reserved name.** See §4.
- **Do not compare a fresh grow to anything.** See §3.
- **Head alignment**: any future grow keeps `d_head = 32`
  (`num_heads = d_model / 32`).
- **The grow's damage is roughly additive across width and depth**
  (§11), so a further grow from Tier-b compounds. If a Tier-c step ever
  happens, do it progressively (width, recover, depth) rather than in
  one jump.
