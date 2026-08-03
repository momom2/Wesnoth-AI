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
- Record the pre-grow holdout CE from the 5M campaign's CSV *before*
  launching, or there is nothing to recover *to*.
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

- **`tier_a_campaign.pt` is a reserved name.** See §4.
- **Do not compare a fresh grow to anything.** See §3.
- **Head alignment**: any future grow keeps `d_head = 32`
  (`num_heads = d_model / 32`).
- **The grow's damage is roughly additive across width and depth**
  (§11), so a further grow from Tier-b compounds. If a Tier-c step ever
  happens, do it progressively (width, recover, depth) rather than in
  one jump.
