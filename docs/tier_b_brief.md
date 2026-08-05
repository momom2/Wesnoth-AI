# Tier-b pre-spend brief (drafted 2026-08-04, T-B measurement PENDING)

One page consolidating what is measured vs pending before the Tier-b
campaign box is rented. Operational mechanics: `docs/tier_b_runbook.md`
(grow → recover → campaign → measure). Decision authority: user
(2026-08-02, "Tier-b directly"), taken knowing the external signal is
unproven — this brief is the final pre-spend evidence check, not a
re-litigation of that decision.

## Measured inputs (done)

- **T-C, raw-policy transfer (2026-08-04, `d302f9d`):** in-lineage
  gain survives WITHOUT search: **+61.2 ±18 Elo** raw-vs-raw
  (163-0-232 / 395 ladder games, one joint BT fit, prediction
  pre-registered). ~Half the with-search gain (+133 ±57) transfers to
  the raw policy. The "distillation-transfer catastrophe" scenario
  (all gain lives in search) is DEAD. Escalation trigger did not fire.
- **Grow candidate exists and is verified:** `tier_b_15m.pt`
  (d384/L8/H12/ff1536, 15.55M), grown from `campaign_live_20260730.pt`
  (decision_step 2,515,896, best measured). Warm-start value MAE
  ~0.226 = known NOT-A-WARM-START starting condition; recovery leg
  (runbook §3) is mandatory before any strength number.
- **External anchor unchanged:** 0-0-30 vs built-in RCA, median leader
  death turn 10. Tier-b is a capacity experiment on an unproven
  external signal (runbook §0) — restated so nobody quotes in-lineage
  Elo as external progress.

## Pending inputs (blocking the spend)

- **T-B, teacher advantage (MEASURED 2026-08-04): KILL, decisively.**
  3,658 paired states / 150 human-corpus games, checkpoint 2,515,896,
  campaign search config, zero error rows. Pooled AUC: raw head
  **0.667**, 32-sim search root value **0.525** — delta **−0.141**
  (game-cluster bootstrap 95% CI [−0.201, −0.079]), and the search is
  worse in EVERY phase (open −0.086, mid −0.203, late −0.127). Not
  merely "≤ +0.02": the search teacher is barely above coin-flip and
  substantially DEGRADES the raw head's outcome prediction. The
  value-distillation channel is dead, and so is "repair the flat value
  head from search values" as a passivity fix direction. Shards:
  scratchpad tb_part{0,50,100}.jsonl; analyzer
  `tools/analyze_teacher_advantage.py`.
- **Side-2 passivity root cause (workflow verdict, adversarially
  verified 2026-08-04):** the asymmetry is confined to the 3 mini maps
  WITHOUT `random_start_time` (deterministic ToD); on the other 4 the
  sides are symmetric. No reward-path asymmetry exists (draws z=0 both
  sides, verified). Mechanism: a self-referential Gumbel PRIOR ratchet
  (~3.9-logit side-2 prior gap vs ~0.5 logits of value restoring
  force; the ±0.37 value bias cancels in ranking). Confounded 3/4
  splits (board size, branching 10-20 vs 40-238, declared-fog
  mismatch) are not separable at N=7 maps; the decisive de-confound is
  forcing `random_start_time` on for the mini pool (changes ToD
  determinism only). Config-level repairs that survived verification:
  mini random-ToD flag, price the clock (`--no-progress-turns`),
  `_rescale_q` floor 1e-8 → ~0.01. NOT a Tier-b blocker per se (mini
  category only), but the ratchet mechanism is category-agnostic.
- **Export-fidelity sweep (DONE 2026-08-04): 538/574 clean; 34 real
  OOS in 3 classes, none blocking after triage.** (1) 2× ladder_fogless
  damage mismatches, both at Tombs of Kesorak's darkened hex (19,12) —
  KNOWN-FIXED by `9133cca` (bare clones dropped `schedules.cfg`, ToD
  macros didn't expand, time_area lawful_bonus defaulted 0); exports
  predate the fix. (2) 30× midgame "found dependent command while
  is_synced=false" — a replay-STRUCTURE bug in the midgame export
  prologue (not physics); root-cause in progress. (3) 2× mini combat
  mismatches vs Tentacle of the Deep — under investigation. Ladder
  (167) and remaining fogless (104): fully clean.

## Cost frame (from the runbook and measured throughput)

Tier-a throughput was ~4,000-7,000 decision-steps/hour with detectable
step gaps of 450k-1M ⇒ ~100-200 box-hours per measurable increment;
Tier-b's ~1.9x forward cost stretches that further. The campaign box
should not start until the three pending inputs above are read.

## Recommendation (filled 2026-08-04, all three inputs measured)

- [x] T-B verdict: **KILL** (delta −0.141, CI [−0.201, −0.079];
      worse in all three phases)
- [x] Value-distillation channel: **kill** — do not carry into Tier-b
- [ ] Passivity fix: **user decision pending** — mini-only symptom,
      but the prior-ratchet mechanism is category-agnostic; the cheap
      pre-campaign options are the mini random-ToD flag and the
      `_rescale_q` floor (both config-level, hours not days)
- [x] Sweep: **no blocking class** — 2 known-fixed (9133cca),
      30 midgame = export-structure bug (validation tooling, not
      training physics), 2 tentacle-combat under investigation
- [x] GO / NO-GO: **GO — user decision 2026-08-05: full move to
      Tier-b; all training on the 15M net from now on unless stated
      otherwise.** Pre-campaign context now in hand: throughput
      program measured (playout-cap landed 2.3x; spool is the shape
      on 3060-class hardware; the campaign GPU should be chosen for
      real batch throughput), distillation damping validated
      (lambda=0.9), per-side normalization live, and the mini-draw
      mechanism identified as an honest mutual-passivity equilibrium
      whose incentive-level repairs (train-draw-tiebreak, no-progress
      clock) await a user ruling before the campaign launches.
