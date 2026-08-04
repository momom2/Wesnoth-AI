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

- **T-B, teacher advantage (RUNNING, eval box, 3 shards):** paired
  AUC of the 32-sim search root value vs the raw head on identical
  human-corpus states, campaign search config, checkpoint 2,515,896.
  Pre-registered bar: **delta ≤ +0.02 ⇒ kill the value-distillation
  channel** before the Tier-b campaign inherits it. Analyzer:
  `tools/analyze_teacher_advantage.py`. Also read directionally: if
  search values DO carry signal the raw head misses, that is the
  repair direction for the flat-value side of mini passivity.
- **Side-2 passivity root cause (workflow RUNNING):** why side 2's
  end_turn prior collapses (late-game 78% vs 40% at p>0.8, conditioned
  beyond value level). If the mechanism is a training-signal artifact
  (draw-tiebreak leakage / Gumbel target asymmetry), it ships with the
  Tier-b campaign unless fixed first — a fix-before-campaign decision
  belongs in this brief once the verdict lands.
- **Export-fidelity sweep (RUNNING, 574 post-fix replays):** zero real
  OOS so far. A new OOS class in live categories would pause the spend
  (sim faithfulness is load-bearing for everything).

## Cost frame (from the runbook and measured throughput)

Tier-a throughput was ~4,000-7,000 decision-steps/hour with detectable
step gaps of 450k-1M ⇒ ~100-200 box-hours per measurable increment;
Tier-b's ~1.9x forward cost stretches that further. The campaign box
should not start until the three pending inputs above are read.

## Recommendation slot (to fill when T-B lands)

- [ ] T-B verdict: ______ (delta, CI, per-phase)
- [ ] Value-distillation channel: keep / kill
- [ ] Passivity fix: pre-campaign / deferred (workflow verdict: ____)
- [ ] Sweep: clean / new OOS class (____)
- [ ] GO / NO-GO to rent the Tier-b campaign box
