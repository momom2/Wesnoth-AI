# Stage C driver's forwards: where they go, what could be saved (2026-10-30)

This is a read-only audit of `tools/pg_stage_c.py` and its evaluator
(`tools/pg_eval.py`), made during the GPU wedge. A fresh subagent did the
reading; the lead checked the plan's formula and the condition for the
second critical batch. Nothing here ran on the GPU. The driver is stage
C's largest wall: the prereg estimated 20-25 h.

**FE** is one forward over every held-out decision (10,000 games at 378
decisions, 3.78M). A half read is 0.5 FE. A KL reading forwards the moved
network over the 5% subsample (pg_eval.py's KL draw), so it is 0.05 FE.

## Where the forwards go

The plan's 11.25 FE is `forward_equivalents(configured_plan(trim=False))`
in `tools/stage_c_plan.py`. It is priced on the worst case.

| Part | Reads | FE |
|---|---|---|
| θ_k, every decision, once | 1 full | 1.0 |
| θ_k's laws on the subsample | 1 KL | 0.05 |
| The five arms, half A each | 5 half | 2.5 |
| The ladder: eps/2 and 2eps on A, the chosen rung on B (eps's A read is cached) | 3 half | 1.5 |
| The gain curve, two points | 2 half | 1.0 |
| The critical batch: 17 Fisher forms × ±s, for 2 batches | 68 KL | 3.4 |
| The KL searches: 5 arms, 2 rungs, 2 curve points, priced at 4 evals each | 36 KL | 1.8 |
| The span Fisher (the pass carries `head_fisher`) | none | 0 |

**What a run should cost.**
- The second critical batch runs only when a linear arm other than reduced
  AdamW wins. `configured_plan` always counts it, on purpose: the plan sets
  the cut, so it prices the worst case.
- Warm searches take 1-2 evaluations, not 4.
- With reduced AdamW winning, the run is about **9.3 FE**.

**The rate.** The forward bench reads 574 positions a second with the
bias fold on. The pg tools run with the fold off, at 528 a second (the
same A/B, `hip_bias_fold_20261029`), so 1 FE takes about **2.0 h**. That
puts the driver at about 19 h expected and 22.5 h in the worst case,
before replay and staging overheads. C0 measures the walk's own rate,
and that rate prices the run.

## What is already efficient

- **The KL search** steps s by the quadratic model's √(target/KL(s)), then
  refines by Illinois on log KL against log s. A cold search takes 2-3
  evaluations where KL(1/|d|) lands within 100× of eps (the toy: 2.6 on
  average), and 4-6 when it lands 10⁴-10⁶ away. Each evaluation is the
  moved network alone on the subsample, about 6 min.
- **Cached:** θ_k's base log-probs and laws, shared by every evaluator;
  each evaluator's readings, by (s, halves); and the eps rung's reading.
- **Warm starts:** the other rungs start from √-scaled steps, and the
  Fisher forms from the whole pass's step scaled by norm. From 0db6ea14
  the gain curve's points do too (0.1-0.2 FE).
- **Halves:** the arms read half A only, and only the chosen rung reads B.

## What remains, ranked (none done)

1. **Several networks per walk.** Every read is its own replay of the
   held-out games, though `_walk` can forward several networks over one
   replay.
   - It would fold the ±s of each Fisher form, the arms with the curve's
     points, and the base into shared walks.
   - The replay volume falls from about 11 to about 5.5 FE. The forwards
     stay the same; the replay's share of the walk, ρ, is saved.
   - ρ is unmeasured. At 0.1-0.3 the saving is 5-15%.
   - It touches the fp16 evaluator, whose hang is not understood, and
     whether 7 model copies fit in the GPU's memory is unverified. **After
     the reboot, with C0's rates in hand.**
2. **One-sided Fisher forms**, 2·KL(s)/s², for the 16 split and deviation
   forms (q_total keeps both sides).
   - Saves 0.8 FE a batch (7-14%).
   - It changes each form by its cubic share, about 2cs/q at KL ≈ eps/4,
     which is unmeasured. That makes it a design change, read before it is
     made.
3. **Two-level subsample for the searches:** bracket on a nested 1% (the
   same draw), and finish on the 5%. Saves 3-4%, and the readings are
   unchanged. The 1% needs its own laws walk.
4. **q_total from the ladder's points** instead of its own readings:
   about 1%.
5. **KL_RTOL 0.01 → 0.05:** 3-4%, but it moves s by up to 2.5%, and the
   paired comparison of close arms could feel that. Last resort.

Rejected: the logged play-time log-probs as θ_k's base (−1 FE). The
log-ratios would mix the served numerics with the recomputed ones,
summed over about 189 decisions a side.

The search counts come from a pure-Python copy of `kl_search` on
KL = ½qs²(1 + as), a cubic share a from 0 to 20%.
