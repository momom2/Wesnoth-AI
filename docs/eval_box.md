# Eval box (decision 2026-08-03)

**Decision (user, 2026-08-03): evaluation runs on a rented box, not on
the laptop.** Operate under this assumption — do not plan, schedule, or
cost any eval as local work.

This is a *separate, cheap, short-lived* box from the Tier-b campaign
box (`docs/tier_b_runbook.md` §4b). Different job, different shape,
different price. Do not conflate them.

---

## 1. Why not local — measured, 2026-08-03

The laptop has **7.6 GB total RAM**. With a browser and chat client
open, **0.58 GB was free and 12.7 GB committed**. Under that pressure a
torch process page-thrashes instead of computing:

> One raw eval game consumed **~1 second of CPU in 9 minutes of wall
> clock** and produced no result file.

That is not "slow", it is *zero throughput plus crash risk* — the
machine had already crashed once that session. It also retroactively
explains CLAUDE.md's parallel-pytest warning: on a 7.6 GB box, one torch
process is most of the memory budget, so "don't run two" was never
really about pytest.

Corollary worth stating plainly: **the local measurement track was never
"free".** It is cheap in money and expensive in a resource this machine
does not have.

## 2. What eval actually needs (and does not)

- **Cores, not GPUs.** `tools/elo_eval_game.py` is **one game per
  process** — the pattern that saturated a 4090 where a central pool
  could not (BACKLOG 2026-07-03). Throughput scales with concurrent
  games, i.e. with cores.
- **A GPU is not merely unnecessary, it is actively counterproductive
  at scale.** Each process builds its own CUDA context (~300-600 MB),
  so N concurrent games cost N contexts and VRAM runs out long before
  the cores are busy. `--device cpu` (added 2026-08-03) exists for
  exactly this; `--device cuda` now REFUSES to fall back silently,
  because an eval that quietly changes device is an eval whose timings
  mean nothing.
- **RAM per concurrent game** is the real sizing constraint.
  `tools/run_elo_batch.py` defaults to a deliberately conservative
  1800 MB/job floor and multiplies it by `--jobs`. **Measure actual
  per-process RSS on the box and set `--min-free-mb` from the
  measurement** rather than inheriting a laptop-derived guess.
- **No learner VRAM requirement**, so none of the Tier-b §4b VRAM
  reasoning applies here.

**Requirements:** many cores (32-64+), RAM ≥ ~1 GB per concurrent job,
modest disk, Python ≥ 3.11. Reliability and duration matter far less
than for a campaign — an eval run is *hours*, and
`tools/run_elo_batch.py` is resumable, so a lost box costs only the
games in flight.

## 3. Runbook

Games accumulate in one directory and are fitted afterwards, so a run
can be split across chunks, boxes, or interruptions freely.

```bash
python tools/run_elo_batch.py --label-a best --spec-a training/checkpoints/campaign_live_20260730.pt --label-b anchor --spec-b training/checkpoints/selfplay_seed_20260718.pt --outdir eval_games/tc_raw --games 400 --mcts-sims 0 --device cpu --jobs 32 --time-budget-min 120
```

Then fit — draws are draws (PURE is the headline; material-sign is
diagnostic only, user decision 2026-07-11):

```bash
python tools/elo_collect.py eval_games/tc_raw
```

Notes that decide whether the numbers mean anything:

- **`--mcts-sims 0` is RAW policy; `32` is training-matched search.
  Never merge the two.** The +133 ±57 in-lineage result was measured
  WITH search; the 0-0-30 RCA anchor was measured RAW.
- **Ladder maps only** for Elo — no mini/drill/midgame. This is the
  locked protocol (2026-07-03), enforced by `test_elo_ladder_maps.py`.
- Sides alternate automatically and seeds derive from the game index,
  so re-running the same command schedules the same games and two
  chunks never collide.
- Pull the checkpoints from HF (`momom2/wesnoth-model-checkpoints`) on the box;
  don't ship them from the laptop.

## 4. The queue this box exists to run

Ordered by information per unit cost (`docs/autonomous_run.md` phase-2
adjudication). **T-A is done**; T-B and T-C are what the box is for.

| | test | what it decides | kill / escalate |
|---|---|---|---|
| **T-A** | value-head AUC probe | DONE 2026-08-03. Head is mediocre, not blind — the stale ~0.50 is dead. Also showed an apparent monotone AUC *regression* across the +133 interval (0.713→0.636 / 0.784→0.727 / 0.683→0.609, n=40 games, paired, significance NOT established). | — |
| **T-B** | teacher-advantage: AUC of the 32-sim search's root value vs the raw head, identical states | whether there is anything to distill into the value head at the campaign's search budget | **kill the value channel at ≤ +0.02** |
| **T-C** | raw-vs-raw in-lineage Elo, the two endpoints of the +133 | whether search gains reach the DEPLOYED raw policy | kill the concern at ≥ +67; escalate to top signal-lever if ~0 (CI within ±50) |

**Why T-C is the one that matters for sequencing:** if the raw policy
did not improve in-lineage, that is a distillation-transfer problem, and
no amount of Tier-b capacity fixes a transfer problem. It is cheap and
it can reorder the whole (a)/(b)/(c) fork in `BACKLOG.md`. Run it before
the campaign box goes live.

## 5. Cost shape

Deliberately not quoting prices — they drift, and the 2026-08-02 Tier-b
search found the quoted price had **doubled in a day** while the offer
ID rotated. Re-search at rental time.

The shape: eval wants a cheap many-core box for *hours*, and T-C's ~400
games are embarrassingly parallel, so wall-clock ≈ games ÷ jobs ×
per-game time. **Per-game time is currently UNMEASURED** — every local
attempt ran under memory pressure and none completed. Measure one game
on the box before sizing the run; do not extrapolate from the laptop.

## Results (2026-08-04)

**T-C COMPLETE: raw-vs-raw in-lineage = +61.2 ± 18 Elo** for
campaign_live_20260730 (2,515,896) over selfplay_seed_20260718
(2,290,529): 163-0-232 over 395 ladder games, --mcts-sims 0, cpu,
sides alternated, seeds by game index. ZERO draws in 395 raw ladder
games. PURE == material-sign here (no draws to score).

Reading vs the pre-set bars: the kill-the-concern bar was >= +67
(half the with-search +133); +61 +-18 lands just under it but the CI
[43, 79] straddles the half-mark and decisively excludes ~0. Verdict:
**raw transfer is REAL and carries roughly half the search-measured
gain** -- the distillation-transfer catastrophe scenario is dead; the
escalate-to-signal-lever trigger (CI within +-50 of 0) did NOT fire.

**T-B COMPLETE: teacher advantage = -0.141 AUC (KILL).** 3,658 paired
states / 150 human-corpus games, checkpoint 2,515,896, campaign search
config (32 sims, tiebreak cap 0.3), zero error rows. Pooled AUC: raw
head 0.667 vs search root value 0.525; delta -0.141, game-cluster
bootstrap 95% CI [-0.201, -0.079]. Per phase: open -0.086, mid -0.203,
late -0.127 -- the search value is WORSE everywhere, not merely under
the +0.02 bar. Consequences: (a) the value-distillation channel stays
dead -- there is nothing to distill at the campaign budget; (b) the
"repair the flat value head from search values" direction for mini
passivity is dead too; (c) NB this measures the visit-weighted root Q
as a *distillation teacher* on human states -- it does not contradict
T-C's finding that search helps *move selection* in self-play.
Instrument: tools/probe_teacher_advantage.py; analyzer:
tools/analyze_teacher_advantage.py; shards archived in the session
scratchpad (tb_part{0,50,100}.jsonl).
