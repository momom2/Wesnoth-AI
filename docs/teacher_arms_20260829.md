# Teacher arms (2026-08-29): is TCS or MCTS the better teacher?

User directive (2026-08-29, autonomous window): "design measurement
legs for figuring out whether MCTS or TCS is the best teacher, and
log anything on the way that might explain degradation — do it
iteratively with deeper measurements until we can figure out why
training didn't work."

## Why this experiment

The leg-5 resume verdict's 2x2 matrix measured TCS **as a play
procedure** ~200 Elo below plain Gumbel MCTS-32 on identical weights
(seed+TCS lost 9-0-31 to seed+MCTS). Training's objective distills
the policy toward TCS-refined play. If the teacher is the weaker
player, every leg trains toward weakness — the leading hypothesis
for the all-proxies-healthy erosion. It has never been tested in
isolation. Separately measured this session: the improvement engine
exists (seed+MCTS-32 beats raw seed 9-0-1, +321 ± 153), so a
correct teacher CAN produce gains in principle.

## Design

Two boxes, one arm each, launched from the **imitation seed**
(`seed_imit_tierb_start.pt`, step 2,809,659, board +223):

- **Arm T (box 49121975): TCS teacher** — the leg-5-resume config
  verbatim: TCS generation (default ON), `--turn-boundary-frame
  mover`, `--abort-k-median 10`, policy anchor v2, mix
  ladder/midgame/fogless 0.6/0.2/0.2, mini 0, sims 32, actor-pool
  topology, no distill-prior-discount. This doubles as the
  **fixed-GBC control leg**: it re-measures the erosion rate under
  the only config change that ships regardless (the GBC label fix —
  every historical erosion number was measured with broken event
  labels).
- **Arm M (box 49121978): MCTS teacher** — identical in every flag
  except `--no-turn-search` (plain Gumbel MCTS-32 generates and is
  distilled). `--abort-k-median 10` armed: this deliberately
  re-enters the regime where turn-truncation (K collapse) was
  originally acquired; a K collapse here is a RESULT (the root
  cause was never TCS-specific), not just a failure.

Known deviations from leg-5-resume, identical across arms so the
teacher contrast is unaffected: fresh boxes/hand-driven launch (no
HF escrow of the campaign; pins pulled to the laptop), and the new
compile+bf16 inference default (generation numerics differ slightly
from all historical legs; flagged in BACKLOG).

## Pre-registered readout

Primary: **Elo-vs-seed slope per arm**, from periodic on-box probe
matches of each saved pin against the seed — 24 games per pin,
both sides MCTS-32 (`--no-turn-search`), the reference frame the
2x2 identified as the stronger procedure and the seed's native
sampling. Probes are INTERNAL slope measurements (bf16+compile
defaults, GPU), not board numbers.

Predictions:
- H-teacher: Arm T's slope is clearly more negative than Arm M's.
  Historical rate for context: leg-5 resume lost ~200 Elo over
  ~144k steps (~-1.4 Elo/1k steps) with all proxies healthy.
- If BOTH arms erode at the historical rate: teacher exonerated →
  the degradation lives elsewhere (candidates, in order: the
  Gumbel micro-tax / target extraction itself; value-trunk drift;
  replay-buffer staleness). Next-deeper measurement is then
  target-vs-prior grading on stored states (does the distill
  target actually rate better than the prior under deep search?).
- If NEITHER arm erodes: the GBC label fix (or the removed
  environment delta) was the channel; re-isolate by reverting the
  label fix on a third cheap arm.

Degradation telemetry logged continuously on both arms (the
"anything that might explain it" channel): per-iteration trainer
history (CE-vs-floor, value_auc, K medians, attack%, draw rate,
loss decomposition), hourly human-holdout CE probe (telemetry
only, CE abort removed by ruling), fork-guard smoke before launch,
per-pin probe matches with per-side forward counts/seconds, and
the sim_self_play banner with the full resolved flag set.

## Ops

- Boxes: 2x RTX 4090, 32 eff cores @ ~5.9GHz (per-core speed
  lesson from the E5-2696 eval box), ~$0.42/h each.
- Cadence: pins every 2 iterations (`--save-every 2`); pin-copy
  loop snapshots the rolling checkpoint; probe loop plays each new
  pin vs the seed on the box's GPU.
- Tripwires: decisive-rate 0.35/20, K-median 10, holdout-stall 60,
  supervised relaunch loop (10 tries), ABORTED_* markers.
- Kill criteria: tripwire abort = arm result recorded; box idle
  with no training process and no marker = investigate, relaunch
  once, else stop the arm.
