# Leg-5 resume verdict (2026-08-26, overnight autonomous run)

**ADDENDUM 2 (same day; supersedes addendum 1, whose "protocol
artifact" reading was wrong).** The full 2x2 sampling matrix, all
40-game matches, sims 32, same seed schedule, 0 no-results:

    pin+MCTS vs seed+MCTS :  9-0-31   seed +208 ± 64
    pin+TCS  vs seed+TCS  : 22-0-18   pin  +34  ± 55
    pin+TCS  vs seed+MCTS :  6-0-34   seed +290 ± 74
    seed+TCS vs seed+MCTS :  9-0-31   MCTS +208 ± 64

Reading: **TCS at play time costs ~200 Elo on the SAME weights**
(row 4 — the seed against itself). Every pin/TCS combination
clusters ~200-290 below seed+MCTS; pin+MCTS lands exactly at
seed+TCS's level. Coherent story: training distilled the policy
toward TCS-refined play — its literal objective — and TCS-refined
play is ~200 Elo weaker than the same prior played through
MCTS-32. The target was worse than the prior at the MACRO level;
the teacher (TCS as a play procedure) is the suspect, ahead of the
Gumbel micro-tax. Addendum 1's 22-0-18 was the pin beating a
TCS-crippled seed.

Caveat: not compute-matched — TCS as configured spends ~3x fewer
forwards per side-turn than MCTS-32 (~113 vs ~384 at K~12; games
2-4 min vs 5-8). The deployment verdict at configured settings
stands; an equal-compute TCS arm is unmeasured. Note TCS was
validated by a per-decision micro-probe (accept 0.64, Δ~2 atoms,
placebo-separated) but never match-validated as a play procedure
until today.

Game files: eval_games/{l5_pin_vs_seed,l5_pin_vs_seed_tcs,
pinTCS_vs_seedMCTS,seedTCS_vs_seedMCTS}. All fits --no-catalog
except row 1 (the catalog protocol).

The original MCTS-32 verdict:

**The leg FAILS the pre-registered Elo gate under the OLD (raw
MCTS-32) protocol: the 3.06M pin lost 9-0-31 to its own seed
(seed +208 ± 64, catalog protocol, 40/40 decisive). Training
stopped, box stopped.**

## Run record

- Box 48670875 (4090, 32 eff EPYC cores, $0.364/h), launched
  2026-08-25 ~16:47Z from configs/leg_l5.json, resumed the escrowed
  checkpoint (step 2,931,890).
- Qualify gate (newly wired into vast_onstart.sh, commit 313d50f):
  PASS, value_auc 0.765 on the entry checkpoint.
- Train banner audited: `--turn-boundary-frame mover`,
  `--abort-k-median 10`, NO `--distill-prior-discount`. Mix
  60/20/20, both anchors, tripwires armed.
- 8 iterations, step 2,931,890 -> 3,076,145 (~16k steps/iter,
  24/24 decisive every iteration, no tripwire ever fired).
- Pin at **3,060,972** (+251k past the 2,809,659 seed) escrowed as
  `tier-b/tier_b_l5_pin3060972.pt` (+ .holdout sidecar).

## The verdict match

`tools/run_elo_batch.py`, catalog protocol (mcts-sims 32), 40
games, seeds by index, 0 no-results, fit by `tools/elo_collect.py`
(PURE, decisive only):

    l5_pin3060972 vs seed: 9-0-31  ->  seed +208.2 ± 64

Game files: `eval_games/l5_pin_vs_seed/` (pulled local). Catalog
updated (`training/metrics/elo_catalog.json`, working tree).

## The finding that matters

**Every internal proxy was healthy while the policy lost ~200 Elo.**
Under the repaired stratified instrument, across the whole leg:
value_auc 0.68-0.73 (floor 0.60, never a single redraw), human-CE
2.48-2.60 (below the 2.90 seed baseline), decisive rate 24/24,
K-median tripwire silent, boundary pairs live. The leg-3 failure
mode (K collapse) and the leg-5-abort mode (value rotation) both
did NOT recur — and strength eroded anyway, at roughly the leg-3/4
rate (~200 Elo per ~250k steps). The erosion channel is invisible
to the entire current telemetry set. X4/X5 (pre-registered against
value rotation) do not obviously apply: the value head read fine
this time.

## Ops notes (for box_specs)

- 15M-vs-15M eval at sims 32 on EPYC CPU cores: ~3 min/turn; only
  turn-6..10 early kills finish inside 40 min (27/40 games timed
  out). Same match on the idle 4090 (`--device cuda --jobs 10`):
  4-8 min/game, 37 games in 33 min, VRAM 3.7-7.4 GB, zero
  timeouts. **When a GPU is idle on the box, run evals on it.**
- Spend this run: ~10h box time ≈ $3.6. Credit remaining ≈ $6.3
  (minus stopped-box storage drain: now THREE stopped boxes).

## State after

- Training DOWN (verdict-stopped, per NEXT ACTIONS item 2:
  "not improved -> stop and rethink").
- Box 48670875 STOPPED (storage). Also stopped: 48108334,
  48607224. All identity is on HF escrow; the boxes hold only
  convenience state.
- Logs local: probe CSV `training/logs/holdout_probe_l5resume.csv`;
  train.log + elo.log in the session scratchpad.
