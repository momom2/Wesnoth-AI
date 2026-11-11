# Pre-registration: a larger full-network step, read by a match (2026-11-11)

Written 2026-11-11 at about 14:15, before any candidate is built. The user
asked for it the same day ("Run the match test"). Nothing else runs on the
desktop: the lean line is stopped
(docs/stage_c_rounds_local_prereg_20261030.md, departure 48).

## Question

A full-network step is small because the KL search sizes it on a few
game-sides where `own2`'s memory diverges (docs/status.md, "What the
numbers do not show"). Sized without them it could be 3 to 8 times larger.
That is reasoned from first-order readings; no game has read what such a
step does.

**Does a full-network step 3 or 8 times stage C's size gain strength over
`own2`, or does the divergence cost it?**

## The candidates

All three come from stage C's saved pass on `own2`
(`C:\Users\momom2\wesnoth_runs\stage_c_local\pass.pt`, 20,000 games) and
its record (`stage_c.json`).
- d is the reduced-AdamW direction (`wesnoth_ai/pg_step.py` `direction`).
- s_F = 7.3605e-05 is the full-network arm's step, and s_H = 0.028278 the
  heads-only arm's. Stage C sized both to a mean KL of 0.00134 a decision.

| label | weights | first-order gain | in Elo |
|---|---|---|---|
| `trunk_headst1` | `own2` + s_H·d on the four head maps: stage C's heads-only arm, the yardstick | 0.1093 ± 0.0067 (stage C, half A) | +38 ± 2 |
| `trunk_full3t1` | `own2` + 3·s_F·d | 0.044 ± 0.017 | +15 ± 6 |
| `trunk_full8t1` | `own2` + 8·s_F·d | 0.118 ± 0.044 | +41 ± 15 |

- **The full arms' forecasts** are stage C's first-order gain at s_F
  (0.0148 ± 0.0055) times the multiple. They assume the gain stays linear
  and the divergence costs nothing. Their SE is that reading's own.
- **In Elo:** a gain G is one seat's change in the outcome (±1), so the
  score is 1/2 + G/2. The rounds agree: round 3 read +134.1 at this
  decode after three heads-only steps.
- **A full arm's gain is the rest of the network's.** At 8·s_F the head
  maps move 2.1% of what they move in the heads-only step (8·s_F/s_H).
- **Built by** `scripts/window_20261029/trunk_step_candidates.py` on
  `feature/local-gpu-stage1`. It records each checkpoint's SHA-256 and the
  norm of what moved in `candidates.json`.

## The matches

    <rocm venv>/python scripts/match_local.py --t1 --label <label> --checkpoint <candidate> ^
        --out C:/Users/momom2/wesnoth_runs/trunk_step_match_20261111/<arm> --games 3200 --seed-base 950000 ^
        --tools-python <.venv>/python --keep-awake

- **Procedure: `raw:t1` on both sides,** as the tail matches
  (docs/tail_local_prereg_20261102.md). It is the decode the rounds train
  at and the unit of the forecasts above. Side B is `own2t1`.
- **Games:** 3,200 decisive an arm, on seeds 950,000-953,199. These are
  the tail matches' first 3,200, so every arm plays the same maps,
  factions and luck.
- **Serving:** the local fp16 server. The code is the run worktree's
  (`wesnoth_ai-run`, detached at 21862a0a).
- **Order:** full8, heads, full3, one at a time.
- **Fit:** tools/elo_collect.py PURE, decisive games only, `own2t1` at 0.

## Expected

- **Standard error:** about 6.2 Elo an arm.
- **Wall:** about 37 minutes a match, so about 2 h in all. The 12,800-game
  tail match of 2026-11-08 took 8,809 s.

## What decides

An arm **gains** when it reads at least +2 SE over `own2`, and **loses**
at −2 SE or under.

- **The yardstick first:** the heads arm must gain (its forecast is 6 SE).
  If it does not, the instrument is in doubt and no full arm is read.
- **Sizing alone unblocks the trunk** if full8 or full3 gains. Stage 2
  follows.
- **The divergence is harmful at that size** if a full arm loses.
- **No readable gain** otherwise: the larger step does not deliver its
  forecast at this resolution. It is reported with its distance from the
  forecast.

## Stage 2 (only if a full arm gains)

One more candidate: the head maps at s_H and every other parameter at
m·s_F, with m the multiple that gained (the higher Elo if both did). It
asks whether the trunk's step adds to a heads-only step, read as mix minus
heads. Its number of games is set from stage 1's readings, in a numbered
departure written before it launches.

## Kill criteria

- A display-driver reset stops the match (match_local, exit 5). It is
  inspected before any new start.
- A match past 60 minutes (1.5 times its estimate) is inspected and cut.
- No-result games above 1% of an arm are recorded with the result. The
  earlier T=1 matches had none.
- The user wants the GPU: a STOP file, and the match resumes from its
  games later.

## What it does not show

- **One step from `own2`.** The pass holds 20,000 games, four times a
  round's, so a round's direction would be noisier.
- **Strength at the reference's decode.** At argmax the rounds' gains read
  about a quarter of their T=1 size. Nothing here is a strength claim, and
  nothing is adopted.
- **Whether a remedy for the loop would do better** (a frozen writer,
  longer windows, a new memory).

## Departures

1. **Stage 2's size, its statistic and a control arm** (written
   2026-11-11 at about 15:45, while full3 plays and before any stage 2
   candidate is built).
   - **Stage 1 so far:** full8 +52.5 ± 6.2 and heads +34.6 ± 6.2 (the
     log below). So full8 gains, and stage 2 runs.
   - **m = 8,** unless full3 reads above full8.
   - **Two candidates,** 3,200 decisive games each on the same seeds,
     mix8 first:
     - `trunk_mix8t1`: the head maps at s_H and every other parameter at
       8·s_F. If the two gains add (0.099 and 0.150 in outcome), it reads
       about +88 Elo.
     - `trunk_heads2t1`, a control added here: the head maps alone at
       stage C's 2ε rung (s = 0.040149; its first-order gain there is
       0.153, about +54 Elo). Without it, "mix beats heads" cannot be told
       from "any larger step beats heads".
   - **The statistic** is the paired difference in score by seed, since
     the arms play the same maps, factions and seats. A seed counts where
     both games are decisive
     (`scripts/window_20261029/trunk_step_pairs.py`, written before these
     games). It was first run on full8 minus heads, after both were
     played: +0.0253 ± 0.0113 (2.2 SE; +17.9 Elo).
   - **What decides:**
     - the trunk's step adds to a heads-only step if mix8 minus heads
       reads at least +2 SE;
     - it adds more than a larger heads step does if mix8 minus heads2
       reads at least +2 SE;
     - otherwise the addition is not shown at this resolution, and it is
       reported with its distance from the additive forecast.
   - **Why 3,200 games:** if the gains add, mix8 leads heads by about 54
     Elo and heads2 by about 34, which is 6 and 4 SE of a difference.
   - **Cost:** about 80 more minutes of the desktop's GPU.

## Result

Pending. The run's log:

- 14:12:52 the pre-registration was committed (29482e10).
- 14:14 the candidates were built (`feature/local-gpu-stage1` f87cfd22;
  `C:\Users\momom2\wesnoth_runs\trunk_step_match_20261111\candidates\`,
  SHA-256 in `candidates.json`: heads 6b899267, full3 c55e9b63, full8
  4a7f2bce).
  - The direction's norms equal stage C's record (32.69 in all, 4.74 on
    the head maps).
  - What moved equals the step times the norm to 2 parts in a million.
  - A second run compared the files on disk with the weights built.
- 14:15:47 full8 started.
- 14:53:51 **full8: +52.5 ± 6.2 Elo** over `own2t1` (1,840 wins to
  1,360, no no-result game; 2,281 s). It gains, at 8.5 SE, and reads
  above its forecast of +41 ± 15.
- 14:54 heads started.
- 15:33:16 **heads: +34.6 ± 6.2 Elo** (1,759 to 1,441, no no-result
  game; 2,345 s). The yardstick gains, at 5.6 SE, on its forecast of
  +38 ± 2, so the full arms are read.
