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
   2026-11-11 at 15:35, while full3 plays and before any stage 2
   candidate is built; committed 15:35:48, d631e88f).
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
2. **heads2 is inspected at 75 minutes, not 60** (written 2026-11-11
   17:08, as heads2 starts).
   - The matches' walls grew: 2,281 s, 2,345 s, 2,609 s and 2,954 s, in
     the order played. No game failed or was replaced in any of them.
   - The desktop is in use beside the matches (a browser, a game launcher
     and a chat client hold most of the CPU time read at 17:08). Not
     read: whether that is the whole cause.
   - So the estimate for heads2 is the last match's 49 minutes, and 1.5
     times it is about 75.
3. **One reading at the reference's decode** (written 2026-11-11 17:58,
   after the result below and before the match starts).
   - **Why:** the result says nothing about strength where the reference
     plays ("What it does not show"). The user has since asked for the KL
     rule to go from the rounds, so the reading bears on what comes next.
   - **The match:** mix8 alone, as `trunk_mix8`, against `own2` at
     `raw:t0+eo-1.5` on both sides (match_local without `--t1`), 3,200
     decisive games on seeds 900,000-903,199, the standard matches' first
     3,200. Into `...\trunk_step_match_20261111\mix8_decode`.
   - **Forecast:** about +22 Elo. The rounds' gains at this decode read
     about a quarter of their T=1 size (round 3: +31.8 against +134.1).
   - **What decides:** mix8 gains at the reference's decode if it reads
     at least +2 SE over `own2` (about +12.4). Under −2 SE it loses
     there. Otherwise the gain is not shown at this resolution.
   - **No heads arm:** one heads-only step reads about +10 there by the
     rounds' record, which 3,200 games cannot resolve.
   - **Wall:** about 57 minutes (round 6's 12,800 took 13,749 s). It is
     inspected at 90.
   - This is still not a strength claim for an adoption: one match of
     3,200, and the candidate is one step from `own2`.

## Result

**Sizing alone unblocks the trunk, and its step adds to the heads'.**
Every pre-registered rule reads on the gaining side. Five matches of 3,200
decisive games, `raw:t1` on both sides, the same seeds, fp16 serving, code
21862a0a; no game was capped, failed or replaced.

| arm | Elo over `own2t1` | forecast | in outcome (2·score − 1) |
|---|---|---|---|
| heads (the yardstick) | +34.6 ± 6.2 | +38 ± 2 | 0.099 |
| full3 | +27.8 ± 6.2 | +15 ± 6 | 0.080 |
| full8 | +52.5 ± 6.2 | +41 ± 15 | 0.150 |
| heads2 (the control) | +48.7 ± 6.2 | +54 | 0.139 |
| mix8 | **+88.2 ± 6.3** | about +88 if the gains add | 0.249 |

- **Stage 1.** The yardstick gains (5.6 SE). Both full arms gain (4.5 and
  8.5 SE), at or above their first-order forecasts. So a full-network
  step 3 or 8 times stage C's size keeps its gain: the memory's
  divergence, which the KL search sized the step on, does not cost it in
  games at these sizes.
- **Stage 2** (the paired score difference by seed):
  - mix8 minus heads **+0.0747 ± 0.0110** (6.8 SE; +53.6 Elo): the
    trunk's step adds to a heads-only step.
  - mix8 minus heads2 **+0.0547 ± 0.0110** (5.0 SE; +39.5 Elo): it adds
    more than a larger heads step does.
  - The gains add: 0.099 and 0.150 forecast 0.249, and mix8 reads 0.249.
- **Larger is better as far as read:** full8 leads full3 by +0.0350 ±
  0.0111 (3.1 SE), and the gain per unit of step falls from 0.027 at 3
  times to 0.019 at 8. Nothing past 8 times was played.
- **At the reference's decode** (departure 3): mix8 reads **+15.9 ± 6.1
  Elo** over `own2` at `raw:t0+eo-1.5` (1,673 to 1,527 over 3,200
  decisive games on seeds 900,000-903,199; 6 no-result games replaced,
  0.2%; 3,314 s). That is 2.6 SE, so it gains there too, on a forecast of
  about +22. It is 18% of its T=1 reading, where the rounds' gains have
  read about a quarter.

What this does not settle, beyond "What it does not show" above:

- **How a round should size the step.** The multiples here were set by
  hand. The 8 came from the step's KL at a given memory (about 2e-5 a
  decision at s_F, so about ε at 8 times): a rule a round could apply,
  not yet built.
- **What a full-network pass costs.** Stage C's ran at 85 decisions a
  second (89,633 s for 7.6 million), against 271 for round 22's
  heads-only pass. At that rate a round's pass goes from 2.5 h to about
  8 h. Whether the trunk's step pays for it per GPU-hour is not read.
- **Whether the gain repeats.** Each further step pushes the memory's
  pattern again; this is one step.
- **Why the divergence is harmless here.** Not read: the candidates'
  memories were not walked.

### The run's log

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
- 15:33 full3 started.
- 16:17:10 **full3: +27.8 ± 6.2 Elo** (1,728 to 1,472, no no-result
  game; 2,609 s). It gains, at 4.5 SE, and reads above its forecast of
  +15 ± 6. It is under full8, so stage 2 keeps m = 8.
- By seed (`trunk_step_pairs.py`, in score): full8 minus full3 +0.0350 ±
  0.0111 (3.1 SE), full8 minus heads +0.0253 ± 0.0113 (2.2 SE), full3
  minus heads −0.0097 ± 0.0112 (−0.9 SE).
- 16:17 the stage 2 candidates were built and all five files checked
  again (mix8 d8ce049c, heads2 e1ad6fff); 16:17:54 mix8 started.
- 17:07:11 **mix8: +88.2 ± 6.3 Elo** (1,998 to 1,202, no no-result
  game; 2,954 s). The additive forecast was about +88.
  - By seed: mix8 minus heads **+0.0747 ± 0.0110** (6.8 SE; +53.6 Elo),
    so the trunk's step adds to a heads-only step (departure 1's first
    rule).
  - mix8 minus full8 +0.0494 ± 0.0111 (4.5 SE; +35.8 Elo).
- 17:07:56 heads2 started (departure 2).
- 17:48:53 **heads2: +48.7 ± 6.2 Elo** (1,823 to 1,377, no no-result
  game; 2,454 s, inside departure 2's 75 minutes).
  - By seed: mix8 minus heads2 **+0.0547 ± 0.0110** (5.0 SE; +39.5 Elo),
    so the trunk's step adds more than a larger heads step does
    (departure 1's second rule).
  - heads2 minus heads +0.0200 ± 0.0111 (1.8 SE); heads2 minus full8
    −0.0053 ± 0.0115 (−0.5 SE).
- The records: `C:\Users\momom2\wesnoth_runs\trunk_step_match_20261111\`
  (an arm a directory: `result.json`, `fit.json`, `games/`), and the
  summary `training/metrics/trunk_step_match_20261111.json` on
  `feature/local-gpu-stage1`.
- 17:57 mix8's match at the reference's decode started (departure 3).
- 18:52:38 **mix8 at the reference's decode: +15.9 ± 6.1 Elo** (1,673 to
  1,527; 6 no-result games replaced; 3,314 s, inside the 90 minutes).
