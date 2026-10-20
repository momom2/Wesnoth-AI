# Pre-registration: `own2` as the base, stage S0(b) (2026-10-20)

Written on `exp/own2-base` before any box for it, for the lead to price and
the user to approve.

## Question

Is `own2` stronger than `own1`, both at the reference decode with 64 memory
slots? The answer picks π_0, the base the near-tie coin rounds start from
(docs/training_design_20261019/current_design.md, 3.1, and section 4,
S0(b)). The record is round 2's R1: `own2` at 64 slots against `own1` at 64,
+16.5 +- 12.3 Elo over 800 decisive games
(docs/memory_own_round2_prereg_20261011.md, "Measured"), an interval that
contains 0.

The match plays 6,400 decisive games, half the design's 12,800, because
this window's box spending is capped at $5 in all and the coin measurement
comes first. A second half of 6,400 games on fresh seeds can be added later
and pooled with this one (see Statistics).

## The match

`scripts/own2_base_box.sh`, records on HF `tier-b/own2_base_20261020`.
PURE, one match:

| side | player | checkpoint | memory |
|---|---|---|---|
| A | `own2` | HF `tier-b/memory_own_round2_20261011/own.pt`, SHA-256 `aea027b2...9129` | 64 slots |
| B | `own1`, the reference (configs/reference_player.json) | HF `tier-b/memory_own_trajectories_20261011/own.pt`, SHA-256 `70925e77...7f80` | 64 slots |

- `own2`'s hash is the one R1's inference server recorded for the checkpoint
  it served (training/metrics/memory_own_round2_20261011/r1_own2_64_vs_own1_64.server_0.json)
  and the one HF's LFS metadata gives for the file (189,557,937 bytes,
  committed 2026-10-12 02:38 UTC, read 2026-10-20 with
  `huggingface_hub.HfApi.get_paths_info`). The box checks both checkpoints
  against their hashes before the match.
- Both sides at `raw:t0+eo-1.5`: argmax, end_turn offset -1.5, no search.
  The Ladder maps with fog, both factions drawn uniformly.
- 6,400 decisive games; a capped game is replaced (up to 750
  replacements); every game is recorded whole. 20 games in flight through
  persistent workers and shared inference (one server per checkpoint on the
  one GPU), eager kernels: R1's configuration.
- Seeds 300000 to 306399; a capped game of slot i is replaced with seed
  300000 + i + k x 1,000,000. On origin/main 41e2f05 the highest seed base
  named in a tracked file is 250000 (an 800-game match), and no tracked file
  names a match seed from 300000 to 312799 modulo 1,000,000 (every
  `"seed":` field and game file name, read with `git ls-tree` and `git
  show`). A second half takes 306400 to 312799.
- The fit (`tools/elo_collect.py --anchor own1`) runs once, on the 6,400
  decisive games. The records on HF carry the game count every 30 minutes
  and the games' tarball; no partial fit is read.

## Statistics

At an even score the fit's standard error on 6,400 decisive games is
(400 / ln 10) / sqrt(6,400 / 4) = 4.34 Elo; the same formula gives the
record's 12.3 at 800 games. The decision reads the fit's own standard
error.

- The rule picks `own2` when the reading exceeds 2 x 4.34 = +8.7 Elo.
- If `own2` is +16.5 Elo stronger, the rule picks it with probability
  Φ((16.5 - 8.7) / 4.34) = 0.964; if the two are equal, with probability
  0.023.
- A reading agrees with the record when it lies within two standard errors
  of the difference: 16.5 +- 2 sqrt(12.3² + 4.34²) = 16.5 +- 26.1, that is
  from -9.6 to +42.6.
- **The second half.** Pooled, the two halves are 12,800 games at a standard
  error of 3.07 Elo, the design's S0(b), and the pooled reading takes the
  same rule. Adding the half only when this reading is inconclusive raises
  the probability of picking `own2` when the two are equal from 0.023 to
  0.038 (two looks at two standard errors, the second on the pooled games,
  a normal model).

## Decision

- **Lower bound above 0** (reading minus two standard errors): π_0 is
  `own2`. The reading is also a PURE strength claim: `own2` beats `own1` at
  `raw:t0+eo-1.5` with 64 slots, by the reading and its standard error.
- **Otherwise:** π_0 is `own1`.

The reference player stays `own1` either way: claims stay against `own1`
even if `own2` becomes the base (current_design.md section 7, ruling (c)).

## What each outcome means

- **Above +8.7, inside the window.** Round 2's training on its own
  trajectories made the 64-slot player stronger, as R1 read. π_0 is `own2`;
  S2's and S3's claims then each take a separate two-network match against
  `own1`.
- **From -8.7 to +8.7.** `own2` is not shown stronger at this resolution;
  R1's +16.5 was mostly chance, or the gain is under about 9 Elo. π_0 is
  `own1`, the base the coin path is priced on, unless the second half is
  played and the pooled reading picks `own2`.
- **Below -8.7.** `own2` is weaker at 64 slots. π_0 is `own1`. Down to -9.6
  the reading still agrees with the record.
- **Outside the window**, any of the above: the two readings disagree at two
  standard errors of their difference, which chance alone produces about
  once in twenty. The match path's changes between R1's stage (commit
  bd5f33d of `exp/memory-own-trajectories`) and this one's are listed with
  the result (`git log bd5f33d..<stage commit> -- tools wesnoth_ai rust
  configs`); the decision follows this match.
- **Cut or failed** (`OWN2_BASE_CUT`, `OWN2_BASE_FAILED` in `status.txt`):
  no reading. The games are on HF, and a re-entry resumes the match in its
  directory: the same stage with a larger `--env BOX_MAX_H=`, or a new
  rental with the same stage.

## Cost

Walls of 800-decisive-game matches with both sides at 64 slots and two
checkpoints, 20 games in flight, on RTX 4090 boxes with an EPYC 7702:

| match | sides | wall | box |
|---|---|---|---|
| R1, round 2 | `own2` against `own1` | 956 s, 801 games | 55454553, $0.463/h |
| C2, confirmation | `own1` against `parity3` | 796 s, 807 games | 55438108, $0.427/h |

(training/metrics/memory_own_round2_20261011/match.walls,
training/metrics/own_confirm_20261011/match.walls.) The estimate scales R1,
the same two checkpoints at the same decode:

| step | time | source |
|---|---|---|
| entry to the first game | 132 s | the confirmation: entry 19:49:00, GPU check passed 19:51:12, tests 48 s |
| the match | 2.12 h | 8 x 956 s; R1's servers ran 939 and 936 s of its 956, so counting the start-up 8 times instead of once adds at most 3 minutes |
| the fit | about half a minute | 4 s for R1's 801 games, times 8 |
| the final upload | a few minutes | the games' tarball about 40 MB: R1's 5.0 MB for 801 games, times 8 |
| the container's start before the entry | | not in the records |

About 2.3 box-hours: **$0.97-1.06** at $0.42-0.46 an hour. R1 itself cost
956 / 800 s x $0.463/h = $1.54e-4 a decisive game, above the design's
$1.17e-4 (an evaluation median, current_design.md section 2).

Bounds (in the script): an attempt starts no game after 165 minutes, 1.29
times the expected 127.5, and is cut 30 minutes later (a game's 20-minute
timeout, the servers' exit); a second attempt plays a short match once more
when the dead-man's switch leaves it at least 10 minutes of budget, after
the 30-minute grace and 15 minutes kept for the fit and the final round.
`BOX_MAX_H` 3.6, 1.57 times the expected box-hours, the least that holds
the budget, the grace and the reserve (3.5 h) after the bring-up: at most
$1.51-1.66 plus the final round. Identical match repeats have differed by
up to 1.8 times in wall (CLAUDE.md, Throughput); at 1.8 times, the budget
stops the match at about 4,600 decisive games (165 of 229.5 minutes),
recorded as `OWN2_BASE_CUT`, and a re-entry resumes it.

`rent_box.py create --hours 2.3` checks funds for 5.1 hours at the offer's
price (`BOX_MAX_H` plus 1.5, the longer of its two bounds): $2.14-2.35.

## Launch

    python tools/stage_code.py --out /tmp/stage_20261020_own2base.tar.gz \
        --script scripts/own2_base_box.sh \
        --upload tier-b/staging/stage_20261020_own2base.tar.gz
    python scripts/rent_box.py search --min-hours 5.1
    python scripts/rent_box.py create OFFER_ID --onstart own2_base_box.sh \
        --stage tier-b/staging/stage_20261020_own2base.tar.gz --hours 2.3 --disk 40
    # once ALL_DONE is on HF
    python tools/pull_box_records.py tier-b/own2_base_20261020 training/metrics/own2_base_20261020

## Measured (2026-10-20)

Box 56955126 (RTX 4090, EPYC 7702, 31 usable cores, $0.571 an hour), stage
`tier-b/staging/stage_20261020_own2base.tar.gz` (commit c16a46e), entry
21:36 UTC, match done 22:40 UTC, about $0.66; records in
`training/metrics/own2_base_20261020/` (the games' tarball stays on HF).
6,402 games, 6,400 decisive, 2 capped, no failure. The match took 3,533 s,
against 7,650 s estimated from R1's wall: this host played 2.2 times R1's
rate, so the cost came in under the estimate.

**Reading: `own2` +22.7 +- 4.4 Elo over `own1`** (PURE, decisive games,
`own2_vs_own1.fit.json`; the MATERIAL-SIGN diagnostic reads +22.6). The
lower 2-SE bound is +14.0, above 0, so the base π_0 is `own2`. The reading
lies inside the record's window (-9.6 to +42.6), in agreement with R1's
+16.5 +- 12.3. As a strength claim against the reference `own1`:
+22.7 +- 4.4 Elo.
