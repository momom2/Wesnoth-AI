# Pre-registration: a simple-loop network at the reference's decode (2026-11-20)

**Status:** registered 2026-11-20, before the match. Results below when
it has played.

## Why

Every match of the simple loop so far has its network sampling at
temperature 1 against the reference at its pinned decode
(`docs/simple_loop_prereg_20261117.md`). That reading mixes the weights
and the decode. The start network reads +114.7 ± 6.5 Elo over `own2` at
the reference's decode (3,200 decisive games) and +39.2 ± 12.4 when it
samples, so sampling cost it about 75 Elo. The loop's sampled readings
have since risen by 50 to 95 Elo. That rise can be new skill, or the
network learning to play well when sampled, which the decode gave it
before. No simple-loop network has been played at the reference's decode.

The user approved the match on 2026-11-20 ("1) Approved.").

## What runs

- **The candidate:** the loop's network after round 208
  (`wesnoth_runs\simple_loop\round_0208\pi.pt`, SHA-256 `34358619…dc6aa`),
  the latest a match has read (+92.3 ± 12.7 sampling).
- **The match:** `scripts/match_local.py --label s0208 --checkpoint <that
  file> --out wesnoth_runs\match_s0208_pure`, under the ROCm venv with
  `HIP_VISIBLE_DEVICES=1`. It is a PURE match: both sides at the
  reference's decode (`raw:t0+eo-1.5`) and 64 memory slots, the candidate
  on side A, 800 decisive games from seed 900000 on the Ladder pool, fp16
  serving.
- **The simple loop is paused for it** (one heavy GPU job at a time): a
  STOP file ends the run after its current tool, and the same command
  resumes it when the match is done. Nothing of the loop changes.

## What is measured

The candidate's Elo over `own2` with its standard error, fitted on the
decisive games (`<out>\result.json`).

## The number that decides

D is the candidate's reading less the start network's +114.7. Its
standard error is the two in quadrature, about 14.5 with a match of 800.

- **D above +3 standard errors (about +44):** the loop improved the
  network's play at the reference's decode.
- **D below −3 standard errors:** it plays worse at that decode than the
  network it started from.
- **Otherwise no change is read.** One match of 800 cannot read a change
  under about 44 Elo, and the result says so.

It is a PURE match of at least 800 decisive games, the kind an adoption
has rested on, but nothing is adopted by it: that is the user's decision.

## What it does not show

- The loop's network was trained to sample with no end-of-turn offset.
  The reference's decode adds an offset of −1.5 to its end-of-turn logit.
  A network that has learned its own end-of-turn timing may be helped or
  hurt by an offset fitted on another network. A second match at plain
  argmax (`raw:t0`) would separate that, and is not part of this run.
- Round 208 is one network. Round 144 read highest when sampling
  (+134.8 ± 13.2) and may differ.

## Kill criteria

1. A display-driver reset stops the match (the tool does it).
2. Past 23 minutes, 1.5 times the README's 15, the match is inspected.

## Results

Not played yet.
