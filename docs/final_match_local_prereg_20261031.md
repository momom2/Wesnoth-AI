# Pre-registration: the autonomous window's final match against own2 (2026-10-31)

Written 2026-10-31, before any round of the local rounds has played
(docs/stage_c_rounds_local_prereg_20261030.md; the window's plan is
docs/autonomy_20261029.md). Sized by the 2026-10-31 morning review.

## Question

Is the policy the window's rounds leave stronger than the reference, `own2`
at `raw:t0+eo-1.5` with its memory at 64 slots, under the standard
procedure (CLAUDE.md, "Eval procedure")? The user returns about 2026-11-08
to this one number.

## The candidate

The rounds' last adopted policy: `policy_after(last round)` in
scripts/rounds_local.py, `round_<k>/pi.pt` for the last round k whose
`round.json` names a candidate. Its SHA-256 is recorded in the match's
records.
- If no round adopted a candidate, nothing is measured: the rounds' own
  records are the window's result.
- P's θ′ (stage C's `planted.pt`) is a planted test step, not an
  improvement, and is never the candidate.

## The match

    set HIP_VISIBLE_DEVICES=1
    <rocm venv>/python scripts/match_local.py --label final_round<k> --checkpoint <round_k>/pi.pt ^
        --out C:/Users/momom2/wesnoth_runs/final_match --games 12800 --seed-base 900000 --keep-awake

- **Procedure:** match_local's standard match. Side B is the reference by
  `tools/reference_player.py --flags b`. Side A is the candidate at the
  same decode and memory. Both sides are raw players at argmax:
  `raw:t0+eo-1.5`.
- **Serving:** the local fp16 server (`infer_fp16` recorded), so these games
  never pool with a box's bf16 matches.
- **Games:** 12,800 decisive games, with capped games replaced as
  match_local does (`--max-extra-games` 1,500).
- **Seeds:** from 900,000. No record names a seed from 900,000 to 999,999
  (match_local's docstring, checked 2026-10-29), and no local match has
  run at that base since.
- **Fit:** `tools/elo_collect.py` PURE, decisive games only, the reference
  anchored at 0.

## Expected

- **Standard error:** about 3.1 Elo. `own2` against `own1` read ±4.4 over
  6,400 games (docs/own2_base_prereg_20261020.md), and the standard error
  scales as one over the square root of the games. The smallest effect read
  at 2 SE is therefore about 6 Elo.
- **Wall:** about 3.8 h: match_local's estimate at 3,450 games an hour,
  plus 5 minutes. Its time budget is 5.9 h (1.5 times the estimate plus 10
  minutes).

## Decision rule

| PURE Elo of the candidate | Reading |
|---|---|
| above +2 SE | stronger than own2 by X ± SE |
| within ±2 SE | no difference resolved at this size |
| below -2 SE | weaker than own2 by X ± SE |

- Adopting a new reference is the user's decision: this match measures, it
  does not adopt.
- No teacher is distilled from the candidate in this window.

## Timing and kill criteria

- **Start no later than 2026-11-07 12:00**, so it ends before the user's
  return. At that hour the rounds get a STOP. A step still running then is
  cut, since the window ends anyway; its loss is recorded.
- **A display reset** stops the match (match_local exits 5). The lead runs
  the GPU health check before any restart, and never runs two faulted
  starts in a row.
- **A short match** (a failed game, the budget) is played again once, as
  match_local does. A second short match is reported short, with its games
  fitted and the shortfall stated.

## What is not done

The design's T=1 claim minus D0 (D0 reads own2t1 at −415.2 ± 5.5) would
have a combined standard error of about 7.8 Elo. No command runs a lone
claim outside the rounds' every-third-round claim, and the standard
procedure above resolves about 2.5 times finer. The round-3 claim, if
played, is reported beside this match, not pooled with it.
