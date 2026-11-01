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

## Forecast (2026-10-31 22:40, before stage C's gate reads)

Expected: **near 0, within ±2 SE**, and if anything slightly negative.
- Stage C's half-A readings at the same KL:
  - heads_only reads +0.1097 ± 0.0076;
  - sgd +0.0128, reduced_adamw +0.0124, muon +0.0147 (each ± about 0.0036).
- The four head temperatures alone (stage_c.json's span: its slopes, and
  its Fisher from the pass) reach √(2·eps·sᵀF⁻¹s) = √(2 · 0.001334 · 3.449) =
  0.096 at that KL, about 88% of heads_only's first-order gain (the 2026-10-31
  evening review).
- So the rounds will mostly sharpen the heads. Sharpening leaves an argmax
  choice where it was, so a match at argmax sees only the rest.
- Sharper actor logits also shrink the fixed −1.5 end_turn offset's relative
  effect, and −1.5 is at the offset curve's peak, so the rest leans
  negative.
- The trained player samples from its policy (Principle 3). Its reading is
  the round-3 T=1 claim against the reference, reported beside this match
  and never pooled with it. This match answers the standard question: is
  the policy stronger where the reference plays, at argmax.

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

- **The launcher starts it when the rounds end** (scripts/chain_local.py
  `--final-out`, 6e5a4286: after rounds_local returns 0 or 3, on the last
  adopted round, its SHA-256 checked against its round.json). The 2026-10-31
  evening review projects that at about Nov 6 08:00. Starting there rather
  than at the deadline leaves time to play a faulted match again.
- **Since 2026-11-01 the rounds run to a deadline** (rounds_local
  `--end-by "2026-11-07 10:00"`, with heads-only passes: the stage C and
  rounds prereg, departures 21-22). The match starts when they end, which
  is no later than about Nov 7 10:00.
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
