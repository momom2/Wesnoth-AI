# Pre-registration: the tail match and the continued rounds (2026-11-02)

Written 2026-11-02 at about 03:30, before any round of the local rounds has
played: the chain still waits on stage C's plant. The window's plan is
docs/autonomy_20261029.md, the rounds' design and departures are
docs/stage_c_rounds_local_prereg_20261030.md, and the final match is
docs/final_match_local_prereg_20261031.md. This answers two review
findings:
- the 2026-11-01 09:43 review: the trained player's own gain may go
  unmeasured;
- the 2026-11-01 21:43 review's M3: the GPU idles 24-36 h after the final
  match.

## Question

The final match reads the rounds' candidate where the reference plays: at
argmax with the end_turn offset −1.5. Its forecast is near 0, because the
rounds mostly sharpen the heads, and sharpening leaves an argmax choice
where it was. The trained player, though, samples from its policy
(Principle 3), and the rounds train it so.

**The tail match asks:** is the rounds' candidate stronger than `own2`
when both play as the rounds train, sampling at temperature 1 with no
end_turn offset?

## The candidate

It is the final match's candidate: the last round that adopted one, its
`pi.pt` SHA-256 checked against its `round.json` again before the match
(scripts/chain_local.py `candidate`). The tail plays only once the final
match has returned 0.

## The match

    <rocm venv>/python scripts/match_local.py --t1 --label final_round<k>t1 --checkpoint <round_k>/pi.pt ^
        --out C:/Users/momom2/wesnoth_runs/tail_match --games 12800 --seed-base 950000 ^
        --tools-python <.venv>/python --keep-awake

- **Procedure: `raw:t1` on both sides.** The flags are `--mcts-sims 0
  --raw-temperature-a 1 --raw-temperature-b 1`, and neither side has an
  end_turn offset. Both sides play 64 memory slots.
  - Side B is the reference's checkpoint (`tools/reference_player.py
    --flags b`), relabelled `own2t1` with its offset dropped
    (match_local's `sampled`).
  - It is the decode of stage 1's self-play and of D0's T=1 side
    (scripts/selfplay_t1_local.py `player_flags`).
- **Serving:** the local fp16 server. These games pool neither with a
  box's matches nor with the final match (another procedure); elo_collect
  refuses mixed estimands in one outdir anyway.
- **Games:** 12,800 decisive, with capped games replaced (at most 1,500).
  Stage 1's 30,000 T=1 self-play games capped none.
- **Seeds: 950,000-962,799,** and replacements at multiples of 1,000,000
  above them.
  - No script or document names a seed from 950,000 to 964,299 (grep,
    2026-11-02).
  - The final match takes 900,000-914,299 at most. The fresh seeds keep
    the two matches' maps, factions and luck independent.
- **Fit:** tools/elo_collect.py PURE, decisive games only, with `own2t1`
  anchored at 0.

## Expected

- **Standard error:** about 3.1 Elo, as the final match's (12,800 decisive
  games near an even score).
- **Wall:** about 2.8-3.7 h.
  - D0 played 12,800 T=1 games against the reference in 2.78 h (one
    network on the server).
  - Stage 1's self-play ran about 4,340 games an hour.
  - match_local's own estimate assumes 3,450 an hour, which gives a time
    budget of 5.9 h.

## Forecast

**The forecast is the rounds' own.** It is ELO_PER_Z (347.4) times the sum
of the adopted rounds' confirmed readings: half B's ΔĴ of each adopted
rung, or its cross-play estimate.
- That is the claims' forecast rule (scripts/rounds_local.py `stops`).
- Its standard error is ELO_PER_Z times the root sum of squares of the
  readings' standard errors.
- Under transitivity the tail estimates the claims' `claim − D0` directly,
  about 2.5 times finer: a combined 7.8 Elo for a claim against D0, 3.1
  here.

The rule is fixed here. Its number is a function of the rounds' records
alone (the readings rounds_local's `confirmed` returns), and it is written
beside the result.

## Decision rule

| PURE Elo of the candidate at T=1 | Reading |
|---|---|
| above +2 SE | the rounds made a stronger sampling player than own2, by X ± SE |
| within ±2 SE | no difference resolved at this size |
| below −2 SE | the rounds made a weaker sampling player, by X ± SE |

- **Against the forecast:** the result's distance from the forecast is
  reported with the combined standard error.
  - More than 2 SE below the forecast: the confirmed gains do not transfer
    to play against own2. That is the design's §6 lower tail, read on the
    final candidate.
  - More than 2 SE above: play gains more than the labels price.
  - Either is reported, not acted on: the window ends.
- **Adoption:** this match measures, it does not adopt. The reference and
  its procedure stay the user's (CLAUDE.md, "Eval procedure"), and the
  standard claim remains the final match.

## The continued rounds

**When they run:** once the tail returns 0, and only if the rounds returned
0. A design stop, a wait or a failure ends the chain there, for the lead
or the user to read.

**What runs:** rounds_local again with the same arguments, resumed from its
markers.
- `--rounds 39` is MAX_ROUNDS, the seed blocks' limit.
- `--end-by "2026-11-08 12:00"` replaces the first run's Nov 7 10:00.
- Each round runs under the rounds' prereg as it stands (departures through
  28), its stops included, with a claim every third round.

**A resume fix comes with it.** rounds_local's `timed` rewrote a finished
round's end at every restart.
- So a resume read every finished round's wall as reaching the restart, and
  `--end-by` refused the next round. A 10-hour pause made a 4-h round read
  22 h.
- Since this branch, the first start and the first end of a round stand.
  This also mattered to any mid-rounds re-arm, the departure-28 fallback
  included.

**The continuation's final match:**
- **When:** a round after the final match's adopted a candidate.
- **What:** the standard match plays it into
  `C:/Users/momom2/wesnoth_runs/final_match_cont`, by the final match's
  procedure, seeds and size: 12,800 games from 900,000. The two candidates
  then play paired games.
- **Without a newer candidate:** nothing more is measured
  (CHAIN_NO_CONTINUE_FINAL).
- **Reporting:** it is reported beside the final match, never pooled with
  it.

**Timing:**
- A round starts only while 1.2 times the longest round fits before the
  deadline, so the continued rounds end by about Nov 8 12:00.
- The continuation's final match takes about 3.8 h (budget 5.9 h). The GPU
  work ends by about Nov 8 16:00-18:00.
- That is before the heartbeat and review crons expire (about Nov 8 22:50)
  and in time for the user's return.
- A STOP file in `rounds_local/` or in a match's directory ends it sooner.

## Kill criteria

- **A display reset** stops a match (match_local exits 5). The chain then
  plays nothing more.
  - The lead runs the GPU health check before any restart, and never two
    faulted starts in a row.
- **A short match** is played again once, as match_local does. A second
  short one is reported short, with its games fitted and the shortfall
  stated.
- **A continued round past 1.5 times its estimate** is inspected and cut
  (the 2026-09-05 rule).

## Measured (2026-11-03 19:28)

The rounds stopped at round 3's claim, and the final match returned 0, so
the chain played the tail on round 3's candidate (`round_03/pi.pt`,
6c4e96f81ade...), 16:47-19:28: 12,800 decisive games from seed 950,000, no
capped game.

**final_round3t1: +134.1 ± 3.3 Elo against own2t1** (PURE, 8,755 won and
4,045 lost; `raw:t1` on both sides, no end_turn offset, 64 memory slots
each, local fp16 serving). **By the decision rule: the rounds made a stronger
sampling player than own2, by 134.1 ± 3.3.**

- **Against the forecast:** the rounds' confirmed readings sum to 0.4077
  (round 1's cross-play 0.1612 ± 0.0099, half B 0.1378 ± 0.0174 and
  0.1087 ± 0.0117), so the forecast is 347.4 × 0.4077 = +141.6 ± 8.1. The
  tail reads 7.5 Elo below it, with a combined SE of 8.7: -0.9 SE. **The
  confirmed gains transfer to play against own2 at T=1.** Neither tail of
  the rule fires.
- **Beside it, never pooled:**
  - the round-3 claim against the reference at its own decode: claim - D0 =
    +118.4 ± 10.7, which fired the transfer stop at 2.2 SE below the
    forecast;
  - the final match at the reference's decode: +31.8 ± 3.1.
  - Under transitivity the tail and claim - D0 estimate the same number.
    They differ by 15.7 Elo, with a combined SE of about 7.7 (2.0 SE), so the
    rounds' gain reads somewhat smaller against the reference at argmax than
    against it sampling. The tail does not test that difference; it is
    reported, not acted on.
- **Adoption:** this match measures; the standard claim is the final match.
