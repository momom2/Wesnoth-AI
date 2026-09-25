# Pre-registration: a value function that ranks candidate turns under obs8 (2026-09-25)

Written before any code for it runs on a box. Approved in principle by
the user on 2026-09-24 ("Turn-ranking value net: Approved in principle")
and ordered on 2026-09-25 ("Proceed").

## Why

Phase 2's teacher is a player that proposes a few whole turns, grades
them, and plays the best (docs/plan_20260904.md 5). Better turns exist:
under the reference, 7 of 60 positions have a sampled alternative turn
better by at least 0.25 in expected outcome
(docs/turn_gap_ref_prereg_20260921.md). Grading by playouts costs
$0.006-0.013 per side-turn, so an 800-game gate of such a player costs
$50-300 (docs/turn_proposer_design_20260905.md); the teacher that can
be matched needs a grader that costs a forward pass. The value head we
have does not rank candidate turns: correlation 0.07 with the playout
mean, within-position residual 0.323 +- 0.059, 2 of 8 winning
alternatives ranked below their base (0.4.2). It was trained on one
human outcome per game through the policy's trunk, which is the likely
cause.

## Question

Does a value function trained on outcome contrasts between candidate
turns of the same position, under `obs8`'s own play, rank candidate
turns well enough to pre-grade them?

## Design

**Positions.** Turn starts of either side from the 800 recorded games
of `obs8` against `terrain` (HF
`tier-b/observation_retrain_20260924/games_obs_e1_vs_terrain.tar.gz`),
rebuilt from their records with the fingerprints checked
(tools/game_record.py `turn_starts`). From turn 2 on the games offer
41,328 turn starts (11 to 199 per game, median 47); up to 15 per game
are drawn with a seed of the game's own, 11,980 in all. Games are split
by a hash of their file name: 630 games (9,438 positions) are fit, 70
(1,046) decide the fit's early stopping, and 100 (1,496) are the
held-out proxy set.

**Candidates.** At each position, as the turn-gap tool proposes them
(tools/turn_gap.py): `obs8`'s own turn at its decode
(`raw:t0+eo-1.5`), two turns sampled at temperature 1, and one
"continue" edit (the base turn without its end_turn, plus one more
argmax action). Candidates whose resulting position equals an earlier
one are dropped. Generation: tools/turn_value_data.py.

**Labels.** One playout per candidate to the end of the game, both
sides `obs8` at `raw:t0.5+eo-1.5`, capped 30 turns after the position
(the turn-gap reference procedure, so the labels and the validation
truth below measure the same thing): +1 win, -1 loss, 0 draw or capped,
from the mover's side.

**What the grader reads.** The position after the candidate turn's last
action and before its end_turn, encoded from the mover's side by
`obs8`'s encoder: exactly what the mover observes when it is about to
end the turn. The existing `value_post` is read after the end_turn,
from the opponent's side, which includes what the opponent sees and the
mover does not; a grader the mover plays with must not. Each candidate
records the commands, recruit rejections and digest of that position
(`pre_end_turn`, tools/turn_gap.py); tools/turn_value.py rebuilds it
from the boundary, checks the digest, and caches `obs8`'s global token
for it. A candidate turn that ends the game is graded by its outcome.

**Arms.** The policy stays byte-identical; the grader has its own
weights and never sends a gradient into the policy.
- A: a linear map from `obs8`'s frozen global token (384 values) to a
  value.
- B: a head of the shape of `obs8`'s value head on the same frozen
  token, initialized from it.

Both fit on the cached tokens (tools/turn_value_fit.py), by squared
error against each playout outcome plus the same error on
within-position centered predictions and outcomes (the ranking term,
weight 1). A: closed-form ridge on standardized tokens, the strength
chosen from 1e-6 to 10 on the early-stopping games. B: AdamW (lr 3e-4,
weight decay 0.01, 128 positions per batch), up to 40 epochs, the epoch
with the lowest early-stopping loss kept (epoch 0 is `obs8`'s head
itself). A fine-tuned copy of the trunk (arm C) is not in this
registration; it is proposed only if A and B fail while the proxy
shows signal.

Reported beside the arms and not judged: `obs8`'s own value head on the
same pre-end_turn state (read offline, `value_reference`, and while
playing, `value_pre`), the existing `value_post` and HP margin, and
the null grader.

## Validation set (the verdict's data)

The 2026-09-23 procedure under `obs8`: tools/turn_gap.py `--reference`
on the first 60 positions of `configs/bench_states.json` (holdout side-2
turn starts), a sequential screen of the base and four samples at
temperature 1 (up to 40 playouts), then the confirmation of each
nominal hit (base and best alternative, up to 160 fresh playouts), seed
25. The grader reads each candidate the way it reads training
candidates. Primary file: the confirmation; secondary: the screen.
These are side-2 turn starts of human games, where the training
positions are turn starts of `obs8`'s own games: a grader that passes
here ranks turns outside the distribution it was fitted on.

## Bars (per arm, on the confirmation file)

The check of docs/turn_gap_ref_prereg_20260921.md ("Pre-grader check"),
with one addition:
- **Pass:** within-position residual SD <= 0.2 with its 2-SE upper
  bound below 0.3; every alternative whose playout gap to its base is
  >= 0.25 ranked above its base; and the residual SD below the null
  grader's (one constant per position) by at least one SE. The addition
  is new: where the true differences between candidates are small, a
  constant can meet the SD bar alone.
- **Fail:** residual SD >= 0.3, or any such alternative ranked below its
  base.
- **Inconclusive** otherwise.
- **Undecided:** fewer than 4 alternatives with a gap >= 0.25 in the
  confirmation; the ranking half cannot be judged and the screen file
  is read as secondary evidence only.
- **Crash barrier, not a verdict:** on the held-out proxy set, the
  correlation between predicted and observed within-position outcome
  differences must be positive at 2 SE. If it is not, the data carry no
  signal the arm can use at this size, and the verdict is not read.

## Predictions

Arm A: residual SD 0.25 to 0.32, P(pass) 0.15. Arm B: 0.22 to 0.30,
P(pass) 0.25. Proxy barrier: passed by both arms, with a within-position
correlation of 0.03 to 0.10. The token was learned to predict human
actions, not to separate turns, and one playout per candidate is noisy:
a playout outcome has an SD near 0.9, so a within-position difference
of 0.1 has a signal-to-noise ratio near 0.08 per candidate pair, and
the 9,438 fit positions give about 25,000 independent contrasts. The
HP margin, a summary the token certainly carries, ranked 3 of 8 large
gaps below their base on 2026-09-23. A fail is more likely than a
pass; the data and the validation set carry over to arm C either way.

## Consequences

A pass makes the cheap teacher buildable: propose, grade with the head,
play the best, gated by an 800-game match against `obs8` (its own
pre-registration). A fail with signal on the proxy proposes arm C; a
fail without signal says the grader needs a different target or much
more data, and the turn-search route is re-priced before anything else.

## Cost

One single-tenant RTX 4090 host, 32 or more effective cores, at about
$0.55-0.70 per hour; the balance is checked before renting.
- Validation set: about 1.25 h (2026-09-23: 7,810 screen playouts in
  2,216 s and 3,160 confirmation playouts in 2,259 s at 24 workers).
- Training data: 11,980 positions x up to 4 candidates, about 46,000
  playouts at about 3.5 a second, about 3.7 h.
- Bring-up, the tests against the built wheel, features (about 46,000
  rebuilt turns through the frozen trunk) and the fits: about 0.5 h.

About 5.5 box-hours, $3.1-3.9. The script (scripts/turn_value_box.sh)
runs the stages in that order, uploads every stage's records as it
goes and the growing data log every 30 minutes, cuts each stage at
twice its estimate (ceiling about 11 box-hours, $6.2-7.8), and stops
the instance at the end (`stop_self`).
