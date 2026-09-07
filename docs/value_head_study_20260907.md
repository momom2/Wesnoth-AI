# Value head against human outcomes, by game phase (2026-09-07)

The user's plan (2026-09-06): measure the value head on the human
corpus before any search or training design depends on it, measure
material the same way, then train the head with material as an
input and see whether it improves. This note holds the first two
measurements and the corpus facts they surfaced.

## Method

Holdout games of the imitation manifest (369 games, never trained
on). At every side's turn start (the state after its init_side, what
the mover faces), the label is whether the side to move went on to
win. Predictors, both from the mover's side: the value head's
expected outcome, and material = sum over the mover's visible units
of cost x HP fraction minus the same over the visible enemy units
(the encoder's own fog filter, wesnoth_ai.visibility).

Same-turn AUC: for each game and turn where both sides' turn-start
states exist, does the predictor rate the eventual winner's state
above the eventual loser's? Averaged per game inside a turn bucket,
then across games; error bar between games. The pooled AUC of the
training evals (0.78 for the seed) mixes turns and rewards knowing
that late positions are decided; this comparison is like for like.
Tool: `tools/analysis/value_head_by_phase.py`; records in
`training/metrics/value_head/`.

## Result: the seed's head, fog-aware corpus (box 50184541, $0.35)

| turns | states | games | same-turn AUC head | same-turn AUC material | pooled AUC head | pooled AUC material | Brier head |
|---|---|---|---|---|---|---|---|
| 1-5 | 3679 | 369 | 0.647 +- 0.018 | 0.563 +- 0.015 | 0.655 | 0.536 | 0.248 |
| 6-10 | 2869 | 337 | 0.767 +- 0.018 | 0.685 +- 0.018 | 0.813 | 0.680 | 0.201 |
| 11-15 | 1482 | 195 | 0.818 +- 0.024 | 0.765 +- 0.025 | 0.870 | 0.798 | 0.175 |
| 16-20 | 645 | 83 | 0.838 +- 0.034 | 0.838 +- 0.030 | 0.904 | 0.866 | 0.152 |
| 21-30 | 343 | 28 | 0.874 +- 0.051 | 0.849 +- 0.044 | 0.860 | 0.822 | 0.179 |
| 31+ | 149 | 8 | 0.932 +- 0.056 | 0.937 +- 0.031 | 0.961 | 0.900 | 0.093 |

Readings:
- The head reads who is ahead, and better than material in the
  early and middle game (turns 1-15: +0.05 to +0.08 AUC over
  material, 3 to 4 SE); the two are equal from turn 16. Its Brier
  score falls from 0.25 (no information) at the start to 0.09 late.
- Half an epoch more of the seed's recipe left the pooled AUC
  unchanged (0.784 -> 0.785 control, 0.781 relevant-set arm; the
  per-phase curve of those arms is not measured).
- These are human-game outcomes under human continuation, not
  optimal play; the early buckets are near the ceiling any predictor
  has on undecided games.
- This measures "who is ahead", not "which of two turns is better".
  The pre-grader run of 2026-09-05 (12 positions, 160-playout truth)
  is the only within-position evidence, and it rests on one position
  where the head misranked a base blunder. "The value head is bad"
  is not established by either measurement.

## Corpus facts found on the way

- The corpus recorded no fog or shroud setting; the encoder assumed
  fog on for every replay. Parsed from the raw replays' [side]
  blocks (`training/metrics/value_head/corpus_fog_shroud.json`):
  fog on 13,784 games (80.5%), fog off 3,230 (18.9%), shroud on 110
  (0.6%, all on standard ladder maps). Ruling: shroud counts as fog;
  fog off with shroud on (20 games) is quarantined. The extractor
  records both flags per side, `replay_dataset.fog_on_for` sets the
  encoder's switch from the two player sides (a scenery side 3 has
  no attribute and must not decide it), the builder and
  `tools/annotate_corpus_fog.py` apply the rule; the corpus is
  annotated (17,104 games, 72 fog-off holdout games).
- The fog-aware encoding moved the seed's holdout numbers by less
  than their error (CE 2.838 -> 2.839, pooled value AUC 0.784 ->
  0.787): the head's reading is robust to seeing the extra enemies.
- A first run of this study measured a random-init net for two hours
  because the checkpoint path did not exist and the loader fell back
  silently. `_load_policy` now refuses a missing path.

## Next, per the plan

1. Train the head with material as an explicit input against the
   identical recipe without it; judge by this per-phase table
   (about $1.5, half an epoch). Not started; needs the input wiring.
2. With a head that reads positions, the self-play design discussion
   (a new turn-level design, or reviving the turn-commit search).
3. Imitation scaling: the fitted player ratings (top quartile +147
   Elo) as a filter, and more replays from the server.
