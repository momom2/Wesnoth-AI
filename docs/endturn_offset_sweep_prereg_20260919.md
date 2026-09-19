# Pre-registration: how far does "act more" go? (2026-09-19)

Follow-up to docs/endturn_rule_prereg_20260919.md, written after its
result and before this box is rented. There the end_turn logit offset
of -1.5 scored p 0.789 +- 0.014 against the reference over 800
decisive games, above the actor-level rule's 0.752, and the screens
read -0.75 at 0.733 and -1.5 at 0.744 over 30-39 decisive: the curve
was still rising at the largest offset tried.

## Question

Where does the offset curve peak, and does the reference lose
anything by never ending a turn while another legal action remains?

## Estimand

- Player A: `raw:t0+eo<x>`, the reference player (`relset`) at
  temperature 0 with the end_turn actor logit offset x in {-2.5, -4,
  -99}; -99 is "act while anything is legal" (end_turn is chosen only
  when it is the only legal action). Player B: `raw:t0`, the reference.
- Match: PURE, sides alternated, Ladder maps, the recommended eval
  path, max 200 turns, decisive results bought to 800 (`--games 800
  --max-extra-games 500`), seed bases 46000 / 46100 / 46200 (disjoint
  from every earlier match). No screens: the offset fires by
  construction.
- Headline per arm: p over decisive games with its standard error
  (0.018 at 800), the capped fraction, decisions per side-turn; the
  three arms next to the -1.5 arm of the first test (0.789, seed base
  43000).

Held fixed: the reference checkpoint, bf16 packed serving on cuda,
combat-oracle alphas 0, the joint-prior argmax for every non-end
decision.

## Reading

The largest p among {-1.5, -2.5, -4, -99} is the offset proposed to
the user as the reference decode, the adjacent arm within 1 SE
counted as a tie in favour of the smaller offset (the smaller change
to the imitation product's own end_turn frequency). A -99 arm within
1 SE of the peak says the prior's end_turn mass carries no
information the argmax needs; a -99 arm well below it says the
end_turn head still marks turns where every remaining action is
worse than passing.

## Predictions

-2.5: p 0.80 (range 0.76-0.84); -4: 0.79 (0.73-0.84); -99: 0.72
(0.60-0.82), the drop coming from forced attacks and moves out of
cover late in a turn. Capped fraction under 0.10 for all three.

Context read after the box was rented and before any result
(`tools/analysis/decisions_per_side_turn.py` over the first 3,000
manifest games, record `endturn_rule_20260919/corpus_decisions_per_side_turn.json`):
the corpus's human side-turn holds 7.50 decisions on average
(end_turn included; median 6), the winner's 9.60 and the loser's
6.00. The reference at argmax plays 6.0, the loser's rate; with the
offset -1.5 it plays 9.25, the winner's. So -2.5 and beyond move past
the rate of the players the product learnt from, which is where the
predicted drop comes from.

## Cost

Three 800-decisive matches at about 9 minutes each on the EPYC 7B13
box class of the first test, bring-up and wheel about 8 minutes: about
40 box-minutes, $0.50 at $0.74 per hour. `scripts/endturn_offset_sweep_box.sh`
runs it end to end and leaves ALL_DONE on HF on every exit.
