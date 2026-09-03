# Minimal self-play loop — spec for review (2026-09-03)

One page. Every sentence is a claim you can reject. Nothing here
runs until you approve it.

## The loop

Repeat:
1. Play N games in the simulator, both sides driven by the SAME
   network plus plain MCTS with S simulations per decision
   (PUCT; no Gumbel root, no tree reuse, no playout caps). Record,
   at every decision, the position and the visit count of each
   legal action.
2. When a game ends, label every recorded position of that game
   with the result from the side-to-move's view: +1 / -1; a game
   that hits the turn cap is discarded (no label, no policy
   target -- it is not a draw, it is unfinished).
3. Take ONE gradient step on all positions from this batch of
   games:
     loss = CE(policy, visit distribution) + c * (V - result)^2
   Adam proposes the update; the applied update is the largest
   fraction (1, 1/2, 1/4, ... 1/64) of it that lowers the loss on
   a fifth of the batch's games held out of the step
   (`tools/step_control.py`, backtracking line search). If none
   does, the step is skipped. The same shrinking also caps the
   value head's mean shift on those held-out states at two C51
   atoms (0.08, the trust-region delta of design_constants.md):
   held-out loss alone accepted a step that swung the level from
   +0.48 to -0.38 (same squared error on the other side of the
   label mean), and search turns a mover-frame level error b into
   a 2b act-vs-end_turn bias. Gradient-norm clip. No replay
   buffer, no per-game reweighting, no label smoothing, no
   auxiliary heads, no anchors, no memory, no extra channels.
   Why the line search: with fresh Adam moments the first update
   is lr * sign(gradient) on EVERY parameter, whatever the gradient
   size (measured 2026-09-03: 66% of 14.8M weights moved by exactly
   lr; the policy collapsed to K median 1 after that one step). A
   step bounded by held-out loss cannot overshoot that way, and it
   has no rate to pick.
4. Save the checkpoint. Every M iterations, play the raw network
   (no search) against the raw seed for G games, and the
   network+search against the seed+search for G games.

That is the whole algorithm (Silver et al. 2017, minus the
distributed parts). The closest thing this codebase has run is the
tier-a campaign (2026-07): MCTS visit-count targets, but with the
51-bin categorical value loss, shaping rewards, the combat oracle
and per-game weighting; it improved against its own lineage (+133)
and not at all against the external opponent. The form above --
scalar squared-error value, result labels only, nothing else -- has
not been run.

## What is deliberately absent, and why

- TCS turn-level targets: the profiler measured their targets
  0.3% away from the prior at every checkpoint. Visit counts are
  used instead because their sharpness comes from averaging
  repeated evaluations, not from a link gain.
- Replay buffer / multi-epoch updates: 16 updates per iteration
  multiplied every systematic tilt in the targets by 16.
- Auxiliary heads, GBC, anchors, value memory, grounding, trust
  region, distill-prior discount, draw tiebreak labels: each was
  added to fix a symptom of the loop above; none was ever shown to
  help in isolation. They are quarantined, not rejected.
- Turn-cap "draw" labels: labelling an unfinished game 0 flattened
  the value head in leg 3; discarding is the honest treatment.

## The two numbers that decide

- raw net vs raw seed (G games): does the prior absorb what search
  found? Prediction: rises above the seed within the budget.
- net+search vs seed+search: does the value head help or hurt the
  search? Prediction: does not fall below the seed.
Kill: raw pin <= seed at the end of the budget, or actions/turn
median < 3 for 3 consecutive iterations (the one tripwire kept).
  Threshold 3, not 10: under plain search, K 5-8 with most games
  decided is play, not collapse (step-scale measurement
  2026-09-03, `training/metrics/step_scale_20260903/`); K 1-2 with
  most games undecided is the degenerate state the tripwire is for.

## Constants to fix before running (your call on each)

- S (simulations per decision): proposal 32 -- the seed+search
  measurement that gave +321 used 32.
- N (games per iteration): proposal 24 (actor-pool sized).
- c (value loss weight): proposal 1.0 with the SQUARED-ERROR value
  loss on a scalar head -- not the 51-bin categorical loss. Reason:
  the profiler showed the value term at 99% of the gradient because
  the categorical loss charges a confident head heavily for a
  coin-flip label; squared error on the mean charges it in
  proportion to the miss.
- Learning rate: 1e-4 is only the proposal's scale; the applied
  step is set by the held-out backtracking above. Clip 1.0.
- Budget: 60 iterations (~1,440 games); pins every 10; G = 40.
- Box: one 4090-class host, ~$0.50/h, ~12-15 h -> ~$7.

## What would make me wrong

If the raw pin does not rise, then MCTS-32's play cannot be
absorbed by this net at this rate -- and the answer is not another
mechanism in this loop but more search compute or more human data
on the seed.
