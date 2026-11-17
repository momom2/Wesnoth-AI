# The training mechanisms, judged on theory (2026-11-11)

The user asked for this after the audit
(`docs/training_dynamics_audit_20261111.md`, whose item numbers are used
here): "If we test everything one by one, we're never gonna get it
sorted. What I want is theoretical considerations. For each mechanism,
consider how it works, whether it's well-founded, whether it's
principled."

Nothing here was measured today. Each verdict says how firm it is:

- **theorem**: it follows from the policy-gradient theorem or a standard
  result, given a stated condition;
- **standard practice**: no derivation, but the mainstream choice with a
  known rationale;
- **judgement**: my reading, where theory gives a direction and not a
  proof.

## The yardstick

Three results do most of the work.

1. **The policy-gradient theorem.** The expected update is unchanged when
   a decision's weight, the outcome z, has something subtracted from it
   that does not depend on the action taken at that decision. Two kinds
   of thing qualify: any function of what happened before the decision,
   and any chance event whose average is zero whatever was decided.
   Subtracting them changes the noise, never the direction on average.
2. **The gradient is local.** It says which way is up at the current
   weights and nothing about how far that holds. Theory gives only loose
   sufficient conditions for a safe step, not usable step sizes.
3. **Noisy gradients favour many small steps.** Past the "critical batch"
   (the number of games at which a step's noise equals its signal), more
   games per step buy almost nothing; more steps do.

## A. The signal

A decision's weight is `z − b − β·S − α·v_t − γ·β·x_t`. Writing S as the
luck before the decision (x_t) plus the luck from the decision on (y_t),
this is `z − b − α·v_t − β·y_t − (1+γ)·β·x_t`. With γ ≈ −0.78 the last
term is small. So the five pieces amount to one idea: **subtract what the
position already promised, and the dice still to come.**

| # | How it works | Well-founded? | Verdict |
|---|---|---|---|
| A3, A5 | Each attack's luck is the realised result less its exact average over the dice, given the position and the attack chosen (`tools/match_luck.py`, after AIVAT, a published method). | **Theorem**, given that the averages are exact, which the core's enumeration is built for (not re-checked). Luck after a decision has zero mean whatever was decided; luck before it is part of the past. | Sound. Keep the idea. |
| A4 | Subtracts the value head's estimate at the decision, logged during play. | **Theorem**: it is a function of the past only. It stays unbiased even when the estimate is stale or wrong; it just removes less noise. | Sound. Keep. |
| A2 | Subtracts the average result for the map, seat and factions. | **Theorem**, but redundant: the value head sees the map, seat and factions too. It removes under 2% of the noise by itself. | Sound and redundant. Drop. |
| A5's γ | A separate coefficient that adds past luck back, because A3 subtracted the whole game's luck and A4 already accounts for the past. | Sound, and an artefact of the parametrisation: it undoes part of A3. | Fold into one form: `z − α·v_t − β·y_t`. |
| A6 | Switches A4 and A5 on or off each round by a significance test. | **Unfounded.** A fitted coefficient already goes to zero when a term is useless; a switch on top only makes the weight's form jump between rounds. | Drop the switch. |
| A1 | Holds 20% of the games out so the coefficients are fitted on games the round does not train on. | The condition is real: coefficients must not depend on the games they weight. But it is met exactly, at no cost, by coefficients fitted on **the previous round's** games (the design itself fitted them on the previous round's held-out set). And with about ten coefficients against thousands of games, fitting on the same games would bias the update by a fraction of a percent. | **Unfounded as built.** Train on every game. |
| A7 | A game with no winner counts 0. | Sound for a zero-sum game. | Keep. |
| A8 | Up to 10 games a chunk may be lost. | A guard. Dropping timed-out games would bias toward short games if it happened; it does not. | Keep as a guard. |

Two further points on the signal:

- **The whole-game outcome is used for every decision,** with no discount
  and no bootstrapping from the value head. That is the unbiased choice
  (theorem). Bootstrapping would bias the update through a value head
  that nothing trains.
- **Nothing trains the value head** (the audit, A4). Once the trunk
  moves, its estimate will drift and A4 will remove less and less noise.
  It stays unbiased. A value loss on the value head alone, cut off from
  the trunk, would keep it sharp without touching the policy's dynamics
  (judgement; not needed to start).

## B. The gradient pass

| # | How it works | Well-founded? | Verdict |
|---|---|---|---|
| B1 | Only the heads' final maps got a gradient. | A restriction to some weights still points uphill, so it is not wrong. But nothing in theory says to freeze the rest; the reason given was noise, measured under the KL rule. | **Unfounded** as a standing rule. Already off. |
| B2 | The gradient through the memory is cut every 16 decisions, at boundaries that follow a clock and not the game. So a decision sees between 0 and 15 decisions of its memory's history, 7.5 on average. | **Biased, by a known mechanism.** Truncation is standard practice when what the memory carries is short-lived next to the window. Here the one thing known about the memory is a pattern held for a hundred decisions or more. The gradient for "what to write" therefore counts the next few decisions and is blind to the rest. That blindness is the trunk account's own link between the pass and the divergence. | **The weakest point left.** See below. |
| B3 | fp16 with an exact power-of-two loss scale. | Numerics. The scale is exact; the rounding is small. | Fine. |
| B4 | A replaced action is trained as ordered. | **Theorem**: the policy chose the order, and what the simulator does with it is part of the environment. | Sound. |
| B5 | Checks that abort the pass. | Guards. The replay identity is exact and cheap. | Keep; restate the limit's reason, which names the removed radius. |

**On B2.** Three options, in order of soundness:

1. Windows that start with a game-side and run to its end (about 250
   decisions against 16): the true gradient. Whether that fits the 8 GB
   card is not known; a whole split in one window does not (departure
   33a). It would also change C1's statistics, which are taken per
   window.
2. No gradient through the memory at all: the update then answers a clean,
   smaller question, "how to act given this memory", and leaves the
   writing as it is.
3. Today's 16: neither. Its bias depends on where a clock happens to cut.

The match test's direction was computed under option 3 and still gained,
so the bias is not fatal. It is the mechanism most likely to matter over
many rounds, since every round pushes the memory by a gradient that
cannot see the push's long-run effect (judgement).

## C. The direction and the step

| # | How it works | Well-founded? | Verdict |
|---|---|---|---|
| C1 | Each weight's gradient is divided by the root of its mean squared per-window gradient: the weight moves by how consistent its gradient is, not by how large. | **Standard practice.** This is Adam's rule, and Adam is what transformers are trained with, because raw gradients differ by orders of magnitude between kinds of layer. The plain gradient is the less defensible choice for this network. | Keep. (This reverses my advice of this afternoon to test it first.) |
| C4 | The heads' direction is averaged with the 8 rounds before, each scaled to unit norm, equal weights. | **Sound in spirit:** it is momentum, the standard answer to a noisy gradient. The form is home-made: unit-norm scaling throws away each round's confidence, the weights are flat, and it covers the heads only. | Replace by ordinary momentum over every weight. |
| C5 | (Built today.) Each group's direction is scaled to unit norm, then to a fixed length. | **Unfounded.** A step of constant length does not shrink when the gradient stops agreeing with itself, so near an optimum it walks at full speed along noise. It also is not what the match tested: the winning candidate was a learning rate times C1's direction. | Replace by a learning rate times the direction, with no rescaling. |
| C3 | Two groups of weights with their own sizes: the heads' final maps and the rest. | **Judgement.** One learning rate is the plain choice. Two is the smallest structure that fits what the match showed: the final maps take a step about 50 times larger per weight than the rest. A crude stand-in for a curvature-aware method. | Keep two learning rates; drop the special list of "head maps" if one rate turns out to serve. |
| C2 | The arm was chosen once by a comparison at equal KL. | Its metric was the KL rule's. | Void; already gone. |

**C1, C4 and C5 are three partial rebuildings of one standard thing:
Adam with momentum and a learning rate.** The design left momentum out
because each step was sized and certified on its own (and so that the
cosine between consecutive directions would read oscillation directly).
With the sizing and the certification gone, the first reason is gone.
Ordinary Adam across rounds is the best-studied optimiser there is, which
is the opposite of dynamics that are hard to analyse.

One detail to fix on the way: the direction's scale carries an accidental
factor, the number of windows per game-side (`wesnoth_ai/pg_step.py`
`second_moment`: "its scale is free ... a constant factor the KL search
removes"). Under a learning rate nothing removes it, so it should be
divided out.

## D. Step size and adoption (removed)

| # | Well-founded? |
|---|---|
| D1 | **Principled in origin, mis-specified here.** Limiting how far the policy moves is the textbook answer to "the gradient is local" (trust-region methods). But the guarantee behind it is a loose sufficient condition, and the rule treated it as necessary. It also sized every step on a mean that 1% of the game-sides dominated. The match shows the harm it guarded against did not occur at 8 times its limit. A learning rate with matches as the guard is the mainstream alternative. |
| D2 | **Unfounded for training.** The cap served the offline evaluator's variance, and bound the training step in most rounds. |
| D4 | Sound statistics that were too weak to decide anything: 17 of 17 steps adopted. |
| D5 | **Sound.** Stochastic gradient methods do not validate each step; a periodic check suffices. |

## E. Round size

**Unfounded now, and probably the largest inefficiency left**
(judgement, on a standard result).

- The 5,000 came from a cost formula with a large fixed cost per round:
  the held-out set, the ladder's forwards and the claims. Those are gone
  or shrunk.
- The critical batch reads 281 games for the heads and 44 for the whole
  network. A 5,000-game round is 18 and 113 times that.
- Past the critical batch, a step on a quarter of the games is nearly as
  good, and there are four of them. With momentum to carry the noise,
  rounds of about 1,000 games should learn several times faster per game.

## F. Gates and stops

| # | Well-founded? | Verdict |
|---|---|---|
| F1, F2 | These test identities that hold if the code is right (the sample matches the law, the luck terms average zero). As tests of the code they are sound. As gates on every round they must fire falsely at their designed rate, 1% a round, and each firing costs hours. | Run them as tests and log them in rounds; do not stop on them. |
| B5's replay identity | Exact, cheap, and it catches the one thing that would silently corrupt a round: a pass that replays a different policy from the one that played. | Keep as a gate. |
| F3, F4 | They compared a forecast with a claim. With no step read before it is taken, there is no forecast. | Moot. |
| F5 | A periodic strength check with arbitrary margins. | Keep one periodic match as the only guard. |

## G. The play side

| # | Well-founded? | Verdict |
|---|---|---|
| G1 | **No result links the two players.** The update improves the player that samples. That its top-choice version improves too is usual and not guaranteed; here it has read a quarter to a fifth as much. | The trained player's own strength is its T=1 reading. Read both. |
| G2, G3 | The top-choice decode compares "end turn", a single entry, with unit actions whose probability is split over type, target and weapon. That favours ending the turn, and the −1.5 offset looks like a patch for it, tuned on a network six generations old (the link is plausible, not verified). | **Unfounded.** A decode that picks the unit first, then its action, needs no offset. |
| G4, G5 | The simulator picks an attack's hex and a move's path. By the game's rules the network can still choose the hex itself, by moving first and attacking from there (not checked against the mask). | A default, not a cap. Leave. |
| G6 | A unit always takes its first advancement. The network has no way round it. | **Against Principle 3.** A cap on play, not on the dynamics. A decision to add when convenient. |
| G9 | After a refused recruit the network decides again on the same input under a different mask. | A hidden input, rare. Either show the bit or correct CLAUDE.md. |
| G12 | The opponent is always the current self. | **A known gap.** Self-play against the latest self alone has no convergence guarantee and can cycle; pools of past selves do. The matches against a fixed reference are the alarm. Leave, and watch. |

One absence belongs here: **nothing keeps the policy exploring.** The
gradient sharpens it toward its own top choices (the gap between its
sampled and top-choice play fell from 415 to about 140 Elo). Theory warns
that a policy which stops sampling an action stops learning about it.
This is not a reason to add a bonus now; it is the thing to watch as the
rounds go on.

## The loop theory would keep

1. Play a round of about 1,000 games. Train on all of them.
2. Weight each decision by `z − α·v_t − β·(dice luck from that decision
   on)`, with α and β from the previous round. No switch, no gate.
3. Take the gradient through the whole network. Say plainly what the
   memory's gradient sees (B2), and lengthen it when the GPU allows.
4. Direction: Adam's per-weight normalisation, with ordinary momentum
   across rounds.
5. Step: a learning rate times the direction, one for the final maps and
   one for the rest. No search, no rescaling.
6. Exact checks abort (the replay identity). Statistical checks are
   logged.
7. One match against the reference every few hours of training is the
   only guard, read at the trained player's own decode and at the
   reference's.

Every piece of this is either a theorem's consequence or the mainstream
practice, so its dynamics are the best-understood there are.

## What this changes in today's work

- The fixed lengths of `feature/fixed-step-rounds` (departure 50) are the
  less principled of two forms. A learning rate is the form the match
  tested.
- My proposal to test the per-weight normalisation first is withdrawn: on
  theory it is the part to keep.
- The 20% of games held out and the per-round statistical gates have no
  theoretical need.

## The user's decisions (2026-11-12)

The user ruled on ten items of this document's summary, in their own
numbering. Nothing is built from them yet.

| # | Item (ids above) | Decision |
|---|---|---|
| 1 | The corrections to the signal (A2-A5) | Decided: the value estimate alone (below). |
| 2 | The 8-round average (C4) | Remove ("Indeed, remove"). This document proposed ordinary momentum in its place. |
| 3 | The games held out (A1) | Agreed: train on every game. |
| 4 | The statistical checks (F1-F4, A6's gate) | Remove from the rounds (below). |
| 5 | The entry switch (A6) | Remove, and the fitted coefficient with it: the weight is `z − v_t` (below). |
| 6 | The fixed step lengths (C5) and the two groups' sizes (C3) | Remove both. In their place: the optimiser's own update, with one learning rate for the whole network ("Just use a single learning rate. Decomplicate things."). In the match test one rate read +27.8 and +52.5 (`full3`, `full8`) and two rates +88.2 (`mix8`). |
| 7 | 5,000 games a round (E) | Reduce to 128 games a step, a number to revise later (below). |
| 8 | The end-of-turn offset (G2, G3) | Remove. |
| 9 | The memory's gradient window (B2) | Read (below); the user's answer is pending. |
| 10 | Self-play against the current self alone (G12) | To discuss. |

### Item 1: are the corrections needed, and should they not be learned?

The user's question: "Do we need it? Isn't it supposed to be learned by
the network rather than corrected by hand?"

What was put to them:

- **None is needed.** No correction changes what is learned (the
  yardstick's first result). Together they are worth 1.85 times the
  games. Round 22's mean squared weight per decision
  (`round_22\corrections.json`), against 1 for the bare result:
  - 0.838 with the baseline and the whole-game luck;
  - 0.603 with the value estimate added;
  - 0.542 with the luck re-timed.
- **They are not built alike.**
  - A4 is the network's own output.
  - A3 and A5 score each fight on seven hand-chosen columns
    (`tools/match_luck.py`, `COLUMNS`): hit points, cost-weighted
    material in two forms, a leader's death, and three saturating values
    of material. Seven coefficients are fitted to results.
  - Their mean is zero whatever is decided, so they are no reward. They
    are still a material score inside the training signal, which
    Principle 3 names.
- **Recommended: A4 alone,** `z − α·v_t`. This withdraws the table's
  "keep the idea" for A3 and A5, on the user's criterion and not on
  soundness.
  - What it costs against A4 with the luck from the decision on is not
    read. The whole-game luck with the baseline removes 16% of the bare
    result's noise, so 10-20% more games is my estimate.
  - A4 alone leans on a value head that nothing trains (above). Keeping
    it means training that head on each round's results.

**The user's answer (2026-11-12): A4 alone** ("We're going for option
2"). So A2, A3 and A5 go, with the luck walk and the fits of β and γ,
and the value head is to be trained on each round's results. Not built
yet.

### Item 4: the statistical checks

As listed to the user. A statistical check has a false-alarm rate; an
exact check has none.

- **Gone with rulings already made:**
  - dice control's slope (the dice terms are out);
  - A6's entry tests and its six I-V tests (the switch is out);
  - the step's confirmation (D4), the transfer stop (F3) and the pause
    (F4), which belonged to the KL rule's evaluator;
  - the one-off gates before the rounds: stage C's, round 1's cross-play,
    and stage P with its planted-signal check (F6, skipped since
    departure 31).
- **Run on every round's games, halting the round on a failure**
  (`tools/selfplay_readout.py`; 18 tests sharing a 1% false-alarm rate):
  - I-sample, five tests: the actions played follow the probabilities the
    player logged (three chi-squares by kind of action, the sampled
    action's position, the temperature score);
  - I-V, twelve tests: no correction leans toward a kind of action (four
    corrections by three kinds). With the value estimate alone, three;
  - dice control, one test (gone, above).

  They fired in 2 of 22 rounds (F1).
- **Planned and never run:** the lean stops (F5), on a claim by 2 SE or an
  argmax match by 1 SE, every 6th round. The claim was the KL rule's
  evaluator's.
- **Exact, so not on this list:** the legal mass and I-count on the games,
  and the pass's checks (B5).

Recommended: no statistical check halts a round. I-sample and the value
estimate's I-V stay as logged readings with a warning; one match against
the reference every few hours of training replaces F5.

**The user's answer (2026-11-12): remove them from the rounds.** The
recommendation to keep two as warnings was withdrawn first, on the user's
objection that such checks look useless:

- the code does not change between rounds, so a check of the code can
  find nothing new in a later round;
- `tests/test_decision_log.py`
  (`test_logging_changes_no_pick_and_the_log_matches_the_law_it_was_played_from`)
  already asserts both properties exactly: the logged log-probability and
  the logged value against the ones recomputed on the position before the
  command.

So no statistical check runs in a round. The exact checks stay, and the
match against the reference is the strength reading.

### Item 5: the fitted coefficients, without the switch

The user asked to verify that fitted coefficients exist and work without
the switch. From each round's `corrections.json` (rounds 2-22,
`C:\Users\momom2\wesnoth_runs\rounds_local`):

- The switch never turned a correction off: both entered in all 21
  rounds.
- The value estimate's coefficient, fitted alone on each of two halves of
  the held-out games, reads 0.78 and 0.78 in round 2 and 0.73 and 0.69 in
  round 22. The halves differ by at most 0.061 in any round.
- Its cut of the noise is 15.5 to 54.0 standard errors in every round,
  and 26-34% of the mean squared weight.
- It drifts down, from 0.78 to about 0.70: the head is frozen while the
  policy moves.
- All of these were fitted with the dice terms in place. For `z − α·v_t`
  alone the coefficient is not read.

Recommended: no fitted coefficient, `z − v_t`. An estimate that equals
the expected result has α = 1, since then E[z·v] = E[v²], and training
the value head on results aims at that. With the optimum at α*, α = 1
keeps (2α* − 1)/α*² of the cut: 82% at 0.70 and 92% at 0.78.

**The user's answer (2026-11-17): no fitted coefficient** ("Approved.
Drop it."). The weight is `z − v_t`. Not built yet.

### Item 7: the round's size

The user asked for the literature, the desktop's budget in games an hour
and the size that trains best.

**The budget.** One GPU plays and trains in turn, and the gradient pass
is the slow part.

- Play: 5,000 games in 89.5 min in round 22 (`round_22.log`), so 3,350
  games an hour, 1.07 s a game at 490 decisions a game.
- A whole-network pass: 85 decisions a second (stage C, 89,633 s for 7.6
  million decisions), so 5.8 s a game at 490 decisions. Round 22's pass,
  the final maps alone, ran at 271.
- Together about 6.8 s a game: about 525 games an hour and 12,600 a day,
  whatever the round's size. At `own2`'s 380 decisions a game it is 5.3 s
  a game (departure 25's prices), about 680 an hour.
- Fixed costs a round today: the pass's dtype floor (462 s in round 22)
  and the processes' starts. The step's search, 14 min, goes with the
  fixed lengths. The model below takes 2 min, which needs the floor read
  once a run and not once a round.
- Not read: the 85 on current code, the rate under a longer memory window
  (item 9), and play's rate with about a hundred games in flight (the
  3,350 is with 768).

**The literature.**

- McCandlish et al. 2018 (arXiv 1812.06162) define the critical batch B
  as the size at which a step is half as efficient per example as a small
  one. With the learning rate at its best for each size, a step on N
  gains N/(N + B) of the most a step can gain, so a game gives B/(N + B)
  of the most a game can give. They find that B grows as training goes on
  and that its estimate is good to an order of magnitude.
- AlphaGo's self-play policy-gradient stage (Silver et al. 2016) is the
  nearest precedent: the game's result as the signal, the value network
  subtracted, 128 games a step and 10,000 steps (read from summaries of
  the paper).
- OpenAI Five (arXiv 1912.06680) used each game about once and found
  stale games costly. Its larger batches bought speed on more hardware,
  less than in proportion (a summary gives 2.5 times for 8 times the
  batch; not read in the paper).

**This project's readings.**

- At `own2`, under the labels then in force, B read 44.3 ± 6.4 games for
  the whole network in the per-weight-normalised direction, and 280.9 ±
  29.9 for the final maps alone (stage C).
- A direction from 5,000 of the pass's 20,000 games read 1.062 ± 0.039 of
  the whole pass's first-order gain (departure 25 of
  `docs/stage_c_rounds_local_prereg_20261030.md`, with its amendment).
- The final maps' signal then fell by a factor of 3 over ten rounds with
  the noise flat (`training/metrics/denoise_20261105.json`): B rises here
  too.
- B under `z − v_t` is not read.

**The model.** Departure 25's own formula, N* = √(c_f·B/c_g), gave 1,111
to 2,198 games at a fixed cost of 6.5 to 11 h a round, and the rule
clamped it up to 5,000. With the fixed cost at 2 min and a game at 6.8 s
it gives 28 games at B 44, 73 at B 300 and 132 at B 1,000. The share of
the best possible
rate that each size keeps (a game's worth, B/(N + B), times the share of
the hour not spent on fixed costs):

| N | at B 44 | at B 300 | at B 1,000 |
|---|---|---|---|
| 32 | 0.37 | 0.58 | 0.63 |
| 64 | 0.32 | 0.65 | 0.74 |
| 128 | 0.22 | 0.62 | 0.78 |
| 250 | 0.14 | 0.51 | 0.75 |
| 500 | 0.08 | 0.36 | 0.64 |
| 1,000 | 0.04 | 0.23 | 0.49 |
| 5,000 | 0.01 | 0.06 | 0.17 |

Recommended: **128 games a step, one step a round, each game used once.**

- It keeps at least 0.60 of the best size's rate at every B from 44 to
  1,000 (0.60, 0.95, 1.00). Against 128, a round of 64 reads 1.42, 1.05
  and 0.95 at the three B, and one of 250 reads 0.62, 0.83 and 0.96.
- A round of 1,000 keeps 0.18 to 0.63 of 128's rate, and one of 5,000
  keeps 0.04 to 0.21.
- With the fixed cost at 1, 4 or 8 min instead of 2, 128 keeps 0.50, 0.72
  and 0.85 of the best at B 44, and 0.90 or more at B 300 and 1,000.
- Under the critical batch, the round's size and the learning rate trade
  one for one (a judgement from the model, not a reading), so the size
  can stay fixed and the learning rate is the one number to tune.
- At 128 a step takes about 17 min: about 87 steps and 11,100 games a
  day.

**The user's answer (2026-11-17): 128 games a step, to be revised later**
("Let's go for batches of 128 games, with a note to revise this number in
the future."). What a revision would read, from the above: the critical
batch under `z − v_t` and as training goes on, the round's fixed cost
once built, play's rate at this many games in flight, and the pass's rate
on the code of the day. Not built yet.

### Item 9: the memory's gradient

**What the pass does** (`tools/pg_pass.py`). It runs 32 game-sides side
by side and cuts the gradient through the memory every 16 time steps of
that schedule (`_run_split` detaches the memories after each window). A
game-side starts from the learned initial memory, so no gradient crosses
from one game-side to the next either. In a full pass each step's forward
is recomputed in the backward (`recompute`), so a window holds its steps'
inputs and memory states and one step's activations.

**What the cut means in a game** (600 games of round 22's play, 1,200
game-sides, `round_22\play\games_s00`):

- A side makes 247 decisions a game on average (median 228, 99th
  percentile 681, longest 995), over 19.9 of its turns: 12.2 a turn.
- So a write is credited with the 0 to 15 decisions after it, 7.5 on
  average: about 0.6 of a turn and 3% of the side's game.
- The memory's one known pattern is held for 7 to 14 turns
  (`docs/status.md`, "The mechanism").
- How to act on the memory is trained at every decision with no cut. The
  cut touches only what is written.

**The literature.**

- A short cut is how the large recurrent self-play systems were trained.
  OpenAI Five applied Adam with truncated backpropagation through time
  over 16 timesteps, 2.1 s of hour-long games (arXiv 1912.06680 and
  1912.06721, read from search summaries). R2D2 uses sequences of 80
  steps.
- The uncut gradient through a long recurrence is known to explode or
  vanish, and clipping its norm is the usual remedy (Pascanu et al. 2013,
  from memory, not re-read). The explosion is measured here: on about 1%
  of game-sides the memory amplifies a gap of 1% to as much as 60%
  (`docs/status.md`).

**The options.**

1. **Keep 16.** No work. It is the practice above, and the only one read
   in games here: the match test's steps were computed under it. Its
   bias: a write is taught by the half turn after it.
2. **The whole game-side.** The true gradient. The pass would run its 32
   game-sides from their starts in place of a clock's windows. Since each
   step is recomputed, it must hold the inputs and memory states of up to
   about 1,000 steps, not their activations. Not measured: whether that
   fits the card (a whole split, 270,000 decisions, did not: departure
   33a), the rate, and fp16's range over chains of hundreds of steps (21
   of 4,926 windows overflowed at 16 steps in round 22's pass, which was
   heads only). It would probably need the clipping above: one more
   number.
3. **No gradient through the memory.** The simplest: the window's number
   goes. The network is trained to act on its memory and never on what to
   write, which then drifts with the trunk.

This corrects section B's verdict on B2 ("the weakest point left",
"neither"): that verdict did not compare the cut with practice. Under the
optimiser's own update the window also stops entering the direction's
statistics (C1's per-window second moment goes), so all it decides is how
far back a write's gradient reaches.

Recommended: **keep 16, as a number to revise,** like the round's size.
It is standard and read in games, and it costs nothing to build. Option 2
is the revision if the memory is to learn anything long, and it changes
the pass alone. The match against the reference is the alarm either way.

The user's answer: pending.
