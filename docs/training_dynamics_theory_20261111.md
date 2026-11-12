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
| 4 | The statistical checks (F1-F4, A6's gate) | The user asked for the list; not yet given. |
| 5 | The entry switch (A6) | Remove. To discuss first: that the fitted coefficients exist and work without it. |
| 6 | The fixed step lengths (C5) | Remove ("I don't understand, but in doubt, I lean towards simplification"). |
| 7 | 5,000 games a round (E) | Reduce. The number is to discuss. |
| 8 | The end-of-turn offset (G2, G3) | Remove. |
| 9 | The memory's gradient window (B2) | To discuss. |
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
