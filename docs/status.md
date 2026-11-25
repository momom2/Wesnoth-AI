# Project status

The project's current state. CLAUDE.md holds only instructions and points
here. Update this file on `main` when a result lands. When a block is
superseded, move it verbatim to `docs/archive/` and index it there.
Earlier status logs: `docs/archive/claude_md_20261109.md` (CLAUDE.md's
"Current status", 2026-10-08 to 2026-10-21) and the files listed in
`docs/archive/README.md`.

Last updated 2026-11-25.

## In one paragraph

Self-play now beats the reference. The policy-iteration rounds, run on the
desktop's RX 6600 from 2026-10-29, took `own2` through 18 adopted rounds.
At `own2`'s own decode, round 18 beats `own2` by +105.7 ± 3.2 Elo, and
paired on the same seeds it leads round 16 by 3.2 SE. Nothing has been
adopted as the new reference yet: that is the user's call. From
2026-11-10 03:03 the line ran lean (departures 43-47): each step is
adopted unread, and evaluation falls from about 32% of a round to about
11% (42% to 18% counting the held-out games). **The line is stopped**
(departure 48). It ended on 2026-11-10 22:29:53 in round 23's games, and
round 22 is its last finished round. Rounds 19-22 were adopted unread;
the 3,200-game claim and argmax match that judge them were due at round
24 and have not been played. One match since reads round 22 level with
round 18 at the reference's decode (+114.7 ± 6.5 against +105.7 ± 3.2;
+1.2 SE paired). The trunk has never trained in a round: the
KL search sizes a full-network step on a few game-sides where the memory
diverges ("What the numbers do not show"). A match on 2026-11-11 shows
that this sizing is what blocks it. At 8 times that size the full-network
step gains +52.5 ± 6.2 Elo over `own2` at T=1, and added to the
heads-only step it reads +88.2 ± 6.3 against the heads' +34.6 ± 6.2 ("A
larger full-network step", under Measured). No round does this yet.

**A simpler loop replaces the rounds** (2026-11-17). The user ruled on ten
of the rounds' mechanisms one by one and said "Launch training."
(`docs/simple_loop_prereg_20261117.md`). A round is now 128 games and one
Adam step on the whole network: the weight is the result less the logged
value estimate, the memory's gradient runs over the whole game, the value
head trains on the result, and 20% of the games are against a past self.
It starts from round 22's network at a learning rate of 1e-5. A match
every 16 rounds reads it against `own2`, the candidate sampling. **Its
registered verdict is "the loop trains"** (2026-11-19): the matches at
rounds 128, 144 and 160 average +120.7 ± 7.5 Elo against +39.2 ± 12.4 at
the start, a difference of +81.5 ± 14.5, 5.6 standard errors against the
registered 3. It is not an adoption claim: the candidate samples while
the reference plays its pinned decode. The run goes on, and since
2026-11-20 it plays from the main checkout (the loop's code is in `main`
since 0.34.0). Since the verdict the level has stepped up and held: the
seven matches of rounds 128 to 224 average +115.4 ± 4.9 and the fifteen
of rounds 240 to 464 average +152.5 ± 3.5 (to 2026-11-23). A line through
the twenty-two from round 128 rises 0.14 ± 0.03 Elo a round; within
rounds 240 to 464 no rise is read (0.02 ± 0.05). Round 464 is the last at
λ = 1.

**The loop's network at the reference's own decode** (2026-11-20,
`docs/loop_pure_match_prereg_20261120.md`). Round 208's network reads
+152.2 ± 13.5 Elo over `own2` in a PURE match of 800 decisive games:
+37.5 ± 15.0 over the start network's +114.7 at that decode, 2.5
standard errors against a registered 3, so no change is read. It leans
to a gain. Nothing is adopted.

**The value estimate was measured** (2026-11-20,
`docs/value_reliability_prereg_20261120.md`), before any weight that
leans on it (λ under 1) is tried. It accounts for 45% of the result's
spread, 2% in a game's first tenth and 89% in its last. It is exact at
the ends, optimistic by about 0.1 in the middle and overconfident in a
game's first third (slope 0.77 ± 0.03), where it fails the registered
rule narrowly. It reads the same in rounds 1 to 32 as in rounds 177 to
208. Today's weight spends half its power on the first three tenths of a
game's decisions.

**Why it errs** (2026-11-21 and 2026-11-23, the same document's two
follow-ups). The value head was fitted on human games' results and read
frozen until this loop, and Adam moves a weight by the learning rate a
step at most. No fault was found in the training code. In rounds 177 to
208 its optimism followed the enemy's faction (+0.20 against Undead,
+0.17 against Northerners), it carried an enemy unit that had gone out
of sight at three quarters of its worth, and its opening prior was too
strong (the result moves 0.54 to 0.66 for each 1 of the estimate at a
game's first decision). It is not blind to hidden enemies. In five of
seven windows it also rises over a side's own turn and falls back across
the enemy's, by up to 0.04. Over all decisions the optimism has fallen
from +0.078 (rounds 177 to 208) to +0.027 (rounds 393 to 424). The user
keeps one learning rate for the whole network (2026-11-21), declined a
one-off refit of the head and proposed continuing at λ = 0.99
(2026-11-23). Read on the logs, that weight has 0.30 to 0.32 of today's
mean square, the result being a third of what the present estimate is
compared with.

**The loop runs at λ = 0.99 since round 465** (2026-11-23 23:58, the
user: "Alright, do it.";
`docs/lambda_prereg_20261123.md`, departure 9 of the simple loop's
registration). The policy's weight becomes generalised advantage
estimation's, fading by the decision; the value head still trains on the
result, and the learning rate and the optimizer's state are unchanged.
The code is in `main` since 0.35.0 (`--lam` on the pass and the runner, 1
by default). The run switched after its match at round 464, and its
command now carries `--lam 0.99`. Round 465 checked out: its weights'
mean square is 0.33 of what the same round reads at λ = 1 (0.3 expected),
and its replay is inside the exact check. The first four matches at 0.99
read +139.8 ± 13.3, +178.1 ± 13.9, +164.9 ± 13.7 and +175.8 ± 13.9
(rounds 480 to 528, to 2026-11-24). They average +164.7 ± 6.8: +12.2 ±
7.7 against the block before the switch, 1.6 standard errors above, which
is no gain read. The early-loss criterion, first read at round 528, is
not met. The value estimate, read again on rounds 497 to 528, is as
before the switch: optimistic by +0.032 ± 0.003 over all decisions
(+0.030 in rounds 433 to 464), with a rise of +0.019 over a side's own
turn that the enemy's turn takes back. The match at round 544 read
+179.2 ± 13.9, and the reading closes there, since the inputs and the
legal actions change at round 545 (next paragraphs): the five matches
average +167.6 ± 6.1, +15.1 ± 7.1 against the fifteen of rounds 240 to
464 (+152.5 ± 3.5), 2.1 standard errors above. The registered verdict
asked for fifteen matches and cannot be read. No run at λ = 1 played
beside it, so the 15 Elo are not shown to come from λ. λ stays at 0.99.

**What the network is not shown of a unit and of a fight** (2026-11-24,
`docs/input_facts_design_20261124.md`). The user watched a game of round
503's network and asked what the network sees. The simulator's rules
checked out: over 11,622 positions the zone-of-control flags are those of
the visible enemies of level 1 or more. But a unit's row holds no level
and no defence, the network sees no number of a fight before it attacks,
and the weapon is scored without the target. Measured on 24 games of
round 513's network, its most likely weapon is beaten by another in 1.0%
of the attacks by a two-weapon unit, and 3.6% of attacks go into a
resistance of 40% or more. The user ruled: "for sure the network needs
that information". The design adds what the sidebar and the attack dialog
show as columns of the unit's row, every new weight starting at zero.

**The user approved it on 2026-11-24, with the attack from a distance
removed** ("Alright, remove attack from a distance. For the rest
approved."), and asked for training to go on with it at once: "No
special learning rate [...] maybe we'll reconsider tomorrow if the
night's training is clearly flawed or stalled."
(`docs/input_facts_prereg_20261125.md`, departure 10 of the simple loop.)
- **Built on the branch `feature/input-facts`, not merged into `main`:**
  the merge is the user's to decide. The run plays from the branch's
  worktree meanwhile (`C:\Users\momom2\wesnoth_worktrees\input-facts`).
  Both venvs hold the branch's core (phase 40), so until the merge a
  local pytest belongs in that worktree: run from the main checkout it
  would test `main`'s Python against the branch's core.
- **A unit attacks only an enemy on a hex that touches its own.** The
  simulator no longer moves a unit before its attack. The reference at
  its pinned decode ordered no such attack in 1,660, so the yardstick
  plays as before.
- **The network reads 232 more numbers of each unit:** its level, its
  defence where it stands and by terrain, its movement costs, and for a
  unit that can attack the numbers of each fight it can start (damage of
  a hit, strikes, chance to hit, for both sides). They go through a
  projection of their own that starts at zero, so the network at the
  switch is round 544's network to the last bit.
- **Not built:** the weapon's score reading the target (the design's
  Part 3), which changes the output and has not been put to the user.
- **The time of day** was already in the input, globally and per hex
  (the user asked on 2026-11-24): the registration lists where.
- **The first night** (rounds 545 to 560, to 2026-11-25 06:29): every
  round's exact check holds (6.8e-6 to 7.7e-6), no warning, the same
  round time. The match at round 560 read +157.4 ± 13.6 (570 of 800),
  0.7 standard errors under the baseline of +167.6 ± 6.1: no change
  read, and the "clearly flawed" criterion is not met.
- **The new weights move at the loop's ordinary pace.** After sixteen
  rounds the new projection's root mean square is 4.6e-5. That is about
  2,300 times under the old projection's size (0.104), but the size is
  the wrong yardstick: the loop made its whole gain (+39 to +168) while
  the whole network moved 4.6e-4 and the old unit projection 2.5e-4, and
  after the loop's first sixteen rounds they stood at 4.4e-5 and 4.3e-5.
  Over rounds 545 to 560 the new projection moved 2.9 times as far as
  the old one. Whether such moves make the network use the new numbers
  is read from the attacks, not from the weights.

**The first reading against an outside opponent** (2026-11-19,
`docs/rca_games_prereg_20261119.md`). The loop's network after round 128
played 100 live games against Wesnoth's default AI at the reference's
decode: 91 wins, 3 losses and 6 stopped, so 91 of 94 finished games
(96.8% ± 1.8%). By the registered number the default AI is saturated as a
yardstick; the margin is one standard error. The six stopped games were
two scenario messages waiting for a click, three differences the board
check made itself, and one defect of the simulator, all fixed on `main`
(0.33.1 and 0.33.2).

## The reference player

`own2` at `raw:t0+eo-1.5` with 64 memory slots, adopted 2026-10-21. It is
pinned in `configs/reference_player.json` (HF
`tier-b/memory_own_round2_20261011/own.pt`, local
`training/checkpoints/own2.pt`). `tools/reference_player.py --flags b`
turns it into run_elo_batch flags. Its lineage: each step is a match of 800
decisive games (6,400 for `own2`) under the observation of its day. Elo
does not chain across observation epochs.

| reference | adopted | over its predecessor |
|---|---|---|
| `own2` | 2026-10-21 | +22.7 ± 4.4 over `own1`: a second round of training on its own games; the memory no longer costs strength |
| `own1` | 2026-10-11 | +88.6 ± 12.7 over `parity3`: trained on its own games, the memory's cost in play cut from 105 to about 22 Elo |
| `parity3` | 2026-10-04 | +82 ± 13 over `parity2` (imitation under the anneal rule) |
| `parity2` | 2026-10-02 | +191 ± 14 over `obs8`: the network observes what a player sees, with a 64-slot memory |
| `obs8` | 2026-09-25 | +73 ± 13 over `terrain` |
| `terrain` | 2026-09-20 | +263 ± 16 over `relset` |
| `relset` | 2026-09-11 | +56 ± 12 over seed2 |
| seed2 | 2026-09-11 | +33 ± 12 over the original imitation seed |

Every reference up to `own2` came from imitation of the human corpus,
followed by fine-tuning on its own recorded games.

## Self-play: the policy-iteration rounds

- **Design:** `docs/training_design_20261019/current_design.md`. The
  pre-registration and its 50 departures are in
  `docs/stage_c_rounds_local_prereg_20261030.md` (main's copy holds 1-41;
  42-50 are on `feature/local-gpu-stage1`). The autonomous window's
  card, plan, reviews and log are in `docs/autonomy_20261029.md`.
- **A round:**
  1. π_k plays itself at T=1: 5,000 games to train on, plus 2,500 held out
     (more in rounds 2-6).
  2. One gradient pass computes the outcome-labelled policy gradient over
     the **policy heads only**; the trunk and the memory do not move.
  3. One step is taken on a KL ladder. It is read on the held-out games
     (half A chooses the step, half B confirms it) and adopted unless half
     B reads it below −2 SE. When the importance weights fail their ESS check,
     a 10,000-game cross-play match decides instead.
  4. Every third round, a claim match plays the round's policy at T=1
     against `own2` at its decode.
- **A lean round** (from round 19, departures 43-47; `--rounds-lean` on
  `feature/lean-rounds`):
  - 5,000 games plus 1,250 held out;
  - the same pass, its floor check on 25 game-sides from round 21
    (departure 46: the check took 16-21% of each pass);
  - a step that is the KL search alone, along the average of the round's
    direction and the 8 before it, to the radius (twice the radius from
    round 25, departure 47). It is adopted unread.
  - Every 6th round, a 3,200-game claim and a 3,200-game argmax match on
    the finals' seeds. The line pauses unless the claim gains 2 combined SE
    over the last claim and the argmax match 1 SE over the last argmax
    reading.
  - About 4.9 h a round (round 19: 4.88 h), about 4.5 h with the smaller
    floor.
- **Code:** `scripts/rounds_local.py` and `scripts/chain_local.py`, over
  `tools/pg_corrections.py`, `tools/pg_pass.py`, `tools/pg_round.py` and
  `tools/pg_eval.py`. The library is `wesnoth_ai/pg_*.py`.
- **Wall time on the desktop:**
  - the games: about 2.25 h;
  - the pass: about 2.7 h;
  - the step: about 1.5 h;
  - a claim (12,800 games): about 3.5 h;
  - so about 6.3 h a round, or 9.7 h with a claim.
- **Games played:** stage 1 played 46,000 (stage C and round 1 read them),
  and rounds 2-18 played 150,000. Evaluation matches played about 210,000
  more.
- **Run state:** stopped ("The line is stopped", below). Its records
  are in `C:\Users\momom2\wesnoth_runs\` (`rounds_local/`, `chain.log`,
  `chain_lean.console.log`). The code ran from the detached worktree
  `wesnoth_ai-run` at 21862a0a (`feature/lean-rounds`), from round 21.
- **Round 19's first games failed a gate** (2026-11-09 20:41, departure
  42). The readout's I-V check read v against recruit at z −3.66, against
  the gates' 3.45.
  - The signal is diffuse: both shards, every map, not a few games. A
    bootstrap reproduces its SE, and over the line's 228 I-V readings z
    reads RMS 1.03. It fits the gates' 1% false-alarm rate, but the gate
    stands on those games.
  - Round 19 was replayed on fresh seeds from 29,500,000 under departure
    42's retry rule, and every gate passed on the fresh games (v against
    recruit z −1.39; round 20's −0.85).
- **The line is stopped** (2026-11-10, departure 48). At 22:29:14 the
  user asked for a stop after the round.
  - Every process of the line ended between 22:29:48 and 22:29:53, five
    minutes into round 23's games. The 64 games then running exited with
    Windows' console-interrupt code (0xC000013A), and the runners left no
    log line. The system's logs show no fault, display reset or restart.
    So the line's console was closed or interrupted. By whom is not
    established; the window's session ran no command then.
  - Round 22 is the last finished round (adopted 22:24:38,
    `round_22\pi.pt`). Round 23 holds 230 game results of 5,000.
  - Rounds 19-22 ran lean and were adopted unread. Round 24's claim and
    argmax match judge departures 43 and 44 (departure 45). They have not
    been played, and neither has a final match on the lean line.
  - **Round 22 read at the reference's decode** (2026-11-11, departure
    49): +114.7 ± 6.5 Elo over `own2` (3,200 decisive games). Paired by
    seed with round 18's final, round 22 leads by +0.0116 ± 0.0097 in
    score (1.2 SE). So the four lean rounds show no loss, and a gain is
    not shown at this size. Round 22's policy is a sound place to go on
    from.
  - It was not relaunched: that is the user's call. A new start resumes
    round 23 from its markers (by the tools' own account; not tried). It
    takes two steps, as the 12:52 arming did:
    - a `STOP` file naming "after round 3" in `rounds_local`, which the
      chain's `--after-stop 3` refuses to start without and moves aside;
    - then `arm_lean47.ps1 -Head 21862a0a`, in
      `C:\Users\momom2\wesnoth_runs\tools\` (outside the repository). The
      branch's `arm_lean.ps1` lacks departure 47's flags.
  - Rounds 23 and 24 with round 24's two matches would take about 12 h
    (reasoned from the rounds' 4.7 h and the final's rate).

### Measured

All matches below ran on the desktop with fp16 serving on both sides,
12,800 decisive games each.

**At the reference's decode** (argmax with the −1.5 end-turn offset, both
sides), against `own2`, seed base 900,000:

| policy | Elo |
|---|---|
| round 3 | +31.8 ± 3.1 |
| round 6 | +48.9 ± 3.1 |
| round 9 | +63.1 ± 3.1 |
| round 12 | +82.9 ± 3.2 |
| round 15 | +96.5 ± 3.2 |
| round 16 | +94.1 ± 3.2 (fresh seeds, base 980,000: **+102.7 ± 3.2**) |
| round 18 | **+105.7 ± 3.2** |

**Claims** (the round's policy at T=1 against `own2` at its decode):

| claim | Elo |
|---|---|
| `own2` itself (D0) | −415.2 ± 5.5 |
| round 3 | −296.8 ± 4.3 |
| round 6 | −204.2 ± 3.6 |
| round 9 | −146.5 ± 3.3 |
| round 12 | −100.8 ± 3.2 |
| round 15 | −43.4 ± 3.1 |
| round 18 | −8.7 ± 3.1 |

Round 3's claim read 2.2 SE under the forecast the rounds' own half-B
readings made, and fired the transfer stop (departure 33 went on past
it). The later claims are recorded as within 2 SE of theirs; the
2026-11-11 audit did not re-check that.

**Both sides at T=1:** round 16 beats `own2` by +429.1 ± 5.7 Elo, round 3
by +134.1 ± 3.3.

**A larger full-network step** (2026-11-11;
`docs/trunk_step_match_prereg_20261111.md`). One step from `own2` along
stage C's pass (20,000 games), each candidate against `own2` at T=1 on
both sides, 3,200 decisive games on the same seeds:

| step | Elo |
|---|---|
| heads only, at its KL-sized step (what a round takes) | +34.6 ± 6.2 |
| heads only, at twice that KL | +48.7 ± 6.2 |
| full network, 3 times its KL-sized step | +27.8 ± 6.2 |
| full network, 8 times | +52.5 ± 6.2 |
| the heads at their step and the rest at 8 times | **+88.2 ± 6.3** |

- Every pre-registered rule read on the gaining side, and each arm read
  at or above its first-order forecast.
- The two steps' gains add: by seed, the last row leads the heads-only
  step by +0.0747 ± 0.0110 in score (6.8 SE) and the larger heads-only
  step by +0.0547 ± 0.0110 (5.0 SE).
- At the reference's decode the last row reads +15.9 ± 6.1 Elo over
  `own2` (3,200 decisive games, 2.6 SE): the one reading there, and not a
  claim for an adoption.
- Not read: a second step, any size past 8 times, and a round's
  5,000-game pass.

### What the numbers do not show

- **At argmax the gains are uneven.**
  - Paired on the same seeds, in score per round:
    - rounds 4-12: about +0.008;
    - rounds 13-15: +0.006;
    - round 16: −0.003 ± 0.005;
    - rounds 17-18: +0.0154 ± 0.0048 together, about +0.008 a round.
  - Off Arcanclave Citadel, round 16 minus round 12 reads only +0.0091 ±
    0.0054. Round 16 wins 560 of its 616 games on that map. Whether that is
    a sound plan or a hole in `own2` is not established.
- **The rounds mostly sharpen the sampled policy toward its own argmax.**
  The gap between the policy at T=1 and at argmax shrank from 415 Elo
  (`own2`) to about 140 (round 15). That headroom is finite.
- **The heads' gradient signal is falling.** Its norm went from 3.97 (round
  2) to about 1.3 (rounds 11-13), while the noise per split stayed flat, so
  each round's 5,000 games buy less (`training/metrics/denoise_20261105.json`).
- **The trunk has never trained in self-play.** The pipeline is not why.
  The memory is why the step is small: on a few game-sides it turns a
  trunk step into a divergence, and the KL search sizes the step on that
  (2026-11-10). **The divergence does not cost a larger step its gain**
  (the match of 2026-11-11, "A larger full-network step" above). So the
  account below explains why the step came out small; it no longer says
  the trunk cannot be trained on this memory. The readings are `training/metrics/trunk_*_20261110.json`
  on `feature/local-gpu-stage1` (69ce95ff), and the account is that
  branch's `docs/autonomy_20261029.md`, the entries of 2026-11-10 14:20
  and 19:07. The 22:01 review's limits are folded in below: the sides
  walked were picked by the step's KL, not sampled, and no reading
  compares the push with another direction of its size.
  - The planted signal reaches the gradient at its size: 0.0222 ± 0.0031
    against 0.0232 planted, in the heads and the trunk alike. Stage C's
    planted check missed (about 6 SE, both seats) on its forecast of how a
    trunk-heavy step moves the attack mass, read through 16-decision
    windows. That is the instrument, not the training.
  - Per √KL, a full-network step gains 0.50 against a heads-only step's
    3.03 (first order, on stage C's half A).
  - **A few game-sides hold a full-network step's KL.** The 1% that move
    most hold 65% of it and the top 5% hold 88% (59% and 84% when each
    decision is classed by its side's decisions of the other parity). A
    heads-only step reads 4% and 16%.
  - **On those sides the memory diverges.** Three were walked decision by
    decision, each network reading the other's memory:
    - `own2` reading the stepped network's memory reproduces 97-99% of the
      step's KL. The stepped network reading `own2`'s memory shows 2e-5 a
      decision, as an ordinary side does.
    - `own2`'s own recurrence amplifies there: a memory gap of about 1%,
      set once, peaks at 60%, 54% and 8%. On a fourth, ordinary side it
      decays.
    - A smaller step does not avoid it. On the one side read, the response
      at 1/64 of the step is linear and still costs about 500 times a
      median side.
    - Nine more sides were walked, down to the 5000th by the step's KL.
      The memory holds 94% to 99% of the KL on those ranked 2nd to 100th,
      81% at the 500th and about half at the 2500th and 5000th.
  - **The mechanism** (13 sides walked, picked by the step's KL rank and
    not sampled; a write's parts read on four):
    - `own2`'s memory holds one learned pattern, the same in every slot
      and on each of the 13 sides. It is up by turn 2 and released some
      turns later (turns 9 to 16 on eight of the nine diverging sides). On
      the ninth it dips at turn 6 and is held again, lower, to about turn
      14. On one side that does not diverge it is raised again from turn 5
      and stays up. Its level follows the game's phase (a rank correlation
      with the turn of −0.67 to −0.96 on twelve sides). The policy
      reads it far more than a random memory direction: a shift of 0.1 of
      the memory's norm along the gap costs a KL of 0.004 to 0.019 a
      decision, against 8e-5 to 4e-4.
    - While the pattern is held, a difference along it is regenerated, not
      stored. The gate keeps 0.67 to 0.84 of it a write and the trunk
      writes back 0.15 to 0.34, so it changes by 0.96 to 1.05 a write. A
      random difference loses 10% to 13% a write.
    - A trunk step writes the pattern higher (+0.008 to +0.012 after one
      write, on all 13 sides). That push is a large part of the gradient.
      The step's part in the span of 27 push gradients (one axis holds
      79.5% of their squared norm) is at a cosine of 0.62 with the
      gradient, and it holds 42.5% of the step's first-order gain on 8.2%
      of its squared norm, read between the pass's two halves. No other
      direction of that size was read beside it.
    - The pushes add up. At decision 100 the gap on the pattern is 116 to
      420 times one write's on the nine diverging sides. The stepped
      memory then releases the pattern 2 to 5 turns late on three of them
      and not within the decisions read (240 at most) on five. On the
      ninth, whose first dip the step does not move, the second hold reads
      +3.5 to +4.9 on turns 11 to 14 against the reference's +1.0 to +2.4.
      Between the two releases the policies differ by a KL of 0.1 to 0.2 a
      decision.
    - Without that part the step loses 98% to 99.7% of its KL on four
      diverging sides outside the span's fit. The part alone brings back
      74% to 88%. So the push is enough to make those sides diverge.
      Whether another 8.2% of the step would do as much was not read (no
      norm-matched control).
    - What is left of the step feeds the same loop: at 3 times its size
      the memory diverges again (the KL grows 18 to 67 times, not 9).
  - **So the step is sized on the memory's divergence.** At a given memory
    it changes the policy by about 2e-5 a decision, 60 to 100 times under
    the 0.00135 it is sized to.
  - **Those sides return no measurable gain** (the top 1%: −0.0035 ±
    0.0046 of the step's 0.0148). Outside the top 5%, the step gains 1.30
    per √KL against the heads' 2.89. So the diverging sides explain a
    factor of about 2.7 of the gap of 6, and a factor of 2.2 remains. On
    its KL at a given memory the trunk step is about as efficient as the
    heads' (2.5 to 3.4 per √KL against 3.0), so that factor may be the
    memory's too. This is reasoned, not measured: the step's first-order
    gain includes what comes through the memory inside a window, and the
    KL at a given memory was read on the 13 sides walked alone.
  - **Not the cause:** the optimizer (sgd, reduced AdamW and muon read
    alike), the fp16 rounding of the stepped weights, or the pass's noise
    (about 0.1% of the trunk step's KL, against 1% of the heads').
  - As the step is sized, on the mean KL over all sides, mixing the trunk
    in adds under 1%. Sized on a KL the diverging sides cannot fill, the
    mix would add at most about 10% (reasoned from the rates above, not
    measured). Without the divergence the step could be between about 3
    and 8 times larger, for a first-order gain of about 0.05 to 0.10
    against the heads-only step's 0.109, also reasoned:
    - 8 times is the step sized on its KL at a given memory (about 2e-5 on
      the 13 sides walked). It is an upper bound, since a memory that does
      not diverge would still move.
    - About 3 to 3.5 times is every side moving as the sides outside the
      top 5% do today (a KL of 1.1e-4 to 1.5e-4 a decision), at their
      measured 1.30 per √KL.
  - Open: which remedy, and it is the user's choice. The candidates:
    - a step sized apart from the divergence, which the match of
      2026-11-11 supports: the heads at their own KL and the rest at its
      KL at a given memory (about 8 times today's step). It needs a
      sizing rule in the rounds and a full-network pass, which ran at
      about a third of a heads-only pass's rate in stage C;
    - a memory with no such loop (`BACKLOG.md`, "Rework the memory
      system");
    - a pass that trains through the loop, past its 16-decision windows;
    - a memory written by a frozen trunk;
    - a step with its push taken out, which gives up 42.5% of the
      first-order gain and still diverges at 3 times the step;
    - heads-only rounds, as now.
  - Not established:
    - whether the push's signal comes through the decision's own policy or
      through the memory inside a 16-decision window;
    - whether a pattern held longer plays better. The pattern follows the
      game's phase, so the step may be moving a clock it has reason to
      move;
    - how the step reads on sampled sides: its KL and gain at a given
      memory, and the step without its push beside a control of the same
      norm;
    - the loop on a current round's games. Every reading is stage C's,
      from `own2`.
  - P stays skipped (departure 31).
- **Serving:** fp16 here against bf16 on a box. The fp16 path matches
  fp32's argmax on 200 of 200 bench positions; bf16 has fewer mantissa
  bits. The two kinds of match are not pooled in one fit.

## Open (for the user unless noted)

- **Merge `feature/input-facts` into `main`** (the new inputs and the
  removal of the attack from a distance, approved 2026-11-24). The run
  plays from the branch's worktree since round 545; `main` still holds
  the code of rounds 1 to 544.
- **How the new weights are trained.** The user set it for discussion on
  2026-11-25, "if the night's training is clearly flawed or stalled". The
  design lists four options (`docs/input_facts_design_20261124.md`,
  "Training the new weights"). The night was not flawed: the match at
  round 560 read +157.4 ± 13.6, 0.7 standard errors under the baseline
  (+167.6 ± 6.1), and the sixteen rounds checked out. Nor is it stalled
  by the weights' reading: the new projection moved 4.6e-5 in sixteen
  rounds, the pace the whole network had in the loop's first sixteen
  (4.4e-5), where the loop's whole gain took a move of 4.6e-4
  (`docs/input_facts_prereg_20261125.md`, "The yardstick"). Whether the
  network comes to use the new numbers is not yet read; the audit's
  counts on a later network would say.
- **How the weapon comes to read the target** (the design's Part 3, not
  yet put to the user). Today a unit's weapon is scored the same
  whichever enemy it strikes.
- **Resume the lean line, or leave it stopped.** It stopped in round 23's
  games on 2026-11-10 ("The line is stopped", above). Rounds 23 and 24
  with round 24's claim and argmax match would give departures 43 and 44
  their registered reading, for about 12 h of the desktop.
- **Adopt round 18, or a later round, as the reference.** The evidence is
  a local match of 12,800 decisive games (+105.7 ± 3.2), and the
  checkpoint is `C:\Users\momom2\wesnoth_runs\rounds_local\round_18\pi.pt`
  (SHA-256 c5796941...). Copies of rounds 3, 15, 16 and 18 are in
  `C:\Users\momom2\wesnoth_checkpoints_window\`.
- **The lean line's code and the stage-1 log commits were never merged.**
  Since 2026-11-20 they are the tags `archive/feature/lean-rounds` (5
  commits, the code the line ran, at 21862a0a) and
  `archive/feature/local-gpu-stage1` (ca21aaa1). The suite passed on the
  desktop at 21862a0a (2,258 passed, 139 skipped); CI has not run on it.
- **The next training step.** The options, each a single factor:
  - more games per step, against the falling signal-to-noise;
  - training the trunk. The match of 2026-11-11 says a full-network step
    sized 8 times larger gains and adds to the heads' step, so a reworked
    memory is not a precondition. The user then asked for the KL rule to
    go entirely (2026-11-11), and it is built, not launched:
    - `feature/fixed-step-rounds` (27cea50a; the desktop suite passed at
      f34d8766, 2,263 passed and 139 skipped): from round 23 the pass is
      the full one and the step has two fixed lengths in weight space,
      0.5 for the head maps and 0.019 for the rest, with no KL read
      (departure 50);
    - a full pass is about 8 h for a round's games at stage C's rate,
      against 2.5 h heads-only. Whether the trunk pays for that per
      GPU-hour, and whether its gain repeats over rounds, is not read;
    - departure 49 reads round 22 at the reference's decode first, and
      the line starts only on the user's word;
  - a step that budgets its KL by decision class. Per decision, recruits
    return about 3.4 times what moves do, while moves take 69-90% of the
    step.
- **What else to take out of training** (`docs/training_dynamics_audit_20261111.md`,
  the audit the user asked for on 2026-11-11). It lists every mechanism
  between "learn from who won" and the training that runs. Apart from the
  match test, none has a reading in games.
  `docs/training_dynamics_theory_20261111.md` judges each on theory, at
  the user's request, and names the loop theory would keep: every game
  trained on, one form of weight, Adam with momentum and a learning rate,
  rounds of about 1,000 games, exact checks only, one periodic match. The
  ones the audit singles out:
  - downstream of the KL rule, so now unsupported: the arm chosen once at
    equal KL, and the 5,000-game round;
  - never measured and now live: the 16-decision window and the
    per-weight normalisation of the direction;
  - something other than the network decides play: the end-of-turn
    offset at the reference's decode (fitted on `relset`), the attack's
    hex, the advancement, and rejected recruit hexes that the mask reads
    and the network cannot see (against CLAUDE.md's Principle 6).

  The user ruled on ten of these on 2026-11-12 (the theory document's
  last section). Nothing is built from the rulings yet:
  - out: the 8-round average, the held-out games, the entry switch, the
    fixed step lengths and the end-of-turn offset;
  - the step is the optimiser's own update with one learning rate for the
    whole network, not a rate for each of two groups;
  - out too: every correction to the signal but the value estimate (the
    map baseline and both dice-luck terms), with the value head to be
    trained on each round's results. The estimate enters as it is, with
    no fitted coefficient (2026-11-17): the weight is `z − v_t`;
  - smaller: a round is 128 games and one step (2026-11-17), a number
    the user wants revised later. The desktop gives about 525 games an
    hour with a whole-network pass;
  - no statistical check runs in a round; the exact checks stay;
  - the gradient through the memory runs over the whole game-side, not
    16 decisions (2026-11-17). It is not clipped: the pass logs a warning
    on a blow-up, and the choice is revisited then. Whether it fits the
    card, and its rate, are not measured;
  - 80% of a round's games are against the current self and 20% against
    past selves (2026-11-17);
  - all ten are ruled on, and the loop built from them was launched on
    2026-11-17 (`docs/simple_loop_prereg_20261117.md`):
    - the code was `feature/simple-loop` (23e4dff8) and is in `main`
      since 0.34.0 (2026-11-20, on the user's word);
    - the whole-game pass fits the card (2.8 GB at its peak) and runs at
      60 decisions a second, so a round takes about 23 minutes: about 62
      rounds and 8,000 games a day;
    - the run is `C:\Users\momom2\wesnoth_runs\simple_loop`; a STOP file
      there ends it, and the same command resumes it;
    - its verdict after round 160 is "the loop trains" (above). What
      follows is the user's: a PURE match of one of its networks at the
      reference's decode, the learning rate, the round's size, or leaving
      it to run.
- **The desktop's wheel is `main`'s again** (2026-11-20): both venvs hold
  the phase-39 wheel, installed when the simple loop moved to the main
  checkout (its departure 8). The run now plays from
  `...\projects\wesnoth_ai` on `main`'s code, so a change to the working
  tree's files there reaches the run at its next tool. Work on a branch
  goes in a worktree of its own.
- **An outside yardstick past the default AI.** At 96.8% the default AI
  cannot track further progress. A stronger opponent or a handicap is the
  user's to choose.
- `main` was pushed on 2026-11-19, on the user's word.
- **The repository is one branch and one folder since 2026-11-20,** on the
  user's word ("So the repo is clean."). `main` is the only branch, here
  and on GitHub, and `...\projects\wesnoth_ai` the only worktree.
  - Every branch `main` holds was deleted.
  - The five branches `main` does not hold were tagged and the tags
    pushed before they were deleted: `archive/feature/lean-rounds`,
    `archive/feature/fixed-step-rounds`,
    `archive/feature/local-gpu-stage1`, `archive/feature/int8-linear` (6
    commits) and `archive/feature/async-replies` (4). Where this file
    names one of those branches, the tag of that name holds it;
    `git worktree add <folder> archive/<name>` brings its files back.
  - The log files the removed folders held untracked are in
    `C:\Users\momom2\wesnoth_runs\worktree_leftovers`.
  - The run's own worktree went last, after the run had moved.

## Standing measured facts (from before the rounds)

- **Memory:** within `own2`, 64 slots against 0 reads +13.0 ± 12.3.
  - In `parity3` the 64 slots cost 105 Elo. A stronger end_turn offset
    recovered 65 of it. The rest was the memory continuing its own
    history's patterns, the causal confusion of history-conditioned
    imitation.
  - Training on its own games removed that cost
    (`docs/memory_own_round2_prereg_20261011.md`).
- **The end_turn logit offset −1.5** is worth about +229 Elo at decode
  (`docs/endturn_rule_prereg_20260919.md`).
- **No cheap turn grader passes**: within-position correlation against
  playout truth is 0.22-0.53 (`docs/turn_value_prereg_20260925.md`).
- **Turn-level gaps exist:** 7 of 60 holdout positions have a sampled turn
  better by at least 0.25 (`docs/turn_gap_ref_prereg_20260921.md`).
- **Search:** Gumbel-MCTS-32 over atomic actions lost to argmax by −124 ±
  58 (`docs/raw_argmax_control_20260904.md`). Every search and playout
  runs on the true state, so a search used in play needs a belief model
  (`docs/hidden_information_20260926.md`). Common random numbers across
  branches were measured dead.

## The simulator

- The Rust core (`rust/wesnoth_core`, phase 39) is the state of record. It
  answers every rule: commands, events, combat, reach, vision,
  observation and encoding.
- Game records are format 5 since 0.33.2: a unit that leaves the board
  leaves the uncovered list, as in the engine. Records of format 4 and
  earlier rebuild under the rule they were played with.
- It was certified against the retired Python applier on 14,376 of 14,376
  replays.
- The engine checks it:
  - the scenario-init oracle, 28 of 28 scenarios;
  - the hidden-units oracle, 54 of 54 positions;
  - live games against the default AI, mirrored from the engine's log on
    the patched 1.18.8 (`tools/wesnoth_build/`). In the 100 games of
    2026-11-19 the boards agreed at every decision of the 94 finished
    games, on all 21 Ladder maps, and one stopped game found the
    simulator's one defect of the series (a dead unit's id left in the
    uncovered list).
- `OBSERVATION_EPOCH` is 11 and `CORPUS_VERSION` 5.

## Compute

| | desktop (RX 6600) | a 4090 box |
|---|---|---|
| self-play generation | about 470-570 decisions a second (about 3,450 games an hour in rounds 18-19, about 490 decisions a game) | about 2,900 decisions a second |
| a training step of `own2` | about 150 positions a second (fp16, HIP kernels) | about 1,480 |
| an 800-decisive-game match | about 15 min | about 17 min |

Sources: `docs/local_machine_use_20261029.md` and `docs/box_specs.md`.

- **The desktop** is free to use at any time; the user says when they want
  the GPU.
- **A box** costs about $0.42-0.58 an hour. The Vast balance read $81.18 on
  2026-10-22, and the budget is about $100 (user, 2026-10-09).
