# Project status

The project's current state. CLAUDE.md holds only instructions and points
here. Update this file on `main` when a result lands. When a block is
superseded, move it verbatim to `docs/archive/` and index it there.
Earlier status logs: `docs/archive/claude_md_20261109.md` (CLAUDE.md's
"Current status", 2026-10-08 to 2026-10-21) and the files listed in
`docs/archive/README.md`.

Last updated 2026-11-10.

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
24 and have not been played. The trunk has never
trained: its pipeline is sound, but on a few game-sides the memory diverges
under a full-network step, and those sides take most of the step's KL for
no gain ("What the numbers do not show").

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
  pre-registration and its 48 departures are in
  `docs/stage_c_rounds_local_prereg_20261030.md` (main's copy holds 1-41;
  42-48 are on `feature/local-gpu-stage1`). The autonomous window's
  card, plan, reviews and log are in `docs/autonomy_20261029.md`.
- **A round:**
  1. π_k plays itself at T=1: 5,000 games to train on, plus 2,500 held out
     (more in rounds 2-6).
  2. One gradient pass computes the outcome-labelled policy gradient over
     the **policy heads only**; the trunk and the memory do not move.
  3. One step is taken on a KL ladder. It is read on the held-out games
     (half A chooses the step, half B confirms it) and adopted if half B
     reads it positive. When the importance weights fail their ESS check,
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
    been played, and neither has a final match on the lean line, so no
    game yet says whether rounds 19-22 gained strength.
  - It was not relaunched: that is the user's call. A new start
    (`scripts/window_20261029/arm_lean.ps1`) resumes round 23 from its
    markers. Rounds 23 and 24 with round 24's two matches would take about
    12 h (reasoned from the rounds' 4.7 h and the final's rate).

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

Every claim landed within 2 SE of the forecast the rounds' own half-B
readings made.

**Both sides at T=1:** round 16 beats `own2` by +429.1 ± 5.7 Elo, round 3
by +134.1 ± 3.3.

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
  (2026-11-10). The readings are `training/metrics/trunk_*_20261110.json`
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

- **Resume the lean line, or leave it stopped.** It stopped in round 23's
  games on 2026-11-10 ("The line is stopped", above). Rounds 23 and 24
  with round 24's claim and argmax match would give departures 43 and 44
  their registered reading, for about 12 h of the desktop.
- **Adopt round 18, or a later round, as the reference.** The evidence is
  a local match of 12,800 decisive games (+105.7 ± 3.2), and the
  checkpoint is `C:\Users\momom2\wesnoth_runs\rounds_local\round_18\pi.pt`
  (SHA-256 c5796941...). Copies of rounds 3, 15, 16 and 18 are in
  `C:\Users\momom2\wesnoth_checkpoints_window\`.
- **Merging `feature/lean-rounds`** (5 commits, the code the line ran,
  at 21862a0a) and the log commits on `feature/local-gpu-stage1` into
  `main`. The suite passed on the desktop at 21862a0a (2,258 passed, 139
  skipped); CI has not run on it.
- **The next training step.** The options, each a single factor:
  - more games per step, against the falling signal-to-noise;
  - training the trunk. Its step's KL must first stop going to the few
    game-sides where the memory diverges: a reworked memory (`BACKLOG.md`)
    or another of the remedies listed under "What the numbers do not
    show". A step sized on a trimmed KL would hide the divergence on those
    sides, not remove it;
  - a step that budgets its KL by decision class. Per decision, recruits
    return about 3.4 times what moves do, while moves take 69-90% of the
    step.
- **Pushing `main`.** It holds the window's merge (0.32.0). No GitHub
  credentials are stored on the desktop.
- Unmerged branches from before the window: `feature/int8-linear` (6
  commits) and `feature/async-replies` (4).

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

- The Rust core (`rust/wesnoth_core`, phase 38) is the state of record. It
  answers every rule: commands, events, combat, reach, vision,
  observation and encoding.
- It was certified against the retired Python applier on 14,376 of 14,376
  replays.
- The engine checks it:
  - the scenario-init oracle, 28 of 28 scenarios;
  - the hidden-units oracle, 54 of 54 positions;
  - live games against the default AI, mirrored from the engine's log on
    the patched 1.18.8 (`tools/wesnoth_build/`).
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
