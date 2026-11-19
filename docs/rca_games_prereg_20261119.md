# Pre-registration: the loop's network against Wesnoth's default AI (2026-11-19)

**Status:** registered 2026-11-19, not launched. The games need the
project's patched Wesnoth 1.18.8 (`tools/wesnoth_build/`), which is not on
the desktop: see "What blocks the launch".

## Why

The user, 2026-11-19: "Let's see how it does against RCA AI."

Every strength number so far is against `own2`, a network of our own. This
is the first reading against an opponent from outside the project. The
default (RCA) AI is weak by players' standards, so the reading is a floor:
a network that does not beat it nearly always is far from a strong
player, and one that does has shown little more than that.

## What runs

- **The player:** the simple loop's network after round 128
  (`wesnoth_runs\simple_loop\round_0128\pi.pt`, SHA-256
  `6bdcae0947c9e81514e4ad62d7efaa0768a671e789be8e844319bea53b39d3aa`), the
  last one a match has read (+107.4 ± 12.9 against `own2` when sampling,
  `docs/simple_loop_prereg_20261117.md`). It plays at the reference's
  decode (`raw:t0+eo-1.5`) with 64 memory slots, on the CPU.
- **The opponent:** the engine's default AI, with no configuration of
  ours.
- **The games:** `tools/live_vs_rca.py --checkpoint <that file> --label
  s0128 --unwatched --seed 20261119`, from `feature/live-rca-checkpoint`
  at 156901ee (the `--checkpoint` option; `main` could only play the
  reference). Each game is a live Wesnoth game, one at a time, in a
  minimized window. It draws a map of the Ladder pool and both factions
  uniformly (`random_setup(category="ladder")`), with the settings a game
  hosted in the lobby gets, and our side alternates.
- **First one trial game,** to check the bridge on this machine. Then
  **100 games.**
- **Beside the training run:** on the CPU at Idle priority, 4 threads at
  most. The GPU stays the run's.

## What is measured

- **The win rate over the games that finish,** with its binomial standard
  error, and the count of each outcome: win, loss, draw (the turn limit)
  and stopped.
- **A stopped game is a fidelity defect, not a result.** The tool compares
  the engine's board with the simulator's at every decision and stops the
  game on a difference (Principle 4). Each one is recorded and its root
  cause looked for.
- **No Elo is fitted and nothing is adopted.** This is not a strength
  claim in the sense of CLAUDE.md, "Strength claims".

## The number that decides

Whether the default AI can serve as an outside yardstick from here on:

- **At 95 wins or more of 100 finished games** it is saturated, and it
  cannot track further progress. An outside yardstick then needs a
  stronger opponent or a handicap, which is the user's to choose.
- **Under 95,** it stays in use: the same 100 games are replayed for a
  later network, and the losses' records are read for what they have in
  common.

## Kill criteria

1. **The trial game stops at its first decision** (the engine plays
   another experience modifier): the build is wrong. Nothing more is
   launched.
2. **Three stopped games in the first ten:** the series ends there and the
   difference is investigated first.
3. **A game past 30 minutes** is inspected.
4. **The training run slows by more than 10%** in decisions a second over
   a pass, or the display driver resets (System event 4101): the games
   end.

## The estimate

- A decision takes 0.06 s on the CPU beside the run (120 decisions at Idle
  priority on 4 threads, 2026-11-19), so about 15 s of a game is the
  network's.
- The engine's share is not known before the trial game. The guess is 2 to
  5 minutes a game, 3 to 8 hours for the 100. Past 1.5 times the trial
  game's rate, the series is inspected.

## What blocks the launch

- **The patched build is not on the desktop.**
  `Desktop\Perso\games\wesnoth-1.18.8-patched\wesnoth.exe` does not exist,
  and no copy was found under the user's folder or on `D:`. With the Steam
  build the tool refuses to play, by design: a command-line game there
  plays 100% experience where a hosted game plays 70%.
- **GitHub no longer holds it.** The workflow last built it on 2026-10-07
  (run 37600290795, on `main`), and its artifact was kept 30 days.
- **A rebuild needs the user's GitHub sign-in** (`gh auth login`, in their
  own terminal: `docs/new_machine.md`, step 2). Then
  `gh workflow run patched-wesnoth.yml --ref main` builds it and
  `gh run download` fetches it (`tools/wesnoth_build/README.md`).
- **The add-on link is missing too:**
  `Documents\My Games\Wesnoth1.18\data\add-ons\wesnoth_ai` does not exist.
  It is made at launch, pointing at the checkout the games run from.

## Results

None yet.
